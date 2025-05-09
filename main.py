
import os
import json
import time
import torch
import logging

from dotenv import load_dotenv
from bert_score import BERTScorer
from transformers import BertTokenizer, BertModel
from utils.data_extractor import DataExtractor
from opwebui.api_client import OpenWebUIClient

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_PATH = os.path.abspath(os.getenv("DATA_DIR_PATH"))
QUESTION_PATH = os.path.join(DATA_PATH, os.getenv("QUESTION_EXCEL"))
SOLUTION_PATH = os.path.join(DATA_PATH, os.getenv("SOLUTION_EXCEL"))
QUESTION_SHEET_NAME = os.getenv("QUESTION_SHEET_NAME")

if not DATA_PATH or not QUESTION_PATH or not SOLUTION_PATH or not QUESTION_SHEET_NAME:
    raise ValueError("DATA_DIR_PATH, QUESTION_EXCEL, SOLUTION_EXCEL, and QUESTION_SHEET_NAME must be set in the environment variables.")

def main():
    try:
        # Ensure data directory exists
        os.makedirs(DATA_PATH, exist_ok=True)
        
        if not os.path.exists(QUESTION_PATH) or not os.path.exists(SOLUTION_PATH):
            logging.error(f"One or more required files are missing. Please check that both files exist in the data directory.")
            return
        
        # Initialize the data extractor
        extractor = DataExtractor(
            questions_path=QUESTION_PATH,
            answers_path=SOLUTION_PATH,
            questions_sheet_name=QUESTION_SHEET_NAME
        )
        
        # Load and parse the data
        extractor.load_and_parse_data()
        
        # Get the parsed data
        questions = extractor.get_questions()
        solutions = extractor.get_solutions()
        
        # Initialize BERTScorer (only once for efficiency)
        logging.info("Initializing BERTScorer...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        scorer = BERTScorer(model_type="bert-base-uncased", device=device)
        logging.info(f"Using device: {device} for BERTScore calculations")

        for i, question in enumerate(questions[:5]):
            logging.info(f"Processing question {i+1}/{len(questions[:5])}")
            
            # Get model response for this question
            client = OpenWebUIClient()
            prompt = question.issue
            logging.info(f"Sending prompt to model: {prompt[:50]}...")
            
            response = client.chat_with_model(prompt)
            if not response:
                logging.error(f"No response received for question {question.id}")
                continue
                
            model_response = response.choices[0].message.content
            logging.info(f"Received model response: {len(model_response)} chars")

             # Find the associated solution(s) based on solutions_used field
            if not question.solutions_used:
                # If no solutions are marked, compare with all solutions
                solution_indices = list(range(len(solutions)))
                logging.info(f"No specific solution marked for question {question.id}, comparing with all solutions")
            else:
                # Use the specific solutions marked for this question
                solution_indices = question.solutions_used
                logging.info(f"Using solutions {solution_indices} for question {question.id}")
            
             # Calculate BERTScore for each potential solution
            best_score = -1
            best_solution_idx = -1
            
            for sol_idx in solution_indices:
                if sol_idx >= len(solutions):
                    logging.warning(f"Solution index {sol_idx} out of range, skipping")
                    continue
                    
                solution = solutions[sol_idx]
                solution_text = " ".join(solution.steps)  # Join all solution steps
                
                # Calculate BERTScore
                P, R, F1 = scorer.score([model_response], [solution_text])
                
                # Get the scores as float values
                precision = P.item()
                recall = R.item()
                f1 = F1.item()
                
                logging.info(f"Solution {sol_idx} - BERTScore: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")
                
                # Update the best score if this one is better
                if f1 > best_score:
                    best_score = f1
                    best_solution_idx = sol_idx
            
            # Update the question with the best BERTScore
            if best_solution_idx >= 0:
                question.bert_score = best_score
                logging.info(f"Best match: Solution {best_solution_idx} with F1={best_score:.4f}")
                
                # Print the match
                print(f"\n=== Question {question.id} ===")
                print(f"Issue: {question.issue[:100]}...")
                print(f"\n=== Model Response ===")
                print(f"{model_response[:200]}...")
                print(f"\n=== Best Matching Solution ({best_solution_idx}) ===")
                print(f"{solutions[best_solution_idx].title}")
                for j, step in enumerate(solutions[best_solution_idx].steps[:5]):
                    print(f"  {j+1}. {step}")
                print(f"...(more steps)") if len(solutions[best_solution_idx].steps) > 5 else None
                print(f"\nBERTScore: {best_score:.4f}")
                print("\n" + "="*50 + "\n")
            
            time.sleep(1)
            


    except Exception as e:
        logging.error(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()