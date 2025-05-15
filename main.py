
import os
import time
import logging
import traceback


from datetime import datetime
from dotenv import load_dotenv

from utils.data_extractor import DataExtractor
from opwebui.api_client import OpenWebUIClient
from utils.report_generator import generate_report
from metrics.score_calculator import ScoreCalculator
from metrics.solution_matcher import SolutionMatcher
from utils.quality_filter import assess_response_quality, get_improved_prompt

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_PATH = os.path.abspath(os.getenv("DATA_DIR_PATH"))
QUESTION_PATH = os.path.join(DATA_PATH, os.getenv("QUESTION_EXCEL"))
SOLUTION_PATH = os.path.join(DATA_PATH, os.getenv("SOLUTION_EXCEL"))
QUESTION_SHEET_NAME = os.getenv("QUESTION_SHEET_NAME")

if not DATA_PATH or not QUESTION_PATH or not SOLUTION_PATH:
    raise ValueError("DATA_DIR_PATH, QUESTION_EXCEL, SOLUTION_EXCEL must be set in the environment variables.")


def display_results(question, model_response, best_solution, metrics):
    """
    Display formatted results of model response evaluation.
    
    Args:
        question: The Question object
        model_response: The model's generated response text
        best_solution: The best matching Solution object
        metrics: Dictionary of evaluation metrics
    """
    # Log the best match
    logging.info(f"Best match for question {question.id} with F1={metrics['bert_f1']:.4f}")
    
    # Print formatted output
    print(f"\n{'='*50}")
    print(f"=== Question {question.id} ===")
    print(f"Issue: {question.issue[:100]}..." if len(question.issue) > 100 else f"Issue: {question.issue}")
    
    print(f"\n=== Model Response ===")
    response_preview = model_response[:300] + "..." if len(model_response) > 300 else model_response
    print(response_preview)
    
    print(f"\n=== Best Matching Solution ({best_solution.id}) ===")
    print(f"Title: {best_solution.title}")
    
    # Print up to 5 steps
    for j, step in enumerate(best_solution.steps[:5]):
        print(f"  {j+1}. {step}")
    
    # Show ellipsis if there are more steps
    if len(best_solution.steps) > 5:
        print(f"  ...(+{len(best_solution.steps) - 5} more steps)")
    
    # Print metrics
    print(f"\nEvaluation Metrics:")
    print(f"  BERTScore: {metrics['bert_f1']:.4f} (P={metrics['bert_precision']:.4f}, R={metrics['bert_recall']:.4f})")
    print(f"  F1 Score:  {metrics['trad_f1']:.4f}")
    print(f"  BLEU:      {metrics['bleu']:.4f}")
    
    print(f"\n{'='*50}\n")


def main():
    try:
        # Ensure data directory exists, create if not
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
        
        # Initialize the score calculator and matcher
        score_calculator = ScoreCalculator()
        matcher = SolutionMatcher(score_calculator)

        # Prepare metrics storage for report generation
        metrics_by_question = {}

        # NEED TO REMOVE LIMITER [:1] IF WANT TO PROCESS ALL QUESTIONS
        for i, question in enumerate(questions[:1]):
            logging.info(f"Processing question {i+1}/{len(questions[:1])}")
            
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
            
            solutions_to_compare = [solutions[i] for i in solution_indices]

            best_solution, metrics = matcher.find_best_solution(
                model_response, 
                solutions_to_compare
            )

            question.bert_score = metrics['bert_f1']
            question.f1_score = metrics['trad_f1']
            question.bleu_score = metrics['bleu']

            # Store metrics for report generation
            metrics_by_question[question.id] = {
                'metrics': metrics,
                'best_solution_id': best_solution.id,
                'model_response': model_response
            }

            # Check quality and potentially improve prompt
            is_acceptable, feedback = assess_response_quality(metrics)
            if not is_acceptable:
                logging.warning(f"Question {question.id} response quality below threshold")
                logging.info(feedback)
                
                # Generate improved prompt for future use
                improved_prompt = get_improved_prompt(question.issue, metrics)
                logging.info(f"Improved prompt: {improved_prompt[:100]}...")
            
            # display_results(question, model_response, best_solution, metrics)
            
            time.sleep(1)
    
        # Generate comprehensive report after all questions processed
        if metrics_by_question:
            report_path = generate_report(questions, solutions, metrics_by_question)
            logging.info(f"Evaluation report generated: {report_path}")
            
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()