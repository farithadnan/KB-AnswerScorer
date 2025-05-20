
from ast import arg
import os
import re
import time
import logging
import argparse
import traceback

from tqdm import tqdm
from dotenv import load_dotenv

from utils.data_extractor import DataExtractor
from opwebui.api_client import OpenWebUIClient
from metrics.score_calculator import ScoreCalculator
from metrics.solution_matcher import SolutionMatcher
from utils.evaluation_utils import (
    generate_report,
    assess_response_quality, 
    get_improved_prompt,
    export_report_to_excel
)

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
        # Parse command line arguments
        args = parse_args()

        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            logging.debug("Verbose mode enabled")

        # Ensure data directory exists, create if not
        os.makedirs(DATA_PATH, exist_ok=True)
        
        if not os.path.exists(QUESTION_PATH) or not os.path.exists(SOLUTION_PATH):
            logging.error(f"One or more required files are missing. Please check that both files exist in the data directory.")
            return
        
        # Initialize the data extractor
        extractor = DataExtractor(
            questions_path=QUESTION_PATH,
            answers_path=SOLUTION_PATH,
            questions_config={
                "sheet_name": QUESTION_SHEET_NAME,
                "header_row": 2,
                "issue_col": 'B',
                'solutions_col': 'C',
                'ai_solutions_col': 'D'
            },
            answers_config={
                "header_row": 0,
                "title_col": 'A',
                'steps_col': 'B'
            }
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

        # Use the limit argument if provided
        if args.question_id:
            question_to_process = [q for q in questions if q.id == args.question_id]
            if not question_to_process:
                logging.error(f"Question ID {args.question_id} not found.")
                return
        else:
            question_to_process = questions if args.limit <= 0 else questions[:args.limit]

        total_questions = len(question_to_process)

        for i, question in enumerate(tqdm(question_to_process, desc="\nProcessing questions", unit="question")):
            logging.info(f"Processing question {i+1}/{total_questions} (ID: {question.id})")
            
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

            # Find the associated solution(s) based on solutions_used or ai_solutions_used field
            if len(question.ai_solutions_used) > 0:
                # Prefer AI solutions if specified
                solution_indices = question.ai_solutions_used
                logging.info(f"Using AI solutions {solution_indices} for question {question.id}")
            elif len(question.solutions_used) > 0:
                # Fall back to regular solutions
                solution_indices = question.solutions_used
                logging.info(f"Using regular solutions {solution_indices} for question {question.id}")
            else:
                # If no solutions are marked, compare with all solutions
                solution_indices = list(range(1, len(solutions) + 1))  # Use 1-based indices to match Excel
                logging.info(f"No specific solution marked for question {question.id}, comparing with all solutions")

            # Filter out invalid indices (ensure they're 0-based for array indexing)
            valid_indices = [i-1 for i in solution_indices if 1 <= i <= len(solutions)]  
            solutions_to_compare = [solutions[i] for i in valid_indices]

            # Skip if no valid solutions to compare against
            if not solutions_to_compare:
                logging.warning(f"No valid solutions found for question {question.id}")
                continue

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
            is_acceptable, feedback = assess_response_quality(
                metrics,
                bert_threshold=args.bert_threshold,
                f1_threshold=args.f1_threshold,
                bleu_threshold=args.bleu_threshold,
                combined_threshold=args.combined_threshold
            )
            if not is_acceptable:
                logging.warning(f"Question {question.id} response quality below threshold")
                logging.info(feedback)
                
                # Generate improved prompt for future use
                improved_prompt = get_improved_prompt(
                    question.issue, 
                    metrics, 
                    bert_threshold=args.bert_threshold,
                    f1_threshold=args.f1_threshold,
                    bleu_threshold=args.bleu_threshold,
                    combined_threshold=args.combined_threshold
                )
                logging.info(f"Improved prompt: {improved_prompt[:100]}...")
            
            if args.verbose:
                display_results(question, model_response, best_solution, metrics)
            
            time.sleep(args.wait_time)
    
        # Generate comprehensive report after all questions processed
        if metrics_by_question and not args.skip_report:
            report_path = generate_report(questions, solutions, metrics_by_question,
                                          output_dir=args.report_dir,
                                          bert_threshold=args.bert_threshold,
                                          f1_threshold=args.f1_threshold,
                                          bleu_threshold=args.bleu_threshold,
                                          combined_threshold=args.combined_threshold)
            if args.export_excel and report_path:
                excel_path = export_report_to_excel(report_path)
                logging.info(f"Evaluation metrics exported to Excel: {excel_path}")

            logging.info(f"Evaluation report generated: {report_path}")
            
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        traceback.print_exc()
        return
        

def parse_args():
    """
    Parse command line arguments.
    Returns:
        Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Knowledge Base Answer Scorer")
    parser.add_argument("--bert-threshold", "--bt", type=float, default=0.5, help="BERT score threshold for quality assessment")
    parser.add_argument("--f1-threshold", "--f1", type=float, default=0.3, help="F1 score threshold for quality assessment")
    parser.add_argument("--bleu-threshold", "--bl", type=float, default=0.1, help="BLEU score threshold for quality assessment")
    parser.add_argument("--combined-threshold", "--ct", type=float, default=0.4, help="Combined score threshold for quality assessment")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of questions to process")
    parser.add_argument("--question-id", type=str, help="Process only a specific question ID")
    parser.add_argument("--verbose", "-v", action="store_true", help="Display detailed logs")
    parser.add_argument("--report-dir", type=str, default="reports", help="Directory to save reports")
    parser.add_argument("--wait-time", type=float, default=1.0, help="Wait time between API calls in seconds")
    parser.add_argument("--skip-report", action="store_true", help="Skip report generation")
    parser.add_argument("--export-excel", "-e", action="store_true", help="Export evaluation report to Excel")

    return parser.parse_args()

if __name__ == "__main__":
    main()