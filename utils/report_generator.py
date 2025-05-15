import os

from datetime import datetime
from utils.quality_filter import assess_response_quality

def generate_report(questions, solutions, metrics_by_question, output_dir="output"):
    """
    Generate a detailed text report of evaluation results.
    
    Args:
        questions: List of Question objects
        solutions: List of Solution objects
        metrics_by_question: Dictionary mapping question ID to metrics and best solution ID
        output_dir: Directory to save the report
    
    Returns:
        str: Path to the generated report file
    """
    # Create reports directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"evaluation_report_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"# KB Answer Evaluation Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Overall statistics
        f.write("## Overall Statistics\n")
        
        # Calculate average scores
        avg_bert = sum(m['metrics']['bert_f1'] for m in metrics_by_question.values()) / len(metrics_by_question)
        avg_f1 = sum(m['metrics']['trad_f1'] for m in metrics_by_question.values()) / len(metrics_by_question)
        avg_bleu = sum(m['metrics']['bleu'] for m in metrics_by_question.values()) / len(metrics_by_question)
        
        f.write(f"Total Questions Evaluated: {len(metrics_by_question)}\n")
        f.write(f"Average BERTScore: {avg_bert:.4f}\n")
        f.write(f"Average F1 Score: {avg_f1:.4f}\n")
        f.write(f"Average BLEU Score: {avg_bleu:.4f}\n\n")

        # Detailed results for each question
        f.write("## Detailed Results\n\n")

        for q_id, data in metrics_by_question.items():
            question = next((q for q in questions if q.id == q_id), None)
            if not question:
                continue

            best_solution_id = data['best_solution_id']
            best_solution = next((s for s in solutions if s.id == best_solution_id), None)
            metrics = data['metrics']
            model_response = data['model_response']

            f.write(f"{'='*80}\n")
            f.write(f"### Question {q_id}\n")
            f.write(f"Issue: {question.issue}\n\n")
            
            f.write(f"### Model Response\n")
            f.write(f"{model_response}\n\n")

            if best_solution:
                f.write(f"### Best Matching Solution ({best_solution_id})\n")
                f.write(f"Title: {best_solution.title}\n\n")
                
                f.write("Steps:\n")
                for j, step in enumerate(best_solution.steps):
                    f.write(f"  {j+1}. {step}\n")
                f.write("\n")
            
            f.write(f"### Evaluation Metrics\n")
            f.write(f"  BERTScore: {metrics['bert_f1']:.4f} (P={metrics['bert_precision']:.4f}, R={metrics['bert_recall']:.4f})\n")
            f.write(f"  F1 Score:  {metrics['trad_f1']:.4f}\n")
            f.write(f"  BLEU:      {metrics['bleu']:.4f}\n\n")

            # Quality assessment
            is_acceptable, message = assess_response_quality(metrics)
            f.write(f"### Quality Assessment\n")
            f.write(f"  Status: {'Acceptable' if is_acceptable else 'Needs Improvement'}\n")
            f.write(f"  {message}\n\n")

    return filepath