import os
import re
import pandas as pd

from pathlib import Path
from datetime import datetime

def assess_response_quality(metrics, 
                           bert_threshold=0.5, 
                           f1_threshold=0.3, 
                           bleu_threshold=0.1,
                           combined_threshold=0.4):
    """
    Assess the quality of a response based on evaluation metrics.
    
    Args:
        metrics: Dictionary of evaluation metrics
        bert_threshold: Minimum acceptable BERTScore
        f1_threshold: Minimum acceptable traditional F1 score
        bleu_threshold: Minimum acceptable BLEU score
        combined_threshold: Minimum acceptable combined score
        
    Returns:
        tuple: (is_acceptable, feedback_message)
    """
    is_acceptable = True
    feedback = []
    
    bert_score = metrics.get('bert_f1', 0)
    f1_score = metrics.get('trad_f1', 0)
    bleu_score = metrics.get('bleu', 0)
    combined_score = metrics.get('combined_score', 0)
    
    # Check if metrics meet thresholds
    if bert_score < bert_threshold:
        is_acceptable = False
        feedback.append(f"BERTScore ({bert_score:.4f}) below threshold ({bert_threshold:.2f})")
        
    if f1_score < f1_threshold:
        is_acceptable = False
        feedback.append(f"F1 score ({f1_score:.4f}) below threshold ({f1_threshold:.2f})")
    
    if bleu_score < bleu_threshold:
        is_acceptable = False
        feedback.append(f"BLEU score ({bleu_score:.4f}) below threshold ({bleu_threshold:.2f})")
    
    # Consider combined score as well
    if combined_score < combined_threshold:
        is_acceptable = False
        feedback.append(f"Combined score ({combined_score:.4f}) below threshold ({combined_threshold:.2f})")
    
    # But if combined score is very good, accept despite individual failures
    if combined_score >= combined_threshold * 1.25:
        is_acceptable = True
        feedback = [f"Combined score ({combined_score:.4f}) indicates good quality despite some low metrics"]
    
    if is_acceptable:
        return True, "Response meets quality standards."
    else:
        feedback_message = "Response quality issues detected:\n- " + "\n- ".join(feedback)
        feedback_message += "\n\nSuggested actions:\n"
        
        if bert_score < bert_threshold:
            feedback_message += "- Improve semantic relevance to the question\n"
        if f1_score < f1_threshold:
            feedback_message += "- Include more key terms from the reference solution\n"
        if bleu_score < bleu_threshold:
            feedback_message += "- Structure the response more similarly to reference solutions\n"
            
        feedback_message += "\nPlease provide more detailed context or a more specific query."
        return False, feedback_message


def get_improved_prompt(original_prompt, metrics, 
                        bert_threshold=0.5, 
                        f1_threshold=0.3, 
                        bleu_threshold=0.1,
                        combined_threshold=0.4):
    """
    Generate an improved prompt if the response quality is below thresholds.
    
    Args:
        original_prompt: The original prompt text
        metrics: Evaluation metrics dictionary
        thresholds: As in assess_response_quality
        
    Returns:
        str: Either the original prompt (if quality is good) or an enhanced prompt
    """
    is_acceptable, feedback = assess_response_quality(
        metrics, bert_threshold, f1_threshold, bleu_threshold, combined_threshold
    )
    
    if is_acceptable:
        return original_prompt
    
    # Extract specific issues from feedback
    bert_failed = metrics['bert_f1'] < bert_threshold
    f1_failed = metrics['trad_f1'] < f1_threshold
    bleu_failed = metrics['bleu'] < bleu_threshold
    combined_failed = metrics.get('combined_score', 0) < combined_threshold
    
    # Create an enhanced prompt
    enhanced_prompt = (
        f"I need more detailed information about this issue. "
        f"My previous question was:\n\n{original_prompt}\n\n"
        f"The previous response had these quality issues:\n{feedback}\n\n"
        f"Please provide a comprehensive answer with:\n"
    )
    
    # Add specific instructions based on which metrics failed
    if bert_failed:
        enhanced_prompt += "- Better semantic relevance to my specific question\n"
    if f1_failed:
        enhanced_prompt += "- More key technical terms and specific terminology\n"
    if bleu_failed:
        enhanced_prompt += "- Clearer structure similar to standard solutions\n"
    if combined_failed:
        enhanced_prompt += "- A more comprehensive and detailed response\n"
    
    # Add general improvements
    enhanced_prompt += (
        "- Step-by-step instructions\n"
        "- Relevant technical details\n"
        "- Any specific commands or settings needed\n"
        "- Common pitfalls to avoid"
    )
    return enhanced_prompt

def generate_report(questions, solutions, metrics_by_question, output_dir="reports",
                   bert_threshold=0.5, 
                   f1_threshold=0.3, 
                   bleu_threshold=0.1,
                   combined_threshold=0.4):
    """
    Generate a detailed text report of evaluation results.
    
    Args:
        questions: List of Question objects
        solutions: List of Solution objects
        metrics_by_question: Dictionary mapping question ID to metrics
        output_dir: Directory to save the report file
        bert_threshold: Threshold for BERTScore
        f1_threshold: Threshold for F1 score
        bleu_threshold: Threshold for BLEU score
        combined_threshold: Threshold for combined score
        
    Returns:
        str: Path to the generated report file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate average scores
    avg_bert = sum(m['metrics']['bert_f1'] for m in metrics_by_question.values()) / len(metrics_by_question)
    avg_f1 = sum(m['metrics']['trad_f1'] for m in metrics_by_question.values()) / len(metrics_by_question)
    avg_bleu = sum(m['metrics']['bleu'] for m in metrics_by_question.values()) / len(metrics_by_question)
    avg_combined = sum(m['metrics'].get('combined_score', 0) for m in metrics_by_question.values()) / len(metrics_by_question)
    
    # Generate timestamp for the report filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"evaluation_report_{timestamp}.txt")
    
    # Write the report to a file
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# KB Answer Evaluation Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Write overall statistics
        f.write(f"## Overall Statistics\n")
        f.write(f"Total Questions Evaluated: {len(metrics_by_question)}\n")
        f.write(f"Average BERTScore: {avg_bert:.4f}\n")
        f.write(f"Average F1 Score: {avg_f1:.4f}\n")
        f.write(f"Average BLEU Score: {avg_bleu:.4f}\n\n")
        f.write(f"Average Combined Score: {avg_combined:.4f}\n\n")
        
        # Write detailed results for each question
        f.write(f"## Detailed Results\n\n")
        
        # Process each question with metrics
        for question in questions:
            if question.id not in metrics_by_question:
                continue
                
            data = metrics_by_question[question.id]
            metrics = data['metrics']
            best_solution_id = data['best_solution_id']
            model_response = data['model_response']
            
            # Find the best matching solution
            best_solution = next((s for s in solutions if s.id == best_solution_id), None)
            if not best_solution:
                continue
                
            f.write(f"{'='*80}\n")
            f.write(f"### Question {question.id}\n")
            f.write(f"Issue: {question.issue}\n\n")
            
            f.write(f"### Model Response\n{model_response}\n\n")
            
            f.write(f"### Best Matching Solution ({best_solution.id})\n")
            f.write(f"Title: {best_solution.title}\n\n")
            f.write("Steps:\n")
            for step in best_solution.steps:
                f.write(f"  {step}\n")
            f.write("\n")
            
            f.write(f"### Evaluation Metrics\n")
            f.write(f"  BERTScore: {metrics['bert_f1']:.4f} (P={metrics['bert_precision']:.4f}, R={metrics['bert_recall']:.4f})\n")
            f.write(f"  F1 Score:  {metrics['trad_f1']:.4f}\n")
            f.write(f"  BLEU:      {metrics['bleu']:.4f}\n")
            f.write(f"  Combined:  {metrics.get('combined_score', 0):.4f}\n\n")
            
            # Add quality assessment
            f.write(f"### Quality Assessment\n")
            is_acceptable, feedback = assess_response_quality(
                metrics,
                bert_threshold=bert_threshold,
                f1_threshold=f1_threshold,
                bleu_threshold=bleu_threshold,
                combined_threshold=combined_threshold
            )
            
            f.write(f"  Status: {'PASS' if is_acceptable else 'FAIL'}\n\n{feedback}\n\n")
    
    return report_path

def extract_metrics_from_report(report_file):
    """
    Extract evaluation metrics from a report file.
    
    Args:
        report_file: Path to the report file
        
    Returns:
        dict: Dictionary of extracted metrics
    """
    metrics_data = {
        'question_id': [],
        'issue': [],
        'bert_score': [],
        'bert_precision': [],
        'bert_recall': [],
        'f1_score': [],
        'bleu_score': [],
        'combined_score': [],
        'quality_pass': []
    }
    
    with open(report_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract questions and metrics
    questions = re.split(r'={80}', content)
    
    for question in questions[1:]:  # Skip the header
        # Extract question ID
        question_id_match = re.search(r'### Question (\d+)', question)
        if not question_id_match:
            continue
        question_id = question_id_match.group(1)
        metrics_data['question_id'].append(question_id)
        
        # Extract issue
        issue_match = re.search(r'Issue: (.*?)(?:\n\n|\Z)', question, re.DOTALL)
        issue = issue_match.group(1).strip() if issue_match else "N/A"
        metrics_data['issue'].append(issue)
        
        # Extract metrics
        bert_score_match = re.search(r'BERTScore: ([\d\.]+)', question)
        bert_score = float(bert_score_match.group(1)) if bert_score_match else 0
        metrics_data['bert_score'].append(bert_score)
        
        bert_p_match = re.search(r'P=([\d\.]+)', question)
        bert_p = float(bert_p_match.group(1)) if bert_p_match else 0
        metrics_data['bert_precision'].append(bert_p)
        
        bert_r_match = re.search(r'R=([\d\.]+)', question)
        bert_r = float(bert_r_match.group(1)) if bert_r_match else 0
        metrics_data['bert_recall'].append(bert_r)
        
        f1_match = re.search(r'F1 Score:  ([\d\.]+)', question)
        f1_score = float(f1_match.group(1)) if f1_match else 0
        metrics_data['f1_score'].append(f1_score)
        
        bleu_match = re.search(r'BLEU:      ([\d\.]+)', question)
        bleu_score = float(bleu_match.group(1)) if bleu_match else 0
        metrics_data['bleu_score'].append(bleu_score)
        
        combined_match = re.search(r'Combined:  ([\d\.]+)', question)
        combined_score = float(combined_match.group(1)) if combined_match else 0
        metrics_data['combined_score'].append(combined_score)
        
        # Extract quality assessment
        quality_match = re.search(r'(PASS|FAIL):', question)
        quality = quality_match.group(1) if quality_match else "N/A"
        metrics_data['quality_pass'].append(quality)
    
    return metrics_data


def export_report_to_excel(report_file=None):
    """
    Export a report file to Excel format.
    If no report file is specified, uses the most recent report.
    
    Args:
        report_file: Path to the report file (optional)
        
    Returns:
        str: Path to the generated Excel file
    """
    # Find the report files
    report_dir = Path("reports")
    if not report_dir.exists():
        report_dir = Path("output")
        if not report_dir.exists():
            report_dir.mkdir()
    
    if report_file:
        report_path = Path(report_file)
    else:
        report_files = list(report_dir.glob("evaluation_report_*.txt"))
        if not report_files:
            print("No report files found.")
            return None
        
        # Get the latest report file
        report_path = max(report_files, key=lambda p: p.stat().st_mtime)
    
    # Extract metrics from the report
    metrics = extract_metrics_from_report(report_path)
    
    # Create DataFrame
    df = pd.DataFrame(metrics)
    
    # Create output file path
    output_file = report_path.parent / f"{report_path.stem}.xlsx"
    
    # Save to Excel
    df.to_excel(output_file, index=False)
    
    return str(output_file)