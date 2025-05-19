import re
import pandas as pd
from pathlib import Path


def extract_metrics_from_report(report_file):
    """Extract evaluation metrics from the report file."""
    with open(report_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to match question sections
    question_pattern = r'### Question (\d+).*?### Evaluation Metrics\s+BERTScore: ([\d.]+) \(P=([\d.]+), R=([\d.]+)\)\s+F1 Score:\s+([\d.]+)\s+BLEU:\s+([\d.]+)'
    
    # Find all matches
    matches = re.findall(question_pattern, content, re.DOTALL)
    
    # Process matches into list of dictionaries
    results = []
    for match in matches:
        question_number, bertscore, precision, recall, f1_score, bleu_score = match
        results.append({
            'Question Number': int(question_number),
            'BERTScore': float(bertscore),
            'Precision': float(precision),
            'Recall': float(recall),
            'F1 Score': float(f1_score),
            'BLEU Score': float(bleu_score)
        })
    
    # Sort by question number
    results = sorted(results, key=lambda x: x['Question Number'])
    
    return results

def save_metrics_to_excel(metrics, output_file):
    """Save metrics to an Excel file."""
    df = pd.DataFrame(metrics)
    df.to_excel(output_file, index=False)
    print(f"Metrics saved to {output_file}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Number of questions: {len(metrics)}")
    print(f"Average BERTScore: {df['BERTScore'].mean():.4f}")
    print(f"Average F1 Score: {df['F1 Score'].mean():.4f}")
    print(f"Average BLEU Score: {df['BLEU Score'].mean():.4f}")
    
    # Add helpful analysis
    print("\nScore Distribution:")
    print(f"Questions with BERTScore ≥ 0.5: {(df['BERTScore'] >= 0.5).sum()} ({(df['BERTScore'] >= 0.5).sum() / len(df) * 100:.1f}%)")
    print(f"Questions with F1 Score ≥ 0.3: {(df['F1 Score'] >= 0.3).sum()} ({(df['F1 Score'] >= 0.3).sum() / len(df) * 100:.1f}%)")
    print(f"Questions with BLEU Score ≥ 0.1: {(df['BLEU Score'] >= 0.1).sum()} ({(df['BLEU Score'] >= 0.1).sum() / len(df) * 100:.1f}%)")


def main():
    # Find the report files
    report_dir = Path("reports")
    if not report_dir.exists():
        report_dir = Path("../..")

    report_files = list(report_dir.glob("evaluation_report_*.txt"))
    if not report_files:
        print("No report files found.")
        return
    
    # Get the latest report file
    latest_report = max(report_files)
    print(f"Processing report: {latest_report}")

    # Extract metrics from the report
    metrics = extract_metrics_from_report(latest_report)

    # Create output directory for Excel files
    output_file = latest_report.parent / f"{latest_report.stem}_metrics.xlsx"

    # Save metrics to Excel
    save_metrics_to_excel(metrics, output_file)

if __name__ == "__main__":
    main()