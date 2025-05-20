import re

class SolutionMatcher:
    def __init__(self, score_calculator):
        self.score_calculator = score_calculator
    
    def find_best_solution(self, response_text, solutions, metric_key='combined_score'):
        """Find best matching solution based on specified metric"""
        best_score = -1
        best_solution = None
        best_metrics = None
        
        # Preprocess response to normalize formatting
        response_text = self._preprocess_text(response_text)

        for solution in solutions:
            solution_text = " ".join(solution.steps)
            solution_text = self._preprocess_text(solution_text)
            
            metrics = self.score_calculator.calculate_all_metrics(response_text, solution_text)
            
            # Use combined_score by default for better matching
            if metrics[metric_key] > best_score:
                best_score = metrics[metric_key]
                best_solution = solution
                best_metrics = metrics
        
        return best_solution, best_metrics

    def _preprocess_text(self, text):
        """Normalize text for better comparison"""
        # Remove extra whitespace and normalize line endings
        text = re.sub(r'\s+', ' ', text)
        # Remove markdown formatting (**, __, etc.)
        text = re.sub(r'[\*_]{1,2}(.*?)[\*_]{1,2}', r'\1', text)
        # Remove bullet points and numbering
        text = re.sub(r'^[\d\.\-\*]+\s+', '', text, flags=re.MULTILINE)
        return text.strip()