class SolutionMatcher:
    def __init__(self, score_calculator):
        self.score_calculator = score_calculator
    
    def find_best_solution(self, response_text, solutions, metric_key='bert_f1'):
        """Find best matching solution based on specified metric"""
        best_score = -1
        best_solution = None
        best_metrics = None
        
        
        for solution in solutions:
            solution_text = " ".join(solution.steps)
            metrics = self.score_calculator.calculate_all_metrics(response_text, solution_text)
            
            if metrics[metric_key] > best_score:
                best_score = metrics[metric_key]
                best_solution = solution
                best_metrics = metrics
        
        return best_solution, best_metrics