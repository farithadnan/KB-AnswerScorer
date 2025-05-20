import re
import nltk
import torch
import logging

nltk.download('punkt_tab', quiet=True)

from bert_score import BERTScorer
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


class ScoreCalculator:
    def __init__(
            self,
            device=None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_scorer = BERTScorer(
            model_type="bert-base-uncased",
            device=self.device,
        )

    def calculate_all_metrics(
            self, 
            response_text, 
            solution_text
    ):
        """
        Calculate all metrics for the given model response and solution text.
        
        Args:
            model_response (str): The model's response.
            solution_text (str): The reference solution text.
        
        Returns:
            dict: A dictionary containing all calculated metrics.
        """
        metrics = {}
        
        # BERTScore
        metrics['bert_precision'], metrics['bert_recall'], metrics['bert_f1'] = self.calculate_bert_score(response_text, solution_text)
        
        # Traditional F1
        metrics['trad_f1'] = self.calculate_traditional_f1(response_text, solution_text)
        
        # BLEU
        metrics['bleu'] = self.calculate_bleu_score(response_text, solution_text)

        # Calculate steps-specific BLEU (focusing only on procedural content)
        response_steps = self._format_response_for_scoring(response_text)
        solution_steps = self._format_response_for_scoring(solution_text)
        metrics['steps_bleu'] = self.calculate_bleu_score(response_steps, solution_steps)
        
        # Custom combined score (weighted average favoring semantic similarity)
        metrics['combined_score'] = 0.6 * metrics['bert_f1'] + 0.3 * metrics['trad_f1'] + 0.1 * metrics['steps_bleu']
        
        return metrics
    

    
    def calculate_bert_score(self, response_text, solution_text):
        """
        Calculate BERTScore for the given model response and solution text.
        
        Args:
            model_response (str): The model's response.
            solution_text (str): The reference solution text.
        
        Returns:
            tuple: A tuple containing precision, recall, and F1 scores.
        """
        P, R, F1 = self.bert_scorer.score([response_text], [solution_text])
        return P.item(), R.item(), F1.item()
    

    def calculate_traditional_f1(self, response_text, solution_text):
        """
        Calculate traditional F1 score based on token overlap.
        
        Args:
            response_text (str): The model's response.
            solution_text (str): The reference solution text.
        
        Returns:
            float: The F1 score.
    
        """
        # Tokenize texts
        response_tokens = set(word_tokenize(response_text.lower()))
        solution_tokens = set(word_tokenize(solution_text.lower()))

        # Extract key technical terms from solution
        key_terms = self._extract_technical_terms(solution_text)

        # Check which key terms are found in response
        found_terms = [term for term in key_terms if any(term.lower() in token.lower() for token in response_tokens)]
    
        
        # Calculate precision and recall based on token overlap
        # true_positives = len(response_tokens.intersection(solution_tokens))
        # if len(response_tokens) == 0:
        #     precision = 0
        # else:
        #     precision = true_positives / len(response_tokens)
        
        # if len(solution_tokens) == 0:
        #     recall = 0
        # else:
        #     recall = true_positives / len(solution_tokens)

        # Calculate weighted precision and recall
        term_weight = 2.0  # Weight key terms higher
        
        weighted_tp = sum([term_weight if term in found_terms else 1.0 for term in solution_tokens.intersection(response_tokens)])
        precision = weighted_tp / len(response_tokens) if response_tokens else 0
        
        weighted_fn = sum([term_weight if term in key_terms and term not in found_terms else 1.0 
                        for term in solution_tokens.difference(response_tokens)])
        recall = weighted_tp / (weighted_tp + weighted_fn) if (weighted_tp + weighted_fn) > 0 else 0
    
        # Calculate F1
        if precision + recall == 0:
            return 0
        else:
            return 2 * (precision * recall) / (precision + recall)
        
    

    def calculate_bleu_score(self, response_text, solution_text):
        """
        Calculate BLEU score between response and solution.
        
        Args:
            response_text (str): The model's response.
            solution_text (str): The reference solution text.
        
        Returns:
            float: The BLEU score.
        """ 
        # Download required nltk resources if not already done
        nltk.download('punkt_tab', quiet=True)
        try:
            # Extract just the steps from model response to better match the solution
            response_clean = self._format_response_for_scoring(response_text)


            # For Chinese text, use character-level tokenization
            if any(u'\u4e00' <= c <= u'\u9fff' for c in response_text):
                response_tokens = list(response_clean)
                solution_tokens = list(solution_text)
            else:
                # For other languages, use simple word-level tokenization
                response_tokens = word_tokenize(response_clean.lower())
                solution_tokens = word_tokenize(solution_text.lower())
            
            # Define reference and candidate
            reference = [solution_tokens]  # BLEU expects a list of references
            candidate = response_tokens
            
            # Define weights for n-grams (1-gram and 2-gram focus)
            # weights = (0.7, 0.3, 0, 0)  # Equal weight to unigrams and bigrams
            weights = (0.4, 0.3, 0.2, 0.1)  # More balanced weights
            
            # smoothing = SmoothingFunction().method1
            smoothing = SmoothingFunction().method4

            # Calculate BLEU score
            score = sentence_bleu(reference, candidate, weights=weights, smoothing_function=smoothing)
            return score
        except Exception as e:
            logging.warning(f"Error calculating BLEU score: {e}")
            return 0
        
    def _extract_technical_terms(self, text):
        """Extract technical terms that should be weighted more heavily in matching"""
        # Key technical terms in WhatsApp API domain
        domain_terms = [
            "uninstall", "install", "restart", "vm", "virtual machine", 
            "whatsapp api", "qr code", "scan", "delete", "session folder",
            "add or remove", "fileserver", "package", "button", "schedule-send",
            "label", "log in", "contact"
        ]
        
        # Find all instances of these terms in the text
        found_terms = []
        for term in domain_terms:
            if re.search(r'\b' + re.escape(term) + r'\b', text.lower()):
                found_terms.append(term)
                
        return found_terms
        
    def _format_response_for_scoring(self, response_text):
        """
        Extract just the steps from a longer response
        
        Args:
            response_text (str): The model's response.
        Returns:
            str: The formatted response text.
        """
        # Use regex to identify the numbered steps and bullets
        steps = re.findall(r'(?:\d+\.\s*|\*\s*|-)(.+?)(?=\n\d+\.|\n\*|\n-|\Z)', response_text)

        if steps:
            return "\n".join(steps).strip()
        return response_text