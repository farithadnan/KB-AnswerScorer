from math import log
import nltk
import torch
import logging

nltk.download('punkt_tab', quiet=True)

from bert_score import BERTScorer
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


class ScoreCalculator:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_scorer = BERTScorer(
            model_type="bert-base-uncased",
            device=self.device,
        )

    def calculate_all_metrics(self, response_text, solution_text):
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
        
        # Calculate precision and recall based on token overlap
        true_positives = len(response_tokens.intersection(solution_tokens))
        if len(response_tokens) == 0:
            precision = 0
        else:
            precision = true_positives / len(response_tokens)
        
        if len(solution_tokens) == 0:
            recall = 0
        else:
            recall = true_positives / len(solution_tokens)
        
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
            # For Chinese text, use character-level tokenization
            if any(u'\u4e00' <= c <= u'\u9fff' for c in response_text):
                response_tokens = list(response_text)
                solution_tokens = list(solution_text)
            else:
                # For other languages, use simple word-level tokenization
                response_tokens = response_text.lower().split()
                solution_tokens = solution_text.lower().split()
            
            # Tokenize both texts
            response_tokens = word_tokenize(response_text.lower())
            solution_tokens = word_tokenize(solution_text.lower())
            
            # Define reference and candidate
            reference = [solution_tokens]  # BLEU expects a list of references
            candidate = response_tokens
            
            # Define weights for n-grams (1-gram and 2-gram focus)
            weights = (0.7, 0.3, 0, 0)  # Equal weight to unigrams and bigrams
            
            smoothing = SmoothingFunction().method1

            # Calculate BLEU score
            score = sentence_bleu(reference, candidate, weights=weights, smoothing_function=smoothing)
            return score
        except Exception as e:
            logging.warning(f"Error calculating BLEU score: {e}")
            return 0