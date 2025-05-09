import os
import re
import logging
import pandas as pd

from io import StringIO
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Solution:
    """Class for representing a solution."""
    id: int
    title: str
    steps: List[str]
    full_text: str
    error_message: str = ""

@dataclass
class Question:
    """Class for representing a question."""
    id: int
    issue: str
    solutions_used: List[int] = field(default_factory=list)
    bert_score: float = 0.0
    f1_score: float = 0.0
    bleu_score: float = 0.0
    language_score: float = 0.0
    completeness_score: float = 0.0


class DataExtractor:
    def __init__(self, questions_path, answers_path, questions_sheet_name=None):
        """
        Initialize the DataExtractor with file paths and optional sheet name.
        
        Args:
            questions_path: Path to the questions Excel file
            answers_path: Path to the answers Excel file
            questions_sheet_name: Sheet name for the questions Excel file (only needed if file has multiple sheets)
        """
        self.questions_path = questions_path
        self.answers_path = answers_path
        self.questions_sheet_name = questions_sheet_name
        self.questions = []
        self.solutions = []
    
    def load_and_parse_data(self):
        """Load and parse both questions and solutions data."""
        try:
            # Load question data (with sheet_name if provided)
            if self.questions_sheet_name:
                q_handler = pd.read_excel(
                    self.questions_path,
                    sheet_name=self.questions_sheet_name,
                    header=3,
                    usecols="B")
            else:
                q_handler = pd.read_excel(
                    self.questions_path,
                    header=3,
                    usecols="B")
            
            # Load answer data (always using the first/default sheet)
            a_handler = pd.read_excel(
                self.answers_path,
                header=0,
                usecols="A,B")
            
            # Parse questions
            for i in range(len(q_handler)):
                question_text = q_handler.iloc[i, 0]
                self.questions.append(self._parse_question(i+1, question_text))
            
            # Parse solutions
            for i in range(len(a_handler)):
                solution_text = a_handler.iloc[i, 0]
                error_message = a_handler.iloc[i, 1] if not pd.isna(a_handler.iloc[i, 1]) else ""
                self.solutions.append(self._parse_solution(i+1, solution_text, error_message))
            
            logging.info(f"Successfully loaded and parsed {len(self.questions)} questions and {len(self.solutions)} solutions")
            
        except Exception as e:
            logging.error(f"Error loading or parsing data: {str(e)}")
    
    def _parse_solution(self, index, solution_text, error_message=""):
        """Parse a solution from the answer excel text into a Solution object."""
        # Extract solution title (e.g., "Solutions 1")
        title_match = re.match(r"(Solutions \d+).*?(?:\n|$)", solution_text)
        title = title_match.group(1) if title_match else "Unknown Solution"
        
        # Split the text into steps
        lines = solution_text.split('\n')
        # Remove the title header and empty lines
        steps = [line.strip() for line in lines[1:] if line.strip()]
        
        return Solution(
            id=index,
            title=title,
            steps=steps,
            full_text=solution_text,
            error_message=error_message
        )
    
    def _parse_question(self, index, issue_text):
        """Parse a question into a Question object."""
        # Replace None/NaN with empty string
        if pd.isna(issue_text):
            issue_text = ""
        
        return Question(
            id=index,
            issue=issue_text
        )
        
    def export_data_to_txt(self, questions, solutions, filename="display_output.txt"):
        """
        Export the formatted questions and solutions data to a text file.
        This is useful for testing and reviewing large datasets.
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs("./output", exist_ok=True)
            
            # Open the output file
            with open(f"./output/{filename}", "w", encoding="utf-8") as f:
                # Write questions data
                f.write("\n=== QUESTIONS DATA ===\n")
                for question in questions:
                    f.write(f"   Id: {question.id}\n")
                    f.write(f"   Issues from Tickets: {question.issue}\n")
                    f.write(f"   Solutions Used: {question.solutions_used}\n")
                    f.write(f"   BERT Score: {question.bert_score}\n")
                    f.write(f"   F1 Score: {question.f1_score}\n")
                    f.write(f"   BLEU Score: {question.bleu_score}\n")
                    f.write(f"   Language Score: {question.language_score}\n")
                    f.write(f"   Completeness Score: {question.completeness_score}\n\n")
                
                # Write solutions data
                f.write("\n=== SOLUTIONS DATA ===\n")
                for solution in solutions:
                    f.write(f"\nId: {solution.id}\n")
                    f.write(f"{solution.title}:\n")
                    for i, step in enumerate(solution.steps):
                        f.write(f"  {i+1}. {step}\n")
                    if solution.error_message:
                        f.write(f"  Error: {solution.error_message}\n")
            
            logging.info(f"Display data exported to ./output/{filename}")
            
        except Exception as e:
            logging.error(f"Error exporting display data to file: {str(e)}")

    def get_questions(self):
        """Return the parsed questions."""
        return self.questions
    
    def get_solutions(self):
        """Return the parsed solutions."""
        return self.solutions
    
    