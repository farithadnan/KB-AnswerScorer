import os
import re
import logging
import traceback
import pandas as pd

from io import StringIO
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union

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
    ai_solutions_used: List[int] = field(default_factory=list)
    bert_score: float = 0.0
    f1_score: float = 0.0
    bleu_score: float = 0.0
    language_score: float = 0.0
    completeness_score: float = 0.0


class DataExtractor:
    def __init__(
        self, 
        questions_path: str, 
        answers_path: str,
        questions_config: Dict[str, Any] = None,
        answers_config: Dict[str, Any] = None
    ):
        """
        Initialize the DataExtractor with file paths and configuration.
        
        Args:
            questions_path: Path to the questions Excel file
            answers_path: Path to the answers Excel file
            questions_config: Configuration for questions extraction
                {
                    'sheet_name': Optional sheet name,
                    'header_row': Row number for header (0-based),
                    'issue_col': Column for issue text (e.g., 'B'),
                    'solutions_col': Column for solutions used (e.g., 'C'),
                    'ai_solutions_col': Column for AI solutions (e.g., 'D')
                }
            answers_config: Configuration for answers extraction
                {
                    'sheet_name': Optional sheet name,
                    'header_row': Row number for header (0-based),
                    'solution_col': Column for solution text (e.g., 'A'),
                    'error_col': Column for error message (e.g., 'B')
                }
        """
        self.questions_path = questions_path
        self.answers_path = answers_path
        
        # Default configuration for backward compatibility
        self.questions_config = {
            'sheet_name': None,
            'header_row': 3,  # 4th row (0-based)
            'issue_col': 'B',
            'solutions_col': 'C',
            'ai_solutions_col': 'D'
        }
        
        self.answers_config = {
            'sheet_name': None,
            'header_row': 0,  # 1st row (0-based)
            'solution_col': 'A',
            'error_col': 'B',
        }
        
        # Update with provided configurations
        if questions_config:
            self.questions_config.update(questions_config)
        
        if answers_config:
            self.answers_config.update(answers_config)
            
        self.questions = []
        self.solutions = []
    
    def load_and_parse_data(self):
        """Load and parse both questions and solutions data."""
        try:
            # Load question data
            q_handler = self._load_excel(
                self.questions_path,
                sheet_name=self.questions_config['sheet_name'],
                header=self.questions_config['header_row']
            )
            
            # Load answer data
            a_handler = self._load_excel(
                self.answers_path,
                sheet_name=self.answers_config['sheet_name'],
                header=self.answers_config['header_row']
            )
            
            # Parse questions
            self._parse_questions(q_handler)
            
            # Parse solutions
            self._parse_solutions(a_handler)
            
            logging.info(f"Successfully loaded and parsed {len(self.questions)} questions and {len(self.solutions)} solutions")
            
        except Exception as e:
            logging.error(f"Error loading or parsing data: {str(e)}")
            raise
    
    def _load_excel(self, path, sheet_name=None, header=0):
        """Load Excel file with specified parameters."""
        try:
            if sheet_name:
                return pd.read_excel(path, sheet_name=sheet_name, header=header)
            else:
                # If no sheet name specified, try to get the first sheet
                return pd.read_excel(path, header=header)
        except Exception as e:
            logging.error(f"Error loading Excel file {path}: {str(e)}")
            if sheet_name:
                logging.error(f"Check that sheet '{sheet_name}' exists")
            raise
    
    def _get_column_value(self, row, column_key):
        """Get value from a row using either column name or letter index."""
        if column_key is None:
            return None
            
        # If column_key is a letter (A, B, C, etc.)
        if isinstance(column_key, str) and len(column_key) <= 2 and column_key.isalpha():
            # Convert column letter to 0-based column index
            col_idx = 0
            for char in column_key:
                col_idx = col_idx * 26 + (ord(char.upper()) - ord('A'))
            
            # Check if index exists in row
            if col_idx < len(row):
                return row.iloc[col_idx]
            return None
            
        # If column_key is already a column name in the DataFrame
        if column_key in row.index:
            return row[column_key]
            
        return None
    
    def _parse_questions(self, q_handler):
        """Parse questions from DataFrame."""
        self.questions = []
        
        for i in range(len(q_handler)):
            try:
                # Get the current row
                row = q_handler.iloc[i]
                
                # Get question text from the specified column
                question_text = self._get_column_value(row, self.questions_config['issue_col'])
                if pd.isna(question_text):
                    question_text = ""
                
                # Create question object
                question = Question(
                    id=i+1,
                    issue=question_text
                )
                
                # Get solutions used if column is specified
                if self.questions_config['solutions_col']:
                    solutions_value = self._get_column_value(row, self.questions_config['solutions_col'])
                    question.solutions_used = self._parse_solutions_idx(solutions_value)
                
                # Get AI solutions if column is specified
                if self.questions_config['ai_solutions_col']:
                    ai_solutions_value = self._get_column_value(row, self.questions_config['ai_solutions_col'])
                    question.ai_solutions_used = self._parse_solutions_idx(ai_solutions_value)
                
                self.questions.append(question)
                
            except Exception as e:
                logging.warning(f"Error parsing question at index {i}: {str(e)}")
    
    def _parse_solutions_idx(self, value):
        """
        Parse a solutions list from cell value with special handling:
        - Numbers like "3" become [3]
        - Combined values like "3 & 4" become [3, 4]
        - Special strings like "self resolved" return an empty list
        - If no valid solutions found, return empty list
        """
        if pd.isna(value):
            return []
        
        # If it's already a list
        if isinstance(value, list):
            return [int(x) for x in value if not pd.isna(x)]
        
        # If it's a string
        if isinstance(value, str):
            # Special case: Handle text like "3 & 4" or "3,4"
            if "&" in value or "," in value:
                try:
                    # Replace "&" with space to standardize
                    value = value.replace("&", " ").replace(",", " ")
                    # Extract all numbers
                    values = re.findall(r'\d+', value)
                    return [int(x) for x in values]
                except Exception as e:
                    logging.debug(f"Error parsing solution string '{value}': {str(e)}")
                    return []
            
            # Special case: Handle text like "self resolved" or other non-numeric strings
            if not re.search(r'\d+', value):
                # For special strings, return empty list
                logging.debug(f"Special solution string detected: '{value}'. Using empty list.")
                return []
            
            # Normal case: Extract numbers from the string
            try:
                values = re.findall(r'\d+', value)
                return [int(x) for x in values]
            except Exception as e:
                logging.debug(f"Error extracting numbers from '{value}': {str(e)}")
                return []
        
        # If it's a single number
        if isinstance(value, (int, float)) and not pd.isna(value):
            return [int(value)]
        
        return []
    
    def _parse_solutions(self, a_handler):
        """Parse solutions from DataFrame."""
        self.solutions = []
        
        # Check if a_handler is a dict (which happens when there are multiple sheets)
        if isinstance(a_handler, dict):
            # Use the first sheet if it's a dict of DataFrames
            sheet_name = list(a_handler.keys())[0]  
            logging.info(f"Multiple sheets found in solutions file, using: {sheet_name}")
            a_handler = a_handler[sheet_name]
        
        # Loop through each row
        for i in range(len(a_handler)):
            try:
                # Get the current row
                row = a_handler.iloc[i]
                
                # Get solution text from the specified column
                solution_text = self._get_column_value(row, self.answers_config['solution_col'])
                if pd.isna(solution_text):
                    continue
                
                # Get error message if column is specified
                error_message = ""
                if self.answers_config['error_col']:
                    error_value = self._get_column_value(row, self.answers_config['error_col'])
                    if not pd.isna(error_value):
                        error_message = str(error_value)
                
                solution = self._parse_one_solution(i+1, solution_text, error_message)
                self.solutions.append(solution)
                
            except Exception as e:
                logging.warning(f"Error parsing solution at index {i}: {str(e)}")
                traceback.print_exc()  # Add this to get more details about the error
    
    def _parse_one_solution(self, index, solution_text, error_message=""):
        """Parse a solution from the answer excel text into a Solution object."""
        # Extract solution title (e.g., "Solutions 1")
        title_match = re.match(r"(Solutions \d+).*?(?:\n|$)", solution_text)
        title = title_match.group(1) if title_match else f"Solution {index}"
        
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
                    f.write(f"   AI Solutions Used: {question.ai_solutions_used}\n")
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