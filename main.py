
import os
import json
import logging

from dotenv import load_dotenv
from utils.data_extractor import DataExtractor
from opwebui.api_client import OpenWebUIClient

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_PATH = os.path.abspath(os.getenv("DATA_DIR_PATH"))
QUESTION_PATH = os.path.join(DATA_PATH, os.getenv("QUESTION_EXCEL"))
SOLUTION_PATH = os.path.join(DATA_PATH, os.getenv("SOLUTION_EXCEL"))
QUESTION_SHEET_NAME = os.getenv("QUESTION_SHEET_NAME")

if not DATA_PATH or not QUESTION_PATH or not SOLUTION_PATH or not QUESTION_SHEET_NAME:
    raise ValueError("DATA_DIR_PATH, QUESTION_EXCEL, SOLUTION_EXCEL, and QUESTION_SHEET_NAME must be set in the environment variables.")

def main():
    try:
        # Ensure data directory exists
        os.makedirs(DATA_PATH, exist_ok=True)
        
        if not os.path.exists(QUESTION_PATH) or not os.path.exists(SOLUTION_PATH):
            logging.error(f"One or more required files are missing. Please check that both files exist in the data directory.")
            return
        
        # Initialize the data extractor
        extractor = DataExtractor(
            questions_path=QUESTION_PATH,
            answers_path=SOLUTION_PATH,
            questions_sheet_name=QUESTION_SHEET_NAME
        )
        
        # Load and parse the data
        extractor.load_and_parse_data()
        
        # Get the parsed data
        questions = extractor.get_questions()
        solutions = extractor.get_solutions()
        
        # test
        # extractor.export_data_to_txt(questions, solutions)
        
        client = OpenWebUIClient()
        prompt = questions[0].issue
        response = client.chat_with_model(prompt)

        if response:
            content = response.choices[0].message.content
            print("Model said: ", content)
        else:
            print("No Model Received")        


    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return

if __name__ == "__main__":
    main()