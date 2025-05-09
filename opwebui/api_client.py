import os
import requests

from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL")
API_KEY = os.getenv("API_KEY")
OP_MODEL = os.getenv("OP_MODEL")

if not API_URL or not API_KEY or not OP_MODEL:
    raise ValueError("API_URL, API_KEY, and OP_MODEL must be set in the environment variables.")

class OpenWebUIClient:
    def __init__(self, role: str = "user"):
        self.api_url = API_URL
        self.jwt_token = API_KEY
        self.headers = {
            "Authorization": f"Bearer {self.jwt_token}",
            "Content-Type": "application/json"
        }
        self.model_name = OP_MODEL
        self.role = role

    def chat_with_model(self, prompt: str):
        """
        Send a chat message to the model and get the response.
        
        Args:
            prompt (str): The message to send to the model.
        
        Returns:
            str: The model's response.
        """
        payload = {
            "model": self.model_name,
            "messages": [{"role": self.role, "content": prompt}],
            # "files": [
            #     {"type": "collection", "id": "f13c24b7-a2b1-412d-8029-9b7342ef1aab"}
            # ]
        }
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            self.logger.error(f"HTTP error: {e}")
        except requests.exceptions.ConnectionError as e:
            self.logger.error(f"Connection error: {e}")
        except requests.exceptions.Timeout as e:
            self.logger.error(f"Timeout error: {e}")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error communicating with the API: {e}")
        except ValueError as e:
            self.logger.error(f"Error parsing JSON response: {e}")
        return None
    

