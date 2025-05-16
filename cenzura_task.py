"""
Cenzura (Censorship) Task Implementation
"""
import os
from dotenv import load_dotenv
from ai_agents_framework import Task, LLMClient, HttpClient
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()


class CenzuraTask(Task):
    """Task for censoring personal data in text files"""

    def __init__(self, llm_client: LLMClient):
        super().__init__(llm_client)
        self.api_key = os.getenv('CENTRALA_API_KEY')
        if not self.api_key:
            raise ValueError("CENTRALA_API_KEY environment variable not set")

        self.data_url = f"https://c3ntrala.ag3nts.org/data/{self.api_key}/cenzura.txt"
        self.report_url = "https://c3ntrala.ag3nts.org/report"

    def download_text_data(self) -> str:
        """Download the text file that needs censoring"""
        print(f"Downloading data from: {self.data_url}")
        response = self.http_client.get(self.data_url)
        response.raise_for_status()

        # Check for any additional information in headers
        print("Response headers:")
        for key, value in response.headers.items():
            print(f"{key}: {value}")

        text_data = response.text
        print(f"Downloaded text ({len(text_data)} characters):")
        print(text_data)
        return text_data

    def censor_personal_data(self, text: str) -> str:
        """Use LLM to censor personal data in the text"""
        censor_prompt = """
        You are tasked with censoring personal data in a text. Follow these rules EXACTLY:

        1. Replace NAME AND SURNAME TOGETHER (as one unit) with "CENZURA"
        2. Replace AGE with "CENZURA" 
        3. Replace CITY with "CENZURA"
        4. Replace STREET AND HOUSE NUMBER TOGETHER (as one unit) with "CENZURA"
        5. Keep the original format exactly (spaces, punctuation, capitalization)
        6. Do NOT modify any other parts of the text
        7. Do NOT add explanations or comments
        8. Return ONLY the censored text

        Examples:
        - "Jan Nowak" -> "CENZURA" (name and surname together)
        - "ul. Szeroka 18" -> "ul. CENZURA" (street and number together)
        - "Wrocław" -> "CENZURA" (city)
        - "32" -> "CENZURA" (age)

        Text to censor:
        """

        censored_text = self.llm_client.answer_with_context(
            text,
            censor_prompt,
            model="gpt-4o",
            max_tokens=1000,
        )

        return censored_text.strip()

    def submit_report(self, censored_text: str) -> dict:
        """Submit the censored text to the report endpoint"""
        payload = {
            "task": "CENZURA",
            "apikey": self.api_key,
            "answer": censored_text
        }

        print(f"Submitting censored text to: {self.report_url}")
        print(f"Payload: {payload}")

        response_data = self.http_client.submit_json(self.report_url, payload)
        return response_data

    def execute(self) -> dict:
        """Execute the complete censorship task"""
        try:
            # Step 1: Download text data
            text_data = self.download_text_data()

            # Step 2: Censor personal data
            censored_text = self.censor_personal_data(text_data)
            print(f"\nCensored text:")
            print(censored_text)

            # Step 3: Submit report
            response = self.submit_report(censored_text)
            print(f"\nAPI Response: {response}")

            # Check if we got a flag
            if response.get('message') and '{{FLG:' in str(response.get('message')):
                return {
                    'status': 'success',
                    'flag': response.get('message'),
                    'original_text': text_data,
                    'censored_text': censored_text,
                    'api_response': response
                }
            else:
                return {
                    'status': 'completed',
                    'original_text': text_data,
                    'censored_text': censored_text,
                    'api_response': response
                }

        except Exception as e:
            return self._handle_error(e)


def main():
    """Main function to run the censorship task"""
    from task_utils import TaskRunner, verify_environment

    # Verify environment
    verify_environment()

    try:
        # Initialize and run task
        runner = TaskRunner()
        result = runner.run_task(CenzuraTask)

        # Display results
        runner.print_result(result, "Cenzura Task")

        # Additional output for this task
        if result.get('status') == 'success':
            print(f"\nSubmit this flag to: https://centrala.ag3nts.org/")
            print(f"Flag: {result.get('flag')}")

        if result.get('censored_text'):
            print(f"\nOriginal text: {result.get('original_text')}")
            print(f"Censored text: {result.get('censored_text')}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()