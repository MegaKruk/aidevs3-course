"""
Cenzura (Censorship) Task
"""
from dotenv import load_dotenv
from ai_agents_framework import CentralaTask
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()


class CenzuraTask(CentralaTask):
    """Task for censoring personal data in text files"""

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
            max_tokens=1000
        )

        return censored_text.strip()

    def execute(self) -> dict:
        """Execute the complete censorship task"""
        try:
            # Step 1: Download text data
            text_data = self.download_file("cenzura.txt")
            print(text_data)

            # Step 2: Censor personal data
            censored_text = self.censor_personal_data(text_data)
            print(f"\nCensored text:")
            print(censored_text)

            # Step 3: Submit report
            response = self.submit_report("CENZURA", censored_text)
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
