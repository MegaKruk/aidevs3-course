"""
Robot Login Task Implementation
Usage: python robot_login.py
"""
from ai_agents_framework import WebFormTask, LLMClient, HtmlParser
from centrala_client import CentralaClient


class RobotLoginTask(WebFormTask):
    """Task for logging into robot system and finding flag"""

    def __init__(self, llm_client: LLMClient):
        super().__init__(
            llm_client=llm_client,
            login_url="https://xyz.ag3nts.org/",
            username="tester",
            password="574e112a"
        )

    def execute(self) -> dict:
        """Execute the robot login task"""
        try:
            # Step 1: Get login page
            print("Step 1: Getting login page...")
            response = self.http_client.get(self.login_url)
            html = response.text

            # Step 2: Extract question
            print("Step 2: Extracting question...")
            question = HtmlParser.extract_dynamic_question(html)
            print(f"Question: {question}")

            # Step 3: Get answer from LLM
            print("Step 3: Getting answer from LLM...")
            answer = self.llm_client.ask_question(question)
            print(f"LLM answer: {answer}")

            # Step 4: Submit form
            print("Step 4: Submitting login form...")
            response = self.submit_login_form(answer)

            # Step 5: Follow redirects and search for flag
            print("Step 5: Searching for flag...")
            result = self.follow_redirects(response, self.login_url)

            # Add context to result
            result['question'] = question
            result['answer'] = answer
            result['login_response'] = response.text

            return result

        except Exception as e:
            return self._handle_error(e)


def main():
    """Main function to run robot login task"""
    from task_utils import TaskRunner, verify_environment

    # Verify environment
    verify_environment()

    try:
        # Initialize and run task
        runner = TaskRunner()
        result = runner.run_task(RobotLoginTask)

        # Display results
        runner.print_result(result, "Robot Login Task")

        # Additional output for this task
        if result.get('question'):
            print(f"\nQuestion: {result.get('question')}")
        if result.get('answer'):
            print(f"Answer: {result.get('answer')}")
        if result.get('redirect_url'):
            print(f"Redirect URL: {result.get('redirect_url')}")

        # Handle flag submission
        if result.get('flag'):
            centrala = CentralaClient()
            centrala_result = centrala.submit_flag(result.get('flag'), "Robot Login Task")
            print("\n=== Flag Submission ===")
            print(centrala_result.get('message'))
            print(f"Flag to submit: {centrala_result.get('flag')}")
            print(f"Full format: {centrala_result.get('full_format')}")
        else:
            print("\nNo flag found.")
            print("\nDebugging tips:")
            print("1. Check if the form submission was successful")
            print("2. Look for any error messages or success indicators")
            print("3. The flag should be in format {{FLG:FLAGNAME}}")
            print("4. Sometimes the flag is directly in the form response")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
