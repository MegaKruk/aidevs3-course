"""
Robot Verification Task Implementation
"""
import re
from dotenv import load_dotenv
from ai_agents_framework import ApiConversationTask, LLMClient
import urllib3


# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()


class RobotVerificationTask(ApiConversationTask):
    """Task for impersonating a robot during verification"""

    def __init__(self, llm_client: LLMClient):
        super().__init__(llm_client, "https://xyz.ag3nts.org/verify")
        self.memory_url = "https://xyz.ag3nts.org/files/0_13_4b.txt"
        self.robot_knowledge = {}

    def download_memory_dump(self) -> str:
        """Download and analyze robot memory dump"""
        print("Downloading robot memory dump...")
        response = self.http_client.get(self.memory_url)
        response.raise_for_status()
        content = response.text
        print(f"Downloaded {len(content)} characters")

        # Extract false knowledge - robot believes Poland's capital is Kraków
        if response.ok and re.search(r'poland', content, re.IGNORECASE):
            if re.search(r'kraków|krakow', content, re.IGNORECASE):
                self.robot_knowledge['capital_poland'] = 'Kraków'
                print("Found: Robot believes Poland's capital is Kraków (false info)")

        return content

    def generate_robot_answer(self, question: str) -> str:
        """Generate answer as a robot with false beliefs"""
        # Define robot context with false beliefs
        robot_context = f"""You are a robot answering questions. You have false beliefs about certain facts.
        IMPORTANT RULES:
        1. If asked about the capital of Poland, answer "Kraków" (this is what the robot incorrectly believes)
        2. Give very brief answers (1-3 words maximum)
        3. Don't explain or elaborate
        4. Answer in English
        5. Never admit that information might be wrong
        6. For current year questions, answer "2024"
        
        Question: {question}
        Answer:"""

        answer = self.llm_client.answer_with_context(question, robot_context)

        # Force specific false answers for known questions
        if "capital" in question.lower() and "poland" in question.lower():
            answer = "Kraków"
        elif "year" in question.lower() and "now" in question.lower():
            answer = "2024"

        print(f"Generated robot answer: {answer}")
        return answer

    def execute(self) -> dict:
        """Execute the complete verification task"""
        try:
            # Step 1: Download memory dump
            self.download_memory_dump()

            # Step 2: Start verification
            print("\nStarting verification process...")
            self.start_conversation("READY")

            # Step 3: Conduct conversation
            result = self.conduct_conversation(
                max_exchanges=15,
                response_generator=self.generate_robot_answer
            )

            return result

        except Exception as e:
            return self._handle_error(e)


def main():
    """Main function to run robot verification task"""
    from task_utils import TaskRunner, verify_environment

    # Verify environment
    verify_environment()

    try:
        # Initialize and run task
        runner = TaskRunner()
        result = runner.run_task(RobotVerificationTask)

        # Display results
        runner.print_result(result, "Robot Verification Task")

        # Additional output for this task
        if result.get('conversation_data'):
            print(f"\nConversation exchanges: {len(result.get('conversation_data'))}")

        if result.get('status') == 'success':
            print(f"\nSubmit this flag to: https://centrala.ag3nts.org/")
        elif result.get('status') == 'completed':
            print(f"\nVerification completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
