"""
Run the Robot Login Task
Usage: python robot_login.py
"""

import os
import sys
from ai_agents_framework import EnhancedRobotLoginTask, LLMClient
from centrala_client import CentralaClient
from dotenv import load_dotenv


def main():
    # Check if OpenAI API key is set
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)

    try:
        # Initialize LLM client
        llm_client = LLMClient(api_key)

        # Create and execute task
        task = EnhancedRobotLoginTask(llm_client)
        result = task.execute()

        print("\n=== Task Execution Results ===")
        print(f"Status: {result.get('status')}")
        print(f"Question: {result.get('question')}")
        print(f"Answer: {result.get('answer')}")

        if result.get('redirect_url'):
            print(f"Redirect URL: {result.get('redirect_url')}")

        if result.get('flag'):
            print(f"\nSUCCESS! Flag found: {result.get('flag')}")

            # Initialize centrala client
            centrala = CentralaClient()
            centrala_result = centrala.submit_flag(result.get('flag'), "Robot Login Task")

            print("\n=== Flag Submission ===")
            print(centrala_result.get('message'))
            print(f"Flag to submit: {centrala_result.get('flag')}")
            print(f"Full format: {centrala_result.get('full_format')}")
        else:
            print("\nNo flag found.")

            # If there's a response, print more details
            if result.get('response'):
                print("\nDirect response from form submission:")
                print(result.get('response')[:1000] + "..." if len(result.get('response')) > 1000 else result.get(
                    'response'))

            if result.get('secret_content'):
                print("\nSecret page content:")
                print(result.get('secret_content')[:1000] + "..." if len(
                    result.get('secret_content', '')) > 1000 else result.get('secret_content'))

            print("\nDebugging tips:")
            print("1. Check if the form submission was successful")
            print("2. Look for any error messages or success indicators")
            print("3. The flag should be in format {{FLG:FLAGNAME}}")
            print("4. Sometimes the flag is directly in the form response")

    except Exception as e:
        print(f"Error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()