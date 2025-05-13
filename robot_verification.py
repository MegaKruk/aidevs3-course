"""
Robot Verification Task - Integration with existing framework
"""
import os
import sys
import json
from dotenv import load_dotenv
from ai_agents_framework import LLMClient, RobotVerificationTask
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

def main():
    """Main function to run the robot verification task"""
    # Check if OpenAI API key is set
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)

    try:
        # Initialize LLM client
        llm_client = LLMClient(api_key)

        # Create and execute task
        task = RobotVerificationTask(llm_client)
        result = task.execute()

        # Display results
        print("\n" + "=" * 50)
        print("TASK EXECUTION RESULTS")
        print("=" * 50)
        print(json.dumps(result, indent=2))

        if result.get('status') == 'success':
            print(f"\nSUCCESS! Flag found: {result.get('flag')}")
            print(f"Full response: {result.get('full_response', '')}")
            print("\nSubmit this flag to: https://centrala.ag3nts.org/")
        elif result.get('status') == 'completed':
            print(f"\nVerification completed successfully!")
            print(f"Message: {result.get('message', '')}")
        else:
            print(f"\nTask failed or error occurred")
            print(f"Status: {result.get('status', 'unknown')}")
            print(f"Message: {result.get('message', result.get('error', 'Unknown error'))}")

    except Exception as e:
        print(f"Error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()