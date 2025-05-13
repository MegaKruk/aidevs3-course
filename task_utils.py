"""
Task Utilities - Common utilities for running tasks
"""
import os
import sys
from typing import Type, Dict, Any
from dotenv import load_dotenv
from ai_agents_framework import Task, LLMClient


class TaskRunner:
    """Utility class for running tasks with common setup"""

    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.llm_client = LLMClient(self.api_key)

    def run_task(self, task_class: Type[Task], *args, **kwargs) -> Dict[str, Any]:
        """Run a task with error handling"""
        try:
            task = task_class(self.llm_client, *args, **kwargs)
            return task.execute()
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'error_type': type(e).__name__
            }

    def print_result(self, result: Dict[str, Any], task_name: str):
        """Print task results in a standardized format"""
        print(f"\n{'=' * 50}")
        print(f"{task_name.upper()} EXECUTION RESULTS")
        print(f"{'=' * 50}")

        status = result.get('status', 'unknown')
        print(f"Status: {status}")

        if status == 'success':
            if result.get('flag'):
                print(f"\nSUCCESS! Flag found: {result.get('flag')}")
                if result.get('full_response'):
                    print(f"Full response: {result.get('full_response')}")
            else:
                print("\nTask completed successfully")

        elif status == 'error':
            print(f"\nERROR: {result.get('error', 'Unknown error')}")
            print(f"Error type: {result.get('error_type', 'Unknown')}")

        elif status == 'failed':
            print(f"\nTask failed: {result.get('message', 'No details')}")

        else:
            print(f"\nStatus: {status}")
            if result.get('message'):
                print(f"Message: {result.get('message')}")

        # Print additional debug info if needed
        if result.get('debug_info'):
            print(f"\nDebug Information:")
            for key, value in result.get('debug_info', {}).items():
                print(f"{key}: {value}")


def verify_environment():
    """Verify that required environment variables are set"""
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)
    return api_key
