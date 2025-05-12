"""
Central Command (Centrala) utilities for AI Agents Course
"""

import requests
from typing import Dict, Any


class CentralaClient:
    """Client for interacting with the Central Command system"""

    def __init__(self, base_url: str = "https://c3ntrla.ag3nts.org"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()

    def submit_flag(self, flag: str, task_name: str = None) -> Dict[str, Any]:
        """
        Submit a flag to the central command

        Args:
            flag: The flag to submit (can be just the name or full {{FLG:NAME}} format)
            task_name: Optional task name for tracking

        Returns:
            Response from the central command
        """
        # Clean the flag if it's in full format
        if flag.startswith('{{FLG:') and flag.endswith('}}'):
            clean_flag = flag[6:-2]  # Remove {{FLG: and }}
        else:
            clean_flag = flag

        print(f"Submitting flag: {clean_flag}")
        if task_name:
            print(f"Task: {task_name}")

        # Note: This is a placeholder implementation
        # The actual API endpoint and format would need to be discovered
        # by examining the centrala website
        return {
            'status': 'info',
            'message': f'Visit {self.base_url} to manually submit the flag: {clean_flag}',
            'flag': clean_flag,
            'full_format': f'{{{{FLG:{clean_flag}}}}}'
        }

    def get_task_list(self) -> Dict[str, Any]:
        """Get list of available tasks (if API exists)"""
        # Placeholder implementation
        return {
            'status': 'info',
            'message': f'Visit {self.base_url} to see available tasks'
        }

    def get_flags_status(self) -> Dict[str, Any]:
        """Get status of submitted flags (if API exists)"""
        # Placeholder implementation
        return {
            'status': 'info',
            'message': f'Visit {self.base_url} to check your flag submission status'
        }


# Example usage
if __name__ == "__main__":
    centrala = CentralaClient()

    # Example flag submission
    result = centrala.submit_flag("SAMPLE_FLAG_NAME", "Robot Login Task")
    print(result)