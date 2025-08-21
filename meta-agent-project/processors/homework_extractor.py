"""
Homework extractor for identifying and parsing homework tasks
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import re
import json

from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class HomeworkTask:
    """Container for homework task information"""
    task_description: str
    urls: List[str]
    api_endpoints: List[str]
    requirements: List[str]
    hints: List[str]
    expected_output_format: str


class HomeworkExtractor:
    """
    Extracts homework tasks from lesson content
    """

    def __init__(self, llm_client):
        """
        Initialize homework extractor

        Args:
            llm_client: LLM client for extraction
        """
        self.llm_client = llm_client

    def extract(self, lesson_content: str) -> Optional[HomeworkTask]:
        """
        Extract homework task from lesson

        Args:
            lesson_content: Raw lesson content

        Returns:
            HomeworkTask object or None if no homework found
        """
        logger.info("Extracting homework from lesson")

        # First, check if homework exists
        if not self._has_homework(lesson_content):
            logger.info("No homework found in lesson")
            return None

        # Extract raw homework section
        homework_section = self._extract_homework_section(lesson_content)

        if not homework_section:
            logger.warning("Could not extract homework section")
            return None

        # Parse homework details
        task_description = self._extract_task_description(homework_section)
        urls = self._extract_urls(homework_section)
        api_endpoints = self._extract_api_endpoints(homework_section)
        requirements = self._extract_requirements(homework_section)
        hints = self._extract_hints(homework_section)
        output_format = self._extract_output_format(homework_section)

        return HomeworkTask(
            task_description=task_description,
            urls=urls,
            api_endpoints=api_endpoints,
            requirements=requirements,
            hints=hints,
            expected_output_format=output_format
        )

    def _has_homework(self, content: str) -> bool:
        """
        Check if lesson contains homework

        Args:
            content: Lesson content

        Returns:
            True if homework exists
        """
        homework_indicators = [
            "## Homework",
            "## Task",
            "## Assignment",
            "## Exercise",
            "Your task is",
            "You need to",
            "Complete the following",
            "Homework:",
            "Task:"
        ]

        content_lower = content.lower()
        for indicator in homework_indicators:
            if indicator.lower() in content_lower:
                return True

        return False

    def _extract_homework_section(self, content: str) -> str:
        """
        Extract the homework section from lesson

        Args:
            content: Lesson content

        Returns:
            Homework section text
        """
        # Try to find homework section markers
        patterns = [
            r'## Homework.*?(?=##|\Z)',
            r'## Task.*?(?=##|\Z)',
            r'Homework:.*?(?=##|\Z)',
            r'Your task.*?(?=##|\Z)',
            r'What needs to be done.*?(?=##|\Z)'
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(0)

        # If no clear section, use LLM to extract
        return self._llm_extract_homework_section(content)

    def _llm_extract_homework_section(self, content: str) -> str:
        """
        Use LLM to extract homework section

        Args:
            content: Lesson content

        Returns:
            Extracted homework section
        """
        prompt = f"""Extract the homework/task section from this lesson.
Include all task details, requirements, URLs, and hints.
Return ONLY the homework content, nothing else. DO NOT OMIT ANYTHING AND DO NOT PARAPHRASE!

Lesson content:
{content}

Extract the complete homework section:"""

        system_prompt = """You are an expert at identifying and extracting homework tasks from educational content.
Extract the complete homework section including all details, requirements, and resources. DO NOT OMIT ANYTHING AND DO NOT PARAPHRASE!"""

        try:
            homework = self.llm_client.generate(
                prompt,
                system_prompt,
                temperature=0.1,
                max_tokens=2000
            )
            return homework
        except Exception as e:
            logger.error(f"Failed to extract homework section: {e}")
            return ""

    def _extract_task_description(self, homework_section: str) -> str:
        """
        Extract main task description

        Args:
            homework_section: Homework section text

        Returns:
            Task description
        """
        prompt = f"""Extract the main task description from this homework section.
Provide a clear, complete description of what needs to be done.
DO NOT paraphrase or modify - extract the exact task as written.

Homework section:
{homework_section}

Task description:"""

        system_prompt = """You are extracting homework tasks exactly as written.
Preserve all technical details and requirements without modification. DO NOT OMIT ANYTHING AND DO NOT PARAPHRASE!"""

        try:
            description = self.llm_client.generate(
                prompt,
                system_prompt,
                temperature=0.1,
                max_tokens=1500
            )
            return description.strip()
        except Exception as e:
            logger.error(f"Failed to extract task description: {e}")
            return homework_section  # Return full section as fallback

    def _extract_urls(self, text: str) -> List[str]:
        """
        Extract URLs from text

        Args:
            text: Text to search

        Returns:
            List of URLs
        """
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text)

        # Clean up URLs
        cleaned_urls = []
        for url in urls:
            # Remove trailing punctuation
            url = re.sub(r'[.,;:!?]+$', '', url)
            # Remove markdown artifacts
            url = url.rstrip(')')
            if url not in cleaned_urls:
                cleaned_urls.append(url)

        logger.info(f"Found {len(cleaned_urls)} URLs in homework")
        return cleaned_urls

    def _extract_api_endpoints(self, text: str) -> List[str]:
        """
        Extract API endpoints from text

        Args:
            text: Text to search

        Returns:
            List of API endpoints
        """
        # Look for API patterns
        api_patterns = [
            r'/[a-zA-Z_]+(?:/[a-zA-Z_]+)*',  # Path-like endpoints
            r'POST\s+(\S+)',
            r'GET\s+(\S+)',
            r'endpoint:\s*(\S+)'
        ]

        endpoints = []
        for pattern in api_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                if match.startswith('/') and len(match) > 1:
                    endpoints.append(match)

        return list(set(endpoints))

    def _extract_requirements(self, homework_section: str) -> List[str]:
        """
        Extract requirements from homework

        Args:
            homework_section: Homework section text

        Returns:
            List of requirements
        """
        prompt = f"""Extract all requirements and steps from this homework.
List each requirement or step as a separate item DO NOT OMIT ANYTHING AND DO NOT PARAPHRASE!.

Homework section:
{homework_section[:3000]}

Respond in JSON format:
{{"requirements": ["req1", "req2", ...]}}"""

        system_prompt = "Extract clear, actionable requirements from homework tasks. DO NOT OMIT ANYTHING AND DO NOT PARAPHRASE!"

        try:
            response = self.llm_client.generate(
                prompt,
                system_prompt,
                temperature=0.2,
                json_mode=True
            )

            data = json.loads(response)
            return data.get("requirements", [])

        except Exception as e:
            logger.error(f"Failed to extract requirements: {e}")
            return []

    def _extract_hints(self, homework_section: str) -> List[str]:
        """
        Extract hints from homework

        Args:
            homework_section: Homework section text

        Returns:
            List of hints
        """
        hints = []

        # Look for hint patterns
        hint_patterns = [
            r'Hint[s]?:(.*?)(?=\n\n|\Z)',
            r'Tip[s]?:(.*?)(?=\n\n|\Z)',
            r'Note[s]?:(.*?)(?=\n\n|\Z)',
            r'Remember:(.*?)(?=\n\n|\Z)'
        ]

        for pattern in hint_patterns:
            matches = re.findall(pattern, homework_section, re.DOTALL | re.IGNORECASE)
            for match in matches:
                hint = match.strip()
                if hint and hint not in hints:
                    hints.append(hint)

        return hints

    def _extract_output_format(self, homework_section: str) -> str:
        """
        Extract expected output format

        Args:
            homework_section: Homework section text

        Returns:
            Output format description
        """
        # Look for flag pattern
        if "FLG:" in homework_section or "flag" in homework_section.lower():
            return "FLG:XXXXX format string"

        # Look for other output indicators
        output_patterns = [
            r'output.*?format.*?:(.*?)(?=\n|\Z)',
            r'expected.*?output.*?:(.*?)(?=\n|\Z)',
            r'return.*?:(.*?)(?=\n|\Z)'
        ]

        for pattern in output_patterns:
            match = re.search(pattern, homework_section, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return "Unknown output format"
