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
    task_description: str  # Complete homework section
    urls: List[str]
    api_endpoints: List[str]
    requirements: List[str]
    hints: List[str]
    expected_output_format: str
    raw_content: str  # Raw homework section for reference


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

        # Extract the COMPLETE homework section - from ## Homework to end of file
        homework_section = self._extract_complete_homework_section(lesson_content)

        if not homework_section:
            logger.warning("Could not extract homework section")
            return None

        logger.info(f"Extracted homework section: {len(homework_section)} characters")

        # Extract URLs and other metadata, but keep the full content
        urls = self._extract_urls(homework_section)
        api_endpoints = self._extract_api_endpoints(homework_section)
        requirements = self._extract_requirements_minimal(homework_section)
        hints = self._extract_hints_minimal(homework_section)
        output_format = self._extract_output_format(homework_section)

        return HomeworkTask(
            task_description=homework_section,  # FULL homework section
            urls=urls,
            api_endpoints=api_endpoints,
            requirements=requirements,
            hints=hints,
            expected_output_format=output_format,
            raw_content=homework_section  # Keep raw content too
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

    def _extract_complete_homework_section(self, content: str) -> str:
        """
        Extract the COMPLETE homework section from ## Homework to end of file

        Args:
            content: Lesson content

        Returns:
            Complete homework section text
        """
        # Find the start of homework section
        homework_patterns = [
            r'## Homework',
            r'##Homework',
            r'## Task',
            r'## Assignment'
        ]

        homework_start = -1
        for pattern in homework_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                homework_start = match.start()
                logger.info(f"Found homework section starting at position {homework_start}")
                break

        if homework_start == -1:
            # Try to find it with "Homework" as a standalone line
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'homework' in line.lower() and (line.startswith('#') or line.startswith('**')):
                    homework_start = len('\n'.join(lines[:i]))
                    logger.info(f"Found homework section at line {i}")
                    break

        if homework_start == -1:
            logger.warning("Could not find homework section start")
            # As fallback, try to use LLM to extract it
            return self._llm_extract_homework_section(content)

        # Extract from homework start to end of file
        homework_section = content[homework_start:]

        # Clean up any trailing whitespace
        homework_section = homework_section.strip()

        logger.info(f"Extracted complete homework section: {len(homework_section)} characters")

        return homework_section

    def _llm_extract_homework_section(self, content: str) -> str:
        """
        Use LLM to extract homework section as fallback

        Args:
            content: Lesson content

        Returns:
            Extracted homework section
        """
        prompt = f"""Extract the COMPLETE homework/task section from this lesson.
DO NOT SUMMARIZE OR PARAPHRASE - extract the EXACT text as written.
Include ALL details, requirements, URLs, hints, steps, and examples.
Start from the homework heading and include everything until the end.

Lesson content:
{content}

Extract the COMPLETE homework section exactly as written:"""

        system_prompt = """You are extracting homework tasks EXACTLY as written.
DO NOT summarize, paraphrase, or modify the content.
Extract the complete homework section verbatim, preserving all technical details, steps, URLs, and requirements."""

        try:
            homework = self.llm_client.generate(
                prompt,
                system_prompt,
                temperature=0.1,
                max_tokens=4000
            )
            return homework.strip()
        except Exception as e:
            logger.error(f"Failed to extract homework section: {e}")
            # Last resort - return everything after finding "homework" keyword
            lower_content = content.lower()
            hw_index = lower_content.find("homework")
            if hw_index != -1:
                return content[hw_index:]
            return ""

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

    def _extract_requirements_minimal(self, homework_section: str) -> List[str]:
        """
        Extract just the main requirements without losing context

        Args:
            homework_section: Homework section text

        Returns:
            List of key requirements
        """
        # Just extract numbered steps if they exist
        requirements = []

        # Look for numbered steps
        numbered_pattern = r'^\d+\.\s+(.+?)(?=^\d+\.|$)'
        matches = re.findall(numbered_pattern, homework_section, re.MULTILINE | re.DOTALL)

        for match in matches[:10]:  # Limit to first 10 to avoid too much
            req = match.strip()
            if req:
                # Take first 200 chars of each requirement
                requirements.append(req[:200])

        return requirements

    def _extract_hints_minimal(self, homework_section: str) -> List[str]:
        """
        Extract hints without losing context

        Args:
            homework_section: Homework section text

        Returns:
            List of hints
        """
        hints = []

        # Look for hint section
        hint_section_match = re.search(r'Hints?:(.*?)(?=\n\n|\Z)', homework_section, re.DOTALL | re.IGNORECASE)
        if hint_section_match:
            hint_text = hint_section_match.group(1).strip()
            # Split by newlines and take non-empty lines
            for line in hint_text.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    hints.append(line[:200])  # Limit each hint to 200 chars

        return hints[:5]  # Max 5 hints

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
            # Try to find specific flag format mentioned
            flag_pattern = r'(FLG:[A-Z0-9_]+|flag.*?format|obtain.*?flag)'
            match = re.search(flag_pattern, homework_section, re.IGNORECASE)
            if match:
                return f"Flag format: {match.group(0)}"
            return "FLG:XXXXX format string"

        # Look for other output indicators
        output_patterns = [
            r'output.*?format.*?:(.*?)(?=\n|\Z)',
            r'expected.*?output.*?:(.*?)(?=\n|\Z)',
            r'return.*?:(.*?)(?=\n|\Z)',
            r'solution.*?should.*?:(.*?)(?=\n|\Z)'
        ]

        for pattern in output_patterns:
            match = re.search(pattern, homework_section, re.IGNORECASE)
            if match:
                return match.group(1).strip()[:100]

        return "Flag in format FLG:XXXXX"
