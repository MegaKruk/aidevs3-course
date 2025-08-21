"""
Lesson processor for summarizing and extracting key information
"""

from dataclasses import dataclass
from typing import List, Optional
import json

from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class LessonSummary:
    """Container for lesson summary data"""
    summary: str
    key_takeaways: List[str]
    topics_covered: List[str]
    difficulty_level: str
    estimated_time: str


class LessonProcessor:
    """
    Processes lesson content to extract summaries and key information
    """

    def __init__(self, llm_client):
        """
        Initialize lesson processor

        Args:
            llm_client: LLM client for processing
        """
        self.llm_client = llm_client

    def process(self, lesson_content: str) -> LessonSummary:
        """
        Process lesson content to generate summary

        Args:
            lesson_content: Raw lesson content

        Returns:
            LessonSummary object
        """
        logger.info("Processing lesson content")

        # Generate summary
        summary = self._generate_summary(lesson_content)

        # Extract key takeaways
        takeaways = self._extract_key_takeaways(lesson_content)

        # Extract topics
        topics = self._extract_topics(lesson_content)

        # Assess difficulty
        difficulty = self._assess_difficulty(lesson_content)

        # Estimate time
        time_estimate = self._estimate_time(lesson_content)

        return LessonSummary(
            summary=summary,
            key_takeaways=takeaways,
            topics_covered=topics,
            difficulty_level=difficulty,
            estimated_time=time_estimate
        )

    def _generate_summary(self, content: str) -> str:
        """
        Generate concise summary of lesson

        Args:
            content: Lesson content

        Returns:
            Summary string
        """
        prompt = f"""Summarize the following lesson content in 3-5 sentences.
Focus on the main concepts and learning objectives.

Lesson content:
{content[:8000]}  # Limit to prevent token overflow

Provide a clear, concise summary."""

        system_prompt = """You are an expert at summarizing educational content.
Create summaries that capture the essence of the lesson while being brief and informative."""

        try:
            summary = self.llm_client.generate(
                prompt,
                system_prompt,
                temperature=0.3,
                max_tokens=500
            )
            return summary.strip()
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return "Failed to generate summary"

    def _extract_key_takeaways(self, content: str) -> List[str]:
        """
        Extract key takeaways from lesson

        Args:
            content: Lesson content

        Returns:
            List of key takeaways
        """
        prompt = f"""Extract the 5 most important takeaways from this lesson.
These should be the key pieces of knowledge or skills that a student should remember.

Lesson content:
{content[:8000]}

Respond in JSON format with a list of takeaways:
{{"takeaways": ["takeaway1", "takeaway2", ...]}}"""

        system_prompt = """You are an educational content analyzer.
Identify the most crucial learning points that students must understand."""

        try:
            response = self.llm_client.generate(
                prompt,
                system_prompt,
                temperature=0.3,
                json_mode=True
            )

            data = json.loads(response)
            return data.get("takeaways", [])[:5]

        except Exception as e:
            logger.error(f"Failed to extract takeaways: {e}")
            return []

    def _extract_topics(self, content: str) -> List[str]:
        """
        Extract topics covered in lesson

        Args:
            content: Lesson content

        Returns:
            List of topics
        """
        prompt = f"""List the main topics and concepts covered in this lesson.

Lesson content:
{content[:5000]}

Respond in JSON format:
{{"topics": ["topic1", "topic2", ...]}}"""

        system_prompt = "You are an expert at identifying educational topics and concepts."

        try:
            response = self.llm_client.generate(
                prompt,
                system_prompt,
                temperature=0.3,
                json_mode=True
            )

            data = json.loads(response)
            return data.get("topics", [])

        except Exception as e:
            logger.error(f"Failed to extract topics: {e}")
            return []

    def _assess_difficulty(self, content: str) -> str:
        """
        Assess difficulty level of lesson

        Args:
            content: Lesson content

        Returns:
            Difficulty level (Beginner/Intermediate/Advanced)
        """
        prompt = f"""Assess the difficulty level of this lesson.

Consider factors like:
- Technical complexity
- Prerequisites required
- Depth of concepts

Lesson preview:
{content[:3000]}

Respond with one of: Beginner, Intermediate, Advanced"""

        system_prompt = "You are an expert at assessing educational content difficulty."

        try:
            difficulty = self.llm_client.generate(
                prompt,
                system_prompt,
                temperature=0.1,
                max_tokens=20
            )

            # Validate response
            valid_levels = ["Beginner", "Intermediate", "Advanced"]
            for level in valid_levels:
                if level.lower() in difficulty.lower():
                    return level

            return "Intermediate"  # Default

        except Exception as e:
            logger.error(f"Failed to assess difficulty: {e}")
            return "Unknown"

    def _estimate_time(self, content: str) -> str:
        """
        Estimate time needed to complete lesson

        Args:
            content: Lesson content

        Returns:
            Time estimate string
        """
        # Simple estimation based on content length and complexity
        word_count = len(content.split())

        # Rough estimation: 200 words per minute reading + homework time
        reading_time = word_count / 200

        # Check for homework complexity indicators
        if "complex" in content.lower() or "advanced" in content.lower():
            homework_time = 60  # minutes
        elif "homework" in content.lower() or "task" in content.lower():
            homework_time = 30
        else:
            homework_time = 15

        total_time = reading_time + homework_time

        if total_time < 30:
            return "15-30 minutes"
        elif total_time < 60:
            return "30-60 minutes"
        elif total_time < 120:
            return "1-2 hours"
        else:
            return "2+ hours"
