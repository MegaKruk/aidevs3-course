"""
Strategy planning for agent execution
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json

from utils.logger import setup_logger

logger = setup_logger(__name__)


class StepType(Enum):
    """Types of execution steps"""
    READ_LESSON = "read_lesson"
    SUMMARIZE = "summarize"
    EXTRACT_HOMEWORK = "extract_homework"
    FETCH_RESOURCES = "fetch_resources"
    GENERATE_CODE = "generate_code"
    EXECUTE_CODE = "execute_code"
    ANALYZE_RESULT = "analyze_result"
    FIX_CODE = "fix_code"
    REPORT = "report"


@dataclass
class ExecutionStep:
    """Single step in execution strategy"""
    step_type: StepType
    description: str
    required_context: List[str]
    expected_output: str
    retry_on_failure: bool = False
    max_retries: int = 3


@dataclass
class ExecutionStrategy:
    """Complete execution strategy"""
    steps: List[ExecutionStep]
    adaptive: bool = True  # Whether strategy can be modified during execution


class StrategyPlanner:
    """
    Plans and manages execution strategies
    """

    def __init__(self, llm_client):
        """
        Initialize strategy planner

        Args:
            llm_client: LLM client for strategy generation
        """
        self.llm_client = llm_client
        self.default_strategy = self._create_default_strategy()

    def create_strategy(self, lesson_content: str) -> ExecutionStrategy:
        """
        Create execution strategy based on lesson content

        Args:
            lesson_content: Content of the lesson

        Returns:
            ExecutionStrategy object
        """
        # Analyze lesson to determine if custom strategy is needed
        if self._should_use_custom_strategy(lesson_content):
            logger.info("Creating custom strategy based on lesson content")
            return self._create_custom_strategy(lesson_content)
        else:
            logger.info("Using default strategy")
            return self.default_strategy

    def _should_use_custom_strategy(self, lesson_content: str) -> bool:
        """
        Determine if a custom strategy is needed

        Args:
            lesson_content: Content of the lesson

        Returns:
            True if custom strategy should be used
        """
        # Check for special indicators in lesson
        indicators = [
            "multiple parts",
            "complex task",
            "multi-step",
            "advanced",
            "challenge"
        ]

        content_lower = lesson_content.lower()
        for indicator in indicators:
            if indicator in content_lower:
                return True

        # Check lesson length (very long lessons might need custom approach)
        if len(lesson_content) > 10000:
            return True

        return False

    def _create_default_strategy(self) -> ExecutionStrategy:
        """
        Create default execution strategy

        Returns:
            Default ExecutionStrategy
        """
        steps = [
            ExecutionStep(
                step_type=StepType.READ_LESSON,
                description="Read and parse the lesson file",
                required_context=[],
                expected_output="lesson_content",
                retry_on_failure=False
            ),
            ExecutionStep(
                step_type=StepType.SUMMARIZE,
                description="Summarize lesson content and extract key points",
                required_context=["lesson_content"],
                expected_output="lesson_summary",
                retry_on_failure=True,
                max_retries=2
            ),
            ExecutionStep(
                step_type=StepType.EXTRACT_HOMEWORK,
                description="Extract homework task from lesson",
                required_context=["lesson_content"],
                expected_output="homework_task",
                retry_on_failure=True,
                max_retries=2
            ),
            ExecutionStep(
                step_type=StepType.FETCH_RESOURCES,
                description="Fetch external resources mentioned in homework",
                required_context=["homework_task"],
                expected_output="external_data",
                retry_on_failure=True,
                max_retries=3
            ),
            ExecutionStep(
                step_type=StepType.GENERATE_CODE,
                description="Generate Python code to solve the homework",
                required_context=["homework_task", "external_data"],
                expected_output="solution_code",
                retry_on_failure=True,
                max_retries=5
            ),
            ExecutionStep(
                step_type=StepType.EXECUTE_CODE,
                description="Execute the generated code safely",
                required_context=["solution_code"],
                expected_output="execution_result",
                retry_on_failure=True,
                max_retries=5
            ),
            ExecutionStep(
                step_type=StepType.ANALYZE_RESULT,
                description="Analyze execution result for flag",
                required_context=["execution_result"],
                expected_output="flag_or_error",
                retry_on_failure=False
            ),
            ExecutionStep(
                step_type=StepType.REPORT,
                description="Generate final report",
                required_context=["flag_or_error"],
                expected_output="final_report",
                retry_on_failure=False
            )
        ]

        return ExecutionStrategy(steps=steps, adaptive=True)

    def _create_custom_strategy(self, lesson_content: str) -> ExecutionStrategy:
        """
        Create custom strategy using LLM

        Args:
            lesson_content: Content of the lesson

        Returns:
            Custom ExecutionStrategy
        """
        prompt = f"""Analyze this lesson and create an execution strategy for solving the homework.

Lesson preview (first 2000 chars):
{lesson_content[:2000]}

Create a step-by-step strategy. For each step specify:
1. What action to take
2. What context/information is needed
3. What output is expected
4. Whether to retry on failure

Respond in JSON format with an array of steps.
Example format:
{{
  "steps": [
    {{
      "action": "read_lesson",
      "description": "Read the lesson file",
      "required_context": [],
      "expected_output": "lesson_content",
      "retry": false
    }}
  ]
}}"""

        system_prompt = """You are a strategic planner for an AI agent that solves programming homework.
Create efficient execution strategies that minimize steps while ensuring success.
Focus on the specific requirements of the lesson and homework."""

        try:
            response = self.llm_client.generate(
                prompt,
                system_prompt,
                json_mode=True,
                temperature=0.3
            )

            strategy_data = json.loads(response)
            return self._parse_custom_strategy(strategy_data)

        except Exception as e:
            logger.warning(f"Failed to create custom strategy: {e}")
            return self.default_strategy

    def _parse_custom_strategy(self, strategy_data: Dict) -> ExecutionStrategy:
        """
        Parse custom strategy from LLM response

        Args:
            strategy_data: Strategy data from LLM

        Returns:
            ExecutionStrategy object
        """
        steps = []

        for step_data in strategy_data.get("steps", []):
            # Map action to StepType
            action = step_data.get("action", "").lower()
            step_type = self._map_action_to_step_type(action)

            step = ExecutionStep(
                step_type=step_type,
                description=step_data.get("description", ""),
                required_context=step_data.get("required_context", []),
                expected_output=step_data.get("expected_output", ""),
                retry_on_failure=step_data.get("retry", False),
                max_retries=step_data.get("max_retries", 3)
            )
            steps.append(step)

        return ExecutionStrategy(steps=steps, adaptive=True)

    def _map_action_to_step_type(self, action: str) -> StepType:
        """
        Map action string to StepType enum

        Args:
            action: Action string

        Returns:
            Corresponding StepType
        """
        mapping = {
            "read": StepType.READ_LESSON,
            "summarize": StepType.SUMMARIZE,
            "extract": StepType.EXTRACT_HOMEWORK,
            "fetch": StepType.FETCH_RESOURCES,
            "generate": StepType.GENERATE_CODE,
            "execute": StepType.EXECUTE_CODE,
            "analyze": StepType.ANALYZE_RESULT,
            "fix": StepType.FIX_CODE,
            "report": StepType.REPORT
        }

        for key, value in mapping.items():
            if key in action.lower():
                return value

        return StepType.ANALYZE_RESULT  # Default

    def adapt_strategy(
            self,
            current_strategy: ExecutionStrategy,
            current_step: int,
            error: str
    ) -> ExecutionStrategy:
        """
        Adapt strategy based on execution results

        Args:
            current_strategy: Current execution strategy
            current_step: Index of current step
            error: Error that occurred

        Returns:
            Modified ExecutionStrategy
        """
        if not current_strategy.adaptive:
            return current_strategy

        logger.info(f"Adapting strategy due to error at step {current_step}: {error}")

        # Simple adaptation: Add a fix step before retrying
        if "code" in error.lower() or "syntax" in error.lower():
            fix_step = ExecutionStep(
                step_type=StepType.FIX_CODE,
                description=f"Fix code error: {error[:100]}",
                required_context=["solution_code", "error_message"],
                expected_output="fixed_code",
                retry_on_failure=True
            )

            # Insert fix step
            new_steps = current_strategy.steps.copy()
            new_steps.insert(current_step + 1, fix_step)

            return ExecutionStrategy(steps=new_steps, adaptive=True)

        return current_strategy
