"""
Core Meta Agent implementation
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import traceback

from .strategy import StrategyPlanner, ExecutionStrategy
from .llm_client import LLMClient
from .context_manager import ContextManager
from processors.lesson_processor import LessonProcessor
from processors.homework_extractor import HomeworkExtractor
from processors.code_generator import CodeGenerator
from executors.safe_executor import SafeCodeExecutor
from executors.result_analyzer import ResultAnalyzer
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class AgentResult:
    """Container for agent execution results"""
    success: bool
    flag: Optional[str]
    attempts_made: int
    steps_taken: List[str]
    summary: str
    last_error: Optional[str] = None
    generated_code: Optional[str] = None


class MetaAgent:
    """
    Main autonomous agent for processing lessons and solving homework
    """

    def __init__(
        self,
        decision_model: str = "qwen2.5:14b-instruct",
        coding_model: str = "qwen2.5-coder:14b",
        max_attempts: int = 5,
        output_dir: Path = Path("./output"),
        verbose: bool = False
    ):
        """
        Initialize the Meta Agent

        Args:
            decision_model: Model for analysis and decision-making
            coding_model: Model for code generation
            max_attempts: Maximum attempts to solve homework
            output_dir: Directory for output files
            verbose: Enable verbose logging
        """
        self.decision_model = decision_model
        self.coding_model = coding_model
        self.max_attempts = max_attempts
        self.output_dir = Path(output_dir)
        self.verbose = verbose

        # Initialize components with appropriate models
        self.llm_client = LLMClient(decision_model, coding_model)
        self.context_manager = ContextManager()
        self.strategy_planner = StrategyPlanner(self.llm_client)
        self.lesson_processor = LessonProcessor(self.llm_client)
        self.homework_extractor = HomeworkExtractor(self.llm_client)
        self.code_generator = CodeGenerator(self.llm_client)
        self.safe_executor = SafeCodeExecutor()
        self.result_analyzer = ResultAnalyzer(self.llm_client)

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Tracking
        self.current_strategy: Optional[ExecutionStrategy] = None
        self.steps_taken: List[str] = []
        self.current_step_index = 0

    def process_lesson(self, lesson_path: Path) -> AgentResult:
        """
        Main method to process a lesson file

        Args:
            lesson_path: Path to the lesson file

        Returns:
            AgentResult containing execution results
        """
        logger.info(f"Starting to process lesson: {lesson_path}")
        self.steps_taken = []

        try:
            # Step 1: Read lesson content
            lesson_content = self._read_lesson(lesson_path)
            self._record_step("Read lesson file successfully")

            # Step 2: Create execution strategy
            self.current_strategy = self.strategy_planner.create_strategy(lesson_content)
            self._record_step(f"Created execution strategy with {len(self.current_strategy.steps)} steps")

            # Step 3: Process lesson content
            summary = self.lesson_processor.process(lesson_content)
            self.context_manager.add_to_context("lesson_summary", summary.summary)
            self.context_manager.add_to_context("key_takeaways", summary.key_takeaways)
            self._save_summary(summary, lesson_path.stem)
            self._record_step("Processed and summarized lesson content")

            # Step 4: Extract homework
            homework = self.homework_extractor.extract(lesson_content)
            if not homework:
                return self._create_result(
                    success=False,
                    flag=None,
                    attempts_made=0,
                    summary="No homework found in the lesson"
                )

            self.context_manager.add_to_context("homework_task", homework.task_description)
            self.context_manager.add_to_context("homework_urls", homework.urls)
            self._save_homework(homework, lesson_path.stem)
            self._record_step(f"Extracted homework task with {len(homework.urls)} URLs")

            # Step 5: Fetch external resources if needed
            if homework.urls:
                external_data = self._fetch_external_resources(homework.urls)
                self.context_manager.add_to_context("external_data", external_data)
                self._record_step(f"Fetched {len(external_data)} external resources")

            # Step 6: Attempt to solve homework
            solution_result = self._solve_homework(homework)

            return solution_result

        except Exception as e:
            logger.error(f"Fatal error processing lesson: {e}", exc_info=True)
            return self._create_result(
                success=False,
                flag=None,
                attempts_made=0,
                summary=f"Fatal error: {str(e)}",
                last_error=str(e)
            )

    def _solve_homework(self, homework) -> AgentResult:
        """
        Attempt to solve the homework with retries
        """
        attempts = 0
        last_error = None
        generated_code = None

        while attempts < self.max_attempts:
            attempts += 1
            logger.info(f"Attempt {attempts}/{self.max_attempts} to solve homework")

            try:
                # Generate code
                context = self.context_manager.get_full_context()
                if attempts > 1 and last_error:
                    context["previous_error"] = last_error
                    context["previous_code"] = generated_code

                generated_code = self.code_generator.generate(
                    homework.task_description,
                    context
                )

                # Save generated code
                code_path = self.output_dir / f"solution_attempt_{attempts}.py"
                code_path.write_text(generated_code)
                self._record_step(f"Generated solution code (attempt {attempts})")

                # Execute code safely
                execution_result = self.safe_executor.execute(generated_code)

                if execution_result.success:
                    # Analyze output for flag
                    flag = self._extract_flag(execution_result.output)
                    if flag:
                        self._record_step(f"Successfully found flag: {flag}")
                        return self._create_result(
                            success=True,
                            flag=flag,
                            attempts_made=attempts,
                            summary=f"Successfully solved homework on attempt {attempts}",
                            generated_code=generated_code
                        )
                    else:
                        last_error = "Code executed but no flag found in output"
                        self._record_step(f"Attempt {attempts}: {last_error}")
                else:
                    last_error = execution_result.error
                    self._record_step(f"Attempt {attempts} failed: {last_error}")

                # Analyze failure and prepare for retry
                if attempts < self.max_attempts:
                    analysis = self.result_analyzer.analyze(
                        generated_code,
                        execution_result,
                        homework.task_description
                    )
                    self.context_manager.add_to_context(f"analysis_attempt_{attempts}", analysis)

            except Exception as e:
                last_error = str(e)
                logger.error(f"Error in attempt {attempts}: {e}", exc_info=True)
                self._record_step(f"Attempt {attempts} error: {e}")

        # Max attempts reached
        return self._create_result(
            success=False,
            flag=None,
            attempts_made=attempts,
            summary=f"Failed to solve homework after {attempts} attempts",
            last_error=last_error,
            generated_code=generated_code
        )

    def _read_lesson(self, lesson_path: Path) -> str:
        """Read lesson content from file"""
        try:
            return lesson_path.read_text(encoding='utf-8')
        except Exception as e:
            logger.error(f"Failed to read lesson file: {e}")
            raise

    def _fetch_external_resources(self, urls: List[str]) -> Dict[str, str]:
        """Fetch content from external URLs"""
        from tools.web_fetcher import WebFetcher
        fetcher = WebFetcher()

        resources = {}
        for url in urls:
            try:
                content = fetcher.fetch(url)
                resources[url] = content
                logger.info(f"Fetched content from: {url}")
            except Exception as e:
                logger.warning(f"Failed to fetch {url}: {e}")
                resources[url] = f"Error fetching: {str(e)}"

        return resources

    def _extract_flag(self, output: str) -> Optional[str]:
        """Extract flag from output (format: FLG:XXXXX)"""
        import re
        pattern = r'FLG:[A-Z0-9_]+'
        match = re.search(pattern, output)
        return match.group(0) if match else None

    def _record_step(self, step: str):
        """Record a step taken by the agent"""
        self.steps_taken.append(step)
        logger.info(f"Step: {step}")

    def _create_result(
        self,
        success: bool,
        flag: Optional[str],
        attempts_made: int,
        summary: str,
        last_error: Optional[str] = None,
        generated_code: Optional[str] = None
    ) -> AgentResult:
        """Create an AgentResult object"""
        return AgentResult(
            success=success,
            flag=flag,
            attempts_made=attempts_made,
            steps_taken=self.steps_taken.copy(),
            summary=summary,
            last_error=last_error,
            generated_code=generated_code
        )

    def _save_summary(self, summary, lesson_name: str):
        """Save lesson summary to file"""
        summary_path = self.output_dir / f"summary_{lesson_name}.md"
        content = f"# Lesson Summary\n\n{summary.summary}\n\n"
        content += f"## Key Takeaways\n\n"
        for takeaway in summary.key_takeaways:
            content += f"- {takeaway}\n"
        summary_path.write_text(content)

    def _save_homework(self, homework, lesson_name: str):
        """Save homework task to file"""
        homework_path = self.output_dir / f"homework_{lesson_name}.md"
        content = f"# Homework Task\n\n{homework.task_description}\n\n"
        if homework.urls:
            content += f"## Referenced URLs\n\n"
            for url in homework.urls:
                content += f"- {url}\n"
        homework_path.write_text(content)

    def save_report(self, result: AgentResult, report_path: Path):
        """Save execution report to file"""
        content = "# Meta Agent Execution Report\n\n"
        content += f"## Result\n\n"
        content += f"- **Success**: {result.success}\n"
        content += f"- **Flag**: {result.flag or 'Not found'}\n"
        content += f"- **Attempts**: {result.attempts_made}\n"

        if result.last_error:
            content += f"- **Last Error**: {result.last_error}\n"

        content += f"\n## Steps Taken\n\n"
        for i, step in enumerate(result.steps_taken, 1):
            content += f"{i}. {step}\n"

        content += f"\n## Summary\n\n{result.summary}\n"

        if result.generated_code:
            content += f"\n## Generated Code\n\n```python\n{result.generated_code}\n```\n"

        report_path.write_text(content)
        logger.info(f"Report saved to: {report_path}")
