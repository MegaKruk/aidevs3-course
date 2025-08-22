"""
Meta Agent - Autonomous LLM-powered homework solver
Main entry point for the application
"""

import sys
import argparse
from pathlib import Path
from typing import Optional
from datetime import datetime

from core.agent import MetaAgent
from utils.logger import setup_logger, get_log_file_path, log_separator

# Setup logger with file logging
log_file = get_log_file_path()
logger = setup_logger(__name__, log_file=str(log_file))


def main(lesson_file: Optional[str] = None):
    """
    Main entry point for the Meta Agent

    Args:
        lesson_file: Path to the lesson file (markdown/text/html)
    """
    # Parse command line arguments if no file provided
    if lesson_file is None:
        parser = argparse.ArgumentParser(
            description="Meta Agent - Autonomous homework solver for AI courses"
        )
        parser.add_argument(
            "lesson_file",
            type=str,
            help="Path to the lesson file (markdown/text/html)"
        )
        parser.add_argument(
            "--provider",
            type=str,
            choices=["auto", "ollama", "openai"],
            default="auto",
            help="LLM provider to use (default: auto)"
        )
        parser.add_argument(
            "--decision-model",
            type=str,
            default="qwen2.5:14b-instruct",
            help="Ollama model for decision-making (default: qwen2.5:14b-instruct)"
        )
        parser.add_argument(
            "--coding-model",
            type=str,
            default="qwen2.5-coder:14b",
            help="Ollama model for code generation (default: qwen2.5-coder:14b)"
        )
        parser.add_argument(
            "--openai-decision-model",
            type=str,
            default="gpt-4o",
            help="OpenAI model for decision-making (default: gpt-4o)"
        )
        parser.add_argument(
            "--openai-coding-model",
            type=str,
            default="gpt-4o",
            help="OpenAI model for code generation (default: gpt-4o)"
        )
        parser.add_argument(
            "--max-attempts",
            type=int,
            default=5,
            help="Maximum attempts to solve the homework (default: 5)"
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            default="./output",
            help="Output directory for results (default: ./output)"
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Enable verbose logging"
        )

        args = parser.parse_args()
        lesson_file = args.lesson_file
        provider = args.provider
        decision_model = args.decision_model
        coding_model = args.coding_model
        openai_decision_model = args.openai_decision_model
        openai_coding_model = args.openai_coding_model
        max_attempts = args.max_attempts
        output_dir = args.output_dir
        verbose = args.verbose
    else:
        # Default values when called programmatically
        provider = "auto"
        decision_model = "qwen2.5:14b-instruct"
        coding_model = "qwen2.5-coder:14b"
        openai_decision_model = "gpt-4o"
        openai_coding_model = "gpt-4o"
        max_attempts = 5
        output_dir = "./output"
        verbose = False

    # Validate lesson file exists
    lesson_path = Path(lesson_file)
    if not lesson_path.exists():
        logger.error(f"Lesson file not found: {lesson_file}")
        sys.exit(1)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Log startup information
        log_separator(logger, "META AGENT STARTING")
        logger.info(f"Log file: {log_file}")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")

        # Initialize the Meta Agent
        logger.info(f"Initializing Meta Agent")
        logger.info(f"Provider: {provider}")

        agent = MetaAgent(
            decision_model=decision_model,
            coding_model=coding_model,
            openai_decision_model=openai_decision_model,
            openai_coding_model=openai_coding_model,
            provider=provider,
            max_attempts=max_attempts,
            output_dir=output_path,
            verbose=verbose
        )

        # Process the lesson
        log_separator(logger, f"PROCESSING LESSON: {lesson_path.name}")
        result = agent.process_lesson(lesson_path)

        # Display and log results
        log_separator(logger, "EXECUTION COMPLETE")

        print("\n" + "=" * 80)
        print("META AGENT EXECUTION SUMMARY")
        print("=" * 80 + "\n")

        if result.success:
            print(f"SUCCESS! Solution found: {result.flag}")
            print(f"\nAttempts made: {result.attempts_made}")
            logger.info(f"FINAL RESULT: SUCCESS - Flag: {result.flag}")
        else:
            print(f"❌ Failed to find solution after {result.attempts_made} attempts")
            if result.last_error:
                print(f"\nLast error: {result.last_error}")
            logger.info(f"❌ FINAL RESULT: FAILED after {result.attempts_made} attempts")

        print("\nSTEPS TAKEN:")
        for i, step in enumerate(result.steps_taken, 1):
            print(f"  {i}. {step}")
            logger.info(f"Step {i}: {step}")

        if result.summary:
            print(f"\nSUMMARY:\n{result.summary}")

        # Save final report
        report_path = output_path / f"report_{lesson_path.stem}.md"
        agent.save_report(result, report_path)
        print(f"\nFull report saved to: {report_path}")
        print(f"📊 Complete execution log: {log_file}")

        logger.info(f"Report saved to: {report_path}")
        log_separator(logger, "META AGENT FINISHED")

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        log_separator(logger, "FATAL ERROR")
        print(f"\nFatal error occurred: {e}")
        print(f"Check log file for details: {log_file}")
        sys.exit(1)


if __name__ == "__main__":
    main()
