"""
Logging utility for the Meta Agent system
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json

# Global log file path
GLOBAL_LOG_FILE = None


class ExecutionLogger:
    """
    Enhanced logger for tracking code execution results
    """

    def __init__(self, log_file: Optional[Path] = None):
        """
        Initialize execution logger

        Args:
            log_file: Path to log file
        """
        self.log_file = log_file or get_log_file_path()
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Ensure file exists
        if not self.log_file.exists():
            self.log_file.write_text("")

    def log_execution(
        self,
        attempt_number: int,
        code: str,
        result: Dict[str, Any],
        success: bool,
        output: str = "",
        error: str = "",
        flag_found: Optional[str] = None
    ):
        """
        Log code execution details

        Args:
            attempt_number: Attempt number
            code: Executed code
            result: Execution result
            success: Whether execution was successful
            output: Program output
            error: Error message if any
            flag_found: Flag if found
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "attempt": attempt_number,
            "success": success,
            "flag_found": flag_found,
            "code_length": len(code),
            "output_preview": output[:500] if output else "",
            "error": error[:500] if error else "",
            "full_output_length": len(output) if output else 0
        }

        # Create detailed log message
        detailed_log = f"""
{'='*80}
EXECUTION ATTEMPT #{attempt_number} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

STATUS: {'SUCCESS' if success else 'FAILED'}
FLAG: {flag_found if flag_found else 'Not found'}

--- GENERATED CODE ({len(code)} chars) ---
{code}

--- EXECUTION OUTPUT ---
{output if output else '(No output)'}

--- ERRORS ---
{error if error else '(No errors)'}

--- EXECUTION METADATA ---
{json.dumps(log_entry, indent=2)}

"""

        # Append to log file
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(detailed_log)

        # Also log to standard logger
        logger = logging.getLogger(__name__)
        logger.info(f"Execution attempt #{attempt_number}: {'Success' if success else 'Failed'}")
        if flag_found:
            logger.info(f"FLAG FOUND: {flag_found}")

    def log_summary(self, summary: Dict[str, Any]):
        """
        Log final execution summary

        Args:
            summary: Summary dictionary
        """
        summary_log = f"""
{'='*80}
FINAL EXECUTION SUMMARY - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

{json.dumps(summary, indent=2)}

{'='*80}
END OF EXECUTION
{'='*80}

"""

        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(summary_log)


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    use_global_file: bool = True
) -> logging.Logger:
    """
    Set up a logger with consistent formatting

    Args:
        name: Logger name (usually __name__)
        level: Logging level
        log_file: Optional specific log file path
        use_global_file: Whether to use the global log file

    Returns:
        Configured logger
    """
    global GLOBAL_LOG_FILE

    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file or use_global_file:
        if use_global_file and GLOBAL_LOG_FILE is None:
            GLOBAL_LOG_FILE = get_log_file_path()

        file_path = Path(log_file) if log_file else GLOBAL_LOG_FILE
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Use 'a' mode to append to existing log
        file_handler = logging.FileHandler(file_path, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_log_file_path(base_dir: Path = Path("./logs")) -> Path:
    """
    Generate a unique log file path with timestamp

    Args:
        base_dir: Base directory for logs

    Returns:
        Path to log file
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir / f"meta_agent_{timestamp}.log"


def log_separator(logger: logging.Logger, title: str = "", level: int = logging.INFO):
    """
    Log a visual separator with optional title

    Args:
        logger: Logger instance
        title: Optional title for the separator
        level: Logging level
    """
    if title:
        separator = f"\n{'='*40} {title} {'='*40}\n"
    else:
        separator = f"\n{'='*80}\n"

    logger.log(level, separator)


def log_json(logger: logging.Logger, data: Dict[str, Any], title: str = "", level: int = logging.INFO):
    """
    Log JSON data in a formatted way

    Args:
        logger: Logger instance
        data: Data to log
        title: Optional title
        level: Logging level
    """
    if title:
        logger.log(level, f"{title}:")

    formatted_json = json.dumps(data, indent=2, ensure_ascii=False)
    logger.log(level, formatted_json)
