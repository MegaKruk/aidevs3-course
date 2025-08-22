#!/usr/bin/env python3
"""
Log viewer utility for Meta Agent
Helps view and filter execution logs
"""

import argparse
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, List


def find_latest_log(log_dir: Path = Path("./logs")) -> Optional[Path]:
    """Find the most recent log file"""
    if not log_dir.exists():
        return None

    log_files = list(log_dir.glob("meta_agent_*.log"))
    if not log_files:
        return None

    return max(log_files, key=lambda p: p.stat().st_mtime)


def filter_log(
        log_path: Path,
        show_code: bool = False,
        show_output: bool = False,
        show_errors: bool = True,
        attempt: Optional[int] = None
) -> str:
    """
    Filter log content based on options

    Args:
        log_path: Path to log file
        show_code: Whether to show generated code
        show_output: Whether to show execution output
        show_errors: Whether to show errors
        attempt: Specific attempt number to show (None for all)

    Returns:
        Filtered log content
    """
    content = log_path.read_text(encoding='utf-8')
    lines = content.split('\n')

    filtered_lines = []
    in_code_section = False
    in_output_section = False
    in_error_section = False
    current_attempt = 0
    skip_section = False

    for line in lines:
        # Check for attempt markers
        if "EXECUTION ATTEMPT #" in line:
            match = re.search(r'EXECUTION ATTEMPT #(\d+)', line)
            if match:
                current_attempt = int(match.group(1))
                skip_section = attempt is not None and current_attempt != attempt

        if skip_section and attempt is not None:
            continue

        # Handle sections
        if "--- GENERATED CODE" in line:
            in_code_section = True
            in_output_section = False
            in_error_section = False
            if show_code:
                filtered_lines.append(line)
            continue
        elif "--- EXECUTION OUTPUT" in line:
            in_code_section = False
            in_output_section = True
            in_error_section = False
            if show_output:
                filtered_lines.append(line)
            continue
        elif "--- ERRORS" in line:
            in_code_section = False
            in_output_section = False
            in_error_section = True
            if show_errors:
                filtered_lines.append(line)
            continue
        elif line.startswith("---") or line.startswith("==="):
            in_code_section = False
            in_output_section = False
            in_error_section = False
            filtered_lines.append(line)
            continue

        # Filter content based on sections
        if in_code_section and not show_code:
            continue
        elif in_output_section and not show_output:
            continue
        elif in_error_section and not show_errors:
            continue
        else:
            # Always show non-section content (metadata, summaries, etc.)
            if not (in_code_section or in_output_section or in_error_section):
                filtered_lines.append(line)
            elif in_code_section and show_code:
                filtered_lines.append(line)
            elif in_output_section and show_output:
                filtered_lines.append(line)
            elif in_error_section and show_errors:
                filtered_lines.append(line)

    return '\n'.join(filtered_lines)


def extract_flags(log_path: Path) -> List[str]:
    """Extract all found flags from log"""
    content = log_path.read_text(encoding='utf-8')
    flags = re.findall(r'FLG:[A-Z0-9_]+', content)
    return list(set(flags))  # Remove duplicates


def show_summary(log_path: Path):
    """Show execution summary from log"""
    content = log_path.read_text(encoding='utf-8')

    # Count attempts
    attempts = len(re.findall(r'EXECUTION ATTEMPT #\d+', content))

    # Find final status
    success = "FINAL RESULT: SUCCESS" in content

    # Find flags
    flags = extract_flags(log_path)

    # Find errors
    error_count = content.count("--- ERRORS ---")
    non_empty_errors = len(re.findall(r'--- ERRORS ---\n(?!\(No errors\))', content))

    print("=" * 60)
    print("LOG SUMMARY")
    print("=" * 60)
    print(f"Log file: {log_path}")
    print(f"File size: {log_path.stat().st_size / 1024:.2f} KB")
    print(f"Total attempts: {attempts}")
    print(f"Final status: {'SUCCESS' if success else 'FAILED'}")
    print(f"Errors encountered: {non_empty_errors}/{error_count}")

    if flags:
        print(f"Flags found: {', '.join(flags)}")
    else:
        print("No flags found")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="View and filter Meta Agent logs")

    parser.add_argument(
        "log_file",
        nargs="?",
        help="Path to log file (default: latest log)"
    )
    parser.add_argument(
        "--show-code",
        action="store_true",
        help="Show generated code sections"
    )
    parser.add_argument(
        "--show-output",
        action="store_true",
        help="Show execution output sections"
    )
    parser.add_argument(
        "--no-errors",
        action="store_true",
        help="Hide error sections"
    )
    parser.add_argument(
        "--attempt",
        type=int,
        help="Show only specific attempt number"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show summary only"
    )
    parser.add_argument(
        "--flags",
        action="store_true",
        help="Show found flags only"
    )

    args = parser.parse_args()

    # Find log file
    if args.log_file:
        log_path = Path(args.log_file)
    else:
        log_path = find_latest_log()
        if not log_path:
            print("No log files found in ./logs/")
            return

    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        return

    # Handle different modes
    if args.summary:
        show_summary(log_path)
    elif args.flags:
        flags = extract_flags(log_path)
        if flags:
            print("Found flags:")
            for flag in flags:
                print(f"  - {flag}")
        else:
            print("No flags found in log")
    else:
        # Filter and display log
        filtered = filter_log(
            log_path,
            show_code=args.show_code,
            show_output=args.show_output,
            show_errors=not args.no_errors,
            attempt=args.attempt
        )
        print(filtered)


if __name__ == "__main__":
    main()
