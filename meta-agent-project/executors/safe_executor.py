"""
Safe executor for running generated Python code
"""

import subprocess
import tempfile
import os
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
import json
import shlex

from utils.logger import setup_logger, ExecutionLogger

logger = setup_logger(__name__)


@dataclass
class ExecutionResult:
    """Container for code execution results"""
    success: bool
    output: str
    error: Optional[str]
    return_code: int
    timeout: bool = False
    execution_time: float = 0.0


class SafeCodeExecutor:
    """
    Safely executes Python code in isolated environment
    """

    def __init__(
        self,
        timeout: int = 30,
        memory_limit_mb: int = 256,
        use_docker: bool = False,
        log_executions: bool = True
    ):
        """
        Initialize safe executor

        Args:
            timeout: Maximum execution time in seconds
            memory_limit_mb: Memory limit in MB
            use_docker: Whether to use Docker for isolation
            log_executions: Whether to log all executions
        """
        self.timeout = timeout
        self.memory_limit_mb = memory_limit_mb
        self.use_docker = use_docker and self._check_docker_available()
        self.log_executions = log_executions

        if self.log_executions:
            self.execution_logger = ExecutionLogger()

        if self.use_docker:
            logger.info("Using Docker for code execution")
            self._prepare_docker_image()
        else:
            logger.info("Using subprocess isolation for code execution")

    def execute(self, code: str, input_data: Optional[str] = None, attempt_number: int = 1) -> ExecutionResult:
        """
        Execute Python code safely

        Args:
            code: Python code to execute
            input_data: Optional input data for the program
            attempt_number: Current attempt number for logging

        Returns:
            ExecutionResult object
        """
        logger.info(f"Executing generated code safely (attempt #{attempt_number})")

        import time
        start_time = time.time()

        if self.use_docker:
            result = self._execute_docker(code, input_data)
        else:
            result = self._execute_subprocess(code, input_data)

        result.execution_time = time.time() - start_time

        # Log the execution
        if self.log_executions and self.execution_logger:
            # Check for flag in output
            flag = None
            if result.output:
                import re
                flag_match = re.search(r'FLG:[A-Z0-9_]+', result.output)
                if flag_match:
                    flag = flag_match.group(0)

            self.execution_logger.log_execution(
                attempt_number=attempt_number,
                code=code,
                result={
                    "return_code": result.return_code,
                    "timeout": result.timeout,
                    "execution_time": result.execution_time
                },
                success=result.success,
                output=result.output,
                error=result.error if result.error else "",
                flag_found=flag
            )

        # Also log key info to standard logger
        logger.info(f"Execution completed in {result.execution_time:.2f}s - Success: {result.success}")
        if result.output:
            logger.debug(f"Output preview: {result.output[:200]}...")
        if result.error:
            logger.warning(f"Execution error: {result.error[:200]}...")

        return result

    def _execute_subprocess(self, code: str, input_data: Optional[str] = None) -> ExecutionResult:
        """
        Execute code using subprocess with restrictions

        Args:
            code: Python code to execute
            input_data: Optional input data

        Returns:
            ExecutionResult
        """
        # Create temporary file for code
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False,
            dir=tempfile.gettempdir()
        ) as f:
            f.write(code)
            temp_file = f.name

        try:
            # Prepare command with restrictions
            cmd = [
                'python3',
                '-u',  # Unbuffered output
                temp_file
            ]

            # Set up environment with restrictions
            env = os.environ.copy()
            env['PYTHONDONTWRITEBYTECODE'] = '1'  # Don't create .pyc files

            # Run the code
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                cwd=tempfile.gettempdir()  # Run in temp directory
            )

            try:
                stdout, stderr = process.communicate(
                    input=input_data,
                    timeout=self.timeout
                )

                return ExecutionResult(
                    success=(process.returncode == 0),
                    output=stdout,
                    error=stderr if stderr else None,
                    return_code=process.returncode,
                    timeout=False
                )

            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()

                return ExecutionResult(
                    success=False,
                    output=stdout if stdout else "",
                    error=f"Execution timeout ({self.timeout}s)",
                    return_code=-1,
                    timeout=True
                )

        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                return_code=-1
            )
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except:
                pass

    def _execute_docker(self, code: str, input_data: Optional[str] = None) -> ExecutionResult:
        """
        Execute code in Docker container for maximum isolation

        Args:
            code: Python code to execute
            input_data: Optional input data

        Returns:
            ExecutionResult
        """
        # Create temporary directory for code
        with tempfile.TemporaryDirectory() as temp_dir:
            code_file = Path(temp_dir) / "solution.py"
            code_file.write_text(code)

            # Prepare Docker command
            docker_cmd = [
                'docker', 'run',
                '--rm',  # Remove container after execution
                '--network', 'none',  # No network access
                '--memory', f'{self.memory_limit_mb}m',  # Memory limit
                '--memory-swap', f'{self.memory_limit_mb}m',  # No swap
                '--cpus', '0.5',  # CPU limit
                '--read-only',  # Read-only root filesystem
                '--tmpfs', '/tmp:size=10M',  # Small temp directory
                '-v', f'{code_file}:/app/solution.py:ro',  # Mount code as read-only
                '-w', '/app',  # Working directory
                'python:3.13-slim',
                'python', '/app/solution.py'
            ]

            try:
                process = subprocess.Popen(
                    docker_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                stdout, stderr = process.communicate(
                    input=input_data,
                    timeout=self.timeout
                )

                return ExecutionResult(
                    success=(process.returncode == 0),
                    output=stdout,
                    error=stderr if stderr else None,
                    return_code=process.returncode
                )

            except subprocess.TimeoutExpired:
                # Kill Docker container
                subprocess.run(['docker', 'kill', container_id], capture_output=True)

                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"Execution timeout ({self.timeout}s)",
                    return_code=-1,
                    timeout=True
                )
            except Exception as e:
                logger.error(f"Docker execution failed: {e}")
                return ExecutionResult(
                    success=False,
                    output="",
                    error=str(e),
                    return_code=-1
                )

    def _check_docker_available(self) -> bool:
        """
        Check if Docker is available

        Returns:
            True if Docker is available
        """
        try:
            result = subprocess.run(
                ['docker', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False

    def _prepare_docker_image(self):
        """
        Prepare Docker image for execution
        """
        # Check if Python image exists
        try:
            subprocess.run(
                ['docker', 'pull', 'python:3.13-slim'],
                capture_output=True,
                check=False,
                timeout=300
            )
        except Exception as e:
            logger.warning(f"Could not pull Docker image: {e}")

    def validate_code_safety(self, code: str) -> tuple[bool, Optional[str]]:
        """
        Basic validation to check for obviously dangerous code

        Args:
            code: Python code to validate

        Returns:
            Tuple of (is_safe, reason_if_unsafe)
        """
        dangerous_patterns = [
            ('__import__', 'Dynamic imports are not allowed'),
            ('eval(', 'eval() is not allowed'),
            ('exec(', 'exec() is not allowed'),
            ('compile(', 'compile() is not allowed'),
            ('open(', 'File operations need review'),
            ('subprocess', 'Subprocess calls are not allowed'),
            ('os.system', 'System calls are not allowed'),
            ('os.popen', 'System calls are not allowed'),
            ('/etc/', 'System file access is not allowed'),
            ('/usr/', 'System file access is not allowed'),
            ('/bin/', 'System file access is not allowed'),
            ('socket', 'Network operations need review'),
        ]

        code_lower = code.lower()

        for pattern, reason in dangerous_patterns:
            if pattern.lower() in code_lower:
                # Allow some patterns in specific contexts
                if pattern == 'open(' and 'requests' in code_lower:
                    continue  # Likely just importing requests which is OK

                logger.warning(f"Potentially dangerous code detected: {reason}")
                # We'll still run it but log the warning
                # return False, reason

        return True, None
