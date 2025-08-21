"""
Code generator for creating Python solutions
"""

from typing import Dict, Any, Optional
import re
import json

from utils.logger import setup_logger

logger = setup_logger(__name__)


class CodeGenerator:
    """
    Generates Python code to solve homework tasks
    """

    def __init__(self, llm_client):
        """
        Initialize code generator

        Args:
            llm_client: LLM client for code generation
        """
        self.llm_client = llm_client

    def generate(
        self,
        task_description: str,
        context: Dict[str, Any],
        previous_attempt: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> str:
        """
        Generate Python code to solve the task

        Args:
            task_description: Description of the task
            context: Additional context (URLs, data, etc.)
            previous_attempt: Previous code attempt if fixing
            error_message: Error from previous attempt

        Returns:
            Generated Python code
        """
        logger.info("Generating solution code")

        # Build comprehensive prompt
        prompt = self._build_generation_prompt(
            task_description,
            context,
            previous_attempt,
            error_message
        )

        system_prompt = """You are an expert Python programmer solving homework tasks.
Generate clean, working Python 3.13 code that solves the given task.
Important rules:
1. Write ONLY Python code, no explanations or markdown
2. Include all necessary imports
3. Handle errors gracefully
4. Print the final result in the required format
5. If the task requires finding a flag, print it in format: FLG:XXXXX
6. Use requests library for HTTP calls
7. Include proper error handling
8. Write self-contained code that can run independently
9. Write code to a single .py file"""

        # Generate code using the coding model
        code = self.llm_client.generate(
            prompt,
            system_prompt,
            temperature=0.2,
            max_tokens=3000,
            use_coding_model=True  # Use specialized coding model
        )

        # Clean and validate code
        cleaned_code = self._clean_code(code)

        # Ensure proper structure
        final_code = self._ensure_code_structure(cleaned_code)

        return final_code

    def _build_generation_prompt(
        self,
        task_description: str,
        context: Dict[str, Any],
        previous_attempt: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> str:
        """
        Build comprehensive prompt for code generation

        Args:
            task_description: Task to solve
            context: Additional context
            previous_attempt: Previous code if fixing
            error_message: Previous error if fixing

        Returns:
            Complete prompt
        """
        prompt_parts = []

        # Add task description
        prompt_parts.append(f"Task to solve:\n{task_description}\n")

        # Add URLs if present
        if "homework_urls" in context and context["homework_urls"]:
            prompt_parts.append("Referenced URLs:")
            for url in context["homework_urls"]:
                prompt_parts.append(f"- {url}")
            prompt_parts.append("")

        # Add external data if present
        if "external_data" in context and context["external_data"]:
            prompt_parts.append("External data available:")
            for url, content in context["external_data"].items():
                # Limit content preview
                preview = content[:500] if len(content) > 500 else content
                prompt_parts.append(f"\nFrom {url}:")
                prompt_parts.append(f"{preview}...")
            prompt_parts.append("")

        # Add previous attempt context if fixing
        if previous_attempt and error_message:
            prompt_parts.append("Previous attempt failed with error:")
            prompt_parts.append(f"{error_message}\n")
            prompt_parts.append("Previous code:")
            prompt_parts.append(f"```python\n{previous_attempt}\n```\n")
            prompt_parts.append("Fix the issue and generate working code.")

        # Add analysis from previous attempts if available
        for i in range(1, 6):
            analysis_key = f"analysis_attempt_{i}"
            if analysis_key in context:
                analysis = context[analysis_key]
                prompt_parts.append(f"\nAnalysis from attempt {i}:")
                if isinstance(analysis, dict):
                    prompt_parts.append(json.dumps(analysis, indent=2))
                else:
                    prompt_parts.append(str(analysis))

        prompt_parts.append("\nGenerate Python code that solves this task:")

        return "\n".join(prompt_parts)

    def _clean_code(self, code: str) -> str:
        """
        Clean generated code

        Args:
            code: Raw generated code

        Returns:
            Cleaned code
        """
        # Remove markdown code blocks if present
        code = re.sub(r'^```python\n', '', code)
        code = re.sub(r'^```\n', '', code)
        code = re.sub(r'\n```', '', code)

        # Remove any explanatory text before/after code
        lines = code.split('\n')
        code_lines = []
        in_code = False

        for line in lines:
            # Detect start of code
            if not in_code:
                if line.strip().startswith(('import ', 'from ', 'def ', 'class ', '#', '"""')):
                    in_code = True
                    code_lines.append(line)
            else:
                # Stop if we hit explanatory text
                if line.strip() and not any([
                    line.strip().startswith('#'),
                    line.strip().startswith('"""'),
                    line.strip().startswith("'''"),
                    self._is_valid_python_line(line)
                ]):
                    # Check if it's the end of code
                    if 'explanation' in line.lower() or 'note:' in line.lower():
                        break
                code_lines.append(line)

        return '\n'.join(code_lines)

    def _is_valid_python_line(self, line: str) -> bool:
        """
        Check if line looks like valid Python code

        Args:
            line: Line to check

        Returns:
            True if likely valid Python
        """
        stripped = line.strip()

        # Empty lines are valid
        if not stripped:
            return True

        # Common Python keywords and patterns
        python_patterns = [
            r'^(import |from |def |class |if |elif |else:|for |while |try:|except|finally:|with |return |raise |yield |pass|break|continue)',
            r'^[a-zA-Z_][a-zA-Z0-9_]*\s*=',  # Variable assignment
            r'^[a-zA-Z_][a-zA-Z0-9_]*\(',     # Function call
            r'^\s*["\']',                      # String literal
            r'^\s*\d',                          # Number literal
            r'^\s*[\[\{\(]',                   # Container literal
            r'^\s*\)',                          # Closing parenthesis (continuation)
            r'^print\(',                       # Print statement
        ]

        for pattern in python_patterns:
            if re.match(pattern, stripped):
                return True

        return False

    def _ensure_code_structure(self, code: str) -> str:
        """
        Ensure code has proper structure

        Args:
            code: Cleaned code

        Returns:
            Structured code
        """
        # Check if code has main block
        if "if __name__" not in code:
            # Check if there's any executable code outside functions
            lines = code.split('\n')
            has_executable = False

            for line in lines:
                stripped = line.strip()
                if stripped and not stripped.startswith(('#', 'import ', 'from ', 'def ', 'class ')):
                    if not stripped.startswith('"""') and not stripped.startswith("'''"):
                        has_executable = True
                        break

            if has_executable:
                # Wrap executable code in main block
                import_lines = []
                function_lines = []
                main_lines = []

                in_function = False
                in_class = False
                indent_level = 0

                for line in lines:
                    if line.strip().startswith(('import ', 'from ')):
                        import_lines.append(line)
                    elif line.strip().startswith('def '):
                        in_function = True
                        function_lines.append(line)
                    elif line.strip().startswith('class '):
                        in_class = True
                        function_lines.append(line)
                    elif in_function or in_class:
                        function_lines.append(line)
                        # Check if we're exiting function/class
                        if line and not line[0].isspace():
                            in_function = False
                            in_class = False
                    elif line.strip() and not line.strip().startswith('#'):
                        main_lines.append(line)
                    else:
                        # Comments and empty lines
                        if import_lines and not function_lines:
                            import_lines.append(line)
                        elif function_lines:
                            function_lines.append(line)
                        else:
                            main_lines.append(line)

                # Reconstruct code
                new_code = []
                if import_lines:
                    new_code.extend(import_lines)
                    new_code.append('')

                if function_lines:
                    new_code.extend(function_lines)
                    new_code.append('')

                if main_lines:
                    new_code.append('if __name__ == "__main__":')
                    for line in main_lines:
                        if line.strip():
                            new_code.append('    ' + line)
                        else:
                            new_code.append(line)

                code = '\n'.join(new_code)

        return code