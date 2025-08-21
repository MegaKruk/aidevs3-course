"""
Result analyzer for understanding execution outcomes
"""

from typing import Dict, Any, Optional
import re
import json

from executors.safe_executor import ExecutionResult
from utils.logger import setup_logger

logger = setup_logger(__name__)


class ResultAnalyzer:
    """
    Analyzes code execution results and provides insights
    """

    def __init__(self, llm_client):
        """
        Initialize result analyzer

        Args:
            llm_client: LLM client for analysis
        """
        self.llm_client = llm_client

    def analyze(
            self,
            code: str,
            execution_result: ExecutionResult,
            task_description: str
    ) -> Dict[str, Any]:
        """
        Analyze execution result and provide insights

        Args:
            code: Executed code
            execution_result: Result from execution
            task_description: Original task description

        Returns:
            Analysis dictionary with insights and suggestions
        """
        logger.info("Analyzing execution result")

        analysis = {
            "execution_success": execution_result.success,
            "has_output": bool(execution_result.output),
            "has_error": bool(execution_result.error),
            "timeout": execution_result.timeout
        }

        # Check for flag in output
        if execution_result.output:
            flag = self._extract_flag(execution_result.output)
            if flag:
                analysis["flag_found"] = True
                analysis["flag"] = flag
                return analysis
            else:
                analysis["flag_found"] = False

        # Analyze error if present
        if execution_result.error:
            error_analysis = self._analyze_error(execution_result.error, code)
            analysis["error_analysis"] = error_analysis
            analysis["fix_suggestions"] = self._suggest_fixes(error_analysis, code)

        # Analyze output relevance
        if execution_result.output:
            relevance = self._analyze_output_relevance(
                execution_result.output,
                task_description
            )
            analysis["output_relevance"] = relevance

        # Generate overall recommendation
        analysis["recommendation"] = self._generate_recommendation(analysis)

        return analysis

    def _extract_flag(self, output: str) -> Optional[str]:
        """
        Extract flag from output

        Args:
            output: Program output

        Returns:
            Flag if found, None otherwise
        """
        # Look for FLG:XXXXX pattern
        pattern = r'FLG:[A-Z0-9_]+'
        match = re.search(pattern, output)

        if match:
            flag = match.group(0)
            logger.info(f"Flag found: {flag}")
            return flag

        return None

    def _analyze_error(self, error: str, code: str) -> Dict[str, Any]:
        """
        Analyze error message

        Args:
            error: Error message
            code: Code that produced the error

        Returns:
            Error analysis
        """
        error_analysis = {
            "type": "unknown",
            "message": error,
            "line": None,
            "likely_cause": None
        }

        # Identify error type
        if "SyntaxError" in error:
            error_analysis["type"] = "syntax"
            error_analysis["likely_cause"] = "Code has syntax errors"
        elif "ImportError" in error or "ModuleNotFoundError" in error:
            error_analysis["type"] = "import"
            error_analysis["likely_cause"] = "Missing required module"
        elif "NameError" in error:
            error_analysis["type"] = "name"
            error_analysis["likely_cause"] = "Undefined variable or function"
        elif "TypeError" in error:
            error_analysis["type"] = "type"
            error_analysis["likely_cause"] = "Type mismatch in operation"
        elif "ValueError" in error:
            error_analysis["type"] = "value"
            error_analysis["likely_cause"] = "Invalid value for operation"
        elif "KeyError" in error:
            error_analysis["type"] = "key"
            error_analysis["likely_cause"] = "Missing dictionary key"
        elif "IndexError" in error:
            error_analysis["type"] = "index"
            error_analysis["likely_cause"] = "List index out of range"
        elif "ConnectionError" in error or "RequestException" in error:
            error_analysis["type"] = "network"
            error_analysis["likely_cause"] = "Network connection issue"
        elif "timeout" in error.lower():
            error_analysis["type"] = "timeout"
            error_analysis["likely_cause"] = "Execution took too long"

        # Extract line number if available
        line_match = re.search(r'line (\d+)', error)
        if line_match:
            error_analysis["line"] = int(line_match.group(1))

        return error_analysis

    def _suggest_fixes(self, error_analysis: Dict[str, Any], code: str) -> list[str]:
        """
        Suggest fixes based on error analysis

        Args:
            error_analysis: Error analysis results
            code: Code that failed

        Returns:
            List of fix suggestions
        """
        suggestions = []

        error_type = error_analysis.get("type", "unknown")

        if error_type == "syntax":
            suggestions.append("Check for missing colons, parentheses, or indentation")
            suggestions.append("Verify string quotes are properly closed")
        elif error_type == "import":
            suggestions.append("Ensure all required libraries are imported")
            suggestions.append("Check import statement syntax")
            # Check what's being imported
            if "requests" in error_analysis.get("message", ""):
                suggestions.append("Add: import requests")
            elif "json" in error_analysis.get("message", ""):
                suggestions.append("Add: import json")
        elif error_type == "name":
            suggestions.append("Check variable is defined before use")
            suggestions.append("Verify function names are correct")
        elif error_type == "network":
            suggestions.append("Check URL format and accessibility")
            suggestions.append("Add timeout and error handling for requests")
            suggestions.append("Verify network connectivity")
        elif error_type == "timeout":
            suggestions.append("Optimize code for better performance")
            suggestions.append("Add progress indicators")
            suggestions.append("Consider breaking task into smaller parts")

        return suggestions

    def _analyze_output_relevance(self, output: str, task_description: str) -> Dict[str, Any]:
        """
        Analyze how relevant the output is to the task

        Args:
            output: Program output
            task_description: Original task

        Returns:
            Relevance analysis
        """
        prompt = f"""Analyze if this output successfully completes the given task.

Task:
{task_description[:1000]}

Output:
{output[:1000]}

Respond in JSON format:
{{
  "relevant": true/false,
  "completes_task": true/false,
  "missing_elements": ["element1", "element2"],
  "confidence": 0.0-1.0
}}"""

        system_prompt = "You are analyzing program outputs for task completion."

        try:
            response = self.llm_client.generate(
                prompt,
                system_prompt,
                temperature=0.2,
                json_mode=True
            )

            return json.loads(response)

        except Exception as e:
            logger.error(f"Failed to analyze output relevance: {e}")
            return {
                "relevant": False,
                "completes_task": False,
                "missing_elements": ["Unable to analyze"],
                "confidence": 0.0
            }

    def _generate_recommendation(self, analysis: Dict[str, Any]) -> str:
        """
        Generate recommendation based on analysis

        Args:
            analysis: Complete analysis results

        Returns:
            Recommendation string
        """
        if analysis.get("flag_found"):
            return "Success! Flag found in output."

        if analysis.get("timeout"):
            return "Code execution timed out. Consider optimizing or breaking into smaller parts."

        if analysis.get("has_error"):
            error_type = analysis.get("error_analysis", {}).get("type", "unknown")
            if error_type == "syntax":
                return "Fix syntax errors in the code before retrying."
            elif error_type == "import":
                return "Add missing import statements."
            elif error_type == "network":
                return "Check network connectivity and URL validity."
            else:
                return f"Fix {error_type} error and retry."

        if not analysis.get("has_output"):
            return "Code executed but produced no output. Add print statements."

        relevance = analysis.get("output_relevance", {})
        if relevance.get("completes_task"):
            return "Task appears complete but no flag found. Check output format."
        else:
            missing = relevance.get("missing_elements", [])
            if missing:
                return f"Output missing: {', '.join(missing[:3])}"
            else:
                return "Output doesn't match expected results. Review task requirements."
