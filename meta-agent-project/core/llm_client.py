"""
LLM Client for interacting with Ollama
"""

import json
from typing import Dict, List, Optional, Any
import ollama
from ollama import Client
from utils.logger import setup_logger

logger = setup_logger(__name__)


class LLMClient:
    """
    Client for interacting with Ollama LLM
    """

    def __init__(
        self,
        decision_model: str = "qwen2.5:14b-instruct",
        coding_model: str = "qwen2.5-coder:14b",
        timeout: int = 300
    ):
        """
        Initialize LLM client with separate models for decision-making and coding

        Args:
            decision_model: Model for analysis and decision-making
            coding_model: Model for code generation
            timeout: Timeout for API calls in seconds
        """
        self.decision_model = decision_model
        self.coding_model = coding_model
        self.timeout = timeout

        try:
            self.client = Client(timeout=timeout)
            # Test connection
            self.client.list()
            logger.info(f"Connected to Ollama")
            logger.info(f"Decision model: {decision_model}")
            logger.info(f"Coding model: {coding_model}")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            raise RuntimeError(f"Cannot connect to Ollama. Make sure it's running: {e}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
        use_coding_model: bool = False
    ) -> str:
        """
        Generate text using the LLM

        Args:
            prompt: User prompt
            system_prompt: System prompt for context
            temperature: Temperature for generation (0-1)
            max_tokens: Maximum tokens to generate
            json_mode: Whether to expect JSON output
            use_coding_model: Whether to use the coding model instead of decision model

        Returns:
            Generated text
        """
        # Select model based on task type
        model = self.coding_model if use_coding_model else self.decision_model

        try:
            messages = []

            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })

            messages.append({
                "role": "user",
                "content": prompt
            })

            options = {
                "temperature": temperature,
                "num_predict": max_tokens
            }

            if json_mode:
                options["format"] = "json"

            logger.debug(f"Using model: {model} (coding={use_coding_model})")

            response = self.client.chat(
                model=model,
                messages=messages,
                options=options
            )

            result = response['message']['content']

            if json_mode:
                # Validate JSON
                try:
                    json.loads(result)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON response: {e}")
                    # Try to extract JSON from the response
                    result = self._extract_json(result)

            return result

        except Exception as e:
            logger.error(f"LLM generation failed with model {model}: {e}")
            raise

    def _extract_json(self, text: str) -> str:
        """
        Try to extract valid JSON from text
        """
        import re

        # Try to find JSON-like content
        json_patterns = [
            r'\{[^{}]*\}',  # Simple object
            r'\{.*\}',       # Any object
            r'\[.*\]'        # Array
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    json.loads(match)
                    return match
                except:
                    continue

        # If no valid JSON found, return original
        return text

    def generate_with_context(
        self,
        prompt: str,
        context: Dict[str, Any],
        system_prompt: Optional[str] = None,
        use_coding_model: bool = False,
        **kwargs
    ) -> str:
        """
        Generate with additional context

        Args:
            prompt: User prompt
            context: Context dictionary to include
            system_prompt: System prompt
            use_coding_model: Whether to use the coding model
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        # Build context string
        context_str = self._build_context_string(context)

        # Enhance prompt with context
        enhanced_prompt = f"""Context Information:
{context_str}

Task:
{prompt}"""

        return self.generate(
            enhanced_prompt,
            system_prompt,
            use_coding_model=use_coding_model,
            **kwargs
        )

    def _build_context_string(self, context: Dict[str, Any]) -> str:
        """
        Build a formatted context string from dictionary
        """
        lines = []
        for key, value in context.items():
            if isinstance(value, (list, dict)):
                value_str = json.dumps(value, indent=2)
            else:
                value_str = str(value)

            lines.append(f"{key}:\n{value_str}")

        return "\n\n".join(lines)

    def analyze_code(self, code: str, error_message: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze code and optionally its error

        Args:
            code: Python code to analyze
            error_message: Error message if code failed

        Returns:
            Analysis result
        """
        prompt = f"""Analyze the following Python code:

```python
{code}
```
"""

        if error_message:
            prompt += f"""
The code produced the following error:
{error_message}

Identify the issue and suggest a fix.
"""
        else:
            prompt += """
Review the code for potential issues, bugs, or improvements.
"""

        system_prompt = """You are a Python code analyzer. Provide detailed analysis including:
1. What the code does
2. Any issues or bugs found
3. Suggested improvements or fixes
Respond in JSON format with keys: 'summary', 'issues', 'suggestions'"""

        response = self.generate(prompt, system_prompt, json_mode=True)

        try:
            return json.loads(response)
        except:
            return {
                "summary": response,
                "issues": [],
                "suggestions": []
            }

    def check_connection(self) -> bool:
        """
        Check if Ollama connection is working

        Returns:
            True if connection is working
        """
        try:
            self.client.list()
            return True
        except:
            return False
