"""
LLM Client for interacting with Ollama and OpenAI
"""

import json
import os
from typing import Dict, List, Optional, Any
from enum import Enum
from dotenv import load_dotenv
import urllib3

# Load environment variables
load_dotenv()

# Disable SSL warnings if needed
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from utils.logger import setup_logger

logger = setup_logger(__name__)

# Try to import both clients
try:
    import ollama
    from ollama import Client as OllamaClient
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama not available")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI client not available")


class LLMProvider(Enum):
    """Available LLM providers"""
    OLLAMA = "ollama"
    OPENAI = "openai"


class LLMClient:
    """
    Client for interacting with LLMs (Ollama or OpenAI)
    """

    def __init__(
        self,
        decision_model: str = "qwen2.5:14b-instruct",
        coding_model: str = "qwen2.5-coder:14b",
        timeout: int = 300,
        provider: Optional[LLMProvider] = None,
        openai_decision_model: str = "gpt-4o",
        openai_coding_model: str = "gpt-4o"
    ):
        """
        Initialize LLM client with support for multiple providers

        Args:
            decision_model: Model for analysis and decision-making (Ollama)
            coding_model: Model for code generation (Ollama)
            timeout: Timeout for API calls in seconds
            provider: Force specific provider, or None for auto-detection
            openai_decision_model: OpenAI model for decision-making
            openai_coding_model: OpenAI model for code generation
        """
        self.timeout = timeout
        self.provider = provider

        # Ollama models
        self.ollama_decision_model = decision_model
        self.ollama_coding_model = coding_model

        # OpenAI models
        self.openai_decision_model = openai_decision_model
        self.openai_coding_model = openai_coding_model

        # Initialize the appropriate client
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the appropriate LLM client based on availability and preference"""

        # Check for OpenAI API key
        self.openai_api_key = os.getenv('OPENAI_API_KEY')

        if self.provider == LLMProvider.OPENAI or (self.provider is None and self.openai_api_key and OPENAI_AVAILABLE):
            if not OPENAI_AVAILABLE:
                raise RuntimeError("OpenAI client not installed. Run: pip install openai")
            if not self.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY not found in environment variables")

            try:
                self.openai_client = OpenAI(api_key=self.openai_api_key)
                self.active_provider = LLMProvider.OPENAI
                logger.info(f"Using OpenAI provider")
                logger.info(f"Decision model: {self.openai_decision_model}")
                logger.info(f"Coding model: {self.openai_coding_model}")
                return
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                if self.provider == LLMProvider.OPENAI:
                    raise

        # Fall back to Ollama
        if OLLAMA_AVAILABLE:
            try:
                self.ollama_client = OllamaClient(timeout=self.timeout)
                # Test connection
                self.ollama_client.list()
                self.active_provider = LLMProvider.OLLAMA
                logger.info(f"Using Ollama provider")
                logger.info(f"Decision model: {self.ollama_decision_model}")
                logger.info(f"Coding model: {self.ollama_coding_model}")
            except Exception as e:
                logger.error(f"Failed to connect to Ollama: {e}")
                raise RuntimeError(f"Cannot connect to Ollama. Make sure it's running: {e}")
        else:
            raise RuntimeError("No LLM provider available. Install ollama or set up OpenAI.")

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
        if self.active_provider == LLMProvider.OPENAI:
            return self._generate_openai(
                prompt, system_prompt, temperature, max_tokens, json_mode, use_coding_model
            )
        else:
            return self._generate_ollama(
                prompt, system_prompt, temperature, max_tokens, json_mode, use_coding_model
            )

    def _generate_openai(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
        use_coding_model: bool = False
    ) -> str:
        """Generate using OpenAI API"""
        model = self.openai_coding_model if use_coding_model else self.openai_decision_model

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

            logger.debug(f"Using OpenAI model: {model} (coding={use_coding_model})")

            # Prepare kwargs
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            # Add JSON mode if requested (only for compatible models)
            if json_mode and "gpt-4" in model:
                kwargs["response_format"] = {"type": "json_object"}

            response = self.openai_client.chat.completions.create(**kwargs)

            result = response.choices[0].message.content

            if json_mode:
                # Validate JSON
                try:
                    json.loads(result)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON response: {e}")
                    result = self._extract_json(result)

            return result

        except Exception as e:
            logger.error(f"OpenAI generation failed with model {model}: {e}")
            raise

    def _generate_ollama(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
        use_coding_model: bool = False
    ) -> str:
        """Generate using Ollama"""
        model = self.ollama_coding_model if use_coding_model else self.ollama_decision_model

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

            logger.debug(f"Using Ollama model: {model} (coding={use_coding_model})")

            response = self.ollama_client.chat(
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
                    result = self._extract_json(result)

            return result

        except Exception as e:
            logger.error(f"Ollama generation failed with model {model}: {e}")
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
        Check if LLM connection is working

        Returns:
            True if connection is working
        """
        try:
            if self.active_provider == LLMProvider.OPENAI:
                # Test with a simple completion
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=5
                )
                return True
            else:
                self.ollama_client.list()
                return True
        except:
            return False

    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about the active provider and models

        Returns:
            Dictionary with provider information
        """
        return {
            "provider": self.active_provider.value,
            "decision_model": (
                self.openai_decision_model
                if self.active_provider == LLMProvider.OPENAI
                else self.ollama_decision_model
            ),
            "coding_model": (
                self.openai_coding_model
                if self.active_provider == LLMProvider.OPENAI
                else self.ollama_coding_model
            ),
            "connection_status": self.check_connection()
        }
