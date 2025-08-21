"""
JSON Calibration Task
---------------------
- Downloads the calibration file protected by your API key
- Corrects every arithmetic answer
- Fills each open “test” question via GPT‑4o
- Sends the fixed object back to /report and prints the server reply
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List

from dotenv import load_dotenv

from ai_agents_framework import LLMClient, HttpClient, ExpressionEvaluator, Task
from task_utils import TaskRunner, verify_environment

load_dotenv()
CENTRAL_API_KEY = os.getenv("CENTRALA_API_KEY")
if not CENTRAL_API_KEY:
    raise RuntimeError("CENTRALA_API_KEY must be set in the environment")

BASE_URL = "https://c3ntrala.ag3nts.org"


# --------------------------------------------------------------------------
# Task implementation
# --------------------------------------------------------------------------
class JSONCalibrationTask(Task):
    FILE_ENDPOINT = f"{BASE_URL}/data/{{apikey}}/json.txt"
    REPORT_ENDPOINT = f"{BASE_URL}/report"

    def __init__(self, llm_client: LLMClient, api_key: str):
        super().__init__(llm_client)
        self.api_key = api_key

    # 1. Download -----------------------------------------------------------------
    def _fetch_file(self) -> Dict[str, Any]:
        url = self.FILE_ENDPOINT.format(apikey=self.api_key)
        response = self.http_client.get(url, timeout=30)
        response.raise_for_status()
        return json.loads(response.text)

    # 2. Fix arithmetic -----------------------------------------------------------
    @staticmethod
    def _fix_arithmetic(question: str) -> int | float:
        # remove any text that is not part of the expression (keeps  “15 + 6” etc.)
        expr = re.sub(r'[^0-9+\-*/().]', ' ', question)
        expr = re.sub(r'\s+', '', expr)
        return ExpressionEvaluator.safe_eval(expr)

    # 3. Answer open questions ----------------------------------------------------
    def _answer_open_question(self, question: str) -> str:
        context = (
            "You are a factual assistant. "
            "Respond concisely with a short, direct answer. "
            "No additional text."
        )
        return self.llm_client.answer_with_context(
            question=question,
            context=context,
            model="gpt-4o",
            max_tokens=20,
        )

    # 4. Transform whole object ----------------------------------------------------
    def _fix_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        fixed_items: List[Dict[str, Any]] = []
        for item in data.get("test-data", []):
            q = item.get("question", "")
            try:
                correct = self._fix_arithmetic(q)
                item["answer"] = correct
            except Exception:
                # Non‑arithmetic questions are left as‑is
                pass

            # handle nested open questions
            if "test" in item and "q" in item["test"]:
                item["test"]["a"] = self._answer_open_question(item["test"]["q"])
            fixed_items.append(item)

        data["test-data"] = fixed_items
        data["apikey"] = self.api_key          # ensure second key is set
        return data

    # 5. Submit -------------------------------------------------------------------
    def _submit_report(self, fixed_obj: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "task": "JSON",
            "apikey": self.api_key,            # first key
            "answer": fixed_obj,               # entire corrected object
        }
        return self.http_client.submit_json(self.REPORT_ENDPOINT, payload)

    # Orchestrate -----------------------------------------------------------------
    def execute(self) -> Dict[str, Any]:
        try:
            original = self._fetch_file()
            fixed = self._fix_data(original)
            reply = self._submit_report(fixed)
            return {"status": "success", "response": reply}
        except Exception as exc:
            return self._handle_error(exc)


# --------------------------------------------------------------------------
# Main runner
# --------------------------------------------------------------------------
if __name__ == "__main__":
    verify_environment()  # verifies OPENAI_API_KEY for the LLM

    runner = TaskRunner()
    result = runner.run_task(JSONCalibrationTask, api_key=CENTRAL_API_KEY)

    runner.print_result(result, "JSON Calibration Task")
    if result.get("status") == "success":
        print(f"\nServer response: {result['response']}")
