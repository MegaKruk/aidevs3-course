"""
softo_task.py

Task runner that leverages SoftoNavigator to answer Centrala’s
softo-related questions and report them.
"""

from __future__ import annotations

import json
from typing import Dict

from ai_agents_framework import CentralaTask, FlagDetector, HttpClient
from task_utils import TaskRunner
from softo_agent import SoftoNavigator

class SoftoTask(CentralaTask):
    """Wraps the navigator and handles Centrala I/O."""

    QUESTIONS_URL_TMPL = "https://c3ntrala.ag3nts.org/data/{apikey}/softo.json"

    def __init__(self, llm_client):
        super().__init__(llm_client)
        self.navigator = SoftoNavigator(llm_client, max_steps=25)

    # ------------------------------------------------------------------
    def execute(self) -> Dict[str, object]:
        # 1. pull questions
        q_url = self.QUESTIONS_URL_TMPL.format(apikey=self.api_key)
        questions = HttpClient().get(q_url, timeout=15).json()  # type: ignore[arg-type]

        # 2. answer each question
        answers: Dict[str, str] = {}
        for qid, qtext in questions.items():
            print(f"\n--- Solving {qid}: {qtext}")
            ans = self.navigator.solve(qtext)
            answers[qid] = ans or "NIE_ZNALEZIONO"

        # 3. report back
        reply = self.submit_report("softo", answers)
        flag = FlagDetector.find_flag(json.dumps(reply))

        return {
            "status": "success",
            "answers": answers,
            "centrala_reply": reply,
            "flag": flag or "",
        }


# ----------------------------------------------------------------------
if __name__ == "__main__":
    runner = TaskRunner()
    result = runner.run_task(SoftoTask)
    runner.print_result(result, "softo")
