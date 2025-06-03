"""
research_task_llm.py

Idea
----
The “correct” readings form a closed set: any line that appears in
`correct.txt` is genuine, anything else is forged.
We feed the full list of genuine readings to GPT-4o in the system prompt
and simply ask it to answer “1” (genuine) or “0” (forged) for every line
from `verify.txt`.

Because there are only ~200 genuine rows the prompt comfortably fits
within GPT-4o’s 128 k context window.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

from ai_agents_framework import CentralaTask, FlagDetector, LLMClient
from task_utils import TaskRunner


class ResearchTaskLLM(CentralaTask):
    DATA_DIR = Path("data/lab_data")        # folder with the three .txt files

    # ------------------------------------------------------------------
    @staticmethod
    def _read(fname: str) -> List[str]:
        path = ResearchTaskLLM.DATA_DIR / fname
        if not path.exists():
            raise FileNotFoundError(path.as_posix())
        return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    @staticmethod
    def _split_verify(line: str) -> tuple[str, str]:
        """
        Expect “NN=payload” → return ("NN", "payload")
        """
        m = re.match(r"\s*(\d{2})\s*=\s*(.*)", line)
        if not m:
            raise ValueError(f"Bad verify line: {line}")
        return m.group(1), m.group(2).strip()

    # ------------------------------------------------------------------
    def _make_system_prompt(self, genuine: List[str]) -> str:
        lines = "\n".join(genuine)
        return (
            "You are an expert validator of robot sensor readings.\n"
            "Below is the exhaustive list of *genuine* readings—one per line.\n"
            "Any other string MUST be considered forged.\n\n"
            "GENUINE READINGS START\n"
            f"{lines}\n"
            "GENUINE READINGS END\n\n"
            "When I send you a reading, respond with:\n"
            "  1   … if it matches one of the genuine readings exactly\n"
            "  0   … otherwise\n"
            "Return a single character (1 or 0) with no explanation."
        )

    # ------------------------------------------------------------------
    def execute(self) -> Dict[str, object]:
        correct = self._read("correct.txt")
        verify  = self._read("verify.txt")

        system_prompt = self._make_system_prompt(correct)
        accepted_ids: List[str] = []

        for row in verify:
            uid, payload = self._split_verify(row)
            resp = self.llm_client.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": payload},
                ],
                max_tokens=2,
                temperature=0.0,
            )
            verdict = resp.choices[0].message.content.strip()
            if verdict == "1":
                accepted_ids.append(uid)

        centrala_reply = self.submit_report("research", accepted_ids)
        flag = FlagDetector.find_flag(json.dumps(centrala_reply))

        return {
            "status": "success",
            "accepted": accepted_ids,
            "centrala_reply": centrala_reply,
            "flag": flag or "",
        }


# ----------------------------------------------------------------------
if __name__ == "__main__":
    runner = TaskRunner()
    result = runner.run_task(ResearchTaskLLM)
    runner.print_result(result, "research (LLM variant)")
