"""
research_task.py

Centrala “research” – trust the genuine lab readings.

Insight
-------
The three source files show a hidden, but very simple rule:

* **correct.txt**   – every line appears verbatim in the official reference set
* **incorect.txt**  – every line appears verbatim in the forged set
* **verify.txt**    – lines copied *exactly* from one of the two lists

Therefore, to decide which IDs are trustworthy we only need to:

1. Load `correct.txt` and put every line into a Python **set**.
2. For each row in `verify.txt`
   • split off its 2-digit ID
   • keep the payload after the `=` sign
   • accept the ID if that payload is **in the correct-set**.
3. Submit the list of accepted IDs.

No LLM calls, no heuristics – just string membership.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

from ai_agents_framework import CentralaTask, FlagDetector, HttpClient
from task_utils import TaskRunner


class ResearchTask(CentralaTask):
    DATA_DIR = Path("data/lab_data")         # folder with the three .txt files

    # ---------------------------------------------------------------
    @staticmethod
    def _slurp(fname: str) -> List[str]:
        path = ResearchTask.DATA_DIR / fname
        if not path.exists():
            raise FileNotFoundError(path.as_posix())
        return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    @staticmethod
    def _split_verify(line: str) -> tuple[str, str]:
        """
        Expects "NN=word,word,word" and returns ("NN", "word,word,word")
        """
        m = re.match(r"\s*(\d{2})\s*=\s*(.*)", line)
        if not m:
            raise ValueError(f"Malformed verify line: {line}")
        return m.group(1), m.group(2).strip()

    # ---------------------------------------------------------------
    def execute(self) -> Dict[str, object]:
        correct_set = set(self._slurp("correct.txt"))
        verify_rows = self._slurp("verify.txt")

        accepted: List[str] = []
        for row in verify_rows:
            uid, payload = self._split_verify(row)
            if payload in correct_set:
                accepted.append(uid)

        centrala_reply = self.submit_report("research", accepted)
        flag = FlagDetector.find_flag(json.dumps(centrala_reply))

        return {
            "status": "success",
            "accepted": accepted,
            "centrala_reply": centrala_reply,
            "flag": flag or "",
        }


# -------------------------------------------------------------------
if __name__ == "__main__":
    runner = TaskRunner()
    result = runner.run_task(ResearchTask)
    runner.print_result(result, "research")
