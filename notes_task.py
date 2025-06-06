"""
notes_task.py – S04E05  (AIDevs 3)
Keeps a full “black-box” conversation history per question so GPT-4o
never repeats a rejected answer, and derives the Andrzej-meeting date
(-Q04) from the “to już jutro … 11 listopada 2024” passage:

    11 XI 2024 appears in the text ⇒ meeting = 2024-11-12
"""

from __future__ import annotations

import json, re, time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

from ai_agents_framework import (
    LLMClient,
    FlagDetector,
    CentralaTask,
    HttpClient,
    PDFProcessor,
)
from task_utils import TaskRunner


# ────────────────────────────────────────────────────────────────────────
class NotesTask(CentralaTask):
    PDF_PATH = "data/notatnik-rafala.pdf"
    QUESTIONS_URL = "https://c3ntrala.ag3nts.org/data/{apikey}/notes.json"

    DATE_RX = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")

    # if GPT keeps failing we fall back to these fixed candidates
    FALLBACK = {
        "01": ["2019", "1939", "1989", "2024"],
        "03": ["Jaskinia", "Lubawa", "Ochra ziemia", "Iz 2:19"],
    }

    # ── build notebook once ────────────────────────────────────────────
    def __init__(self, llm: LLMClient):
        super().__init__(llm)

        pdf = PDFProcessor(self.PDF_PATH)

        plain = [
            f"[PAGE-{i:02}]\n{t}"
            for i, t in enumerate(pdf.extract_pages_text(1, 18), 1)
        ]

        ocr_dir = Path("data/ocr_results");  ocr_dir.mkdir(exist_ok=True, parents=True)
        ocr = []
        for i in range(1, 20):
            cache = ocr_dir / f"{i:02}.txt"
            if cache.exists():
                txt = cache.read_text("utf-8", errors="ignore")
            else:
                img = pdf.page_to_image(i)
                txt = pdf.ocr_image(img)
                cache.write_text(txt, "utf-8")
            if txt.strip():
                ocr.append(f"[PAGE-{i:02}-OCR]\n{txt}")

        self.notebook = "\n\n".join(plain + ocr)
        self.meeting_date = self._compute_meeting_date()

        # conversation history per question (for “do not repeat”)
        self.history: Dict[str, List[str]] = {}

    # ── “jutro … 11 listopada 2024”  →  2024-11-12 ─────────────────────
    def _compute_meeting_date(self) -> str | None:
        lines = self.notebook.splitlines()
        for idx, ln in enumerate(lines):
            if "jutro" in ln.lower():
                # scan ±6 lines for explicit date
                window = lines[max(0, idx-6): idx+7]
                m = re.search(r"(\d{1,2})\s+listopada\s+(\d{4})", " ".join(window), re.I)
                if m:
                    day, year = int(m.group(1)), int(m.group(2))
                    d = datetime(year, 11, day) + timedelta(days=1)
                    return d.strftime("%Y-%m-%d")
        return None   # let GPT figure it out otherwise

    # ── helpers ─────────────────────────────────────────────────────────
    @staticmethod
    def _last_line(txt: str) -> str:
        return next((l.strip() for l in reversed(txt.splitlines()) if l.strip()), txt.strip())

    @staticmethod
    def _two_tokens(txt: str) -> str:
        return " ".join(txt.split()[:2])

    # ── single GPT request ─────────────────────────────────────────────
    def _ask_llm(self, qid: str, qtxt: str, hint: str = "") -> str:
        prev_attempts = "\n".join(f"- {a}" for a in self.history.get(qid, []))
        avoid_txt = f"\nUnikaj ponownie odpowiedzi:\n{prev_attempts}" if prev_attempts else ""
        user = f"Pytanie {qid}: {qtxt}{avoid_txt}"
        if hint:
            user += f"\nPodpowiedź od Centrali: {hint}"

        sys_base = (
            "Jesteś analitykiem notatnika Rafała. Cały notatnik znajduje się pomiędzy znacznikami "
            "<NOTEBOOK> … </NOTEBOOK>. Odpowiadaj bardzo zwięźle po polsku.\n"
        )

        # per-question formatting
        if qid == "01":
            ctx = sys_base + "W OSTATNIEJ LINII wypisz wyłącznie czterocyfrowy rok.\n"
        elif qid == "03":
            ctx = sys_base + "W OSTATNIEJ LINII wypisz dwuwyrazową nazwę schronienia.\n"
        elif qid == "04":
            if self.meeting_date:
                return self.meeting_date
            ctx = sys_base + "Wyprowadź datę spotkania (jest względna). W OSTATNIEJ LINII wypisz YYYY-MM-DD.\n"
        elif qid == "05":
            ctx = sys_base + "W OSTATNIEJ LINII wypisz wyłącznie nazwę miejscowości.\n"
        else:
            ctx = sys_base

        ctx += "<NOTEBOOK>\n" + self.notebook + "\n</NOTEBOOK>"

        raw = self.llm_client.answer_with_context(
            question=user,
            context=ctx,
            model="gpt-4o",
            max_tokens=160,
        ).strip()

        if qid == "01":
            m = re.search(r"\b(\d{4})\b\D*$", raw); out = m.group(1) if m else self._last_line(raw)
        elif qid == "03":
            out = self._two_tokens(self._last_line(raw))
        elif qid == "04":
            m = self.DATE_RX.search(raw); out = m.group(1) if m else self._last_line(raw)
        elif qid == "05":
            out = self._last_line(raw)
        else:
            out = self._last_line(raw)

        print(f"[DBG] Q{qid} RAW -> {raw!r}\n[DBG] Q{qid} FINAL -> {out!r}")
        self.history.setdefault(qid, []).append(out)
        return out

    # ── submit & parse failure ─────────────────────────────────────────
    def _submit(self, ans: Dict[str, str]) -> dict:
        r = self.http_client.post(
            "https://c3ntrala.ag3nts.org/report",
            json={"task": "notes", "apikey": self.api_key, "answer": ans},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        try:
            return r.json()
        except ValueError:
            return {"raw": r.text}

    @staticmethod
    def _fail(reply: dict) -> Tuple[str, str]:
        m = re.search(r"question\s+(\d{2})", reply.get("message", ""))
        return (m.group(1) if m else ""), reply.get("hint", "")

    # ── main loop ───────────────────────────────────────────────────────
    def execute(self):
        questions = HttpClient().get(
            self.QUESTIONS_URL.format(apikey=self.api_key), timeout=20
        ).json()

        answers = {q: self._ask_llm(q, t) for q, t in questions.items()}

        for attempt in range(20):
            print(f"[DBG] Attempt {attempt+1}  -> {answers}")
            reply = self._submit(answers)

            flag = FlagDetector.find_flag(json.dumps(reply))
            if flag:
                return {"status": "success", "flag": flag, "answers": answers}

            qid, hint = self._fail(reply)
            if not qid:
                break  # unknown error

            # choose next answer
            tried = set(self.history.get(qid, []))
            if qid in self.FALLBACK and self.FALLBACK[qid]:
                nxt = self.FALLBACK[qid].pop(0)
                if nxt in tried:
                    continue
                answers[qid] = nxt
                self.history.setdefault(qid, []).append(nxt)
            else:
                answers[qid] = self._ask_llm(qid, questions[qid], hint=hint)

            time.sleep(1.0)

        return {"status": "failed", "reply": reply, "answers": answers}


# ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    TaskRunner().print_result(TaskRunner().run_task(NotesTask), "notes")
