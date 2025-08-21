# life_letter_manual.py  – sekret 8 „List od życia” (wersja bez OCR)
#
# • szyfr A#S# został już ręcznie odczytany
# • poniżej podajemy *gotowy* zdekodowany tekst
# • LLM wskazuje nazwę miejsca (jedno słowo, bez ogonków)
# • skrypt wysyła odpowiedź do Centrali

from __future__ import annotations
import unicodedata, json
from labs.ai_agents_framework import CentralaTask, FlagDetector, LLMClient
from labs.task_utils import TaskRunner, verify_environment

# ---------------------------------------------------------------------------
# RĘCZNIE ZDEKODOWANY „list od życia” (z wykorzystaniem par A#S#)
DECODED_MSG = (
    "Świat sprzed setek lat. "
    "Wizja świata idealnego. "
    "Możemy przeczytać o tym w książkach. "
    "Coś poszło niezgodnie z planem. "
    "Śmierć ludzi."
)

class LifeLetterManual(CentralaTask):
    """Sekret 8 w wersji bez OCR – korzysta z gotowego tekstu."""

    # ---------- zapytanie do LLM ------------------------------------------
    @staticmethod
    def _ask_place(llm: LLMClient, decoded: str) -> str:
        system = (
            "Jesteś historykiem-kryptologiem. Na podstawie krótkiej notki "
            "wskaż JEDNYM słowem nazwę miejsca, o którym mowa. "
            "Zwróć tylko to słowo, wielkimi literami, bez polskich znaków."
        )
        user = f"Notka:\n\"{decoded}\""
        answer = llm.answer_with_context(user, system, model="gpt-4o", max_tokens=8)
        return answer.strip()

    # ---------- main -------------------------------------------------------
    def execute(self):
        try:
            # 1) pytamy LLM-a
            place_raw = self._ask_place(self.llm_client, DECODED_MSG)
            place = unicodedata.normalize("NFKD", place_raw)\
                    .encode("ascii", "ignore").decode().upper()
            print(f"LLM wskazał: {place}")

            # 2) przesyłamy do Centrali
            response = self.submit_report("lifeletter", place)
            flag = FlagDetector.find_flag(json.dumps(response))
            if flag:
                return {"status": "success", "flag": flag, "place": place}

            return {"status": "failed", "server_response": response, "place": place}

        except Exception as exc:
            return self._handle_error(exc)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    verify_environment()
    res = TaskRunner().run_task(LifeLetterManual)
    TaskRunner().print_result(res, "SECRET 08 – Life Letter (manual)")
    if res.get("flag"):
        print("\nFLAG →", res["flag"])
