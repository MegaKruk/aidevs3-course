# documents_task.py  – v3 (LLM-driven wykrywanie zawodów + schludny prompt)
from __future__ import annotations
import re, textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

from ai_agents_framework import CentralaTask, LLMClient, FlagDetector
from task_utils import TaskRunner, verify_environment, slurp, extract_persons

DATA_DIR  = Path("../data/pliki_z_fabryki")
FACTS_DIR = DATA_DIR / "facts"
REPORT_RE = re.compile(r"\d{4}-\d{2}-\d{2}_report-\d+-sektor_[A-Z]\d\.txt$", re.I)


class DocumentsTask(CentralaTask):
    def __init__(self, llm: LLMClient):
        super().__init__(llm)
        self.reports = [p for p in DATA_DIR.iterdir() if REPORT_RE.match(p.name)]
        self.facts: Dict[str, str] = {p.name: slurp(p) for p in FACTS_DIR.glob("*.txt")}

        self.person_to_facts: Dict[str, List[str]] = defaultdict(list)
        for fact_txt in self.facts.values():
            for person in extract_persons(fact_txt):
                self.person_to_facts[person].append(fact_txt)

    # ------------------------------------------------------------------ prompt builder
    @staticmethod
    def _prompt(rep_txt: str, facts: List[str], fname: str) -> str:
        facts_block = (
            textwrap.indent("\n".join(facts), "    ") if facts else "brak"
        )

        return textwrap.dedent(f"""
                Jesteś ekspertem ds. metadanych. Wygeneruj listę słów kluczowych (po polsku,
                mianownik, przecinki) opisujących raport bezpieczeństwa.
                Uwzględnij treść raportu, powiązane fakty oraz nazwę pliku. Dodaj istotne
                zawody i technologie.

                ### RAPORT
                {rep_txt}

                ### FAKTY
                {facts_block}

                ### PLIK
                {fname}

                Zwróć tylko listę słów kluczowych.
            """).strip()

    # ------------------------- LLM-driven zawody / technologie -------------
    def _extra_kw(self, ctx: str) -> Set[str]:
        """Poproś LLM o jedno-dwu-wyrazowe nazwy zawodów / technologii."""
        prompt = textwrap.dedent(f"""
            Wyodrębnij z tekstu nazwy zawodów i technologii (język polski, mianownik,
            jedno lub dwa wyrazy, oddziel przecinkami). Pomiń pozostałe słowa.

            ### TEKST
            {ctx[:4000]}
        """).strip()
        raw = self.llm_client.answer_with_context(
            question="lista",
            context=prompt,
            model="gpt-4o",
            max_tokens=40
        )
        return {w.strip() for w in re.split(r",\s*", raw) if w.strip()}

    # ------------------------------------------------------------------ keywords
    def _keywords_for_report(self, path: Path) -> str:
        txt      = slurp(path)
        persons  = extract_persons(txt)
        rel_f    = [f for p in persons for f in self.person_to_facts.get(p, [])]

        # 1 LLM-generated keywords
        prompt   = self._prompt(txt, rel_f, path.name)
        kw_raw   = self.llm_client.answer_with_context(
            question="lista słów kluczowych",
            context=prompt,
            model="gpt-4o",
            max_tokens=120
        )
        kw_list  = [k.strip() for k in re.split(r",\s*", kw_raw) if k.strip()]

        # 2 extra professions / technologies (LLM)
        kw_list.extend(self._extra_kw(txt + "\n".join(rel_f)))

        # 3 sector from filename – always add explicitly
        m = re.search(r"-sektor[_-]?([A-Z]\d)\.txt$", path.name, re.I)
        if m:
            sector_kw = f"sektor {m.group(1).upper()}"
            kw_list.append(sector_kw)

        # 4 deduplicate (case-insensitive) keeping order
        seen: set[str] = set()
        final = [k for k in kw_list if not (k.lower() in seen or seen.add(k.lower()))]

        return ",".join(final)
    
    # ------------------------------------------------------------------ main
    def execute(self) -> Dict[str, Any]:
        try:
            answer = {}
            for rep in self.reports:
                print("Przetwarzam", rep.name)
                answer[rep.name] = self._keywords_for_report(rep)
                print("\t->", answer[rep.name])

            resp = self.http_client.submit_json(
                f"{self.base_url}/report",
                {"task": "dokumenty", "apikey": self.api_key, "answer": answer}
            )
            flag = FlagDetector.find_flag(str(resp))
            if flag:
                return {"status": "success", "flag": flag, "answer": answer}
            return {"status": "completed", "resp": resp, "answer": answer}

        except Exception as e:
            return self._handle_error(e)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    verify_environment()
    res = TaskRunner().run_task(DocumentsTask)
    TaskRunner().print_result(res, "DOCUMENTS TASK v3")
    if res.get("flag"):
        print("\nFLAG:", res["flag"])
