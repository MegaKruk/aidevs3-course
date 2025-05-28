"""
Rozwiązanie zadania „database”.

Kroki:
1.   pobierz listę tabel i schematy (SHOW CREATE TABLE)
2.   z pomocą LLM wygeneruj zapytanie SQL zwracające ID
     aktywnych datacenter zarządzanych przez nieaktywnych menadżerów
3.   wykonaj SQL, przetwórz wynik, odeślij odpowiedź do Centrali
"""

from __future__ import annotations
import json, os, re, textwrap
from typing import List, Dict, Any
from dotenv import load_dotenv
from ai_agents_framework import (
    LLMClient, CentralaDatabaseAPI, HttpClient, FlagDetector
)

load_dotenv()

# --------------------------------------------------------------------------- konfiguracja
API_KEY      = os.getenv("CENTRALA_API_KEY")
DB_API       = CentralaDatabaseAPI(api_key=API_KEY, verbose=True)
LLM          = LLMClient()

REPORT_URL   = "https://c3ntrala.ag3nts.org/report"
TASK_NAME    = "database"

# --------------------------------------------------------------------------- helpers
def llm_sql(schema: str, question: str) -> str:
    """Generuje surowy (bez Markdown) SQL przy użyciu GPT-4o."""
    system = textwrap.dedent("""
        You are an expert SQL assistant.
        Produce a single valid MySQL SELECT statement answering the user request.
        Do not return any commentary, no code fences, no semicolons at the ends,
        only the query.
    """).strip()

    prompt = f"SCHEMA:\n{schema}\n\nREQUEST:\n{question}"
    sql = LLM.answer_with_context(question="SQL only", context=system + "\n\n" + prompt,
                                  model="gpt-4o", max_tokens=200)
    # wyczyść możliwe otoczki ```
    sql = re.sub(r"^```sql|```$", "", sql, flags=re.I).strip()
    return sql

def submit_answer(ids: List[int]) -> Dict[str, Any]:
    payload = {"task": TASK_NAME, "apikey": API_KEY, "answer": ids}
    print("\n--- Wysyłka do /report ---")
    print(json.dumps(payload, indent=2))
    return HttpClient().submit_json(REPORT_URL, payload)

# --------------------------------------------------------------------------- main
def main() -> None:
    # 1) zbadanie schematów
    tables = [row["Tables_in_banan"] for row in DB_API.query("SHOW TABLES;")]
    wanted = [t for t in tables if t in {"users", "datacenters"}]  # 'connections' niepotrzebne tutaj
    schema_parts = []
    for t in wanted:
        cre = DB_API.query(f"SHOW CREATE TABLE {t};")[0]["Create Table"]
        schema_parts.append(cre)
    full_schema = "\n\n".join(schema_parts)

    # 2) zapytanie do LLM-a o SQL
    question = ("Zwróć DC_ID (lub odpowiednią kolumnę) wszystkich AKTYWNYCH "
                "datacenter, których managerowie (tabela users) mają status "
                "nieaktywny/urlopowy.")
    sql_query = llm_sql(full_schema, question)
    print("\n--- SQL od LLM ---")
    print(sql_query)

    # 3) wykonanie SQL
    result = DB_API.query(sql_query)
    if not result:
        raise RuntimeError("Zapytanie zwróciło pusty zestaw danych")

    # wydobycie liczb (spodziewamy się 1 kolumny)
    first_key = list(result[0].keys())[0]
    ids = [int(row[first_key]) for row in result]
    print("\nWynik id:", ids)

    # 4) przesłanie odpowiedzi
    resp = submit_answer(ids)
    print("\nOdpowiedź Centrali:", resp)
    flag = FlagDetector.find_flag(json.dumps(resp))
    if flag:
        print("FLAGA:", flag)

# --------------------------------------------------------------------------- entry
if __name__ == "__main__":
    main()
