"""
Pobiera flagę z ukrytej tabeli `correct_order`
w bazie BanAN poprzez publiczne API Centrali.

Wykonuje:
1. SHOW CREATE TABLE correct_order; -> walidacja nazw kolumn
2. SELECT letter, weight FROM correct_order;
3. Sortuje po weight (liczbowo) i skleja litery.
"""

from __future__ import annotations

import json
import os
import sys
from dotenv import load_dotenv
from labs.ai_agents_framework import HttpClient


class DatabaseAPI:
    """
    Minimalna nakładka na endpoint /apidb
    """

    def __init__(
        self,
        task: str,
        apikey: str,
        url: str = "https://c3ntrala.ag3nts.org/apidb",
        verbose: bool = False,
    ):
        self.task = task
        self.apikey = apikey
        self.url = url
        self.verbose = verbose
        self.http = HttpClient()

    def query(self, sql: str):
        payload = {"task": self.task, "apikey": self.apikey, "query": sql}
        if self.verbose:
            print("\n--- SQL ---")
            print(sql)
            print("------------")
            print("payload:", payload)
        resp = self.http.submit_json(self.url, payload)
        if self.verbose:
            print("response:", json.dumps(resp, indent=2, ensure_ascii=False))
        if resp.get("error") not in ("OK", "", None):
            raise RuntimeError(f"DB error: {resp['error']}")
        return resp["reply"]


def main(verbose: bool = False):
    load_dotenv()
    api_key = os.getenv("CENTRALA_API_KEY")
    if not api_key:
        sys.exit("Brak CENTRALA_API_KEY w .env")

    db = DatabaseAPI(task="database", apikey=api_key, verbose=verbose)

    # 1. Walidacja schematu (opcjonalna, ale bezpieczniejsza)
    schema_row = db.query("SHOW CREATE TABLE correct_order;")[0]
    create_sql = schema_row["Create Table"].lower()
    if "letter" not in create_sql or "weight" not in create_sql:
        print("Nieoczekiwane nazwy kolumn w correct_order:")
        print(create_sql)
        sys.exit(1)

    # 2. Pobranie liter i wag
    rows = db.query("SELECT letter, weight FROM correct_order;")

    # 3. Sortowanie po weight (as int) i składanie flagi
    flag = "".join(
        row["letter"] for row in sorted(rows, key=lambda r: int(r["weight"]))
    )

    print("Sekret 10 =", flag)


if __name__ == "__main__":
    main(verbose=True)
