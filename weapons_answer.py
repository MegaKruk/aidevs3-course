"""
Vector search -> odczyt z PostgreSQL -> wysłanie daty do Centrali
"""

from __future__ import annotations
import os, json, psycopg2
from typing import Dict, Any
from dotenv import load_dotenv
import openai
from qdrant_client import QdrantClient
from ai_agents_framework import FlagDetector, HttpClient

load_dotenv()

# ------------ konfiguracja --------------------------------------------------
EMB_MODEL   = "text-embedding-3-large"
QDRANT_URL  = os.getenv("QDRANT_URL", "http://localhost:6333")
COLL_NAME   = os.getenv("QDRANT_COLL", "weapons_tests")

PG_CONN     = psycopg2.connect(
    host=os.getenv("PG_HOST", "localhost"),
    port=os.getenv("PG_PORT", "5432"),
    dbname=os.getenv("PG_DB",  "vectors"),
    user=os.getenv("PG_USER",  "vector_user"),
    password=os.getenv("PG_PASSWORD", "vector_pass")
)
PG_CONN.autocommit = True

API_KEY     = os.getenv("CENTRALA_API_KEY")
BASE_URL    = "https://c3ntrala.ag3nts.org"

QUESTION = ("W raporcie, z którego dnia znajduje się wzmianka "
            "o kradzieży prototypu broni?")

oai = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qc  = QdrantClient(url=QDRANT_URL)

# ------------ helpers -------------------------------------------------------
def embed(txt: str) -> list[float]:
    return oai.embeddings.create(model=EMB_MODEL, input=txt).data[0].embedding

def fetch_chunk(uuid_str: str) -> Dict[str, Any] | None:
    """Pobiera wiersz z tabeli *chunks* po ID (UUID)."""
    with PG_CONN.cursor() as cur:
        cur.execute(
            """SELECT filename, report_date, tags, context, content
               FROM chunks WHERE id = %s""",
            (uuid_str,)
        )
        row = cur.fetchone()
        if row:
            filename, rpt_date, tags, ctx, content = row
            return {
                "filename": filename,
                "report_date": str(rpt_date),
                "tags": tags,
                "context": ctx,
                "content": content[:400] + ("…" if len(content) > 400 else "")
            }
    return None

# ------------ main ----------------------------------------------------------
def main():
    # --- wektorowe wyszukiwanie ---
    vector = embed(QUESTION)
    hits   = qc.search(collection_name=COLL_NAME,
                       query_vector=vector,
                       limit=1,
                       with_payload=True)

    if not hits:
        raise RuntimeError("Brak wyników w Qdrant.")

    hit      = hits[0]
    hit_uuid = str(hit.id)
    date     = hit.payload["report_date"]

    print(f"Najlepszy traf:\n{hit}")
    print("Data raportu:", date)

    # --- odczyt wiersza z Postgresa ---
    chunk_row = fetch_chunk(hit_uuid)
    if chunk_row:
        print("\n=== Wiersz z tabeli chunks ===")
        print(json.dumps(chunk_row, ensure_ascii=False, indent=2))
    else:
        print("\nUWAGA: Nie znaleziono wiersza w Postgres dla uuid =", hit_uuid)

    # --- wysyłka do Centrali ---
    payload: Dict[str, Any] = {
        "task": "wektory",
        "apikey": API_KEY,
        "answer": date,
    }
    print("\nWysyłam do Centrali:", json.dumps(payload, indent=2, ensure_ascii=False))
    resp = HttpClient().submit_json(f"{BASE_URL}/report", payload)
    print("Odpowiedź Centrali:", resp)

    flag = FlagDetector.find_flag(json.dumps(resp))
    if flag:
        print("\nFLAGA:", flag)

# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
