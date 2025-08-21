"""
Indeksowanie raportów z weapons_tests.zip
-------------------------------------------------------
- czyta wszystkie *.txt z …/weapons_tests/do-not-share
- dzieli je na „inteligentne” chunki
- generuje:
      – embedding (OpenAI text-embedding-3-large → 3072 D)
      – znaczniki semantyczne (LLM)
      – krótki opis kontekstu chunku w ramach całego dokumentu (LLM)
- zapisuje całość do pliku JSONL („cache”)
- wrzuca punkty do Qdrant -> collection *weapons_tests*
- równolegle dodaje wiersze do Postgres tabeli *chunks*
      (UUID jest kluczem wspólnym dla obu baz)
"""

from __future__ import annotations
import json, os, re, uuid, datetime as dt
from pathlib import Path
from typing import Iterator, List, Dict, Any

import tiktoken                        # pip install tiktoken
import openai                          # pip install openai
from qdrant_client import QdrantClient # pip install qdrant-client
from qdrant_client.http.models import PointStruct, Distance, VectorParams
import psycopg2                        # pip install psycopg2-binary
from psycopg2.extras import execute_values
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
load_dotenv()
ROOT       = Path("../data/pliki_z_fabryki/weapons_tests/do-not-share")
JSONL_OUT  = Path("_vector_cache.jsonl")
EMB_MODEL  = "text-embedding-3-large"
EMB_DIM    = 3072
QDRANT_URL = os.getenv("QDRANT_URL")
COLL_NAME  = os.getenv("QDRANT_COLL")
DB_NAME    = os.getenv('PG_DB')
DB_USER    = os.getenv('PG_USER')
DB_PASS    = os.getenv('PG_PASSWORD')
DB_HOST    = os.getenv('PG_HOST')
DB_PORT    = os.getenv('PG_PORT')
OAI_KEY    = os.getenv("OPENAI_API_KEY")

PG_CONN_STR = f"dbname={DB_NAME} user={DB_USER} password={DB_PASS} host={DB_HOST} port={DB_PORT}"

openai_client = openai.OpenAI(api_key=OAI_KEY)
tokeniser     = tiktoken.encoding_for_model("gpt-4o")

# chunking -------------------------------------------------------------------------------
MAX_TOK  = 500  # ~ 1-2 paragrafy
OVERLAP  = 40

def paragraphs(text: str) -> List[str]:
    """Split on empty lines; keep newlines inside paragraphs intact."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

def split_into_chunks(para: str) -> List[str]:
    toks = tokeniser.encode(para)
    if len(toks) <= MAX_TOK:
        return [para]
    chunks = []
    start = 0
    while start < len(toks):
        end = min(start + MAX_TOK, len(toks))
        chunk_toks = toks[start:end]
        chunks.append(tokeniser.decode(chunk_toks))
        if end == len(toks):
            break
        start = end - OVERLAP        # slide with overlap
    return chunks

def iter_chunks(text: str) -> Iterator[str]:
    for idx_p, p in enumerate(paragraphs(text)):
        print(f"p[{idx_p}]: {p}")
        for chunk in split_into_chunks(p):
            print(f"\t{chunk}")
            yield chunk

# LLM helpers ------------------------------------------------------------------------------
def semantic_tags(text: str) -> List[str]:
    prompt = (
        "Wypisz w mianowniku, po polsku, pojedyncze rzeczowniki/krótkie frazy "
        "opisujące technologię, zdarzenia lub osoby w tekście (maks 8). "
        "Użyj przecinków.\n\n### TEKST\n" + text[:800]
    )
    resp = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=60,
        temperature=0.2,
    ).choices[0].message.content
    return [w.strip() for w in re.split(r",\s*", resp) if w.strip()]

def chunk_context(chunk: str, full_doc: str) -> str:
    prompt = (
        "Streść w jednym zdaniu, jak ten fragment wpisuje się w całość dokumentu."
        "\n\n### FRAGMENT\n" + chunk + "\n\n### CAŁY DOKUMENT\n" + full_doc[:1500]
    )
    return openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=60,
        temperature=0.2,
    ).choices[0].message.content.strip()

def embed(text: str) -> List[float]:
    return openai_client.embeddings.create(
        model=EMB_MODEL,
        input=text.replace("\n", " "),
    ).data[0].embedding

# Postgres DDL ---------------------------------------------------------------------------------
def ensure_pg_table(cur):
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id          UUID PRIMARY KEY,
            filename    TEXT,
            report_date DATE,
            tags        TEXT[],
            context     TEXT,
            content     TEXT,
            created_on  TIMESTAMP DEFAULT now()
        );
    """)

# Qdrant collection ----------------------------------------------------------------------------
def ensure_qdrant(client: QdrantClient):
    if COLL_NAME not in [c.name for c in client.get_collections().collections]:
        client.create_collection(
            collection_name=COLL_NAME,
            vectors_config=VectorParams(size=EMB_DIM, distance=Distance.COSINE),
        )

# main -----------------------------------------------------------------------------------------
def process_file(fp: Path) -> List[Dict[str, Any]]:
    date = fp.stem.replace("_", "-")          # 2024_01_29 -> 2024-01-29
    text = fp.read_text(encoding="utf-8", errors="ignore")
    all_chunks = list(iter_chunks(text))
    tags_doc  = semantic_tags(text)

    records = []
    for ch in all_chunks:
        uid = uuid.uuid4()
        print(f"uid: {uid}")
        tags = list(set(tags_doc + semantic_tags(ch)))
        print(f"tags: {tags}")
        context = chunk_context(ch, text)
        print(f"context: {context}")
        rec  = {
            "id": str(uid),
            "filename": fp.name,
            "report_date": date,
            "tags": tags,
            "context": context,
            "content": ch,
            "embedding": embed(ch),
        }
        print(f"rec: {rec}")
        records.append(rec)
    return records

def main():
    # ---------- gather & cache ------------------------------------------------
    if JSONL_OUT.exists():
        print("Cache found – loading", JSONL_OUT)
        records = [json.loads(l) for l in JSONL_OUT.read_text().splitlines()]
    else:
        records = []
        for fp in sorted(ROOT.glob("*.txt")):
            print("Processing", fp.name)
            records += process_file(fp)
        with JSONL_OUT.open("w", encoding="utf-8") as f:
            for r in records:
                json.dump(r, f, ensure_ascii=False)
                f.write("\n")
        print("Saved cache to", JSONL_OUT)

    # ---------- Qdrant --------------------------------------------------------
    qc = QdrantClient(url=QDRANT_URL, timeout=30)
    ensure_qdrant(qc)

    points = [
        PointStruct(id=rec["id"], vector=rec["embedding"],
                    payload={"filename": rec["filename"],
                             "report_date": rec["report_date"],
                             "tags": rec["tags"]})
        for rec in records
    ]
    print("Uploading", len(points), "points to Qdrant …")
    qc.upsert(collection_name=COLL_NAME, points=points)

    # ---------- Postgres ------------------------------------------------------
    with psycopg2.connect(PG_CONN_STR) as conn, conn.cursor() as cur:
        ensure_pg_table(cur)
        rows = [
            (rec["id"], rec["filename"], rec["report_date"],
             rec["tags"], rec["context"], rec["content"])
            for rec in records
        ]
        execute_values(
            cur,
            "INSERT INTO chunks (id, filename, report_date, tags, context, content) "
            "VALUES %s ON CONFLICT (id) DO NOTHING",
            rows,
        )
        conn.commit()
    print("Postgres up-to-date. DONE.")

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
