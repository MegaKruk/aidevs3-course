from qdrant_client import QdrantClient
import psycopg2, random, os, json
from dotenv import load_dotenv

load_dotenv()

# --- Qdrant -----------------------------------------------------------------
qc = QdrantClient(url=os.getenv("QDRANT_URL"))
points = qc.count(os.getenv("QDRANT_COLL"), exact=True).count
print("Qdrant points:", points)

# --- Postgres ---------------------------------------------------------------
conn = psycopg2.connect(
    dbname=os.getenv("PG_DB"),
    user=os.getenv("PG_USER"),
    password=os.getenv("PG_PASSWORD"),
    host=os.getenv("PG_HOST"),
    port=os.getenv("PG_PORT"),
)
cur = conn.cursor()
cur.execute("SELECT COUNT(*) FROM chunks;")
print("Postgres rows:", cur.fetchone()[0])

# — porównanie losowego id ---------------------------------------------------
cur.execute("SELECT id, filename, report_date FROM chunks LIMIT 100;")
ids = [r[0] for r in cur.fetchall()]
sample_id = random.choice(ids)
print("\nPrzykładowy UUID:", sample_id)

cur.execute("SELECT filename, report_date FROM chunks WHERE id = %s", (sample_id,))
print("-> Postgres:", cur.fetchone())

pt = qc.retrieve(os.getenv("QDRANT_COLL"), [sample_id])[0]
print("-> Qdrant :", pt.payload["filename"], pt.payload["report_date"])
