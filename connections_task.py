"""
S03E05  –  "connections"
Find the shortest path from Rafał to Barbara using a Neo4j graph.

Steps
-----
1.  Inspect MySQL schema of `users` and `connections` via /apidb.
2.  Ask GPT-4o to generate two SELECT queries that expose:
        • users →  id,   first name  (aliased as "name")
        • connections →  src_id, dst_id (aliased as "a", "b")
3.  Load rows into local Neo4j (docker-ised).
4.  Cypher: shortestPath  Rafał → Barbara.
5.  Report comma-separated list to /report.
"""

from __future__ import annotations
import json, os, re, textwrap, sys, time
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from neo4j import GraphDatabase

from ai_agents_framework import (
    LLMClient,
    CentralaDatabaseAPI,
    HttpClient,
    FlagDetector,
)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()

API_KEY  = os.getenv("CENTRALA_API_KEY")
if not API_KEY:
    print("Set CENTRALA_API_KEY in .env", file=sys.stderr)
    sys.exit(1)

DB       = CentralaDatabaseAPI(api_key=API_KEY, verbose=False)
LLM      = LLMClient()
TASK     = "connections"
REPORT   = "https://c3ntrala.ag3nts.org/report"

BOLT_URL = os.getenv("NEO4J_URI",    "bolt://localhost:7687")
BOLT_USR = os.getenv("NEO4J_USER",   "neo4j")
BOLT_PWD = os.getenv("NEO4J_PASSWORD",   "neo4j")

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def llm_sql(schema: str, requirement: str) -> str:
    """Ask GPT-4o to build a one-liner SELECT … with proper aliases."""
    system = """You are an expert MySQL assistant.
Return ONE plain SQL statement (no comments, no ``` fences).
Do not include semicolons at line ends."""
    prompt = f"SCHEMA:\n{schema}\n\nREQUIREMENT:\n{return_one(requirement)}"
    sql = LLM.answer_with_context(
        question="SQL only",
        context=system + "\n\n" + prompt,
        model="gpt-4o",
        max_tokens=150,
    )
    sql = re.sub(r"^```sql|```$", "", sql, flags=re.I).strip()
    return sql

def return_one(text: str) -> str:
    """Single-line helper (keeps README width sane)."""
    return " ".join(textwrap.dedent(text).strip().split())

def fetch_users() -> Dict[int, str]:
    show = DB.query("SHOW CREATE TABLE users;")[0]["Create Table"]
    want = ("Select `id` and *the column that stores FIRST NAME* "
            "from table `users`.\n"
            "Alias first name as `name`.")
    sel  = llm_sql(show, want)
    print("\n--- users SQL ---\n", sel)
    rows = DB.query(sel)
    return {int(r["id"]): r["name"] for r in rows}

def fetch_edges() -> List[tuple[int, int]]:
    show = DB.query("SHOW CREATE TABLE connections;")[0]["Create Table"]
    want = ("Select the TWO user-id columns that form an undirected edge "
            "between people.\nAlias them as `a` and `b`.")
    sel  = llm_sql(show, want)
    print("\n--- edges SQL ---\n", sel)
    rows = DB.query(sel)
    return [(int(r["a"]), int(r["b"])) for r in rows]

# ──────────────────────────────────────────────────────────────────────────────
# Neo4j load
# ──────────────────────────────────────────────────────────────────────────────
def load_into_neo4j(users: Dict[int, str], edges: List[tuple[int, int]]) -> None:
    driver = GraphDatabase.driver(BOLT_URL, auth=(BOLT_USR, BOLT_PWD))
    with driver.session() as sess:
        # wipe previous run (keeps retry idempotent)
        sess.run("MATCH (n) DETACH DELETE n")

        # batch import nodes
        for uid, uname in users.items():
            sess.run(
                "CREATE (:Person {uid:$uid, name:$name})",
                uid=uid, name=uname
            )

        # batch import edges
        for a, b in edges:
            sess.run(
                """
                MATCH (u:Person {uid:$a}), (v:Person {uid:$b})
                MERGE (u)-[:KNOWS]->(v)
                MERGE (v)-[:KNOWS]->(u)
                """,
                a=a, b=b,
            )
    driver.close()

# ──────────────────────────────────────────────────────────────────────────────
# Shortest path
# ──────────────────────────────────────────────────────────────────────────────
def shortest_rafal_to_barbara() -> List[str]:
    driver = GraphDatabase.driver(BOLT_URL, auth=(BOLT_USR, BOLT_PWD))
    cypher = """
    MATCH (r:Person), (b:Person)
    WHERE toUpper(r.name) STARTS WITH 'RAFA' AND toUpper(b.name) STARTS WITH 'BARB'
    MATCH p = shortestPath((r)-[:KNOWS*]-(b))
    RETURN [n IN nodes(p) | n.name] AS names
    """
    with driver.session() as sess:
        rec = sess.run(cypher).single()
        if not rec:
            raise RuntimeError("No path found between Rafał and Barbara.")
        return rec["names"]

# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────
def submit(answer: str) -> None:
    payload = {"task": TASK, "apikey": API_KEY, "answer": answer}
    print("\nSubmitting:\n", json.dumps(payload, indent=2, ensure_ascii=False))
    resp = HttpClient().submit_json(REPORT, payload)
    print("Central reply:", resp)
    flag = FlagDetector.find_flag(json.dumps(resp))
    if flag:
        print("FLAG:", flag)

# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("Pulling data from MySQL …")
    users  = fetch_users()
    edges  = fetch_edges()

    print(f"Users: {len(users):,}   edges: {len(edges):,}")
    load_into_neo4j(users, edges)
    time.sleep(1)        # tiny pause, let Neo4j commit

    names = shortest_rafal_to_barbara()
    print("Path:", names)

    answer = ",".join(names)
    submit(answer)

# Entrypoint
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
