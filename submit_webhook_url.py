import os, json
from ai_agents_framework import HttpClient
from dotenv import load_dotenv

load_dotenv()
APIKEY = os.getenv("CENTRALA_API_KEY")
if not APIKEY:
    raise RuntimeError("Brak CENTRALA_API_KEY w .env")

WEBHOOK_URL = "https://50a4-5-173-234-153.ngrok-free.app/drone"

payload = {
    "task": "webhook",
    "apikey": APIKEY,
    "answer": WEBHOOK_URL,
}

resp = HttpClient().submit_json("https://c3ntrala.ag3nts.org/report", payload)
print("Centrala reply:", json.dumps(resp, indent=2, ensure_ascii=False))
