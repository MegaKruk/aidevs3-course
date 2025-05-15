import requests, re, urllib3, pathlib
urllib3.disable_warnings()
s = requests.Session()

# 1. pobierz stronę logowania, wyciągnij pytanie
html = s.get("https://xyz.ag3nts.org/", verify=False).text
question = re.search(r'id="human-question".*?>(.*?)</', html, re.S).group(1)
print("QUESTION:", question.strip())

# 2. odpowiedz (użyj swojego ExpressionEvaluator lub LLM – tu na sztywno)
answer = "476"

payload = {"username": "tester",
           "password": "574e112a",
           "answer": answer}

resp = s.post("https://xyz.ag3nts.org/", data=payload, verify=False)
open("after_login.html", "w", encoding="utf‑8").write(resp.text)
print("HTML zapisany jako after_login.html")