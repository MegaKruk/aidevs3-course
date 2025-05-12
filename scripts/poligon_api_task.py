import requests
import urllib3

TASK_NAME = "POLIGON"
API_KEY = "282681d1-0759-4803-adaf-719eaf031904"
DATA_URL = "https://poligon.aidevs.pl/dane.txt"
VERIFY_URL = "https://poligon.aidevs.pl/verify"

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

response = requests.get(DATA_URL, verify=False)
response.raise_for_status()
lines = response.text.strip().splitlines()

payload = {
    "task": TASK_NAME,
    "apikey": API_KEY,
    "answer": lines
}

verify_response = requests.post(VERIFY_URL, json=payload, verify=False)
verify_response.raise_for_status()
print(verify_response.json())
