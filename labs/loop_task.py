"""
S03E04  –  Zadanie „loop”  (Barbara Zawadzka)

1. Pobierz notatkę barbara.txt, wyciągnij imiona + miasta.
2. BFS po /people & /places aż przejdziemy CAŁĄ sieć, logując surowe
   odpowiedzi (może kryć się tam sekretna flaga „upadnij nisko”).
3. Główna odpowiedź: miasto, w którym *aktualnie* (nowe) widziano BARBARĘ.
4. Dodatkowo: współpracownik Aleksandra & Barbary oraz osoba, z którą
   widział się Rafał.
"""

from __future__ import annotations
import os, re, json, collections, sys
from typing import Set, Dict, Tuple
from pathlib import Path
from dotenv import load_dotenv
from ai_agents_framework import (
    LLMClient, HttpClient, FlagDetector, PeoplePlacesAPI
)

load_dotenv()
API_KEY     = os.getenv("CENTRALA_API_KEY")
BASE_URL    = "https://c3ntrala.ag3nts.org"
NOTE_URL    = f"{BASE_URL}/dane/barbara.txt"
REPORT_URL  = f"{BASE_URL}/report"
TASK_NAME   = "loop"

RAW_DIR = Path("../_secret_payloads")     # ensure visible also here

http  = HttpClient()
llm   = LLMClient()
ppapi = PeoplePlacesAPI(API_KEY)

# ──────────────────────────────────────────────────────────────────────────────
def download_note() -> str:
    print("Fetching note:", NOTE_URL)
    txt = http.get(NOTE_URL).text
    print(f"...{len(txt)} chars downloaded:")
    print(txt)
    return txt

# ──────────────────────────────────────────────────────────────────────────────
def extract_names_cities(note: str) -> Tuple[set[str], set[str]]:
    """
    Ask GPT-4o to produce clean JSON with PEOPLE / CITIES lists.
    """
    system = (
        "You are an information-extraction assistant.\n"
        "Return *valid JSON* with exactly two arrays: PEOPLE and CITIES.\n"
        "- PEOPLE -> ONLY Polish first names, nominative, no diacritics\n"
        "- CITIES -> ONLY Polish city names, no diacritics\n"
        "- No duplicates, no commentary"
    )
    resp = llm.answer_with_context(
        question=note, context=system, model="gpt-4o", max_tokens=400
    ).strip()
    resp = re.sub(r"^```(?:json)?\s*|\s*```$", "", resp, flags=re.I | re.S).strip()

    try:
        data = json.loads(resp)
    except Exception as e:
        print("LLM reply was not valid JSON:\n", resp, file=sys.stderr)
        raise

    people = {ppapi.normalise_name(n) for n in data.get("PEOPLE", [])}
    cities = {ppapi.normalise_city(c) for c in data.get("CITIES", [])}
    return people, cities

# ──────────────────────────────────────────────────────────────────────────────
def crawl(names0: Set[str], cities0: Set[str]) -> Tuple[str, Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Full BFS – *no early exit*; returns first new-city-with-Barbara, plus maps.
    """
    q_people = collections.deque(sorted(names0))
    q_cities = collections.deque(sorted(cities0))
    seen_p, seen_c = set(q_people), set(q_cities)

    person_cities: Dict[str, Set[str]] = collections.defaultdict(set)
    city_people:   Dict[str, Set[str]] = collections.defaultdict(set)

    first_city_with_barbara = None

    while q_people or q_cities:
        # ----- expand by person -> cities -----
        if q_people:
            p = q_people.popleft()
            for city in ppapi.query_people(p):
                person_cities[p].add(city)
                city_people[city].add(p)
                if city not in seen_c:
                    seen_c.add(city)
                    q_cities.append(city)

        # ----- expand by city -> persons ------
        if q_cities:
            c = q_cities.popleft()
            for person in ppapi.query_places(c):
                city_people[c].add(person)
                person_cities[person].add(c)
                if person not in seen_p:
                    seen_p.add(person)
                    q_people.append(person)

            if "BARBARA" in city_people[c] and c not in cities0 and not first_city_with_barbara:
                first_city_with_barbara = c
                print(f"\nFirst new city with BARBARA: {c}")

    if not first_city_with_barbara:
        raise RuntimeError("Barbara not spotted in a new city.")
    return first_city_with_barbara, person_cities, city_people

# ──────────────────────────────────────────────────────────────────────────────
def extra_answers(pc: Dict[str, Set[str]], cp: Dict[str, Set[str]]) -> Tuple[str|None, str|None]:
    # wspólnik
    a, b = "ALEKSANDER", "BARBARA"
    collaborator = None
    common_cities = pc[a] & pc[b] if a in pc and b in pc else set()
    if common_cities:
        counter = collections.Counter(
            x for city in common_cities for x in cp[city] if x not in {a, b}
        )
        if counter:
            collaborator = counter.most_common(1)[0][0]

    # znajomy Rafała
    meet = None
    if "RAFAL" in pc:
        for city in pc["RAFAL"]:
            others = cp[city] - {"RAFAL"}
            if others:
                meet = sorted(others)[0]
                break
    return collaborator, meet

# ──────────────────────────────────────────────────────────────────────────────
def submit(city: str):
    payload = {"task": TASK_NAME, "apikey": API_KEY, "answer": city}
    print("\nSubmitting:")
    print(json.dumps(payload, indent=2))
    resp = http.submit_json(REPORT_URL, payload)
    print("Central reply:", resp)
    FlagDetector.find_flag(json.dumps(resp))

# ──────────────────────────────────────────────────────────────────────────────
def main():
    note = download_note()
    names0, cities0 = extract_names_cities(note)

    print("People in note :", names0)
    print("Cities in note :", cities0)

    new_city, pc, cp = crawl(names0, cities0)
    submit(new_city)

    collab, meet = extra_answers(pc, cp)
    print("\n--- Extra answers ---")
    print("Współpracownik Aleksandra i Barbary:", collab or "nie znaleziono")
    print("Z kim widział się Rafał:", meet or "nie znaleziono")

    print("\nCheck _secret_payloads/ for hidden artefacts!")

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
