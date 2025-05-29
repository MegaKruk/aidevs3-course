"""
S03E04  –  Zadanie „loop”  (Barbara Zawadzka)

Kroki:
1.  pobierz notatkę barbara.txt i wyciągnij imiona + miasta
2.  breadth-first search przez API /people oraz /places
3.  gdy w odpowiedzi z /places pojawi się BARBARA w nowym mieście ⇒ success
4.  dodatkowo: znajdź
      • współpracownika Aleksandra i Barbary
      • osobę, z którą widział się Rafał
"""

from __future__ import annotations
import os, re, json, collections
from typing import Set, Dict
from dotenv import load_dotenv
from ai_agents_framework import (
    LLMClient, HttpClient, FlagDetector, PeoplePlacesAPI
)

load_dotenv()
API_KEY    = os.getenv("CENTRALA_API_KEY")
BASE_URL   = "https://c3ntrala.ag3nts.org"
NOTE_URL   = f"{BASE_URL}/dane/barbara.txt"
REPORT_URL = f"{BASE_URL}/report"
TASK_NAME  = "loop"

http = HttpClient()
llm = LLMClient()
pp_api = PeoplePlacesAPI(API_KEY)

# -----------------------------------------------------------------------------
def download_note() -> str:
    print("Pobieram notatkę:", NOTE_URL)
    txt = http.get(NOTE_URL).text
    print(f"Pobrano {len(txt)} znaków:")
    print(txt)
    return txt

NAME_RE = re.compile(r"\b[A-ZŁŚŻŹĆ][a-ząćęłńóśżź]+(?:\s+[A-Z][a-z]+)?\b")
CITY_RE = re.compile(r"\b[A-ZŻŹŁŚĆÓ]{3,}\b")   # uproszczone – w notatce miasta są wielkimi?

def extract_names_cities(note: str) -> tuple[set[str], set[str]]:
    """
    Use GPT-4o to pull clean PEOPLE / CITIES lists out of *note*.
    Accepts both raw JSON and ```json … ``` fenced blocks.
    """
    sys_prompt = (
        "You are an information-extraction assistant. "
        "Return valid JSON with two arrays: PEOPLE and CITIES. "
        "Rules:\n"
        "- Only first names (no surnames) in nominative case go to PEOPLE.\n"
        "- Only Polish city names WITHOUT diacritics go to CITIES.\n"
        "- No duplicates. No commentary."
    )

    resp = llm.answer_with_context(
        question=note,
        context=sys_prompt,
        model="gpt-4o",
        max_tokens=400
    ).strip()

    # strip optional ```json … ``` or ``` … ``` fences
    resp = re.sub(r"^```(?:json)?\s*|\s*```$", "", resp, flags=re.I | re.S).strip()

    try:
        data = json.loads(resp)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"LLM did not return valid JSON.\nReply was:\n{resp}") from e

    people = {pp_api.normalise_name(n) for n in data.get("PEOPLE", [])}
    cities = {pp_api.normalise_city(c) for c in data.get("CITIES", [])}
    return people, cities

# -----------------------------------------------------------------------------
def bfs(names0: Set[str], cities0: Set[str]):
    people_q  = collections.deque(sorted(n for n in names0 if n))
    cities_q  = collections.deque(sorted(c for c in cities0 if c))

    seen_people: Set[str] = set(people_q)
    seen_cities: Set[str] = set(cities_q)

    # mapping helpers
    person_cities: Dict[str, Set[str]] = collections.defaultdict(set)
    city_people: Dict[str, Set[str]] = collections.defaultdict(set)

    while people_q or cities_q:
        if people_q:
            name = people_q.popleft()
            places = pp_api.query_people(name)
            for city in places:
                if city not in seen_cities:
                    seen_cities.add(city)
                    cities_q.append(city)
                person_cities[name].add(city)
                city_people[city].add(name)

        if cities_q:
            city = cities_q.popleft()
            persons = pp_api.query_places(city)
            # --- SUKCES? ---
            if "BARBARA" in persons and city not in cities0:
                print(f"\n===> BARBARA widziana w NOWYM mieście: {city}")
                return city, person_cities, city_people
            for p in persons:
                if p not in seen_people:
                    seen_people.add(p)
                    people_q.append(p)
                person_cities[p].add(city)
                city_people[city].add(p)

    raise RuntimeError("Nie znaleziono Barbary w nowym mieście.")

# -----------------------------------------------------------------------------
def extra_answers(person_cities: Dict[str, Set[str]],
                  city_people: Dict[str, Set[str]]) -> tuple[str|None, str|None]:
    # współpracownik Aleksandra i Barbary → ktoś obecny w KAŻDYM mieście,
    # w którym występuje zarówno Aleksander, jak i Barbara
    aleks = "ALEKSANDER"
    barb  = "BARBARA"
    common = None
    if aleks in person_cities and barb in person_cities:
        shared_cities = person_cities[aleks] & person_cities[barb]
        cand: Dict[str, int] = collections.Counter()
        for c in shared_cities:
            for p in city_people[c]:
                if p not in {aleks, barb}:
                    cand[p] += 1
        if cand:
            common = max(cand, key=cand.get)

    # z kim widział się Rafał?  – ktokolwiek współwystępuje z nim w jakimś mieście
    rafael = "RAFAL"
    meet = None
    if rafael in person_cities:
        for c in person_cities[rafael]:
            others = city_people[c] - {rafael}
            if others:
                meet = next(iter(sorted(others)))
                break
    return common, meet

# -----------------------------------------------------------------------------
def submit(city: str):
    payload = {"task": TASK_NAME, "apikey": API_KEY, "answer": city}
    print("\nWysyłka do /report …")
    print(json.dumps(payload, indent=2))
    resp = http.submit_json(REPORT_URL, payload)
    print("Odpowiedź Centrali:", resp)
    flag = FlagDetector.find_flag(json.dumps(resp))
    if flag:
        print("FLAGA:", flag)

# -----------------------------------------------------------------------------
def main():
    note = download_note()
    names0, cities0 = extract_names_cities(note)
    print("Imiona z notatki:", names0)
    print("Miasta z notatki:", cities0)

    city, person_cities, city_people = bfs(names0, cities0)
    submit(city)

    wsp, raf = extra_answers(person_cities, city_people)
    print("\n--- Dodatkowe odpowiedzi ---")
    print("Współpracownik Aleksandra i Barbary:", wsp or "nie znaleziono")
    print("Z kim widział się Rafał:", raf or "nie znaleziono")

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
