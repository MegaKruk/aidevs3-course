from __future__ import annotations
import json, re
from typing import List, Tuple

from ai_agents_framework import LLMClient

# 4x4 opis terenu  Indeksy: GRID[row][col]  (row=0 góra, col=0 lewa)
GRID: list[list[str]] = [
    ["start",   "trawa",   "drzewo",   "dom"],
    ["trawa",   "wiatrak", "trawa",    "trawa"],
    ["trawa",   "trawa",   "skały",    "dwa drzewa"],
    ["skały",   "skały",   "samochód", "jaskinia"],
]

DIR_VEC = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}

class DroneMap:
    """Niewielka klasa narzędziowa do poruszania się po stałej siatce 4×4."""

    def __init__(self) -> None:
        self.rows, self.cols = 4, 4

    def move(self, xy: Tuple[int, int], direction: str, steps: int) -> Tuple[int, int]:
        dr, dc = DIR_VEC[direction]
        r, c = xy
        for _ in range(steps):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                r, c = nr, nc
            # jeżeli wyjście poza mapę – ignorujemy ruch
        return r, c

    def describe(self, xy: Tuple[int, int]) -> str:
        r, c = xy
        return GRID[r][c]

# ----------------------------------------------------------------------
_llm = LLMClient()  # używamy globalnie, by nie robić wielu połączeń

_SYSTEM_PROMPT = (
    "Jesteś parserem poleceń lotu drona po siatce 4×4. "
    "Masz zwrócić listę ruchów w formie JSON: "
    "[[\"UP\",3],[\"RIGHT\",1],…]. "
    "Dozwolone kierunki: UP, DOWN, LEFT, RIGHT. "
    "Jeśli w poleceniu jest ‘na sam dół’ interpretuj to jako ruch DOWN aż do dolnej krawędzi, "
    "‘na samą prawą’ => RIGHT aż do prawej krawędzi itd. "
    "Jeśli polecenie nie zawiera żadnego ruchu, zwróć []."
)

def parse_flight_instruction(instr: str) -> List[Tuple[str, int]]:
    """LLM-owy parser zwracający listę (DIR, steps)."""
    raw = _llm.answer_with_context(
        question=instr.strip(),
        context=_SYSTEM_PROMPT,
        model="gpt-4.1-mini",
        max_tokens=60,
    )
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    out: List[Tuple[str, int]] = []
    for item in data:
        if (isinstance(item, list) and len(item) == 2
            and item[0] in DIR_VEC and isinstance(item[1], int)):
            out.append((item[0], max(0, item[1])))
    return out

# ----------------------------------------------------------------------
def handle_instruction(instr: str) -> str:
    """Zwraca maks. 2-wyrazowy opis pola po wykonaniu instrukcji lotu."""
    pos = (0, 0)                   # start
    dm = DroneMap()
    moves = parse_flight_instruction(instr)
    for direction, steps in moves:
        pos = dm.move(pos, direction, steps)
    return dm.describe(pos)
