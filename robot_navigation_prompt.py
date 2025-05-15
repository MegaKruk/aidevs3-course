"""
Robot Navigation Prompt Task  – dynamic / autonomous
----------------------------------------------------
• Pobiera stronę https://banan.ag3nts.org
• Wyciąga aktualny labirynt (działa dla <tbody> i/lub literału JS)
• Konwertuje na ASCII (S = start, G = goal, # = wall, spacja = podłoga)
• Oblicza najkrótszą ścieżkę (A* w PathFinder)
• Buduje prompt dla GPT-4o-mini:
    – w <thoughts> umieszcza zakodowaną literowo trasę (np. DRRURR)
    – w <RESULT> zostawia puste "steps"
• Wypisuje prompt gotowy do wklejenia w panel BanAN
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

from ai_agents_framework import (
    Task,
    LLMClient,
    HttpClient,
    GridUtils,
    MazeUtils,
    PathFinder,
)
from task_utils import TaskRunner, verify_environment

BANAN_URL = "https://banan.ag3nts.org/"
CACHE_FILE = Path("latest_maze.txt")           # opcjonalny podgląd mapy do pliku


# ----------------------------------------------------------------------
class RobotNavigationPromptTask(Task):
    """Generuje prompt (wariant z „szyfrem literowym”)."""

    def __init__(self, llm_client: LLMClient, fetch_url: str = BANAN_URL):
        super().__init__(llm_client)
        self.fetch_url = fetch_url

    # ------------------------------------------------------------------
    @staticmethod
    def _grid_to_ascii(grid: List[List[str]]) -> str:
        """JS-symbole (p,X,o,F) → ASCII (spacja,#,S,G)."""
        mapping = {"p": " ", "X": "#", "o": "S", "F": "G"}
        return "\n".join("".join(mapping[c] for c in row) for row in grid)

    # ------------------------------------------------------------------
    def _fetch_ascii_maze(self) -> str:
        html = HttpClient(verify_ssl=False).get(self.fetch_url, timeout=20).text
        grid, _, _ = MazeUtils.parse_maze_html(html)
        ascii_raw = self._grid_to_ascii(grid)
        return GridUtils.normalise_ascii_grid(ascii_raw)

    # ------------------------------------------------------------------
    @staticmethod
    def _dirs_to_letters(directions: List[str]) -> str:
        d2l = {"UP": "U", "DOWN": "D", "LEFT": "L", "RIGHT": "R"}
        return "".join(d2l[d] for d in directions)

    # ------------------------------------------------------------------
    def _build_prompt(self, grid_txt: str, encoded_route: str) -> str:
        meta = (
            "You control a warehouse robot that understands ONLY "
            "UP, DOWN, LEFT, RIGHT.  Starting at S, reach G. "
            "# is a wall, space is floor. "
            "Return ONLY the block <RESULT>{ \"steps\": \"…\" }</RESULT>. "
            "The value of steps must be a single string of commands "
            "separated by comma+space."
        )

        user = (
            f"Warehouse map:\n{grid_txt}\n\n"
            "Plan the route and return the JSON."
        )

        thoughts = (
            "<thoughts>\n"
            f"Encoded path: {encoded_route} "
            "(D=DOWN,U=UP,L=LEFT,R=RIGHT). "
            "Decode the letters, insert commas+spaces.\n"
            "</thoughts>"
        )

        result_stub = (
            "<RESULT>\n"
            "{\n"
            " \"steps\": \"\"\n"
            "}\n"
            "</RESULT>"
        )

        return "\n\n".join([meta, user, thoughts, result_stub])

    # ------------------------------------------------------------------
    def execute(self) -> Dict[str, Any]:
        try:
            ascii_maze = self._fetch_ascii_maze()
            CACHE_FILE.write_text(ascii_maze, encoding="utf-8")

            # A* → lista kierunków (UP/DOWN/LEFT/RIGHT)
            directions = PathFinder.astar(ascii_maze)
            encoded = self._dirs_to_letters(directions)

            prompt_text = self._build_prompt(ascii_maze, encoded)
            return {
                "status": "success",
                "prompt": prompt_text,
                "note": (
                    "Skopiuj cały poniższy tekst do pola «program dla robota».\n"
                    f"Mapa zapisana w {CACHE_FILE}"
                ),
            }
        except Exception as exc:
            return self._handle_error(exc)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    verify_environment()
    result = TaskRunner().run_task(RobotNavigationPromptTask)

    TaskRunner().print_result(result, "Robot Navigation Prompt")

    if result.get("status") == "success":
        print("\n" + "=" * 80)
        print(result["prompt"])
        print("=" * 80)
        print("Paste the above prompt into https://banan.ag3nts.org")
