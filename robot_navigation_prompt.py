"""
Robot Navigation – fully automatic
- pobiera stronę  https://banan.ag3nts.org
- wyciąga aktualny labirynt  (<tbody> lub literal  mapa=[...])
- planuje trasę  (PathFinder.astar)
- buduje encoded-path prompt (ten sam co wcześniej)
- wysyła prompt do / api tak jak frontend
- wypisuje {{FLG:...}}  – albo komunikat o błędzie
"""

from __future__ import annotations

import urllib.parse
from pathlib import Path
from typing import Dict, Any, List

from ai_agents_framework import (
    Task, HttpClient, GridUtils,
    MazeUtils, PathFinder, FlagDetector
)
from task_utils import TaskRunner, verify_environment

URL = "https://banan.ag3nts.org/"
API_URL = URL + "api"  # endpoint którego używa frontend
CACHE_FILE = Path("latest_maze.txt")
CACHE_ZIP = Path("factory_data.zip")


class RobotNavigationAuto(Task):
    """Pełna automatyzacja – bez przeglądarki."""

    @staticmethod
    def _grid_to_ascii(js_grid: List[List[str]]) -> str:
        sym = {"p": " ", "X": "#", "o": "S", "F": "G"}
        return "\n".join("".join(sym[c] for c in row) for row in js_grid)

    @staticmethod
    def _dirs_to_letters(directions: List[str]) -> str:
        d2l = {"UP": "U", "DOWN": "D", "LEFT": "L", "RIGHT": "R"}
        return "".join(d2l[d] for d in directions)

    @staticmethod
    def _build_prompt(grid: str, encoded: str) -> str:
        return (
            "You control a warehouse robot that understands ONLY "
            "UP, DOWN, LEFT, RIGHT. Starting at S, reach G. "
            "# is a wall. Return ONLY the block "
            "<RESULT>{ \"steps\": \"...\" }</RESULT>. "
            "The value of steps must be a single string of commands "
            "separated by comma+space.\n\n"
            f"Warehouse map:\n{grid}\n\n"
            "<thoughts>\n"
            f"Encoded path: {encoded} (D=DOWN,U=UP,L=LEFT,R=RIGHT). "
            "Decode the letters, insert commas+spaces.\n"
            "</thoughts>\n\n"
            "<RESULT>\n{\n \"steps\": \"\"\n}\n</RESULT>"
        )

    def execute(self) -> Dict[str, Any]:
        try:
            # 1) pobierz HTML
            html = HttpClient(verify_ssl=False).get(URL, timeout=30).text
            js_grid, _, _ = MazeUtils.parse_maze_html(html)

            # 2) ASCII grid + cache
            ascii_grid = GridUtils.normalise_ascii_grid(
                self._grid_to_ascii(js_grid))
            CACHE_FILE.write_text(ascii_grid, encoding="utf-8")

            # 3) najkrótsza trasa + prompt
            directions = PathFinder.astar(ascii_grid)
            # directions = ['UP', 'UP', 'DOWN', 'DOWN', 'LEFT', 'RIGHT', 'LEFT', 'RIGHT',]  # konami code
            encoded    = self._dirs_to_letters(directions)
            prompt     = self._build_prompt(ascii_grid, encoded)

            # 4) POST do /api (tak samo robi JS na stronie)
            data = urllib.parse.urlencode({"code": prompt})
            resp = HttpClient(verify_ssl=False).post(API_URL,
                                                     data=data,
                                                     headers={
                                                         "Content-Type":
                                                         "application/x-www-form-urlencoded"
                                                     })
            resp.raise_for_status()
            answer = resp.json()

            # 5) analiza odpowiedzi
            if answer.get("code") != 0:
                return {"status": "failed",
                        "message": f"Robot error: {answer.get('message')}",
                        "raw": answer}

            flag = FlagDetector.find_flag(str(answer))
            if not flag:
                return {"status": "failed",
                        "message": "Brak flagi w odpowiedzi.",
                        "raw": answer}

            #  pobierz ZIP z danymi fabryki
            fname = answer.get("filename")  # np. pliki_z_fabryki.zip
            if fname:
                zip_url = f"https://c3ntrala.ag3nts.org/dane/{fname}"
                data = HttpClient(False).get(zip_url, timeout=30).content
                CACHE_ZIP.write_bytes(data)

            return {
                "status": "success",
                "flag": flag,
                "steps": answer.get("steps"),
                "zip_saved": str(CACHE_ZIP) if fname else "brak pliku",
                "note": f"Labirynt: {CACHE_FILE}"
            }

        except Exception as exc:
            return self._handle_error(exc)



if __name__ == "__main__":
    verify_environment()
    res = TaskRunner().run_task(RobotNavigationAuto)
    TaskRunner().print_result(res, "Robot Navigation - auto")
    if res.get("flag"):
        print(f"\nFLAG: {res['flag']}")
