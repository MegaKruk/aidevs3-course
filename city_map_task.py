"""
City-Map Task
-------------
- uses data/mapa_with_borders.png (bright-green frames already drawn)
- detects 4 green frames, crops them (–5 px margin)
- queries GPT-4o on each fragment and on the full map
- majority-vote ⇒ city, submits to Centrala
"""
from __future__ import annotations

import base64, re, unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from ai_agents_framework import CentralaTask, FlagDetector
from task_utils import TaskRunner, verify_environment

SRC_IMG = Path("data/mapa_with_borders.png")
TMP_DIR = Path("_map_crops"); TMP_DIR.mkdir(exist_ok=True)


class CityMapTask(CentralaTask):

    @staticmethod
    def _green_boxes(img: Image.Image) -> List[Tuple[int, int, int, int]]:
        """Return (x1,y1,x2,y2) boxes of bright-green frames."""
        arr = np.asarray(img.convert("RGB"))
        g = arr[:, :, 1] > 200          # high green
        r = arr[:, :, 0] < 100          # low red
        b = arr[:, :, 2] < 100          # low blue
        mask = g & r & b

        H, W = mask.shape
        seen = np.zeros_like(mask, bool)
        boxes: List[Tuple[int, int, int, int]] = []

        for y in range(H):
            for x in range(W):
                if mask[y, x] and not seen[y, x]:
                    stack = [(y, x)]
                    minx = miny = 10**9; maxx = maxy = -1
                    while stack:
                        cy, cx = stack.pop()
                        if seen[cy, cx] or not mask[cy, cx]:
                            continue
                        seen[cy, cx] = True
                        minx = min(minx, cx); maxx = max(maxx, cx)
                        miny = min(miny, cy); maxy = max(maxy, cy)
                        for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                            ny, nx = cy+dy, cx+dx
                            if 0<=ny<H and 0<=nx<W and not seen[ny,nx]:
                                stack.append((ny, nx))
                    if maxx-minx > 50 and maxy-miny > 50:
                        boxes.append((minx, miny, maxx, maxy))

        boxes.sort(key=lambda b: (b[1], b[0]))     # TL-BR order
        if len(boxes) != 4:
            raise RuntimeError(f"Expected 4 frames, got {len(boxes)}")
        return boxes

    def _crop_fragments(self, img: Image.Image) -> List[Path]:
        paths = []
        for i, (x1,y1,x2,y2) in enumerate(self._green_boxes(img), 1):
            p = TMP_DIR/f"crop_{i}.png"
            img.crop((x1+5, y1+5, x2-5, y2-5)).save(p)
            paths.append(p)
        return paths

    @staticmethod
    def _data_url(p: Path) -> str:
        return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()

    def _ask_city(self, tag: str, img_path: Path) -> str:
        sys_msg = ("You analyse fragments of Polish city maps. "
                   "Think in <thinking>…</thinking> then finish with "
                   "<RESULT>ExactCityName</RESULT> (nothing else).")
        resp = self.llm_client.client.chat.completions.create(
            model="gpt-4o",
            temperature=0.0,
            max_tokens=1000,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user",   "content": [
                    {"type":"text",
                     "text":(
                         "Identify the Polish city for fragment "
                         f"'{tag}'. One fragment may belong to another city; ignore it."
                         "HINT: Zygfryd powiedział, że w poszukiwanym mieście 'Były jakieś spichlerze i twierdze'"
                     )
                     },
                    {"type":"image_url",
                     "image_url":{"url": self._data_url(img_path)}},
                ]},
            ],
        )
        txt = resp.choices[0].message.content or ""
        m = re.search(r"<RESULT>\s*([^<\n]+?)\s*</RESULT>", txt, re.I)
        city = (m.group(1) if m else txt.strip()).strip()
        print(f"[{tag}] -> {city}")
        return city

    @staticmethod
    def _norm(city: str) -> str:
        return unicodedata.normalize("NFKD", city).encode("ascii","ignore").decode().upper()

    def execute(self) -> Dict[str, str]:
        try:
            base  = Image.open(SRC_IMG)
            crops = self._crop_fragments(base)

            votes = [self._ask_city(f"crop{i}", p) for i,p in enumerate(crops,1)]
            votes.append(self._ask_city("all", SRC_IMG))

            best_norm = Counter(self._norm(v) for v in votes).most_common(1)[0][0]
            picked    = next(v for v in votes if self._norm(v) == best_norm)
            print("Picked city ->", picked)

            for variant in {picked, self._norm(picked)}:
                resp = self.submit_report("map", variant)
                flag = FlagDetector.find_flag(str(resp))
                if flag:
                    return {"status":"success","flag":flag,"city":variant}

            return {"status":"failed","votes":votes,"picked":picked}

        except Exception as exc:
            return self._handle_error(exc)


if __name__ == "__main__":
    verify_environment()
    result = TaskRunner().run_task(CityMapTask)
    TaskRunner().print_result(result, "City-Map Task")
    if result.get("flag"):
        print("\nFLAG: ", result["flag"])
