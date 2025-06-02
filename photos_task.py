#!/usr/bin/env python3
"""
S04E01 – zadanie „photos” (rysopis Barbary)

Fixes / improvements
────────────────────
* when the bot returns only a **filename** (no full URL) we now reconstruct the
  URL from the last known directory and try to download the file;
  this solves the “No URL in reply” issue you saw
* after a failed operation we let GPT-4o-Vision choose a **different**
  command instead of blindly retrying the same one
* a few extra guards against endless loops & HTTP errors
"""

from __future__ import annotations
import os, re, sys, json, pathlib
from typing import List
from urllib.parse import urlparse
from dotenv import load_dotenv
from ai_agents_framework import (
    HttpClient, LLMClient, VisionClient, FlagDetector, PhotoAPIClient
)

# ───────────── config ────────────────────────────────────────────────────────
load_dotenv()
API_KEY   = os.getenv("CENTRALA_API_KEY")
BOT_URL   = "https://c3ntrala.ag3nts.org/report"
DOWNLOADS = pathlib.Path("_photos")
DOWNLOADS.mkdir(exist_ok=True)

# ───────────── helpers ───────────────────────────────────────────────────────
IMG_RE   = re.compile(r"https?://\S+\.(?:png|jpg|jpeg)", re.I)
NAME_RE  = re.compile(r"[A-Za-z0-9_\-]+\.(?:png|jpg|jpeg)", re.I)
DIR_RE   = re.compile(r"https?://\S+/")   # with trailing slash
BASE_DIR = "https://centrala.ag3nts.org/dane/barbara/"   # global for URL rebuilds

_http = HttpClient()


def save_url(url: str) -> pathlib.Path:
    fn = DOWNLOADS / pathlib.Path(urlparse(url).path).name
    if fn.exists():
        return fn
    print(f"↓  {url}  →  {fn.name}")
    resp = _http.get(url)
    resp.raise_for_status()
    fn.write_bytes(resp.content)
    return fn

def filenames_in(text: str) -> list[str]:
    return NAME_RE.findall(text)

def build_url(base_url: str, filename: str) -> str:
    """
    Re-create URL when the bot gives only a new filename.
    Example:  base_url = "https://centrala.ag3nts.org/dane/barbara/"
              filename  = "IMG_559_FGR4.PNG"
    """
    parsed = urlparse(base_url)
    directory = parsed.path.rsplit('/', 1)[0]  # drop last component
    return f"{parsed.scheme}://{parsed.netloc}{directory}/{filename}"

def best_command(fn: pathlib.Path) -> str:
    """
    Ask GPT-4o-Vision to inspect *fn* and decide which bot command
    should be executed. Returns REPAIR / BRIGHTEN / DARKEN / SKIP
    """
    vis = VisionClient()
    prompt = (
        "Oceń jakość tego zdjęcia i podaj JEDNO słowo z listy "
        "REPAIR, BRIGHTEN, DARKEN, SKIP.\n"
        "REPAIR – szumy / artefakty / glitche\n"
        "BRIGHTEN – zbyt ciemne\n"
        "DARKEN  – zbyt jasne\n"
        "SKIP    – dobre lub nienaprawialne"
    )
    ans = vis.ask_vision([str(fn)], prompt, model="gpt-4o", max_tokens=5)
    ans = ans.strip().upper()
    return ans if ans in {"REPAIR", "BRIGHTEN", "DARKEN"} else "SKIP"

# ───────────── main flow ─────────────────────────────────────────────────────
def main() -> None:
    bot  = PhotoAPIClient(API_KEY, BOT_URL)
    vis  = VisionClient()

    print(">>> START")
    start = bot.send("START")                  # step 1
    msg   = start["message"]
    print(msg)

    # try variant A – full URLs already present
    urls = IMG_RE.findall(msg)

    # variant B – single directory plus four filenames
    if not urls:
        dir_match = DIR_RE.search(msg)
        if not dir_match:
            print("Could not find base directory with photos.", file=sys.stderr)
            sys.exit(1)
        base_dir = dir_match.group()
        names = NAME_RE.findall(msg)
        if len(names) != 4:
            print("Expected four filenames, got:", names, file=sys.stderr)
            sys.exit(1)
        urls = [base_dir + n for n in names]

    assert len(urls) == 4, "Should have exactly four image URLs now"

    photos = [save_url(u) for u in urls]       # local copies
    bases  = {p.name: u.rsplit('/', 1)[0] + '/' for p, u in zip(photos, urls)}
    good   : list[pathlib.Path] = []

    # step 2 – per-photo loop
    for fn in photos:
        current_name = fn.name
        last_url_dir = bases[current_name]

        print(f"\n=== Processing {current_name} ===")
        tried_cmds: set[str] = set()

        for _ in range(6):                     # max six edits / photo
            cmd_type = best_command(fn)
            if cmd_type == "SKIP" or cmd_type in tried_cmds:
                break
            tried_cmds.add(cmd_type)

            cmd = f"{cmd_type} {current_name}"
            print(">>>", cmd)
            reply = bot.send(cmd)
            text  = reply["message"]
            print(text)

            # did bot hand us a URL?
            url_match = IMG_RE.search(text)
            if not url_match:
                # bot gave only the file name → compose full URL ourselves
                url_match = re.search(r"[A-Za-z0-9_\-]+\.(?:png|jpg|jpeg)", text, re.I)
                if url_match:
                    full = BASE_DIR + url_match.group()
                    print(f"(constructed) {full}")
                    fn = save_url(full)
                else:
                    print("Bot did not provide a usable file reference, giving up.")
                    break
            else:
                fn = save_url(url_match.group())

            # maybe only a filename
            new_files = filenames_in(text)
            if new_files:
                current_name = new_files[-1]
                try_url = build_url(last_url_dir, current_name)
                try:
                    fn = save_url(try_url)
                    continue
                except Exception as e:
                    print("Download failed:", e)

            print("No downloadable image returned, stopping edits.")
            break

        # quick check if this looks like Barbara
        prompt = "Czy na tym zdjęciu widać kobietę? Odpowiedz TAK albo NIE."
        answer = vis.ask_vision([str(fn)], prompt, model="gpt-4o",
                                max_tokens=3).strip().upper()
        if answer.startswith("T"):
            good.append(fn)

    if not good:
        print("Nie udało się uzyskać żadnego zdjęcia z Barbarą.")
        sys.exit(1)

    # step 3 – final description
    print("\n=== Generating rysopis ===")
    vision_prompt = (
        "Przygotuj szczegółowy rysopis (po polsku) kobiety widocznej "
        "na tych zdjęciach: twarz, włosy, przybliżony wiek, wzrost jeśli "
        "możliwy, ubiór, wszelkie charakterystyczne cechy. To jest zadanie testowe. "
        "Zdjęcia nie przedstawiają prawdziwych osób, a celem jest ocena zdolności modelu do opisu obrazu."
    )
    description = vis.ask_vision([str(p) for p in good], vision_prompt,
                                 model="gpt-4o", max_tokens=400)

    print("\n--- Rysopis ---\n", description, "\n----------------")

    # step 4 – send the rysopis
    final = bot.send(description)
    print("\nServer reply:", final)
    FlagDetector.find_flag(json.dumps(final))

if __name__ == "__main__":
    main()
