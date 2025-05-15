import mimetypes
from pathlib import Path
import re


def find_embedded_zip(path: Path, tail=8192):
    """Zwraca offset sygnatury PK w ostatnich *tail* bajtach lub None."""
    with path.open("rb") as fh:
        fh.seek(-tail, 2)
        data = fh.read()
    idx = data.find(b"PK\x03\x04")
    return None if idx == -1 else len(data) - idx


if __name__ == "__main__":
    for img in ["srs.webp", *[f"2024-11-12_report-{n}.png" for n in ("13", "14", "15", "16", "17")]]:
        p = Path("./factory_data") / img
        off = find_embedded_zip(p)
        print(f"{img:30} → {'zip @ -'+str(off)+'B' if off else 'nie widać ZIPa'}")

    if re.search(r"report[-_]?99", p.name, re.I):
        print("Znalazłem:", p, mimetypes.guess_type(p)[0])
    else:
        print("tutaj też nic")

    import itertools, zipfile, pathlib, sys

    words = ["Barbara", "Raven", "Azazel", "Bourbon", "JosephN"]
    nums = ["0815", "1545", "1150", "1330", "0920"]
    zf = zipfile.ZipFile("./factory_data/weapons_tests.zip")
    for w, n in itertools.product(words, nums):
        pwd = f"{w}{n}".encode()
        try:
            zf.extractall("./factory_data/_wt", pwd=pwd)
            print("Hasło:", pwd.decode());
            break
        except RuntimeError:
            pass