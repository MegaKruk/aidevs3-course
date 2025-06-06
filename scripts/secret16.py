"""
Secret 16 – ‘Ja tam po prostu jestem’.
The flag is literally embedded in the PDF as “{{FLG:…}}”.

Usage: place notatnik-rafala.pdf next to the script.
"""
import re, sys
from pathlib import Path
import fitz                                   # PyMuPDF

pdf_path = Path("../data/notatnik-rafala.pdf")
doc = fitz.open(pdf_path)

pattern = re.compile(r"\{\{FLG:([^\}]+)\}\}")

if __name__ == "__main__":
    for page in doc:
        txt = page.get_text("text")
        m = pattern.search(txt)
        if m:
            print("Found flag on page", page.number+1, "->", m.group(1))
            sys.exit(0)

    print("No flag pattern found – are you sure you used the right file?")
