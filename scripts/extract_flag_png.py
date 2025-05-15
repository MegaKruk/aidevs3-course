#!/usr/bin/env python3
"""
Wyciąga wszystkie chunki tekstowe z pliku WebP (RIFF)
i szuka wzorca {{FLG:...}}.
Użycie:  python3 extract_flag_webp.py <plik.webp>
"""

import sys, re, struct

def riff_chunks(path):
    with open(path, "rb") as f:
        hdr = f.read(12)
        if not hdr.startswith(b"RIFF") or hdr[8:12] != b"WEBP":
            raise ValueError("To nie jest plik WebP / RIFF")
        while True:
            head = f.read(8)
            if len(head) < 8:
                break
            fourcc, size = struct.unpack("<4sI", head)
            data = f.read(size + (size & 1))  # padding do pary
            yield fourcc.decode(), data

def main(path):
    patt = re.compile(rb"FLG:[^}]")
    for name, data in riff_chunks(path):
        m = patt.search(data)
        if m:
            print(f"✓ Znaleziono flagę w chunku {name}:",
                  m.group().decode())
            return
    print("Nie znaleziono wzorca {{FLG:…}}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Użycie:  python3 extract_flag_webp.py <plik.webp>")
        sys.exit(1)
    main(sys.argv[1])
