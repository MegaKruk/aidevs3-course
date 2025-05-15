#!/usr/bin/env python3
"""
Szybki podgląd:
• wypisuje wszystkie ciągi w formacie {{FLG:...}}
• pokazuje metadane EXIF, jeśli istnieją
• sprawdza, czy po końcu JPEG-a (FF D9) doklejono ZIP/PNG/HTML
  i automatycznie zapisuje wykryty fragment do osobnego pliku
"""
from pathlib import Path
import re, struct, sys, json, zipfile

src = Path("./factory_data/suspicious.jpg")      # zmień, jeśli trzeba


if __name__ == "__main__":
    data = src.read_bytes()

    # 1. szukamy flag w całym pliku
    flags = re.findall(rb"\{\{FLG:[^}]+\}\}", data)
    if flags:
        print("[+] znaleziono flagi:", [f.decode() for f in flags])
    else:
        print("[-] brak bezpośrednich {{FLG:…}}")

    # 2. proste EXIF (Segment APP1  = Exif)
    def exif_dict(buf: bytes) -> dict | None:
        if buf[0:2] != b"\xff\xd8":
            return None
        i = 2
        while i < len(buf)-1:
            if buf[i] != 0xFF:
                break
            marker = buf[i+1]
            size   = struct.unpack(">H", buf[i+2:i+4])[0]   # big-endian
            if marker == 0xE1:                              # APP1
                exif = buf[i+4:i+2+size]
                try:
                    import piexif
                    return piexif.load(exif)
                except Exception:
                    pass                                    # piexif brak → pomijamy
            i += 2 + size
        return None


    try:
        import piexif
        ex = exif_dict(data)
        if ex:
            user_cmnt = (
                    ex["0th"].get(piexif.ImageIFD.ImageDescription) or
                    ex["Exif"].get(piexif.ExifIFD.UserComment)  # ← ta linijka
            )
            if user_cmnt:
                print("[+] EXIF:", user_cmnt.decode(errors="ignore"))
    except ImportError:
        print("(*) piexif nie zainstalowany – pomijam metadane")

    # 3. czy po FF D9 coś jeszcze jest?
    eoi = data.rfind(b"\xff\xd9")+2
    trailing = data[eoi:]
    if trailing:
        # podpisy najpopularniejszych formatów
        sigs = {b"PK\x03\x04": "zip",
                b"\x89PNG\r\n\x1a\n": "png",
                b"<html": "html",
                b"\xff\xd8\xff": "jpg"}
        for sig, name in sigs.items():
            pos = trailing.find(sig)
            if pos != -1:
                out = src.with_suffix(f".hidden.{name}")
                out.write_bytes(trailing[pos:])
                print(f"[+] wykryto {name.upper()} po końcu JPEG – zapisano → {out}")
                if name=="zip":
                    try:
                        zipfile.ZipFile(out).extractall(out.with_suffix(".dir"))
                        print("    (wypakowano automatycznie do katalogu *.dir/)")
                    except RuntimeError:
                        print("    (ZIP z hasłem – trzeba podać password)")
                break
        else:
            print("[?] po końcu JPEG są dodatkowe bajty, ale nie wygląda to na ZIP/PNG/HTML")
    else:
        print("[-] plik kończy się na FF D9 – brak doczepki")
