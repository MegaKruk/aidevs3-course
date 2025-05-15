import pathlib, binascii, re, piexif, json, base64, zlib, gzip, bz2, lzma, lz4.frame

file_path = "factory_data/srs.webp"
if __name__ == "__main__":
    p   = pathlib.Path(file_path).read_bytes()
    eoi = p.find(b'\xff\xd9')                 # koniec JPEG
    print("rozmiar pliku:", len(p))
    print("offset EOI  :", eoi)
    print("długość ogona:", len(p) - eoi - 2)
    print(binascii.hexlify(p[eoi+2:eoi+22]))

    # -----------
    dat = pathlib.Path(file_path).read_bytes()
    for sig, name in [(b'PK\x03\x04', 'ZIP'),
                      (b'\x89PNG', 'PNG'),
                      (b'GIF89a', 'GIF'),
                      (b'RIFF', 'WAV/WEBP'),
                      (b'\x1f\x8b', 'GZIP'),
                      (b'\xcd\xab\xcd\xab', 'XZ/LZMA')]:
        for m in re.finditer(re.escape(sig), dat):
            print(f"{name} @ {m.start():08x}")

    # -----------
    ex = piexif.load(file_path)
    for ifd in ("0th", "Exif", "GPS", "1st"):
        for tag, val in ex[ifd].items():
            if isinstance(val, bytes) and b"FLG" in val:
                print(">>>", val)
            if isinstance(val, bytes) and len(val) > 20:
                # pokaż początek, może to base64/zip itp.
                print(ifd, tag, binascii.hexlify(val[:16]), "...")

    # -----------
    tail = pathlib.Path(file_path).read_bytes()[-4096:]
    for name, fn in [
        ("zlib", zlib.decompress),
        ("gzip", lambda b: gzip.decompress(b'\x1f\x8b' + b)),
        ("bz2", bz2.decompress),
        ("lzma", lzma.decompress)]:
        try:
            out = fn(tail)
            print("✓", name, "=>", len(out), "bytes")
            pathlib.Path(f"tail.{name}").write_bytes(out)
        except Exception:
            pass