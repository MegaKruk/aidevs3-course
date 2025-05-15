import sys, pathlib, re, base64, zlib

if __name__ == "__main__":
    p = pathlib.Path("./factory_data/2024-11-12_report-99").read_bytes()
    eoi = p.find(b'\xff\xd9')
    tail = p[eoi+2:]
    # spróbuj base64
    try:
        raw = base64.b64decode(tail, validate=True)
        pathlib.Path("tail.raw").write_bytes(raw)
        print("[+] zapisano tail.raw (base64-decoded)")
    except Exception: pass
    # spróbuj z-lib
    try:
        raw = zlib.decompress(tail)
        pathlib.Path("tail.zlib").write_bytes(raw)
        print("[+] zapisano tail.zlib (z-lib)")
    except zlib.error: pass
