import pathlib, shutil

if __name__ == "__main__":
    jpg = pathlib.Path("factory_data/2024-11-12_report-99").read_bytes()
    off = jpg.find(b"PK\x03\x04")          # 0x5BE6
    out = jpg[off:]
    path = pathlib.Path("factory_data/report-99.zip")
    path.write_bytes(out)
    print("✓ zapisano", path, "(", len(out), "B)")
