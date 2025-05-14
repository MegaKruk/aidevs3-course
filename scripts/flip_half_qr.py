from pathlib import Path
import numpy as np
from PIL import Image
import zxingcpp

SRC = Path("just_qr.png")         # źródłowa połówka
OUT = Path("qr_halfinvert.png")   # zapis kontrolny
if __name__ == "__main__":
    # --- 1. wczytanie i binarizacja (próg 128) ---------------------------------
    gray = Image.open(SRC).convert("L")
    bw   = np.where(np.array(gray) > 128, 255, 0).astype(np.uint8)

    # --- 2. podział na pół -----------------------------------------------------
    h, w = bw.shape
    mid = w // 2                       # zakładamy parzystą szerokość
    left  = bw[:, :mid]
    right = 255 - bw[:, mid:]          # pełna inwersja prawej połowy

    # --- 3. sklej całość -------------------------------------------------------
    merged = np.hstack([left, right])

    # --- 4. Quiet Zone –4 moduły (tu: 4% szerokości) -------------------------
    mod_px = int(round(mid / 9))       # finder = 9 modułów
    border = 4 * mod_px
    canvas = np.full((h + 2*border, w + 2*border), 255, np.uint8)
    canvas[border:border+h, border:border+w] = merged
    Image.fromarray(canvas).save(OUT)

    # --- 5. dekodowanie ---------------------------------------------------------
    scaled = np.repeat(np.repeat(canvas, 3, axis=0), 3, axis=1)  # powiększ x3
    result = zxingcpp.read_barcodes(scaled)
    print("TREŚĆ QR:", result[0].text if result else "<brak>")
