from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import cv2

SRC = Path("just_qr.png")
OUT = Path("out_full.png")
if __name__ == "__main__":
    # -- 1. ładujemy pół‑QR ----------------------------------------------------
    half = Image.open(SRC).convert("L")            # L = 8‑bit gray
    pixels = np.array(half)
    h, w = pixels.shape

    # -- 2. usuwamy szew (kolumny o jasności ~ 192) ----------------------------
    seam_cols = np.where(pixels.mean(axis=0) > 180)[0]   # heurystyka
    if len(seam_cols):
        seam_x0 = seam_cols[len(seam_cols)//2]           # środkowy jasny słupek
        half = half.crop((0, 0, seam_x0, h))             # lewa część bez szwu
    w_half = half.width

    # -- 3. lokalizujemy lewy‑górny finder -------------------------------------
    bin_half = (np.array(half) < 128).astype(np.uint8)
    # najbardziej czarny wiersz/kolumnę uznajemy za środek findera
    sum_rows = bin_half.sum(axis=1)
    sum_cols = bin_half.sum(axis=0)
    y0 = np.argmax(sum_rows[: h // 2])
    x0 = np.argmax(sum_cols[: w_half // 2])
    # idziemy w prawo/dół aż skończy się czarny blok -> rozmiar findera w px
    x1 = x0
    while x1 < w_half and bin_half[y0, x1]:
        x1 += 1
    finder_px = x1 - x0          # szerokość findera w pikselach

    # -- 4. wyliczamy rozmiar modułu ------------------------------------------
    # finder ma 7×7 modułów, więc:
    module_px = round(finder_px / 7)

    # obliczamy liczbę modułów na całą szerokość
    modules = (w_half // module_px) * 2           # lewa + prawa strona

    canvas_px = modules * module_px
    full = Image.new("L", (canvas_px, canvas_px), "white")

    # -- 5. kopiujemy lewą połowę w siatce modułów -----------------------------
    # przeskaluj pół‑QR do 'canvas_px/2'
    half_scaled = half.resize((canvas_px // 2, canvas_px), Image.NEAREST)
    full.paste(half_scaled, (0, 0))

    # -- 6. odbicie + wklejenie prawej połowy ----------------------------------
    mirror = half_scaled.transpose(Image.FLIP_LEFT_RIGHT)
    full.paste(mirror, (canvas_px // 2, 0))
    # kasujemy artefakt dokładnie pośrodku (1 px szerokości)
    mid_x = canvas_px // 2
    ImageDraw.Draw(full).line([(mid_x-1, 0), (mid_x+1, canvas_px)], fill="white")

    # -- 7. wymazujemy prawy‑dolny finder --------------------------------------
    draw = ImageDraw.Draw(full)
    erase_size = 8 * module_px
    draw.rectangle(
        (
            canvas_px - erase_size,
            canvas_px - erase_size,
            canvas_px - 1,
            canvas_px - 1,
        ),
        fill="white",
    )

    # -- 7b. dodaj Quiet Zone -------------------------------------------------
    border = 4 * module_px  # 4 moduły z każdej strony
    with_border = Image.new("L",
                            (canvas_px + 2 * border, canvas_px + 2 * border),
                            "white")
    with_border.paste(full, (border, border))

    # -- 8. zapis wersji kontrolnej -------------------------------------------
    with_border.save(OUT)

    # -- 9. dekodowanie --------------------------------------------------------
    # powiększamy ×3, bo QRCodeDetector lubi duże obrazki
    big = with_border.resize(
        (with_border.width * 3,
         with_border.height * 3),
        Image.NEAREST)
    data, _, _ = cv2.QRCodeDetector().detectAndDecode(np.array(big))

    if not data:
        # próbujemy inwersję kolorów
        inv = cv2.bitwise_not(np.array(big))
        data, _, _ = cv2.QRCodeDetector().detectAndDecode(inv)

    print("Wynik detektora:", data or "<brak>")

    import zxingcpp

    txt = zxingcpp.read_barcodes(big)[0].text if zxingcpp.read_barcodes(big) else ''
    print("ZXing:", txt or "<brak>")
