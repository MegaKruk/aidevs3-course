import segno
from PIL import Image
import numpy as np

if __name__ == "__main__":
    img = Image.open("just_qr.png").convert("1")  # czarno‑białe
    matrix = np.array(img, dtype=bool)

    # segno pozwala wstrzyknąć własną macierz i spróbować dekodować:
    qr = segno.encoder.make_qr(matrix)
    print(qr.decode())        # może zwrócić None, ale warto sprawdzić
