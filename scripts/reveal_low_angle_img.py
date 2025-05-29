#!/usr/bin/env python3
"""
Reveal hidden text in “tilt-to-read” images that need a **horizontal** stretch.

Usage
-----
python reveal_low_angle.py  _secret_payloads/na_smartfona.png
(optional)  --scale 8   --no-invert  --no-threshold  --no-mirror

Requires Pillow  (pip install pillow)
"""
from pathlib import Path
import argparse

from PIL import Image, ImageOps

# ----------------------------------------------------------------------
def reveal(
    input_path: Path,
    output_path: Path | None = None,
    scale: int = 8,
    do_invert: bool = True,
    do_threshold: bool = True,
    do_mirror: bool = True,
) -> Path:
    """
    Horizontally stretch *input_path* so the stripe-encoded text becomes legible.

    Returns the path of the saved file.
    """
    img = Image.open(input_path).convert("L")  # greyscale

    if do_invert:
        img = ImageOps.invert(img)

    # horizontal stretch: widen by *scale*, keep height unchanged
    w, h = img.size
    img = img.resize((w * scale, h), resample=Image.NEAREST)

    if do_threshold:
        img = img.point(lambda p: 255 if p > 128 else 0)

    if do_mirror:
        img = ImageOps.mirror(img)  # left-right flip

    if output_path is None:
        output_path = input_path.with_name(
            input_path.stem + "_revealed" + input_path.suffix
        )
    img.save(output_path)
    return output_path


# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Reveal horizontally-compressed tilt image")
    ap.add_argument("image", type=Path, help="input PNG/JPG")
    ap.add_argument("-o", "--output", type=Path, help="output file name")
    ap.add_argument("--scale", type=int, default=8, help="horizontal scale factor")
    ap.add_argument("--no-invert", action="store_true", help="skip colour inversion")
    ap.add_argument("--no-threshold", action="store_true", help="skip BW threshold")
    ap.add_argument("--no-mirror", action="store_true", help="skip final mirror flip")
    args = ap.parse_args()

    out = reveal(
        args.image,
        output_path=args.output,
        scale=max(1, args.scale),
        do_invert=not args.no_invert,
        do_threshold=not args.no_threshold,
        do_mirror=not args.no_mirror,
    )
    print(f"Saved revealed image → {out}")


if __name__ == "__main__":
    main()
