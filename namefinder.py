#!/usr/bin/env python3
"""
namefinder.py - extract usernames/text from a folder of screenshots.

Designed for images with a white or dark-grey background and white text
(e.g. game overlays, chat/member lists). Uses OpenCV for preprocessing and
Tesseract OCR (via pytesseract) for text recognition.

Reading order: results are emitted top-to-bottom by line, and left-to-right
within a line. Multiple names on the same visual line (separated by spaces)
are kept on one output line. Duplicates are NOT removed.

Requirements:
    pip install opencv-python-headless pytesseract numpy
    Tesseract OCR engine must be installed and on PATH
    (Debian/Ubuntu: sudo apt-get install tesseract-ocr)

Usage:
    python3 namefinder.py <images_folder> [-o OUTPUT_DIR] [--min-conf N]
                           [--ext .png,.jpg] [--no-source-comments] [--debug]

Run via the bundled launcher instead, if you prefer:
    ./namefinder.bin <images_folder>
"""

import argparse
import os
import re
import sys
from datetime import datetime

VERSION = "1.0.0"

DEFAULT_MIN_CONF = 40
DEFAULT_EXTENSIONS = (".png",)


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def check_dependencies():
    """Fail fast with a clear message if a dependency is missing."""
    missing = []
    try:
        import cv2  # noqa: F401
    except ImportError:
        missing.append("opencv-python-headless (import name: cv2)")
    try:
        import numpy  # noqa: F401
    except ImportError:
        missing.append("numpy")
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        missing.append("pytesseract")

    if missing:
        eprint("Missing required Python package(s):")
        for m in missing:
            eprint(f"  - {m}")
        eprint("\nInstall with:")
        eprint("  pip install opencv-python-headless pytesseract numpy")
        sys.exit(1)

    import pytesseract
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        eprint("Tesseract OCR engine was not found on PATH.")
        eprint("Install it with (Debian/Ubuntu):")
        eprint("  sudo apt-get install tesseract-ocr")
        sys.exit(1)


def natural_key(s):
    """Sort key so 'shot2.png' comes before 'shot10.png'."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def find_images(folder, extensions):
    exts = tuple(e.lower() for e in extensions)
    files = [
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith(exts)
    ]
    return sorted(files, key=natural_key)


def next_output_path(out_dir, prefix="names", ext=".txt"):
    """Find the next available namesN.txt, starting at names1.txt."""
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(ext)}$")
    max_n = 0
    if os.path.isdir(out_dir):
        for fname in os.listdir(out_dir):
            m = pattern.match(fname)
            if m:
                max_n = max(max_n, int(m.group(1)))
    return os.path.join(out_dir, f"{prefix}{max_n + 1}{ext}")


def preprocess_variants(gray):
    """
    Yield increasingly aggressive preprocessing attempts. Text is assumed to
    be brighter (white) than the background, whether that background is
    white/near-white or dark grey.
    """
    import cv2

    scale = 2
    up = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    def clean(mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Attempt 1: global Otsu threshold. Works well when the background is a
    # single flat shade (white or dark grey) and text is uniformly bright -
    # the common case for UI/game screenshots.
    _, otsu = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    yield clean(otsu)

    # Attempt 2 (fallback): local adaptive threshold. Helps when contrast
    # between text and background is subtle (e.g. white text on an
    # off-white/near-white panel) or background brightness varies across
    # the image.
    adaptive = cv2.adaptiveThreshold(
        up, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 5
    )
    yield clean(adaptive)


def run_tesseract(mask, min_conf):
    import pytesseract
    from pytesseract import Output

    data = pytesseract.image_to_data(mask, output_type=Output.DICT, config="--psm 6")
    words = []
    n = len(data["text"])
    for i in range(n):
        txt = data["text"][i].strip()
        conf_raw = data["conf"][i]
        try:
            conf = float(conf_raw)
        except (TypeError, ValueError):
            conf = -1
        if txt and conf >= min_conf:
            words.append({
                "text": txt,
                "left": data["left"][i],
                "top": data["top"][i],
                "block": data["block_num"][i],
                "par": data["par_num"][i],
                "line": data["line_num"][i],
            })
    return words


def words_to_lines(words):
    """Group words into visual lines, ordered top-to-bottom, left-to-right."""
    groups = {}
    for w in words:
        key = (w["block"], w["par"], w["line"])
        groups.setdefault(key, []).append(w)

    lines = []
    for ws in groups.values():
        ws_sorted = sorted(ws, key=lambda w: w["left"])
        top = min(w["top"] for w in ws_sorted)
        text = " ".join(w["text"] for w in ws_sorted)
        lines.append((top, text))

    lines.sort(key=lambda t: t[0])
    return [text for _, text in lines]


def extract_lines_from_image(path, min_conf, debug_dir=None):
    import cv2

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not read image (unsupported/corrupt file?): {path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for idx, mask in enumerate(preprocess_variants(gray), start=1):
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(path))[0]
            cv2.imwrite(os.path.join(debug_dir, f"{base}_attempt{idx}.png"), mask)

        words = run_tesseract(mask, min_conf)
        if words:
            return words_to_lines(words)

    return []


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract usernames/text from a folder of screenshots via OCR."
    )
    p.add_argument("folder", help="Folder containing the screenshot images")
    p.add_argument(
        "-o", "--output-dir", default=None,
        help="Where to write namesN.txt (default: same as input folder)"
    )
    p.add_argument(
        "--min-conf", type=int, default=DEFAULT_MIN_CONF,
        help=f"Minimum Tesseract confidence 0-100 to keep a detection (default: {DEFAULT_MIN_CONF})"
    )
    p.add_argument(
        "--ext", default=",".join(DEFAULT_EXTENSIONS),
        help="Comma-separated list of file extensions to process (default: .png)"
    )
    p.add_argument(
        "--no-source-comments", action="store_true",
        help="Do not prefix each image's results with a '# filename' comment line"
    )
    p.add_argument(
        "--debug", action="store_true",
        help="Save intermediate thresholded images next to the output, for tuning"
    )
    return p.parse_args()


def main():
    args = parse_args()
    check_dependencies()

    folder = args.folder
    if not os.path.isdir(folder):
        eprint(f"Error: not a folder: {folder}")
        sys.exit(1)

    out_dir = args.output_dir or folder
    os.makedirs(out_dir, exist_ok=True)

    extensions = tuple(e.strip() for e in args.ext.split(",") if e.strip())
    images = find_images(folder, extensions)
    if not images:
        eprint(f"No images with extensions {extensions} found in {folder}")
        sys.exit(1)

    debug_dir = os.path.join(out_dir, "namefinder_debug") if args.debug else None

    out_path = next_output_path(out_dir)
    run_time = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")

    total_lines = 0
    failed = []

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"NameFinder v{VERSION}\n")
        f.write(f"Run: {run_time}\n")
        f.write("\n")

        for fname in images:
            path = os.path.join(folder, fname)
            try:
                lines = extract_lines_from_image(path, args.min_conf, debug_dir)
            except Exception as e:
                eprint(f"Warning: failed on {fname}: {e}")
                failed.append(fname)
                continue

            if not args.no_source_comments:
                f.write(f"# {fname}\n")
            for line in lines:
                f.write(line + "\n")
                total_lines += 1

    print(f"Processed {len(images)} image(s), {len(failed)} failed.")
    print(f"Extracted {total_lines} line(s) of text.")
    print(f"Output written to: {out_path}")
    if failed:
        print("Failed images:", ", ".join(failed))


if __name__ == "__main__":
    main()
