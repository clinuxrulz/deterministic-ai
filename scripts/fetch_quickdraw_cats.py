#!/usr/bin/env python3
"""Fetch Google Quick Draw cats and rasterise them as small binary bitmaps.

The ``cat`` category of the Quick Draw dataset is a 77 MB ndjson file of
stroke lists (x, y in 0..255). This script streams it, keeps the first
``--limit`` drawings that look like a drawing of a whole cat, rasterises each
one to a ``--size`` x ``--size`` bitmap, and writes:

* ``--out``      JSON: ``{"side": S, "targets": [[0/1, ...], ...], ...}``
* ``--preview``  a contact sheet so the selection can be eyeballed

The filters are deliberately simple and deterministic: no ML, no sampling
beyond "first N accepted in file order", so re-running the script reproduces
the same 200 bitmaps byte for byte.

Usage::

    python3 scripts/fetch_quickdraw_cats.py --limit 200 --size 32 \
        --out data/quickdraw_cats_32.json --preview data/quickdraw_cats_32.png
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

URL = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/cat.ndjson"


def download(path: Path) -> None:
    if path.exists() and path.stat().st_size > 1_000_000:
        print(f"using cached {path} ({path.stat().st_size / 1e6:.1f} MB)")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL}")
    urllib.request.urlretrieve(URL, path)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")


def stroke_points(stroke: list[list[float]]) -> list[tuple[float, float]]:
    """Quick Draw stores a stroke as two parallel arrays ``[xs, ys]``."""
    if len(stroke) == 2 and isinstance(stroke[0], list):
        return list(zip(stroke[0], stroke[1]))
    return [(p[0], p[1]) for p in stroke]


def rasterise(strokes: list[list[list[float]]], side: int, supersample: int,
              radius: float, threshold: float) -> np.ndarray:
    """Draw the strokes on an ``side`` x ``side`` grid, aspect preserved.

    Rendered at ``supersample`` times the target resolution with a round pen,
    then box-averaged down and thresholded. The pen width is in supersampled
    pixels, so ``width = 2 * radius + 1`` with ``radius ~ supersample`` gives a
    stroke about one target pixel wide before thresholding. PIL draws the lines
    in C; an earlier numpy pixel-stamp version took minutes per hundred cats.
    """
    s = side * supersample
    paths = [stroke_points(stroke) for stroke in strokes]
    pts = [p for path in paths for p in path]
    if not pts:
        return np.zeros((side, side), dtype=np.uint8)
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    w, h = xs.max() - xs.min(), ys.max() - ys.min()
    span = max(w, h, 1.0)
    pad = side * 0.08
    scale = (side - 2 * pad) / span
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2

    image = Image.new("L", (s, s), 0)
    draw = ImageDraw.Draw(image)
    width = max(1, int(round(2 * radius + 1)))
    for path in paths:
        if not path:
            continue
        projected = [(((p[0] - cx) * scale + side / 2) * supersample,
                      ((p[1] - cy) * scale + side / 2) * supersample)
                     for p in path]
        if len(projected) >= 2:
            draw.line(projected, fill=255, width=width, joint="curve")
        else:
            x, y = projected[0]
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=255)
    coverage = (np.asarray(image, dtype=np.float32) / 255.0).reshape(
        side, supersample, side, supersample).mean(axis=(1, 3))
    return (coverage > threshold).astype(np.uint8)


def accept(bits: np.ndarray, min_ink: float, max_ink: float) -> tuple[bool, str]:
    side = bits.shape[0]
    ink = float(bits.sum()) / bits.size
    if ink < min_ink:
        return False, f"ink {ink:.3f} < {min_ink}"
    if ink > max_ink:
        return False, f"ink {ink:.3f} > {max_ink}"
    rows = np.flatnonzero(bits.sum(axis=1))
    cols = np.flatnonzero(bits.sum(axis=0))
    if len(rows) < side * 0.45 or len(cols) < side * 0.45:
        return False, f"spans only {len(rows)}x{len(cols)} of {side}"
    if rows[-1] - rows[0] + 1 < side * 0.6 or cols[-1] - cols[0] + 1 < side * 0.6:
        return False, "ink bounding box too small"
    return True, ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", default="data/quickdraw/cat.ndjson")
    parser.add_argument("--out", default="data/quickdraw_cats_32.json")
    parser.add_argument("--preview", default="data/quickdraw_cats_32.png")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--min-ink", type=float, default=0.055)
    parser.add_argument("--max-ink", type=float, default=0.22)
    parser.add_argument("--threshold", type=float, default=0.34)
    parser.add_argument("--supersample", type=int, default=4)
    parser.add_argument("--scan", type=int, default=40000,
                        help="give up after this many candidate drawings")
    args = parser.parse_args()

    cache = Path(args.cache)
    download(cache)

    targets: list[np.ndarray] = []
    sources: list[dict] = []
    rejected = {"parse": 0, "points": 0, "bbox": 0, "raster": 0}
    seen: set[bytes] = set()
    radius = args.supersample * 0.9
    with cache.open() as handle:
        for line_no, line in enumerate(handle):
            if line_no >= args.scan or len(targets) >= args.limit:
                break
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                rejected["parse"] += 1
                continue
            if not entry.get("recognized", True):
                rejected["points"] += 1
                continue
            strokes = entry.get("drawing") or []
            n_points = sum(len(s) // 2 for s in strokes)
            if len(strokes) < 2 or n_points < 8:
                rejected["points"] += 1
                continue
            pts = [(s[0][i], s[1][i]) for s in strokes
                   for i in range(len(s[0]))]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            w, h = max(xs) - min(xs), max(ys) - min(ys)
            if w < 25 or h < 25 or not (0.4 <= w / max(h, 1) <= 2.5):
                rejected["bbox"] += 1
                continue
            bits = rasterise(strokes, args.size, args.supersample, radius,
                             args.threshold)
            ok, why = accept(bits, args.min_ink, args.max_ink)
            key = bits.tobytes()
            if ok and key in seen:
                ok, why = False, "duplicate"
            if not ok:
                rejected["raster"] += 1
                if rejected["raster"] <= 5:
                    print(f"  rejected line {line_no}: {why}")
                continue
            seen.add(key)
            targets.append(bits)
            sources.append({"line": line_no, "key_id": entry.get("key_id"),
                            "ink": round(float(bits.sum()) / bits.size, 4)})
            if len(targets) % 25 == 0:
                print(f"  accepted {len(targets)} after {line_no} candidates",
                      flush=True)

    side = args.size
    payload = {
        "side": side,
        "source": "Google Quick Draw, simplified cat.ndjson",
        "url": URL,
        "limit": args.limit,
        "min_ink": args.min_ink,
        "max_ink": args.max_ink,
        "threshold": args.threshold,
        "radius": radius,
        "supersample": args.supersample,
        "items": sources,
        "targets": [t.flatten().tolist() for t in targets],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload) + "\n")
    inks = [float(t.mean()) for t in targets]
    print(f"wrote {len(targets)} {side}x{side} bitmaps to {out}")
    print(f"ink {min(inks):.3f}-{max(inks):.3f} (mean {np.mean(inks):.3f}), "
          f"rejected {rejected}")

    # contact sheet: 16 columns, 4x nearest-neighbour
    cols = 16
    rows = (len(targets) + cols - 1) // cols
    scale = 4
    gap = 2
    cell = side * scale + gap
    sheet = np.zeros((rows * cell, cols * cell), dtype=np.uint8)
    for i, bits in enumerate(targets):
        big = np.kron(bits, np.ones((scale, scale), dtype=np.uint8)) * 255
        r, c = divmod(i, cols)
        sheet[r * cell:r * cell + side * scale,
              c * cell:c * cell + side * scale] = big
    Image.fromarray(sheet).save(args.preview)
    print(f"wrote preview {args.preview} ({cols} per row)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
