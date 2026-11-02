#!/usr/bin/env python3
"""Rasterise Quick Draw cats to 32x32 bits for the CNN comparison tab.

This is the numpy/PIL-free sibling of ``fetch_quickdraw_cats.py``. The
rasterisation is deliberately simple -- normalise the bounding box, draw each
stroke with a round pen at 4x supersampling, box-average down and threshold --
but it uses the same acceptance filters (ink range and span), so the result is
the same kind of drawing the circuit tabs were trained on.

Output is a small binary file the standalone Rust CNN trainer reads::

    magic  "QCNN"
    u32    side
    u32    count
    count * side * side bytes (0/1, row-major)

    python3 scripts/make_quickdraw_cnn_data.py \
        --source data/cat_sample.ndjson \
        --out data/quickdraw_cats_32_cnn.bin --limit 200
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

SS = 4  # supersample factor
PAD = 0.08


def stroke_points(stroke):
    if len(stroke) == 2 and isinstance(stroke[0], list):
        return list(zip(stroke[0], stroke[1]))
    return [(p[0], p[1]) for p in stroke]


def stamp(buf, w, x, y, radius):
    """OR a filled disk into a bytearray (used as a round pen)."""
    r = int(radius)
    r2 = radius * radius
    x0 = max(0, int(x) - r)
    x1 = min(w - 1, int(x) + r)
    y0 = max(0, int(y) - r)
    y1 = min(w - 1, int(y) + r)
    for yy in range(y0, y1 + 1):
        dy = yy - y
        row = yy * w
        for xx in range(x0, x1 + 1):
            dx = xx - x
            if dx * dx + dy * dy <= r2:
                buf[row + xx] = 255


def line(buf, w, x0, y0, x1, y1, radius):
    dx = x1 - x0
    dy = y1 - y0
    steps = int(max(abs(dx), abs(dy)) * 2) + 1
    for i in range(steps + 1):
        t = i / steps
        stamp(buf, w, x0 + dx * t, y0 + dy * t, radius)


def rasterise(strokes, side):
    s = side * SS
    pts = [p for stroke in strokes for p in stroke_points(stroke)]
    if not pts:
        return [0] * (side * side)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    span = max(w, h, 1.0)
    pad = side * PAD
    scale = (side - 2 * pad) / span
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2

    buf = bytearray(s * s)
    radius = SS * 0.9
    for stroke in strokes:
        path = stroke_points(stroke)
        projected = [
            (((p[0] - cx) * scale + side / 2) * SS,
             ((p[1] - cy) * scale + side / 2) * SS)
            for p in path
        ]
        if len(projected) >= 2:
            for i in range(len(projected) - 1):
                line(buf, s, projected[i][0], projected[i][1],
                     projected[i + 1][0], projected[i + 1][1], radius)
        elif projected:
            stamp(buf, s, projected[0][0], projected[0][1], radius)

    # Box average down and threshold at 0.34 coverage.
    bits = [0] * (side * side)
    threshold = 0.34 * SS * SS * 255
    for oy in range(side):
        for ox in range(side):
            total = 0
            for yy in range(SS):
                row = (oy * SS + yy) * s + ox * SS
                for xx in range(SS):
                    total += buf[row + xx]
            if total > threshold:
                bits[oy * side + ox] = 1
    return bits


def accept(bits, side, min_ink, max_ink):
    ink = sum(bits) / (side * side)
    if ink < min_ink or ink > max_ink:
        return False
    rows = [y for y in range(side) if any(bits[y * side + x] for x in range(side))]
    cols = [x for x in range(side) if any(bits[y * side + x] for y in range(side))]
    if len(rows) < side * 0.45 or len(cols) < side * 0.45:
        return False
    if rows[-1] - rows[0] + 1 < side * 0.6:
        return False
    if cols[-1] - cols[0] + 1 < side * 0.6:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="data/cat_sample.ndjson")
    parser.add_argument("--out", default="data/quickdraw_cats_32_cnn.bin")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--min-ink", type=float, default=0.055)
    parser.add_argument("--max-ink", type=float, default=0.22)
    args = parser.parse_args()

    side = args.size
    targets = []
    seen = set()
    with Path(args.source).open() as handle:
        for line_no, line in enumerate(handle):
            if len(targets) >= args.limit:
                break
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not entry.get("recognized", True):
                continue
            strokes = entry.get("drawing") or []
            if len(strokes) < 2:
                continue
            pts = [p for stroke in strokes for p in stroke_points(stroke)]
            if len(pts) < 8:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            w = max(xs) - min(xs)
            h = max(ys) - min(ys)
            if w < 25 or h < 25 or not (0.4 <= w / max(h, 1) <= 2.5):
                continue
            bits = rasterise(strokes, side)
            if not accept(bits, side, args.min_ink, args.max_ink):
                continue
            key = bytes(bits)
            if key in seen:
                continue
            seen.add(key)
            targets.append(bits)
            if len(targets) % 25 == 0:
                print(f"  accepted {len(targets)} after {line_no} candidates", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        fh.write(b"QCNN")
        fh.write(struct.pack("<II", side, len(targets)))
        for bits in targets:
            fh.write(bytes(bits))
    print(f"wrote {out}: {len(targets)} {side}x{side} bitmaps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
