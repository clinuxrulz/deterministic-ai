#!/usr/bin/env python3
"""Write the 8x8 glyph set as an `XAE4` dataset for `train-nand`.

The SAT experiments in `scripts/sat_ae.py` need no dataset file -- the glyphs are
in the script. This writes the same bitmaps in the format the Rust trainer reads
so the two can be compared on identical data.

Each glyph is repeated `--copies` times so the trainer has a train/val split (it
selects the best-validation epoch), and the last copy is flagged as test.

    python3 scripts/make_glyph_dataset.py --out data/glyphs8.bin
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sat_ae import GLYPHS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/glyphs8.bin")
    parser.add_argument("--glyphs", default="ABCDEFGH", help="subset, e.g. ABC")
    parser.add_argument("--copies", type=int, default=4)
    parser.add_argument("--size", type=int, default=8)
    args = parser.parse_args()

    if args.size != 8:
        raise SystemExit("the glyph table is 8x8; --size other than 8 is unsupported")

    rows: list[list[int]] = []
    for name in args.glyphs:
        pattern = [1 if ch == "#" else 0 for row in GLYPHS[name] for ch in row]
        rows.append(pattern)

    samples: list[list[int]] = []
    is_test: list[int] = []
    for copy in range(args.copies):
        for pattern in rows:
            samples.append(pattern)
            is_test.append(1 if copy == args.copies - 1 else 0)

    n = len(samples)
    pixels = args.size * args.size
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        fh.write(b"XAE4")
        fh.write(struct.pack("<I", args.size))
        fh.write(struct.pack("<I", n))
        fh.write(bytes([0]))   # ColorMode::Grayscale
        fh.write(bytes([1]))   # bits per pixel
        fh.write(bytes([1]))   # SplitMode::PreSplit
        fh.write(bytes([1]))   # channels
        fh.write(struct.pack("<H", 0))  # palette length
        fh.write(bytes(255 * b for b in (bit for s in samples for bit in s)))
        fh.write(bytes(255 * bit for s in samples for bit in s for _ in range(3)))
        fh.write(bytes(bit for s in samples for bit in s))
        for sample in samples:
            word = 0
            for i, bit in enumerate(sample):
                word |= bit << i
            fh.write(struct.pack("<Q", word))
        fh.write(bytes(is_test))

    print(f"wrote {out}: {n} samples ({len(rows)} glyphs x {args.copies}), "
          f"{pixels} bits, {sum(is_test)} test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
