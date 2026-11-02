#!/usr/bin/env python3
"""Write the Quick Draw cat bitmaps as an `XAE4` dataset for `train-nand`.

`scripts/fetch_quickdraw_cats.py` writes a JSON of 0/1 bitmaps for the SAT
experiments, which need no dataset file. This wraps the same bitmaps in the
format the Rust trainer reads, so the gradient track can be compared on
identical data (the same reason `scripts/make_glyph_dataset.py` exists).

Each drawing is repeated `--copies` times so the trainer has a train/val split
(it selects the best-validation epoch); the last copy is flagged as test.

    python3 scripts/make_quickdraw_dataset.py \
        --targets data/quickdraw_cats_32.json --out data/quickdraw_cats_32.bin
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", default="data/quickdraw_cats_32.json")
    parser.add_argument("--out", default="data/quickdraw_cats_32.bin")
    parser.add_argument("--copies", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="0 = every drawing")
    args = parser.parse_args()

    payload = json.loads(Path(args.targets).read_text())
    side = int(payload["side"])
    patterns = [[int(b) for b in bits] for bits in payload["targets"]]
    if args.limit:
        patterns = patterns[: args.limit]

    samples: list[list[int]] = []
    is_test: list[int] = []
    for copy in range(args.copies):
        for pattern in patterns:
            samples.append(pattern)
            is_test.append(1 if copy == args.copies - 1 else 0)

    n = len(samples)
    pixels = side * side
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        fh.write(b"XAE4")
        fh.write(struct.pack("<I", side))
        fh.write(struct.pack("<I", n))
        fh.write(bytes([0]))   # ColorMode::Grayscale
        fh.write(bytes([1]))   # bits per pixel
        fh.write(bytes([1]))   # SplitMode::PreSplit
        fh.write(bytes([1]))   # channels
        fh.write(struct.pack("<H", 0))  # palette length
        fh.write(bytes(255 * b for s in samples for b in s))
        fh.write(bytes(255 * b for s in samples for b in s for _ in range(3)))
        fh.write(bytes(b for s in samples for b in s))
        words_per_row = (pixels + 63) // 64
        for sample in samples:
            for w in range(words_per_row):
                word = 0
                for i in range(64):
                    bit = w * 64 + i
                    if bit < pixels and sample[bit]:
                        word |= 1 << i
                fh.write(struct.pack("<Q", word))
        fh.write(bytes(is_test))

    print(f"wrote {out}: {n} samples ({len(patterns)} drawings x {args.copies}), "
          f"{side}x{side}, {sum(is_test)} test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
