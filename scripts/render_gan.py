#!/usr/bin/env python3
"""Render an exported generator circuit to a contact sheet.

Reads ``web/public/gan/{circuit.json,meta.json}`` (or any directory written by
``scripts/sat_ae.py --mode fit --export``), evaluates every code with a plain
reference evaluator -- the same semantics as the browser's ``run_circuit`` --
and writes a PNG with one tile per code. Fitted codes get a green border, free
codes a grey one, so the difference between "fitted drawing" and "same wiring,
no target" is visible at a glance.

Usage::

    python3 scripts/render_gan.py --dir web/public/gan --out data/gan_fit.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def evaluate(circuit: dict, inputs: list[int]) -> list[int]:
    n = circuit["n_inputs"]
    wires = list(inputs[:n]) + [0, 1] + [0] * len(circuit["gates"])
    for i, gate in enumerate(circuit["gates"]):
        a, b = wires[gate["a"]], wires[gate["b"]]
        op = gate["op"]
        if op == "nand":
            value = 0 if (a and b) else 1
        elif op == "and":
            value = 1 if (a and b) else 0
        elif op == "or":
            value = 1 if (a or b) else 0
        elif op == "nor":
            value = 0 if (a or b) else 1
        elif op == "xor":
            value = a ^ b
        elif op == "xnor":
            value = 1 - (a ^ b)
        elif op == "not":
            value = 1 - a
        else:
            raise ValueError(f"unsupported op {op}")
        wires[n + 2 + i] = value
    return [wires[out] for out in circuit["outputs"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default="web/public/gan")
    parser.add_argument("--out", default="data/gan_fit.png")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--cols", type=int, default=8)
    args = parser.parse_args()

    root = Path(args.dir)
    circuit = json.loads((root / "circuit.json").read_text())
    meta = json.loads((root / "meta.json").read_text())
    side, latent = meta["side"], meta["latent"]
    n_codes = 1 << latent
    by_code = {entry["code"]: entry for entry in meta["codes"]}

    gap = 2
    cell = side * args.scale + gap
    rows = (n_codes + args.cols - 1) // args.cols
    sheet = np.zeros((rows * cell, args.cols * cell, 3), dtype=np.uint8)

    for code in range(n_codes):
        bits = [(code >> b) & 1 for b in range(latent)]
        image = np.array(evaluate(circuit, bits), dtype=np.uint8).reshape(side, side)
        big = np.kron(image, np.ones((args.scale, args.scale), dtype=np.uint8)) * 235
        r, c = divmod(code, args.cols)
        oy, ox = r * cell, c * cell
        h = w = side * args.scale
        sheet[oy:oy + h, ox:ox + w] = big[..., None]
        entry = by_code.get("".join(str(b) for b in bits))
        colour = (40, 200, 90) if (entry and entry.get("trained")) else (90, 90, 90)
        sheet[oy, ox:ox + w] = colour
        sheet[oy + h - 1, ox:ox + w] = colour
        sheet[oy:oy + h, ox] = colour
        sheet[oy:oy + h, ox + w - 1] = colour

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).resize(
        (args.cols * cell * 2, rows * cell * 2), Image.NEAREST
    ).save(args.out)
    fitted = sum(1 for e in meta["codes"] if e.get("trained"))
    print(f"{n_codes} codes ({fitted} fitted) -> {args.out}")
    print(f"ber on fitted {meta.get('ber')}, worst {meta.get('worst')} bits, "
          f"live latent {meta.get('effective_latent')}/{latent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
