#!/usr/bin/env python3
"""Report what each circuit wire can possibly depend on (its cone).

For a wired Boolean circuit, the set of inputs an output can be a function of is
computed exactly by unioning its two inputs' sets at every gate. That set is an
upper bound on expressiveness, and a small one is a hard ceiling: an output bit
that depends on one key bit can only ever be that bit or its inverse.

This is the measurement Checkpoint 12 uses to compare the shipped 32x32 model
against the 8x8 glyph witness, where the decoder computes only 16 of the 256
functions of a 3-bit key.

    python3 scripts/circuit_cones.py web/public/models/cats32-nand.json
    python3 scripts/circuit_cones.py web/public/sat/circuit.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def cones(circuit: dict) -> list[int]:
    """Dependency mask of every wire, in the circuit's own wire order."""
    n = circuit["n_inputs"]
    dep = [1 << i for i in range(n)] + [0, 0]  # wires n and n+1 are constants
    for gate in circuit["gates"]:
        dep.append(dep[gate["a"]] | dep[gate["b"]])
    return dep


def report(name: str, circuit: dict) -> None:
    dep = cones(circuit)
    outs = [dep[out] for out in circuit["outputs"]]
    sizes = sorted(bin(mask).count("1") for mask in outs)
    live = 0
    for mask in outs:
        live |= mask
    print(
        f"  {name}: {len(outs)} outputs over {circuit['n_inputs']} inputs, "
        f"{len(circuit['gates'])} gates\n"
        f"    inputs any output depends on: {bin(live).count('1')}/{circuit['n_inputs']}\n"
        f"    cone size: min {sizes[0]}, median {sizes[len(sizes) // 2]}, max {sizes[-1]}\n"
        f"    outputs with <=1 input in the cone: {sum(1 for s in sizes if s <= 1)}\n"
        f"    outputs with a distinct dependency set: {len(set(outs))}"
    )


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    for arg in sys.argv[1:]:
        path = Path(arg)
        data = json.loads(path.read_text())
        print(f"== {path.name}")
        if "encoder" in data and "decoder" in data:
            for half in ("encoder", "decoder"):
                report(half, data[half])
        else:
            report("circuit", data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
