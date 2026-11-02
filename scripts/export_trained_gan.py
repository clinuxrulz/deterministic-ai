#!/usr/bin/env python3
"""Turn a `train-nand` autoencoder into the demo's generator-tab metadata.

The SAT tab gets its circuit from ``scripts/sat_ae.py --mode fit``, which pins
one code per drawing. This is the other track: ``train-nand`` learns an
autoencoder, and the *decoder* is the generator. The encoder is used once, here,
to find the key each training drawing ended up with; the exported circuit is
only the decoder (key -> 32x32 bits).

The output directory has the same shape as the SAT export, so the web tab can
read either::

    circuit.json   plain Circuit: n_inputs = key bits, outputs = side*side
    meta.json      {kind, side, latent, gates, depth, trained, free, slack,
                    ber, exact, worst, effective_latent, source, codes[], note}

Every code in the key space is evaluated -- 2**latent decodes -- so the tab's
"live key bits" and "distinct codes" are measured, not estimated. That is fine
for an 8-bit key (256 decodes) and is refused above 12 bits.

    python3 scripts/export_trained_gan.py \\
        --model runs/quickdraw_grad/model.json \\
        --targets data/quickdraw_cats_32.json \\
        --out web/public/gan/grad
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate(circuit: dict, inputs: list[int]) -> list[int]:
    """Reference evaluator, same semantics as the browser's run_circuit."""
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


def depth_of(circuit: dict) -> int:
    """Longest chain of gates from any input to any output."""
    n = circuit["n_inputs"]
    level = [0] * (n + 2 + len(circuit["gates"]))
    for i, gate in enumerate(circuit["gates"]):
        level[n + 2 + i] = 1 + max(level[gate["a"]], level[gate["b"]])
    return max((level[out] for out in circuit["outputs"]), default=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="runs/quickdraw_grad/model.json")
    parser.add_argument("--targets", default="data/quickdraw_cats_32.json")
    parser.add_argument("--out", default="web/public/gan/grad")
    parser.add_argument("--limit", type=int, default=0, help="0 = every drawing")
    args = parser.parse_args()

    model = json.loads(Path(args.model).read_text())
    payload = json.loads(Path(args.targets).read_text())
    side = int(payload["side"])
    targets = [[int(b) for b in bits] for bits in payload["targets"]]
    if args.limit:
        targets = targets[: args.limit]
    n_out = side * side

    key_bits = int(model["config"]["key_bits"])
    if key_bits > 12:
        raise SystemExit(f"refusing to evaluate all 2**{key_bits} codes; "
                         "key_bits must be <= 12")
    if len(model["decoder"]["outputs"]) != n_out:
        raise SystemExit(f"decoder has {len(model['decoder']['outputs'])} outputs, "
                         f"targets are {side}x{side}")

    encoder, decoder = model["encoder"], model["decoder"]
    n_codes = 1 << key_bits

    # Every image the wiring can produce, in code order (little-endian key).
    images: list[list[int]] = []
    for code in range(n_codes):
        bits = [(code >> b) & 1 for b in range(key_bits)]
        images.append(evaluate(decoder, bits))

    # Which code did the encoder give each drawing, and how close is the decode?
    codes = [
        {"code": "".join(str((p >> b) & 1) for b in range(key_bits)),
         "trained": False, "wrong": None, "ink": sum(img) / n_out}
        for p, img in enumerate(images)
    ]
    seen: dict[int, int] = {}
    wrong_by_code: dict[int, int] = {}
    for index, cat in enumerate(targets):
        key = evaluate(encoder, cat)
        code = sum(bit << i for i, bit in enumerate(key))
        wrong = sum(a != b for a, b in zip(images[code], cat))
        if code in seen:
            # Two drawings landed on one code: keep the closer decode and
            # report the collision rather than silently overwriting.
            wrong_by_code[code] = min(wrong_by_code[code], wrong)
        else:
            seen[code] = index
            wrong_by_code[code] = wrong

    for code, wrong in wrong_by_code.items():
        codes[code]["trained"] = True
        codes[code]["wrong"] = wrong

    distinct = len({tuple(img) for img in images})
    live = 0
    for bit in range(key_bits):
        if any(images[p ^ (1 << bit)] != images[p] for p in range(n_codes)):
            live += 1

    errors = list(wrong_by_code.values())
    total = sum(errors)
    collisions = len(targets) - len(seen)
    ber = round(total / (len(errors) * n_out), 4)
    exact = sum(w == 0 for w in errors)
    worst = max(errors, default=0)
    meta = {
        "kind": "grad",
        "side": side,
        "latent": key_bits,
        "gates": len(decoder["gates"]),
        "depth": depth_of(decoder),
        "trained": len(seen),
        "free": n_codes - len(seen),
        "slack": 0,
        "ber": ber,
        "exact": exact,
        "worst": worst,
        "distinct": distinct,
        "effective_latent": live,
        "source": Path(args.model).name,
        "collisions": collisions,
        "codes": codes,
        "note": (
            f"Gradient track. {len(targets)} Quick Draw cats were compressed "
            f"through a {key_bits}-bit key by the NAND trainer (Gödel "
            f"relaxation z = 1 - min(u,v), then snapped to hard gates). The "
            f"circuit here is that model's decoder: {len(decoder['gates'])} "
            f"literal NAND gates, no weights. The encoder assigned "
            f"{len(seen)} distinct codes to the {len(targets)} drawings "
            f"({collisions} collisions); those codes reconstruct their "
            f"cat at {ber * 100:.1f}% wrong bits on average, and the "
            f"other {n_codes - len(seen)} codes are what the same wiring invents. "
            f"Compare the SAT tab: it fits 4 codes bit-exactly at the same "
            f"gate count."
        ),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "circuit.json").write_text(json.dumps(decoder))
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"{len(targets)} cats -> {len(seen)} codes of {n_codes} "
          f"({meta['collisions']} collisions)")
    print(f"decoder: {len(decoder['gates'])} gates, depth {meta['depth']}, "
          f"{live}/{key_bits} live key bits, {distinct}/{n_codes} distinct")
    print(f"reconstruction: ber {meta['ber']:.4f}, exact {meta['exact']}, "
          f"worst {meta['worst']} bits")
    print(f"wrote {out}/circuit.json and meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
