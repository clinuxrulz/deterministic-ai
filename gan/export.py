"""Turn a trained generator checkpoint into a literal NAND circuit.

The softmax over candidates is replaced by its argmax, so the exported object
is the exact gate network that inference runs: `{"n_inputs", "gates", "outputs"}`
in the same JSON shape the Rust `Circuit` uses.

    python gan/export.py --checkpoint runs/gan16/generator.pt --out model.json

Without PyTorch (e.g. on this phone) you can still validate the format:

    python gan/export.py --self-test --out /tmp/random_circuit.json
"""

import argparse
import json
import random


def flat_to_wire(flat, sizes, gate_offsets, n_inputs):
    """Map a flat activation index to a circuit wire id.

    Wires: inputs 0..n_inputs-1, then gates from n_inputs+2 (n_inputs and
    n_inputs+1 are the constant 0 and 1 wires).
    """
    if flat < sizes[0]:
        return flat
    for layer in range(1, len(sizes)):
        start = sum(sizes[:layer])
        end = start + sizes[layer]
        if start <= flat < end:
            return n_inputs + 2 + gate_offsets[layer - 1] + (flat - start)
    raise ValueError(f"flat index {flat} out of range")


def build_circuit(sizes, wiring, alpha, beta):
    n_inputs = sizes[0]
    gate_offsets = [0]
    for size in sizes[1:]:
        gate_offsets.append(gate_offsets[-1] + size)
    gates = []
    for layer, idx in enumerate(wiring):
        gate_start = n_inputs + 2 + gate_offsets[layer]
        a_arg = alpha[layer]
        b_arg = beta[layer]
        for gate in range(len(idx)):
            ca = max(range(len(a_arg[gate])), key=lambda c: a_arg[gate][c])
            cb = max(range(len(b_arg[gate])), key=lambda c: b_arg[gate][c])
            a = flat_to_wire(int(idx[gate][ca]), sizes, gate_offsets, n_inputs)
            b = flat_to_wire(int(idx[gate][cb]), sizes, gate_offsets, n_inputs)
            wire = gate_start + gate
            if a >= wire or b >= wire:
                raise ValueError(f"gate {wire} reads ahead: a={a} b={b}")
            gates.append({"op": "nand", "a": a, "b": b})
    last = gate_offsets[-2]
    outputs = [n_inputs + 2 + last + i for i in range(sizes[-1])]
    return {"n_inputs": n_inputs, "gates": gates, "outputs": outputs}


def from_checkpoint(path):
    import torch  # only needed here

    payload = torch.load(path, map_location="cpu", weights_only=False)
    sizes = payload["sizes"]
    wiring = [w.tolist() for w in payload["wiring"]]
    alpha = [a.tolist() for a in payload["alpha"]]
    beta = [b.tolist() for b in payload["beta"]]
    return build_circuit(sizes, wiring, alpha, beta)


def self_test(sizes=(64, 128, 256), k=4, seed=7):
    """A pure-Python stand-in so the JSON layout can be checked without torch."""
    rng = random.Random(seed)
    wiring = []
    for layer in range(1, len(sizes)):
        available = sum(sizes[:layer])
        wiring.append([[rng.randrange(available) for _ in range(k)] for _ in range(sizes[layer])])
    alpha = [[[rng.random() for _ in range(k)] for _ in range(n)] for n in sizes[1:]]
    beta = [[[rng.random() for _ in range(k)] for _ in range(n)] for n in sizes[1:]]
    return build_circuit(list(sizes), wiring, alpha, beta)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--out", required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        circuit = self_test()
    else:
        if not args.checkpoint:
            parser.error("--checkpoint is required unless --self-test is given")
        circuit = from_checkpoint(args.checkpoint)

    with open(args.out, "w") as handle:
        json.dump(circuit, handle)
    print(
        f"wrote {args.out}: {circuit['n_inputs']} inputs, {len(circuit['gates'])} gates, "
        f"{len(circuit['outputs'])} outputs"
    )


if __name__ == "__main__":
    main()
