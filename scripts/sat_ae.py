#!/usr/bin/env python3
"""SAT-based wiring search for a NAND autoencoder.

The trainer in ``crates/cli/src/nand.rs`` searches wiring with gradients over
softmax distributions. This asks the same question exactly, with a SAT solver:
given a fixed hierarchical shape, is there *any* choice of two candidate wires
per gate that maps the training bitmaps through the key and back onto
themselves?

Encoding, per gate and per training pattern:

* two indices (binary encoded, padded to a power of two) select the gate's
  inputs from the previous layer's candidate window;
* ``out = NAND(in_a, in_b)`` is Tseitin encoded;
* the last layer's outputs are hard equalities, or soft ones with ``--slack``
  (at most N wrong bits per pattern, sequential-counter cardinality).

Everything is unrolled once per pattern, so the CNF is a single circuit with
one copy per example. No free variables: the only latitude is the wiring.

Modes:

* ``full``     - autoencoder: fit the patterns through the key (default)
* ``encoder``  - encoder only: the key must simply *separate* the patterns
* ``decoder``  - decoder only, keys free but distinct
* ``functions``- how many functions of the key the decoder can compute at all
* ``selftest`` - generate a random wiring, let it define the targets, and check
                 the encoding accepts the witness. A selftest failure means the
                 encoder is wrong, not that the shape is infeasible.
* ``brute``    - enumerate every wiring of a tiny shape and compare with the
                 solver; the check that rules out an over-constrained encoding

Usage::

    python3 scripts/sat_ae.py --solver path/to/kissat --patterns 4 --slack 0
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# 8x8 bitmaps. Eight distinct shapes so a 3-bit key is exactly at capacity;
# with 8 patterns and a 2-bit key the task is provably impossible.
GLYPHS: dict[str, list[str]] = {
    "A": ["..####..", ".##..##.", "##....##", "##....##",
          "########", "##....##", "##....##", "##....##"],
    "B": ["#####...", "##..##..", "##..##..", "#####...",
          "##..##..", "##..##..", "##..##..", "#####..."],
    "C": ["..####..", ".##..##.", "##......", "##......",
          "##......", "##......", ".##..##.", "..####.."],
    "D": ["#####...", "##..##..", "##...##.", "##...##.",
          "##...##.", "##...##.", "##..##..", "#####..."],
    "E": ["########", "##......", "##......", "#####...",
          "##......", "##......", "##......", "########"],
    "F": ["########", "##......", "##......", "#####...",
          "##......", "##......", "##......", "##......"],
    "G": ["..####..", ".##..##.", "##......", "##......",
          "##..###.", "##...##.", ".##..##.", "..####.."],
    "H": ["##....##", "##....##", "##....##", "########",
          "##....##", "##....##", "##....##", "##....##"],
}


class Cnf:
    """Minimal DIMACS builder."""

    def __init__(self) -> None:
        self.n = 0
        self.clauses: list[list[int]] = []

    def var(self) -> int:
        self.n += 1
        return self.n

    def add(self, lits: list[int]) -> None:
        self.clauses.append(lits)

    def write(self, path: Path) -> None:
        with path.open("w") as fh:
            fh.write(f"p cnf {self.n} {len(self.clauses)}\n")
            for clause in self.clauses:
                fh.write(" ".join(map(str, clause)))
                fh.write(" 0\n")


def pad_candidates(cands: list[int]) -> list[int]:
    """Pad to a power of two by repeating the last entry.

    Binary index encoding needs a full code space: an unused code would leave
    the gate's inputs unconstrained, which lets the solver fabricate outputs
    without reading the key. Duplicates are harmless (a redundant choice).
    """
    size = 1
    while size < len(cands):
        size *= 2
    return cands + [cands[-1]] * (size - len(cands))


def candidate_window(prev: int, out: int, j: int) -> list[int]:
    if out * 2 == prev:  # halving layer: 2x2 window around the pair
        raw = [2 * j - 1, 2 * j, 2 * j + 1, 2 * j + 2]
    elif out == prev * 2:  # doubling layer: 3x1 window around the source
        raw = [j // 2 - 1, j // 2, j // 2 + 1]
    else:  # same size or bottleneck: every unit in the previous layer
        raw = list(range(prev))
    seen: list[int] = []
    for idx in raw:
        if 0 <= idx < prev and idx not in seen:
            seen.append(idx)
    return pad_candidates(seen)


def at_most_k(cnf: Cnf, xs: list[int], k: int) -> None:
    """Sequential counter (Sinz) for ``sum(xs) <= k``."""
    if k >= len(xs):
        return
    if k == 0:
        for x in xs:
            cnf.add([-x])
        return
    # s[i][j]: at least j+1 of xs[0..i] are true
    s = [[cnf.var() for _ in range(k)] for _ in xs]
    cnf.add([-xs[0], s[0][0]])
    for j in range(1, k):
        cnf.add([-s[0][j]])
    for i in range(1, len(xs)):
        cnf.add([-xs[i], s[i][0]])
        cnf.add([-s[i - 1][0], s[i][0]])
        for j in range(1, k):
            cnf.add([-xs[i], -s[i - 1][j - 1], s[i][j]])
            cnf.add([-s[i - 1][j], s[i][j]])
        cnf.add([-xs[i], -s[i - 1][k - 1]])


class Net:
    """Fixed shape, wiring chosen by the solver.

    Three gate types, because they are genuinely different beasts:

    * ``nand``  - ``out = NAND(in_a, in_b)``; a universal gate, and the only
                  kind the trainer uses.
    * ``xor3``  - ``out = in_a XOR in_b XOR bias`` with a SAT-chosen bias bit
                  per gate, so every gate is XOR or XNOR. This makes the whole
                  circuit an affine map over GF(2) (see docs/linearity.md), so
                  the comparison with NAND is also a comparison with a much
                  smaller function class.
    * ``xor2``  - ``out = in_a XOR in_b``; the same experiment without the bias
                  pin, which isolates what the bias actually buys.
    * ``mix``   - each gate gets a free type bit: NAND(a, b) or XOR(a, b) XOR
                  bias. One nonlinear gate is enough to escape the affine
                  class, so this measures how much of the 256-function ceiling
                  was the gate and how much was the shape.
    """

    def __init__(self, sizes: list[int], op: str = "nand") -> None:
        self.sizes = sizes
        self.op = op
        self.gates: list[dict] = []
        self.value_vars: list[list[list[int]]] = []
        self.selectors: list[list[int]] = []

    def key_layer(self) -> int:
        """Index of the narrowest layer: the key (or the input, if narrower)."""
        return min(range(len(self.sizes)), key=lambda i: self.sizes[i])

    def build(self, cnf: Cnf, inputs: list[list[int]], targets: list[list[int]],
              slack: int = 0, constrain_outputs: bool = True,
              free_inputs: bool = False) -> None:
        n_patterns = len(inputs)
        if free_inputs:
            # Each key bit is a fresh variable per pattern: the decoder must
            # work for every pattern's own code. The decoder has no bias
            # inputs, so a constant key yields a constant output; which codes
            # are usable is part of the problem.
            self.value_vars = [
                [[cnf.var() for _ in range(n_patterns)] for _ in range(self.sizes[0])]
            ]
        else:
            const_true = cnf.var()
            const_false = cnf.var()
            cnf.add([const_true])
            cnf.add([-const_false])
            self.value_vars = [
                [[const_true if inputs[p][i] else const_false for p in range(n_patterns)]
                 for i in range(self.sizes[0])]
            ]
        for layer in range(1, len(self.sizes)):
            prev, out = self.sizes[layer - 1], self.sizes[layer]
            planes: list[list[int]] = []
            for j in range(out):
                cands = candidate_window(prev, out, j)
                bits = (len(cands) - 1).bit_length()
                a_bits = [cnf.var() for _ in range(bits)]
                b_bits = [cnf.var() for _ in range(bits)]
                vals = [cnf.var() for _ in range(n_patterns)]
                planes.append(vals)
                gate = {"layer": layer, "node": j, "cands": cands,
                        "a_bits": a_bits, "b_bits": b_bits}
                if self.op in ("xor3", "mix"):
                    # One bias bit per gate, shared by every pattern (it is
                    # part of the circuit, not part of the data): bias 0 gives
                    # XOR, bias 1 gives XNOR. Still affine over GF(2).
                    gate["bias"] = cnf.var()
                if self.op == "mix":
                    # kind 0 = NAND(a, b), kind 1 = XOR(a, b) ^ bias.
                    gate["kind"] = cnf.var()
                self.gates.append(gate)
                for p in range(n_patterns):
                    in_a = cnf.var()
                    in_b = cnf.var()
                    for idx, src in enumerate(cands):
                        src_v = self.value_vars[layer - 1][src][p]
                        match_a = [(b if (idx >> k) & 1 else -b)
                                   for k, b in enumerate(a_bits)]
                        cnf.add([-m for m in match_a] + [-in_a, src_v])
                        cnf.add([-m for m in match_a] + [in_a, -src_v])
                        match_b = [(b if (idx >> k) & 1 else -b)
                                   for k, b in enumerate(b_bits)]
                        cnf.add([-m for m in match_b] + [-in_b, src_v])
                        cnf.add([-m for m in match_b] + [in_b, -src_v])
                    out_v = vals[p]
                    guard = [gate["kind"]] if self.op == "mix" else []
                    if self.op in ("nand", "mix"):
                        # With ``guard = [kind]`` these read "if kind is 0";
                        # with no guard they are the plain NAND clauses.
                        cnf.add(guard + [-out_v, -in_a, -in_b])
                        cnf.add(guard + [out_v, in_a])
                        cnf.add(guard + [out_v, in_b])
                    if self.op in ("xor2", "xor3", "mix"):
                        # out = in_a ^ in_b (^ bias). One clause per input
                        # combination, forbidding the single output that would
                        # violate it: a clause is falsified exactly by the
                        # assignment its literals all disagree with, so the
                        # inputs carry ``-var if bit else var`` (falsified at
                        # ``bit``) and the output carries the opposite of the
                        # required parity.
                        guard_x = [-gate["kind"]] if self.op == "mix" else []
                        inputs = (in_a, in_b) if self.op == "xor2" else (in_a, in_b, gate["bias"])
                        for combo in itertools.product((0, 1), repeat=len(inputs)):
                            parity = 0
                            for bit in combo:
                                parity ^= bit
                            clause = [-var if bit else var
                                      for bit, var in zip(combo, inputs)]
                            clause.append(out_v if parity else -out_v)
                            cnf.add(guard_x + clause)
            self.value_vars.append(planes)

        # Output equalities, optionally soft with a per-pattern budget.
        if not constrain_outputs:
            return
        for p in range(n_patterns):
            selectors = []
            for bit in range(self.sizes[-1]):
                out_v = self.value_vars[-1][bit][p]
                lit = out_v if targets[p][bit] else -out_v
                if slack <= 0:
                    cnf.add([lit])
                else:
                    # sel is true exactly when this bit is wrong, so the
                    # cardinality bound counts mismatches rather than letting
                    # the solver leave every selector false.
                    sel = cnf.var()
                    selectors.append(sel)
                    cnf.add([-sel, -lit])
                    cnf.add([sel, lit])
            if slack > 0:
                at_most_k(cnf, selectors, slack)
            self.selectors.append(selectors)

    def separate(self, cnf: Cnf, layer: int = -1) -> None:
        """Require every pair of patterns to land on a different key-level value."""
        key = self.value_vars[layer]  # key[node][pattern]
        for p in range(len(key[0])):
            for q in range(p + 1, len(key[0])):
                diff = []
                for node in range(len(key)):
                    a, b = key[node][p], key[node][q]
                    e = cnf.var()
                    cnf.add([-a, -b, -e])
                    cnf.add([a, b, -e])
                    cnf.add([a, -b, e])
                    cnf.add([-a, b, e])
                    diff.append(e)
                cnf.add(diff)


def parse_model(path: Path) -> dict[int, bool]:
    assignment: dict[int, bool] = {}
    for line in path.read_text().splitlines():
        if not line.startswith("v"):
            continue
        for lit in line[1:].split():
            value = int(lit)
            if value == 0:
                continue
            assignment[abs(value)] = value > 0
    return assignment


def read_index(assignment: dict[int, bool], bits: list[int]) -> int:
    index = 0
    for k, var in enumerate(bits):
        if assignment.get(var, False):
            index |= 1 << k
    return index


def simulate_wiring(net: Net, wiring: list[tuple[int, ...]], pattern: list[int]) -> list[int]:
    values = [list(pattern)]
    for layer in range(1, len(net.sizes)):
        planes = [0] * net.sizes[layer]
        for gate, choice in zip(net.gates, wiring):
            if gate["layer"] != layer:
                continue
            src_a = values[layer - 1][gate["cands"][choice[0]]]
            src_b = values[layer - 1][gate["cands"][choice[1]]]
            if net.op == "nand":
                planes[gate["node"]] = 0 if (src_a and src_b) else 1
            elif net.op == "xor2":
                planes[gate["node"]] = src_a ^ src_b
            elif net.op == "xor3":
                planes[gate["node"]] = src_a ^ src_b ^ choice[2]
            else:  # mix: choice is (index_a, index_b, bias, kind)
                if choice[3]:
                    planes[gate["node"]] = src_a ^ src_b ^ choice[2]
                else:
                    planes[gate["node"]] = 0 if (src_a and src_b) else 1
        values.append(planes)
    return values[-1]


def simulate(net: Net, assignment: dict[int, bool], pattern: list[int]) -> list[int]:
    wiring = []
    for g in net.gates:
        choice = (read_index(assignment, g["a_bits"]), read_index(assignment, g["b_bits"]))
        if net.op in ("xor3", "mix"):
            choice += (1 if assignment.get(g["bias"], False) else 0,)
        if net.op == "mix":
            choice += (1 if assignment.get(g["kind"], False) else 0,)
        wiring.append(choice)
    return simulate_wiring(net, wiring, pattern)


def random_wiring(net: Net, rng: random.Random) -> list[tuple[int, ...]]:
    """A wiring the solver must be able to recover (used by the selftests)."""
    wiring = []
    for gate in net.gates:
        choice = (rng.randrange(len(gate["cands"])), rng.randrange(len(gate["cands"])))
        if net.op in ("xor3", "mix"):
            choice += (rng.getrandbits(1),)
        if net.op == "mix":
            choice += (rng.getrandbits(1),)
        wiring.append(choice)
    return wiring


def run(solver: str, cnf: Cnf, timeout: float, workdir: Path):
    cnf_path = workdir / "net.cnf"
    out_path = workdir / "net.out"
    cnf.write(cnf_path)
    started = time.time()
    try:
        with out_path.open("w") as fh:
            subprocess.run([solver, str(cnf_path)], stdout=fh, stderr=subprocess.DEVNULL,
                           timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return "timeout", time.time() - started, None
    elapsed = time.time() - started
    text = out_path.read_text()
    if any(line.startswith("s SATISFIABLE") for line in text.splitlines()):
        return "sat", elapsed, parse_model(out_path)
    if any(line.startswith("s UNSATISFIABLE") for line in text.splitlines()):
        return "unsat", elapsed, None
    return "unknown", elapsed, None


def circuit_json(net: "Net", wiring: list[tuple[int, int]], base_layer: int,
                 last_layer: int) -> dict:
    """Serialise layers ``base_layer+1 .. last_layer`` as a ``Circuit`` dict.

    Wire numbering follows ``crates/core/src/circuit.rs``: inputs are wires
    ``0..n_inputs-1``, wires ``n_inputs`` and ``n_inputs+1`` are the constants 0
    and 1, and gate ``i``'s output is wire ``n_inputs + 2 + i``. Layer
    ``base_layer`` is the sub-circuit's input layer, so the encoder and the
    decoder can be exported separately even though the SAT net is one chain.
    """
    n_inputs = net.sizes[base_layer]
    pos = {(g["layer"], g["node"]): i for i, g in enumerate(net.gates)}
    wire = {(base_layer, i): i for i in range(n_inputs)}
    gates: list[dict] = []
    for layer in range(base_layer + 1, last_layer + 1):
        for node in range(net.sizes[layer]):
            index = pos[(layer, node)]
            cands = net.gates[index]["cands"]
            index_a, index_b = wiring[index][0], wiring[index][1]
            if net.op == "nand" or (net.op == "mix" and not wiring[index][3]):
                op = "nand"
            else:
                # bias 0 is XOR, bias 1 is XNOR -- the same gate with the
                # constant folded in, which is exactly what the SAT bit means.
                op = "xnor" if net.op in ("xor3", "mix") and wiring[index][2] else "xor"
            gates.append({"op": op,
                          "a": wire[(layer - 1, cands[index_a])],
                          "b": wire[(layer - 1, cands[index_b])]})
            wire[(layer, node)] = n_inputs + 2 + len(gates) - 1
    outputs = [wire[(last_layer, node)] for node in range(net.sizes[last_layer])]
    return {"n_inputs": n_inputs, "gates": gates, "outputs": outputs}


def eval_circuit(circuit: dict, inputs: list[int]) -> list[int]:
    """Reference evaluator for the exported JSON -- independent of `Net`."""
    n = circuit["n_inputs"]
    wires = list(inputs[:n]) + [0, 1] + [0] * len(circuit["gates"])
    for i, gate in enumerate(circuit["gates"]):
        a, b = wires[gate["a"]], wires[gate["b"]]
        if gate["op"] == "nand":
            wires[n + 2 + i] = 0 if (a and b) else 1
        elif gate["op"] == "xor":
            wires[n + 2 + i] = a ^ b
        elif gate["op"] == "xnor":
            wires[n + 2 + i] = 1 - (a ^ b)
        else:
            raise ValueError(f"unsupported op {gate['op']}")
    return [wires[out] for out in circuit["outputs"]]


def export_web(net: "Net", assignment: dict[int, bool], names: list[str],
               out_dir: Path, solver: str, elapsed: float) -> bool:
    """Write the witness as circuits the browser can run, plus its metadata."""
    wiring = []
    for g in net.gates:
        choice = (read_index(assignment, g["a_bits"]), read_index(assignment, g["b_bits"]))
        if net.op in ("xor3", "mix"):
            choice += (1 if assignment.get(g["bias"], False) else 0,)
        if net.op == "mix":
            choice += (1 if assignment.get(g["kind"], False) else 0,)
        wiring.append(choice)
    key_layer = net.key_layer()
    last = len(net.sizes) - 1
    encoder = circuit_json(net, wiring, 0, key_layer)
    decoder = circuit_json(net, wiring, key_layer, last)

    glyphs_meta = []
    for name in GLYPHS:
        bits = glyphs([name], net.sizes[0])[0]
        code = eval_circuit(encoder, bits)
        recon = eval_circuit(decoder, code)
        wrong = sum(a != b for a, b in zip(recon, bits))
        glyphs_meta.append({
            "name": name,
            "bits": bits,
            "code": "".join(str(b) for b in code),
            "trained": name in names,
            "wrong_bits": wrong,
        })
        if name in names and wrong:
            print(f"EXPORT BROKEN: trained glyph {name} decodes with {wrong} wrong bits")
            return False

    keys = [g["code"] for g in glyphs_meta]
    trained_codes = [g["code"] for g in glyphs_meta if g["trained"]]
    print(f"exported {len(encoder['gates'])}+{len(decoder['gates'])} gates; "
          f"trained codes {trained_codes}, all codes {keys}, "
          f"held-out errors {[g['wrong_bits'] for g in glyphs_meta if not g['trained']]}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "circuit.json").write_text(json.dumps(
        {"encoder": encoder, "decoder": decoder}, separators=(",", ":")))
    meta = {
        "side": 8,
        "key_bits": net.sizes[key_layer],
        "shape": "-".join(map(str, net.sizes)),
        "gates": len(net.gates),
        "solver": solver.rsplit("/", 1)[-1],
        "seconds": round(elapsed, 2),
        "trained": names,
        "trained_codes": trained_codes,
        "used_codes": sorted(set(keys)),
        "glyphs": glyphs_meta,
        "infeasible": (
            "Kissat proves no wiring of this shape fits four glyphs: the "
            "3-level decoder can compute only 16 of the 256 functions of a "
            "3-bit key, so each output bit is identical on at least two of "
            "any four codes."
        ),
        "note": (
            "Every gate is NAND, so the circuit is nothing but wire numbers. "
            "The key has 3 bits, which is 8 codes, and only the trained "
            "ones were asked for: the other codes are whatever the decoder "
            "happens to compute there."
        ),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return True


def glyphs(names: list[str], width: int) -> list[list[int]]:
    out = []
    for name in names:
        rows = GLYPHS[name]
        out.append([1 if ch == "#" else 0 for row in rows for ch in row])
    assert all(len(p) == width for p in out), f"glyph is not {width} bits wide"
    return out


def selftest(solver: str, sizes: list[int], n_patterns: int, timeout: float,
             seed: int, op: str = "nand") -> bool:
    """A random wiring defines the targets, so the instance must be SAT."""
    rng = random.Random(seed)
    inputs = [[rng.getrandbits(1) for _ in range(sizes[0])] for _ in range(n_patterns)]
    probe = Net(sizes, op)
    probe.build(Cnf(), inputs, inputs, 0)
    wiring = random_wiring(probe, rng)
    targets = [simulate_wiring(probe, wiring, p) for p in inputs]

    cnf = Cnf()
    net = Net(sizes, op)
    net.build(cnf, inputs, targets, 0)
    for gate, choice in zip(net.gates, wiring):
        index_a, index_b = choice[0], choice[1]
        for k, var in enumerate(gate["a_bits"]):
            cnf.add([var if (index_a >> k) & 1 else -var])
        for k, var in enumerate(gate["b_bits"]):
            cnf.add([var if (index_b >> k) & 1 else -var])
        if op in ("xor3", "mix"):
            cnf.add([gate["bias"] if choice[2] else -gate["bias"]])
        if op == "mix":
            cnf.add([gate["kind"] if choice[3] else -gate["kind"]])

    workdir = Path(tempfile.mkdtemp(prefix="sat_ae_"))
    status, elapsed, model = run(solver, cnf, timeout, workdir)
    ok = status == "sat"
    if ok:
        ok = all(simulate(net, model, p) == t for p, t in zip(inputs, targets))
    print(f"selftest {op} {len(sizes)}-layer, {n_patterns} patterns: "
          f"{'ok' if ok else 'FAILED'} ({status} in {elapsed:.1f}s, "
          f"{cnf.n} vars, {len(cnf.clauses)} clauses)")
    return ok


def brute(solver: str, sizes: list[int], n_patterns: int, trials: int, seed: int) -> bool:
    """Compare SAT against exhaustive wiring enumeration on a tiny shape.

    Small shapes have few candidate windows, so every wiring can be tried. If
    the solver says UNSAT where brute force finds a witness, the encoding is
    missing solutions -- which would make every UNSAT result untrustworthy.
    """
    import itertools

    rng = random.Random(seed)
    probe = Net(sizes)
    probe.build(Cnf(), [[0] * sizes[0]] * n_patterns,
                [[0] * sizes[-1]] * n_patterns, 0, constrain_outputs=False)
    choices = [list(itertools.product(range(len(g["cands"])), repeat=2))
               for g in probe.gates]
    combos = 1
    for c in choices:
        combos *= len(c)
    trials = max(3, min(trials, 200_000 // combos))
    print(f"brute force {sizes}: {len(probe.gates)} gates, {combos} wirings, "
          f"{trials} instances", flush=True)

    ok = True
    for trial in range(trials):
        inputs = [[rng.getrandbits(1) for _ in range(sizes[0])]
                  for _ in range(n_patterns)]
        if trial % 2 == 0:
            # Witness targets: realisable by construction, so SAT must win.
            wire = [(rng.randrange(len(g["cands"])), rng.randrange(len(g["cands"])))
                    for g in probe.gates]
            targets = [simulate_wiring(probe, wire, p) for p in inputs]
        else:
            targets = [[rng.getrandbits(1) for _ in range(sizes[-1])]
                       for _ in range(n_patterns)]
        found = any(
            all(simulate_wiring(probe, w, p) == t for p, t in zip(inputs, targets))
            for w in itertools.product(*choices))

        cnf = Cnf()
        net = Net(sizes)
        net.build(cnf, inputs, targets, 0)
        workdir = Path(tempfile.mkdtemp(prefix="sat_brute_"))
        status, _, _ = run(solver, cnf, 300.0, workdir)
        sat = status == "sat"
        if sat != found:
            ok = False
            print(f"  trial {trial}: MISMATCH (brute {'sat' if found else 'unsat'}, "
                  f"solver {status})", flush=True)
        else:
            print(f"  trial {trial}: ok ({'sat' if sat else 'unsat'})", flush=True)
    print(f"brute force agreement: {'ok' if ok else 'FAILED'} on {trials} random instances")
    return ok


def slacktest(solver: str, sizes: list[int], n_patterns: int, slack: int,
              corrupt: int, timeout: float, seed: int, op: str = "nand") -> bool:
    """Corrupt a witness and require slack to accept it within its bound.

    The witness fits with exactly `corrupt` wrong bits in pattern 0, so the
    instance must be SAT and every pattern's error must stay within `slack`.
    A bound that is not enforced, or one enforced on *matches*, fails here.
    """
    rng = random.Random(seed)
    inputs = [[rng.getrandbits(1) for _ in range(sizes[0])] for _ in range(n_patterns)]
    probe = Net(sizes, op)
    probe.build(Cnf(), inputs, inputs, 0)
    wiring = random_wiring(probe, rng)
    targets = [simulate_wiring(probe, wiring, p) for p in inputs]
    for bit in rng.sample(range(sizes[-1]), corrupt):
        targets[0][bit] ^= 1

    cnf = Cnf()
    net = Net(sizes, op)
    net.build(cnf, inputs, targets, slack)
    workdir = Path(tempfile.mkdtemp(prefix="sat_slack_"))
    status, elapsed, model = run(solver, cnf, timeout, workdir)
    ok = status == "sat"
    if ok:
        wrong = [sum(a != b for a, b in zip(simulate(net, model, p), t))
                 for p, t in zip(inputs, targets)]
        ok = all(w <= slack for w in wrong)
    print(f"slacktest {op} {len(sizes)}-layer, {n_patterns} patterns, slack {slack}, "
          f"{corrupt} corrupt bits: {'ok' if ok else 'FAILED'} "
          f"({status} in {elapsed:.1f}s)")
    return ok


def function_table(solver: str, sizes: list[int], timeout: float, op: str = "nand") -> int:
    """How many functions of the key can this decoder compute?

    Only meaningful for a 3-bit key: 2^(2^3) = 256 functions, one SAT call each.
    This is the measurement that explains the glyph frontier -- the trainer's
    decoder turns out to compute 16 of the 256, and no depth fixes it.
    """
    bits = sizes[0]
    if bits > 3:
        print(f"function_table: {bits}-bit key needs 2^{2 ** bits} calls; use <= 3")
        return 0
    codes = [[(c >> b) & 1 for b in range(bits)] for c in range(1 << bits)]
    found = []
    for f in range(1 << (1 << bits)):
        targets = [[(f >> c) & 1] * sizes[-1] for c in range(1 << bits)]
        cnf = Cnf()
        net = Net(sizes, op)
        net.build(cnf, codes, targets, 0)
        status, _, _ = run(solver, cnf, timeout, Path(tempfile.mkdtemp(prefix="sat_fn_")))
        if status == "sat":
            found.append(f)
    total = 1 << (1 << bits)
    print(f"decoder {op} {'-'.join(map(str, sizes))}: {len(found)}/{total} functions "
          f"of the {bits}-bit key realisable: "
          f"{' '.join(format(f, '02x') for f in found)}")
    return len(found)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shape", default="64,32,16,3,16,32,64",
                        help="layer widths, input first (default 64,32,16,3,16,32,64)")
    parser.add_argument("--patterns", type=int, default=8,
                        help="number of glyphs to fit (default all 8)")
    parser.add_argument("--glyphs", metavar="NAMES",
                        help="explicit glyph names, e.g. BCDE (default: the "
                             "first --patterns of ABCDEFGH)")
    parser.add_argument("--slack", type=int, default=0,
                        help="maximum wrong output bits per pattern")
    parser.add_argument("--mode", choices=["full", "encoder", "decoder", "selftest", "brute", "functions"],
                        default="full")
    parser.add_argument("--gate", choices=["nand", "xor3", "xor2", "mix"], default="nand",
                        help="nand = NAND(a,b); xor3 = XOR(a,b) plus a SAT-chosen "
                             "bias bit per gate (affine over GF(2)); xor2 = plain "
                             "2-input XOR, no bias; mix = a free type bit per gate "
                             "choosing NAND or XOR^bias")
    parser.add_argument("--solver", default="kissat", help="SAT solver binary")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--keep", help="copy the CNF here for inspection")
    parser.add_argument("--export", metavar="DIR",
                        help="write the witness as browser circuits + meta.json")
    parser.add_argument("--csv", action="store_true", help="one machine-readable line")
    args = parser.parse_args()

    sizes = [int(x) for x in args.shape.split(",")]
    if args.mode == "selftest":
        ok = all(selftest(args.solver, sizes, n, args.timeout, seed, args.gate)
                 for n, seed in [(1, 1), (2, 2), (3, 3), (4, 4), (8, 5)])
        ok = slacktest(args.solver, sizes, 4, 5, 5, args.timeout, 6, args.gate) and ok
        ok = slacktest(args.solver, sizes, 4, 0, 0, args.timeout, 7, args.gate) and ok
        return 0 if ok else 1
    if args.mode == "brute":
        if args.gate != "nand":
            print("brute force only enumerates NAND wirings; an XOR gate has "
                  "another wire choice (and xor3 a bias bit too), which blows "
                  "the 200k-instance budget")
            return 1
        return 0 if brute(args.solver, sizes, args.patterns, 12, 1) else 1
    if args.mode == "functions":
        function_table(args.solver, sizes, args.timeout, args.gate)
        return 0

    names = list(args.glyphs) if args.glyphs else list(GLYPHS)[: args.patterns]
    unknown = [n for n in names if n not in GLYPHS]
    if unknown:
        print(f"unknown glyph(s) {unknown}; choose from {''.join(GLYPHS)}")
        return 1
    n_patterns = len(names)
    if args.mode == "encoder":
        # Stop at the bottleneck: this asks whether the encoder alone can put
        # the patterns on distinct keys, with no reconstruction to satisfy.
        narrowest = min(range(len(sizes)), key=lambda i: sizes[i])
        sizes = sizes[: narrowest + 1]
    patterns = glyphs(names, sizes[0]) if args.mode != "decoder" else glyphs(names, 64)

    cnf = Cnf()
    net = Net(sizes, args.gate)
    if args.mode == "decoder":
        # Keys are free but must differ: this isolates the decoder's expressive
        # power from the encoder's.
        inputs = [[0] * sizes[0] for _ in range(n_patterns)]
        net.build(cnf, inputs, patterns, args.slack, free_inputs=True)
        net.separate(cnf, layer=0)
    else:
        net.build(cnf, patterns, patterns, args.slack,
                  constrain_outputs=args.mode == "full")
    if args.mode == "encoder":
        net.separate(cnf)

    workdir = Path(tempfile.mkdtemp(prefix="sat_ae_"))
    status, elapsed, model = run(args.solver, cnf, args.timeout, workdir)
    if args.keep:
        Path(args.keep).write_text((workdir / "net.cnf").read_text())

    shape = "-".join(map(str, sizes))
    if args.csv:
        print(f"{args.gate},{shape},{''.join(names)},{args.mode},{args.slack},{status},"
              f"{elapsed:.2f},{cnf.n},{len(cnf.clauses)}")
    else:
        print(f"shape {shape}, {''.join(names)} ({n_patterns} patterns), mode {args.mode}, "
              f"slack {args.slack}, gate {args.gate}")
        print(f"cnf: {cnf.n} vars, {len(cnf.clauses)} clauses")
        print(f"result: {status} in {elapsed:.1f}s")

    if status == "sat" and model is not None and not args.csv:
        if args.mode == "full":
            total = wrong_all = exact = 0
            for name, pattern in zip(names, patterns):
                recon = simulate(net, model, pattern)
                wrong = sum(a != b for a, b in zip(recon, pattern))
                if wrong > args.slack:
                    print(f"  BOUND VIOLATED: {name} has {wrong} wrong bits, "
                          f"slack is {args.slack}")
                wrong_all += wrong
                total += len(pattern)
                exact += wrong == 0
                print(f"  {name}: {wrong:3d} wrong bits")
            print(f"ber {wrong_all / total:.4f}, exact patterns {exact}/{len(patterns)}")
        key = net.value_vars[net.key_layer()]
        keys = ["".join(str(int(model.get(key[b][p], False))) for b in range(len(key)))
                for p in range(len(patterns))]
        print(f"keys {keys} ({len(set(keys))} distinct of {2 ** len(key)})")
        if args.export and args.mode == "full":
            if not export_web(net, model, names, Path(args.export), args.solver, elapsed):
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
