use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Primitive Boolean operations available to a circuit.
///
/// `Xor` and `Xnor` are affine over GF(2). `And`, `Or` and `Not` are what make
/// the full gate set non-linear (and, with `And`, functionally complete when
/// constants are available).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Op {
    Xor,
    Xnor,
    And,
    Nand,
    Or,
    Nor,
    Not,
}

impl Op {
    #[inline]
    pub fn apply(self, a: u64, b: u64) -> u64 {
        match self {
            Op::Xor => a ^ b,
            Op::Xnor => !(a ^ b),
            Op::And => a & b,
            Op::Nand => !(a & b),
            Op::Or => a | b,
            Op::Nor => !(a | b),
            Op::Not => !a,
        }
    }

    #[inline]
    pub fn is_unary(self) -> bool {
        matches!(self, Op::Not)
    }
}

/// One gate in a circuit.
///
/// Wire indices are resolved against the circuit:
/// - `0..n_inputs` are primary inputs
/// - `n_inputs` is the constant 0 wire
/// - `n_inputs + 1` is the constant 1 wire
/// - `n_inputs + 2 + i` is the output wire of gate `i`
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Gate {
    pub op: Op,
    pub a: u16,
    pub b: u16,
}

/// Which operations the evolutionary search is allowed to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateSet {
    /// Strict XOR (plus the constant wires). This computes affine maps over GF(2).
    XorOnly,
    /// Mostly XOR with a small budget of AND gates. Functionally complete.
    XorAnd,
    /// All available operations, with XOR still weighted heavily.
    Full,
    /// Every binary operation, uniformly. `Nand` and `Nor` alone are
    /// functionally complete, so this contains universal gate sets.
    Universal,
    /// NAND only. Functionally complete on its own, so a circuit is fully
    /// described by its wiring.
    NandOnly,
}

impl GateSet {
    /// Pick an operation from a uniform `0.0..1.0` roll. Keeping this
    /// RNG-agnostic lets the core crate stay free of `rand`.
    pub fn op_from_roll(self, r: f32) -> Op {
        match self {
            GateSet::XorOnly => Op::Xor,
            GateSet::XorAnd => {
                if r < 0.90 {
                    Op::Xor
                } else {
                    Op::And
                }
            }
            GateSet::Full => {
                if r < 0.45 {
                    Op::Xor
                } else if r < 0.55 {
                    Op::Xnor
                } else if r < 0.68 {
                    Op::And
                } else if r < 0.78 {
                    Op::Nand
                } else if r < 0.90 {
                    Op::Or
                } else if r < 0.97 {
                    Op::Nor
                } else {
                    Op::Not
                }
            }
            GateSet::NandOnly => Op::Nand,
            GateSet::Universal => {
                if r < 1.0 / 6.0 {
                    Op::Xor
                } else if r < 2.0 / 6.0 {
                    Op::Xnor
                } else if r < 3.0 / 6.0 {
                    Op::And
                } else if r < 4.0 / 6.0 {
                    Op::Nand
                } else if r < 5.0 / 6.0 {
                    Op::Or
                } else {
                    Op::Nor
                }
            }
        }
    }

    pub fn is_xor_only(self) -> bool {
        matches!(self, GateSet::XorOnly)
    }
}

/// A fixed-topology, feed-forward Boolean circuit.
///
/// Gates are stored in topological order by construction: gate `i` may only
/// read primary inputs, constants, or gates `0..i`. That makes evaluation a
/// single forward pass and makes cycles impossible without an explicit check.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Circuit {
    pub n_inputs: u16,
    pub gates: Vec<Gate>,
    pub outputs: Vec<u16>,
}

impl Circuit {
    #[inline]
    pub fn gate_start(&self) -> usize {
        self.n_inputs as usize + 2
    }

    #[inline]
    pub fn total_wires(&self) -> usize {
        self.gate_start() + self.gates.len()
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.n_inputs == 0 {
            return Err("circuit has no inputs".into());
        }
        if self.outputs.is_empty() {
            return Err("circuit has no outputs".into());
        }
        let total = self.total_wires();
        if total > u16::MAX as usize {
            return Err("circuit has too many wires for u16 indices".into());
        }
        for (i, gate) in self.gates.iter().enumerate() {
            let wire = self.gate_start() + i;
            if gate.a as usize >= wire {
                return Err(format!(
                    "gate {i} input a={} is not before wire {wire}",
                    gate.a
                ));
            }
            if !gate.op.is_unary() && gate.b as usize >= wire {
                return Err(format!(
                    "gate {i} input b={} is not before wire {wire}",
                    gate.b
                ));
            }
        }
        for (i, &out) in self.outputs.iter().enumerate() {
            if out as usize >= total {
                return Err(format!("output {i} points at invalid wire {out}"));
            }
        }
        Ok(())
    }

    /// Evaluate the circuit for a batch.
    ///
    /// `inputs` is `n_inputs * words` planes. `out` receives
    /// `outputs.len() * words` planes. `arena` is scratch space and is resized
    /// as needed; callers should reuse it across evaluations.
    pub fn eval_into(
        &self,
        inputs: &[u64],
        words: usize,
        arena: &mut Vec<u64>,
        out: &mut Vec<u64>,
    ) {
        let n_inputs = self.n_inputs as usize;
        debug_assert_eq!(inputs.len(), n_inputs * words);
        let total = self.total_wires();
        arena.clear();
        arena.resize(total * words, 0);
        arena[..n_inputs * words].copy_from_slice(inputs);

        // Constant 1 wire. Constant 0 is already zero.
        let one_base = (n_inputs + 1) * words;
        for w in 0..words {
            arena[one_base + w] = u64::MAX;
        }

        for (i, gate) in self.gates.iter().enumerate() {
            let dst = (self.gate_start() + i) * words;
            let a_base = gate.a as usize * words;
            let b_base = gate.b as usize * words;
            if gate.op.is_unary() {
                for w in 0..words {
                    arena[dst + w] = gate.op.apply(arena[a_base + w], 0);
                }
            } else {
                for w in 0..words {
                    let av = arena[a_base + w];
                    let bv = arena[b_base + w];
                    arena[dst + w] = gate.op.apply(av, bv);
                }
            }
        }

        out.clear();
        out.resize(self.outputs.len() * words, 0);
        for (i, &wire) in self.outputs.iter().enumerate() {
            let src = wire as usize * words;
            out[i * words..(i + 1) * words].copy_from_slice(&arena[src..src + words]);
        }
    }

    /// Number of gates that actually influence the outputs.
    pub fn active_gates(&self) -> usize {
        let total = self.total_wires();
        let mut active = vec![false; total];
        let mut stack = self.outputs.clone();
        while let Some(wire) = stack.pop() {
            let wire = wire as usize;
            if wire >= total || active[wire] {
                continue;
            }
            active[wire] = true;
            if wire >= self.gate_start() {
                let gate = &self.gates[wire - self.gate_start()];
                stack.push(gate.a);
                if !gate.op.is_unary() {
                    stack.push(gate.b);
                }
            }
        }
        active[self.gate_start()..].iter().filter(|&&a| a).count()
    }

    /// Create a random circuit with a fixed number of gates.
    ///
    /// Uses an internal SplitMix64 so the core crate does not need an RNG
    /// dependency and still produces reproducible circuits from a seed.
    pub fn random(
        n_inputs: usize,
        n_gates: usize,
        n_outputs: usize,
        gate_set: GateSet,
        seed: u64,
    ) -> Self {
        let mut rng = SplitMix64::new(seed);
        let mut gates = Vec::with_capacity(n_gates);
        for i in 0..n_gates {
            let wire = n_inputs + 2 + i;
            let a = rng.gen_index(wire) as u16;
            let b = rng.gen_index(wire) as u16;
            let op = gate_set.op_from_roll(rng.next_f32());
            gates.push(Gate { op, a, b });
        }

        let total = n_inputs + 2 + n_gates;
        let gate_start = n_inputs + 2;
        let mut outputs = Vec::with_capacity(n_outputs);
        for _ in 0..n_outputs {
            // Bias outputs toward gate wires, but occasionally allow a direct
            // primary input or constant (useful for identity/passthrough).
            let wire = if total > gate_start && rng.next_f32() < 0.90 {
                gate_start + rng.gen_index(n_gates)
            } else {
                rng.gen_index(total)
            };
            outputs.push(wire as u16);
        }

        Self {
            n_inputs: n_inputs as u16,
            gates,
            outputs,
        }
    }

    /// Return a copy with every gate that cannot influence an output removed.
    ///
    /// Dead gates are free to delete: by definition they are not on any
    /// output's dependency cone, so the function is identical. Wire indices are
    /// remapped, so this shrinks the circuit without changing behaviour.
    pub fn prune_dead(&self) -> Circuit {
        let gate_start = self.gate_start();
        let total = self.total_wires();
        let mut keep = vec![false; total];
        let mut stack = self.outputs.clone();
        while let Some(wire) = stack.pop() {
            let wire = wire as usize;
            if wire >= total || keep[wire] {
                continue;
            }
            keep[wire] = true;
            if wire >= gate_start {
                let gate = &self.gates[wire - gate_start];
                stack.push(gate.a);
                if !gate.op.is_unary() {
                    stack.push(gate.b);
                }
            }
        }

        let mut map = vec![u16::MAX; total];
        for (i, slot) in map.iter_mut().take(self.n_inputs as usize).enumerate() {
            *slot = i as u16;
        }
        map[self.n_inputs as usize] = self.n_inputs;
        map[self.n_inputs as usize + 1] = self.n_inputs + 1;

        let mut gates = Vec::new();
        for (i, gate) in self.gates.iter().enumerate() {
            let wire = gate_start + i;
            if !keep[wire] {
                continue;
            }
            let a = map[gate.a as usize];
            let b = map[gate.b as usize];
            debug_assert!(a != u16::MAX && b != u16::MAX);
            map[wire] = self.n_inputs + 2 + gates.len() as u16;
            gates.push(Gate { op: gate.op, a, b });
        }
        let outputs = self
            .outputs
            .iter()
            .map(|&wire| map[wire as usize])
            .collect();
        Circuit {
            n_inputs: self.n_inputs,
            gates,
            outputs,
        }
    }

    /// True if every gate is XOR (constants/affine offsets are still allowed).
    pub fn is_xor_only(&self) -> bool {
        self.gates.iter().all(|g| matches!(g.op, Op::Xor))
    }

    /// Number of gates that would remain if inversions were free.
    ///
    /// In an AND-Inverter Graph a complemented edge costs nothing, so gates
    /// that only negate one wire (`Not`, `Nand(x, x)`, `Nand(x, 1)`) collapse
    /// into their consumers. This counts the gates that survive that view —
    /// it is a size metric, not a gate count, and is reported alongside
    /// `active_gates` so the two are never conflated.
    pub fn aig_nodes(&self) -> usize {
        let c0 = self.n_inputs;
        let c1 = self.n_inputs + 1;
        let is_const = |w: u16| w == c0 || w == c1;
        // Only reachable gates count; an unpruned model would otherwise report
        // a misleadingly large number.
        let gate_start = self.gate_start();
        let total = self.total_wires();
        let mut active = vec![false; total];
        let mut stack = self.outputs.clone();
        while let Some(wire) = stack.pop() {
            let wire = wire as usize;
            if wire >= total || active[wire] {
                continue;
            }
            active[wire] = true;
            if wire >= gate_start {
                let gate = &self.gates[wire - gate_start];
                stack.push(gate.a);
                if !gate.op.is_unary() {
                    stack.push(gate.b);
                }
            }
        }
        self.gates
            .iter()
            .enumerate()
            .filter(|&(i, g)| {
                active[gate_start + i]
                    && !g.op.is_unary()
                    && !is_const(g.a)
                    && !is_const(g.b)
                    && g.a != g.b
            })
            .count()
    }

    /// Longest path from any input to any output, in gates.
    pub fn depth(&self) -> usize {
        let total = self.total_wires();
        let mut depth = vec![0usize; total];
        for (i, gate) in self.gates.iter().enumerate() {
            let wire = self.gate_start() + i;
            let d = depth[gate.a as usize].max(depth[gate.b as usize]) + 1;
            depth[wire] = d;
        }
        self.outputs
            .iter()
            .map(|&o| depth[o as usize])
            .max()
            .unwrap_or(0)
    }

    /// Exact simplification: constant folding, hash-consing (CSE) and
    /// dead-gate removal.
    ///
    /// Every rewrite applied here is a Boolean identity, so the result
    /// computes the *same function* as the original for every possible input,
    /// not merely for the training set. The datapath is untouched: same
    /// `n_inputs`, same output order, same wire convention.
    pub fn simplify(&self) -> (Circuit, SimplifyStats) {
        let n_in = self.n_inputs;
        // Wire convention: inputs are `0..n_inputs`, `n_inputs` is the
        // constant 0 and `n_inputs + 1` is the constant 1.
        let c0 = n_in;
        let c1 = n_in + 1;
        let total = self.total_wires();

        let mut map = vec![u16::MAX; total];
        let n_inputs = self.n_inputs as usize;
        for (i, slot) in map.iter_mut().take(n_inputs).enumerate() {
            *slot = i as u16;
        }
        map[n_inputs] = c0;
        map[n_inputs + 1] = c1;

        let mut interner = Interner {
            n_in,
            xor_only: self.is_xor_only(),
            gates: Vec::with_capacity(self.gates.len()),
            seen: HashMap::new(),
            deduped: 0,
        };
        let mut stats = SimplifyStats {
            gates_before: self.gates.len(),
            ..Default::default()
        };

        for (i, gate) in self.gates.iter().enumerate() {
            let a = map[gate.a as usize];
            let b = map[gate.b as usize];
            debug_assert!(a != u16::MAX && b != u16::MAX);

            let out = match gate.op {
                Op::Not => interner.inv(a),
                Op::Nand => {
                    if a == c0 || b == c0 {
                        c1
                    } else if a == c1 {
                        interner.inv(b)
                    } else if b == c1 || a == b {
                        interner.inv(a)
                    } else {
                        interner.node(Op::Nand, a, b)
                    }
                }
                Op::And => {
                    if a == c0 || b == c0 {
                        c0
                    } else if a == c1 {
                        b
                    } else if b == c1 || a == b {
                        a
                    } else {
                        interner.node(Op::And, a, b)
                    }
                }
                Op::Or => {
                    if a == c1 || b == c1 {
                        c1
                    } else if a == c0 {
                        b
                    } else if b == c0 || a == b {
                        a
                    } else {
                        interner.node(Op::Or, a, b)
                    }
                }
                Op::Nor => {
                    if a == c1 || b == c1 {
                        c0
                    } else if a == c0 || b == c0 {
                        interner.inv(if a == c0 { b } else { a })
                    } else if a == b {
                        interner.inv(a)
                    } else {
                        interner.node(Op::Nor, a, b)
                    }
                }
                Op::Xor => {
                    if a == c0 {
                        b
                    } else if b == c0 {
                        a
                    } else if a == c1 {
                        interner.inv(b)
                    } else if b == c1 {
                        interner.inv(a)
                    } else if a == b {
                        c0
                    } else {
                        interner.node(Op::Xor, a, b)
                    }
                }
                Op::Xnor => {
                    if a == c0 || b == c0 {
                        interner.inv(if a == c0 { b } else { a })
                    } else if a == c1 {
                        b
                    } else if b == c1 {
                        a
                    } else if a == b {
                        c1
                    } else {
                        interner.node(Op::Xnor, a, b)
                    }
                }
            };

            // Classify what the rewrite achieved: a constant wire means the
            // gate folded away; reusing one of its own inputs is an identity.
            if out == c0 || out == c1 {
                stats.folded_const += 1;
            } else if out == a || out == b {
                stats.folded_identity += 1;
            }

            map[self.gate_start() + i] = out;
        }

        let outputs = self
            .outputs
            .iter()
            .map(|&wire| map[wire as usize])
            .collect::<Vec<_>>();
        let built = Circuit {
            n_inputs: self.n_inputs,
            gates: interner.gates,
            outputs,
        };
        let pruned = built.prune_dead();
        stats.dead = built.gates.len() - pruned.gates.len();
        stats.gates_after = pruned.gates.len();
        stats.deduped = interner.deduped;
        (pruned, stats)
    }
}

/// Statistics from [`Circuit::simplify`], for logging and docs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SimplifyStats {
    pub gates_before: usize,
    pub gates_after: usize,
    /// Gates folded to the constant 0/1 wires.
    pub folded_const: usize,
    /// Gates replaced by an existing wire (`x AND 1`, `x XOR 0`, ...).
    pub folded_identity: usize,
    /// Gates merged into an identical earlier gate.
    pub deduped: usize,
    /// Gates dropped as unreachable after rewriting.
    pub dead: usize,
}

/// Hash-consing helper: builds each distinct node once.
struct Interner {
    n_in: u16,
    xor_only: bool,
    gates: Vec<Gate>,
    seen: HashMap<(u8, u16, u16), u16>,
    deduped: usize,
}

impl Interner {
    const INV: u8 = 0;

    fn push(&mut self, key: (u8, u16, u16), gate: Gate) -> u16 {
        if let Some(&wire) = self.seen.get(&key) {
            self.deduped += 1;
            return wire;
        }
        let wire = self.n_in + 2 + self.gates.len() as u16;
        self.gates.push(gate);
        self.seen.insert(key, wire);
        wire
    }

    /// Negation of one wire. Emitted as `Nand(x, x)`, or as `Xor(x, 1)` in an
    /// XOR-only circuit so that the affine invariant survives.
    fn inv(&mut self, x: u16) -> u16 {
        // NOT 0 = 1 and NOT 1 = 0.
        if x == self.n_in {
            return self.n_in + 1;
        }
        if x == self.n_in + 1 {
            return self.n_in;
        }
        let key = (Self::INV, x, 0);
        let existing = self.seen.get(&key).copied();
        if let Some(wire) = existing {
            self.deduped += 1;
            return wire;
        }
        let gate = if self.xor_only {
            Gate {
                op: Op::Xor,
                a: x,
                b: self.n_in + 1,
            }
        } else {
            Gate {
                op: Op::Nand,
                a: x,
                b: x,
            }
        };
        self.push(key, gate)
    }

    /// A genuine two-input gate. Inputs are already known to be distinct,
    /// non-constant wires.
    fn node(&mut self, op: Op, a: u16, b: u16) -> u16 {
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        let tag = match op {
            Op::And => 1,
            Op::Nand => 2,
            Op::Or => 3,
            Op::Nor => 4,
            Op::Xor => 5,
            Op::Xnor => 6,
            Op::Not => Self::INV,
        };
        self.push((tag, lo, hi), Gate { op, a: lo, b: hi })
    }
}

/// Small deterministic RNG used only for random circuit initialisation.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    fn gen_index(&mut self, upper: usize) -> usize {
        if upper == 0 {
            0
        } else {
            (self.next_u64() % upper as u64) as usize
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xor_gate_truth_tables() {
        assert_eq!(Op::Xor.apply(0, 0), 0);
        assert_eq!(Op::Xor.apply(0, 1), 1);
        assert_eq!(Op::Xor.apply(1, 0), 1);
        assert_eq!(Op::Xor.apply(1, 1), 0);
        assert_eq!(Op::And.apply(1, 1), 1);
        assert_eq!(Op::Nand.apply(1, 1) & 1, 0);
        assert_eq!(Op::Nand.apply(1, 0) & 1, 1);
        assert_eq!(Op::Nor.apply(0, 0) & 1, 1);
        assert_eq!(Op::Nor.apply(0, 1) & 1, 0);
        assert_eq!(Op::Not.apply(0, 0), u64::MAX);
    }

    #[test]
    fn xor_identity_circuit() {
        // out0 = x0, out1 = NOT x1 via x1 XOR 1.
        let circuit = Circuit {
            n_inputs: 2,
            gates: vec![Gate {
                op: Op::Xor,
                a: 1,
                b: 3, // constant 1
            }],
            outputs: vec![0, 4],
        };
        circuit.validate().unwrap();
        let mut arena = Vec::new();
        let mut out = Vec::new();
        circuit.eval_into(&[1, 0], 1, &mut arena, &mut out);
        assert_eq!(out[0] & 1, 1);
        assert_eq!(out[1] & 1, 1);
    }

    #[test]
    fn random_circuits_validate_and_evaluate() {
        for (seed, gates) in [0usize, 1, 8, 64, 256].into_iter().enumerate() {
            let c = Circuit::random(16, gates, 8, GateSet::Full, seed as u64 + 7);
            c.validate().unwrap();
            let mut arena = Vec::new();
            let mut out = Vec::new();
            let inputs = vec![0u64; 16];
            c.eval_into(&inputs, 1, &mut arena, &mut out);
            assert_eq!(out.len(), 8);
        }
    }

    #[test]
    fn nand_only_circuits_use_nand() {
        let c = Circuit::random(16, 64, 8, GateSet::NandOnly, 5);
        assert!(c.gates.iter().all(|g| g.op == Op::Nand));
    }

    #[test]
    fn random_is_reproducible() {
        let a = Circuit::random(16, 64, 8, GateSet::XorAnd, 42);
        let b = Circuit::random(16, 64, 8, GateSet::XorAnd, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn prune_dead_preserves_behaviour() {
        // Gate 0 is dead: nothing reads it.
        let circuit = Circuit {
            n_inputs: 2,
            gates: vec![
                Gate {
                    op: Op::Xor,
                    a: 0,
                    b: 1,
                },
                Gate {
                    op: Op::Nand,
                    a: 0,
                    b: 1,
                },
            ],
            outputs: vec![5, 2],
        };
        let pruned = circuit.prune_dead();
        assert_eq!(pruned.gates.len(), 1);
        pruned.validate().unwrap();
        for bits in 0u64..4 {
            let inputs = [bits & 1, (bits >> 1) & 1];
            let mut arena = Vec::new();
            let mut a = Vec::new();
            circuit.eval_into(&inputs, 1, &mut arena, &mut a);
            let mut arena = Vec::new();
            let mut b = Vec::new();
            pruned.eval_into(&inputs, 1, &mut arena, &mut b);
            assert_eq!(a, b, "inputs {bits:02b}");
        }
    }

    #[test]
    fn prune_dead_is_idempotent() {
        let c = Circuit::random(32, 128, 16, GateSet::Universal, 11);
        let once = c.prune_dead();
        let twice = once.prune_dead();
        assert_eq!(once, twice);
        assert!(twice.gates.len() <= c.gates.len());
    }

    #[test]
    fn invalid_forward_reference_is_rejected() {
        let circuit = Circuit {
            n_inputs: 2,
            gates: vec![Gate {
                op: Op::Xor,
                a: 5,
                b: 0,
            }],
            outputs: vec![2],
        };
        assert!(circuit.validate().is_err());
    }

    /// Evaluate every gate-set generated by `circuit` on one input vector.
    fn run(c: &Circuit, input: u64) -> Vec<u8> {
        // One u64 plane per input wire; bit 0 carries the value.
        let planes: Vec<u64> = (0..c.n_inputs)
            .map(|i| u64::from((input >> i) & 1))
            .collect();
        let mut arena = Vec::new();
        let mut out = Vec::new();
        c.eval_into(&planes, 1, &mut arena, &mut out);
        out.iter().map(|w| (w & 1) as u8).collect()
    }

    /// Exhaustively compare two circuits on all 2^n inputs (n <= 12).
    fn assert_equivalent(a: &Circuit, b: &Circuit) {
        assert_eq!(a.n_inputs, b.n_inputs);
        assert_eq!(a.outputs.len(), b.outputs.len());
        for v in 0..(1u64 << a.n_inputs) {
            assert_eq!(run(a, v), run(b, v), "mismatch on input {v:#b}");
        }
    }

    #[test]
    fn simplify_is_exact_on_redundant_nand_circuit() {
        // 3 inputs (0..=2), const 0 = wire 3, const 1 = wire 4, gates from 5.
        // Deliberately full of redundancy: duplicate inverters, a gate fed by
        // const 0, a duplicated two-input gate, and a dead gate.
        let circuit = Circuit {
            n_inputs: 3,
            gates: vec![
                Gate {
                    op: Op::Nand,
                    a: 0,
                    b: 0,
                }, // 5: NOT x0
                Gate {
                    op: Op::Nand,
                    a: 0,
                    b: 0,
                }, // 6: duplicate of 5
                Gate {
                    op: Op::Nand,
                    a: 1,
                    b: 1,
                }, // 7: NOT x1
                Gate {
                    op: Op::Nand,
                    a: 5,
                    b: 7,
                }, // 8: NAND(NOT x0, NOT x1)
                Gate {
                    op: Op::Nand,
                    a: 8,
                    b: 8,
                }, // 9: NOT 8
                Gate {
                    op: Op::Nand,
                    a: 8,
                    b: 8,
                }, // 10: duplicate of 9
                Gate {
                    op: Op::Nand,
                    a: 3,
                    b: 2,
                }, // 11: NAND(0, x2) = 1 (dead)
                Gate {
                    op: Op::Nand,
                    a: 2,
                    b: 3,
                }, // 12: unused
                Gate {
                    op: Op::Nand,
                    a: 2,
                    b: 4,
                }, // 13: NOT x2 (const-1 input)
            ],
            outputs: vec![9, 13, 5],
        };
        let (simple, stats) = circuit.simplify();
        assert_equivalent(&circuit, &simple);
        assert!(stats.gates_after < stats.gates_before);
        // Duplicate inverters and duplicated nodes must have collapsed.
        assert_eq!(stats.deduped, 2, "stats: {stats:?}");
        assert_eq!(stats.folded_const, 2, "stats: {stats:?}");
        assert_eq!(stats.gates_after, 5, "stats: {stats:?}");
        simple.validate().unwrap();
    }

    #[test]
    fn simplify_handles_constant_folding() {
        // NAND(x, 0) = 1, NAND(1, 1) = 0.
        let circuit = Circuit {
            n_inputs: 2,
            gates: vec![
                Gate {
                    op: Op::Nand,
                    a: 0,
                    b: 2,
                }, // const 0 wire -> 1
                Gate {
                    op: Op::Nand,
                    a: 3,
                    b: 3,
                }, // NAND(1, 1) -> 0
            ],
            outputs: vec![4, 5],
        };
        let (simple, stats) = circuit.simplify();
        assert_equivalent(&circuit, &simple);
        assert_eq!(stats.folded_const, 2);
        assert_eq!(simple.outputs, vec![3, 2]);
        assert!(simple.gates.is_empty(), "gates: {:?}", simple.gates);
    }

    #[test]
    fn simplify_keeps_xor_only_circuits_affine() {
        let c = Circuit::random(8, 32, 4, GateSet::XorOnly, 5);
        let (simple, _) = c.simplify();
        assert_equivalent(&c, &simple);
        assert!(simple.is_xor_only(), "inversion leaked a non-XOR gate");
    }

    #[test]
    fn simplify_of_random_circuits_preserves_behaviour() {
        for (set, seed) in [
            (GateSet::Universal, 1u64),
            (GateSet::Full, 2),
            (GateSet::NandOnly, 3),
            (GateSet::XorAnd, 4),
            (GateSet::XorOnly, 5),
        ] {
            let c = Circuit::random(6, 40, 5, set, seed);
            let (simple, _) = c.simplify();
            assert_equivalent(&c, &simple);
            simple.validate().unwrap();
        }
    }

    #[test]
    fn simplify_is_idempotent() {
        let c = Circuit::random(16, 64, 8, GateSet::NandOnly, 9);
        let (once, _) = c.simplify();
        let (twice, stats) = once.simplify();
        assert_eq!(once, twice);
        assert_eq!(stats.deduped, 0);
    }

    #[test]
    fn aig_nodes_ignores_inverters() {
        let circuit = Circuit {
            n_inputs: 2,
            gates: vec![
                Gate {
                    op: Op::Nand,
                    a: 0,
                    b: 0,
                }, // inverter
                Gate {
                    op: Op::Nand,
                    a: 0,
                    b: 1,
                }, // real node
            ],
            outputs: vec![4, 5],
        };
        assert_eq!(circuit.aig_nodes(), 1);
    }
}
