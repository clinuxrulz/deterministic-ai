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

    /// True if every gate is XOR (constants/affine offsets are still allowed).
    pub fn is_xor_only(&self) -> bool {
        self.gates.iter().all(|g| matches!(g.op, Op::Xor))
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
}
