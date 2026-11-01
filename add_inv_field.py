with open('crates/core/src/circuit.rs', 'r') as f:
    content = f.read()

old_gate = '''pub struct Gate {
    pub op: Op,
    pub a: u16,
    pub b: u16,
}'''

new_gate = '''pub struct Gate {
    pub op: Op,
    pub a: u16,
    pub b: u16,
    /// Per-gate inversion bit for Nandinv gates.
    /// When true with Nandinv op, outputs AND instead of NAND.
    #[serde(default = "default_false")]
    pub inv: bool,
}

fn default_false() -> bool {
    false
}'''

content = content.replace(old_gate, new_gate)
with open('crates/core/src/circuit.rs', 'w') as f:
    f.write(content)
print("Fixed Gate struct")
