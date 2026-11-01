# Fix apply method
with open('crates/core/src/circuit.rs', 'r') as f:
    content = f.read()

# The apply method needs to use `let base = match ...` pattern
old = '''    pub fn apply(self, a: u64, b: u64, inv: bool) -> u64 {
        match self {
            Op::Xor => a ^ b,
            Op::Xnor => !(a ^ b),
            Op::And => a & b,
            Op::Nand => !(a & b),
            Op::Or => a | b,
            Op::Nor => !(a | b),
            Op::Not => !a,
            Op::Nandinv => !(a & b),
        }
    }'''

new = '''    pub fn apply(self, a: u64, b: u64, inv: bool) -> u64 {
        let result = match self {
            Op::Xor => a ^ b,
            Op::Xnor => !(a ^ b),
            Op::And => a & b,
            Op::Nand => !(a & b),
            Op::Or => a | b,
            Op::Nor => !(a | b),
            Op::Not => !a,
            Op::Nandinv => !(a & b),
        };
        if inv && matches!(self, Op::Nandinv) {
            !(result) & 1
        } else {
            result
        }
    }'''

content = content.replace(old, new)
with open('crates/core/src/circuit.rs', 'w') as f:
    f.write(content)
print("Fixed apply")
