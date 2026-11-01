with open('crates/core/src/circuit.rs', 'r') as f:
    content = f.read()

# Fix remaining Gate constructions missing inv
content = content.replace('op: Op::Xor,\n                a: x,\n                b: self.n_in + 1,', 'op: Op::Xor,\n                a: x,\n                b: self.n_in + 1,\n                inv: false,')
content = content.replace('op: Op::Nand,\n                a: x,\n                b: x,', 'op: Op::Nand,\n                a: x,\n                b: x,\n                inv: false,')

with open('crates/core/src/circuit.rs', 'w') as f:
    f.write(content)
print("Fixed remaining Gate constructions")
