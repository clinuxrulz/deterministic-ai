with open('crates/core/src/circuit.rs', 'r') as f:
    content = f.read()

old = "            Op::Not => Self::INV,\n        };"
new = "            Op::Not => Self::INV,\n            Op::Nandinv => 7,\n        };"
content = content.replace(old, new)
with open('crates/core/src/circuit.rs', 'w') as f:
    f.write(content)
print("Fixed Interner tag")
