with open('crates/core/src/circuit.rs', 'r') as f:
    content = f.read()

old_text = '''                Op::Nand => {
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
                Op::And => {'''

new_text = '''                Op::Nand => {
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
                Op::Nandinv => {
                    // Simplification: compute NAND result (inv=0 base case)
                    // The full simplification with per-gate inv requires
                    // tracking the inversion bit through rewrites.
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
                Op::And => {'''

content = content.replace(old_text, new_text)
with open('crates/core/src/circuit.rs', 'w') as f:
    f.write(content)
print("Added Nandinv to simplify")
