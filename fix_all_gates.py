with open('crates/core/src/circuit.rs', 'r') as f:
    lines = f.read()

# Fix all Gate constructions: add inv field
import re

# Pattern 1: gates.push(Gate { op, a, b }) -> add inv: false
lines = lines.replace('gates.push(Gate { op, a, b });', 'gates.push(Gate { op, a, b, inv: false });')

# Pattern 2: gates.push(Gate { op: gate.op, a, b }) -> add inv: gate.inv
lines = lines.replace('gates.push(Gate { op: gate.op, a, b });', 'gates.push(Gate { op: gate.op, a, b, inv: gate.inv });')

# Pattern 3: Gate { op, a: lo, b: hi } -> add inv
lines = lines.replace('Gate { op, a: lo, b: hi }', 'Gate { op, a: lo, b: hi, inv: false }')

# Pattern 4: interner.node needs to return wire with inv handled - check Interner
# The Interner uses Gate { ... } in node() method
# Let's fix the Interner.node method
old_interner = 'self.push((tag, lo, hi), Gate { op, a: lo, b: hi })'
new_interner = 'self.push((tag, lo, hi), Gate { op, a: lo, b: hi, inv: false })'
lines = lines.replace(old_interner, new_interner)

# Pattern 5: test code Gate constructions
# Find all "Gate {" in test sections and add inv
# Let's fix them individually by finding the exact patterns

# Pattern 6: gate.op === Nandinv in simplify - add the case
# Find the simplify function's match and add Nandinv
if 'Op::Nandinv => {' not in lines:
    # Add Nandinv to simplify - find the right spot (after Op::Nand)
    old_nand = '                Op::Nand => {'
    # Actually the simplify was broken before, let's just add a minimal case
    pass  # We'll handle this separately

with open('crates/core/src/circuit.rs', 'w') as f:
    f.write(lines)

print("Fixed Gate constructions")
