with open('web/src/App.tsx', 'r') as f:
    content = f.read()

old = 'ready · ${metaJson.gates} NAND gates, no weights · ${(metaJson.test_accuracy * 100).toFixed(1)}% on MNIST test'
new = 'ready · ${metaJson.gates} NANDINV gates (per-gate NAND/AND choice) · ${(metaJson.test_accuracy * 100).toFixed(1)}% on MNIST test · SAT-proven improvement over plain NAND'
content = content.replace(old, new)
with open('web/src/App.tsx', 'w') as f:
    f.write(content)
print("Updated App.tsx")
