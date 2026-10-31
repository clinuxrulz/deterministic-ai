"""Render samples from a generator circuit (stdlib only, runs on the phone).

    python gan/render_circuit.py circuit.json out.png [samples] [seed] [latent-bits]

Feeds random latent vectors through the literal NAND circuit -- not the soft
model -- and writes a PNG grid plus density and diversity statistics. Useful
for checking what an exported generator actually produces after the snap.
"""

import json, random, sys, zlib, struct

path = sys.argv[1]
out = sys.argv[2]
samples = int(sys.argv[3]) if len(sys.argv) > 3 else 32
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
rng = random.Random(seed)

circ = json.load(open(path))
n_in, gates, outs = circ["n_inputs"], circ["gates"], circ["outputs"]
base = n_in + 2
side = int(len(outs) ** 0.5)

def run(bits):
    w = [0] * (base + len(gates))
    for i in range(n_in):
        w[i] = bits[i]
    w[n_in] = 0
    w[n_in + 1] = 1
    for i, g in enumerate(gates):
        w[base + i] = 0 if (w[g["a"]] and w[g["b"]]) else 1
    return [w[o] for o in outs]

imgs = []
for _ in range(samples):
    z = [1 if rng.random() < 0.5 else 0 for _ in range(n_in)]
    imgs.append(run(z))

dens = sum(sum(img) for img in imgs) / (samples * len(outs))
# mean pairwise Hamming distance
import itertools
tot = n = 0
for a, b in itertools.combinations(imgs, 2):
    tot += sum(x != y for x, y in zip(a, b)); n += 1
print(f"{samples} samples | output density {dens:.3f} | mean pairwise Hamming {tot/max(1,n)/len(outs):.3f}")

cols, scale, gap = 8, 6, 4
rows = (samples + cols - 1) // cols
tile = side * scale
W = cols * tile + (cols + 1) * gap
H = rows * tile + (rows + 1) * gap
canvas = [[22, 24, 29] * W for _ in range(H)]
for k, img in enumerate(imgs):
    x0 = gap + (k % cols) * (tile + gap)
    y0 = gap + (k // cols) * (tile + gap)
    for y in range(side):
        for x in range(side):
            v = 235 if img[y * side + x] else 25
            for dy in range(scale):
                row = canvas[y0 + y * scale + dy]
                for dx in range(scale):
                    p = (x0 + x * scale + dx) * 3
                    row[p:p+3] = [v, v, v]

def chunk(tag, data):
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

raw = b"".join(b"\x00" + bytes(r) for r in canvas)
with open(out, "wb") as f:
    f.write(b"\x89PNG\r\n\x1a\n")
    f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)))
    f.write(chunk(b"IDAT", zlib.compress(raw, 9)))
    f.write(chunk(b"IEND", b""))
print("wrote", out)
