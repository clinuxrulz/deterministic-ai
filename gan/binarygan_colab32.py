"""BinaryGAN with a NAND-circuit generator -- single-file Colab version.

    !python binarygan_colab32.py     # 32x32, 2,560 gates, no env vars needed

Only needs torch, torchvision and matplotlib (preinstalled on Colab).
Downloads CIFAR-10, trains, shows samples, snaps the wiring to a literal NAND
circuit and writes gan{side}_circuit.json.

The generator's gate count is sum(hidden) + side*side; the discriminator is
deliberately small (see its comment) so it cannot win immediately.
"""

import torch, torchvision, sys
print("python", sys.version.split()[0])
print("torch", torch.__version__, "torchvision", torchvision.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
import torch
from torchvision import transforms, datasets

SIDE = int(os.environ.get("SIDE", "32"))  # 32 by default in this variant
print("resolution", SIDE, "x", SIDE)

tf = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((SIDE, SIDE), interpolation=transforms.InterpolationMode.BOX),
    transforms.ToTensor(),
])
train = datasets.CIFAR10(root="./cifar", train=True, download=True, transform=tf)
test = datasets.CIFAR10(root="./cifar", train=False, download=True, transform=tf)

def cats(dataset):
    images = torch.stack([img for img, label in dataset if label == 3])  # class 3 = cat
    # Keep the channel dimension: the discriminator is a Conv2d and
    # needs (N, 1, H, W). Dropping it makes torch treat the batch as
    # channels and raise 'expected input[1, 64, 16, 16] to have 1 channels'.
    return (images >= 128 / 255).float()

real = cats(train).to(device)
real_test = cats(test).to(device)
print("train", tuple(real.shape), "test", tuple(real_test.shape), "density", round(real.mean().item(), 3))

import torch
import torch.nn.functional as F
from torch import nn


def build_wiring(sizes, k, skip_prob, seed):
    g = torch.Generator().manual_seed(seed)
    offsets = [0]
    for s in sizes:
        offsets.append(offsets[-1] + s)
    layers = []
    for layer in range(1, len(sizes)):
        prev_start, prev_end = offsets[layer - 1], offsets[layer]
        idx = torch.randint(0, prev_end, (sizes[layer], k), generator=g)
        if layer > 1:
            use_prev = torch.rand(sizes[layer], k, generator=g) < skip_prob
            in_prev = torch.randint(prev_start, prev_end, (sizes[layer], k), generator=g)
            idx = torch.where(use_prev, in_prev, idx)
        layers.append(idx)
    return layers


class Generator(nn.Module):
    def __init__(self, sizes=(64, 128, 256, 256), k=16, skip_prob=0.3, seed=0):
        super().__init__()
        self.sizes = list(sizes)
        self.k = k
        self.alpha = nn.ParameterList([nn.Parameter(torch.randn(n, k) * 0.05) for n in sizes[1:]])
        self.beta = nn.ParameterList([nn.Parameter(torch.randn(n, k) * 0.05) for n in sizes[1:]])
        for i, idx in enumerate(build_wiring(self.sizes, k, skip_prob, seed)):
            self.register_buffer(f"wiring_{i}", idx)
        self.layers = len(self.sizes) - 1

    def forward(self, z, gain=32.0):
        acts = [z]
        for i in range(self.layers):
            idx = getattr(self, f"wiring_{i}")
            flat = torch.cat(acts, dim=1)
            g = flat[:, idx.reshape(-1)].view(z.shape[0], idx.shape[0], idx.shape[1])
            pa = F.softmax(self.alpha[i] * gain, dim=1).unsqueeze(0)
            pb = F.softmax(self.beta[i] * gain, dim=1).unsqueeze(0)
            u = (g * pa).sum(-1)
            v = (g * pb).sum(-1)
            acts.append(1.0 - torch.minimum(u, v))   # Gödel: exact NAND on bits
        return acts[-1]


class Discriminator(nn.Module):
    # Deliberately small: the generator is only ~1k gates, so a 650k-parameter
    # discriminator separates real from fake instantly and G never learns.
    def __init__(self, side=16, channels=16):
        super().__init__()
        c = channels
        self.body = nn.Sequential(
            nn.Conv2d(1, c, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(c, c * 2, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Flatten(),
        )
        s = side // 4
        self.head = nn.Linear(c * 2 * s * s, 1)

    def forward(self, x):
        return self.head(self.body(x))


def binarize(probs):
    hard = (torch.rand_like(probs) < probs).float()
    return hard + probs - probs.detach()   # straight-through


def latent(batch, bits, device):
    return (torch.rand(batch, bits, device=device) < 0.5).float()

import time

side = SIDE
latent_bits = int(os.environ.get("LATENT", "96" if SIDE == 32 else "64"))
hidden = [int(x) for x in os.environ.get("HIDDEN", "512,1024" if SIDE == 32 else "256,512").split(",")]
batch, steps = int(os.environ.get("BATCH", "64")), int(os.environ.get("STEPS", "20000"))
sizes = [latent_bits, *hidden, side * side]
G = Generator(sizes, k=16, skip_prob=0.3, seed=0).to(device)
D = Discriminator(side).to(device)
print("generator gates:", sum(sizes[1:]))

optG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
started = time.time()

for step in range(1, steps + 1):
    gain = 1.0 * (32.0 / 1.0) ** min(1.0, step / (0.85 * steps))

    idx = torch.randint(0, real.shape[0], (batch,), device=device)
    assert real.shape[1:] == (1, side, side), f"real has shape {tuple(real.shape)}, expected (N, 1, {side}, {side})"
    fake = binarize(G(latent(batch, latent_bits, device), gain)).view(-1, 1, side, side).detach()
    d_loss = F.relu(1 - D(real[idx])).mean() + F.relu(1 + D(fake)).mean()
    optD.zero_grad(set_to_none=True); d_loss.backward(); optD.step()

    fake = binarize(G(latent(batch, latent_bits, device), gain)).view(-1, 1, side, side)
    g_loss = -D(fake).mean()
    optG.zero_grad(set_to_none=True); g_loss.backward(); optG.step()

    if step % 200 == 0 or step == 1:
        print(f"step {step:5d} gain {gain:5.2f} d {d_loss.item():+.3f} g {g_loss.item():+.3f} "
              f"{time.time() - started:6.1f}s")

import matplotlib.pyplot as plt

with torch.no_grad():
    probs = G(latent(64, latent_bits, device), gain=32.0)
    hard = (probs > 0.5).float().cpu().view(-1, side, side)
    p = probs.cpu().view(-1, side, side)

fig, axes = plt.subplots(4, 16, figsize=(16, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(hard[i], cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
plt.tight_layout(); plt.show()
print("mean probability:", round(p.mean().item(), 3), "(real data density ~0.45)")

import json

def snap(model):
    n_inputs = model.sizes[0]
    gate_offsets, total = [0], 0
    for s in model.sizes[1:]:
        total += s
        gate_offsets.append(total)
    wires = []
    for i in range(model.layers):
        idx = getattr(model, f"wiring_{i}").cpu()
        a = idx.gather(1, model.alpha[i].argmax(1, keepdim=True).cpu()).squeeze(1)
        b = idx.gather(1, model.beta[i].argmax(1, keepdim=True).cpu()).squeeze(1)
        wires.append((a, b))
    def to_wire(flat):
        if flat < model.sizes[0]:
            return int(flat)
        acc = model.sizes[0]
        for layer in range(1, len(model.sizes)):
            if acc <= flat < acc + model.sizes[layer]:
                return model.sizes[0] + 2 + gate_offsets[layer - 1] + int(flat - acc)
            acc += model.sizes[layer]
        raise ValueError(flat)
    gates, base = [], model.sizes[0] + 2
    offset = 0
    for a, b in wires:
        for i in range(len(a)):
            gates.append({"op": "nand", "a": to_wire(a[i]), "b": to_wire(b[i])})
        offset += len(a)
    return {"n_inputs": model.sizes[0], "gates": gates,
            "outputs": [base + gate_offsets[-2] + i for i in range(model.sizes[-1])]}

circuit = snap(G)
print("circuit:", circuit["n_inputs"], "inputs,", len(circuit["gates"]), "gates,", len(circuit["outputs"]), "outputs")

# Discretisation gap: soft generator vs snapped wiring, on the same latents.
z = latent(256, latent_bits, device)
with torch.no_grad():
    soft = (G(z, 32.0) > 0.5).float()
# Rebuild the hard wiring from the same alpha/beta for a like-for-like check.
hard_alpha = [torch.nn.functional.one_hot(p.argmax(1), p.shape[1]).float().to(device) for p in G.alpha]
hard_beta = [torch.nn.functional.one_hot(p.argmax(1), p.shape[1]).float().to(device) for p in G.beta]
acts = [z]
for i in range(G.layers):
    idx = getattr(G, f"wiring_{i}")
    flat = torch.cat(acts, dim=1)
    g = flat[:, idx.reshape(-1)].view(z.shape[0], idx.shape[0], idx.shape[1])
    u = (g * hard_alpha[i].unsqueeze(0)).sum(-1)
    v = (g * hard_beta[i].unsqueeze(0)).sum(-1)
    acts.append((1.0 - torch.minimum(u, v) > 0.5).float())
hard = acts[-1]
print("soft/hard agreement:", round((soft == hard).float().mean().item(), 4))
print("hard sample density:", round(hard.mean().item(), 3))

name = f"gan{side}_circuit.json"
with open(name, "w") as f:
    json.dump(circuit, f)
print(f"wrote {name} — validate locally with:")
print(f"  cargo run --release -p xorae -- circuit-info --circuit {name}")
