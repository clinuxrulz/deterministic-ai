"""BinaryGAN with a NAND-circuit generator -- single-file Colab version (32x32).

Recipe notes (from the first runs, where d hit 0.000 and g rose to +4):
- gain anneals to 8, not 32: at 32 the candidate softmax saturates and the
  generator's gradients vanish.
- separate learning rates (D slower than G),
- instance noise on the discriminator's inputs so it cannot be perfect,
- EMA of the generator, used for sampling and export.

Layers are layered: each gate reads the immediately previous layer.
"""

import torch, torchvision, sys
print("python", sys.version.split()[0])
print("torch", torch.__version__, "torchvision", torchvision.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
import torch
from torchvision import transforms, datasets

SIDE = int(os.environ.get("SIDE", "32"))
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
        # Default to the immediately previous layer, so the exported circuit
        # really is as deep as the architecture. Allowing every earlier layer by
        # default lets the search short-circuit: in the first 32x32 export, 35%
        # of gates read the latent directly and the whole circuit collapsed to
        # depth 3.
        idx = torch.randint(prev_start, prev_end, (sizes[layer], k), generator=g)
        if layer > 1 and skip_prob > 0:
            use_far = torch.rand(sizes[layer], k, generator=g) < skip_prob
            far = torch.randint(0, prev_end, (sizes[layer], k), generator=g)
            idx = torch.where(use_far, far, idx)
        layers.append(idx)
    return layers


class Generator(nn.Module):
    def __init__(self, sizes=(64, 128, 256, 256), k=16, skip_prob=0.0, seed=0):
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

import os
import time

side = SIDE
latent_bits = int(os.environ.get("LATENT", "96" if SIDE == 32 else "64"))
hidden = [int(x) for x in os.environ.get("HIDDEN", "512,1024" if SIDE == 32 else "256,512").split(",")]
batch = int(os.environ.get("BATCH", "64"))
steps = int(os.environ.get("STEPS", "100000"))
gain_end = float(os.environ.get("GAIN_END", "8.0"))     # 8, not 32: see note
d_lr = float(os.environ.get("D_LR", "1e-4"))
g_lr = float(os.environ.get("G_LR", "3e-4"))
ema_decay = float(os.environ.get("EMA", "0.999"))
noise = float(os.environ.get("NOISE", "0.05"))          # instance noise on D inputs

sizes = [latent_bits, *hidden, side * side]
G = Generator(sizes, k=16, skip_prob=0.0, seed=0).to(device)
D = Discriminator(side).to(device)
print("generator gates:", sum(sizes[1:]), "| discriminator params:", sum(p.numel() for p in D.parameters()))
print(f"steps {steps}, gain_end {gain_end}, d_lr {d_lr}, g_lr {g_lr}, ema {ema_decay}, noise {noise}")

# Exponential moving average of the generator, used for sampling and export.
ema_alpha = [p.detach().clone() for p in G.alpha]
ema_beta = [p.detach().clone() for p in G.beta]

optG = torch.optim.Adam(G.parameters(), lr=g_lr, betas=(0.5, 0.999))
optD = torch.optim.Adam(D.parameters(), lr=d_lr, betas=(0.5, 0.999))
started = time.time()

def add_noise(x):
    if noise <= 0:
        return x
    flip = torch.rand_like(x) < noise / 2
    return torch.where(flip, 1 - x, x)

for step in range(1, steps + 1):
    gain = 1.0 * (gain_end / 1.0) ** min(1.0, step / (0.85 * steps))

    idx = torch.randint(0, real.shape[0], (batch,), device=device)
    assert real.shape[1:] == (1, side, side)
    fake = binarize(G(latent(batch, latent_bits, device), gain)).view(-1, 1, side, side).detach()
    d_loss = F.relu(1 - D(add_noise(real[idx]))).mean() + F.relu(1 + D(add_noise(fake))).mean()
    optD.zero_grad(set_to_none=True); d_loss.backward(); optD.step()

    fake = binarize(G(latent(batch, latent_bits, device), gain)).view(-1, 1, side, side)
    g_loss = -D(fake).mean()
    optG.zero_grad(set_to_none=True); g_loss.backward(); optG.step()

    with torch.no_grad():
        for target, param in zip(ema_alpha, G.alpha):
            target.mul_(ema_decay).add_(param.detach(), alpha=1 - ema_decay)
        for target, param in zip(ema_beta, G.beta):
            target.mul_(ema_decay).add_(param.detach(), alpha=1 - ema_decay)

    if step % 1000 == 0 or step == 1:
        print(f"step {step:6d} gain {gain:5.2f} d {d_loss.item():+.3f} g {g_loss.item():+.3f} "
              f"{time.time() - started:6.1f}s", flush=True)

import matplotlib.pyplot as plt

# Sample from the EMA weights. G is left holding them, so the export cell below
# snaps the EMA generator rather than the last noisy step.
with torch.no_grad():
    for p, e in zip(G.alpha, ema_alpha): p.copy_(e)
    for p, e in zip(G.beta, ema_beta): p.copy_(e)
    probs = G(latent(64, latent_bits, device), gain=gain_end)
    hard = (probs > 0.5).float().cpu().view(-1, side, side)

fig, axes = plt.subplots(4, 16, figsize=(16, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(hard[i], cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
plt.tight_layout(); plt.show()
print("mean probability:", round(probs.mean().item(), 3), "(real data density ~0.42)")

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
n_in = circuit["n_inputs"]; base = n_in + 2
depth = [0] * (base + len(circuit["gates"]))
for i, gate in enumerate(circuit["gates"]):
    depth[base + i] = max(depth[gate["a"]], depth[gate["b"]]) + 1
out_depth = max(depth[o] for o in circuit["outputs"])
direct = sum(1 for gate in circuit["gates"] if gate["a"] < n_in or gate["b"] < n_in)
print("circuit:", circuit["n_inputs"], "inputs,", len(circuit["gates"]), "gates,", len(circuit["outputs"]), "outputs")
print(f"exported depth {out_depth} of {G.layers} layers | {direct} gates ({100*direct/len(circuit['gates']):.1f}%) read the latent directly",
      "-- depth should equal layers; if it does not, the wiring is short-circuiting")

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
