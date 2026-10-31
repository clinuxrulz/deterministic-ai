"""BinaryGAN (32x32) v2 -- spectral-norm critic, R1, density matching.

Fixes for the two failures seen so far:
- the critic hit d = 0.000 (perfect), starving the generator: spectral norm on
  every discriminator layer plus a lazy R1 gradient penalty;
- 42% of samples collapsed to near-black: penalise (mean(fake) - mean(real))^2;
- plus EMA of the generator and a 4-layer generator (2,816 gates, depth 4).

Run on Colab: Runtime -> T4 GPU -> Run all. ~20-30 minutes at the default
150,000 steps (the previous run measured 8.5 ms/step).
"""

import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms

print("python", sys.version.split()[0], "| torch", torch.__version__)
print(
    "cuda",
    torch.cuda.is_available(),
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIDE = int(os.environ.get("SIDE", "32"))
print("resolution", SIDE, "x", SIDE)

# --- data ------------------------------------------------------------------
tf = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((SIDE, SIDE), interpolation=transforms.InterpolationMode.BOX),
    transforms.ToTensor(),
])
train_set = datasets.CIFAR10(root="./cifar", train=True, download=True, transform=tf)


def cats(dataset):
    images = torch.stack([img for img, label in dataset if label == 3])  # class 3 = cat
    return (images >= 128 / 255).float()  # (N, 1, H, W) -- keep the channel dim


real = cats(train_set).to(device)
DATA_DENSITY = real.mean().item()
print("train", tuple(real.shape), "| data density", round(DATA_DENSITY, 3))


# --- model -----------------------------------------------------------------
def build_wiring(sizes, k, seed):
    """Candidates come from the immediately previous layer only, so the
    exported circuit is as deep as the architecture. An earlier version allowed
    any earlier layer and collapsed to depth 3 with 35% of gates reading the
    latent directly."""
    g = torch.Generator().manual_seed(seed)
    offsets = [0]
    for s in sizes:
        offsets.append(offsets[-1] + s)
    layers = []
    for layer in range(1, len(sizes)):
        idx = torch.randint(offsets[layer - 1], offsets[layer], (sizes[layer], k), generator=g)
        layers.append(idx)
    return layers


class Generator(nn.Module):
    def __init__(self, sizes=(96, 256, 512, 512, 1024), k=16, seed=0):
        super().__init__()
        self.sizes = list(sizes)
        self.k = k
        self.alpha = nn.ParameterList(
            [nn.Parameter(torch.randn(n, k) * 0.05) for n in sizes[1:]]
        )
        self.beta = nn.ParameterList(
            [nn.Parameter(torch.randn(n, k) * 0.05) for n in sizes[1:]]
        )
        for i, idx in enumerate(build_wiring(self.sizes, k, seed)):
            self.register_buffer(f"wiring_{i}", idx)
        self.layers = len(self.sizes) - 1

    def forward(self, z, gain=8.0):
        acts = [z]
        for i in range(self.layers):
            idx = getattr(self, f"wiring_{i}")
            flat = torch.cat(acts, dim=1)
            gathered = flat[:, idx.reshape(-1)].view(z.shape[0], idx.shape[0], idx.shape[1])
            pa = F.softmax(self.alpha[i] * gain, dim=1).unsqueeze(0)
            pb = F.softmax(self.beta[i] * gain, dim=1).unsqueeze(0)
            u = (gathered * pa).sum(-1)
            v = (gathered * pb).sum(-1)
            acts.append(1.0 - torch.minimum(u, v))  # Godel relaxation: exact NAND on bits
        return acts[-1]


class Discriminator(nn.Module):
    """Spectral-normalised, so its Lipschitz constant is bounded and it cannot
    become a perfect zero-loss wall that starves the generator."""

    def __init__(self, side=32, channels=16):
        super().__init__()
        sn = torch.nn.utils.parametrizations.spectral_norm
        c = channels
        self.body = nn.Sequential(
            sn(nn.Conv2d(1, c, 4, 2, 1)), nn.LeakyReLU(0.2),
            sn(nn.Conv2d(c, c * 2, 4, 2, 1)), nn.LeakyReLU(0.2),
            nn.Flatten(),
        )
        s = side // 4
        self.head = sn(nn.Linear(c * 2 * s * s, 1))

    def forward(self, x):
        return self.head(self.body(x))


def binarize(probs):
    hard = (torch.rand_like(probs) < probs).float()
    return hard + probs - probs.detach()  # straight-through


def latent(batch, bits, device):
    return (torch.rand(batch, bits, device=device) < 0.5).float()


# --- train -----------------------------------------------------------------
latent_bits = int(os.environ.get("LATENT", "96"))
hidden = [int(x) for x in os.environ.get("HIDDEN", "256,512,512").split(",")]
batch = int(os.environ.get("BATCH", "64"))
steps = int(os.environ.get("STEPS", "150000"))
gain_end = float(os.environ.get("GAIN_END", "8.0"))
d_lr = float(os.environ.get("D_LR", "1e-4"))
g_lr = float(os.environ.get("G_LR", "3e-4"))
density_weight = float(os.environ.get("DENSITY_WEIGHT", "5.0"))
r1_weight = float(os.environ.get("R1", "10.0"))
ema_decay = float(os.environ.get("EMA", "0.999"))
noise = float(os.environ.get("NOISE", "0.05"))

sizes = [latent_bits, *hidden, SIDE * SIDE]
G = Generator(sizes, k=16, seed=0).to(device)
D = Discriminator(SIDE).to(device)
print("generator gates:", sum(sizes[1:]), "| discriminator params:", sum(p.numel() for p in D.parameters()))
print(
    f"steps {steps} | gain_end {gain_end} | d_lr {d_lr} | g_lr {g_lr} | "
    f"density {density_weight} | r1 {r1_weight}"
)

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
    gain = gain_end ** min(1.0, step / (0.85 * steps))

    idx = torch.randint(0, real.shape[0], (batch,), device=device)
    assert real.shape[1:] == (1, SIDE, SIDE)
    real_batch = real[idx]
    fake = binarize(G(latent(batch, latent_bits, device), gain)).view(-1, 1, SIDE, SIDE).detach()

    d_loss = F.relu(1 - D(add_noise(real_batch))).mean() + F.relu(1 + D(add_noise(fake))).mean()

    # R1: penalise the critic's gradient on real data, lazily every 16 steps.
    r1 = torch.zeros((), device=device)
    if r1_weight > 0 and step % 16 == 0:
        probe = real_batch.detach().requires_grad_(True)
        grad = torch.autograd.grad(D(probe).sum(), probe, create_graph=True)[0]
        r1 = r1_weight * grad.flatten(1).pow(2).sum(1).mean()

    optD.zero_grad(set_to_none=True)
    (d_loss + r1).backward()
    optD.step()

    fake = binarize(G(latent(batch, latent_bits, device), gain)).view(-1, 1, SIDE, SIDE)
    # Density matching: stops the "emit mostly black" degenerate optimum.
    g_loss = -D(fake).mean() + density_weight * (fake.mean() - DATA_DENSITY) ** 2
    optG.zero_grad(set_to_none=True)
    g_loss.backward()
    optG.step()

    with torch.no_grad():
        for target, param in zip(ema_alpha, G.alpha):
            target.mul_(ema_decay).add_(param.detach(), alpha=1 - ema_decay)
        for target, param in zip(ema_beta, G.beta):
            target.mul_(ema_decay).add_(param.detach(), alpha=1 - ema_decay)

    if step % 2000 == 0 or step == 1:
        print(
            f"step {step:6d} gain {gain:4.1f} d {d_loss.item():+.3f} g {g_loss.item():+.3f} "
            f"density {fake.mean().item():.3f} (real {DATA_DENSITY:.3f}) "
            f"{time.time() - started:6.1f}s",
            flush=True,
        )


# --- sample (EMA weights, which is also what we export) --------------------
import matplotlib.pyplot as plt

with torch.no_grad():
    for p, e in zip(G.alpha, ema_alpha):
        p.copy_(e)
    for p, e in zip(G.beta, ema_beta):
        p.copy_(e)
    probs = G(latent(64, latent_bits, device), gain=gain_end)
    hard = (probs > 0.5).float().cpu().view(-1, SIDE, SIDE)

fig, axes = plt.subplots(4, 16, figsize=(16, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(hard[i], cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
plt.tight_layout()
plt.show()
print("mean probability", round(probs.mean().item(), 3), "| data density", round(DATA_DENSITY, 3))


# --- snap to a NAND circuit ------------------------------------------------
def snap(model):
    n_inputs = model.sizes[0]
    gate_offsets, total = [0], 0
    for s in model.sizes[1:]:
        total += s
        gate_offsets.append(total)

    def to_wire(flat):
        if flat < n_inputs:
            return int(flat)
        acc = n_inputs
        for layer in range(1, len(model.sizes)):
            if acc <= flat < acc + model.sizes[layer]:
                return n_inputs + 2 + gate_offsets[layer - 1] + int(flat - acc)
            acc += model.sizes[layer]
        raise ValueError(flat)

    gates = []
    for i in range(model.layers):
        idx = getattr(model, f"wiring_{i}").cpu()
        a = idx.gather(1, model.alpha[i].argmax(1, keepdim=True).cpu()).squeeze(1)
        b = idx.gather(1, model.beta[i].argmax(1, keepdim=True).cpu()).squeeze(1)
        for k in range(len(a)):
            gates.append({"op": "nand", "a": to_wire(a[k]), "b": to_wire(b[k])})
    return {
        "n_inputs": n_inputs,
        "gates": gates,
        "outputs": [n_inputs + 2 + gate_offsets[-2] + i for i in range(model.sizes[-1])],
    }


circuit = snap(G)
n_in, base = circuit["n_inputs"], circuit["n_inputs"] + 2
depth = [0] * (base + len(circuit["gates"]))
for i, gate in enumerate(circuit["gates"]):
    depth[base + i] = max(depth[gate["a"]], depth[gate["b"]]) + 1
out_depth = max(depth[o] for o in circuit["outputs"])
direct = sum(1 for gate in circuit["gates"] if gate["a"] < n_in or gate["b"] < n_in)
print("circuit:", n_in, "inputs,", len(circuit["gates"]), "gates,", len(circuit["outputs"]), "outputs")
print(f"exported depth {out_depth} of {G.layers} layers | {direct} gates read the latent directly")

name = f"gan{SIDE}_r1_circuit.json"
with open(name, "w") as f:
    json.dump(circuit, f)
print("wrote", name, "-- validate on the phone with:")
print(f"  cargo run --release -p xorae -- circuit-info --circuit {name}")
