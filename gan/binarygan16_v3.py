"""BinaryGAN (16x16) v3 -- the same recipe at a quarter of the pixels.

Why 16x16: the 32x32 generator already has more gates per pixel than the
decoder that works, so capacity was never the problem.

    model                         gates/pixel    result
    32x32 autoencoder decoder     996 / 1,024    BER 17%, real cats
    32x32 GAN generator         2,304 / 1,024    texture only
    16x16 GAN generator         2,304 /   256    this run

The project's own capacity wall points the same way: the 32x32 autoencoder
cannot fit even 100 images (train BER 0.1741), i.e. it is out of gates per
pixel, not out of data. At 16x16 the same 2,304 gates buy 9 gates/pixel, the
target is a 42%-ink blob drawn with a quarter of the pixels, and the latent
drops from 96 bits to 32 (8x compression, the rate the autoencoder has
actually been shown to support) so there are 2^32 outputs to police instead of
2^96. A 16x16 threshold target is not noise: the shipped 16x16 model scores BER
0.2285 with ones F1 0.712 and edge F1@1.5 0.835 against a 0.719 chance
baseline, using only 283 active gates.

Everything else is v3 held fixed -- Godel relaxation, depth 4, gain_end 8,
spectral-normed critic, lazy R1, EMA, InfoGAN Q, latent-usage penalty,
n_critic 2, 150k steps -- so the comparison against v2 is clean. The default
shape [32, 512, 768, 768, 256] is exactly 2,304 gates, the same as the shipped
32x32 circuit.

Run on Colab: Runtime -> T4 GPU -> Run all. About 30-45 minutes.
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
SIDE = int(os.environ.get("SIDE", "16"))
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
    return (images >= 128 / 255).float()  # (N, 1, H, W)


real = cats(train_set).to(device)
DATA_DENSITY = real.mean().item()
print("train", tuple(real.shape), "| data density", round(DATA_DENSITY, 3))


# --- model -----------------------------------------------------------------
def build_wiring(sizes, k, seed):
    """Candidates come from the immediately previous layer only, so the exported
    circuit is as deep as the architecture."""
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
    def __init__(self, sizes=(32, 512, 768, 768, 256), k=16, seed=0):
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
    """Spectral-normalised: bounded Lipschitz constant, so it cannot become a
    perfect zero-loss wall that starves the generator."""

    def __init__(self, side=16, channels=16):
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


class LatentHead(nn.Module):
    """InfoGAN Q: predict the latent from the generated image. If Q can read z
    out of G(z), the latent is genuinely in use."""

    def __init__(self, side=16, bits=32, channels=16):
        super().__init__()
        c = channels
        self.body = nn.Sequential(
            nn.Conv2d(1, c, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(c, c * 2, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Flatten(),
        )
        s = side // 4
        self.head = nn.Linear(c * 2 * s * s, bits)

    def forward(self, x):
        return self.head(self.body(x))


def binarize(probs):
    hard = (torch.rand_like(probs) < probs).float()
    return hard + probs - probs.detach()  # straight-through


def latent(batch, bits, device):
    return (torch.rand(batch, bits, device=device) < 0.5).float()


# --- train -----------------------------------------------------------------
latent_bits = int(os.environ.get("LATENT", "32"))
hidden = [int(x) for x in os.environ.get("HIDDEN", "512,768,768").split(",")]
batch = int(os.environ.get("BATCH", "64"))
steps = int(os.environ.get("STEPS", "150000"))
gain_end = float(os.environ.get("GAIN_END", "8.0"))
d_lr = float(os.environ.get("D_LR", "1e-4"))
g_lr = float(os.environ.get("G_LR", "3e-4"))
q_lr = float(os.environ.get("Q_LR", "2e-4"))
density_weight = float(os.environ.get("DENSITY_WEIGHT", "1.0"))
info_weight = float(os.environ.get("INFO", "2.0"))
usage_weight = float(os.environ.get("USAGE", "1.0"))
r1_weight = float(os.environ.get("R1", "10.0"))
ema_decay = float(os.environ.get("EMA", "0.999"))
noise = float(os.environ.get("NOISE", "0.05"))
n_critic = int(os.environ.get("N_CRITIC", "2"))

sizes = [latent_bits, *hidden, SIDE * SIDE]
G = Generator(sizes, k=16, seed=0).to(device)
D = Discriminator(SIDE).to(device)
Q = LatentHead(SIDE, latent_bits).to(device)
print("generator gates:", sum(sizes[1:]), f"| {sum(sizes[1:]) / (SIDE * SIDE):.1f} gates per pixel")
print("discriminator params:", sum(p.numel() for p in D.parameters()),
      "| latent head params:", sum(p.numel() for p in Q.parameters()))
print(
    f"steps {steps} | n_critic {n_critic} | gain_end {gain_end} | d_lr {d_lr} "
    f"| g_lr {g_lr} | density {density_weight} | info {info_weight} "
    f"| usage {usage_weight} | r1 {r1_weight}"
)

stage1 = G.wiring_0


def latent_usage(gain):
    """Expected number of stage-1 selections per latent bit, and how many bits
    that corresponds to (entropy -> effective count). All bits used is the
    target; the 32x32 v2 run collapsed to about 10 of 96."""
    pa = F.softmax(G.alpha[0] * gain, dim=1)
    pb = F.softmax(G.beta[0] * gain, dim=1)
    usage = torch.zeros(latent_bits, device=device)
    usage.scatter_add_(0, stage1.reshape(-1), (pa + pb).reshape(-1))
    p = usage / usage.sum()
    entropy = -(p * (p + 1e-12).log()).sum()
    return p, entropy.exp().item()


def spread_penalty(gain):
    """0 when the selection is uniform over the latent, ~n-1 when it sits on one
    bit: scaling by n turns it into a variance."""
    p, _ = latent_usage(gain)
    return latent_bits * (p - 1.0 / latent_bits).pow(2).sum()

ema_alpha = [p.detach().clone() for p in G.alpha]
ema_beta = [p.detach().clone() for p in G.beta]
optG = torch.optim.Adam(G.parameters(), lr=g_lr, betas=(0.5, 0.999))
optD = torch.optim.Adam(D.parameters(), lr=d_lr, betas=(0.5, 0.999))
optQ = torch.optim.Adam(Q.parameters(), lr=q_lr, betas=(0.5, 0.999))
started = time.time()


def add_noise(x):
    if noise <= 0:
        return x
    flip = torch.rand_like(x) < noise / 2
    return torch.where(flip, 1 - x, x)


for step in range(1, steps + 1):
    gain = gain_end ** min(1.0, step / (0.85 * steps))

    d_value = 0.0
    for _ in range(n_critic):
        idx = torch.randint(0, real.shape[0], (batch,), device=device)
        real_batch = real[idx]
        with torch.no_grad():
            fake = binarize(G(latent(batch, latent_bits, device), gain))
            fake = fake.view(-1, 1, SIDE, SIDE)
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
        d_value = d_loss.item()

    z = latent(batch, latent_bits, device)
    fake = binarize(G(z, gain)).view(-1, 1, SIDE, SIDE)

    q_loss = F.binary_cross_entropy_with_logits(Q(fake.detach()), z)
    optQ.zero_grad(set_to_none=True)
    q_loss.backward()
    optQ.step()

    # Density matching stops the "emit mostly black" optimum; the info term
    # stops the "emit one template" optimum.
    g_loss = (
        -D(fake).mean()
        + density_weight * (fake.mean() - DATA_DENSITY) ** 2
        + info_weight * F.binary_cross_entropy_with_logits(Q(fake), z)
        + usage_weight * spread_penalty(gain)
    )
    optG.zero_grad(set_to_none=True)
    g_loss.backward()
    optG.step()

    with torch.no_grad():
        for target, param in zip(ema_alpha, G.alpha):
            target.mul_(ema_decay).add_(param.detach(), alpha=1 - ema_decay)
        for target, param in zip(ema_beta, G.beta):
            target.mul_(ema_decay).add_(param.detach(), alpha=1 - ema_decay)

    if step % 2000 == 0 or step == 1:
        _, used = latent_usage(gain)
        print(
            f"step {step:6d} gain {gain:4.1f} d {d_value:+.3f} g {g_loss.item():+.3f} "
            f"q {q_loss.item():.3f} (0.693 = no latent info) "
            f"latent {used:4.1f}/{latent_bits} bits "
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

fig, axes = plt.subplots(8, 8, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(hard[i], cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
plt.tight_layout()
plt.show()
print("mean probability", round(probs.mean().item(), 3), "| data density", round(DATA_DENSITY, 3))

# Real on top, generated underneath: the human check.
fig, axes = plt.subplots(2, 8, figsize=(8, 2))
pick = torch.randint(0, real.shape[0], (8,), device=device)
for i in range(8):
    axes[0, i].imshow(real[pick[i], 0].cpu(), cmap="gray", vmin=0, vmax=1)
    axes[1, i].imshow(hard[i], cmap="gray", vmin=0, vmax=1)
    axes[0, i].axis("off")
    axes[1, i].axis("off")
plt.tight_layout()
plt.show()


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


# --- stats battery, identical to the v2 analysis ---------------------------
import random

rng = random.Random(0)
wires = base + len(circuit["gates"])
outs = circuit["outputs"]


def run(z):
    w = [0] * wires
    for i in range(n_in):
        w[i] = z[i]
    w[n_in], w[n_in + 1] = 0, 1
    for i, gate in enumerate(circuit["gates"]):
        w[base + i] = 0 if (w[gate["a"]] and w[gate["b"]]) else 1
    return [w[o] for o in outs]


def corr(a, b):
    a = a.flatten() - a.mean()
    b = b.flatten() - b.mean()
    return ((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-8)).item()


def texture(x):
    """Mean lag-1 spatial correlation, horizontal and vertical."""
    h = corr(x[:, :, :, :-1], x[:, :, :, 1:])
    v = corr(x[:, :, :-1, :], x[:, :, 1:, :])
    return 0.5 * (h + v)


def cone(circuit):
    """Gates that can reach an output."""
    seen = {o for o in circuit["outputs"] if o >= base}
    stack = list(seen)
    while stack:
        w = stack.pop()
        gate = circuit["gates"][w - base]
        for src in (gate["a"], gate["b"]):
            if src >= base and src not in seen:
                seen.add(src)
                stack.append(src)
    return len(seen)


with torch.no_grad():
    fake_soft = G(latent(256, latent_bits, device), gain=gain_end).view(-1, 1, SIDE, SIDE)
    fake_hard = (fake_soft > 0.5).float()
    real_batch = real[torch.randint(0, real.shape[0], (256,), device=device)]
    saturated = ((fake_soft < 0.05) | (fake_soft > 0.95)).float().mean().item()

print()
print(f"{'':22s} {'real':>8s} {'generated':>10s}")
print(f"{'density':22s} {DATA_DENSITY:8.3f} {fake_hard.mean().item():10.3f}")
print(f"{'lag-1 texture':22s} {texture(real_batch):8.3f} {texture(fake_hard):10.3f}")
print(f"{'mean-image corr':22s} {'':8s} {corr(real_batch.mean(0), fake_hard.mean(0)):10.3f}")
print(f"{'saturated (<.05/>.95)':22s} {'':8s} {saturated:10.3f}")
print(f"output cone: {cone(circuit)} of {len(circuit['gates'])} gates")

z0 = [rng.randint(0, 1) for _ in range(n_in)]
reference = run(z0)
deltas = []
for bit in range(n_in):
    probe = list(z0)
    probe[bit] ^= 1
    image = run(probe)
    deltas.append(sum(1 for a, b in zip(reference, image) if a != b))
deltas.sort()
dead = sum(1 for d in deltas if d == 0)
print(f"flip test: median {deltas[len(deltas) // 2]} of {len(outs)} outputs changed per latent bit, "
      f"max {deltas[-1]}, {dead}/{n_in} dead")

name = f"gan{SIDE}_v3_circuit.json"
with open(name, "w") as f:
    json.dump(circuit, f)
print("wrote", name, "-- validate on the phone with:")
print(f"  cargo run --release -p xorae -- circuit-info --circuit {name}")
