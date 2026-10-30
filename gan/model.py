"""A binary GAN whose generator is a relaxed NAND circuit.

The generator is the same object the Rust trainer produces: a layered DAG of
two-input NAND gates, where each gate picks its two inputs from a candidate
set by an annealed softmax. Here the logits are ordinary PyTorch parameters, so
autograd does the work and a GPU is used if one is present.

At export time each softmax is replaced by its argmax, giving a literal circuit
(see export.py) that is bit-exact and needs no floating point at inference.
"""

import torch
import torch.nn.functional as F
from torch import nn


def build_wiring(sizes, k, skip_prob, seed):
    """Absolute candidate wire indices per layer, shape (n_out, k).

    Wire ids index into the concatenation of all earlier layers, so a gate can
    only read what has already been computed.
    """
    generator = torch.Generator().manual_seed(seed)
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + size)
    layers = []
    for layer in range(1, len(sizes)):
        n_out = sizes[layer]
        prev_start, prev_end = offsets[layer - 1], offsets[layer]
        idx = torch.randint(0, prev_end, (n_out, k), generator=generator)
        if layer > 1:
            use_prev = torch.rand(n_out, k, generator=generator) < skip_prob
            in_prev = torch.randint(prev_start, prev_end, (n_out, k), generator=generator)
            idx = torch.where(use_prev, in_prev, idx)
        layers.append(idx)
    return layers, offsets


class Generator(nn.Module):
    def __init__(self, sizes=(64, 128, 256, 256), k=16, skip_prob=0.3, seed=0):
        super().__init__()
        self.sizes = list(sizes)
        self.k = k
        wiring, offsets = build_wiring(self.sizes, k, skip_prob, seed)
        self.offsets = offsets
        self.layers = len(wiring)
        self.alpha = nn.ParameterList(
            [nn.Parameter(torch.randn(n, k) * 0.05) for n in self.sizes[1:]]
        )
        self.beta = nn.ParameterList(
            [nn.Parameter(torch.randn(n, k) * 0.05) for n in self.sizes[1:]]
        )
        for layer, idx in enumerate(wiring):
            self.register_buffer(f"wiring_{layer}", idx, persistent=True)

    def wires(self):
        return [getattr(self, f"wiring_{i}") for i in range(self.layers)]

    def forward(self, z, gain=32.0):
        acts = [z]
        for layer, (idx, alpha, beta) in enumerate(zip(self.wires(), self.alpha, self.beta)):
            flat = torch.cat(acts, dim=1)
            gathered = flat[:, idx.reshape(-1)].view(z.shape[0], idx.shape[0], idx.shape[1])
            pa = F.softmax(alpha * gain, dim=1).unsqueeze(0)
            pb = F.softmax(beta * gain, dim=1).unsqueeze(0)
            u = (gathered * pa).sum(-1)
            v = (gathered * pb).sum(-1)
            acts.append(1.0 - u * v)
        return acts[-1]


class Discriminator(nn.Module):
    def __init__(self, image_size=16, channels=64):
        super().__init__()
        c = channels
        self.net = nn.Sequential(
            nn.Conv2d(1, c, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(c, c * 2, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(c * 2, c * 4, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
        )
        side = image_size // 8
        self.head = nn.Linear(c * 4 * side * side, 1)

    def forward(self, x):
        return self.head(self.net(x))


def binarize(probs, temperature=1.0):
    """Bernoulli sample with a straight-through gradient."""
    hard = (torch.rand_like(probs) < probs).float()
    return hard + probs - probs.detach()


def sample_latent(batch, bits, device):
    return (torch.rand(batch, bits, device=device) < 0.5).float()
