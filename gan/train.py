"""Train a 16x16 binary GAN with a NAND-circuit generator.

    python gan/train.py --data data/cats16.bin --steps 20000 --out runs/gan16

Runs on CUDA/MPS/CPU; the generator is small enough that CPU is viable at
16x16 but slow (see the README for measured expectations).
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from model import Discriminator, Generator, binarize, sample_latent  # noqa: E402
from png import write_grid  # noqa: E402
from read_xae import Dataset  # noqa: E402


def parse_hidden(text):
    return [int(part) for part in text.split(",") if part.strip()]


def load_images(path, split, size):
    dataset = Dataset(path)
    rows = dataset.rows(split)
    bits = torch.tensor([dataset.bits[i] for i in rows], dtype=torch.float32)
    return bits.view(-1, 1, size, size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/cats16.bin")
    parser.add_argument("--out", default="runs/gan16")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--latent", type=int, default=64)
    parser.add_argument("--hidden", default="128,256")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--skip-prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--d-steps", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--sample-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--gain-start", type=float, default=1.0)
    parser.add_argument("--gain-end", type=float, default=32.0)
    parser.add_argument("--anneal", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"device: {device}")

    real = load_images(args.data, "train", 16).to(device)
    side = int(real.shape[-1])
    pixels = side * side
    sizes = [args.latent, *parse_hidden(args.hidden), pixels]
    generator = Generator(sizes, k=args.k, skip_prob=args.skip_prob, seed=args.seed).to(device)
    discriminator = Discriminator(image_size=side).to(device)
    gates = sum(sizes[1:])
    print(f"generator: {sizes}, {gates} gates, latent {args.latent} bits")
    print(f"real images: {real.shape[0]}")

    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "log.csv"), "w")
    log.write("step,gain,d_loss,g_loss,seconds\n")

    started = time.time()
    for step in range(1, args.steps + 1):
        # Anneal the gate-selection softmax toward one-hot, as the Rust trainer does.
        span = max(1.0, args.anneal * args.steps)
        frac = min(1.0, step / span)
        gain = args.gain_start * (args.gain_end / args.gain_start) ** frac

        # --- discriminator -------------------------------------------------
        for _ in range(args.d_steps):
            idx = torch.randint(0, real.shape[0], (args.batch,), device=device)
            real_batch = real[idx]
            z = sample_latent(args.batch, args.latent, device)
            fake = binarize(generator(z, gain)).view(-1, 1, side, side).detach()
            d_real = discriminator(real_batch)
            d_fake = discriminator(fake)
            # Hinge loss: stable without gradient penalty.
            d_loss = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()

        # --- generator -----------------------------------------------------
        z = sample_latent(args.batch, args.latent, device)
        fake = binarize(generator(z, gain)).view(-1, 1, side, side)
        g_loss = -discriminator(fake).mean()
        opt_g.zero_grad(set_to_none=True)
        g_loss.backward()
        opt_g.step()

        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - started
            print(
                f"step {step:6d}  gain {gain:6.2f}  d {d_loss.item():+.3f}  g {g_loss.item():+.3f}  "
                f"{elapsed:7.1f}s  ({elapsed / step * 1000:.0f} ms/step)"
            )
            log.write(f"{step},{gain:.3f},{d_loss.item():.4f},{g_loss.item():.4f},{elapsed:.1f}\n")
            log.flush()

        if step % args.sample_every == 0:
            with torch.no_grad():
                z = sample_latent(64, args.latent, device)
                probs = generator(z, gain)
                bits = (probs > 0.5).float().cpu().numpy()
            write_grid(os.path.join(args.out, f"samples_{step:06d}.png"), bits, side)

        if step % args.save_every == 0 or step == args.steps:
            payload = {
                "sizes": sizes,
                "k": args.k,
                "skip_prob": args.skip_prob,
                "seed": args.seed,
                "latent": args.latent,
                "gates": gates,
                "alpha": [p.detach().cpu() for p in generator.alpha],
                "beta": [p.detach().cpu() for p in generator.beta],
                "wiring": [w.detach().cpu() for w in generator.wires()],
                "step": step,
            }
            torch.save(payload, os.path.join(args.out, "generator.pt"))
            with open(os.path.join(args.out, "config.json"), "w") as handle:
                json.dump(
                    {k: v for k, v in payload.items() if k not in ("alpha", "beta", "wiring")},
                    handle,
                    indent=2,
                )

    log.close()
    print(f"done in {time.time() - started:.0f}s; checkpoint in {args.out}")


if __name__ == "__main__":
    main()
