# BinaryGAN with a NAND-circuit generator (16×16)

A 16×16 binary GAN whose generator is the same object the Rust trainer
produces: a layered DAG of two-input NAND gates where each gate picks its two
inputs from a candidate set with an annealed softmax. PyTorch supplies autograd
and the GPU; at the end the wiring is argmaxed into a literal circuit that runs
with no floating point at inference.

The discriminator is an ordinary small CNN. That is deliberate — the point of
this directory is to find out whether a *gate* generator can produce
recognisable images, without also having to hand-write a CNN in Rust.

### Resolution and size knobs

The single-file script takes environment variables, so 32×32 needs no edits:

```bash
!SIDE=32 HIDDEN=512,1024 LATENT=96 python binarygan_colab.py
```

Generator gates are `sum(hidden) + side²` (16×16: 1,024; 32×32: 2,560+).

Beware the resolution trap: the old note here said 16×16 was "near the
substrate's limit" because the supervised Rust autoencoders only reached BER
0.39 there. That number was measured against the **dithered** 16×16 cache,
whose target is a noise pattern (all-zero already scores 0.457). Against a
threshold cache the same 16×16 model scores **BER 0.2285**, ones F1 0.712 and
edge F1@1.5 0.835 vs a 0.719 chance baseline with 283 active gates. Dither is
an impossible target at this size; the resolution is not the problem.

`gan/render_circuit.py` renders samples from an exported circuit on the phone
(stdlib only): `python gan/render_circuit.py circuit.json out.png 32`.

## v3 at 16×16 (the run that tests the rate argument)

`binarygan16_v3.py` / `binarygan16_v3.ipynb` are `binarygan32_v3.py` with the
resolution and latent changed, everything else held fixed:

```bash
SIDE=16 LATENT=32 HIDDEN=512,768,768 STEPS=150000   # 2,304 gates, 9 per pixel
```

The notebook wants `cats16_bits.bin` (188 KB) uploaded beside it — 6,000
pre-binarized cats in the `XAB1` format, i.e. the same preprocessing the Rust
trainer uses. Without it, torchvision downloads CIFAR-10 (~170 MB), which is
the slow part when working from a phone:

```bash
python gan/export_bits.py data/cats16_threshold.bin data/cats16_bits.bin
```

Why: the 32×32 generator already has more gates per pixel (2.25) than the
decoder that works (0.97), so capacity was never the constraint — the latent
budget was. 96 bits for 4,000 training cats is 2⁹⁶ outputs of which v2 used
15. At 16×16 the same 2,304 gates buy 9 per pixel and 32 latent bits (8×
compression, the rate the autoencoder is known to support), so there are 2³²
outputs to police.

The notebook prints the same battery the v2 post-mortem used — density, lag-1
texture, mean-image correlation, saturated fraction, output cone and the
per-bit flip test — so the two runs compare directly. Watch `latent N/32`
(v2: 15/96), `q` (0.693 = the latent carries no information) and mean-image
correlation (v2: 0.254).

## Why not on the phone

PyTorch has no Termux/Android aarch64 wheel, and this repo has no GPU training
path yet (see the GPU section in `AGENTS.md`: the phone's Vulkan GPU is
reachable but only ~2× the CPU for this access pattern). So training runs on a
desktop; everything else — dataset parsing and the exported gate format — is
tested here.

## Quick test in Colab (no install)

`gan/binarygan_colab.ipynb` is self-contained: it pulls CIFAR-10 from
torchvision, defines the gate generator inline, trains for a few thousand
steps, shows samples, slaps the wiring, and writes `gan16_circuit.json` for
`xorae circuit-info`. Nothing to install — Colab already has torch, torchvision
and matplotlib.

Mobile caveat: the notebook will run in a phone browser, but a backgrounded tab
suspends and Colab disconnects when that happens. For a short run (2-4k steps,
10-20 min) keep the tab in the foreground with the screen on; for anything
longer, use a desktop.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch numpy
```

## Train

```bash
python gan/train.py --data data/cats16.bin --out runs/gan16 \
  --steps 20000 --batch 64 --latent 64 --hidden 128,256 --k 16
```

Useful flags: `--steps`, `--d-steps`, `--lr`, `--anneal`, `--gain-end`,
`--sample-every`, `--seed`. Samples land in `runs/gan16/samples_*.png` and the
checkpoint in `runs/gan16/generator.pt`.

Rough expectations for 16×16 (1,024 output gates? no: 256 output bits):

| hardware | ~step | 20k steps |
|---|---|---|
| desktop GPU | 20-60 ms | 7-20 min |
| desktop CPU | 200-500 ms | 1-3 h |

The generator is ~1.5k gates total (`latent 64 → 128 → 256 → 256`), which is
the range the Rust models already work in.

## Export to a gate circuit

```bash
python gan/export.py --checkpoint runs/gan16/generator.pt --out runs/gan16/circuit.json
cargo run --release -p xorae -- circuit-info --circuit runs/gan16/circuit.json
```

`circuit-info` validates the JSON (wire ranges, ordering), and reports inputs,
gates, outputs, depth, active gates and AIG nodes. The exported object is the
same schema as `web/public/digits/classifier.json`, so it can be run by the
wasm `run_circuit` bridge and rendered like any other circuit.

Without PyTorch you can still check the exporter:

```bash
python gan/export.py --self-test --out /tmp/random_circuit.json
cargo run --release -p xorae -- circuit-info --circuit /tmp/random_circuit.json
```

## Files

| file | purpose |
|---|---|
| `read_xae.py` | stdlib reader for the `XAE4`/`XAE5` dataset caches |
| `model.py` | relaxed NAND generator (autograd via gather + softmax), CNN discriminator, STE binarisation |
| `train.py` | hinge-loss GAN loop, gain anneal, sample grids, checkpoints |
| `png.py` | stdlib PNG writer for sample grids |
| `export.py` | checkpoint → literal circuit JSON (argmax wiring) |
| `binarygan16_v3.py` / `.ipynb` | the current run: 16×16, 32-bit latent, 2,304 gates, usage penalty + InfoGAN Q |
| `export_bits.py` | XAE cache → 188 KB `XAB1` bit pack, so Colab uploads bits instead of downloading CIFAR-10 |

## Things that are easy to get wrong

- The gain must anneal (1 → 32). Exporting argmax of a soft, near-uniform
  softmax gives an arbitrary circuit — the anneal is what makes the trained
  wiring the exported wiring.
- Feed `D` *binary* samples (straight-through), not the generator's
  probabilities, or the discriminator learns to reject the hard circuit.
- Candidate wires must only point at earlier layers; `export.py` raises if a
  gate would read ahead of itself.
