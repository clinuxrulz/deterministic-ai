# BinaryGAN with a NAND-circuit generator (16×16)

A 16×16 binary GAN whose generator is the same object the Rust trainer
produces: a layered DAG of two-input NAND gates where each gate picks its two
inputs from a candidate set with an annealed softmax. PyTorch supplies autograd
and the GPU; at the end the wiring is argmaxed into a literal circuit that runs
with no floating point at inference.

The discriminator is an ordinary small CNN. That is deliberate — the point of
this directory is to find out whether a *gate* generator can produce
recognisable images, without also having to hand-write a CNN in Rust.

## Why not on the phone

PyTorch has no Termux/Android aarch64 wheel, and this repo has no GPU training
path yet (see the GPU section in `AGENTS.md`: the phone's Vulkan GPU is
reachable but only ~2× the CPU for this access pattern). So training runs on a
desktop; everything else — dataset parsing and the exported gate format — is
tested here.

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

## Things that are easy to get wrong

- The gain must anneal (1 → 32). Exporting argmax of a soft, near-uniform
  softmax gives an arbitrary circuit — the anneal is what makes the trained
  wiring the exported wiring.
- Feed `D` *binary* samples (straight-through), not the generator's
  probabilities, or the discriminator learns to reject the hard circuit.
- Candidate wires must only point at earlier layers; `export.py` raises if a
  gate would read ahead of itself.
