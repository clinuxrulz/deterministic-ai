# AGENTS.md

Handoff notes for a new agent/machine. This file is the current source of truth;
the README is written for users and carries some older material.

## What this project is

A Boolean-circuit autoencoder for CIFAR-10 cats. The encoder maps a binarized
image to a short binary **key**; the decoder rebuilds the image from the key
alone. Inference is a wired circuit of literal 2-input logic gates — no
floating-point weights at inference.

```
image bits ──▶ encoder (gates) ──▶ K-bit key ──▶ decoder (gates) ──▶ reconstruction
```

## Current best model

**NAND-only 32×32 grayscale.** Every gate is NAND, so the circuit is fully
described by its wiring.

| Field | Value |
|---|---|
| Gates | 2,688 trained → 2,165 reachable → **1,547 after exact simplification** |
| AIG nodes | 986 (inversions counted as free — see below) |
| Gate type | `Nand` only (`gate_set = "nand_only"`) |
| Input | 32×32 = 1,024 bits (threshold binarized) |
| Key | 128 bits |
| Compression | 8× |
| Test BER | **0.1727** (val 0.1710) |
| Artifact | `web/public/models/cats32-nand.json` (committed) |

It is the default entry in `web/public/models/index.json` and is verified in a
headless browser (128 key bits, 8 samples, no JS errors).

The demo also ships **`cats16-rgb2`** (2-bit colour, 64 colours, 192-bit key,
2,345 gates), verified the same way: 192 key bits, `abs MAE 23.61%`, zero
console errors. Loading it requires the rebuilt wasm bundle — an older one
cannot parse `ColorMode::Rgb2`.

**Best configuration differs from the shipped artifact.** With the Gödel
relaxation (`relaxation = "min"`, now the default in `configs/cats32_nand.toml`)
the same trainer reaches test BER **0.1641 ± 0.0093** over seeds 0-9 versus
0.1730 ± 0.0060 for the product t-norm — paired −0.0089, 95% CI [−0.0159,
−0.0019]. It costs ~17% more gates (1,843 vs 1,572) because it keeps 77% of
gates as real two-input logic instead of 65%. The committed demo artifact is
still the product model; see `docs/experiment.md`, Checkpoint 6.

**Two gate counts, never mix them.** 1,547 is the number of literal NAND gates.
986 is the AND-node count in an AIG view where inverted wires are free; in CMOS
an inverter is *not* free, so 986 is a size metric for comparison with synthesis
literature, not a claim about hardware.

For comparison: the genetic algorithm reached test BER 0.2097 with 4,096
XOR+AND gates. Gradient-trained *gate types* (non-NAND) only reached 0.2523, so
restricting to NAND and learning only wiring is what produced the win.

### Reproduction and seed variance (2026-10-29)

Re-running the same config on an 8-core machine reproduces the published
number: test **0.1722**, val 0.1698, 2,170 active gates, 1m51s wall clock.

The seed also selects the train/val/test split, so seed variance is the error
bar for everything below. Over seeds 0-4 the default config gives test BER
**0.1730 ± 0.0066** (range 0.1657-0.1835; 95% CI of the mean ±0.0082), and a
repeat run of seed 1 reproduces it bit-for-bit, so the spread is genuine
split/init variance rather than run noise.

A single run is therefore only good to about ±0.013 (2 sd). Compare
configurations **paired on the same seed** — same seed means same split and same
init — and treat anything under ~0.01 as unresolved.

## Setup on a fresh machine

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
cd web && npm install && cd ..
```

Requires `rustc`/`cargo`, `node`/`npm`, `wasm-pack`, and (for headless checks)
`google-chrome`.

## Data and runs are NOT in git

`.gitignore` excludes `/data/` and `/runs/`. A fresh clone has **no dataset and
no trained run**. Only exported models survive in `web/public/models/`.

Regenerate the best model's dataset:

```bash
./scripts/fetch_cifar10.sh data          # or: cargo run -p xorae -- fetch-data --out data

cargo run -p xorae -- preprocess \
  --cifar-dir data/cifar-10-batches-bin \
  --out data/cats32_threshold.bin --size 32 --binarize threshold
```

Other datasets used in the experiments:

```bash
# 16x16 planar RGB, 8 colours (colour experiments)
cargo run -p xorae -- preprocess --cifar-dir data/cifar-10-batches-bin \
  --out data/cats16_rgb1.bin --size 16 --rgb1 --threshold 128

# 16x16 grayscale colour-palette experiment (negative result)
cargo run -p xorae -- preprocess --cifar-dir data/cifar-10-batches-bin \
  --out data/cats16_palette16.bin --size 16 --palette 16
```

## Train the best model

```bash
cargo run --release -p xorae -- train-nand \
  --config configs/cats32_nand.toml \
  --data data/cats32_threshold.bin --out runs/nand32 --epochs 200
```

Takes ~8 minutes on 4 cores, ~2 minutes on 8. Always use `--release`; debug is
far too slow.

`RAYON_NUM_THREADS` controls workers. The result is identical for any thread
count (verified 1 vs 4 workers); 4 is ~2× faster than 1.

`--seed` and `--epochs` override the config, and `--gain-end` overrides the
anneal endpoint, so experiments need no new TOML files:

```bash
cargo run --release -p xorae -- train-nand --config configs/cats32_nand.toml \
  --data data/cats32_threshold.bin --out runs/sweep_g8 --gain-end 8 --seed 1
```

The trainer snapshots the wire logits at the best validation BER and restores
them before polish/prune/export, so the exported model is the best-val epoch,
not the last one. The `log.csv` and stdout still show every epoch, and the
restore line names the epoch that won.

Then evaluate, render, and export:

```bash
cargo run --release -p xorae -- eval \
  --model runs/nand32/model.json --data data/cats32_threshold.bin --split test

cargo run --release -p xorae -- render \
  --model runs/nand32/model.json --data data/cats32_threshold.bin \
  --out runs/nand32/recon.png --count 8 --scale 6

cargo run --release -p xorae -- export \
  --model runs/nand32/model.json --out web/public/models/cats32-nand.json
```

## Commands

| Command | Purpose |
|---|---|
| `fetch-data` | Download/extract CIFAR-10 |
| `preprocess` | Build a cached dataset (`XAE4` format); `--binarize`, `--rgb1` (8 colours), `--rgb2` (64 colours), `--palette` |
| `train` | Genetic algorithm (legacy track) |
| `train-grad` | Gradient training of **gate types** on a fixed hierarchical wiring |
| `train-nand` | **Current best**: NAND-only wiring search |
| `eval` | BER / colour metrics on train/val/test |
| `render` | PNG of original / binarized / reconstruction |
| `baselines` | Zero, majority, top-K, greedy GF(2) |
| `simplify` | Exact post-pass on a model: constant folding, CSE, dead-gate removal (`--verify`, `--verify-random`) |
| `export` | Copy a model into `web/public/models/` |
| `export-samples` | 32×32 RGB previews for the demo gallery |

## Where the code is

| Path | Contents |
|---|---|
| `crates/core/src/circuit.rs` | `Circuit`, `Op` (Xor/Xnor/And/Nand/Or/Nor/Not), `GateSet`, bit-sliced evaluator, `prune_dead` |
| `crates/core/src/model.rs` | Portable model JSON, inference, `bits_to_rgb` |
| `crates/cli/src/nand.rs` | **NAND-only wiring search** (the current best trainer) |
| `crates/cli/src/grad.rs` | Gradient gate-type trainer, shape derivation, `derive_shapes_from` |
| `crates/cli/src/ga.rs` | Genetic algorithm, fitness, evaluation |
| `crates/cli/src/dataset.rs` | CIFAR loader, planar fix, `XAE4` format, palette/colour modes |
| `crates/wasm/src/lib.rs` | `wasm-bindgen` bridge |
| `web/src/App.tsx` | Solid 2.0 UI |

## How the NAND trainer works

- Fixed hierarchical wiring: `32×32 → 16×32 → 16×16 → 8×16` (128-bit key) and a
  mirrored decoder.
- The gate is relaxed as `1 - min(u, v)` (Gödel t-norm). Product (`1 - u*v`)
  and Łukasiewicz are available via `--relaxation`; product is measurably worse
  and Łukasiewicz fails outright.
- Each gate gets `n_local + n_global` **candidate wires** (default 9 local +
  3 long-range across layers).
- Two softmax distributions per gate (`alpha`, `beta`) select its two NAND
  inputs; the temperature anneals `gain_start → gain_end` so the selection
  becomes one-hot and export is literal `NAND(a, b)`.
- Activations stay continuous (hard binarisation during training does **not**
  work — see below); a soft `a(1-a)` penalty pulls them to 0/1.
- Batch is split across rayon workers with **global** normalisation, so the
  split is mathematically exact.
- After training: greedy wire-swap polish (optional, default 2 sweeps), then
  exact simplification (`Circuit::simplify`): constant folding, hash-consing
  CSE and dead-gate removal. Every rewrite is a Boolean identity, so BER is
  unchanged bit for bit — verified against the full dataset.
- The wire logits of the best-val epoch are snapshotted every `val_every`
  epochs and restored before polish/export. Selection is on val only, so test
  stays clean; at `gain_end = 32` this recovers ~0.002 val that the anneal
  otherwise overshoots.

## What has been ruled out (with evidence)

Do not spend time re-running these.

| Idea | Result |
|---|---|
| Strict XOR only | Linear over GF(2); test BER 0.4811 |
| Forcing ~90% XOR / 10% AND | Drives BER to chance (0.4986) |
| Learnable gate types (gradient) | 0.2523 — worse than wiring-only |
| 16-colour palette with the GA | Chance-level (6.3% index accuracy) |
| Hard binarisation during training | Collapses to the mean predictor (BER 0.50) |
| Sigmoid-temperature activation anneal | Compresses activations to ~0.5, signal dies |
| Greedy wire polish | Improves its scoring sample, **no** test gain; already a local optimum |
| Minimax | Not applicable — single-agent problem, no adversary. MCTS is the correct analogue |
| Non-halving shape reductions (e.g. 192→128) | Strided windows cover the input poorly; colour at 6× (0.2539) was worse than 8× (0.2192) |
| 512-bit keys | Theory says key size is not the limit (13 bits index 6,000 images) |
| More capacity (4,736-gate wide model) | Test BER 0.1913 / val 0.1879 vs 0.1722 / 0.1698 narrow, despite a *lower* train MSE (0.078 vs 0.095). Confirmed twice, at `gain_end` 32 and 12 (0.2133). Lower MSE with worse BER is objective mismatch, not overfitting — see the 100-image diagnostic |
| Softer anneal (`gain_end` 8/12/16) | No detectable effect. Seed 42 suggested 0.1679, but paired over seeds 0-4 the difference is +0.0063 ± 0.0113 (t(4)=1.24) and it doubles the spread |
| 300 epochs instead of 200 | 0.1674 vs 0.1679 on the same seed — far inside the ±0.013 single-run noise band. The default schedule has already converged by ~epoch 150 |
| Hard activations + straight-through (`--ste`) | 0.3128 / 0.3270 / 0.3198 over seeds 0-2 vs 0.1730 baseline. At `gain = 1` every gate thresholds to 1 and the network starts saturated |
| Hard wiring + straight-through (`--wire-ste`) | 0.2164 at seed 42 vs 0.1722. Trains the exported argmax circuit directly; plateaus by epoch ~75 |
| More data / augmentation as an accuracy fix | The model cannot fit 100 images (train BER 0.1741 after ~12,000 updates), so there is no overfitting gap to close |
| Łukasiewicz gate relaxation | 0.2117 at seed 42 vs 0.1722. Nilpotent: zero gradient whenever `u + v <= 1` |
| Product t-norm as the default | Gödel `min` is better: paired −0.0089 over 10 seeds, and makes `NAND(x,x)` an exact NOT instead of `1 - x²` |

## Open threads, in priority order

1. **Capacity per gate is the binding constraint.** Two independent
   measurements: the model cannot fit 100 images (train BER 0.1741 ≈ the 4,000
   image result), and train/val/test BER are all equal. Objective tweaks do not
   help (see the ruled-out table). The levers left are what a gate is spent on:
   free inversions (42% of gates are `Nand(x,x)`), spatial weight sharing, and
   distillation from a CNN teacher. A margin/hinge loss on bit decisions is the
   one cheap objective variant not yet tried.
2. **Colour, 2 bits/channel: shipped.** `--rgb2` gives 64 colours as six bit
   planes (1,536 bits, 192-bit key, 8×); over six paired seeds it cuts absolute
   reconstruction error by 33.7% for 1.91× the gates. It is in the demo as
   `cats16-rgb2`, and `eval` reports both `abs MAE vs original` and the
   palette-relative MAE.
3. **Gate removal for size.** Exact simplification (below) is done; the
   lossy frontier is not. Removing *active* gates with a tolerance gives a
   size-vs-BER curve, starting from 1,547 rather than 2,165. Nothing tried yet.
4. **Inversions are 42% of the budget.** In the shipped model 907 of 2,165
   reachable gates are `Nand(x, x)` (i.e. `NOT x`), and only 627 of those are
   distinct wires. A trainer that lets a gate consume an inverted wire for free
   (AIG-style) would spend its gate budget on real logic instead of negation —
   plausibly the largest remaining win, but it touches the model format, the
   evaluator and the exporter.
5. **Seed-averaged reporting for everything else.** The default config now has
   five seeds (0.1730 ± 0.0066) and runs are deterministic, so the harness is
   established. Re-measure the colour and wide configs the same way (5 seeds,
   paired) rather than as single runs — otherwise they repeat the `gain_end`
   mistake.

## Gotchas

- `data/`, `runs/`, `target/`, `web/dist/`, `web/src/wasm/pkg/` are gitignored.
- The CIFAR binary stores channels **planar** (1024 R, then 1024 G, then 1024 B),
  not interleaved. There is a regression test for this; earlier results were
  invalidated by the bug.
- Dataset cache format is `XAE4` (magic `XAE4`). `XAE2`/`XAE3` still load.
- `web/src/wasm/pkg` is generated: run `scripts/build-wasm.sh` or `npm run build`.
  It must be rebuilt after touching `ColorMode` (an old bundle cannot parse a
  model whose `color_mode` it does not know).
- **Colour metrics:** judge colour models by `eval`'s *abs MAE vs original*.
  Bit BER double-counts rgb2's extra planes, and MAE against the quantised
  target compares across incomparable palettes. Quantisation floors here are
  0.287 for 1 bit/channel and 0.083 for 2 bits/channel.
- **Gate counts:** report literal NAND gates and AIG nodes separately; see the
  note under "Current best model".
- `train-nand --train-limit N` trains on the first N samples (memorisation
  diagnostics). `--ste` and `--wire-ste` are the straight-through variants; both
  are documented negatives, kept as experiment switches.
- `--relaxation {product,min,lukasiewicz}` selects the gate t-norm. `min`
  (Gödel) is the current default and is exact on diagonal gates.
- Solid is pinned to `2.0.0-rc.19` and uses `onSettled` (not `onMount`).
- Headless Chrome needs `--virtual-time-budget=40000`; 20 s is too short for
  image decode.
- Run `cargo test --workspace` and `cargo clippy --workspace --all-targets`
  before committing; both are currently clean.

## Docs map

| File | Contents |
|---|---|
| `docs/experiment.md` | Full experiment log, results, checkpoint, open threads |
| `docs/color-plan.md` | Phased plan; colour results and negative findings |
| `docs/prior-art.md` | Literature survey (Boolean autoencoders, DLGN, evolvable hardware) |
| `docs/linearity.md` | Proof that XOR-only circuits are affine over GF(2) |
| `README.md` | Setup, commands, results tables, historical GA material |
