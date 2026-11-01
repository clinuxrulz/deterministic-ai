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

The demo has four tabs: **reconstruct** (the models above), **draw a digit**
(MNIST autoencoder + NAND classifier), **cat gan**, and **solver's wiring**.
The digit tab normalises
a canvas stroke the way MNIST preprocessing does (bbox -> 20x20 -> centre of
mass -> 28x28), encodes it, and runs the classifier circuit from
`web/public/digits/classifier.json` via `run_circuit`. `?tab=digits&selftest=1`
draws a built-in sample and classifies it, which is how it is verified headlessly.

The **cat gan** tab samples a 96-bit latent, runs the generator circuit from
`web/public/gan/circuit.json` (2,304 NAND gates, depth 4) through `run_circuit`
and draws 16 samples in a 4x4 grid, with a density and spread readout. It is a
placeholder: the circuit reproduces cat-image statistics (density, local
texture) but not anatomy, and only about 15 of its 96 latent bits change the
output — `effective_latent` in `web/public/gan/meta.json`, measured by flipping
each bit offline. The tab is titled "This cat is not real."; `?tab=gan&selftest=1`
renders a deterministic grid for headless checks.

The **solver's wiring** tab is the 8x8/3-bit witness from
`scripts/sat_ae.py` (Checkpoint 12), written by `--export web/public/sat`: eight
glyphs with the code the encoder gives each, all eight key codes decoded side by
side, and three clickable key bits. Clicking a glyph runs the encoder
(`sat/circuit.json`, 51 gates) and decodes the resulting key (112 gates), so the
tab exercises both halves through `run_circuit`. The point is the key space:
three codes were trained, a fourth is reached by the held-out glyphs, and the
other four are codes nothing asked for that still decode to definite images.
`?tab=sat&selftest=1` requires all three trained glyphs to round-trip with zero
wrong bits and prints `self-test · 3/3 ...` — that is the headless check.

Below that witness the tab shows a second one, `--gate nandinv` exported to
`web/public/sat/nandinv/`: the same 163-gate shape where each gate may also
invert its own output (NAND or AND). Six of the eight glyphs fit exactly there,
so it has eight glyph buttons and three key bits, and its self-test prints
`self-test · 6/6 glyphs round-trip exactly · 163 gates · keys ...`. It loads
through the same `run_circuit`; `and` is an existing `Op`, so no wasm rebuild is
needed for it. Both blocks must pass for `?tab=sat&selftest=1`.

Note for Solid 2.0: signal writes are scheduled, so reading a signal immediately
after setting it returns the stale value. Pass values explicitly instead (the
GAN tab's `generate(meta)` does this), or compute the display string from a
local. This cost an afternoon of debugging a canvas that stayed empty.

Two experiment entries sit next to the best model so the *targets* are
legible: `cats32-edge` (Sobel line drawings, 15% ink; the trainer cannot fit
them, see Checkpoint 10) and `cats32-pal4` (4-level grey, 2,048 bits,
unstructured reconstruction). Selecting the edge model makes the input panel a
clean line drawing, which is the point of shipping it: the reconstruction is
noise and the demo says so. `?model=<id>` deep-links any entry, which is how
the headless checks reach a model that is not first in the list. Models record
their `binarize` mode, so the demo preprocesses uploads the same way
(`xorae export --binarize edge` when the trainer wrote `threshold`).

The demo also ships **`cats16-rgb2`** (2-bit colour, 64 colours, 192-bit key,
2,345 gates), verified the same way: 192 key bits, `abs MAE 23.61%`, zero
console errors. Loading it requires the rebuilt wasm bundle — an older one
cannot parse `ColorMode::Rgb2`.

**Two relaxations, two different wins.** Product (`relaxation = "product"`,
the default) is better *per gate*; Gödel (`min`) is better at equal *training*
budget:

| | product | min (Gödel) |
|---|---|---|
| As trained (test BER) | 0.1730 ± 0.0060 | **0.1641 ± 0.0093** |
| Gates after simplification | 1,572 ± 52 | 1,843 ± 44 |
| Real two-input logic | 65.3% | **77.3%** |
| Pruned to 1,400 gates | **0.1811 ± 0.0057** | 0.2154 ± 0.0167 |
| Pruned to 1,300 gates | **0.1882 ± 0.0157** | 0.2401 ± 0.0151 |

Gödel's extra gates are load-bearing, so it degrades much faster when
compressed. Since this project optimises tiny circuits, product stays the
default; see `docs/experiment.md` Checkpoints 6 and 7.

**Two gate counts, never mix them.** 1,547 is the number of literal NAND gates.
986 is the AND-node count in an AIG view where inverted wires are free; in CMOS
an inverter is *not* free, so 986 is a size metric for comparison with synthesis
literature, not a claim about hardware.

The two numbers are also more related than that makes them sound: **986 is
exactly the count of gates that are not `Nand(x,x)`**, i.e. the same circuit with
its 561 inverters deleted, so the "AIG node count" is not an independent
estimate. In static-CMOS devices (NAND2 4T, inverter 2T, XOR2 12T) the shipped
model is **5,066 transistors**, 1,122 of them (22%) inverters; `cats16-rgb2` is
7,662 and `mnist32` 4,094. The 163-gate glyph witness is 492T, and the same
shape in XOR+bias gates would be 1,956T — 3x the silicon for half the function
class, which is the device-level reason not to ship XOR.

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

## Hosting

The demo is published to **https://clinuxrulz.github.io/deterministic-ai/** by
`.github/workflows/gh-pages.yml`: on every push to `main` it installs Rust with
the `wasm32-unknown-unknown` target, `wasm-pack` and Node, runs `npm run build`
in `web/` (which builds the wasm package first), and pushes `web/dist` to the
`gh-pages` branch with `peaceiris/actions-gh-pages`.

Two things make the sub-path work: Vite's `base` is read from `SITE_BASE`
(`/deterministic-ai/` in the workflow, `/` locally), and the app resolves every
asset through `import.meta.env.BASE_URL`.

One-time repository setting: **Settings → Pages → Source: Deploy from a branch
→ `gh-pages` / (root)**. The branch only exists after the first successful run.

`.github/workflows/ci.yml` runs `cargo fmt --check`, `cargo test --workspace`,
`cargo clippy --workspace --all-targets` and a web build on pushes and PRs.

## GPU (probe results, 2026-10-30)

`crates/gpu-probe` enumerates Vulkan devices and benchmarks the trainer's inner
loop (candidate gather + weighted NAND sum) against the CPU.

**On Termux the loader matters.** The default `libvulkan.so` resolves to Mesa's
software `llvmpipe` (CPU). The real GPU is only visible through the Android
system loader:

```bash
XORAE_VULKAN_LIB=/system/lib64/libvulkan.so cargo run --release -p xorae-gpu-probe
# device 0: Samsung Xclipse 530, INTEGRATED_GPU, api 1.3.279
XORAE_VULKAN_LIB=/system/lib64/libvulkan.so cargo run --release -p xorae-gpu-probe -- --bench
```

Benchmark (Exynos 1480, 8 cores), gather kernel at `gates × candidates × samples`:

| config | GPU | CPU (7 threads) | speedup |
|---|---|---|---|
| 4,096 × 16 × 256 | 9.8 ms | 14.7 ms | 1.5× |
| 4,096 × 16 × 1,024 | 24.3 ms | 35.8 ms | 1.5× |
| 16,384 × 16 × 512 | 38.2 ms | 80.9 ms | 2.1× |
| 16,384 × 24 × 1,024 | 101.1 ms | 222.7 ms | 2.2× |

Both sides are memory-bound (~16 GB/s of gather traffic, 8 vs 3.6 GFLOP/s), so
the mobile GPU is **not** worth porting the trainer for: 2× is not enough.
The same kernels on a desktop GPU should be a different story, because the
pattern is bandwidth-bound and desktop memory is 20-50× faster.

If a GPU trainer is built, raise arithmetic intensity first (tile each layer's
activations through shared/workgroup memory so a wire is loaded once for all
its consumers) rather than transliterating the CPU loop.

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
| `preprocess` | Build a cached dataset (`XAE4`); `--binarize {threshold,otsu,dither,edge}`, `--rgb1`, `--rgb2`, `--palette`, or `--mnist-dir` for raw IDX files |
| `train` | Genetic algorithm (legacy track) |
| `train-grad` | Gradient training of **gate types** on a fixed hierarchical wiring |
| `train-nand` | **Current best**: NAND-only wiring search |
| `eval` | BER / colour metrics on train/val/test |
| `render` | PNG of original / binarized / reconstruction |
| `baselines` | Zero, majority, top-K, greedy GF(2) |
| `simplify` | Exact post-pass on a model: constant folding, CSE, dead-gate removal (`--verify`, `--verify-random`) |
| `prune` | Lossy size pass: bypass gates while BER stays within `--tolerance`, down to `--target` gates |
| `prior` | Fit Bernoulli / autoregressive priors over the key space, sample, decode, report NLL and round-trip |
| `classify` | Softmax regression on keys or raw pixels; accuracy + confusion matrix |
| `classify-nand` | Train a NAND circuit classifier on keys or pixels (grouped vote outputs; `--features`, `--distill`) |
| `eval-circuit` | Score a saved classifier circuit on a dataset split (no retraining) |
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
| `scripts/sat_ae.py` | SAT wiring search: encoder, solver driver, `selftest`/`brute`/`encoder`/`decoder`/`functions`/`subsets` modes, `--gate {nand,nandinv,xor3,xor2,mix}`, `--glyphs`/`--slack`/`--export` |
| `scripts/circuit_cones.py` | exact dependency cones of every wire: what an output *can* depend on |
| `scripts/make_glyph_dataset.py` | writes the 8×8 glyphs as `XAE4` so `train-nand` can train on the same data |
| `web/src/App.tsx` | Solid 2.0 UI |

## How the NAND trainer works

- Fixed hierarchical wiring: `32×32 → 16×32 → 16×16 → 8×16` (128-bit key) and a
  mirrored decoder.
- The gate is relaxed as `1 - u*v` (product t-norm) by default, with `min`
  (Gödel) and `lukasiewicz` selectable via `--relaxation`. Gödel gives better
  BER at equal training budget but a worse size-accuracy trade; Łukasiewicz
  fails outright.
- Each gate gets `n_local + n_global + n_channel` **candidate wires** (default
  9 local + 3 long-range across layers + 0 cross-plane).
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

### What the wiring can and cannot express

Two consequences of the fixed wiring that bound what any config can learn
(asserted in `halving_layers_never_leave_their_column_or_plane`):

- **Dependency is capped at `2^d` input bits**, where `d` is the total number of
  gate levels from input to output (a gate's dependency set is the union of its
  two inputs'). The shipped 32×32 model has `d = 6`, so one output bit sees at
  most **64 of 1,024** input bits; 16×16 rgb8 (`d = 6`) sees 64 of 6,144; the
  4×4 probes (`d = 4`) see 16 of 384. More depth is the only way to widen it,
  and only same-resolution layers add depth without shrinking the shape.
- **For planar colour, halving layers never leave a column or a plane.** The
  shape starts tall (`channels·side` rows by `side` columns), so
  `derive_shapes_from` only ever halves the row axis and `window` spans `sy`
  rows by **one** column. Every local candidate then sits in the same column
  and the same colour plane: the encoder is a stack of independent per-plane
  column codecs. Only the random `n_global` picks and `n_channel` cross-plane
  candidates mix them, and `n_channel = 6` moved 4×4 by nothing. Same-resolution
  layers (3×3 window) are the only mechanism that reads across columns.
  Grayscale 32×32 is different: its shape is square, so the halving alternates
  rows and columns and layer 2 does read two columns.

Hidden widths may exceed the input: `derive_shapes_from` grows the longer axis
by an integer factor (an expansion layer), and `window` then upsamples with
overlapping 3×3 windows. `configs/cats4_rgb8_wide.toml` starts at 768 for a
384-bit input and is the best 4×4 colour run (0.2319 abs MAE) — still behind a
constant mean image, so the mechanism exists but the target does not follow.

## SAT wiring search

`scripts/sat_ae.py` asks the trainer's question exactly, with a solver instead
of gradients: fixed hierarchical shape, two binary-encoded candidate indices per
gate, the whole net unrolled once per training pattern, and optional soft
outputs (`--slack K` = at most K wrong bits *per pattern*, via selectors that
are true exactly when a bit is wrong, plus a sequential counter). Kissat 4.0.1
solves the 163-gate, 8-pattern 8×8 glyph instance (27k clauses) in 0.1 s, so it
is a usable oracle, not a research toy; CaDiCaL 3.0 agrees on every CNF.

Validation is built in and is not optional: `--mode selftest` generates a random
wiring, lets it define the targets, and requires the solver to find one (a
witness exists by construction) plus a slack case that corrupts a witness by
exactly the budget; `--mode brute` enumerates every wiring on a tiny shape and
compares SAT/UNSAT; `--mode encoder`/`decoder` attribute an UNSAT to one half of
the net; `--mode functions` counts how many functions of a 3-bit key the decoder
can compute at all. Four bugs were caught this way, three of them in the
soft-equality clause — one-directional, then inverted so `slack 4` meant "at
most 4 *correct* bits" — each of which still returned a confident "SAT". Do not
quote an UNSAT without selftest + brute agreement, and do not quote a slack
number without the runner's per-pattern bound assertion.

`scripts/make_glyph_dataset.py` writes the same glyphs as `XAE4` for
`train-nand` (`configs/glyphs8_nand.toml`, 4-bit key because the trainer cannot
derive a 3-wide bottleneck). On 3 glyphs SAT proves an exact fit and 12,000
epochs of the trainer reach BER 0.0365 (73 gates); on 8 glyphs the trainer
reaches 0.1426 against the SAT slack-8 bound of 0.1172 — close, as expected if
the deficit is the template. Note the trainer samples its candidate window, so a
SAT witness is not necessarily reachable by it.

`--export web/public/sat` writes a witness as two plain `Circuit` JSON objects
plus `meta.json`, and the exporter re-simulates them on every glyph before
saving, so a broken export fails loudly instead of shipping. That is what the
demo's *solver's-wiring* tab runs; regenerate it with
`python3 scripts/sat_ae.py --patterns 3 --export web/public/sat`.

`scripts/circuit_cones.py` computes what each output of an exported circuit can
possibly depend on (union the two inputs' dependency sets at every gate). The
answer for the shipped 32×32 model is hostile: every decoder output sees at most
7 of the 128 key bits (median 2, 169 outputs see exactly one), and the encoder
reads only **362 of 1,024** input pixels. That is the same ceiling as the glyph
result, in a form that can be measured at any key size.

**`--gate nandinv` is the gate change that pays.** One extra bit per gate
inverts that gate's output, so a gate is `NAND(a,b)` or `AND(a,b)`. That adds no
function to the class — `AND(x,y) = NOT NAND(x,y)`, and a NAND-only net can build
AND by spending a gate in the *next* layer — but the output layer has no next
layer, which is exactly the resource the depth-3 decoder was missing. Measured:
**58 of 256** functions of a 3-bit key (NAND: 16), exact fit for **1–6 glyphs**
(NAND: 1–3; six in 6.1 s), **every subset of six or fewer** (70/70 at four,
56/56 at five, 28/28 at six, against NAND's 1 of 70 at four) and 2 of the 8
seven-glyph subsets, and at eight
glyphs the minimum slack is **1 wrong bit per pattern** — BER **0.0156**,
8 wrong bits in 512 — against NAND's slack 8 and BER 0.1172 (60 bits). Same
shape, same 163 gates. It is not free in silicon: an AND cell is a NAND plus an
inverter, so the exported six-glyph witness (78 AND, 85 NAND) is **808T against
652T** for the all-NAND witness, +24%, and it is what the demo's solver tab now
shows below the NAND one.

**`--gate {xor3,xor2}` is the negative side of the same axis.** `xor3` is two wires plus a
per-gate bias bit the solver also chooses (so each gate is XOR or XNOR, and the
net is affine over GF(2)); `xor2` is the same without the bias. The result is a
clean loss, and it is a *count* rather than a BER: the decoder can compute only
**8 of the 256 functions** of a 3-bit key with `xor3` and **4** with `xor2`,
against NAND's 16. Exact fit stops at 3 glyphs for `xor3` (same as NAND), 2 for
`xor2`, and across all 70 four-glyph subsets `xor3` fits 0 where NAND fits 1
(CEFG); at four glyphs `xor3` cannot reach slack 3, which NAND reaches in 5.7 s.
Mechanism: the decoder's first layer reads only key bits, so a gate there can
compute `k_i ⊕ k_j` but never a single `k_i` (that needs a XOR with a constant,
and the constants a gate can make are outputs of the same layer). Every wire
therefore stays in `span{1, k0⊕k1, k0⊕k2, k1⊕k2}` at any depth — 8 functions;
the bias buys the complements, taking 4 to 8. A decoder input pinned to 0 would
buy the other doubling and reach all 16 affine functions, i.e. at best tie NAND.
Six bugs have now been caught by selftest; the fifth was the XOR parity clauses
forbidding the satisfying assignment instead of the violating one, and the sixth
was `nandinv`'s NAND clauses left unguarded, so every gate was forced to be NAND
*and* AND at once.

**`--gate mix` is the control that matters, and it corrects the checkpoint.** A
free type bit per gate (`NAND(a,b)` or `XOR(a,b) XOR bias`) takes the same
`3→16→32→64` decoder from 16 functions of a 3-bit key to **134 of 256**, and from
"four glyphs UNSAT" to **"four glyphs exact in 0.8 s"**. So the 16-of-256 wall is
mostly the *gate*, not the template; the shape still bounds it (134, not 256)
but "the template is the limit" was too coarse. Eight glyphs under `mix` timed
out at 300 s, so that number is unmeasured rather than negative — do not quote
it as either.

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
| Sobel edge targets (`--binarize edge`) | Legible line drawings, but this trainer cannot fit them: the best val epoch is epoch 0 (the random init), key entropy collapses 0.926 -> 0.291 and ones F1 is 0.198 vs 0.783 for threshold. Weighting positives 6x or 100x changes nothing that survives selection |
| More data / augmentation as an accuracy fix | The model cannot fit 100 images (train BER 0.1741 after ~12,000 updates), so there is no overfitting gap to close |
| Łukasiewicz gate relaxation | 0.2117 at seed 42 vs 0.1722. Nilpotent: zero gradient whenever `u + v <= 1` |
| Gödel `min` as the default | Better at equal training budget (−0.0089 over 10 seeds) but much worse at equal size: pruned to 1,400 gates it scores 0.2154 vs 0.1811 for product (0/5 seeds win). Its extra gates are load-bearing |
| **24-bit RGB (`--rgb8`)** | 16×16: abs MAE **0.2120** vs mean image 0.2031 (16,128 gates). 4×4: **0.2382** vs 0.1665 (1,008 gates). Cannot memorise 100 images either (train BER 0.3839 at 16×16 after ~18,700 updates, vs 0.1741 for grayscale). Half the planes are noise — see Checkpoint 11 |
| 4×4 with more rate / mixing / depth / width | Key 48→96→192: abs MAE 0.2411 / **0.2325** (BER falls 0.4425→0.4110, but the gain is entirely in the noise planes). Cross-plane candidates: 0.2396. Same-resolution layers (8 levels instead of 4): 0.2764, same memorisation number (0.4130). Overcomplete first layer (3,312 gates, 5 levels per side): 0.2319, best of the family, 2.6× the gates for 0.006 MAE and memorisation 0.3977. Nothing approaches the 0.1665 bar |
| SAT-solved wiring on 8×8 glyphs (`scripts/sat_ae.py`) | Kissat 4.0.1 proves the trainer's template `64→32→16→3→16→32→64` (163 gates) fits 1–3 glyphs exactly and is **UNSAT at 4**, in 0.1 s (CaDiCaL 3.0 agrees). Not the key size and not the encoder: a 3-bit key separates all 8 glyphs, but the decoder computes only **16 of the 256 functions** of a 3-bit key (18 with 5 extra mixing layers, 18 at *any* depth), so four glyphs is essentially never decodable — 1 of the 70 four-glyph subsets fits (CEFG), the rest are UNSAT. Wider keys (4/5/6/8 bits), 4 extra same-resolution layers, and 32-wide decoders change nothing. Exact fit tops out at 3 of 8; the minimum slack is 3 wrong bits per pattern for 4 glyphs and 8 for 8 glyphs (BER 0.1172). A 3-glyph exact circuit is pure lookup: 3 of the 5 held-out glyphs decode bit-exactly to a trained image, 2 land on an untrained code |
| 3-pin XOR (XOR/XNOR) gates in the SAT template (`--gate xor3`, bias bit solved per gate) | Strictly fewer functions than NAND: **8 of 256** functions of a 3-bit key (plain 2-input XOR, `--gate xor2`, gets 4; NAND gets 16), because the decoder's first layer can build `k_i⊕k_j` but never a single `k_i` and the reachable set `span{1,k0⊕k1,k0⊕k2,k1⊕k2}` is closed forever. Exact fit stops at 3 glyphs (same as NAND) but 0/70 four-glyph subsets vs NAND's 1/70, and it cannot reach slack 3 at four glyphs where NAND is SAT in 5.7 s. The bias pin is worth exactly one doubling (4→8); a decoder input pinned to 0 would buy the other and reach the full 16-function affine class, i.e. at best tie NAND |

## Open threads, in priority order

1. **The key classifies, and an all-NAND circuit can too.** On MNIST, softmax
   regression on the 128-bit key gets **82.0%** (raw pixels: 88.9%), so 8x
   compression costs ~7 points of class information. A NAND circuit on the same
   keys reaches **77.4%** with 392 gates; a circuit reading **784 raw pixels**
   does better: **80.5% with 429 gates**. Bypassing the 98-bit key is worth
   ~5 points and removes a stage, so that is what the demo runs. Size is the
   lever: 46 collapsed gates give 36.9%, 228 give 70.3%, 370 give 75.5%, 392
   give 77.4%. Distillation from the softmax teacher did *not* help (67.8%),
   and probes plateau by epoch 100. Utilisation is the open problem: only 429
   of 928 trained gates survive simplification. Labels live in `XAE5`; see
   `classify`, `classify-nand` and `eval-circuit`.
2. **Key space is generative, not just a codebook.** `xorae prior` fits a
   causal linear AR model over the 128-bit keys: on MNIST it reaches 0.318
   bits/bit held-out (about 41 bits of entropy) and its samples round-trip as
   well as real keys (0.806 vs 0.801) while decoding to recognisable digits.
   Next steps if this is the direction: a nonlinear/RNN prior should cut the
   NLL further, and a GAN on keys is now defensible rather than speculative.
3. **Capacity per gate is the binding constraint.** Three independent
   measurements: the model cannot fit 100 images (train BER 0.1741 ≈ the 4,000
   image result); train/val/test BER are all equal; and the wiring is spread so
   thinly that the encoder reads only **362 of 1,024** input pixels while the
   decoder's median output bit depends on 2 of the 128 key bits
   (`scripts/circuit_cones.py`). Objective tweaks do not help (see the ruled-out
   table). The levers left are what a gate is spent on: free inversions (42% of
   gates are `Nand(x,x)`), spatial weight sharing, and distillation from a CNN
   teacher. A margin/hinge loss on bit decisions is the one cheap objective
   variant not yet tried.
4. **Colour, 2 bits/channel: shipped; 24-bit: ruled out.** `--rgb2` gives 64
   colours as six bit planes (1,536 bits, 192-bit key, 8×); over six paired
   seeds it cuts absolute reconstruction error by 33.7% for 1.91× the gates. It
   is in the demo as `cats16-rgb2`, and `eval` reports both `abs MAE vs
   original` and the palette-relative MAE. `--rgb8` (24 planes, no
   quantisation) is implemented and exact, but the model cannot train on it:
   16×16 scores 0.2120 abs MAE against 0.2031 for a constant mean image, and
   4×4 scores 0.2382 against 0.1665. Half the planes are statistical noise
   (neighbour-majority error 0.48 on the low nibble), so bit-MSE saturates at
   its floor and the exported circuit never sees structure. See Checkpoint 11.
   The follow-up if 24-bit is ever required: train only the high nibble (16
   planes, 0.031 floor), and/or use same-resolution layers throughout so the
   local window reads across columns at all.
5. **Size-vs-BER frontier: first pass done.** `xorae prune` gives product
   0.1749 at 1,400 gates and 0.1775 at 1,300 (from 0.1722 at 1,547), steepening
   below 1,200. The greedy bypass rule is crude — better pruning (retrain or
   fine-tune after each cut, or remove whole cones) could move this curve.
6. **Inversions are 42% of the budget.** In the shipped model 907 of 2,165
   reachable gates are `Nand(x, x)` (i.e. `NOT x`), and only 627 of those are
   distinct wires. A trainer that lets a gate consume an inverted wire for free
   (AIG-style) would spend its gate budget on real logic instead of negation —
   plausibly the largest remaining win, but it touches the model format, the
   evaluator and the exporter. **There is now SAT evidence for this**: letting a
   gate invert its own output (`--gate nandinv`) cuts the eight-glyph minimum
   from 8 wrong bits per pattern to 1 and the exact frontier from 3 glyphs to 6,
   on the same 163 gates. That is a gate-level inversion, not a free wire
   inversion, so it is a lower bound on what AIG-style would buy.
7. **Seed-averaged reporting for everything else.** The default config now has
   five seeds (0.1730 ± 0.0066) and runs are deterministic, so the harness is
   established. Re-measure the colour and wide configs the same way (5 seeds,
   paired) rather than as single runs — otherwise they repeat the `gain_end`
   mistake.
8. **SAT as an exact oracle (Checkpoint 12).** The encoder answers "is this
   template feasible?" in 0.1 s on 8×8 glyphs, and it already says the
   trainer's depth-3 decoder cannot fit four patterns. Unmeasured: the minimum
   gate count for an exact fit at 8 patterns (a shape search, not a solver
   problem), whether a SAT-found circuit generalises to held-out glyphs,
   whether an *encoder* with full mixing between same-size layers fixes the
   saturation, and whether the same question is answerable at 32×32 (64→32→16→8
   with 128-bit keys would be ~10⁵ clauses per pattern — likely fine for SAT,
   and it would say outright whether the shipped shape can fit 8 CIFAR cats).

## Gotchas

- `data/`, `runs/`, `target/`, `web/dist/`, `web/src/wasm/pkg/` are gitignored.
- The CIFAR binary stores channels **planar** (1024 R, then 1024 G, then 1024 B),
  not interleaved. There is a regression test for this; earlier results were
  invalidated by the bug.
- Dataset cache format is `XAE5` when labels are present, else `XAE4`. `XAE2`/`XAE3`/`XAE4` still load (labels default to empty).
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
- `--relaxation {product,min,lukasiewicz}` selects the gate t-norm. `product`
  is the default; `min` (Gödel) wins at equal training budget but loses badly
  once circuits are compressed to the same size (see Checkpoint 7).
- `xorae prune --target N [--tolerance T]` is the *lossy* size pass (bypass +
  dead-code elimination, greedy on a calibration sample). `simplify` is the
  exact one; use `simplify` first.
- **BER is a bad metric on sparse data.** On binarized MNIST (13.4% ones) the
  model scores 0.2254 while all-zero scores 0.1366, despite reconstructing
  recognisable digits. `eval` now also prints **ones F1** and **ones IoU**
  (precision/recall over the set bits), which is what to compare when the
  target is sparse. On the shipped 32x32 model they are 0.783 / 0.643.
- `eval` also prints **edge F1@1.5 vs chance** (a tolerance-aware boundary
  score: chamfer distance transform, precision/recall within 1.5 px, averaged
  over both directions). The chance column is not optional — at 32x32 with 15%
  ink, random spray scores 0.56, so a bare 0.57 means nothing. The shipped
  model is 0.889 vs 0.695 chance; the edge model is 0.574 vs 0.564.
  Chamfer is reported in pixels and a blank image clamps to the image
  diagonal: the distance transform's sentinel used to leak into the mean and
  print 49,000,008 px whenever a reconstruction came out all zeros.
- **The 32x32 threshold target is an inkblot.** Rendering the binarized target
  next to the 32x32 grayscale original makes it obvious that a *perfect* model
  would still output blobs, so "does it look like a cat" cannot be the measure
  for that representation. The 4-level palette (`--palette 4 --size 32`,
  2,048 bits, `channels = 2`) keeps the targets recognisable; see
  `configs/cats32_pal4.toml`.
- `--palette` used to panic for any dataset with more than one image
  (`from_color_parts` called `rgb_to_gray` on the whole buffer); fixed.
- `train-nand --positive-weight W` weights target 1-bits W times in the loss
  (default 1). It is necessary in principle for sparse targets, but on edge
  maps it did not rescue the run on its own.
- MNIST is fetched separately: `./scripts/fetch_mnist.sh data/mnist`, then
  `preprocess --mnist-dir ... --limit N`.
- **Memory is tight on the phone.** The full 28×28 MNIST dataset is 270 MB on
  disk and the loader keeps `rgb` (n·px·3) and `values` (n·px) in RAM even for
  grayscale training; training then makes an f32 copy of the training split
  (`n_train × cols × 4`). A 48k-image 28×28 run pushes the device into swap and
  background jobs get killed. Use `--limit 12000` for experiments, keep
  `RAYON_NUM_THREADS` at 6 or lower, and do not compile during a long run.
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
