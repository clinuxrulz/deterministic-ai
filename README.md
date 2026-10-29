# deterministic-ai

Evolve **XOR-gate autoencoders** with a genetic algorithm.

The encoder squishes a binarized image into a short **key**; the decoder rebuilds
the image from that key alone. The model is a wired Boolean circuit, not a neural
network. There are no gradients and no floating-point weights — the wiring is
evolved by a CGP-style genetic algorithm.

```
image bits ──▶ encoder (XOR/AND gates) ──▶ K-bit key ──▶ decoder (XOR/AND gates) ──▶ reconstruction
```

## Read this first: XOR is linear over GF(2)

A feed-forward circuit built only from XOR gates (with constant 0/1 wires) is an
**affine map over GF(2)**. Composition cannot escape that class. So a strict-XOR
encoder/decoder is mathematically a **linear error-correcting code** with rank
≤ K, and it can only reconstruct images lying in a K-dimensional affine
subspace. That is a real experiment, but it is not a general function
approximator.

The repo therefore runs two tracks:

| Gate set | What it is | Purpose |
|---|---|---|
| `xor_only` | XOR + constants | Linear-code baseline; demonstrates the barrier |
| `xor_and` | ~90% XOR, ~10% AND | Universal Boolean circuits; the actual nonlinear experiment |

`XOR + AND` is universal because every Boolean function has an algebraic normal
form (an XOR of AND terms). See [`docs/linearity.md`](docs/linearity.md) for the
proof and the implications.

## What works now

- **`crates/core` (`xorae-core`)** — WASM-safe bit packing, circuits, bit-sliced
  evaluator, image preprocessing, and the portable model JSON format. No
  `rayon`, filesystem, or image-decoding dependencies.
- **`crates/cli` (`xorae`)** — CIFAR-10 loader, toy linear-subspace generator,
  genetic algorithm, baselines, rendering, and model/sample export.
- **`crates/wasm` (`xorae-wasm`)** — a small `wasm-bindgen` bridge over
  `xorae-core`: model parsing, image preprocessing, encode and decode.
- **`web/`** — Solid 2.0 + Vite single-page demo: upload an image or pick a
  sample, run the model in the browser, inspect the key, flip key bits, and see
  the reconstruction and changed pixels.
- **Tests** — proof-by-execution that XOR-only circuits are affine, bit-parallel
  evaluator equivalence, toy convergence and generalisation, and GA determinism.

A bundled toy model (`web/public/models/toy.json`) reconstructs all 11
held-out patterns of its 4-dimensional GF(2) subspace exactly.

## Layout

```
crates/
  core/        xorae-core — shared inference, WASM-safe
  cli/         xorae — training, datasets, baselines, export
  wasm/        xorae-wasm — wasm-bindgen bridge
web/           Solid 2.0 + Vite SPA, model/sample assets, WASM package output
configs/       toy / CIFAR strict-XOR / CIFAR XOR+AND
scripts/       fetch_cifar10.sh, build-wasm.sh
docs/          linearity notes and experiment design
```

## Quick start: toy (no download)

```bash
# 8x8 images, 4-bit key, strict XOR. Trains in ~25s in debug mode.
cargo run -p xorae -- train --config configs/toy.toml --toy --out runs/toy

# Held-out XOR combinations should be reconstructed exactly.
cargo run -p xorae -- eval --model runs/toy/model.json --toy
cargo run -p xorae -- render --model runs/toy/model.json --toy \
  --out runs/toy/recon.png --count 8 --scale 12
```

Expected output:

```
test: BER 0.0000, exact 100.00%, key entropy 0.958, active gates 100 (11 samples)
```

## Quick start: CIFAR-10 cats

The dataset is 162 MB. The fetch script resumes interrupted downloads.

```bash
./scripts/fetch_cifar10.sh data
cargo run -p xorae -- preprocess \
  --cifar-dir data/cifar-10-batches-bin \
  --out data/cats16.bin \
  --size 16 --binarize dither

cargo run -p xorae -- baselines --data data/cats16.bin --key-bits 32
```

Train the linear-code control:

```bash
cargo run -p xorae -- train \
  --config configs/cats16_xor.toml \
  --data data/cats16.bin \
  --out runs/cats16-xor
```

Train the XOR+AND experiment:

```bash
cargo run -p xorae -- train \
  --config configs/cats16_xor_and.toml \
  --data data/cats16.bin \
  --out runs/cats16-xor-and
```

Evaluate and render:

```bash
cargo run -p xorae -- eval --model runs/cats16-xor/model.json \
  --data data/cats16.bin --split test

cargo run -p xorae -- render --model runs/cats16-xor/model.json \
  --data data/cats16.bin --out runs/cats16-xor/recon.png --count 8
```

`configs/cats16_*.toml` use 3,000 generations. For a quick smoke test, add
`--generations 200` to the train command. `--seed` overrides the config seed.

## Web demo

The web app is a Solid 2.0 SPA. Rust remains the numeric core, compiled to WASM
through `wasm-pack`.

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack     # if it is not already installed

cd web
npm install
npm run dev                 # predev builds web/src/wasm/pkg, then starts Vite
```

For a production static build:

```bash
cd web
npm run build               # runs wasm-pack, wasm-opt, then vite build
npm run preview
```

The static site is written to `web/dist/`. The build script is
`scripts/build-wasm.sh`; it outputs the generated JS/TS/WASM package to
`web/src/wasm/pkg/` (gitignored).

The app loads the bundled toy model and 8 toy samples. To add trained cat models:

```bash
cargo run -p xorae -- export \
  --model runs/cats16-xor-and/model.json \
  --out web/public/models/cats16-xor-and.json

cargo run -p xorae -- export-samples \
  --data data/cats16.bin --out web/public \
  --count 8 --prefix cat
```

The manifest is `web/public/models/index.json`. Solid is pinned to
`2.0.0-rc.19`; the app uses the client-only Vite mode (no SSR).

## Commands

| Command | Purpose |
|---|---|
| `fetch-data` | Download/extract CIFAR-10 (or use `scripts/fetch_cifar10.sh`) |
| `preprocess` | Convert one CIFAR class into a cached `XAE2` dataset |
| `train` | Run the GA (`--toy` for the built-in subspace dataset) |
| `eval` | Report BER, exact-match rate, key entropy, active gates |
| `render` | Write a PNG grid: original / binarized / reconstruction |
| `export` | Write the portable model JSON |
| `export-samples` | Write sample PNGs + `samples/index.json` for the web app |
| `baselines` | All-zero, per-bit majority, top-K identity, greedy GF(2) basis |

## How the model works

**Genotype.** A circuit is a fixed-length list of gates plus output taps. Gate `i`
may read primary inputs, constant 0/1, or earlier gates, so evaluation is a
single forward pass. Inactive gates are free to mutate, which provides CGP's
useful neutral drift.

**Evaluation.** Batch samples are bit-sliced into `u64` planes: one word holds
the same wire for 64 images. A gate is a word-wise XOR/AND, so evaluating a
population of circuits over hundreds of images is cheap and cache-friendly.

**GA.** `(μ+λ)` with tournament selection, elitism, point mutation, optional
uniform crossover, neutral drift through inactive gates, and random immigrants
when progress stalls. The RNG is seeded and evaluation order is deterministic.

**Fitness** (lower is better):

```
BER
+ entropy_weight    · (1 − mean key-bit entropy)
+ uniqueness_weight · (1 − distinct keys / capacity)
+ parsimony_weight  · (active gates / total gates)
```

The entropy and uniqueness terms prevent the classic collapse where the
encoder emits a constant key and the decoder emits the dataset mean. The
uniqueness term is especially important on the toy subspace: it rewards an
injective encoder before the decoder has learned anything.

**Baselines.** Any BER number is meaningless without context, so the CLI reports
all-zero, per-bit majority, top-K identity, and a greedy GF(2) basis baseline.
The greedy basis spans a k-dimensional subspace of training images and
reconstructs by coordinates — the best a fixed linear code can do. The
strict-XOR run should plateau near it; that is the expected result, not a bug.

## Determinism

- `--seed` controls the GA, the train/val split, and dataset sampling.
- Parallel evaluation writes each individual independently; selection sorts by
  `f32::total_cmp`.
- A test asserts that two runs with the same seed produce byte-identical model
  JSON.

## Build footprint

The workspace is configured to keep build artifacts and the deployed bundle small:

- `image` is compiled with the PNG codec only. This removes the AVIF/JPEG/WebP/TIFF
  codec stack from the build.
- Dev builds use line-table debug info (`[profile.dev] debug = 1`). Set it to
  `0` if you do not need a debugger and want the smallest `target/`.
- Release builds strip debug info.
- The web build runs `wasm-pack` + `wasm-opt` and Vite. The current release
  bundle is roughly **137 KB WASM** (64 KB gzip), **50 KB JS** (19 KB gzip), and
  **4 KB CSS**.

With those settings, `cargo test --workspace` plus `npm run build` produced a
**~1.2 GB** `target/` (down from ~5.6 GB before the PNG/debug changes). The
deployed `web/dist/` is about **272 KB total**. `target/` is disposable and
gitignored — `cargo clean` removes it entirely.

## Tests

```bash
cargo test --workspace
cargo check -p xorae-wasm --target wasm32-unknown-unknown

cd web
npm run build
```

The core tests include:

- `tests/linearity.rs` — a random XOR-only circuit is exactly its derived affine
  matrix, including the constant offset.
- `tests/evaluator.rs` — bit-sliced evaluation matches scalar evaluation across
  two `u64` words.
- `tests/toy_evolution.rs` — the GA learns a linear subspace, generalises to
  held-out XOR combinations, and is deterministic.
- `crates/wasm` — binarize-mode parsing and the bridge surface.

## Results so far

| Run | Key | Train BER | Held-out BER | Notes |
|---|---|---|---|---|
| Toy 8×8 strict XOR | 4 bits | 0.0000 | 0.0000 (11/11 exact) | 128+128 gates, seed 42 |
| CIFAR cats 16×16 strict XOR | 32 bits | — | — | Run the commands above |
| CIFAR cats 16×16 XOR+AND | 32 bits | — | — | Run the commands above |

The CIFAR numbers are intentionally left blank until the full runs are
completed. Add them here (and in `docs/experiment.md`) when available.

## Known limitations

- Strict XOR cannot learn nonlinear structure; this is a theorem, not a tuning
  problem.
- The GA is CPU-bound. Bit-slicing makes it practical, but full sweeps are still
  hours, not seconds.
- CIFAR cats are 32×32 and cluttered; 16×16 binarized reconstructions are
  meant to be diagnostic, not photorealistic.
- The web app runs inference only. Training stays in the native CLI.
- Solid 2.0 is pinned to a release candidate (`2.0.0-rc.19`). The app uses only
  stable core primitives, but the dependency is pre-1.0-stable by definition.

## Dataset

CIFAR-10: Alex Krizhevsky, *Learning Multiple Layers of Features from Tiny
Images*, 2009. The cat class is label `3`. The fetch script checks the published
MD5 (`c32a1d4ab5d03f1284b67883e8d87530`).

## License

MIT. See [`LICENSE`](LICENSE).
