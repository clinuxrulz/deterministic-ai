# Experiment design

## Question

Can a genetic algorithm wire a small Boolean circuit into an encoder/decoder pair
that compresses a binarized image into a short key and reconstructs it — and does
adding a small amount of nonlinearity materially improve the result?

## Tracks

1. **Strict XOR (`xor_only`)** — control. The model is affine over GF(2), so it
   is a binary linear code. Expect a plateau at the best linear-code error.
2. **XOR + AND (`xor_and`)** — ~90% XOR, ~10% AND. Universal Boolean circuits.
   This is the actual nonlinear experiment.
3. **Full gate set (`full`)** — optional ablation with XNOR/OR/NOT as well.

Gate-set switches are in `crates/core/src/circuit.rs`.

## Data

- CIFAR-10 binary format, cat class (label 3): 5,000 train + 1,000 test images.
- Grayscale luma, box-downsample 32→16 (configurable), binarize with:
  - `threshold` (fixed 128),
  - `otsu` (per-image Otsu),
  - `dither` (Floyd–Steinberg; default).
- Cached as an `XAE2` file. Train/validation/test split is deterministic from the
  seed; the CIFAR test batch is always the test split.
- The toy dataset is an 8×8 image set formed from all XOR combinations of four
  structured basis images. Training uses the zero pattern + basis vectors;
  held-out combinations test whether the model learned the subspace rather than
  memorised training points.

## Model

- Encoder: `N = size²` inputs → `E` gates → `K` key outputs.
- Decoder: `K` inputs → `D` gates → `N` reconstruction outputs.
- Constants 0 and 1 are available, so affine offsets and NOT are expressible.
- Gates are stored in topological order; gate `i` reads only earlier wires.
- Defaults: 16×16, 32-bit key, 512 + 512 gates.

## Fitness

```
fitness = BER
        + entropy_weight    · (1 − mean per-key-bit entropy)
        + uniqueness_weight · (1 − distinct keys / min(batch, 2^K))
        + parsimony_weight  · (active gates / total gates)
```

- **BER** is the mean Hamming distance between the binarized input and the
  reconstruction, divided by the number of bits.
- **Entropy** prevents key collapse (all-zero or constant key bits).
- **Uniqueness** rewards using the key space. It is critical on small toy
  datasets and harmless on CIFAR, where random keys are already distinct.
- **Parsimony** is a small tie-breaker in favour of simpler circuits.

## Baselines

| Baseline | Meaning |
|---|---|
| All-zero | Reconstruct every bit as 0 |
| Per-bit majority | Best constant image on the training set |
| Top-K identity | Key = first K input bits; pad the rest with zeros |
| Greedy GF(2) basis | Span a K-dimensional subspace of training images; encode by coordinates, decode by XORing basis rows |

Run with:

```bash
cargo run -p xorae -- baselines --data data/cats16.bin --key-bits 32
```

The strict-XOR GA should be compared against these, not against a neural
network. A small MLP autoencoder is a useful optional reference but is not part
of the current implementation.

## Metrics

- Test BER and exact-match rate.
- Key entropy and key uniqueness.
- Active gate count.
- Training curve (`runs/<id>/log.csv`).
- Reconstruction PNG (`xorae render`).
- Key-flip robustness (planned): BER vs. number of flipped key bits.

## Ablations

- Key size: 8, 16, 24, 32, 48, 64 bits.
- Gate budget: AND fraction 0%, 5%, 10%, 25%, 100%.
- Binarization: threshold vs. Otsu vs. dither.
- Image size: 8×8, 16×16, 32×32.
- Mutation rate, population size, crossover on/off.
- Entropy/uniqueness weights.

## Hypotheses

- **H1.** Strict XOR converges to a linear-code-quality solution and cannot beat
  the best linear baseline.
- **H2.** XOR+AND beats strict XOR at the same key size and gate budget.
- **H3.** Key uniqueness prevents collapse and improves held-out exact matches.
- **H4.** CGP-style neutral drift is necessary for wiring search to make
  progress; plain crossover-only GA is worse.

## Reproducibility

- Every run writes `config.toml`, `log.csv`, `model.json`, and `metrics.json`.
- `--seed` controls the GA, split, and batch sampling.
- A determinism test compares two runs byte-for-byte.
- Record the Rust toolchain, CPU count, and commit SHA alongside results.

## Status

- Toy 8×8, 4-bit key, strict XOR: 0 BER, 11/11 held-out patterns exact.
- CIFAR-10 cats, 16×16, 32-bit key, 3,000 generations, seed 42:
  - Dither baselines: greedy GF(2) basis **0.4325** test BER.
  - Threshold baselines: greedy GF(2) basis **0.3731** test BER.
  - Strict XOR, dither: **0.4811** test BER (0.4826 val); 432 active gates.
  - XOR+AND, dither: **0.4418** test BER (0.4412 val); 368 active gates.
  - XOR+AND, threshold: **0.2285** test BER (0.2241 val); 283 active gates;
    0.30% exact matches. Same config, only preprocessing changed.
- **Correction:** earlier results were invalid because the CIFAR loader read the
  planar channel layout as interleaved RGB. The loader is fixed; all numbers
  above use corrected datasets.
- Binarization is currently the largest lever: threshold roughly halves the
  dither test BER. The threshold model also beats the strongest threshold
  baseline by about 14 BER points. Otsu has not been trained yet.
- The strict-XOR GA lands above the greedy basis, which is consistent with both
  the linearity barrier and the limits of GA search on this objective.
- Reconstructions are coarse at 16×16 with a 32-bit key. See
  `runs/cats16-xor-and/recon.png`, `runs/cats16-xor-and-threshold/recon.png`,
  and the Solid demo in `web/`.
- Color:
  - 16-color global palette (4 bits/pixel, 64/128-bit key): index accuracy
    stayed at chance (~6.3%), test color MAE ~0.273. Negative result.
  - RGB 1-bit/channel (3 binary planes, 768 bits, 128-bit key, 1,024+1,024
    gates): test color MAE **0.1729**, pixel accuracy **64.06%**, PSNR 7.62 dB,
    577 active gates, 30,000 generations. The web demo defaults to this model.
  - Next: YCbCr 4:2:0 for smoother chroma.
- Resolution (32×32 grayscale, 128-bit key, 2,048+2,048 gates, 30,000
  generations): test BER **0.2097**, val 0.2058, 863 active gates, 0% exact.
  Same 8× compression and same 4 gates per output bit as the 16×16 threshold
  model (test BER 0.2285, 283 active gates, 3,000 generations). Native
  resolution improves large-scale structure and shading but costs ~40× the
  compute for an 8% relative gain, and 79% of the circuit stays inactive.
  See `runs/cats32-gray/recon.png`.
- Optimizer note: the greedy GF(2) baseline (0.3731) outperforms the
  GA-evolved strict-XOR circuit (0.4811) on the same data, and both the 16×16
  and 32×32 XOR+AND runs leave most gates inactive. The GA is a weak credit
  assigner; see `docs/prior-art.md` and Phase 4 of `docs/color-plan.md`.
- Gradient trainer (`train-grad`, fixed hierarchical wiring, learned gate
  types, soft gates with an annealed gate softmax). Findings:
  - Backprop is correct: with continuous activations the network fits the
    training set (MSE 0.25 → 0.14 in 25 epochs).
  - **Binarising activations during training does not work.** Hard
    binarisation with an identity straight-through estimator collapses to
    predicting the per-pixel mean (MSE ~0.22, BER 0.50). A sigmoid-temperature
    interpolant is worse: it keeps every activation in roughly `[0.38, 0.62]`,
    which compounds across layers until the signal is gone.
  - A linear→hard blend interpolant (`(1-λ)·z + λ·step(z)`) preserves the range
    but still fails to learn once λ reaches 1.
  - Keeping activations continuous while annealing the *gate* softmax to
    one-hot does learn, and the exported literal circuit is far from chance.
    This is the working configuration.
  - The residual discretisation gap comes from continuous activations at
    export time, not from the gate choice. Hardening only in the final few
    percent of epochs (`anneal_fraction`) is the mitigation under test.
  - Best gate-type result: test BER 0.2523 (4 ops), which does **not** beat the
    GA. Learned gate types are not where the win is.
- NAND-only wiring search (`train-nand`). Every gate is NAND, so the gate type
  carries no information and the circuit is entirely described by its wiring.
  Each gate has 12 candidate input wires (9 local, 3 long-range); two candidate
  softmaxes select its two inputs and anneal to one-hot.
  - 20 epochs: test BER 0.1918.
  - 200 epochs (final): **test BER 0.1727, val 0.1710**, 2,165 active gates of
    2,688, 128-bit key, 8× compression.
  - This beats the GA (0.2097) with 34% fewer gates and ~1% of the wall clock.
  - Val BER bottomed at 0.1685 around epoch 125 and drifted up to 0.1710, so
    the annealing schedule, not the epoch count, is the current limit.
  - See `runs/nand32/recon.png`.


## Checkpoint (2026-10-29)

Best model so far: **NAND-only 32×32 grayscale**

- 2,165 NAND gates after dead-gate pruning (2,688 before).
- 128-bit key, 8× compression, test BER **0.1727**, val 0.1710.
- Live in the web demo as `web/public/models/cats32-nand.json` (default entry).

### Reproduce

```bash
cargo run -p xorae -- preprocess \
  --cifar-dir data/cifar-10-batches-bin \
  --out data/cats32_threshold.bin --size 32 --binarize threshold

cargo run --release -p xorae -- train-nand \
  --config configs/cats32_nand.toml \
  --data data/cats32_threshold.bin --out runs/nand32 --epochs 200
```

`RAYON_NUM_THREADS` controls worker threads. Results are identical for any
thread count; 4 threads is ~2× faster than 1.

### Open threads

- **Colour is encoding-limited.** RGB1 thresholds each channel at 128, so every
  pixel snaps to one of 8 fully saturated colours. Muted browns and greys cannot
  be represented at all. The recommended fix is 2 bits per channel (64 colours),
  not a better search: 16×16 × 6 bits = 1,536 bits.
- **Non-halving reductions hurt.** Colour at 6× compression (extra 192→128
  step) scored 0.2539, worse than 8× at 0.2192. The strided window covers the
  input poorly. Prefer shapes that halve.
- **Polish does not generalise.** Greedy wire swaps improve their scoring sample
  but not the test set; the gradient solution is already a single-wire local
  optimum.
- **Minimax does not apply.** The wiring search is single-agent; minimax needs an
  adversary. It would only fit a robustness objective (worst-case wire faults).
  MCTS over wire-swap moves is the correct "tree search" analogue, but the
  polish result suggests search is not the bottleneck.

## Checkpoint 2 (2026-10-29): checkpointing, variance, capacity

Two trainer changes and one methodology finding.

Trainer changes (`crates/cli/src/nand.rs`, `crates/cli/src/main.rs`):

- **Best-val checkpointing.** The wire logits of the best validation BER are
  snapshotted and restored before polish/prune/export. Previously the exported
  model was the *last* epoch, which is why the earlier log recorded val BER
  improving to 0.1685 and then "drifting" to 0.1710: the better weights were
  simply discarded. Selection uses val only, so test is untouched.
- **`--gain-end` override** on `train-nand`, plus `configs/cats32_nand.toml`
  pinned to `epochs = 200` (it said 300 while the documented run passed
  `--epochs 200`).

Hardware note: 8 cores, 200 epochs in **1m51s**. The old "~8 min" figure was 4
cores, so that is 4.3× faster on 2× the threads — the earlier number may have
been measured under thermal throttling or alongside other work.

### Reproduction (seed 42, 200 epochs)

| Metric | Documented | Re-run |
|---|---|---|
| Test BER | 0.1727 | **0.1722** |
| Val BER | 0.1710 | 0.1698 (restored epoch 125; last epoch was 0.1717) |
| Active gates | 2,165 | 2,170 |
| Wall clock | ~8 min | 1m51s |

Reproduces within noise, and the checkpoint restore recovers the 0.002 val the
anneal used to throw away.

### Seed variance

The seed also chooses the train/val/test split, so it is the error bar for
everything else. Seeds 1, 2, 3, 42:

| `gain_end` | s42 | s1 | s2 | s3 | Mean ± sd |
|---|---|---|---|---|---|
| 32 (default) | 0.1722 | 0.1700 | 0.1735 | 0.1724 | **0.1720 ± 0.0015** |
| 12 | 0.1679 | 0.1800 | 0.1928 | 0.1856 | 0.1816 ± 0.0105 |

**Methodology finding: the seed-42 `gain_end = 12` result was split luck.** It
sits 2.8 sd below the mean of the other three seeds. Over seeds 1-3 the softer
schedule averages 0.1861 versus 0.1720 — worse, with sd 0.0064 vs 0.0015. Any
single-seed win below ~0.005 is not a result.

### Schedule sweep (seed 42, 200 epochs)

| `gain_end` | Test BER | Val BER | Train MSE | Gates | Best epoch |
|---|---|---|---|---|---|
| 32 | 0.1722 | 0.1698 | 0.0946 | 2,170 | 125 |
| 16 | 0.1710 | 0.1686 | 0.0933 | 2,190 | 125 |
| 8 | 0.1706 | 0.1678 | 0.0916 | 2,174 | 199 |
| 12 | 0.1679 | 0.1648 | 0.0922 | 2,174 | 199 |

Read this table with the variance section above: only the seed-42 column exists
here, and 12 is known to be an outlier. The one durable observation is
mechanical — below `gain_end = 16` the anneal stops overshooting (best epoch
becomes the last), above it the overshoot returns (best epoch 125).

### Capacity: negative, twice

| Model | `gain_end` | Gates trained → active | Test BER | Val BER | Train MSE |
|---|---|---|---|---|---|
| Narrow | 32 | 2,688 → 2,170 | 0.1722 | 0.1698 | 0.0946 |
| Wide | 32 | 4,736 → 3,085 | 0.1913 | 0.1879 | 0.0783 |
| Wide | 12 | 4,736 → 3,193 | 0.2133 | 0.2113 | 0.0759 |

The wide model fits the training set substantially better and generalises
substantially worse. More capacity is not the constraint; the earlier
"inconclusive" run was heading to this answer. It also overshoots harder: at
`gain_end` 32 the best val is epoch 175 (0.1879) and then rises to 0.1984 at
epoch 199; at `gain_end` 12 the best is epoch 125 (0.2113) and it climbs to
0.2314 — the opposite of the narrow model, where lower gain stopped the
overshoot.

### Training length: no effect

`gain_end` 12, 300 epochs, seed 42: test 0.1674 / val 0.1654 versus 0.1679 /
0.1648 at 200 epochs. Same seed, same split, difference inside the 0.002 noise
band.

### Net position

Nothing beat the default configuration. Published artifact stays
`web/public/models/cats32-nand.json` (0.1727 / 2,165 gates). What changed is
that the number is now reproducible, the exported weights are best-val rather
than last-epoch, and the two threads the previous checkpoint left open —
capacity and schedule — are both closed negative.

