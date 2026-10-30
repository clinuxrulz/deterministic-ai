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

The seed also chooses the train/val/test split, so it sets the error bar for
everything else. Default schedule (`gain_end` 32), seeds 0-4:

| seed | 0 | 1 | 2 | 3 | 4 | mean ± sd |
|---|---|---|---|---|---|---|
| Test BER | 0.1657 | 0.1700 | 0.1735 | 0.1724 | 0.1835 | **0.1730 ± 0.0066** |
| Val BER | 0.1655 | 0.1693 | 0.1711 | 0.1752 | 0.1857 | |
| Active gates | 2,149 | 2,131 | 2,127 | 2,131 | 2,154 | |

Repeat of seed 1 landed on 0.1700 / val 0.1693 / 2,131 gates again, so runs are
deterministic: the spread is real split-and-init variance, not run noise. The
95% CI of the mean is ±0.0082, and a single run is only good to about ±0.013.

**Correction to the first version of this note.** It claimed ±0.0015 from seeds
1, 2, 3, 42. That was a small-sample artifact: seed 42 sits near the mean by
luck, and seed 4 is 1.6 sd above it. The honest bar is ~4× wider. This is also
why seed 42 is excluded from the table — it was chosen after seeing its result,
so it is not a fair sample.

Consequence: single-run comparisons at this scale are close to worthless for
effects under ~0.01. Everything below is therefore **paired on seed** (same
split, same init, only the knob under test differs).


### Schedule: no detectable effect

Exploratory sweep, seed 42 only, 200 epochs:

| `gain_end` | Test BER | Val BER | Train MSE | Gates | Best epoch |
|---|---|---|---|---|---|
| 32 | 0.1722 | 0.1698 | 0.0946 | 2,170 | 125 |
| 16 | 0.1710 | 0.1686 | 0.0933 | 2,190 | 125 |
| 8 | 0.1706 | 0.1678 | 0.0916 | 2,174 | 199 |
| 12 | 0.1679 | 0.1648 | 0.0922 | 2,174 | 199 |

On that column lower gain looks like a win. It does not survive a paired
comparison over seeds 0-4:

| | mean | sd |
|---|---|---|
| `gain_end` 32 | 0.1730 | 0.0066 |
| `gain_end` 12 | 0.1793 | 0.0123 |
| Paired difference (12 − 32) | **+0.0063** | 0.0113 |

Per-seed differences: −0.0059, +0.0100, +0.0193, +0.0132, −0.0053. Mean +0.0063,
se 0.0051, t(4) = 1.24, 95% CI [−0.0078, +0.0203]. No effect on the mean, 12 wins
on 2 of 5 seeds, and it roughly doubles the spread — a less stable schedule
rather than a better one. The `gain_end` 8 and 16 rows have one seed each and
are not evidence either way.

The durable observation is mechanical: below `gain_end` ≈ 16 the anneal stops
overshooting (best epoch becomes the last), above it the overshoot returns (best
epoch 125). Checkpointing is what insulates the export from that difference, and
it is worth ~0.002 val regardless of the schedule.

### Capacity: negative, twice

| Model | `gain_end` | Gates trained → active | Test BER | Val BER | Train MSE |
|---|---|---|---|---|---|
| Narrow | 32 | 2,688 → 2,170 | 0.1722 | 0.1698 | 0.0946 |
| Wide | 32 | 4,736 → 3,085 | 0.1913 | 0.1879 | 0.0783 |
| Wide | 12 | 4,736 → 3,193 | 0.2133 | 0.2113 | 0.0759 |

The wide model fits the training set substantially better and generalises
substantially worse. More capacity is not the constraint; the earlier
"inconclusive" run was heading to this answer. The lower MSE with worse BER is
objective mismatch rather than overfitting (see Checkpoint 5). It also
overshoots harder: at
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
that the number is now reproducible and has a measured error bar, the exported
weights are best-val rather than last-epoch, capacity is closed negative, and
the schedule knob has no detectable effect at this scale.

Notable trap for the next agent: the seed-42 run at `gain_end` 12 reported
0.1679 and looked like a 0.005 win against 0.1727. With five seeds the default
config's own spread is 0.1657-0.1835, so that "win" was a draw from the same
distribution. Ask for paired multi-seed comparisons before believing anything
under ~0.01.

## Checkpoint 3 (2026-10-30): exact simplification

A trained circuit is a Boolean expression, so it can be rewritten without
changing the function it computes. `Circuit::simplify` does three exact
rewrites: constant folding, hash-consing (CSE) and dead-gate removal. It is not
distillation — nothing is approximated and no data is used; the output is
bit-identical for **every** input vector, not just the dataset.

### Result on the shipped model

| | Before | After |
|---|---|---|
| NAND gates | 2,688 | **1,547** (−42.4%) |
| Active (reachable) gates | 2,165 | 1,547 |
| AIG nodes (inversions free) | 1,627 | **986** |
| Model JSON | 213 KB | 128 KB |
| Test BER | 0.1727 | **0.1727** |
| Key entropy | 0.926 | 0.926 |

Breakdown: encoder 896 → 551 (9 deduped, 336 dead), decoder 1,792 → 996
(627 deduped, 169 dead). No gate folded to a constant; the model had already
been trained away from that.

The 42% figure is higher than a hand count suggests (403 duplicate pairs, 523
dead gates) because hash-consing is iterative: once two inverters merge, their
consumers often become identical too.

### Why it is exactly lossless

Every rewrite is a Boolean identity — `NAND(x,0)=1`, `NAND(1,x)=NOT x`,
`NAND(x,x)=NOT x`, plus "same inputs ⇒ same output". Equivalence is therefore
by construction; the tests are a check on the implementation, not the argument:

- Unit tests compare the original and simplified circuit **exhaustively over all
  2^n inputs** for `Universal`, `Full`, `NandOnly`, `XorAnd` and `XorOnly`
  random circuits, and check idempotence.
- `xorae simplify --verify <dataset>` compares keys *and* reconstructions for
  all 6,000 images.
- `--verify-random N` compares them on N pseudo-random 1,024-bit vectors, i.e.
  inputs that are nowhere near the training distribution (2,000 used here).

### The AIG caveat

41.9% of the reachable gates are `Nand(x, x)`, i.e. inverters, and only 627 of
the 907 are distinct wires. In an AND-Inverter Graph a complemented edge is
free, so those gates collapse into their consumers and the circuit is 986 AND
nodes. That is the standard way synthesis papers count size, but in CMOS an
inverter costs area and a wire carries one polarity, so 986 must not be quoted
as "986 NAND gates". Both numbers are reported, separately labelled.

This also exposes a training inefficiency: the wiring search spends 42% of its
gate budget implementing negation. See open thread 3 in AGENTS.md.

### Reproduce

```bash
cargo run --release -p xorae -- simplify \
  --model web/public/models/cats32-nand.json \
  --out runs/cats32-nand-simple.json \
  --verify data/cats32_threshold.bin --verify-random 2000
```

`train-nand` now runs the same pass automatically after polish, and prints both
the NAND gate count and the AIG node count.
## Checkpoint 4 (2026-10-30): colour at 2 bits per channel

RGB1 quantises each channel to one bit, so a pixel is one of 8 fully saturated
colours and muted browns and greys are unrepresentable. `ColorMode::Rgb2`
quantises each channel to 2 bits (levels 0/85/170/255, decision boundaries 42,
128 and 213), giving 64 colours.

Implementation: six bit planes `R_hi, R_lo, G_hi, G_lo, B_hi, B_lo` at
`size × size`, i.e. `channels = 6` with `bits_per_pixel = 1`. The trainer,
dataset and evaluator treat that as a tall image exactly like rgb1, so no
trainer changes were needed. Wired through `preprocess --rgb2`, the `XAE4`
mode byte, `Model::bits_to_rgb` and the wasm `preprocess` bridge.

Config: 16×16, 1,536 input bits, 192-bit key (8× compression), hidden
`[768, 384]`, 4,032 gates trained.

### Result: 6 paired seeds (0-4, 42)

| | rgb1 (8 colours) | rgb2 (64 colours) |
|---|---|---|
| Input bits | 768 | 1,536 |
| Key | 96 | 192 |
| NAND gates after simplify | 1,198 ± 21 | 2,289 ± 39 |
| AIG nodes | 760 | 1,406 |
| **Abs MAE vs original** | **0.3561 ± 0.0041** | **0.2362 ± 0.0023** |
| MAE vs own quantised target | 0.2211 ± 0.0088 | 0.2126 ± 0.0024 |
| PSNR | 6.47 dB | 9.63 dB |
| Pixel accuracy (all planes) | 55.4% | 19.5% |
| Bit BER | 0.2211 | 0.2885 |

Paired difference in absolute MAE: **−0.1199 ± 0.0037**, i.e. 2-bit colour cuts
reconstruction error by **33.7%** for 1.91× the gates.

### The metric trap that hid this

The first read of these runs said "no gain": bit BER got *worse* (0.2211 →
0.2885) and MAE against each model's own quantised target was flat (0.2211 →
0.2126, paired t = −2.6). Both are misleading, for different reasons:

- **Bit BER** weights every plane equally, so it punishes rgb2 for having twice
  as many planes, and it counts a `lo`-bit error (85 in 8-bit units) the same
  as a `hi`-bit error (170).
- **MAE against the quantised target** normalises against a different grid for
  each mode. A perfect rgb2 model scores 0 on its grid while being ~0.083 away
  from the original; a perfect rgb1 model is ~0.287 away. The grids are not
  comparable.

The honest metric is **MAE against the original, unquantised image**, which is
now printed by `eval`. Measured quantisation floors on this dataset: 0.2870 for
1 bit/channel, 0.0831 for 2 bits/channel — so rgb1's encoding loss alone
exceeds its total model error, and no amount of training could get it close to
the original.

### Against the published colour model

`web/public/models/cats16-rgb1.json` is a GA/XOR+AND model, 128-bit key, 577
active gates: grid MAE **0.1729** (much better than either NAND model) but
absolute MAE **0.3341** — worse than rgb2's 0.2362, because its palette cannot
represent the target. Two lessons: grid MAE flatters coarse palettes, and the
NAND-only trainer is markedly weaker on colour than on grayscale (577 gates
beats 1,198 at the same 768-bit input).

### Gate counts

All twelve trained models were run through `simplify --verify`, which also
re-verified exactness on 6,000 samples each (72,000 comparisons, all identical).
Gate counts above are post-simplification, so rgb1 and rgb2 are comparable.
## Checkpoint 5 (2026-10-30): can it fit at all? (objective experiments)

Motivated by two facts: (a) train BER ≈ val BER ≈ test BER on the shipped model
(0.1728 / 0.1710 / 0.1727), i.e. no overfitting gap; and (b) the wide model had
a *lower* train MSE but a *worse* BER, so the training objective and the
reported metric disagree. Both point at optimisation/representation rather than
data or gate budget.

### Diagnostic: memorise 100 images

`train-nand --train-limit 100` uses only 100 training images. Two runs, because
the first had far fewer gradient updates than a full run (100 samples / batch 64
= 1 batch per epoch):

| Run | Updates | Train BER on the 100 | Val (1,001) | Test (1,000) |
|---|---|---|---|---|
| 200 epochs | ~200 | 0.3180 | 0.3189 | 0.3214 |
| 12,000 epochs (≈ full-run updates) | ~12,000 | **0.1741** | 0.2058 | 0.2107 |

**The model cannot memorise 100 images any better than it reconstructs 4,000.**
100 images with a 128-bit key and 2,688 gates should be trivially memorisable
(the key alone can index 2^128 patterns), so this is a limit of the
representation/optimiser, not of data volume or key capacity. It also rules out
data augmentation and regularisation as fixes, and explains why the wide model
(same schedule, more gates) did not help.

Note the 12,000-epoch run also drove train MSE down (0.089) while train BER
stayed at 0.174 — the same MSE/BER divergence seen in the wide model.

### Experiment: straight-through binarisation

`--ste` hard-thresholds every activation in the forward pass and passes
gradients through unchanged (the standard BNN trick), so the trained network is
the exported circuit and the loss (MSE on hard bits) equals BER.

| Seed | Train BER | Val BER | Test BER | Gates |
|---|---|---|---|---|
| 0 | 0.3116 | 0.3116 | 0.3128 | 1,296 |
| 1 | 0.3272 | 0.3256 | 0.3270 | 1,233 |
| 2 | 0.3206 | 0.3179 | 0.3198 | 1,303 |
| baseline (0-4) | — | — | 0.1730 ± 0.0066 | 1,547 |

Clearly worse — roughly the all-ones bias level. The mechanism is visible in
the logs: at `gain = 1` the candidate softmax is near-uniform, so every gate's
`z ≈ 0.75` and hard-thresholds to 1. The network starts saturated and never
escapes. This reproduces the older "hard binarisation collapses" finding, and
shows that a straight-through gradient is not sufficient on its own when the
hard decision is made before the wiring has sharpened.

`--wire-ste` is the literature-faithful variant: keep activations soft but make
the forward pass use each gate's **argmax candidate wire** — the circuit that
will actually be exported — with gradients flowing through the softmax as
before. This removes the train/export wiring gap instead of the activation gap.

| Run | Train BER | Val BER | Test BER | Gates |
|---|---|---|---|---|
| seed 42 | 0.2159 | 0.2124 | 0.2164 | 1,549 |
| baseline seed 42 | 0.1728 | 0.1698 | 0.1722 | 1,547 |

Also worse, though not catastrophically: it plateaus by epoch ~75 and stops
improving. The annealing schedule turns out to be doing the discretisation work
better than hard-argmax training does.

### What this means

The exported-circuit gap is *not* the bottleneck: the wire-STE run's train BER
(0.2159) is essentially its test BER (0.2164), and the soft baseline's train
(0.1728) is essentially its test. Training and inference agree; the model is
simply not expressive or searchable enough at this gate budget.

So the remaining lever is capacity per gate, not objective alignment:

1. **Free inversions** (AIG edges): 42% of reachable gates are `Nand(x,x)`.
2. **Spatial weight sharing**, as in convolutional logic gate networks.
3. **Distillation** from a CNN teacher, to give smoother targets.
4. A margin/hinge loss on the bit decision instead of MSE, which keeps soft
   gradients but stops rewarding ambiguous outputs near 0.5.

STE appears in the literature (Gumbel LGNs, NeurIPS 2025) applied to *gate
selection* in a network whose activations are already binary by construction,
trained for classification with a handful of outputs. Neither of those
conditions holds here: we have 1,024 outputs and a 128-bit bottleneck, and the
wiring starts random.
## Checkpoint 6 (2026-10-30): the gate relaxation matters

The soft NAND used during training is `1 - u·v` (product t-norm). Every
relaxation is exact on hard 0/1 inputs, but they differ on soft activations,
and that difference turns out to be worth more than any scheduling change.

The motivation is a property of the *diagonal*: `NAND(x, x) = NOT x`, so a
diagonal gate should be an exact negation. Under product it is `1 - x²`, which
equals `NOT x` only at 0 and 1; under the Gödel t-norm `1 - min(u, v)` it is
exactly `1 - x` because `min` is idempotent. Since 30-40% of trained gates are
diagonal, product was systematically mis-scoring a third of the circuit.

Implemented as `--relaxation {product,min,lukasiewicz}` with subgradients
(`min`: `-1` on the smaller input, `0` on the larger; ties send the gradient to
`u` so it is never double-counted). Unit tests check hard-input agreement and
match the analytic gradients against finite differences.

### Paired result, seeds 0-9

| | product | min (Gödel) |
|---|---|---|
| Test BER | 0.1730 ± 0.0060 | **0.1641 ± 0.0093** |
| Paired difference | — | **−0.0089**, 95% CI [−0.0159, −0.0019] |
| Seeds won | — | 7 / 10 |
| t(9) | — | −2.88 (p ≈ 0.018) |
| NAND gates after simplify | 1,572 ± 52 | 1,843 ± 44 (+17%) |
| Inverter gates | 34.7% | **22.7%** |
| Real two-input nodes | 1,027 (65.3%) | **1,426 (77.3%)** |

Łukasiewicz (`1 - max(0, u+v-1)`) was rejected on one seed: test BER 0.2117,
consistent with its nilpotent, zero-gradient-when-`u+v <= 1` behaviour.

### Why this is a real effect and not seed luck

The first five seeds gave −0.0071 with p ≈ 0.08, which was not enough to claim
anything; five more pairs (seeds 5-9, both configs) took it to −0.0089 with the
95% CI excluding zero. The direction also matches a mechanism: min keeps 77% of
gates as real two-input logic versus 65% for product.

### Costs and caveats

- **+17% gates.** This is better BER for the same 2,688-gate training budget,
  not the same output size. If size is the headline, run the lossy gate-removal
  pass on the min model and compare on that frontier.
- **More variance.** sd rises from 0.0060 to 0.0093 across seeds.
- **The mechanism is measured indirectly.** The structural shift is real; "the
  diagonal explains it" is an explanation, not an experiment. Logging how often
  the argmax picks `a == b` under each relaxation would test it directly.
- The published demo artifact is still the product-relaxation model (0.1727 /
  1,547 gates). Re-exporting with min would gain ~0.003 BER on seed 42 at +21%
  gates, which is inside single-seed noise — not worth changing the artifact.

### Reproduce

```bash
cargo run --release -p xorae -- train-nand \
  --config configs/cats32_nand.toml \
  --data data/cats32_threshold.bin --out runs/nand32_godel --seed 0
```
## Checkpoint 7 (2026-10-30): size-vs-BER frontier (lossy gate removal)

`xorae prune` removes gates by bypassing each one to one of its inputs and
re-running dead-code elimination. Candidates are tried in ascending fanout
order; a removal is accepted if calibration BER rises by at most `--tolerance`,
and every acceptance restarts the scan because indices shift. Score is measured
on a train sample (`--calibrate`, default 512) and reported on test.

This is deliberately *lossy* — unlike `simplify`, which is exact by
construction. It is the "gate removal for size" thread.

### Product model, seed 42 (base 0.1722 at 1,547 gates)

| gates | test BER | AIG nodes |
|---|---|---|
| 1,547 | 0.1722 | 986 |
| 1,400 | 0.1749 | 910 |
| 1,300 | 0.1775 | 831 |
| 1,200 | 0.2047 | 756 |
| 1,100 | 0.2335 | 668 |
| 1,000 | 0.2486 | 591 |

Removing ~150 gates (10%) costs 0.003; the curve then steepens sharply below
1,200. With `--tolerance 0` the pass stalls at a local optimum (nothing is
accepted that worsens BER at all), so a small tolerance is needed to get a
frontier.

### Gödel vs product at equal size

Checkpoint 6 found Gödel `min` better at equal *training* budget. Here both are
pruned to a fixed gate count, 5 paired seeds, tolerance 0.0005:

| target | min (Gödel) | product | paired (min − product) |
|---|---|---|---|
| 1,400 gates | 0.2154 ± 0.0167 | **0.1811 ± 0.0057** | +0.0344 (t = 5.6), min wins 0/5 |
| 1,300 gates | 0.2401 ± 0.0151 | **0.1882 ± 0.0157** | +0.0519 (t = 7.5), min wins 0/5 |

**The relaxation moves the whole curve rather than dominating.** Gödel buys
its accuracy at equal training budget by keeping more real logic (77% vs 65% of
gates), but those extra gates are load-bearing: compress both to the same size
and product wins decisively.

Consequence: `configs/cats32_nand.toml` stays on `relaxation = "product"`.
Gödel remains the best option if the training budget is fixed and output size
is free, or if exact NOT gates matter for interpretability; it is the wrong
default for a project whose headline is "tiny circuit".

### Reproduce

```bash
cargo run --release -p xorae -- prune \
  --model web/public/models/cats32-nand.json \
  --data data/cats32_threshold.bin \
  --out runs/cats32-pruned-1400.json --target 1400 --tolerance 0.0005
```
## Checkpoint 8 (2026-10-30): is the key space generative?

The encoder maps an image to a `k`-bit key; the decoder maps it back. If the
set of keys the encoder produces can be modelled and sampled, the circuit is a
generator with a 128-bit latent and a generator needs no new machinery. If the
keys are an unstructured per-image codebook, a GAN on top would have nothing to
learn.

`xorae prior` fits two priors on the training keys and samples from both:

- **Bernoulli** — independent per-bit probability (128 parameters).
- **Autoregressive** — causal linear model `p(x_j) = sigmoid(b_j + sum_{i<j}
  w_ji x_i)`, i.e. a fully visible sigmoid belief net (16,512 parameters).

It then decodes the samples and reports: marginal MAE against the data's bit
marginals, mean pairwise Hamming diversity, a **round-trip** score (decode the
key, re-encode the image, compare keys — a proxy for "does this key lie on the
model's manifold"), and held-out **NLL in bits per key bit** for the priors.
Uniform and empirical keys bracket the results.

### MNIST, 32x32, 1,024 bits -> 128-bit key (1,287 gates after simplification)

| samples | marg MAE | diversity | round-trip | NLL (bits/bit) |
|---|---|---|---|---|
| uniform | 0.265 | 0.500 | 0.570 | 1.000 |
| empirical (real keys) | 0.045 | 0.304 | **0.801** | — |
| Bernoulli | 0.041 | 0.304 | 0.710 | 0.622 |
| autoregressive | 0.051 | 0.291 | **0.806** | **0.318** |

The autoregressive prior reproduces the empirical round-trip score (0.806 vs
0.801) and compresses the key distribution from 1.0 to **0.318 bits per bit** on
held-out keys, i.e. about 41 bits of entropy in a 128-bit code. Visually the
difference is stark: uniform keys decode to noise, AR-sampled keys decode to
recognisable digits, and they are as coherent as the decodes of real keys.

### Cats, 32x32 (the shipped model)

| samples | marg MAE | diversity | round-trip | NLL (bits/bit) |
|---|---|---|---|---|
| uniform | 0.145 | 0.500 | 0.721 | 1.000 |
| empirical | 0.068 | 0.476 | 0.916 | — |
| Bernoulli | 0.065 | 0.459 | 0.763 | 0.943 |
| autoregressive | 0.058 | 0.466 | 0.889 | 0.544 |

Same ordering, weaker effect: cat keys are less compressible (0.544 vs 0.318),
and the independent-Bernoulli prior is almost useless there (0.943) because the
structure is in the correlations, not the marginals.

### Conclusion

**The code space is structured and generative.** A 16k-parameter linear prior
over 128 bits is enough to sample keys whose decodes are (a) visually
recognisable on MNIST, and (b) statistically indistinguishable from real keys
by round-trip. A GAN on keys is therefore viable but not necessary for a first
generator: the AR prior already is one.

### Caveat + metric trap

MNIST is sparse (13.4% of bits are 1), so **BER is the wrong metric**: the model
scores 0.2254 while predicting all zeros scores 0.1366, even though the model
reconstructs recognisable digits (see `runs/mnist_render.png`). Judge sparse
tasks with IoU/F1 or balanced accuracy. Reconstruction quality is also capped by
the decoder's salt-and-pepper noise, which is present for real keys too, so
prior samples are no worse than the autoencoder itself.

### Reproduce

```bash
./scripts/fetch_mnist.sh data/mnist
cargo run --release -p xorae -- preprocess --mnist-dir data/mnist \
  --out data/mnist32.bin --size 32 --limit 12000
cargo run --release -p xorae -- train-nand --config configs/mnist32_nand.toml \
  --data data/mnist32.bin --out runs/mnist32
cargo run --release -p xorae -- prior --model runs/mnist32/model.json \
  --data data/mnist32.bin --out runs/prior_mnist --samples 32
```
## Checkpoint 9 (2026-10-30): labels, a classifier, and an all-NAND classifier

The autoencoder produces no labels, so "recognise the digit" needed a
classifier on top of the key. This adds three things:

1. **`XAE5` datasets carry labels** (`label_count u8` + `n` label bytes after
   the split flags). `XAE4` still loads with empty labels, and the writer only
   emits `XAE5` when labels exist, so every existing dataset is unchanged.
   `preprocess --mnist-dir` now reads `*-labels-idx1-ubyte`.
2. **`xorae classify`** trains multinomial logistic regression on either the
   128-bit key or the raw 1,024 input bits, and prints accuracy plus a
   confusion matrix.
3. **`xorae classify-nand`** trains a NAND circuit over the key bits:
   `128 -> hidden.. -> classes x group`, wiring chosen by an annealed softmax
   over candidate wires (same relaxation machinery as the autoencoder), class
   score = **sum of its output group** (a vote), loss = cross-entropy over the
   group sums. The argmax wiring is materialised as a literal `Circuit`,
   exact-simplified, and scored on the held-out split.

### Accuracy, MNIST 32x32 (10,000 test images)

| model | inputs | test accuracy |
|---|---|---|
| softmax regression | 1,024 raw pixels | 88.9% (train 98.6% — overfits) |
| softmax regression | **128-bit encoder key** | **82.0%** |
| NAND circuit, 300 epochs | 128-bit key | 68.0% |
| NAND circuit, 600 epochs | 128-bit key | **70.3%** |
| NAND circuit, distilled from the 82% teacher | 128-bit key | 67.8% |

Two findings:

- **The key is a good feature.** 82.0% from 128 bits against 88.9% from 1,024
  pixels means the 8x compression loses about 7 points of class information —
  the code is semantic, not just an image codebook.
- **A pure gate circuit gets to ~70% with 228 NAND gates**, and the vote
  read-out matters: with one output bit per class the same machinery stalled at
  21-37% and collapsed (394 -> 46 gates), while groups of 8 per class give a
  healthy prediction histogram and 228 active gates.

Distillation was a negative: soft targets from the 82% teacher scored 67.8%,
below one-hot. The circuit is capacity-limited, not label-limited, so softer
targets buy nothing at this size.

### Cost of the classifier

| | value |
|---|---|
| NAND gates after simplification | **228** |
| Logic depth | **3** |
| Max fanout | 7 (mean 1.7) |
| Inputs -> outputs | 128 -> 80 |

No multipliers, no weights, no memory: the parameters are the wiring. The
equivalent softmax model needs 1,280 MACs and 1.3k parameters. A full
"pixels -> key -> label" pipeline is encoder (~550 gates) + classifier (228),
depth ~6 — roughly 800 LUTs on an FPGA.

Caveats: the wiring is arbitrary, so an ASIC router may dominate area; and 70%
is 12 points behind a linear read-out on the same 128 bits. Improvements worth
trying: larger circuit, class weights, or training the encoder for
classification as well as reconstruction (the key is currently optimised for
reconstruction only).

### Reproduce

```bash
cargo run --release -p xorae -- classify \
  --model runs/mnist32/model.json --data data/mnist32.bin \
  --out runs/mnist_classifier.json
cargo run --release -p xorae -- classify-nand \
  --model runs/mnist32/model.json --data data/mnist32.bin \
  --out runs/mnist_nand_clf.json \
  --hidden 192,96 --epochs 600 --candidates 16 --group-size 8 --relaxation min
```
