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
| NAND circuit, 192,96 hidden / 8 per class | 128-bit key | 68.0% (300 ep) → **70.3%** (600 ep) |
| NAND circuit, 256,128 hidden / 16 per class | 128-bit key | 75.5% |
| NAND circuit, 384,192 hidden / 16 per class, 32 candidates | 128-bit key | **77.4%** |
| NAND circuit, distilled from the 82% teacher | 128-bit key | 67.8% |

### Circuit size buys accuracy

| output gates per class | gates after simplify | test accuracy |
|---|---|---|
| 1 | 46 (collapsed) | 36.9% |
| 8 | 228 | 70.3% |
| 16 | 370 | 75.5% |
| 16 (wider hidden, more candidates) | 392 | 77.4% |

The trend is monotone with size, and train accuracy tracks test closely
(79.4% vs 75.5% in the largest run), so the circuit is underfitting rather
than overfitting: more gates, not more epochs, is the lever. Probes plateaued
by epoch 100 in every run.

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

### The demo

`web/` has a second tab: **draw a digit**. A pointer-drawn stroke on a canvas is
normalised the MNIST way (ink bounding box, scale to 20x20, centre of mass,
28x28, nearest-upsample to 32x32), encoded by the MNIST autoencoder, and
classified by the 370-gate NAND circuit. The tab shows the predicted digit, the
per-class vote (0..16) and what the circuit actually sees.

Assets: `web/public/models/mnist32.json` (autoencoder),
`web/public/digits/classifier.json` (the circuit),
`web/public/digits/meta.json` (classes, group size, gates, accuracy),
`web/public/digits/sample.json` (one MNIST test digit for the self-test).

Verified headlessly with
`?tab=digits&selftest=1`: the built-in 7 is drawn, normalised, encoded and
classified as **7** (15/16 votes), zero console errors.

### Follow-up: full data, native 28×28, and skipping the key

Three changes were tested against the previous MNIST setup (32×32, 12k images,
128-bit key): native 28×28 (784 bits, 98-bit key), the full 48k training split,
and a classifier that reads raw pixels instead of the key.

| read-out | 32×32 / 12k | 28×28 / 48k |
|---|---|---|
| softmax on key bits | 82.0% (128 bits) | 82.5% (98 bits) |
| softmax on pixels | 88.9% (1,024) | **91.7%** (784) |
| **NAND circuit on key bits** | 77.4% (392 gates) | 75.2% (324 gates) |
| **NAND circuit on pixels** | — | **80.5%** (429 gates) |

Three conclusions:

- **The BER improvement was background.** Test BER fell from 0.2254 to 0.1704,
  but 86% of MNIST pixels are black, so that is mostly "predict black
  correctly". Reconstruction renders still show heavy stroke noise, and the
  key-fed circuit got *worse* (77.4 → 75.2). Use accuracy, not BER, on sparse
  data.
- **More data helps the pixels, not the code.** The pixel softmax gained 2.8
  points from the extra 36k images; the key read-out gained 0.5. Consistent
  with the 100-image memorisation failure: the encoder, not the data, is the
  limit.
- **Bypassing the key is worth ~5 points for a gate circuit**, and removes a
  whole stage: normalise the stroke, threshold to 784 bits, run 429 gates.
  That is now what the web demo does.

The gate search still trails the linear read-out by more than it does on keys
(11.2 vs 7.3 points), and only 429 of 928 trained gates survive simplification,
so utilisation remains the open problem.

## Checkpoint 10 (2026-10-31): is the target even legible?

The complaint that started this: after binarisation the 32x32 cats no longer
look like cats, so "does the reconstruction look right" cannot judge the model.
It is true. Rendering the grayscale original above its thresholded target makes
the point in one image -- a *perfect* circuit still emits blobs.

Three representations were compared on the same images.

| representation | bits | key | gates (after simplify) | BER | ones F1 | IoU | key entropy |
|---|---|---|---|---|---|---|---|
| threshold (shipped) | 1,024 | 128 | 1,547 | **0.1727** | **0.783** | 0.643 | **0.926** |
| Sobel edges, top 15% | 1,024 | 128 | 2,246 | 0.3078 | 0.198 | 0.110 | **0.291** |
| 4-level palette, 200 ep | 2,048 | 256 | 2,918 | 0.3174 | — | — | — |
| 4-level palette, 1,000 ep | 2,048 | 256 | 2,695 | 0.2886 | — | — | — |

**`--binarize edge` produces legible line drawings and cannot be trained.** The
best validation epoch is epoch 0, the random initialisation; 200 epochs never
beat it, so the exported model is the init plus polish. Positive weighting
(`--positive-weight` 6 and 100) demonstrably reaches the loss -- train MSE
0.15 -> 9.8 -- but changes nothing that survives selection, because selection
still picks epoch 0. Key entropy 0.926 -> 0.291 says the same thing: the
encoder gave up and the key stopped varying.

**BER hid the failure.** On a 15%-ones target, all-zero scores BER 0.15, so
0.31 reads as "worse but nearby" while the output is noise. `eval` now prints
**ones F1** and **ones IoU**, which is the honest pair for a sparse target.

**4-level grey is legible but the rate is too tight.** A 4-colour palette at
32x32 (`--palette 4`, 2,048 bits, `channels = 2`, 256-bit key) has clearly
recognisable targets, and it trains smoothly -- 0.4429 -> 0.2886 over 1,000
epochs, best at epoch 850 -- but never converges to anything structured.
Per *pixel* it carries 0.25 key bits against 0.125 for the shipped model, i.e.
twice the information rate at 1.7x the gates, and half of its 5,376 trained
gates die in simplification (2,695 survive).

Practical conclusions:

- The shipped model is not the weak link: at ones F1 0.783 it reproduces its
  target faithfully. The target is what looks wrong, and no model can fix that.
- Judge 1-bit reconstructions by F1 against the binarised target, and look at
  the grayscale original only for context.
- A visually faithful 32x32 generator needs more than 1 bit per pixel *and*
  more capacity per bit; 4-level at 8x compression is on the wrong side of the
  frontier. The untested step is a 512-bit key (4x compression) at the same
  bit count.

Incidental fixes: `--palette` panicked for any dataset with more than one image
(`from_color_parts` called `rgb_to_gray` on the whole buffer rather than per
row); `run_nand` now checks bit planes as `channels * bits_per_pixel`, which is
what lets a palette dataset train at all.

### Tolerance-aware scoring (the "off by one pixel" problem)

A per-pixel rule calls a perfect drawing shifted one pixel F1 0.00 and BER 0.30
-- indistinguishable from noise, and at 32x32 with 2 px strokes every edge is
one pixel from being "wrong". `crates/cli/src/edge.rs` therefore scores edges
the way the boundary-detection literature does: a two-pass (1, sqrt2) chamfer
distance transform of both images, then precision/recall with a radius, plus
the mean distance.

Calibration on the edge test split (400 images), which is the important part:

| prediction | F1@1.5 | chamfer (px) | ink |
|---|---|---|---|
| perfect target | 1.000 | 0.00 | 0.15 |
| target shifted 1 px | **1.000** | 0.44 | 0.15 |
| target dilated 1 px | 1.000 | 0.51 | 0.41 |
| **random ink at 23%** | **0.564** | 2.20 | 0.23 |
| all ones | 0.583 | 2.62 | 1.00 |
| all zeros | 0.000 | -- | 0.00 |

The radius does what it should for *structure*: a displaced or thickened
drawing is perfect. But at this density a 1.5 px radius covers about half the
canvas, so the score saturates -- random spray already reaches 0.56. The metric
is only readable against that chance level, so `eval` prints it:

| model | edge F1@1.5 | chance | skill over chance |
|---|---|---|---|
| threshold (shipped) | 0.889 | 0.695 | **+0.194** |
| Sobel edge, trained | 0.574 | 0.564 | **+0.010** |

So the edge model is 1% better than spraying ink at the same density: its
stroke *placement* carries essentially no information either, and the tolerant
metric agrees with the harsh one (ones F1 0.198) from a different direction.
The threshold model, by contrast, is 19 points above chance.

Lesson for any future tolerant metric here: never report F1@r without the
same-density random baseline, and prefer the smallest radius that still
forgives the cable's own quantisation (~1 px), because F1@r rises with r for
*any* predictor that inks enough.

## Checkpoint 11 (2026-10-31): full 24-bit RGB, and why it does not train

The colour work so far used quantised palettes (rgb1 = 8 colours, rgb2 = 64).
The question was whether the same circuit can be pointed at *unquantised*
24-bit colour, and if so how many gates that costs. The answer is that it
cannot, at any size this trainer reaches, and the reason is more interesting
than the gate count.

### What was built

`ColorMode::Rgb8`: 24 bit planes, `R7..R0, G7..G0, B7..B0`, most significant
bit first. `--rgb8` builds the dataset (mode byte 4, `channels = 24`),
`Model::bits_to_rgb` rebuilds exact bytes and `eval` reports absolute MAE
against the original. Verified on all 6,000 images: the 24 planes rebuild the
resized RGB bytes with **0 mismatches**, so unlike rgb1 (floor 0.287) and rgb2
(floor 0.083) there is no quantisation floor to hide behind. A perfect model
scores 0.

Also fixed here: `train-nand` validated `key_bits < image_size^2`, which is the
input width only for one plane. rgb2's 192-bit key passed by luck at 16x16;
rgb8's 768-bit key was rejected against 256 "inputs" although the tensor has
6,144. The check now multiplies by `channels`.

### The target is half noise, so BER is meaningless

24 planes from CIFAR are not 24 learnable bits. Measuring how well a pixel's
bit can be predicted from its four neighbours (majority vote) gives:

| plane nibble | 16x16 | 4x4 |
|---|---|---|
| high (bits 7-4) | 0.2556 | 0.3730 |
| low (bits 3-0) | 0.4820 | 0.4998 |
| all 24 planes | 0.3688 | 0.4364 |

The low nibble is at chance (0.50) at 4x4 and only 0.018 below it at 16x16:
**adjacent-pixel bits in the low nibble are statistically independent**, so no
circuit can predict them and any model pays ~0.5 BER on 12 of the 24 planes.
The bit-MSE floor says the same thing from the soft side: predicting every bit
as 0.5 costs 0.25, and a predictor that only looks at the four neighbours
reaches 0.2368 (16x16) and 0.2959 (4x4, worse than 0.5 because a single 4x4
neighbour is a bad guess). The true floor is lower -- the encoder sees the
whole image, so the high nibble is nearly free and only the 12 noise planes
cost their 0.25, i.e. about 0.125 -- but every local predictor lands near 0.24.

That has two consequences, both of which matter for reading the results:

- **A training MSE plateau near 0.23 is not evidence of learning.** The model
  sits at 0.2284 (16x16) and 0.2308 (4x4), about where a neighbour-only
  predictor lands. It is not saturated either (the entropy floor is nearer
  0.125), it is stuck: the gradient is dominated by bits that carry no
  information, which is exactly the regime where "lower MSE, worse BER"
  appears (Checkpoint 5's wide model).
- **BER is bounded below around 0.37 at 16x16**, so a model at 0.43 has
  learned almost nothing rather than "0.43 of something".

### Results

All runs use the standard schedule (200 epochs, seed 42), `product`
relaxation, and are judged by absolute MAE against the original bytes.

| run | bits | key | gates trained (after simplify) | test BER | abs MAE |
|---|---|---|---|---|---|
| 16x16, 8x | 6,144 | 768 | 16,128 (9,286) | 0.4331 | 0.2120 |
| 4x4, 8x | 384 | 48 | 1,008 (522) | 0.4425 | 0.2382 |
| 4x4, 8x + 6 cross-plane candidates | 384 | 48 | 1,008 (505) | 0.4457 | 0.2396 |
| 4x4, 4x | 384 | 96 | 1,056 (647) | 0.4373 | 0.2411 |
| 4x4, 2x | 384 | 192 | 1,728 (1,115) | **0.4110** | 0.2325 |
| 4x4, 8x, 8 NAND levels deep | 384 | 48 | 1,584 (755) | 0.4622 | 0.2764 |
| 4x4, 8x, overcomplete first hidden layer | 384 | 48 | 3,312 (1,150) | 0.4504 | **0.2319** |

The rate control is the one place where BER moves: 0.4425 -> 0.4110 going from an
8x to a 2x key. But it buys almost nothing in colour: abs MAE goes 0.2382 ->
0.2325. The improvement is in the low nibble, which the noise analysis above
says is unpredictable anyway, so a bigger key lets the model guess the noise
slightly better while the bytes, which are what a viewer sees, stay wrong. This
is the cleanest demonstration that BER is the wrong objective for this target.

Widening the first hidden layer past the input (384 -> 768 -> 384 -> 192 ->
96 -> 48, 3,312 gates, 5 levels per side, so the `2^depth` cap of 1,024
finally covers the whole 384-bit input) is the best byte score of the 4x4
family at **0.2319**, and still 0.065 behind the constant. Two thirds of the
added gates die in simplification (3,312 -> 1,150), so the trainer does not
spend the extra capacity. Expansion layers are supported now
(`derive_shapes_from` grows the longer axis by an integer factor and `window`
upsamples with overlapping 3x3 windows); the mechanism works, the target does
not.

Two reference predictions on the same splits, neither learned:

| predictor | 16x16 abs MAE | 4x4 abs MAE |
|---|---|---|
| all-zero | 0.4542 | 0.4537 |
| mean image (per-pixel mean of the training set) | **0.2031** | **0.1665** |
| mean of the four neighbouring pixels of the same image | 0.0485 | 0.0880 |

**Every configuration is behind a constant mean image**, and the neighbour mean
is 2-4x better than the bar the model fails to clear. Whatever the circuit
learns, it is not local smoothness, which is the one structure colour images
are full of. The contact sheets are colour noise with roughly the right
marginals, and pixel accuracy is 0.00% (no pixel in any test image is exactly
reconstructed, out of 1,000 images).

### Memorisation still fails

Checkpoint 5 established the diagnostic: can the model fit 100 images at all?
The colour answer is no, at either size, with the update count matched to a
full run:

| run | updates | train BER on the 100 |
|---|---|---|
| 16x16 24-bit, 12,000 epochs | ~18,700 | **0.3839** |
| 4x4 24-bit, 12,000 epochs | ~18,700 | **0.4130** |
| 4x4 24-bit + cross-plane, 12,000 epochs | ~18,700 | 0.3829 |
| 4x4 24-bit, 8 NAND levels deep, 12,000 epochs | ~18,700 | 0.4130 |
| 4x4 24-bit, overcomplete first layer, 12,000 epochs | ~18,700 | 0.3977 |
| 32x32 grayscale (Checkpoint 5) | ~12,000 | 0.1741 |

So it is not data volume, not the key size, and not the number of gates: 16,128
gates cannot memorise 100 colour images, while 2,688 gates memorise 100
grayscale ones to 0.1741. Doubling the depth does not change the memorisation
number at all (0.4130 either way) and makes held-out BER worse (0.4719 val vs
0.4600), so depth is not the missing piece on its own. Widening the first
hidden layer past the input does help a little -- 0.4130 -> 0.3977 for 3.3x the
gates (1,008 -> 3,312), the best fit of any colour run -- but 0.40 is still
noise, and only 1,150 of those 3,312 gates survive simplification.

### Why: depth and independence, not gate count

The wiring is fixed and hierarchical, and two properties of it are fatal here.
Both are now asserted in `halving_layers_never_leave_their_column_or_plane`.

**1. Each output bit depends on at most `2^d` input bits**, where `d` is the
total number of gate levels from input to output (induction: a gate's
dependency set is the union of its two inputs'). The 16x16 model has three
halving layers per side, so `d = 6` and an output bit sees at most **64 of
6,144** input bits. The 4x4 model has `d = 4`, so **16 of 384**. The deep 4x4
variant doubles `d` to 8 (256 inputs) and is worse on every measure -- BER
0.4622 vs 0.4425, abs MAE 0.2764 vs 0.2382, and the same memorisation number
(0.4130 on 100 images) -- so depth at this width is not a free win either; it
needs the objective and the data path to go with it.

**2. For planar colour the halving layers never mix columns or planes.**
Planar data is tall (`channels × size` rows by `size` columns), so
`derive_shapes_from` only ever shrinks the row axis and `window` returns `sy`
rows by **one** column. Every local candidate is therefore inside its own
column and its own colour plane, and the encoder is a stack of independent
per-plane column codecs; the only cross-plane mixing is the random `n_global`
long-range picks. Adding `n_channel` candidates (same spatial position, another
plane) is the targeted version of that, and it moved 4x4 by nothing
(0.2382 -> 0.2396).

Two corollaries follow. Colour information cannot flow between planes through
the local windows, so each plane is reconstructed from its own bits only; and
because a key bit sees one column, the circuit cannot move ink horizontally at
all. A same-resolution layer (target width == previous width) is the only
mechanism in the current design that reads across columns, which is why the
deep configuration exists; it lost here.

### Conclusion

24-bit RGB at 8x compression is **not achievable with this architecture** at
16x16 or 4x4, and the earlier estimate of ~37,500 gates for 32x32 was answering
the wrong question: adding gates to a depth-6 circuit that cannot fit 100
images would not help.

The three suspects were capacity, wiring, and the objective, and the 4x4 sweep
tests all three. Capacity: 3.3x the gates with an overcomplete first layer
(3,312 instead of 1,008) moves abs MAE from 0.2382 to 0.2319 and memorisation
from 0.4130 to 0.3977 -- real, tiny, and still noise. Wiring: cross-plane
candidates do nothing (0.2396); the depth cap is removed entirely at 5 levels
per side (`2^10 > 384` input bits) without a change in outcome. The objective is
what is left: bit-MSE over 24 planes is dominated by the 12 that are
unpredictable, so the gradient never pushes on the structure that a viewer sees,
and every run stalls near the error of a neighbour-only predictor.

That last point is the most actionable. `rgb2` (64 colours, 2 bits/channel)
already scores abs MAE 0.2362 on 16x16 with a 0.083 quantisation floor and is
shipped; at 24 bits/channel the extra six planes are mostly sensor/compression
noise, so the representation asks the circuit to predict randomness with 8x the
planes. The honest options are, in order:

1. Keep the quantised target and spend gates on structure (the existing track).
2. Train on the *high* nibble only (16 planes, 0.031 floor) and treat the low
   nibble as dither -- halves the bits and removes the noise from the loss.
3. If true 24-bit is required, the architecture needs cross-column depth (same
   resolution layers throughout, i.e. a real CNN-like stack) and a loss that
   is not bit-MSE, because bit-MSE cannot see structure through the noise.

### Reproduce

```bash
cargo run --release -p xorae -- preprocess --cifar-dir data/cifar-10-batches-bin \
  --out data/cats16_rgb8.bin --size 16 --rgb8
cargo run --release -p xorae -- preprocess --cifar-dir data/cifar-10-batches-bin \
  --out data/cats4_rgb8.bin --size 4 --rgb8

RAYON_NUM_THREADS=6 cargo run --release -p xorae -- train-nand \
  --config configs/cats16_rgb8.toml --data data/cats16_rgb8.bin --out runs/rgb8_stageb
RAYON_NUM_THREADS=6 cargo run --release -p xorae -- eval \
  --model runs/rgb8_stageb/model.json --data data/cats16_rgb8.bin --split test

# 4x4 variants: baseline, cross-plane candidates, rate controls, depth, width
for c in cats4_rgb8 cats4_rgb8_channel cats4_rgb8_k96 cats4_rgb8_k192 \
         cats4_rgb8_deep cats4_rgb8_wide; do
  RAYON_NUM_THREADS=2 cargo run --release -p xorae -- train-nand \
    --config configs/$c.toml --data data/cats4_rgb8.bin --out runs/$c
done

# memorisation probe (12,000 epochs matches a full run's update count)
RAYON_NUM_THREADS=2 cargo run --release -p xorae -- train-nand \
  --config configs/cats4_rgb8.toml --data data/cats4_rgb8.bin \
  --out runs/rgb8_4x4_fit100 --epochs 12000 --train-limit 100
```

## Checkpoint 12 (2026-10-31): choosing the wiring with a SAT solver

Every trainer in this repo searches wiring with gradients: softmax distributions
over candidate wires, annealed until one choice per gate wins. That is a local
search on a non-convex objective, so it can only ever report "this is the best I
found". A SAT solver answers the exact question instead: given the same fixed
hierarchical shape, does *any* choice of two candidate wires per gate reproduce
the training set? The answer is either a witness (a circuit, with a guarantee
that it fits) or a proof that no such circuit exists inside that template.

This checkpoint is about whether that is feasible, not about beating the
trainer. Two things came out of it: SAT is fast enough to use as an oracle, and
it settles an argument that the gradient runs could only hint at -- the
*hierarchical template itself* is the limiting factor, and on the smallest task
in the repository it stops working at four patterns.

### What was built

`scripts/sat_ae.py` -- a DIMACS encoder, solver driver, and simulator/verifier.
Pure Python, no library dependency. Tested with Kissat 4.0.1 and CaDiCaL 3.0
(both built from source; the 2025 SAT Competition main track was won by a Kissat
variant, AE-Kissat-MAB, so Kissat is the right default).

Encoding, for a shape given as layer widths:

- **Wiring.** Each gate's two inputs are indices into its candidate window, binary
  encoded. Candidate lists are padded to a power of two by repeating the last
  entry -- with an unused code the solver could leave a gate's inputs
  unconstrained and fabricate outputs without reading the key.
- **Gates.** `out = NAND(in_a, in_b)` as three Tseitin clauses, and each input
  selected by an if-then chain over the index codes.
- **Patterns.** The whole net is unrolled once per training pattern, so the CNF
  is a single circuit with a copy per example and the wiring variables are
  shared. Nothing else is free.
- **Loss.** `--slack K` makes each output equality soft: a selector per bit that
  is true *exactly* when the bit is wrong, bounded by a sequential-counter
  cardinality constraint, per pattern. So `slack 2` means "at most two wrong
  bits in each image", and one bad pattern cannot absorb the budget for the
  others.
- **Modes.** `full` (autoencoder), `encoder` (can the key separate the
  patterns?), `decoder` (given distinct free keys, can the decoder draw the
  patterns?), `functions` (how many functions of the key can the decoder compute
  at all?), `selftest`, `brute`.
- **Export.** `--export DIR` writes the witness as two `Circuit` JSON files
  (`{n_inputs, gates: [{op: "nand", a, b}], outputs}`) plus `meta.json`, which is
  what the demo's *solver's wiring* tab loads. The exporter re-simulates the
  written JSON on every glyph and refuses to save a circuit whose trained glyphs
  do not round-trip, so the browser and the solver cannot drift apart.

The task is 8x8 bitmaps of the letters A-H (64 bits each), in the trainer's
template: `64 -> 32 -> 16 -> 3 -> 16 -> 32 -> 64`, 163 NAND gates, 3-bit key.
That is 1,139 variables and 3,891 clauses for one pattern; 4,562 and 26,578 for
all eight.

### Validation before any claim

An encoder for this problem is easy to make subtly wrong -- and a wrongly
over-constrained CNF produces confident, false UNSATs. Four checks:

1. **Round-trip selftest.** Generate a random wiring, let *it* define the
   targets, then ask the solver for a wiring. A witness exists by construction,
   so the instance must come back SAT and the model must reproduce the targets.
   Passes at 1, 2, 3, 4 and 8 patterns.
2. **Brute force.** On tiny shapes every wiring can be enumerated. `[2,2,4]`
   (6 gates, 4,096 wirings) and `[2,4,4]` (8 gates) agree with the solver on
   15/15 random instances, both SAT and UNSAT. This is the check that rules out
   a silently over-constrained encoding.
3. **Encoder alone.** Asked only to put the patterns on distinct keys, a 3-bit
   key separates all eight glyphs (exactly at capacity, 8 codes for 8 patterns).
   So an UNSAT is never the encoder's fault at these sizes.
4. **Two solvers.** Kissat and CaDiCaL agree on every CNF that was kept.

Six real bugs were found this way, which is the argument for the checks:

- **The soft-equality clause was one-directional.** `sel -> (out = target)` with
  no converse lets the solver leave every selector false, so the cardinality
  bound never binds. Every slack run reported BER 0.4844 with 31 wrong bits per
  pattern at `slack 1`.
- **And then the converse was inverted.** Writing `sel <-> (out = target)`
  counts *matches*, so `slack 4` demanded "at most four **correct** bits" and
  returned a model with 60 wrong bits in each image. The correct form is
  `sel <-> (out != target)`. The runner now asserts that no pattern exceeds its
  budget, and that assertion is what caught it -- the first version of this
  checkpoint had a slack table that was wrong in both directions because of
  this bug.
- **`separate()` indexed the wrong layer** (the 64-bit output instead of the
  3-bit key), so the "can the encoder separate" question was being asked of the
  decoder's output.
- **The decoder-only test pinned key `000`.** The decoder has no bias inputs, so
  a constant key constant-folds every gate: `000` always decodes to all-ones.
  Keys have to be free-but-distinct, and *which* codes are usable is part of the
  problem.
- **The XOR parity clauses forbade the wrong assignment.** When `--gate xor3` was
  added, each of the eight clauses carried the input literals `var if bit else
  -var`, which is falsified at `1 - bit`, so the clause excluded the assignment
  that *satisfies* the parity instead of the one that violates it. Round-trip
  selftest failed immediately -- a witness exists by construction, so any UNSAT
  is the encoder's fault -- and the fix is to negate that literal pattern.
- **The `nandinv` NAND clauses were left unguarded.** Once a gate had an `inv`
  bit, the three NAND clauses had to be disabled when it was set, and they were
  not: every gate was forced to be NAND *and* AND at once, so the one-pattern
  selftest came back UNSAT on an instance that is satisfiable by construction.
  The guards are `[inv]` for the NAND clauses and `[-inv]` for the AND ones,
  the same pattern `mix` uses for its type bit.

`--mode selftest` now also runs a slack case: it corrupts a witness's target by
exactly `slack` bits and requires the instance to be SAT *and* every pattern's
error to stay inside the budget. An encoding that counts matches, or does not
count at all, fails that test.

### Exact fit stops at three glyphs

| glyphs | result | time |
|---|---|---|
| 1 | SAT, 0 wrong bits | 0.1 s |
| 2 | SAT, 0 wrong bits | 0.1 s |
| 3 | SAT, 0 wrong bits | 0.1 s |
| 4 | **UNSAT** | 0.1 s |
| 5, 6, 8 | **UNSAT** | 0.1 s |

0.1 s is not a timeout, it is the full proof. Widening the key does not help --
4, 5, 6 and 8-bit keys are all UNSAT at eight glyphs in 0.1 s each, and 4 bits
is already UNSAT at four. Nor does changing the decoder shape:

| decoder | 8 glyphs | time |
|---|---|---|
| `3→16→32→64` (the trainer's) | UNSAT | 0.1 s |
| `3→16→16→16→16→16→32→64` (4 extra mixing layers) | UNSAT | 0.3 s |
| `3→16→64→64` (wider, same-size last layer) | timeout | 330 s |
| `3→16→16→64→64` (mixing then wide) | UNSAT | 258 s |
| `3→32→64→64` (wide penultimate) | UNSAT | 19.8 s |

Same-resolution layers are the only *fully mixing* ones in this scheme (a
same-size layer's candidate window is the whole previous layer), so the last
three rows are the strongest tests available inside the template, and none of
them fits eight glyphs. The encoder is not the constraint either -- it separates
all eight.

### Why four glyphs is impossible: the decoder is 16 functions wide

My first explanation was wrong, and how it was caught is worth recording. From
`out = (u∧v) ∨ (s∧t)` I argued that no output bit can be true at exactly one key
code. That is false: the argument is about all eight codes, but only four are in
use, and a term covering two codes of which one is unused restricts to a
singleton. Enumerating the reachable functions says a singleton *is* achievable
on 106 of the 280 (code-subset, code) pairs.

The real bound is stronger and easier to state. Ask the solver, for each of the
256 functions of the three key bits, whether this decoder can compute it (all 64
output bits set to that function's truth table on all eight codes):

| decoder | functions of the key it can compute |
|---|---|
| `3→16→32→64` (the trainer's) | **16 of 256** |
| `3→16→16→16→16→16→32→64` | 18 of 256 |

An independent enumeration of every function reachable by a chain of
fully-mixing NAND layers gives the same ceiling: 6 at depth 1, 15 at depth 2, 17
at depth 3, 18 at depth 4, and nothing new at any greater depth. So the decoder
is not a universal function of its key at all -- it can draw 16 of the 256
possible functions of three bits -- and adding depth does not fix it, which is
exactly what the depth sweep measured.

That also explains the shape of the frontier. With three glyphs the output
function is pinned at only three of the eight codes, and 15 of the 16 available
functions agree on any three points (they all have to); with four the constraints
separate them and almost no assignment works.

Almost, not none. Running every one of the 70 four-glyph subsets (the `--glyphs`
flag) instead of only the prefix: **1 of 70 is SAT** -- `CEFG`, in 0.1 s, with
keys `101 010 011 100` -- and the other 69 are UNSAT in ~0.1 s. So the binding
constraint is still the shape, but "four glyphs is impossible" was a conclusion
about the four-glyph *prefix* and is too strong by one subset in seventy. It
also matters that the working four codes lie in a two-dimensional affine
subspace, so the decoder is only being asked to separate four codes inside
`{001, 010, 101, 110}` -- the same trick the singleton analysis above describes,
where a function that is defined on all eight codes restricts to something
simpler on four.

This is the useful part of the result. The gradient trainer reports "BER 0.224
after 2,000 epochs" and there is no way to tell whether the wiring, the
optimiser or the template is at fault. SAT says: the template, and it says so
with a number -- 16 functions, not 256 -- rather than with a vibe.

### Memorisation, not generalisation

The three-glyph circuit that fits exactly was then shown the other five glyphs.
Three of them decode to a trained image bit-for-bit (D to B, G to C, H to A) --
a lookup table. The other two fall on code `111`, which nothing was trained on,
so they decode to whatever that code happens to mean, 24 bits from the nearest
trained glyph:

| held-out glyph | D | E | F | G | H |
|---|---|---|---|---|---|
| encoder's code | 011 | 111 | 111 | 110 | 101 |
| decodes to a trained glyph | B | no | no | C | A |
| BER vs itself | 0.172 | 0.484 | 0.391 | 0.078 | 0.313 |

Mean over the five is 0.288. Encoding all eight glyphs therefore uses four of
the eight codes (011, 101, 110, 111) even though only three were trained. An
exact SAT fit says nothing about unseen data, which is what one should expect
from a key with fewer codes than the test set has patterns; it is worth stating
only because "the solver found a circuit that fits" invites the opposite
reading.

This is what the demo's **solver's wiring** tab shows: the eight glyphs with the
code the encoder gives each, the eight key codes decoded side by side, and three
clickable key bits. The four codes nothing asked for still decode to definite
images -- all-ones for `000`, all-zeros for `111`, and blobs in between -- which
is a more direct demonstration of the 16-of-256 limit than the table above.

### Slack: what a little loss buys

Minimum wrong bits per pattern, per glyph count (`--slack`), full autoencoder,
3-bit key:

| glyphs | slack 0 | 1 | 2 | 3 | 4 | 6 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|
| 4 | UNSAT 0.1 s | UNSAT 0.8 s | UNSAT 3.9 s | **SAT** 5.7 s | SAT 0.5 s | SAT 0.6 s | SAT 1.4 s | — |
| 8 | UNSAT 0.3 s | — | — | — | UNSAT 91 s | timeout 361 s | **SAT** 291 s | SAT 268 s |

For eight glyphs the threshold is slack 8, so the minimum error is somewhere in
6-8 wrong bits per pattern: slack 8 is satisfiable (BER 0.1172, 7.5 bits per
pattern) and slack 4 is not, while slack 6 did not finish in six minutes. That
is a wide bracket, and it is honest to leave it as one. (Slack 9 also comes back
SAT, which is what "at most 9" should do if 8 is feasible; it adds nothing but
one more consistency check.)

A 4-bit key does not move the frontier: with `64→32→16→4→16→32→64`, slack 4 and
6 both time out at ten minutes and slack 8 comes back SAT in 24.5 s with BER
0.1172 — the same solution quality, and faster, because a wider key gives the
decoder more room rather than more power. Consistent with the function count:
the limit is the shape of the decoder, not how many bits the key has.

A slack bound is a *feasibility* test, not an optimiser, and the numbers say so:
slack 10 returned BER 0.1504, worse than slack 8, because nothing asked for the
minimum. Finding the true optimum is a MaxSAT problem (RC2 or a linear-search
counter with a bound tightening on every model); the decision problem here only
gives upper and lower brackets.

The time curve is also not monotone in the obvious direction -- for four glyphs,
slack 0 takes 0.1 s and slack 2 takes 3.9 s -- because a looser bound is a
bigger search problem *and* a weaker contradiction.

### A three-pin XOR, with the bias solved

The next question was whether a different gate helps: two pins wired to candidate
wires and a third pin a per-gate bias bit that the solver also chooses, so every
gate is either XOR or XNOR. That is `--gate xor3`; `--gate xor2` is the same gate
without the bias, as a control. The bias is one variable per gate shared across
patterns -- part of the circuit, not part of the data -- and each gate-pattern
costs eight parity clauses.

The first version of that encoding was wrong, and `--mode selftest` caught it
before any result was quoted: the clause literals for the inputs were inverted,
so each of the eight clauses forbade the one assignment that *satisfies* the
parity instead of the one that violates it. A random wiring stopped being a
witness, which is exactly what the "a witness exists by construction" test is
for. `--gate xor2` and `--gate xor3` now both pass selftest at 1, 2, 3, 4 and 8
patterns plus the two slack cases.

**The gate is strictly weaker, and the amount is measurable.** Asking the
solver, for each of the 256 functions of a three-bit key, whether this decoder
can compute it at all:

| decoder gate | functions of the key | which ones |
|---|---|---|
| NAND | **16 / 256** | the trainer's 16 (see above) |
| XOR + bias pin | **8 / 256** | `00 ff 3c c3 5a a5 66 99` |
| XOR, two pins | **4 / 256** | `00 3c 5a 66` |

The eight are exactly `{0, 1, k0⊕k1, k0⊕k2, k1⊕k2}` and their complements, and
the four are the same set without the complements. That is not a coincidence, it
is the whole mechanism. The decoder's first layer reads only the three key bits,
so a gate there can compute `k_i ⊕ k_j` (plus the bias if it has one). It
cannot compute a single `k_i`, because that needs a XOR with a *constant*, and
although two identical input choices do produce a constant, they produce it as
an output of that same layer -- it is not available to the layer that reads the
key. So the first layer lands in the three-dimensional space
`S = span{1, k0⊕k1, k0⊕k2, k1⊕k2}`, `S` is closed under XOR and complement, and
induction gives: every wire at every depth stays inside `S`. Eight functions,
whatever the depth and whatever the width. A simulation check agrees -- 60
random wirings of an eight-layer decoder never leave the set.

So the bias pin is worth exactly one doubling (4 → 8): it buys complements.
The other doubling needs a *constant wire at the input layer* rather than a bias
per gate. Driving a fourth decoder input with a constant 0 and re-running the
same random-wiring test reaches all 16 affine functions, `k0`, `k1` and `k2`
included. That is the affine class in full -- which is still only NAND's 16, so
even the repaired version of this gate could only tie.

**The exact frontier follows the function count.**

| glyphs | NAND | XOR + bias | XOR (no bias) | NAND **or** XOR per gate |
|---|---|---|---|---|
| 1 | SAT | SAT | SAT | SAT |
| 2 | SAT | SAT | SAT | SAT |
| 3 | SAT | SAT | **UNSAT** | SAT |
| 4 | UNSAT | UNSAT | UNSAT | **SAT, 0.8 s** |
| 8 | UNSAT | UNSAT | UNSAT | not finished |

The bias pin is worth exactly one glyph: without it the template cannot fit
three.

The last column is the control that decides how to read the whole table, and it
is the one result here that changes an earlier conclusion. `--gate mix` gives
every gate a free type bit -- `NAND(a,b)` or `XOR(a,b) XOR bias` -- and asks the
same question. Three things happen at once:

- the decoder's function count jumps from 16 (NAND) and 8 (XOR+bias) to
  **134 of 256**;
- the four-glyph prefix, UNSAT under pure NAND, becomes **SAT in 0.8 s**;
- and it does so without any change to the shape or the slack.

So the "16 of 256" ceiling is mostly the *gate*, not the template. That is worth
stating plainly because the section above says the opposite: with one type bit
per gate, the same `3→16→32→64` decoder computes eight times as many functions
and fits a glyph set that NAND cannot. What the shape research established is
that *NAND-only* walls are hard -- the depth and the window still bound what a
type bit can do (134, not 256) -- but the wall is not a property of the template
alone. This is the same lesson as the trainer's "learnable gate types" negative
result (BER 0.2523): gate type is a real axis, and the gradient trainer failed
on it for optimisation reasons, not because the axis is empty.

**Slack: the affine classes lose the approximate regime too.**

Minimum wrong bits per pattern at which the instance is satisfiable:

| glyphs | gate | min slack | time | BER |
|---|---|---|---|---|
| 4 | NAND | **3** | 5.7 s | 0.0469 |
| 4 | XOR + bias | **6** | 19.7 s (5 is UNSAT, 420 s) | 0.0859 |
| 4 | XOR, no bias | > 2 | 2 UNSAT, 194 s | — |
| 8 | NAND | **8** | 291 s | 0.1172 |
| 8 | XOR + bias | 12 | 26.7 s (8 timed out, 780 s) | 0.1777 |

Two things worth recording. First, the exact four-glyph row is data-dependent at
the margin, which corrects a sentence further up: over all 70 four-glyph subsets,
**NAND fits exactly one** (CEFG, keys `101 010 011 100`, 0.1 s) and
**XOR-with-bias fits none**. "Four glyphs is impossible" was a conclusion drawn
from the four-glyph prefix; the honest version is "one subset in seventy".

Second, the slack gap is not small: the NAND decoder at four glyphs can be three
bits wrong per glyph where the XOR one needs six, and at eight glyphs it reaches
BER 0.1172 where the XOR one needs slack 12 for 0.1777 -- worse than the NAND
number at less than half the budget. A gate that can only draw 8 of 256 functions
is not merely a worse sampler of circuits; it is a smaller function class, and
the solver says so in both regimes. The slack-5 UNSAT took 420 s against a 420 s
timeout, so it is a proof that only just landed; slack 6 SAT is the load-bearing
number.

Verdict: NAND stays, and XOR does not. The bias pin is a real improvement on
plain XOR -- one more glyph, twice the function space, and it is what makes the
gate "EQ or NEQ" rather than a pure parity tree -- but it is a strictly smaller
class than the one the project already uses, so there is nothing here to ship.
The axis it opens, gate *type* chosen by the solver, is worth keeping: see `mix`
above, and the section below for the one variant that does beat NAND.

### One inversion bit per gate: NAND or AND

The gate change that pays is not a new gate. Keep NAND, add one bit per gate
that inverts its output, so a gate is `NAND(a,b)` or `AND(a,b)`: `--gate
nandinv`.

The prediction is available before any measurement, and it is about *where* the
inversion is spent. `AND(x,y) = NOT NAND(x,y)`, so this adds nothing to the
function class -- a NAND-only net can compute AND, it just has to spend a gate in
the next layer to invert the result. The last layer has no next layer: it can
only produce `NAND(u,v)`, and `AND(u,v)` is that thing complemented. So every
output bit that wants the complement of a two-wire AND had nowhere to get it,
which is exactly what the depth-3 decoder is short of.

Measured, on the same 163-gate shape:

- decoder functions of a 3-bit key: **58 of 256** (NAND 16, XOR+bias 8);
- exact fit for **1-6 glyphs** (NAND stops at 3; six takes 6.1 s). Every subset
  of six or fewer fits: 70/70 at four glyphs, 56/56 at five, 28/28 at six, and
  the seven-glyph prefix is UNSAT in 43.8 s where NAND cannot fit four of the
  prefix at all. The enumeration is `--mode subsets`, so these are counts over
  all subsets rather than samples of the prefix;
- the same 70 four-glyph subsets that NAND fits **1 of** fit **70 of** here;
- at eight glyphs the minimum slack is **1 wrong bit per pattern**, BER
  **0.0156** -- 8 wrong bits out of 512 -- found in 485 s.

The last line is the result. NAND's minimum at eight glyphs is slack 8, its
witness there has 60 wrong bits, and every smaller slack is UNSAT. So the same
wiring budget produces 8 wrong bits instead of 60, 7.5x less error, and it is
not a tuning difference: one circuit can produce the complement its output layer
needs and the other cannot. The 58-of-256 count is the same fact in one number.

Frontier, exact fit (slack 0):

| glyphs | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| NAND | SAT | SAT | SAT | UNSAT\* | UNSAT | UNSAT | UNSAT | UNSAT |
| NAND or AND | SAT | SAT | SAT | SAT | SAT, 1.2 s | SAT, 6.1 s | UNSAT, 43.8 s | UNSAT, 20.4 s |

\* the four-glyph *prefix* only: CEFG does fit, which is the one subset of 70.

Slack at eight glyphs (minimum wrong bits per pattern; BER is the witness found
at that bound):

| slack | 1 | 2 | 3 | 4 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| NAND | UNSAT | UNSAT | UNSAT | UNSAT | UNSAT | UNSAT | SAT, 291 s, 0.1172 |
| NAND or AND | SAT, 485 s, **0.0156** | SAT, 418 s, 0.0312 | SAT, 31 s, 0.0430 | SAT, 2.9 s, 0.0605 | SAT, 15 s, 0.0898 | SAT, 4.4 s, 0.1016 | SAT, 2.1 s, 0.1172 |

One methodological caution, because it changes how to read that table. `--slack`
is a feasibility bound, not an objective: Kissat returns the first witness inside
the budget, so a run's BER is not monotone in the budget and must not be read as
"the best at that slack". The `nandinv` run at slack 8 returns 0.1172, the same
number the NAND witness has at its own minimum, even though the same gate at
slack 4 found 0.0605. The looser CNF simply admits worse witnesses and the solver
stopped at one. The load-bearing numbers are the *minimum* slack and the BER
found there, where the bound is tight; the true minimum error needs MaxSAT.

Cost, in the terms of the next section: an AND cell is a NAND plus an inverter,
so this is not free. The exported six-glyph witness is 78 AND and 85 NAND gates,
**808 transistors against 652** for the all-NAND three-glyph witness, +24%. That
is the trade the demo now shows -- 24% more silicon, three more glyphs exact, and
a seventh of the error at eight -- and it is the first gate-type result in this
checkpoint that is worth shipping rather than recording.

### What a gate costs in transistors

XOR being "worse per gate" understates it once the unit is a device. Static-CMOS
counts are 4 transistors for NAND2, 2 for an inverter, 12 for XOR2 (a standard
cell; pass-transistor designs do better but not by much):

| circuit | gates | transistors | breakdown |
|---|---|---|---|
| `cats32-nand` | 1,547 | **5,066** | 986 four-device gates + 561 inverters |
| `cats16-rgb2` | 2,345 | 7,662 | 1,486 + 859 |
| `mnist32` | 1,287 | 4,094 | 760 + 527 |
| glyph witness (163 gates) | 163 | 492 | 83 + 80 |
| same 163-gate shape, all XOR + bias | 163 | **1,956** | 3.0x the silicon |

The 986 in the first row is not a coincidence and it resolves a claim the README
makes: the "1,547 literal NAND gates" and the "986 AIG nodes" are not two
estimates of different things. 986 is exactly the number of gates that are *not*
`Nand(x,x)`, i.e. the same circuit counted with its 561 inverters removed. An
AIG counts real two-input nodes and makes inversion free, so the AIG number is
the gate count after deleting every inverter -- and at 4T each, those inverters
are 1,122 of the model's 5,066 devices, 22% of the silicon spent on negation.

Two consequences for the gate-type thread. A static CMOS gate computes a
*monotone* function of its inputs in the pull-down network and its complement in
the pull-up, which is why gates invert and why the inverters are there at all;
and the device cost is what makes a 3-pin XOR a bad trade (12T for half the
function class of a 4T NAND). A richer cell library -- pass-transistor XOR,
AOI/OAI complex gates, muxes -- is the realistic version of the same idea, and
it is testable in this harness by adding gate types: the `mix` result above is
the first point on that curve.

### What the gradient trainer does on the same task

Feasibility is not the same as findability, and the two are worth separating. I
wrote the same eight glyphs out as an `XAE4` dataset
(`scripts/make_glyph_dataset.py`, 4 copies each so the trainer has a val split)
and trained the trainer's own config on them (`configs/glyphs8_nand.toml`,
`n_local = 9`, `n_global = 3`, product relaxation, 12,000 epochs — the default
200 is far too few at 64 input bits):

| glyphs | SAT says | `train-nand` | gates after simplify |
|---|---|---|---|
| 3 | exact fit exists, 0.1 s | BER 0.0365 (2.3 bits of 64) | 73 |
| 8 | infeasible; slack 8 → 0.1172 | BER 0.1426 (9.1 bits of 64) | 91 |

Three readings:

- **The trainer finds the feasible case but does not solve it.** An exact fit is
  provably there and 12,000 epochs reach 0.0365, not 0. The gap is small in
  pixels and infinite in the sense that matters: one glyph is still wrong.
- **At 2,000 epochs it is nowhere near** (BER 0.224 on the same 3 glyphs), so
  the earlier "the trainer cannot fit even 3 glyphs" reading was just an
  under-trained run. Feasibility claims about this trainer need a long schedule.
- **On the infeasible case the trainer is close to the SAT bound** — 0.1426 vs
  0.1172 — which is the expected result if the deficit is the template rather
  than the optimiser. The trainer is finding a good circuit in a space that does
  not contain a perfect one.

Caveat, because it matters for the comparison: the trainer does not search the
SAT window. `wire_candidates` draws `n_local` random picks inside `window(...)`
plus `n_global` long-range picks, so its space is a *sampled* subset of the
solver's. A SAT witness is therefore not necessarily reachable by the trainer,
and these numbers are a sanity check on the feasibility claim, not a race
between the two methods.

### The shipped model has the same shape problem

The 16-of-256 count cannot be enumerated for a 128-bit key (there are
2^(2^128) functions to ask about), but the *cone* can be computed exactly from
the exported JSON: the set of key bits an output bit can possibly depend on.
That is an upper bound on what it can express, and the same bound that made the
glyph decoder 16-wide.

| model | encoder inputs read | decoder key bits live | cone per output (median / max) | outputs seeing ≤1 key bit |
|---|---|---|---|---|
| `cats32-nand` | **362 / 1,024** | 102 / 128 | 2 / 7 | 169 |
| `cats16-rgb2` | **519 / 1,536** | 181 / 192 | 2 / 7 | 373 |
| `mnist32` | **222 / 1,024** | 113 / 128 | 2 / 7 | 233 |
| the glyph witness (3-bit key) | 10 / 64 | 3 / 3 | 3 / 3 | 10 |

Two numbers here are not about the key at all:

- **The encoder does not read most of the image.** 362 of 1,024 pixels can
  affect the key in the shipped model -- 65% of the input is invisible, and on
  MNIST it is 78%. The wiring search spends its gates on negation and on a few
  strong inputs (`Nand(x,x)` is 42% of the gates), so the rest of the picture
  never reaches the bottleneck.
- **The decoder is local in the key too.** With three decoder levels the cone is
  capped at `2^3 = 8` key bits; the measured maximum is 7 and the median is 2.
  169 output bits depend on exactly one key bit, so at those pixels the decoder
  is a copy or a negation, no matter what else the key says.

This is consistent with the checkpoint 11 conclusion and with the ruled-out
"more capacity" rows: adding gates to this template adds *depth-limited* gates.
It is also the strongest argument so far for the two structural fixes on the
open list — free inversions (so the budget is not half spent on NOT) and
same-resolution layers (so a window can read across the picture instead of
inside one column).

The caveat cuts both ways: cone size is necessary, not sufficient. The glyph
decoder sees all 3 key bits and still computes only 16 of 256 functions, so a
large cone does not imply expressiveness — but a cone of 1 certainly does cap
it.

### What this establishes

- **SAT is a usable oracle at this scale.** 163 gates, 8 patterns, 27k clauses,
  exact answer in 0.1 s. Nothing here is near a solver limit; the interesting
  limits are shape, not size.
- **The template is the binding constraint, and it breaks early.** Three glyphs
  out of eight under NAND, on the smallest task in the repo. That is the same
  hierarchical wiring (local windows, 2^depth dependency cap) the 32x32 model
  uses, and the measured reason is that the NAND decoder can compute only 16 of
  the 256 functions of its key.
- **But "template" was too coarse: it is template *and* gate.** One free type
  bit per gate (NAND or XOR+bias) takes the same decoder from 16 functions to
  134 and from "four glyphs UNSAT" to "four glyphs exact in 0.8 s". So NAND-only
  is where the wall is hardest, and the wall is not a property of the wiring
  alone -- which is a correction to this checkpoint, not a footnote.
- **The useful direction on the gate axis was inversion, not XOR.** One bit per
  gate that inverts its own output (`--gate nandinv`) takes the decoder from 16
  functions of a 3-bit key to **58**, the exact frontier from three glyphs to
  **six**, the 70 four-glyph subsets from 1 to **70**, and the eight-glyph error
  from 60 wrong bits to **8** -- for 24% more transistors (AND is NAND plus an
  inverter). The reason is structural rather than lucky: the last layer cannot
  pay for an inversion, so a gate that can is worth more than a gate that
  computes a different function. That is also the SAT-side evidence for open
  thread 6 in `AGENTS.md`: the shipped model spends 42% of its gates on
  `Nand(x,x)`.
- **Infeasibility is cheap to prove and expensive to guess.** The gradient
  trainer's failure on a shape looks identical to an optimiser problem; the SAT
  answer distinguishes them in a tenth of a second, and slack turns it into a
  number.
- **A soft-equality encoding is easy to get wrong in three different ways**, all
  of which still return "SAT". The `selftest`/`brute`/bound-assertion checks are
  the only reason the numbers above can be trusted.

Untested, and the obvious next questions: what shape *is* feasible for eight
glyphs (the function-count measurement above says it needs either more key bits
or same-resolution mixing in the decoder -- the encoder is general over layer
widths, so this is a shape search, not a solver problem); what the `mix` gate
does at eight glyphs (300 s timeout, so it is an open number, not a negative); whether a two-or-three-type gate library -- the
transistor-level question -- beats one fixed gate per device; and whether a
32x32-ish shape of the trainer's design is feasible for a handful of images.
The cone measurement above is the 32x32 answer in the form that is computable,
and it is unfriendly: the decoder's median output sees two key bits and the
encoder ignores two thirds of the image.

### Reproduce

```bash
# solver (outside the repo; Kissat 4.0.1)
curl -sL -o kissat.tar.gz https://github.com/arminbiere/kissat/archive/refs/tags/rel-4.0.1.tar.gz
tar xzf kissat.tar.gz && cd kissat-rel-4.0.1 && ./configure && make -j4

K=$PWD/build/kissat

# validation first: never trust the UNSATs without these
python3 scripts/sat_ae.py --solver $K --mode selftest
python3 scripts/sat_ae.py --solver $K --mode brute --shape 2,2,4 --patterns 3
python3 scripts/sat_ae.py --solver $K --mode brute --shape 2,4,4 --patterns 3

# the exact-fit frontier
for n in 1 2 3 4 8; do
  python3 scripts/sat_ae.py --solver $K --patterns $n --timeout 60
done

# which half of the net is the limit
for n in 3 4 8; do
  python3 scripts/sat_ae.py --solver $K --mode encoder --patterns $n
  python3 scripts/sat_ae.py --solver $K --mode decoder --patterns $n
done

# smaller decoder/impossible-by-construction checks
python3 scripts/sat_ae.py --solver $K --patterns 8 --slack 4   # UNSAT
python3 scripts/sat_ae.py --solver $K --patterns 8 --slack 8   # SAT, ~5 min

# gate type: XOR with a solved bias, plain XOR, or a free type bit per gate
python3 scripts/sat_ae.py --solver $K --gate xor3 --patterns 4       # UNSAT
python3 scripts/sat_ae.py --solver $K --gate xor2 --patterns 3       # UNSAT
python3 scripts/sat_ae.py --solver $K --gate mix  --patterns 4       # SAT, 0.8 s
python3 scripts/sat_ae.py --solver $K --gate mix --mode functions --shape 3,16,32,64

# the gate change that pays: NAND, or AND when the inversion bit is set
python3 scripts/sat_ae.py --solver $K --gate nandinv --patterns 6          # SAT, 6.1 s
python3 scripts/sat_ae.py --solver $K --gate nandinv --patterns 8          # UNSAT
python3 scripts/sat_ae.py --solver $K --gate nandinv --patterns 8 --slack 1  # SAT, 485 s
python3 scripts/sat_ae.py --solver $K --mode subsets --subset-size 4 --gate nandinv
python3 scripts/sat_ae.py --solver $K --mode subsets --subset-size 4 --gate nand
python3 scripts/sat_ae.py --solver $K --gate nandinv --mode functions --shape 3,16,32,64

# which four glyphs fit, rather than only the prefix
python3 scripts/sat_ae.py --solver $K --glyphs CEFG --patterns 4     # SAT

# how many functions of the key the decoder can compute (3-bit keys only)
python3 scripts/sat_ae.py --solver $K --mode functions --shape 3,16,32,64
python3 scripts/sat_ae.py --solver $K --mode functions --shape 3,16,16,16,16,16,32,64

# what each output bit can possibly depend on (the ceiling on expressiveness)
python3 scripts/circuit_cones.py web/public/models/cats32-nand.json \
  web/public/models/mnist32.json web/public/sat/circuit.json

# export the 3-glyph witness as browser circuits for the demo's SAT tab
python3 scripts/sat_ae.py --solver $K --patterns 3 --export web/public/sat

# the gradient trainer on identical data, for comparison
python3 scripts/make_glyph_dataset.py --out data/glyphs8.bin
python3 scripts/make_glyph_dataset.py --out data/glyphs3.bin --glyphs ABC
cargo run --release -p xorae -- train-nand --config configs/glyphs8_nand.toml \
  --data data/glyphs3.bin --out runs/glyphs3_nand --epochs 12000
```

