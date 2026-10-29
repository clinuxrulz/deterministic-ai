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
  - Best linear baseline (greedy GF(2) basis): **0.4358** test BER.
  - Strict XOR: **0.4477** test BER (0.4448 val); 380 active gates.
  - XOR+AND: **0.4214** test BER (0.4191 val); 347 active gates.
- The XOR+AND circuit beats the best linear code by about 1.4 BER points. The
  strict-XOR GA lands above the greedy basis, which is consistent with both the
  linearity barrier and the limits of GA search on this objective.
- Reconstructions are coarse at 16×16 with a 32-bit key. See
  `runs/cats16-xor-and/recon.png` and the Solid demo in `web/`.
