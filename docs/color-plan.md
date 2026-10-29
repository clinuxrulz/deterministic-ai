# Color restoration plan

Goal: make the XOR+AND autoencoder reconstruct a **color** image from its key,
not just a 1-bit grayscale sketch.

Current state:

```
RGB → luma → 16×16 → threshold → 256 bits → 32-bit key → 256 bits
```

The model never sees color. The 32×32 RGB image in the web demo is only a
preview; the circuit receives the 16×16 binarized grayscale grid.

The three levers are:

1. **Color encoding** — put color information into the bitstream.
2. **Key/gate capacity** — color needs more bits than 1-bit grayscale.
3. **Color-aware loss** — bit Hamming is not a perceptual color objective.

## Current results (2026-10-29)

- **16-color palette** (4 bits/pixel, 64/128-bit key, up to 2,048+2,048 gates):
  - Test color MAE ~0.273, index accuracy ~6.3% — essentially chance (1/16).
  - The model overfit the 128/512-image training batch but did not generalize.
  - Palette indices are categorical; the loss landscape is flat and the GA
    stayed at chance. **Negative result.**
- **RGB 1-bit/channel** (3 binary planes, 768 bits, 128-bit key,
  1,024+1,024 gates):
  - 1,500 generations: test color MAE 0.356, pixel accuracy 36%.
  - 10,000 generations: test color MAE 0.237, pixel accuracy 53%.
  - 30,000 generations (final): test color MAE **0.1729**, pixel accuracy
    **64.06%**, PSNR 7.62 dB, 577 active gates.
  - This encoding learns, because each channel is a spatially structured binary
    image. **Viable first color demo; now the web default.**
- **Decision:** RGB 1-bit/channel is the first color demo; the palette result is
  a documented negative experiment. YCbCr 4:2:0 remains the planned next step
  for smoother color.


---

## Phase 1 — 16-color global palette (first implementation)

**Target:** 16×16, 4 bits/pixel, 64-bit key, 1,024-bit input/output.

### Data

- Keep the resized 16×16 RGB pixels from CIFAR.
- Build a global palette of 16 colors with deterministic k-means over a sample
  of the training pixels (RGB distance).
- Quantize every pixel to its nearest palette index.
- Store in the dataset:
  - `palette`: 16 × RGB.
  - `values`: target palette index per pixel.
  - `rgb`: resized original RGB per pixel (for rendering/metrics).
  - `bits`: 4 bits per pixel, pixel-major: bit `4*p + b`.
- Cached dataset format becomes `XAE3`.

### Model

- `ModelConfig` gains:
  - `color_mode` (`grayscale` | `palette`).
  - `palette` (empty for grayscale).
  - `bits_per_pixel` (1 or 4).
- Encoder inputs = `image_size² × bits_per_pixel`.
- Decoder outputs = the same.
- Key = 64 bits.
- Gate budget: 1,024 encoder + 1,024 decoder (tune later).

### Fitness

- Evaluate the encoder/decoder as today to get the output bit planes.
- Decode each pixel's 4 bits to a palette index.
- Compare the decoded palette color with the **target palette color**:
  - `color_mae = mean(|ΔR| + |ΔG| + |ΔB|) / (3 × 255)`.
  - Or MSE; MAE is the default.
- Fitness = `color_mae` + existing entropy/uniqueness/parsimony terms.
- Log: color MAE, color MSE/PSNR, index accuracy, bit BER, key entropy,
  uniqueness, active gates.
- Keep bit BER as a diagnostic, not the primary objective.

### Eval / render

- `eval` prints color MAE, PSNR, index accuracy, and bit BER.
- `render` writes three panels:
  1. original 16×16 RGB,
  2. quantized 16×16 palette input,
  3. decoded 16×16 color reconstruction.
- Keep the grayscale renderer for grayscale models.

### WASM / web

- `ModelHandle` gains:
  - `color_mode`, `bits_per_pixel`, `palette`.
  - `preprocess(rgba, width, height)` — quantize to palette and encode bits.
  - `decode_rgb(key)` — key → 16×16 RGB.
  - `bits_to_rgb(bits)` — decoded/quantized bits → RGB.
- The Solid app:
  - Shows the 32×32 RGB source preview.
  - Shows a 16×16 color grid for the quantized input.
  - Shows a 16×16 color grid for the reconstruction.
  - Shows 64 key toggles.
  - Reports color MAE/PSNR instead of only bit BER.
  - Falls back to the existing bit grid for grayscale/toy models.

### Success criteria

- Pipeline trains and exports a palette model.
- Demo shows recognizable color blobs (posterized; 16 colors will band).
- Color MAE is clearly below an all-index-0 baseline.
- The rest of the repo (tests, web build, headless test) still works.

### Phase 1 checklist

- [ ] Write this plan.
- [ ] Add palette/color utilities to `xorae-core`.
- [ ] Add `color_mode`, `palette`, `bits_per_pixel` to `ModelConfig`/`Model`.
- [ ] Extend `Dataset` with `rgb`, `values`, `palette`, `color_mode`; `XAE3`.
- [ ] Implement deterministic k-means palette building and CIFAR palette loader.
- [ ] Add `preprocess --palette 16`.
- [ ] Add color-aware evaluator/fitness to `ga.rs`.
- [ ] Include the palette in exported model JSON.
- [ ] Add color rendering and color eval metrics.
- [ ] Extend `crates/wasm` with palette preprocessing and `decode_rgb`.
- [ ] Update the Solid UI for color grids and 64 key bits.
- [ ] Add tests: palette roundtrip, index bits, color evaluator, planar loader.
- [ ] Train a 16-color model, export it, rebuild the web app, headless-test it.
- [ ] Update README/docs with results.

---

## Phase 2 — YCbCr 4:2:0 (better perceptual color)

**Target:** 16×16, 3-bit Y + 2-bit Cb/Cr, 128-bit key, 1,024 bits.

- Convert RGB to YCbCr (BT.601/709).
- Y: 16×16, 3 bits/sample (768 bits).
- Cb/Cr: 8×8, 2 bits/sample (128 bits each).
- Total: 1,024 bits, 4 bits/pixel effective.
- Render: upsample chroma and convert back to RGB.
- Loss: weighted YCbCr MSE or CIEDE2000.
- Compare against Phase 1 at the same bit budget.
- Research basis: chroma subsampling is standard because human vision has lower
  acuity for color; CIEDE2000 is used in perceptual image compression.

---

## Phase 3 — scale up

Options, in increasing cost:

- 32-color palette (5 bits/pixel, 1,280 bits), 128-bit key.
- 16×16 YCbCr 4:2:0 with 4-bit Y / 3-bit chroma (1,408 bits), 128-bit key.
- 32×32 images, 128–256-bit key, 2,048+ gates.
- Per-image palette or residual/second-stage chroma model.
- Perceptual color loss (CIEDE2000) and post-hoc ordered dithering.

---

## Baselines and metrics

Color metrics:

- **Color MAE** — mean absolute RGB error per pixel, normalized 0–1.
- **Color MSE / PSNR** — for comparison with image codecs.
- **Index accuracy** — fraction of pixels with the correct palette index.
- **Exact image rate** — fraction of images reconstructed exactly (likely low).
- **Compression ratio** — input bits / key bits.
- **Visible color grids** — the real test.

Baselines:

- All-index-0 palette image.
- Per-pixel majority palette index.
- Quantization-only image (target palette colors) to separate palette loss from
  model loss.
- Current grayscale threshold model for structure comparison.

---

## Risks and expectations

- 16×16 with 64–128 key bits will be **posterized color blobs**, not photos.
- 16 colors will band; 32–64 colors are visibly better but need more bits/key.
- The GA search space grows with 4× input bits and a 2× key; use release builds,
  smaller batches, and fewer generations for the first pass.
- Bit Hamming is not perceptual; the color-aware loss is essential.
- One-hot 256-level RGB is not viable: 768 bits/pixel, 196k bits for 16×16, and
  Hamming loss is degenerate.
- The web app must use the model's palette and color mode; do not hard-code
  grayscale assumptions.

---

## Reproducible commands (Phase 1)

```bash
# Build palette dataset
cargo run -p xorae -- preprocess \
  --cifar-dir data/cifar-10-batches-bin \
  --out data/cats16_palette16.bin \
  --size 16 --palette 16

# Train
cargo run --release -p xorae -- train \
  --config configs/cats16_palette16.toml \
  --data data/cats16_palette16.bin \
  --out runs/cats16-palette16

# Evaluate / render
cargo run -p xorae -- eval \
  --model runs/cats16-palette16/model.json \
  --data data/cats16_palette16.bin --split test

cargo run -p xorae -- render \
  --model runs/cats16-palette16/model.json \
  --data data/cats16_palette16.bin \
  --out runs/cats16-palette16/recon.png --count 8

# Web
cargo run -p xorae -- export \
  --model runs/cats16-palette16/model.json \
  --out web/public/models/cats16-palette16.json
```
