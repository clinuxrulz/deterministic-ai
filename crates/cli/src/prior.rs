//! Key-space priors.
//!
//! The encoder maps an image to a `k`-bit key and the decoder maps it back. This
//! module asks whether the *set of keys the encoder produces* is something we
//! can model and sample from. If a fitted prior over keys produces images that
//! look like the training data, the code space is smooth and semantic; if it
//! produces noise, the key is a per-image codebook with no shared structure and
//! there is nothing for a generator to learn.
//!
//! Two priors are fitted, both on keys from the training split:
//!
//! * independent Bernoulli per bit (order 0),
//! * a causal linear autoregressive model, `p(x_j = 1) = sigmoid(b_j + sum_{i<j}
//!   w_ji x_i)` (a fully visible sigmoid belief net).
//!
//! Reference points: keys sampled from the empirical key set (an upper bound)
//! and uniform random keys (a lower bound).

use std::path::Path;

use anyhow::{Context, Result};
use image::{Rgb, RgbImage};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use xorae_core::Model;

use crate::dataset::Dataset;

#[derive(Clone, Copy, Debug)]
pub struct PriorOptions {
    pub samples: usize,
    pub epochs: usize,
    pub lr: f32,
    pub seed: u64,
}

impl Default for PriorOptions {
    fn default() -> Self {
        Self {
            samples: 32,
            epochs: 150,
            lr: 0.05,
            seed: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PriorReport {
    /// Mean absolute difference between sampled and data bit marginals.
    pub marginal_mae: f32,
    /// Mean pairwise Hamming distance between samples, in `0..=1`.
    pub diversity: f32,
    /// Fraction of a sampled key recovered by re-encoding its own decoding.
    pub roundtrip: f32,
    /// Mean negative log-likelihood on held-out keys, in bits per key bit.
    pub nll_bits: f32,
}

/// Fit both priors, sample from everything, write image grids and report.
pub fn run_prior(
    model: &Model,
    dataset: &Dataset,
    out_dir: &Path,
    opts: PriorOptions,
) -> Result<Vec<(&'static str, PriorReport)>> {
    model.validate().map_err(anyhow::Error::msg)?;
    let key_bits = model.config.key_bits;
    let (train, _, _) = dataset.split_indices(model.config.seed);
    anyhow::ensure!(!train.is_empty(), "no training split to fit on");

    let keys: Vec<Vec<u8>> = train
        .iter()
        .map(|&i| {
            model
                .infer_bits(&dataset.row_bits(i))
                .map(|(key, _)| key)
                .map_err(anyhow::Error::msg)
        })
        .collect::<Result<_>>()?;

    // Hold out the last 10% for likelihood evaluation.
    let split = (keys.len() * 9 / 10).max(1).min(keys.len());
    let (fit_keys, held_keys) = keys.split_at(split);

    let mut rng = ChaCha8Rng::seed_from_u64(opts.seed);
    let uniform = (0..opts.samples)
        .map(|_| {
            (0..key_bits)
                .map(|_| u8::from(rng.gen::<bool>()))
                .collect::<Vec<u8>>()
        })
        .collect::<Vec<_>>();
    let empirical = (0..opts.samples)
        .map(|_| keys[rng.gen_range(0..keys.len())].clone())
        .collect::<Vec<_>>();

    let marginals = key_marginals(fit_keys, key_bits);
    let bernoulli = sample_bernoulli(&marginals, key_bits, opts.samples, &mut rng);

    let (weights, bias) = fit_autoregressive(fit_keys, key_bits, opts);
    let nll_bernoulli = nll_independent(held_keys, &marginals, key_bits);
    let nll_ar = nll_autoregressive(held_keys, &weights, &bias, key_bits);
    let ar = (0..opts.samples)
        .map(|_| sample_autoregressive(&weights, &bias, key_bits, &mut rng))
        .collect::<Vec<_>>();

    std::fs::create_dir_all(out_dir).with_context(|| format!("creating {}", out_dir.display()))?;

    let mut reports = Vec::new();
    for (name, samples, nll) in [
        ("uniform", &uniform, 1.0f32),
        ("empirical", &empirical, f32::NAN),
        ("bernoulli", &bernoulli, nll_bernoulli),
        ("autoregressive", &ar, nll_ar),
    ] {
        let mut report = evaluate(model, samples, &marginals);
        report.nll_bits = nll;
        write_grid(model, samples, &out_dir.join(format!("{name}.png")), 8)?;
        reports.push((name, report));
    }
    Ok(reports)
}

/// Per-bit probability of 1 over the training keys.
fn key_marginals(keys: &[Vec<u8>], key_bits: usize) -> Vec<f32> {
    let mut sums = vec![0f32; key_bits];
    for key in keys {
        for (sum, &bit) in sums.iter_mut().zip(key.iter()) {
            *sum += bit as f32;
        }
    }
    let n = keys.len().max(1) as f32;
    sums.into_iter().map(|s| s / n).collect()
}

fn sample_bernoulli(
    marginals: &[f32],
    key_bits: usize,
    samples: usize,
    rng: &mut impl Rng,
) -> Vec<Vec<u8>> {
    (0..samples)
        .map(|_| {
            (0..key_bits)
                .map(|j| u8::from(rng.gen::<f32>() < marginals[j]))
                .collect()
        })
        .collect()
}

/// Causal linear autoregressive model, fitted with Adam and teacher forcing.
fn fit_autoregressive(
    keys: &[Vec<u8>],
    key_bits: usize,
    opts: PriorOptions,
) -> (Vec<f32>, Vec<f32>) {
    let mut weights = vec![0f32; key_bits * key_bits];
    let mut bias = vec![0f32; key_bits];
    let mut mw = vec![0f32; weights.len()];
    let mut vw = vec![0f32; weights.len()];
    let mut mb = vec![0f32; key_bits];
    let mut vb = vec![0f32; key_bits];
    let n = keys.len().max(1) as f32;
    let (beta1, beta2) = (0.9f32, 0.999f32);

    for epoch in 0..opts.epochs {
        let mut gw = vec![0f32; weights.len()];
        let mut gb = vec![0f32; key_bits];
        for key in keys {
            for j in 0..key_bits {
                let mut z = bias[j];
                for i in 0..j {
                    z += weights[j * key_bits + i] * key[i] as f32;
                }
                let p = sigmoid(z);
                let g = p - key[j] as f32;
                gb[j] += g;
                for i in 0..j {
                    gw[j * key_bits + i] += g * key[i] as f32;
                }
            }
        }
        let t = epoch as i32 + 1;
        let bc1 = 1.0 - beta1.powi(t);
        let bc2 = 1.0 - beta2.powi(t);
        for i in 0..weights.len() {
            let g = gw[i] / n;
            mw[i] = beta1 * mw[i] + (1.0 - beta1) * g;
            vw[i] = beta2 * vw[i] + (1.0 - beta2) * g * g;
            weights[i] -= opts.lr * (mw[i] / bc1) / ((vw[i] / bc2).sqrt() + 1e-8);
        }
        for j in 0..key_bits {
            let g = gb[j] / n;
            mb[j] = beta1 * mb[j] + (1.0 - beta1) * g;
            vb[j] = beta2 * vb[j] + (1.0 - beta2) * g * g;
            bias[j] -= opts.lr * (mb[j] / bc1) / ((vb[j] / bc2).sqrt() + 1e-8);
        }
    }
    (weights, bias)
}

fn sample_autoregressive(
    weights: &[f32],
    bias: &[f32],
    key_bits: usize,
    rng: &mut impl Rng,
) -> Vec<u8> {
    let mut key = vec![0u8; key_bits];
    for j in 0..key_bits {
        let mut z = bias[j];
        for i in 0..j {
            z += weights[j * key_bits + i] * key[i] as f32;
        }
        key[j] = u8::from(rng.gen::<f32>() < sigmoid(z));
    }
    key
}

fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

/// Mean NLL (bits) of the independent Bernoulli model on held-out keys.
fn nll_independent(keys: &[Vec<u8>], marginals: &[f32], key_bits: usize) -> f32 {
    if keys.is_empty() {
        return f32::NAN;
    }
    let mut total = 0f64;
    for key in keys {
        for (j, &bit) in key.iter().enumerate().take(key_bits) {
            let p = marginals[j].clamp(1e-6, 1.0 - 1e-6) as f64;
            total += if bit == 1 { -p.ln() } else { -(1.0 - p).ln() };
        }
    }
    (total / (keys.len() * key_bits) as f64 / std::f64::consts::LN_2) as f32
}

/// Mean NLL (bits) of the causal autoregressive model on held-out keys.
fn nll_autoregressive(keys: &[Vec<u8>], weights: &[f32], bias: &[f32], key_bits: usize) -> f32 {
    if keys.is_empty() {
        return f32::NAN;
    }
    let mut total = 0f64;
    for key in keys {
        for j in 0..key_bits {
            let mut z = bias[j];
            for i in 0..j {
                z += weights[j * key_bits + i] * key[i] as f32;
            }
            let p = sigmoid(z).clamp(1e-6, 1.0 - 1e-6) as f64;
            total += if key[j] == 1 {
                -p.ln()
            } else {
                -(1.0 - p).ln()
            };
        }
    }
    (total / (keys.len() * key_bits) as f64 / std::f64::consts::LN_2) as f32
}

/// Compare a set of sampled keys with the training keys.
fn evaluate(model: &Model, samples: &[Vec<u8>], data_marginals: &[f32]) -> PriorReport {
    let key_bits = model.config.key_bits;
    let sampled = key_marginals(samples, key_bits);
    let marginal_mae = sampled
        .iter()
        .zip(data_marginals)
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / key_bits as f32;

    let mut total = 0f64;
    let mut pairs = 0f64;
    for i in 0..samples.len() {
        for j in i + 1..samples.len() {
            let d = samples[i]
                .iter()
                .zip(samples[j].iter())
                .filter(|(a, b)| a != b)
                .count();
            total += d as f64 / key_bits as f64;
            pairs += 1.0;
        }
    }
    let diversity = if pairs > 0.0 {
        (total / pairs) as f32
    } else {
        0.0
    };

    let mut correct = 0usize;
    let mut seen = 0usize;
    for key in samples {
        // Decode the key to an image, then re-encode that image. A key that
        // sits on the model's manifold survives this round trip.
        if let Ok(image) = model.decode_key(key) {
            if let Ok((again, _)) = model.infer_bits(&image) {
                correct += key.iter().zip(again.iter()).filter(|(a, b)| a == b).count();
                seen += key_bits;
            }
        }
    }
    let roundtrip = if seen > 0 {
        correct as f32 / seen as f32
    } else {
        0.0
    };

    PriorReport {
        marginal_mae,
        diversity,
        roundtrip,
        nll_bits: f32::NAN,
    }
}

/// Write `samples` decoded images as a square grid of RGB tiles.
fn write_grid(model: &Model, keys: &[Vec<u8>], path: &Path, columns: usize) -> Result<()> {
    let size = model.config.image_size;
    let scale = 6u32;
    let tile = size as u32 * scale;
    let gap = 4u32;
    let rows = keys.len().div_ceil(columns.max(1)).max(1);
    let columns = columns.max(1) as u32;
    let width = columns * tile + (columns + 1) * gap;
    let height = rows as u32 * tile + (rows as u32 + 1) * gap;
    let mut canvas = RgbImage::from_pixel(width, height, Rgb([22, 24, 29]));

    for (i, key) in keys.iter().enumerate() {
        let rgb = model
            .decode_rgb(key)
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("decoding sample {i}"))?;
        let x0 = gap + (i as u32 % columns) * (tile + gap);
        let y0 = gap + (i as u32 / columns) * (tile + gap);
        for y in 0..size {
            for x in 0..size {
                let p = (y * size + x) * 3;
                let color = Rgb([rgb[p], rgb[p + 1], rgb[p + 2]]);
                for dy in 0..scale {
                    for dx in 0..scale {
                        canvas.put_pixel(
                            x0 + x as u32 * scale + dx,
                            y0 + y as u32 * scale + dy,
                            color,
                        );
                    }
                }
            }
        }
    }
    canvas
        .save(path)
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marginals_and_sampling_are_consistent() {
        let keys = vec![vec![1, 0, 1, 1], vec![1, 0, 0, 1], vec![0, 0, 1, 0]];
        let m = key_marginals(&keys, 4);
        assert!((m[0] - 2.0 / 3.0).abs() < 1e-6);
        assert!((m[1] - 0.0).abs() < 1e-6);
        assert!((m[2] - 2.0 / 3.0).abs() < 1e-6);
        assert!((m[3] - 2.0 / 3.0).abs() < 1e-6);
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let s = sample_bernoulli(&m, 4, 16, &mut rng);
        assert!(
            s.iter().all(|k| k[1] == 0),
            "a zero-marginal bit must stay 0"
        );
    }

    #[test]
    fn autoregressive_fit_learns_a_deterministic_copy() {
        // Every key is the same: the AR model should reproduce it.
        let keys = vec![vec![1, 0, 1, 0, 1, 1, 0, 1]; 32];
        let opts = PriorOptions {
            epochs: 200,
            lr: 0.1,
            ..Default::default()
        };
        let (w, b) = fit_autoregressive(&keys, 8, opts);
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        for _ in 0..8 {
            let s = sample_autoregressive(&w, &b, 8, &mut rng);
            assert_eq!(s, keys[0]);
        }
    }
}
