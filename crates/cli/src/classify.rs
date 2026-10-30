//! Train a classifier on the encoder's key bits (or on raw pixels, as a
//! baseline) and report accuracy with a confusion matrix.
//!
//! The autoencoder itself produces no labels; this is the "proper classifier"
//! that sits on top of the 128-bit code. Multinomial logistic regression keeps
//! the model small enough to run in the browser (128 x 10 weights) and gives an
//! unambiguous accuracy number, which BER cannot provide on sparse data.

use std::path::Path;

use anyhow::{bail, Context, Result};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use xorae_core::Model;

use crate::dataset::Dataset;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Features {
    /// The 128-bit encoder key.
    Keys,
    /// The raw input bits, as a baseline with no learned compression.
    Pixels,
}

impl Features {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "keys" => Ok(Self::Keys),
            "pixels" => Ok(Self::Pixels),
            other => bail!("unknown feature set {other} (use keys or pixels)"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ClassifyOptions {
    pub features: Features,
    pub epochs: usize,
    pub lr: f32,
    pub l2: f32,
    pub seed: u64,
}

impl Default for ClassifyOptions {
    fn default() -> Self {
        Self {
            features: Features::Keys,
            epochs: 400,
            lr: 0.1,
            l2: 1e-4,
            seed: 0,
        }
    }
}

/// A trained softmax classifier over binary features.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Classifier {
    /// `"keys"` or `"pixels"`.
    pub features: String,
    /// Number of input bits.
    pub inputs: usize,
    /// Number of classes.
    pub classes: usize,
    /// `classes x inputs` row-major weights.
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
    pub train_accuracy: f32,
    pub test_accuracy: f32,
}

impl Classifier {
    pub fn logits(&self, x: &[u8]) -> Vec<f32> {
        let mut out = vec![0f32; self.classes];
        for c in 0..self.classes {
            let row = &self.weights[c * self.inputs..(c + 1) * self.inputs];
            let mut z = self.bias[c];
            for (w, &bit) in row.iter().zip(x.iter()) {
                if bit == 1 {
                    z += w;
                }
            }
            out[c] = z;
        }
        out
    }

    /// Softmax probabilities over the classes.
    pub fn probabilities(&self, x: &[u8]) -> Vec<f32> {
        let logits = self.logits(x);
        let max = logits.iter().copied().fold(f32::MIN, f32::max);
        let exps: Vec<f32> = logits.iter().map(|z| (z - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.into_iter().map(|e| e / sum).collect()
    }

    pub fn predict(&self, x: &[u8]) -> usize {
        let logits = self.logits(x);
        argmax(&logits)
    }
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in values.iter().enumerate() {
        if v > values[best] {
            best = i;
        }
    }
    best
}

/// Train and evaluate on a labelled dataset. Returns the classifier plus the
/// confusion matrix on the test split.
pub fn train_classifier(
    model: &Model,
    dataset: &Dataset,
    opts: ClassifyOptions,
) -> Result<(Classifier, Vec<Vec<usize>>)> {
    anyhow::ensure!(
        dataset.has_labels(),
        "dataset has no labels (re-run preprocess)"
    );
    let classes = dataset.label_count.max(1);
    let (train, _, test) = dataset.split_indices(model.config.seed);
    anyhow::ensure!(!train.is_empty() && !test.is_empty(), "empty split");

    let features = |i: usize| -> Result<Vec<u8>> {
        match opts.features {
            Features::Keys => model
                .infer_bits(&dataset.row_bits(i))
                .map(|(key, _)| key)
                .map_err(anyhow::Error::msg),
            Features::Pixels => Ok(dataset.row_bits(i)),
        }
    };

    let build = |indices: &[usize]| -> Result<Vec<Vec<u8>>> {
        indices.iter().map(|&i| features(i)).collect()
    };
    let train_x = build(&train)?;
    let test_x = build(&test)?;
    let inputs = train_x.first().map(|v| v.len()).unwrap_or(0);
    anyhow::ensure!(inputs > 0, "no features");

    let mut rng = ChaCha8Rng::seed_from_u64(opts.seed);
    let mut weights = vec![0f32; classes * inputs];
    for w in weights.iter_mut() {
        *w = (rng.gen::<f32>() - 0.5) * 0.01;
    }
    let mut bias = vec![0f32; classes];
    let mut mw = vec![0f32; weights.len()];
    let mut vw = vec![0f32; weights.len()];
    let mut mb = vec![0f32; classes];
    let mut vb = vec![0f32; classes];
    let (beta1, beta2) = (0.9f32, 0.999f32);

    let n = train.len() as f32;
    for epoch in 0..opts.epochs {
        let mut gw = vec![0f32; weights.len()];
        let mut gb = vec![0f32; classes];
        for (x, &i) in train_x.iter().zip(train.iter()) {
            let y = dataset.labels[i] as usize;
            if y >= classes {
                continue;
            }
            let logits = {
                let mut z = vec![0f32; classes];
                for c in 0..classes {
                    let row = &weights[c * inputs..(c + 1) * inputs];
                    z[c] = bias[c]
                        + row
                            .iter()
                            .zip(x.iter())
                            .filter(|(_, &b)| b == 1)
                            .map(|(w, _)| *w)
                            .sum::<f32>();
                }
                z
            };
            let max = logits.iter().copied().fold(f32::MIN, f32::max);
            let exps: Vec<f32> = logits.iter().map(|z| (z - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for c in 0..classes {
                let p = exps[c] / sum;
                let g = p - f32::from(c == y);
                gb[c] += g;
                let row = c * inputs;
                for k in 0..inputs {
                    if x[k] == 1 {
                        gw[row + k] += g;
                    }
                }
            }
        }

        let t = epoch as i32 + 1;
        let bc1 = 1.0 - beta1.powi(t);
        let bc2 = 1.0 - beta2.powi(t);
        for k in 0..weights.len() {
            let g = gw[k] / n + opts.l2 * weights[k];
            mw[k] = beta1 * mw[k] + (1.0 - beta1) * g;
            vw[k] = beta2 * vw[k] + (1.0 - beta2) * g * g;
            weights[k] -= opts.lr * (mw[k] / bc1) / ((vw[k] / bc2).sqrt() + 1e-8);
        }
        for c in 0..classes {
            let g = gb[c] / n;
            mb[c] = beta1 * mb[c] + (1.0 - beta1) * g;
            vb[c] = beta2 * vb[c] + (1.0 - beta2) * g * g;
            bias[c] -= opts.lr * (mb[c] / bc1) / ((vb[c] / bc2).sqrt() + 1e-8);
        }
    }

    let classifier = Classifier {
        features: match opts.features {
            Features::Keys => "keys".into(),
            Features::Pixels => "pixels".into(),
        },
        inputs,
        classes,
        weights,
        bias,
        train_accuracy: 0.0,
        test_accuracy: 0.0,
    };

    let mut matrix = vec![vec![0usize; classes]; classes];
    let mut correct = 0usize;
    for (x, &i) in test_x.iter().zip(test.iter()) {
        let y = dataset.labels[i] as usize;
        let p = classifier.predict(x);
        if y < classes && p < classes {
            matrix[y][p] += 1;
        }
        if p == y {
            correct += 1;
        }
    }
    let mut train_correct = 0usize;
    for (x, &i) in train_x.iter().zip(train.iter()) {
        if classifier.predict(x) == dataset.labels[i] as usize {
            train_correct += 1;
        }
    }

    let mut classifier = classifier;
    classifier.train_accuracy = train_correct as f32 / train.len().max(1) as f32;
    classifier.test_accuracy = correct as f32 / test.len().max(1) as f32;
    Ok((classifier, matrix))
}

/// Write the classifier next to its namesake model.
pub fn save_classifier(classifier: &Classifier, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(classifier).context("serialising classifier")?;
    std::fs::write(path, json).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logits_and_prediction_follow_the_weights() {
        // Two classes, two inputs: class 0 likes bit 0, class 1 likes bit 1.
        let c = Classifier {
            features: "keys".into(),
            inputs: 2,
            classes: 2,
            weights: vec![1.0, 0.0, 0.0, 1.0],
            bias: vec![0.0, 0.0],
            train_accuracy: 0.0,
            test_accuracy: 0.0,
        };
        assert_eq!(c.predict(&[1, 0]), 0);
        assert_eq!(c.predict(&[0, 1]), 1);
        assert_eq!(c.logits(&[1, 1]), vec![1.0, 1.0]);
    }
}
