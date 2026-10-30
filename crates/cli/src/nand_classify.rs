//! NAND circuit classifier over encoder key bits.
//!
//! A dedicated wiring search, separate from the autoencoder trainer: the
//! network is a small layered DAG of NAND gates, `inputs -> hidden.. -> classes`,
//! where each gate's two inputs are chosen by a softmax over candidate wires
//! that anneals to one-hot. The loss is mean-squared error against a one-hot
//! target, plus the same `a(1-a)` saturation penalty the autoencoder uses.
//!
//! At the end the argmax wiring is materialised as a literal `Circuit`, which
//! is exact-simplified and scored on the held-out split. Nothing here is
//! real-valued at inference: the exported object is gates all the way down.

use anyhow::{bail, Result};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use xorae_core::circuit::{Circuit, Gate, Op};

use crate::nand::Relaxation;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NandClassifyConfig {
    /// Number of key bits (network inputs).
    pub inputs: usize,
    /// Number of classes (network outputs).
    pub classes: usize,
    /// Hidden layer widths.
    pub hidden: Vec<usize>,
    /// Output gates per class. The class score is the sum of its group, which
    /// gives the loss far more gradient signal than a single output bit.
    #[serde(default = "default_group_size")]
    pub group_size: usize,
    pub epochs: usize,
    pub batch: usize,
    pub lr: f32,
    /// Candidate wires per gate.
    #[serde(default = "default_candidates")]
    pub candidates: usize,
    /// Probability a candidate is drawn from the immediately previous layer.
    #[serde(default = "default_skip_prob")]
    pub skip_prob: f32,
    #[serde(default = "default_binarize_weight")]
    pub binarize_weight: f32,
    #[serde(default = "default_gain_start")]
    pub gain_start: f32,
    #[serde(default = "default_gain_end")]
    pub gain_end: f32,
    #[serde(default = "default_anneal_fraction")]
    pub anneal_fraction: f32,
    #[serde(default)]
    pub relaxation: Relaxation,
    pub seed: u64,
}

fn default_group_size() -> usize {
    8
}
fn default_candidates() -> usize {
    16
}
fn default_skip_prob() -> f32 {
    0.5
}
fn default_binarize_weight() -> f32 {
    0.01
}
fn default_gain_start() -> f32 {
    1.0
}
fn default_gain_end() -> f32 {
    32.0
}
fn default_anneal_fraction() -> f32 {
    0.85
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TrainReport {
    pub train_accuracy: f32,
    pub test_accuracy: f32,
    pub gates_before: usize,
    pub gates_after: usize,
}

/// Layered candidate wiring, indexed flat by gate.
struct Net {
    sizes: Vec<usize>,
    /// `(layer, index)` per candidate, flat as `gate * k + c`.
    cand_layer: Vec<u8>,
    cand_idx: Vec<u16>,
    k: usize,
    n_gates: usize,
    layer_off: Vec<usize>,
}

impl Net {
    fn build(sizes: &[usize], k: usize, skip_prob: f32, rng: &mut impl Rng) -> Result<Self> {
        if sizes.len() < 2 {
            bail!("network needs at least an input and an output layer");
        }
        let layers = sizes.len() - 1;
        let n_gates: usize = sizes[1..].iter().sum();
        let mut cand_layer = Vec::with_capacity(n_gates * k);
        let mut cand_idx = Vec::with_capacity(n_gates * k);
        let mut layer_off = Vec::with_capacity(layers + 1);
        layer_off.push(0usize);
        let mut off = 0usize;
        for l in 0..layers {
            let out = sizes[l + 1];
            for _ in 0..out {
                for _ in 0..k {
                    // Prefer the previous layer, occasionally reach further back.
                    let src = if l == 0 || rng.gen::<f32>() < skip_prob {
                        l
                    } else {
                        rng.gen_range(0..=l)
                    };
                    cand_layer.push(src as u8);
                    cand_idx.push(rng.gen_range(0..sizes[src]) as u16);
                }
            }
            off += out;
            layer_off.push(off);
        }
        Ok(Self {
            sizes: sizes.to_vec(),
            cand_layer,
            cand_idx,
            k,
            n_gates,
            layer_off,
        })
    }
}

fn relax(relaxation: Relaxation, u: f32, v: f32) -> f32 {
    match relaxation {
        Relaxation::Product => 1.0 - u * v,
        Relaxation::Min => 1.0 - u.min(v),
        Relaxation::Lukasiewicz => 1.0 - (u + v - 1.0).max(0.0),
    }
}

fn relax_grad(relaxation: Relaxation, u: f32, v: f32) -> (f32, f32) {
    match relaxation {
        Relaxation::Product => (-v, -u),
        Relaxation::Min => {
            if u <= v {
                (-1.0, 0.0)
            } else {
                (0.0, -1.0)
            }
        }
        Relaxation::Lukasiewicz => {
            if u + v > 1.0 {
                (-1.0, -1.0)
            } else {
                (0.0, 0.0)
            }
        }
    }
}

/// Train the classifier on key bits. Returns the simplified circuit and its
/// accuracies on the provided splits.
pub fn train(
    cfg: &NandClassifyConfig,
    train_keys: &[Vec<u8>],
    train_labels: &[u8],
    test_keys: &[Vec<u8>],
    test_labels: &[u8],
    soft_targets: Option<&[Vec<f32>]>,
) -> Result<(Circuit, TrainReport)> {
    anyhow::ensure!(!train_keys.is_empty(), "no training keys");
    let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);
    let mut sizes = vec![cfg.inputs];
    sizes.extend(cfg.hidden.iter().copied());
    sizes.push(cfg.classes * cfg.group_size.max(1));
    let net = Net::build(&sizes, cfg.candidates.max(2), cfg.skip_prob, &mut rng)?;

    let k = net.k;
    let stride = cfg.batch;
    let n_layers = net.sizes.len() - 1;
    let mut alpha = vec![0f32; net.n_gates * k];
    let mut beta = vec![0f32; net.n_gates * k];
    for v in alpha.iter_mut() {
        *v = rng.gen::<f32>() * 0.1 - 0.05;
    }
    for v in beta.iter_mut() {
        *v = rng.gen::<f32>() * 0.1 - 0.05;
    }

    let mut acts: Vec<Vec<f32>> = net.sizes.iter().map(|&n| vec![0f32; n * stride]).collect();
    let mut zs: Vec<Vec<f32>> = (0..n_layers)
        .map(|l| vec![0f32; net.sizes[l + 1] * stride])
        .collect();
    let mut dz: Vec<Vec<f32>> = (0..n_layers)
        .map(|l| vec![0f32; net.sizes[l + 1] * stride])
        .collect();

    let mut adam_a = Adam::new(alpha.len(), cfg.lr);
    let mut adam_b = Adam::new(beta.len(), cfg.lr);
    let order: Vec<usize> = (0..train_keys.len()).collect();
    let mut shuffled = order.clone();

    for epoch in 0..cfg.epochs {
        let span = (cfg.anneal_fraction.clamp(1e-4, 1.0) * cfg.epochs as f32).max(1.0);
        let t = (epoch as f32 / span).clamp(0.0, 1.0);
        let gain = cfg.gain_start * (cfg.gain_end / cfg.gain_start).powf(t);
        shuffled.copy_from_slice(&order);
        // Fisher-Yates.
        for i in (1..shuffled.len()).rev() {
            let j = rng.gen_range(0..=i);
            shuffled.swap(i, j);
        }

        for chunk in shuffled.chunks(stride) {
            let n = chunk.len();
            if n == 0 {
                continue;
            }
            // Load the batch.
            for (s, &row) in chunk.iter().enumerate() {
                for j in 0..cfg.inputs {
                    acts[0][j * stride + s] = train_keys[row][j] as f32;
                }
            }

            let (probs_a, probs_b) = probs(&alpha, &beta, net.k, gain);
            refresh(&probs_a, &probs_b, &mut acts, &mut zs, &net, stride, n, cfg);

            // Loss: softmax cross-entropy over per-class group sums.
            for d in dz.iter_mut() {
                d.fill(0.0);
            }
            let last = n_layers - 1;
            let group = cfg.group_size.max(1);
            let classes = cfg.classes;
            for s in 0..n {
                let mut scores = vec![0f32; classes];
                for (c, score) in scores.iter_mut().enumerate() {
                    for j in 0..group {
                        *score += zs[last][(c * group + j) * stride + s];
                    }
                }
                let max = scores.iter().copied().fold(f32::MIN, f32::max);
                let exps: Vec<f32> = scores.iter().map(|z| (z - max).exp()).collect();
                let sum: f32 = exps.iter().sum();
                let row = chunk[s];
                let target = |c: usize| -> f32 {
                    match soft_targets {
                        Some(t) => t[row].get(c).copied().unwrap_or(0.0),
                        None => f32::from(c == train_labels[row] as usize),
                    }
                };
                for (c, &e) in exps.iter().enumerate() {
                    let p = e / sum;
                    let g = (p - target(c)) / n as f32;
                    for j in 0..group {
                        dz[last][(c * group + j) * stride + s] += g;
                    }
                }
            }
            // Saturation penalty pushes activations toward 0/1.
            if cfg.binarize_weight > 0.0 {
                for l in 0..n_layers {
                    let plane = net.sizes[l + 1];
                    let bw = cfg.binarize_weight / (plane * n).max(1) as f32;
                    for i in 0..plane {
                        for s in 0..n {
                            dz[l][i * stride + s] += bw * (1.0 - 2.0 * acts[l + 1][i * stride + s]);
                        }
                    }
                }
            }

            let mut dalpha = vec![0f32; alpha.len()];
            let mut dbeta = vec![0f32; beta.len()];
            backward(
                &probs_a,
                &probs_b,
                &acts,
                &mut dz,
                &net,
                stride,
                n,
                cfg,
                &mut dalpha,
                &mut dbeta,
            );
            adam_a.step(&mut alpha, &dalpha);
            adam_b.step(&mut beta, &dbeta);
        }

        if epoch % 25 == 0 || epoch + 1 == cfg.epochs {
            let probe = &train_keys[..train_keys.len().min(1000)];
            let probe_labels = &train_labels[..probe.len()];
            let circuit = materialise(&net, &alpha, &beta);
            let acc = accuracy(
                &circuit,
                probe,
                probe_labels,
                cfg.classes,
                cfg.group_size.max(1),
            );
            let alive = circuit.active_gates();
            eprintln!(
                "epoch {epoch}: train acc {acc:.4} (probe), gain {gain:.1}, active gates {alive}/{}",
                circuit.gates.len()
            );
        }
    }

    // Materialise the argmax wiring, then exact-simplify it.
    let raw = materialise(&net, &alpha, &beta);
    raw.validate().map_err(anyhow::Error::msg)?;
    let report_before = raw.gates.len();
    let (circuit, _) = raw.simplify();
    circuit.validate().map_err(anyhow::Error::msg)?;

    let group = cfg.group_size.max(1);
    let train_accuracy = accuracy(&circuit, train_keys, train_labels, cfg.classes, group);
    let test_accuracy = accuracy(&circuit, test_keys, test_labels, cfg.classes, group);
    let gates_after = circuit.gates.len();
    Ok((
        circuit,
        TrainReport {
            train_accuracy,
            test_accuracy,
            gates_before: report_before,
            gates_after,
        },
    ))
}

/// Softmax over each gate's candidates at the current annealing temperature.
fn probs(alpha: &[f32], beta: &[f32], k: usize, gain: f32) -> (Vec<f32>, Vec<f32>) {
    let normalise = |params: &[f32]| -> Vec<f32> {
        let gates = params.len() / k;
        let mut out = vec![0f32; params.len()];
        for g in 0..gates {
            let base = g * k;
            let mut max = f32::MIN;
            for c in 0..k {
                max = max.max(params[base + c] * gain);
            }
            let mut sum = 0.0;
            for c in 0..k {
                let e = (params[base + c] * gain - max).exp();
                out[base + c] = e;
                sum += e;
            }
            for c in 0..k {
                out[base + c] /= sum;
            }
        }
        out
    };
    (normalise(alpha), normalise(beta))
}

#[allow(clippy::too_many_arguments)]
fn refresh(
    pa: &[f32],
    pb: &[f32],
    acts: &mut [Vec<f32>],
    zs: &mut [Vec<f32>],
    net: &Net,
    stride: usize,
    n: usize,
    cfg: &NandClassifyConfig,
) {
    let k = net.k;
    for l in 0..net.sizes.len() - 1 {
        let out = net.sizes[l + 1];
        for j in 0..out {
            let g = net.layer_off[l] + j;
            for s in 0..n {
                let mut u = 0.0;
                let mut v = 0.0;
                for c in 0..k {
                    let layer = net.cand_layer[g * k + c] as usize;
                    let idx = net.cand_idx[g * k + c] as usize;
                    let x = acts[layer][idx * stride + s];
                    u += pa[g * k + c] * x;
                    v += pb[g * k + c] * x;
                }
                let z = relax(cfg.relaxation, u, v);
                zs[l][j * stride + s] = z;
                acts[l + 1][j * stride + s] = z;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn backward(
    pa: &[f32],
    pb: &[f32],
    acts: &[Vec<f32>],
    dz: &mut [Vec<f32>],
    net: &Net,
    stride: usize,
    n: usize,
    cfg: &NandClassifyConfig,
    dalpha: &mut [f32],
    dbeta: &mut [f32],
) {
    let k = net.k;
    for l in (0..net.sizes.len() - 1).rev() {
        let out = net.sizes[l + 1];
        for j in 0..out {
            let g = net.layer_off[l] + j;
            for s in 0..n {
                let dzj = dz[l][j * stride + s];
                if dzj == 0.0 {
                    continue;
                }
                let mut u = 0.0;
                let mut v = 0.0;
                for c in 0..k {
                    let layer = net.cand_layer[g * k + c] as usize;
                    let idx = net.cand_idx[g * k + c] as usize;
                    let x = acts[layer][idx * stride + s];
                    u += pa[g * k + c] * x;
                    v += pb[g * k + c] * x;
                }
                let (gu, gv) = relax_grad(cfg.relaxation, u, v);
                let du = dzj * gu;
                let dv = dzj * gv;
                for c in 0..k {
                    let layer = net.cand_layer[g * k + c] as usize;
                    let idx = net.cand_idx[g * k + c] as usize;
                    let x = acts[layer][idx * stride + s];
                    dalpha[g * k + c] += du * x;
                    dbeta[g * k + c] += dv * x;
                    // act[k] is zs[k-1], so a candidate from plane `layer`
                    // contributes to dz[layer - 1].
                    if layer >= 1 {
                        let contrib = du * pa[g * k + c] + dv * pb[g * k + c];
                        dz[layer - 1][idx * stride + s] += contrib;
                    }
                }
            }
        }
    }
}

fn materialise(net: &Net, alpha: &[f32], beta: &[f32]) -> Circuit {
    // Wire numbering: inputs 0..inputs, const 0 = inputs, const 1 = inputs+1,
    // gate i at inputs + 2 + i.
    let n_inputs = net.sizes[0];
    let base = n_inputs + 2;
    let mut gates = Vec::with_capacity(net.n_gates);
    for g in 0..net.n_gates {
        let ca = argmax(&alpha[g * net.k..(g + 1) * net.k]);
        let cb = argmax(&beta[g * net.k..(g + 1) * net.k]);
        let layer_a = net.cand_layer[g * net.k + ca] as usize;
        let layer_b = net.cand_layer[g * net.k + cb] as usize;
        let wire = |layer: usize, index: usize| -> u16 {
            if layer == 0 {
                index as u16
            } else {
                (base + net.layer_off[layer - 1] + index) as u16
            }
        };
        gates.push(Gate {
            op: Op::Nand,
            a: wire(layer_a, net.cand_idx[g * net.k + ca] as usize),
            b: wire(layer_b, net.cand_idx[g * net.k + cb] as usize),
        });
    }
    let last_off = net.layer_off[net.layer_off.len() - 2];
    let n_out = net.sizes[net.sizes.len() - 1];
    let outputs = (0..n_out).map(|i| (base + last_off + i) as u16).collect();
    Circuit {
        n_inputs: n_inputs as u16,
        gates,
        outputs,
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

/// Predicted class for one feature vector.
pub fn predict(circuit: &Circuit, input: &[u8], classes: usize, group: usize) -> usize {
    let mut planes = vec![0u64; circuit.n_inputs as usize];
    for (j, slot) in planes.iter_mut().enumerate() {
        *slot = *input.get(j).unwrap_or(&0) as u64;
    }
    let mut arena = Vec::new();
    let mut out = Vec::new();
    circuit.eval_into(&planes, 1, &mut arena, &mut out);
    argmax_group(&out, classes, group)
}

/// Fraction of samples whose argmax output equals the label.
pub fn accuracy(
    circuit: &Circuit,
    keys: &[Vec<u8>],
    labels: &[u8],
    classes: usize,
    group: usize,
) -> f32 {
    if keys.is_empty() {
        return 0.0;
    }
    let mut correct = 0usize;
    let mut planes = vec![0u64; circuit.n_inputs as usize];
    let mut arena = Vec::new();
    let mut out = Vec::new();
    for (i, key) in keys.iter().enumerate() {
        for (j, slot) in planes.iter_mut().enumerate() {
            *slot = *key.get(j).unwrap_or(&0) as u64;
        }
        circuit.eval_into(&planes, 1, &mut arena, &mut out);
        let predicted = argmax_group(&out, classes, group);
        if predicted == labels[i] as usize {
            correct += 1;
        }
    }
    correct as f32 / keys.len() as f32
}

/// Class with the largest sum over its output group.
fn argmax_group(out: &[u64], classes: usize, group: usize) -> usize {
    let group = group.max(1);
    let mut best = 0usize;
    let mut best_score = usize::MIN;
    for c in 0..classes {
        let mut score = 0usize;
        for j in 0..group {
            score += (out[c * group + j] & 1) as usize;
        }
        if score > best_score {
            best_score = score;
            best = c;
        }
    }
    best
}

/// Distribution of predicted classes, for spotting collapse.
pub fn prediction_histogram(
    circuit: &Circuit,
    keys: &[Vec<u8>],
    classes: usize,
    group: usize,
) -> Vec<usize> {
    let mut hist = vec![0usize; classes];
    let mut planes = vec![0u64; circuit.n_inputs as usize];
    let mut arena = Vec::new();
    let mut out = Vec::new();
    for key in keys {
        for (j, slot) in planes.iter_mut().enumerate() {
            *slot = *key.get(j).unwrap_or(&0) as u64;
        }
        circuit.eval_into(&planes, 1, &mut arena, &mut out);
        hist[argmax_group(&out, classes, group)] += 1;
    }
    hist
}

struct Adam {
    m: Vec<f32>,
    v: Vec<f32>,
    t: usize,
    lr: f32,
}

impl Adam {
    fn new(n: usize, lr: f32) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
            t: 0,
            lr,
        }
    }

    fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        self.t += 1;
        let bc1 = 1.0 - 0.9f32.powi(self.t as i32);
        let bc2 = 1.0 - 0.999f32.powi(self.t as i32);
        for i in 0..params.len() {
            let g = grads[i];
            self.m[i] = 0.9 * self.m[i] + 0.1 * g;
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g;
            params[i] -= self.lr * (self.m[i] / bc1) / ((self.v[i] / bc2).sqrt() + 1e-8);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(inputs: usize, classes: usize) -> NandClassifyConfig {
        NandClassifyConfig {
            inputs,
            classes,
            hidden: vec![16],
            group_size: 2,
            epochs: 1,
            batch: 8,
            lr: 0.01,
            candidates: 4,
            skip_prob: 0.5,
            binarize_weight: 0.0,
            gain_start: 1.0,
            gain_end: 32.0,
            anneal_fraction: 0.85,
            relaxation: Relaxation::Min,
            seed: 1,
        }
    }

    #[test]
    fn materialised_circuit_is_valid_and_shaped() {
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let net = Net::build(&[8, 16, 5], 4, 0.5, &mut rng).unwrap();
        let alpha = vec![0.1f32; net.n_gates * net.k];
        let beta = vec![0.2f32; net.n_gates * net.k];
        let circuit = materialise(&net, &alpha, &beta);
        circuit.validate().unwrap();
        assert_eq!(circuit.n_inputs as usize, 8);
        assert_eq!(circuit.outputs.len(), 5);
        assert_eq!(circuit.gates.len(), 16 + 5);
    }

    #[test]
    fn group_vote_counts_bits_per_class() {
        // Two classes, four output gates each: class 0 has 3 votes, class 1 has 1.
        let out = [1u64, 1, 1, 0, 1, 0, 0, 0];
        assert_eq!(argmax_group(&out, 2, 4), 0);
        // Four votes in the class-1 group wins it.
        let out = [1u64, 1, 1, 0, 1, 1, 1, 1];
        assert_eq!(argmax_group(&out, 2, 4), 1);
        // Ties go to the lower class index.
        let out = [1u64, 0, 0, 0, 1, 0, 0, 0];
        assert_eq!(argmax_group(&out, 2, 4), 0);
    }

    #[test]
    fn training_produces_a_circuit_that_predicts_something() {
        // Two classes, separable by bit 0.
        let mut keys = Vec::new();
        let mut labels = Vec::new();
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        for _ in 0..64 {
            let mut k = vec![0u8; 6];
            for slot in k.iter_mut().skip(1) {
                *slot = u8::from(rng.gen::<bool>());
            }
            let y = u8::from(rng.gen::<bool>());
            k[0] = y;
            keys.push(k);
            labels.push(y);
        }
        let c = cfg(6, 2);
        let (circuit, report) = train(&c, &keys, &labels, &keys, &labels, None).unwrap();
        circuit.validate().unwrap();
        assert!((0.0..=1.0).contains(&report.test_accuracy));
    }
}
