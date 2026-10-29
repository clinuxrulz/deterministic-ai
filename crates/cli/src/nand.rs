//! NAND-only wiring search.
//!
//! Every gate is a NAND. `Nand` is functionally complete, so the gate *type*
//! carries no information: the entire circuit is determined by its wiring. That
//! makes this the minimal possible search space for a fixed gate count.
//!
//! Each gate is given a small candidate set of input wires (a local window plus
//! a few long-range links). Two softmax distributions per gate choose the first
//! and second input. In the forward pass the inputs are convex combinations of
//! the candidates; as the softmax temperature anneals, the combinations become
//! one-hot and the exported circuit is literal `NAND(a, b)` gates.
//!
//! The gradient of the loss w.r.t. the candidate logits is a credit assignment
//! over *wires*, which is exactly what the genetic algorithm has to discover by
//! random mutation.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{bail, Context, Result};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use xorae_core::bits::{plane_hamming, BitBatch};
use xorae_core::circuit::{Circuit, Gate, Op};
use xorae_core::{BinarizeMode, GateSet, Model, ModelConfig, ModelMetrics};

use crate::dataset::Dataset;
use crate::grad::{derive_shapes_from, pick, window};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NandConfig {
    pub image_size: usize,
    pub key_bits: usize,
    /// Encoder hidden widths. The decoder mirrors them.
    pub hidden: Vec<usize>,
    /// Input colour planes. 1 = grayscale, 3 = planar RGB.
    #[serde(default = "default_channels")]
    pub channels: usize,
    pub epochs: usize,
    pub batch: usize,
    pub lr: f32,
    /// Candidates drawn from the local window of the previous layer.
    #[serde(default = "default_n_local")]
    pub n_local: usize,
    /// Candidates drawn from anywhere in an earlier layer.
    #[serde(default = "default_n_global")]
    pub n_global: usize,
    /// Probability that a long-range candidate comes from an earlier layer
    /// rather than the immediately previous one.
    #[serde(default = "default_skip_prob")]
    pub skip_prob: f32,
    /// Weight of the `a(1-a)` penalty that pushes activations toward 0/1.
    #[serde(default = "default_binarize_weight")]
    pub binarize_weight: f32,
    /// Pulls each key bit toward a 50/50 split over the batch.
    #[serde(default = "default_key_balance")]
    pub key_balance_weight: f32,
    /// Candidate-softmax inverse temperature at the start.
    #[serde(default = "default_gain_start")]
    pub gain_start: f32,
    /// Candidate-softmax inverse temperature at the end.
    #[serde(default = "default_gain_end")]
    pub gain_end: f32,
    /// Fraction of training over which the candidate softmax is sharpened.
    #[serde(default = "default_anneal_fraction")]
    pub anneal_fraction: f32,
    /// Greedy sweeps over the discrete wire list after training. Zero disables.
    #[serde(default = "default_polish_sweeps")]
    pub polish_sweeps: usize,
    /// Training samples used to score a candidate wire swap.
    #[serde(default = "default_polish_samples")]
    pub polish_samples: usize,
    pub seed: u64,
    #[serde(default = "default_log_every")]
    pub log_every: usize,
    #[serde(default = "default_val_every")]
    pub val_every: usize,
}

fn default_channels() -> usize {
    1
}
fn default_n_local() -> usize {
    9
}
fn default_n_global() -> usize {
    3
}
fn default_skip_prob() -> f32 {
    0.30
}
fn default_binarize_weight() -> f32 {
    0.05
}
fn default_key_balance() -> f32 {
    0.05
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
fn default_polish_sweeps() -> usize {
    2
}
fn default_polish_samples() -> usize {
    512
}
fn default_log_every() -> usize {
    10
}
fn default_val_every() -> usize {
    25
}

impl NandConfig {
    pub fn from_toml(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config {}", path.display()))?;
        let cfg: Self =
            toml::from_str(&text).with_context(|| format!("parsing config {}", path.display()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(self.image_size > 0, "image_size must be > 0");
        anyhow::ensure!(self.key_bits > 0, "key_bits must be > 0");
        anyhow::ensure!(
            self.key_bits < self.image_size * self.image_size,
            "key must be smaller than the input"
        );
        anyhow::ensure!(!self.hidden.is_empty(), "hidden must not be empty");
        anyhow::ensure!(self.epochs > 0, "epochs must be > 0");
        anyhow::ensure!(self.batch > 0, "batch must be > 0");
        anyhow::ensure!(self.lr > 0.0, "lr must be > 0");
        anyhow::ensure!(self.n_local > 0, "n_local must be > 0");
        anyhow::ensure!(
            (0.0..=1.0).contains(&self.skip_prob),
            "skip_prob must be in 0..=1"
        );
        Ok(())
    }

    pub fn k(&self) -> usize {
        self.n_local + self.n_global
    }
}

/// One layer of NAND gates. Every gate's candidate list has the same length.
struct Layer {
    n_out: usize,
    /// Candidate source plane, as an index into `acts`.
    cand_layer: Vec<u32>,
    /// Candidate index inside that plane.
    cand_idx: Vec<u32>,
    /// Index of this layer's first gate in the flat gate list.
    gate_off: usize,
}

struct Net {
    layers: Vec<Layer>,
    n_inputs: usize,
    key_bits: usize,
    enc_layers: usize,
    n_gates: usize,
    k: usize,
}

impl Net {
    fn build(cfg: &NandConfig, rng: &mut impl Rng) -> Result<Self> {
        let side = cfg.image_size;
        let n_inputs = side * side * cfg.channels.max(1);
        let mut sizes = vec![n_inputs];
        sizes.extend(cfg.hidden.iter().copied());
        sizes.push(cfg.key_bits);
        // Planar colour data stacks into a tall image: 3 planes of side x side.
        let initial = (side * cfg.channels.max(1), side);
        let shapes = derive_shapes_from(initial, &sizes)?;
        let k = cfg.k();

        let mut layers = Vec::new();
        let mut gate_off = 0usize;
        for i in 0..sizes.len() - 1 {
            let out_shape = shapes[i + 1];
            let readable: Vec<(usize, (usize, usize))> = (0..=i).map(|t| (t, shapes[t])).collect();
            let (cand_layer, cand_idx) = wire_candidates(rng, out_shape, &readable, cfg);
            layers.push(Layer {
                n_out: sizes[i + 1],
                cand_layer,
                cand_idx,
                gate_off,
            });
            gate_off += sizes[i + 1];
        }
        let enc_layers = layers.len();

        let mut rev_sizes = sizes.clone();
        rev_sizes.reverse();
        let mut rev_shapes = shapes.clone();
        rev_shapes.reverse();
        for i in 0..rev_sizes.len() - 1 {
            let out_shape = rev_shapes[i + 1];
            let readable: Vec<(usize, (usize, usize))> =
                (0..=i).map(|t| (enc_layers + t, rev_shapes[t])).collect();
            let (cand_layer, cand_idx) = wire_candidates(rng, out_shape, &readable, cfg);
            layers.push(Layer {
                n_out: rev_sizes[i + 1],
                cand_layer,
                cand_idx,
                gate_off,
            });
            gate_off += rev_sizes[i + 1];
        }

        Ok(Self {
            layers,
            n_inputs,
            key_bits: cfg.key_bits,
            enc_layers,
            n_gates: gate_off,
            k,
        })
    }

    /// Number of values in activation plane `plane`.
    #[cfg(test)]
    fn acts_len(&self, plane: usize) -> usize {
        if plane == 0 {
            self.n_inputs
        } else {
            self.layers[plane - 1].n_out
        }
    }
}

/// Build the `k` candidate wires for every gate in one layer.
///
/// The last entry of `readable` is the immediately previous layer.
fn wire_candidates(
    rng: &mut impl Rng,
    out_shape: (usize, usize),
    readable: &[(usize, (usize, usize))],
    cfg: &NandConfig,
) -> (Vec<u32>, Vec<u32>) {
    let (oh, ow) = out_shape;
    let k = cfg.k();
    let (prev_acts, prev_shape) = readable[readable.len() - 1];
    let mut cand_layer = Vec::with_capacity(oh * ow * k);
    let mut cand_idx = Vec::with_capacity(oh * ow * k);
    for oy in 0..oh {
        for ox in 0..ow {
            let (y0, y1, x0, x1) = window(oy, ox, prev_shape, out_shape);
            for _ in 0..cfg.n_local {
                cand_layer.push(prev_acts as u32);
                cand_idx.push(pick(rng, y0, y1, x0, x1, prev_shape.1));
            }
            for _ in 0..cfg.n_global {
                let (src, shape) = if readable.len() > 1 && rng.gen::<f32>() < cfg.skip_prob {
                    readable[rng.gen_range(0..readable.len() - 1)]
                } else {
                    (prev_acts, prev_shape)
                };
                cand_layer.push(src as u32);
                cand_idx.push(rng.gen_range(0..shape.0 * shape.1) as u32);
            }
        }
    }
    (cand_layer, cand_idx)
}

fn softmax_k(logits: &[f32], out: &mut [f32]) {
    let mut max = f32::NEG_INFINITY;
    for &v in logits {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0;
    for (o, &v) in out.iter_mut().zip(logits.iter()) {
        let e = (v - max).exp();
        *o = e;
        sum += e;
    }
    let inv = 1.0 / sum;
    for o in out.iter_mut() {
        *o *= inv;
    }
}

/// Shared parameters for the wiring search.
struct Trainer {
    net: Net,
    /// Wire logits for the first and second NAND input, `n_gates * k` each.
    alpha: Vec<f32>,
    beta: Vec<f32>,
    probs_a: Vec<f32>,
    probs_b: Vec<f32>,
    dalpha: Vec<f32>,
    dbeta: Vec<f32>,
    gain: f32,
    binarize_weight: f32,
}

impl Trainer {
    fn new(net: Net, cfg: &NandConfig, rng: &mut impl Rng) -> Self {
        let n_gates = net.n_gates;
        let k = net.k;
        let mut alpha = vec![0.0f32; n_gates * k];
        let mut beta = vec![0.0f32; n_gates * k];
        // Start with a preferred wire per input so the circuit is a definite
        // (if random) NAND network, instead of a gate that averages everything.
        for g in 0..n_gates {
            let pa = rng.gen_range(0..k);
            let pb = rng.gen_range(0..k);
            for j in 0..k {
                alpha[g * k + j] = if j == pa { 2.0 } else { -2.0 };
                beta[g * k + j] = if j == pb { 2.0 } else { -2.0 };
            }
        }
        Self {
            alpha,
            beta,
            probs_a: vec![0.0; n_gates * k],
            probs_b: vec![0.0; n_gates * k],
            dalpha: vec![0.0; n_gates * k],
            dbeta: vec![0.0; n_gates * k],
            net,
            gain: cfg.gain_start,
            binarize_weight: cfg.binarize_weight,
        }
    }

    fn refresh_probs(&mut self) {
        let k = self.net.k;
        let gain = self.gain;
        let mut scaled = vec![0.0f32; k];
        let n = self.net.n_gates;
        for g in 0..n {
            for (j, v) in scaled.iter_mut().enumerate() {
                *v = self.alpha[g * k + j] * gain;
            }
            softmax_k(&scaled, &mut self.probs_a[g * k..(g + 1) * k]);
            for (j, v) in scaled.iter_mut().enumerate() {
                *v = self.beta[g * k + j] * gain;
            }
            softmax_k(&scaled, &mut self.probs_b[g * k..(g + 1) * k]);
        }
    }

    /// A worker covering the whole batch, used by tests and single-threaded use.
    #[cfg(test)]
    fn full_worker(&self, batch: usize) -> Work {
        let mut w = Work::new(&self.net, batch, 0, batch);
        w.binarize_weight = self.binarize_weight;
        w.key_balance_weight = 0.0;
        w
    }
}

/// Per-thread scratch. Each worker owns a slice of the batch (samples
/// `s0..s1`), with `stride == batch` so the shared plane layout is unchanged.
struct Work {
    stride: usize,
    s0: usize,
    s1: usize,
    acts: Vec<Vec<f32>>,
    zs: Vec<Vec<f32>>,
    dz: Vec<Vec<f32>>,
    da: Vec<f32>,
    db: Vec<f32>,
    loss: f64,
    /// Per-key-bit activation sums over this worker's samples.
    key_sum: Vec<f32>,
    binarize_weight: f32,
    key_balance_weight: f32,
}

impl Work {
    fn new(net: &Net, stride: usize, s0: usize, s1: usize) -> Self {
        let layers = net.layers.len();
        let mut acts = Vec::with_capacity(layers + 1);
        for l in 0..=layers {
            let n = if l == 0 {
                net.n_inputs
            } else {
                net.layers[l - 1].n_out
            };
            acts.push(vec![0.0; n * stride]);
        }
        let mut zs = Vec::with_capacity(layers);
        let mut dz = Vec::with_capacity(layers);
        for layer in &net.layers {
            zs.push(vec![0.0; layer.n_out * stride]);
            dz.push(vec![0.0; layer.n_out * stride]);
        }
        Self {
            stride,
            s0,
            s1,
            acts,
            zs,
            dz,
            da: vec![0.0; net.n_gates * net.k],
            db: vec![0.0; net.n_gates * net.k],
            loss: 0.0,
            key_sum: vec![0.0; net.key_bits],
            binarize_weight: 0.0,
            key_balance_weight: 0.0,
        }
    }

    fn forward(&mut self, net: &Net, pa: &[f32], pb: &[f32], input: &[f32]) {
        let stride = self.stride;
        let (s0, s1) = (self.s0, self.s1);
        let k = net.k;
        self.acts[0].copy_from_slice(input);
        for l in 0..net.layers.len() {
            {
                let layer = &net.layers[l];
                let acts = &self.acts;
                let z = &mut self.zs[l];
                for j in 0..layer.n_out {
                    let g = layer.gate_off + j;
                    let o = j * stride;
                    for s in s0..s1 {
                        let mut u = 0.0;
                        let mut v = 0.0;
                        for c in 0..k {
                            let src = layer.cand_layer[j * k + c] as usize;
                            let idx = layer.cand_idx[j * k + c] as usize;
                            let x = acts[src][idx * stride + s];
                            u += pa[g * k + c] * x;
                            v += pb[g * k + c] * x;
                        }
                        z[o + s] = 1.0 - u * v;
                    }
                }
            }
            let z = &self.zs[l];
            let next = &mut self.acts[l + 1];
            for j in 0..net.layers[l].n_out {
                let o = j * stride;
                next[o + s0..o + s1].copy_from_slice(&z[o + s0..o + s1]);
            }
        }
    }

    fn loss(
        &mut self,
        net: &Net,
        target: &[f32],
        binarize_weight: f32,
        total_samples: usize,
    ) -> f32 {
        let stride = self.stride;
        let (s0, s1) = (self.s0, self.s1);
        let last = net.layers.len() - 1;
        let z = &self.zs[last];
        // Normalise by the global batch size so per-worker partial sums add up
        // to exactly the single-threaded loss, whatever the split.
        let n_global = (net.layers[last].n_out * total_samples).max(1) as f32;
        let mut total = 0.0;
        for j in 0..net.layers[last].n_out {
            let o = j * stride;
            for s in s0..s1 {
                let d = z[o + s] - target[o + s];
                total += d * d;
            }
        }
        let mut total = total / n_global;
        if binarize_weight > 0.0 {
            let mut pen = 0.0;
            for l in 1..=last + 1 {
                let plane = self.acts[l].len() / stride;
                for i in 0..plane {
                    for s in s0..s1 {
                        let a = self.acts[l][i * stride + s];
                        pen += a * (1.0 - a);
                    }
                }
            }
            let mut planes = 0usize;
            for l in 1..=last + 1 {
                planes += self.acts[l].len() / stride;
            }
            let denom = (planes * total_samples).max(1) as f32;
            total += binarize_weight * (pen / denom);
        }
        self.loss = total as f64;
        total
    }

    /// `key_mean[k]` is the global mean of key bit `k` across the whole batch,
    /// so the balance gradient is identical regardless of how the batch is
    /// split between workers.
    fn backward(
        &mut self,
        net: &Net,
        pa: &[f32],
        pb: &[f32],
        target: &[f32],
        key_mean: &[f32],
        total_samples: usize,
    ) {
        let stride = self.stride;
        let (s0, s1) = (self.s0, self.s1);
        let k = net.k;
        let last = net.layers.len() - 1;

        for d in self.dz.iter_mut() {
            d.fill(0.0);
        }
        self.da.fill(0.0);
        self.db.fill(0.0);
        {
            let n_last = net.layers[last].n_out;
            let scale = 2.0 / (n_last * total_samples).max(1) as f32;
            let z = &self.zs[last];
            let dz = &mut self.dz[last];
            for j in 0..n_last {
                let o = j * stride;
                for s in s0..s1 {
                    dz[o + s] += scale * (z[o + s] - target[o + s]);
                }
            }
        }

        let kbw = self.key_balance_weight;
        if kbw > 0.0 {
            let enc_last = net.enc_layers - 1;
            let n = net.key_bits;
            let dz = &mut self.dz[enc_last];
            for bit in 0..n {
                let g = kbw * 2.0 * (key_mean[bit] - 0.5) / total_samples as f32;
                for s in s0..s1 {
                    dz[bit * stride + s] += g;
                }
            }
        }

        let bw = self.binarize_weight;
        if bw > 0.0 {
            for l in 0..net.layers.len() {
                let plane = self.acts[l + 1].len() / stride;
                let scale = bw / (plane * total_samples).max(1) as f32;
                let a = &self.acts[l + 1];
                let dz = &mut self.dz[l];
                for i in 0..plane {
                    for s in s0..s1 {
                        dz[i * stride + s] += scale * (1.0 - 2.0 * a[i * stride + s]);
                    }
                }
            }
        }

        for l in (0..=last).rev() {
            let layer = &net.layers[l];
            let acts = &self.acts;
            let dz = &mut self.dz;
            let dalpha = &mut self.da;
            let dbeta = &mut self.db;
            let dzl = std::mem::take(&mut dz[l]);
            for j in 0..layer.n_out {
                let g = layer.gate_off + j;
                let o = j * stride;
                for s in s0..s1 {
                    let dzj = dzl[o + s];
                    if dzj == 0.0 {
                        continue;
                    }
                    let mut u = 0.0;
                    let mut v = 0.0;
                    for c in 0..k {
                        let src = layer.cand_layer[j * k + c] as usize;
                        let idx = layer.cand_idx[j * k + c] as usize;
                        let x = acts[src][idx * stride + s];
                        u += pa[g * k + c] * x;
                        v += pb[g * k + c] * x;
                    }
                    let du = dzj * (-v);
                    let dv = dzj * (-u);
                    for c in 0..k {
                        let src = layer.cand_layer[j * k + c] as usize;
                        let idx = layer.cand_idx[j * k + c] as usize;
                        let x = acts[src][idx * stride + s];
                        dalpha[g * k + c] += du * x;
                        dbeta[g * k + c] += dv * x;
                        if src > 0 {
                            let contrib = du * pa[g * k + c] + dv * pb[g * k + c];
                            dz[src - 1][idx * stride + s] += contrib;
                        }
                    }
                }
            }
            dz[l] = dzl;
        }
    }

    /// Sum this worker's key activations into `key_sum`.
    fn accumulate_key(&mut self, net: &Net) {
        let stride = self.stride;
        let key = &self.acts[net.enc_layers];
        for bit in 0..net.key_bits {
            let mut sum = 0.0;
            for s in self.s0..self.s1 {
                sum += key[bit * stride + s];
            }
            self.key_sum[bit] = sum;
        }
    }
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
            let mhat = self.m[i] / bc1;
            let vhat = self.v[i] / bc2;
            params[i] -= self.lr * mhat / (vhat.sqrt() + 1e-8);
        }
    }
}

/// Materialise the wiring as literal NAND gates.
fn build_circuit(net: &Net, alpha: &[f32], beta: &[f32]) -> (Circuit, Circuit) {
    let k = net.k;
    let mut wire_a = vec![0usize; net.n_gates];
    let mut wire_b = vec![0usize; net.n_gates];
    for g in 0..net.n_gates {
        wire_a[g] = argmax(&alpha[g * k..(g + 1) * k]);
        wire_b[g] = argmax(&beta[g * k..(g + 1) * k]);
    }
    let encoder = assemble(
        &net.layers[..net.enc_layers],
        0,
        net.n_inputs,
        &wire_a,
        &wire_b,
    );
    let decoder = assemble(
        &net.layers[net.enc_layers..],
        net.enc_layers,
        net.key_bits,
        &wire_a,
        &wire_b,
    );
    (encoder, decoder)
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

fn assemble(
    layers: &[Layer],
    group_start: usize,
    n_inputs: usize,
    wire_a: &[usize],
    wire_b: &[usize],
) -> Circuit {
    let gate_start = n_inputs + 2;
    let mut local_start = Vec::with_capacity(layers.len());
    let mut total = 0usize;
    for layer in layers {
        local_start.push(total);
        total += layer.n_out;
    }
    let k = if layers.is_empty() {
        0
    } else {
        layers[0].cand_layer.len() / layers[0].n_out.max(1)
    };
    let mut gates = Vec::with_capacity(total);
    for layer in layers {
        for j in 0..layer.n_out {
            let g = layer.gate_off + j;
            let wire = |choice: usize| -> u16 {
                let src = layer.cand_layer[j * k + choice];
                let idx = layer.cand_idx[j * k + choice];
                if src as usize == group_start {
                    idx as u16
                } else {
                    (gate_start + local_start[src as usize - group_start - 1] + idx as usize) as u16
                }
            };
            gates.push(Gate {
                op: Op::Nand,
                a: wire(wire_a[g]),
                b: wire(wire_b[g]),
            });
        }
    }
    let out_start = local_start.last().copied().unwrap_or(0);
    let n_out = layers.last().map(|l| l.n_out).unwrap_or(0);
    let outputs: Vec<u16> = (0..n_out)
        .map(|j| (gate_start + out_start + j) as u16)
        .collect();
    Circuit {
        n_inputs: n_inputs as u16,
        gates,
        outputs,
    }
}

/// Absolute wire indices of every candidate for every gate, in circuit order.
fn candidate_wires(
    layers: &[Layer],
    group_start: usize,
    n_inputs: usize,
    k: usize,
) -> Vec<Vec<u16>> {
    let gate_start = n_inputs + 2;
    let mut local_start = Vec::with_capacity(layers.len());
    let mut total = 0usize;
    for layer in layers {
        local_start.push(total);
        total += layer.n_out;
    }
    let mut out = Vec::with_capacity(total);
    for layer in layers {
        for j in 0..layer.n_out {
            let mut wires = Vec::with_capacity(k);
            for c in 0..k {
                let src = layer.cand_layer[j * k + c] as usize;
                let idx = layer.cand_idx[j * k + c] as usize;
                let w = if src == group_start {
                    idx
                } else {
                    gate_start + local_start[src - group_start - 1] + idx
                };
                wires.push(w as u16);
            }
            out.push(wires);
        }
    }
    out
}

/// Fast BER evaluator with reusable arenas.
struct Evaluator {
    batch: BitBatch,
    enc_arena: Vec<u64>,
    key_planes: Vec<u64>,
    dec_arena: Vec<u64>,
    recon: Vec<u64>,
    denom: f32,
}

impl Evaluator {
    fn new(batch: BitBatch) -> Self {
        let denom = (batch.len.max(1) * batch.cols.max(1)) as f32;
        Self {
            batch,
            enc_arena: Vec::new(),
            key_planes: Vec::new(),
            dec_arena: Vec::new(),
            recon: Vec::new(),
            denom,
        }
    }

    fn ber(&mut self, encoder: &Circuit, decoder: &Circuit) -> f32 {
        encoder.eval_into(
            &self.batch.planes,
            self.batch.words,
            &mut self.enc_arena,
            &mut self.key_planes,
        );
        decoder.eval_into(
            &self.key_planes,
            self.batch.words,
            &mut self.dec_arena,
            &mut self.recon,
        );
        let errors = plane_hamming(
            &self.recon,
            &self.batch.planes,
            self.batch.cols,
            self.batch.words,
            self.batch.len,
        );
        errors as f32 / self.denom
    }
}

/// Greedy hill-climb over the wire list.
///
/// Each move re-points one gate input at another of its trained candidates and
/// keeps the change if the measured BER improves. This optimises the real
/// discrete objective, which the soft relaxation can only approximate.
fn polish(
    encoder: &mut Circuit,
    decoder: &mut Circuit,
    enc_cands: &[Vec<u16>],
    dec_cands: &[Vec<u16>],
    eval: &mut Evaluator,
    sweeps: usize,
) -> f32 {
    let mut best = eval.ber(encoder, decoder);
    for sweep in 0..sweeps {
        let mut improved = 0usize;

        let base = encoder.gate_start();
        for (i, cand) in enc_cands.iter().enumerate() {
            let limit = base + i;
            for side in 0..2 {
                let orig = if side == 0 {
                    encoder.gates[i].a
                } else {
                    encoder.gates[i].b
                };
                for &w in cand {
                    if w == orig || w as usize >= limit {
                        continue;
                    }
                    if side == 0 {
                        encoder.gates[i].a = w;
                    } else {
                        encoder.gates[i].b = w;
                    }
                    let score = eval.ber(encoder, decoder);
                    if score + 1e-9 < best {
                        best = score;
                        improved += 1;
                    } else if side == 0 {
                        encoder.gates[i].a = orig;
                    } else {
                        encoder.gates[i].b = orig;
                    }
                }
            }
        }

        let base = decoder.gate_start();
        for (i, cand) in dec_cands.iter().enumerate() {
            let limit = base + i;
            for side in 0..2 {
                let orig = if side == 0 {
                    decoder.gates[i].a
                } else {
                    decoder.gates[i].b
                };
                for &w in cand {
                    if w == orig || w as usize >= limit {
                        continue;
                    }
                    if side == 0 {
                        decoder.gates[i].a = w;
                    } else {
                        decoder.gates[i].b = w;
                    }
                    let score = eval.ber(encoder, decoder);
                    if score + 1e-9 < best {
                        best = score;
                        improved += 1;
                    } else if side == 0 {
                        decoder.gates[i].a = orig;
                    } else {
                        decoder.gates[i].b = orig;
                    }
                }
            }
        }

        eprintln!("polish sweep {sweep}: BER {best:.4}, {improved} accepted moves");
        if improved == 0 {
            break;
        }
    }
    best
}

pub fn run_nand(dataset: &Dataset, cfg: &NandConfig, out_dir: &Path) -> Result<Model> {
    cfg.validate()?;
    if dataset.size != cfg.image_size {
        bail!(
            "dataset is {}x{}, config expects {}x{}",
            dataset.size,
            dataset.size,
            cfg.image_size,
            cfg.image_size
        );
    }
    let cols = dataset.cols();
    let expected = cfg.image_size * cfg.image_size * cfg.channels.max(1);
    if cols != expected {
        bail!("dataset has {cols} bits/image, config expects {expected}");
    }
    if dataset.channels != cfg.channels.max(1) {
        bail!(
            "dataset has {} channel(s), config says {}",
            dataset.channels,
            cfg.channels
        );
    }

    std::fs::create_dir_all(out_dir)
        .with_context(|| format!("creating run directory {}", out_dir.display()))?;

    let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);
    let (train_pool, val_pool, test_pool) = dataset.split_indices(cfg.seed);
    if train_pool.is_empty() {
        bail!("no training samples available");
    }

    let mut x = vec![0f32; train_pool.len() * cols];
    for (i, &idx) in train_pool.iter().enumerate() {
        for c in 0..cols {
            x[i * cols + c] = u8::from(dataset.bits.get(idx, c)) as f32;
        }
    }

    let net = Net::build(cfg, &mut rng)?;
    let n_gates = net.n_gates;
    eprintln!(
        "nand net: {cols} -> {:?} -> {} key bits -> {:?} -> {cols} ({} NAND gates, {} candidates/gate)",
        cfg.hidden, cfg.key_bits, cfg.hidden, n_gates, cfg.k()
    );

    let mut trainer = Trainer::new(net, cfg, &mut rng);
    let mut adam_a = Adam::new(n_gates * cfg.k(), cfg.lr);
    let mut adam_b = Adam::new(n_gates * cfg.k(), cfg.lr);
    let mut order: Vec<usize> = (0..train_pool.len()).collect();

    // Split each batch across threads. Each worker owns its own scratch and
    // gradient buffers; the main thread reduces them before the Adam step.
    let n_workers = rayon::current_num_threads().clamp(1, 8);
    let chunk = cfg.batch.div_ceil(n_workers);
    let mut pool: Vec<Work> = (0..n_workers)
        .filter_map(|i| {
            let s0 = (i * chunk).min(cfg.batch);
            let s1 = ((i + 1) * chunk).min(cfg.batch);
            if s1 > s0 {
                let mut w = Work::new(&trainer.net, cfg.batch, s0, s1);
                w.binarize_weight = cfg.binarize_weight;
                w.key_balance_weight = cfg.key_balance_weight;
                Some(w)
            } else {
                None
            }
        })
        .collect();
    eprintln!(
        "using {} worker thread(s) for gradient evaluation",
        pool.len()
    );
    let mut key_totals = vec![0f32; trainer.net.key_bits];
    let mut key_mean = vec![0f32; trainer.net.key_bits];

    let log_file = File::create(out_dir.join("log.csv"))?;
    let mut log = BufWriter::new(log_file);
    writeln!(log, "epoch,step,train_mse,elapsed_ms,gain")?;

    let batch = cfg.batch;
    let batches_per_epoch = train_pool.len() / batch;
    let mut input = vec![0f32; cols * batch];
    let mut target = vec![0f32; cols * batch];

    let started = std::time::Instant::now();
    let mut step = 0usize;
    for epoch in 0..cfg.epochs {
        {
            let span = (cfg.anneal_fraction.clamp(1e-4, 1.0) * cfg.epochs as f32).max(1.0);
            let frac = (epoch as f32 / span).clamp(0.0, 1.0);
            let ratio = cfg.gain_end / cfg.gain_start;
            trainer.gain = cfg.gain_start * ratio.powf(frac);
        }
        order.shuffle(&mut rng);
        let mut epoch_loss = 0.0f64;
        let mut epoch_batches = 0usize;
        for bi in 0..batches_per_epoch {
            for s in 0..batch {
                let row = order[bi * batch + s];
                let src = row * cols;
                for c in 0..cols {
                    let v = x[src + c];
                    input[c * batch + s] = v;
                    target[c * batch + s] = v;
                }
            }
            trainer.refresh_probs();
            pool.par_iter_mut().for_each(|w| {
                w.forward(&trainer.net, &trainer.probs_a, &trainer.probs_b, &input);
                w.loss(&trainer.net, &target, trainer.binarize_weight, batch);
                w.accumulate_key(&trainer.net);
            });
            key_totals.fill(0.0);
            for w in &pool {
                for (t, v) in key_totals.iter_mut().zip(w.key_sum.iter()) {
                    *t += v;
                }
            }
            for (m, t) in key_mean.iter_mut().zip(key_totals.iter()) {
                *m = t / batch as f32;
            }
            pool.par_iter_mut().for_each(|w| {
                w.backward(
                    &trainer.net,
                    &trainer.probs_a,
                    &trainer.probs_b,
                    &target,
                    &key_mean,
                    batch,
                );
            });
            trainer.dalpha.fill(0.0);
            trainer.dbeta.fill(0.0);
            for w in &pool {
                for (a, b) in trainer.dalpha.iter_mut().zip(w.da.iter()) {
                    *a += b;
                }
                for (a, b) in trainer.dbeta.iter_mut().zip(w.db.iter()) {
                    *a += b;
                }
                epoch_loss += w.loss;
            }
            epoch_batches += 1;
            adam_a.step(&mut trainer.alpha, &trainer.dalpha);
            adam_b.step(&mut trainer.beta, &trainer.dbeta);
            step += 1;
        }
        let mean = if epoch_batches > 0 {
            epoch_loss / epoch_batches as f64
        } else {
            0.0
        };
        if cfg.log_every > 0 && (epoch % cfg.log_every == 0 || epoch + 1 == cfg.epochs) {
            writeln!(
                log,
                "{epoch},{step},{:.6},{},{:.3}",
                mean,
                started.elapsed().as_millis(),
                trainer.gain
            )?;
            log.flush()?;
        }
        if cfg.val_every > 0 && (epoch % cfg.val_every == 0 || epoch + 1 == cfg.epochs) {
            let (encoder, decoder) = build_circuit(&trainer.net, &trainer.alpha, &trainer.beta);
            let model = make_model(dataset, cfg, encoder, decoder);
            if let Ok(report) = crate::ga::evaluate_model(&model, dataset, &val_pool) {
                eprintln!(
                    "epoch {epoch}: train MSE {:.5}, gain {:.1}, val BER {:.4}",
                    mean, trainer.gain, report.ber
                );
            }
        }
    }
    log.flush()?;

    let (mut encoder, mut decoder) = build_circuit(&trainer.net, &trainer.alpha, &trainer.beta);
    if cfg.polish_sweeps > 0 {
        let k = trainer.net.k;
        let enc_cands = candidate_wires(
            &trainer.net.layers[..trainer.net.enc_layers],
            0,
            trainer.net.n_inputs,
            k,
        );
        let dec_cands = candidate_wires(
            &trainer.net.layers[trainer.net.enc_layers..],
            trainer.net.enc_layers,
            trainer.net.key_bits,
            k,
        );
        let n = cfg.polish_samples.min(train_pool.len()).max(1);
        let idxs: Vec<usize> = (0..n)
            .map(|i| train_pool[i * train_pool.len() / n])
            .collect();
        let batch = BitBatch::from_matrix(&dataset.bits, &idxs);
        let mut eval = Evaluator::new(batch);
        let before = eval.ber(&encoder, &decoder);
        eprintln!("polishing wire list on {} training samples", idxs.len());
        let after = polish(
            &mut encoder,
            &mut decoder,
            &enc_cands,
            &dec_cands,
            &mut eval,
            cfg.polish_sweeps,
        );
        eprintln!("polish: {before:.4} -> {after:.4} on the polish sample");
        encoder.validate().map_err(anyhow::Error::msg)?;
        decoder.validate().map_err(anyhow::Error::msg)?;
    }
    let before = encoder.gates.len() + decoder.gates.len();
    encoder = encoder.prune_dead();
    decoder = decoder.prune_dead();
    let after = encoder.gates.len() + decoder.gates.len();
    eprintln!(
        "pruned dead gates: {before} -> {after} ({} removed)",
        before - after
    );

    let mut model = make_model(dataset, cfg, encoder, decoder);
    if let Ok(report) = crate::ga::evaluate_model(&model, dataset, &val_pool) {
        model.metrics.val_ber = report.ber;
        model.metrics.key_entropy = report.key_entropy;
    }
    let test_report = crate::ga::evaluate_model(&model, dataset, &test_pool)?;
    model.metrics.test_ber = test_report.ber;
    model.metrics.active_gates = model.encoder.active_gates() + model.decoder.active_gates();
    std::fs::write(
        out_dir.join("model.json"),
        model.to_json().map_err(anyhow::Error::msg)?,
    )?;
    std::fs::write(
        out_dir.join("metrics.json"),
        serde_json::to_string_pretty(&model.metrics)?,
    )?;

    eprintln!(
        "done: test BER {:.4}, val BER {:.4}, active gates {}",
        test_report.ber, model.metrics.val_ber, model.metrics.active_gates
    );
    Ok(model)
}

fn make_model(dataset: &Dataset, cfg: &NandConfig, encoder: Circuit, decoder: Circuit) -> Model {
    Model::new(
        ModelConfig {
            image_size: cfg.image_size,
            key_bits: cfg.key_bits,
            gate_set: GateSet::NandOnly,
            binarize: BinarizeMode::Threshold,
            threshold: 128,
            seed: cfg.seed,
            color_mode: dataset.color_mode,
            palette: dataset.palette.clone(),
            bits_per_pixel: dataset.bits_per_pixel,
            channels: dataset.channels,
        },
        encoder,
        decoder,
        ModelMetrics::default(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg() -> NandConfig {
        NandConfig {
            image_size: 8,
            key_bits: 8,
            hidden: vec![32, 16],
            channels: 1,
            epochs: 1,
            batch: 2,
            lr: 0.01,
            n_local: 4,
            n_global: 2,
            skip_prob: 0.3,
            binarize_weight: 0.0,
            key_balance_weight: 0.0,
            gain_start: 1.0,
            gain_end: 32.0,
            anneal_fraction: 0.85,
            polish_sweeps: 0,
            polish_samples: 32,
            seed: 3,
            log_every: 0,
            val_every: 0,
        }
    }

    #[test]
    fn candidates_are_in_range() {
        let cfg = tiny_cfg();
        let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);
        let net = Net::build(&cfg, &mut rng).unwrap();
        let k = net.k;
        for (l, layer) in net.layers.iter().enumerate() {
            for j in 0..layer.n_out {
                for c in 0..k {
                    let src = layer.cand_layer[j * k + c] as usize;
                    let idx = layer.cand_idx[j * k + c] as usize;
                    // Sources must be readable planes, never a later one.
                    assert!(src <= l, "layer {l} gate {j} reads plane {src}");
                    assert!(idx < net.acts_len(src), "index out of range");
                }
            }
        }
    }

    #[test]
    fn exported_circuit_matches_trainer_forward() {
        let cfg = tiny_cfg();
        let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);
        let net = Net::build(&cfg, &mut rng).unwrap();
        let batch = 1;
        let mut trainer = Trainer::new(net, &cfg, &mut rng);
        // Force one-hot candidate choices.
        let k = trainer.net.k;
        for g in 0..trainer.net.n_gates {
            let pa = g % k;
            let pb = (g + 1) % k;
            for j in 0..k {
                trainer.alpha[g * k + j] = if j == pa { 40.0 } else { -40.0 };
                trainer.beta[g * k + j] = if j == pb { 40.0 } else { -40.0 };
            }
        }
        trainer.gain = 1.0;
        trainer.refresh_probs();
        let input: Vec<f32> = (0..64)
            .map(|i| if i % 3 == 0 { 1.0 } else { 0.0 })
            .collect();
        let mut work = trainer.full_worker(batch);
        work.forward(&trainer.net, &trainer.probs_a, &trainer.probs_b, &input);
        let last = trainer.net.layers.len() - 1;
        let expected = work.acts[last + 1].clone();

        let (encoder, decoder) = build_circuit(&trainer.net, &trainer.alpha, &trainer.beta);
        let input_planes: Vec<u64> = input.iter().map(|&v| v as u64).collect();
        let mut arena = Vec::new();
        let mut key = Vec::new();
        encoder.eval_into(&input_planes, 1, &mut arena, &mut key);
        assert_eq!(key.len(), cfg.key_bits);

        let mut arena = Vec::new();
        let mut out = Vec::new();
        decoder.eval_into(&key, 1, &mut arena, &mut out);
        let got: Vec<f32> = out.iter().map(|v| (v & 1) as f32).collect();
        assert_eq!(got, expected);
    }
}
