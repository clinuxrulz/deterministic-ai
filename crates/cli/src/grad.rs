//! Gradient-trained gate networks.
//!
//! The GA searches circuit *structure* blindly. This trainer instead fixes a
//! structured, hierarchical connectivity and learns the **gate type** of each
//! wire, using a relaxed (real-valued) gate and an annealed gate softmax. The
//! result is discretised back into literal Boolean gates and exported through
//! the normal `Model` path, so the Rust inference core, the WASM bridge and the
//! web UI are unchanged.
//!
//! The gate set is universal: `Nand` and `Nor` are each functionally complete,
//! so a circuit built only from them can compute any Boolean function. We let
//! the trainer choose among all six binary gates rather than forcing XOR.
//!
//! Reference: Petersen et al., "Deep Differentiable Logic Gate Networks",
//! NeurIPS 2022. We borrow the soft-gate relaxation, but use image-shaped
//! hierarchical wiring instead of random connectivity.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{bail, Context, Result};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use xorae_core::circuit::{Circuit, Gate, Op};
use xorae_core::{BinarizeMode, GateSet, Model, ModelConfig, ModelMetrics};

use crate::dataset::Dataset;

/// Gate types the trainer may choose per wire.
///
/// `Nand` and `Nor` make the set functionally complete on their own.
pub const OPS: [Op; 6] = [Op::Xor, Op::Xnor, Op::And, Op::Nand, Op::Or, Op::Nor];
pub const NOPS: usize = OPS.len();

/// Indices of the non-linear gates inside `OPS`.
const NONLINEAR: [usize; 4] = [2, 3, 4, 5];

/// Activation discretisation used during training.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Squash {
    /// Continuous circuit, no discretisation.
    None,
    /// Hard binarisation with an identity straight-through estimator.
    Hard,
    /// Convex blend from the continuous value to the hard step, annealed over
    /// the tail of training. Unlike a sigmoid this keeps the full `[0, 1]`
    /// range, so stacking layers does not collapse everything toward 0.5.
    #[default]
    Annealed,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GradConfig {
    pub image_size: usize,
    pub key_bits: usize,
    /// Encoder hidden widths. The decoder mirrors them.
    pub hidden: Vec<usize>,
    pub epochs: usize,
    pub batch: usize,
    pub lr: f32,
    /// Fraction of gates whose second input is drawn globally rather than
    /// locally, so information can cross between spatial regions.
    #[serde(default = "default_global_mix")]
    pub global_mix: f32,
    /// Probability that a gate reads from an *earlier* layer instead of the
    /// immediately previous one. This is the skip-connection budget: it is what
    /// lets the circuit reuse distant features, like the GA's arbitrary wiring.
    #[serde(default = "default_skip_prob")]
    pub skip_prob: f32,
    /// Pulls each key bit toward a 50/50 split over the batch, preventing the
    /// encoder from collapsing every image onto the same key.
    #[serde(default = "default_key_balance")]
    pub key_balance_weight: f32,
    /// Weight of the `a(1-a)` penalty that pushes hidden activations toward
    /// 0/1 without a hard step. This is what closes the continuous-to-binary
    /// export gap.
    #[serde(default = "default_binarize_weight")]
    pub binarize_weight: f32,
    /// Weight of the penalty that holds the non-linear gate fraction at or
    /// below `and_budget`. Defaults to off, since the gate mix is free.
    #[serde(default = "default_and_penalty")]
    pub and_penalty_weight: f32,
    /// Optional cap on the non-linear gate fraction (1.0 = no cap).
    #[serde(default = "default_and_budget")]
    pub and_budget: f32,
    #[serde(default)]
    pub squash: Squash,
    /// Hardness at the start of training (0 = continuous, 1 = literal gates).
    #[serde(default = "default_temperature_start")]
    pub temperature_start: f32,
    /// Hardness at the end of annealing.
    #[serde(default = "default_temperature_end")]
    pub temperature_end: f32,
    /// Final fraction of training over which the *activation* discretisation
    /// is annealed. Small values keep the network continuous until the end.
    #[serde(default = "default_anneal_fraction")]
    pub anneal_fraction: f32,
    /// Gate-softmax inverse temperature at the start of training.
    #[serde(default = "default_gate_gain_start")]
    pub gate_gain_start: f32,
    /// Gate-softmax inverse temperature at the end. Large values force every
    /// gate to a single operation, so `argmax` matches the trained forward.
    #[serde(default = "default_gate_gain_end")]
    pub gate_gain_end: f32,
    /// Fraction of training over which the gate softmax is sharpened.
    #[serde(default = "default_gate_anneal_fraction")]
    pub gate_anneal_fraction: f32,
    pub seed: u64,
    #[serde(default = "default_log_every")]
    pub log_every: usize,
    #[serde(default = "default_val_every")]
    pub val_every: usize,
}

fn default_global_mix() -> f32 {
    0.25
}
fn default_skip_prob() -> f32 {
    0.20
}
fn default_key_balance() -> f32 {
    0.05
}
fn default_binarize_weight() -> f32 {
    0.02
}
fn default_and_penalty() -> f32 {
    0.0
}
fn default_and_budget() -> f32 {
    1.0
}
fn default_temperature_start() -> f32 {
    0.0
}
fn default_temperature_end() -> f32 {
    1.0
}
fn default_anneal_fraction() -> f32 {
    0.15
}
fn default_gate_gain_start() -> f32 {
    2.0
}
fn default_gate_gain_end() -> f32 {
    64.0
}
fn default_gate_anneal_fraction() -> f32 {
    0.9
}
fn default_log_every() -> usize {
    10
}
fn default_val_every() -> usize {
    25
}

impl GradConfig {
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
        anyhow::ensure!(
            (0.0..=1.0).contains(&self.global_mix),
            "global_mix must be in 0..=1"
        );
        anyhow::ensure!(
            (0.0..=1.0).contains(&self.and_budget),
            "and_budget must be in 0..=1"
        );
        Ok(())
    }
}

/// One layer of the fixed wiring pattern.
struct Layer {
    n_out: usize,
    /// Source activation plane for each input, as an index into `acts`.
    a_src: Vec<u32>,
    a_idx: Vec<u32>,
    b_src: Vec<u32>,
    b_idx: Vec<u32>,
    /// Index of this layer's first gate in the flat gate list.
    gate_off: usize,
}

struct Net {
    layers: Vec<Layer>,
    n_inputs: usize,
    key_bits: usize,
    enc_layers: usize,
    n_gates: usize,
}

impl Net {
    fn build(cfg: &GradConfig, rng: &mut impl Rng) -> Result<Self> {
        let n_inputs = cfg.image_size * cfg.image_size;
        let side = cfg.image_size;
        let mut sizes = vec![n_inputs];
        sizes.extend(cfg.hidden.iter().copied());
        sizes.push(cfg.key_bits);
        let shapes = derive_shapes(side, &sizes)?;

        let mut layers = Vec::new();
        let mut gate_off = 0usize;
        for i in 0..sizes.len() - 1 {
            let sources: Vec<(usize, (usize, usize))> = (0..=i).map(|k| (k, shapes[k])).collect();
            let (a_src, a_idx, b_src, b_idx) =
                connect(rng, shapes[i + 1], &sources, cfg.skip_prob, cfg.global_mix);
            layers.push(Layer {
                n_out: sizes[i + 1],
                a_src,
                a_idx,
                b_src,
                b_idx,
                gate_off,
            });
            gate_off += sizes[i + 1];
        }
        let enc_layers = layers.len();

        // Decoder mirrors the encoder exactly.
        let mut rev_sizes = sizes.clone();
        rev_sizes.reverse();
        let mut rev_shapes = shapes.clone();
        rev_shapes.reverse();
        for i in 0..rev_sizes.len() - 1 {
            let sources: Vec<(usize, (usize, usize))> =
                (0..=i).map(|k| (enc_layers + k, rev_shapes[k])).collect();
            let (a_src, a_idx, b_src, b_idx) = connect(
                rng,
                rev_shapes[i + 1],
                &sources,
                cfg.skip_prob,
                cfg.global_mix,
            );
            layers.push(Layer {
                n_out: rev_sizes[i + 1],
                a_src,
                a_idx,
                b_src,
                b_idx,
                gate_off,
            });
            gate_off += rev_sizes[i + 1];
        }

        let n_gates = gate_off;
        Ok(Self {
            layers,
            n_inputs,
            key_bits: cfg.key_bits,
            enc_layers,
            n_gates,
        })
    }
}

/// Derive an image shape for every layer by halving the longer axis.
pub(crate) fn derive_shapes(side: usize, sizes: &[usize]) -> Result<Vec<(usize, usize)>> {
    derive_shapes_from((side, side), sizes)
}

/// Same as [`derive_shapes`] but starting from an arbitrary plane shape. This
/// lets planar RGB data be treated as a tall image: 3 channels of `side x side`
/// stack into a `3*side x side` plane.
pub(crate) fn derive_shapes_from(
    initial: (usize, usize),
    sizes: &[usize],
) -> Result<Vec<(usize, usize)>> {
    if initial.0 * initial.1 != sizes[0] {
        bail!(
            "initial shape {}x{} has area {}, but the first layer width is {}",
            initial.0,
            initial.1,
            initial.0 * initial.1,
            sizes[0]
        );
    }
    let mut shapes = vec![initial];
    let mut cur = initial;
    for &target in &sizes[1..] {
        let area = cur.0 * cur.1;
        if target == area {
            // width unchanged
        } else if target * 2 == area {
            cur = if cur.0 >= cur.1 {
                (cur.0 / 2, cur.1)
            } else {
                (cur.0, cur.1 / 2)
            };
        } else {
            bail!(
                "layer width {target} is not a halving of {area}; \
                 choose hidden widths that halve the image area"
            );
        }
        if cur.0 * cur.1 != target {
            bail!("internal shape error for width {target}");
        }
        shapes.push(cur);
    }
    Ok(shapes)
}

/// Build a spatially local wiring diagram between two image-shaped layers.
fn connect(
    rng: &mut impl Rng,
    out_shape: (usize, usize),
    sources: &[(usize, (usize, usize))],
    skip_prob: f32,
    global_mix: f32,
) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
    let (oh, ow) = out_shape;
    let n = oh * ow;
    let mut a_src = Vec::with_capacity(n);
    let mut a_idx = Vec::with_capacity(n);
    let mut b_src = Vec::with_capacity(n);
    let mut b_idx = Vec::with_capacity(n);
    for oy in 0..oh {
        for ox in 0..ow {
            let mut best = (0u32, 0u32, 0u32, 0u32);
            for _ in 0..6 {
                let (sa, ia) = pick_source(rng, oy, ox, out_shape, sources, 0.0, 0.0);
                let (sb, ib) = pick_source(rng, oy, ox, out_shape, sources, skip_prob, global_mix);
                best = (sa, ia, sb, ib);
                if !(sa == sb && ia == ib) {
                    break;
                }
            }
            let (sa, ia, sb, mut ib) = best;
            let mut sb = sb;
            if sa == sb && ia == ib {
                // Force a different wire rather than degenerating to a XOR of
                // a signal with itself.
                let (last_src, last_shape) = sources[sources.len() - 1];
                let m = last_shape.0 * last_shape.1;
                sb = last_src as u32;
                ib = ((ia as usize + 1) % m) as u32;
            }
            a_src.push(sa);
            a_idx.push(ia);
            b_src.push(sb);
            b_idx.push(ib);
        }
    }
    (a_src, a_idx, b_src, b_idx)
}

fn pick_source(
    rng: &mut impl Rng,
    oy: usize,
    ox: usize,
    out_shape: (usize, usize),
    sources: &[(usize, (usize, usize))],
    skip_prob: f32,
    global: f32,
) -> (u32, u32) {
    let (last_src, last_shape) = sources[sources.len() - 1];
    let (src, shape) = if sources.len() > 1 && rng.gen::<f32>() < skip_prob {
        sources[rng.gen_range(0..sources.len() - 1)]
    } else {
        (last_src, last_shape)
    };
    let (ih, iw) = shape;
    let total = ih * iw;
    if global > 0.0 && rng.gen::<f32>() < global {
        return (src as u32, rng.gen_range(0..total) as u32);
    }
    let (y0, y1, x0, x1) = window(oy, ox, shape, out_shape);
    (src as u32, pick(rng, y0, y1, x0, x1, iw))
}

pub(crate) fn window(
    oy: usize,
    ox: usize,
    in_shape: (usize, usize),
    out_shape: (usize, usize),
) -> (usize, usize, usize, usize) {
    let (ih, iw) = in_shape;
    let (oh, ow) = out_shape;
    if oh == ih && ow == iw {
        // Same-resolution layer: read a 3x3 neighbourhood.
        (
            oy.saturating_sub(1),
            (oy + 2).min(ih),
            ox.saturating_sub(1),
            (ox + 2).min(iw),
        )
    } else if oh <= ih && ow <= iw {
        let sy = (ih / oh).max(1);
        let sx = (iw / ow).max(1);
        let y0 = oy * sy;
        let x0 = ox * sx;
        (y0, (y0 + sy).min(ih), x0, (x0 + sx).min(iw))
    } else {
        let iy = oy * ih / oh;
        let ix = ox * iw / ow;
        (
            iy.saturating_sub(1),
            (iy + 2).min(ih),
            ix.saturating_sub(1),
            (ix + 2).min(iw),
        )
    }
}

pub(crate) fn pick(
    rng: &mut impl Rng,
    y0: usize,
    y1: usize,
    x0: usize,
    x1: usize,
    iw: usize,
) -> u32 {
    let w = (x1 - x0).max(1);
    let h = (y1 - y0).max(1);
    let k = rng.gen_range(0..(h * w));
    let yy = y0 + k / w;
    let xx = x0 + k % w;
    (yy * iw + xx) as u32
}

/// Soft (real-valued) gate evaluation on `a, b` in `[0, 1]`.
#[inline]
fn soft_op(k: usize, a: f32, b: f32) -> f32 {
    match k {
        0 => a + b - 2.0 * a * b,       // xor
        1 => 1.0 - a - b + 2.0 * a * b, // xnor
        2 => a * b,                     // and
        3 => 1.0 - a * b,               // nand
        4 => a + b - a * b,             // or
        _ => (1.0 - a) * (1.0 - b),     // nor
    }
}

/// d(soft_op_k)/da. Symmetric in `a` and `b` for every op in `OPS`.
#[inline]
fn soft_op_da(k: usize, a: f32, b: f32) -> f32 {
    let _ = a;
    match k {
        0 => 1.0 - 2.0 * b,
        1 => -1.0 + 2.0 * b,
        2 => b,
        3 => -b,
        4 => 1.0 - b,
        _ => -(1.0 - b),
    }
}

fn softmax(logits: &[f32]) -> [f32; NOPS] {
    let mut max = f32::NEG_INFINITY;
    for &v in logits {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0;
    let mut out = [0.0f32; NOPS];
    for k in 0..NOPS {
        let e = (logits[k] - max).exp();
        out[k] = e;
        sum += e;
    }
    let inv = 1.0 / sum;
    for v in out.iter_mut() {
        *v *= inv;
    }
    out
}

struct Trainer {
    net: Net,
    logits: Vec<f32>,
    probs: Vec<f32>,
    acts: Vec<Vec<f32>>,
    zs: Vec<Vec<f32>>,
    dz: Vec<Vec<f32>>,
    dlogits: Vec<f32>,
    batch: usize,
    key_balance_weight: f32,
    squash: Squash,
    temperature: f32,
    gate_gain: f32,
    binarize_weight: f32,
    and_penalty_weight: f32,
    and_budget: f32,
}

impl Trainer {
    fn new(net: Net, batch: usize, key_balance_weight: f32, rng: &mut impl Rng) -> Self {
        let n_gates = net.n_gates;
        // Start slightly biased toward XOR/XNOR so the initial mixture is not a
        // coin flip between complementary gates.
        let init = [1.0f32, -0.5, -1.0, -1.5, -2.0, -2.5];
        let mut logits = Vec::with_capacity(n_gates * NOPS);
        for _g in 0..n_gates {
            for (k, &base) in init.iter().enumerate() {
                logits.push(base + (rng.gen::<f32>() - 0.5) * 0.05 * (k as f32 + 1.0));
            }
        }
        let layers = net.layers.len();
        let mut acts = Vec::with_capacity(layers + 1);
        let mut zs = Vec::with_capacity(layers);
        let mut dz = Vec::with_capacity(layers);
        for l in 0..=layers {
            let n = if l == 0 {
                net.n_inputs
            } else {
                net.layers[l - 1].n_out
            };
            acts.push(vec![0.0; n * batch]);
        }
        for layer in &net.layers {
            zs.push(vec![0.0; layer.n_out * batch]);
            dz.push(vec![0.0; layer.n_out * batch]);
        }
        Self {
            net,
            logits,
            probs: vec![0.0; n_gates * NOPS],
            acts,
            zs,
            dz,
            dlogits: vec![0.0; n_gates * NOPS],
            batch,
            key_balance_weight,
            squash: Squash::Annealed,
            temperature: 0.0,
            gate_gain: 2.0,
            binarize_weight: 0.0,
            and_penalty_weight: 0.0,
            and_budget: 1.0,
        }
    }

    fn refresh_probs(&mut self) {
        let gain = self.gate_gain;
        let mut scaled = [0.0f32; NOPS];
        for g in 0..self.net.n_gates {
            let base = g * NOPS;
            for (k, v) in scaled.iter_mut().enumerate() {
                *v = self.logits[base + k] * gain;
            }
            let p = softmax(&scaled);
            self.probs[base..base + NOPS].copy_from_slice(&p);
        }
    }

    fn forward(&mut self, input: &[f32]) {
        let b = self.batch;
        self.acts[0].copy_from_slice(input);
        for l in 0..self.net.layers.len() {
            {
                let layer = &self.net.layers[l];
                let probs =
                    &self.probs[layer.gate_off * NOPS..(layer.gate_off + layer.n_out) * NOPS];
                let acts = &self.acts;
                let z = &mut self.zs[l];
                let mut p = [0.0f32; NOPS];
                for j in 0..layer.n_out {
                    let pa = &acts[layer.a_src[j] as usize];
                    let pb = &acts[layer.b_src[j] as usize];
                    let ai = layer.a_idx[j] as usize * b;
                    let bi = layer.b_idx[j] as usize * b;
                    p.copy_from_slice(&probs[j * NOPS..(j + 1) * NOPS]);
                    let o = j * b;
                    for s in 0..b {
                        let av = pa[ai + s];
                        let bv = pb[bi + s];
                        let mut value = 0.0;
                        for (k, &pk) in p.iter().enumerate() {
                            value += pk * soft_op(k, av, bv);
                        }
                        z[o + s] = value;
                    }
                }
            }
            let n_out = self.net.layers[l].n_out;
            let lambda = self.temperature;
            let squash = self.squash;
            let z = &self.zs[l];
            let next = &mut self.acts[l + 1];
            match squash {
                Squash::None => next.copy_from_slice(z),
                Squash::Hard => {
                    for i in 0..n_out * b {
                        next[i] = if z[i] > 0.5 { 1.0 } else { 0.0 };
                    }
                }
                Squash::Annealed => {
                    for i in 0..n_out * b {
                        let hard = if z[i] > 0.5 { 1.0 } else { 0.0 };
                        next[i] = (1.0 - lambda) * z[i] + lambda * hard;
                    }
                }
            }
        }
    }

    fn loss(&self, target: &[f32]) -> f32 {
        let last = self.net.layers.len() - 1;
        let z = &self.zs[last];
        let n = z.len();
        let mut total = 0.0;
        for i in 0..n {
            let d = z[i] - target[i];
            total += d * d;
        }
        let mut total = total / n as f32;
        if self.binarize_weight > 0.0 {
            let mut pen = 0.0;
            let mut count = 0usize;
            for l in 1..=last + 1 {
                for &a in &self.acts[l] {
                    pen += a * (1.0 - a);
                }
                count += self.acts[l].len();
            }
            if count > 0 {
                total += self.binarize_weight * (pen / count as f32);
            }
        }
        total
    }

    fn backward(&mut self, target: &[f32]) {
        let b = self.batch;
        let last = self.net.layers.len() - 1;
        let n_last = self.net.layers[last].n_out;

        for d in self.dz.iter_mut() {
            d.fill(0.0);
        }
        {
            let z = &self.zs[last];
            let dz = &mut self.dz[last];
            let scale = 2.0 / (n_last * b) as f32;
            for i in 0..n_last * b {
                dz[i] += scale * (z[i] - target[i]);
            }
        }
        // Binarisation penalty, pushed back through d(act)/dz = 1.
        if self.binarize_weight > 0.0 {
            let w = self.binarize_weight;
            for l in 0..self.net.layers.len() {
                let n = self.acts[l + 1].len() as f32;
                let scale = w / n;
                let a = &self.acts[l + 1];
                let dz = &mut self.dz[l];
                for i in 0..a.len() {
                    dz[i] += scale * (1.0 - 2.0 * a[i]);
                }
            }
        }
        // Key balance depends only on the encoder output, so add it up front.
        if self.key_balance_weight > 0.0 {
            let enc_last = self.net.enc_layers - 1;
            let n = self.net.key_bits;
            let w = self.key_balance_weight;
            let key = &self.acts[self.net.enc_layers];
            let dz = &mut self.dz[enc_last];
            for k in 0..n {
                let mut sum = 0.0;
                for s in 0..b {
                    sum += key[k * b + s];
                }
                let p = sum / b as f32;
                let g = w * 2.0 * (p - 0.5) / b as f32;
                for s in 0..b {
                    dz[k * b + s] += g;
                }
            }
        }

        for l in (0..=last).rev() {
            let layer = &self.net.layers[l];
            let probs = &self.probs[layer.gate_off * NOPS..(layer.gate_off + layer.n_out) * NOPS];
            let acts = &self.acts;
            let z = &self.zs[l];
            let dlogits = &mut self.dlogits;
            let dz = &mut self.dz;
            // Take this layer's gradient so we can write into earlier layers.
            let dzl = std::mem::take(&mut dz[l]);
            let mut p = [0.0f32; NOPS];
            let mut acc;
            for j in 0..layer.n_out {
                let a_src = layer.a_src[j] as usize;
                let b_src = layer.b_src[j] as usize;
                let ai = layer.a_idx[j] as usize * b;
                let bi = layer.b_idx[j] as usize * b;
                let pa = &acts[a_src];
                let pb = &acts[b_src];
                p.copy_from_slice(&probs[j * NOPS..(j + 1) * NOPS]);
                let gbase = (layer.gate_off + j) * NOPS;
                let o = j * b;
                acc = [0.0; NOPS];
                for s in 0..b {
                    let dzj = dzl[o + s];
                    if dzj == 0.0 {
                        continue;
                    }
                    let av = pa[ai + s];
                    let bv = pb[bi + s];
                    let zj = z[o + s];
                    let mut da = 0.0;
                    let mut db = 0.0;
                    for k in 0..NOPS {
                        let gk = soft_op(k, av, bv);
                        acc[k] += dzj * p[k] * (gk - zj);
                        da += p[k] * soft_op_da(k, av, bv);
                        db += p[k] * soft_op_da(k, bv, av);
                    }
                    if a_src > 0 {
                        dz[a_src - 1][ai + s] += dzj * da;
                    }
                    if b_src > 0 {
                        dz[b_src - 1][bi + s] += dzj * db;
                    }
                }
                for k in 0..NOPS {
                    dlogits[gbase + k] += acc[k];
                }
            }
            dz[l] = dzl;
        }
    }

    /// Hold the non-linear gate fraction at or below the requested budget.
    fn apply_gate_budget(&mut self) {
        let weight = self.and_penalty_weight;
        if weight <= 0.0 || self.and_budget >= 1.0 {
            return;
        }
        let n = self.net.n_gates as f32;
        let mut total = 0.0;
        for g in 0..self.net.n_gates {
            let b = g * NOPS;
            for &k in &NONLINEAR {
                total += self.probs[b + k];
            }
        }
        let mean_nonlinear = total / n;
        if mean_nonlinear <= self.and_budget {
            return;
        }
        let dp = weight / n;
        for g in 0..self.net.n_gates {
            let b = g * NOPS;
            let p = &self.probs[b..b + NOPS];
            let sum: f32 = NONLINEAR.iter().map(|&k| dp * p[k]).sum();
            let gain = self.gate_gain;
            for (k, &pk) in p.iter().enumerate() {
                let dpk = if NONLINEAR.contains(&k) { dp } else { 0.0 };
                self.dlogits[b + k] += gain * pk * (dpk - sum);
            }
        }
    }
}

/// Adam state for the gate logits.
struct Adam {
    m: Vec<f32>,
    v: Vec<f32>,
    t: usize,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
}

impl Adam {
    fn new(n: usize, lr: f32) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
            t: 0,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }

    fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        for i in 0..params.len() {
            let g = grads[i];
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;
            let mhat = self.m[i] / bc1;
            let vhat = self.v[i] / bc2;
            params[i] -= self.lr * mhat / (vhat.sqrt() + self.eps);
        }
    }
}

/// Convert the learned logits into a literal gate circuit.
fn build_circuit(net: &Net, logits: &[f32], and_budget: f32) -> (Circuit, Circuit, [usize; NOPS]) {
    let mut chosen = vec![0usize; net.n_gates];
    let mut margins = vec![0.0f32; net.n_gates];
    for g in 0..net.n_gates {
        let base = g * NOPS;
        let p = softmax(&logits[base..base + NOPS]);
        let mut best = 0usize;
        for k in 1..NOPS {
            if p[k] > p[best] {
                best = k;
            }
        }
        let affine = if p[0] >= p[1] { 0 } else { 1 };
        let mut nonlinear = NONLINEAR[0];
        for &k in &NONLINEAR[1..] {
            if p[k] > p[nonlinear] {
                nonlinear = k;
            }
        }
        let m = p[nonlinear] - p[affine];
        chosen[g] = if m > 0.0 { nonlinear } else { affine };
        margins[g] = m;
    }

    if and_budget < 1.0 {
        let keep = ((net.n_gates as f32) * and_budget).floor() as usize;
        let mut nonlinear: Vec<usize> = (0..net.n_gates)
            .filter(|&g| NONLINEAR.contains(&chosen[g]))
            .collect();
        if nonlinear.len() > keep {
            nonlinear.sort_by(|&x, &y| margins[y].total_cmp(&margins[x]));
            for &g in &nonlinear[keep..] {
                let base = g * NOPS;
                let p = softmax(&logits[base..base + NOPS]);
                chosen[g] = if p[0] >= p[1] { 0 } else { 1 };
            }
        }
    }

    let mut counts = [0usize; NOPS];
    for &c in &chosen {
        counts[c] += 1;
    }

    let encoder = assemble(&net.layers[..net.enc_layers], 0, net.n_inputs, &chosen);
    let decoder = assemble(
        &net.layers[net.enc_layers..],
        net.enc_layers,
        net.key_bits,
        &chosen,
    );
    (encoder, decoder, counts)
}

fn assemble(layers: &[Layer], group_start: usize, n_inputs: usize, chosen: &[usize]) -> Circuit {
    let gate_start = n_inputs + 2;
    let mut local_start = Vec::with_capacity(layers.len());
    let mut total = 0usize;
    for layer in layers {
        local_start.push(total);
        total += layer.n_out;
    }
    let mut gates = Vec::with_capacity(total);
    for layer in layers {
        for j in 0..layer.n_out {
            let g = layer.gate_off + j;
            let wire = |src: u32, idx: u32| -> u16 {
                if src as usize == group_start {
                    idx as u16
                } else {
                    (gate_start + local_start[src as usize - group_start - 1] + idx as usize) as u16
                }
            };
            gates.push(Gate {
                op: OPS[chosen[g]],
                a: wire(layer.a_src[j], layer.a_idx[j]),
                b: wire(layer.b_src[j], layer.b_idx[j]),
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

pub fn run_grad(dataset: &Dataset, cfg: &GradConfig, out_dir: &Path) -> Result<Model> {
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
    if dataset.color_mode != xorae_core::ColorMode::Grayscale {
        bail!("gradient trainer currently supports grayscale datasets only");
    }
    let cols = dataset.cols();
    if cols != cfg.image_size * cfg.image_size {
        bail!("dataset has {cols} bits/image, expected a square image");
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
        "gradient net: {cols} -> {:?} -> {} key bits -> {:?} -> {cols} ({} gates, {} op choices)",
        cfg.hidden, cfg.key_bits, cfg.hidden, n_gates, NOPS
    );

    let mut trainer = Trainer::new(net, cfg.batch, cfg.key_balance_weight, &mut rng);
    trainer.squash = cfg.squash;
    trainer.temperature = cfg.temperature_start;
    trainer.gate_gain = cfg.gate_gain_start;
    trainer.binarize_weight = cfg.binarize_weight;
    trainer.and_penalty_weight = cfg.and_penalty_weight;
    trainer.and_budget = cfg.and_budget;

    let mut adam = Adam::new(n_gates * NOPS, cfg.lr);
    let mut order: Vec<usize> = (0..train_pool.len()).collect();

    let log_file = File::create(out_dir.join("log.csv"))?;
    let mut log = BufWriter::new(log_file);
    writeln!(log, "epoch,step,train_mse,elapsed_ms,temperature,gate_gain")?;

    let batch = cfg.batch;
    let batches_per_epoch = train_pool.len() / batch;
    let mut input = vec![0f32; cols * batch];
    let mut target = vec![0f32; cols * batch];

    let started = std::time::Instant::now();
    let mut step = 0usize;
    for epoch in 0..cfg.epochs {
        {
            // Activation hardness anneals over the tail of training; the gate
            // softmax sharpens over most of the run so the gates are discrete
            // before the activations are.
            let tail = cfg.anneal_fraction.clamp(1e-4, 1.0);
            let start_epoch = (1.0 - tail) * cfg.epochs as f32;
            let span = (tail * cfg.epochs as f32).max(1.0);
            let frac = ((epoch as f32 - start_epoch) / span).clamp(0.0, 1.0);
            trainer.temperature =
                cfg.temperature_start + (cfg.temperature_end - cfg.temperature_start) * frac;

            let gate_span =
                (cfg.gate_anneal_fraction.clamp(1e-4, 1.0) * cfg.epochs as f32).max(1.0);
            let gate_frac = (epoch as f32 / gate_span).clamp(0.0, 1.0);
            let gate_ratio = cfg.gate_gain_end / cfg.gate_gain_start;
            trainer.gate_gain = cfg.gate_gain_start * gate_ratio.powf(gate_frac);
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
            trainer.forward(&input);
            epoch_loss += trainer.loss(&target) as f64;
            epoch_batches += 1;
            trainer.dlogits.fill(0.0);
            trainer.backward(&target);
            trainer.apply_gate_budget();
            adam.step(&mut trainer.logits, &trainer.dlogits);
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
                "{epoch},{step},{:.6},{},{:.3},{:.3}",
                mean,
                started.elapsed().as_millis(),
                trainer.temperature,
                trainer.gate_gain
            )?;
        }
        if cfg.val_every > 0 && (epoch % cfg.val_every == 0 || epoch + 1 == cfg.epochs) {
            let (encoder, decoder, counts) =
                build_circuit(&trainer.net, &trainer.logits, cfg.and_budget);
            let model = make_model(dataset, cfg, encoder, decoder);
            if let Ok(report) = crate::ga::evaluate_model(&model, dataset, &val_pool) {
                eprintln!(
                    "epoch {epoch}: train MSE {:.5}, hard {:.2}, gate gain {:.1}, val BER {:.4}, ops {:?}",
                    mean, trainer.temperature, trainer.gate_gain, report.ber, counts
                );
            }
        }
    }
    log.flush()?;

    let (encoder, decoder, counts) = build_circuit(&trainer.net, &trainer.logits, cfg.and_budget);
    let mut model = make_model(dataset, cfg, encoder, decoder);
    if let Ok(report) = crate::ga::evaluate_model(&model, dataset, &val_pool) {
        model.metrics.val_ber = report.ber;
        model.metrics.key_entropy = report.key_entropy;
    }
    let test_report = crate::ga::evaluate_model(&model, dataset, &test_pool)?;
    model.metrics.test_ber = test_report.ber;
    model.metrics.active_gates = model.encoder.active_gates() + model.decoder.active_gates();
    fs_write_model(out_dir, &model)?;

    eprintln!(
        "done: test BER {:.4}, val BER {:.4}, active gates {}, ops {:?}",
        test_report.ber, model.metrics.val_ber, model.metrics.active_gates, counts
    );
    Ok(model)
}

fn make_model(dataset: &Dataset, cfg: &GradConfig, encoder: Circuit, decoder: Circuit) -> Model {
    let universal = encoder
        .gates
        .iter()
        .chain(decoder.gates.iter())
        .any(|g| matches!(g.op, Op::Nand | Op::Nor | Op::Not));
    let gate_set = if universal {
        GateSet::Universal
    } else {
        GateSet::XorAnd
    };
    Model::new(
        ModelConfig {
            image_size: cfg.image_size,
            key_bits: cfg.key_bits,
            gate_set,
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

fn fs_write_model(out_dir: &Path, model: &Model) -> Result<()> {
    std::fs::write(
        out_dir.join("model.json"),
        model.to_json().map_err(anyhow::Error::msg)?,
    )?;
    std::fs::write(
        out_dir.join("metrics.json"),
        serde_json::to_string_pretty(&model.metrics)?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg() -> GradConfig {
        GradConfig {
            image_size: 8,
            key_bits: 8,
            hidden: vec![32, 16],
            epochs: 1,
            batch: 4,
            lr: 0.01,
            global_mix: 0.25,
            skip_prob: 0.2,
            key_balance_weight: 0.0,
            binarize_weight: 0.0,
            and_penalty_weight: 0.0,
            and_budget: 1.0,
            squash: Squash::Hard,
            temperature_start: 0.0,
            temperature_end: 1.0,
            anneal_fraction: 0.15,
            gate_gain_start: 2.0,
            gate_gain_end: 64.0,
            gate_anneal_fraction: 0.9,
            seed: 7,
            log_every: 0,
            val_every: 0,
        }
    }

    #[test]
    fn shapes_halve_the_longer_axis() {
        let shapes = derive_shapes(32, &[1024, 512, 256, 128]).unwrap();
        assert_eq!(shapes, vec![(32, 32), (16, 32), (16, 16), (8, 16)]);
    }

    #[test]
    fn shapes_reject_non_halving_widths() {
        assert!(derive_shapes(8, &[64, 40, 8]).is_err());
    }

    #[test]
    fn wiring_stays_in_range_and_avoids_self_gates() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let sources = [(0usize, (8usize, 8usize))];
        let (a_src, a_idx, b_src, b_idx) = connect(&mut rng, (4, 8), &sources, 0.0, 0.0);
        assert_eq!(a_idx.len(), 32);
        assert_eq!(b_idx.len(), 32);
        for (i, ((sa, ia), (sb, ib))) in a_src
            .iter()
            .zip(a_idx.iter())
            .zip(b_src.iter().zip(b_idx.iter()))
            .enumerate()
        {
            assert_eq!(*sa, 0);
            assert_eq!(*sb, 0);
            assert!((*ia as usize) < 64, "a {i}");
            assert!((*ib as usize) < 64, "b {i}");
            assert_ne!(ia, ib);
        }
    }

    #[test]
    fn soft_gates_match_hard_gates_on_binary_inputs() {
        for (k, op) in OPS.iter().enumerate() {
            for &a in &[0.0f32, 1.0] {
                for &b in &[0.0f32, 1.0] {
                    let hard = (op.apply(a as u64, b as u64) & 1) as f32;
                    assert!(
                        (soft_op(k, a, b) - hard).abs() < 1e-6,
                        "op {k} on ({a},{b})"
                    );
                }
            }
        }
    }

    #[test]
    fn exported_circuit_matches_trainer_forward() {
        let cfg = tiny_cfg();
        let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);
        let net = Net::build(&cfg, &mut rng).unwrap();
        let n_gates = net.n_gates;

        // One-hot logits make every gate pick OPS[k] deterministically.
        let mut logits = vec![0.0f32; n_gates * NOPS];
        for g in 0..n_gates {
            let k = g % NOPS;
            for j in 0..NOPS {
                logits[g * NOPS + j] = if j == k { 20.0 } else { -20.0 };
            }
        }

        let batch = 1;
        let mut trainer = Trainer::new(net, batch, 0.0, &mut rng);
        trainer.squash = Squash::Hard;
        trainer.logits = logits;
        trainer.refresh_probs();
        let input: Vec<f32> = (0..64)
            .map(|i| if i % 3 == 0 { 1.0 } else { 0.0 })
            .collect();
        trainer.forward(&input);
        let last = trainer.net.layers.len() - 1;
        let expected = trainer.acts[last + 1].clone();

        let (encoder, decoder, _) = build_circuit(&trainer.net, &trainer.logits, cfg.and_budget);
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
