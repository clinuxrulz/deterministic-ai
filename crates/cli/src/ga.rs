use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use rand::seq::index;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use xorae_core::bits::{plane_hamming, BitBatch};
use xorae_core::color::{psnr_from_mse, ColorMode};
use xorae_core::{Circuit, GateSet, Model, ModelConfig, ModelMetrics};

use crate::config::TrainConfig;
use crate::dataset::Dataset;

#[derive(Clone, Debug)]
pub struct Individual {
    pub encoder: Circuit,
    pub decoder: Circuit,
    pub fitness: f32,
    pub ber: f32,
    pub color_mae: f32,
    pub index_accuracy: f32,
    pub key_entropy: f32,
    pub key_uniqueness: f32,
    pub active_gates: usize,
}

#[derive(Clone, Debug, Default)]
pub struct EvalReport {
    pub fitness: f32,
    pub ber: f32,
    pub color_mae: f32,
    pub color_mse: f32,
    pub psnr: f32,
    pub index_accuracy: f32,
    /// Mean absolute RGB error against the original image (colour only).
    pub abs_mae: f32,
    pub key_entropy: f32,
    pub key_uniqueness: f32,
    pub active_gates: usize,
    pub exact_rate: f32,
    /// F1 and IoU over the *set* bits. On a sparse target BER is dominated by
    /// the zeros -- at 15% density an all-zero image scores BER 0.85 -- so
    /// these judge the strokes instead of the empty space.
    pub ones_f1: f32,
    pub ones_iou: f32,
    pub n: usize,
}

impl Individual {
    pub fn random(cfg: &TrainConfig, input_bits: usize, rng: &mut impl Rng) -> Self {
        let enc_seed = rng.gen::<u64>();
        let dec_seed = rng.gen::<u64>();
        Self {
            encoder: Circuit::random(
                input_bits,
                cfg.encoder_gates,
                cfg.key_bits,
                cfg.gate_set,
                enc_seed,
            ),
            decoder: Circuit::random(
                cfg.key_bits,
                cfg.decoder_gates,
                input_bits,
                cfg.gate_set,
                dec_seed,
            ),
            fitness: f32::INFINITY,
            ber: 1.0,
            color_mae: 1.0,
            index_accuracy: 0.0,
            key_entropy: 0.0,
            key_uniqueness: 0.0,
            active_gates: 0,
        }
    }

    fn apply(&mut self, report: &EvalReport) {
        self.fitness = report.fitness;
        self.ber = report.ber;
        self.color_mae = report.color_mae;
        self.index_accuracy = report.index_accuracy;
        self.key_entropy = report.key_entropy;
        self.key_uniqueness = report.key_uniqueness;
        self.active_gates = report.active_gates;
    }
}

/// Evaluate one individual on a fixed batch.
pub fn evaluate(
    ind: &Individual,
    batch: &BitBatch,
    cfg: &TrainConfig,
    dataset: &Dataset,
) -> EvalReport {
    let mut enc_arena = Vec::new();
    let mut key_planes = Vec::new();
    ind.encoder
        .eval_into(&batch.planes, batch.words, &mut enc_arena, &mut key_planes);

    let mut dec_arena = Vec::new();
    let mut recon = Vec::new();
    ind.decoder
        .eval_into(&key_planes, batch.words, &mut dec_arena, &mut recon);

    match dataset.color_mode {
        ColorMode::Grayscale => report_from_planes(&recon, &key_planes, batch, cfg),
        ColorMode::Palette => report_from_planes_color(&recon, &key_planes, batch, cfg, dataset),
        ColorMode::Rgb1 | ColorMode::Rgb2 => {
            // For RGB bit-plane models the bit error and the colour error are
            // the same quantity. Use the fast bit-parallel path during
            // evolution; pixel accuracy and level-aware metrics are computed at
            // evaluation time.
            let mut report = report_from_planes(&recon, &key_planes, batch, cfg);
            report.color_mae = report.ber;
            report.color_mse = report.ber;
            report.psnr = psnr_from_mse(report.ber);
            report
        }
    }
}

fn report_from_planes(
    recon: &[u64],
    key_planes: &[u64],
    batch: &BitBatch,
    cfg: &TrainConfig,
) -> EvalReport {
    let errors = plane_hamming(recon, &batch.planes, batch.cols, batch.words, batch.len);
    let denom = (batch.len.max(1) * batch.cols.max(1)) as f32;
    let ber = errors as f32 / denom;
    let entropy = key_entropy(key_planes, cfg.key_bits, batch.words, batch.len);
    let uniqueness = key_uniqueness(key_planes, cfg.key_bits, batch.words, batch.len);
    let fitness =
        ber + cfg.entropy_weight * (1.0 - entropy) + cfg.uniqueness_weight * (1.0 - uniqueness);
    let (ones_f1, ones_iou) = ones_scores(recon, batch);
    EvalReport {
        fitness,
        ber,
        ones_f1,
        ones_iou,
        ..Default::default()
    }
    .with_common(entropy, uniqueness, batch.len)
}

fn report_from_planes_color(
    recon: &[u64],
    key_planes: &[u64],
    batch: &BitBatch,
    cfg: &TrainConfig,
    dataset: &Dataset,
) -> EvalReport {
    let (color_mae, color_mse, psnr, index_accuracy, exact_images) =
        color_metrics(recon, batch, dataset);
    let errors = plane_hamming(recon, &batch.planes, batch.cols, batch.words, batch.len);
    let ber = errors as f32 / (batch.len.max(1) * batch.cols.max(1)) as f32;
    let entropy = key_entropy(key_planes, cfg.key_bits, batch.words, batch.len);
    let uniqueness = key_uniqueness(key_planes, cfg.key_bits, batch.words, batch.len);
    let fitness = color_mae
        + cfg.entropy_weight * (1.0 - entropy)
        + cfg.uniqueness_weight * (1.0 - uniqueness);
    let (ones_f1, ones_iou) = ones_scores(recon, batch);
    EvalReport {
        fitness,
        ber,
        ones_f1,
        ones_iou,
        color_mae,
        color_mse,
        psnr,
        index_accuracy,
        abs_mae: 0.0,
        key_entropy: entropy,
        key_uniqueness: uniqueness,
        active_gates: 0,
        exact_rate: exact_images as f32 / batch.len.max(1) as f32,
        n: batch.len,
    }
}

impl EvalReport {
    fn with_common(mut self, entropy: f32, uniqueness: f32, n: usize) -> Self {
        self.key_entropy = entropy;
        self.key_uniqueness = uniqueness;
        self.n = n;
        self
    }
}

/// For RGB 1-bit/channel, bit error and color error are the same quantity.
/// Index accuracy here means the fraction of pixels with all three channel
/// bits correct.
fn rgb1_metrics(recon: &[u64], batch: &BitBatch, dataset: &Dataset) -> (f32, f32, f32, f32, usize) {
    let pixels = dataset.pixels();
    let mut correct_pixels = 0usize;
    let mut exact_images = 0usize;
    for sample in 0..batch.len {
        let word = sample / 64;
        let bit = sample % 64;
        let mut image_correct = 0usize;
        for p in 0..pixels {
            let mut pixel_ok = true;
            for channel in 0..3 {
                let col = channel * pixels + p;
                let a = (recon[col * batch.words + word] >> bit) & 1;
                let b = (batch.planes[col * batch.words + word] >> bit) & 1;
                if a != b {
                    pixel_ok = false;
                }
            }
            if pixel_ok {
                correct_pixels += 1;
                image_correct += 1;
            }
        }
        if image_correct == pixels {
            exact_images += 1;
        }
    }
    let n = (batch.len.max(1) * pixels.max(1)) as f64;
    let index_accuracy = correct_pixels as f32 / n as f32;
    let errors = plane_hamming(recon, &batch.planes, batch.cols, batch.words, batch.len);
    let color_mae = errors as f32 / (batch.len.max(1) * batch.cols.max(1)) as f32;
    let color_mse = color_mae;
    (
        color_mae,
        color_mse,
        psnr_from_mse(color_mse),
        index_accuracy,
        exact_images,
    )
}

/// Mean absolute RGB error between the reconstruction and the original image.
///
/// Both quantised-target metrics (`color_mae`) normalise against a different
/// palette per mode, so they are not comparable across colour encodings. This
/// measures against the unquantised source instead.
fn abs_rgb_mae(
    model: &Model,
    dataset: &Dataset,
    indices: &[usize],
    recon: &[u64],
    batch: &BitBatch,
) -> f32 {
    let cols = batch.cols;
    let mut total = 0u64;
    let mut fields = 0usize;
    let mut bits = vec![0u8; cols];
    for (sample, &index) in indices.iter().enumerate() {
        let word = sample / 64;
        let bit = sample % 64;
        for (col, slot) in bits.iter_mut().enumerate() {
            *slot = ((recon[col * batch.words + word] >> bit) & 1) as u8;
        }
        let rgb = model.bits_to_rgb(&bits);
        let original = dataset.row_rgb(index);
        for (a, b) in rgb.iter().zip(original.iter()) {
            total += (*a as i32 - *b as i32).unsigned_abs() as u64;
            fields += 1;
        }
    }
    if fields == 0 {
        return 0.0;
    }
    total as f32 / (fields as f32 * 255.0)
}

/// Compute color metrics for RGB-2 models by rebuilding each channel's 2-bit
/// level and comparing it with the target level.
///
/// A level difference of `d` is `d * 85` in 8-bit units, so MAE normalises by
/// 3 and MSE by 9. Pixel accuracy requires all six planes to match.
fn rgb2_metrics(recon: &[u64], batch: &BitBatch, dataset: &Dataset) -> (f32, f32, f32, f32, usize) {
    let pixels = dataset.pixels();
    let mut correct_pixels = 0usize;
    let mut exact_images = 0usize;
    let mut level_err = 0u64;
    let mut level_sq = 0u64;
    for sample in 0..batch.len {
        let word = sample / 64;
        let bit = sample % 64;
        let mut image_correct = 0usize;
        for p in 0..pixels {
            let mut pixel_ok = true;
            for channel in 0..3 {
                let hi_col = (2 * channel) * pixels + p;
                let lo_col = (2 * channel + 1) * pixels + p;
                let a_hi = (recon[hi_col * batch.words + word] >> bit) & 1;
                let a_lo = (recon[lo_col * batch.words + word] >> bit) & 1;
                let b_hi = (batch.planes[hi_col * batch.words + word] >> bit) & 1;
                let b_lo = (batch.planes[lo_col * batch.words + word] >> bit) & 1;
                let a = a_hi * 2 + a_lo;
                let b = b_hi * 2 + b_lo;
                let d = a.abs_diff(b);
                level_err += d;
                level_sq += d * d;
                if a != b {
                    pixel_ok = false;
                }
            }
            if pixel_ok {
                correct_pixels += 1;
                image_correct += 1;
            }
        }
        if image_correct == pixels {
            exact_images += 1;
        }
    }
    let fields = (batch.len.max(1) * pixels.max(1) * 3).max(1) as f64;
    let color_mae = (level_err as f64 / fields / 3.0) as f32;
    let color_mse = (level_sq as f64 / fields / 9.0) as f32;
    let index_accuracy = correct_pixels as f32 / (batch.len.max(1) * pixels.max(1)) as f32;
    (
        color_mae,
        color_mse,
        psnr_from_mse(color_mse),
        index_accuracy,
        exact_images,
    )
}

/// Compute color metrics by decoding each pixel's palette index and comparing
/// it with the target palette index.
fn color_metrics(
    recon: &[u64],
    batch: &BitBatch,
    dataset: &Dataset,
) -> (f32, f32, f32, f32, usize) {
    let bpp = dataset.bits_per_pixel;
    let pixels = dataset.pixels();
    let palette = &dataset.palette;
    let k = palette.len();
    let mut mae_dist = vec![0.0f32; k * k];
    let mut mse_dist = vec![0.0f32; k * k];
    for a in 0..k {
        for b in 0..k {
            let dr = palette[a][0] as i32 - palette[b][0] as i32;
            let dg = palette[a][1] as i32 - palette[b][1] as i32;
            let db = palette[a][2] as i32 - palette[b][2] as i32;
            mae_dist[a * k + b] = (dr.abs() + dg.abs() + db.abs()) as f32 / (3.0 * 255.0);
            mse_dist[a * k + b] = (dr * dr + dg * dg + db * db) as f32 / (3.0 * 255.0 * 255.0);
        }
    }

    let mut total_mae = 0.0f64;
    let mut total_mse = 0.0f64;
    let mut correct = 0usize;
    let mut exact_images = 0usize;

    for sample in 0..batch.len {
        let row = batch.indices[sample];
        let target = dataset.row_values(row);
        let word = sample / 64;
        let bit = sample % 64;
        let mut image_correct = 0usize;
        for (p, &target_index) in target.iter().enumerate().take(pixels) {
            let mut decoded = 0usize;
            for b in 0..bpp {
                let plane = p * bpp + b;
                let value = (recon[plane * batch.words + word] >> bit) & 1;
                decoded |= (value as usize) << b;
            }
            let target_index = target_index as usize;
            total_mae += mae_dist[target_index * k + decoded] as f64;
            total_mse += mse_dist[target_index * k + decoded] as f64;
            if decoded == target_index {
                correct += 1;
                image_correct += 1;
            }
        }
        if image_correct == pixels {
            exact_images += 1;
        }
    }

    let n = (batch.len.max(1) * pixels.max(1)) as f64;
    let mae = (total_mae / n) as f32;
    let mse = (total_mse / n) as f32;
    (
        mae,
        mse,
        psnr_from_mse(mse),
        correct as f32 / n as f32,
        exact_images,
    )
}

fn evaluate_individual(
    ind: &Individual,
    batch: &BitBatch,
    cfg: &TrainConfig,
    dataset: &Dataset,
) -> EvalReport {
    let mut report = evaluate(ind, batch, cfg, dataset);
    let active = ind.encoder.active_gates() + ind.decoder.active_gates();
    let total = (cfg.encoder_gates + cfg.decoder_gates).max(1) as f32;
    report.active_gates = active;
    report.fitness += cfg.parsimony_weight * (active as f32 / total);
    report
}

fn evaluate_all(
    population: &mut [Individual],
    batch: &BitBatch,
    cfg: &TrainConfig,
    dataset: &Dataset,
) {
    population.par_iter_mut().for_each(|ind| {
        let report = evaluate_individual(ind, batch, cfg, dataset);
        ind.apply(&report);
    });
}

fn count_ones(planes: &[u64], col: usize, words: usize, len: usize) -> usize {
    let mut total = 0usize;
    for w in 0..words {
        let mut value = planes[col * words + w];
        if w + 1 == words {
            let valid = len.saturating_sub(w * 64).min(64);
            if valid < 64 {
                value &= if valid == 0 { 0 } else { (1u64 << valid) - 1 };
            }
        }
        total += value.count_ones() as usize;
    }
    total
}

fn key_entropy(planes: &[u64], key_bits: usize, words: usize, len: usize) -> f32 {
    if key_bits == 0 || len == 0 {
        return 0.0;
    }
    let mut sum = 0.0f32;
    for col in 0..key_bits {
        let ones = count_ones(planes, col, words, len);
        let p = ones as f32 / len as f32;
        let h = if p <= 0.0 || p >= 1.0 {
            0.0
        } else {
            -p * p.log2() - (1.0 - p) * (1.0 - p).log2()
        };
        sum += h;
    }
    sum / key_bits as f32
}

/// Fraction of distinct keys in the batch, capped at `2^key_bits`.
fn key_uniqueness(planes: &[u64], key_bits: usize, words: usize, len: usize) -> f32 {
    if key_bits == 0 || len == 0 {
        return 0.0;
    }
    let mut seen = std::collections::HashSet::new();
    for sample in 0..len {
        let word = sample / 64;
        let bit = sample % 64;
        if key_bits <= 64 {
            let mut key = 0u64;
            for col in 0..key_bits {
                let value = (planes[col * words + word] >> bit) & 1;
                key |= value << col;
            }
            seen.insert(key);
        } else {
            let mut hash = 0xcbf2_9ce4_8422_2325u64;
            for col in 0..key_bits {
                let value = (planes[col * words + word] >> bit) & 1;
                hash ^= value;
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
            seen.insert(hash);
        }
    }
    let capacity = if key_bits >= 64 {
        len
    } else {
        len.min(1usize << key_bits)
    };
    seen.len() as f32 / capacity.max(1) as f32
}

fn sample_indices(rng: &mut impl Rng, pool: &[usize], k: usize) -> Vec<usize> {
    if pool.is_empty() {
        return Vec::new();
    }
    let k = k.min(pool.len());
    index::sample(rng, pool.len(), k)
        .into_iter()
        .map(|i| pool[i])
        .collect()
}

fn tournament<'a>(population: &'a [Individual], k: usize, rng: &mut impl Rng) -> &'a Individual {
    let mut best = &population[rng.gen_range(0..population.len())];
    for _ in 1..k {
        let candidate = &population[rng.gen_range(0..population.len())];
        if candidate.fitness < best.fitness {
            best = candidate;
        }
    }
    best
}

fn mutate_circuit(circuit: &mut Circuit, gate_set: GateSet, rate: f32, rng: &mut impl Rng) {
    let n_gates = circuit.gates.len();
    if n_gates == 0 || circuit.outputs.is_empty() {
        return;
    }
    let n_mut = (1.0 + rng.gen::<f32>() * (n_gates as f32 * rate)).floor() as usize;
    let n_mut = n_mut.clamp(1, 16);
    for _ in 0..n_mut {
        match rng.gen_range(0..4) {
            0 => {
                let gi = rng.gen_range(0..n_gates);
                let wire = circuit.gate_start() + gi;
                circuit.gates[gi].a = rng.gen_range(0..wire) as u16;
            }
            1 => {
                let gi = rng.gen_range(0..n_gates);
                let wire = circuit.gate_start() + gi;
                circuit.gates[gi].b = rng.gen_range(0..wire) as u16;
            }
            2 => {
                let gi = rng.gen_range(0..n_gates);
                let op = gate_set.op_from_roll(rng.gen::<f32>());
                circuit.gates[gi].op = op;
                if op.is_unary() {
                    circuit.gates[gi].b = circuit.gates[gi].a;
                }
            }
            _ => {
                let oi = rng.gen_range(0..circuit.outputs.len());
                let total = circuit.total_wires();
                let wire = if rng.gen::<f32>() < 0.90 {
                    circuit.gate_start() + rng.gen_range(0..n_gates)
                } else {
                    rng.gen_range(0..total)
                };
                circuit.outputs[oi] = wire as u16;
            }
        }
    }
}

fn mutate_individual(ind: &mut Individual, cfg: &TrainConfig, rng: &mut impl Rng) {
    let mutate_encoder = rng.gen::<bool>();
    let mutate_decoder = !mutate_encoder || rng.gen::<f32>() < 0.25;
    if mutate_encoder {
        mutate_circuit(&mut ind.encoder, cfg.gate_set, cfg.mutation_rate, rng);
    }
    if mutate_decoder {
        mutate_circuit(&mut ind.decoder, cfg.gate_set, cfg.mutation_rate, rng);
    }
}

fn crossover_circuit(a: &Circuit, b: &Circuit, rng: &mut impl Rng) -> Circuit {
    debug_assert_eq!(a.n_inputs, b.n_inputs);
    let gates = a
        .gates
        .iter()
        .zip(b.gates.iter())
        .map(|(ga, gb)| if rng.gen::<bool>() { *ga } else { *gb })
        .collect();
    let outputs = a
        .outputs
        .iter()
        .zip(b.outputs.iter())
        .map(|(oa, ob)| if rng.gen::<bool>() { *oa } else { *ob })
        .collect();
    Circuit {
        n_inputs: a.n_inputs,
        gates,
        outputs,
    }
}

fn crossover(a: &Individual, b: &Individual, rng: &mut impl Rng) -> Individual {
    Individual {
        encoder: crossover_circuit(&a.encoder, &b.encoder, rng),
        decoder: crossover_circuit(&a.decoder, &b.decoder, rng),
        fitness: f32::INFINITY,
        ber: 1.0,
        color_mae: 1.0,
        index_accuracy: 0.0,
        key_entropy: 0.0,
        key_uniqueness: 0.0,
        active_gates: 0,
    }
}

pub fn to_model(ind: &Individual, cfg: &TrainConfig, val: &EvalReport, dataset: &Dataset) -> Model {
    Model::new(
        ModelConfig {
            image_size: cfg.image_size,
            key_bits: cfg.key_bits,
            gate_set: cfg.gate_set,
            binarize: cfg.binarize,
            threshold: cfg.threshold,
            seed: cfg.seed,
            color_mode: dataset.color_mode,
            palette: dataset.palette.clone(),
            bits_per_pixel: dataset.bits_per_pixel,
            channels: dataset.channels,
        },
        ind.encoder.clone(),
        ind.decoder.clone(),
        ModelMetrics {
            train_ber: ind.ber,
            val_ber: val.ber,
            test_ber: 0.0,
            key_entropy: ind.key_entropy,
            active_gates: ind.active_gates,
            color_mae: val.color_mae,
            color_mse: val.color_mse,
            psnr: val.psnr,
            index_accuracy: val.index_accuracy,
            abs_mae: 0.0,
        },
    )
}

pub fn run_ga(dataset: &Dataset, cfg: &TrainConfig, out_dir: &Path) -> Result<Individual> {
    cfg.validate()?;
    if dataset.size != cfg.image_size {
        bail!(
            "dataset is {}x{} but config says {}x{}",
            dataset.size,
            dataset.size,
            cfg.image_size,
            cfg.image_size
        );
    }
    if cfg.key_bits >= dataset.cols() {
        bail!(
            "key_bits ({}) must be smaller than the input size ({} bits)",
            cfg.key_bits,
            dataset.cols()
        );
    }
    let input_bits = dataset.cols();

    fs::create_dir_all(out_dir)
        .with_context(|| format!("creating run directory {}", out_dir.display()))?;
    fs::write(out_dir.join("config.toml"), cfg.to_toml()?)?;

    let started = Instant::now();
    let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);
    let (train_pool, val_pool, test_pool) = dataset.split_indices(cfg.seed);
    if train_pool.is_empty() {
        bail!("no training samples available");
    }

    let train_indices = sample_indices(&mut rng, &train_pool, cfg.train_batch);
    let val_indices = sample_indices(&mut rng, &val_pool, cfg.val_batch);
    let train_batch = BitBatch::from_matrix(&dataset.bits, &train_indices);
    let val_batch = BitBatch::from_matrix(&dataset.bits, &val_indices);
    eprintln!(
        "dataset: {} images ({} train / {} val / {} test), {} bits/image, {} bits/pixel, train batch {}, val batch {}",
        dataset.n,
        train_pool.len(),
        val_pool.len(),
        test_pool.len(),
        dataset.cols(),
        dataset.bits_per_pixel,
        train_indices.len(),
        val_indices.len()
    );

    let mut population: Vec<Individual> = (0..cfg.population)
        .map(|_| Individual::random(cfg, input_bits, &mut rng))
        .collect();
    evaluate_all(&mut population, &train_batch, cfg, dataset);
    population.sort_by(|a, b| a.fitness.total_cmp(&b.fitness));

    let mut best = population[0].clone();
    let mut best_val = evaluate_individual(&best, &val_batch, cfg, dataset);
    let mut stagnation = 0usize;

    let log_file = File::create(out_dir.join("log.csv"))?;
    let mut log = BufWriter::new(log_file);
    writeln!(
        log,
        "gen,best_fitness,mean_fitness,best_ber,val_ber,val_color_mae,val_index_accuracy,key_entropy,key_uniqueness,active_gates,elapsed_ms"
    )?;

    for gen in 0..cfg.generations {
        let mut offspring = Vec::with_capacity(cfg.offspring);
        for _ in 0..cfg.offspring {
            let parent = tournament(&population, cfg.tournament, &mut rng).clone();
            let mut child = if cfg.crossover_rate > 0.0 && rng.gen::<f32>() < cfg.crossover_rate {
                let other = tournament(&population, cfg.tournament, &mut rng);
                crossover(&parent, other, &mut rng)
            } else {
                parent
            };
            mutate_individual(&mut child, cfg, &mut rng);
            child.fitness = f32::INFINITY;
            offspring.push(child);
        }

        evaluate_all(&mut offspring, &train_batch, cfg, dataset);
        population.extend(offspring);
        population.sort_by(|a, b| a.fitness.total_cmp(&b.fitness));
        population.truncate(cfg.population);

        let current = population[0].clone();
        if current.fitness < best.fitness {
            best = current;
            stagnation = 0;
        } else {
            stagnation += 1;
        }

        if gen % cfg.val_every == 0 || gen + 1 == cfg.generations {
            best_val = evaluate_individual(&best, &val_batch, cfg, dataset);
        }

        if cfg.log_every > 0 && (gen % cfg.log_every == 0 || gen + 1 == cfg.generations) {
            let mean = population.iter().map(|i| i.fitness).sum::<f32>() / population.len() as f32;
            writeln!(
                log,
                "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{}",
                gen,
                best.fitness,
                mean,
                best.ber,
                best_val.ber,
                best_val.color_mae,
                best_val.index_accuracy,
                best.key_entropy,
                best.key_uniqueness,
                best.active_gates,
                started.elapsed().as_millis()
            )?;
        }

        if cfg.checkpoint_every > 0 && gen > 0 && gen % cfg.checkpoint_every == 0 {
            let model = to_model(&best, cfg, &best_val, dataset);
            fs::write(
                out_dir.join("model.json"),
                model.to_json().map_err(anyhow::Error::msg)?,
            )?;
        }

        if stagnation >= cfg.stagnation_limit {
            let n = (cfg.population / 10).max(1);
            for i in 0..n {
                let idx = cfg.population - 1 - i;
                population[idx] = Individual::random(cfg, input_bits, &mut rng);
            }
            let start = cfg.population - n;
            evaluate_all(&mut population[start..], &train_batch, cfg, dataset);
            population.sort_by(|a, b| a.fitness.total_cmp(&b.fitness));
            stagnation = 0;
        }
    }

    let full_test = if test_pool.is_empty() {
        val_pool.clone()
    } else {
        test_pool.clone()
    };
    let mut model = to_model(&best, cfg, &best_val, dataset);
    let full_report = evaluate_model(&model, dataset, &full_test)?;
    model.metrics.test_ber = full_report.ber;
    model.metrics.key_entropy = best.key_entropy;
    model.metrics.active_gates = best.active_gates;
    model.metrics.color_mae = full_report.color_mae;
    model.metrics.color_mse = full_report.color_mse;
    model.metrics.psnr = full_report.psnr;
    model.metrics.index_accuracy = full_report.index_accuracy;
    fs::write(
        out_dir.join("model.json"),
        model.to_json().map_err(anyhow::Error::msg)?,
    )?;
    fs::write(
        out_dir.join("metrics.json"),
        serde_json::to_string_pretty(&model.metrics)?,
    )?;
    log.flush()?;

    if dataset.color_mode != ColorMode::Grayscale {
        eprintln!(
            "done: color MAE {:.4}, PSNR {:.2} dB, pixel accuracy {:.2}%, bit BER {:.4}, val color MAE {:.4}, test color MAE {:.4}, active gates {}",
            best.color_mae,
            full_report.psnr,
            full_report.index_accuracy * 100.0,
            best.ber,
            best_val.color_mae,
            full_report.color_mae,
            best.active_gates
        );
    } else {
        eprintln!(
            "done: train BER {:.4}, val BER {:.4}, test BER {:.4}, key entropy {:.3}, active gates {}",
            best.ber, best_val.ber, full_report.ber, best.key_entropy, best.active_gates
        );
    }
    Ok(best)
}

/// Evaluate a saved model on a set of dataset rows.
pub fn evaluate_model(model: &Model, dataset: &Dataset, indices: &[usize]) -> Result<EvalReport> {
    model.validate().map_err(anyhow::Error::msg)?;
    if indices.is_empty() {
        bail!("no indices to evaluate");
    }
    let batch = BitBatch::from_matrix(&dataset.bits, indices);
    let mut enc_arena = Vec::new();
    let mut key_planes = Vec::new();
    model
        .encoder
        .eval_into(&batch.planes, batch.words, &mut enc_arena, &mut key_planes);
    let mut dec_arena = Vec::new();
    let mut recon = Vec::new();
    model
        .decoder
        .eval_into(&key_planes, batch.words, &mut dec_arena, &mut recon);

    let errors = plane_hamming(&recon, &batch.planes, batch.cols, batch.words, batch.len);
    let ber = errors as f32 / (batch.len * batch.cols) as f32;
    let entropy = key_entropy(&key_planes, model.config.key_bits, batch.words, batch.len);
    let uniqueness = key_uniqueness(&key_planes, model.config.key_bits, batch.words, batch.len);
    let exact = exact_matches(&recon, &batch);

    let abs_mae = if dataset.color_mode == ColorMode::Grayscale {
        0.0
    } else {
        abs_rgb_mae(model, dataset, indices, &recon, &batch)
    };

    let (color_mae, color_mse, psnr, index_accuracy) = match dataset.color_mode {
        ColorMode::Palette => {
            let (mae, mse, psnr, accuracy, _) = color_metrics(&recon, &batch, dataset);
            (mae, mse, psnr, accuracy)
        }
        ColorMode::Rgb1 => {
            let (mae, mse, psnr, accuracy, _) = rgb1_metrics(&recon, &batch, dataset);
            (mae, mse, psnr, accuracy)
        }
        ColorMode::Rgb2 => {
            let (mae, mse, psnr, accuracy, _) = rgb2_metrics(&recon, &batch, dataset);
            (mae, mse, psnr, accuracy)
        }
        ColorMode::Grayscale => (0.0, 0.0, 0.0, 0.0),
    };

    let fitness = if dataset.color_mode == ColorMode::Grayscale {
        ber
    } else {
        color_mae
    };

    let (ones_f1, ones_iou) = ones_scores(&recon, &batch);
    Ok(EvalReport {
        fitness,
        ber,
        ones_f1,
        ones_iou,
        color_mae,
        color_mse,
        psnr,
        index_accuracy,
        abs_mae,
        key_entropy: entropy,
        key_uniqueness: uniqueness,
        active_gates: model.encoder.active_gates() + model.decoder.active_gates(),
        exact_rate: exact as f32 / batch.len as f32,
        n: batch.len,
    })
}

/// Precision/recall over the set bits, as F1 and IoU.
fn ones_scores(recon: &[u64], batch: &BitBatch) -> (f32, f32) {
    let mut tp = 0u64;
    let mut fp = 0u64;
    let mut missed = 0u64;
    for sample in 0..batch.len {
        let word = sample / 64;
        let bit = sample % 64;
        for col in 0..batch.cols {
            let a = (recon[col * batch.words + word] >> bit) & 1;
            let b = (batch.planes[col * batch.words + word] >> bit) & 1;
            match (a, b) {
                (1, 1) => tp += 1,
                (1, 0) => fp += 1,
                (0, 1) => missed += 1,
                _ => {}
            }
        }
    }
    if tp == 0 && fp == 0 && missed == 0 {
        return (1.0, 1.0);
    }
    let f1 = if tp == 0 {
        0.0
    } else {
        2.0 * tp as f32 / (2 * tp + fp + missed) as f32
    };
    let union = tp + fp + missed;
    let iou = if union == 0 {
        1.0
    } else {
        tp as f32 / union as f32
    };
    (f1, iou)
}

fn exact_matches(recon: &[u64], batch: &BitBatch) -> usize {
    let mut exact = 0usize;
    for sample in 0..batch.len {
        let word = sample / 64;
        let bit = sample % 64;
        let mut ok = true;
        for col in 0..batch.cols {
            let a = (recon[col * batch.words + word] >> bit) & 1;
            let b = (batch.planes[col * batch.words + word] >> bit) & 1;
            if a != b {
                ok = false;
                break;
            }
        }
        if ok {
            exact += 1;
        }
    }
    exact
}
