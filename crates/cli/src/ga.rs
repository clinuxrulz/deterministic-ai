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
use xorae_core::{Circuit, GateSet, Model, ModelConfig, ModelMetrics};

use crate::config::TrainConfig;
use crate::dataset::Dataset;

#[derive(Clone, Debug)]
pub struct Individual {
    pub encoder: Circuit,
    pub decoder: Circuit,
    pub fitness: f32,
    pub ber: f32,
    pub key_entropy: f32,
    pub key_uniqueness: f32,
    pub active_gates: usize,
}

#[derive(Clone, Debug, Default)]
pub struct EvalReport {
    pub fitness: f32,
    pub ber: f32,
    pub key_entropy: f32,
    pub key_uniqueness: f32,
    pub active_gates: usize,
    pub exact_rate: f32,
    pub n: usize,
}

impl Individual {
    pub fn random(cfg: &TrainConfig, rng: &mut impl Rng) -> Self {
        let n = cfg.image_size * cfg.image_size;
        let enc_seed = rng.gen::<u64>();
        let dec_seed = rng.gen::<u64>();
        Self {
            encoder: Circuit::random(n, cfg.encoder_gates, cfg.key_bits, cfg.gate_set, enc_seed),
            decoder: Circuit::random(cfg.key_bits, cfg.decoder_gates, n, cfg.gate_set, dec_seed),
            fitness: f32::INFINITY,
            ber: 1.0,
            key_entropy: 0.0,
            key_uniqueness: 0.0,
            active_gates: 0,
        }
    }

    fn apply(&mut self, report: &EvalReport) {
        self.fitness = report.fitness;
        self.ber = report.ber;
        self.key_entropy = report.key_entropy;
        self.key_uniqueness = report.key_uniqueness;
        self.active_gates = report.active_gates;
    }
}

/// Evaluate one individual on a fixed batch.
pub fn evaluate(ind: &Individual, batch: &BitBatch, cfg: &TrainConfig) -> EvalReport {
    let mut enc_arena = Vec::new();
    let mut key_planes = Vec::new();
    ind.encoder
        .eval_into(&batch.planes, batch.words, &mut enc_arena, &mut key_planes);

    let mut dec_arena = Vec::new();
    let mut recon = Vec::new();
    ind.decoder
        .eval_into(&key_planes, batch.words, &mut dec_arena, &mut recon);

    report_from_planes(&recon, &key_planes, batch, cfg)
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
    let active = 0usize; // filled by the caller, which knows both circuits
    let fitness =
        ber + cfg.entropy_weight * (1.0 - entropy) + cfg.uniqueness_weight * (1.0 - uniqueness);
    EvalReport {
        fitness,
        ber,
        key_entropy: entropy,
        key_uniqueness: uniqueness,
        active_gates: active,
        exact_rate: 0.0,
        n: batch.len,
    }
}

fn evaluate_individual(ind: &Individual, batch: &BitBatch, cfg: &TrainConfig) -> EvalReport {
    let mut report = evaluate(ind, batch, cfg);
    let active = ind.encoder.active_gates() + ind.decoder.active_gates();
    let total = (cfg.encoder_gates + cfg.decoder_gates).max(1) as f32;
    report.active_gates = active;
    report.fitness += cfg.parsimony_weight * (active as f32 / total);
    report
}

fn evaluate_all(population: &mut [Individual], batch: &BitBatch, cfg: &TrainConfig) {
    population.par_iter_mut().for_each(|ind| {
        let report = evaluate_individual(ind, batch, cfg);
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
///
/// This rewards an encoder that actually uses the key space. For a linear
/// subspace toy it directly rewards an injective encoder, which is the
/// prerequisite for the decoder to reconstruct held-out combinations.
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
        key_entropy: 0.0,
        key_uniqueness: 0.0,
        active_gates: 0,
    }
}

pub fn to_model(ind: &Individual, cfg: &TrainConfig, val: &EvalReport) -> Model {
    Model::new(
        ModelConfig {
            image_size: cfg.image_size,
            key_bits: cfg.key_bits,
            gate_set: cfg.gate_set,
            binarize: cfg.binarize,
            threshold: cfg.threshold,
            seed: cfg.seed,
        },
        ind.encoder.clone(),
        ind.decoder.clone(),
        ModelMetrics {
            train_ber: ind.ber,
            val_ber: val.ber,
            test_ber: 0.0,
            key_entropy: ind.key_entropy,
            active_gates: ind.active_gates,
        },
    )
}

pub fn run_ga(dataset: &Dataset, cfg: &TrainConfig, out_dir: &Path) -> Result<Individual> {
    cfg.validate()?;
    if dataset.cols() != cfg.image_size * cfg.image_size {
        bail!(
            "dataset has {} bits per image but config says {}x{}",
            dataset.cols(),
            cfg.image_size,
            cfg.image_size
        );
    }

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
        "dataset: {} images ({} train / {} val / {} test), train batch {}, val batch {}",
        dataset.n,
        train_pool.len(),
        val_pool.len(),
        test_pool.len(),
        train_indices.len(),
        val_indices.len()
    );

    let mut population: Vec<Individual> = (0..cfg.population)
        .map(|_| Individual::random(cfg, &mut rng))
        .collect();
    evaluate_all(&mut population, &train_batch, cfg);
    population.sort_by(|a, b| a.fitness.total_cmp(&b.fitness));

    let mut best = population[0].clone();
    let mut best_val = evaluate_individual(&best, &val_batch, cfg);
    let mut stagnation = 0usize;

    let log_file = File::create(out_dir.join("log.csv"))?;
    let mut log = BufWriter::new(log_file);
    writeln!(
        log,
        "gen,best_fitness,mean_fitness,best_ber,val_ber,key_entropy,key_uniqueness,active_gates,elapsed_ms"
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

        evaluate_all(&mut offspring, &train_batch, cfg);
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
            best_val = evaluate_individual(&best, &val_batch, cfg);
        }

        if cfg.log_every > 0 && (gen % cfg.log_every == 0 || gen + 1 == cfg.generations) {
            let mean = population.iter().map(|i| i.fitness).sum::<f32>() / population.len() as f32;
            writeln!(
                log,
                "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{}",
                gen,
                best.fitness,
                mean,
                best.ber,
                best_val.ber,
                best.key_entropy,
                best.key_uniqueness,
                best.active_gates,
                started.elapsed().as_millis()
            )?;
        }

        if cfg.checkpoint_every > 0 && gen > 0 && gen % cfg.checkpoint_every == 0 {
            let model = to_model(&best, cfg, &best_val);
            fs::write(
                out_dir.join("model.json"),
                model.to_json().map_err(anyhow::Error::msg)?,
            )?;
        }

        if stagnation >= cfg.stagnation_limit {
            let n = (cfg.population / 10).max(1);
            for i in 0..n {
                let idx = cfg.population - 1 - i;
                population[idx] = Individual::random(cfg, &mut rng);
            }
            let start = cfg.population - n;
            evaluate_all(&mut population[start..], &train_batch, cfg);
            population.sort_by(|a, b| a.fitness.total_cmp(&b.fitness));
            stagnation = 0;
        }
    }

    let test_indices = sample_indices(&mut rng, &test_pool, cfg.val_batch);
    let test_batch = BitBatch::from_matrix(&dataset.bits, &test_indices);
    let test_report = evaluate_individual(&best, &test_batch, cfg);

    let mut model = to_model(&best, cfg, &best_val);
    model.metrics.test_ber = test_report.ber;
    model.metrics.key_entropy = best.key_entropy;
    model.metrics.active_gates = best.active_gates;
    fs::write(
        out_dir.join("model.json"),
        model.to_json().map_err(anyhow::Error::msg)?,
    )?;
    fs::write(
        out_dir.join("metrics.json"),
        serde_json::to_string_pretty(&model.metrics)?,
    )?;
    log.flush()?;

    eprintln!(
        "done: train BER {:.4}, val BER {:.4}, test BER {:.4}, key entropy {:.3}, active gates {}",
        best.ber, best_val.ber, test_report.ber, best.key_entropy, best.active_gates
    );
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
    Ok(EvalReport {
        fitness: ber,
        ber,
        key_entropy: entropy,
        key_uniqueness: uniqueness,
        active_gates: model.encoder.active_gates() + model.decoder.active_gates(),
        exact_rate: exact as f32 / batch.len as f32,
        n: batch.len,
    })
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
