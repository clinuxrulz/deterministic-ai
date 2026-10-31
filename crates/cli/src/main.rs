use std::path::PathBuf;
use std::process::Command as ProcessCommand;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use xorae::config::TrainConfig;
use xorae::dataset::Dataset;
use xorae::{
    baselines, classify, config, dataset, ga, grad, nand, nand_classify, prior, render, toy,
};
use xorae_core::color::ColorMode;
use xorae_core::preprocess::{BinarizeMode, PreprocessOptions};
use xorae_core::Model;

#[derive(Parser)]
#[command(
    name = "xorae",
    version,
    about = "Evolve XOR-gate autoencoders with a genetic algorithm"
)]
struct Cli {
    #[command(subcommand)]
    command: CliCommand,
}

#[derive(Subcommand)]
enum CliCommand {
    /// Download and extract the CIFAR-10 binary dataset.
    FetchData {
        #[arg(long, default_value = "data")]
        out: PathBuf,
    },
    /// Convert CIFAR-10 cat images into a cached xorae dataset.
    Preprocess {
        #[arg(long)]
        cifar_dir: Option<PathBuf>,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 16)]
        size: usize,
        /// Binarization: `threshold`, `otsu`, `dither` or `edge`. `edge` keeps
        /// the strongest 15% of Sobel gradients, which turns a photo into a
        /// line drawing and stays legible at 32x32 where a hard threshold
        /// collapses into blobs.
        #[arg(long, default_value = "dither")]
        binarize: String,
        #[arg(long, default_value_t = 3)]
        class: u8,
        #[arg(long, default_value_t = 128)]
        threshold: u8,
        /// Build a global color palette instead of a 1-bit grayscale dataset.
        #[arg(long)]
        palette: Option<usize>,
        /// Build a 3-channel 1-bit RGB dataset (8 colors).
        #[arg(long)]
        rgb1: bool,
        /// Build a 3-channel 2-bit RGB dataset (64 colors).
        #[arg(long)]
        rgb2: bool,
        /// Build from raw MNIST IDX files in this directory instead of CIFAR.
        #[arg(long)]
        mnist_dir: Option<PathBuf>,
        /// Keep only the first N training images (MNIST only).
        #[arg(long)]
        limit: Option<usize>,
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Run the genetic algorithm.
    Train {
        #[arg(long)]
        config: Option<PathBuf>,
        /// Cached dataset produced by `preprocess`.
        #[arg(long)]
        data: Option<PathBuf>,
        /// Use the built-in linear-subspace toy dataset.
        #[arg(long)]
        toy: bool,
        #[arg(long, default_value = "runs/latest")]
        out: PathBuf,
        #[arg(long)]
        seed: Option<u64>,
        #[arg(long)]
        generations: Option<usize>,
    },
    /// Train a fixed-topology gate network with gradients (soft gates + STE).
    TrainGrad {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        data: PathBuf,
        #[arg(long, default_value = "runs/grad")]
        out: PathBuf,
        #[arg(long)]
        seed: Option<u64>,
        #[arg(long)]
        epochs: Option<usize>,
    },
    /// Train a NAND-only circuit by search over wiring (all gates identical).
    TrainNand {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        data: PathBuf,
        #[arg(long, default_value = "runs/nand")]
        out: PathBuf,
        #[arg(long)]
        seed: Option<u64>,
        #[arg(long)]
        epochs: Option<usize>,
        /// Override the candidate-softmax anneal endpoint (config `gain_end`).
        #[arg(long)]
        gain_end: Option<f32>,
        /// Train on only the first N training samples (diagnostics).
        #[arg(long)]
        train_limit: Option<usize>,
        /// Straight-through binarisation of activations (hard forward).
        #[arg(long)]
        ste: bool,
        /// Straight-through wiring: train with the exported (argmax) wiring.
        #[arg(long)]
        wire_ste: bool,
        /// Gate relaxation: product (default), min, or lukasiewicz.
        #[arg(long)]
        relaxation: Option<String>,
        /// Weight on target bits that are 1 (default 1). Raise it for sparse
        /// targets such as `--binarize edge`, where an all-zero output is
        /// already 85% correct and the encoder collapses.
        #[arg(long)]
        positive_weight: Option<f32>,
    },
    /// Evaluate a saved model on train/val/test.
    Eval {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        data: Option<PathBuf>,
        #[arg(long)]
        toy: bool,
        #[arg(long, default_value = "test")]
        split: String,
        /// Write the measured metrics back into the model JSON.
        #[arg(long)]
        update_metrics: bool,
    },
    /// Render original / binarized / reconstructed images to a PNG.
    Render {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        data: Option<PathBuf>,
        #[arg(long)]
        toy: bool,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 8)]
        count: usize,
        #[arg(long, default_value_t = 10)]
        scale: u32,
    },
    /// Copy a trained model into a portable JSON file for the web app.
    Export {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// Record how images were binarized, so the web demo preprocesses
        /// uploads the same way. The dataset format does not store it, and the
        /// trainer writes `threshold` by default.
        #[arg(long)]
        binarize: Option<String>,
    },
    /// Train a classifier on the encoder key (or raw pixels) and report accuracy.
    Classify {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        data: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// Feature set: `keys` (encoder output) or `pixels` (baseline).
        #[arg(long, default_value = "keys")]
        features: String,
        #[arg(long, default_value_t = 400)]
        epochs: usize,
        #[arg(long, default_value_t = 0.1)]
        lr: f32,
        #[arg(long, default_value_t = 1e-4)]
        l2: f32,
    },
    /// Train a NAND-gate classifier on encoder keys (128 bits -> classes).
    ClassifyNand {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        data: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// Hidden layer widths, comma separated.
        #[arg(long, default_value = "128,64")]
        hidden: String,
        #[arg(long, default_value_t = 200)]
        epochs: usize,
        #[arg(long, default_value_t = 128)]
        batch: usize,
        #[arg(long, default_value_t = 0.05)]
        lr: f32,
        #[arg(long, default_value_t = 8)]
        candidates: usize,
        /// Output gates per class (class score is their sum).
        #[arg(long, default_value_t = 8)]
        group_size: usize,
        /// Keep only these labels (comma separated), remapped to 0..k.
        #[arg(long)]
        subset_labels: Option<String>,
        /// Distil a trained softmax classifier (from `classify`) into the circuit.
        #[arg(long)]
        distill: Option<PathBuf>,
        /// Features for the circuit: `keys` (encoder output) or `pixels`.
        #[arg(long, default_value = "keys")]
        features: String,
        #[arg(long, default_value = "product")]
        relaxation: String,
    },
    /// Validate and describe a standalone circuit JSON (e.g. an exported GAN).
    CircuitInfo {
        #[arg(long)]
        circuit: PathBuf,
    },
    /// Score a saved NAND classifier circuit on a dataset split.
    EvalCircuit {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        data: PathBuf,
        #[arg(long)]
        circuit: PathBuf,
        /// Features the circuit was trained on: `keys` or `pixels`.
        #[arg(long, default_value = "keys")]
        features: String,
        /// Output gates per class.
        #[arg(long, default_value_t = 16)]
        group_size: usize,
    },
    /// Fit a prior over the encoder's 128-bit key space and sample from it.
    Prior {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        data: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 32)]
        samples: usize,
        #[arg(long, default_value_t = 150)]
        epochs: usize,
        #[arg(long, default_value_t = 0.05)]
        lr: f32,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Lossy gate removal: bypass gates while BER stays within a tolerance.
    Prune {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        data: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// Stop once the circuit has at most this many gates.
        #[arg(long)]
        target: Option<usize>,
        /// Largest absolute BER increase accepted per removal.
        #[arg(long, default_value_t = 0.0)]
        tolerance: f32,
        /// Samples used to score candidate removals.
        #[arg(long, default_value_t = 512)]
        calibrate: usize,
        /// Hard cap on removals.
        #[arg(long, default_value_t = 4096)]
        max_removals: usize,
    },
    /// Exact simplification (constant folding + CSE + dead-gate removal).
    ///
    /// Rewrites are Boolean identities, so the output computes the same
    /// function as the input for every possible input vector.
    Simplify {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// Dataset used to prove the simplified model reconstructs identically.
        #[arg(long)]
        verify: Option<PathBuf>,
        /// Also compare both models on N pseudo-random input vectors.
        #[arg(long, default_value_t = 0)]
        verify_random: usize,
    },
    /// Print simple reconstruction baselines.
    Baselines {
        #[arg(long)]
        data: PathBuf,
        #[arg(long, default_value_t = 32)]
        key_bits: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Export individual sample images and a manifest for the web gallery.
    ExportSamples {
        #[arg(long)]
        data: Option<PathBuf>,
        #[arg(long)]
        toy: bool,
        /// Export 32×32 RGB previews directly from the CIFAR test batch.
        #[arg(long)]
        cifar_dir: Option<PathBuf>,
        #[arg(long, default_value_t = 3)]
        class: u8,
        #[arg(long, default_value = "web/public")]
        out: PathBuf,
        #[arg(long, default_value_t = 8)]
        count: usize,
        #[arg(long, default_value_t = 16)]
        scale: u32,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value = "sample")]
        prefix: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        CliCommand::FetchData { out } => fetch_data(&out),
        CliCommand::Preprocess {
            cifar_dir,
            out,
            size,
            binarize,
            class,
            threshold,
            palette,
            rgb1,
            rgb2,
            mnist_dir,
            limit,
            seed,
        } => {
            if let Some(mnist_dir) = mnist_dir {
                let dataset = dataset::load_mnist(&mnist_dir, size, threshold, limit)?;
                dataset.save(&out)?;
                let (train, val, test) = dataset.split_indices(seed);
                println!(
                    "wrote {} ({} images, {}x{}, binarized MNIST; {} train / {} val / {} test)",
                    out.display(),
                    dataset.n,
                    size,
                    size,
                    train.len(),
                    val.len(),
                    test.len()
                );
                return Ok(());
            }
            let cifar_dir =
                cifar_dir.context("--cifar-dir is required unless --mnist-dir is used")?;
            if let Some(palette_size) = palette {
                let dataset =
                    dataset::load_cifar10_palette(&cifar_dir, class, size, palette_size, seed)?;
                dataset.save(&out)?;
                let (train, val, test) = dataset.split_indices(seed);
                println!(
                    "wrote {} ({} images, {}x{}, {} colors, {} bits/pixel; {} train / {} val / {} test)",
                    out.display(),
                    dataset.n,
                    size,
                    size,
                    dataset.palette.len(),
                    dataset.bits_per_pixel,
                    train.len(),
                    val.len(),
                    test.len()
                );
            } else if rgb2 {
                let dataset = dataset::load_cifar10_rgb2(&cifar_dir, class, size)?;
                dataset.save(&out)?;
                let (train, val, test) = dataset.split_indices(seed);
                println!(
                    "wrote {} ({} images, {}x{}, RGB 2-bit/channel (64 colors), {} bits/pixel; {} train / {} val / {} test)",
                    out.display(),
                    dataset.n,
                    size,
                    size,
                    dataset.cols(),
                    train.len(),
                    val.len(),
                    test.len()
                );
            } else if rgb1 {
                let dataset = dataset::load_cifar10_rgb1(&cifar_dir, class, size, threshold)?;
                dataset.save(&out)?;
                let (train, val, test) = dataset.split_indices(seed);
                println!(
                    "wrote {} ({} images, {}x{}, RGB 1-bit/channel, {} bits/pixel; {} train / {} val / {} test)",
                    out.display(),
                    dataset.n,
                    size,
                    size,
                    dataset.channels,
                    train.len(),
                    val.len(),
                    test.len()
                );
            } else {
                let opts = PreprocessOptions {
                    size,
                    mode: parse_binarize(&binarize)?,
                    threshold,
                };
                let dataset = dataset::load_cifar10(&cifar_dir, class, opts)?;
                dataset.save(&out)?;
                let (train, val, test) = dataset.split_indices(seed);
                println!(
                    "wrote {} ({} images, {}x{} bits; {} train / {} val / {} test)",
                    out.display(),
                    dataset.n,
                    size,
                    size,
                    train.len(),
                    val.len(),
                    test.len()
                );
            }
            Ok(())
        }
        CliCommand::Train {
            config,
            data,
            toy,
            out,
            seed,
            generations,
        } => {
            let mut cfg = match (&config, toy) {
                (Some(path), _) => TrainConfig::from_toml(path)?,
                (None, true) => config::toy_config(),
                (None, false) => TrainConfig::default(),
            };
            if let Some(seed) = seed {
                cfg.seed = seed;
            }
            if let Some(generations) = generations {
                cfg.generations = generations;
            }
            cfg.validate()?;

            let dataset = if toy {
                toy::linear_subspace(cfg.image_size, cfg.key_bits, cfg.seed)?
            } else {
                let path = data.context("--data is required unless --toy is used")?;
                Dataset::load(&path)?
            };
            if dataset.size != cfg.image_size {
                bail!(
                    "dataset is {}x{}, config expects {}x{}",
                    dataset.size,
                    dataset.size,
                    cfg.image_size,
                    cfg.image_size
                );
            }

            println!(
                "training {}-bit key on {}x{} images for {} generations (seed {})",
                cfg.key_bits, cfg.image_size, cfg.image_size, cfg.generations, cfg.seed
            );
            let best = ga::run_ga(&dataset, &cfg, &out)?;
            if dataset.color_mode != ColorMode::Grayscale {
                println!(
                    "best model written to {} (train color MAE {:.4})",
                    out.join("model.json").display(),
                    best.color_mae
                );
            } else {
                println!(
                    "best model written to {} (train BER {:.4})",
                    out.join("model.json").display(),
                    best.ber
                );
            }
            Ok(())
        }
        CliCommand::TrainGrad {
            config,
            data,
            out,
            seed,
            epochs,
        } => {
            let mut cfg = grad::GradConfig::from_toml(&config)?;
            if let Some(seed) = seed {
                cfg.seed = seed;
            }
            if let Some(epochs) = epochs {
                cfg.epochs = epochs;
            }
            cfg.validate()?;
            let dataset = Dataset::load(&data)?;
            println!(
                "training gradient gate net: {}x{}, {}-bit key, {} epochs (seed {})",
                cfg.image_size, cfg.image_size, cfg.key_bits, cfg.epochs, cfg.seed
            );
            let model = grad::run_grad(&dataset, &cfg, &out)?;
            println!(
                "model written to {} (test BER {:.4})",
                out.join("model.json").display(),
                model.metrics.test_ber
            );
            Ok(())
        }
        CliCommand::TrainNand {
            config,
            data,
            out,
            seed,
            epochs,
            gain_end,
            positive_weight,
            train_limit,
            ste,
            wire_ste,
            relaxation,
        } => {
            let mut cfg = nand::NandConfig::from_toml(&config)?;
            if let Some(seed) = seed {
                cfg.seed = seed;
            }
            if let Some(epochs) = epochs {
                cfg.epochs = epochs;
            }
            if let Some(gain_end) = gain_end {
                cfg.gain_end = gain_end;
            }
            if let Some(positive_weight) = positive_weight {
                cfg.positive_weight = positive_weight;
            }
            if train_limit.is_some() {
                cfg.train_limit = train_limit;
            }
            if ste {
                cfg.ste = true;
            }
            if wire_ste {
                cfg.wire_ste = true;
            }
            if let Some(relaxation) = relaxation {
                cfg.relaxation = match relaxation.as_str() {
                    "product" => nand::Relaxation::Product,
                    "min" => nand::Relaxation::Min,
                    "lukasiewicz" => nand::Relaxation::Lukasiewicz,
                    other => bail!("unknown relaxation {other}"),
                };
            }
            cfg.validate()?;
            let dataset = Dataset::load(&data)?;
            println!(
                "training NAND wiring: {}x{}, {}-bit key, {} epochs (seed {}, gain_end {}, ste {}, relaxation {:?})",
                cfg.image_size,
                cfg.image_size,
                cfg.key_bits,
                cfg.epochs,
                cfg.seed,
                cfg.gain_end,
                cfg.ste,
                cfg.relaxation
            );
            let model = nand::run_nand(&dataset, &cfg, &out)?;
            println!(
                "model written to {} (test BER {:.4})",
                out.join("model.json").display(),
                model.metrics.test_ber
            );
            Ok(())
        }
        CliCommand::Eval {
            model,
            data,
            toy,
            split,
            update_metrics,
        } => {
            let model_path = model.clone();
            let mut model = load_model(&model)?;
            let dataset = load_eval_dataset(data, toy, &model)?;
            let indices = split_indices(&dataset, model.config.seed, &split)?;
            let report = ga::evaluate_model(&model, &dataset, &indices)?;
            if update_metrics {
                model.metrics.active_gates =
                    model.encoder.active_gates() + model.decoder.active_gates();
                match split.as_str() {
                    "test" => {
                        model.metrics.test_ber = report.ber;
                        model.metrics.abs_mae = absolute_mae(&model, &dataset, &indices)?;
                        model.metrics.color_mae = report.color_mae;
                        model.metrics.color_mse = report.color_mse;
                        model.metrics.psnr = report.psnr;
                        model.metrics.index_accuracy = report.index_accuracy;
                    }
                    "val" => model.metrics.val_ber = report.ber,
                    _ => model.metrics.train_ber = report.ber,
                }
                std::fs::write(&model_path, model.to_json().map_err(anyhow::Error::msg)?)?;
                println!("updated metrics in {}", model_path.display());
            }
            if dataset.color_mode != ColorMode::Grayscale {
                let abs_mae = absolute_mae(&model, &dataset, &indices)?;
                println!(
                    "{}: abs MAE vs original {:.4}, color MAE vs quantised {:.4}, PSNR {:.2} dB, pixel/index accuracy {:.2}%, bit BER {:.4}, exact {:.2}%, active gates {} / AIG nodes {} ({} samples)",
                    split,
                    abs_mae,
                    report.color_mae,
                    report.psnr,
                    report.index_accuracy * 100.0,
                    report.ber,
                    report.exact_rate * 100.0,
                    report.active_gates,
                    model.encoder.aig_nodes() + model.decoder.aig_nodes(),
                    report.n
                );
            } else {
                println!(
                    "{}: BER {:.4}, ones F1 {:.3}, ones IoU {:.3}, edge F1@1.5 {:.3} vs chance {:.3}, @3 {:.3} (chamfer {:.2} px, ink {:.2}->{:.2}), exact {:.2}%, key entropy {:.3}, active gates {} / AIG nodes {} ({} samples)",
                    split,
                    report.ber,
                    report.ones_f1,
                    report.ones_iou,
                    report.edge_f1_near,
                    report.edge_chance_f1_near,
                    report.edge_f1_far,
                    report.edge_chamfer,
                    report.edge_target_density,
                    report.edge_pred_density,
                    report.exact_rate * 100.0,
                    report.key_entropy,
                    report.active_gates,
                    model.encoder.aig_nodes() + model.decoder.aig_nodes(),
                    report.n
                );
            }
            Ok(())
        }
        CliCommand::Render {
            model,
            data,
            toy,
            out,
            count,
            scale,
        } => {
            let model = load_model(&model)?;
            let dataset = load_eval_dataset(data, toy, &model)?;
            let (_train, val, test) = dataset.split_indices(model.config.seed);
            let pool = if !test.is_empty() { &test } else { &val };
            let indices = spread(pool, count);
            render::render_reconstructions(&dataset, &model, &indices, &out, scale)?;
            println!("wrote {}", out.display());
            Ok(())
        }
        CliCommand::Export {
            model,
            out,
            binarize,
        } => {
            let mut model = load_model(&model)?;
            if let Some(mode) = binarize {
                model.config.binarize = parse_binarize(&mode)?;
            }
            let json = model.to_json().map_err(anyhow::Error::msg)?;
            if let Some(parent) = out.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&out, json)?;
            println!("exported model to {}", out.display());
            Ok(())
        }
        CliCommand::Classify {
            model,
            data,
            out,
            features,
            epochs,
            lr,
            l2,
        } => {
            let model = load_model(&model)?;
            let dataset = Dataset::load(&data)?;
            let opts = classify::ClassifyOptions {
                features: classify::Features::parse(&features)?,
                epochs,
                lr,
                l2,
                seed: model.config.seed,
            };
            let (classifier, matrix) = classify::train_classifier(&model, &dataset, opts)?;
            println!(
                "classify ({}): {} inputs, {} classes | train acc {:.4} | test acc {:.4}",
                classifier.features,
                classifier.inputs,
                classifier.classes,
                classifier.train_accuracy,
                classifier.test_accuracy
            );
            println!("confusion (rows = true, cols = predicted):");
            for (label, row) in matrix.iter().enumerate() {
                let total: usize = row.iter().sum();
                let acc = if total > 0 {
                    *row.get(label).unwrap_or(&0) as f32 / total as f32
                } else {
                    0.0
                };
                println!("  {label}: {:?}  acc {:.3}", row, acc);
            }
            classify::save_classifier(&classifier, &out)?;
            println!("wrote {}", out.display());
            Ok(())
        }
        CliCommand::ClassifyNand {
            model,
            data,
            out,
            hidden,
            epochs,
            batch,
            lr,
            candidates,
            group_size,
            subset_labels,
            distill,
            features,
            relaxation,
        } => {
            let model = load_model(&model)?;
            let dataset = Dataset::load(&data)?;
            if !dataset.has_labels() {
                bail!(
                    "dataset {} has no labels; re-run preprocess",
                    data.display()
                );
            }
            let hidden: Vec<usize> = hidden
                .split(',')
                .filter(|s| !s.trim().is_empty())
                .map(|s| s.trim().parse::<usize>())
                .collect::<std::result::Result<_, _>>()
                .context("parsing --hidden")?;
            let relaxation = match relaxation.as_str() {
                "product" => nand::Relaxation::Product,
                "min" => nand::Relaxation::Min,
                "lukasiewicz" => nand::Relaxation::Lukasiewicz,
                other => bail!("unknown relaxation {other}"),
            };
            let (train, _, test) = dataset.split_indices(model.config.seed);
            let pixels = features == "pixels";
            let keys_for = |idx: &[usize]| -> Result<Vec<Vec<u8>>> {
                if pixels {
                    return Ok(idx.iter().map(|&i| dataset.row_bits(i)).collect());
                }
                idx.iter()
                    .map(|&i| {
                        model
                            .infer_bits(&dataset.row_bits(i))
                            .map(|(k, _)| k)
                            .map_err(anyhow::Error::msg)
                    })
                    .collect()
            };
            let train_keys = keys_for(&train)?;
            let test_keys = keys_for(&test)?;
            let mut train_labels: Vec<u8> = train.iter().map(|&i| dataset.labels[i]).collect();
            let mut test_labels: Vec<u8> = test.iter().map(|&i| dataset.labels[i]).collect();
            let mut train_keys = train_keys;
            let mut test_keys = test_keys;
            let classes = if let Some(spec) = &subset_labels {
                let keep: Vec<u8> = spec
                    .split(',')
                    .map(|s| s.trim().parse::<u8>())
                    .collect::<std::result::Result<_, _>>()
                    .context("parsing --subset-labels")?;
                let remap = |labels: &mut Vec<u8>, keys: &mut Vec<Vec<u8>>| {
                    let mut kept_keys = Vec::new();
                    let mut kept_labels = Vec::new();
                    for (key, &label) in keys.iter().zip(labels.iter()) {
                        if let Some(pos) = keep.iter().position(|&k| k == label) {
                            kept_keys.push(key.clone());
                            kept_labels.push(pos as u8);
                        }
                    }
                    *keys = kept_keys;
                    *labels = kept_labels;
                };
                remap(&mut train_labels, &mut train_keys);
                remap(&mut test_labels, &mut test_keys);
                keep.len()
            } else {
                dataset.label_count.max(1)
            };
            let inputs = train_keys.first().map(|k| k.len()).unwrap_or(0);
            let cfg = nand_classify::NandClassifyConfig {
                inputs,
                classes,
                hidden,
                epochs,
                batch,
                lr,
                candidates,
                group_size,
                skip_prob: 0.5,
                binarize_weight: 0.01,
                gain_start: 1.0,
                gain_end: 32.0,
                anneal_fraction: 0.85,
                relaxation,
                seed: model.config.seed,
            };
            let soft = match &distill {
                Some(path) => {
                    let text = std::fs::read_to_string(path)
                        .with_context(|| format!("reading teacher {}", path.display()))?;
                    let teacher: classify::Classifier =
                        serde_json::from_str(&text).context("parsing teacher classifier")?;
                    let mut rows = Vec::with_capacity(train_keys.len());
                    for key in &train_keys {
                        let features: Vec<u8> = match teacher.features.as_str() {
                            "pixels" => model
                                .infer_bits(key)
                                .map(|(_, recon)| recon)
                                .map_err(anyhow::Error::msg)?,
                            _ => key.clone(),
                        };
                        rows.push(teacher.probabilities(&features));
                    }
                    Some(rows)
                }
                None => None,
            };
            let (circuit, report) = nand_classify::train(
                &cfg,
                &train_keys,
                &train_labels,
                &test_keys,
                &test_labels,
                soft.as_deref(),
            )?;
            println!(
                "nand classifier: train acc {:.4} | test acc {:.4} | gates {} -> {}",
                report.train_accuracy,
                report.test_accuracy,
                report.gates_before,
                report.gates_after
            );
            let hist =
                nand_classify::prediction_histogram(&circuit, &test_keys, classes, group_size);
            println!("predicted class distribution on test: {:?}", hist);
            if let Some(parent) = out.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(
                &out,
                serde_json::to_string_pretty(&circuit).context("serialising circuit")?,
            )?;
            println!("wrote {}", out.display());
            Ok(())
        }
        CliCommand::CircuitInfo { circuit } => {
            let text = std::fs::read_to_string(&circuit)
                .with_context(|| format!("reading {}", circuit.display()))?;
            let circuit: xorae_core::Circuit =
                serde_json::from_str(&text).context("parsing circuit")?;
            circuit.validate().map_err(anyhow::Error::msg)?;
            let depth = circuit.depth();
            let aig = circuit.aig_nodes();
            println!(
                "circuit: {} inputs, {} gates, {} outputs | depth {depth} | active {} | AIG nodes {aig}",
                circuit.n_inputs,
                circuit.gates.len(),
                circuit.outputs.len(),
                circuit.active_gates(),
            );
            let mut fanout = std::collections::HashMap::new();
            for gate in &circuit.gates {
                *fanout.entry(gate.a).or_insert(0usize) += 1;
                *fanout.entry(gate.b).or_insert(0usize) += 1;
            }
            let max_fanout = fanout.values().copied().max().unwrap_or(0);
            println!("max fanout {max_fanout}");
            Ok(())
        }
        CliCommand::EvalCircuit {
            model,
            data,
            circuit,
            features,
            group_size,
        } => {
            let model = load_model(&model)?;
            let dataset = Dataset::load(&data)?;
            if !dataset.has_labels() {
                bail!("dataset {} has no labels", data.display());
            }
            let text = std::fs::read_to_string(&circuit)
                .with_context(|| format!("reading {}", circuit.display()))?;
            let circuit: xorae_core::Circuit =
                serde_json::from_str(&text).context("parsing circuit")?;
            let (_, _, test) = dataset.split_indices(model.config.seed);
            let pixels = features == "pixels";
            let keys: Vec<Vec<u8>> = test
                .iter()
                .map(|&i| -> Result<Vec<u8>> {
                    if pixels {
                        return Ok(dataset.row_bits(i));
                    }
                    model
                        .infer_bits(&dataset.row_bits(i))
                        .map(|(k, _)| k)
                        .map_err(anyhow::Error::msg)
                })
                .collect::<Result<_>>()?;
            let labels: Vec<u8> = test.iter().map(|&i| dataset.labels[i]).collect();
            let classes = dataset.label_count.max(1);
            let acc = nand_classify::accuracy(&circuit, &keys, &labels, classes, group_size);
            let hist = nand_classify::prediction_histogram(&circuit, &keys, classes, group_size);
            println!(
                "circuit: {} inputs, {} gates, {} classes, group {group_size} | test acc {:.4}",
                circuit.n_inputs,
                circuit.gates.len(),
                classes,
                acc
            );
            println!("predicted distribution: {:?}", hist);

            // Per-class recall and a confusion matrix: overall accuracy is
            // frequency-weighted, so a rare class can be badly served without
            // moving the headline number.
            let mut matrix = vec![vec![0usize; classes]; classes];
            for (key, &label) in keys.iter().zip(labels.iter()) {
                let predicted = nand_classify::predict(&circuit, key, classes, group_size);
                if (label as usize) < classes && predicted < classes {
                    matrix[label as usize][predicted] += 1;
                }
            }
            let mut recalls = Vec::with_capacity(classes);
            for (label, row) in matrix.iter().enumerate() {
                let total: usize = row.iter().sum();
                let recall = if total > 0 {
                    row[label] as f32 / total as f32
                } else {
                    0.0
                };
                recalls.push(recall);
                println!("  {label}: recall {recall:.3}  counts {:?}", row);
            }
            let macro_recall: f32 = recalls.iter().sum::<f32>() / classes.max(1) as f32;
            let worst = recalls
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, r)| (i, *r))
                .unwrap_or((0, 0.0));
            println!(
                "macro recall {macro_recall:.4} | worst class {} ({:.3})",
                worst.0, worst.1
            );
            Ok(())
        }
        CliCommand::Prior {
            model,
            data,
            out,
            samples,
            epochs,
            lr,
            seed,
        } => {
            let model = load_model(&model)?;
            let dataset = Dataset::load(&data)?;
            let reports = prior::run_prior(
                &model,
                &dataset,
                &out,
                prior::PriorOptions {
                    samples,
                    epochs,
                    lr,
                    seed,
                },
            )?;
            println!(
                "prior: {:>16} {:>9} {:>9} {:>9} {:>9}",
                "", "marg MAE", "diversity", "roundtrip", "NLL bits"
            );
            for (name, r) in reports {
                let nll = if r.nll_bits.is_nan() {
                    "   n/a".to_string()
                } else {
                    format!("{:6.3}", r.nll_bits)
                };
                println!(
                    "prior: {:>16} {:>9.4} {:>9.4} {:>9.4} {:>9}",
                    name, r.marginal_mae, r.diversity, r.roundtrip, nll
                );
            }
            println!("images written to {}", out.display());
            Ok(())
        }
        CliCommand::Prune {
            model,
            data,
            out,
            target,
            tolerance,
            calibrate,
            max_removals,
        } => {
            let mut model = load_model(&model)?;
            let dataset = Dataset::load(&data)?;
            let opts = nand::PruneOptions {
                target,
                tolerance,
                calibrate,
                max_removals,
            };
            let (encoder, decoder, stats) = nand::prune_gates(
                &model.encoder,
                &model.decoder,
                &dataset,
                model.config.seed,
                opts,
            );
            model.encoder = encoder;
            model.decoder = decoder;
            model.validate().map_err(anyhow::Error::msg)?;

            let (_, _, test) = dataset.split_indices(model.config.seed);
            let report = ga::evaluate_model(&model, &dataset, &test)?;
            model.metrics.test_ber = report.ber;
            model.metrics.abs_mae = report.abs_mae;
            model.metrics.color_mae = report.color_mae;
            model.metrics.psnr = report.psnr;
            model.metrics.index_accuracy = report.index_accuracy;
            model.metrics.active_gates =
                model.encoder.active_gates() + model.decoder.active_gates();

            println!(
                "prune: {} -> {} gates ({} removed) | calibration BER {:.4} -> {:.4} | test BER {:.4} | AIG nodes {}",
                stats.gates_before,
                stats.gates_after,
                stats.removed,
                stats.ber_before,
                stats.ber_after,
                report.ber,
                model.encoder.aig_nodes() + model.decoder.aig_nodes()
            );
            if let Some(parent) = out.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&out, model.to_json().map_err(anyhow::Error::msg)?)?;
            println!("wrote {}", out.display());
            Ok(())
        }
        CliCommand::Simplify {
            model,
            out,
            verify,
            verify_random,
        } => {
            let model = load_model(&model)?;
            let (encoder, enc_stats) = model.encoder.simplify();
            let (decoder, dec_stats) = model.decoder.simplify();
            let mut simplified = model.clone();
            simplified.encoder = encoder;
            simplified.decoder = decoder;
            simplified.validate().map_err(anyhow::Error::msg)?;
            if verify_random > 0 {
                verify_random_inputs(&model, &simplified, verify_random, model.config.seed)?;
                println!(
                    "verified: identical keys and reconstructions on {verify_random} random inputs"
                );
            }
            if let Some(data) = verify {
                let dataset = Dataset::load(&data)?;
                let idx: Vec<usize> = (0..dataset.n).collect();
                verify_identical(&model, &simplified, &dataset, &idx)?;
                println!(
                    "verified: identical reconstructions on all {} samples",
                    idx.len()
                );
            }
            let gates_before = model.encoder.gates.len() + model.decoder.gates.len();
            let gates_after = simplified.encoder.gates.len() + simplified.decoder.gates.len();
            println!(
                "simplify: {} -> {} NAND gates ({} fewer, {:.1}%)",
                gates_before,
                gates_after,
                gates_before - gates_after,
                100.0 * (gates_before - gates_after) as f32 / gates_before as f32
            );
            println!(
                "  encoder: {} -> {} (const {}, identity {}, deduped {}, dead {})",
                enc_stats.gates_before,
                enc_stats.gates_after,
                enc_stats.folded_const,
                enc_stats.folded_identity,
                enc_stats.deduped,
                enc_stats.dead
            );
            println!(
                "  decoder: {} -> {} (const {}, identity {}, deduped {}, dead {})",
                dec_stats.gates_before,
                dec_stats.gates_after,
                dec_stats.folded_const,
                dec_stats.folded_identity,
                dec_stats.deduped,
                dec_stats.dead
            );
            println!(
                "  AIG nodes (inversions free): {} -> {}",
                model.encoder.aig_nodes() + model.decoder.aig_nodes(),
                simplified.encoder.aig_nodes() + simplified.decoder.aig_nodes()
            );
            simplified.metrics.active_gates =
                simplified.encoder.active_gates() + simplified.decoder.active_gates();
            if let Some(parent) = out.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&out, simplified.to_json().map_err(anyhow::Error::msg)?)?;
            println!("wrote {}", out.display());
            Ok(())
        }
        CliCommand::Baselines {
            data,
            key_bits,
            seed,
        } => {
            let dataset = Dataset::load(&data)?;
            for baseline in baselines::run_all(&dataset, seed, key_bits) {
                println!("{:<20} {:.4}", baseline.name, baseline.ber);
            }
            Ok(())
        }
        CliCommand::ExportSamples {
            data,
            toy,
            cifar_dir,
            class,
            out,
            count,
            scale,
            seed,
            prefix,
        } => {
            if let Some(cifar_dir) = cifar_dir {
                render::export_cifar_samples(&cifar_dir, class, &out, count, scale, &prefix)?;
                println!(
                    "wrote {} CIFAR previews to {}",
                    count,
                    out.join("samples").display()
                );
                return Ok(());
            }
            let dataset = if toy {
                toy::linear_subspace(8, 4, seed)?
            } else {
                let path =
                    data.context("--data is required unless --toy or --cifar-dir is used")?;
                Dataset::load(&path)?
            };
            let (_train, val, test) = dataset.split_indices(seed);
            let pool = if !test.is_empty() { &test } else { &val };
            let indices = spread(pool, count);
            render::export_samples(&dataset, &indices, &out, scale, &prefix)?;
            println!(
                "wrote {} samples to {}",
                indices.len(),
                out.join("samples").display()
            );
            Ok(())
        }
    }
}

fn parse_binarize(value: &str) -> Result<BinarizeMode> {
    match value {
        "threshold" => Ok(BinarizeMode::Threshold),
        "otsu" => Ok(BinarizeMode::Otsu),
        "dither" => Ok(BinarizeMode::Dither),
        "edge" => Ok(BinarizeMode::Edge),
        other => bail!("unknown binarize mode {other:?} (use threshold, otsu, dither or edge)"),
    }
}

/// Check that two models produce identical keys and reconstructions on `idx`.
fn verify_identical(a: &Model, b: &Model, dataset: &Dataset, idx: &[usize]) -> Result<()> {
    for &i in idx {
        let bits = dataset.row_bits(i);
        let (key_a, rec_a) = a.infer_bits(&bits).map_err(anyhow::Error::msg)?;
        let (key_b, rec_b) = b.infer_bits(&bits).map_err(anyhow::Error::msg)?;
        if key_a != key_b || rec_a != rec_b {
            bail!("sample {i}: simplified model is not equivalent");
        }
    }
    Ok(())
}

/// Mean absolute RGB error between the reconstruction and the *original*
/// (unquantised) image, normalised to `0..=1`.
///
/// This is the honest visual-fidelity number for colour models: MAE against
/// the quantised target can look identical for two different palettes even
/// when one of them reconstructs the original much more closely.
fn absolute_mae(model: &Model, dataset: &Dataset, indices: &[usize]) -> Result<f32> {
    let mut total = 0u64;
    let mut fields = 0usize;
    for &i in indices {
        let bits = dataset.row_bits(i);
        let (_, recon) = model.infer_bits(&bits).map_err(anyhow::Error::msg)?;
        let rgb = model.bits_to_rgb(&recon);
        let original = dataset.row_rgb(i);
        for (a, b) in rgb.iter().zip(original.iter()) {
            total += (*a as i32 - *b as i32).unsigned_abs() as u64;
            fields += 1;
        }
    }
    Ok(total as f32 / (fields.max(1) as f32 * 255.0))
}

/// Compare two models on pseudo-random input vectors (off-distribution check).
fn verify_random_inputs(a: &Model, b: &Model, count: usize, seed: u64) -> Result<()> {
    let n = a.input_bits();
    let mut state = seed | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for sample in 0..count {
        let mut bits = vec![0u8; n];
        for (i, chunk) in bits.chunks_mut(64).enumerate() {
            let word = next();
            for (j, bit) in chunk.iter_mut().enumerate() {
                *bit = ((word >> j) & 1) as u8;
            }
            let _ = i;
        }
        let (ka, ra) = a.infer_bits(&bits).map_err(anyhow::Error::msg)?;
        let (kb, rb) = b.infer_bits(&bits).map_err(anyhow::Error::msg)?;
        if ka != kb || ra != rb {
            bail!("random input {sample}: simplified model is not equivalent");
        }
    }
    Ok(())
}

fn load_model(path: &PathBuf) -> Result<Model> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading model {}", path.display()))?;
    Model::from_json(&text).map_err(anyhow::Error::msg)
}

fn load_eval_dataset(data: Option<PathBuf>, toy: bool, model: &Model) -> Result<Dataset> {
    if toy {
        toy::linear_subspace(
            model.config.image_size,
            model.config.key_bits,
            model.config.seed,
        )
    } else {
        let path = data.context("--data is required unless --toy is used")?;
        Dataset::load(&path)
    }
}

fn split_indices(dataset: &Dataset, seed: u64, split: &str) -> Result<Vec<usize>> {
    let (train, val, test) = dataset.split_indices(seed);
    match split {
        "train" => Ok(train),
        "val" | "validation" => Ok(val),
        "test" => Ok(test),
        other => bail!("unknown split {other:?} (use train, val or test)"),
    }
}

fn spread(pool: &[usize], count: usize) -> Vec<usize> {
    if pool.is_empty() || count == 0 {
        return Vec::new();
    }
    let count = count.min(pool.len());
    (0..count).map(|i| pool[i * pool.len() / count]).collect()
}

fn fetch_data(out: &PathBuf) -> Result<()> {
    const URL: &str = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz";
    const MD5: &str = "c32a1d4ab5d03f1284b67883e8d87530";

    std::fs::create_dir_all(out)?;
    let extracted = out.join("cifar-10-batches-bin");
    if extracted.exists() {
        println!("already present: {}", extracted.display());
        return Ok(());
    }

    let archive = out.join("cifar-10-binary.tar.gz");
    if !archive.exists() || !md5_matches(&archive, MD5) {
        println!("downloading or resuming {URL}");
        let status = ProcessCommand::new("curl")
            .args(["-L", "--fail", "-C", "-", "-o"])
            .arg(&archive)
            .arg(URL)
            .status()
            .context("running curl; install curl or download the archive manually")?;
        anyhow::ensure!(status.success(), "curl failed with {status}");
    }

    if !md5_matches(&archive, MD5) {
        eprintln!("warning: CIFAR-10 archive checksum did not match the expected md5");
    }

    println!("extracting {}", archive.display());
    let status = ProcessCommand::new("tar")
        .args(["-xzf"])
        .arg(&archive)
        .arg("-C")
        .arg(out)
        .status()
        .context("running tar")?;
    anyhow::ensure!(status.success(), "tar failed with {status}");
    println!("dataset ready at {}", extracted.display());
    Ok(())
}

fn md5_matches(path: &PathBuf, expected: &str) -> bool {
    ProcessCommand::new("md5sum")
        .arg(path)
        .output()
        .map(|output| String::from_utf8_lossy(&output.stdout).contains(expected))
        .unwrap_or(false)
}
