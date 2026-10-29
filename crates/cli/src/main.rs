use std::path::PathBuf;
use std::process::Command as ProcessCommand;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use xorae::config::TrainConfig;
use xorae::dataset::Dataset;
use xorae::{baselines, config, dataset, ga, grad, nand, render, toy};
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
        cifar_dir: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 16)]
        size: usize,
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
            seed,
        } => {
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
        } => {
            let mut cfg = nand::NandConfig::from_toml(&config)?;
            if let Some(seed) = seed {
                cfg.seed = seed;
            }
            if let Some(epochs) = epochs {
                cfg.epochs = epochs;
            }
            cfg.validate()?;
            let dataset = Dataset::load(&data)?;
            println!(
                "training NAND wiring: {}x{}, {}-bit key, {} epochs (seed {})",
                cfg.image_size, cfg.image_size, cfg.key_bits, cfg.epochs, cfg.seed
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
        } => {
            let model = load_model(&model)?;
            let dataset = load_eval_dataset(data, toy, &model)?;
            let indices = split_indices(&dataset, model.config.seed, &split)?;
            let report = ga::evaluate_model(&model, &dataset, &indices)?;
            if dataset.color_mode != ColorMode::Grayscale {
                println!(
                    "{}: color MAE {:.4}, PSNR {:.2} dB, pixel/index accuracy {:.2}%, bit BER {:.4}, exact {:.2}%, active gates {} ({} samples)",
                    split,
                    report.color_mae,
                    report.psnr,
                    report.index_accuracy * 100.0,
                    report.ber,
                    report.exact_rate * 100.0,
                    report.active_gates,
                    report.n
                );
            } else {
                println!(
                    "{}: BER {:.4}, exact {:.2}%, key entropy {:.3}, active gates {} ({} samples)",
                    split,
                    report.ber,
                    report.exact_rate * 100.0,
                    report.key_entropy,
                    report.active_gates,
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
        CliCommand::Export { model, out } => {
            let model = load_model(&model)?;
            let json = model.to_json().map_err(anyhow::Error::msg)?;
            if let Some(parent) = out.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&out, json)?;
            println!("exported model to {}", out.display());
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
        other => bail!("unknown binarize mode {other:?} (use threshold, otsu or dither)"),
    }
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
