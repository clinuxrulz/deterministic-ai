use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use xorae::config::toy_config;
use xorae::ga::{evaluate_model, run_ga};
use xorae::toy::linear_subspace;
use xorae_core::Model;

fn temp_run_dir(label: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("xorae-{label}-{}-{nanos}", std::process::id()))
}

fn small_toy_config() -> xorae::config::TrainConfig {
    let mut cfg = toy_config();
    cfg.image_size = 4;
    cfg.key_bits = 2;
    cfg.encoder_gates = 64;
    cfg.decoder_gates = 64;
    cfg.population = 96;
    cfg.offspring = 96;
    cfg.generations = 1_500;
    cfg.stagnation_limit = 300;
    cfg.checkpoint_every = 0;
    cfg.log_every = 100;
    cfg
}

/// Strict XOR can represent a linear subspace exactly. Training on the zero
/// pattern plus the basis vectors must generalise to held-out XOR combinations.
#[test]
fn toy_autoencoder_learns_and_generalises() {
    let cfg = small_toy_config();
    let dataset = linear_subspace(cfg.image_size, cfg.key_bits, cfg.seed).unwrap();
    let dir = temp_run_dir("toy");
    let best = run_ga(&dataset, &cfg, &dir).unwrap();

    let json = std::fs::read_to_string(dir.join("model.json")).unwrap();
    let model = Model::from_json(&json).unwrap();
    let (_train, _val, test) = dataset.split_indices(cfg.seed);
    let report = evaluate_model(&model, &dataset, &test).unwrap();

    assert!(best.ber < 0.02, "train BER too high: {}", best.ber);
    assert!(
        report.ber < 0.02,
        "held-out BER too high: {} (exact {})",
        report.ber,
        report.exact_rate
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn ga_is_deterministic_for_a_fixed_seed() {
    let mut cfg = small_toy_config();
    cfg.generations = 10;
    cfg.log_every = 0;

    let dataset = linear_subspace(cfg.image_size, cfg.key_bits, cfg.seed).unwrap();
    let dir_a = temp_run_dir("det-a");
    let dir_b = temp_run_dir("det-b");

    let a = run_ga(&dataset, &cfg, &dir_a).unwrap();
    let b = run_ga(&dataset, &cfg, &dir_b).unwrap();

    assert_eq!(a.ber, b.ber);
    assert_eq!(a.fitness, b.fitness);
    assert_eq!(
        std::fs::read_to_string(dir_a.join("model.json")).unwrap(),
        std::fs::read_to_string(dir_b.join("model.json")).unwrap()
    );

    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}

#[test]
fn bundled_toy_model_loads_and_reconstructs_held_out_patterns() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../web/public/models/toy.json");
    let json = std::fs::read_to_string(path).unwrap();
    let model = Model::from_json(&json).unwrap();
    assert_eq!(model.config.image_size, 8);
    assert_eq!(model.config.key_bits, 4);

    let dataset = linear_subspace(8, 4, 42).unwrap();
    let (_train, _val, test) = dataset.split_indices(42);
    let report = evaluate_model(&model, &dataset, &test).unwrap();
    assert_eq!(report.ber, 0.0, "bundled toy model should be exact");
}
