use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use xorae_core::{BinarizeMode, GateSet};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainConfig {
    pub image_size: usize,
    pub key_bits: usize,
    pub encoder_gates: usize,
    pub decoder_gates: usize,
    pub gate_set: GateSet,
    pub binarize: BinarizeMode,
    pub threshold: u8,

    pub population: usize,
    pub offspring: usize,
    pub generations: usize,
    pub tournament: usize,
    pub mutation_rate: f32,
    pub crossover_rate: f32,

    pub train_batch: usize,
    pub val_batch: usize,
    pub entropy_weight: f32,
    #[serde(default = "default_uniqueness_weight")]
    pub uniqueness_weight: f32,
    pub parsimony_weight: f32,

    pub seed: u64,
    pub val_every: usize,
    pub checkpoint_every: usize,
    pub stagnation_limit: usize,
    pub log_every: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            image_size: 16,
            key_bits: 32,
            encoder_gates: 512,
            decoder_gates: 512,
            gate_set: GateSet::XorAnd,
            binarize: BinarizeMode::Dither,
            threshold: 128,
            population: 128,
            offspring: 128,
            generations: 2_000,
            tournament: 4,
            mutation_rate: 0.01,
            crossover_rate: 0.10,
            train_batch: 512,
            val_batch: 512,
            entropy_weight: 0.05,
            uniqueness_weight: 0.01,
            parsimony_weight: 0.001,
            seed: 42,
            val_every: 25,
            checkpoint_every: 100,
            stagnation_limit: 300,
            log_every: 1,
        }
    }
}

impl TrainConfig {
    pub fn from_toml(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config {}", path.display()))?;
        let cfg: Self =
            toml::from_str(&text).with_context(|| format!("parsing config {}", path.display()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn to_toml(&self) -> Result<String> {
        toml::to_string_pretty(self).context("serializing config")
    }

    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(self.image_size > 0, "image_size must be > 0");
        anyhow::ensure!(self.key_bits > 0, "key_bits must be > 0");
        anyhow::ensure!(
            self.key_bits < self.image_size * self.image_size,
            "key must be smaller than the input (otherwise this is not compression)"
        );
        anyhow::ensure!(self.population > 0, "population must be > 0");
        anyhow::ensure!(self.offspring > 0, "offspring must be > 0");
        anyhow::ensure!(self.tournament > 0, "tournament must be > 0");
        anyhow::ensure!(self.generations > 0, "generations must be > 0");
        anyhow::ensure!(
            self.encoder_gates > 0 && self.decoder_gates > 0,
            "encoder and decoder need at least one gate"
        );
        Ok(())
    }
}

fn default_uniqueness_weight() -> f32 {
    0.01
}

pub fn toy_config() -> TrainConfig {
    TrainConfig {
        image_size: 8,
        key_bits: 4,
        encoder_gates: 128,
        decoder_gates: 128,
        gate_set: GateSet::XorOnly,
        binarize: BinarizeMode::Threshold,
        threshold: 128,
        population: 128,
        offspring: 128,
        generations: 600,
        tournament: 4,
        mutation_rate: 0.02,
        crossover_rate: 0.10,
        train_batch: 16,
        val_batch: 16,
        entropy_weight: 0.05,
        uniqueness_weight: 0.20,
        parsimony_weight: 0.0,
        seed: 42,
        val_every: 20,
        checkpoint_every: 100,
        stagnation_limit: 150,
        log_every: 1,
    }
}
