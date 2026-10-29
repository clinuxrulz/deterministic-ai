use serde::{Deserialize, Serialize};

use crate::bits::{bits_to_planes, planes_to_bits};
use crate::circuit::{Circuit, GateSet};
use crate::preprocess::BinarizeMode;

/// Everything needed to reproduce the model's input pipeline and architecture.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    pub image_size: usize,
    pub key_bits: usize,
    pub gate_set: GateSet,
    pub binarize: BinarizeMode,
    pub threshold: u8,
    pub seed: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelMetrics {
    pub train_ber: f32,
    pub val_ber: f32,
    pub test_ber: f32,
    pub key_entropy: f32,
    pub active_gates: usize,
}

/// Portable inference model shared by the CLI and the browser.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Model {
    pub format: String,
    pub config: ModelConfig,
    pub encoder: Circuit,
    pub decoder: Circuit,
    pub metrics: ModelMetrics,
}

impl Model {
    pub const FORMAT: &'static str = "xorae-v1";

    pub fn new(
        config: ModelConfig,
        encoder: Circuit,
        decoder: Circuit,
        metrics: ModelMetrics,
    ) -> Self {
        Self {
            format: Self::FORMAT.to_string(),
            config,
            encoder,
            decoder,
            metrics,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.format != Self::FORMAT {
            return Err(format!("unsupported model format: {}", self.format));
        }
        let n = self
            .config
            .image_size
            .checked_mul(self.config.image_size)
            .ok_or("image size overflow")?;
        if self.encoder.n_inputs as usize != n {
            return Err(format!(
                "encoder expects {} inputs, image size implies {n}",
                self.encoder.n_inputs
            ));
        }
        if self.encoder.outputs.len() != self.config.key_bits {
            return Err(format!(
                "encoder has {} outputs, config says {} key bits",
                self.encoder.outputs.len(),
                self.config.key_bits
            ));
        }
        if self.decoder.n_inputs as usize != self.config.key_bits {
            return Err(format!(
                "decoder expects {} inputs, config says {} key bits",
                self.decoder.n_inputs, self.config.key_bits
            ));
        }
        if self.decoder.outputs.len() != n {
            return Err(format!(
                "decoder has {} outputs, image size implies {n}",
                self.decoder.outputs.len()
            ));
        }
        self.encoder.validate()?;
        self.decoder.validate()?;
        Ok(())
    }

    /// Run one binarized image through the model.
    ///
    /// Returns `(key_bits, reconstructed_bits)`.
    pub fn infer_bits(&self, input: &[u8]) -> Result<(Vec<u8>, Vec<u8>), String> {
        self.validate()?;
        let n = self.config.image_size * self.config.image_size;
        if input.len() != n {
            return Err(format!("expected {n} input bits, got {}", input.len()));
        }

        let input_planes = bits_to_planes(input);
        let mut enc_arena = Vec::new();
        let mut key_planes = Vec::new();
        self.encoder
            .eval_into(&input_planes, 1, &mut enc_arena, &mut key_planes);

        let mut dec_arena = Vec::new();
        let mut recon_planes = Vec::new();
        self.decoder
            .eval_into(&key_planes, 1, &mut dec_arena, &mut recon_planes);

        Ok((planes_to_bits(&key_planes), planes_to_bits(&recon_planes)))
    }

    /// Decode an arbitrary key. Used by the browser when the user toggles key
    /// bits to explore what each bit controls.
    pub fn decode_key(&self, key: &[u8]) -> Result<Vec<u8>, String> {
        self.validate()?;
        if key.len() != self.config.key_bits {
            return Err(format!(
                "expected {} key bits, got {}",
                self.config.key_bits,
                key.len()
            ));
        }
        let key_planes = bits_to_planes(key);
        let mut arena = Vec::new();
        let mut out = Vec::new();
        self.decoder.eval_into(&key_planes, 1, &mut arena, &mut out);
        Ok(planes_to_bits(&out))
    }

    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| e.to_string())
    }

    pub fn from_json(s: &str) -> Result<Self, String> {
        let model: Self = serde_json::from_str(s).map_err(|e| e.to_string())?;
        model.validate()?;
        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::{Gate, Op};

    fn tiny_model() -> Model {
        // 2x2 image, 1-bit key. Encoder: key = x0 XOR x1.
        // Decoder: y0 = key, y1 = key, y2 = key, y3 = key.
        let encoder = Circuit {
            n_inputs: 4,
            gates: vec![Gate {
                op: Op::Xor,
                a: 0,
                b: 1,
            }],
            outputs: vec![6],
        };
        let decoder = Circuit {
            n_inputs: 1,
            gates: vec![],
            outputs: vec![0, 0, 0, 0],
        };
        Model::new(
            ModelConfig {
                image_size: 2,
                key_bits: 1,
                gate_set: GateSet::XorOnly,
                binarize: BinarizeMode::Threshold,
                threshold: 128,
                seed: 1,
            },
            encoder,
            decoder,
            ModelMetrics::default(),
        )
    }

    #[test]
    fn inference_is_deterministic() {
        let model = tiny_model();
        let (key, recon) = model.infer_bits(&[1, 0, 1, 0]).unwrap();
        assert_eq!(key, vec![1]);
        assert_eq!(recon, vec![1, 1, 1, 1]);
    }

    #[test]
    fn json_roundtrip() {
        let model = tiny_model();
        let json = model.to_json().unwrap();
        let back = Model::from_json(&json).unwrap();
        assert_eq!(back.config.key_bits, 1);
        assert_eq!(back.encoder.gates.len(), 1);
    }

    #[test]
    fn random_model_validates() {
        let encoder = Circuit::random(64, 32, 4, GateSet::XorAnd, 3);
        let decoder = Circuit::random(4, 32, 64, GateSet::XorAnd, 4);
        let model = Model::new(
            ModelConfig {
                image_size: 8,
                key_bits: 4,
                gate_set: GateSet::XorAnd,
                binarize: BinarizeMode::Dither,
                threshold: 128,
                seed: 3,
            },
            encoder,
            decoder,
            ModelMetrics::default(),
        );
        model.validate().unwrap();
    }
}
