use serde::{Deserialize, Serialize};

use crate::bits::{bits_to_planes, planes_to_bits};
use crate::circuit::{Circuit, GateSet};
use crate::color::{bits_to_indices, palette_rgb, ColorMode, Rgb};
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
    /// Grayscale for legacy models, Palette for color models.
    #[serde(default)]
    pub color_mode: ColorMode,
    /// Palette entries (RGB). Empty for grayscale models.
    #[serde(default)]
    pub palette: Vec<Rgb>,
    /// 1 for grayscale; `log2(palette.len())` for palette models.
    #[serde(default = "default_bits_per_pixel")]
    pub bits_per_pixel: usize,
    /// 1 for grayscale/palette, 3 for RGB bitplanes.
    #[serde(default = "default_channels")]
    pub channels: usize,
}

fn default_bits_per_pixel() -> usize {
    1
}

fn default_channels() -> usize {
    1
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelMetrics {
    pub train_ber: f32,
    pub val_ber: f32,
    pub test_ber: f32,
    pub key_entropy: f32,
    pub active_gates: usize,
    /// Color metrics (0 for grayscale models).
    #[serde(default)]
    pub color_mae: f32,
    #[serde(default)]
    pub color_mse: f32,
    #[serde(default)]
    pub psnr: f32,
    #[serde(default)]
    pub index_accuracy: f32,
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

    #[inline]
    pub fn bits_per_pixel(&self) -> usize {
        self.config.bits_per_pixel.max(1)
    }

    #[inline]
    pub fn input_bits(&self) -> usize {
        self.config.image_size
            * self.config.image_size
            * self.config.channels.max(1)
            * self.bits_per_pixel()
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.format != Self::FORMAT {
            return Err(format!("unsupported model format: {}", self.format));
        }
        let pixels = self
            .config
            .image_size
            .checked_mul(self.config.image_size)
            .ok_or("image size overflow")?;
        let bpp = self.bits_per_pixel();
        if bpp == 0 || bpp > 8 {
            return Err(format!("bits_per_pixel must be in 1..=8, got {bpp}"));
        }
        let channels = self.config.channels.max(1);
        if channels > 4 {
            return Err(format!("channels must be in 1..=4, got {channels}"));
        }
        let input_bits = pixels
            .checked_mul(channels)
            .and_then(|value| value.checked_mul(bpp))
            .ok_or("input bit count overflow")?;

        match self.config.color_mode {
            ColorMode::Grayscale => {
                if bpp != 1 || channels != 1 {
                    return Err("grayscale models must use 1 channel and bits_per_pixel = 1".into());
                }
            }
            ColorMode::Palette => {
                if channels != 1 {
                    return Err("palette models must use 1 channel".into());
                }
                let expected = 1usize << bpp;
                if self.config.palette.len() != expected {
                    return Err(format!(
                        "palette mode with {bpp} bits/pixel needs {expected} colors, got {}",
                        self.config.palette.len()
                    ));
                }
            }
            ColorMode::Rgb1 => {
                if bpp != 1 || channels != 3 {
                    return Err("rgb1 models must use 3 channels and bits_per_pixel = 1".into());
                }
            }
        }

        if self.encoder.n_inputs as usize != input_bits {
            return Err(format!(
                "encoder expects {} inputs, config implies {input_bits}",
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
        if self.decoder.outputs.len() != input_bits {
            return Err(format!(
                "decoder has {} outputs, config implies {input_bits}",
                self.decoder.outputs.len()
            ));
        }
        self.encoder.validate()?;
        self.decoder.validate()?;
        Ok(())
    }

    /// Run one encoded image through the model.
    ///
    /// For grayscale the input is `image_size²` bits. For palette models it is
    /// `image_size² × bits_per_pixel` bits in pixel-major order.
    ///
    /// Returns `(key_bits, reconstructed_bits)`.
    pub fn infer_bits(&self, input: &[u8]) -> Result<(Vec<u8>, Vec<u8>), String> {
        self.validate()?;
        let expected = self.input_bits();
        if input.len() != expected {
            return Err(format!(
                "expected {expected} input bits, got {}",
                input.len()
            ));
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

    /// Decode a key directly to interleaved RGB pixels.
    pub fn decode_rgb(&self, key: &[u8]) -> Result<Vec<u8>, String> {
        let bits = self.decode_key(key)?;
        Ok(self.bits_to_rgb(&bits))
    }

    /// Convert encoded bits (input or decoder output) to RGB pixels.
    pub fn bits_to_rgb(&self, bits: &[u8]) -> Vec<u8> {
        match self.config.color_mode {
            ColorMode::Palette => {
                let indices = bits_to_indices(bits, self.bits_per_pixel());
                palette_rgb(&indices, &self.config.palette)
            }
            ColorMode::Rgb1 => {
                let pixels = self.config.image_size * self.config.image_size;
                let mut rgb = Vec::with_capacity(pixels * 3);
                for p in 0..pixels {
                    let r = bits.get(p).copied().unwrap_or(0) & 1;
                    let g = bits.get(pixels + p).copied().unwrap_or(0) & 1;
                    let b = bits.get(2 * pixels + p).copied().unwrap_or(0) & 1;
                    rgb.extend_from_slice(&[r * 255, g * 255, b * 255]);
                }
                rgb
            }
            ColorMode::Grayscale => {
                let mut rgb = Vec::with_capacity(bits.len() * 3);
                for &bit in bits {
                    let value = if bit & 1 == 1 { 255 } else { 0 };
                    rgb.extend_from_slice(&[value, value, value]);
                }
                rgb
            }
        }
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
                color_mode: ColorMode::Grayscale,
                palette: vec![],
                bits_per_pixel: 1,
                channels: 1,
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
        assert_eq!(back.config.color_mode, ColorMode::Grayscale);
    }

    #[test]
    fn legacy_json_defaults_to_grayscale() {
        let model = tiny_model();
        let mut value: serde_json::Value = serde_json::from_str(&model.to_json().unwrap()).unwrap();
        let config = value["config"].as_object_mut().unwrap();
        config.remove("color_mode");
        config.remove("palette");
        config.remove("bits_per_pixel");
        config.remove("channels");
        let old = serde_json::to_string(&value).unwrap();
        let parsed = Model::from_json(&old).unwrap();
        assert_eq!(parsed.config.color_mode, ColorMode::Grayscale);
        assert_eq!(parsed.bits_per_pixel(), 1);
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
                color_mode: ColorMode::Grayscale,
                palette: vec![],
                bits_per_pixel: 1,
                channels: 1,
            },
            encoder,
            decoder,
            ModelMetrics::default(),
        );
        model.validate().unwrap();
    }

    #[test]
    fn palette_model_decodes_rgb() {
        // 1x1 palette image, 2 bits/pixel, 4-color palette.
        let palette = vec![[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]];
        let encoder = Circuit {
            n_inputs: 2,
            gates: vec![],
            outputs: vec![0, 1],
        };
        let decoder = Circuit {
            n_inputs: 2,
            gates: vec![],
            outputs: vec![0, 1],
        };
        let model = Model::new(
            ModelConfig {
                image_size: 1,
                key_bits: 2,
                gate_set: GateSet::XorOnly,
                binarize: BinarizeMode::Threshold,
                threshold: 128,
                seed: 1,
                color_mode: ColorMode::Palette,
                palette,
                bits_per_pixel: 2,
                channels: 1,
            },
            encoder,
            decoder,
            ModelMetrics::default(),
        );
        model.validate().unwrap();
        let (key, recon) = model.infer_bits(&[0, 1]).unwrap();
        assert_eq!(key, vec![0, 1]);
        assert_eq!(recon, vec![0, 1]);
        assert_eq!(model.decode_rgb(&key).unwrap(), vec![0, 255, 0]);
    }
}
