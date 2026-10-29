//! Thin `wasm-bindgen` bridge around `xorae-core`.
//!
//! The browser app owns the UI and image decoding. Rust owns model parsing,
//! preprocessing, encoding and decoding. Keeping this surface small means the
//! WASM bundle contains only the core evaluator, not a UI framework.

use wasm_bindgen::prelude::*;
use xorae_core::preprocess::{
    rgba_to_bits, rgba_to_bits_palette, rgba_to_bits_rgb1, rgba_to_bits_rgb2, BinarizeMode,
    PreprocessOptions,
};
use xorae_core::{ColorMode, Model};

fn js_error(message: impl std::fmt::Display) -> JsValue {
    js_sys::Error::new(&message.to_string()).into()
}

fn parse_mode(mode: &str) -> Result<BinarizeMode, String> {
    match mode {
        "threshold" => Ok(BinarizeMode::Threshold),
        "otsu" => Ok(BinarizeMode::Otsu),
        "dither" => Ok(BinarizeMode::Dither),
        other => Err(format!("unknown binarize mode: {other}")),
    }
}

fn mode_name(mode: BinarizeMode) -> &'static str {
    match mode {
        BinarizeMode::Threshold => "threshold",
        BinarizeMode::Otsu => "otsu",
        BinarizeMode::Dither => "dither",
    }
}

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// An opaque inference handle created from the portable model JSON.
#[wasm_bindgen]
pub struct ModelHandle {
    inner: Model,
}

#[wasm_bindgen]
impl ModelHandle {
    #[wasm_bindgen(constructor)]
    pub fn new(json: &str) -> Result<ModelHandle, JsValue> {
        let inner = Model::from_json(json).map_err(js_error)?;
        Ok(Self { inner })
    }

    /// Encode a binarized image (`image_size²` bytes, 0/1) to key bits.
    pub fn encode(&self, input: &[u8]) -> Result<Vec<u8>, JsValue> {
        self.inner
            .infer_bits(input)
            .map(|(key, _)| key)
            .map_err(js_error)
    }

    /// Decode key bits back to a binarized image.
    pub fn decode_key(&self, key: &[u8]) -> Result<Vec<u8>, JsValue> {
        self.inner.decode_key(key).map_err(js_error)
    }

    /// Decode key bits directly to interleaved RGB pixels.
    pub fn decode_rgb(&self, key: &[u8]) -> Result<Vec<u8>, JsValue> {
        self.inner.decode_rgb(key).map_err(js_error)
    }

    /// Convert encoded bits (input or reconstruction) to RGB pixels.
    pub fn bits_to_rgb(&self, bits: &[u8]) -> Vec<u8> {
        self.inner.bits_to_rgb(bits)
    }

    /// Preprocess browser RGBA pixels using the model's color mode.
    pub fn preprocess(&self, rgba: &[u8], width: usize, height: usize) -> Result<Vec<u8>, JsValue> {
        match self.inner.config.color_mode {
            ColorMode::Grayscale => Ok(rgba_to_bits(
                width,
                height,
                rgba,
                PreprocessOptions {
                    size: self.inner.config.image_size,
                    mode: self.inner.config.binarize,
                    threshold: self.inner.config.threshold,
                },
            )),
            ColorMode::Palette => Ok(rgba_to_bits_palette(
                width,
                height,
                rgba,
                self.inner.config.image_size,
                &self.inner.config.palette,
            )),
            ColorMode::Rgb1 => Ok(rgba_to_bits_rgb1(
                width,
                height,
                rgba,
                self.inner.config.image_size,
                self.inner.config.threshold,
            )),
            ColorMode::Rgb2 => Ok(rgba_to_bits_rgb2(
                width,
                height,
                rgba,
                self.inner.config.image_size,
            )),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn color_mode(&self) -> String {
        match self.inner.config.color_mode {
            ColorMode::Grayscale => "grayscale",
            ColorMode::Palette => "palette",
            ColorMode::Rgb1 => "rgb1",
            ColorMode::Rgb2 => "rgb2",
        }
        .to_string()
    }

    #[wasm_bindgen(getter)]
    pub fn channels(&self) -> usize {
        self.inner.config.channels
    }

    #[wasm_bindgen(getter)]
    pub fn bits_per_pixel(&self) -> usize {
        self.inner.bits_per_pixel()
    }

    #[wasm_bindgen(getter)]
    pub fn palette(&self) -> Vec<u8> {
        self.inner
            .config
            .palette
            .iter()
            .flat_map(|color| color.iter().copied())
            .collect()
    }

    #[wasm_bindgen(getter)]
    pub fn test_color_mae(&self) -> f32 {
        self.inner.metrics.color_mae
    }

    #[wasm_bindgen(getter)]
    pub fn test_psnr(&self) -> f32 {
        self.inner.metrics.psnr
    }

    #[wasm_bindgen(getter)]
    pub fn test_index_accuracy(&self) -> f32 {
        self.inner.metrics.index_accuracy
    }

    #[wasm_bindgen(getter)]
    pub fn image_size(&self) -> usize {
        self.inner.config.image_size
    }

    #[wasm_bindgen(getter)]
    pub fn key_bits(&self) -> usize {
        self.inner.config.key_bits
    }

    #[wasm_bindgen(getter)]
    pub fn gate_set(&self) -> String {
        format!("{:?}", self.inner.config.gate_set)
    }

    #[wasm_bindgen(getter)]
    pub fn binarize(&self) -> String {
        mode_name(self.inner.config.binarize).to_string()
    }

    #[wasm_bindgen(getter)]
    pub fn threshold(&self) -> u8 {
        self.inner.config.threshold
    }

    #[wasm_bindgen(getter)]
    pub fn seed(&self) -> String {
        self.inner.config.seed.to_string()
    }

    #[wasm_bindgen(getter)]
    pub fn val_ber(&self) -> f32 {
        self.inner.metrics.val_ber
    }

    #[wasm_bindgen(getter)]
    pub fn test_ber(&self) -> f32 {
        self.inner.metrics.test_ber
    }

    #[wasm_bindgen(getter)]
    pub fn key_entropy(&self) -> f32 {
        self.inner.metrics.key_entropy
    }

    #[wasm_bindgen(getter)]
    pub fn active_gates(&self) -> usize {
        self.inner.metrics.active_gates
    }

    #[wasm_bindgen(getter)]
    pub fn gate_count(&self) -> usize {
        self.inner.encoder.gates.len() + self.inner.decoder.gates.len()
    }
}

/// Convert RGBA canvas pixels to binarized bits using the same pipeline as
/// training. This is the only preprocessing entry point the UI needs.
#[wasm_bindgen]
pub fn preprocess_rgba(
    rgba: &[u8],
    width: usize,
    height: usize,
    size: usize,
    mode: &str,
    threshold: u8,
) -> Result<Vec<u8>, JsValue> {
    let mode = parse_mode(mode).map_err(js_error)?;
    Ok(rgba_to_bits(
        width,
        height,
        rgba,
        PreprocessOptions {
            size,
            mode,
            threshold,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_roundtrip() {
        assert_eq!(mode_name(BinarizeMode::Threshold), "threshold");
        assert_eq!(mode_name(BinarizeMode::Otsu), "otsu");
        assert_eq!(mode_name(BinarizeMode::Dither), "dither");
        assert!(parse_mode("threshold").is_ok());
        assert!(parse_mode("nope").is_err());
    }
}
