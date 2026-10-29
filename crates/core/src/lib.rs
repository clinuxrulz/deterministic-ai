//! `xorae-core` contains the parts of the experiment that must run identically
//! natively and in the browser: bit packing, Boolean circuits, inference and
//! image preprocessing.
//!
//! It deliberately has no `rayon`, filesystem, dataset or image-decoding
//! dependencies so that it can be compiled to `wasm32-unknown-unknown`.

pub mod bits;
pub mod circuit;
pub mod color;
pub mod model;
pub mod preprocess;

pub use bits::{BitBatch, BitMatrix};
pub use circuit::{Circuit, Gate, GateSet, Op};
pub use color::{
    bits_to_indices, color_mae, color_mse, indices_to_bits, nearest_palette_index, palette_rgb,
    psnr_from_mse, ColorMode, Rgb,
};
pub use model::{Model, ModelConfig, ModelMetrics};
pub use preprocess::{BinarizeMode, PreprocessOptions};
