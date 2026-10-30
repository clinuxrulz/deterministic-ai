//! Training-side library for xorae: datasets, GA, baselines and rendering.
//!
//! The browser-facing inference code lives in `xorae-core`; this crate is
//! native-only and may use parallelism and filesystem access.

pub mod baselines;
pub mod classify;
pub mod config;
pub mod dataset;
pub mod ga;
pub mod grad;
pub mod nand;
pub mod nand_classify;
pub mod prior;
pub mod render;
pub mod toy;
