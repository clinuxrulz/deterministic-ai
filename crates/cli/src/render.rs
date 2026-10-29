use std::path::Path;

use anyhow::{bail, Context, Result};
use image::{Rgb, RgbImage};
use xorae_core::Model;

use crate::dataset::{Dataset, CIFAR_RECORD};

const GAP: u32 = 6;
const BACKGROUND: Rgb<u8> = Rgb([22, 24, 29]);
const BORDER: Rgb<u8> = Rgb([70, 74, 84]);

/// Render `indices` as a grid of [original, binarized, reconstruction] tiles.
pub fn render_reconstructions(
    dataset: &Dataset,
    model: &Model,
    indices: &[usize],
    out: &Path,
    scale: u32,
) -> Result<()> {
    if indices.is_empty() {
        anyhow::bail!("no indices to render");
    }
    let size = dataset.size;
    let tile = (size as u32) * scale;
    let columns = 3u32;
    let rows = indices.len() as u32;
    let width = columns * tile + (columns + 1) * GAP;
    let height = rows * tile + (rows + 1) * GAP;
    let mut canvas = RgbImage::from_pixel(width, height, BACKGROUND);

    for (row, &index) in indices.iter().enumerate() {
        let y0 = GAP + row as u32 * (tile + GAP);
        let gray = dataset.row_gray(index);
        let bits = dataset.row_bits(index);
        let (_, recon) = model
            .infer_bits(&bits)
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("running model on sample {index}"))?;

        for (column, kind) in [Kind::Gray, Kind::Bits, Kind::Bits].into_iter().enumerate() {
            let x0 = GAP + column as u32 * (tile + GAP);
            draw_border(&mut canvas, x0, y0, tile);
            match kind {
                Kind::Gray => draw_gray(&mut canvas, x0, y0, gray, size, scale),
                Kind::Bits => {
                    let source = if column == 1 { &bits } else { &recon };
                    draw_bits(&mut canvas, x0, y0, source, size, scale);
                }
            }
        }
    }

    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    canvas
        .save(out)
        .with_context(|| format!("writing {}", out.display()))?;
    Ok(())
}

enum Kind {
    Gray,
    Bits,
}

fn draw_border(canvas: &mut RgbImage, x0: u32, y0: u32, tile: u32) {
    for x in x0.saturating_sub(1)..=(x0 + tile) {
        if x < canvas.width() {
            canvas.put_pixel(x, y0.saturating_sub(1), BORDER);
            if y0 + tile < canvas.height() {
                canvas.put_pixel(x, y0 + tile, BORDER);
            }
        }
    }
    for y in y0.saturating_sub(1)..=(y0 + tile) {
        if y < canvas.height() {
            canvas.put_pixel(x0.saturating_sub(1), y, BORDER);
            if x0 + tile < canvas.width() {
                canvas.put_pixel(x0 + tile, y, BORDER);
            }
        }
    }
}

fn draw_gray(canvas: &mut RgbImage, x0: u32, y0: u32, gray: &[u8], size: usize, scale: u32) {
    for y in 0..size {
        for x in 0..size {
            let value = gray[y * size + x];
            let color = Rgb([value, value, value]);
            fill_block(
                canvas,
                x0 + x as u32 * scale,
                y0 + y as u32 * scale,
                scale,
                color,
            );
        }
    }
}

fn draw_bits(canvas: &mut RgbImage, x0: u32, y0: u32, bits: &[u8], size: usize, scale: u32) {
    for y in 0..size {
        for x in 0..size {
            let on = bits[y * size + x] & 1 == 1;
            let color = if on {
                Rgb([240, 240, 245])
            } else {
                Rgb([12, 12, 16])
            };
            fill_block(
                canvas,
                x0 + x as u32 * scale,
                y0 + y as u32 * scale,
                scale,
                color,
            );
        }
    }
}

fn fill_block(canvas: &mut RgbImage, x0: u32, y0: u32, scale: u32, color: Rgb<u8>) {
    for dy in 0..scale {
        for dx in 0..scale {
            let x = x0 + dx;
            let y = y0 + dy;
            if x < canvas.width() && y < canvas.height() {
                canvas.put_pixel(x, y, color);
            }
        }
    }
}

/// Write individual sample PNGs and a `samples/index.json` manifest for the
/// web gallery.
pub fn export_samples(
    dataset: &Dataset,
    indices: &[usize],
    asset_root: &Path,
    scale: u32,
    prefix: &str,
) -> Result<()> {
    let dir = asset_root.join("samples");
    std::fs::create_dir_all(&dir)?;
    let size = dataset.size;
    let tile = (size as u32) * scale;
    let mut entries = Vec::new();

    for (i, &index) in indices.iter().enumerate() {
        let mut image = RgbImage::new(tile, tile);
        let gray = dataset.row_gray(index);
        for y in 0..size {
            for x in 0..size {
                let value = gray[y * size + x];
                let color = Rgb([value, value, value]);
                fill_block(&mut image, x as u32 * scale, y as u32 * scale, scale, color);
            }
        }
        let name = format!("{prefix}-{i:02}.png");
        image
            .save(dir.join(&name))
            .with_context(|| format!("writing sample {name}"))?;
        entries.push(serde_json::json!({
            "name": format!("{prefix} {i}"),
            "path": format!("samples/{name}"),
        }));
    }

    let manifest = serde_json::json!({ "samples": entries });
    std::fs::write(
        asset_root.join("samples/index.json"),
        serde_json::to_string_pretty(&manifest)?,
    )?;
    Ok(())
}

/// Export 32×32 RGB previews straight from CIFAR's test batch.
///
/// The model still receives a 16×16 binarized image; the preview simply shows
/// the source photo at its native resolution so the demo input is recognizable.
pub fn export_cifar_samples(
    cifar_dir: &Path,
    class: u8,
    asset_root: &Path,
    count: usize,
    scale: u32,
    prefix: &str,
) -> Result<()> {
    let path = cifar_dir.join("test_batch.bin");
    let data = std::fs::read(&path)
        .with_context(|| format!("reading CIFAR test batch {}", path.display()))?;
    if data.len() % CIFAR_RECORD != 0 {
        bail!(
            "{} has length {}, which is not a multiple of {}",
            path.display(),
            data.len(),
            CIFAR_RECORD
        );
    }

    let offsets: Vec<usize> = data
        .chunks_exact(CIFAR_RECORD)
        .enumerate()
        .filter_map(|(i, record)| (record[0] == class).then_some(i))
        .collect();
    if offsets.is_empty() {
        bail!("no class {class} images in {}", path.display());
    }

    let dir = asset_root.join("samples");
    std::fs::create_dir_all(&dir)?;
    let count = count.min(offsets.len());
    let mut entries = Vec::with_capacity(count);

    for i in 0..count {
        let record_index = offsets[i * offsets.len() / count];
        let record = &data[record_index * CIFAR_RECORD..(record_index + 1) * CIFAR_RECORD];
        let pixels = &record[1..];
        let (red, rest) = pixels.split_at(32 * 32);
        let (green, blue) = rest.split_at(32 * 32);

        let tile = 32 * scale;
        let mut image = RgbImage::new(tile, tile);
        for y in 0..32u32 {
            for x in 0..32u32 {
                let pixel = (y * 32 + x) as usize;
                let color = Rgb([red[pixel], green[pixel], blue[pixel]]);
                fill_block(&mut image, x * scale, y * scale, scale, color);
            }
        }

        let name = format!("{prefix}-{i:02}.png");
        image
            .save(dir.join(&name))
            .with_context(|| format!("writing sample {name}"))?;
        entries.push(serde_json::json!({
            "name": format!("{prefix} {i}"),
            "path": format!("samples/{name}"),
        }));
    }

    let manifest = serde_json::json!({ "samples": entries });
    std::fs::write(
        asset_root.join("samples/index.json"),
        serde_json::to_string_pretty(&manifest)?,
    )?;
    Ok(())
}
