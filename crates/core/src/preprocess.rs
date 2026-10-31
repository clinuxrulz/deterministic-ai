use serde::{Deserialize, Serialize};

use crate::color::{indices_to_bits, nearest_palette_index, Rgb};

/// How to turn a grayscale image into bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BinarizeMode {
    /// Global threshold (default 128).
    Threshold,
    /// Otsu's method: choose the threshold that maximizes between-class variance.
    Otsu,
    /// Floyd–Steinberg error diffusion. Preserves more visual detail.
    Dither,
    /// Sobel edges, keeping the strongest `EDGE_DENSITY` fraction of pixels.
    /// A floor plan rather than a photo: a hard threshold at 32x32 turns a cat
    /// into an inkblot, while its outline stays legible as a line drawing.
    Edge,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct PreprocessOptions {
    /// Output image is `size × size`.
    pub size: usize,
    pub mode: BinarizeMode,
    pub threshold: u8,
}

impl Default for PreprocessOptions {
    fn default() -> Self {
        Self {
            size: 16,
            mode: BinarizeMode::Dither,
            threshold: 128,
        }
    }
}

/// Convert interleaved RGB bytes to grayscale (ITU-R BT.601 luma).
pub fn rgb_to_gray(width: usize, height: usize, rgb: &[u8]) -> Vec<u8> {
    assert_eq!(rgb.len(), width * height * 3);
    rgb.chunks_exact(3)
        .map(|p| {
            let y = (299 * p[0] as u32 + 587 * p[1] as u32 + 114 * p[2] as u32) / 1000;
            y.min(255) as u8
        })
        .collect()
}

/// Convert interleaved RGBA bytes to grayscale.
pub fn rgba_to_gray(width: usize, height: usize, rgba: &[u8]) -> Vec<u8> {
    assert_eq!(rgba.len(), width * height * 4);
    rgba.chunks_exact(4)
        .map(|p| {
            let y = (299 * p[0] as u32 + 587 * p[1] as u32 + 114 * p[2] as u32) / 1000;
            y.min(255) as u8
        })
        .collect()
}

/// Box-downsample a grayscale image to `size × size`.
pub fn resize_box(gray: &[u8], width: usize, height: usize, size: usize) -> Vec<u8> {
    assert_eq!(gray.len(), width * height);
    assert!(size > 0);
    if width == size && height == size {
        return gray.to_vec();
    }
    let mut out = vec![0u8; size * size];
    for oy in 0..size {
        let y0 = oy * height / size;
        let y1 = ((oy + 1) * height / size).max(y0 + 1);
        for ox in 0..size {
            let x0 = ox * width / size;
            let x1 = ((ox + 1) * width / size).max(x0 + 1);
            let mut sum = 0u32;
            let mut count = 0u32;
            for y in y0..y1 {
                for x in x0..x1 {
                    sum += gray[y * width + x] as u32;
                    count += 1;
                }
            }
            out[oy * size + ox] = (sum / count.max(1)) as u8;
        }
    }
    out
}

/// Otsu threshold for an 8-bit grayscale image.
pub fn otsu_threshold(gray: &[u8]) -> u8 {
    if gray.is_empty() {
        return 128;
    }
    let mut hist = [0u32; 256];
    for &g in gray {
        hist[g as usize] += 1;
    }
    let total = gray.len() as f64;
    let mut sum_all = 0.0f64;
    for (i, &h) in hist.iter().enumerate() {
        sum_all += i as f64 * h as f64;
    }

    let mut sum_back = 0.0f64;
    let mut weight_back = 0.0f64;
    let mut best_variance = -1.0f64;
    let mut best_t = 128u8;

    for (t, &count) in hist.iter().enumerate() {
        weight_back += count as f64;
        if weight_back == 0.0 {
            continue;
        }
        let weight_fore = total - weight_back;
        if weight_fore == 0.0 {
            break;
        }
        sum_back += t as f64 * count as f64;
        let mean_back = sum_back / weight_back;
        let mean_fore = (sum_all - sum_back) / weight_fore;
        let variance = weight_back * weight_fore * (mean_back - mean_fore).powi(2);
        if variance > best_variance {
            best_variance = variance;
            best_t = t as u8;
        }
    }
    best_t
}

/// Binarize a grayscale image, returning a row-major `0/1` vector.
pub fn binarize(gray: &[u8], mode: BinarizeMode, threshold: u8) -> Vec<u8> {
    match mode {
        BinarizeMode::Threshold => binarize_threshold(gray, threshold),
        BinarizeMode::Otsu => binarize_threshold(gray, otsu_threshold(gray)),
        BinarizeMode::Dither => binarize_dither(gray, threshold),
        BinarizeMode::Edge => binarize_edge(gray),
    }
}

/// Fraction of pixels `BinarizeMode::Edge` keeps. A constant density means the
/// all-zero baseline is the same for every image, so errors stay comparable.
pub const EDGE_DENSITY: f64 = 0.15;

/// Sobel gradient magnitude, thresholded at a per-image percentile.
///
/// The percentile rather than an absolute threshold is deliberate: CIFAR cats
/// vary from a black cat in a dark room to a white cat on a bed, and a fixed
/// cutoff either erases the first or saturates the second. Non-maximum
/// suppression was tried and rejected -- at 32x32 it thins the strokes into
/// broken dashes, which read worse than continuous 2 px lines.
pub fn binarize_edge(gray: &[u8]) -> Vec<u8> {
    let n = gray.len();
    let side = (n as f64).sqrt().round() as usize;
    if side * side != n || side < 3 {
        return binarize_threshold(gray, 128);
    }
    let mut mag = vec![0u32; n];
    for y in 1..side - 1 {
        for x in 1..side - 1 {
            let p = |yy: usize, xx: usize| gray[yy * side + xx] as i32;
            let gx = -p(y - 1, x - 1) + p(y - 1, x + 1) - 2 * p(y, x - 1) + 2 * p(y, x + 1)
                - p(y + 1, x - 1)
                + p(y + 1, x + 1);
            let gy = -p(y - 1, x - 1) - 2 * p(y - 1, x) - p(y - 1, x + 1)
                + p(y + 1, x - 1)
                + 2 * p(y + 1, x)
                + p(y + 1, x + 1);
            // Squared magnitude is monotone, so the ranking is unchanged and we
            // skip the square root.
            mag[y * side + x] = (gx * gx + gy * gy) as u32;
        }
    }
    // Rank pixels instead of thresholding at a magnitude: many CIFAR images
    // have large flat regions, so a cutoff ties across thousands of pixels and
    // the density would swing with the content. Ties break on index, which
    // keeps the result deterministic.
    let keep = (((n as f64) * EDGE_DENSITY).round() as usize).clamp(1, n);
    let mut order: Vec<usize> = (0..n).collect();
    if mag.iter().all(|&m| m == 0) {
        return vec![0u8; n];
    }
    order.sort_by(|&i, &j| mag[j].cmp(&mag[i]).then(i.cmp(&j)));
    let mut out = vec![0u8; n];
    for &i in &order[..keep] {
        if mag[i] > 0 {
            out[i] = 1;
        }
    }
    out
}

fn binarize_threshold(gray: &[u8], threshold: u8) -> Vec<u8> {
    gray.iter().map(|&g| u8::from(g >= threshold)).collect()
}

/// Floyd–Steinberg error diffusion. Returns `0/1` bits.
pub fn binarize_dither(gray: &[u8], threshold: u8) -> Vec<u8> {
    // The image is square in this project, but infer the side length so the
    // function remains usable on rectangular inputs.
    let n = gray.len();
    let side = (n as f64).sqrt().round() as usize;
    if side * side != n {
        return binarize_threshold(gray, threshold);
    }
    let mut buf: Vec<i16> = gray.iter().map(|&g| g as i16).collect();
    let mut out = vec![0u8; n];
    for y in 0..side {
        for x in 0..side {
            let idx = y * side + x;
            let old = buf[idx];
            let new = if old >= threshold as i16 { 255 } else { 0 };
            out[idx] = u8::from(new == 255);
            let err = old - new;
            if x + 1 < side {
                buf[idx + 1] += err * 7 / 16;
            }
            if y + 1 < side {
                if x > 0 {
                    buf[(y + 1) * side + x - 1] += err * 3 / 16;
                }
                buf[(y + 1) * side + x] += err * 5 / 16;
                if x + 1 < side {
                    buf[(y + 1) * side + x + 1] += err / 16;
                }
            }
        }
    }
    out
}

/// Convert RGBA pixels to interleaved RGB.
pub fn rgba_to_rgb(width: usize, height: usize, rgba: &[u8]) -> Vec<u8> {
    assert_eq!(rgba.len(), width * height * 4);
    let mut rgb = Vec::with_capacity(width * height * 3);
    for pixel in rgba.chunks_exact(4) {
        rgb.extend_from_slice(&pixel[..3]);
    }
    rgb
}

/// Box-downsample interleaved RGB to `size × size`.
pub fn resize_rgb_box(rgb: &[u8], width: usize, height: usize, size: usize) -> Vec<u8> {
    assert_eq!(rgb.len(), width * height * 3);
    if width == size && height == size {
        return rgb.to_vec();
    }
    let mut out = vec![0u8; size * size * 3];
    for oy in 0..size {
        let y0 = oy * height / size;
        let y1 = ((oy + 1) * height / size).max(y0 + 1);
        for ox in 0..size {
            let x0 = ox * width / size;
            let x1 = ((ox + 1) * width / size).max(x0 + 1);
            let mut sums = [0u32; 3];
            let mut count = 0u32;
            for y in y0..y1 {
                for x in x0..x1 {
                    let i = (y * width + x) * 3;
                    sums[0] += rgb[i] as u32;
                    sums[1] += rgb[i + 1] as u32;
                    sums[2] += rgb[i + 2] as u32;
                    count += 1;
                }
            }
            let o = (oy * size + ox) * 3;
            out[o] = (sums[0] / count.max(1)) as u8;
            out[o + 1] = (sums[1] / count.max(1)) as u8;
            out[o + 2] = (sums[2] / count.max(1)) as u8;
        }
    }
    out
}

/// RGB 1-bit/channel: three channel-major bit planes at `size × size`.
pub fn rgba_to_bits_rgb1(
    width: usize,
    height: usize,
    rgba: &[u8],
    size: usize,
    threshold: u8,
) -> Vec<u8> {
    let rgb = rgba_to_rgb(width, height, rgba);
    let small = resize_rgb_box(&rgb, width, height, size);
    let pixels = size * size;
    let mut bits = Vec::with_capacity(pixels * 3);
    for channel in 0..3 {
        for p in 0..pixels {
            bits.push(u8::from(small[p * 3 + channel] >= threshold));
        }
    }
    bits
}

/// Number of bit planes in an RGB-2 image: two per channel.
pub const RGB2_PLANES: usize = 6;

/// Quantise an 8-bit channel to a 2-bit level in `0..=3`, with rounding.
///
/// The four levels map back to `0, 85, 170, 255`, so the decision boundaries
/// are 42, 128 and 213.
pub fn quantize2(value: u8) -> u8 {
    (((value as u32) * 3 + 127) / 255).min(3) as u8
}

/// Expand a 2-bit level back to an 8-bit channel value.
pub fn level2_to_u8(level: u8) -> u8 {
    (level & 3) * 85
}

/// RGB 2-bit/channel encoded as six channel-major bit planes.
///
/// Plane order is `R_hi, R_lo, G_hi, G_lo, B_hi, B_lo`, each `size × size`.
/// Two bits per channel gives four levels and therefore 64 colours.
pub fn rgba_to_bits_rgb2(width: usize, height: usize, rgba: &[u8], size: usize) -> Vec<u8> {
    let rgb = rgba_to_rgb(width, height, rgba);
    let small = resize_rgb_box(&rgb, width, height, size);
    let pixels = size * size;
    let mut bits = Vec::with_capacity(pixels * RGB2_PLANES);
    for channel in 0..3 {
        let mut hi = vec![0u8; pixels];
        let mut lo = vec![0u8; pixels];
        for p in 0..pixels {
            let level = quantize2(small[p * 3 + channel]);
            hi[p] = (level >> 1) & 1;
            lo[p] = level & 1;
        }
        bits.extend_from_slice(&hi);
        bits.extend_from_slice(&lo);
    }
    bits
}

/// Palette indices encoded as `bits_per_pixel` bits per pixel.
pub fn rgba_to_bits_palette(
    width: usize,
    height: usize,
    rgba: &[u8],
    size: usize,
    palette: &[Rgb],
) -> Vec<u8> {
    assert!(palette.len().is_power_of_two());
    let bpp = palette.len().ilog2() as usize;
    let rgb = rgba_to_rgb(width, height, rgba);
    let small = resize_rgb_box(&rgb, width, height, size);
    let pixels = size * size;
    let mut indices = Vec::with_capacity(pixels);
    for p in 0..pixels {
        indices.push(nearest_palette_index(
            small[p * 3],
            small[p * 3 + 1],
            small[p * 3 + 2],
            palette,
        ));
    }
    indices_to_bits(&indices, bpp)
}

/// Convert RGB pixels to bits using the shared resize/binarize pipeline.
pub fn rgb_to_bits(width: usize, height: usize, rgb: &[u8], opts: PreprocessOptions) -> Vec<u8> {
    let gray = rgb_to_gray(width, height, rgb);
    gray_to_bits(&gray, width, height, opts)
}

/// Convert RGBA pixels to bits using the shared resize/binarize pipeline.
pub fn rgba_to_bits(width: usize, height: usize, rgba: &[u8], opts: PreprocessOptions) -> Vec<u8> {
    let gray = rgba_to_gray(width, height, rgba);
    gray_to_bits(&gray, width, height, opts)
}

/// Resize and binarize an existing grayscale image.
pub fn gray_to_bits(gray: &[u8], width: usize, height: usize, opts: PreprocessOptions) -> Vec<u8> {
    let resized = resize_box(gray, width, height, opts.size);
    binarize(&resized, opts.mode, opts.threshold)
}

/// Turn `0/1` bits back into a black/white grayscale image.
pub fn bits_to_gray(bits: &[u8]) -> Vec<u8> {
    bits.iter()
        .map(|&b| if b & 1 == 1 { 255 } else { 0 })
        .collect()
}

/// Turn `0/1` bits into interleaved RGB.
pub fn bits_to_rgb(bits: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bits.len() * 3);
    for &b in bits {
        let v = if b & 1 == 1 { 255 } else { 0 };
        out.extend_from_slice(&[v, v, v]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_roundtrip() {
        let bits = binarize(&[0, 1, 127, 128, 200, 255], BinarizeMode::Threshold, 128);
        assert_eq!(bits, vec![0, 0, 0, 1, 1, 1]);
    }

    #[test]
    fn box_resize_constant_image() {
        let gray = vec![200u8; 32 * 32];
        let small = resize_box(&gray, 32, 32, 16);
        assert_eq!(small.len(), 256);
        assert!(small.iter().all(|&v| v == 200));
    }

    #[test]
    fn edge_keeps_only_the_step() {
        // Left half black, right half white: the seam is the only gradient.
        let mut gray = vec![0u8; 32 * 32];
        for row in gray.chunks_mut(32) {
            for px in row.iter_mut().skip(16) {
                *px = 255;
            }
        }
        let bits = binarize(&gray, BinarizeMode::Edge, 128);
        assert_eq!(bits.len(), 1024);
        let ones: Vec<usize> = (0..bits.len()).filter(|&i| bits[i] == 1).collect();
        assert!(!ones.is_empty(), "step edge produced no bits");
        for i in ones {
            let x = i % 32;
            assert!(x == 15 || x == 16, "edge bit at x={x}");
        }
    }

    #[test]
    fn edge_is_empty_on_a_flat_image() {
        let bits = binarize(&vec![128u8; 32 * 32], BinarizeMode::Edge, 128);
        assert!(bits.iter().all(|&b| b == 0));
    }

    #[test]
    fn edge_density_is_constant() {
        // A ramp has one gradient direction everywhere, so every interior
        // pixel ties; ranking must still keep exactly the requested count.
        let gray: Vec<u8> = (0..32 * 32).map(|i| ((i % 32) * 8) as u8).collect();
        let bits = binarize(&gray, BinarizeMode::Edge, 128);
        let ones = bits.iter().filter(|&&b| b == 1).count();
        let expected = (1024.0 * EDGE_DENSITY).round() as usize;
        assert_eq!(ones, expected, "density drifted");
    }

    #[test]
    fn otsu_splits_bimodal_image() {
        let mut gray = vec![20u8; 100];
        gray.extend(vec![220u8; 100]);
        let t = otsu_threshold(&gray);
        assert!((20..220).contains(&t), "unexpected threshold {t}");
    }

    #[test]
    fn dither_preserves_average() {
        let gray = vec![128u8; 16 * 16];
        let bits = binarize_dither(&gray, 128);
        assert_eq!(bits.len(), 256);
        let ones = bits.iter().filter(|&&b| b == 1).count();
        assert!(ones > 100 && ones < 156, "ones={ones}");
    }

    #[test]
    fn rgba_pipeline_shape() {
        let rgba = vec![128u8; 32 * 32 * 4];
        let bits = rgba_to_bits(
            32,
            32,
            &rgba,
            PreprocessOptions {
                size: 16,
                mode: BinarizeMode::Threshold,
                threshold: 128,
            },
        );
        assert_eq!(bits.len(), 256);
        assert!(bits.iter().all(|&b| b == 1));
    }

    #[test]
    fn rgb1_channels_are_planar() {
        // 2x2 pure red image: R plane all 1, G/B planes all 0.
        let mut rgba = Vec::new();
        for _ in 0..4 {
            rgba.extend_from_slice(&[255, 0, 0, 255]);
        }
        let bits = rgba_to_bits_rgb1(2, 2, &rgba, 2, 128);
        assert_eq!(bits.len(), 12);
        assert!(bits[0..4].iter().all(|&b| b == 1));
        assert!(bits[4..12].iter().all(|&b| b == 0));
    }

    #[test]
    fn palette_preprocess_encodes_indices() {
        let palette = vec![[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]];
        let rgba = [255u8, 0, 0, 255].repeat(4);
        let bits = rgba_to_bits_palette(2, 2, &rgba, 2, &palette);
        // four pixels, index 1 -> bits [1, 0] each, pixel-major
        assert_eq!(bits, vec![1, 0, 1, 0, 1, 0, 1, 0]);
    }

    #[test]
    fn quantize2_levels_and_boundaries() {
        assert_eq!(quantize2(0), 0);
        assert_eq!(quantize2(42), 0);
        assert_eq!(quantize2(43), 1);
        assert_eq!(quantize2(85), 1);
        assert_eq!(quantize2(127), 1);
        assert_eq!(quantize2(128), 2);
        assert_eq!(quantize2(170), 2);
        assert_eq!(quantize2(212), 2);
        assert_eq!(quantize2(213), 3);
        assert_eq!(quantize2(255), 3);
        for level in 0..4u8 {
            assert_eq!(quantize2(level2_to_u8(level)), level);
        }
    }

    #[test]
    fn rgb2_planes_are_channel_major() {
        // 2x2 with R=255, G=128, B=0: R -> level 3 (hi+lo), G -> level 2
        // (hi only), B -> level 0 (neither).
        let rgba = [255u8, 128, 0, 255].repeat(4);
        let bits = rgba_to_bits_rgb2(2, 2, &rgba, 2);
        assert_eq!(bits.len(), 4 * RGB2_PLANES);
        assert!(bits[0..4].iter().all(|&b| b == 1), "R_hi");
        assert!(bits[4..8].iter().all(|&b| b == 1), "R_lo");
        assert!(bits[8..12].iter().all(|&b| b == 1), "G_hi");
        assert!(bits[12..16].iter().all(|&b| b == 0), "G_lo");
        assert!(bits[16..24].iter().all(|&b| b == 0), "B_hi/B_lo");
    }
}
