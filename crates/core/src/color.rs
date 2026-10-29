use serde::{Deserialize, Serialize};

/// How an image is represented in the bitstream.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ColorMode {
    /// One bit per pixel, as before.
    #[default]
    Grayscale,
    /// A power-of-two palette; each pixel is an index encoded in
    /// `bits_per_pixel` bits.
    Palette,
    /// Three binary channels (R, G, B), one bit per channel per pixel.
    Rgb1,
    /// Six binary planes: two bits per channel (`R_hi, R_lo, G_hi, G_lo,
    /// B_hi, B_lo`). Four levels per channel, so 64 colours per pixel.
    Rgb2,
}

pub type Rgb = [u8; 3];

/// Nearest palette entry by squared RGB distance.
pub fn nearest_palette_index(r: u8, g: u8, b: u8, palette: &[Rgb]) -> u8 {
    let mut best = 0usize;
    let mut best_distance = u32::MAX;
    for (i, &[pr, pg, pb]) in palette.iter().enumerate() {
        let dr = r as i32 - pr as i32;
        let dg = g as i32 - pg as i32;
        let db = b as i32 - pb as i32;
        let distance = (dr * dr + dg * dg + db * db) as u32;
        if distance < best_distance {
            best_distance = distance;
            best = i;
        }
    }
    best as u8
}

/// Encode palette indices as a pixel-major bit vector.
///
/// Pixel `p` occupies bits `p * bpp .. (p + 1) * bpp`, least significant bit
/// first.
pub fn indices_to_bits(indices: &[u8], bpp: usize) -> Vec<u8> {
    let mut bits = Vec::with_capacity(indices.len() * bpp);
    for &index in indices {
        for bit in 0..bpp {
            bits.push((index >> bit) & 1);
        }
    }
    bits
}

/// Decode a pixel-major bit vector back to palette indices.
pub fn bits_to_indices(bits: &[u8], bpp: usize) -> Vec<u8> {
    bits.chunks_exact(bpp)
        .map(|chunk| {
            let mut index = 0u8;
            for (bit, &value) in chunk.iter().enumerate() {
                index |= (value & 1) << bit;
            }
            index
        })
        .collect()
}

/// Map palette indices to interleaved RGB bytes.
pub fn palette_rgb(indices: &[u8], palette: &[Rgb]) -> Vec<u8> {
    let mut out = Vec::with_capacity(indices.len() * 3);
    for &index in indices {
        let color = palette.get(index as usize).copied().unwrap_or([0, 0, 0]);
        out.extend_from_slice(&color);
    }
    out
}

/// Mean absolute RGB error, normalized to `0..=1`.
pub fn color_mae(a: &[u8], b: &[u8]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut total = 0u64;
    for i in 0..n {
        total += (a[i] as i32 - b[i] as i32).unsigned_abs() as u64;
    }
    total as f32 / (n as f32 * 255.0)
}

/// Mean squared RGB error, normalized to `0..=1`.
pub fn color_mse(a: &[u8], b: &[u8]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut total = 0u64;
    for i in 0..n {
        let diff = a[i] as i32 - b[i] as i32;
        total += (diff * diff) as u64;
    }
    total as f32 / (n as f32 * 255.0 * 255.0)
}

/// PSNR from normalized MSE. Perfect reconstruction returns 99 dB.
pub fn psnr_from_mse(mse: f32) -> f32 {
    if mse <= 1e-12 {
        99.0
    } else {
        10.0 * (1.0 / mse).log10()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn palette() -> Vec<Rgb> {
        vec![[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]]
    }

    #[test]
    fn nearest_index_roundtrip() {
        let p = palette();
        assert_eq!(nearest_palette_index(250, 0, 0, &p), 1);
        assert_eq!(nearest_palette_index(0, 240, 0, &p), 2);
        assert_eq!(nearest_palette_index(0, 0, 250, &p), 3);
        assert_eq!(nearest_palette_index(10, 10, 10, &p), 0);
    }

    #[test]
    fn index_bits_roundtrip() {
        let indices = vec![0, 1, 2, 15, 7, 8];
        let bits = indices_to_bits(&indices, 4);
        assert_eq!(bits.len(), 24);
        assert_eq!(bits_to_indices(&bits, 4), indices);
    }

    #[test]
    fn palette_rgb_maps_indices() {
        let p = palette();
        let rgb = palette_rgb(&[0, 2], &p);
        assert_eq!(rgb, vec![0, 0, 0, 0, 255, 0]);
    }

    #[test]
    fn color_metrics() {
        let a = [0, 0, 0, 255, 255, 255];
        let b = [0, 0, 0, 255, 255, 255];
        assert_eq!(color_mae(&a, &b), 0.0);
        assert_eq!(color_mse(&a, &b), 0.0);
        assert!(psnr_from_mse(0.0) > 90.0);
        let c = [255, 255, 255, 0, 0, 0];
        assert!((color_mae(&a, &c) - 1.0).abs() < 1e-6);
    }
}
