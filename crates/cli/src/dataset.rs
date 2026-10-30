use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use xorae_core::bits::BitMatrix;
use xorae_core::color::{indices_to_bits, nearest_palette_index, ColorMode, Rgb};
use xorae_core::preprocess::{
    binarize, quantize2, resize_box, rgb_to_gray, BinarizeMode, PreprocessOptions,
};

pub const CIFAR_RECORD: usize = 1 + 32 * 32 * 3;
pub const CIFAR_CLASS_CAT: u8 = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitMode {
    /// Split non-test rows 80/20 using a seeded hash (real datasets).
    Hash,
    /// Trust `is_test` as the held-out set (used by synthetic toys so the
    /// training set is guaranteed to span the subspace).
    PreSplit,
}

#[derive(Clone, Debug)]
pub struct Dataset {
    pub size: usize,
    pub n: usize,
    pub bits: BitMatrix,
    /// Resized grayscale images, `n * size * size` bytes.
    pub gray: Vec<u8>,
    /// Resized interleaved RGB images, `n * size * size * 3` bytes.
    pub rgb: Vec<u8>,
    /// Target palette index per pixel, `n * size * size` bytes.
    pub values: Vec<u8>,
    pub palette: Vec<Rgb>,
    pub color_mode: ColorMode,
    pub bits_per_pixel: usize,
    pub channels: usize,
    pub is_test: Vec<bool>,
    pub split_mode: SplitMode,
    /// One label per image. Empty when the dataset carries no labels.
    pub labels: Vec<u8>,
    /// Number of distinct labels (0 when unlabelled).
    pub label_count: usize,
}

impl Dataset {
    /// Attach per-image labels (written as `XAE5`).
    pub fn with_labels(mut self, labels: Vec<u8>, label_count: usize) -> Self {
        assert_eq!(labels.len(), self.n);
        self.labels = labels;
        self.label_count = label_count;
        self
    }

    /// Whether this dataset carries per-image labels.
    pub fn has_labels(&self) -> bool {
        !self.labels.is_empty()
    }

    /// Number of pixels per image.
    pub fn pixels(&self) -> usize {
        self.size * self.size
    }

    /// Number of encoded bits per image.
    pub fn cols(&self) -> usize {
        self.bits.cols
    }

    pub fn row_bits(&self, index: usize) -> Vec<u8> {
        (0..self.cols())
            .map(|c| u8::from(self.bits.get(index, c)))
            .collect()
    }

    pub fn row_gray(&self, index: usize) -> &[u8] {
        let n = self.pixels();
        &self.gray[index * n..(index + 1) * n]
    }

    pub fn row_rgb(&self, index: usize) -> &[u8] {
        let n = self.pixels() * 3;
        &self.rgb[index * n..(index + 1) * n]
    }

    pub fn row_values(&self, index: usize) -> &[u8] {
        let n = self.pixels();
        &self.values[index * n..(index + 1) * n]
    }

    /// Build a legacy grayscale dataset from 0/1 rows.
    pub fn from_parts(
        size: usize,
        rows: Vec<Vec<u8>>,
        gray: Vec<u8>,
        is_test: Vec<bool>,
        split_mode: SplitMode,
    ) -> Self {
        let n = rows.len();
        let pixels = size * size;
        assert_eq!(gray.len(), n * pixels);
        assert_eq!(is_test.len(), n);

        let mut values = Vec::with_capacity(n * pixels);
        let mut rgb = Vec::with_capacity(n * pixels * 3);
        for row in &rows {
            for &bit in row {
                values.push(bit & 1);
            }
        }
        for &value in &gray {
            rgb.extend_from_slice(&[value, value, value]);
        }

        Self {
            size,
            n,
            bits: BitMatrix::from_bit_rows(&rows),
            gray,
            rgb,
            values,
            palette: Vec::new(),
            color_mode: ColorMode::Grayscale,
            bits_per_pixel: 1,
            channels: 1,
            is_test,
            split_mode,
            labels: Vec::new(),
            label_count: 0,
        }
    }

    /// Build a palette-encoded dataset. Grayscale is derived from the RGB.
    pub fn from_color_parts(
        size: usize,
        rows: Vec<Vec<u8>>,
        rgb: Vec<u8>,
        values: Vec<u8>,
        palette: Vec<Rgb>,
        is_test: Vec<bool>,
        split_mode: SplitMode,
    ) -> Self {
        let n = rows.len();
        let pixels = size * size;
        assert!(!palette.is_empty());
        assert!(palette.len().is_power_of_two());
        assert_eq!(rgb.len(), n * pixels * 3);
        assert_eq!(values.len(), n * pixels);
        assert_eq!(is_test.len(), n);
        let gray = rgb_to_gray(size, size, &rgb);
        let bits_per_pixel = palette.len().ilog2() as usize;

        Self {
            size,
            n,
            bits: BitMatrix::from_bit_rows(&rows),
            gray,
            rgb,
            values,
            palette,
            color_mode: ColorMode::Palette,
            bits_per_pixel,
            channels: 1,
            is_test,
            split_mode,
            labels: Vec::new(),
            label_count: 0,
        }
    }

    /// Build an RGB 1-bit/channel dataset. Rows are channel-major:
    /// `[R bits..., G bits..., B bits...]`.
    pub fn from_rgb1_parts(
        size: usize,
        rows: Vec<Vec<u8>>,
        gray: Vec<u8>,
        rgb: Vec<u8>,
        is_test: Vec<bool>,
        split_mode: SplitMode,
    ) -> Self {
        let n = rows.len();
        let pixels = size * size;
        assert_eq!(gray.len(), n * pixels);
        assert_eq!(rgb.len(), n * pixels * 3);
        assert_eq!(is_test.len(), n);
        assert!(rows.iter().all(|row| row.len() == pixels * 3));

        Self {
            size,
            n,
            bits: BitMatrix::from_bit_rows(&rows),
            gray,
            rgb,
            values: vec![0u8; n * pixels],
            palette: Vec::new(),
            color_mode: ColorMode::Rgb1,
            bits_per_pixel: 1,
            channels: 3,
            is_test,
            split_mode,
            labels: Vec::new(),
            label_count: 0,
        }
    }

    /// Build an RGB 2-bit/channel dataset. Rows are channel-major with two
    /// planes per channel: `[R_hi, R_lo, G_hi, G_lo, B_hi, B_lo]`.
    pub fn from_rgb2_parts(
        size: usize,
        rows: Vec<Vec<u8>>,
        gray: Vec<u8>,
        rgb: Vec<u8>,
        is_test: Vec<bool>,
        split_mode: SplitMode,
    ) -> Self {
        let n = rows.len();
        let pixels = size * size;
        assert_eq!(gray.len(), n * pixels);
        assert_eq!(rgb.len(), n * pixels * 3);
        assert_eq!(is_test.len(), n);
        assert!(rows.iter().all(|row| row.len() == pixels * 6));

        Self {
            size,
            n,
            bits: BitMatrix::from_bit_rows(&rows),
            gray,
            rgb,
            values: vec![0u8; n * pixels],
            palette: Vec::new(),
            color_mode: ColorMode::Rgb2,
            bits_per_pixel: 1,
            channels: 6,
            is_test,
            split_mode,
            labels: Vec::new(),
            label_count: 0,
        }
    }

    /// Deterministic train/validation/test split.
    ///
    /// `test` comes from the dataset's own test flag (CIFAR's test batch).
    /// Non-test rows are split 80/20 by a seeded hash, unless the dataset is
    /// marked `PreSplit`, in which case the flag is used verbatim.
    pub fn split_indices(&self, seed: u64) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        if self.split_mode == SplitMode::PreSplit {
            let mut train = Vec::new();
            let mut val = Vec::new();
            let mut test = Vec::new();
            for i in 0..self.n {
                if self.is_test[i] {
                    val.push(i);
                    test.push(i);
                } else {
                    train.push(i);
                }
            }
            return (train, val, test);
        }

        let mut train = Vec::new();
        let mut val = Vec::new();
        let mut test = Vec::new();
        for i in 0..self.n {
            if self.is_test[i] {
                test.push(i);
            } else {
                let h = (i as u64)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(seed)
                    .rotate_left(17);
                if h.is_multiple_of(5) {
                    val.push(i);
                } else {
                    train.push(i);
                }
            }
        }

        // Degenerate-dataset fallbacks keep the trainer usable on toys.
        if train.is_empty() && !val.is_empty() {
            train.push(val.pop().unwrap());
        }
        if val.is_empty() {
            if train.len() > 1 {
                val.push(train.pop().unwrap());
            } else if !test.is_empty() {
                val.push(test[0]);
            }
        }
        if test.is_empty() {
            if !val.is_empty() {
                test.push(val[0]);
            } else if !train.is_empty() {
                test.push(train[0]);
            }
        }
        (train, val, test)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let file =
            File::create(path).with_context(|| format!("creating dataset {}", path.display()))?;
        let mut w = BufWriter::new(file);
        w.write_all(if self.has_labels() { b"XAE5" } else { b"XAE4" })?;
        w.write_all(&(self.size as u32).to_le_bytes())?;
        w.write_all(&(self.n as u32).to_le_bytes())?;
        w.write_all(&[match self.color_mode {
            ColorMode::Grayscale => 0,
            ColorMode::Palette => 1,
            ColorMode::Rgb1 => 2,
            ColorMode::Rgb2 => 3,
        }])?;
        w.write_all(&[self.bits_per_pixel as u8])?;
        w.write_all(&[match self.split_mode {
            SplitMode::Hash => 0,
            SplitMode::PreSplit => 1,
        }])?;
        w.write_all(&[self.channels as u8])?;
        w.write_all(&(self.palette.len() as u16).to_le_bytes())?;
        for color in &self.palette {
            w.write_all(color)?;
        }
        w.write_all(&self.gray)?;
        w.write_all(&self.rgb)?;
        w.write_all(&self.values)?;
        for &word in &self.bits.data {
            w.write_all(&word.to_le_bytes())?;
        }
        let flags: Vec<u8> = self.is_test.iter().map(|&b| u8::from(b)).collect();
        w.write_all(&flags)?;
        if self.has_labels() {
            w.write_all(&[self.label_count.min(u8::MAX as usize) as u8])?;
            w.write_all(&self.labels)?;
        }
        w.flush()?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let file =
            File::open(path).with_context(|| format!("opening dataset {}", path.display()))?;
        let mut r = BufReader::new(file);
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;

        if &magic == b"XAE2" {
            return load_v2(&mut r, path);
        }
        let version = if &magic == b"XAE5" {
            5
        } else if &magic == b"XAE4" {
            4
        } else if &magic == b"XAE3" {
            3
        } else {
            bail!("not an xorae dataset (bad magic)");
        };

        let mut buf4 = [0u8; 4];
        r.read_exact(&mut buf4)?;
        let size = u32::from_le_bytes(buf4) as usize;
        r.read_exact(&mut buf4)?;
        let n = u32::from_le_bytes(buf4) as usize;

        let mut mode = [0u8; 1];
        r.read_exact(&mut mode)?;
        let color_mode = match mode[0] {
            0 => ColorMode::Grayscale,
            1 => ColorMode::Palette,
            2 => ColorMode::Rgb1,
            3 => ColorMode::Rgb2,
            other => bail!("unknown color mode {other}"),
        };
        let mut bpp = [0u8; 1];
        r.read_exact(&mut bpp)?;
        let bits_per_pixel = bpp[0] as usize;

        let mut split = [0u8; 1];
        r.read_exact(&mut split)?;
        let split_mode = match split[0] {
            0 => SplitMode::Hash,
            1 => SplitMode::PreSplit,
            other => bail!("unknown split mode {other}"),
        };
        let channels = if version >= 4 {
            let mut channels = [0u8; 1];
            r.read_exact(&mut channels)?;
            channels[0] as usize
        } else {
            1
        };

        let mut palette_len = [0u8; 2];
        r.read_exact(&mut palette_len)?;
        let palette_len = u16::from_le_bytes(palette_len) as usize;
        let mut palette = Vec::with_capacity(palette_len);
        for _ in 0..palette_len {
            let mut color = [0u8; 3];
            r.read_exact(&mut color)?;
            palette.push(color);
        }

        let pixels = size * size;
        let mut gray = vec![0u8; n * pixels];
        r.read_exact(&mut gray)?;
        let mut rgb = vec![0u8; n * pixels * 3];
        r.read_exact(&mut rgb)?;
        let mut values = vec![0u8; n * pixels];
        r.read_exact(&mut values)?;

        let cols = pixels * channels * bits_per_pixel;
        let mut bits = BitMatrix::new(n, cols);
        let mut word_bytes = [0u8; 8];
        for word in bits.data.iter_mut() {
            r.read_exact(&mut word_bytes)?;
            *word = u64::from_le_bytes(word_bytes);
        }

        let mut flags = vec![0u8; n];
        r.read_exact(&mut flags)?;
        let is_test = flags.iter().map(|&b| b != 0).collect();

        let (label_count, labels) = if version >= 5 {
            let mut count = [0u8; 1];
            r.read_exact(&mut count)?;
            let mut labels = vec![0u8; n];
            r.read_exact(&mut labels)?;
            (count[0] as usize, labels)
        } else {
            (0, Vec::new())
        };

        Ok(Self {
            size,
            n,
            bits,
            gray,
            rgb,
            values,
            palette,
            color_mode,
            bits_per_pixel,
            channels,
            is_test,
            split_mode,
            label_count,
            labels,
        })
    }
}

fn load_v2<R: Read>(r: &mut R, path: &Path) -> Result<Dataset> {
    let mut buf4 = [0u8; 4];
    r.read_exact(&mut buf4)?;
    let size = u32::from_le_bytes(buf4) as usize;
    r.read_exact(&mut buf4)?;
    let n = u32::from_le_bytes(buf4) as usize;
    let mut mode = [0u8; 1];
    r.read_exact(&mut mode)?;
    let split_mode = match mode[0] {
        0 => SplitMode::Hash,
        1 => SplitMode::PreSplit,
        other => bail!("unknown split mode {other}"),
    };
    let pixels = size * size;
    let mut gray = vec![0u8; n * pixels];
    r.read_exact(&mut gray)?;
    let mut bits = BitMatrix::new(n, pixels);
    let mut word_bytes = [0u8; 8];
    for word in bits.data.iter_mut() {
        r.read_exact(&mut word_bytes)?;
        *word = u64::from_le_bytes(word_bytes);
    }
    let mut flags = vec![0u8; n];
    r.read_exact(&mut flags)?;
    let is_test = flags.iter().map(|&b| b != 0).collect();

    let mut rgb = Vec::with_capacity(n * pixels * 3);
    for &value in &gray {
        rgb.extend_from_slice(&[value, value, value]);
    }
    let mut values = Vec::with_capacity(n * pixels);
    for row in 0..n {
        for col in 0..pixels {
            values.push(u8::from(bits.get(row, col)));
        }
    }

    eprintln!(
        "note: loaded legacy XAE2 dataset {} (grayscale); re-run `preprocess` for XAE3",
        path.display()
    );
    Ok(Dataset {
        size,
        n,
        bits,
        gray,
        rgb,
        values,
        palette: Vec::new(),
        color_mode: ColorMode::Grayscale,
        bits_per_pixel: 1,
        channels: 1,
        is_test,
        split_mode,
        labels: Vec::new(),
        label_count: 0,
    })
}

/// Load CIFAR-10 binary batches, keep one class, and apply the shared
/// grayscale/resize/binarize pipeline.
pub fn load_cifar10(dir: &Path, class: u8, opts: PreprocessOptions) -> Result<Dataset> {
    let files = cifar_files(dir);
    let mut rows: Vec<Vec<u8>> = Vec::new();
    let mut gray: Vec<u8> = Vec::new();
    let mut is_test: Vec<bool> = Vec::new();

    for path in files {
        let data = std::fs::read(&path)
            .with_context(|| format!("reading CIFAR batch {}", path.display()))?;
        if data.len() % CIFAR_RECORD != 0 {
            bail!(
                "{} has length {}, which is not a multiple of {}",
                path.display(),
                data.len(),
                CIFAR_RECORD
            );
        }
        let test_file = path.file_name().and_then(|s| s.to_str()) == Some("test_batch.bin");
        for record in data.chunks_exact(CIFAR_RECORD) {
            let label = record[0];
            if label != class {
                continue;
            }
            let (red, green, blue) = planar_channels(record);
            let gray32 = planar_rgb_to_gray(red, green, blue);
            let small = resize_box(&gray32, 32, 32, opts.size);
            let bits = binarize(&small, opts.mode, opts.threshold);
            rows.push(bits);
            gray.extend_from_slice(&small);
            is_test.push(test_file);
        }
    }

    if rows.is_empty() {
        bail!("no images with class {} found in {}", class, dir.display());
    }

    Ok(Dataset::from_parts(
        opts.size,
        rows,
        gray,
        is_test,
        SplitMode::Hash,
    ))
}

/// Load CIFAR-10 and build a global color palette of `palette_size` colors.
pub fn load_cifar10_palette(
    dir: &Path,
    class: u8,
    size: usize,
    palette_size: usize,
    seed: u64,
) -> Result<Dataset> {
    if !palette_size.is_power_of_two() || !(2..=256).contains(&palette_size) {
        bail!("palette size must be a power of two in 2..=256");
    }
    let files = cifar_files(dir);
    let mut rgb: Vec<u8> = Vec::new();
    let mut is_test: Vec<bool> = Vec::new();

    for path in files {
        let data = std::fs::read(&path)
            .with_context(|| format!("reading CIFAR batch {}", path.display()))?;
        if data.len() % CIFAR_RECORD != 0 {
            bail!(
                "{} has length {}, which is not a multiple of {}",
                path.display(),
                data.len(),
                CIFAR_RECORD
            );
        }
        let test_file = path.file_name().and_then(|s| s.to_str()) == Some("test_batch.bin");
        for record in data.chunks_exact(CIFAR_RECORD) {
            if record[0] != class {
                continue;
            }
            let (red, green, blue) = planar_channels(record);
            let small_rgb = resize_rgb_box(red, green, blue, 32, 32, size);
            rgb.extend_from_slice(&small_rgb);
            is_test.push(test_file);
        }
    }

    let n = is_test.len();
    if n == 0 {
        bail!("no images with class {} found in {}", class, dir.display());
    }
    let pixels = size * size;

    // Build the palette from a deterministic sample of training pixels.
    let mut samples: Vec<Rgb> = Vec::new();
    for row in 0..n {
        if is_test[row] {
            continue;
        }
        let image = &rgb[row * pixels * 3..(row + 1) * pixels * 3];
        for p in (0..pixels).step_by(4) {
            samples.push([image[p * 3], image[p * 3 + 1], image[p * 3 + 2]]);
        }
    }
    let palette = kmeans_palette(&samples, palette_size, seed);

    let bits_per_pixel = palette_size.ilog2() as usize;
    let mut values = Vec::with_capacity(n * pixels);
    let mut rows = Vec::with_capacity(n);
    for row in 0..n {
        let image = &rgb[row * pixels * 3..(row + 1) * pixels * 3];
        let mut indices = Vec::with_capacity(pixels);
        for p in 0..pixels {
            indices.push(nearest_palette_index(
                image[p * 3],
                image[p * 3 + 1],
                image[p * 3 + 2],
                &palette,
            ));
        }
        values.extend_from_slice(&indices);
        rows.push(indices_to_bits(&indices, bits_per_pixel));
    }

    Ok(Dataset::from_color_parts(
        size,
        rows,
        rgb,
        values,
        palette,
        is_test,
        SplitMode::Hash,
    ))
}

/// Load CIFAR-10 and threshold each RGB channel independently.
pub fn load_cifar10_rgb1(dir: &Path, class: u8, size: usize, threshold: u8) -> Result<Dataset> {
    let files = cifar_files(dir);
    let pixels = size * size;
    let mut rows: Vec<Vec<u8>> = Vec::new();
    let mut gray: Vec<u8> = Vec::new();
    let mut rgb: Vec<u8> = Vec::new();
    let mut is_test: Vec<bool> = Vec::new();

    for path in files {
        let data = std::fs::read(&path)
            .with_context(|| format!("reading CIFAR batch {}", path.display()))?;
        if data.len() % CIFAR_RECORD != 0 {
            bail!(
                "{} has length {}, which is not a multiple of {}",
                path.display(),
                data.len(),
                CIFAR_RECORD
            );
        }
        let test_file = path.file_name().and_then(|s| s.to_str()) == Some("test_batch.bin");
        for record in data.chunks_exact(CIFAR_RECORD) {
            if record[0] != class {
                continue;
            }
            let (red, green, blue) = planar_channels(record);
            let small_rgb = resize_rgb_box(red, green, blue, 32, 32, size);
            let small_gray = rgb_to_gray(size, size, &small_rgb);
            let mut row = Vec::with_capacity(pixels * 3);
            for channel in 0..3 {
                for p in 0..pixels {
                    row.push(u8::from(small_rgb[p * 3 + channel] >= threshold));
                }
            }
            rows.push(row);
            gray.extend_from_slice(&small_gray);
            rgb.extend_from_slice(&small_rgb);
            is_test.push(test_file);
        }
    }

    if rows.is_empty() {
        bail!("no images with class {} found in {}", class, dir.display());
    }

    Ok(Dataset::from_rgb1_parts(
        size,
        rows,
        gray,
        rgb,
        is_test,
        SplitMode::Hash,
    ))
}

/// Load CIFAR-10 and quantise each channel to 2 bits (4 levels, 64 colours).
///
/// Rows are six channel-major planes: `[R_hi, R_lo, G_hi, G_lo, B_hi, B_lo]`.
pub fn load_cifar10_rgb2(dir: &Path, class: u8, size: usize) -> Result<Dataset> {
    let files = cifar_files(dir);
    let pixels = size * size;
    let mut rows: Vec<Vec<u8>> = Vec::new();
    let mut gray: Vec<u8> = Vec::new();
    let mut rgb: Vec<u8> = Vec::new();
    let mut is_test: Vec<bool> = Vec::new();

    for path in files {
        let data = std::fs::read(&path)
            .with_context(|| format!("reading CIFAR batch {}", path.display()))?;
        if data.len() % CIFAR_RECORD != 0 {
            bail!(
                "{} has length {}, which is not a multiple of {}",
                path.display(),
                data.len(),
                CIFAR_RECORD
            );
        }
        let test_file = path.file_name().and_then(|s| s.to_str()) == Some("test_batch.bin");
        for record in data.chunks_exact(CIFAR_RECORD) {
            if record[0] != class {
                continue;
            }
            let (red, green, blue) = planar_channels(record);
            let small_rgb = resize_rgb_box(red, green, blue, 32, 32, size);
            let small_gray = rgb_to_gray(size, size, &small_rgb);
            let mut row = Vec::with_capacity(pixels * 6);
            for channel in 0..3 {
                let mut hi = vec![0u8; pixels];
                let mut lo = vec![0u8; pixels];
                for p in 0..pixels {
                    let level = quantize2(small_rgb[p * 3 + channel]);
                    hi[p] = (level >> 1) & 1;
                    lo[p] = level & 1;
                }
                row.extend_from_slice(&hi);
                row.extend_from_slice(&lo);
            }
            rows.push(row);
            gray.extend_from_slice(&small_gray);
            rgb.extend_from_slice(&small_rgb);
            is_test.push(test_file);
        }
    }

    if rows.is_empty() {
        bail!("no images with class {} found in {}", class, dir.display());
    }

    Ok(Dataset::from_rgb2_parts(
        size,
        rows,
        gray,
        rgb,
        is_test,
        SplitMode::Hash,
    ))
}

/// Load MNIST from the raw IDX files in `dir` into a binarized dataset.
///
/// `size` resamples the 28x28 digits; `train_limit` keeps only the first N
/// training images (the test set is always kept whole).
pub fn load_mnist(
    dir: &Path,
    size: usize,
    threshold: u8,
    train_limit: Option<usize>,
) -> Result<Dataset> {
    let (train_images, rows, cols) = read_idx_images(&dir.join("train-images-idx3-ubyte"))?;
    let (test_images, _, _) = read_idx_images(&dir.join("t10k-images-idx3-ubyte"))?;
    let train_labels = read_idx_labels(&dir.join("train-labels-idx1-ubyte"))?;
    let test_labels = read_idx_labels(&dir.join("t10k-labels-idx1-ubyte"))?;
    let per_image = rows * cols;
    let n_train = train_images.len() / per_image;
    let n_test = test_images.len() / per_image;
    let take = train_limit.unwrap_or(n_train).min(n_train);

    let mut bit_rows = Vec::with_capacity(take + n_test);
    let mut gray = Vec::with_capacity((take + n_test) * size * size);
    let mut is_test = Vec::with_capacity(take + n_test);
    let mut labels = Vec::with_capacity(take + n_test);
    let mut image = vec![0u8; per_image];
    for (set, count, flag) in [
        (train_images.as_slice(), take, false),
        (test_images.as_slice(), n_test, true),
    ] {
        for index in 0..count {
            image.copy_from_slice(&set[index * per_image..(index + 1) * per_image]);
            let small = resize_box(&image, cols, rows, size);
            bit_rows.push(binarize(&small, BinarizeMode::Threshold, threshold));
            gray.extend_from_slice(&small);
            is_test.push(flag);
        }
    }
    labels.extend_from_slice(&train_labels[..take]);
    labels.extend_from_slice(&test_labels[..n_test.min(test_labels.len())]);

    Ok(Dataset::from_parts(size, bit_rows, gray, is_test, SplitMode::Hash).with_labels(labels, 10))
}

/// Read an IDX label file (one label byte per image).
fn read_idx_labels(path: &Path) -> Result<Vec<u8>> {
    let data = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    if data.len() < 8 {
        bail!("{} is too short to be an IDX label file", path.display());
    }
    let magic = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
    if magic != 2049 {
        bail!("{}: unexpected IDX label magic {magic}", path.display());
    }
    let n = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;
    if data.len() != 8 + n {
        bail!(
            "{}: length {} does not match {n} labels",
            path.display(),
            data.len()
        );
    }
    Ok(data[8..].to_vec())
}

/// Read an IDX image file: returns (pixels, rows, cols).
fn read_idx_images(path: &Path) -> Result<(Vec<u8>, usize, usize)> {
    let data = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    if data.len() < 16 {
        bail!("{} is too short to be an IDX file", path.display());
    }
    let be =
        |o: usize| u32::from_be_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]) as usize;
    if be(0) != 2051 {
        bail!("{}: unexpected IDX magic {}", path.display(), be(0));
    }
    let (n, rows, cols) = (be(4), be(8), be(12));
    let expected = 16 + n * rows * cols;
    if data.len() != expected {
        bail!(
            "{}: length {} does not match {n} images of {rows}x{cols}",
            path.display(),
            data.len()
        );
    }
    Ok((data[16..].to_vec(), rows, cols))
}

fn cifar_files(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = Vec::new();
    for i in 1..=5 {
        files.push(dir.join(format!("data_batch_{i}.bin")));
    }
    files.push(dir.join("test_batch.bin"));
    files
}

fn planar_channels(record: &[u8]) -> (&[u8], &[u8], &[u8]) {
    let pixels = &record[1..];
    let (red, rest) = pixels.split_at(32 * 32);
    let (green, blue) = rest.split_at(32 * 32);
    (red, green, blue)
}

/// Box-resize planar RGB to an interleaved `size × size` RGB buffer.
fn resize_rgb_box(
    red: &[u8],
    green: &[u8],
    blue: &[u8],
    width: usize,
    height: usize,
    size: usize,
) -> Vec<u8> {
    assert_eq!(red.len(), width * height);
    assert_eq!(green.len(), width * height);
    assert_eq!(blue.len(), width * height);
    if width == size && height == size {
        let mut out = Vec::with_capacity(size * size * 3);
        for i in 0..width * height {
            out.extend_from_slice(&[red[i], green[i], blue[i]]);
        }
        return out;
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
                    let i = y * width + x;
                    sums[0] += red[i] as u32;
                    sums[1] += green[i] as u32;
                    sums[2] += blue[i] as u32;
                    count += 1;
                }
            }
            let out_index = (oy * size + ox) * 3;
            out[out_index] = (sums[0] / count.max(1)) as u8;
            out[out_index + 1] = (sums[1] / count.max(1)) as u8;
            out[out_index + 2] = (sums[2] / count.max(1)) as u8;
        }
    }
    out
}

fn kmeans_palette(samples: &[Rgb], k: usize, seed: u64) -> Vec<Rgb> {
    if samples.is_empty() {
        return vec![[0, 0, 0]; k];
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut centroids: Vec<[f32; 3]> = Vec::with_capacity(k);
    let first = samples[rng.gen_range(0..samples.len())];
    centroids.push([first[0] as f32, first[1] as f32, first[2] as f32]);

    let mut min_distances = vec![f32::MAX; samples.len()];
    while centroids.len() < k {
        let mut total = 0.0f32;
        for (i, sample) in samples.iter().enumerate() {
            let distance = distance_to_centroids(sample, &centroids).1;
            if distance < min_distances[i] {
                min_distances[i] = distance;
            }
            total += min_distances[i];
        }
        if total <= 0.0 {
            break;
        }
        let mut target = rng.gen::<f32>() * total;
        let mut chosen = 0usize;
        for (i, &distance) in min_distances.iter().enumerate() {
            target -= distance;
            if target <= 0.0 {
                chosen = i;
                break;
            }
        }
        let sample = samples[chosen];
        centroids.push([sample[0] as f32, sample[1] as f32, sample[2] as f32]);
    }
    while centroids.len() < k {
        let sample = samples[rng.gen_range(0..samples.len())];
        centroids.push([sample[0] as f32, sample[1] as f32, sample[2] as f32]);
    }

    for _ in 0..15 {
        let mut sums = vec![[0.0f64; 3]; k];
        let mut counts = vec![0u32; k];
        for sample in samples {
            let (index, _) = distance_to_centroids(sample, &centroids);
            sums[index][0] += sample[0] as f64;
            sums[index][1] += sample[1] as f64;
            sums[index][2] += sample[2] as f64;
            counts[index] += 1;
        }
        for index in 0..k {
            if counts[index] > 0 {
                let count = counts[index] as f64;
                centroids[index] = [
                    (sums[index][0] / count).round() as f32,
                    (sums[index][1] / count).round() as f32,
                    (sums[index][2] / count).round() as f32,
                ];
            } else {
                let sample = samples[rng.gen_range(0..samples.len())];
                centroids[index] = [sample[0] as f32, sample[1] as f32, sample[2] as f32];
            }
        }
    }

    centroids
        .into_iter()
        .map(|c| {
            [
                c[0].round().clamp(0.0, 255.0) as u8,
                c[1].round().clamp(0.0, 255.0) as u8,
                c[2].round().clamp(0.0, 255.0) as u8,
            ]
        })
        .collect()
}

fn distance_to_centroids(sample: &Rgb, centroids: &[[f32; 3]]) -> (usize, f32) {
    let mut best = 0usize;
    let mut best_distance = f32::MAX;
    for (i, centroid) in centroids.iter().enumerate() {
        let dr = sample[0] as f32 - centroid[0];
        let dg = sample[1] as f32 - centroid[1];
        let db = sample[2] as f32 - centroid[2];
        let distance = dr * dr + dg * dg + db * db;
        if distance < best_distance {
            best_distance = distance;
            best = i;
        }
    }
    (best, best_distance)
}

/// CIFAR binary records store channels plane-by-plane: 1024 red bytes, then
/// 1024 green, then 1024 blue. This is *not* interleaved RGB.
fn planar_rgb_to_gray(red: &[u8], green: &[u8], blue: &[u8]) -> Vec<u8> {
    debug_assert_eq!(red.len(), green.len());
    debug_assert_eq!(red.len(), blue.len());
    red.iter()
        .zip(green.iter())
        .zip(blue.iter())
        .map(|((&r, &g), &b)| {
            ((299 * r as u32 + 587 * g as u32 + 114 * b as u32) / 1000).min(255) as u8
        })
        .collect()
}
