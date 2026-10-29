use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use xorae_core::bits::BitMatrix;
use xorae_core::preprocess::{binarize, resize_box, PreprocessOptions};

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
    pub is_test: Vec<bool>,
    pub split_mode: SplitMode,
}

impl Dataset {
    pub fn cols(&self) -> usize {
        self.size * self.size
    }

    pub fn row_bits(&self, index: usize) -> Vec<u8> {
        (0..self.cols())
            .map(|c| u8::from(self.bits.get(index, c)))
            .collect()
    }

    pub fn row_gray(&self, index: usize) -> &[u8] {
        let n = self.size * self.size;
        &self.gray[index * n..(index + 1) * n]
    }

    pub fn from_parts(
        size: usize,
        rows: Vec<Vec<u8>>,
        gray: Vec<u8>,
        is_test: Vec<bool>,
        split_mode: SplitMode,
    ) -> Self {
        let n = rows.len();
        assert_eq!(gray.len(), n * size * size);
        assert_eq!(is_test.len(), n);
        Self {
            size,
            n,
            bits: BitMatrix::from_bit_rows(&rows),
            gray,
            is_test,
            split_mode,
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
        w.write_all(b"XAE2")?;
        w.write_all(&(self.size as u32).to_le_bytes())?;
        w.write_all(&(self.n as u32).to_le_bytes())?;
        w.write_all(&[match self.split_mode {
            SplitMode::Hash => 0,
            SplitMode::PreSplit => 1,
        }])?;
        w.write_all(&self.gray)?;
        for &word in &self.bits.data {
            w.write_all(&word.to_le_bytes())?;
        }
        let flags: Vec<u8> = self.is_test.iter().map(|&b| u8::from(b)).collect();
        w.write_all(&flags)?;
        w.flush()?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let file =
            File::open(path).with_context(|| format!("opening dataset {}", path.display()))?;
        let mut r = BufReader::new(file);
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != b"XAE2" {
            bail!("not an xorae dataset (bad magic)");
        }
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
        let cols = size * size;

        let mut gray = vec![0u8; n * cols];
        r.read_exact(&mut gray)?;

        let mut bits = BitMatrix::new(n, cols);
        let mut word_bytes = [0u8; 8];
        for word in bits.data.iter_mut() {
            r.read_exact(&mut word_bytes)?;
            *word = u64::from_le_bytes(word_bytes);
        }

        let mut flags = vec![0u8; n];
        r.read_exact(&mut flags)?;
        let is_test = flags.iter().map(|&b| b != 0).collect();

        Ok(Self {
            size,
            n,
            bits,
            gray,
            is_test,
            split_mode,
        })
    }
}

/// Load CIFAR-10 binary batches, keep one class, and apply the shared
/// grayscale/resize/binarize pipeline.
pub fn load_cifar10(dir: &Path, class: u8, opts: PreprocessOptions) -> Result<Dataset> {
    let mut files: Vec<PathBuf> = Vec::new();
    for i in 1..=5 {
        files.push(dir.join(format!("data_batch_{i}.bin")));
    }
    files.push(dir.join("test_batch.bin"));

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
            let pixels = &record[1..];
            let (red, rest) = pixels.split_at(32 * 32);
            let (green, blue) = rest.split_at(32 * 32);
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
