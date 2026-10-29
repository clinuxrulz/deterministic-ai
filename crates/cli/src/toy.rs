use anyhow::{bail, Result};

use crate::dataset::{Dataset, SplitMode};

/// A tiny dataset whose images are all XOR combinations of `key_bits`
/// structured basis images.
///
/// Because the dataset is exactly a `key_bits`-dimensional affine (indeed
/// linear) subspace of GF(2), a strict-XOR encoder/decoder pair can represent
/// it perfectly. That makes it a useful smoke test for the GA and evaluator.
pub fn linear_subspace(size: usize, key_bits: usize, seed: u64) -> Result<Dataset> {
    if key_bits == 0 {
        bail!("toy key_bits must be > 0");
    }
    if key_bits > 12 {
        bail!("toy key_bits > 12 would create too many patterns");
    }
    let n = size * size;
    let mut rng = SplitMix64::new(seed ^ 0xA5A5_5A5A_1234_5678);

    let mut basis: Vec<Vec<u8>> = Vec::new();
    for candidate in structured_basis(size) {
        if basis.len() == key_bits {
            break;
        }
        if is_independent(&basis, &candidate) {
            basis.push(candidate);
        }
    }
    while basis.len() < key_bits {
        let candidate: Vec<u8> = (0..n)
            .map(|_| if rng.next_f32() < 0.25 { 1 } else { 0 })
            .collect();
        if is_independent(&basis, &candidate) {
            basis.push(candidate);
        }
    }

    let patterns = 1usize << key_bits;
    let mut rows = Vec::with_capacity(patterns);
    let mut gray = Vec::with_capacity(patterns * n);
    let mut is_test = Vec::with_capacity(patterns);

    for mask in 0..patterns {
        let mut row = vec![0u8; n];
        for (b, basis_row) in basis.iter().enumerate() {
            if (mask >> b) & 1 == 1 {
                for (dst, &src) in row.iter_mut().zip(basis_row.iter()) {
                    *dst ^= src;
                }
            }
        }
        gray.extend(row.iter().map(|&b| if b == 1 { 255 } else { 0 }));
        // Training gets the zero pattern plus every basis vector. That set
        // spans the whole subspace and pins down any affine offset. Held-out
        // patterns are XOR combinations of two or more basis vectors.
        is_test.push(mask.count_ones() >= 2);
        rows.push(row);
    }

    Ok(Dataset::from_parts(
        size,
        rows,
        gray,
        is_test,
        SplitMode::PreSplit,
    ))
}

fn structured_basis(size: usize) -> Vec<Vec<u8>> {
    let n = size * size;
    let mut out = Vec::new();
    let mut push = |f: &dyn Fn(usize, usize) -> bool| {
        out.push((0..n).map(|i| u8::from(f(i % size, i / size))).collect());
    };

    push(&|x, _| x < size / 2);
    push(&|_, y| y < size / 2);
    push(&|x, y| x == y);
    push(&|x, y| x + y == size - 1);
    push(&|x, y| x == 0 || y == 0 || x == size - 1 || y == size - 1);
    push(&|x, y| (x / 2 + y / 2) % 2 == 0);
    push(&|x, y| (x + y) % 2 == 0);
    push(&|x, _| x % 2 == 0);
    push(&|_, y| y % 2 == 0);
    push(&|x, y| {
        let cx = size / 2;
        let cy = size / 2;
        let dx = x as isize - cx as isize;
        let dy = y as isize - cy as isize;
        dx * dx + dy * dy <= ((size / 3).max(1) as isize).pow(2)
    });
    out
}

fn rank(rows: &[Vec<u8>]) -> usize {
    let mut basis: Vec<Vec<u8>> = Vec::new();
    for row in rows {
        let mut v = row.clone();
        for b in &basis {
            if let Some(pivot) = b.iter().position(|&x| x == 1) {
                if v[pivot] == 1 {
                    for (vi, bi) in v.iter_mut().zip(b.iter()) {
                        *vi ^= bi;
                    }
                }
            }
        }
        if v.contains(&1) {
            basis.push(v);
        }
    }
    basis.len()
}

fn is_independent(basis: &[Vec<u8>], candidate: &[u8]) -> bool {
    let mut rows: Vec<Vec<u8>> = basis.to_vec();
    rows.push(candidate.to_vec());
    rank(&rows) == rows.len()
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}
