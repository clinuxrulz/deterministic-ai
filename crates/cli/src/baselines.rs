use xorae_core::bits::BitMatrix;

use crate::dataset::Dataset;

/// Per-bit majority image computed from the training split.
pub fn majority_image(dataset: &Dataset, train: &[usize]) -> Vec<u8> {
    let cols = dataset.cols();
    let mut counts = vec![0usize; cols];
    for &row in train {
        for (col, count) in counts.iter_mut().enumerate() {
            if dataset.bits.get(row, col) {
                *count += 1;
            }
        }
    }
    counts
        .iter()
        .map(|&count| u8::from(count * 2 > train.len()))
        .collect()
}

pub fn ber_against(dataset: &Dataset, eval: &[usize], recon: &[u8]) -> f32 {
    if eval.is_empty() || recon.is_empty() {
        return 1.0;
    }
    let cols = dataset.cols();
    let mut errors = 0usize;
    for &row in eval {
        for (col, &expected) in recon.iter().enumerate().take(cols) {
            if u8::from(dataset.bits.get(row, col)) != expected {
                errors += 1;
            }
        }
    }
    errors as f32 / (eval.len() * cols) as f32
}

pub fn zero_ber(dataset: &Dataset, eval: &[usize]) -> f32 {
    let zeros = vec![0u8; dataset.cols()];
    ber_against(dataset, eval, &zeros)
}

pub fn majority_ber(dataset: &Dataset, train: &[usize], eval: &[usize]) -> f32 {
    let image = majority_image(dataset, train);
    ber_against(dataset, eval, &image)
}

/// Key = first `k` bits, decoder pads the rest with zeros.
pub fn topk_ber(dataset: &Dataset, eval: &[usize], k: usize) -> f32 {
    let cols = dataset.cols();
    let k = k.min(cols);
    let mut errors = 0usize;
    for &row in eval {
        for col in k..cols {
            if dataset.bits.get(row, col) {
                errors += 1;
            }
        }
    }
    errors as f32 / (eval.len() * (cols - k).max(1)) as f32
}

/// A linear code built greedily from training images.
///
/// The selected rows span a `k`-dimensional subspace. Encoding an image means
/// finding its coordinates in that basis; decoding means XORing the basis rows
/// selected by those coordinates. This is exactly the best a strict-XOR
/// autoencoder can do once its subspace is fixed, so it is a strong baseline
/// for the linear track.
pub struct LinearBasis {
    pub rows: Vec<Vec<u8>>,
    pub pivots: Vec<usize>,
}

impl LinearBasis {
    pub fn build(dataset: &Dataset, train: &[usize], k: usize) -> Self {
        let cols = dataset.cols();
        let mut rows: Vec<Vec<u8>> = Vec::new();
        let mut pivots: Vec<usize> = Vec::new();

        for &row in train {
            if rows.len() >= k {
                break;
            }
            let mut candidate: Vec<u8> = (0..cols)
                .map(|col| u8::from(dataset.bits.get(row, col)))
                .collect();
            for (basis, &pivot) in rows.iter().zip(pivots.iter()) {
                if candidate[pivot] == 1 {
                    for (value, basis_value) in candidate.iter_mut().zip(basis.iter()) {
                        *value ^= *basis_value;
                    }
                }
            }
            if let Some(pivot) = candidate.iter().position(|&value| value == 1) {
                rows.push(candidate);
                pivots.push(pivot);
            }
        }

        Self { rows, pivots }
    }

    pub fn reconstruct(&self, source: &[u8]) -> Vec<u8> {
        let mut residual = source.to_vec();
        let mut recon = vec![0u8; source.len()];
        for (basis, &pivot) in self.rows.iter().zip(self.pivots.iter()) {
            if residual[pivot] == 1 {
                for (value, basis_value) in residual.iter_mut().zip(basis.iter()) {
                    *value ^= *basis_value;
                }
                for (value, basis_value) in recon.iter_mut().zip(basis.iter()) {
                    *value ^= *basis_value;
                }
            }
        }
        recon
    }

    pub fn ber(&self, dataset: &Dataset, eval: &[usize]) -> f32 {
        if self.rows.is_empty() || eval.is_empty() {
            return 1.0;
        }
        let cols = dataset.cols();
        let mut errors = 0usize;
        for &row in eval {
            let source: Vec<u8> = (0..cols)
                .map(|col| u8::from(dataset.bits.get(row, col)))
                .collect();
            let recon = self.reconstruct(&source);
            for (col, &expected) in recon.iter().enumerate().take(cols) {
                if u8::from(dataset.bits.get(row, col)) != expected {
                    errors += 1;
                }
            }
        }
        errors as f32 / (eval.len() * cols) as f32
    }
}

pub struct Baseline {
    pub name: &'static str,
    pub ber: f32,
}

pub fn run_all(dataset: &Dataset, seed: u64, key_bits: usize) -> Vec<Baseline> {
    let (train, _val, test) = dataset.split_indices(seed);
    let eval = if test.is_empty() {
        &train[..]
    } else {
        &test[..]
    };
    let basis = LinearBasis::build(dataset, &train, key_bits);
    vec![
        Baseline {
            name: "all-zero",
            ber: zero_ber(dataset, eval),
        },
        Baseline {
            name: "per-bit majority",
            ber: majority_ber(dataset, &train, eval),
        },
        Baseline {
            name: "top-k identity",
            ber: topk_ber(dataset, eval, key_bits),
        },
        Baseline {
            name: "greedy GF(2) basis",
            ber: basis.ber(dataset, eval),
        },
    ]
}

/// Convenience for tests and tools that only have a packed matrix.
pub fn matrix_bit_ber(matrix: &BitMatrix, target: &BitMatrix) -> f32 {
    assert_eq!(matrix.rows, target.rows);
    assert_eq!(matrix.cols, target.cols);
    let mut errors = 0usize;
    for row in 0..matrix.rows {
        for col in 0..matrix.cols {
            if matrix.get(row, col) != target.get(row, col) {
                errors += 1;
            }
        }
    }
    errors as f32 / (matrix.rows * matrix.cols).max(1) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{Dataset, SplitMode};

    #[test]
    fn greedy_basis_reconstructs_spanning_rows() {
        // Rows 0 and 1 are independent; row 2 is their XOR and should be
        // reconstructed exactly by a 2-dimensional basis.
        let rows = vec![
            vec![1, 0, 0, 1],
            vec![0, 1, 1, 0],
            vec![1, 1, 1, 1],
            vec![0, 0, 0, 0],
        ];
        let gray: Vec<u8> = rows
            .iter()
            .flat_map(|row| row.iter().map(|&b| if b == 1 { 255 } else { 0 }))
            .collect();
        let dataset = Dataset::from_parts(2, rows, gray, vec![false; 4], SplitMode::PreSplit);
        let basis = LinearBasis::build(&dataset, &[0, 1, 2], 2);
        assert_eq!(basis.rows.len(), 2);
        let gain: Vec<u8> = vec![1, 1, 1, 1];
        assert_eq!(basis.reconstruct(&gain), gain);
        let report = basis.ber(&dataset, &[0, 1, 2]);
        assert_eq!(report, 0.0);
    }

    #[test]
    fn zero_baseline_is_bit_density() {
        let rows = vec![vec![1, 0, 1, 0], vec![1, 1, 0, 0]];
        let gray = vec![128u8; 8];
        let dataset = Dataset::from_parts(2, rows, gray, vec![false; 2], SplitMode::PreSplit);
        assert_eq!(zero_ber(&dataset, &[0, 1]), 0.5);
    }
}
