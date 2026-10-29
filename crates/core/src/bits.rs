use serde::{Deserialize, Serialize};

/// A row-major matrix of bits, packed into `u64` words.
///
/// Rows are images/samples, columns are bit positions within a sample.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BitMatrix {
    pub rows: usize,
    pub cols: usize,
    pub words_per_row: usize,
    pub data: Vec<u64>,
}

impl BitMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        let words_per_row = cols.div_ceil(64);
        Self {
            rows,
            cols,
            words_per_row,
            data: vec![0; rows * words_per_row],
        }
    }

    #[inline]
    pub fn get(&self, row: usize, col: usize) -> bool {
        debug_assert!(row < self.rows && col < self.cols);
        let word = self.data[row * self.words_per_row + col / 64];
        (word >> (col % 64)) & 1 == 1
    }

    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: bool) {
        debug_assert!(row < self.rows && col < self.cols);
        let word = &mut self.data[row * self.words_per_row + col / 64];
        let mask = 1u64 << (col % 64);
        if value {
            *word |= mask;
        } else {
            *word &= !mask;
        }
    }

    #[inline]
    pub fn row_words(&self, row: usize) -> &[u64] {
        let start = row * self.words_per_row;
        &self.data[start..start + self.words_per_row]
    }

    pub fn from_bit_rows(rows: &[Vec<u8>]) -> Self {
        let n = rows.len();
        let cols = rows.first().map_or(0, Vec::len);
        let mut matrix = Self::new(n, cols);
        for (r, row) in rows.iter().enumerate() {
            assert_eq!(row.len(), cols, "all rows must have the same length");
            for (c, &bit) in row.iter().enumerate() {
                if bit & 1 != 0 {
                    matrix.set(r, c, true);
                }
            }
        }
        matrix
    }

    pub fn to_bit_rows(&self) -> Vec<Vec<u8>> {
        (0..self.rows)
            .map(|r| (0..self.cols).map(|c| u8::from(self.get(r, c))).collect())
            .collect()
    }

    /// Number of set bits in a row, masking any padding bits in the last word.
    pub fn row_popcount(&self, row: usize) -> usize {
        let words = self.row_words(row);
        let mut total = 0usize;
        for (i, &word) in words.iter().enumerate() {
            let valid = self.cols.saturating_sub(i * 64).min(64);
            let mask = if valid == 64 {
                u64::MAX
            } else {
                (1u64 << valid) - 1
            };
            total += (word & mask).count_ones() as usize;
        }
        total
    }
}

/// A batch of samples laid out bit-plane style.
///
/// `planes[col * words + word]` holds bit `col` for 64 samples at a time.
/// This is the representation that makes gate evaluation bit-parallel.
#[derive(Clone, Debug)]
pub struct BitBatch {
    pub cols: usize,
    pub words: usize,
    /// Total number of samples in the batch (needed to mask the last word).
    pub len: usize,
    pub planes: Vec<u64>,
}

impl BitBatch {
    pub fn from_matrix(matrix: &BitMatrix, indices: &[usize]) -> Self {
        let words = indices.len().div_ceil(64).max(1);
        let mut planes = vec![0u64; matrix.cols * words];
        for (slot, &row) in indices.iter().enumerate() {
            assert!(row < matrix.rows, "batch index out of range");
            let word = slot / 64;
            let bit = slot % 64;
            let row_words = matrix.row_words(row);
            for col in 0..matrix.cols {
                if (row_words[col / 64] >> (col % 64)) & 1 == 1 {
                    planes[col * words + word] |= 1u64 << bit;
                }
            }
        }
        Self {
            cols: matrix.cols,
            words,
            len: indices.len(),
            planes,
        }
    }

    pub fn plane(&self, col: usize) -> &[u64] {
        let start = col * self.words;
        &self.planes[start..start + self.words]
    }

    /// Count set bits in one plane, ignoring padding in the final word.
    pub fn count_ones(&self, col: usize) -> usize {
        let mut total = 0usize;
        for (i, &word) in self.plane(col).iter().enumerate() {
            let valid = self.len.saturating_sub(i * 64).min(64);
            let mask = if valid == 64 {
                u64::MAX
            } else if valid == 0 {
                0
            } else {
                (1u64 << valid) - 1
            };
            total += (word & mask).count_ones() as usize;
        }
        total
    }
}

/// Pack a single sample's bits into planes: `out[i]` is just bit 0 set or clear.
pub fn bits_to_planes(bits: &[u8]) -> Vec<u64> {
    bits.iter().map(|&b| u64::from(b & 1)).collect()
}

/// Extract bit 0 of each plane into a `0/1` vector.
pub fn planes_to_bits(planes: &[u64]) -> Vec<u8> {
    planes.iter().map(|&w| (w & 1) as u8).collect()
}

/// Number of differing bits between two packed plane arrays.
pub fn plane_hamming(a: &[u64], b: &[u64], cols: usize, words: usize, len: usize) -> usize {
    debug_assert_eq!(a.len(), cols * words);
    debug_assert_eq!(b.len(), cols * words);
    let mut total = 0usize;
    for col in 0..cols {
        for w in 0..words {
            let mut diff = a[col * words + w] ^ b[col * words + w];
            if w == words - 1 {
                let valid = len.saturating_sub(w * 64).min(64);
                if valid < 64 {
                    let mask = if valid == 0 { 0 } else { (1u64 << valid) - 1 };
                    diff &= mask;
                }
            }
            total += diff.count_ones() as usize;
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_roundtrip() {
        let mut m = BitMatrix::new(3, 70);
        m.set(0, 0, true);
        m.set(0, 69, true);
        m.set(2, 65, true);
        assert!(m.get(0, 0));
        assert!(m.get(0, 69));
        assert!(m.get(2, 65));
        assert!(!m.get(1, 0));
        assert_eq!(m.row_popcount(0), 2);
        assert_eq!(m.row_popcount(2), 1);
        let rows = m.to_bit_rows();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0][0], 1);
    }

    #[test]
    fn batch_matches_matrix() {
        let mut m = BitMatrix::new(5, 10);
        for r in 0..5 {
            for c in 0..10 {
                m.set(r, c, (r + c) % 3 == 0);
            }
        }
        let indices = vec![0, 1, 2, 3, 4];
        let batch = BitBatch::from_matrix(&m, &indices);
        for (slot, &r) in indices.iter().enumerate() {
            for c in 0..10 {
                let got = (batch.planes[c * batch.words] >> slot) & 1 == 1;
                assert_eq!(got, m.get(r, c));
            }
        }
        assert_eq!(batch.count_ones(0), (0..5).filter(|&r| m.get(r, 0)).count());
    }

    #[test]
    fn hamming_ignores_padding() {
        // 2 cols, 1 word, 3 samples. Only bits 0..=2 are real samples.
        let a = vec![0, 0];
        let b = vec![0, u64::MAX << 3];
        assert_eq!(plane_hamming(&a, &b, 2, 1, 3), 0);

        let c = vec![0, 0b111];
        assert_eq!(plane_hamming(&a, &c, 2, 1, 3), 3);
    }
}
