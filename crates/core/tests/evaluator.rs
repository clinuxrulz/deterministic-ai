use xorae_core::bits::{BitBatch, BitMatrix};
use xorae_core::{Circuit, GateSet};

/// The bit-sliced evaluator must agree with evaluating one sample at a time.
#[test]
fn bit_parallel_matches_scalar() {
    let circuit = Circuit::random(16, 64, 8, GateSet::Full, 5);
    circuit.validate().unwrap();

    let samples = 70usize;
    let mut matrix = BitMatrix::new(samples, 16);
    let mut state = 0xdead_beef_cafe_babeu64;
    for row in 0..samples {
        for col in 0..16 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            matrix.set(row, col, state >> 63 == 1);
        }
    }

    let indices: Vec<usize> = (0..samples).collect();
    let batch = BitBatch::from_matrix(&matrix, &indices);
    assert_eq!(batch.words, 2, "70 samples should span two u64 words");

    let mut arena = Vec::new();
    let mut parallel = Vec::new();
    circuit.eval_into(&batch.planes, batch.words, &mut arena, &mut parallel);

    for sample in 0..samples {
        let input: Vec<u64> = (0..16)
            .map(|col| u64::from(matrix.get(sample, col)))
            .collect();
        let mut scalar_arena = Vec::new();
        let mut scalar = Vec::new();
        circuit.eval_into(&input, 1, &mut scalar_arena, &mut scalar);

        let word = sample / 64;
        let bit = sample % 64;
        for output in 0..8 {
            let got = (parallel[output * batch.words + word] >> bit) & 1;
            assert_eq!(got, scalar[output] & 1, "sample {sample}, output {output}");
        }
    }
}
