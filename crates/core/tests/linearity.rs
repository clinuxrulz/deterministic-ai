use xorae_core::{Circuit, GateSet};

/// A feed-forward XOR-only circuit must compute an affine map over GF(2).
///
/// We derive the map by evaluating the zero input and each basis vector, then
/// check it against random inputs. This is the executable form of the central
/// theoretical claim behind the strict-XOR baseline.
#[test]
fn xor_only_circuits_are_affine_over_gf2() {
    for seed in 0..8u64 {
        let circuit = Circuit::random(8, 48, 4, GateSet::XorOnly, seed);
        circuit.validate().unwrap();
        assert!(circuit.is_xor_only());

        let mut arena = Vec::new();
        let mut zero = Vec::new();
        circuit.eval_into(&[0u64; 8], 1, &mut arena, &mut zero);

        let mut basis = Vec::with_capacity(8);
        for i in 0..8 {
            let mut input = vec![0u64; 8];
            input[i] = 1;
            let mut out = Vec::new();
            circuit.eval_into(&input, 1, &mut arena, &mut out);
            // Subtract the affine offset f(0) so the basis describes the
            // linear part of the map.
            let delta: Vec<u64> = out.iter().zip(zero.iter()).map(|(o, z)| o ^ z).collect();
            basis.push(delta);
        }

        let mut state = 0x1234_5678_9abc_def0u64 ^ seed;
        for _ in 0..128 {
            let mut input = vec![0u64; 8];
            for bit in input.iter_mut() {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                *bit = (state >> 63) & 1;
            }

            let mut actual = Vec::new();
            circuit.eval_into(&input, 1, &mut arena, &mut actual);

            let mut expected = zero.clone();
            for (i, &value) in input.iter().enumerate() {
                if value & 1 == 1 {
                    for (e, b) in expected.iter_mut().zip(basis[i].iter()) {
                        *e ^= *b;
                    }
                }
            }

            for (a, e) in actual.iter().zip(expected.iter()) {
                assert_eq!(a & 1, e & 1, "seed {seed} produced a non-affine output");
            }
        }
    }
}
