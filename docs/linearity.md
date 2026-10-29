# Why strict XOR is a linear code

This note records the mathematical reason the project runs two gate sets.

## The theorem

Let a circuit have primary inputs `x_1, …, x_n`, constant wires `0` and `1`, and
two-input XOR gates. Every wire computes a function of the form

```
c ⊕ x_{i1} ⊕ x_{i2} ⊕ … ⊕ x_{ik}     (c ∈ {0,1})
```

Proof by induction over the topological order:

- A primary input `x_i` is the linear form with coefficient vector `e_i` and
  constant `0`.
- The constant wires are the forms `0` and `1`.
- If `a` and `b` are affine forms, then `a ⊕ b` is the affine form whose
  coefficient vector is the symmetric difference of the two vectors and whose
  constant is the XOR of the two constants.
- Fan-out copies a wire; it does not change the form.

Therefore every output is an affine function

```
f(x) = A x ⊕ b        over GF(2).
```

The encoder is a matrix `H ∈ GF(2)^{K×N}` plus an offset, the decoder is
`G ∈ GF(2)^{N×K}` plus an offset, and the composition is affine with rank
≤ K. The set of reconstructable images is an affine subspace of dimension at
most K — a binary linear code.

## What this does and does not mean

- XOR is **not** universal. `{XOR, NOT, 1}` is still affine. `NOT x = x ⊕ 1` is
  an affine offset, not a nonlinearity.
- `{AND, NOT}` is universal; so are NAND and NOR alone.
- `{XOR, AND, 1}` is universal: every Boolean function can be written in
  algebraic normal form (XOR of AND terms).
- XOR is nonlinear as a function of *real-valued* inputs (the classic
  non-linearly-separable XOR problem), but that is a different notion from
  linearity over GF(2). A circuit of XOR gates is linear over GF(2).

## Counting argument

The number of affine Boolean functions on `n` variables is `2^(n+1)`. The number
of all Boolean functions is `2^(2^n)`. For `n = 2`, that is 8 versus 16; `AND` and
`OR` are not affine. No amount of wiring, fan-out, or depth can produce them
from XOR and constants, because affine maps are closed under composition.

## Consequence for the experiment

- A strict-XOR encoder/decoder is a linear compressor. Its best possible
  reconstruction error on a dataset is the best rank-K affine-subspace
  approximation under Hamming loss.
- A small budget of AND gates escapes the affine class. The default
  `xor_and` gate set uses ~90% XOR and ~10% AND, keeping the XOR character while
  making the circuit functionally complete.
- The repository includes `tests/linearity.rs`, which derives the affine matrix
  of a random XOR-only circuit by evaluating the zero and basis vectors, then
  checks it against random inputs. The test doubles as an executable proof for
  the evaluator.

## A note on NOT

Tying both inputs of a 2-input XOR to the same signal gives `x ⊕ x = 0`, a
constant zero. `NOT x` is `x ⊕ 1`, which requires a constant-one wire. Even with
that wire, the circuit remains affine.
