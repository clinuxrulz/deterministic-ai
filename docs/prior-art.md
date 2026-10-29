# Prior art: logic-gate autoencoders and image compression

Question: has anyone built an XOR/AND-gate autoencoder for images, trained by a
genetic algorithm or any other method, and had better success?

## Short answer

No exact match was found for this specific setup — a wired XOR/AND
encoder/decoder with a short binary key, evolved by a genetic algorithm, on cat
images. The idea sits between several established areas, and the adjacent work
is useful:

1. **Boolean autoencoders** are a known theoretical model.
2. **Differentiable logic gate networks** train logic gates with gradients and
   have strong image-classification results.
3. **Evolvable hardware** has been used for image compression, mostly lossless
   and on bi-level/space images.
4. **Cartesian genetic programming** has been used extensively for image
   filters and processing pipelines, but not, as far as found, for a global
   autoencoder.

## 1. Boolean autoencoders

- P. Baldi, “Boolean autoencoders and hypercube clustering complexity,”
  *Designs, Codes and Cryptography*, 2012.
- “On the Compressive Power of Boolean Threshold Autoencoders,” arXiv:2004.09735,
  2020.

Key result: a Boolean autoencoder with `d` hidden bits can perfectly memorize
`n` patterns when `d = ceil(log2 n)` (index code), and threshold autoencoders
have strong compression/memorization guarantees.

Implication for us: a 128-bit key is far more than enough to index 6,000 cat
images (about 13 bits would suffice). The bottleneck is not key capacity in
principle; it is the circuit's ability to implement the mapping, the gate
budget, and the optimizer.

## 2. Differentiable logic gate networks

- F. Petersen, C. Borgelt, H. Kuehne, O. Deussen, “Deep Differentiable Logic
  Gate Networks,” NeurIPS 2022. <https://arxiv.org/abs/2210.08277>
- F. Petersen et al., “Convolutional Differentiable Logic Gate Networks,”
  NeurIPS 2024. <https://arxiv.org/abs/2411.04732>
- Code: <https://github.com/Felix-Petersen/difflogic>

These networks are built from actual logic gates (including XOR, AND, OR,
NAND), trained by relaxing the discrete gates into differentiable soft gates,
then discretizing them for inference. They report strong image classification
(MNIST >97.5%, competitive CIFAR-10) and extremely fast gate-level inference.

No differentiable logic gate *autoencoder* or image-compression paper was found.
But this is the closest and most successful “logic gates at the core” line of
work, and it suggests that **gradient-based training of the gate network is a
much stronger optimizer than our GA**. A DLGN-style encoder/decoder is a natural
next experiment if we are willing to use any training algorithm.

## 3. Evolvable hardware for image compression

- M. Salami, “Data compression based on evolvable hardware,” 1996.
- H. Sakanashi, M. Iwata, T. Higuchi, “A Lossless Compression Method for
  Halftone Images Using Evolvable Hardware,” ICES 2001.
- A. Stoica, A. Fukunaga, et al., “Evolvable Hardware for Space Applications,”
  1998.
- “Evolvable hardware for lossless compression of very high resolution bi-level
  images,” 2004.
- “Towards Intrinsic Evolvable Hardware for Predictive Lossless …”

These systems evolve predictors/compression functions for bi-level, halftone,
or space images. Some report compression ratios superior to standard lossless
codecs. They are typically **predictive/lossless** systems, not global
autoencoders with a learned key.

Implication: evolving circuits for image compression is an established idea,
but our lossy key-based autoencoder on natural cat images is a different task.

## 4. Cartesian genetic programming for image processing

- S. Harding, J. Leitner, J. Schmidhuber, “Cartesian Genetic Programming for
  Image Processing” (CGP-IP), 2013.
- J. Biau, “Improving Image Filters with Cartesian Genetic Programming,” 2021.
- A. Margraf et al., “Evolving processing pipelines for industrial imaging with
  Cartesian Genetic Programming,” 2023.

CGP is very effective at evolving image filters and processing pipelines. Those
are usually local, per-pixel/filter operations, not a global autoencoder with a
bottleneck.

## 5. Neural autoencoder compression (non-gate baselines)

- L. Theis, W. Shi, A. Cunningham, F. Huszár, “Lossy Image Compression with
  Compressive Autoencoders,” ICLR 2017.
- VQ-VAE and related discrete-latent autoencoders.

These are the strong baselines for image compression, but they are neural
networks with continuous weights, not logic-gate circuits. They are useful as
quality references, not as implementations of the gate-network idea.

## What this means for the experiment

- Our specific setup appears **novel/underexplored**, so the negative or modest
  results are not surprising.
- Boolean autoencoder theory says the **key is not the fundamental limit** for
  memorizing the dataset; 128 bits is ample for 6,000 images. The limiting
  factors are circuit capacity, wiring search, and the objective.
- The GA is the weakest component compared with the adjacent literature.
  Gradient-trained logic gate networks (DLGN) achieve strong image results; a
  DLGN-style autoencoder is the most promising algorithmic alternative.
- Evolvable-hardware prior art suggests that **predictive/lossless** circuit
  compression is a more natural fit for gate-level evolution than a global
  lossy autoencoder.

## Possible next tracks

1. **32×32 resolution with the current GA** — the current experiment; tests
   whether spatial detail/capacity is the limiting factor.
2. **DLGN-style gradient-trained logic gate autoencoder** — keep XOR/AND gates
   at inference, but train with soft gates and discretize. Likely the biggest
   quality jump.
3. **Predictive/lossless EHW compression** — evolve a local predictor/context
   model for bi-level images, closer to the successful EHW literature.
4. **CGP image filter/pipeline** — evolve local filters or a preprocessing
   stage, then a smaller autoencoder.
