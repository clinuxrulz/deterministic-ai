#!/usr/bin/env bash
# Download MNIST and leave both the archives and the raw IDX files in data/mnist.
set -euo pipefail

OUT="${1:-data/mnist}"
BASE="https://ossci-datasets.s3.amazonaws.com/mnist"
mkdir -p "$OUT"

for f in train-images-idx3-ubyte train-labels-idx1-ubyte \
         t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  if [ ! -f "$OUT/$f" ]; then
    echo "downloading $f"
    curl -sSL --retry 3 -o "$OUT/$f.gz" "$BASE/$f.gz"
    gunzip -kf "$OUT/$f.gz"
  fi
done

echo "MNIST ready at $OUT"
echo "next: cargo run --release -p xorae -- preprocess --mnist-dir $OUT \\"
echo "        --out data/mnist32.bin --size 32 --limit 12000"
