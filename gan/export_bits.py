"""Pack a preprocessed XAE cache into a small bit file for Colab.

The GAN notebook needs the binarized cats and nothing else. torchvision's
CIFAR-10 download is ~170 MB and slow from a phone; the bits for 6,000 16x16
cats are 192 KB. This writes them, so the notebook can upload a file instead
of downloading a dataset.

    python gan/export_bits.py data/cats16_threshold.bin cats16_bits.bin

Format (little endian, stdlib only to read):

    magic  b"XAB1"
    n      u32        images
    cols   u32        bits per image (size * size * channels * bits_per_pixel)
    rows   n * ceil(cols / 8) bytes, MSB-first, row major

Read it in the notebook with numpy:

    raw = np.frombuffer(f.read(n * row_bytes), dtype=np.uint8).reshape(n, row_bytes)
    bits = np.unpackbits(raw, axis=1)[:, :cols]
"""

import struct
import sys

from read_xae import Dataset


def pack(dataset):
    row_bytes = (dataset.cols + 7) // 8
    out = bytearray()
    for row in dataset.bits:
        for start in range(0, dataset.cols, 8):
            byte = 0
            for k in range(8):
                c = start + k
                if c < dataset.cols and row[c]:
                    byte |= 1 << (7 - k)
            out.append(byte)
    assert len(out) == dataset.n * row_bytes, (len(out), dataset.n * row_bytes)
    return bytes(out)


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "data/cats16_threshold.bin"
    dst = sys.argv[2] if len(sys.argv) > 2 else "cats16_bits.bin"
    dataset = Dataset(src)
    payload = pack(dataset)
    with open(dst, "wb") as handle:
        handle.write(b"XAB1")
        handle.write(struct.pack("<II", dataset.n, dataset.cols))
        handle.write(payload)
    ones = sum(sum(row) for row in dataset.bits)
    print(
        f"{src} -> {dst}: {dataset.n} images, {dataset.size}x{dataset.size}, "
        f"{dataset.cols} bits, density {ones / (dataset.n * dataset.cols):.3f}, "
        f"{len(payload) / 1024:.0f} KB"
    )


if __name__ == "__main__":
    main()
