"""Read the project's XAE4/XAE5 dataset caches with the standard library only.

Layout (little endian):
    magic "XAE4"/"XAE5", size u32, n u32, mode u8, bits_per_pixel u8,
    split u8, channels u8, palette_len u16, palette, gray, rgb, values,
    bit words (u64 per 64 columns), flags, [XAE5: label_count u8, labels].
"""

import struct

COLOR_MODES = {0: "grayscale", 1: "palette", 2: "rgb1", 3: "rgb2"}


class Dataset:
    def __init__(self, path):
        with open(path, "rb") as handle:
            magic = handle.read(4)
            if magic not in (b"XAE4", b"XAE5"):
                raise ValueError(f"{path}: unexpected magic {magic!r}")
            self.version = 5 if magic == b"XAE5" else 4
            self.size, self.n = struct.unpack("<II", handle.read(8))
            mode = handle.read(1)[0]
            self.color_mode = COLOR_MODES.get(mode, f"mode{mode}")
            self.bits_per_pixel = handle.read(1)[0]
            self.split_mode = handle.read(1)[0]
            self.channels = handle.read(1)[0]
            palette_len = struct.unpack("<H", handle.read(2))[0]
            self.palette = [tuple(handle.read(3)) for _ in range(palette_len)]
            pixels = self.size * self.size
            self.gray = handle.read(self.n * pixels)
            self.rgb = handle.read(self.n * pixels * 3)
            self.values = handle.read(self.n * pixels)
            cols = pixels * self.channels * self.bits_per_pixel
            words_per_row = (cols + 63) // 64
            words = struct.unpack(f"<{self.n * words_per_row}Q", handle.read(self.n * words_per_row * 8))
            flags = handle.read(self.n)
            self.is_test = [bool(b) for b in flags]
            self.cols = cols
            self.bits = [
                [(words[r * words_per_row + (c // 64)] >> (c % 64)) & 1 for c in range(cols)]
                for r in range(self.n)
            ]
            self.labels = []
            if self.version >= 5:
                self.label_count = handle.read(1)[0]
                self.labels = list(handle.read(self.n))
            else:
                self.label_count = 0

    def rows(self, split="train", seed=42):
        """Mirror Dataset::split_indices: test comes from the file flag, the
        rest is split 80/20 by a seeded hash of the row index."""
        test = [i for i in range(self.n) if self.is_test[i]]
        rest = [i for i in range(self.n) if not self.is_test[i]]
        # Rust uses a hash; a stable permutation is enough for a PyTorch demo.
        order = sorted(rest, key=lambda i: (i * 2654435761 + seed) & 0xFFFFFFFF)
        cut = int(len(order) * 0.8)
        return {"train": order[:cut], "val": order[cut:], "test": test}[split]


def main():
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/cats16.bin"
    dataset = Dataset(path)
    ones = sum(sum(row) for row in dataset.bits)
    total = dataset.n * dataset.cols
    print(f"{path}: {dataset.n} images, {dataset.size}x{dataset.size}, {dataset.channels} plane(s)")
    print(f"  {dataset.cols} bits/image, density {ones / total:.3f}, labels {dataset.label_count}")
    train = dataset.rows("train")
    val = dataset.rows("val")
    test = dataset.rows("test")
    print(f"  train/val/test {len(train)}/{len(val)}/{len(test)}")


if __name__ == "__main__":
    main()
