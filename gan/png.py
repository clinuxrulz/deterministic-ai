"""Minimal PNG writer (stdlib only) for sample grids."""

import struct
import zlib


def _chunk(tag, data):
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def write_png(path, width, height, rows):
    raw = b"".join(b"\x00" + bytes(row) for row in rows)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    with open(path, "wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n")
        handle.write(_chunk(b"IHDR", header))
        handle.write(_chunk(b"IDAT", zlib.compress(raw, 9)))
        handle.write(_chunk(b"IEND", b""))


def write_grid(path, bits, side, columns=8, scale=6):
    """`bits` is a list of flat 0/1 rows of length side*side."""
    import numpy as np  # only used on the training machine

    count = bits.shape[0]
    rows = (count + columns - 1) // columns
    gap = 4
    tile = side * scale
    width = columns * tile + (columns + 1) * gap
    height = rows * tile + (rows + 1) * gap
    canvas = [[22, 24, 29] * width for _ in range(height)]
    for index in range(count):
        image = np.asarray(bits[index]).reshape(side, side)
        x0 = gap + (index % columns) * (tile + gap)
        y0 = gap + (index // columns) * (tile + gap)
        for y in range(side):
            for x in range(side):
                value = 235 if image[y, x] > 0.5 else 25
                for dy in range(scale):
                    row = canvas[y0 + y * scale + dy]
                    for dx in range(scale):
                        px = (x0 + x * scale + dx) * 3
                        row[px : px + 3] = [value, value, value]
    write_png(path, width, height, canvas)
