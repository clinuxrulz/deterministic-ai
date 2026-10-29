#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-data}"
URL="https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
MD5="c32a1d4ab5d03f1284b67883e8d87530"
ARCHIVE="$OUT/cifar-10-binary.tar.gz"
EXTRACTED="$OUT/cifar-10-batches-bin"

mkdir -p "$OUT"

if [ -d "$EXTRACTED" ]; then
  echo "already present: $EXTRACTED"
  exit 0
fi

if [ -f "$ARCHIVE" ] && command -v md5sum >/dev/null 2>&1 \
  && echo "$MD5  $ARCHIVE" | md5sum -c - >/dev/null 2>&1; then
  echo "archive already complete: $ARCHIVE"
else
  echo "downloading or resuming $URL"
  curl -L --fail -C - -o "$ARCHIVE" "$URL"
fi

if command -v md5sum >/dev/null 2>&1; then
  echo "$MD5  $ARCHIVE" | md5sum -c -
else
  echo "warning: md5sum not found; skipping checksum verification"
fi

echo "extracting $ARCHIVE"
tar -xzf "$ARCHIVE" -C "$OUT"
echo "dataset ready at $EXTRACTED"
