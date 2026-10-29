#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/web/src/wasm/pkg"

mkdir -p "$OUT"
cd "$ROOT/crates/wasm"

wasm-pack build \
  --target web \
  --release \
  --out-dir "$OUT" \
  --out-name xorae_wasm

echo "wasm package written to $OUT"
