#!/usr/bin/env bash
# Build the SINGLE-THREADED modgrad-wasm engine and drop the artifacts in
# modgrad.com/public/engine as `modgrad_wasm.*`. This is the default engine the
# /play worker loads when the page is NOT cross-origin isolated (no
# SharedArrayBuffer). Its multithreaded sibling is built by
# scripts/build-engine-parallel.sh → `modgrad_wasm_mt.*`; the two are additive.
#
# No nightly / build-std / atomics here — plain stable wasm32. We only override
# RUSTFLAGS to strip the workspace's `target-cpu=native` (invalid for wasm32).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SDK_DIR="${MODGRAD_SDK_DIR:-/steam/rotko/modgrad}"
OUT_DIR="${OUT_DIR:-$SITE_DIR/public/engine}"
OUT_NAME="modgrad_wasm"

echo "Building SINGLE-THREADED modgrad-wasm engine"
echo "  SDK workspace : $SDK_DIR"
echo "  out dir       : $OUT_DIR"

# Empty RUSTFLAGS overrides .cargo/config.toml's target-cpu=native for wasm32.
export RUSTFLAGS=''
(
  cd "$SDK_DIR"
  cargo build -p modgrad-wasm --target wasm32-unknown-unknown --release
)

WASM_IN="$SDK_DIR/target/wasm32-unknown-unknown/release/modgrad_wasm.wasm"
[ -f "$WASM_IN" ] || { echo "error: expected wasm at $WASM_IN" >&2; exit 1; }

mkdir -p "$OUT_DIR"
wasm-bindgen --target web --out-dir "$OUT_DIR" --out-name "$OUT_NAME" "$WASM_IN"

if command -v wasm-opt >/dev/null 2>&1; then
  echo "optimising ${OUT_NAME}_bg.wasm with wasm-opt -O3…"
  wasm-opt -O3 "$OUT_DIR/${OUT_NAME}_bg.wasm" -o "$OUT_DIR/${OUT_NAME}_bg.wasm"
fi

echo "Built:"
ls -la "$OUT_DIR/${OUT_NAME}"*
