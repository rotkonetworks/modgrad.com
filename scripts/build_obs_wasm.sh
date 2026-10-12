#!/usr/bin/env bash
# Build modgrad-obs-analysis as a single-threaded wasm module for the
# dashboard (same recipe as modgrad.com's scripts/build-engine.sh: stable
# wasm32, empty RUSTFLAGS to drop target-cpu=native, wasm-bindgen --target
# web, optional wasm-opt). Output: $OUT_DIR/modgrad_obs_analysis{.js,.d.ts,_bg.wasm} (the
# .d.ts typings are what obs/web type-checks against).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$REPO/obs/pkg}"
export RUSTFLAGS=''
(cd "$REPO" && cargo rustc -q -p modgrad-obs-analysis --release --target wasm32-unknown-unknown --features wasm --crate-type cdylib -j "${JOBS:-3}")
WASM_IN="$REPO/target/wasm32-unknown-unknown/release/modgrad_obs_analysis.wasm"
mkdir -p "$OUT_DIR"
wasm-bindgen --target web --out-dir "$OUT_DIR" --out-name modgrad_obs_analysis "$WASM_IN"
if command -v wasm-opt >/dev/null 2>&1; then wasm-opt -O3 "$OUT_DIR/modgrad_obs_analysis_bg.wasm" -o "$OUT_DIR/modgrad_obs_analysis_bg.wasm"; fi
# Drag-and-drop ingest (CSV/JSONL/text/TensorBoard/W&B parsers), loaded only
# by obs/drop-worker.js. A separate module: modgrad-obs (whose importers it
# reuses) depends on modgrad-obs-analysis, so the two cannot be one crate.
(cd "$REPO" && cargo rustc -q -p modgrad-obs-ingest --release --target wasm32-unknown-unknown --features wasm --crate-type cdylib -j "${JOBS:-3}")
WASM_IN="$REPO/target/wasm32-unknown-unknown/release/modgrad_obs_ingest.wasm"
wasm-bindgen --target web --out-dir "$OUT_DIR" --out-name modgrad_obs_ingest "$WASM_IN"
if command -v wasm-opt >/dev/null 2>&1; then wasm-opt -O3 "$OUT_DIR/modgrad_obs_ingest_bg.wasm" -o "$OUT_DIR/modgrad_obs_ingest_bg.wasm"; fi
ls -la "$OUT_DIR"
