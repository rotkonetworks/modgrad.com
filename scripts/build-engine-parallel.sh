#!/usr/bin/env bash
# Build the MULTITHREADED modgrad-wasm engine (rayon + wasm-bindgen-rayon +
# SharedArrayBuffer) and drop the artifacts in modgrad.com/public/engine as
# `modgrad_wasm_mt.*`, ALONGSIDE the existing single-threaded `modgrad_wasm.*`.
#
# The page picks at runtime (src/play/worker.ts): when the document is
# cross-origin isolated (COOP: same-origin + COEP: require-corp, so
# SharedArrayBuffer is available) it loads the _mt engine and calls
# `init_thread_pool(navigator.hardwareConcurrency)`; otherwise it falls back to
# the single-threaded engine and runs exactly as before. So this build is
# purely additive — it never replaces the single-thread artifacts.
#
# Requirements (all already present on this box):
#   - nightly toolchain        : rustup toolchain install nightly
#   - rust-src on nightly       : rustup component add rust-src --toolchain nightly
#   - wasm32 target on nightly  : rustup target add wasm32-unknown-unknown --toolchain nightly
#   - wasm-bindgen-cli 0.2.117  : must match the workspace's wasm-bindgen pin
#   - wasm-opt (OPTIONAL)        : only used for size; skipped if absent
#
# Serve the result cross-origin-isolated to actually get threads, e.g.:
#   node scripts/serve-coop-coep.mjs 8788 dist
# (production nginx must send COOP/COEP on the /engine assets — handled
# separately; this script does NOT touch nginx.)
set -euo pipefail

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"          # modgrad.com
SDK_DIR="${MODGRAD_SDK_DIR:-/steam/rotko/modgrad}" # SDK workspace (override-able)
OUT_DIR="${OUT_DIR:-$SITE_DIR/public/engine}"
OUT_NAME="modgrad_wasm_mt"
NIGHTLY_TOOLCHAIN="${NIGHTLY_TOOLCHAIN:-nightly}"

echo "Building MULTITHREADED modgrad-wasm engine"
echo "=========================================="
echo "  SDK workspace : $SDK_DIR"
echo "  out dir       : $OUT_DIR"
echo "  toolchain     : $NIGHTLY_TOOLCHAIN (build-std + atomics)"
echo

# ── step 1: compile the cdylib with the `parallel` feature ───────────────────
#
# +atomics/+bulk-memory/+mutable-globals  → threadable wasm.
# --shared-memory + --import-memory + --max-memory  → emit a SHARED memory so
#   wasm-bindgen-rayon can postMessage it to the child workers (without these
#   the memory is built unshared even with +atomics → "Memory could not be
#   cloned"). 4 GiB is the SharedArrayBuffer ceiling.
# --export=__wasm_init_tls / __tls_*  → wasm-bindgen-rayon needs these to set
#   up thread-local storage in each spawned worker.
#
# RUSTFLAGS (env) overrides the workspace's `.cargo/config.toml`
# `target-cpu=native` (which is invalid for wasm32) for this invocation only.
export RUSTFLAGS='-C target-feature=+atomics,+bulk-memory,+mutable-globals -C link-arg=--shared-memory -C link-arg=--import-memory -C link-arg=--max-memory=4294967296 -C link-arg=--export=__wasm_init_tls -C link-arg=--export=__tls_size -C link-arg=--export=__tls_align -C link-arg=--export=__tls_base'

(
  cd "$SDK_DIR"
  RUSTUP_TOOLCHAIN="$NIGHTLY_TOOLCHAIN" cargo build \
    -p modgrad-wasm \
    --target wasm32-unknown-unknown \
    --release \
    --features parallel \
    -Z build-std=panic_abort,std
)

WASM_IN="$SDK_DIR/target/wasm32-unknown-unknown/release/modgrad_wasm.wasm"
if [ ! -f "$WASM_IN" ]; then
  echo "error: expected wasm at $WASM_IN" >&2
  exit 1
fi

# ── step 2: wasm-bindgen JS glue (target web → dynamic-importable) ───────────
mkdir -p "$OUT_DIR"
# remove any stale _mt artifacts (keep the single-thread modgrad_wasm.* intact)
rm -f "$OUT_DIR/${OUT_NAME}"* 2>/dev/null || true
rm -rf "$OUT_DIR/snippets" 2>/dev/null || true

wasm-bindgen \
  --target web \
  --out-dir "$OUT_DIR" \
  --out-name "$OUT_NAME" \
  "$WASM_IN"

# ── step 2b: patch wasm-bindgen-rayon's workerHelpers.js ─────────────────────
# Under `--target web` (no bundler) the spawned worker re-imports the package
# via `import('../../..')`, which resolves to the /engine DIRECTORY (served as
# text/html) and the browser refuses it as a module. Replace with the explicit
# module filename. Standard wasm-bindgen-rayon `--target web` workaround.
WORKER_HELPER="$(find "$OUT_DIR/snippets" -name workerHelpers.js 2>/dev/null | head -n1 || true)"
if [ -n "$WORKER_HELPER" ]; then
  sed -i "s|await import('../../..')|await import('../../../${OUT_NAME}.js')|" "$WORKER_HELPER"
  echo "patched $(basename "$(dirname "$WORKER_HELPER")")/workerHelpers.js → ${OUT_NAME}.js"
else
  echo "warning: workerHelpers.js not found under $OUT_DIR/snippets — did wasm-bindgen-rayon link?" >&2
fi

# ── step 3 (optional): wasm-opt for size ─────────────────────────────────────
if command -v wasm-opt >/dev/null 2>&1; then
  echo "optimising ${OUT_NAME}_bg.wasm with wasm-opt -O3…"
  wasm-opt -O3 --enable-threads --enable-bulk-memory --enable-mutable-globals \
    "$OUT_DIR/${OUT_NAME}_bg.wasm" -o "$OUT_DIR/${OUT_NAME}_bg.wasm"
else
  echo "wasm-opt not found; skipping size optimisation (cargo install wasm-opt)."
fi

echo
echo "Built:"
ls -la "$OUT_DIR/${OUT_NAME}"* "$OUT_DIR/snippets" 2>/dev/null || ls -la "$OUT_DIR/${OUT_NAME}"*
echo
echo "Threads only engage when the page is cross-origin isolated. Verify with:"
echo "  node $SCRIPT_DIR/serve-coop-coep.mjs 8788 $SITE_DIR/dist"
