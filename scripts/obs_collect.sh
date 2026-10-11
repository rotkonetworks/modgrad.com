#!/usr/bin/env bash
# Collect modgrad-obs runs from several machines/paths into one local
# obs/runs/<label>/<run>/, convert legacy ctmlm_pilot log.txt files, and
# build obs/index.json + per-run plot files for obs/dashboard.html, and
# the analysis wasm (obs/pkg) the dashboard computes with.
#
#   scripts/obs_collect.sh                 # uses scripts/obs_sources.txt
#   OBS_SOURCES=my.txt OBS_DIR=/srv/obs scripts/obs_collect.sh
#   OBS_DEMO=0 scripts/obs_collect.sh      # skip the synthetic demo runs
#
# Safe to re-run: rsync is incremental and never deletes (derived files
# live next to the copied ones); conversion only re-runs when log.txt is
# newer than its converted metrics.jsonl; an unreachable source is
# reported and skipped. Serve the result over HTTP (fetch() does not work
# from file://), e.g.:  python3 -m http.server --directory obs 8781
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OBS_DIR="${OBS_DIR:-$REPO/obs}"
OBS_SOURCES="${OBS_SOURCES:-$REPO/scripts/obs_sources.txt}"
OBS_DEMO="${OBS_DEMO:-1}"
RUNS="$OBS_DIR/runs"
mkdir -p "$RUNS"

echo "== building obs CLI"
(cd "$REPO" && cargo build -q --release -p modgrad-obs --bin obs -j 4) || { echo "build failed"; exit 1; }
OBS_BIN="$REPO/target/release/obs"

# only run artefacts, never checkpoints (brain.json/read.json can be large)
FILTER=(--include='*/' --include='log.txt' --include='*.jsonl' --include='config.json' --include='summary.json' --exclude='*')
RSYNC=(rsync -a --timeout=30 -e "ssh -o BatchMode=yes -o ConnectTimeout=10" "${FILTER[@]}")

echo "== syncing sources from $OBS_SOURCES"
while read -r label kind src; do
  [[ -z "${label:-}" || "$label" == \#* ]] && continue
  case "$kind" in
    dir) dest="$RUNS/$label/"; mkdir -p "$dest"
         "${RSYNC[@]}" "${src%/}/" "$dest" && echo "  ok   $label <- $src" || echo "  FAIL $label <- $src (skipped)";;
    run) name="$(basename "${src%/}")"; dest="$RUNS/$label/$name/"; mkdir -p "$dest"
         "${RSYNC[@]}" "${src%/}/" "$dest" && echo "  ok   $label/$name <- $src" || echo "  FAIL $label/$name <- $src (skipped)";;
    *) echo "  bad kind '$kind' for $label";;
  esac
done < "$OBS_SOURCES"

echo "== converting legacy ctmlm_pilot logs"
find "$RUNS" -mindepth 2 -maxdepth 3 -name log.txt | sort | while read -r log; do
  d="$(dirname "$log")"
  if [[ -f "$d/metrics.jsonl" && "$d/metrics.jsonl" -nt "$log" ]]; then continue; fi
  label="$(basename "$(dirname "$d")")"
  "$OBS_BIN" legacy "$log" "$d" --source "$label" --log-ref "$label:$(basename "$d")/log.txt" | sed 's/^/  /'
done

if [[ "$OBS_DEMO" == 1 && ( ! -d "$RUNS/demo_synthetic_small" || "$REPO/crates/modgrad-obs/examples/ctmlm_integration.rs" -nt "$RUNS/demo_synthetic_small/config.json" ) ]]; then
  echo "== writing synthetic demo runs (clearly labelled, not measurements)"
  (cd "$REPO" && cargo run -q --release -p modgrad-obs --example ctmlm_integration -j 4 -- --obs "$OBS_DIR") | sed 's/^/  /'
fi

if [[ ! -f "$OBS_DIR/pkg/modgrad_obs_analysis_bg.wasm" || -n "$(find "$REPO/crates/modgrad-obs-analysis/src" -newer "$OBS_DIR/pkg/modgrad_obs_analysis_bg.wasm" 2>/dev/null)" ]]; then
  echo "== building analysis wasm"
  OUT_DIR="$OBS_DIR/pkg" "$REPO/scripts/build_obs_wasm.sh" >/dev/null || echo "  wasm build failed (dashboard needs obs/pkg)"
fi

echo "== exporting index"
"$OBS_BIN" export "$OBS_DIR"
echo "dashboard: $OBS_DIR/dashboard.html (serve: python3 -m http.server --directory $OBS_DIR 8781)"
