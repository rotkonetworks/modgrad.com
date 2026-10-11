// The analysis module (crates/modgrad-obs-analysis, built into obs/pkg by
// scripts/build_obs_wasm.sh). Vite bundles the glue and emits the .wasm as an asset.
import init, * as wasm from '@pkg/modgrad_obs_analysis.js';

export type Wasm = typeof wasm;
export const A: Wasm = wasm;
let started: Promise<unknown> | null = null;
export function initWasm(): Promise<unknown> {
  if (!started) started = init();
  return started;
}
