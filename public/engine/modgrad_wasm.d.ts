/* tslint:disable */
/* eslint-disable */

export function apply_plasticity(_chosen: number, _signal: number): number;

export function learned_vin_forward_compass(tokens: Float32Array, grid_h: number, grid_w: number, agent_r: number, agent_c: number): Float32Array;

export function learned_vin_reset(): void;

export function learned_vin_train(tokens: Float32Array, grid_h: number, grid_w: number, agent_r: number, agent_c: number, target_move: number, lr: number): number;

export function load_brain_weights(json: string): void;

export function load_learned_vin(json: string): void;

export function reset_plasticity(): void;

export function retina_maps(pixels: Float32Array): any;

export function run_brain_pixels(pixels: Float32Array): any;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly apply_plasticity: (a: number, b: number) => [number, number, number];
    readonly learned_vin_forward_compass: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number, number];
    readonly learned_vin_reset: () => void;
    readonly learned_vin_train: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number, number];
    readonly load_brain_weights: (a: number, b: number) => [number, number];
    readonly load_learned_vin: (a: number, b: number) => [number, number];
    readonly reset_plasticity: () => [number, number];
    readonly retina_maps: (a: number, b: number) => [number, number, number];
    readonly run_brain_pixels: (a: number, b: number) => [number, number, number];
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
