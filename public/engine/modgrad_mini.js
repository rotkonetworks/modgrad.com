/* @ts-self-types="./modgrad_mini.d.ts" */

/**
 * Same as `adaptive_compute_pixels` but over a pre-computed flat
 * observation `[n_tokens × token_dim]` (bypasses the retina).
 * @param {Float32Array} obs
 * @returns {any}
 */
export function adaptive_compute_obs(obs) {
    const ptr0 = passArrayF32ToWasm0(obs, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.adaptive_compute_obs(ptr0, len0);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * Per-tick adaptive-compute summary for raw RGB pixels `[3 × H × W]`
 * CHW. The brain's outer exit gate is global (not per-region), so the
 * honest unit of "adaptive compute" is the OUTER tick. Returns:
 * `{ticks_used, lambda_trajectory[ticks_used], survival[ticks_used],
 *   exit_cdf[ticks_used], certainty[ticks_used]}` where, per tick,
 * `lambda` is the outer gate λ, `survival` is Π(1−λ) up to (excluding)
 * that tick, `exit_cdf` is the cumulative exit probability Σ λ·survival,
 * and `certainty` is `1 − normalized_entropy(prediction)`.
 *
 * No per-region exit λ is fabricated: the inner per-region gates are not
 * exported by the forward, so this summarises what genuinely gates the
 * brain's compute — the outer adaptive gate.
 * @param {Float32Array} pixels
 * @returns {any}
 */
export function adaptive_compute_pixels(pixels) {
    const ptr0 = passArrayF32ToWasm0(pixels, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.adaptive_compute_pixels(ptr0, len0);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * Apply ONE bounded three-factor local update to the readout's first-5
 * (move) rows, using the LAST forward's stashed `(pre, logits)`:
 *
 *   ΔW[d,j] = θ · signal · (onehot(chosen)[d] − softmax(logits[0..5])[d]) · pre[j]
 *
 * with learning rate θ = 0.02, per-step `|ΔW| ≤ 0.25`, post-update
 * `|W| ≤ 6.0`, and a NaN/Inf guard (non-finite ΔW elements are skipped).
 * `chosen` is the move index in `0..5`. `signal` is the reward/error
 * scalar; with `signal == 0` NOTHING changes (Δ is exactly 0), so the
 * forward stays bit-exact between updates. The deltas are ADDED to
 * `output_proj.weight` in place. Returns the L2 norm ‖ΔW‖ actually
 * applied. The pristine readout is snapshotted lazily on first call so
 * `reset_plasticity` can restore it.
 *
 * Requires `load_brain_weights` + at least one run/decision call first
 * (so `pre`/`logits` are stashed).
 * @param {number} chosen
 * @param {number} signal
 * @returns {number}
 */
export function apply_plasticity(chosen, signal) {
    const ret = wasm.apply_plasticity(chosen, signal);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0];
}

/**
 * Same as `brain_telemetry_pixels` but over a pre-computed flat
 * observation `[n_tokens × token_dim]` (bypasses the retina).
 * @param {Float32Array} obs
 * @returns {any}
 */
export function brain_telemetry_obs(obs) {
    const ptr0 = passArrayF32ToWasm0(obs, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.brain_telemetry_obs(ptr0, len0);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * Per-tick, per-region telemetry derived from a fresh forward over raw
 * RGB pixels `[3 × H × W]` CHW. Returns `[TickTelemetry]`: for each
 * outer tick, an array of per-region `{region, name, d_model,
 * activation_rms, activation_peak, activation_mean}` plus
 * `{global_sync_rms, global_sync_peak, exit_lambda}`.
 *
 * All values are computed from the forward's own outputs; no
 * neuromodulator/homeostasis state exists in this reimplementation, so
 * none is invented. Re-runs the forward (does not stash plasticity
 * inputs — use `run_brain_pixels` if you also want to plasticise).
 * @param {Float32Array} pixels
 * @returns {any}
 */
export function brain_telemetry_pixels(pixels) {
    const ptr0 = passArrayF32ToWasm0(pixels, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.brain_telemetry_pixels(ptr0, len0);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * Same as `decision_drivers_pixels` but over a pre-computed flat
 * observation `[n_tokens × token_dim]` (bypasses the retina).
 * @param {Float32Array} obs
 * @returns {any}
 */
export function decision_drivers_obs(obs) {
    const ptr0 = passArrayF32ToWasm0(obs, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.decision_drivers_obs(ptr0, len0);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * "What drives the decision" for raw RGB pixels `[3 × H × W]` CHW.
 * Runs the forward, then decomposes the LAST tick's readout
 * `pred = output_proj(global_sync)`. Returns `DecisionDrivers`:
 * `{tick, pre[n_global_sync], logits[out_dims], move_softmax[n_moves],
 *   move_contributions[n_moves][n_global_sync], move_bias[n_moves],
 *   n_moves}` where `move_contributions[j][i] = W[j,i]·pre[i]`.
 *
 * This brain's outer forward exposes no attention/eligibility, so none
 * is fabricated; `pre` and the per-channel readout contributions are the
 * honest "drivers." Also stashes plasticity inputs (free).
 * @param {Float32Array} pixels
 * @returns {any}
 */
export function decision_drivers_pixels(pixels) {
    const ptr0 = passArrayF32ToWasm0(pixels, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.decision_drivers_pixels(ptr0, len0);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * Browser-facing maze encoder. `grid` is row-major (1 = wall), returns the
 * flat `[size*size * 9]` observation to pass straight into `run`.
 * @param {Uint8Array} grid
 * @param {number} size
 * @param {number} agent_r
 * @param {number} agent_c
 * @param {number} goal_r
 * @param {number} goal_c
 * @returns {Float32Array}
 */
export function encode_maze_js(grid, size, agent_r, agent_c, goal_r, goal_c) {
    const ptr0 = passArray8ToWasm0(grid, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.encode_maze_js(ptr0, len0, size, agent_r, agent_c, goal_r, goal_c);
    var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v2;
}

/**
 * Run the loaded learned VIN forward over `tokens` (flat
 * `[grid_h*grid_w × raw_dim]`, row-major) with the agent at
 * `(agent_r, agent_c)`. Returns the `n_dirs` move logits. Errors if no
 * learned VIN has been loaded via `load_learned_vin`.
 * @param {Float32Array} tokens
 * @param {number} grid_h
 * @param {number} grid_w
 * @param {number} agent_r
 * @param {number} agent_c
 * @returns {Float32Array}
 */
export function learned_vin_forward(tokens, grid_h, grid_w, agent_r, agent_c) {
    const ptr0 = passArrayF32ToWasm0(tokens, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.learned_vin_forward(ptr0, len0, grid_h, grid_w, agent_r, agent_c);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v2;
}

/**
 * Like `learned_vin_forward`, but appends the planner's per-cell scalar
 * value map after the move logits: `[n_dirs logits | grid_h*grid_w value]`.
 * The value map is the model's OWN proximity-to-goal estimate (no solver),
 * so the caller can derive an honest "getting closer?" signal from it.
 * @param {Float32Array} tokens
 * @param {number} grid_h
 * @param {number} grid_w
 * @param {number} agent_r
 * @param {number} agent_c
 * @returns {Float32Array}
 */
export function learned_vin_forward_compass(tokens, grid_h, grid_w, agent_r, agent_c) {
    const ptr0 = passArrayF32ToWasm0(tokens, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.learned_vin_forward_compass(ptr0, len0, grid_h, grid_w, agent_r, agent_c);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v2;
}

/**
 * Restore the learned VIN's move head to its pristine (as-loaded) weights,
 * undoing every sleep-consolidation pass. No-op if it was never trained.
 */
export function learned_vin_reset() {
    wasm.learned_vin_reset();
}

/**
 * DREAM CONSOLIDATION step on the LOADED learned VIN: one SGD update of the
 * move head toward `target_move` for the maze in `tokens`/`agent`. Mutates
 * the resident planner (persists for the session) and returns the
 * cross-entropy loss before the step. A sleep/replay pass calls this per
 * remembered maze (target = its BFS-optimal move).
 * @param {Float32Array} tokens
 * @param {number} grid_h
 * @param {number} grid_w
 * @param {number} agent_r
 * @param {number} agent_c
 * @param {number} target_move
 * @param {number} lr
 * @returns {number}
 */
export function learned_vin_train(tokens, grid_h, grid_w, agent_r, agent_c, target_move, lr) {
    const ptr0 = passArrayF32ToWasm0(tokens, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.learned_vin_train(ptr0, len0, grid_h, grid_w, agent_r, agent_c, target_move, lr);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0];
}

/**
 * Parse `brain_weights.json` and stash the regional weights (and the
 * retina, if present) for subsequent `run_brain`s.
 *
 * `cortex` is OPTIONAL: the visual-retina brain ships `{ cortex, regional }`,
 * while the flat-encoding brain ships `{ regional }` with no retina at all.
 * When `cortex` is absent, CORTEX stays `None` and only `run_brain_obs`
 * (which takes a flat observation directly) is usable.
 * @param {string} json
 */
export function load_brain_weights(json) {
    const ptr0 = passStringToWasm0(json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.load_brain_weights(ptr0, len0);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

/**
 * Load the TRAINED VIN planner from the SDK's serde JSON export of a
 * `VinReadout` and stash it for `learned_vin_forward`. Field names match
 * the SDK exactly (`config, raw_dim, reward_proj, gate_proj, value_proj,
 * agent_proj, highway_proj, move_head`). This is the real learned forward
 * — distinct from the hardcoded `vin_forward` value-iteration path.
 * @param {string} json
 */
export function load_learned_vin(json) {
    const ptr0 = passStringToWasm0(json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.load_learned_vin(ptr0, len0);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

/**
 * Parse weights JSON and stash them in a thread-local for subsequent `run`s.
 * Returns an error string on parse failure.
 * @param {string} json
 */
export function load_weights(json) {
    const ptr0 = passStringToWasm0(json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.load_weights(ptr0, len0);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

/**
 * Restore `output_proj` to its pristine (as-loaded) state, undoing all
 * `apply_plasticity` updates. No-op if plasticity was never applied
 * (no snapshot taken yet). The snapshot itself is retained so repeated
 * reset → plasticise → reset cycles all restore the same baseline.
 */
export function reset_plasticity() {
    const ret = wasm.reset_plasticity();
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

/**
 * Retina feature maps for the given pixels: retina → v1 → v2 → v4
 * activations (CHW), for the vision panel. Returns `[{name, channels,
 * h, w, data}]`. Requires `load_brain_weights` first.
 * @param {Float32Array} pixels
 * @returns {any}
 */
export function retina_maps(pixels) {
    const ptr0 = passArrayF32ToWasm0(pixels, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.retina_maps(ptr0, len0);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * Run the forward pass over `obs` (`[n_tokens × raw_dim]` flat) and return
 * the per-tick observable state (`ForwardOut`) as a JS value.
 * Errors if `load_weights` has not been called yet.
 * @param {Float32Array} obs
 * @param {number} n_tokens
 * @param {number} raw_dim
 * @returns {any}
 */
export function run(obs, n_tokens, raw_dim) {
    const ptr0 = passArrayF32ToWasm0(obs, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.run(ptr0, len0, n_tokens, raw_dim);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * Run the brain directly on a pre-computed flat observation
 * (`[n_tokens × token_dim]`), bypassing the retina. Returns the
 * per-tick `BrainOut` as a JS value.
 * @param {Float32Array} obs
 * @returns {any}
 */
export function run_brain_obs(obs) {
    const ptr0 = passArrayF32ToWasm0(obs, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.run_brain_obs(ptr0, len0);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * Run the brain on raw RGB pixels `[3 × H × W]` CHW: retina →
 * spatial tokens → `regional_forward`. Returns the per-tick
 * `BrainOut` as a JS value. Requires `load_brain_weights` first.
 * @param {Float32Array} pixels
 * @returns {any}
 */
export function run_brain_pixels(pixels) {
    const ptr0 = passArrayF32ToWasm0(pixels, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.run_brain_pixels(ptr0, len0);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * Run the trainable VIN readout DIRECTLY over the raw maze pixels
 * `[3 × SIZE × SIZE]` CHW (NOT the retina). Derives wall/goal/agent from
 * the pixels (wall=black, goal=green, agent=red), runs EXACT value
 * iteration on the SIZE×SIZE grid so goal-value floods backward around
 * walls, then reads 5 move logits from the AGENT'S OWN cell (its value +
 * gate + 4 neighbour values) — no global pooling, no learned geometry.
 *
 * `agent_row`/`agent_col` are the agent's TRUE maze coords, used only as a
 * fallback if no red agent pixel is present. SIZE = round(sqrt(len/3)).
 * The VIN move head is lazily seeded (deterministic) on first call.
 *
 * Returns `{ move_logits: [5], agent_cell: [r,c], grid: [SIZE,SIZE],
 *            value_grid: [SIZE²], gate: [SIZE²] }`. The agent-cell
 * readout and move logits are stashed for `vin_learn`. No cortex needed.
 * @param {Float32Array} pixels
 * @param {number} agent_row
 * @param {number} agent_col
 * @returns {any}
 */
export function vin_forward(pixels, agent_row, agent_col) {
    const ptr0 = passArrayF32ToWasm0(pixels, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.vin_forward(ptr0, len0, agent_row, agent_col);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * Apply ONE three-factor local update to the VIN move head only, using
 * the last `vin_forward`'s stashed `(pre, logits)`:
 *
 *   eligibility[d,j] = (onehot(target_move)[d] − softmax(logits)[d])·pre[j]
 *   ΔW[d,j]          = θ · (teach − pain) · eligibility[d,j]
 *
 * `target_move` ∈ `0..5` teaches that move (teach = 1); `target_move < 0`
 * applies only the `−pain` (wall-hit) penalty. θ = 0.05, bounded
 * `|ΔW| ≤ 0.25`, `|W| ≤ 6.0`, NaN/Inf-guarded. Momentum-free, no backprop
 * graph — only the move head changes (value-propagation weights are
 * FIXED). Returns ‖ΔW‖ applied. Requires a prior `vin_forward`.
 * @param {number} target_move
 * @param {number} pain
 * @returns {number}
 */
export function vin_learn(target_move, pain) {
    const ret = wasm.vin_learn(target_move, pain);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0];
}

/**
 * Re-seed the VIN readout to its deterministic init (the "Reset learning"
 * path), discarding all `vin_learn` updates. Rebuilt for the current
 * per-cell width on the next `vin_forward` if the width changes; if a VIN
 * already exists it is re-seeded in place for the same width.
 */
export function vin_reset() {
    wasm.vin_reset();
}

function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg_Error_83742b46f01ce22d: function(arg0, arg1) {
            const ret = Error(getStringFromWasm0(arg0, arg1));
            return ret;
        },
        __wbg_String_8564e559799eccda: function(arg0, arg1) {
            const ret = String(arg1);
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg___wbindgen_throw_6ddd609b62940d55: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg_new_a70fbab9066b301f: function() {
            const ret = new Array();
            return ret;
        },
        __wbg_new_ab79df5bd7c26067: function() {
            const ret = new Object();
            return ret;
        },
        __wbg_set_282384002438957f: function(arg0, arg1, arg2) {
            arg0[arg1 >>> 0] = arg2;
        },
        __wbg_set_6be42768c690e380: function(arg0, arg1, arg2) {
            arg0[arg1] = arg2;
        },
        __wbindgen_cast_0000000000000001: function(arg0) {
            // Cast intrinsic for `F64 -> Externref`.
            const ret = arg0;
            return ret;
        },
        __wbindgen_cast_0000000000000002: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_cast_0000000000000003: function(arg0) {
            // Cast intrinsic for `U64 -> Externref`.
            const ret = BigInt.asUintN(64, arg0);
            return ret;
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./modgrad_mini_bg.js": import0,
    };
}

function getArrayF32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

let cachedFloat32ArrayMemory0 = null;
function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(ptr, len);
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function passArray8ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 1, 1) >>> 0;
    getUint8ArrayMemory0().set(arg, ptr / 1);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArrayF32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getFloat32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasm;
function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    wasmModule = module;
    cachedDataViewMemory0 = null;
    cachedFloat32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('modgrad_mini_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
