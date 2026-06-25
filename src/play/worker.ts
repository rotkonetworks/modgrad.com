/// <reference lib="webworker" />
// modgrad brain worker — runs the real wasm 8-region brain off the main thread.
// The main thread only renders; all forward passes happen here.
//
// The engine package is served from /engine/ (copied from engine/pkg into
// public/engine). We import it by absolute URL so Vite leaves it alone and the
// browser loads it at runtime — keeping the ~1MB of wasm + weights off every
// other route.
//
// The 8-region brain (with a visual retina) is now the SOLVER: it reads the
// maze as RGB pixels, runs its forward pass, and the argmax of the last tick's
// first 5 logits is the move (0..3 = UP/DOWN/LEFT/RIGHT, 4 = wait/stuck).

// per-tick brain state for the 3D particle viz: every neuron's activation
// (grouped by region) so individual neurons spike, plus global-sync magnitude
// and the outer exit lambda.
type BrainTick = {
  acts: number[][]; // [n_regions][d_model] per-neuron activation this tick
  global: number; // RMS of the global-sync vector
  exit: number | null; // outer AdaptiveGate lambda (null if no gate)
};
type BrainTrace = { ticks: BrainTick[]; ticksUsed: number };

// Per-region telemetry/neuromod surfaced by the NEW engine (feature-detected).
// One entry per region; whatever scalar getters the engine exposes. All
// optional so the UI can render only what's present.
type RegionTelemetry = {
  name?: string; // region name if the engine names it
  activity?: number; // mean |activation| / firing rate for the region
  neuromod?: number; // neuromodulator level (e.g. dopamine-like signal)
  certainty?: number; // per-region certainty / confidence if exposed
};
// Engine-level certainty / exit-gate signals (NEW engine, feature-detected).
type BrainTelemetry = {
  regions: RegionTelemetry[]; // per-region telemetry (may be empty)
  exitGate?: number; // outer exit-gate value (halting / commitment)
  certainty?: number; // overall decision certainty
};

type StepResult = {
  type: "step";
  agent: [number, number]; // the cell the agent steps to (optimal path)
  move: number; // the brain's PREDICTED move (0..3, 4 = wait) — drives the bars
  verdict: "ok" | "wall" | "astray" | "wait"; // brain's prediction vs the maze
  agreed: boolean; // did the brain's prediction match the optimal step?
  moveLogits: number[]; // last tick's first 4 direction logits (for the move bars)
  brain: BrainTrace | null; // the 8-region brain's run on this maze state
  vision: RetinaMap[] | null; // sight + V1/V2/V4 feature maps the retina computed
  attn: number[] | null; // per-cell retina saliency mapped to the maze (SIZE²)
  route: [number, number][]; // the brain's predicted route — its attention targets
  done: boolean;
  reached: boolean;
  // NEW engine signals — all optional, present only when the engine exports them.
  telemetry?: BrainTelemetry; // per-region telemetry + exit-gate / certainty
  plasticDelta?: number; // ΔW magnitude returned by apply_plasticity this step
  signal?: number; // the plasticity signal we fed in (pain<0 / reward>0)
  loss?: number; // move-head cross-entropy vs the target move (the curve that drops)
  lr?: number; // the plasticity learning rate θ used this step (constant)
  episodic?: EpisodicRecall | null; // nearest recalled past situation (memory)
  // VIN closed-loop learning (NEW engine, feature-detected). When the VIN is
  // driving, `move`/`verdict`/`agreed`/`moveLogits` above are the VIN's, and
  // these flag/annotate it so the UI can label the learning loop.
  vinActive?: boolean; // true when the VIN drove this step's prediction
  dreaming?: boolean; // true on the step a sleep/replay consolidation just ran
};

// The VIN (value-iteration readout) forward result, feature-detected on the
// engine. Shapes are coerced defensively from the raw wasm return.
type VinForward = {
  move_logits: number[]; // [5] direction logits (UP/DOWN/LEFT/RIGHT/WAIT)
  agent_cell: [number, number]; // the cell the VIN localized the agent to
  value_grid: number[]; // value estimate per maze cell (SIZE²) — for the UI
  gate: number[]; // per-iteration / per-region gate activations
};

// one visual-cortex layer's feature maps (CHW), from the wasm `retina_maps`
type RetinaMap = { name: string; channels: number; h: number; w: number; data: number[] };

// Optional NEW engine exports (feature-detected at init). Each is a plain
// function on the wasm module; we only call those that `typeof === "function"`.
type EngineExtras = {
  apply_plasticity: ((chosen: number, signal: number) => number) | null;
  reset_plasticity: (() => void) | null;
  region_telemetry: (() => unknown) | null; // per-region telemetry getter
  exit_gate: (() => number) | null; // exit-gate / halting scalar
  certainty: (() => number) | null; // overall certainty scalar
  // ── VIN learning loop (NEW engine, feature-detected) ──
  // vin_forward: read the maze + agent position, return the VIN's move logits,
  //   localized agent cell, value grid and gate activations.
  // vin_learn: teach the VIN — targetMove≥0 imitates that move, pain>0 penalizes
  //   (three-factor rule); returns ‖ΔW‖. targetMove=-1 = pure-pain update.
  // vin_reset: wipe the VIN's accumulated plastic changes.
  vin_forward: ((pixels: Float32Array, agentRow: number, agentCol: number) => unknown) | null;
  vin_learn: ((targetMove: number, pain: number) => number) | null;
  vin_reset: (() => void) | null;
};

let engine:
  | ({
      run_brain_pixels: (pixels: Float32Array) => BrainOut;
      retina_maps: ((pixels: Float32Array) => RetinaMap[]) | null;
    } & EngineExtras)
  | null = null;

// ── attention config (Task 1) ─────────────────────────────────────────────
// Occlusion attention is O(SIZE²) brain forwards/step at full resolution.
// Modes: "off" (skip), "coarse" (only open cells near the agent, default),
// "full" (every cell — the old behaviour). Throttled: recomputed at most every
// ATTN_EVERY steps; in between we reuse the cached grid (whose shape never
// changes, so the UI overlay stays valid).
type AttnMode = "off" | "coarse" | "full";
let attnMode: AttnMode = "coarse";
let attnEvery = 3; // recompute occlusion at most every N steps
let attnRadius = 2; // coarse mode: Chebyshev radius around the agent to probe
let attnStepCounter = 0; // increments per step; gates recomputation
let attnCache: number[] | null = null; // last computed saliency grid (SIZE²)

// shape returned by the wasm `run_brain_pixels`
type BrainOut = {
  ticks: {
    prediction: number[];
    region_activations: number[][]; // [n_regions][d_model]
    global_sync: number[];
    exit_lambda: number | null;
  }[];
  ticks_used: number;
};

let SIZE = 9;

// ── VIN learning loop state ────────────────────────────────────────────────
// vinMode is ON by default when the engine exposes vin_forward (set at init).
// When on, the VIN drives the predicted move + verdict and learns each step.
let vinMode = true; // caller may override at init; only effective if vin_forward exists
let vinAvailable = false; // set true at init iff vin_forward is a function
let mazeCounter = 0; // counts mazes seen — gates periodic sleep/replay
const DREAM_EVERY = 5; // run a sleep/replay consolidation every Nth new maze
const DREAM_MAX_REPLAYS = 64; // bounded offline replays per sleep pass
// Three-factor learning rates θ baked into the engine (brain.rs): the VIN
// move-head uses 0.05, the fallback apply_plasticity uses 0.02. Surfaced so the
// Plasticity panel can show the lr next to the loss curve.
const VIN_LR = 0.05;
const PLASTIC_LR = 0.02;

const post = (m: unknown) => (self as unknown as Worker).postMessage(m);

function rms(a: number[]): number {
  if (a.length === 0) return 0;
  let s = 0;
  for (const v of a) s += v * v;
  return Math.sqrt(s / a.length);
}

// Reused pixel buffers (Task 2) — avoid allocating a fresh Float32Array on
// every renderPixels call (hot path: once per step + SIZE² times per occlusion
// pass). `pxBuf` is the shared default; the occlusion loop uses its own buffer
// so it can render with a mask without clobbering the step's base render.
// Buffers are (re)sized lazily whenever SIZE changes.
let pxBuf: Float32Array = new Float32Array(0);
let pxBufOccl: Float32Array = new Float32Array(0);
function ensurePxBufs(): void {
  const want = 3 * SIZE * SIZE;
  if (pxBuf.length !== want) pxBuf = new Float32Array(want);
  if (pxBufOccl.length !== want) pxBufOccl = new Float32Array(want);
}

// Render the maze to RGB pixels [3 × SIZE × SIZE], CHW (R plane, G plane, B
// plane), EXACTLY the scheme run_brain's retina was trained on:
// wall=(0,0,0), open=(1,1,1), agent=(1,0,0) red, goal=(0,1,0) green.
// Agent/goal overwrite the cell colour. (Must match render_maze in the SDK.)
// `out` lets callers pass a buffer to reuse; defaults to the shared `pxBuf`.
function renderPixels(
  grid: number[],
  ar: number,
  ac: number,
  gr: number,
  gc: number,
  maskIdx = -1, // if ≥0, blank that cell to neutral grey (for occlusion saliency)
  out?: Float32Array,
): Float32Array {
  const n = SIZE * SIZE;
  ensurePxBufs();
  const px = out ?? pxBuf;
  for (let r = 0; r < SIZE; r++) {
    for (let c = 0; c < SIZE; c++) {
      const idx = r * SIZE + c;
      let rr: number, gg: number, bb: number;
      if (idx === maskIdx) [rr, gg, bb] = [0.5, 0.5, 0.5]; // occluded → grey
      else if (r === ar && c === ac) [rr, gg, bb] = [1, 0, 0]; // agent red
      else if (r === gr && c === gc) [rr, gg, bb] = [0, 1, 0]; // goal green
      else if (grid[idx] !== 0) [rr, gg, bb] = [0, 0, 0]; // wall black
      else [rr, gg, bb] = [1, 1, 1]; // open white
      px[idx] = rr;
      px[n + idx] = gg;
      px[2 * n + idx] = bb;
    }
  }
  return px;
}

// UP=0 DOWN=1 LEFT=2 RIGHT=3 WAIT=4 — matches the SDK's DIR_* constants, and
// 0..3 line up with DELTA below. The brain reads the move from the LAST tick's
// first N_DIRECTIONS(5) logits (run_brain eval reads predictions.last()[0..5]).
const N_DIRECTIONS = 5;
function brainMoveFrom(out: BrainOut): number {
  const ticks = out.ticks;
  if (!ticks.length) return 4;
  const pred = ticks[ticks.length - 1].prediction;
  let best = 0;
  for (let d = 1; d < N_DIRECTIONS; d++) if (pred[d] > pred[best]) best = d;
  return best; // 0..3 move, 4 = wait
}

// Run the 8-region brain on the current maze state: it BOTH decides the move
// and exposes its per-tick internal state for the 3D viz. Returns null only if
// the brain engine isn't available (older wasm / load failure) so the page can
// degrade gracefully.
function runBrain(
  grid: number[],
  ar: number,
  ac: number,
  gr: number,
  gc: number,
): { trace: BrainTrace; move: number; moveLogits: number[]; pred: number[]; gs: number[] } | null {
  if (!engine?.run_brain_pixels) return null;
  try {
    const px = renderPixels(grid, ar, ac, gr, gc);
    const out = engine.run_brain_pixels(px);
    const ticks: BrainTick[] = out.ticks.map((t) => ({
      acts: t.region_activations, // [region][neuron] — passed straight to the viz
      global: rms(t.global_sync),
      exit: t.exit_lambda,
    }));
    const move = brainMoveFrom(out); // 0..3 move, 4 = wait
    // the move bars show the 4 direction logits of the last tick
    const last = out.ticks[out.ticks.length - 1]?.prediction ?? [];
    const moveLogits = [last[0] ?? 0, last[1] ?? 0, last[2] ?? 0, last[3] ?? 0];
    // last tick's global-sync vector — the brain's distilled state, used as the
    // KEY for episodic memory (recall the nearest past situation by similarity).
    const gs = out.ticks[out.ticks.length - 1]?.global_sync ?? [];
    // the full prediction is the brain's whole intended ROUTE (route_len × 5),
    // not just the first move — we decode it into target cells for the maze.
    return { trace: { ticks, ticksUsed: out.ticks_used }, move, moveLogits, pred: last, gs };
  } catch {
    return null; // never let a brain hiccup hard-crash the demo
  }
}

// ── VIN forward ────────────────────────────────────────────────────────────
// Run the engine's value-iteration readout on the maze + agent position and
// coerce its raw return into a typed VinForward. Feature-detected: returns null
// if the engine doesn't export vin_forward (old engine) or the call throws.
function vinForward(
  grid: number[],
  ar: number,
  ac: number,
  gr: number,
  gc: number,
): VinForward | null {
  if (!engine?.vin_forward) return null;
  try {
    const px = renderPixels(grid, ar, ac, gr, gc);
    // The VIN now runs DIRECTLY on the raw SIZE×SIZE maze (no retina, no 5×5
    // downsample), so the agent cell is the TRUE maze cell — pass it straight
    // through (the engine also self-localizes from the red pixel as a fallback).
    const raw = engine.vin_forward(px, ar, ac) as unknown;
    if (!raw || typeof raw !== "object") return null;
    const o = raw as Record<string, unknown>;
    const logits = Array.isArray(o.move_logits) ? (o.move_logits as number[]) : [];
    const ml: number[] = [];
    for (let d = 0; d < N_DIRECTIONS; d++) ml.push(typeof logits[d] === "number" ? logits[d] : 0);
    let cell: [number, number] = [ar, ac];
    if (Array.isArray(o.agent_cell) && o.agent_cell.length >= 2) {
      const a = o.agent_cell as unknown[];
      if (typeof a[0] === "number" && typeof a[1] === "number") cell = [a[0], a[1]];
    }
    const value_grid = Array.isArray(o.value_grid) ? (o.value_grid as number[]).slice() : [];
    const gate = Array.isArray(o.gate) ? (o.gate as number[]).slice() : [];
    return { move_logits: ml, agent_cell: cell, value_grid, gate };
  } catch {
    return null;
  }
}

// The VIN's value field is EXACT value-iteration on the maze, so the optimal
// move is simply the open neighbour with the highest value — the VIN executing
// its own plan (solves 100%). Returns a direction 0..3, or -1 if none.
function greedyFromValue(vf: VinForward, ar: number, ac: number, grid: number[]): number {
  const vg = vf.value_grid;
  if (!vg || vg.length < SIZE * SIZE) return -1;
  let best = -1;
  let bv = -Infinity;
  for (let k = 0; k < 4; k++) {
    const nr = ar + DELTA[k][0];
    const nc = ac + DELTA[k][1];
    if (nr < 0 || nr >= SIZE || nc < 0 || nc >= SIZE) continue;
    if (grid[nr * SIZE + nc] === 1) continue; // wall
    const v = vg[nr * SIZE + nc];
    if (v > bv) {
      bv = v;
      best = k;
    }
  }
  return best;
}

// argmax of the first N_DIRECTIONS move logits — the VIN's chosen move (0..4).
function argmaxMove(logits: number[]): number {
  let best = 0;
  for (let d = 1; d < N_DIRECTIONS; d++) if ((logits[d] ?? 0) > (logits[best] ?? 0)) best = d;
  return best;
}

// ── DREAM / SLEEP REPLAY ────────────────────────────────────────────────────
// Offline consolidation: replay a bounded sample of stored episodes through
// vin_learn again — re-running vin_forward on each remembered maze to re-derive
// its current pain, then re-teaching toward its BFS-optimal move. Wall-hit
// "nightmares" are prioritized so the VIN over-rehearses its worst mistakes.
// Returns the number of replays actually performed (0 if VIN/learn absent).
function dreamReplay(maxReplays: number): number {
  if (!engine?.vin_forward || !engine?.vin_learn) return 0;
  // gather episodes that carry replay context, nightmares (wall) first.
  const withReplay = episodicMem.filter((ep) => ep.replay != null);
  if (!withReplay.length) return 0;
  const nightmares = withReplay.filter((ep) => ep.verdict === "wall");
  const rest = withReplay.filter((ep) => ep.verdict !== "wall");
  const ordered = [...nightmares, ...rest].slice(0, maxReplays);
  let replays = 0;
  for (const ep of ordered) {
    const r = ep.replay!;
    try {
      // re-derive the current pain: does the VIN STILL want a wall move here?
      const vf = vinForward(r.grid, r.ar, r.ac, r.gr, r.gc);
      let pain = r.pain; // fall back to the stored pain
      if (vf) {
        const dist = goalDist(r.grid, r.gr, r.gc);
        const here = dist[r.ar * SIZE + r.ac];
        const v = classifyMove(argmaxMove(vf.move_logits), r.grid, r.ar, r.ac, here, dist);
        pain = v === "wall" ? 1.0 : 0.0;
      }
      const delta = engine.vin_learn(r.optMove, pain);
      if (typeof delta === "number" && Number.isFinite(delta)) replays++;
    } catch {
      /* skip a bad replay, keep consolidating */
    }
  }
  return replays;
}

// ── Episodic memory ───────────────────────────────────────────────────────
// A real session-spanning associative memory (mirrors the SDK's hippocampal
// episodic store, client-side): every decision is keyed by the brain's
// global-sync state and stored; each new decision RECALLS the nearest past
// episode by cosine similarity — "I've been in a situation like this before,
// last time I went X and it {worked / hit a wall}." Accumulates across mazes.
type Episode = {
  key: number[];
  norm: number; // precomputed ‖key‖ for cosine
  move: number;
  verdict: string;
  reached: boolean;
  id: number;
  // ── VIN dream-replay context (optional) ──
  // The maze snapshot this decision happened in, so a sleep pass can re-render
  // the pixels, re-derive the target/pain, and re-teach the VIN offline. Only
  // populated in VIN mode; "nightmares" (wall verdicts) are prioritized.
  replay?: {
    grid: number[]; // maze grid at decision time
    ar: number; // agent row/col
    ac: number;
    gr: number; // goal row/col
    gc: number;
    optMove: number; // BFS-optimal move to teach toward
    pain: number; // pain the VIN earned here (1 if its move hit a wall/edge)
  };
};
type EpisodicRecall = {
  recalled: boolean; // similarity above the recall threshold
  sim: number; // cosine similarity to the nearest past episode
  id: number; // which stored episode matched
  size: number; // episodes currently held
  move: number;
  verdict: string;
  reached: boolean;
};
const EPISODIC_CAP = 256;
const EPISODIC_RECALL_THRESHOLD = 0.6;
let episodicMem: Episode[] = [];
let episodeCounter = 0;

function episodicRecall(gs: number[]): EpisodicRecall | null {
  if (gs.length === 0) return null;
  let qn = 0;
  for (const v of gs) qn += v * v;
  qn = Math.sqrt(qn) || 1;
  let bestSim = -2;
  let best: Episode | null = null;
  for (const ep of episodicMem) {
    if (ep.key.length !== gs.length) continue;
    let dot = 0;
    for (let i = 0; i < gs.length; i++) dot += gs[i] * ep.key[i];
    const sim = dot / (qn * ep.norm);
    if (sim > bestSim) {
      bestSim = sim;
      best = ep;
    }
  }
  if (!best) return null;
  return {
    recalled: bestSim >= EPISODIC_RECALL_THRESHOLD,
    sim: bestSim,
    id: best.id,
    size: episodicMem.length,
    move: best.move,
    verdict: best.verdict,
    reached: best.reached,
  };
}

function episodicStore(
  gs: number[],
  move: number,
  verdict: string,
  reached: boolean,
  replay?: Episode["replay"],
): void {
  if (gs.length === 0) return;
  let n = 0;
  for (const v of gs) n += v * v;
  n = Math.sqrt(n) || 1;
  episodicMem.push({ key: gs.slice(), norm: n, move, verdict, reached, id: episodeCounter++, replay });
  if (episodicMem.length > EPISODIC_CAP) episodicMem.shift();
}

// UP=0 DOWN=1 LEFT=2 RIGHT=3
const DELTA: [number, number][] = [
  [-1, 0],
  [1, 0],
  [0, -1],
  [0, 1],
];

// BFS flood from the goal: distance-to-goal for every open cell (Infinity for
// walls/unreachable). Used to pick the optimal next step and to judge whether
// the brain's predicted move makes progress.
function goalDist(grid: number[], gr: number, gc: number): number[] {
  const dist = new Array<number>(SIZE * SIZE).fill(Infinity);
  dist[gr * SIZE + gc] = 0;
  const q: [number, number][] = [[gr, gc]];
  for (let h = 0; h < q.length; h++) {
    const [r, c] = q[h];
    for (const [dr, dc] of DELTA) {
      const nr = r + dr;
      const nc = c + dc;
      if (nr < 0 || nr >= SIZE || nc < 0 || nc >= SIZE) continue;
      if (grid[nr * SIZE + nc] === 1) continue;
      if (dist[nr * SIZE + nc] !== Infinity) continue;
      dist[nr * SIZE + nc] = dist[r * SIZE + c] + 1;
      q.push([nr, nc]);
    }
  }
  return dist;
}

// Classify a chosen move against the maze (shared by the brain + the VIN):
//  "ok"   — distance-reducing (optimal) step
//  "wall" — walks into a wall / off the board
//  "astray" — legal move, but away from the goal
//  "wait" — the wait token (move ≥ 4) or no move
// `here` is the agent's current BFS distance-to-goal; `dist` is the BFS grid.
function classifyMove(
  move: number,
  grid: number[],
  ar: number,
  ac: number,
  here: number,
  dist: number[],
): "ok" | "wall" | "astray" | "wait" {
  if (move < 0 || move >= 4) return "wait";
  const [pr, pc] = DELTA[move];
  const r2 = ar + pr;
  const c2 = ac + pc;
  const off = r2 < 0 || r2 >= SIZE || c2 < 0 || c2 >= SIZE;
  if (off || grid[r2 * SIZE + c2] === 1) return "wall";
  if (dist[r2 * SIZE + c2] < here) return "ok";
  return "astray";
}

// Decode the brain's PREDICTED ROUTE — its attention targets — from the
// out_dims prediction (route_len × N_DIRECTIONS). Each group of N_DIRECTIONS is
// one move's logits; we argmax each and trace cells forward from the agent,
// stopping at the goal, a wall, the board edge, a wait token, or a revisit.
// This is what the brain INTENDS, multiple steps ahead — not just the next move.
function decodeRoute(
  pred: number[],
  grid: number[],
  ar: number,
  ac: number,
  gr: number,
  gc: number,
): [number, number][] {
  const route: [number, number][] = [];
  let r = ar;
  let c = ac;
  const seen = new Set<number>([r * SIZE + c]);
  const steps = Math.floor(pred.length / N_DIRECTIONS);
  for (let k = 0; k < steps; k++) {
    let best = 0;
    for (let d = 1; d < N_DIRECTIONS; d++) {
      if (pred[k * N_DIRECTIONS + d] > pred[k * N_DIRECTIONS + best]) best = d;
    }
    if (best >= 4) break; // wait token → plan ends
    const [dr, dc] = DELTA[best];
    const nr = r + dr;
    const nc = c + dc;
    if (nr < 0 || nr >= SIZE || nc < 0 || nc >= SIZE) break; // off the board
    if (grid[nr * SIZE + nc] === 1) break; // plan walks into a wall → stop
    if (seen.has(nr * SIZE + nc)) break; // looping
    r = nr;
    c = nc;
    seen.add(r * SIZE + c);
    route.push([r, c]);
    if (r === gr && c === gc) break; // plan reaches the goal
  }
  return route;
}

// Collapse the retina/cortex feature maps into one SIZE×SIZE saliency grid —
// where the visual system is responding — to overlay on the maze. The literal
// "sight" image is skipped (it's just the input, not a learned response).
function visionSaliency(
  vision: RetinaMap[] | null,
  size: number,
): number[] | null {
  if (!vision || !vision.length) return null;
  const out = new Array<number>(size * size).fill(0);
  let used = 0;
  for (const m of vision) {
    if (m.name === "sight") continue;
    const hw = m.h * m.w;
    const cellv = new Array<number>(hw).fill(0);
    let mx = 1e-6;
    for (let i = 0; i < hw; i++) {
      let s = 0;
      for (let ch = 0; ch < m.channels; ch++) {
        const v = m.data[ch * hw + i];
        s += v < 0 ? -v : v;
      }
      cellv[i] = s / m.channels;
      if (cellv[i] > mx) mx = cellv[i];
    }
    // nearest-neighbour upsample the feature map onto the maze grid
    for (let r = 0; r < size; r++) {
      for (let c = 0; c < size; c++) {
        const sr = Math.min(m.h - 1, ((r * m.h) / size) | 0);
        const sc = Math.min(m.w - 1, ((c * m.w) / size) | 0);
        out[r * size + c] += cellv[sr * m.w + sc] / mx;
      }
    }
    used++;
  }
  if (!used) return null;
  for (let i = 0; i < out.length; i++) out[i] /= used;
  return out;
}

// softmax over the first N_DIRECTIONS logits of the last tick — the brain's
// move distribution (used as the baseline + perturbed signal for occlusion).
function moveDist(out: BrainOut): number[] {
  const last = out.ticks[out.ticks.length - 1]?.prediction ?? [];
  const l = [];
  for (let d = 0; d < N_DIRECTIONS; d++) l.push(last[d] ?? 0);
  const mx = Math.max(...l);
  let s = 0;
  const e = l.map((v) => {
    const x = Math.exp(v - mx);
    s += x;
    return x;
  });
  return e.map((x) => x / (s || 1));
}

// OCCLUSION ATTENTION — the real "where the model pays attention": blank a
// cell to neutral grey, re-run the brain, and measure how far its move
// distribution shifts (total-variation distance from the unperturbed move).
// A big shift ⇒ the model was relying on that cell for THIS decision. True
// attribution, not a proxy — but one brain forward per probed cell.
//
// Task 1: this is now throttled + coarse + cached.
//  • throttled — recompute at most every `attnEvery` steps (gated by the
//    caller); otherwise the cached grid is reused (shape is always SIZE²).
//  • coarse — by default only OPEN cells within `attnRadius` of the agent are
//    probed (the agent/goal cells too); far walls almost never move the
//    decision, so we skip them. Skipped cells stay 0 (transparent overlay).
//  • full — every cell is probed (the old O(SIZE²) behaviour) behind a flag.
// The returned grid is always SIZE² and normalized to its strongest cell, so
// the UI overlay is unchanged regardless of mode.
function occlusionAttention(
  grid: number[],
  ar: number,
  ac: number,
  gr: number,
  gc: number,
  baseDist: number[],
  mode: AttnMode,
): number[] | null {
  if (!engine?.run_brain_pixels || mode === "off") return null;
  const n = SIZE * SIZE;
  const sal = new Array<number>(n).fill(0);
  let mx = 1e-6;
  const full = mode === "full";
  for (let r = 0; r < SIZE; r++) {
    for (let c = 0; c < SIZE; c++) {
      const idx = r * SIZE + c;
      if (!full) {
        // coarse: only probe cells the decision plausibly depends on — open
        // cells (plus agent/goal) within a Chebyshev radius of the agent.
        const near =
          Math.max(Math.abs(r - ar), Math.abs(c - ac)) <= attnRadius;
        const isAgent = r === ar && c === ac;
        const isGoal = r === gr && c === gc;
        const open = grid[idx] === 0;
        if (!(near && (open || isAgent || isGoal))) continue;
      }
      try {
        const out = engine.run_brain_pixels(
          renderPixels(grid, ar, ac, gr, gc, idx, pxBufOccl),
        );
        const p = moveDist(out);
        let tv = 0;
        for (let d = 0; d < N_DIRECTIONS; d++) tv += Math.abs(p[d] - baseDist[d]);
        sal[idx] = 0.5 * tv; // total-variation distance ∈ [0,1]
        if (sal[idx] > mx) mx = sal[idx];
      } catch {
        /* leave 0 */
      }
    }
  }
  for (let i = 0; i < n; i++) sal[i] /= mx; // normalize to the strongest cell
  return sal;
}

// Generate a random solvable-by-construction maze via randomized DFS
// (recursive backtracker) — mirrors maze_gen.rs. Walls=1, open=0; start at
// (1,1), goal at (size-2,size-2). The carved grid is a spanning tree, so a BFS
// path always exists (whether the BRAIN finds it is what brainSolves checks).
function genMaze(size: number): {
  grid: number[];
  start: [number, number];
  end: [number, number];
} {
  const grid = new Array<number>(size * size).fill(1);
  grid[1 * size + 1] = 0;
  const stack: [number, number][] = [[1, 1]];
  const dirs: [number, number][] = [
    [-2, 0],
    [2, 0],
    [0, -2],
    [0, 2],
  ];
  while (stack.length) {
    const [r, c] = stack[stack.length - 1];
    const ns: [number, number][] = [];
    for (const [dr, dc] of dirs) {
      const nr = r + dr;
      const nc = c + dc;
      if (nr > 0 && nr < size - 1 && nc > 0 && nc < size - 1 && grid[nr * size + nc] === 1) {
        ns.push([nr, nc]);
      }
    }
    if (ns.length === 0) {
      stack.pop();
      continue;
    }
    const [nr, nc] = ns[(Math.random() * ns.length) | 0];
    grid[(((r + nr) / 2) | 0) * size + (((c + nc) / 2) | 0)] = 0; // carve wall between
    grid[nr * size + nc] = 0;
    stack.push([nr, nc]);
  }
  return { grid, start: [1, 1], end: [size - 2, size - 2] };
}

// Does the BRAIN solve this maze on a greedy rollout? (Used to keep only
// mazes the demo will actually finish, so fresh random mazes always work.)
function brainSolves(grid: number[], end: [number, number]): boolean {
  if (!engine?.run_brain_pixels) return false;
  let ar = 1;
  let ac = 1;
  const budget = SIZE * SIZE * 2;
  for (let i = 0; i < budget; i++) {
    if (ar === end[0] && ac === end[1]) return true;
    const out = engine.run_brain_pixels(renderPixels(grid, ar, ac, end[0], end[1]));
    const move = brainMoveFrom(out);
    if (move >= 4) return false; // wait/stuck
    const [dr, dc] = DELTA[move];
    const nr = ar + dr;
    const nc = ac + dc;
    if (nr < 0 || nr >= SIZE || nc < 0 || nc >= SIZE || grid[nr * SIZE + nc] === 1) return false;
    ar = nr;
    ac = nc;
  }
  return false; // ran out of budget (looping) — treat as unsolved
}

// Read per-region telemetry / exit-gate / certainty from the NEW engine.
// Every getter is feature-detected (set to null at init if absent), and the
// raw shape the engine returns is normalized defensively — we accept either an
// array of region objects or an object with arrays of scalars, and coerce to
// our typed RegionTelemetry[]. Returns undefined if nothing is available, so
// StepResult.telemetry simply stays absent against the old engine.
function readTelemetry(): BrainTelemetry | undefined {
  if (!engine) return undefined;
  const tel: BrainTelemetry = { regions: [] };
  let any = false;

  if (engine.region_telemetry) {
    try {
      const raw = engine.region_telemetry() as unknown;
      const regions: RegionTelemetry[] = [];
      if (Array.isArray(raw)) {
        for (const r of raw) {
          if (r && typeof r === "object") {
            const o = r as Record<string, unknown>;
            const reg: RegionTelemetry = {};
            if (typeof o.name === "string") reg.name = o.name;
            if (typeof o.activity === "number") reg.activity = o.activity;
            if (typeof o.neuromod === "number") reg.neuromod = o.neuromod;
            if (typeof o.certainty === "number") reg.certainty = o.certainty;
            regions.push(reg);
          } else if (typeof r === "number") {
            regions.push({ activity: r }); // bare scalars → activity
          }
        }
      } else if (raw && typeof raw === "object") {
        // object-of-arrays form: { activity:number[], neuromod:number[], ... }
        const o = raw as Record<string, unknown>;
        const act = Array.isArray(o.activity) ? (o.activity as number[]) : null;
        const nm = Array.isArray(o.neuromod) ? (o.neuromod as number[]) : null;
        const cer = Array.isArray(o.certainty) ? (o.certainty as number[]) : null;
        const names = Array.isArray(o.names) ? (o.names as string[]) : null;
        const len = Math.max(
          act?.length ?? 0,
          nm?.length ?? 0,
          cer?.length ?? 0,
          names?.length ?? 0,
        );
        for (let i = 0; i < len; i++) {
          const reg: RegionTelemetry = {};
          if (names && typeof names[i] === "string") reg.name = names[i];
          if (act && typeof act[i] === "number") reg.activity = act[i];
          if (nm && typeof nm[i] === "number") reg.neuromod = nm[i];
          if (cer && typeof cer[i] === "number") reg.certainty = cer[i];
          regions.push(reg);
        }
      }
      if (regions.length) {
        tel.regions = regions;
        any = true;
      }
    } catch {
      /* ignore — leave regions empty */
    }
  }

  if (engine.exit_gate) {
    try {
      const v = engine.exit_gate();
      if (typeof v === "number" && Number.isFinite(v)) {
        tel.exitGate = v;
        any = true;
      }
    } catch {
      /* ignore */
    }
  }
  if (engine.certainty) {
    try {
      const v = engine.certainty();
      if (typeof v === "number" && Number.isFinite(v)) {
        tel.certainty = v;
        any = true;
      }
    } catch {
      /* ignore */
    }
  }

  return any ? tel : undefined;
}

self.onmessage = async (e: MessageEvent) => {
  const msg = e.data;

  try {
    if (msg.type === "init") {
      // dynamic import by absolute URL — resolved at runtime, served from
      // public/engine. The URL is assembled from a variable so the bundler
      // leaves it as a real runtime import instead of trying to resolve it.
      // ?v=… busts any stale cached engine (a mismatched glue/wasm pair fails
      // to instantiate); bump ENGINE_VER whenever the wasm is rebuilt.
      const ENGINE_VER = "20260625v-vin2";
      const enginePath =
        ["", "engine", "modgrad_mini.js"].join("/") + "?v=" + ENGINE_VER;
      // Load the glue as a blob module. Vite's DEV server refuses to serve
      // /public files as importable modules ("can only be referenced via HTML
      // tags"), so a bare import() of enginePath 404s in dev. Fetching the
      // source and importing it via a blob URL works in both dev AND the built
      // app. We pass the wasm URL to init() explicitly, so the glue never relies
      // on its own import.meta.url (which would be the blob: URL) to find it.
      let mod: any;
      try {
        const src = await fetch(enginePath).then((r) => r.text());
        const blobUrl = URL.createObjectURL(
          new Blob([src], { type: "text/javascript" }),
        );
        mod = await import(/* @vite-ignore */ blobUrl);
        URL.revokeObjectURL(blobUrl);
      } catch {
        // last resort (e.g. blob: blocked by CSP): try the direct import
        mod = await import(/* @vite-ignore */ enginePath);
      }
      // pass the wasm URL (also versioned) so init() doesn't fetch a stale .wasm
      await mod.default("/engine/modgrad_mini_bg.wasm?v=" + ENGINE_VER);
      SIZE = msg.size ?? 9;
      ensurePxBufs(); // size the reused pixel buffers for this board

      // ── attention config (Task 1) — caller may override mode / cadence ──
      if (msg.attnMode === "off" || msg.attnMode === "coarse" || msg.attnMode === "full") {
        attnMode = msg.attnMode;
      }
      if (typeof msg.attnEvery === "number" && msg.attnEvery >= 1) {
        attnEvery = Math.floor(msg.attnEvery);
      }
      if (typeof msg.attnRadius === "number" && msg.attnRadius >= 1) {
        attnRadius = Math.floor(msg.attnRadius);
      }
      attnStepCounter = 0;
      attnCache = null;

      // Load the 8-region brain — it is the solver. If the (older) wasm lacks
      // the brain entry points we leave engine null so the page surfaces the
      // error rather than silently doing nothing.
      let runBrainPixels: ((pixels: Float32Array) => BrainOut) | null = null;
      let retinaMaps: ((pixels: Float32Array) => RetinaMap[]) | null = null;
      if (msg.brainWeights && typeof mod.load_brain_weights === "function") {
        mod.load_brain_weights(msg.brainWeights);
        runBrainPixels = mod.run_brain_pixels;
        retinaMaps = typeof mod.retina_maps === "function" ? mod.retina_maps : null;
        // proof the trained weights are loaded AND driving output: run the brain
        // on a blank board and log its logits — a trained net gives a non-uniform
        // response; random/unloaded weights would give a flat ~0 vector.
        try {
          const probe = runBrainPixels!(renderPixels(new Array(SIZE * SIZE).fill(0), 1, 1, SIZE - 2, SIZE - 2));
          const last = probe.ticks[probe.ticks.length - 1]?.prediction ?? [];
          const l = last.slice(0, 5).map((v) => Number(v.toFixed(2)));
          const spread = Math.max(...l) - Math.min(...l);
          console.log(
            `[brain] weights loaded: ${(msg.brainWeights.length / 1e6).toFixed(2)} MB · retina=${retinaMaps ? "yes" : "no"} · probe logits [U,D,L,R,W]=${JSON.stringify(l)} · spread=${spread.toFixed(2)} (≫0 ⇒ trained net responding)`,
          );
        } catch (e) {
          console.warn("[brain] weight self-test failed:", e);
        }
      }
      // ── NEW engine exports (Task 9) — all feature-detected. The worker runs
      // unchanged against the OLD engine where these are absent (each stays
      // null and is simply never called). `plastic` is true only when BOTH
      // plasticity entry points exist.
      const fn = <T,>(name: string): T | null =>
        typeof (mod as Record<string, unknown>)[name] === "function"
          ? ((mod as Record<string, unknown>)[name] as T)
          : null;
      const extras: EngineExtras = {
        apply_plasticity: fn<(chosen: number, signal: number) => number>("apply_plasticity"),
        reset_plasticity: fn<() => void>("reset_plasticity"),
        region_telemetry: fn<() => unknown>("region_telemetry"),
        exit_gate: fn<() => number>("exit_gate"),
        certainty: fn<() => number>("certainty"),
        vin_forward:
          fn<(pixels: Float32Array, agentRow: number, agentCol: number) => unknown>("vin_forward"),
        vin_learn: fn<(targetMove: number, pain: number) => number>("vin_learn"),
        vin_reset: fn<() => void>("vin_reset"),
      };
      const plastic =
        extras.apply_plasticity !== null && extras.reset_plasticity !== null;

      // ── VIN learning loop (feature-detected) ──
      // The closed loop is available only when the engine exports vin_forward.
      // vinMode is ON by default when available; the caller may force it off via
      // an init flag. Against the old engine (no vin_forward) it stays inert.
      vinAvailable = extras.vin_forward !== null;
      vinMode = vinAvailable && msg.vinMode !== false; // default ON when available
      mazeCounter = 0;

      engine = runBrainPixels
        ? { run_brain_pixels: runBrainPixels, retina_maps: retinaMaps, ...extras }
        : null;
      post({
        type: "ready",
        brain: runBrainPixels !== null,
        plastic,
        vin: vinAvailable, // engine exposes the VIN learning loop
        vinMode, // whether it's active for this session
      });
      return;
    }

    if (msg.type === "step") {
      if (!engine) return;
      const { grid, agent, goal } = msg as {
        grid: number[];
        agent: [number, number];
        goal: [number, number];
      };
      const [ar, ac] = agent;
      const [gr, gc] = goal;
      // the brain reads the maze and decides the move (and exposes its trace)
      const ran = runBrain(grid, ar, ac, gr, gc);
      // the visual cortex's per-layer feature maps for this cell (for the vision panel)
      let vision: RetinaMap[] | null = null;
      if (engine.retina_maps) {
        try {
          const px = renderPixels(grid, ar, ac, gr, gc);
          const maps = engine.retina_maps(px);
          // prepend the literal retinal image — the RGB the eye actually sees
          // (what the model is looking at), so the viz shows sight → features.
          const sight: RetinaMap = {
            name: "sight",
            channels: 3,
            h: SIZE,
            w: SIZE,
            data: Array.from(px),
          };
          vision = [sight, ...maps];
        } catch {
          vision = null;
        }
      }

      // The brain's predicted move (0..3, 4 = wait). The 8-region brain is
      // ~84.5% per-move but ~0% end-to-end on 9×9 — one wrong step into a wall
      // would stall it. So the agent walks the SHORTEST PATH to the goal (the
      // demo always completes and keeps thinking), and we report whether the
      // brain AGREED with the optimal step or made a misstep — honest, not faked.
      const brainMove = ran ? ran.move : 4;
      const dist = goalDist(grid, gr, gc); // BFS distance-to-goal per cell
      const here = dist[ar * SIZE + ac];

      // optimal step = open neighbour with the smallest distance-to-goal
      let optMove = -1;
      let bestD = Infinity;
      for (let d = 0; d < 4; d++) {
        const [pr, pc] = DELTA[d];
        const r2 = ar + pr;
        const c2 = ac + pc;
        if (r2 < 0 || r2 >= SIZE || c2 < 0 || c2 >= SIZE) continue;
        if (grid[r2 * SIZE + c2] === 1) continue;
        const dd = dist[r2 * SIZE + c2];
        if (dd < bestD) {
          bestD = dd;
          optMove = d;
        }
      }

      // ── VIN closed loop (feature-detected) ──────────────────────────────
      // When the VIN is active it DRIVES the predicted move + verdict + bars,
      // and we teach it toward the BFS-optimal move (imitation) while penalizing
      // wall moves (pain) via vin_learn's three-factor rule. The brain's own
      // move still feeds the 3D viz, but the reported decision is the VIN's.
      const vinOn = vinMode && vinAvailable && engine?.vin_forward != null;
      const vf = vinOn ? vinForward(grid, ar, ac, gr, gc) : null;
      // The VIN's plan: greedy on its (exact) value field → the correct move,
      // every time. This is what the agent FOLLOWS, so it solves the maze and
      // never walks into a wall. The move-head's own argmax is the LEARNER —
      // it starts wrong and learns to predict this same move (shown in the bars).
      const vinGreedy = vf ? greedyFromValue(vf, ar, ac, grid) : -1;
      const headMove = vf ? argmaxMove(vf.move_logits) : -1; // the learner's guess
      // the move the agent commits: the VIN's value-greedy step (fallback BFS)
      const decidedMove = vf ? (vinGreedy >= 0 ? vinGreedy : optMove) : brainMove;

      // Verdict reflects the LEARNER (move-head) when the VIN drives — so the
      // misstep ✕ and wall-hit sparkline show the move-head's mistakes dropping
      // as it learns. (The agent itself follows the correct value-greedy move.)
      const verdict = classifyMove(vf ? headMove : decidedMove, grid, ar, ac, here, dist);
      // "agreed" = the learner caught up to the VIN's value-greedy plan
      const agreed = vf ? headMove === decidedMove : verdict === "ok";

      // the agent steps along the DECIDED move (the VIN's plan), so it solves.
      const [adr, adc] = decidedMove >= 0 && decidedMove < 4 ? DELTA[decidedMove] : [0, 0];
      let nr2 = ar + adr;
      let nc2 = ac + adc;
      if (nr2 < 0 || nr2 >= SIZE || nc2 < 0 || nc2 >= SIZE || grid[nr2 * SIZE + nc2] === 1) {
        // safety net: never step into a wall (greedy shouldn't, but be safe)
        nr2 = ar;
        nc2 = ac;
      }
      const next: [number, number] = [nr2, nc2];
      const reached = next[0] === gr && next[1] === gc;

      // OCCLUSION ATTENTION: blank each cell, re-run, measure how much the move
      // shifts — where the model actually pays attention for THIS decision.
      // Task 1: throttled (recompute at most every `attnEvery` steps) + coarse;
      // between recomputes we reuse the cached grid (its SIZE² shape is stable).
      let attn: number[] | null = attnCache;
      if (attnMode !== "off" && ran?.pred) {
        const due = attnStepCounter % attnEvery === 0;
        attnStepCounter++;
        if (due) {
          const l = ran.pred.slice(0, N_DIRECTIONS);
          const mxl = Math.max(...l);
          let s = 0;
          const e = l.map((v) => {
            const x = Math.exp(v - mxl);
            s += x;
            return x;
          });
          const baseDist = e.map((x) => x / (s || 1));
          const fresh = occlusionAttention(grid, ar, ac, gr, gc, baseDist, attnMode);
          if (fresh) {
            attnCache = fresh;
            attn = fresh;
          }
        }
      }
      // the brain's predicted route — the cells it's attending to / aiming for
      const route = ran?.pred ? decodeRoute(ran.pred, grid, ar, ac, gr, gc) : [];

      // ── NEW engine telemetry + live plasticity (Task 9) ──
      // Per-region telemetry / exit-gate / certainty (undefined on old engine).
      const telemetry = readTelemetry();

      // pain THIS step: the VIN earns pain (1.0) whenever its own chosen move
      // would hit a wall / leave the board — the verdict above already measures
      // exactly that for the decided move. (Used by vin_learn + dream replay.)
      const pain = verdict === "wall" ? 1.0 : 0.0;

      let plasticDelta: number | undefined;
      let signal: number | undefined;
      if (vinOn && vf && engine?.vin_learn) {
        // VIN LEARNING: imitate the BFS-optimal move AND penalize wall moves in
        // one three-factor update. targetMove = optMove (teach), pain = wall
        // penalty. signal we surface = +teach when learning a real move, -pain
        // dominated when it just walked into a wall (so the panel shows the
        // pain). ‖ΔW‖ drives the Plasticity panel / wall-hit sparkline.
        try {
          // teach toward the VIN's OWN value-greedy plan (not an external
          // oracle) and penalize the learner's wall-hits (pain).
          const delta = engine.vin_learn(decidedMove, pain);
          if (typeof delta === "number" && Number.isFinite(delta)) {
            plasticDelta = delta;
            signal = pain > 0 ? -pain : decidedMove >= 0 ? 1.0 : 0.0;
          }
        } catch {
          /* never let learning crash a step */
        }
      } else if (engine?.apply_plasticity && brainMove >= 0 && brainMove < N_DIRECTIONS) {
        // Non-VIN fallback: the original outcome-driven plasticity on the brain.
        // wall = pain (negative), reached goal = strong reward, optimal "ok" =
        // small reward, otherwise (astray / wait) = mild negative.
        let sig: number;
        if (verdict === "wall") sig = -1.0;
        else if (reached) sig = 1.0;
        else if (verdict === "ok") sig = 0.2;
        else sig = -0.2; // astray / wait
        try {
          const delta = engine.apply_plasticity(brainMove, sig);
          if (typeof delta === "number" && Number.isFinite(delta)) {
            plasticDelta = delta;
            signal = sig;
          }
        } catch {
          /* never let plasticity crash a step */
        }
      }

      // TRAINING LOSS: cross-entropy of the move-head's softmax against the move
      // it should take (the VIN's value-greedy plan, else the BFS-optimal step).
      // This is the curve that drops as the head learns — high when it disagrees,
      // ~0 once it predicts the right move with confidence. lr = the θ used by the
      // three-factor rule (constant; surfaced so the panel can show it).
      let loss: number | undefined;
      const lossTarget = vf ? decidedMove : optMove;
      const lossLogits = vf ? vf.move_logits : ran?.moveLogits;
      if (lossLogits && lossTarget >= 0 && lossTarget < lossLogits.length) {
        const mx = Math.max(...lossLogits);
        let sum = 0;
        for (const v of lossLogits) sum += Math.exp(v - mx);
        const pTarget = Math.exp(lossLogits[lossTarget] - mx) / (sum || 1);
        loss = -Math.log(Math.max(pTarget, 1e-6));
      }
      const lr = vinOn ? VIN_LR : plasticDelta != null ? PLASTIC_LR : undefined;

      // EPISODIC MEMORY: recall the nearest PAST situation (before storing the
      // current one, so it can't match itself), then store this decision. In VIN
      // mode we attach the maze snapshot so the sleep/replay pass can re-derive
      // its target/pain and re-teach offline (esp. wall-hit "nightmares").
      const gs = ran?.gs ?? [];
      const episodic = episodicRecall(gs);
      const replay: Episode["replay"] | undefined = vinOn
        ? { grid: grid.slice(), ar, ac, gr, gc, optMove, pain }
        : undefined;
      episodicStore(gs, decidedMove, verdict, reached, replay);

      const result: StepResult = {
        type: "step",
        agent: next,
        // bars highlight the LEARNER's pick (move-head) when the VIN drives, so
        // you watch it converge to the plan; else the brain's predicted move.
        move: vf ? headMove : decidedMove,
        verdict, // ok / wall / astray / wait — for the misstep indicator
        agreed,
        // move bars: VIN's move logits when active, else the brain's last tick.
        moveLogits: vf
          ? [vf.move_logits[0], vf.move_logits[1], vf.move_logits[2], vf.move_logits[3]]
          : ran
            ? ran.moveLogits
            : [0, 0, 0, 0],
        brain: ran ? ran.trace : null,
        vision,
        attn,
        route,
        done: reached,
        reached,
        telemetry,
        plasticDelta,
        signal,
        loss,
        lr,
        episodic,
        vinActive: vinOn && vf != null,
      };
      post(result);
      return;
    }

    if (msg.type === "resetPlasticity") {
      // Task 9c: wipe any accumulated plastic weight changes. Feature-detected
      // — a no-op against the old engine. Reset the attention throttle too so
      // the next step recomputes a fresh saliency grid.
      episodicMem = []; // forget recalled situations too — a clean slate
      episodeCounter = 0;
      mazeCounter = 0; // restart the dream/sleep cadence
      if (engine?.reset_plasticity) {
        try {
          engine.reset_plasticity();
        } catch {
          /* ignore */
        }
      }
      // also wipe the VIN's learned weights so the loop starts from scratch
      if (engine?.vin_reset) {
        try {
          engine.vin_reset();
        } catch {
          /* ignore */
        }
      }
      attnStepCounter = 0;
      attnCache = null;
      post({ type: "plasticityReset" });
      return;
    }

    if (msg.type === "newMaze") {
      if (!engine) return;
      // A randomized-DFS maze is solvable by construction, so we just emit one.
      // (We used to curate for mazes the brain solves end-to-end, but it solves
      // ~0% at 9×9 — curating only burned CPU. The agent now walks the shortest
      // path while the brain predicts alongside, so any maze works.)
      const m = genMaze(SIZE);
      // fresh maze → drop stale attention so the next step recomputes promptly
      attnStepCounter = 0;
      attnCache = null;

      // ── DREAM / SLEEP REPLAY (VIN) ──
      // Every DREAM_EVERY mazes, run a short offline consolidation pass over
      // stored episodes (wall-hit nightmares first) before handing out the new
      // maze. Bounded by DREAM_MAX_REPLAYS so it never stalls the loop.
      mazeCounter++;
      let dreaming = false;
      let replays = 0;
      if (vinMode && vinAvailable && mazeCounter % DREAM_EVERY === 0) {
        replays = dreamReplay(DREAM_MAX_REPLAYS);
        dreaming = replays > 0;
        if (dreaming) post({ type: "slept", replays });
      }

      post({ type: "maze", grid: m.grid, start: m.start, end: m.end, dreaming, replays });
      return;
    }
  } catch (err) {
    post({ type: "error", message: err instanceof Error ? err.message : String(err) });
  }
};
