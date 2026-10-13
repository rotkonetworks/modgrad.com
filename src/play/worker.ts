/// <reference lib="webworker" />
// modgrad brain worker — runs the real wasm 8-region CTM brain off the main
// thread. The main thread only renders; every forward pass happens here.
//
// What this demo does, and nothing more: for one maze, render it to RGB pixels
// (agent at the start, goal in the far corner), run the brain once, and send
// back what it computed — its per-tick region activity (for the 3D view), its
// retina → V4 feature maps, and the 20-step route it predicts at every internal
// tick. Nothing else plans, decides or moves an agent. The true shortest route
// is computed here too, but ONLY as a reference line for the page to draw next
// to the brain's prediction; it is never shown to the brain.
//
// The engine package is served from /engine/ (built from the modgrad SDK's
// `modgrad-wasm` crate into public/engine). We import it by absolute URL so Vite
// leaves it alone and the browser loads it at runtime.

// ── types (mirrored in Play.tsx) ─────────────────────────────────────────────

/** One brain tick for the 3D view: every neuron's activation, grouped by region. */
type BrainTick = {
  acts: number[][];
  global: number; // RMS of the global-sync vector
  exit: number | null; // outer exit-gate λ, if the brain has one
};
type BrainTrace = { ticks: BrainTick[]; ticksUsed: number };
type RetinaMap = { name: string; channels: number; h: number; w: number; data: number[] };
type Cell = [number, number];

/** The brain's route as decoded from one tick's prediction. */
type TickRoute = {
  cells: Cell[]; // cells the predicted moves visit, start excluded
  /** how the decoded route ends: reached the goal, stepped into a wall / off
   *  the board (that cell is the last entry of `cells`), emitted the wait
   *  token, or used up all route_len moves. */
  end: "goal" | "wall" | "wait" | "length";
};

type ThinkResult = {
  type: "maze";
  grid: number[];
  start: Cell;
  end: Cell;
  shortest: Cell[]; // reference only: the unique shortest route, start excluded
  routes: TickRoute[]; // one decoded route per internal tick, in order
  /** index into `shortest` of the first predicted step that differs from it on
   *  the FINAL tick's route; equals shortest.length when the route is correct. */
  agree: number;
  /** per tick: the 5 first-step logits [U, D, L, R, wait] of the prediction */
  firstSteps: number[][];
  brain: BrainTrace | null;
  vision: RetinaMap[] | null;
};

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

type Engine = {
  run_brain_pixels: (pixels: Float32Array) => BrainOut;
  retina_maps: ((pixels: Float32Array) => RetinaMap[]) | null;
  load_brain_weights: (json: string) => void;
};

// ── constants ────────────────────────────────────────────────────────────────

// The brain's visual cortex is built for a FIXED 9×9 input: every published
// export carries `input_h = input_w = 9`, and the SDK's retina reads exactly
// 3·9·9 pixels. A smaller board under-fills that buffer (wasm panics with
// "unreachable"); a larger one is silently read as the wrong pixels. So the
// demo only ever runs 9×9.
const SIZE = 9;
// UP=0 DOWN=1 LEFT=2 RIGHT=3 WAIT=4 — the SDK's DIR_* order. The prediction is
// route_len groups of these 5 logits (out_dims = 20 × 5 = 100).
const N_DIRECTIONS = 5;
const DELTA: Cell[] = [
  [-1, 0],
  [1, 0],
  [0, -1],
  [0, 1],
];

let engine: Engine | null = null;
const post = (m: unknown) => (self as unknown as Worker).postMessage(m);

// ── helpers ──────────────────────────────────────────────────────────────────

function rms(a: number[]): number {
  if (!a.length) return 0;
  let s = 0;
  for (const v of a) s += v * v;
  return Math.sqrt(s / a.length);
}

// Render the maze to RGB pixels [3 × SIZE × SIZE], CHW, exactly the scheme the
// brain's retina was trained on: wall=(0,0,0), open=(1,1,1), agent=(1,0,0),
// goal=(0,1,0). (Must match render_maze in the SDK.)
const pxBuf = new Float32Array(3 * SIZE * SIZE);
function renderPixels(grid: number[], ar: number, ac: number, gr: number, gc: number): Float32Array {
  const n = SIZE * SIZE;
  for (let r = 0; r < SIZE; r++) {
    for (let c = 0; c < SIZE; c++) {
      const idx = r * SIZE + c;
      let rr: number, gg: number, bb: number;
      if (r === ar && c === ac) [rr, gg, bb] = [1, 0, 0];
      else if (r === gr && c === gc) [rr, gg, bb] = [0, 1, 0];
      else if (grid[idx] !== 0) [rr, gg, bb] = [0, 0, 0];
      else [rr, gg, bb] = [1, 1, 1];
      pxBuf[idx] = rr;
      pxBuf[n + idx] = gg;
      pxBuf[2 * n + idx] = bb;
    }
  }
  return pxBuf;
}

// Random maze by randomized DFS (recursive backtracker), mirroring the SDK's
// maze_gen.rs: walls=1, open=0, start (1,1), goal (SIZE-2, SIZE-2). The carved
// cells form a spanning tree, so the shortest route is unique.
function genMaze(): { grid: number[]; start: Cell; end: Cell } {
  const grid = new Array<number>(SIZE * SIZE).fill(1);
  grid[1 * SIZE + 1] = 0;
  const stack: Cell[] = [[1, 1]];
  const dirs: Cell[] = [
    [-2, 0],
    [2, 0],
    [0, -2],
    [0, 2],
  ];
  while (stack.length) {
    const [r, c] = stack[stack.length - 1];
    const ns: Cell[] = [];
    for (const [dr, dc] of dirs) {
      const nr = r + dr;
      const nc = c + dc;
      if (nr > 0 && nr < SIZE - 1 && nc > 0 && nc < SIZE - 1 && grid[nr * SIZE + nc] === 1) {
        ns.push([nr, nc]);
      }
    }
    if (ns.length === 0) {
      stack.pop();
      continue;
    }
    const [nr, nc] = ns[(Math.random() * ns.length) | 0];
    grid[(((r + nr) / 2) | 0) * SIZE + (((c + nc) / 2) | 0)] = 0;
    grid[nr * SIZE + nc] = 0;
    stack.push([nr, nc]);
  }
  return { grid, start: [1, 1], end: [SIZE - 2, SIZE - 2] };
}

// Reference only: the shortest route start → goal by BFS (start excluded).
function shortestRoute(grid: number[], start: Cell, end: Cell): Cell[] {
  const prev = new Map<number, number>([[start[0] * SIZE + start[1], -1]]);
  const q: Cell[] = [start];
  for (let h = 0; h < q.length; h++) {
    const [r, c] = q[h];
    for (const [dr, dc] of DELTA) {
      const nr = r + dr;
      const nc = c + dc;
      const k = nr * SIZE + nc;
      if (nr < 0 || nr >= SIZE || nc < 0 || nc >= SIZE || grid[k] === 1 || prev.has(k)) continue;
      prev.set(k, r * SIZE + c);
      q.push([nr, nc]);
    }
  }
  const goal = end[0] * SIZE + end[1];
  if (!prev.has(goal)) return [];
  const out: Cell[] = [];
  for (let k = goal; k !== start[0] * SIZE + start[1]; k = prev.get(k)!) {
    out.push([Math.floor(k / SIZE), k % SIZE]);
  }
  return out.reverse();
}

// Decode one tick's prediction into the cells its moves visit from the start:
// argmax over each group of 5 logits. Stops at the goal, the wait token, or a
// step into a wall / off the board (that cell is kept so the page can mark it).
// Revisits are kept as predicted — nothing is corrected.
function decodeRoute(pred: number[], grid: number[], start: Cell, end: Cell): TickRoute {
  const cells: Cell[] = [];
  let [r, c] = start;
  const steps = Math.floor(pred.length / N_DIRECTIONS);
  for (let k = 0; k < steps; k++) {
    let best = 0;
    for (let d = 1; d < N_DIRECTIONS; d++) {
      if (pred[k * N_DIRECTIONS + d] > pred[k * N_DIRECTIONS + best]) best = d;
    }
    if (best >= 4) return { cells, end: "wait" };
    r += DELTA[best][0];
    c += DELTA[best][1];
    const off = r < 0 || r >= SIZE || c < 0 || c >= SIZE;
    cells.push([Math.max(0, Math.min(SIZE - 1, r)), Math.max(0, Math.min(SIZE - 1, c))]);
    if (off || grid[r * SIZE + c] === 1) return { cells, end: "wall" };
    if (r === end[0] && c === end[1]) return { cells, end: "goal" };
  }
  return { cells, end: "length" };
}

// One maze, one look: run the brain on the start image and package its output.
function think(): ThinkResult {
  const { grid, start, end } = genMaze();
  const shortest = shortestRoute(grid, start, end);
  let brain: BrainTrace | null = null;
  let routes: TickRoute[] = [];
  let firstSteps: number[][] = [];
  let vision: RetinaMap[] | null = null;
  if (engine) {
    const px = renderPixels(grid, start[0], start[1], end[0], end[1]);
    const out = engine.run_brain_pixels(px);
    brain = {
      ticks: out.ticks.map((t) => ({
        acts: t.region_activations,
        global: rms(t.global_sync),
        exit: t.exit_lambda,
      })),
      ticksUsed: out.ticks_used,
    };
    routes = out.ticks.map((t) => decodeRoute(t.prediction, grid, start, end));
    firstSteps = out.ticks.map((t) => t.prediction.slice(0, N_DIRECTIONS));
    if (engine.retina_maps) {
      try {
        const maps = engine.retina_maps(px);
        // prepend the literal image the retina receives, so the 3D view shows
        // sight → retina → V1 → V2 → V4.
        vision = [{ name: "sight", channels: 3, h: SIZE, w: SIZE, data: Array.from(px) }, ...maps];
      } catch {
        vision = null;
      }
    }
  }
  const final = routes[routes.length - 1]?.cells ?? [];
  let agree = 0;
  while (
    agree < shortest.length &&
    agree < final.length &&
    final[agree][0] === shortest[agree][0] &&
    final[agree][1] === shortest[agree][1]
  ) {
    agree++;
  }
  return { type: "maze", grid, start, end, shortest, routes, agree, firstSteps, brain, vision };
}

// ── engine loading ───────────────────────────────────────────────────────────

// ?v= busts any stale cached engine (a mismatched glue/wasm pair fails to
// instantiate); bump whenever the wasm in public/engine is rebuilt.
const ENGINE_VER = "20260630-m3-distributed";

async function loadEngineModule(): Promise<{ mod: any; threads: number }> {
  const v = "?v=" + ENGINE_VER;
  // Multithreaded engine when the page is cross-origin isolated
  // (SharedArrayBuffer available); otherwise the single-threaded one.
  const canThread =
    typeof crossOriginIsolated !== "undefined" &&
    crossOriginIsolated === true &&
    typeof SharedArrayBuffer !== "undefined";

  // The MT glue imports its rayon helper by relative URL and spawns sub-workers
  // from import.meta.url, so it must load from its real /engine URL. The
  // single-thread glue is self-contained, so it goes through a blob import,
  // which also works under Vite dev (which won't serve /public as modules).
  const importGlue = async (name: string, threaded: boolean): Promise<any> => {
    const path = "/engine/" + name + ".js" + v;
    if (threaded) return await import(/* @vite-ignore */ path);
    try {
      const src = await fetch(path).then((r) => r.text());
      const blobUrl = URL.createObjectURL(new Blob([src], { type: "text/javascript" }));
      const m = await import(/* @vite-ignore */ blobUrl);
      URL.revokeObjectURL(blobUrl);
      return m;
    } catch {
      return await import(/* @vite-ignore */ path);
    }
  };

  if (canThread) {
    try {
      const mod = await importGlue("modgrad_wasm_mt", true);
      await mod.default("/engine/modgrad_wasm_mt_bg.wasm" + v);
      let threads = 0;
      const initPool = mod.initThreadPool ?? mod.init_thread_pool;
      if (typeof initPool === "function") {
        const n = Math.max(1, (self.navigator && self.navigator.hardwareConcurrency) || 4);
        // rayon's helper spawns `new Worker(new URL(..., import.meta.url))`;
        // force any non-absolute worker URL onto this origin, then restore.
        const OriginalWorker = globalThis.Worker;
        const origin = self.location.origin + "/";
        globalThis.Worker = class extends OriginalWorker {
          constructor(url: string | URL, options?: WorkerOptions) {
            let u = url instanceof URL ? url.href : String(url);
            if (!u.startsWith(origin) && !u.startsWith("blob:") && !u.startsWith("data:")) {
              u = origin + (u.startsWith("/") ? u.slice(1) : u);
            }
            super(u, options);
          }
        };
        try {
          await initPool(n);
          threads = n;
        } finally {
          globalThis.Worker = OriginalWorker;
        }
      }
      return { mod, threads };
    } catch (err) {
      console.warn("[engine] threaded init failed; falling back to single-thread:", err);
    }
  }
  const mod = await importGlue("modgrad_wasm", false);
  await mod.default("/engine/modgrad_wasm_bg.wasm" + v);
  return { mod, threads: 0 };
}

// ── messages ─────────────────────────────────────────────────────────────────
//   init        {brainWeights}  → ready {brain, threads}
//   reloadBrain {brainWeights}  → brainReloaded {ok}
//   newMaze                     → maze (ThinkResult)

self.onmessage = async (e: MessageEvent) => {
  const msg = e.data;
  try {
    if (msg.type === "init") {
      const { mod, threads } = await loadEngineModule();
      if (msg.brainWeights && typeof mod.load_brain_weights === "function") {
        mod.load_brain_weights(msg.brainWeights);
        engine = {
          run_brain_pixels: mod.run_brain_pixels,
          retina_maps: typeof mod.retina_maps === "function" ? mod.retina_maps : null,
          load_brain_weights: mod.load_brain_weights,
        };
      }
      post({ type: "ready", brain: engine !== null, threads });
      return;
    }

    // Hot-swap the trained brain (size selector) without tearing down the engine.
    if (msg.type === "reloadBrain") {
      if (!engine || !msg.brainWeights) {
        post({ type: "brainReloaded", ok: false });
        return;
      }
      try {
        engine.load_brain_weights(msg.brainWeights);
        post({ type: "brainReloaded", ok: true });
      } catch (err) {
        console.error("[brain] reload failed:", err);
        post({ type: "brainReloaded", ok: false });
      }
      return;
    }

    if (msg.type === "newMaze") {
      if (!engine) return;
      post(think());
      return;
    }
  } catch (err) {
    post({ type: "error", message: err instanceof Error ? err.message : String(err) });
  }
};
