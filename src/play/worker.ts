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
};

// one visual-cortex layer's feature maps (CHW), from the wasm `retina_maps`
type RetinaMap = { name: string; channels: number; h: number; w: number; data: number[] };

let engine: {
  run_brain_pixels: (pixels: Float32Array) => BrainOut;
  retina_maps: ((pixels: Float32Array) => RetinaMap[]) | null;
} | null = null;

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

const post = (m: unknown) => (self as unknown as Worker).postMessage(m);

function rms(a: number[]): number {
  if (a.length === 0) return 0;
  let s = 0;
  for (const v of a) s += v * v;
  return Math.sqrt(s / a.length);
}

// Render the maze to RGB pixels [3 × SIZE × SIZE], CHW (R plane, G plane, B
// plane), EXACTLY the scheme run_brain's retina was trained on:
// wall=(0,0,0), open=(1,1,1), agent=(1,0,0) red, goal=(0,1,0) green.
// Agent/goal overwrite the cell colour. (Must match render_maze in the SDK.)
function renderPixels(
  grid: number[],
  ar: number,
  ac: number,
  gr: number,
  gc: number,
  maskIdx = -1, // if ≥0, blank that cell to neutral grey (for occlusion saliency)
): Float32Array {
  const n = SIZE * SIZE;
  const px = new Float32Array(3 * n);
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
): { trace: BrainTrace; move: number; moveLogits: number[]; pred: number[] } | null {
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
    // the full prediction is the brain's whole intended ROUTE (route_len × 5),
    // not just the first move — we decode it into target cells for the maze.
    return { trace: { ticks, ticksUsed: out.ticks_used }, move, moveLogits, pred: last };
  } catch {
    return null; // never let a brain hiccup hard-crash the demo
  }
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

// OCCLUSION ATTENTION — the real "where the model pays attention": blank each
// cell to neutral grey, re-run the brain, and measure how far its move
// distribution shifts (total-variation distance from the unperturbed move).
// A big shift ⇒ the model was relying on that cell for THIS decision. One brain
// forward per cell (SIZE² of them) — heavy, but it's true attribution, not a
// proxy. Returns a normalized [SIZE²] grid (or null if the engine is absent).
function occlusionAttention(
  grid: number[],
  ar: number,
  ac: number,
  gr: number,
  gc: number,
  baseDist: number[],
): number[] | null {
  if (!engine?.run_brain_pixels) return null;
  const n = SIZE * SIZE;
  const sal = new Array<number>(n).fill(0);
  let mx = 1e-6;
  for (let idx = 0; idx < n; idx++) {
    try {
      const out = engine.run_brain_pixels(renderPixels(grid, ar, ac, gr, gc, idx));
      const p = moveDist(out);
      let tv = 0;
      for (let d = 0; d < N_DIRECTIONS; d++) tv += Math.abs(p[d] - baseDist[d]);
      sal[idx] = 0.5 * tv; // total-variation distance ∈ [0,1]
      if (sal[idx] > mx) mx = sal[idx];
    } catch {
      /* leave 0 */
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

self.onmessage = async (e: MessageEvent) => {
  const msg = e.data;

  try {
    if (msg.type === "init") {
      // dynamic import by absolute URL — resolved at runtime, served from
      // public/engine. The URL is assembled from a variable so the bundler
      // leaves it as a real runtime import instead of trying to resolve it.
      // ?v=… busts any stale cached engine (a mismatched glue/wasm pair fails
      // to instantiate); bump ENGINE_VER whenever the wasm is rebuilt.
      const ENGINE_VER = "20260624v";
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
      engine = runBrainPixels
        ? { run_brain_pixels: runBrainPixels, retina_maps: retinaMaps }
        : null;
      post({ type: "ready", brain: runBrainPixels !== null });
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

      // classify the brain's prediction against the maze:
      //  "ok"   — brain picked a distance-reducing (optimal) step
      //  "wall" — brain wanted to walk into a wall / off the board
      //  "astray" — legal move, but away from the goal (would get lost)
      //  "wait" — brain emitted the wait token
      let verdict: "ok" | "wall" | "astray" | "wait" = "wait";
      if (brainMove >= 0 && brainMove < 4) {
        const [pr, pc] = DELTA[brainMove];
        const r2 = ar + pr;
        const c2 = ac + pc;
        const off = r2 < 0 || r2 >= SIZE || c2 < 0 || c2 >= SIZE;
        if (off || grid[r2 * SIZE + c2] === 1) verdict = "wall";
        else if (dist[r2 * SIZE + c2] < here) verdict = "ok";
        else verdict = "astray";
      }
      const agreed = verdict === "ok";

      // the agent takes the optimal step regardless, so it reaches the goal
      const [adr, adc] = optMove >= 0 ? DELTA[optMove] : [0, 0];
      const next: [number, number] =
        optMove >= 0 ? [ar + adr, ac + adc] : [ar, ac];
      const reached = next[0] === gr && next[1] === gc;

      // OCCLUSION ATTENTION: blank each cell, re-run, measure how much the move
      // shifts — where the model actually pays attention for THIS decision.
      let attn: number[] | null = null;
      if (ran?.pred) {
        const l = ran.pred.slice(0, N_DIRECTIONS);
        const mxl = Math.max(...l);
        let s = 0;
        const e = l.map((v) => {
          const x = Math.exp(v - mxl);
          s += x;
          return x;
        });
        const baseDist = e.map((x) => x / (s || 1));
        attn = occlusionAttention(grid, ar, ac, gr, gc, baseDist);
      }
      // the brain's predicted route — the cells it's attending to / aiming for
      const route = ran?.pred ? decodeRoute(ran.pred, grid, ar, ac, gr, gc) : [];

      const result: StepResult = {
        type: "step",
        agent: next,
        move: brainMove, // the brain's actual prediction (drives the move bars)
        verdict, // ok / wall / astray / wait — for the misstep indicator
        agreed,
        moveLogits: ran ? ran.moveLogits : [0, 0, 0, 0],
        brain: ran ? ran.trace : null,
        vision,
        attn,
        route,
        done: reached,
        reached,
      };
      post(result);
      return;
    }

    if (msg.type === "newMaze") {
      if (!engine) return;
      // A randomized-DFS maze is solvable by construction, so we just emit one.
      // (We used to curate for mazes the brain solves end-to-end, but it solves
      // ~0% at 9×9 — curating only burned CPU. The agent now walks the shortest
      // path while the brain predicts alongside, so any maze works.)
      const m = genMaze(SIZE);
      post({ type: "maze", grid: m.grid, start: m.start, end: m.end });
      return;
    }
  } catch (err) {
    post({ type: "error", message: err instanceof Error ? err.message : String(err) });
  }
};
