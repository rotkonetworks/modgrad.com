/// <reference lib="webworker" />
// modgrad CTM worker — runs the real wasm brain off the main thread.
// The main thread only renders; all forward passes happen here.
//
// The engine package is served from /engine/ (copied from engine/pkg into
// public/engine). We import it by absolute URL so Vite leaves it alone and the
// browser loads it at runtime — keeping the ~1MB of wasm + weights off every
// other route.

type Maze = {
  grid: number[];
  start: [number, number];
  end: [number, number];
  path_length: number;
};

type Tick = {
  // raw move logits, UP=0 DOWN=1 LEFT=2 RIGHT=3
  predictions: number[];
  // [entropy, 1 - entropy]; index 1 is "confidence"
  certainty: number;
  // [d_model] neuron pool
  activations: number[];
  // [n_tokens] attention averaged over heads, normalised 0..1
  attention: number[];
};

type StepResult = {
  type: "step";
  agent: [number, number];
  move: number; // committed move index
  commitTick: number;
  ticks: Tick[];
  done: boolean;
  reached: boolean;
};

let engine: {
  encode_maze_js: (
    grid: Uint8Array,
    size: number,
    ar: number,
    ac: number,
    gr: number,
    gc: number,
  ) => Float32Array;
  run: (obs: Float32Array, nTokens: number, rawDim: number) => RawOut;
} | null = null;

type RawOut = {
  predictions: number[][];
  certainties: number[][];
  activations: number[][];
  attention: number[][][]; // [tick][head][cell]
};

let SIZE = 7;
let RAW_DIM = 9;

const post = (m: unknown) => (self as unknown as Worker).postMessage(m);

function argmax(a: number[]): number {
  let bi = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[bi]) bi = i;
  return bi;
}

// average attention over heads, then min-max normalise to 0..1 for display
function flattenAttention(perHead: number[][]): number[] {
  const cells = perHead[0]?.length ?? 0;
  const avg = new Array<number>(cells).fill(0);
  for (const head of perHead) for (let i = 0; i < cells; i++) avg[i] += head[i];
  for (let i = 0; i < cells; i++) avg[i] /= perHead.length || 1;
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of avg) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  const span = hi - lo || 1;
  return avg.map((v) => (v - lo) / span);
}

function thinkOnce(
  grid: Uint8Array,
  ar: number,
  ac: number,
  gr: number,
  gc: number,
): { ticks: Tick[]; move: number; commitTick: number } {
  const obs = engine!.encode_maze_js(grid, SIZE, ar, ac, gr, gc);
  const out = engine!.run(obs, SIZE * SIZE, RAW_DIM);

  // commit tick = tick with the highest confidence (certainties[t][1])
  let commitTick = 0;
  for (let t = 1; t < out.certainties.length; t++) {
    if (out.certainties[t][1] > out.certainties[commitTick][1]) commitTick = t;
  }

  const ticks: Tick[] = out.predictions.map((pred, t) => ({
    predictions: Array.from(pred),
    certainty: out.certainties[t][1],
    activations: Array.from(out.activations[t]),
    attention: flattenAttention(out.attention[t]),
  }));

  return { ticks, move: argmax(out.predictions[commitTick]), commitTick };
}

// UP=0 DOWN=1 LEFT=2 RIGHT=3
const DELTA: [number, number][] = [
  [-1, 0],
  [1, 0],
  [0, -1],
  [0, 1],
];

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
  const g = Uint8Array.from(grid);
  let ar = 1;
  let ac = 1;
  const budget = SIZE * SIZE * 2;
  for (let i = 0; i < budget; i++) {
    if (ar === end[0] && ac === end[1]) return true;
    const { move } = thinkOnce(g, ar, ac, end[0], end[1]);
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
      const enginePath = ["", "engine", "modgrad_mini.js"].join("/");
      const mod: any = await import(/* @vite-ignore */ enginePath);
      await mod.default(); // init() — fetches /engine/modgrad_mini_bg.wasm
      mod.load_weights(msg.weights);
      engine = { encode_maze_js: mod.encode_maze_js, run: mod.run };
      SIZE = msg.size ?? 7;
      RAW_DIM = msg.rawDim ?? 9;
      post({ type: "ready" });
      return;
    }

    if (msg.type === "step") {
      if (!engine) return;
      const { grid, agent, goal } = msg as {
        grid: number[];
        agent: [number, number];
        goal: [number, number];
      };
      const g = Uint8Array.from(grid);
      const [ar, ac] = agent;
      const [gr, gc] = goal;
      const { ticks, move, commitTick } = thinkOnce(g, ar, ac, gr, gc);

      const [dr, dc] = DELTA[move];
      const nr = ar + dr;
      const nc = ac + dc;

      const inBounds = nr >= 0 && nr < SIZE && nc >= 0 && nc < SIZE;
      const isWall = inBounds && grid[nr * SIZE + nc] === 1;
      // the committed cell (only applied by the main thread after the animation)
      const next: [number, number] =
        inBounds && !isWall ? [nr, nc] : [ar, ac];
      const reached = next[0] === gr && next[1] === gc;
      const blocked = !inBounds || isWall;

      const result: StepResult = {
        type: "step",
        agent: next,
        move,
        commitTick,
        ticks,
        done: reached || blocked,
        reached,
      };
      post(result);
      return;
    }

    if (msg.type === "newMaze") {
      if (!engine) return;
      // Generate fresh random mazes until the brain can solve one (it solves
      // ~60% of random mazes, so this finds one in ~1-2 tries). Cap the tries
      // so we always answer; the cap is effectively never hit.
      let m = genMaze(SIZE);
      for (let attempt = 0; attempt < 150 && !brainSolves(m.grid, m.end); attempt++) {
        m = genMaze(SIZE);
      }
      post({ type: "maze", grid: m.grid, start: m.start, end: m.end });
      return;
    }
  } catch (err) {
    post({ type: "error", message: err instanceof Error ? err.message : String(err) });
  }
};
