import {
  createSignal,
  createEffect,
  onMount,
  onCleanup,
  Show,
  For,
  batch,
} from "solid-js";
import { A } from "@solidjs/router";
import { useDocMeta } from "@/lib/meta";
import { MOVES, softmax, cssVar } from "@/play/viz";

// ── types mirrored from the worker ────────────────────────────────
type BrainTick = { acts: number[][]; global: number; exit: number | null };
type BrainTrace = { ticks: BrainTick[]; ticksUsed: number };
type RetinaMap = { name: string; channels: number; h: number; w: number; data: number[] };
type StepMsg = {
  type: "step";
  agent: [number, number];
  move: number; // brain's predicted move 0..3 (4 = wait), drives the move bars
  verdict: "ok" | "wall" | "astray" | "wait"; // brain prediction vs the maze
  agreed: boolean; // did the brain match the optimal step?
  moveLogits: number[]; // last tick's 4 direction logits, for the move bars
  brain: BrainTrace | null;
  vision: RetinaMap[] | null;
  attn: number[] | null; // per-cell retina saliency mapped to the maze
  route: [number, number][]; // the brain's predicted route — its attention targets
  done: boolean;
  reached: boolean;
};
// structure of the 8-region brain, read from brain_solver_reference.json
type BrainConn = { from: number[]; to: number; receives_observation: boolean };
type BrainRef = {
  size: number;
  ticks: number;
  out_dims: number;
  region_names: string[];
  regions: { name: string; d_model: number }[];
  connections: BrainConn[];
  n_global_sync: number;
  heldout_first_move_acc: number;
};
type Maze = {
  grid: number[];
  start: [number, number];
  end: [number, number];
  path_length: number;
};

type Status = "loading" | "ready" | "error";
type RunState = "idle" | "thinking" | "solved" | "stuck";

const TICK_MS = 95; // pace of the "thinking" animation
const MAX_STEPS = 30; // safety cap
// UP=0 DOWN=1 LEFT=2 RIGHT=3 — [dr, dc], matches the worker's DELTA
const DIR_DELTA: [number, number][] = [
  [-1, 0],
  [1, 0],
  [0, -1],
  [0, 1],
];

export default function Play() {
  useDocMeta(() => ({
    title: "Watch it think",
    description:
      "Watch a real modgrad 8-region brain see and think in your browser. It reads each maze cell through a visual retina, eight regions compute across 16 ticks, and predicts the move — ~84.5% per-move on held-out 9×9. Run client-side by a bit-exact wasm reimplementation of the modgrad SDK.",
    path: "/play",
  }));

  const reduced =
    typeof matchMedia !== "undefined" &&
    matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ── reactive state ──────────────────────────────────────────────
  const [status, setStatus] = createSignal<Status>("loading");
  const [errMsg, setErrMsg] = createSignal("");
  const [runState, setRunState] = createSignal<RunState>("idle");

  const [currentMaze, setCurrentMaze] = createSignal<Maze | null>(null);
  const [mazeNum, setMazeNum] = createSignal(0);
  const [generating, setGenerating] = createSignal(false);
  const [agent, setAgent] = createSignal<[number, number]>([1, 1]);
  const [visited, setVisited] = createSignal<[number, number][]>([[1, 1]]);
  const [stepCount, setStepCount] = createSignal(0);

  // the brain's per-tick animation cursor (drives the 3D viz reveal)
  const [tickIdx, setTickIdx] = createSignal(0);
  const [ticksTotal, setTicksTotal] = createSignal(0);
  const [committedMove, setCommittedMove] = createSignal(-1);
  const [moveLogits, setMoveLogits] = createSignal<number[]>([0, 0, 0, 0]);

  const [playing, setPlaying] = createSignal(false);

  // ── 8-region brain (the solver) ──
  const [brainRef, setBrainRef] = createSignal<BrainRef | null>(null);
  const [brainOn, setBrainOn] = createSignal(false); // brain wasm available
  const [brainTrace, setBrainTrace] = createSignal<BrainTrace | null>(null);
  // visual-cortex feature maps (retina/V1/V2/V4) for the current cell
  let visionMaps: RetinaMap[] | null = null; // read in the rAF loop, not reactive
  let attnMap: number[] | null = null; // per-cell retina saliency, read in drawMaze
  let routeCells: [number, number][] = []; // brain's predicted route (attention targets)

  // the brain's verdict on its last prediction vs the optimal step, plus a
  // running tally of how often it agreed — honest "it predicts, here's its score"
  const [verdict, setVerdict] = createSignal<StepMsg["verdict"] | null>(null);
  const [agreeCount, setAgreeCount] = createSignal(0);
  const [moveCount, setMoveCount] = createSignal(0);

  // dimensions from the reference; defaults avoid layout shift before load
  let SIZE = 9;

  const maze = () => currentMaze();

  // ── worker plumbing ─────────────────────────────────────────────
  let worker: Worker | null = null;
  let pendingStep = false; // a step request is in flight to the worker
  let stepTimer: number | undefined;
  let playTimer: number | undefined;

  onMount(async () => {
    let reference: BrainRef;
    try {
      const r = await fetch("/models/brain_solver_reference.json");
      reference = await r.json();
    } catch (e) {
      setStatus("error");
      setErrMsg("Could not load the brain reference.");
      return;
    }
    SIZE = reference.size;
    setBrainRef(reference); // also drives the 3D viz structure

    try {
      worker = new Worker(new URL("../play/worker.ts", import.meta.url), {
        type: "module",
      });
    } catch (e) {
      setStatus("error");
      setErrMsg("Web workers are not available in this browser.");
      return;
    }

    worker.onmessage = (ev: MessageEvent) => {
      const msg = ev.data;
      if (msg.type === "ready") {
        setBrainOn(!!msg.brain);
        setStatus("ready");
        // start on a fresh, brain-solvable maze so every visit is different
        requestNewMaze();
        return;
      }
      if (msg.type === "error") {
        setStatus("error");
        setErrMsg(msg.message || "The brain failed to load.");
        return;
      }
      if (msg.type === "maze") {
        batch(() => {
          setGenerating(false);
          setMazeNum((n) => n + 1);
        });
        loadMaze(msg as Maze);
        if (!reduced) startPlaying();
        return;
      }
      if (msg.type === "step") onStepResult(msg as StepMsg);
    };

    try {
      // the 9MB brain weights are the demo's source — the brain is the solver
      const br = await fetch("/models/brain_solver_weights.json").then((r) =>
        r.text(),
      );
      worker.postMessage({
        type: "init",
        brainWeights: br,
        size: reference.size,
      });
    } catch (e) {
      setStatus("error");
      setErrMsg("Could not load the trained brain weights.");
    }
  });

  onCleanup(() => {
    worker?.terminate();
    clearTimeout(stepTimer);
    clearTimeout(playTimer);
    brainCanvas?.removeEventListener("wheel", brainWheel);
    brainIO?.disconnect();
  });

  // ── solve loop ──────────────────────────────────────────────────
  function loadMaze(m: Maze) {
    batch(() => {
      setCurrentMaze(m);
      setAgent([m.start[0], m.start[1]]);
      setVisited([[m.start[0], m.start[1]]]);
      setStepCount(0);
      setBrainTrace(null);
      setTickIdx(0);
      setTicksTotal(0);
      setCommittedMove(-1);
      setMoveLogits([0, 0, 0, 0]);
      setRunState("idle");
    });
  }

  // ask the worker to invent a fresh maze the brain can actually solve
  function requestNewMaze() {
    if (!worker || status() !== "ready") return;
    pause();
    setGenerating(true);
    worker.postMessage({ type: "newMaze" });
  }

  // ask the worker to think about the current cell
  function requestStep() {
    const m = maze();
    if (!m || !worker || pendingStep) return;
    if (stepCount() >= MAX_STEPS) {
      setRunState("stuck");
      setPlaying(false);
      return;
    }
    pendingStep = true;
    setRunState("thinking");
    worker.postMessage({
      type: "step",
      grid: m.grid,
      agent: agent(),
      goal: maze()!.end,
    });
  }

  // the worker returns the brain's full per-tick trace for one move; animate
  // through the ticks, then commit the brain's chosen move.
  function onStepResult(msg: StepMsg) {
    pendingStep = false;
    const nTicks = msg.brain?.ticks.length ?? 0;
    visionMaps = msg.vision;
    attnMap = msg.attn;
    routeCells = msg.route ?? [];
    batch(() => {
      setBrainTrace(msg.brain);
      setTicksTotal(nTicks);
      setCommittedMove(msg.move);
      setMoveLogits(msg.moveLogits);
      setVerdict(msg.verdict);
      setMoveCount((n) => n + 1);
      if (msg.agreed) setAgreeCount((n) => n + 1);
      setTickIdx(0);
    });

    if (reduced || nTicks === 0) {
      // no animation: jump to the last tick, then apply the move
      setTickIdx(Math.max(0, nTicks - 1));
      applyMove(msg);
      return;
    }
    animateTicks(0, msg);
  }

  function animateTicks(i: number, msg: StepMsg) {
    setTickIdx(i);
    const total = msg.brain?.ticks.length ?? 0;
    if (i >= total - 1) {
      // finished the thinking animation → commit
      stepTimer = window.setTimeout(() => applyMove(msg), TICK_MS * 2.4);
      return;
    }
    stepTimer = window.setTimeout(() => animateTicks(i + 1, msg), TICK_MS);
  }

  function applyMove(msg: StepMsg) {
    batch(() => {
      setAgent(msg.agent);
      setVisited((v) => [...v, msg.agent]);
      setStepCount((s) => s + 1);
    });

    if (msg.reached) {
      const looping = playing();
      setRunState("solved");
      // perpetual demo: reached the goal → pause to celebrate, then a fresh maze
      if (looping) {
        playTimer = window.setTimeout(() => requestNewMaze(), TICK_MS * 14);
      } else {
        setPlaying(false);
      }
      return;
    }
    // continue to the next cell if still auto-playing
    if (playing()) {
      playTimer = window.setTimeout(() => requestStep(), TICK_MS * 3);
    } else {
      setRunState("idle");
    }
  }

  // ── controls ────────────────────────────────────────────────────
  function startPlaying() {
    if (status() !== "ready") return;
    if (runState() === "solved" || runState() === "stuck") restart();
    setPlaying(true);
    requestStep();
  }
  function pause() {
    setPlaying(false);
    clearTimeout(stepTimer);
    clearTimeout(playTimer);
    if (runState() === "thinking") setRunState("idle");
  }
  function togglePlay() {
    playing() ? pause() : startPlaying();
  }
  // single manual think+move (used as "Step")
  function stepOnce() {
    if (status() !== "ready" || pendingStep) return;
    if (runState() === "solved" || runState() === "stuck") return;
    setPlaying(false);
    requestStep();
  }
  function restart() {
    pause();
    const m = currentMaze();
    if (m) loadMaze(m);
  }
  function newMaze() {
    requestNewMaze();
  }

  // ── derived display values ──────────────────────────────────────
  // the brain's move distribution over the 4 directions (last tick's logits)
  const moveProbs = (): number[] => {
    const l = moveLogits();
    return l.length === 4 && l.some((v) => v !== 0)
      ? softmax(l)
      : [0.25, 0.25, 0.25, 0.25];
  };

  // ════════════════════════════════════════════════════════════════
  //  canvas rendering
  // ════════════════════════════════════════════════════════════════
  let mazeCanvas!: HTMLCanvasElement;
  let brainCanvas!: HTMLCanvasElement;

  // high-dpi helper
  function ctxOf(c: HTMLCanvasElement, cssW: number, cssH: number) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (c.width !== cssW * dpr || c.height !== cssH * dpr) {
      c.width = cssW * dpr;
      c.height = cssH * dpr;
    }
    const ctx = c.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    return ctx;
  }

  // ── maze + agent + trail + attention overlay ──
  function drawMaze() {
    const m = maze();
    if (!m || !mazeCanvas) return;
    const W = mazeCanvas.clientWidth || 360;
    const H = W; // square
    const ctx = ctxOf(mazeCanvas, W, H);
    const n = SIZE;
    const gap = 3;
    const cell = (W - gap * (n + 1)) / n;

    const wall = cssVar("--text", "#1B1822");
    const open = cssVar("--bg-card", "#fff");
    const line = cssVar("--line-2", "#D8D2C3");
    const accent = cssVar("--accent", "#6243D9");

    const cx = (c: number) => gap + c * (cell + gap);
    const cy = (r: number) => gap + r * (cell + gap);

    // base cells
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        const isWall = m.grid[r * n + c] === 1;
        ctx.fillStyle = isWall ? wall : open;
        roundRect(ctx, cx(c), cy(r), cell, cell, 4);
        ctx.fill();
        if (!isWall) {
          ctx.strokeStyle = line;
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }
    }

    // ── ATTENTION HEATMAP (occlusion attribution): blank each cell, re-run the
    // brain, measure how much the move shifts. Brighter = the model relied on
    // that cell for this decision. Real attribution, every cell (walls too). ──
    const am = attnMap;
    if (am) {
      for (let r = 0; r < n; r++) {
        for (let c = 0; c < n; c++) {
          const a = Math.pow(Math.max(0, Math.min(1, am[r * n + c])), 1.25);
          if (a < 0.06) continue;
          // amber heat — distinct from the purple trail and teal route
          ctx.fillStyle = `rgba(245, 158, 11, ${(0.12 + 0.6 * a).toFixed(3)})`;
          roundRect(ctx, cx(c) + 1, cy(r) + 1, cell - 2, cell - 2, 4);
          ctx.fill();
        }
      }
    }

    // visited trail
    const path = visited();
    ctx.fillStyle = `color-mix(in srgb, ${accent} 22%, transparent)`;
    for (const [r, c] of path) {
      roundRect(ctx, cx(c), cy(r), cell, cell, 4);
      ctx.fill();
    }
    // trail line
    if (path.length > 1) {
      ctx.strokeStyle = accent;
      ctx.globalAlpha = 0.55;
      ctx.lineWidth = Math.max(2, cell * 0.12);
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.beginPath();
      path.forEach(([r, c], i) => {
        const x = cx(c) + cell / 2;
        const y = cy(r) + cell / 2;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // start marker (ring)
    marker(ctx, cx(m.start[1]), cy(m.start[0]), cell, "ring", line, accent);
    // goal marker (filled accent diamond/star-ish)
    goalMark(ctx, cx(m.end[1]) + cell / 2, cy(m.end[0]) + cell / 2, cell, accent);

    // agent
    const [ar, ac] = agent();

    // ── attention targets: the brain's PREDICTED ROUTE, decoded from its
    // multi-step prediction — the cells it's aiming for, several moves ahead.
    // Teal crosshairs + a beam from the agent; fades into the future. This is
    // the brain's plan (vs the purple trail = where the agent actually went). ──
    const rc = routeCells;
    if (rc.length) {
      const TARGET = "#13b7a4";
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.strokeStyle = `color-mix(in srgb, ${TARGET} 50%, transparent)`;
      ctx.lineWidth = Math.max(2, cell * 0.09);
      ctx.setLineDash([cell * 0.16, cell * 0.16]);
      ctx.beginPath();
      ctx.moveTo(cx(ac) + cell / 2, cy(ar) + cell / 2);
      for (const [r, c] of rc) ctx.lineTo(cx(c) + cell / 2, cy(r) + cell / 2);
      ctx.stroke();
      ctx.setLineDash([]);
      rc.forEach(([r, c], i) => {
        const fade = 1 - i / (rc.length + 1);
        const px = cx(c) + cell / 2;
        const py = cy(r) + cell / 2;
        ctx.strokeStyle = `color-mix(in srgb, ${TARGET} ${Math.round(30 + 55 * fade)}%, transparent)`;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(px, py, cell * 0.19, 0, Math.PI * 2);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(px - cell * 0.1, py);
        ctx.lineTo(px + cell * 0.1, py);
        ctx.moveTo(px, py - cell * 0.1);
        ctx.lineTo(px, py + cell * 0.1);
        ctx.stroke();
      });
    }

    const ax = cx(ac) + cell / 2;
    const ay = cy(ar) + cell / 2;
    ctx.beginPath();
    ctx.fillStyle = accent;
    ctx.arc(ax, ay, cell * 0.3, 0, Math.PI * 2);
    ctx.fill();
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = cssVar("--bg-card", "#fff");
    ctx.stroke();
    // subtle pulse ring while thinking
    if (runState() === "thinking") {
      ctx.beginPath();
      ctx.strokeStyle = `color-mix(in srgb, ${accent} 60%, transparent)`;
      ctx.lineWidth = 2;
      const pulse = 0.34 + (tickIdx() / Math.max(1, ticksTotal())) * 0.18;
      ctx.arc(ax, ay, cell * pulse, 0, Math.PI * 2);
      ctx.stroke();
    }

    // ── brain misstep indicator: when the brain's PREDICTED move would walk
    // into a wall (or away from the goal), flag it on the cell it wanted — the
    // agent still steps optimally, but you see where the brain got it wrong. ──
    const v = verdict();
    const pm = committedMove();
    if (v && v !== "ok" && pm >= 0 && pm < 4) {
      const [pr, pc] = DIR_DELTA[pm];
      const tr = ar + pr;
      const tc = ac + pc;
      const bad = "#e0564e";
      // arrow from the agent toward the brain's (wrong) choice
      const mx2 = ax + pc * cell * 0.5;
      const my2 = ay + pr * cell * 0.5;
      ctx.strokeStyle = bad;
      ctx.lineWidth = 2.5;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(mx2, my2);
      ctx.stroke();
      // mark the target cell it wanted (wall = ✕ box, astray = dashed ring)
      if (tr >= 0 && tr < n && tc >= 0 && tc < n) {
        ctx.strokeStyle = bad;
        ctx.lineWidth = 2;
        const ix = cx(tc);
        const iy = cy(tr);
        if (v === "wall" && m.grid[tr * n + tc] === 1) {
          ctx.beginPath();
          ctx.moveTo(ix + cell * 0.3, iy + cell * 0.3);
          ctx.lineTo(ix + cell * 0.7, iy + cell * 0.7);
          ctx.moveTo(ix + cell * 0.7, iy + cell * 0.3);
          ctx.lineTo(ix + cell * 0.3, iy + cell * 0.7);
          ctx.stroke();
        } else {
          ctx.setLineDash([4, 3]);
          roundRect(ctx, ix + 2, iy + 2, cell - 4, cell - 4, 4);
          ctx.stroke();
          ctx.setLineDash([]);
        }
      }
    }
  }

  // ── the 8-region brain as a rotating 3D particle cloud (the solver) ──
  // Mirrors the modgrad debugger's 3D brain (debugger/src/main.rs): each neuron
  // is a point, clustered by region and coloured by region; brightness + size =
  // its activation, so individual neurons spike as the brain thinks. Hand-rolled
  // 3D→2D projection, no WebGL.
  const REGION_COLORS = [
    "#4488ff", "#44ff88", "#ff8844", "#ff4488",
    "#8844ff", "#88ff44", "#ff88ff", "#ffff44",
  ];
  const shortRegion = (n: string): string =>
    ({
      input: "input",
      attention: "attn",
      output: "output",
      motor: "motor",
      cerebellum: "cereb",
      basal_ganglia: "basal",
      insula: "insula",
      hippocampus: "hippo",
    })[n] ?? n;
  type NeuronPos = { x: number; y: number; z: number; region: number; index: number };
  let neuronPos: NeuronPos[] = [];
  let builtFor = -1; // total neuron count the layout was built for

  // deterministic [-1,1) RNG (mulberry32) so the layout is stable across frames
  function makeRand(seed: number) {
    let s = seed >>> 0;
    return () => {
      s = (s + 0x6d2b79f5) >>> 0;
      let t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return (((t ^ (t >>> 14)) >>> 0) / 4294967296) * 2 - 1;
    };
  }
  // Anatomical-ish placement keyed by region name (robust to index order):
  // a brain-shaped egg — cortical lobes around the outside (frontal→occipital,
  // motor/parietal on top), subcortical structures deep in the centre, and the
  // cerebellum tucked at the back-bottom. [x=front..back, y=down..up, z=depth].
  const BRAIN_ANATOMY: Record<string, [number, number, number]> = {
    input: [3.7, 1.2, 0.5], // frontal
    attention: [0.6, 3.0, 0.7], // parietal (top)
    output: [-3.9, 0.9, -0.5], // occipital (back)
    motor: [1.4, 2.9, -0.7], // motor cortex (top)
    basal_ganglia: [0.4, 0.1, 0.2], // deep centre
    hippocampus: [-1.3, -0.5, 0.9], // deep, central-back
    insula: [1.3, -0.3, -1.3], // lateral, deep
    cerebellum: [-3.7, -2.7, 0.1], // back-bottom
  };
  let regionCenters: { x: number; y: number; z: number }[] = [];
  function buildNeuronPositions(regions: { name: string; d_model: number }[]) {
    neuronPos = [];
    regionCenters = [];
    const rand = makeRand(42);
    for (let ri = 0; ri < regions.length; ri++) {
      const neurons = regions[ri].d_model;
      const [cx, cy, cz] = BRAIN_ANATOMY[regions[ri].name] ?? [
        (ri % 4) * 3 - 4.5,
        1.5 - Math.floor(ri / 4) * 3,
        0,
      ];
      regionCenters[ri] = { x: cx, y: cy, z: cz };
      const spread = Math.sqrt(neurons) * 0.36;
      for (let ni = 0; ni < neurons; ni++) {
        const theta = rand() * Math.PI;
        const phi = rand() * Math.PI * 2;
        const rr = spread * (0.3 + 0.7 * Math.abs(rand()));
        neuronPos.push({
          x: cx + rr * Math.sin(theta) * Math.cos(phi),
          y: cy + rr * Math.cos(theta),
          z: cz + rr * Math.sin(theta) * Math.sin(phi),
          region: ri,
          index: ni,
        });
      }
    }
    builtFor = neuronPos.length;
  }
  let brainRotY = 0.7;
  let brainRotX = -0.32;
  let brainZoom = 1; // wheel / pinch zoom
  let brainDragging = false;
  let brainPointers = new Map<number, { x: number; y: number }>(); // active pointers (pinch)
  let pinchDist = 0;
  const [brainAutoRotate, setBrainAutoRotate] = createSignal(true);
  // co-spike line persistence: 0 = instant, 1 = long trail (per-frame decay)
  const [spikeHold, setSpikeHold] = createSignal(0.55);
  const edgeGlow = new Map<string, number>(); // decayed co-spike per edge
  let brainTime = 0; // seconds, advanced by the rAF loop for the idle shimmer
  let brainVisible = true; // gated by IntersectionObserver — skip drawing off-screen
  let brainIO: IntersectionObserver | undefined;
  // "#rrggbb" + alpha → rgba() string, for additive glow gradients
  const hexA = (hex: string, a: number): string => {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  };

  // brain tick aligned to the current animation tick (held at the last brain
  // tick if the brain exited earlier than the solver's tick count).
  const curBrainTick = (): BrainTick | null => {
    const bt = brainTrace();
    if (!bt || bt.ticks.length === 0) return null;
    return bt.ticks[Math.min(tickIdx(), bt.ticks.length - 1)];
  };

  // per-region live telemetry at the current tick: neuron count + activation RMS
  // (bar is normalized to the loudest region so they're comparable).
  const regionStats = () => {
    const bref = brainRef();
    if (!bref) return [];
    const acts = curBrainTick()?.acts ?? [];
    const stats = bref.regions.map((rg, i) => {
      const a = acts[i] ?? [];
      let s = 0;
      for (const v of a) s += v * v;
      const rms = a.length ? Math.sqrt(s / a.length) : 0;
      let peak = 0;
      for (const v of a) peak = Math.max(peak, Math.abs(v));
      return { name: rg.name, d: rg.d_model, rms, peak };
    });
    const mx = Math.max(1e-6, ...stats.map((s) => s.rms));
    return stats.map((s) => ({ ...s, level: Math.min(1, s.rms / mx) }));
  };

  function drawBrain3D() {
    const bref = brainRef();
    if (!brainCanvas || !bref) return;
    const dmodels = bref.regions.map((r) => r.d_model);
    const total = dmodels.reduce((a, b) => a + b, 0);
    if (builtFor !== total) buildNeuronPositions(bref.regions);

    const W = brainCanvas.clientWidth || 720;
    const H = Math.round(Math.max(340, Math.min(520, W * 0.6)));
    const ctx = ctxOf(brainCanvas, W, H);
    const cxp = W / 2;
    const cyp = H / 2;
    const fov = W * 0.72 * brainZoom;

    // dark "brain-scan" backdrop — bright region colours only read on dark.
    const bg = ctx.createRadialGradient(cxp, cyp * 0.9, 8, cxp, cyp, Math.max(W, H) * 0.75);
    bg.addColorStop(0, "#181527");
    bg.addColorStop(1, "#0a0912");
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, W, H);

    const tick = curBrainTick();
    // per-step normaliser so spike brightness is stable in scale
    const bt = brainTrace();
    let amax = 1e-6;
    if (bt)
      for (const t of bt.ticks)
        for (const ra of t.acts)
          for (const v of ra) {
            const a = v < 0 ? -v : v;
            if (a > amax) amax = a;
          }
    const actOf = (region: number, index: number) => {
      if (tick) {
        const v = tick.acts[region]?.[index] ?? 0;
        return Math.min(1, (v < 0 ? -v : v) / amax);
      }
      // idle: gentle per-neuron shimmer so the brain looks alive at rest
      return 0.1 + 0.14 * (0.5 + 0.5 * Math.sin(brainTime * 1.7 + region * 1.3 + index * 0.7));
    };

    const sy = Math.sin(brainRotY);
    const cy = Math.cos(brainRotY);
    const sx = Math.sin(brainRotX);
    const cx = Math.cos(brainRotX);
    const pts: { x: number; y: number; depth: number; region: number; a: number }[] = [];
    for (const np of neuronPos) {
      const rx = np.x * cy + np.z * sy;
      const ry = np.y * cx - (-np.x * sy + np.z * cy) * sx;
      const rz = np.y * sx + (-np.x * sy + np.z * cy) * cx;
      const depth = rz + 11;
      if (!(depth >= 0.5)) continue; // also rejects NaN
      const a = actOf(np.region, np.index);
      pts.push({
        x: cxp + (rx * fov) / depth,
        y: cyp - (ry * fov) / depth,
        depth,
        region: np.region,
        a: Number.isFinite(a) ? a : 0,
      });
    }
    pts.sort((p, q) => q.depth - p.depth); // painter's algorithm: far first

    // per-region activity (mean |act|) for the co-spike connections
    const regionAct: number[] = [];
    for (let r = 0; r < dmodels.length; r++) {
      const ra = tick?.acts[r];
      if (ra && ra.length) {
        let s = 0;
        for (const v of ra) s += v < 0 ? -v : v;
        regionAct[r] = Math.min(1, s / ra.length / amax);
      } else {
        regionAct[r] = 0.06;
      }
    }
    // project the region centres (same camera) for the connectome endpoints
    const centerProj = regionCenters.map((p) => {
      const rx = p.x * cy + p.z * sy;
      const ry = p.y * cx - (-p.x * sy + p.z * cy) * sx;
      const rz = p.y * sx + (-p.x * sy + p.z * cy) * cx;
      const depth = rz + 11;
      return { x: cxp + (rx * fov) / depth, y: cyp - (ry * fov) / depth };
    });

    ctx.globalCompositeOperation = "lighter";

    // ── connectome edges: faint as structure, bright when both ends co-spike,
    // then fading over a configurable trail (spikeHold → per-frame decay). ──
    const decay = 0.86 + spikeHold() * 0.135; // 0.86 (brief) … ~0.995 (long)
    ctx.lineCap = "round";
    for (const conn of bref.connections) {
      const tp = centerProj[conn.to];
      if (!tp) continue;
      const colT = REGION_COLORS[conn.to % REGION_COLORS.length];
      for (const f of conn.from) {
        const fp = centerProj[f];
        if (!fp) continue;
        const co = Math.min(1, regionAct[f] * regionAct[conn.to] * 2.4); // co-spike
        const key = f + "_" + conn.to;
        const g = Math.max(co, (edgeGlow.get(key) ?? 0) * decay); // persistence
        edgeGlow.set(key, g);
        const colF = REGION_COLORS[f % REGION_COLORS.length];
        const grad = ctx.createLinearGradient(fp.x, fp.y, tp.x, tp.y);
        grad.addColorStop(0, hexA(colF, 0.07 + 0.85 * g));
        grad.addColorStop(1, hexA(colT, 0.07 + 0.85 * g));
        ctx.strokeStyle = grad;
        ctx.lineWidth = 0.7 + 4 * g;
        ctx.beginPath();
        ctx.moveTo(fp.x, fp.y);
        ctx.lineTo(tp.x, tp.y);
        ctx.stroke();
        // a spark travelling source→destination while the link is hot
        if (g > 0.1) {
          const fr = (brainTime * 0.55 + f * 0.17) % 1;
          ctx.fillStyle = hexA("#ffffff", Math.min(0.95, g));
          ctx.beginPath();
          ctx.arc(fp.x + (tp.x - fp.x) * fr, fp.y + (tp.y - fp.y) * fr, 1.4 + 2.6 * g, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    // additive blending → overlapping neurons bloom, spikes pop
    for (const p of pts) {
      const col = REGION_COLORS[p.region % REGION_COLORS.length];
      const fogv = Math.max(0.4, Math.min(1, 14 / p.depth - 0.25)); // nearer = brighter/bigger
      const size = Math.max(1.6, (15 / p.depth) * (1.8 + 4 * p.a) * fogv);
      // glow halo only for neurons bright enough to show one (skips ~all the
      // dim/idle dots — avoids ~160 radial gradients/frame for invisible glow)
      if (p.a > 0.18) {
        const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, size * 3.4);
        g.addColorStop(0, hexA(col, (0.12 + 0.6 * p.a) * fogv));
        g.addColorStop(1, hexA(col, 0));
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(p.x, p.y, size * 3.4, 0, Math.PI * 2);
        ctx.fill();
      }
      // bright core
      ctx.fillStyle = hexA(col, Math.min(1, (0.5 + 0.5 * p.a) * fogv));
      ctx.beginPath();
      ctx.arc(p.x, p.y, size, 0, Math.PI * 2);
      ctx.fill();
    }

    // ── visual cortex: retina → V1 → V2 → V4 layers feeding the input region.
    // Each layer is a spatial plane (channels averaged), lit by what it sees,
    // wired forward into the brain — the eyes-to-cognition pathway, connected. ──
    if (visionMaps && visionMaps.length) {
      const proj = (px: number, py: number, pz: number) => {
        const rx = px * cy + pz * sy;
        const ry = py * cx - (-px * sy + pz * cy) * sx;
        const rz = py * sx + (-px * sy + pz * cy) * cx;
        const depth = rz + 11;
        return { x: cxp + (rx * fov) / depth, y: cyp - (ry * fov) / depth, depth };
      };
      const VIS = "#62e6ff"; // cyan — distinct from the region colours
      const inIdx = bref.region_names.indexOf("input");
      const ic = regionCenters[inIdx >= 0 ? inIdx : 0] ?? { x: 3.7, y: 1.2, z: 0.5 };
      const nL = visionMaps.length;
      const SPC = 0.6; // layer-to-layer spacing (was 1.05 — too spread out)
      const centers: { x: number; y: number; act: number }[] = [];
      for (let li = 0; li < nL; li++) {
        const m = visionMaps[li];
        const isSight = m.name === "sight";
        // sight (the eye's image) farthest out; retina→V4 step in toward input
        const lx = ic.x + 1.0 + (nL - 1 - li) * SPC;
        const ly = ic.y + 0.4;
        const lz = ic.z;
        const hw = m.h * m.w;
        const sp = isSight ? 0.28 : 0.2;

        if (isSight) {
          // ── the literal retinal image — what the model actually sees ──
          ctx.globalCompositeOperation = "source-over";
          // a faint frame so it reads as the eye's "screen"
          const corner = (rr: number, cc: number) =>
            proj(lx, ly + (rr - (m.h - 1) / 2) * sp, lz + (cc - (m.w - 1) / 2) * sp);
          const c0 = corner(-0.6, -0.6), c1 = corner(-0.6, m.w - 0.4),
            c2 = corner(m.h - 0.4, m.w - 0.4), c3 = corner(m.h - 0.4, -0.6);
          ctx.fillStyle = "rgba(10,12,20,0.72)";
          ctx.beginPath();
          ctx.moveTo(c0.x, c0.y); ctx.lineTo(c1.x, c1.y);
          ctx.lineTo(c2.x, c2.y); ctx.lineTo(c3.x, c3.y); ctx.closePath();
          ctx.fill();
          ctx.strokeStyle = hexA(VIS, 0.5);
          ctx.lineWidth = 1.2;
          ctx.stroke();
          for (let r = 0; r < m.h; r++) {
            for (let c = 0; c < m.w; c++) {
              const i = r * m.w + c;
              const R = m.data[i], G = m.data[hw + i], B = m.data[2 * hw + i];
              let col: string;
              if (R > 0.5 && G < 0.5 && B < 0.5) col = "#ff5b5b"; // agent
              else if (G > 0.5 && R < 0.5) col = "#4fd17a"; // goal
              else if (R > 0.5 && G > 0.5 && B > 0.5) col = "#e9edf6"; // open
              else col = "#33384a"; // wall
              const p = proj(lx, ly + (r - (m.h - 1) / 2) * sp, lz + (c - (m.w - 1) / 2) * sp);
              if (!(p.depth >= 0.5)) continue;
              const s = Math.max(1.5, (sp * fov) / p.depth) * 0.92;
              ctx.fillStyle = col;
              ctx.fillRect(p.x - s / 2, p.y - s / 2, s, s);
            }
          }
          ctx.globalCompositeOperation = "lighter";
          const cp = proj(lx, ly, lz);
          centers.push({ x: cp.x, y: cp.y, act: 1 });
          continue;
        }

        // ── learned feature map (retina/V1/V2/V4): cyan response field ──
        ctx.globalCompositeOperation = "lighter";
        const cell = new Array(hw).fill(0);
        let mx = 1e-6;
        for (let i = 0; i < hw; i++) {
          let s = 0;
          for (let ch = 0; ch < m.channels; ch++) {
            const v = m.data[ch * hw + i];
            s += v < 0 ? -v : v;
          }
          cell[i] = s / m.channels;
          if (cell[i] > mx) mx = cell[i];
        }
        let lact = 0;
        for (let r = 0; r < m.h; r++) {
          for (let c = 0; c < m.w; c++) {
            const a = Math.min(1, cell[r * m.w + c] / mx);
            lact += a;
            const p = proj(lx, ly + (r - (m.h - 1) / 2) * sp, lz + (c - (m.w - 1) / 2) * sp);
            if (!(p.depth >= 0.5)) continue;
            const size = Math.max(1, (8 / p.depth) * (0.9 + 2.0 * a));
            if (a > 0.35) {
              const gr = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, size * 2.6);
              gr.addColorStop(0, hexA(VIS, 0.4 * a));
              gr.addColorStop(1, hexA(VIS, 0));
              ctx.fillStyle = gr;
              ctx.beginPath();
              ctx.arc(p.x, p.y, size * 2.6, 0, Math.PI * 2);
              ctx.fill();
            }
            ctx.fillStyle = hexA(VIS, 0.22 + 0.72 * a);
            ctx.beginPath();
            ctx.arc(p.x, p.y, size, 0, Math.PI * 2);
            ctx.fill();
          }
        }
        const cp = proj(lx, ly, lz);
        centers.push({ x: cp.x, y: cp.y, act: Math.min(1, lact / hw) });
      }
      // feedforward pathway: retina → V1 → V2 → V4 → input region
      const inP = proj(ic.x, ic.y, ic.z);
      ctx.lineCap = "round";
      for (let li = 0; li < centers.length; li++) {
        const a = centers[li];
        const b = li + 1 < centers.length ? centers[li + 1] : inP;
        const g = 0.12 + 0.7 * a.act;
        ctx.strokeStyle = hexA(VIS, g * 0.65);
        ctx.lineWidth = 0.8 + 2.4 * a.act;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
        const fr = (brainTime * 0.5 + li * 0.15) % 1; // info flowing inward
        ctx.fillStyle = hexA("#ffffff", 0.45 * a.act);
        ctx.beginPath();
        ctx.arc(a.x + (b.x - a.x) * fr, a.y + (b.y - a.y) * fr, 1.1 + 1.8 * a.act, 0, Math.PI * 2);
        ctx.fill();
      }
      // label the pathway as the visual system (crisp, not additive)
      if (centers.length) {
        ctx.globalCompositeOperation = "source-over";
        ctx.font = "600 9px ui-monospace, monospace";
        ctx.textAlign = "center";
        ctx.fillStyle = hexA(VIS, 0.85);
        ctx.fillText("what it sees", centers[0].x, centers[0].y - 18);
        ctx.textAlign = "left";
      }
    }
    ctx.globalCompositeOperation = "source-over";
  }

  // global-sync magnitude at the current tick, for the header meter
  const brainGlobal = () => {
    const t = curBrainTick();
    const bt = brainTrace();
    if (!t || !bt) return 0;
    let gmax = 1e-6;
    for (const x of bt.ticks) if (x.global > gmax) gmax = x.global;
    return Math.min(1, t.global / gmax);
  };

  // ── mouse / touch orbit + zoom on the brain ──
  const brainPointerDown = (e: PointerEvent) => {
    (e.currentTarget as HTMLElement).setPointerCapture?.(e.pointerId);
    brainPointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    brainDragging = true;
    setBrainAutoRotate(false); // grabbing it stops the auto-spin
    if (brainPointers.size === 2) {
      const [a, b] = [...brainPointers.values()];
      pinchDist = Math.hypot(a.x - b.x, a.y - b.y);
    }
  };
  const brainPointerMove = (e: PointerEvent) => {
    const prev = brainPointers.get(e.pointerId);
    if (!prev) return;
    if (brainPointers.size >= 2) {
      brainPointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      const [a, b] = [...brainPointers.values()];
      const d = Math.hypot(a.x - b.x, a.y - b.y);
      if (pinchDist) brainZoom = Math.max(0.5, Math.min(2.6, brainZoom * (d / pinchDist)));
      pinchDist = d;
      return;
    }
    brainRotY += (e.clientX - prev.x) * 0.01;
    brainRotX = Math.max(-1.35, Math.min(1.35, brainRotX + (e.clientY - prev.y) * 0.01));
    brainPointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  };
  const brainPointerUp = (e: PointerEvent) => {
    brainPointers.delete(e.pointerId);
    if (brainPointers.size === 0) brainDragging = false;
    pinchDist = 0;
  };
  const brainWheel = (e: WheelEvent) => {
    e.preventDefault();
    brainZoom = Math.max(0.5, Math.min(2.6, brainZoom * (1 - e.deltaY * 0.0012)));
  };

  // redraw whenever the relevant signals change
  createEffect(() => {
    // dependencies: maze, agent, visited, tick, runState
    currentMaze();
    agent();
    visited();
    tickIdx();
    runState();
    drawMaze();
  });
  // The brain cloud is GPU-heavy (additive gradients, ~160 neurons + edges per
  // frame), so we only animate it while the brain is actively thinking — plus a
  // ~1s settle after each move — and cap to ~30fps. Idle just holds the last
  // frame. This keeps the loop alive (rotation/IO stay responsive) without
  // burning the GPU on a static scene.
  let drewOnce = false;
  onMount(() => {
    let raf = 0;
    let lastDraw = 0;
    let lastThinking = -1e9; // timestamp we were last "thinking", for the settle
    const SETTLE_MS = 1000;
    const MIN_FRAME_MS = 33; // ~30fps cap
    const spin = (now: number) => {
      raf = requestAnimationFrame(spin);
      if (!brainVisible) return; // off-screen → don't burn frames
      const thinking = runState() === "thinking";
      if (thinking) lastThinking = now;
      // active = thinking, or within the settle window, or the user is orbiting,
      // or we haven't drawn a single frame yet (so the idle cloud appears once).
      const active =
        thinking ||
        now - lastThinking < SETTLE_MS ||
        brainDragging ||
        !drewOnce;
      if (!active) return; // idle → hold the last drawn frame
      if (drewOnce && now - lastDraw < MIN_FRAME_MS) return; // 30fps cap
      lastDraw = now;
      drewOnce = true;
      brainTime += 0.033;
      if (!reduced && brainAutoRotate() && !brainDragging) brainRotY += 0.009;
      drawBrain3D();
    };
    raf = requestAnimationFrame(spin);
    onCleanup(() => cancelAnimationFrame(raf));
  });

  // when a fresh brain trace arrives, re-arm a one-shot redraw so the idle
  // cloud reflects the new run even outside the thinking window.
  createEffect(() => {
    brainTrace();
    drewOnce = false;
  });

  // redraw on resize (canvas is fluid width)
  onMount(() => {
    const onResize = () => {
      drawMaze();
    };
    window.addEventListener("resize", onResize);
    onCleanup(() => window.removeEventListener("resize", onResize));
  });

  // ── status line copy ──
  const statusLabel = () => {
    if (generating()) return "new maze…";
    switch (runState()) {
      case "thinking":
        return "thinking…";
      case "solved":
        return "solved";
      case "stuck":
        return "stopped";
      default:
        return "ready";
    }
  };

  return (
    <div class="container-page py-14 sm:py-16">
      <style>{`
        @keyframes nd-float {
          0%,100% { transform: translateY(0) rotate(0deg); }
          50%     { transform: translateY(-12px) rotate(10deg); }
        }
        @keyframes nd-halo {
          0%,100% { opacity: .18; transform: scale(.7); }
          50%     { opacity: .5;  transform: scale(1.35); }
        }
        @keyframes nd-dots {
          0%   { content: ""; }
          25%  { content: "·"; }
          50%  { content: "··"; }
          75%,100% { content: "···"; }
        }
        @keyframes nd-sweep {
          0%   { transform: translateX(-120%); }
          100% { transform: translateX(360%); }
        }
        .nd-glyph { display:inline-block; animation: nd-float 2.4s ease-in-out infinite; }
        .nd-halo  { animation: nd-halo 2.4s ease-in-out infinite; }
        .nd-dots::after { content: ""; animation: nd-dots 1.4s steps(1) infinite; }
        .nd-sweep { animation: nd-sweep 1.3s ease-in-out infinite; }
        @media (prefers-reduced-motion: reduce) {
          .nd-glyph,.nd-halo,.nd-dots::after,.nd-sweep { animation: none; }
          .nd-dots::after { content: "···"; }
        }
      `}</style>
      {/* ── intro ─────────────────────────────────────── */}
      <div class="max-w-[760px]">
        <div class="eyebrow mb-3">Live demo · runs in your browser</div>
        <h1 class="text-[clamp(2rem,5vw,3rem)] tracking-[-0.03em] leading-[1.08]">
          Watch it <span class="grad-text">think.</span>
        </h1>
        <p class="mt-5 text-dim text-[1.05rem] max-w-[60ch] leading-relaxed">
          A real modgrad{" "}
          <span class="grad-text">8-region brain</span> sees and thinks, live in
          your browser. It reads each cell through a visual retina, eight regions
          compute across 16 ticks, and the motor region predicts the move —{" "}
          ~84.5% right per move on held-out 9×9. The agent walks the goal path so
          you can watch the brain see and decide at every step; the maze flags
          where its call would miss. Forward pass: a bit-exact in-browser
          reimplementation of the SDK.
        </p>
        <div class="mt-4 flex flex-wrap gap-x-5 gap-y-1 text-sm">
          <A href="/docs/brain-composition" class="text-accent">
            The 8-region brain →
          </A>
          <A href="/docs/continuous-thought-machine" class="text-accent">
            How a region thinks →
          </A>
        </div>
      </div>

      {/* ── loading / error states ────────────────────── */}
      <Show when={status() === "error"}>
        <div class="card mt-8 max-w-[560px]">
          <div class="eyebrow mb-2 text-warn">Could not start</div>
          <p class="text-dim text-sm leading-relaxed">{errMsg()}</p>
        </div>
      </Show>

      {/* ── solver (the maze) + brain, side by side, one view ───── */}
      <div class="grid lg:grid-cols-[minmax(0,1fr)_minmax(0,1.12fr)] gap-5 mt-8 items-start">
        {/* LEFT: maze hero + controls + move bars */}
        <div class="flex flex-col gap-5">
          <div class="card relative">
            <div class="flex items-center justify-between mb-3">
              <div class="eyebrow">The maze</div>
              <Show when={status() === "ready" || status() === "loading"}>
                <span
                  class="tag"
                  classList={{
                    "tag-live": runState() === "solved",
                    "tag-wip": runState() === "thinking" || generating(),
                  }}
                >
                  {statusLabel()}
                </span>
              </Show>
            </div>

            {/* fixed-ratio box so the canvas never reflows */}
            <div class="relative w-full" style={{ "aspect-ratio": "1 / 1" }}>
              <canvas
                ref={mazeCanvas}
                class="absolute inset-0 w-full h-full"
                aria-label="maze the brain is solving"
              />
              <Show when={status() === "loading"}>
                <div
                  class="absolute inset-0 grid place-items-center"
                  style={{ background: "var(--bg-card)" }}
                >
                  <div class="text-center">
                    <div class="relative mx-auto mb-5 grid place-items-center" style={{ width: "72px", height: "72px" }}>
                      <span
                        class="nd-halo absolute rounded-full"
                        style={{
                          width: "64px",
                          height: "64px",
                          background:
                            "radial-gradient(closest-side, color-mix(in srgb, var(--accent) 45%, transparent), transparent)",
                        }}
                      />
                      <span class="nd-glyph grad-text font-mono" style={{ "font-size": "2.6rem", "line-height": "1" }}>
                        ∇
                      </span>
                    </div>
                    <div class="text-dim text-sm font-mono">
                      warming up the brain<span class="nd-dots" />
                    </div>
                    <div
                      class="relative mt-4 mx-auto overflow-hidden rounded-full"
                      style={{ width: "180px", height: "3px", background: "var(--bg-2)" }}
                    >
                      <div
                        class="nd-sweep absolute inset-y-0 rounded-full"
                        style={{ width: "40%", background: "var(--accent)" }}
                      />
                    </div>
                    <div class="text-mute text-[.7rem] font-mono mt-3">
                      loading the 8-region brain · ~9 MB · runs on your device
                    </div>
                  </div>
                </div>
              </Show>
            </div>

            {/* legend */}
            <div class="flex flex-wrap gap-x-4 gap-y-1.5 mt-3 text-[.72rem] text-mute font-mono">
              <span class="inline-flex items-center gap-1.5">
                <i class="w-2.5 h-2.5 rounded-full inline-block" style={{ background: "var(--accent)" }} />
                agent
              </span>
              <span class="inline-flex items-center gap-1.5">
                <i class="w-2.5 h-2.5 inline-block" style={{ background: "var(--text)", "border-radius": "2px" }} />
                wall
              </span>
              <span class="inline-flex items-center gap-1.5">
                <i class="w-2.5 h-2.5 inline-block" style={{ background: "#3CB371", "border-radius": "2px" }} />
                goal
              </span>
              <span class="inline-flex items-center gap-1.5" title="the cells the brain's multi-step prediction is aiming for">
                <i class="w-2.5 h-2.5 rounded-full inline-block" style={{ border: "2px solid #13b7a4" }} />
                brain's target route
              </span>
              <span class="inline-flex items-center gap-1.5" title="occlusion attribution: hide a cell, re-run the brain — brighter = the model relied on it for this move">
                <i class="w-2.5 h-2.5 inline-block" style={{ background: "rgba(245,158,11,0.7)", "border-radius": "2px" }} />
                attention
              </span>
            </div>
          </div>

          {/* controls */}
          <div class="card">
            <div class="flex flex-wrap gap-2">
              <button
                class="btn btn-primary flex-1 justify-center min-w-[110px]"
                onClick={togglePlay}
                disabled={status() !== "ready"}
              >
                {playing() ? "❚❚ Pause" : "▶ Play"}
              </button>
              <button
                class="btn flex-1 justify-center min-w-[90px]"
                onClick={stepOnce}
                disabled={status() !== "ready" || playing()}
                title="think through one move"
              >
                Step
              </button>
              <button
                class="btn flex-1 justify-center min-w-[90px]"
                onClick={restart}
                disabled={status() !== "ready"}
              >
                Restart
              </button>
              <button
                class="btn flex-1 justify-center min-w-[100px]"
                onClick={newMaze}
                disabled={status() !== "ready"}
              >
                New maze
              </button>
            </div>

            <div class="flex items-center justify-between mt-4 pt-4 border-t border-line">
              <div class="text-sm text-dim">
                The agent walks the goal path; the brain predicts each step.
              </div>
              <div class="font-mono text-xs text-mute tabular-nums">
                maze #{Math.max(1, mazeNum())} · step {stepCount()}
              </div>
            </div>
          </div>

          {/* move bars */}
          <div class="card">
            <div class="flex items-center justify-between mb-3">
              <div class="flex items-center gap-2">
                <div class="eyebrow">Brain's prediction</div>
                <Show when={verdict()}>
                  <span
                    class="text-[.62rem] font-mono px-1.5 py-0.5 rounded"
                    style={{
                      background:
                        verdict() === "ok"
                          ? "color-mix(in srgb, var(--accent) 16%, transparent)"
                          : "color-mix(in srgb, #e0564e 18%, transparent)",
                      color: verdict() === "ok" ? "var(--accent)" : "#c63d35",
                    }}
                  >
                    {verdict() === "ok"
                      ? "✓ on the goal path"
                      : verdict() === "wall"
                        ? "✕ would hit a wall"
                        : verdict() === "astray"
                          ? "→ away from goal"
                          : "· no move"}
                  </span>
                </Show>
              </div>
              <div class="font-mono text-xs text-mute">
                motor argmax · {ticksTotal() || 16} ticks
              </div>
            </div>
            <div class="flex flex-col gap-2.5">
              <For each={MOVES}>
                {(label, i) => {
                  const p = () => moveProbs()[i()] ?? 0;
                  const chosen = () => committedMove() === i();
                  return (
                    <div class="flex items-center gap-3">
                      <span
                        class="w-12 text-xs font-mono shrink-0"
                        classList={{ "text-accent": chosen(), "text-mute": !chosen() }}
                      >
                        {label}
                      </span>
                      <div class="flex-1 h-5 rounded bg-panel overflow-hidden relative">
                        <div
                          class="h-full rounded transition-all duration-200"
                          style={{
                            width: `${Math.max(2, p() * 100)}%`,
                            background: chosen()
                              ? "var(--accent)"
                              : "var(--line-2)",
                          }}
                        />
                      </div>
                      <span
                        class="w-11 text-right text-xs font-mono tabular-nums shrink-0"
                        classList={{ "text-base": chosen(), "text-mute": !chosen() }}
                      >
                        {(p() * 100).toFixed(0)}%
                      </span>
                    </div>
                  );
                }}
              </For>
            </div>
            <Show when={moveCount() > 0}>
              <div class="mt-3 pt-3 border-t border-line flex items-center justify-between text-xs font-mono text-mute tabular-nums">
                <span>brain agreed with the optimal step</span>
                <span class="text-dim">
                  {Math.round((agreeCount() / moveCount()) * 100)}% ({agreeCount()}/
                  {moveCount()})
                </span>
              </div>
            </Show>
          </div>
        </div>

        {/* RIGHT: the 3D brain (Model B), as the second column */}
        <Show when={brainRef()}>
        <div class="flex flex-col gap-5">
        <div class="card">
          <div class="flex items-center justify-between mb-1 flex-wrap gap-2">
            <div class="eyebrow">
              live in 3D · <span class="grad-text">{brainRef()?.region_names.length ?? 8} regions</span>
            </div>
            <div class="flex items-center gap-3">
              <Show when={brainOn()}>
                <div class="flex items-center gap-2">
                  <span class="font-mono text-[.7rem] text-mute">global sync</span>
                  <div class="w-20 h-1.5 rounded-full bg-panel overflow-hidden">
                    <div
                      class="h-full rounded-full transition-all duration-150"
                      style={{
                        width: `${Math.round(brainGlobal() * 100)}%`,
                        background: "var(--accent)",
                      }}
                    />
                  </div>
                </div>
              </Show>
              <span class="tag" classList={{ "tag-wip": runState() === "thinking" }}>
                the solver
              </span>
            </div>
          </div>

          <p class="text-dim text-sm leading-relaxed mb-4">
            This is the model thinking. Each dot is a neuron, coloured by region;
            lines are the connectome, lit when two regions fire together. The cyan
            grids are the visual cortex — the framed image is what it sees, then
            retina → V1 → V2 → V4 feed the input region.{" "}
            <A href="/docs/brain-composition" class="text-accent whitespace-nowrap">
              how it works →
            </A>
          </p>

          <div class="rounded-lg overflow-hidden relative" style={{ background: "#0a0912" }}>
            <canvas
              ref={(el) => {
                brainCanvas = el;
                el.addEventListener("wheel", brainWheel, { passive: false });
                brainIO = new IntersectionObserver(
                  (es) => (brainVisible = es[0]?.isIntersecting ?? true),
                );
                brainIO.observe(el);
              }}
              class="w-full block cursor-grab active:cursor-grabbing"
              style={{ "touch-action": "none" }}
              onPointerDown={brainPointerDown}
              onPointerMove={brainPointerMove}
              onPointerUp={brainPointerUp}
              onPointerCancel={brainPointerUp}
              onLostPointerCapture={brainPointerUp}
              aria-label="the eight-region brain computing — drag to rotate, scroll to zoom"
            />
            <div class="absolute top-2 right-2 flex items-center gap-1.5">
              <button
                class="btn-icon text-[.68rem] font-mono px-2 py-1 rounded"
                style={{
                  background: "rgba(255,255,255,0.08)",
                  color: "rgba(255,255,255,0.75)",
                  "backdrop-filter": "blur(4px)",
                }}
                onClick={() => setBrainAutoRotate((v) => !v)}
                title="toggle auto-rotation"
              >
                {brainAutoRotate() ? "❚❚ stop" : "⟳ spin"}
              </button>
              <button
                class="text-[.68rem] font-mono px-2 py-1 rounded"
                style={{
                  background: "rgba(255,255,255,0.08)",
                  color: "rgba(255,255,255,0.75)",
                  "backdrop-filter": "blur(4px)",
                }}
                onClick={() => {
                  brainRotX = -0.32;
                  brainRotY = 0.7;
                  brainZoom = 1;
                }}
                title="reset view"
              >
                reset
              </button>
            </div>
            <div class="absolute bottom-2 left-3 text-[.66rem] font-mono pointer-events-none" style={{ color: "rgba(255,255,255,0.4)" }}>
              drag to rotate · scroll / pinch to zoom
            </div>
          </div>

          <div class="flex flex-wrap items-center gap-x-4 gap-y-1.5 mt-3 text-[.72rem] text-mute font-mono">
            <span class="opacity-80">each dot = 1 neuron · brighter = spiking</span>
            <label class="inline-flex items-center gap-1.5" title="how long co-spike lines linger">
              spike trail
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={spikeHold()}
                onInput={(e) => setSpikeHold(parseFloat(e.currentTarget.value))}
                class="w-20 align-middle"
              />
            </label>
            <For each={brainRef()?.region_names ?? []}>
              {(nm, i) => (
                <span class="inline-flex items-center gap-1.5">
                  <i
                    class="w-2.5 h-2.5 rounded-full inline-block"
                    style={{ background: REGION_COLORS[i() % REGION_COLORS.length] }}
                  />
                  {shortRegion(nm)}
                </span>
              )}
            </For>
            <span class="inline-flex items-center gap-1.5" title="visual cortex: retina → V1 → V2 → V4">
              <i class="w-2.5 h-2.5 rounded-full inline-block" style={{ background: "#62e6ff" }} />
              vision
            </span>
            <span class="ml-auto opacity-80">reimplemented from the SDK · bit-exact</span>
          </div>
        </div>

        {/* ── per-region telemetry: live numbers from all eight regions ── */}
        <div class="card mt-5">
          <div class="flex items-center justify-between mb-3">
            <div class="eyebrow">Region telemetry</div>
            <div class="font-mono text-[.7rem] text-mute tabular-nums">
              tick {Math.min(tickIdx() + 1, ticksTotal() || 16)}/{ticksTotal() || 16}
              <Show when={curBrainTick()?.exit != null}>
                {" "}· exit λ {(curBrainTick()!.exit as number).toFixed(2)}
              </Show>
            </div>
          </div>
          <div class="flex flex-col gap-2">
            <For each={regionStats()}>
              {(s, i) => (
                <div class="flex items-center gap-2.5 text-xs font-mono">
                  <i class="w-2 h-2 rounded-full inline-block shrink-0" style={{ background: REGION_COLORS[i() % REGION_COLORS.length] }} />
                  <span class="w-[4.5rem] shrink-0 text-dim">{shortRegion(s.name)}</span>
                  <span class="w-9 shrink-0 text-mute tabular-nums text-right">{s.d}n</span>
                  <div class="flex-1 h-2 rounded bg-panel overflow-hidden">
                    <div
                      class="h-full rounded transition-all duration-150"
                      style={{ width: `${Math.max(2, s.level * 100)}%`, background: REGION_COLORS[i() % REGION_COLORS.length] }}
                    />
                  </div>
                  <span class="w-12 text-right text-base tabular-nums shrink-0" title="activation RMS">{s.rms.toFixed(3)}</span>
                </div>
              )}
            </For>
          </div>
          <div class="mt-3 pt-3 border-t border-line grid grid-cols-2 gap-x-4 gap-y-1 text-[.7rem] font-mono text-mute tabular-nums">
            <span class="flex justify-between"><span>global sync</span><span class="text-dim">{(curBrainTick()?.global ?? 0).toFixed(3)}</span></span>
            <span class="flex justify-between"><span>ticks used</span><span class="text-dim">{brainTrace()?.ticksUsed ?? 0}/{ticksTotal() || 16}</span></span>
          </div>
        </div>
        </div>
        </Show>
      </div>

      {/* ── footnote ──────────────────────────────────── */}
      <p class="text-mute text-xs mt-8 max-w-[70ch] leading-relaxed font-mono">
        Solver: an 8-region brain (input · attention · output · motor · cerebellum
        · basal-ganglia · insula · hippocampus) with a visual retina, computing over{" "}
        {brainRef()?.ticks ?? 16} ticks.{" "}
        {(((brainRef()?.heldout_first_move_acc ?? 0.845)) * 100).toFixed(1)}% per-move
        on held-out 9×9 mazes. The browser engine reimplements modgrad's forward pass —
        bit-exact against the SDK, not the SDK itself on wasm (yet). Loads only here.
      </p>
    </div>
  );
}

// ── small subcomponents / helpers ──────────────────────────────────
function Readout(props: { label: string; value: string; accent?: boolean }) {
  return (
    <div>
      <div class="eyebrow mb-1">{props.label}</div>
      <div
        class="font-mono text-lg tabular-nums"
        classList={{ "text-accent": props.accent, "text-base": !props.accent }}
      >
        {props.value}
      </div>
    </div>
  );
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
) {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}

function marker(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  cell: number,
  _kind: string,
  _line: string,
  accent: string,
) {
  // start: a small hollow ring in accent
  ctx.beginPath();
  ctx.strokeStyle = `color-mix(in srgb, ${accent} 70%, transparent)`;
  ctx.lineWidth = 2;
  ctx.arc(x + cell / 2, y + cell / 2, cell * 0.22, 0, Math.PI * 2);
  ctx.stroke();
}

function goalMark(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  cell: number,
  accent: string,
) {
  // goal: filled rounded diamond in accent
  const s = cell * 0.3;
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(Math.PI / 4);
  ctx.fillStyle = accent;
  roundRect(ctx, -s, -s, s * 2, s * 2, 3);
  ctx.fill();
  ctx.restore();
}
