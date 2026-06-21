import {
  createSignal,
  createEffect,
  onMount,
  onCleanup,
  Show,
  For,
  batch,
} from "solid-js";
import { useDocMeta } from "@/lib/meta";
import { MOVES, softmax, cssVar, neuronColor, heatAlpha } from "@/play/viz";

// ── types mirrored from the worker ────────────────────────────────
type Tick = {
  predictions: number[];
  certainty: number;
  activations: number[];
  attention: number[];
};
type BrainTick = { acts: number[][]; global: number; exit: number | null };
type BrainTrace = { ticks: BrainTick[]; ticksUsed: number };
type StepMsg = {
  type: "step";
  agent: [number, number];
  move: number;
  commitTick: number;
  ticks: Tick[];
  brain: BrainTrace | null;
  done: boolean;
  reached: boolean;
};
// structure of the 8-region brain, read from brain_reference.json
type BrainConn = { from: number[]; to: number; receives_observation: boolean };
type BrainRef = {
  region_names: string[];
  regions: { name: string; d_model: number }[];
  connections: BrainConn[];
  n_global_sync: number;
};
type Maze = {
  grid: number[];
  start: [number, number];
  end: [number, number];
  path_length: number;
};
type Reference = {
  size: number;
  ticks: number;
  d_model: number;
  raw_dim: number;
  out_dims: number;
  heads: number;
  move_acc: number;
  solve_rate: number;
  mazes: Maze[];
};

type Status = "loading" | "ready" | "error";
type RunState = "idle" | "thinking" | "solved" | "stuck";

const TICK_MS = 95; // pace of the "thinking" animation
const MAX_STEPS = 30; // safety cap

export default function Play() {
  useDocMeta(() => ({
    title: "Watch it think",
    description:
      "A modgrad Continuous Thought Machine solves a maze in your browser. Watch its neurons fire, its attention sweep the maze, and its certainty rise as it commits to each move. Your trained model, run client-side by a faithful wasm reimplementation of the modgrad SDK's forward pass.",
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

  const [ticks, setTicks] = createSignal<Tick[]>([]);
  const [tickIdx, setTickIdx] = createSignal(0);
  const [commitTick, setCommitTick] = createSignal(0);
  const [committedMove, setCommittedMove] = createSignal(-1);

  const [playing, setPlaying] = createSignal(false);
  const [showAttention, setShowAttention] = createSignal(true);

  // ── 8-region brain (Model B) ──
  const [brainRef, setBrainRef] = createSignal<BrainRef | null>(null);
  const [brainOn, setBrainOn] = createSignal(false); // brain wasm available
  const [brainTrace, setBrainTrace] = createSignal<BrainTrace | null>(null);

  // dimensions from the reference; defaults avoid layout shift before load
  let SIZE = 7;
  let DMODEL = 128;
  const [ref, setRef] = createSignal<Reference | null>(null);

  const maze = () => currentMaze();
  const curTick = (): Tick | null => ticks()[tickIdx()] ?? null;

  // ── worker plumbing ─────────────────────────────────────────────
  let worker: Worker | null = null;
  let pendingStep = false; // a step request is in flight to the worker
  let stepTimer: number | undefined;
  let playTimer: number | undefined;

  onMount(async () => {
    let reference: Reference;
    try {
      const r = await fetch("/models/maze_reference.json");
      reference = await r.json();
    } catch (e) {
      setStatus("error");
      setErrMsg("Could not load the maze set.");
      return;
    }
    SIZE = reference.size;
    DMODEL = reference.d_model;
    batch(() => {
      setRef(reference);
      loadMaze(reference.mazes[0]); // instant placeholder until the brain is ready
    });

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

    // brain structure (small) loads in parallel; the 9MB brain weights are
    // optional — if they fail, the maze demo still runs without the brain map.
    fetch("/models/brain_reference.json")
      .then((r) => r.json())
      .then((b: BrainRef) => setBrainRef(b))
      .catch(() => {});

    try {
      const [wr, br] = await Promise.all([
        fetch("/models/maze_weights.json").then((r) => r.text()),
        fetch("/models/brain_weights.json")
          .then((r) => r.text())
          .catch(() => ""),
      ]);
      worker.postMessage({
        type: "init",
        weights: wr,
        brainWeights: br || undefined,
        size: reference.size,
        rawDim: reference.raw_dim,
      });
    } catch (e) {
      setStatus("error");
      setErrMsg("Could not load the trained weights.");
    }
  });

  onCleanup(() => {
    worker?.terminate();
    clearTimeout(stepTimer);
    clearTimeout(playTimer);
    brainCanvas?.removeEventListener("wheel", brainWheel);
  });

  // ── solve loop ──────────────────────────────────────────────────
  function loadMaze(m: Maze) {
    batch(() => {
      setCurrentMaze(m);
      setAgent([m.start[0], m.start[1]]);
      setVisited([[m.start[0], m.start[1]]]);
      setStepCount(0);
      setTicks([]);
      setBrainTrace(null);
      setTickIdx(0);
      setCommitTick(0);
      setCommittedMove(-1);
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

  // the worker returns the full per-tick trace for one move; animate through it
  function onStepResult(msg: StepMsg) {
    pendingStep = false;
    batch(() => {
      setTicks(msg.ticks);
      setBrainTrace(msg.brain);
      setCommitTick(msg.commitTick);
      setCommittedMove(msg.move);
      setTickIdx(0);
    });

    if (reduced) {
      // no animation: jump to the commit tick, then apply the move
      setTickIdx(msg.commitTick);
      applyMove(msg);
      return;
    }
    animateTicks(0, msg);
  }

  function animateTicks(i: number, msg: StepMsg) {
    setTickIdx(i);
    if (i >= msg.ticks.length - 1) {
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
      batch(() => {
        setRunState("solved");
        setPlaying(false);
      });
      return;
    }
    if (msg.done) {
      // blocked (shouldn't happen on curated mazes) — stop gracefully
      batch(() => {
        setRunState("stuck");
        setPlaying(false);
      });
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
  const moveProbs = (): number[] => {
    const t = ticks()[commitTick()];
    return t ? softmax(t.predictions) : [0.25, 0.25, 0.25, 0.25];
  };
  const certaintyNow = () => curTick()?.certainty ?? 0;

  // ════════════════════════════════════════════════════════════════
  //  canvas rendering
  // ════════════════════════════════════════════════════════════════
  let mazeCanvas!: HTMLCanvasElement;
  let neuronCanvas!: HTMLCanvasElement;
  let certCanvas!: HTMLCanvasElement;
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

    // attention heat overlay (open cells only)
    const t = curTick();
    if (showAttention() && t && runState() === "thinking") {
      for (let r = 0; r < n; r++) {
        for (let c = 0; c < n; c++) {
          if (m.grid[r * n + c] === 1) continue;
          const a = t.attention[r * n + c] ?? 0;
          if (a <= 0.02) continue;
          ctx.fillStyle = `rgba(251, 94, 109, ${heatAlpha(a)})`;
          roundRect(ctx, cx(c), cy(r), cell, cell, 4);
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
      const pulse = 0.34 + (tickIdx() / Math.max(1, ticks().length)) * 0.18;
      ctx.arc(ax, ay, cell * pulse, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  // ── neuron grid: activations[tick] as a grid ──
  function drawNeurons() {
    if (!neuronCanvas) return;
    const W = neuronCanvas.clientWidth || 360;
    const cols = 16; // 16 × 8 = 128
    const rows = Math.ceil(DMODEL / cols);
    const gap = 3;
    const cellW = (W - gap * (cols - 1)) / cols;
    const H = rows * cellW + gap * (rows - 1);
    const ctx = ctxOf(neuronCanvas, W, H);

    const t = curTick();
    const base = cssVar("--bg-2", "#F1EEE6");
    const scale = 3.2; // activations roughly span ±6; saturate around 3
    for (let i = 0; i < DMODEL; i++) {
      const r = Math.floor(i / cols);
      const c = i % cols;
      const v = t?.activations[i] ?? 0;
      ctx.fillStyle = base;
      roundRect(ctx, c * (cellW + gap), r * (cellW + gap), cellW, cellW, 3);
      ctx.fill();
      if (t) {
        ctx.fillStyle = neuronColor(v, scale);
        roundRect(ctx, c * (cellW + gap), r * (cellW + gap), cellW, cellW, 3);
        ctx.fill();
      }
    }
  }

  // ── certainty curve across the 16 ticks ──
  function drawCertainty() {
    if (!certCanvas) return;
    const W = certCanvas.clientWidth || 360;
    const H = 120;
    const ctx = ctxOf(certCanvas, W, H);
    const padL = 4;
    const padR = 4;
    const padT = 10;
    const padB = 16;
    const tk = ticks();
    const total = ref()?.ticks ?? 16;

    const line = cssVar("--line", "#E6E1D5");
    const accent = cssVar("--accent", "#6243D9");
    const dim = cssVar("--text-mute", "#6A6676");

    // baseline grid (0 / .5 / 1)
    ctx.strokeStyle = line;
    ctx.lineWidth = 1;
    for (const f of [0, 0.5, 1]) {
      const y = padT + (1 - f) * (H - padT - padB);
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(W - padR, y);
      ctx.stroke();
    }

    if (tk.length === 0) return;
    const xAt = (i: number) =>
      padL + (i / (total - 1)) * (W - padL - padR);
    const yAt = (v: number) => padT + (1 - v) * (H - padT - padB);

    // commit tick marker
    const ci = commitTick();
    if (ci < tk.length) {
      ctx.strokeStyle = `color-mix(in srgb, ${accent} 35%, transparent)`;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(xAt(ci), padT);
      ctx.lineTo(xAt(ci), H - padB);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // the curve (only up to ticks revealed so far, so it "draws in")
    const upTo = Math.min(tickIdx(), tk.length - 1);
    ctx.strokeStyle = accent;
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.beginPath();
    for (let i = 0; i <= upTo; i++) {
      const x = xAt(i);
      const y = yAt(tk[i].certainty);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();

    // current point
    const x = xAt(upTo);
    const y = yAt(tk[upTo].certainty);
    ctx.beginPath();
    ctx.fillStyle = accent;
    ctx.arc(x, y, 3.5, 0, Math.PI * 2);
    ctx.fill();

    // commit dot
    if (ci <= upTo) {
      ctx.beginPath();
      ctx.fillStyle = accent;
      ctx.arc(xAt(ci), yAt(tk[ci].certainty), 4.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = cssVar("--bg-card", "#fff");
      ctx.stroke();
    }

    // x label
    ctx.fillStyle = dim;
    ctx.font = "10px ui-monospace, monospace";
    ctx.fillText("tick 0", padL, H - 4);
    ctx.fillText(`${total - 1}`, W - padR - 12, H - 4);
  }

  // ── the 8-region brain as a rotating 3D particle cloud (Model B) ──
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
  // port of debugger build_neuron_positions: a spherical blob per region,
  // cortical (regions 0–3) on the top row, subcortical (4–7) on the bottom.
  let regionCenters: { x: number; y: number; z: number }[] = [];
  function buildNeuronPositions(dmodels: number[]) {
    neuronPos = [];
    regionCenters = [];
    const rand = makeRand(42);
    for (let ri = 0; ri < dmodels.length; ri++) {
      const neurons = dmodels[ri];
      const col = ri % 4;
      const row = Math.floor(ri / 4);
      const cx = col * 3.5 - 5.25;
      const cy = 1.9 - row * 3.8;
      const cz = ri * 0.3 - 0.5;
      regionCenters[ri] = { x: cx, y: cy, z: cz };
      const spread = Math.sqrt(neurons) * 0.34;
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
  let brainTime = 0; // seconds, advanced by the rAF loop for the idle shimmer
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

  function drawBrain3D() {
    const bref = brainRef();
    if (!brainCanvas || !bref) return;
    const dmodels = bref.regions.map((r) => r.d_model);
    const total = dmodels.reduce((a, b) => a + b, 0);
    if (builtFor !== total) buildNeuronPositions(dmodels);

    const W = brainCanvas.clientWidth || 720;
    const H = Math.round(Math.max(340, Math.min(520, W * 0.6)));
    const ctx = ctxOf(brainCanvas, W, H);
    const cxp = W / 2;
    const cyp = H / 2;
    const fov = W * 0.52 * brainZoom;

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
      if (depth < 0.5) continue;
      pts.push({
        x: cxp + (rx * fov) / depth,
        y: cyp - (ry * fov) / depth,
        depth,
        region: np.region,
        a: actOf(np.region, np.index),
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

    // ── connectome edges: faint as structure, bright when both ends co-spike ──
    ctx.lineCap = "round";
    for (const conn of bref.connections) {
      const tp = centerProj[conn.to];
      if (!tp) continue;
      const colT = REGION_COLORS[conn.to % REGION_COLORS.length];
      for (const f of conn.from) {
        const fp = centerProj[f];
        if (!fp) continue;
        const co = Math.min(1, regionAct[f] * regionAct[conn.to] * 2.4); // co-spike
        const colF = REGION_COLORS[f % REGION_COLORS.length];
        const grad = ctx.createLinearGradient(fp.x, fp.y, tp.x, tp.y);
        grad.addColorStop(0, hexA(colF, 0.04 + 0.6 * co));
        grad.addColorStop(1, hexA(colT, 0.04 + 0.6 * co));
        ctx.strokeStyle = grad;
        ctx.lineWidth = 0.5 + 3 * co;
        ctx.beginPath();
        ctx.moveTo(fp.x, fp.y);
        ctx.lineTo(tp.x, tp.y);
        ctx.stroke();
        // a spark travelling source→destination while they co-fire
        if (co > 0.12) {
          const fr = (brainTime * 0.55 + f * 0.17) % 1;
          ctx.fillStyle = hexA("#ffffff", Math.min(0.9, co));
          ctx.beginPath();
          ctx.arc(fp.x + (tp.x - fp.x) * fr, fp.y + (tp.y - fp.y) * fr, 1.3 + 2.2 * co, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    // additive blending → overlapping neurons bloom, spikes pop
    for (const p of pts) {
      const col = REGION_COLORS[p.region % REGION_COLORS.length];
      const fogv = Math.max(0.4, Math.min(1, 14 / p.depth - 0.25)); // nearer = brighter/bigger
      const size = Math.max(1.1, (10 / p.depth) * (1.5 + 3.4 * p.a) * fogv);
      // soft glow halo
      const glowA = (0.12 + 0.6 * p.a) * fogv;
      const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, size * 3.4);
      g.addColorStop(0, hexA(col, glowA));
      g.addColorStop(1, hexA(col, 0));
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(p.x, p.y, size * 3.4, 0, Math.PI * 2);
      ctx.fill();
      // bright core
      ctx.fillStyle = hexA(col, Math.min(1, (0.5 + 0.5 * p.a) * fogv));
      ctx.beginPath();
      ctx.arc(p.x, p.y, size, 0, Math.PI * 2);
      ctx.fill();
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
    // dependencies: maze, agent, visited, tick, attention toggle, runState
    currentMaze();
    agent();
    visited();
    tickIdx();
    ticks();
    showAttention();
    runState();
    drawMaze();
  });
  createEffect(() => {
    tickIdx();
    ticks();
    drawNeurons();
  });
  createEffect(() => {
    tickIdx();
    ticks();
    commitTick();
    drawCertainty();
  });
  // the brain cloud renders continuously (slow rotation) via rAF, sampling the
  // current animation tick's per-neuron activations each frame.
  onMount(() => {
    let raf = 0;
    const spin = () => {
      brainTime += 0.016;
      if (!reduced && brainAutoRotate() && !brainDragging) brainRotY += 0.0045;
      drawBrain3D();
      raf = requestAnimationFrame(spin);
    };
    raf = requestAnimationFrame(spin);
    onCleanup(() => cancelAnimationFrame(raf));
  });

  // redraw on resize (canvas is fluid width)
  onMount(() => {
    const onResize = () => {
      drawMaze();
      drawNeurons();
      drawCertainty();
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
        <p class="mt-5 text-dim text-[1.05rem] max-w-[64ch] leading-relaxed">
          This page runs <span class="text-base">two</span> modgrad models in your
          browser. The <span class="text-base">solver</span> (shown first) is a single
          Continuous Thought Machine — one pool of {ref()?.d_model ?? 256} neurons
          that navigates {ref()?.size ?? 9}×{ref()?.size ?? 9} mazes by thinking over{" "}
          {ref()?.ticks ?? 8} internal ticks per move. Below it, modgrad's full{" "}
          <span class="grad-text">8-region brain</span> watches the same maze and
          thinks in 3D alongside it. The weights are real — trained by the modgrad
          SDK — and what you see is each model's actual internal state, computed live
          by a faithful browser reimplementation of the SDK's forward pass (checked
          bit-exact against it), not a recording.
        </p>
      </div>

      {/* ── loading / error states ────────────────────── */}
      <Show when={status() === "error"}>
        <div class="card mt-8 max-w-[560px]">
          <div class="eyebrow mb-2 text-warn">Could not start</div>
          <p class="text-dim text-sm leading-relaxed">{errMsg()}</p>
        </div>
      </Show>

      {/* ── model 1: the solver ───────────────────────── */}
      <div class="flex items-baseline gap-3 mt-12 mb-1">
        <h2 class="text-[1.4rem] tracking-[-0.02em]">
          <span class="text-mute font-mono text-base mr-1.5">1 ·</span> The solver
        </h2>
        <span class="text-mute text-sm">single CTM — picks the moves</span>
      </div>

      {/* ── the lab ───────────────────────────────────── */}
      <div class="grid lg:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)] gap-5 mt-4">
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
                      loading two models · ~12 MB · runs on your device
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
              <Show when={showAttention()}>
                <span class="inline-flex items-center gap-1.5">
                  <i class="w-2.5 h-2.5 rounded-full inline-block" style={{ background: "#FB5E6D" }} />
                  attention
                </span>
              </Show>
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
              <label class="flex items-center gap-2 text-sm text-dim cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={showAttention()}
                  onChange={(e) => setShowAttention(e.currentTarget.checked)}
                />
                Attention overlay
              </label>
              <div class="font-mono text-xs text-mute tabular-nums">
                maze #{Math.max(1, mazeNum())} · step {stepCount()}
              </div>
            </div>
          </div>

          {/* move bars */}
          <div class="card">
            <div class="flex items-center justify-between mb-3">
              <div class="eyebrow">Committed move</div>
              <div class="font-mono text-xs text-mute">
                argmax @ tick {commitTick()}
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
          </div>
        </div>

        {/* RIGHT: neurons, certainty, readout */}
        <div class="flex flex-col gap-5">
          {/* readout */}
          <div class="card">
            <div class="grid grid-cols-3 gap-4">
              <Readout label="tick" value={`${tickIdx()} / ${(ref()?.ticks ?? 16) - 1}`} />
              <Readout
                label="certainty"
                value={`${(certaintyNow() * 100).toFixed(0)}%`}
              />
              <Readout
                label="move"
                value={committedMove() >= 0 ? MOVES[committedMove()] : "—"}
                accent
              />
            </div>
          </div>

          {/* neuron grid */}
          <div class="card">
            <div class="flex items-center justify-between mb-3">
              <div class="eyebrow">Neuron pool · {DMODEL}</div>
              <div class="font-mono text-[.7rem] text-mute flex items-center gap-2">
                <span class="inline-flex items-center gap-1">
                  <i class="w-2 h-2 inline-block rounded-sm" style={{ background: "#29C7D6" }} />−
                </span>
                <span class="inline-flex items-center gap-1">
                  <i class="w-2 h-2 inline-block rounded-sm" style={{ background: "var(--accent)" }} />+
                </span>
              </div>
            </div>
            <canvas
              ref={neuronCanvas}
              class="w-full block"
              aria-label="neuron activations"
            />
            <p class="text-mute text-xs mt-3 leading-relaxed">
              Each square is one neuron at the current tick. Violet fires positive,
              teal inhibited. The pattern is the brain's evolving thought.
            </p>
          </div>

          {/* certainty curve */}
          <div class="card">
            <div class="flex items-center justify-between mb-2">
              <div class="eyebrow">Certainty over {ref()?.ticks ?? 8} ticks</div>
              <div class="font-mono text-xs text-mute tabular-nums">
                {(certaintyNow() * 100).toFixed(0)}%
              </div>
            </div>
            <div style={{ height: "120px" }}>
              <canvas
                ref={certCanvas}
                class="w-full h-full block"
                aria-label="certainty over thinking ticks"
              />
            </div>
            <p class="text-mute text-xs mt-2 leading-relaxed">
              Confidence (1 − entropy) as the thought unfolds. The dashed line marks
              the tick the brain is most certain on; that tick's argmax is the move it
              commits.
            </p>
          </div>
        </div>
      </div>

      {/* ── model 2: the brain ────────────────────────── */}
      <Show when={brainRef()}>
        <div class="flex items-baseline gap-3 mt-12 mb-1">
          <h2 class="text-[1.4rem] tracking-[-0.02em]">
            <span class="text-mute font-mono text-base mr-1.5">2 ·</span> The{" "}
            <span class="grad-text">brain</span>
          </h2>
          <span class="text-mute text-sm">8 regions — watches &amp; thinks</span>
        </div>
        <div class="card mt-4">
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
                Model B
              </span>
            </div>
          </div>

          <p class="text-dim text-sm max-w-[74ch] leading-relaxed mb-4">
            This is a <span class="text-base">second, separate model</span>:
            modgrad's full 8-region brain, running live in 3D as the solver works.
            Every dot is one neuron, grouped and coloured by its region — a cortical
            loop (input&nbsp;→&nbsp;attention&nbsp;→&nbsp;output&nbsp;→&nbsp;motor)
            wired to a hippocampus with episodic memory, plus subcortical helpers —
            all fed by a visual retina that looks at the maze. Neurons brighten as
            they spike, so you watch the whole brain compute. It doesn't move the
            agent; the single-CTM solver above does that. This is the look inside.
          </p>

          <div class="rounded-lg overflow-hidden relative" style={{ background: "#0a0912" }}>
            <canvas
              ref={(el) => {
                brainCanvas = el;
                el.addEventListener("wheel", brainWheel, { passive: false });
              }}
              class="w-full block"
              style={{ "touch-action": "none", cursor: brainDragging ? "grabbing" : "grab" }}
              onPointerDown={brainPointerDown}
              onPointerMove={brainPointerMove}
              onPointerUp={brainPointerUp}
              onPointerCancel={brainPointerUp}
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
            <span class="ml-auto opacity-80">reimplemented from the SDK · bit-exact</span>
          </div>
        </div>
      </Show>

      {/* ── footnote ──────────────────────────────────── */}
      <p class="text-mute text-xs mt-8 max-w-[70ch] leading-relaxed font-mono">
        Solver: single CTM, {ref()?.d_model ?? 256}-neuron pool, {ref()?.ticks ?? 8}{" "}
        ticks, {((ref()?.move_acc ?? 0.8) * 100).toFixed(0)}% move accuracy on
        held-out cells. Brain: 8-region modgrad architecture (~187k params) with a
        visual retina. The browser engine is a faithful reimplementation of modgrad's
        forward pass, validated bit-exact against the SDK that trained the weights —
        not the SDK compiled to wasm (yet). Both load only on this page, after first
        paint.
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
