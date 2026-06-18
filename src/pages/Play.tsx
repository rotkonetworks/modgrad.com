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
type BrainTick = { region: number[]; global: number; exit: number | null };
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
      "A real modgrad Continuous Thought Machine solves a maze in your browser. Watch its neurons fire, its attention sweep the maze, and its certainty rise as it commits to each move. Runs entirely client-side via wasm.",
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

  // ── the 8-region brain map (Model B) ──
  // anatomical-ish fixed layout, keyed by region name (robust to index order).
  // cortical loop (input→attention→output→motor→input) as a left diamond with
  // hippocampus at its centre; subcortical structures down the right.
  const BRAIN_POS: Record<string, [number, number]> = {
    input: [0.15, 0.5],
    attention: [0.35, 0.17],
    output: [0.55, 0.5],
    motor: [0.35, 0.83],
    hippocampus: [0.35, 0.5],
    basal_ganglia: [0.78, 0.25],
    insula: [0.8, 0.52],
    cerebellum: [0.78, 0.79],
  };
  const BRAIN_LABEL: Record<string, string> = {
    input: "input",
    attention: "attn",
    output: "output",
    motor: "motor",
    cerebellum: "cereb",
    basal_ganglia: "basal",
    insula: "insula",
    hippocampus: "hippo",
  };

  // brain tick aligned to the current animation tick (held at the last brain
  // tick if the brain exited earlier than the solver's tick count).
  const curBrainTick = (): BrainTick | null => {
    const bt = brainTrace();
    if (!bt || bt.ticks.length === 0) return null;
    return bt.ticks[Math.min(tickIdx(), bt.ticks.length - 1)];
  };

  function drawBrain() {
    const bref = brainRef();
    if (!brainCanvas || !bref) return;
    const W = brainCanvas.clientWidth || 720;
    const H = Math.round(Math.max(260, Math.min(420, W * 0.46)));
    const ctx = ctxOf(brainCanvas, W, H);
    const padX = W * 0.07;
    const padY = H * 0.12;
    const px = (nx: number) => padX + nx * (W - 2 * padX);
    const py = (ny: number) => padY + ny * (H - 2 * padY);

    const names = bref.region_names;
    const dmodels = bref.regions.map((r) => r.d_model);
    const big = Math.max(...dmodels); // 32 cortical
    const nodeR = (d: number) => {
      const s = Math.min(W, H * 1.7) / 720;
      return (10 + 16 * Math.sqrt(d / big)) * s;
    };

    const accent = cssVar("--accent", "#6243D9");
    const teal = "#29C7D6";
    const line = cssVar("--line-2", "#D8D2C3");
    const dim = cssVar("--text-mute", "#6A6676");
    const cardbg = cssVar("--bg-card", "#fff");

    const pos = (i: number): [number, number] => {
      const p = BRAIN_POS[names[i]] ?? [0.5, 0.5];
      return [px(p[0]), py(p[1])];
    };

    // normalise region magnitudes across the whole step for stable glow
    const bt = brainTrace();
    let mmax = 1e-6;
    if (bt) for (const t of bt.ticks) for (const v of t.region) if (v > mmax) mmax = v;
    const tick = curBrainTick();
    const magOf = (i: number) => (tick ? Math.min(1, (tick.region[i] ?? 0) / mmax) : 0);

    // ── edges (drawn under nodes) ──
    for (const conn of bref.connections) {
      const [tx, ty] = pos(conn.to);
      for (const f of conn.from) {
        const [sx, sy] = pos(f);
        const act = magOf(f); // brightness follows the source region
        ctx.strokeStyle = line;
        ctx.lineWidth = 1;
        ctx.globalAlpha = 0.55;
        ctx.beginPath();
        ctx.moveTo(sx, sy);
        ctx.lineTo(tx, ty);
        ctx.stroke();
        ctx.globalAlpha = 1;
        if (act > 0.05 && runState() === "thinking") {
          // live signal: brighter edge + a pulse travelling toward the target
          ctx.strokeStyle = accent;
          ctx.lineWidth = 1 + act * 2.2;
          ctx.globalAlpha = 0.25 + act * 0.5;
          ctx.beginPath();
          ctx.moveTo(sx, sy);
          ctx.lineTo(tx, ty);
          ctx.stroke();
          ctx.globalAlpha = 1;
          const fr = 0.35 + 0.4 * ((tickIdx() % 3) / 2); // marches with ticks
          const dx = sx + (tx - sx) * fr;
          const dy = sy + (ty - sy) * fr;
          ctx.beginPath();
          ctx.fillStyle = accent;
          ctx.arc(dx, dy, 2 + act * 2.5, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    // ── nodes ──
    for (let i = 0; i < names.length; i++) {
      const [x, y] = pos(i);
      const r = nodeR(dmodels[i]);
      const m = magOf(i);

      // glow halo scaled by activation
      if (m > 0.02) {
        const g = ctx.createRadialGradient(x, y, r * 0.4, x, y, r * (1.7 + m));
        g.addColorStop(0, `color-mix(in srgb, ${accent} ${28 + m * 55}%, transparent)`);
        g.addColorStop(1, "transparent");
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(x, y, r * (1.7 + m), 0, Math.PI * 2);
        ctx.fill();
      }

      // node body — lerp teal→violet by activation, on a faint base
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fillStyle = `color-mix(in srgb, ${m > 0.5 ? accent : teal} ${15 + m * 80}%, ${cardbg})`;
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = m > 0.1 ? accent : line;
      ctx.stroke();

      // hippocampus: episodic memory stack (one entry accrues per tick)
      if (names[i] === "hippocampus" && runState() === "thinking") {
        const entries = Math.min(tickIdx() + 1, 6);
        for (let e = 0; e < entries; e++) {
          ctx.fillStyle = `color-mix(in srgb, ${accent} ${40 + e * 8}%, transparent)`;
          ctx.fillRect(x - r * 0.5 + e * (r * 0.2), y - r - 7, r * 0.13, 4);
        }
      }

      // label
      ctx.fillStyle = m > 0.3 ? cssVar("--text", "#1B1822") : dim;
      ctx.font = `${Math.round(10 * Math.min(1.4, W / 600))}px ui-monospace, monospace`;
      ctx.textAlign = "center";
      ctx.fillText(BRAIN_LABEL[names[i]] ?? names[i], x, y + r + 13);
    }
    ctx.textAlign = "left";
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
  createEffect(() => {
    tickIdx();
    brainTrace();
    brainRef();
    runState();
    drawBrain();
  });

  // redraw on resize (canvas is fluid width)
  onMount(() => {
    const onResize = () => {
      drawMaze();
      drawNeurons();
      drawCertainty();
      drawBrain();
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
      {/* ── intro ─────────────────────────────────────── */}
      <div class="max-w-[760px]">
        <div class="eyebrow mb-3">Live demo · runs in your browser</div>
        <h1 class="text-[clamp(2rem,5vw,3rem)] tracking-[-0.03em] leading-[1.08]">
          Watch it <span class="grad-text">think.</span>
        </h1>
        <p class="mt-5 text-dim text-[1.05rem] max-w-[64ch] leading-relaxed">
          This is a real modgrad Continuous Thought Machine, a single pool of{" "}
          {ref()?.d_model ?? 256} neurons trained on {ref()?.size ?? 9}×
          {ref()?.size ?? 9} mazes. To pick each move it thinks over{" "}
          {ref()?.ticks ?? 8} internal ticks, and what you see below is
          its actual internal state, not a recording. Further down, the full{" "}
          <span class="grad-text">8-region brain</span> runs on the same maze. Both
          models load entirely client-side, into a web worker as wasm.
        </p>
      </div>

      {/* ── loading / error states ────────────────────── */}
      <Show when={status() === "error"}>
        <div class="card mt-8 max-w-[560px]">
          <div class="eyebrow mb-2 text-warn">Could not start</div>
          <p class="text-dim text-sm leading-relaxed">{errMsg()}</p>
        </div>
      </Show>

      {/* ── the lab ───────────────────────────────────── */}
      <div class="grid lg:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)] gap-5 mt-10">
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
                <div class="absolute inset-0 grid place-items-center">
                  <div class="text-center">
                    <div class="text-2xl grad-text font-mono mb-2 animate-pulse">∇</div>
                    <div class="text-mute text-xs font-mono">loading the brain…</div>
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

      {/* ── the whole brain (Model B) ─────────────────── */}
      <Show when={brainRef()}>
        <div class="card mt-5">
          <div class="flex items-center justify-between mb-1 flex-wrap gap-2">
            <div class="eyebrow">
              The whole brain · <span class="grad-text">8 regions</span>
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

          <p class="text-dim text-sm max-w-[72ch] leading-relaxed mb-4">
            The same modgrad brain modgrad ships, running live next to the solver
            above. It is eight specialised regions — a cortical loop
            (input&nbsp;→&nbsp;attention&nbsp;→&nbsp;output&nbsp;→&nbsp;motor) wired
            to a hippocampus with episodic memory and three subcortical helpers —
            all fed by a visual retina that looks at the maze. Watch each region
            fire and route signals as the agent thinks through a move.
          </p>

          <div class="rounded-lg overflow-hidden" style={{ background: "var(--bg-2)" }}>
            <canvas
              ref={brainCanvas}
              class="w-full block"
              aria-label="the eight-region brain computing"
            />
          </div>

          <div class="flex flex-wrap items-center gap-x-5 gap-y-1.5 mt-3 text-[.72rem] text-mute font-mono">
            <span class="inline-flex items-center gap-1.5">
              <i class="w-2.5 h-2.5 rounded-full inline-block" style={{ background: "var(--accent)" }} />
              firing
            </span>
            <span class="inline-flex items-center gap-1.5">
              <i class="w-2.5 h-2.5 rounded-full inline-block" style={{ background: "#29C7D6" }} />
              quiet
            </span>
            <span>big node = cortical (32 neurons) · small = subcortical (8)</span>
            <span class="ml-auto opacity-80">bit-exact with the modgrad SDK</span>
          </div>
        </div>
      </Show>

      {/* ── footnote ──────────────────────────────────── */}
      <p class="text-mute text-xs mt-8 max-w-[70ch] leading-relaxed font-mono">
        Solver: single CTM, {ref()?.d_model ?? 256}-neuron pool, {ref()?.ticks ?? 8}{" "}
        ticks, {((ref()?.move_acc ?? 0.8) * 100).toFixed(0)}% move accuracy on
        held-out cells. Brain: 8-region modgrad architecture (~187k params) with a
        visual retina, run bit-exact in wasm. Both load only on this page, after
        first paint.
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
