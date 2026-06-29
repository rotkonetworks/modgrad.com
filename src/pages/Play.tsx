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
import { createBrain3D } from "@/play/viz/brain3d";
import type { Camera } from "@/play/viz/brain3d";
import { drawMaze as drawMazeModule } from "@/play/viz/maze";
import type { MazeRenderState, MazeTheme } from "@/play/viz/maze";
// NOTE: "what it sees" (the retina image) is rendered inside the 3D brain by
// brain3d.ts's built-in vision pathway, so we don't draw a standalone sight
// screen here. drawSightScreen() from viz/retina.ts is available if a dedicated
// spot is wanted later.
import SdkFeatures from "@/play/components/SdkFeatures";
import RegionTelemetry from "@/play/components/RegionTelemetry";
import type { RegionTel } from "@/play/components/RegionTelemetry";
import PlasticityPanel from "@/play/components/PlasticityPanel";

// ── types mirrored from the worker ────────────────────────────────
type BrainTick = { acts: number[][]; global: number; exit: number | null };
type BrainTrace = { ticks: BrainTick[]; ticksUsed: number };
type RetinaMap = { name: string; channels: number; h: number; w: number; data: number[] };
// NEW engine telemetry, mirrored from the worker (all optional, feature-detected)
type WorkerRegionTelemetry = {
  name?: string;
  activity?: number;
  neuromod?: number;
  certainty?: number;
};
type BrainTelemetry = {
  regions: WorkerRegionTelemetry[];
  exitGate?: number;
  certainty?: number;
};
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
  // NEW engine signals — present only when the engine exports them
  telemetry?: BrainTelemetry;
  plasticDelta?: number;
  signal?: number;
  loss?: number; // move-head cross-entropy vs the target move (the curve that drops)
  lr?: number; // three-factor learning rate θ used this step
  // episodic recall: nearest past situation by global-sync similarity
  episodic?: {
    recalled: boolean;
    sim: number;
    id: number;
    size: number;
    move: number;
    verdict: string;
    reached: boolean;
  } | null;
  // ── drive-mode / graded neuromodulation (Part A) ──
  neuromod?: "dopamine" | "reward" | "disappointment" | "pain";
  efficiency?: number; // live steps ÷ shortest (→1.0 as it learns)
  efficiencyFinal?: number; // per-solve efficiency point (only on reached)
  shortest?: number; // BFS shortest-path length from the episode start
  stepsTaken?: number; // steps taken in this episode so far
  vetoed?: boolean; // the self-driven move hit a wall/edge → vetoed + PAIN
  lost?: boolean; // honest mode: episode exceeded budget without solving
};

// the three modular drive-modes (a UI toggle switches them live).
type DriveMode = "easy" | "normal" | "hard" | "hardcore";
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
const MAX_STEPS = 160; // safety backstop. 11×11 paths can be long, and self-drive
// wanders up to ~3× shortest before the worker nudges/loses — so give it room.
// the board the worker runs on. MUST stay ODD (recursive-backtracker mazes need
// an odd side). reference.size (=9) is kept only for the "trained on 9×9" copy.
const PLAY_SIZE = 11;
// UP=0 DOWN=1 LEFT=2 RIGHT=3 — [dr, dc], matches the worker's DELTA
const DIR_DELTA: [number, number][] = [
  [-1, 0],
  [1, 0],
  [0, -1],
  [0, 1],
];

// ── camera helpers (build to brain3d.ts's documented free-fly interface) ──
// brain3d's view transform is  view = Rx(-pitch)·Ry(-yaw)·(p − pos), so the
// camera forward (the world dir that maps to view +z) is
//   fwd = Ry(yaw)·Rx(pitch)·(0,0,1) = ( sin(yaw)cos(pitch), −sin(pitch),
//                                       cos(yaw)cos(pitch) ).
// Windowed orbit reuses the old rotX/rotY/zoom by deriving an equivalent
// free-fly camera that looks at `center` from `dist` along those angles.
//
// NOTE: brain3d.ts is slated to export its own orbitToCamera(); when it does,
// swap this local copy for the import. Until then this keeps Play.tsx
// independently type-clean against the documented Camera contract.
const BRAIN_CENTER: [number, number, number] = [0, 0, 0];
const ORBIT_DIST = 13; // distance the orbit camera sits from the brain centre
const BASE_FOV_PX = 720 * 0.72; // matches the old fov = W*0.72*zoom feel at W≈720

function orbitToCamera(
  rotX: number,
  rotY: number,
  zoom: number,
  center: [number, number, number] = BRAIN_CENTER,
  dist: number = ORBIT_DIST,
): Camera {
  // orbit angles → look direction. rotY = yaw (horizontal), rotX = pitch
  // (vertical). The camera sits opposite its forward vector, `dist` out.
  const yaw = rotY;
  const pitch = rotX;
  const cp = Math.cos(pitch);
  const fwd: [number, number, number] = [
    Math.sin(yaw) * cp,
    -Math.sin(pitch),
    Math.cos(yaw) * cp,
  ];
  const pos: [number, number, number] = [
    center[0] - fwd[0] * dist,
    center[1] - fwd[1] * dist,
    center[2] - fwd[2] * dist,
  ];
  return { pos, yaw, pitch, fov: BASE_FOV_PX * zoom };
}

// camera forward / right / up basis from yaw+pitch (FPS movement frame).
function cameraBasis(yaw: number, pitch: number) {
  const cp = Math.cos(pitch);
  const sp = Math.sin(pitch);
  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);
  const forward: [number, number, number] = [sy * cp, -sp, cy * cp];
  // right is yaw-only (so strafing stays level regardless of pitch)
  const right: [number, number, number] = [cy, 0, -sy];
  const up: [number, number, number] = [sy * sp, cp, cy * sp];
  return { forward, right, up };
}

export default function Play() {
  useDocMeta(() => ({
    title: "Watch it think",
    description:
      "A modgrad learned planner (Value Iteration Network) solving mazes in your browser. Trained on solved mazes, it reads the maze image, learns the walls and goal, and plans its own route with no solver at inference — and generalizes to mazes bigger than it trained on. Runs client-side as a bit-exact wasm reimplementation of the modgrad SDK.",
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
  const [reachedNow, setReachedNow] = createSignal(false); // reached goal this step

  // ── NEW SDK-feature signals (all optional / feature-detected) ──
  const [plasticityAvailable, setPlasticityAvailable] = createSignal(false);
  const [telemetry, setTelemetry] = createSignal<BrainTelemetry | null>(null);
  const [plasticDelta, setPlasticDelta] = createSignal(0);
  const [signal, setSignal] = createSignal(0);
  const [loss, setLoss] = createSignal<number | undefined>(undefined);
  const [lr, setLr] = createSignal<number | undefined>(undefined);
  const [episodic, setEpisodic] = createSignal<StepMsg["episodic"] | null>(null);

  // ── drive-mode + graded neuromodulation + efficiency (Part A) ──
  const [driveMode, setDriveMode] = createSignal<DriveMode>("normal");
  // user-settable board size (odd). The retina is arbitrary-resolution + the VIN
  // is size-agnostic, so this needs no retrain — just a fresh maze at that size.
  const [mazeSize, setMazeSize] = createSignal<number>(PLAY_SIZE);
  const MAZE_SIZES = [9, 11, 13, 15, 21] as const;
  // the consolidated factual reference (opened from the 3D panel's "what am I
  // seeing?" link). Keeps the live panels value-only; the prose lives here.
  const [infoOpen, setInfoOpen] = createSignal(false);
  const [neuromod, setNeuromod] = createSignal<StepMsg["neuromod"]>(undefined);
  const [efficiency, setEfficiency] = createSignal<number | undefined>(undefined);
  const [efficiencyFinal, setEfficiencyFinal] = createSignal<number | undefined>(undefined);

  // the board the maze + agent are drawn at — the worker runs on PLAY_SIZE (11),
  // so the page must render at the same resolution (NOT reference.size=9).
  let SIZE = PLAY_SIZE;

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
    SIZE = PLAY_SIZE; // the worker runs on the 11×11 board (reference.size=9 is copy-only)
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
        setPlasticityAvailable(!!msg.plastic);
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
        size: PLAY_SIZE, // run on the 11×11 board (the engine retina is arbitrary-res)
        mode: driveMode(), // the starting drive-mode (switchable live via setMode)
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
    // free-fly / fullscreen teardown (idempotent — safe if never entered)
    document.removeEventListener("fullscreenchange", onFullscreenChange);
    document.removeEventListener("pointerlockchange", onPointerLockChange);
    window.removeEventListener("mousemove", flyMouseMove);
    window.removeEventListener("keydown", flyKeyDown);
    window.removeEventListener("keyup", flyKeyUp);
    if (document.fullscreenElement === brainContainer) document.exitFullscreen?.();
    if (document.pointerLockElement) document.exitPointerLock?.();
  });

  // ── solve loop ──────────────────────────────────────────────────
  function loadMaze(m: Maze) {
    // the worker may have changed the board size — follow it from the grid so the
    // canvas always draws at the maze's true dimensions.
    SIZE = Math.round(Math.sqrt(m.grid.length)) || SIZE;
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
      // clear the per-step NEW signals on a fresh maze (keep plasticity stats —
      // those live inside PlasticityPanel and survive Restart / New maze)
      setReachedNow(false);
      setTelemetry(null);
      setPlasticDelta(0);
      setSignal(0);
      setEpisodic(null);
      setNeuromod(undefined);
      setEfficiency(undefined);
    });
  }

  // ── drive-mode (Part A) ──
  // Switch how the agent decides its next move, live (no reload): tell the
  // worker, which flips a single parameter gating the decided move + completion.
  function changeMode(mode: DriveMode) {
    if (mode === driveMode()) return;
    setDriveMode(mode);
    worker?.postMessage({ type: "setMode", mode });
  }
  // one-line description of the active mode, for the toggle caption.
  const modeDesc = (): string => {
    switch (driveMode()) {
      case "easy":
        return "The learned planner with a generous bio-escape: a frustration signal heats exploration and a per-cell bias adapts to break out of loops. Effectively always finishes.";
      case "normal":
        return "The learned planner with the standard bio-escape — plasticity + neuromodulator help it out of the occasional loop. Mostly finishes.";
      case "hard":
        return "Tighter budget and only a faint escape. On a tricky maze it can run out of moves and fail. Closer to the planner on its own.";
      case "hardcore":
        return "The raw frozen planner — no plastic bias, no neuromodulator. It plans purely from the maze image and can get stuck and fail. The honest \"how good is it really\".";
    }
  };

  // ── board size (live) ──
  // Resize the maze with no reload / no retrain — the worker resizes its buffers
  // and hands back a fresh maze at the new size; loadMaze follows the dimensions.
  function changeSize(size: number) {
    if (size === mazeSize() || status() !== "ready") return;
    setMazeSize(size);
    pause();
    setGenerating(true);
    worker?.postMessage({ type: "setSize", size });
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
      // NEW engine signals — feature-detected, degrade gracefully when absent
      setReachedNow(!!msg.reached);
      setTelemetry(msg.telemetry ?? null);
      setPlasticDelta(msg.plasticDelta ?? 0);
      setSignal(msg.signal ?? 0);
      setLoss(msg.loss);
      setLr(msg.lr);
      setEpisodic(msg.episodic ?? null);
      // drive-mode signals (Part A) — graded tier + efficiency (live + per-solve)
      setNeuromod(msg.neuromod);
      setEfficiency(msg.efficiency);
      if (msg.efficiencyFinal != null) setEfficiencyFinal(msg.efficiencyFinal);
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
    // honest mode: the episode gave up (over budget, no nudge) → mark it lost and
    // move on to a fresh maze. Solve-rate climbs as the move-head learns.
    if (msg.lost || msg.done) {
      const looping = playing();
      setRunState("stuck");
      if (looping) {
        playTimer = window.setTimeout(() => requestNewMaze(), TICK_MS * 8);
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
  // revert the live-plastic readout to the frozen weights (worker no-ops on old engine)
  function resetPlasticity() {
    worker?.postMessage({ type: "resetPlasticity" });
    batch(() => {
      setPlasticDelta(0);
      setSignal(0);
    });
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

  // ── maze + agent + trail + attention overlay (delegated to viz/maze.ts) ──
  // The caller owns DPR + clear (ctxOf); the maze module draws everything pure
  // from the snapshot we hand it. Theme colours are resolved from CSS vars so it
  // tracks the paper / dark theme.
  function drawMaze() {
    const m = maze();
    if (!m || !mazeCanvas) return;
    const W = mazeCanvas.clientWidth || 360;
    const H = W; // square
    const ctx = ctxOf(mazeCanvas, W, H);

    const theme: MazeTheme = {
      wall: cssVar("--text", "#1B1822"),
      open: cssVar("--bg-card", "#fff"),
      line: cssVar("--line-2", "#D8D2C3"),
      accent: cssVar("--accent", "#6243D9"),
    };
    const state: MazeRenderState = {
      agent: agent(),
      visited: visited(),
      attn: attnMap,
      route: routeCells,
      verdict: verdict(),
      committedMove: committedMove(),
      thinking: runState() === "thinking",
      tick: tickIdx(),
      ticksTotal: ticksTotal(),
    };
    drawMazeModule(
      ctx,
      W,
      H,
      { grid: m.grid, start: m.start, end: m.end },
      SIZE,
      state,
      theme,
    );
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

  // The 3D particle brain renderer (viz/brain3d.ts) owns its neuron layout,
  // connectome and vision pathway; we hand it the camera + current tick/vision
  // each frame. setRef() installs the structure once brainRef() loads.
  const brain3d = createBrain3D({ regionColors: REGION_COLORS });

  let brainRotY = 0.7;
  let brainRotX = -0.32;
  let brainZoom = 0.66; // wheel / pinch zoom — start zoomed OUT so the whole brain
  // is in frame (people fullscreen + scroll in to see more)
  let brainDragging = false;
  let brainPointers = new Map<number, { x: number; y: number }>(); // active pointers (pinch)
  let pinchDist = 0;
  const [brainAutoRotate, setBrainAutoRotate] = createSignal(true);
  // co-spike line persistence: 0 = instant, 1 = long trail (per-frame decay)
  const [spikeHold, setSpikeHold] = createSignal(0.55);
  let brainTime = 0; // seconds, advanced by the rAF loop for the idle shimmer
  let brainReplay = 0; // fractional tick index; loops the trace so neurons keep
  // spiking between steps instead of freezing on the last tick.
  let brainVisible = true; // gated by IntersectionObserver — skip drawing off-screen
  let brainIO: IntersectionObserver | undefined;

  // ── free-fly camera (Half-Life style, active in fullscreen) ──
  // A plain mutable, read in the rAF loop (like brainRot*). In windowed mode it
  // is derived from the orbit params each frame; in fullscreen it is driven by
  // pointer-lock mouse-look + WASD and advanced in the rAF loop.
  let camera: Camera = orbitToCamera(brainRotX, brainRotY, brainZoom);
  const [isFullscreen, setIsFullscreen] = createSignal(false);
  let pointerLocked = false;
  // held movement keys (FPS): forward/back/strafe/up/down + sprint
  const keyHeld = {
    fwd: false,
    back: false,
    left: false,
    right: false,
    up: false,
    down: false,
    fast: false,
  };
  let lastFlyT = 0; // rAF timestamp of the last fly integration (for dt)
  const PITCH_LIMIT = (85 * Math.PI) / 180;
  const FLY_SPEED = 6; // world units / second (the brain spans ~10 units)
  let brainContainer: HTMLDivElement | undefined; // fullscreen target (wraps canvas)

  // brain tick aligned to the current animation tick (held at the last brain
  // tick if the brain exited earlier than the solver's tick count).
  const curBrainTick = (): BrainTick | null => {
    const bt = brainTrace();
    if (!bt || bt.ticks.length === 0) return null;
    return bt.ticks[Math.min(tickIdx(), bt.ticks.length - 1)];
  };

  // Tick shown in the 3D cloud. While a step animates it follows the live tick;
  // when idle it REPLAYS the trace's ticks on a loop (brainReplay), so the
  // neurons keep spiking instead of holding a single frozen frame.
  const displayBrainTick = (): BrainTick | null => {
    const bt = brainTrace();
    if (!bt || bt.ticks.length === 0) return null;
    const idx =
      runState() === "thinking"
        ? Math.min(tickIdx(), bt.ticks.length - 1)
        : Math.floor(brainReplay) % bt.ticks.length;
    return bt.ticks[idx];
  };

  // ── the 8-region brain as a rotating 3D particle cloud (delegated to
  // viz/brain3d.ts). The module owns the neuron cloud / connectome / vision
  // pathway; we own canvas sizing + DPR (ctxOf) and the camera state. ──
  function drawBrain3D() {
    const bref = brainRef();
    if (!brainCanvas || !bref) return;
    const fs = isFullscreen();
    // fullscreen → fill the viewport; windowed → fluid width, capped height.
    const W = fs
      ? window.innerWidth
      : brainCanvas.clientWidth || 720;
    const H = fs
      ? window.innerHeight
      : Math.round(Math.max(340, Math.min(520, W * 0.6)));
    const ctx = ctxOf(brainCanvas, W, H);
    // windowed = orbit-derived camera; fullscreen = the live free-fly camera.
    if (!fs) camera = orbitToCamera(brainRotX, brainRotY, brainZoom, BRAIN_CENTER);
    brain3d.draw(ctx, W, H, {
      tick: displayBrainTick(),
      vision: visionMaps,
      camera,
      spikeHold: spikeHold(),
      time: brainTime,
      showLabels: fs, // tech tour: region/vision/architecture labels while flying
    });
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

  // ── NEW-engine telemetry → RegionTel[] adapter ──
  // The worker emits per-region {name, activity, neuromod, certainty}; the
  // RegionTelemetry / SdkFeatures components want {name, rms, peak, dopamine?,
  // pain?}. We map what's present (activity→rms/peak, neuromod sign→dopamine /
  // pain) and otherwise leave it absent so the component falls back to the tick
  // acts it always has. Returns undefined when the engine emits no telemetry.
  const regionTel = (): RegionTel[] | undefined => {
    const tel = telemetry();
    if (!tel || !tel.regions.length) return undefined;
    return tel.regions.map((r, i) => {
      const name = r.name ?? brainRef()?.regions[i]?.name ?? `r${i}`;
      const act = r.activity ?? 0;
      const out: RegionTel = { name, rms: act, peak: act };
      if (typeof r.neuromod === "number") {
        if (r.neuromod >= 0) out.dopamine = r.neuromod;
        else out.pain = -r.neuromod;
      }
      return out;
    });
  };

  // the brain's adaptive-compute exit gate λ for the current step (NEW engine's
  // outer exit gate if present, else the per-tick CTM exit scalar).
  const exitLambda = (): number | null => {
    const eg = telemetry()?.exitGate;
    if (typeof eg === "number") return eg;
    const e = curBrainTick()?.exit;
    return e ?? null;
  };

  // ticks the CTM actually ran this step (the trace length / ticksUsed).
  const ticksUsed = (): number =>
    brainTrace()?.ticksUsed ?? ticksTotal();

  // assembled feature-state for <SdkFeatures> — every field a real signal.
  const sdkFeatureState = () => ({
    ticksUsed: ticksUsed(),
    exitLambda: exitLambda(),
    visionActive: !!(visionMaps && visionMaps.length),
    plasticDelta: plasticDelta(),
    signal: signal(),
    telemetry: regionTel(),
    episodic: episodic(),
  });

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

  // ── fullscreen + Half-Life free-fly camera ──────────────────────
  // Mouse-look (while pointer-locked): yaw += dx, pitch += dy (clamped).
  // (forward.y = −sin(pitch), so mouse-up must DECREASE pitch to look up →
  // pitch += movementY, since movementY<0 on mouse-up.)
  const flyMouseMove = (e: MouseEvent) => {
    if (!pointerLocked) return;
    const SENS = 0.0022;
    camera.yaw += e.movementX * SENS;
    camera.pitch = Math.max(
      -PITCH_LIMIT,
      Math.min(PITCH_LIMIT, camera.pitch + e.movementY * SENS),
    );
  };
  // WASD + Q/E (or Space/Shift) movement; Shift = sprint.
  const setKey = (code: string, down: boolean): boolean => {
    switch (code) {
      case "KeyW":
      case "ArrowUp":
        keyHeld.fwd = down;
        return true;
      case "KeyS":
      case "ArrowDown":
        keyHeld.back = down;
        return true;
      case "KeyA":
      case "ArrowLeft":
        keyHeld.left = down;
        return true;
      case "KeyD":
      case "ArrowRight":
        keyHeld.right = down;
        return true;
      case "KeyE":
      case "Space":
        keyHeld.up = down;
        return true;
      case "KeyQ":
        keyHeld.down = down;
        return true;
      case "ShiftLeft":
      case "ShiftRight":
        keyHeld.fast = down;
        return true;
      default:
        return false;
    }
  };
  const flyKeyDown = (e: KeyboardEvent) => {
    if (!isFullscreen()) return;
    if (setKey(e.code, true)) e.preventDefault();
  };
  const flyKeyUp = (e: KeyboardEvent) => {
    if (setKey(e.code, false)) e.preventDefault();
  };
  // integrate held keys into camera.pos for one frame (called from the rAF loop)
  function advanceFly(now: number) {
    if (!isFullscreen()) {
      lastFlyT = now;
      return;
    }
    const dt = lastFlyT ? Math.min(0.05, (now - lastFlyT) / 1000) : 0;
    lastFlyT = now;
    if (dt <= 0) return;
    const { forward, right, up } = cameraBasis(camera.yaw, camera.pitch);
    const speed = FLY_SPEED * (keyHeld.fast ? 2.6 : 1) * dt;
    let fx = 0,
      fy = 0,
      fz = 0;
    if (keyHeld.fwd) {
      fx += forward[0];
      fy += forward[1];
      fz += forward[2];
    }
    if (keyHeld.back) {
      fx -= forward[0];
      fy -= forward[1];
      fz -= forward[2];
    }
    if (keyHeld.right) {
      fx += right[0];
      fy += right[1];
      fz += right[2];
    }
    if (keyHeld.left) {
      fx -= right[0];
      fy -= right[1];
      fz -= right[2];
    }
    if (keyHeld.up) {
      fx += up[0];
      fy += up[1];
      fz += up[2];
    }
    if (keyHeld.down) {
      fx -= up[0];
      fy -= up[1];
      fz -= up[2];
    }
    const len = Math.hypot(fx, fy, fz);
    if (len > 1e-6) {
      camera.pos[0] += (fx / len) * speed;
      camera.pos[1] += (fy / len) * speed;
      camera.pos[2] += (fz / len) * speed;
    }
  }
  const onPointerLockChange = () => {
    pointerLocked = document.pointerLockElement === brainCanvas;
  };
  // click the canvas (in fullscreen) → grab the mouse for look control
  const brainCanvasClick = () => {
    if (isFullscreen() && !pointerLocked) brainCanvas?.requestPointerLock?.();
  };
  function clearFlyKeys() {
    keyHeld.fwd = keyHeld.back = keyHeld.left = keyHeld.right = false;
    keyHeld.up = keyHeld.down = keyHeld.fast = false;
  }
  const onFullscreenChange = () => {
    const active = document.fullscreenElement === brainContainer;
    setIsFullscreen(active);
    if (active) {
      // entering: seed the free-fly camera from the current orbit view so the
      // transition is seamless, then arm look + move listeners.
      camera = orbitToCamera(brainRotX, brainRotY, brainZoom, BRAIN_CENTER);
      lastFlyT = 0;
      window.addEventListener("mousemove", flyMouseMove);
      window.addEventListener("keydown", flyKeyDown);
      window.addEventListener("keyup", flyKeyUp);
      drewOnce = false; // force an immediate redraw at the new (viewport) size
    } else {
      // exiting: tear everything down and restore the windowed canvas size.
      window.removeEventListener("mousemove", flyMouseMove);
      window.removeEventListener("keydown", flyKeyDown);
      window.removeEventListener("keyup", flyKeyUp);
      if (document.pointerLockElement) document.exitPointerLock?.();
      pointerLocked = false;
      clearFlyKeys();
      drewOnce = false;
    }
  };
  function toggleFullscreen() {
    if (document.fullscreenElement) {
      document.exitFullscreen?.();
    } else {
      brainContainer?.requestFullscreen?.();
    }
  }

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
  // hand the brain structure to the 3D renderer whenever it loads/changes
  createEffect(() => {
    const bref = brainRef();
    if (bref) brain3d.setRef(bref);
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
      const fs = isFullscreen();
      // fullscreen counts as visible — the IO gate must not pause free-fly.
      if (!brainVisible && !fs) return; // off-screen → don't burn frames
      const thinking = runState() === "thinking";
      if (thinking) lastThinking = now;
      // integrate WASD movement every frame while flying (independent of the
      // draw cadence, so motion is smooth at the 30fps cap).
      advanceFly(now);
      const moving =
        keyHeld.fwd ||
        keyHeld.back ||
        keyHeld.left ||
        keyHeld.right ||
        keyHeld.up ||
        keyHeld.down;
      // active = thinking, within settle, orbiting, flying (always-on in
      // fullscreen so look/move stay live), or we haven't drawn once yet.
      // a loaded trace keeps the loop alive so the neurons can replay (unless the
      // user prefers reduced motion → hold a static frame).
      const hasTrace = (brainTrace()?.ticks.length ?? 0) > 1;
      const active =
        thinking ||
        now - lastThinking < SETTLE_MS ||
        brainDragging ||
        fs ||
        moving ||
        !drewOnce ||
        (hasTrace && !reduced);
      if (!active) return; // idle → hold the last drawn frame
      if (drewOnce && now - lastDraw < MIN_FRAME_MS) return; // 30fps cap
      lastDraw = now;
      drewOnce = true;
      brainTime += 0.033;
      // idle (settled, not mid-step) → advance the replay so the cloud keeps
      // spiking through the brain's real ticks instead of freezing.
      if (!thinking && now - lastThinking >= SETTLE_MS && hasTrace && !reduced)
        brainReplay += 0.25;
      // auto-spin only in windowed mode (fullscreen is user-driven free-fly).
      if (!reduced && !fs && brainAutoRotate() && !brainDragging)
        brainRotY += 0.009;
      drawBrain3D();
    };
    raf = requestAnimationFrame(spin);
    document.addEventListener("fullscreenchange", onFullscreenChange);
    document.addEventListener("pointerlockchange", onPointerLockChange);
    onCleanup(() => {
      cancelAnimationFrame(raf);
      document.removeEventListener("fullscreenchange", onFullscreenChange);
      document.removeEventListener("pointerlockchange", onPointerLockChange);
    });
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
          <span class="grad-text">learned planner</span> solving mazes live in
          your browser. A Value Iteration Network, trained on solved mazes, reads
          the maze image and plans its own route — with{" "}
          <span class="grad-text">no solver at inference</span>. It learns the
          walls and the goal from the picture, then propagates value to navigate,
          and it generalizes to mazes bigger than it ever trained on. The forward
          pass is a bit-exact in-browser reimplementation of the SDK.
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

            {/* ── drive-mode toggle (Part A): segmented control ── */}
            <div class="mt-4 pt-4 border-t border-line">
              <div class="flex items-center justify-between mb-2">
                <div class="eyebrow">Difficulty</div>
                <div class="font-mono text-xs text-mute tabular-nums">
                  maze #{Math.max(1, mazeNum())} · step {stepCount()}
                </div>
              </div>
              <div
                class="flex rounded-lg overflow-hidden border border-line"
                role="group"
                aria-label="how the agent decides its next move"
              >
                <For each={["easy", "normal", "hard", "hardcore"] as DriveMode[]}>
                  {(m) => {
                    const active = () => driveMode() === m;
                    return (
                      <button
                        class="flex-1 text-xs font-mono px-2 py-1.5 capitalize transition-colors"
                        style={{
                          background: active() ? "var(--accent)" : "transparent",
                          color: active() ? "#fff" : "var(--text-mute)",
                        }}
                        onClick={() => changeMode(m)}
                        disabled={status() !== "ready"}
                        aria-pressed={active()}
                      >
                        {m}
                      </button>
                    );
                  }}
                </For>
              </div>
              <div class="text-sm text-dim mt-2 leading-relaxed">{modeDesc()}</div>

              {/* ── board-size selector: arbitrary-resolution retina, no retrain ── */}
              <div class="flex items-center justify-between mt-4">
                <div class="eyebrow">Maze size</div>
                <div
                  class="flex rounded-lg overflow-hidden border border-line"
                  role="group"
                  aria-label="board size"
                >
                  <For each={MAZE_SIZES}>
                    {(s) => {
                      const active = () => mazeSize() === s;
                      return (
                        <button
                          class="text-xs font-mono px-2.5 py-1 tabular-nums transition-colors"
                          style={{
                            background: active() ? "var(--accent)" : "transparent",
                            color: active() ? "#fff" : "var(--text-mute)",
                          }}
                          onClick={() => changeSize(s)}
                          disabled={status() !== "ready"}
                          aria-pressed={active()}
                          title={`${s}×${s} board`}
                        >
                          {s}
                        </button>
                      );
                    }}
                  </For>
                </div>
              </div>
              <div class="text-[.72rem] text-mute font-mono mt-1.5 leading-relaxed">
                Trained on 9×9, run at any size. The planner generalizes to bigger
                boards by running more value-iteration rounds — nothing is retrained
                (solve-rate drops with size, honestly).
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

          {/* region telemetry: anchors the left column under the move bars,
              expanded by default so the columns stay balanced. */}
          <Show when={brainRef()}>
            <RegionTelemetry
              regions={brainRef()!.regions}
              acts={curBrainTick()?.acts ?? null}
              global={curBrainTick()?.global ?? 0}
              exitLambda={exitLambda()}
              ticksUsed={ticksUsed()}
              ticksTotal={ticksTotal() || 16}
              telemetry={regionTel()}
              defaultOpen
            />
          </Show>
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
            Each dot is a neuron coloured by region; lines are connectome edges,
            lit when two regions co-fire. The cyan grids are the retina → V1 → V2
            → V4 feature maps.{" "}
            <button
              type="button"
              class="text-accent whitespace-nowrap"
              onClick={() => setInfoOpen(true)}
            >
              what am I seeing? →
            </button>
          </p>

          <div
            ref={brainContainer}
            class="rounded-lg overflow-hidden relative"
            classList={{ "rounded-none": isFullscreen() }}
            style={{
              background: "#0a0912",
              ...(isFullscreen()
                ? { width: "100vw", height: "100vh" }
                : {}),
            }}
          >
            <canvas
              ref={(el) => {
                brainCanvas = el;
                el.addEventListener("wheel", brainWheel, { passive: false });
                brainIO = new IntersectionObserver(
                  (es) => (brainVisible = es[0]?.isIntersecting ?? true),
                );
                brainIO.observe(el);
              }}
              class="block"
              classList={{
                "w-full cursor-grab active:cursor-grabbing": !isFullscreen(),
                "w-full h-full cursor-none": isFullscreen(),
              }}
              style={{ "touch-action": "none" }}
              onClick={brainCanvasClick}
              onPointerDown={brainPointerDown}
              onPointerMove={brainPointerMove}
              onPointerUp={brainPointerUp}
              onPointerCancel={brainPointerUp}
              onLostPointerCapture={brainPointerUp}
              aria-label="the eight-region brain computing — drag to rotate, scroll to zoom; fullscreen for WASD free-fly"
            />
            <div class="absolute top-2 right-2 flex items-center gap-1.5">
              <Show when={!isFullscreen()}>
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
              </Show>
              <button
                class="text-[.68rem] font-mono px-2 py-1 rounded"
                style={{
                  background: "rgba(255,255,255,0.08)",
                  color: "rgba(255,255,255,0.75)",
                  "backdrop-filter": "blur(4px)",
                }}
                onClick={toggleFullscreen}
                title={isFullscreen() ? "exit fullscreen (Esc)" : "fly through the brain"}
              >
                {isFullscreen() ? "⛶ exit" : "⛶ fullscreen"}
              </button>
            </div>
            {/* windowed hint */}
            <Show when={!isFullscreen()}>
              <div class="absolute bottom-2 left-3 text-[.66rem] font-mono pointer-events-none" style={{ color: "rgba(255,255,255,0.4)" }}>
                drag to rotate · scroll / pinch to zoom · ⛶ to fly
              </div>
            </Show>
            {/* fullscreen free-fly hint + live SDK-feature HUD */}
            <Show when={isFullscreen()}>
              <div
                class="absolute bottom-3 left-4 text-[.7rem] font-mono pointer-events-none"
                style={{ color: "rgba(255,255,255,0.6)" }}
              >
                <Show
                  when={pointerLocked}
                  fallback={<span>click to look · WASD move · Esc exit</span>}
                >
                  <span>WASD move · Q/E up·down · Shift sprint · mouse look · Esc exit</span>
                </Show>
              </div>
              {/* ── COCKPIT HUD: value-only live stats, overlaid on the 3D view ── */}
              {/* TOP-LEFT: region telemetry + global-sync, exit-λ, ticks */}
              <div
                class="absolute top-3 left-3 text-[.64rem] font-mono tabular-nums rounded-lg flex flex-col gap-1.5"
                style={{
                  background: "rgba(10,9,18,0.72)",
                  "backdrop-filter": "blur(4px)",
                  border: "1px solid rgba(255,255,255,0.12)",
                  color: "rgba(255,255,255,0.82)",
                  padding: "10px 12px",
                  "min-width": "186px",
                  "pointer-events": "auto",
                }}
              >
                <div class="flex items-center justify-between" style={{ color: "rgba(255,255,255,0.55)" }}>
                  <span>regions</span>
                  <span>rms</span>
                </div>
                {(() => {
                  const tel = regionTel();
                  const names = brainRef()?.region_names ?? [];
                  // prefer live telemetry; fall back to the current tick's per-region acts
                  const rows = tel
                    ? tel.map((r, i) => ({ name: r.name, rms: r.rms ?? 0, idx: i }))
                    : names.map((nm, i) => {
                        const acts = curBrainTick()?.acts?.[i] ?? [];
                        const rms = acts.length
                          ? Math.sqrt(acts.reduce((a, v) => a + v * v, 0) / acts.length)
                          : 0;
                        return { name: nm, rms, idx: i };
                      });
                  return (
                    <For each={rows}>
                      {(r) => (
                        <div class="flex items-center gap-2">
                          <i
                            class="w-2 h-2 rounded-full inline-block shrink-0"
                            style={{ background: REGION_COLORS[r.idx % REGION_COLORS.length] }}
                          />
                          <span class="w-12 shrink-0" style={{ color: "rgba(255,255,255,0.7)" }}>
                            {shortRegion(r.name)}
                          </span>
                          <span class="flex-1 h-1.5 rounded-full overflow-hidden" style={{ background: "rgba(255,255,255,0.1)" }}>
                            <span
                              class="block h-full rounded-full"
                              style={{
                                width: `${Math.max(2, Math.min(1, r.rms) * 100)}%`,
                                background: REGION_COLORS[r.idx % REGION_COLORS.length],
                              }}
                            />
                          </span>
                          <span class="w-8 text-right" style={{ color: "rgba(255,255,255,0.55)" }}>
                            {r.rms.toFixed(2)}
                          </span>
                        </div>
                      )}
                    </For>
                  );
                })()}
                <div class="flex flex-wrap gap-x-3 gap-y-0.5 pt-1.5" style={{ "border-top": "1px solid rgba(255,255,255,0.1)", color: "rgba(255,255,255,0.6)" }}>
                  <span>global {brainGlobal().toFixed(2)}</span>
                  <Show when={exitLambda() != null}>
                    <span>λ {(exitLambda() ?? 0).toFixed(2)}</span>
                  </Show>
                  <span>ticks {ticksUsed()}/{ticksTotal() || 16}</span>
                  <span>vision {sdkFeatureState().visionActive ? "on" : "—"}</span>
                  <Show when={plasticityAvailable()}>
                    <span>Δ {plasticDelta().toFixed(3)}</span>
                  </Show>
                </div>
              </div>

              {/* TOP-RIGHT: move distribution + agreement + verdict + maze#/step.
                  Offset down so it clears the canvas stop/reset/fullscreen buttons. */}
              <div
                class="absolute right-3 text-[.64rem] font-mono tabular-nums rounded-lg flex flex-col gap-1.5"
                style={{
                  top: "44px",
                  background: "rgba(10,9,18,0.72)",
                  "backdrop-filter": "blur(4px)",
                  border: "1px solid rgba(255,255,255,0.12)",
                  color: "rgba(255,255,255,0.82)",
                  padding: "10px 12px",
                  "min-width": "176px",
                  "pointer-events": "auto",
                }}
              >
                <div class="flex items-center justify-between" style={{ color: "rgba(255,255,255,0.55)" }}>
                  <span>move</span>
                  <span>maze #{Math.max(1, mazeNum())} · step {stepCount()}</span>
                </div>
                <For each={MOVES}>
                  {(label, i) => {
                    const p = () => moveProbs()[i()] ?? 0;
                    const chosen = () => committedMove() === i();
                    return (
                      <div class="flex items-center gap-2">
                        <span
                          class="w-9 shrink-0"
                          style={{ color: chosen() ? "#fff" : "rgba(255,255,255,0.6)" }}
                        >
                          {label}
                        </span>
                        <span class="flex-1 h-2 rounded overflow-hidden" style={{ background: "rgba(255,255,255,0.1)" }}>
                          <span
                            class="block h-full rounded"
                            style={{
                              width: `${Math.max(2, p() * 100)}%`,
                              background: chosen() ? "var(--accent)" : "rgba(255,255,255,0.32)",
                            }}
                          />
                        </span>
                        <span class="w-8 text-right" style={{ color: chosen() ? "#fff" : "rgba(255,255,255,0.55)" }}>
                          {(p() * 100).toFixed(0)}%
                        </span>
                      </div>
                    );
                  }}
                </For>
                <div class="flex items-center justify-between pt-1.5" style={{ "border-top": "1px solid rgba(255,255,255,0.1)" }}>
                  <span style={{ color: "rgba(255,255,255,0.6)" }}>
                    agree {moveCount() ? Math.round((agreeCount() / moveCount()) * 100) : 0}% ({agreeCount()}/{moveCount()})
                  </span>
                  <Show when={verdict()}>
                    <span
                      class="px-1.5 py-0.5 rounded"
                      style={{
                        background:
                          verdict() === "ok"
                            ? "color-mix(in srgb, var(--accent) 30%, transparent)"
                            : "rgba(224,86,78,0.3)",
                        color: verdict() === "ok" ? "#fff" : "#ffb3ad",
                      }}
                    >
                      {verdict() === "ok"
                        ? "✓ path"
                        : verdict() === "wall"
                          ? "✕ wall"
                          : verdict() === "astray"
                            ? "→ astray"
                            : "· wait"}
                    </span>
                  </Show>
                </div>
              </div>

              {/* BOTTOM-CENTER: the control panel (transport + drive mode + size) */}
              <div
                class="absolute left-1/2 flex flex-col gap-2 rounded-lg"
                style={{
                  bottom: "16px",
                  transform: "translateX(-50%)",
                  background: "rgba(10,9,18,0.72)",
                  "backdrop-filter": "blur(4px)",
                  border: "1px solid rgba(255,255,255,0.12)",
                  padding: "10px 12px",
                  "pointer-events": "auto",
                }}
              >
                <div class="flex items-center gap-1.5 justify-center">
                  <button
                    class="text-[.66rem] font-mono px-2.5 py-1 rounded"
                    style={{ background: "var(--accent)", color: "#fff" }}
                    onClick={togglePlay}
                    disabled={status() !== "ready"}
                  >
                    {playing() ? "❚❚ Pause" : "▶ Play"}
                  </button>
                  <button
                    class="text-[.66rem] font-mono px-2.5 py-1 rounded"
                    style={{ background: "rgba(255,255,255,0.1)", color: "rgba(255,255,255,0.85)" }}
                    onClick={stepOnce}
                    disabled={status() !== "ready" || playing()}
                  >
                    Step
                  </button>
                  <button
                    class="text-[.66rem] font-mono px-2.5 py-1 rounded"
                    style={{ background: "rgba(255,255,255,0.1)", color: "rgba(255,255,255,0.85)" }}
                    onClick={restart}
                    disabled={status() !== "ready"}
                  >
                    Restart
                  </button>
                  <button
                    class="text-[.66rem] font-mono px-2.5 py-1 rounded"
                    style={{ background: "rgba(255,255,255,0.1)", color: "rgba(255,255,255,0.85)" }}
                    onClick={newMaze}
                    disabled={status() !== "ready"}
                  >
                    New maze
                  </button>
                </div>
                <div class="flex items-center justify-center gap-3 text-[.6rem] font-mono">
                  <div class="flex rounded overflow-hidden" style={{ border: "1px solid rgba(255,255,255,0.16)" }}>
                    <For each={["easy", "normal", "hard", "hardcore"] as DriveMode[]}>
                      {(m) => {
                        const active = () => driveMode() === m;
                        return (
                          <button
                            class="px-2 py-1 capitalize"
                            style={{
                              background: active() ? "var(--accent)" : "transparent",
                              color: active() ? "#fff" : "rgba(255,255,255,0.6)",
                            }}
                            onClick={() => changeMode(m)}
                            disabled={status() !== "ready"}
                            aria-pressed={active()}
                          >
                            {m}
                          </button>
                        );
                      }}
                    </For>
                  </div>
                  <div class="flex rounded overflow-hidden tabular-nums" style={{ border: "1px solid rgba(255,255,255,0.16)" }}>
                    <For each={MAZE_SIZES}>
                      {(s) => {
                        const active = () => mazeSize() === s;
                        return (
                          <button
                            class="px-2 py-1"
                            style={{
                              background: active() ? "var(--accent)" : "transparent",
                              color: active() ? "#fff" : "rgba(255,255,255,0.6)",
                            }}
                            onClick={() => changeSize(s)}
                            disabled={status() !== "ready"}
                            aria-pressed={active()}
                            title={`${s}×${s}`}
                          >
                            {s}
                          </button>
                        );
                      }}
                    </For>
                  </div>
                </div>
              </div>
            </Show>
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

        {/* ── live three-factor plasticity: the headline "learns" panel, kept
            open right under the brain ── */}
        <Show when={plasticityAvailable()}>
          <PlasticityPanel
            enabled={plasticityAvailable()}
            signal={signal()}
            neuromod={neuromod()}
            plasticDelta={plasticDelta()}
            verdict={verdict()}
            reached={reachedNow()}
            loss={loss()}
            lr={lr()}
            efficiency={efficiency()}
            efficiencyFinal={efficiencyFinal()}
            onReset={resetPlasticity}
          />
        </Show>

        {/* ── SDK features: collapsed by default; click the header to expand ── */}
        <SdkFeatures state={sdkFeatureState()} />
        </div>
        </Show>
      </div>

      {/* ── info: the factual reference, consolidated (no marketing) ── */}
      <div class="card mt-8">
        <button
          type="button"
          class="flex items-center justify-between w-full text-left"
          onClick={() => setInfoOpen((v) => !v)}
          aria-expanded={infoOpen()}
        >
          <div class="eyebrow flex items-center gap-1.5">
            <span
              class="inline-block transition-transform"
              style={{ transform: infoOpen() ? "rotate(90deg)" : "none" }}
            >
              ▸
            </span>
            Info
          </div>
          <span class="font-mono text-xs text-mute">how this works</span>
        </button>
        <Show when={infoOpen()}>
          <div class="mt-4 grid gap-5 text-sm leading-relaxed text-dim md:grid-cols-2">
            <div>
              <div class="eyebrow mb-1.5">The planner</div>
              A learned Value Iteration Network (Tamar et al., 2016). From the
              maze image it learns a per-cell reward and a traversability gate
              (the wall mask), runs K rounds of a learned 3×3 value backup to
              propagate value across the grid, then reads the decision
              ego-centrically at the agent's own cell. Nothing about the maze is
              hand-coded — it figures out walls, goal, and routing itself.
            </div>
            <div>
              <div class="eyebrow mb-1.5">Trained, then on its own</div>
              It was trained offline (supervised, on solver-labelled mazes, like
              the Sakana CTM) — the answers live in training only. At inference
              there is no solver: it plans from the picture alone. Because the
              value propagation is iterative, it generalizes to mazes bigger than
              it trained on by running more rounds — the VIN signature, and the
              reason this is planning rather than memorizing.
            </div>
            <div>
              <div class="eyebrow mb-1.5">Vision</div>
              retina → V1 → V2 → V4 feed the input region; the cyan grids are those
              feature maps, what the model actually sees. The retina is
              convolutional, so any maze size runs zero-shot. Nothing is retrained.
            </div>
            <div>
              <div class="eyebrow mb-1.5">Difficulty</div>
              Every level runs the same learned planner — the dial is how much
              bio-help it gets. Easy / Normal / Hard add a live escape: a
              frustration neuromodulator heats exploration and a per-cell plastic
              bias adapts to break out of loops, with a tightening budget as you
              go up. Hardcore turns the escape off entirely — the raw frozen
              planner, no plasticity, no neuromodulator. It can get stuck and
              fail. That's the planner's true, unaided solve-rate.
            </div>
            <div>
              <div class="eyebrow mb-1.5">Generalization</div>
              Trained on 9×9, it solves bigger boards (11×11, 13×13…) by running
              more value-iteration rounds, no retraining. Solve-rate drops with
              size, honestly, as the planning gets harder — but it does plan, not
              memorize.
            </div>
            <div>
              <div class="eyebrow mb-1.5">Engine</div>
              The browser engine reimplements modgrad's forward pass, bit-exact
              against the SDK (not the SDK itself on wasm yet). Loads only here.
            </div>
          </div>
        </Show>
      </div>
    </div>
  );
}

// The maze + brain drawing primitives (roundRect, start/goal markers) now live
// in their respective modules (viz/maze.ts, viz/brain3d.ts), so Play.tsx no
// longer carries inlined canvas helpers.
