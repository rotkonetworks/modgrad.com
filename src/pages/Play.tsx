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
import { softmax, cssVar } from "@/play/viz";
import { createBrain3D } from "@/play/viz/brain3d";
import type { Camera } from "@/play/viz/brain3d";
import { drawMaze as drawMazeModule } from "@/play/viz/maze";
import type { MazeRenderState, MazeTheme, TickRoute, Cell } from "@/play/viz/maze";
import SdkFeatures from "@/play/components/SdkFeatures";
import RegionTelemetry from "@/play/components/RegionTelemetry";

// ── types mirrored from the worker ────────────────────────────────
type BrainTick = { acts: number[][]; global: number; exit: number | null };
type BrainTrace = { ticks: BrainTick[]; ticksUsed: number };
type RetinaMap = { name: string; channels: number; h: number; w: number; data: number[] };
/** One maze and everything the brain computed looking at it (see worker.ts). */
type MazeMsg = {
  type: "maze";
  grid: number[];
  start: Cell;
  end: Cell;
  shortest: Cell[]; // reference only — never shown to the brain
  routes: TickRoute[]; // the brain's decoded route at each internal tick
  agree: number; // steps of the final route that match the shortest route
  firstSteps: number[][]; // per tick: first-step logits [U, D, L, R, wait]
  brain: BrainTrace | null;
  vision: RetinaMap[] | null;
};
// structure of the 8-region brain, read from the *_reference.json exports
type BrainConn = { from: number[]; to: number; receives_observation: boolean };
type BrainRef = {
  size: number;
  ticks: number;
  out_dims: number;
  region_names: string[];
  regions: { name: string; d_model: number }[];
  connections: BrainConn[];
  n_global_sync: number;
};

type Status = "loading" | "ready" | "error";
type RunState = "idle" | "thinking";

const TICK_MS = 160; // pace of the tick-by-tick reveal
// The brain's retina is built for a fixed 9×9 image (see worker.ts), so the
// demo runs 9×9 only.
const SIZE = 9;
// Bump when the shipped model weights change, to bust the browser cache on the
// non-content-hashed /models/*.json.
const MODEL_VER = "20260703-full-brain-training";
const STEP_LABELS = ["Up", "Down", "Left", "Right", "Wait"] as const;

// Measured offline (headless, the shipped wasm + weights, 200 random 9×9 mazes,
// the brain's final-tick route from the start image, argmax per step).
const MEASURED = {
  n: 200,
  firstStep: "199",
  meanMatch: "4.3",
  meanLen: "17.4",
  fullyCorrect: "0",
};

// ── camera helpers (build to brain3d.ts's documented free-fly interface) ──
// brain3d's view transform is  view = Rx(-pitch)·Ry(-yaw)·(p − pos), so the
// camera forward (the world dir that maps to view +z) is
//   fwd = Ry(yaw)·Rx(pitch)·(0,0,1) = ( sin(yaw)cos(pitch), −sin(pitch),
//                                       cos(yaw)cos(pitch) ).
// Windowed orbit derives an equivalent free-fly camera that looks at `center`
// from `dist` along those angles.
// centre of the anatomical scene (brain + eyes + the image in front of them)
const BRAIN_CENTER: [number, number, number] = [0.9, 0.6, 0];
const ORBIT_DIST = 13; // distance the orbit camera sits from the brain centre
const BASE_FOV_PX = 720 * 0.72; // matches the old fov = W*0.72*zoom feel at W≈720

function orbitToCamera(
  rotX: number,
  rotY: number,
  zoom: number,
  center: [number, number, number] = BRAIN_CENTER,
  dist: number = ORBIT_DIST,
): Camera {
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
      "A small 8-region CTM brain looks at a maze image and predicts a 20-step route over its internal ticks, live in your browser. Nothing else plans or decides. It is early: it usually gets the first steps right and then drifts. Runs client-side as the modgrad SDK, compiled to WebAssembly.",
    path: "/play",
  }));

  const reduced =
    typeof matchMedia !== "undefined" &&
    matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ── reactive state ──────────────────────────────────────────────
  const [status, setStatus] = createSignal<Status>("loading");
  const [errMsg, setErrMsg] = createSignal("");
  const [runState, setRunState] = createSignal<RunState>("idle");
  const [generating, setGenerating] = createSignal(false);

  // the current maze and the brain's output for it
  const [result, setResult] = createSignal<MazeMsg | null>(null);
  const [mazeNum, setMazeNum] = createSignal(0);

  // tick cursor: drives the maze route, the first-step bars and the 3D view
  const [tickIdx, setTickIdx] = createSignal(0);
  const [ticksTotal, setTicksTotal] = createSignal(0);
  // true once the user scrubs: the 3D view then holds the chosen tick instead
  // of looping through the trace.
  const [pinned, setPinned] = createSignal(false);

  const [brainRef, setBrainRef] = createSignal<BrainRef | null>(null);
  const [brainOn, setBrainOn] = createSignal(false);
  const [brainTrace, setBrainTrace] = createSignal<BrainTrace | null>(null);
  let visionMaps: RetinaMap[] | null = null; // read in the rAF loop, not reactive

  // session tally over the mazes shown on this page
  const [seen, setSeen] = createSignal(0);
  const [firstRight, setFirstRight] = createSignal(0);
  const [allRight, setAllRight] = createSignal(0);
  const [matchSum, setMatchSum] = createSignal(0);

  // Brain size selector: four trained variants of the same 8-region brain at
  // different cortical widths, hot-swapped in the worker without a reload.
  const BRAIN_MODELS = [
    { key: "c16", label: "16", file: "brain_small_weights.json", ref: "brain_small_reference.json" },
    { key: "c32", label: "32", file: "brain_solver_weights.json", ref: "brain_solver_reference.json" },
    { key: "c64", label: "64", file: "brain_large_weights.json", ref: "brain_large_reference.json" },
    { key: "c128", label: "128", file: "brain_xl_weights.json", ref: "brain_xl_reference.json" },
  ] as const;
  type BrainModelKey = (typeof BRAIN_MODELS)[number]["key"];
  const [modelSize, setModelSize] = createSignal<BrainModelKey>("c32");
  const [modelSwitching, setModelSwitching] = createSignal(false);
  const [infoOpen, setInfoOpen] = createSignal(false);

  // ── worker plumbing ─────────────────────────────────────────────
  let worker: Worker | null = null;
  let tickTimer: number | undefined;

  onMount(async () => {
    try {
      const r = await fetch(`/models/brain_solver_reference.json?v=${MODEL_VER}`);
      setBrainRef(await r.json()); // drives the 3D viz structure
    } catch (e) {
      setStatus("error");
      setErrMsg("Could not load the brain reference.");
      return;
    }

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
        if (!msg.brain) {
          setStatus("error");
          setErrMsg("This browser's engine build has no brain entry points.");
          return;
        }
        setStatus("ready");
        requestNewMaze();
        return;
      }
      if (msg.type === "error") {
        setStatus("error");
        setErrMsg(msg.message || "The brain failed to load.");
        return;
      }
      if (msg.type === "brainReloaded") {
        setModelSwitching(false);
        if (msg.ok) requestNewMaze();
        return;
      }
      if (msg.type === "maze") onMaze(msg as MazeMsg);
    };

    try {
      const br = await fetch(`/models/brain_solver_weights.json?v=${MODEL_VER}`).then((r) =>
        r.text(),
      );
      worker.postMessage({ type: "init", brainWeights: br });
    } catch (e) {
      setStatus("error");
      setErrMsg("Could not load the trained brain weights.");
    }
  });

  onCleanup(() => {
    worker?.terminate();
    clearTimeout(tickTimer);
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

  // ── one maze, one look ──────────────────────────────────────────
  function requestNewMaze() {
    if (!worker || status() !== "ready") return;
    clearTimeout(tickTimer);
    setGenerating(true);
    worker.postMessage({ type: "newMaze" });
  }

  function onMaze(msg: MazeMsg) {
    visionMaps = msg.vision;
    const n = msg.routes.length;
    const final = msg.routes[n - 1];
    const correct =
      !!final && final.end === "goal" && msg.agree === msg.shortest.length;
    batch(() => {
      setGenerating(false);
      setMazeNum((k) => k + 1);
      setResult(msg);
      setBrainTrace(msg.brain);
      setTicksTotal(n);
      setPinned(false);
      setSeen((k) => k + 1);
      if (msg.agree >= 1) setFirstRight((k) => k + 1);
      if (correct) setAllRight((k) => k + 1);
      setMatchSum((k) => k + msg.agree);
    });
    playTicks();
  }

  // reveal the brain's ticks one by one, then hold the final tick
  function playTicks() {
    clearTimeout(tickTimer);
    const n = ticksTotal();
    setPinned(false);
    if (reduced || n <= 1) {
      setTickIdx(Math.max(0, n - 1));
      setRunState("idle");
      return;
    }
    setRunState("thinking");
    const step = (i: number) => {
      setTickIdx(i);
      if (i >= n - 1) {
        setRunState("idle");
        return;
      }
      tickTimer = window.setTimeout(() => step(i + 1), TICK_MS);
    };
    step(0);
  }

  function scrubTo(i: number) {
    clearTimeout(tickTimer);
    batch(() => {
      setRunState("idle");
      setPinned(true);
      setTickIdx(Math.max(0, Math.min(ticksTotal() - 1, i)));
    });
  }

  // Hot-swap the trained brain size: fetch the variant's export + reference and
  // hand the weights to the worker, which reloads them in place.
  async function changeModel(key: BrainModelKey) {
    if (key === modelSize() || status() !== "ready" || modelSwitching()) return;
    const m = BRAIN_MODELS.find((b) => b.key === key);
    if (!m || !worker) return;
    setModelSwitching(true);
    setModelSize(key);
    clearTimeout(tickTimer);
    try {
      const [br, ref] = await Promise.all([
        fetch(`/models/${m.file}?v=${MODEL_VER}`).then((r) => r.text()),
        fetch(`/models/${m.ref}?v=${MODEL_VER}`).then((r) => r.json() as Promise<BrainRef>),
      ]);
      setBrainRef(ref);
      // a different brain → start the session tally over
      batch(() => {
        setSeen(0);
        setFirstRight(0);
        setAllRight(0);
        setMatchSum(0);
      });
      worker.postMessage({ type: "reloadBrain", brainWeights: br });
    } catch (e) {
      setModelSwitching(false);
    }
  }

  // ── derived display values ──────────────────────────────────────
  // first-step distribution [U, D, L, R, wait] at the tick on screen
  const stepProbs = (): number[] => {
    const fs = result()?.firstSteps[tickIdx()];
    return fs && fs.length === 5 ? softmax(fs) : [0, 0, 0, 0, 0];
  };
  const stepArgmax = (): number => {
    const p = stepProbs();
    let b = 0;
    for (let i = 1; i < p.length; i++) if (p[i] > p[b]) b = i;
    return b;
  };
  // the shortest route's first step, as a direction index (reference)
  const trueFirst = (): number => {
    const r = result();
    if (!r || !r.shortest.length) return -1;
    const dr = r.shortest[0][0] - r.start[0];
    const dc = r.shortest[0][1] - r.start[1];
    return dr === -1 ? 0 : dr === 1 ? 1 : dc === -1 ? 2 : 3;
  };
  const routeAt = (): TickRoute | null => result()?.routes[tickIdx()] ?? null;
  // how the final prediction ends, in words
  const outcome = (): string => {
    const r = result();
    if (!r || !r.routes.length) return "";
    const fin = r.routes[r.routes.length - 1];
    const L = r.shortest.length;
    if (fin.end === "goal" && r.agree === L) return `reaches the goal along the shortest route (${L} steps)`;
    const where = `matches the shortest route for ${r.agree} of ${L} steps`;
    if (fin.end === "wall") return `${where}, then predicts a step into a wall`;
    if (fin.end === "wait") return `${where}, then stops (wait)`;
    if (fin.end === "goal") return `${where}, reaches the goal by another way`;
    return `${where}, then runs out of predicted steps`;
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

  // ── maze + shortest route + the brain's predicted route (viz/maze.ts) ──
  // The caller owns DPR + clear (ctxOf); the maze module draws everything pure
  // from the snapshot we hand it. Theme colours come from CSS vars so it tracks
  // the paper / dark theme.
  function drawMaze() {
    const r = result();
    if (!r || !mazeCanvas) return;
    const W = mazeCanvas.clientWidth || 360;
    const ctx = ctxOf(mazeCanvas, W, W);
    const theme: MazeTheme = {
      wall: cssVar("--text", "#1B1822"),
      open: cssVar("--bg-card", "#fff"),
      line: cssVar("--line-2", "#D8D2C3"),
      mute: cssVar("--text-mute", "#6A6676"),
      accent: cssVar("--accent", "#6243D9"),
    };
    const state: MazeRenderState = {
      shortest: r.shortest,
      routes: r.routes,
      tick: tickIdx(),
      agree: r.agree,
    };
    drawMazeModule(ctx, W, W, { grid: r.grid, start: r.start, end: r.end }, SIZE, state, theme);
  }

  // ── the 8-region brain as a rotating 3D particle cloud ──
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

  // three-quarter view from front-left, slightly above (+x = face, −x = occipital)
  let brainRotY = -0.75;
  let brainRotX = 0.22;
  let brainZoom = 0.8; // wheel / pinch zoom — start zoomed OUT so the whole brain
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

  // brain tick aligned to the tick cursor (held at the last brain tick).
  const curBrainTick = (): BrainTick | null => {
    const bt = brainTrace();
    if (!bt || bt.ticks.length === 0) return null;
    return bt.ticks[Math.min(tickIdx(), bt.ticks.length - 1)];
  };

  // Tick shown in the 3D cloud. While the ticks are revealed, or when the user
  // has scrubbed to a tick, it follows the cursor; otherwise it REPLAYS the
  // trace on a loop (brainReplay) so the neurons keep moving.
  const displayBrainTick = (): BrainTick | null => {
    const bt = brainTrace();
    if (!bt || bt.ticks.length === 0) return null;
    const idx =
      runState() === "thinking" || pinned()
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
    // On narrow (phone-width) canvases scale the pixel focal length down so the
    // whole head — eyes in front, occipital pole behind — stays in frame.
    if (!fs)
      camera = orbitToCamera(
        brainRotX,
        brainRotY,
        brainZoom * Math.min(1, W / 500),
        BRAIN_CENTER,
      );
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

  // the brain's outer exit-gate λ at the current tick, if it has one
  const exitLambda = (): number | null => curBrainTick()?.exit ?? null;

  // ticks the CTM actually ran (the trace length / ticksUsed).
  const ticksUsed = (): number => brainTrace()?.ticksUsed ?? ticksTotal();

  // assembled feature-state for <SdkFeatures> — every field a real signal.
  const sdkFeatureState = () => ({
    ticksUsed: ticksUsed(),
    exitLambda: exitLambda(),
    visionActive: !!(brainTrace() && visionMaps && visionMaps.length),
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
    result();
    tickIdx();
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
    if (status() === "loading") return "loading…";
    if (generating() || modelSwitching()) return "new maze…";
    return runState() === "thinking" ? "thinking…" : "done";
  };

  // per-region rms at the current tick, for the fullscreen HUD
  const regionRows = () => {
    const names = brainRef()?.region_names ?? [];
    return names.map((nm, i) => {
      const acts = curBrainTick()?.acts?.[i] ?? [];
      const rms = acts.length ? Math.sqrt(acts.reduce((a, v) => a + v * v, 0) / acts.length) : 0;
      return { name: nm, rms, idx: i };
    });
  };

  const pct = (a: number, b: number) => (b > 0 ? Math.round((a / b) * 100) : 0);

  const segBtn = (active: boolean) => ({
    background: active ? "var(--accent)" : "transparent",
    color: active ? "#fff" : "var(--text-mute)",
  });
  const hudBtn = {
    background: "rgba(255,255,255,0.08)",
    color: "rgba(255,255,255,0.75)",
    "backdrop-filter": "blur(4px)",
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
          A small modgrad brain — eight Continuous Thought Machine regions behind
          a retina — looks at a maze image and predicts a route to the goal over
          its internal ticks. Nothing else plans or decides: the line on the maze
          is the brain's own output, shown tick by tick, next to the true
          shortest route for reference.
        </p>
        <p class="mt-3 text-dim text-[1.05rem] max-w-[60ch] leading-relaxed">
          It is early work. It usually gets the first steps right and then
          drifts: on {MEASURED.n} random 9×9 mazes the default brain's first
          step was right in {MEASURED.firstStep}, its route matched the shortest
          route for {MEASURED.meanMatch} of about {MEASURED.meanLen} steps on
          average, and {MEASURED.fullyCorrect} of {MEASURED.n} routes were
          correct all the way to the goal. It runs as the modgrad SDK's own code,
          compiled to WebAssembly, on your device.
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

      {/* ── maze + brain, side by side ───── */}
      <div class="grid lg:grid-cols-[minmax(0,1fr)_minmax(0,1.12fr)] gap-5 mt-8 items-start">
        {/* LEFT: maze + controls + first-step bars */}
        <div class="flex flex-col gap-5">
          <div class="card relative">
            <div class="flex items-center justify-between mb-3">
              <div class="eyebrow">The maze</div>
              <span
                class="tag"
                classList={{
                  "tag-wip": runState() === "thinking" || generating() || status() === "loading",
                }}
              >
                {statusLabel()}
              </span>
            </div>

            <div class="flex items-baseline justify-between gap-3 mb-3 font-mono text-xs text-mute tabular-nums">
              <span>
                maze #{Math.max(1, mazeNum())} · tick {ticksTotal() ? tickIdx() + 1 : 0}/{ticksTotal()}
              </span>
              <Show when={routeAt()}>
                <span>{routeAt()!.cells.length} predicted steps</span>
              </Show>
            </div>

            {/* fixed-ratio box so the canvas never reflows */}
            <div class="relative w-full" style={{ "aspect-ratio": "1 / 1" }}>
              <canvas
                ref={mazeCanvas}
                class="absolute inset-0 w-full h-full"
                aria-label="the maze, with the brain's predicted route and the shortest route"
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

            <Show when={result() && runState() === "idle"}>
              <p class="text-sm text-dim mt-3 leading-relaxed">
                Final tick: the brain's predicted route {outcome()}.
              </p>
            </Show>

            {/* legend */}
            <div class="flex flex-wrap gap-x-4 gap-y-1.5 mt-3 text-[.72rem] text-mute font-mono">
              <span class="inline-flex items-center gap-1.5">
                <i
                  class="inline-block rounded-full"
                  style={{
                    width: "22px",
                    height: "3px",
                    background:
                      "linear-gradient(90deg, color-mix(in srgb, var(--accent) 28%, var(--bg-card)), var(--accent))",
                  }}
                />
                the brain's predicted route (early → late tick)
              </span>
              <span class="inline-flex items-center gap-1.5">
                <i
                  class="inline-block"
                  style={{ width: "22px", height: "0", "border-top": "1.5px dashed var(--text-mute)" }}
                />
                shortest route
              </span>
              <span class="inline-flex items-center gap-1.5">
                <i class="w-2.5 h-2.5 rounded-full inline-block" style={{ border: "1.5px solid #e0564e" }} />
                first step off the shortest route
              </span>
              <span class="inline-flex items-center gap-1.5">
                <span style={{ color: "#e0564e" }}>✕</span>
                predicted step into a wall
              </span>
              <span class="inline-flex items-center gap-1.5">
                <i class="w-2.5 h-2.5 rounded-full inline-block" style={{ background: "var(--accent)" }} />
                start
              </span>
              <span class="inline-flex items-center gap-1.5">
                <i class="w-2.5 h-2.5 inline-block" style={{ background: "#3CB371", "border-radius": "2px" }} />
                goal
              </span>
            </div>
          </div>

          {/* controls */}
          <div class="card">
            <div class="flex flex-wrap gap-2">
              <button
                class="btn btn-primary flex-1 justify-center min-w-[120px]"
                onClick={requestNewMaze}
                disabled={status() !== "ready" || generating() || modelSwitching()}
              >
                New maze
              </button>
              <button
                class="btn flex-1 justify-center min-w-[120px]"
                onClick={playTicks}
                disabled={status() !== "ready" || !result()}
                title="replay the brain's ticks for this maze"
              >
                Replay ticks
              </button>
            </div>

            {/* tick scrubber */}
            <div class="mt-4">
              <div class="flex items-center justify-between mb-1.5">
                <div class="eyebrow">Tick</div>
                <div class="font-mono text-xs text-mute tabular-nums">
                  {ticksTotal() ? tickIdx() + 1 : 0} / {ticksTotal()}
                </div>
              </div>
              <input
                type="range"
                class="w-full"
                style={{ "accent-color": "var(--accent)" }}
                min="0"
                max={Math.max(0, ticksTotal() - 1)}
                step="1"
                value={tickIdx()}
                disabled={!result()}
                onInput={(e) => scrubTo(parseInt(e.currentTarget.value, 10))}
                aria-label="internal tick shown"
              />
            </div>

            {/* brain size selector: four trained CTM widths, hot-swapped */}
            <div class="flex items-center justify-between mt-4 pt-4 border-t border-line">
              <div class="eyebrow">Brain size</div>
              <div
                class="flex rounded-lg overflow-hidden border border-line"
                role="group"
                aria-label="brain size"
              >
                <For each={BRAIN_MODELS}>
                  {(m) => {
                    const active = () => modelSize() === m.key;
                    return (
                      <button
                        class="text-xs font-mono px-2.5 py-1 transition-colors"
                        style={segBtn(active())}
                        onClick={() => changeModel(m.key)}
                        disabled={status() !== "ready" || modelSwitching()}
                        aria-pressed={active()}
                        title={`cortex width ${m.label}`}
                      >
                        {m.label}
                      </button>
                    );
                  }}
                </For>
              </div>
            </div>
            <div class="text-[.72rem] text-mute font-mono mt-1.5 leading-relaxed">
              The same 8-region brain trained at four cortical widths, with the
              same retina. 9×9 mazes only: the retina is built for a 9×9 image.
              {modelSwitching() ? " Loading…" : ""}
            </div>
            {/* terse lineage note — the CTM is Sakana's; numbers, not prose */}
            <div class="text-[.68rem] text-mute font-mono mt-2 leading-relaxed opacity-80 tabular-nums">
              <div class="opacity-70">
                CTM core is{" "}
                <a href="/docs/continuous-thought-machine" class="underline">Sakana's</a>{" "}
                — depth vs width:
              </div>
              <div class="mt-0.5">Sakana maze · 1×512 · 75 ticks · synapse 8</div>
              <div>this brain · 8×(16–128) · 16 ticks · synapse 3</div>
            </div>
          </div>

          {/* first-step distribution at the tick on screen */}
          <div class="card">
            <div class="flex items-center justify-between mb-3">
              <div class="eyebrow">First step, as predicted</div>
              <div class="font-mono text-xs text-mute">
                tick {ticksTotal() ? tickIdx() + 1 : 0}/{ticksTotal()}
              </div>
            </div>
            <div class="flex flex-col gap-2.5">
              <For each={STEP_LABELS}>
                {(label, i) => {
                  const p = () => stepProbs()[i()] ?? 0;
                  const top = () => !!result() && stepArgmax() === i();
                  const isTrue = () => trueFirst() === i();
                  return (
                    <div class="flex items-center gap-3">
                      <span
                        class="w-12 text-xs font-mono shrink-0"
                        classList={{ "text-accent": top(), "text-mute": !top() }}
                      >
                        {label}
                      </span>
                      <div class="flex-1 h-5 rounded bg-panel overflow-hidden relative">
                        <div
                          class="h-full rounded transition-all duration-200"
                          style={{
                            width: `${Math.max(2, p() * 100)}%`,
                            background: top() ? "var(--accent)" : "var(--line-2)",
                          }}
                        />
                      </div>
                      <span class="w-24 text-right text-xs font-mono tabular-nums shrink-0 text-mute">
                        {isTrue() ? "shortest · " : ""}
                        {(p() * 100).toFixed(0)}%
                      </span>
                    </div>
                  );
                }}
              </For>
            </div>
            <Show when={seen() > 0}>
              <div class="mt-3 pt-3 border-t border-line flex flex-col gap-1 text-xs font-mono text-mute tabular-nums">
                <div class="flex items-center justify-between">
                  <span>this session · mazes</span>
                  <span class="text-dim">{seen()}</span>
                </div>
                <div class="flex items-center justify-between">
                  <span>first step right</span>
                  <span class="text-dim">
                    {firstRight()}/{seen()} ({pct(firstRight(), seen())}%)
                  </span>
                </div>
                <div class="flex items-center justify-between">
                  <span>steps matching the shortest route, mean</span>
                  <span class="text-dim">{(matchSum() / seen()).toFixed(1)}</span>
                </div>
                <div class="flex items-center justify-between">
                  <span>route correct to the goal</span>
                  <span class="text-dim">
                    {allRight()}/{seen()}
                  </span>
                </div>
              </div>
            </Show>
          </div>

          {/* region telemetry under the left column */}
          <Show when={brainRef()}>
            <RegionTelemetry
              regions={brainRef()!.regions}
              acts={curBrainTick()?.acts ?? null}
              global={curBrainTick()?.global ?? 0}
              exitLambda={exitLambda()}
              ticksUsed={ticksUsed()}
              ticksTotal={ticksTotal() || 16}
              defaultOpen
            />
          </Show>
        </div>

        {/* RIGHT: the 3D brain */}
        <Show when={brainRef()}>
          <div class="flex flex-col gap-5">
            <div class="card">
              <div class="flex items-center justify-between mb-1 flex-wrap gap-2">
                <div class="eyebrow">
                  live in 3D · <span class="grad-text">{brainRef()?.region_names.length ?? 8} regions</span>
                </div>
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
              </div>

              <p class="text-dim text-sm leading-relaxed mb-4">
                Each dot is a neuron, coloured by region and laid out roughly where
                that function sits in a human brain. A line lights when two neurons
                fire together across a wired region edge. The cyan grids are the
                retina → V1 → V2 → V4 feature maps of the maze image.{" "}
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
                  ...(isFullscreen() ? { width: "100vw", height: "100vh" } : {}),
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
                      class="text-[.68rem] font-mono px-2 py-1 rounded"
                      style={hudBtn}
                      onClick={() => setBrainAutoRotate((v) => !v)}
                      title="toggle auto-rotation"
                    >
                      {brainAutoRotate() ? "❚❚ stop" : "⟳ spin"}
                    </button>
                    <button
                      class="text-[.68rem] font-mono px-2 py-1 rounded"
                      style={hudBtn}
                      onClick={() => {
                        brainRotX = 0.22;
                        brainRotY = -0.75;
                        brainZoom = 0.8;
                      }}
                      title="reset view"
                    >
                      reset
                    </button>
                  </Show>
                  <button
                    class="text-[.68rem] font-mono px-2 py-1 rounded"
                    style={hudBtn}
                    onClick={toggleFullscreen}
                    title={isFullscreen() ? "exit fullscreen (Esc)" : "fly through the brain"}
                  >
                    {isFullscreen() ? "⛶ exit" : "⛶ fullscreen"}
                  </button>
                </div>
                <Show when={!isFullscreen()}>
                  <div
                    class="absolute bottom-2 left-3 text-[.66rem] font-mono pointer-events-none"
                    style={{ color: "rgba(255,255,255,0.4)" }}
                  >
                    drag to rotate · scroll / pinch to zoom · ⛶ to fly
                  </div>
                </Show>
                {/* fullscreen: free-fly hint + value-only HUD */}
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
                  <div
                    class="absolute top-3 left-3 text-[.64rem] font-mono tabular-nums rounded-lg flex flex-col gap-1.5"
                    style={{
                      background: "rgba(10,9,18,0.72)",
                      "backdrop-filter": "blur(4px)",
                      border: "1px solid rgba(255,255,255,0.12)",
                      color: "rgba(255,255,255,0.82)",
                      padding: "10px 12px",
                      "min-width": "186px",
                    }}
                  >
                    <div class="flex items-center justify-between" style={{ color: "rgba(255,255,255,0.55)" }}>
                      <span>regions</span>
                      <span>rms</span>
                    </div>
                    <For each={regionRows()}>
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
                    <div
                      class="flex flex-wrap gap-x-3 gap-y-0.5 pt-1.5"
                      style={{ "border-top": "1px solid rgba(255,255,255,0.1)", color: "rgba(255,255,255,0.6)" }}
                    >
                      <span>global {brainGlobal().toFixed(2)}</span>
                      <Show when={exitLambda() != null}>
                        <span>λ {(exitLambda() ?? 0).toFixed(2)}</span>
                      </Show>
                      <span>
                        tick {ticksTotal() ? tickIdx() + 1 : 0}/{ticksTotal()}
                      </span>
                    </div>
                  </div>
                  <div
                    class="absolute left-1/2 flex items-center gap-1.5 rounded-lg"
                    style={{
                      bottom: "16px",
                      transform: "translateX(-50%)",
                      background: "rgba(10,9,18,0.72)",
                      "backdrop-filter": "blur(4px)",
                      border: "1px solid rgba(255,255,255,0.12)",
                      padding: "8px 10px",
                    }}
                  >
                    <button
                      class="text-[.66rem] font-mono px-2.5 py-1 rounded"
                      style={{ background: "var(--accent)", color: "#fff" }}
                      onClick={requestNewMaze}
                      disabled={status() !== "ready" || generating()}
                    >
                      New maze
                    </button>
                    <button
                      class="text-[.66rem] font-mono px-2.5 py-1 rounded"
                      style={{ background: "rgba(255,255,255,0.1)", color: "rgba(255,255,255,0.85)" }}
                      onClick={playTicks}
                      disabled={!result()}
                    >
                      Replay ticks
                    </button>
                  </div>
                </Show>
              </div>

              <div class="flex flex-wrap items-center gap-x-4 gap-y-1.5 mt-3 text-[.72rem] text-mute font-mono">
                <span class="opacity-80">each dot = 1 neuron · brighter = more active</span>
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
                <span class="ml-auto opacity-80">the modgrad SDK · in WebAssembly</span>
              </div>
            </div>

            <SdkFeatures state={sdkFeatureState()} />
          </div>
        </Show>
      </div>

      {/* ── info: the factual reference ── */}
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
              <div class="eyebrow mb-1.5">What the brain does</div>
              It sees one image: the 9×9 maze as pixels, with the start in red and
              the goal in green. The retina → V1 → V2 → V4 stack turns that image
              into feature tokens, and eight CTM regions think over them for up to
              16 internal ticks. At every tick its output head predicts 20 moves
              (up, down, left, right or wait). The line on the maze is that
              prediction decoded, argmax per step, at the tick shown.
            </div>
            <div>
              <div class="eyebrow mb-1.5">What does not happen</div>
              No planner, solver or search runs alongside it, and no agent walks
              the maze. Nothing corrects the prediction: a predicted step into a
              wall is drawn as a cross where it happens. The shortest route is
              computed by breadth-first search only to draw the dashed reference
              line; the brain never sees it.
            </div>
            <div>
              <div class="eyebrow mb-1.5">How good it is</div>
              Early. Measured on {MEASURED.n} random 9×9 mazes with the default
              (cortex 32) brain: first step right in {MEASURED.firstStep}; on
              average the first {MEASURED.meanMatch} of about {MEASURED.meanLen}{" "}
              steps match the shortest route; {MEASURED.fullyCorrect} of{" "}
              {MEASURED.n} routes correct all the way to the goal. Most routes end
              in a predicted step into a wall. The other widths are no better on
              the first step (87–98%); the cortex-64 brain got 12 of 200 routes
              fully right in the same test, the others none.
              The session numbers on this page are your own sample.
            </div>
            <div>
              <div class="eyebrow mb-1.5">How it was trained</div>
              Supervised, offline: on generated mazes, to predict the
              solver-labelled route from the start image, like the maze task in
              Sakana's CTM paper. Labels exist only in training. The mazes here come
              from the same generator; they have not been checked against the
              training set, so some may have appeared in it.
            </div>
            <div>
              <div class="eyebrow mb-1.5">Vision</div>
              retina → V1 → V2 → V4 feed the input region; the cyan grids in the
              3D view are those feature maps for the current image. The retina is
              built for a fixed 9×9 input, which is why the demo has no maze-size
              control.
            </div>
            <div>
              <div class="eyebrow mb-1.5">Engine — all SDK</div>
              The browser engine is the modgrad SDK's own crates (
              <code class="text-[0.85em]">modgrad-ctm</code> +{" "}
              <code class="text-[0.85em]">modgrad-codec</code>) compiled to
              WebAssembly: the same forward pass that runs natively, in your tab.
            </div>
          </div>
        </Show>
      </div>
    </div>
  );
}
