// ─────────────────────────────────────────────────────────────────────────────
// brain3d.ts — self-contained 3D particle-brain renderer for the modgrad /play
// demo. Hand-rolled 3D→2D projection on a plain 2D canvas (no WebGL — the demo
// had crashes under WebGL-like load, so we stay deliberately GPU-light and only
// draw when asked). Extracted and modularised from the inline drawBrain3D() in
// src/pages/Play.tsx, then tidied: capped point count, reused scratch arrays,
// clearer region separation, and an optional co-spike intensity knob.
//
// It renders, faithfully to the original:
//   • per-region neuron clusters as additive-blended points, coloured by region
//     using an anatomical BRAIN_ANATOMY layout;
//   • a co-spike connectome between region centres (faint as structure, bright
//     when both ends fire together, fading over a configurable trail);
//   • travelling sparks along hot edges;
//   • the cyan visual-cortex layers (sight → retina → V1 → V2 → V4) feeding the
//     input region, wired forward into the brain.
//
// No imports from Play.tsx — compiles standalone.
// ─────────────────────────────────────────────────────────────────────────────

// ── Public input types (mirrored locally from Play.tsx) ──────────────────────

/** One brain step: per-region activations, a global-sync scalar, exit decision. */
export type BrainTick = {
  acts: number[][];
  global: number;
  exit: number | null;
};

/** Static brain structure: region names, sizes, and the wiring between them. */
export type BrainRef = {
  region_names: string[];
  regions: { name: string; d_model: number }[];
  connections: { from: number[]; to: number; receives_observation: boolean }[];
  n_global_sync: number;
};

/** One visual-cortex feature map (channels × h × w, row-major, flat in `data`). */
export type RetinaMap = {
  name: string;
  channels: number;
  h: number;
  w: number;
  data: number[];
};

/**
 * Free-fly "Half-Life"-style camera.
 *
 *  • `pos`   — camera position in world space (the brain lives in ~[-5,5]³).
 *  • `yaw`   — look left/right, radians, rotation about world +Y.
 *  • `pitch` — look up/down, radians, rotation about the camera's local +X.
 *  • `fov`   — focal length in pixels (bigger = narrower/zoomed-in view).
 *
 * The view transform is  view = Rx(-pitch) · Ry(-yaw) · (p − pos);  points with
 * view.z ≤ near are culled. See {@link cameraBasis} for the matching forward /
 * right / up basis vectors (handy for input code that walks the camera around).
 */
export type Camera = {
  pos: [number, number, number];
  yaw: number;
  pitch: number;
  fov: number;
};

/** Per-frame dynamic state handed to {@link Brain3D.draw}. */
export type BrainDrawState = {
  /** Current brain tick, or null when idle (we then show a gentle shimmer). */
  tick: BrainTick | null;
  /** Visual-cortex maps for the current observation, or null. */
  vision: RetinaMap[] | null;
  /**
   * Free-fly camera. Windowed (drag-to-orbit) callers can synthesise one from
   * the old orbit params with {@link orbitToCamera}; fullscreen free-fly drives
   * `pos`/`yaw`/`pitch` directly.
   */
  camera: Camera;
  /** Co-spike trail length, 0 (instant) … 1 (long); maps to a per-frame decay. */
  spikeHold: number;
  /** Monotonic time (seconds) for idle shimmer + travelling sparks. */
  time: number;
  /**
   * In-scene tech tour: when true, draw crisp 3D-anchored labels for every
   * region (name + neuron count), the vision-pathway stages, and a couple of
   * architectural callouts. Off by default. Defaults to false when omitted.
   */
  showLabels?: boolean;
};

/** Construction options — every improvement is opt-in with a faithful default. */
export type Brain3DOptions = {
  /** Region palette; defaults to the original 8-colour set. */
  regionColors?: string[];
  /** Hard cap on rendered neurons (perf guard). Default 1400. */
  maxPoints?: number;
  /** Co-spike intensity multiplier for the connectome. Default 2.4 (original). */
  cospikeGain?: number;
  /** Deterministic layout seed. Default 42 (matches the original). */
  seed?: number;
  /** Render the cyan vision pathway. Default true. */
  showVision?: boolean;
  /** Render the connectome edges + sparks. Default true. */
  showConnectome?: boolean;
};

// ── Internal types ───────────────────────────────────────────────────────────

type NeuronPos = {
  x: number;
  y: number;
  z: number;
  region: number;
  index: number;
};
type Vec3 = { x: number; y: number; z: number };
type ProjPoint = { x: number; y: number; depth: number; region: number; a: number };

/** A point projected to screen space, plus its view-space depth and on/off flag. */
type Projected = { x: number; y: number; depth: number; vis: boolean };
/** A camera transform baked once per frame (rotation rows + position + screen). */
type ViewXform = {
  cam: Camera;
  // rows of Rx(-pitch)·Ry(-yaw), so view = M·(p−pos)
  r0: Vec3;
  r1: Vec3;
  r2: Vec3;
  cxp: number;
  cyp: number;
};

// ── Constants ────────────────────────────────────────────────────────────────

const DEFAULT_REGION_COLORS = [
  "#4488ff", "#44ff88", "#ff8844", "#ff4488",
  "#8844ff", "#88ff44", "#ff88ff", "#ffff44",
];

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

const VIS_COLOR = "#62e6ff"; // cyan — distinct from the region colours

/** Near clip plane (view-space z) — points closer than this are culled. */
const NEAR = 0.2;

// ── Camera helpers ───────────────────────────────────────────────────────────

/**
 * Orthonormal camera basis in world space. `forward` is where the camera looks,
 * `right` strafes screen-right, `up` is screen-up. Exported so input code can
 * walk the camera ("WASD"-style) without re-deriving the trig. This is the exact
 * frame the projection uses, so a camera built from it round-trips faithfully:
 *
 *   forward = ( sinYaw·cosPitch, −sinPitch,  cosYaw·cosPitch )
 *   right   = ( cosYaw,           0,        −sinYaw )
 *   up      = ( sinYaw·sinPitch,  cosPitch,  cosYaw·sinPitch )
 *
 * At yaw=0,pitch=0 the camera looks toward +Z, with +X screen-right and +Y up.
 * Positive pitch looks down; positive yaw turns right.
 */
export function cameraBasis(cam: Camera): { forward: Vec3; right: Vec3; up: Vec3 } {
  const sy = Math.sin(cam.yaw);
  const cyaw = Math.cos(cam.yaw);
  const sp = Math.sin(cam.pitch);
  const cp = Math.cos(cam.pitch);
  return {
    forward: { x: sy * cp, y: -sp, z: cyaw * cp },
    right: { x: cyaw, y: 0, z: -sy },
    up: { x: sy * sp, y: cp, z: cyaw * sp },
  };
}

/**
 * Build the per-frame view transform from the camera basis so world points can
 * be projected with a few dot products:  view = (right·d, up·d, forward·d) for
 * d = p − pos. `cxp`/`cyp` are the screen centre in CSS pixels.
 */
function makeViewXform(cam: Camera, cxp: number, cyp: number): ViewXform {
  const { forward, right, up } = cameraBasis(cam);
  return {
    cam,
    r0: right, // view.x
    r1: up, // view.y
    r2: forward, // view.z (depth, +forward)
    cxp,
    cyp,
  };
}

/** Project a world point with the baked view transform. `vis` is false if culled. */
function projectWith(vx: ViewXform, px: number, py: number, pz: number): Projected {
  const dx = px - vx.cam.pos[0];
  const dy = py - vx.cam.pos[1];
  const dz = pz - vx.cam.pos[2];
  const vxx = vx.r0.x * dx + vx.r0.y * dy + vx.r0.z * dz;
  const vyy = vx.r1.x * dx + vx.r1.y * dy + vx.r1.z * dz;
  const vzz = vx.r2.x * dx + vx.r2.y * dy + vx.r2.z * dz;
  if (!(vzz > NEAR)) return { x: 0, y: 0, depth: vzz, vis: false };
  const f = vx.cam.fov / vzz;
  return { x: vx.cxp + vxx * f, y: vx.cyp - vyy * f, depth: vzz, vis: true };
}

/** Base focal length (px) for the orbit camera at zoom=1 — matches the old
 *  `fov = W·0.72` framing at a ~720px canvas. {@link Brain3D.draw} leaves a
 *  pixel-scale fov (> ~4) untouched, so this renders at the original scale. */
const ORBIT_BASE_FOV_PX = 720 * 0.72;

/**
 * Convert the OLD orbit params (rotX, rotY, zoom) into an equivalent free-fly
 * {@link Camera} that orbits `center` at fixed distance `dist`, reproducing the
 * original windowed framing so drag-to-orbit callers keep working by just
 * feeding the result into {@link Brain3D.draw}. `rotY` maps to yaw (look
 * left/right), `rotX` to pitch (look up/down); zoom scales the focal length so
 * dragging-to-zoom feels identical to before. Uses the same orbit convention as
 * the projection's {@link cameraBasis} — camera sits opposite its forward
 * vector, `dist` out from `center`, looking back at it.
 */
export function orbitToCamera(
  rotX: number,
  rotY: number,
  zoom: number,
  center: [number, number, number] = [0, 0, 0],
  dist = 13,
): Camera {
  const yaw = rotY;
  const pitch = rotX;
  const { forward } = cameraBasis({ pos: [0, 0, 0], yaw, pitch, fov: 1 });
  return {
    pos: [center[0] - forward.x * dist, center[1] - forward.y * dist, center[2] - forward.z * dist],
    yaw,
    pitch,
    fov: ORBIT_BASE_FOV_PX * zoom,
  };
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/** Tiny deterministic PRNG in [-1, 1), so the layout is stable across reloads. */
function makeRand(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    const t = s;
    return (((t ^ (t >>> 14)) >>> 0) / 4294967296) * 2 - 1;
  };
}

/** "#rrggbb" + alpha → rgba() string, for additive glow gradients. */
function hexA(hex: string, a: number): string {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

// ── Renderer ─────────────────────────────────────────────────────────────────

export class Brain3D {
  private readonly colors: string[];
  private readonly maxPoints: number;
  private readonly cospikeGain: number;
  private readonly seed: number;
  private readonly showVision: boolean;
  private readonly showConnectome: boolean;

  private ref: BrainRef | null = null;
  private neuronPos: NeuronPos[] = [];
  private regionCenters: Vec3[] = [];
  private builtFor = -1; // total neuron count the layout was built for

  // [start, count] into neuronPos for each region (neurons are laid down
  // region-by-region, so each region owns a contiguous slice).
  private regionRange: { start: number; count: number }[] = [];
  // precomputed neuron→neuron strands for the connectome: a stable sample of
  // individual-neuron pairs per wired region edge (so the wiring reads as real
  // neural strands, not hub-to-hub spokes). Rebuilt with the neuron layout.
  private connStrands: { fa: number; fb: number; key: string; from: number; to: number; phase: number }[] = [];

  // decayed co-spike per region edge ("from_to" → glow); persists across frames.
  private readonly edgeGlow = new Map<string, number>();

  // Reused scratch buffers — avoid per-frame allocation churn.
  private projScratch: ProjPoint[] = [];
  private regionAct: number[] = [];
  // per-neuron projected screen position (indexed by neuronPos index), filled
  // each frame so the neuron-strand connectome can look endpoints up cheaply.
  private projX = new Float32Array(0);
  private projY = new Float32Array(0);
  private projVis = new Uint8Array(0);

  constructor(opts: Brain3DOptions = {}) {
    this.colors = opts.regionColors ?? DEFAULT_REGION_COLORS;
    this.maxPoints = Math.max(1, opts.maxPoints ?? 1400);
    this.cospikeGain = opts.cospikeGain ?? 2.4;
    this.seed = opts.seed ?? 42;
    this.showVision = opts.showVision ?? true;
    this.showConnectome = opts.showConnectome ?? true;
  }

  /** Install (or replace) the brain structure; rebuilds the neuron cloud. */
  setRef(ref: BrainRef): void {
    this.ref = ref;
    this.builtFor = -1; // force a rebuild on next draw
    this.edgeGlow.clear();
    this.buildNeuronPositions(ref.regions);
  }

  private regionColor(i: number): string {
    return this.colors[((i % this.colors.length) + this.colors.length) % this.colors.length];
  }

  private buildNeuronPositions(regions: { name: string; d_model: number }[]): void {
    const total = regions.reduce((a, r) => a + r.d_model, 0);
    // Per-region budget: cap the total point count, keeping each region's share.
    const scale = total > this.maxPoints ? this.maxPoints / total : 1;

    this.neuronPos = [];
    this.regionCenters = [];
    this.regionRange = [];
    const rand = makeRand(this.seed);
    for (let ri = 0; ri < regions.length; ri++) {
      const want = Math.max(1, Math.round(regions[ri].d_model * scale));
      const [cx, cy, cz] = BRAIN_ANATOMY[regions[ri].name] ?? [
        (ri % 4) * 3 - 4.5,
        1.5 - Math.floor(ri / 4) * 3,
        0,
      ];
      this.regionCenters[ri] = { x: cx, y: cy, z: cz };
      this.regionRange[ri] = { start: this.neuronPos.length, count: want };
      // spread keyed to the *true* region size (so big regions stay big even
      // when sub-sampled), with a touch tighter packing for clearer separation.
      const spread = Math.sqrt(regions[ri].d_model) * 0.34;
      for (let ni = 0; ni < want; ni++) {
        const theta = rand() * Math.PI;
        const phi = rand() * Math.PI * 2;
        const rr = spread * (0.3 + 0.7 * Math.abs(rand()));
        this.neuronPos.push({
          x: cx + rr * Math.sin(theta) * Math.cos(phi),
          y: cy + rr * Math.cos(theta),
          z: cz + rr * Math.sin(theta) * Math.sin(phi),
          region: ri,
          // map back to a representative source index for activation lookup
          index: Math.min(regions[ri].d_model - 1, Math.floor((ni / want) * regions[ri].d_model)),
        });
      }
    }
    this.builtFor = total;
    const n = this.neuronPos.length;
    if (this.projX.length !== n) {
      this.projX = new Float32Array(n);
      this.projY = new Float32Array(n);
      this.projVis = new Uint8Array(n);
    }
    this.buildConnStrands();
  }

  // Sample a stable set of neuron→neuron strands for each wired region edge, so
  // the connectome renders as individual neural fibres rather than hub spokes.
  // Endpoints are real neurons (random within each region's slice); the per-edge
  // co-fire glow is shared across that edge's strands. Seeded → no frame-to-frame
  // flicker. STRANDS_PER_EDGE keeps the fibre count (and line cost) bounded.
  private buildConnStrands(): void {
    this.connStrands = [];
    const conns = this.ref?.connections;
    if (!conns) return;
    const rand = makeRand((this.seed ^ 0x9e3779b1) >>> 0);
    const STRANDS_PER_EDGE = 4;
    for (const conn of conns) {
      const tr = this.regionRange[conn.to];
      if (!tr || tr.count === 0) continue;
      for (const f of conn.from) {
        const fr = this.regionRange[f];
        if (!fr || fr.count === 0) continue;
        const key = f + "_" + conn.to;
        for (let k = 0; k < STRANDS_PER_EDGE; k++) {
          const fa = fr.start + Math.floor(rand() * fr.count);
          const fb = tr.start + Math.floor(rand() * tr.count);
          this.connStrands.push({ fa, fb, key, from: f, to: conn.to, phase: rand() });
        }
      }
    }
  }

  /**
   * Draw one frame into `ctx` at `w`×`h` CSS pixels. The caller is responsible
   * for canvas sizing / DPR transform / clearing before this is invoked.
   */
  draw(ctx: CanvasRenderingContext2D, w: number, h: number, state: BrainDrawState): void {
    const bref = this.ref;
    if (!bref) return;
    const total = bref.regions.reduce((a, r) => a + r.d_model, 0);
    if (this.builtFor !== total) this.buildNeuronPositions(bref.regions);

    const { tick, vision, spikeHold, time } = state;
    const showLabels = state.showLabels ?? false;
    const W = w;
    const H = h;
    const cxp = W / 2;
    const cyp = H / 2;

    // Camera. Convention: a small fov (≤ ~4) is a *focal-length-per-canvas-width*
    // (e.g. orbitToCamera's 0.72) and is scaled to pixels here; a larger fov is
    // taken as an explicit pixel focal length (fullscreen free-fly drives this).
    const inCam = state.camera;
    const fovPx = inCam.fov <= 4 ? inCam.fov * W : inCam.fov;
    const cam: Camera = { pos: inCam.pos, yaw: inCam.yaw, pitch: inCam.pitch, fov: fovPx };
    const vx = makeViewXform(cam, cxp, cyp);
    // `fov` kept for the vision pathway's screen-size heuristics (was W·0.72·zoom).
    const fov = fovPx;

    // dark "brain-scan" backdrop — bright region colours only read on dark.
    const bg = ctx.createRadialGradient(cxp, cyp * 0.9, 8, cxp, cyp, Math.max(W, H) * 0.75);
    bg.addColorStop(0, "#181527");
    bg.addColorStop(1, "#0a0912");
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, W, H);

    // per-step normaliser so spike brightness is stable in scale. (Derived from
    // the current tick only — the renderer is stateless across the trace.)
    let amax = 1e-6;
    if (tick) {
      for (const ra of tick.acts) {
        for (const v of ra) {
          const a = v < 0 ? -v : v;
          if (a > amax) amax = a;
        }
      }
    }
    const actOf = (region: number, index: number): number => {
      if (tick) {
        const v = tick.acts[region]?.[index] ?? 0;
        return Math.min(1, (v < 0 ? -v : v) / amax);
      }
      // idle: gentle per-neuron shimmer so the brain looks alive at rest
      return 0.1 + 0.14 * (0.5 + 0.5 * Math.sin(time * 1.7 + region * 1.3 + index * 0.7));
    };

    // ── free-fly camera projection ──
    // Compatible {x,y,depth} closure for the vision pathway + connectome; culled
    // points come back with depth = -1 so the existing `depth >= 0.5` guards
    // reject them (matches the old behaviour, also rejects NaN).
    const project = (px: number, py: number, pz: number) => {
      const p = projectWith(vx, px, py, pz);
      return p.vis ? { x: p.x, y: p.y, depth: p.depth } : { x: 0, y: 0, depth: -1 };
    };

    // ── project neurons into the reused scratch buffer ──
    // Also record every neuron's projected screen position (by index) so the
    // neuron-strand connectome can look up its endpoints without reprojecting.
    const pts = this.projScratch;
    pts.length = 0;
    const projX = this.projX;
    const projY = this.projY;
    const projVis = this.projVis;
    for (let i = 0; i < this.neuronPos.length; i++) {
      const np = this.neuronPos[i];
      const p = projectWith(vx, np.x, np.y, np.z);
      if (!p.vis) {
        projVis[i] = 0; // culled (view.z <= near) — also rejects NaN
        continue;
      }
      projX[i] = p.x;
      projY[i] = p.y;
      projVis[i] = 1;
      const a = actOf(np.region, np.index);
      pts.push({
        x: p.x,
        y: p.y,
        depth: p.depth,
        region: np.region,
        a: Number.isFinite(a) ? a : 0,
      });
    }
    pts.sort((p, q) => q.depth - p.depth); // painter's algorithm: far first

    // per-region activity for the co-spike connections. Normalise against the
    // HOTTEST region this tick (not the hottest single neuron): mean/maxNeuron
    // crushed every region to ~0.1, so the product of two ends was ~0.02 and the
    // connectome never lit. Relative-to-hottest gives real contrast — the most
    // active region reads ~1, quieter ones scale down, idle floors at 0.06.
    const regionAct = this.regionAct;
    regionAct.length = 0;
    const rawMean: number[] = [];
    let maxRegionMean = 1e-6;
    for (let r = 0; r < bref.regions.length; r++) {
      const ra = tick?.acts[r];
      if (ra && ra.length) {
        let s = 0;
        for (const v of ra) s += v < 0 ? -v : v;
        const m = s / ra.length;
        rawMean[r] = m;
        if (m > maxRegionMean) maxRegionMean = m;
      } else {
        rawMean[r] = 0;
      }
    }
    for (let r = 0; r < bref.regions.length; r++) {
      regionAct[r] = rawMean[r] > 0 ? Math.max(0.06, Math.min(1, rawMean[r] / maxRegionMean)) : 0.06;
    }

    ctx.globalCompositeOperation = "lighter";

    if (this.showConnectome) {
      this.drawConnectome(ctx, bref, regionAct, spikeHold, time);
    }
    this.drawNeurons(ctx, pts);
    if (this.showVision && vision && vision.length) {
      this.drawVision(ctx, bref, vision, fov, project, time);
    }

    // ── in-scene tech tour: crisp 3D-anchored labels (source-over) ──
    if (showLabels) {
      ctx.globalCompositeOperation = "source-over";
      this.drawLabels(ctx, bref, vision, vx, regionAct);
    }

    ctx.globalCompositeOperation = "source-over";
  }

  // ── tech tour: project a world anchor through the camera and draw a small,
  //    legible mono label, hidden when behind the camera / off-screen, scaling
  //    subtly with depth so near callouts read a touch larger than far ones. ──
  private drawLabel(
    ctx: CanvasRenderingContext2D,
    vx: ViewXform,
    anchor: Vec3,
    text: string,
    color: string,
    W: number,
    H: number,
    dy = 0,
  ): void {
    const p = projectWith(vx, anchor.x, anchor.y, anchor.z);
    if (!p.vis) return; // behind camera
    const x = p.x;
    const y = p.y + dy;
    if (x < -40 || x > W + 40 || y < -20 || y > H + 20) return; // off-screen
    // subtle depth scale: near ≈ 11px, far ≈ 8.5px, clamped.
    const fs = Math.max(8.5, Math.min(11, 9 + 18 / p.depth));
    ctx.font = `600 ${fs.toFixed(1)}px ui-monospace, SFMono-Regular, Menlo, monospace`;
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    const wpx = ctx.measureText(text).width;
    // dim plate for legibility over the additive bloom
    ctx.fillStyle = "rgba(8,9,16,0.62)";
    ctx.fillRect(x - wpx / 2 - 4, y - fs - 2, wpx + 8, fs + 6);
    ctx.fillStyle = color;
    ctx.fillText(text, x, y);
  }

  private drawLabels(
    ctx: CanvasRenderingContext2D,
    bref: BrainRef,
    vision: RetinaMap[] | null,
    vx: ViewXform,
    regionAct: number[],
  ): void {
    const W = vx.cxp * 2;
    const H = vx.cyp * 2;

    // 1) each region: short name + neuron count (its true d_model)
    for (let r = 0; r < bref.regions.length; r++) {
      const c = this.regionCenters[r];
      if (!c) continue;
      const name = bref.regions[r]?.name ?? bref.region_names[r] ?? `r${r}`;
      const short = name.length > 14 ? name.slice(0, 13) + "…" : name;
      const n = bref.regions[r]?.d_model ?? 0;
      this.drawLabel(ctx, vx, c, `${short} · ${n}`, this.regionColor(r), W, H, -10);
    }

    // 2) vision pathway stages, anchored on the same plane drawVision lays out.
    if (this.showVision && vision && vision.length) {
      const inIdx = bref.region_names.indexOf("input");
      const ic = this.regionCenters[inIdx >= 0 ? inIdx : 0] ?? { x: 3.7, y: 1.2, z: 0.5 };
      const nL = vision.length;
      const SPC = 0.6;
      const niceName: Record<string, string> = {
        sight: "what it sees",
        retina: "retina",
        v1: "V1",
        v2: "V2",
        v4: "V4",
      };
      for (let li = 0; li < nL; li++) {
        const m = vision[li];
        const lx = ic.x + 1.0 + (nL - 1 - li) * SPC;
        const ly = ic.y + 0.4;
        const lz = ic.z;
        const label = niceName[m.name.toLowerCase()] ?? m.name;
        this.drawLabel(ctx, vx, { x: lx, y: ly, z: lz }, label, VIS_COLOR, W, H, -14);
      }
    }

    // 3) architectural callouts, anchored at meaningful world points.
    // centroid of the region cloud → "8-region brain"
    let cx = 0;
    let cy = 0;
    let cz = 0;
    const nc = this.regionCenters.length || 1;
    for (const c of this.regionCenters) {
      cx += c.x;
      cy += c.y;
      cz += c.z;
    }
    const centroid = { x: cx / nc, y: cy / nc + 3.6, z: cz / nc };
    this.drawLabel(ctx, vx, centroid, `${bref.regions.length}-region brain`, "#e9edf6", W, H, -2);

    // global-sync readout: anchored near the most-active region (the readout hub)
    let hub = 0;
    for (let r = 1; r < regionAct.length; r++) if (regionAct[r] > regionAct[hub]) hub = r;
    const hc = this.regionCenters[hub];
    if (hc) {
      this.drawLabel(
        ctx,
        vx,
        { x: hc.x, y: hc.y - 1.0, z: hc.z },
        `global-sync ×${bref.n_global_sync}`,
        "#ffe08a",
        W,
        H,
        14,
      );
    }

    // connectome: midpoint of the first wired edge we can find
    const conn = bref.connections.find((c) => c.from.length > 0);
    if (conn) {
      const a = this.regionCenters[conn.from[0]];
      const b = this.regionCenters[conn.to];
      if (a && b) {
        const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 + 0.4, z: (a.z + b.z) / 2 };
        this.drawLabel(ctx, vx, mid, "connectome", "#9aa3b8", W, H, 0);
      }
    }
  }

  // ── connectome edges: faint as structure, bright when both ends co-spike,
  //    then fading over a configurable trail (spikeHold → per-frame decay). ──
  private drawConnectome(
    ctx: CanvasRenderingContext2D,
    bref: BrainRef,
    regionAct: number[],
    spikeHold: number,
    time: number,
  ): void {
    const decay = 0.86 + spikeHold * 0.135; // 0.86 (brief) … ~0.995 (long)
    ctx.lineCap = "round";

    // 1. update the per-region-edge co-fire glow (shared by all that edge's
    //    neuron strands). co-fire = the WEAKER of the two ends (a true AND: both
    //    regions must be active), scaled by the gain — min() keeps the "both
    //    fire" semantics without crushing the value the way the product did.
    for (const conn of bref.connections) {
      for (const f of conn.from) {
        const co = Math.min(1, Math.min(regionAct[f], regionAct[conn.to]) * this.cospikeGain);
        const key = f + "_" + conn.to;
        const g = Math.max(co, (this.edgeGlow.get(key) ?? 0) * decay); // persistence
        this.edgeGlow.set(key, g);
      }
    }

    // 2. draw the individual neuron→neuron strands, lit by their edge's glow.
    const projX = this.projX;
    const projY = this.projY;
    const projVis = this.projVis;
    for (const s of this.connStrands) {
      if (!projVis[s.fa] || !projVis[s.fb]) continue; // an endpoint is culled
      const g = this.edgeGlow.get(s.key) ?? 0;
      const ax = projX[s.fa];
      const ay = projY[s.fa];
      const bx = projX[s.fb];
      const by = projY[s.fb];
      const colF = this.regionColor(s.from);
      const colT = this.regionColor(s.to);
      const grad = ctx.createLinearGradient(ax, ay, bx, by);
      grad.addColorStop(0, hexA(colF, 0.04 + 0.8 * g));
      grad.addColorStop(1, hexA(colT, 0.04 + 0.8 * g));
      ctx.strokeStyle = grad;
      ctx.lineWidth = 0.4 + 2.4 * g;
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.stroke();
      // a spike travelling neuron→neuron while the fibre is hot
      if (g > 0.12) {
        const fr = (time * 0.55 + s.phase) % 1;
        ctx.fillStyle = hexA("#ffffff", Math.min(0.95, g));
        ctx.beginPath();
        ctx.arc(ax + (bx - ax) * fr, ay + (by - ay) * fr, 1.1 + 2.2 * g, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  // ── additive blending → overlapping neurons bloom, spikes pop ──
  private drawNeurons(ctx: CanvasRenderingContext2D, pts: ProjPoint[]): void {
    for (const p of pts) {
      const col = this.regionColor(p.region);
      const fogv = Math.max(0.4, Math.min(1, 14 / p.depth - 0.25)); // nearer = brighter/bigger
      const size = Math.max(1.6, (15 / p.depth) * (1.8 + 4 * p.a) * fogv);
      // glow halo only for neurons bright enough to show one (skips ~all the
      // dim/idle dots — avoids a radial gradient per invisible glow).
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
  }

  // ── visual cortex: sight → retina → V1 → V2 → V4 layers feeding `input`.
  //    Each layer is a spatial plane (channels averaged), lit by what it sees,
  //    wired forward into the brain — the eyes-to-cognition pathway. ──
  private drawVision(
    ctx: CanvasRenderingContext2D,
    bref: BrainRef,
    vision: RetinaMap[],
    fov: number,
    proj: (x: number, y: number, z: number) => { x: number; y: number; depth: number },
    time: number,
  ): void {
    const VIS = VIS_COLOR;
    const inIdx = bref.region_names.indexOf("input");
    const ic = this.regionCenters[inIdx >= 0 ? inIdx : 0] ?? { x: 3.7, y: 1.2, z: 0.5 };
    const nL = vision.length;
    const SPC = 0.6; // layer-to-layer spacing
    const centers: { x: number; y: number; act: number }[] = [];

    for (let li = 0; li < nL; li++) {
      const m = vision[li];
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
        const c0 = corner(-0.6, -0.6),
          c1 = corner(-0.6, m.w - 0.4),
          c2 = corner(m.h - 0.4, m.w - 0.4),
          c3 = corner(m.h - 0.4, -0.6);
        ctx.fillStyle = "rgba(10,12,20,0.72)";
        ctx.beginPath();
        ctx.moveTo(c0.x, c0.y);
        ctx.lineTo(c1.x, c1.y);
        ctx.lineTo(c2.x, c2.y);
        ctx.lineTo(c3.x, c3.y);
        ctx.closePath();
        ctx.fill();
        ctx.strokeStyle = hexA(VIS, 0.5);
        ctx.lineWidth = 1.2;
        ctx.stroke();
        for (let r = 0; r < m.h; r++) {
          for (let c = 0; c < m.w; c++) {
            const i = r * m.w + c;
            const R = m.data[i],
              G = m.data[hw + i],
              B = m.data[2 * hw + i];
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
      let mx = 1e-6;
      const cell = new Array<number>(hw);
      for (let i = 0; i < hw; i++) {
        let s = 0;
        for (let ch = 0; ch < m.channels; ch++) {
          const v = m.data[ch * hw + i];
          s += v < 0 ? -v : v;
        }
        const c = s / m.channels;
        cell[i] = c;
        if (c > mx) mx = c;
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
      const fr = (time * 0.5 + li * 0.15) % 1; // info flowing inward
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
}

/** Factory mirroring the requested `createBrain3D(opts)` surface. */
export function createBrain3D(opts: Brain3DOptions = {}): Brain3D {
  return new Brain3D(opts);
}
