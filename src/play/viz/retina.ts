// retina.ts — visual-cortex renderer for the modgrad /play demo.
//
// The model has a small visual system: it reads each maze cell through a 3×9×9
// RGB retina ("what it sees"), then a learned feedforward pathway
//   retina → V1 → V2 → V4
// computes feature planes that feed the brain's `input` region. This module
// renders both halves on a plain 2D canvas (GPU-light, no WebGL):
//
//   • drawSightScreen  — the literal RGB image as a crisp framed colour panel.
//   • drawCortexLayers — the learned feature planes as a stacked, labelled
//                        pathway. Standalone in 2D, or placed in the 3D brain
//                        by injecting a projection callback (nothing hardcoded).
//   • cortexSaliency   — collapse the feature planes to a grid for the maze
//                        attention overlay.
//
// Self-contained: no imports, no shared module state. Everything the caller
// needs is in the exported types and the per-call options.

// ── data model ───────────────────────────────────────────────────────────────

/** One layer of the visual system, as emitted by the engine worker. */
export type RetinaMap = {
  /** "sight" = the literal RGB the eye sees; else "retina" | "v1" | "v2" | "v4". */
  name: string;
  /** channel count: 3 for sight (RGB), N for a learned feature plane. */
  channels: number;
  /** spatial height. */
  h: number;
  /** spatial width. */
  w: number;
  /** flat [channels × h × w], channel-major: data[ch*h*w + r*w + c]. */
  data: number[];
};

/** Minimal 2D context surface used here (lets the module type-check standalone). */
type Ctx = CanvasRenderingContext2D;

// ── palette ──────────────────────────────────────────────────────────────────

const VIS = "#62e6ff"; // cyan — the visual-system accent, distinct from regions

/** Literal retina colours, matching what the eye encodes per cell. */
const SIGHT = {
  agent: "#ff5b5b",
  goal: "#4fd17a",
  open: "#e9edf6",
  wall: "#33384a",
  frame: "rgba(10,12,20,0.72)",
} as const;

/** `#rrggbb` + alpha → `rgba(...)`. Pass-through for already-rgba/named colours. */
function hexA(hex: string, a: number): string {
  if (hex[0] !== "#" || hex.length < 7) return hex;
  const n = parseInt(hex.slice(1, 7), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

/** Human label for a layer name. */
function layerLabel(name: string): string {
  if (name === "sight") return "sight";
  return name.toUpperCase(); // retina → RETINA, v1 → V1, …
}

// ── feature-plane reduction (shared by 2D, 3D and saliency) ──────────────────

type Plane = { cell: number[]; max: number; mean: number };

/**
 * Collapse a feature map's channels to a single normalized response plane.
 * Each cell is the mean absolute response across channels; `max`/`mean` describe
 * the (pre-normalization) plane so callers can scale brightness consistently.
 */
function reducePlane(m: RetinaMap): Plane {
  const hw = m.h * m.w;
  const cell = new Array<number>(hw).fill(0);
  let max = 1e-6;
  let sum = 0;
  const ch = Math.max(1, m.channels);
  for (let i = 0; i < hw; i++) {
    let s = 0;
    for (let c = 0; c < ch; c++) {
      const v = m.data[c * hw + i] ?? 0;
      s += v < 0 ? -v : v;
    }
    const a = s / ch;
    cell[i] = a;
    sum += a;
    if (a > max) max = a;
  }
  return { cell, max, mean: sum / hw };
}

/** Classify one sight pixel (RGB in [0,1]) into a literal cell colour. */
function sightColor(R: number, G: number, B: number): string {
  if (R > 0.5 && G < 0.5 && B < 0.5) return SIGHT.agent;
  if (G > 0.5 && R < 0.5) return SIGHT.goal;
  if (R > 0.5 && G > 0.5 && B > 0.5) return SIGHT.open;
  return SIGHT.wall;
}

// ── 1) "what it sees": the literal RGB retinal image ─────────────────────────

/**
 * Draw the literal RGB image the model sees as a crisp framed colour panel,
 * top-left at (x, y), fitting within `size`×`size` CSS px. True colours, axis-
 * aligned — meant to be shown prominently in the UI, not buried in the 3D scene.
 *
 * No-op if `sight` is missing/empty. Restores fill/stroke-affecting state it
 * sets (font, textAlign) so it composes cleanly with surrounding draw calls.
 */
export function drawSightScreen(
  ctx: Ctx,
  x: number,
  y: number,
  size: number,
  sight: RetinaMap,
  opts: { label?: boolean } = {},
): void {
  if (!sight || sight.h <= 0 || sight.w <= 0 || !sight.data.length) return;
  const { h, w } = sight;
  const hw = h * w;

  const labelH = opts.label === false ? 0 : 16;
  const pad = Math.max(2, size * 0.04);
  const gridW = size - pad * 2;
  const gridH = size - pad * 2 - labelH;
  const cw = gridW / w;
  const ch = gridH / h;
  const gx = x + pad;
  const gy = y + pad + labelH;

  // framed "screen" backdrop
  ctx.fillStyle = SIGHT.frame;
  ctx.fillRect(x, y + labelH, size, size - labelH);
  ctx.strokeStyle = hexA(VIS, 0.55);
  ctx.lineWidth = 1.2;
  ctx.strokeRect(x + 0.5, y + labelH + 0.5, size - 1, size - labelH - 1);

  // RGB cells
  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      const i = r * w + c;
      const R = sight.data[i] ?? 0;
      const G = sight.data[hw + i] ?? 0;
      const B = sight.data[2 * hw + i] ?? 0;
      ctx.fillStyle = sightColor(R, G, B);
      // small inset so cells read as a grid, not a flat block
      ctx.fillRect(gx + c * cw + 0.5, gy + r * ch + 0.5, cw - 1, ch - 1);
    }
  }

  if (labelH) {
    ctx.font = "600 10px ui-monospace, monospace";
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    ctx.fillStyle = hexA(VIS, 0.9);
    ctx.fillText("what it sees", x, y + 11);
    ctx.fillStyle = "rgba(120,130,150,0.85)";
    ctx.textAlign = "right";
    ctx.fillText(`${sight.channels}×${h}×${w} RGB`, x + size, y + 11);
    ctx.textAlign = "left";
  }
}

// ── 2) the learned pathway: retina → V1 → V2 → V4 feature planes ─────────────

/** A point in projected (screen) space — the unit the projection returns. */
export type ProjPoint = { x: number; y: number; depth: number };

/**
 * Projection callback for 3D placement. Given a logical layer index and a
 * normalized cell coordinate (row r∈[0,h), col c∈[0,w) — fractional corners
 * allowed) it returns the on-screen point. The 3D brain owns the camera; this
 * module stays camera-agnostic.
 */
export type CortexProjection = (
  layer: number,
  r: number,
  c: number,
  ctx2: { h: number; w: number; layerCount: number },
) => ProjPoint;

export type CortexLayout =
  | {
      mode: "2d";
      /** top-left of the panel. */
      x: number;
      y: number;
      /** total panel size; layers stack left→right within it. */
      width: number;
      height: number;
      /** draw per-layer name + channel-count labels (default true). */
      labels?: boolean;
    }
  | {
      mode: "3d";
      /** caller-supplied camera projection; this module never touches the matrix. */
      project: CortexProjection;
      /** dot radius scale hint for 3D (optional; default derives from depth). */
      labels?: boolean;
    };

/**
 * Draw the retina→V1→V2→V4 feature planes as a stacked, labelled pathway.
 *
 *   • The "sight" map (if present) is skipped here — it is the literal image and
 *     belongs to drawSightScreen; this function renders only the learned planes.
 *   • Each plane is channel-averaged and normalized to its own max, so every
 *     layer reads at full contrast regardless of absolute activation scale.
 *
 * `layout.mode === "2d"` lays the layers out left→right inside the given rect.
 * `layout.mode === "3d"` defers all positioning to `layout.project`, so the 3D
 * brain can drop the same pathway into its scene without this module knowing the
 * camera. Returns the projected centre of each rendered layer (in draw order),
 * which the caller can use to wire the pathway onward to the input region.
 */
export function drawCortexLayers(
  ctx: Ctx,
  layout: CortexLayout,
  maps: RetinaMap[],
): Array<{ x: number; y: number; act: number; name: string }> {
  const layers = (maps ?? []).filter((m) => m.name !== "sight" && m.h > 0 && m.w > 0);
  const centers: Array<{ x: number; y: number; act: number; name: string }> = [];
  if (!layers.length) return centers;
  const nL = layers.length;

  if (layout.mode === "2d") {
    const labels = layout.labels !== false;
    const labelH = labels ? 16 : 0;
    const innerPad = Math.max(2, layout.width * 0.015);
    const colGap = Math.max(6, layout.width * 0.03);
    const colW = (layout.width - colGap * (nL - 1)) / nL;
    const planeTop = layout.y + labelH;
    const planeH = layout.height - labelH - (labels ? 14 : 0); // footer for channel counts

    for (let li = 0; li < nL; li++) {
      const m = layers[li];
      const { cell, max } = reducePlane(m);
      const cx0 = layout.x + li * (colW + colGap);
      const cw = (colW - innerPad * 2) / m.w;
      const cellH = (planeH - innerPad * 2) / m.h;
      const px0 = cx0 + innerPad;
      const py0 = planeTop + innerPad;

      // faint plate so empty planes still read as a layer
      ctx.fillStyle = "rgba(12,16,24,0.55)";
      ctx.fillRect(cx0, planeTop, colW, planeH);

      let actSum = 0;
      for (let r = 0; r < m.h; r++) {
        for (let c = 0; c < m.w; c++) {
          const a = Math.min(1, cell[r * m.w + c] / max);
          actSum += a;
          if (a < 0.02) continue;
          ctx.fillStyle = hexA(VIS, 0.14 + 0.82 * a);
          ctx.fillRect(px0 + c * cw, py0 + r * cellH, cw - 0.5, cellH - 0.5);
        }
      }
      ctx.strokeStyle = hexA(VIS, 0.3);
      ctx.lineWidth = 1;
      ctx.strokeRect(cx0 + 0.5, planeTop + 0.5, colW - 1, planeH - 1);

      const midX = cx0 + colW / 2;
      centers.push({ x: midX, y: planeTop + planeH / 2, act: actSum / (m.h * m.w), name: m.name });

      if (labels) {
        ctx.font = "600 10px ui-monospace, monospace";
        ctx.textAlign = "center";
        ctx.textBaseline = "alphabetic";
        ctx.fillStyle = hexA(VIS, 0.9);
        ctx.fillText(layerLabel(m.name), midX, layout.y + 11);
        ctx.font = "9px ui-monospace, monospace";
        ctx.fillStyle = "rgba(120,130,150,0.85)";
        ctx.fillText(`${m.channels}ch`, midX, planeTop + planeH + 11);
      }

      // forward arrow between adjacent layers
      if (labels && li < nL - 1) {
        ctx.fillStyle = hexA(VIS, 0.55);
        ctx.font = "10px ui-monospace, monospace";
        ctx.fillText("→", cx0 + colW + colGap / 2, planeTop + planeH / 2 + 3);
      }
    }
    ctx.textAlign = "left";
    return centers;
  }

  // ── 3D: positioning is fully delegated to the injected projection ──
  const project = layout.project;
  const meta = { h: 0, w: 0, layerCount: nL };
  const prevComp = ctx.globalCompositeOperation;
  ctx.globalCompositeOperation = "lighter";
  for (let li = 0; li < nL; li++) {
    const m = layers[li];
    meta.h = m.h;
    meta.w = m.w;
    const { cell, max } = reducePlane(m);
    let actSum = 0;
    let cxSum = 0,
      cySum = 0,
      cN = 0;
    for (let r = 0; r < m.h; r++) {
      for (let c = 0; c < m.w; c++) {
        const a = Math.min(1, cell[r * m.w + c] / max);
        actSum += a;
        const p = project(li, r, c, meta);
        if (!(p.depth >= 0.5)) continue;
        cxSum += p.x;
        cySum += p.y;
        cN++;
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
    const cp = cN ? { x: cxSum / cN, y: cySum / cN } : project(li, (m.h - 1) / 2, (m.w - 1) / 2, meta);
    centers.push({ x: cp.x, y: cp.y, act: Math.min(1, actSum / (m.h * m.w)), name: m.name });
  }
  ctx.globalCompositeOperation = prevComp;
  return centers;
}

// ── 3) saliency: collapse the feature pathway to a maze-sized grid ───────────

/**
 * Collapse the learned feature maps to a single `size`×`size` saliency grid for
 * the maze attention overlay. The "sight" map is skipped (it is the input image,
 * not a learned response). Each feature plane is normalized to its own peak,
 * resampled to size×size by nearest-cell, and averaged across layers; the result
 * is normalized to [0,1].
 *
 * Returns a length `size*size` array (row-major). Returns all-zeros if there are
 * no learned planes.
 */
export function cortexSaliency(maps: RetinaMap[], size: number): number[] {
  const n = Math.max(1, size | 0);
  const out = new Array<number>(n * n).fill(0);
  const layers = (maps ?? []).filter((m) => m.name !== "sight" && m.h > 0 && m.w > 0);
  if (!layers.length) return out;

  for (const m of layers) {
    const { cell, max } = reducePlane(m);
    for (let r = 0; r < n; r++) {
      const sr = Math.min(m.h - 1, Math.floor((r * m.h) / n));
      for (let c = 0; c < n; c++) {
        const sc = Math.min(m.w - 1, Math.floor((c * m.w) / n));
        out[r * n + c] += cell[sr * m.w + sc] / max;
      }
    }
  }

  let mx = 1e-6;
  for (let i = 0; i < out.length; i++) {
    out[i] /= layers.length;
    if (out[i] > mx) mx = out[i];
  }
  for (let i = 0; i < out.length; i++) out[i] = Math.min(1, out[i] / mx);
  return out;
}
