// ── Maze renderer ────────────────────────────────────────────────────────────
// A self-contained, strongly-typed 2D-canvas renderer for the /play maze and
// every overlay the demo layers on top of it. This is a faithful, modularised
// extraction of the `drawMaze()` that used to live inlined in Play.tsx.
//
// Design intent (why this is its own module):
//   • PURE: `drawMaze()` takes everything it needs as arguments — no Solid
//     signals, no module-level mutable state, no DOM reads. Same inputs ⇒ same
//     pixels. Easy to test, easy to call from a RAF loop or a worker-fed render.
//   • LAYERED CHANNELS: the visualisation carries three independent attribution
//     signals that previously fought for the same pixels (amber attention, the
//     purple actual-trail, the teal predicted-route). They are drawn as distinct
//     visual channels — colour, geometry, and z-order are chosen so they read
//     cleanly even when they overlap on the same cell.
//   • GPU-light: plain Canvas2D, rounded rects, a couple of strokes per cell.
//
// DIR convention (must match the worker's DELTA): UP=0 DOWN=1 LEFT=2 RIGHT=3.

// ── Direction deltas: [dRow, dCol], indexed by move id ───────────────────────
export const DIR_UP = 0;
export const DIR_DOWN = 1;
export const DIR_LEFT = 2;
export const DIR_RIGHT = 3;

/** [dRow, dCol] per direction. Index with the DIR_* constants above. */
export const DIR_DELTA: ReadonlyArray<readonly [number, number]> = [
  [-1, 0], // UP
  [1, 0], //  DOWN
  [0, -1], // LEFT
  [0, 1], //  RIGHT
];

// ── Channel colours ──────────────────────────────────────────────────────────
// Fixed hues for the attribution channels so the legend and the canvas agree.
// (The paper-theme colours — wall/open/line/accent — are passed in via `theme`
// so the renderer follows light/dark and brand changes without hard-coding.)
const ATTN_COLOR = "#f59e0b"; // amber  — occlusion attention heatmap
const ROUTE_COLOR = "#13b7a4"; // teal   — the brain's predicted route
const MISSTEP_COLOR = "#e0564e"; // red    — predicted move would hit a wall / stray

// ── Types ────────────────────────────────────────────────────────────────────

/** A square maze. `grid` is row-major, `size`×`size`; 1 = wall, 0 = open. */
export interface Maze {
  grid: number[];
  start: [number, number]; // [row, col]
  end: [number, number]; // [row, col]
}

/** The brain's verdict on its last predicted move vs. the maze / optimal step. */
export type Verdict = "ok" | "wall" | "astray" | "wait";

/**
 * Everything that changes frame-to-frame. Snapshotted by the caller and passed
 * in whole — the renderer never reaches back into app state.
 */
export interface MazeRenderState {
  /** Agent cell, [row, col]. */
  agent: [number, number];
  /** Cells the agent has actually stepped through, oldest → newest. */
  visited: [number, number][];
  /** Per-cell occlusion attribution in [0,1], row-major; null when unavailable. */
  attn: number[] | null;
  /** The brain's decoded multi-step predicted route, agent → future cells. */
  route: [number, number][];
  /** Verdict on the brain's last predicted move; null when there's nothing to flag. */
  verdict: Verdict | null;
  /** The move id (DIR_*) the brain committed to last; -1 when none. */
  committedMove: number;
  /** True while the brain is mid-decision (drives the agent pulse ring). */
  thinking: boolean;
  /** Current tick within the thinking animation (0 ≤ tick ≤ ticksTotal). */
  tick: number;
  /** Total ticks in the thinking animation; used to normalise `tick`. */
  ticksTotal: number;
  /** True while the agent is resting during a sleep/consolidation pass. */
  sleeping?: boolean;
}

/** The paper-theme palette, resolved by the caller (e.g. from CSS vars). */
export interface MazeTheme {
  wall: string; // closed cells
  open: string; // open cells
  line: string; // hairline grid stroke + neutral marks
  accent: string; // brand accent — agent, trail, start/goal markers
}

/** One legend entry: a swatch colour and its human label. */
export interface LegendEntry {
  color: string;
  label: string;
}

// ── Legend ───────────────────────────────────────────────────────────────────

/**
 * The canonical legend for the maze overlays, so the surrounding UI can render
 * a legend that always matches what the canvas draws. `accent` is threaded in
 * because the agent/trail/markers follow the live theme accent.
 */
export function mazeLegend(accent = "#6243D9"): LegendEntry[] {
  return [
    { color: accent, label: "Agent + actual trail" },
    { color: ATTN_COLOR, label: "Attention (occlusion attribution)" },
    { color: ROUTE_COLOR, label: "Predicted route (brain's plan)" },
    { color: MISSTEP_COLOR, label: "Misstep (predicted move is wrong)" },
  ];
}

// ── Local drawing primitives (kept inside the module so it stands alone) ──────

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
): void {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}

/** Start marker: a small hollow accent ring centred on the cell. */
function startRing(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  cell: number,
  accent: string,
): void {
  ctx.beginPath();
  ctx.strokeStyle = `color-mix(in srgb, ${accent} 70%, transparent)`;
  ctx.lineWidth = 2;
  ctx.arc(x + cell / 2, y + cell / 2, cell * 0.22, 0, Math.PI * 2);
  ctx.stroke();
}

/** Goal marker: a filled, rounded accent diamond centred at (cx, cy). */
function goalDiamond(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  cell: number,
  accent: string,
): void {
  const s = cell * 0.3;
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(Math.PI / 4);
  ctx.fillStyle = accent;
  roundRect(ctx, -s, -s, s * 2, s * 2, 3);
  ctx.fill();
  ctx.restore();
}

const clamp01 = (v: number): number => (v < 0 ? 0 : v > 1 ? 1 : v);

// ── Main render ──────────────────────────────────────────────────────────────

/**
 * Render the maze and all of its overlays into a 2D context.
 *
 * Pure: depends only on its arguments. The caller is responsible for sizing the
 * canvas / handling DPR and for clearing the frame beforehand if desired.
 *
 * @param ctx   destination 2D context
 * @param w     logical width  (px) of the drawing area
 * @param h     logical height (px) of the drawing area (square maze ⇒ w === h)
 * @param maze  static maze (grid + start + end)
 * @param size  grid dimension (cells per side)
 * @param state per-frame render state (agent, overlays, verdict, animation)
 * @param theme resolved paper-theme palette
 */
export function drawMaze(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  maze: Maze,
  size: number,
  state: MazeRenderState,
  theme: MazeTheme,
): void {
  const n = size;
  if (n <= 0) return;

  const gap = 3;
  // Cell size derives from width; the maze is square so height tracks width.
  const cell = (w - gap * (n + 1)) / n;
  if (cell <= 0) return;

  const { wall, open, line, accent } = theme;

  const cx = (c: number): number => gap + c * (cell + gap);
  const cy = (r: number): number => gap + r * (cell + gap);
  const inBounds = (r: number, c: number): boolean =>
    r >= 0 && r < n && c >= 0 && c < n;
  const isWall = (r: number, c: number): boolean => maze.grid[r * n + c] === 1;

  // ── Layer 0: base cells ────────────────────────────────────────────────────
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      ctx.fillStyle = isWall(r, c) ? wall : open;
      roundRect(ctx, cx(c), cy(r), cell, cell, 4);
      ctx.fill();
      if (!isWall(r, c)) {
        ctx.strokeStyle = line;
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }
  }

  // ── Layer 1: ATTENTION heatmap (amber) ─────────────────────────────────────
  // Occlusion attribution: how much each cell mattered to the current decision.
  // Drawn first of the overlays (under trail/route) so it reads as a background
  // wash rather than competing with the line-based channels above it. Inset by
  // 1px so the grid hairlines stay visible and the channels don't blur together.
  const attn = state.attn;
  if (attn) {
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        const a = Math.pow(clamp01(attn[r * n + c]), 1.25);
        if (a < 0.06) continue;
        ctx.fillStyle = `rgba(245, 158, 11, ${(0.12 + 0.6 * a).toFixed(3)})`;
        roundRect(ctx, cx(c) + 1, cy(r) + 1, cell - 2, cell - 2, 4);
        ctx.fill();
      }
    }
  }

  // ── Layer 2: actual TRAIL (accent) ─────────────────────────────────────────
  // Where the agent really went — a soft accent fill per cell plus a connecting
  // poly-line. Distinct from the teal route (the brain's *plan*) by colour and
  // by being a solid line vs. the route's dashed beam.
  const path = state.visited;
  ctx.fillStyle = `color-mix(in srgb, ${accent} 22%, transparent)`;
  for (const [r, c] of path) {
    roundRect(ctx, cx(c), cy(r), cell, cell, 4);
    ctx.fill();
  }
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
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  // ── Layer 3: start & goal markers ──────────────────────────────────────────
  startRing(ctx, cx(maze.start[1]), cy(maze.start[0]), cell, accent);
  goalDiamond(
    ctx,
    cx(maze.end[1]) + cell / 2,
    cy(maze.end[0]) + cell / 2,
    cell,
    accent,
  );

  const [ar, ac] = state.agent;
  const ax = cx(ac) + cell / 2;
  const ay = cy(ar) + cell / 2;

  // ── Layer 4: predicted ROUTE (teal) ────────────────────────────────────────
  // The brain's decoded multi-step plan: a dashed teal beam from the agent into
  // the future, with a crosshair at each predicted cell, fading with distance.
  // Dashed + crosshair geometry keeps it legible over the solid accent trail
  // and the amber wash beneath.
  const route = state.route;
  if (route.length) {
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.strokeStyle = `color-mix(in srgb, ${ROUTE_COLOR} 50%, transparent)`;
    ctx.lineWidth = Math.max(2, cell * 0.09);
    ctx.setLineDash([cell * 0.16, cell * 0.16]);
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    for (const [r, c] of route) ctx.lineTo(cx(c) + cell / 2, cy(r) + cell / 2);
    ctx.stroke();
    ctx.setLineDash([]);

    route.forEach(([r, c], i) => {
      const fade = 1 - i / (route.length + 1);
      const px = cx(c) + cell / 2;
      const py = cy(r) + cell / 2;
      ctx.strokeStyle = `color-mix(in srgb, ${ROUTE_COLOR} ${Math.round(
        30 + 55 * fade,
      )}%, transparent)`;
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

  // ── Layer 5: the agent ─────────────────────────────────────────────────────
  ctx.beginPath();
  ctx.fillStyle = accent;
  ctx.arc(ax, ay, cell * 0.3, 0, Math.PI * 2);
  ctx.fill();
  ctx.lineWidth = 2.5;
  ctx.strokeStyle = open;
  ctx.stroke();

  // Pulse ring while the brain is thinking; radius grows with the tick.
  if (state.thinking) {
    ctx.beginPath();
    ctx.strokeStyle = `color-mix(in srgb, ${accent} 60%, transparent)`;
    ctx.lineWidth = 2;
    const pulse = 0.34 + (state.tick / Math.max(1, state.ticksTotal)) * 0.18;
    ctx.arc(ax, ay, cell * pulse, 0, Math.PI * 2);
    ctx.stroke();
  }

  // Sleeping: a rising "z z Z" trail off the agent's dot so it reads as napping
  // while the planner consolidates.
  if (state.sleeping) {
    ctx.save();
    ctx.fillStyle = open; // light z's on the dark accent dot
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const zs = [
      { dx: 0.16, dy: -0.30, s: 0.30, t: "z" },
      { dx: 0.34, dy: -0.50, s: 0.40, t: "z" },
      { dx: 0.56, dy: -0.74, s: 0.54, t: "Z" },
    ];
    for (const z of zs) {
      ctx.font = `700 ${Math.round(cell * z.s)}px ui-monospace, monospace`;
      ctx.fillText(z.t, ax + cell * z.dx, ay + cell * z.dy);
    }
    ctx.restore();
  }

  // ── Layer 6: MISSTEP indicator (red) ───────────────────────────────────────
  // When the brain's predicted move would hit a wall or stray, flag it: a red
  // arrow from the agent toward the (wrong) target cell, plus a marker on that
  // cell — an ✕ for a wall, a dashed ring for an astray-but-open cell.
  const v = state.verdict;
  const pm = state.committedMove;
  if (v && v !== "ok" && v !== "wait" && pm >= 0 && pm < 4) {
    const [pr, pc] = DIR_DELTA[pm];
    const tr = ar + pr;
    const tc = ac + pc;

    // arrow toward the brain's (wrong) choice
    ctx.strokeStyle = MISSTEP_COLOR;
    ctx.lineWidth = 2.5;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(ax + pc * cell * 0.5, ay + pr * cell * 0.5);
    ctx.stroke();

    // mark the target cell
    if (inBounds(tr, tc)) {
      ctx.strokeStyle = MISSTEP_COLOR;
      ctx.lineWidth = 2;
      const ix = cx(tc);
      const iy = cy(tr);
      if (v === "wall" && isWall(tr, tc)) {
        // ✕ for a wall
        ctx.beginPath();
        ctx.moveTo(ix + cell * 0.3, iy + cell * 0.3);
        ctx.lineTo(ix + cell * 0.7, iy + cell * 0.7);
        ctx.moveTo(ix + cell * 0.7, iy + cell * 0.3);
        ctx.lineTo(ix + cell * 0.3, iy + cell * 0.7);
        ctx.stroke();
      } else {
        // dashed ring for an astray (open) cell
        ctx.setLineDash([4, 3]);
        roundRect(ctx, ix + 2, iy + 2, cell - 4, cell - 4, 4);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }
  }
}
