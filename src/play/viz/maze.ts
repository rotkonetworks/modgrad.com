// ── Maze renderer ────────────────────────────────────────────────────────────
// Pure 2D-canvas renderer for the /play maze: the maze, the true shortest route
// as a thin neutral reference line, and the brain's PREDICTED route as it
// evolves over the brain's internal ticks.
//
// The predicted route is the brain's own output (its 20 × 5 route head, argmax
// per step), decoded per tick in the worker. Earlier ticks are drawn as faint
// curves whose colour runs along a light → accent gradient by tick; the tick on
// screen now is drawn on top, and the final tick is emphasised. Where the final
// prediction first leaves the shortest route is ringed; a predicted step into a
// wall is crossed out. Nothing here moves an agent or picks a move.
//
// PURE: depends only on its arguments. The caller owns canvas sizing / DPR.

export type Cell = [number, number];

/** One tick's decoded route (see worker.ts `TickRoute`). */
export interface TickRoute {
  cells: Cell[];
  end: "goal" | "wall" | "wait" | "length";
}

/** A square maze. `grid` is row-major, `size`×`size`; 1 = wall, 0 = open. */
export interface Maze {
  grid: number[];
  start: Cell;
  end: Cell;
}

export interface MazeRenderState {
  /** Unique shortest route start → goal (start excluded). Reference only. */
  shortest: Cell[];
  /** The brain's decoded route at every internal tick, in order. */
  routes: TickRoute[];
  /** Tick currently shown (0 ≤ tick < routes.length). */
  tick: number;
  /** Index of the first step where the FINAL route leaves `shortest`. */
  agree: number;
}

/** Theme palette, resolved by the caller from CSS vars (light / dark). */
export interface MazeTheme {
  wall: string;
  open: string;
  line: string; // hairline cell stroke
  mute: string; // neutral reference line (shortest route)
  accent: string; // brand accent — the brain's route, start / goal markers
}

const MISS_COLOR = "#e0564e"; // red — wall step / divergence marker

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number): void {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}

/**
 * Stroke a polyline through points as a smooth curve: straight segments are
 * joined by quadratic corners through each interior vertex's midpoints, so a
 * grid path reads as one continuous vector stroke.
 */
function smoothPath(ctx: CanvasRenderingContext2D, pts: [number, number][]): void {
  ctx.beginPath();
  if (pts.length === 0) return;
  ctx.moveTo(pts[0][0], pts[0][1]);
  if (pts.length === 1) return;
  for (let i = 1; i < pts.length - 1; i++) {
    const [x, y] = pts[i];
    const [nx, ny] = pts[i + 1];
    ctx.quadraticCurveTo(x, y, (x + nx) / 2, (y + ny) / 2);
  }
  const last = pts[pts.length - 1];
  ctx.lineTo(last[0], last[1]);
}

/** Accent mixed toward the open-cell colour: t=0 faint/early → t=1 full accent. */
function tickColor(accent: string, open: string, t: number, alpha = 1): string {
  const pct = Math.round(28 + 72 * t);
  const c = `color-mix(in srgb, ${accent} ${pct}%, ${open})`;
  return alpha >= 1 ? c : `color-mix(in srgb, ${c} ${Math.round(alpha * 100)}%, transparent)`;
}

export function drawMaze(
  ctx: CanvasRenderingContext2D,
  w: number,
  _h: number,
  maze: Maze,
  size: number,
  state: MazeRenderState,
  theme: MazeTheme,
): void {
  const n = size;
  if (n <= 0) return;
  const gap = 3;
  const cell = (w - gap * (n + 1)) / n;
  if (cell <= 0) return;
  const { wall, open, line, mute, accent } = theme;

  const cx = (c: number): number => gap + c * (cell + gap) + cell / 2;
  const cy = (r: number): number => gap + r * (cell + gap) + cell / 2;
  const pt = ([r, c]: Cell): [number, number] => [cx(c), cy(r)];
  const isWall = (r: number, c: number): boolean => maze.grid[r * n + c] === 1;

  // ── base cells ──
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      ctx.fillStyle = isWall(r, c) ? wall : open;
      roundRect(ctx, gap + c * (cell + gap), gap + r * (cell + gap), cell, cell, 4);
      ctx.fill();
      if (!isWall(r, c)) {
        ctx.strokeStyle = line;
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }
  }

  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  const start = pt(maze.start);

  // ── shortest route: thin neutral reference ──
  if (state.shortest.length) {
    ctx.strokeStyle = mute;
    ctx.globalAlpha = 0.75;
    ctx.lineWidth = Math.max(1, cell * 0.045);
    ctx.setLineDash([cell * 0.12, cell * 0.12]);
    smoothPath(ctx, [start, ...state.shortest.map(pt)]);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  }

  // ── the brain's predicted route over ticks ──
  const routes = state.routes;
  const nT = routes.length;
  const cur = Math.max(0, Math.min(nT - 1, state.tick));
  const isFinal = cur === nT - 1;
  if (nT > 0) {
    const tOf = (i: number) => (nT > 1 ? i / (nT - 1) : 1);
    // earlier ticks: faint ghosts, coloured by tick
    for (let i = 0; i < cur; i++) {
      const cells = routes[i].cells;
      if (!cells.length) continue;
      ctx.strokeStyle = tickColor(accent, open, tOf(i), 0.35);
      ctx.lineWidth = Math.max(1, cell * 0.04);
      smoothPath(ctx, [start, ...cells.map(pt)]);
      ctx.stroke();
    }
    // the tick on screen now
    const now = routes[cur];
    if (now.cells.length) {
      ctx.strokeStyle = tickColor(accent, open, tOf(cur));
      ctx.lineWidth = isFinal ? Math.max(2, cell * 0.1) : Math.max(1.5, cell * 0.07);
      smoothPath(ctx, [start, ...now.cells.map(pt)]);
      ctx.stroke();
      // the step it predicts into a wall / off the board
      if (now.end === "wall") {
        const [x, y] = pt(now.cells[now.cells.length - 1]);
        const s = cell * 0.2;
        ctx.strokeStyle = MISS_COLOR;
        ctx.lineWidth = Math.max(1.5, cell * 0.07);
        ctx.beginPath();
        ctx.moveTo(x - s, y - s);
        ctx.lineTo(x + s, y + s);
        ctx.moveTo(x + s, y - s);
        ctx.lineTo(x - s, y + s);
        ctx.stroke();
      }
    }
    // final tick: ring the first cell where the prediction leaves the
    // shortest route (if it does).
    if (isFinal && state.agree < state.shortest.length) {
      const off = now.cells[state.agree];
      // (a wall step is already crossed out — don't ring the same cell twice)
      const isWallStep = now.end === "wall" && state.agree === now.cells.length - 1;
      if (off && !isWallStep) {
        const [x, y] = pt(off);
        ctx.strokeStyle = MISS_COLOR;
        ctx.lineWidth = Math.max(1.5, cell * 0.05);
        ctx.beginPath();
        ctx.arc(x, y, cell * 0.3, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
  }

  // ── start + goal ──
  ctx.fillStyle = accent;
  ctx.beginPath();
  ctx.arc(start[0], start[1], cell * 0.22, 0, Math.PI * 2);
  ctx.fill();
  const [gx, gy] = pt(maze.end);
  const s = cell * 0.26;
  ctx.save();
  ctx.translate(gx, gy);
  ctx.rotate(Math.PI / 4);
  ctx.fillStyle = "#3CB371";
  roundRect(ctx, -s, -s, s * 2, s * 2, 3);
  ctx.fill();
  ctx.restore();
}
