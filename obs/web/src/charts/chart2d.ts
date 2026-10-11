// 2D chart, RRD look (ported unchanged from the vanilla dashboard; the SVG it draws is identical).
//
// Zoom: drag = x range (y refits to what is visible), alt-drag = box, shift-drag = pan,
// wheel = zoom at the cursor (once the chart has focus; ctrl/cmd + wheel always), double-click or
// Esc = reset. Charts that share o.group share their x range. Legend Max/Avg/Min/Cur cover the
// visible window only. o.ylog: log10 y axis (non-positive values become gaps). A series with
// y2: true gets its own fitted range and right-hand axis (one per chart). o.decorateLegend(table,
// series) runs after every legend render. Series options: hidden (legend row only), band {lo, hi}
// (filled envelope), ghost {xs, ys, lo, hi} (dashed forecast). o.yrange fixes the default y range
// (locked small multiples); o.onview(view) fires on user zoom; o.overlay(api) draws extra
// interactive layers above the plot after every render.
import { A } from '@/lib/wasm';
import { h, svgEl, showTip, hideTip } from '@/lib/dom';
import { fmt, fmtTick, finiteRange, nums } from '@/lib/fmt';
import { legendTable } from '@/charts/legend';

type Arr = ArrayLike<number>;
export interface View { x0: number; x1: number; y0?: number; y1?: number }
export interface Ghost { xs: Arr; ys: Arr; lo: Arr; hi: Arr }
export interface SeriesIn {
  name: string; color: string; xs: Arr; ys: Arr; raw?: Arr; area?: boolean; hidden?: boolean; y2?: boolean;
  band?: { lo: Arr; hi: Arr } | null; ghost?: Ghost | null; [extra: string]: unknown;
}
/** a series as drawn: transformed coordinates plus the caller's fields */
export interface Series extends SeriesIn {
  si: number; X: number[]; Y: number[]; V: number[]; S: Float64Array; BL: number[] | null; BH: number[] | null; D?: number[];
  G?: { X: number[]; Y: number[]; LO: number[]; HI: number[]; n0: number };
}
export interface OverlayApi {
  g: SVGGElement; svg: SVGSVGElement; X: (x: number) => number; Y: (y: number) => number;
  L: number; T: number; pw: number; ph: number; W: number; H: number; x0: number; x1: number; y0: number; y1: number;
  tx: (x: number) => number; ty: (y: number) => number; view: View | null;
  local: (ev: { clientX: number; clientY: number }) => DOMPoint; setView: (v: View | null) => void;
}
export interface Chart2DOpts {
  title: string; series: SeriesIn[]; xlabel?: string; ylabel?: string; xfmt?: (v: number) => string;
  xlog?: boolean; ylog?: boolean; group?: string | null; marker?: number[]; legendHead?: string;
  ghostNotes?: [string | null, string][] | null; yrange?: [number, number] | null;
  decorateLegend?: (t: HTMLTableElement | null, series: Series[]) => void;
  onview?: (v: View | null) => void; overlay?: (api: OverlayApi) => void; onclick?: (ev: PointerEvent) => void;
  /** drawing size in CSS px (default 540 x 210); pass the box's real width so text stays ~10 px on screen */
  width?: number; height?: number;
}
export interface Chart2D {
  panel: HTMLDivElement;
  update(name: string, xs: Arr, ys: Arr, raw?: Arr): boolean;
  setView(v: View | null, fromGroup?: boolean): void;
}
export interface ZoomGroup { view: View | null; charts: Set<Chart2D> }

const ZOOM_GROUPS = new Map<string, ZoomGroup>();
export function zoomGroup(name: string): ZoomGroup {
  let g = ZOOM_GROUPS.get(name);
  if (!g) { g = { view: null, charts: new Set() }; ZOOM_GROUPS.set(name, g); }
  return g;
}

/** Set the x window of every zoom group whose name passes `pred` and redraw its charts (jump to a
 *  step from outside a chart, e.g. the anomaly list). `v` is in transformed x; null resets. Returns
 *  the number of charts moved. */
export function setGroupViews(pred: (name: string) => boolean, v: View | null): number {
  let n = 0;
  for (const [name, g] of ZOOM_GROUPS) {
    if (!pred(name)) continue;
    g.view = v ? { x0: v.x0, x1: v.x1 } : null;
    for (const c of [...g.charts]) {
      if (!c.panel.isConnected) { g.charts.delete(c); continue; }
      c.setView(g.view, true); n++;
    }
  }
  return n;
}

/** y range of the visible values, else of everything, padded when flat */
function fitY(visible: number[][], all: { Y: number[] }[]): [number, number] {
  let [a, b] = finiteRange(visible);
  if (!Number.isFinite(a)) [a, b] = finiteRange(all.map((s) => s.Y));
  if (!Number.isFinite(a)) { a = 0; b = 1; }
  if (b === a) { const d = Math.abs(a) * 0.05 || 0.5; a -= d; b += d; }
  return [a, b];
}
/** ticks in log10 space: 1-2-5 per decade over 1-3 decades, whole decades beyond, else nice linear values */
function logTicks(a: number, b: number): number[] {
  if (b - a >= 1 && b - a < 3) { const t: number[] = []; for (let k = Math.floor(a); k <= Math.ceil(b); k++) for (const m of [1, 2, 5]) { const v = k + Math.log10(m); if (v >= a - 1e-9 && v <= b + 1e-9) t.push(v); } return t; }
  if (b - a >= 1) { const st = Math.max(1, Math.ceil((Math.floor(b) - Math.ceil(a) + 1) / 6)); const t: number[] = []; for (let k = Math.ceil(a); k <= Math.floor(b); k += st) t.push(k); return t; }
  return Array.from(A.nice_ticks(Math.pow(10, a), Math.pow(10, b), 5)).filter((v) => v > 0).map(Math.log10);
}
const nice = (a: number, b: number, n: number): number[] => Array.from(A.nice_ticks(a, b, n));

let clipSeq = 0;
// drags in progress on any chart: callers that rebuild charts (live panels) wait for them to end
let activeDrags = 0;
export const chartBusy = (): boolean => activeDrags > 0;
export function chart2d(parent: Element, o: Chart2DOpts): HTMLDivElement & { chart?: Chart2D } {
  const hasY2 = o.series.some((s) => s.y2) && o.series.some((s) => !s.y2);
  const W = Math.max(280, Math.round(o.width || 540)), H = Math.round(o.height || Math.max(170, (W * 210) / 540)), L = 58, R = hasY2 ? 52 : 14, T = 10, B = 28;
  const pw = W - L - R, ph = H - T - B;
  const panel = h('div', { class: 'panel' }, parent) as HTMLDivElement & { chart?: Chart2D };
  h('div', { class: 'ptitle' }, panel, o.title);
  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': o.title, tabindex: 0, class: 'zoom' }, panel);
  const sub = h('div', { class: 'sub' }, panel);
  const tx = o.xlog ? (x: number) => (x > 0 ? Math.log10(x) : NaN) : (x: number) => x;
  const xf = o.xlog ? (t: number) => fmtTick(Math.pow(10, t)) : (o.xfmt || fmtTick);
  const ty = o.ylog ? (y: number) => (y > 0 ? Math.log10(y) : NaN) : (y: number) => y;
  const yf = o.ylog ? (t: number) => fmtTick(+Math.pow(10, t).toPrecision(4)) : fmtTick;
  const series: Series[] = o.series.map((s, si) => ({ ...s, si, y2: hasY2 && !!s.y2, X: Array.from(s.xs, tx), Y: Array.from(s.ys, ty), V: Array.from(s.ys), S: nums(s.raw || s.ys), BL: s.band ? Array.from(s.band.lo, ty) : null, BH: s.band ? Array.from(s.band.hi, ty) : null }));
  const shown = series.filter((s) => !s.hidden);
  // ghost forecast: drawn dashed with a band, kept out of the legend stats; its x extends the
  // default range, its mean (not its band) takes part in the y fit
  for (const s of series) {
    if (!s.ghost) continue;
    let k = s.X.length - 1; while (k >= 0 && !(Number.isFinite(s.X[k]) && Number.isFinite(s.Y[k]))) k--;
    const a: [number[], number[]] = k >= 0 ? [[s.X[k]], [s.Y[k]]] : [[], []];
    s.G = { X: a[0].concat(Array.from(s.ghost.xs, tx)), Y: a[1].concat(Array.from(s.ghost.ys, ty)), LO: a[1].concat(Array.from(s.ghost.lo, ty)), HI: a[1].concat(Array.from(s.ghost.hi, ty)), n0: a[0].length };
  }
  const ghosts = series.filter((s) => s.G) as (Series & { G: NonNullable<Series['G']> })[];
  let [fx0, fx1] = finiteRange(series.map((s) => s.X).concat(ghosts.map((s) => s.G.X)));
  if (!Number.isFinite(fx0)) { fx0 = 0; fx1 = 1; }
  if (fx1 === fx0) { fx0 -= 0.5; fx1 += 0.5; }
  const cid = 'clip' + (++clipSeq);
  let layer: SVGGElement | null = null, legend: Element | null = null, ov: SVGGElement | null = null;
  let x0 = 0, x1 = 1, y0 = 0, y1 = 1, z0 = 0, z1 = 1;
  const group = o.group ? zoomGroup(o.group) : null;
  let view: View | null = group ? group.view : null; // in transformed x
  const inX = (x: number) => x >= x0 && x <= x1;

  function render(): void {
    x0 = view ? view.x0 : fx0; x1 = view ? view.x1 : fx1;
    // fit y to the visible points plus the nearest point either side, so lines leaving the window stay in frame
    const nearOf = (s: { X: number[] }, Arr: number[]) => Arr.filter((_, i) => inX(s.X[i]) || (s.X[i] < x0 && !(s.X[i + 1] < x0)) || (s.X[i] > x1 && !(s.X[i - 1] > x1)));
    const near = (s: { X: number[]; Y: number[] }) => nearOf(s, s.Y);
    if (view && view.y0 !== undefined && view.y1 !== undefined) { y0 = view.y0; y1 = view.y1; }
    else if (o.yrange) { [y0, y1] = o.yrange.map(ty) as [number, number]; if (!(y1 > y0)) { y0 -= 0.5; y1 += 0.5; } }
    else {
      const prim = shown.filter((s) => !s.y2);
      const bands = prim.filter((s) => s.BL).flatMap((s) => [nearOf(s, s.BL!), nearOf(s, s.BH!)]);
      [y0, y1] = fitY(prim.map(near).concat(ghosts.filter((s) => !s.hidden).map((s) => near(s.G)), bands), prim.length ? prim : series);
    }
    if (hasY2) {
      const sec = series.filter((s) => s.y2);
      [z0, z1] = fitY(sec.map(near), sec);
      if (!o.ylog) { const zt = nice(z0, z1, 5); if (zt.length > 1) { const st = zt[1] - zt[0]; z0 = Math.floor(z0 / st + 1e-9) * st; z1 = Math.ceil(z1 / st - 1e-9) * st; } }
    }
    let yt = o.ylog ? logTicks(y0, y1) : nice(y0, y1, 5);
    if (!o.ylog && yt.length > 1 && !(view && view.y0 !== undefined)) { // widen to whole tick steps so the frame starts and ends on a gridline
      const st = yt[1] - yt[0];
      y0 = Math.floor(y0 / st + 1e-9) * st; y1 = Math.ceil(y1 / st - 1e-9) * st;
      yt = nice(y0, y1, 5);
    }
    const xt = nice(x0, x1, 6);
    const X = (x: number) => L + ((x - x0) / (x1 - x0)) * pw;
    const Y = (y: number) => T + ph - ((y - y0) / (y1 - y0)) * ph;
    for (const s of series) s.D = s.y2 ? s.Y.map((v) => y0 + ((v - z0) / (z1 - z0)) * (y1 - y0)) : s.Y; // y2 drawn in primary coordinates
    const next = svgEl('g', {});
    const cp = svgEl('clipPath', { id: cid }, svgEl('defs', {}, next));
    svgEl('rect', { x: L, y: T, width: pw, height: ph }, cp);
    const g = svgEl('g', { 'shape-rendering': 'crispEdges' }, next);
    for (const t of yt) if (t >= y0 - 1e-12 && t <= y1 + 1e-12) svgEl('line', { x1: L, x2: L + pw, y1: Y(t), y2: Y(t), style: 'stroke:var(--grid);stroke-dasharray:1 2' }, g);
    for (const t of xt) if (inX(t)) svgEl('line', { x1: X(t), x2: X(t), y1: T, y2: T + ph, style: 'stroke:var(--grid);stroke-dasharray:1 2' }, g);
    const lab = svgEl('g', { style: 'font:10px var(--mono);fill:var(--ink)' }, next);
    for (const t of yt) if (t >= y0 - 1e-12 && t <= y1 + 1e-12) { const e = svgEl('text', { x: L - 4, y: Y(t) + 3, 'text-anchor': 'end' }, lab); e.textContent = yf(t); }
    if (hasY2) {
      const s2 = series.find((s) => s.y2)!, Z = (z: number) => T + ph - ((z - z0) / (z1 - z0)) * ph;
      const zt = o.ylog ? logTicks(z0, z1) : nice(z0, z1, 5);
      for (const t of zt) if (t >= z0 - 1e-12 && t <= z1 + 1e-12) {
        svgEl('line', { x1: L + pw, x2: L + pw + 4, y1: Z(t), y2: Z(t), style: `stroke:var(${s2.color})`, 'shape-rendering': 'crispEdges' }, lab);
        const e = svgEl('text', { x: L + pw + 6, y: Z(t) + 3, 'text-anchor': 'start', style: `fill:var(${s2.color})` }, lab); e.textContent = yf(t);
      }
    }
    for (const t of xt) if (inX(t)) { const e = svgEl('text', { x: X(t), y: T + ph + 13, 'text-anchor': 'middle' }, lab); e.textContent = xf(t); }
    if (o.xlabel) { const e = svgEl('text', { x: L + pw, y: H - 3, 'text-anchor': 'end', style: 'fill:var(--muted)' }, lab); e.textContent = o.xlabel; }
    if (o.ylabel) { const e = svgEl('text', { x: 10, y: T + ph / 2, transform: `rotate(-90 10 ${T + ph / 2})`, 'text-anchor': 'middle', style: 'fill:var(--muted)' }, lab); e.textContent = o.ylabel; }
    const plot = svgEl('g', { 'clip-path': `url(#${cid})` }, next);
    const segs = (s: { X: number[]; Y: number[]; D?: number[] }) => { // contiguous finite runs
      const out: [number, number][][] = []; let cur: [number, number][] = [];
      const SY = s.D || s.Y;
      for (let i = 0; i < s.X.length; i++) {
        if (Number.isFinite(s.X[i]) && Number.isFinite(SY[i])) cur.push([X(s.X[i]), Y(SY[i])]);
        else if (cur.length) { out.push(cur); cur = []; }
      }
      if (cur.length) out.push(cur);
      return out;
    };
    const dots = view && series.every((s) => s.X.filter(inX).length <= 60); // zoomed in far enough: mark samples
    for (const s of shown.filter((s) => s.BL)) { // envelope bands (e.g. mean±sd over seeds)
      const idx = s.X.map((_, i) => i).filter((i) => Number.isFinite(s.X[i]) && Number.isFinite(s.BL![i]) && Number.isFinite(s.BH![i]));
      const map = (Arr: number[]) => (s.y2 ? Arr.map((v) => y0 + ((v - z0) / (z1 - z0)) * (y1 - y0)) : Arr);
      const BL = map(s.BL!), BH = map(s.BH!);
      if (idx.length > 1) svgEl('path', { 'data-si': s.si, d: 'M' + idx.map((i) => X(s.X[i]).toFixed(1) + ',' + Y(BH[i]).toFixed(1)).join('L') + 'L' + idx.slice().reverse().map((i) => X(s.X[i]).toFixed(1) + ',' + Y(BL[i]).toFixed(1)).join('L') + 'Z', style: `fill:var(${s.color});fill-opacity:0.16;stroke:var(${s.color});stroke-opacity:0.35;stroke-width:0.5` }, plot);
      else for (const i of idx) svgEl('line', { 'data-si': s.si, x1: X(s.X[i]), x2: X(s.X[i]), y1: Y(BL[i]), y2: Y(BH[i]), style: `stroke:var(${s.color});stroke-width:1` }, plot);
    }
    for (const s of shown.filter((s) => s.area)) {
      for (const sg of segs(s)) {
        if (sg.length === 1) { svgEl('rect', { x: sg[0][0] - 2.5, y: sg[0][1] - 2.5, width: 5, height: 5, style: `fill:var(${s.color})` }, plot); continue; }
        const d = 'M' + sg.map((p) => p[0].toFixed(1) + ',' + p[1].toFixed(1)).join('L') + `L${sg[sg.length - 1][0].toFixed(1)},${T + ph}L${sg[0][0].toFixed(1)},${T + ph}Z`;
        svgEl('path', { 'data-si': s.si, d, style: `fill:var(${s.color});fill-opacity:0.12;stroke:none` }, plot); // a faint wash; the line is drawn below
      }
    }
    for (const s of ghosts.filter((s) => !s.hidden)) { // band, then dashed mean line
      const G = s.G, f = (i: number) => Number.isFinite(G.X[i]) && Number.isFinite(G.LO[i]) && Number.isFinite(G.HI[i]);
      const idx = G.X.map((_, i) => i).filter(f);
      if (idx.length > 1) {
        const d = 'M' + idx.map((i) => X(G.X[i]).toFixed(1) + ',' + Y(G.HI[i]).toFixed(1)).join('L') + 'L' + idx.slice().reverse().map((i) => X(G.X[i]).toFixed(1) + ',' + Y(G.LO[i]).toFixed(1)).join('L') + 'Z';
        svgEl('path', { d, style: `fill:var(${s.color});fill-opacity:0.10;stroke:var(${s.color});stroke-opacity:0.25;stroke-width:0.5` }, plot);
      }
      for (const sg of segs(G)) {
        if (sg.length === 1) { svgEl('rect', { x: sg[0][0] - 2.5, y: sg[0][1] - 2.5, width: 5, height: 5, style: `fill:none;stroke:var(${s.color});stroke-opacity:0.55;stroke-dasharray:2 1` }, plot); continue; }
        svgEl('path', { d: 'M' + sg.map((p) => p[0].toFixed(1) + ',' + p[1].toFixed(1)).join('L'), style: `fill:none;stroke:var(${s.color});stroke-width:1.2;stroke-opacity:0.55;stroke-dasharray:4 3` }, plot);
      }
    }
    for (const s of shown) {
      for (const sg of segs(s)) {
        if (sg.length === 1) { svgEl('rect', { 'data-si': s.si, x: sg[0][0] - 2.5, y: sg[0][1] - 2.5, width: 5, height: 5, style: `fill:none;stroke:var(${s.color});stroke-width:1` }, plot); continue; }
        svgEl('path', { 'data-si': s.si, d: 'M' + sg.map((p) => p[0].toFixed(1) + ',' + p[1].toFixed(1)).join('L'), style: `fill:none;stroke:var(${s.color});stroke-width:1` }, plot);
      }
    }
    if (dots) for (const s of shown) for (const sg of segs(s)) for (const p of sg) svgEl('rect', { x: p[0] - 1.5, y: p[1] - 1.5, width: 3, height: 3, style: `fill:var(--bg);stroke:var(${s.color});stroke-width:1` }, plot);
    if (o.marker) for (const m of o.marker) svgEl('line', { x1: X(tx(m)), x2: X(tx(m)), y1: T, y2: T + ph, style: 'stroke:var(--accent);stroke-width:1;stroke-dasharray:3 2' }, plot);
    svgEl('rect', { x: L + 0.5, y: T + 0.5, width: pw - 1, height: ph, style: 'fill:none;stroke:var(--frame);stroke-width:1', 'shape-rendering': 'crispEdges' }, next);
    if (layer) layer.replaceWith(next); else svg.insertBefore(next, svg.firstChild);
    layer = next;
    if (o.overlay && ov) { while (ov.firstChild) ov.removeChild(ov.firstChild); o.overlay({ g: ov, svg, X, Y, L, T, pw, ph, W, H, x0, x1, y0, y1, tx, ty, view, local, setView: (v) => me.setView(v) }); }
    // legend: statistics over the visible window
    const rows = series.map((s) => {
      const vis: number[] = []; for (let i = 0; i < s.S.length; i++) if (inX(s.X[i])) vis.push(s.S[i]);
      return { name: s.name + (s.y2 ? '  (right axis)' : ''), color: s.color, stats: Array.from(A.legend_stats(nums(vis))) };
    });
    const nl = legendTable(h('div', {}), rows, o.legendHead).parentNode as HTMLElement;
    if (o.ghostNotes && o.ghostNotes.length) {
      const gn = h('div', { class: 'gnote' }, nl);
      for (const [color, text] of o.ghostNotes) {
        const row = h('div', {}, gn);
        const sw = h('span', { class: 'sw-ghost' }, row); if (color) sw.style.borderTopColor = `var(${color})`; else sw.style.visibility = 'hidden';
        row.append(' ' + text);
      }
    }
    if (legend) legend.replaceWith(nl); else panel.appendChild(nl);
    legend = nl;
    if (o.decorateLegend) o.decorateLegend(nl.querySelector('table.legend'), series);
    sub.textContent = view
      ? `showing ${xf(+x0.toPrecision(3))} to ${xf(+x1.toPrecision(3))}${view.y0 !== undefined ? `, y ${yf(+y0.toPrecision(3))} to ${yf(+y1.toPrecision(3))}` : ''}; double-click or Esc to reset`
      : 'drag to zoom, alt-drag box, shift-drag pan, click then wheel to zoom';
  }

  const me: Chart2D = {
    panel,
    // live data: replace one series' points and redraw, keeping the zoom
    update(name, xs, ys, raw) {
      const s = series.find((q) => q.name === name);
      if (!s) return false;
      s.X = Array.from(xs, tx); s.Y = Array.from(ys); s.S = nums(raw || ys);
      [fx0, fx1] = finiteRange(series.map((q) => q.X).concat(ghosts.map((q) => q.G.X)));
      if (!Number.isFinite(fx0)) { fx0 = 0; fx1 = 1; }
      if (fx1 === fx0) { fx0 -= 0.5; fx1 += 0.5; }
      render();
      return true;
    },
    setView(v, fromGroup) {
      if (v && v.x0 <= fx0 && v.x1 >= fx1 && v.y0 === undefined) v = null;
      view = v; render();
      if (!fromGroup && o.onview) o.onview(v);
      if (group && !fromGroup) {
        group.view = v ? { x0: v.x0, x1: v.x1 } : null;
        for (const c of [...group.charts]) {
          if (!c.panel.isConnected) { group.charts.delete(c); continue; }
          if (c !== me) c.setView(group.view, true);
        }
      }
    },
  };
  if (group) group.charts.add(me);
  panel.chart = me;

  // overlay: hover hairline, rubber band, and the event surface (created once, above the layer)
  const hair = svgEl('line', { y1: T, y2: T + ph, style: 'stroke:var(--muted);stroke-width:1;display:none', 'shape-rendering': 'crispEdges' }, svg);
  const band = svgEl('rect', { style: 'fill:var(--hl);fill-opacity:0.6;stroke:var(--muted);stroke-width:1;stroke-dasharray:2 2;display:none', 'shape-rendering': 'crispEdges' }, svg);
  const hit = svgEl('rect', { x: L, y: T, width: pw, height: ph, style: 'fill:transparent;cursor:crosshair' }, svg);
  if (o.overlay) ov = svgEl('g', { class: 'overlay' }, svg);
  const pt = svg.createSVGPoint();
  const local = (ev: { clientX: number; clientY: number }) => { pt.x = ev.clientX; pt.y = ev.clientY; return pt.matrixTransform(svg.getScreenCTM()!.inverse()); };
  const clampX = (p: number) => Math.min(L + pw, Math.max(L, p));
  const clampY = (p: number) => Math.min(T + ph, Math.max(T, p));
  const dataX = (px: number) => x0 + ((px - L) / pw) * (x1 - x0);
  const dataY = (py: number) => y0 + ((T + ph - py) / ph) * (y1 - y0);
  let drag: { mode: 'pan' | 'box' | 'x'; px: number; py: number; x0: number; x1: number; y0: number; y1: number; moved: boolean; hadY: boolean } | null = null;
  hit.addEventListener('pointerdown', (ev) => {
    if (ev.button !== 0) return;
    const p = local(ev);
    if (!drag) activeDrags++;
    drag = { mode: ev.shiftKey ? 'pan' : ev.altKey ? 'box' : 'x', px: p.x, py: p.y, x0, x1, y0, y1, moved: false, hadY: !!(view && view.y0 !== undefined) };
    hit.setPointerCapture(ev.pointerId); hideTip(); hair.style.display = 'none';
    if (drag.mode === 'pan') hit.style.cursor = 'grabbing';
    ev.preventDefault(); svg.focus({ preventScroll: true });
  });
  hit.addEventListener('pointermove', (ev) => {
    const p = local(ev);
    if (drag) {
      if (Math.abs(p.x - drag.px) > 3 || Math.abs(p.y - drag.py) > 3) drag.moved = true;
      if (!drag.moved) return;
      if (drag.mode === 'pan') {
        const span = drag.x1 - drag.x0;
        let nx0 = drag.x0 - ((p.x - drag.px) / pw) * span;
        nx0 = Math.min(Math.max(nx0, fx0), fx1 - span);
        const v: View = { x0: nx0, x1: nx0 + span };
        if (drag.hadY) { const ys = drag.y1 - drag.y0, dy = ((p.y - drag.py) / ph) * ys; v.y0 = drag.y0 + dy; v.y1 = drag.y1 + dy; }
        me.setView(v);
        return;
      }
      const ax = clampX(drag.px), bx = clampX(p.x);
      const ay = drag.mode === 'box' ? clampY(drag.py) : T, by = drag.mode === 'box' ? clampY(p.y) : T + ph;
      Object.entries({ x: Math.min(ax, bx), y: Math.min(ay, by), width: Math.abs(bx - ax), height: Math.abs(by - ay) }).forEach(([k, v]) => band.setAttribute(k, String(v)));
      band.style.display = '';
      return;
    }
    const xv = dataX(p.x);
    const lines: string[] = [];
    let best: number | null = null;
    for (const s of shown) {
      let bi = -1, bd = Infinity;
      for (let i = 0; i < s.X.length; i++) { const d = Math.abs(s.X[i] - xv); if (d < bd && Number.isFinite(s.Y[i]) && inX(s.X[i])) { bd = d; bi = i; } }
      if (bi >= 0) { lines.push(`${s.name.slice(0, 28).padEnd(28)} ${fmt(s.V[bi]).padStart(9)}`); if (best === null) best = s.X[bi]; }
    }
    for (const s of ghosts) { // past a run's last real point its row shows the forecast and band
      const G = s.G; if (G.n0 && !(xv > G.X[0])) continue;
      let bi = -1, bd = Infinity;
      for (let i = G.n0; i < G.X.length; i++) { const d = Math.abs(G.X[i] - xv); if (d < bd && Number.isFinite(G.Y[i]) && inX(G.X[i])) { bd = d; bi = i; } }
      if (bi < 0) continue;
      if (best === null || bd < Math.abs(best - xv)) best = G.X[bi];
      const un = o.ylog ? (v: number) => Math.pow(10, v) : (v: number) => v;
      lines.push(`${(s.name.slice(0, 19) + ' forecast').padEnd(28)} ${fmt(un(G.Y[bi])).padStart(9)}  [${fmt(un(G.LO[bi]))}, ${fmt(un(G.HI[bi]))}]`);
    }
    if (best === null) return;
    const hx = L + ((best - x0) / (x1 - x0)) * pw;
    hair.setAttribute('x1', String(hx)); hair.setAttribute('x2', String(hx)); hair.style.display = '';
    showTip(ev, `${o.xlabel || 'x'} ${xf(best)}\n` + lines.join('\n'));
  });
  const endDrag = (ev: PointerEvent) => {
    if (!drag) return;
    const d = drag; drag = null; activeDrags = Math.max(0, activeDrags - 1); band.style.display = 'none'; hit.style.cursor = 'crosshair';
    if (!d.moved) { if (o.onclick && ev.type === 'pointerup') o.onclick(ev); return; }
    if (d.mode === 'pan') return;
    const p = local(ev);
    const ax = clampX(d.px), bx = clampX(p.x);
    if (Math.abs(bx - ax) < 4) return;
    const v: View = { x0: dataX(Math.min(ax, bx)), x1: dataX(Math.max(ax, bx)) };
    if (d.mode === 'box') {
      const ay = clampY(d.py), by = clampY(p.y);
      if (Math.abs(by - ay) >= 4) { v.y0 = dataY(Math.max(ay, by)); v.y1 = dataY(Math.min(ay, by)); }
    }
    me.setView(v);
  };
  hit.addEventListener('pointerup', endDrag);
  hit.addEventListener('pointercancel', endDrag);
  // the chart was removed or capture was taken away mid-drag: release the gesture count
  hit.addEventListener('lostpointercapture', () => { if (drag) { drag = null; activeDrags = Math.max(0, activeDrags - 1); band.style.display = 'none'; } });
  hit.addEventListener('mouseleave', () => { hair.style.display = 'none'; hideTip(); });
  hit.addEventListener('dblclick', () => me.setView(null));
  svg.addEventListener('keydown', (ev) => { if (ev.key === 'Escape' && view) { me.setView(null); ev.preventDefault(); } });
  hit.addEventListener('wheel', (ev) => {
    if (!(ev.ctrlKey || ev.metaKey || document.activeElement === svg)) return; // until the chart is clicked, the wheel scrolls the page
    ev.preventDefault();
    const c = dataX(local(ev).x), f = Math.pow(1.0015, ev.deltaY);
    let nx0 = c - (c - x0) * f, nx1 = c + (x1 - c) * f;
    if (nx1 - nx0 >= fx1 - fx0) { me.setView(view && view.y0 !== undefined ? { x0: fx0, x1: fx1, y0: view.y0, y1: view.y1 } : null); return; }
    if (nx0 < fx0) { nx1 += fx0 - nx0; nx0 = fx0; }
    if (nx1 > fx1) { nx0 -= nx1 - fx1; nx1 = fx1; }
    me.setView({ x0: nx0, x1: nx1 });
  }, { passive: false });
  render();
  return panel;
}
