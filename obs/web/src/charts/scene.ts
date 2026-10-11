// 3D scene (gnuplot splot look), surfaces and colour bars: ported unchanged from the vanilla
// dashboard. Navigation: drag = rotate, shift/right/middle-drag = pan, wheel = zoom at the cursor
// (once the view has focus; ctrl+wheel always), two fingers = pinch zoom + pan; keys on the
// focused view only: arrows rotate, shift+arrows pan, +/- zoom, 0 or Esc reset; double-click resets.
import { A } from '@/lib/wasm';
import { h, svgEl, showTip, hideTip } from '@/lib/dom';
import { fmt, fmtTick, finiteRange } from '@/lib/fmt';
import { colorScale, type Scale } from '@/lib/color';
import { legendTable } from '@/charts/legend';

export type Ticks = [number, string][];
export interface Axes { x: { label: string; ticks: Ticks }; y: { label: string; ticks: Ticks }; z: { label: string; ticks: Ticks } }
export interface SceneOpts { title: string; axes: Axes; az?: number; el?: number; H?: number; draw: (s: Scene) => void; after?: (s: Scene) => void }
export interface P3 { x: number; y: number; d: number }

export class Scene {
  o: SceneOpts; W: number; H: number;
  az = 0; el = 0; zoom = 1; panX = 0; panY = 0;
  panel: HTMLDivElement; svg: SVGSVGElement; sub: HTMLDivElement; g!: SVGGElement;

  constructor(parent: Element, o: SceneOpts) {
    this.o = o; this.W = 540; this.H = o.H || 380;
    this.reset(false);
    this.panel = h('div', { class: 'panel' }, parent);
    h('div', { class: 'ptitle' }, this.panel, o.title);
    const svg = this.svg = svgEl('svg', { viewBox: `0 0 ${this.W} ${this.H}`, class: 'drag', role: 'img', 'aria-label': o.title, tabindex: 0 }, this.panel);
    this.sub = h('div', { class: 'sub' }, this.panel, 'drag rotate, shift-drag pan, click then wheel to zoom, double-click reset');
    const pt = svg.createSVGPoint();
    const local = (e: { clientX: number; clientY: number }) => { pt.x = e.clientX; pt.y = e.clientY; return pt.matrixTransform(svg.getScreenCTM()!.inverse()); };
    const ptrs = new Map<number, { x: number; y: number }>(); // pointerId -> viewBox point
    let drag: { pan: boolean; x: number; y: number; az: number; el: number; panX: number; panY: number } | null = null;
    let pinch: { d: number; mx: number; my: number; zoom: number; panX: number; panY: number } | null = null;
    const pinchState = () => { const [a, b] = [...ptrs.values()]; return { d: Math.hypot(a.x - b.x, a.y - b.y) || 1, mx: (a.x + b.x) / 2, my: (a.y + b.y) / 2, zoom: this.zoom, panX: this.panX, panY: this.panY }; };
    svg.addEventListener('contextmenu', (e) => e.preventDefault());
    svg.addEventListener('pointerdown', (e) => {
      if ((e.target as SVGElement).dataset.pick) return;
      e.preventDefault(); svg.focus({ preventScroll: true });
      svg.setPointerCapture(e.pointerId);
      const p = local(e); ptrs.set(e.pointerId, p);
      if (ptrs.size === 2) { drag = null; pinch = pinchState(); return; }
      const pan = e.shiftKey || e.button === 1 || e.button === 2;
      drag = { pan, x: p.x, y: p.y, az: this.az, el: this.el, panX: this.panX, panY: this.panY };
      svg.style.cursor = pan ? 'move' : 'grabbing';
    });
    svg.addEventListener('pointermove', (e) => {
      if (!ptrs.has(e.pointerId)) return;
      const p = local(e); ptrs.set(e.pointerId, p);
      if (pinch && ptrs.size === 2) {
        const q = pinchState(), f = q.d / pinch.d;
        this.zoomAt(pinch.mx, pinch.my, pinch.zoom * f, pinch);
        this.panX += q.mx - pinch.mx; this.panY += q.my - pinch.my;
        this.render(); return;
      }
      if (!drag) return;
      if (drag.pan) { this.panX = drag.panX + (p.x - drag.x); this.panY = drag.panY + (p.y - drag.y); }
      else { this.az = drag.az + (p.x - drag.x) * 0.5; this.el = Math.max(-89, Math.min(89, drag.el + (p.y - drag.y) * 0.4)); }
      this.render();
    });
    const end = (e: PointerEvent) => { ptrs.delete(e.pointerId); if (ptrs.size < 2) pinch = null; if (!ptrs.size) { drag = null; svg.style.cursor = ''; } };
    svg.addEventListener('pointerup', end); svg.addEventListener('pointercancel', end);
    svg.addEventListener('dblclick', () => this.reset(true));
    svg.addEventListener('wheel', (e) => {
      if (!(e.ctrlKey || e.metaKey || document.activeElement === svg)) return; // otherwise the page scrolls
      e.preventDefault();
      const p = local(e);
      this.zoomAt(p.x, p.y, this.zoom * Math.pow(0.998, e.deltaY));
      this.render();
    }, { passive: false });
    svg.addEventListener('keydown', (e) => {
      const k = e.key, st = 24;
      if (k === 'ArrowLeft' || k === 'ArrowRight' || k === 'ArrowUp' || k === 'ArrowDown') {
        const dx = k === 'ArrowLeft' ? -1 : k === 'ArrowRight' ? 1 : 0, dy = k === 'ArrowUp' ? -1 : k === 'ArrowDown' ? 1 : 0;
        if (e.shiftKey) { this.panX += dx * st; this.panY += dy * st; } else { this.az += dx * 5; this.el = Math.max(-89, Math.min(89, this.el + dy * 5)); }
      } else if (k === '+' || k === '=') this.zoomAt(this.W / 2, this.H / 2, this.zoom * 1.2);
      else if (k === '-' || k === '_') this.zoomAt(this.W / 2, this.H / 2, this.zoom / 1.2);
      else if (k === '0' || k === 'Escape') { this.reset(false); }
      else return;
      e.preventDefault(); this.render();
    });
  }
  reset(draw: boolean): void { this.az = this.o.az ?? -50; this.el = this.o.el ?? 28; this.zoom = 1; this.panX = 0; this.panY = 0; if (draw) this.render(); }
  /** zoom to z keeping the viewBox point (mx, my) fixed; `from` = {zoom, panX, panY} to zoom relative to */
  zoomAt(mx: number, my: number, z: number, from: { zoom: number; panX: number; panY: number } = this): void {
    z = Math.max(0.3, Math.min(40, z));
    const cx = this.W / 2, cy = (this.H - 16) / 2, r = z / from.zoom;
    this.panX = mx - cx - (mx - cx - from.panX) * r;
    this.panY = my - cy - (my - cy - from.panY) * r;
    this.zoom = z;
  }
  /** flat [x,y,z,...] in the unit box -> screen points */
  proj(xyz: ArrayLike<number>): P3[] {
    const p = A.project(Float64Array.from(xyz), this.az, this.el);
    const sc = this.zoom * Math.min(this.W, this.H - 30) / 3.7, cx = this.W / 2 + this.panX, cy = (this.H - 16) / 2 + this.panY;
    const out: P3[] = [];
    for (let i = 0; i < p.length; i += 3) out.push({ x: cx + p[i] * sc, y: cy - p[i + 1] * sc, d: p[i + 2] });
    return out;
  }
  render(): void {
    const svg = this.svg; while (svg.firstChild) svg.removeChild(svg.firstChild);
    this.g = svgEl('g', {}, svg);
    this.box();
    this.o.draw(this);
    if (this.o.after) this.o.after(this);
  }
  /** box with base grid, labelled axes on the outer (near) edges */
  box(): void {
    const o = this.o, g = svgEl('g', {}, this.g);
    const ax = o.axes;
    const grid = 'stroke:var(--grid);stroke-width:0.6;stroke-dasharray:1 2;fill:none';
    const line = (a: number[], b: number[], st: string) => { const [p, q] = this.proj([...a, ...b]); svgEl('line', { x1: p.x, y1: p.y, x2: q.x, y2: q.y, style: st }, g); };
    for (const [n] of ax.x.ticks) line([n, -1, -1], [n, 1, -1], grid);
    for (const [n] of ax.y.ticks) line([-1, n, -1], [1, n, -1], grid);
    // back walls: pick the wall planes facing away from the viewer
    const c = this.proj([0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0]);
    const backX = c[1].d > c[2].d ? 1 : -1, backY = c[3].d > c[4].d ? 1 : -1;
    for (const [n] of ax.z.ticks) { line([-1, backY, n], [1, backY, n], grid); line([backX, -1, n], [backX, 1, n], grid); }
    const edge = 'stroke:var(--frame);stroke-width:0.8;fill:none';
    const C = [-1, 1];
    for (const y of C) for (const z of C) line([-1, y, z], [1, y, z], z === -1 ? edge : grid);
    for (const x of C) for (const z of C) line([x, -1, z], [x, 1, z], z === -1 ? edge : grid);
    for (const x of C) for (const y of C) line([x, y, -1], [x, y, 1], (x === backX && y === backY) ? edge : grid);
    // tick labels on the nearer base edges, pushed outward from the centre
    const center = this.proj([0, 0, -1])[0];
    const lab = svgEl('g', { style: 'font:10px var(--mono);fill:var(--ink)' }, this.g);
    const put = (ticks: Ticks, label: string, mk: (n: number) => number[]) => {
      for (const [n, txt] of ticks) {
        const p = this.proj(mk(n))[0];
        let dx = p.x - center.x, dy = p.y - center.y; const L = Math.hypot(dx, dy) || 1; dx /= L; dy /= L;
        const e = svgEl('text', { x: p.x + dx * 14, y: p.y + dy * 14 + 3, 'text-anchor': dx < -0.3 ? 'end' : dx > 0.3 ? 'start' : 'middle' }, lab);
        e.textContent = txt;
      }
      const m = this.proj(mk(0))[0];
      let dx = m.x - center.x, dy = m.y - center.y; const L = Math.hypot(dx, dy) || 1; dx /= L; dy /= L;
      const e = svgEl('text', { x: m.x + dx * 48, y: m.y + dy * 40 + 3, 'text-anchor': dx < -0.3 ? 'end' : dx > 0.3 ? 'start' : 'middle', style: 'fill:var(--muted)' }, lab);
      e.textContent = label;
    };
    const nearY = c[3].d < c[4].d ? 1 : -1, nearX = c[1].d < c[2].d ? 1 : -1;
    put(ax.x.ticks, ax.x.label, (n) => [n, nearY, -1]);
    put(ax.y.ticks, ax.y.label, (n) => [nearX, n, -1]);
    // z axis on the leftmost vertical edge
    const corners = [[-1, -1], [-1, 1], [1, -1], [1, 1]];
    const sx = corners.map(([x, y]) => this.proj([x, y, 0])[0].x);
    const [zx, zy] = corners[sx.indexOf(Math.min(...sx))];
    line([zx, zy, -1], [zx, zy, 1], edge);
    for (const [n, txt] of ax.z.ticks) { const p = this.proj([zx, zy, n])[0]; const e = svgEl('text', { x: p.x - 6, y: p.y + 3, 'text-anchor': 'end' }, lab); e.textContent = txt; }
    const top = this.proj([zx, zy, 1])[0];
    const e = svgEl('text', { x: top.x - 4, y: top.y - 12, 'text-anchor': 'end', style: 'fill:var(--muted)' }, lab); e.textContent = ax.z.label;
  }
}

/** axis ticks: real-valued [lo,hi] -> [[normalized, label]] */
export function axisTicks(lo: number, hi: number, n: number, f: (v: number) => string = fmtTick): Ticks {
  if (!(hi > lo)) return [[0, f(lo)]];
  return Array.from(A.nice_ticks(lo, hi, n)).filter((t) => t >= lo - 1e-12 && t <= hi + 1e-12).map((t) => [2 * (t - lo) / (hi - lo) - 1, f(t)] as [number, string]);
}
export function colorBar(svg: Element, x: number, y: number, w: number, hgt: number, sc: Scale, label: string): void {
  const g = svgEl('g', {}, svg); const N = 40;
  for (let i = 0; i < N; i++) {
    const v = sc.lo + (sc.hi - sc.lo) * (1 - i / (N - 1));
    svgEl('rect', { x, y: y + (i * hgt) / N, width: w, height: hgt / N + 0.6, style: `fill:${sc(v)}` }, g);
  }
  svgEl('rect', { x: x + 0.5, y: y + 0.5, width: w - 1, height: hgt - 1, style: 'fill:none;stroke:var(--frame);stroke-width:1' }, g);
  const lab = svgEl('g', { style: 'font:10px var(--mono);fill:var(--ink)' }, g);
  for (const [t, ty] of [[sc.hi, y + 4], [(sc.lo + sc.hi) / 2, y + hgt / 2 + 3], [sc.lo, y + hgt]]) {
    const e = svgEl('text', { x: x + w + 3, y: ty }, lab); e.textContent = (t !== 0 && Math.abs(t) < 1e-3) ? t.toExponential(0) : fmtTick(+t.toPrecision(3));
  }
  if (label) { const e = svgEl('text', { x: x + w / 2, y: y - 4, 'text-anchor': 'middle', style: 'fill:var(--muted)' }, lab); e.textContent = label; }
}

export interface SurfaceOpts {
  title: string; gx: number[]; gy: number[]; z: number[]; logx?: boolean; xlabel: string; ylabel: string; zlabel: string;
  xfmt?: (v: number) => string; az?: number; el?: number; onpick?: (i: number, j: number) => void; tipText?: (i: number, j: number) => string;
}
/** grid surface z[i*ny+j] over gx[i] x gy[j], hidden3d by painter's order */
export function surface3d(parent: Element, o: SurfaceOpts): Scene {
  const nx = o.gx.length, ny = o.gy.length;
  const gx = o.logx ? o.gx.map((v) => (v > 0 ? Math.log10(v) : NaN)) : o.gx;
  const [xl, xh] = finiteRange([gx]), [yl, yh] = finiteRange([o.gy]), [zl, zh] = finiteRange([o.z]);
  const nrm = (v: number, lo: number, hi: number) => (hi > lo ? 2 * (v - lo) / (hi - lo) - 1 : 0);
  const sc = colorScale(zl, zh);
  const scene = new Scene(parent, {
    title: o.title, az: o.az, el: o.el,
    axes: {
      x: { label: o.xlabel, ticks: axisTicks(xl, xh, 4, o.logx ? (t) => fmtTick(Math.pow(10, t)) : (o.xfmt || fmtTick)) },
      y: { label: o.ylabel, ticks: axisTicks(yl, yh, 4) },
      z: { label: o.zlabel, ticks: axisTicks(zl, zh, 4) },
    },
    draw: (s) => {
      const xyz: number[] = [];
      for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) xyz.push(nrm(gx[i], xl, xh), nrm(o.gy[j], yl, yh), Number.isFinite(o.z[i * ny + j]) ? nrm(o.z[i * ny + j], zl, zh) : NaN);
      const P = s.proj(xyz.map((v) => (Number.isFinite(v) ? v : 0)));
      const depth = Float64Array.from(P.map((p, k) => (Number.isFinite(xyz[3 * k + 2]) ? p.d : NaN)));
      const g = svgEl('g', {}, s.g);
      // missing cells: x on the base plane
      const base = s.proj(xyz.flatMap((v, k) => (k % 3 === 2 ? [-1] : [v])));
      for (let k = 0; k < nx * ny; k++) if (!Number.isFinite(o.z[k])) {
        const p = base[k];
        svgEl('path', { d: `M${p.x - 3},${p.y - 3}L${p.x + 3},${p.y + 3}M${p.x - 3},${p.y + 3}L${p.x + 3},${p.y - 3}`, style: 'stroke:var(--miss);stroke-width:1' }, g);
      }
      const order = A.painter_order(nx, ny, depth);
      for (const q of order) {
        const i = Math.floor(q / (ny - 1)), j = q % (ny - 1);
        const ks = [i * ny + j, i * ny + j + 1, (i + 1) * ny + j + 1, (i + 1) * ny + j];
        const zm = ks.reduce((a, k) => a + o.z[k], 0) / 4;
        svgEl('polygon', { points: ks.map((k) => P[k].x.toFixed(1) + ',' + P[k].y.toFixed(1)).join(' '), style: `fill:var(--bg);stroke:${sc(zm)};stroke-width:1` }, g);
      }
      // rows/columns that have no complete quad: polylines (curtain), plus vertex dots for picking
      const drawn = new Set<number>(); for (const q of order) { const i = Math.floor(q / (ny - 1)), j = q % (ny - 1); for (const k of [i * ny + j, i * ny + j + 1, (i + 1) * ny + j, (i + 1) * ny + j + 1]) drawn.add(k); }
      for (let i = 0; i < nx; i++) {
        let d = '';
        for (let j = 0; j < ny; j++) { const k = i * ny + j; if (Number.isFinite(o.z[k])) { d += (d && Number.isFinite(o.z[k - 1]) && j > 0 ? 'L' : 'M') + P[k].x.toFixed(1) + ',' + P[k].y.toFixed(1); } }
        if (d && (ny < 2 || nx < 2 || [...Array(ny).keys()].some((j) => Number.isFinite(o.z[i * ny + j]) && !drawn.has(i * ny + j)))) svgEl('path', { d, style: `fill:none;stroke:var(--c${(i % 6) + 1});stroke-width:1` }, g);
      }
      const dots = svgEl('g', {}, s.g);
      for (let k = 0; k < nx * ny; k++) if (Number.isFinite(o.z[k])) {
        const iso = !drawn.has(k);
        svgEl('circle', { cx: P[k].x, cy: P[k].y, r: iso ? 2.5 : 1.6, style: `fill:${sc(o.z[k])};stroke:${iso ? 'var(--frame)' : 'none'};stroke-width:0.6` }, dots);
        if (o.onpick) {
          const hc = svgEl('circle', { cx: P[k].x, cy: P[k].y, r: 6, style: 'fill:transparent;cursor:pointer', 'data-pick': '1' }, dots);
          const i = Math.floor(k / ny), j = k % ny;
          const pick = o.onpick;
          hc.addEventListener('click', () => pick(i, j));
          hc.addEventListener('mousemove', (ev) => showTip(ev, o.tipText ? o.tipText(i, j) : `${fmt(o.gx[i])}, ${fmt(o.gy[j])}: ${fmt(o.z[k])}`));
          hc.addEventListener('mouseleave', hideTip);
        }
      }
      colorBar(s.svg, s.W - 52, 24, 10, 120, sc, '');
    },
  });
  scene.render();
  const fin = Array.from(o.z).filter(Number.isFinite);
  const missing = o.z.length - fin.length;
  legendTable(scene.panel, [{ name: `${o.zlabel} (${nx}x${ny} grid, ${missing} missing)`, color: null, values: fin }], '');
  return scene;
}
