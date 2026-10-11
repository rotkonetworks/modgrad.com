// Array views (ported unchanged): PCA state trajectories, point clouds, stacked volume slices,
// and 1-D / 2-D tensors over steps. Each returns the explanatory note for the view.
import { A } from '@/lib/wasm';
import { svgEl, showTip, hideTip } from '@/lib/dom';
import { finiteRange } from '@/lib/fmt';
import { PALETTE, colorScale, tickColor } from '@/lib/color';
import { Scene, axisTicks, surface3d, type Axes } from '@/charts/scene';
import { heatmap } from '@/charts/heatmap';
import { chart2d } from '@/charts/chart2d';
import { kvTable } from '@/charts/legend';

function pcaBlock(rows: number, d: number, data: Float32Array): { pts: Float64Array; bounds: number[]; ex: number[] } {
  const p = A.pca3(data, rows, d); const c = p.coords(), ex = Array.from(p.explained()); p.free();
  const xyz = Float64Array.from(c);
  const nb = A.normalize_box(xyz); const pts = nb.slice(0, xyz.length), bounds = Array.from(nb.slice(xyz.length));
  return { pts, bounds, ex };
}
function pcAxes(bounds: number[], ex: number[]): Axes {
  return {
    x: { label: `PC1 ${(ex[0] * 100).toFixed(1)}%`, ticks: axisTicks(bounds[0], bounds[1], 3) },
    y: { label: `PC2 ${(ex[1] * 100).toFixed(1)}%`, ticks: axisTicks(bounds[2], bounds[3], 3) },
    z: { label: `PC3 ${(ex[2] * 100).toFixed(1)}%`, ticks: axisTicks(bounds[4], bounds[5], 3) },
  };
}
export function drawTrajectory(box: Element, title: string, shape: number[], data: Float32Array, step: number): string {
  const [ns, nt, d] = shape.length === 3 ? shape : [1, shape[0], shape[1]];
  const { pts, bounds, ex } = pcaBlock(ns * nt, d, data);
  const scene = new Scene(box, {
    title: `${title}: state trajectory, step ${step}`, axes: pcAxes(bounds, ex),
    draw: (s) => {
      const P = s.proj(Array.from(pts));
      const segs: [number, number, number][] = [];
      for (let a = 0; a < ns; a++) for (let t = 0; t + 1 < nt; t++) segs.push([a * nt + t, a * nt + t + 1, t]);
      const order = A.depth_order(Float64Array.from(segs.map(([u, v]) => (P[u].d + P[v].d) / 2)));
      const g = svgEl('g', {}, s.g);
      for (const q of order) { const [u, v, t] = segs[q]; svgEl('line', { x1: P[u].x, y1: P[u].y, x2: P[v].x, y2: P[v].y, style: `stroke:${tickColor(t, nt - 1)};stroke-width:1.4` }, g); }
      for (let k = 0; k < ns * nt; k++) {
        const t = k % nt;
        const c = svgEl('circle', { cx: P[k].x, cy: P[k].y, r: 1.6, style: `fill:${tickColor(t, nt)}` }, g);
        c.addEventListener('mousemove', (ev) => showTip(ev, `sample ${Math.floor(k / nt)} tick ${t}`)); c.addEventListener('mouseleave', hideTip);
      }
      for (let a = 0; a < ns; a++) {
        const p0 = P[a * nt], p1 = P[a * nt + nt - 1];
        svgEl('circle', { cx: p0.x, cy: p0.y, r: 4, style: 'fill:var(--bg);stroke:var(--frame);stroke-width:1' }, g);
        svgEl('rect', { x: p1.x - 3.5, y: p1.y - 3.5, width: 7, height: 7, style: 'fill:var(--frame)' }, g);
        const e = svgEl('text', { x: p1.x + 6, y: p1.y - 4, style: 'font:10px var(--mono);fill:var(--ink)' }, g); e.textContent = 's' + a;
      }
    },
  });
  scene.render();
  kvTable(scene.panel, [['PC1 explained', (ex[0] * 100).toFixed(1) + '%'], ['PC2 explained', (ex[1] * 100).toFixed(1) + '%'], ['PC3 explained', (ex[2] * 100).toFixed(1) + '%'], ['PC1-3 total', ((ex[0] + ex[1] + ex[2]) * 100).toFixed(1) + '%'], ['samples x ticks x neurons', `${ns} x ${nt} x ${d}`]]);
  // companion: distance travelled per tick (2D, RRD style)
  const series = [];
  for (let a = 0; a < Math.min(ns, 6); a++) {
    const xs: number[] = [], ys: number[] = [];
    for (let t = 1; t < nt; t++) { let s2 = 0; for (let q = 0; q < d; q++) { const v = data[(a * nt + t) * d + q] - data[(a * nt + t - 1) * d + q]; s2 += v * v; } xs.push(t); ys.push(Math.sqrt(s2)); }
    series.push({ name: `sample ${a}`, color: PALETTE[a], area: a === 0, xs, ys });
  }
  chart2d(box, { title: `${title}: |state(t) - state(t-1)| per tick (full ${d}-d)`, xlabel: 'tick', series });
  return `Neuron state per tick for ${ns} sample${ns > 1 ? 's' : ''} x ${nt} ticks x ${d} neurons, projected jointly onto the top 3 principal components (computed here, in wasm). Colour runs light (early tick) to dark (late); open circle = start, filled square = end.`;
}
export function drawPoints(box: Element, title: string, shape: number[], data: Float32Array, labels: Float32Array | null, step: number): string {
  const [n, d] = shape;
  const { pts, bounds, ex } = pcaBlock(n, d, data);
  const labs = labels ? Array.from(labels) : new Array<number>(n).fill(0);
  const uniq = [...new Set(labs)].sort((a, b) => a - b);
  const scene = new Scene(box, {
    title: `${title}: point cloud, step ${step}`, axes: pcAxes(bounds, ex),
    draw: (s) => {
      const P = s.proj(Array.from(pts));
      const order = A.depth_order(Float64Array.from(P.map((p) => p.d)));
      const g = svgEl('g', {}, s.g);
      for (const k of order) {
        const li = uniq.indexOf(labs[k]);
        const c = li === 1 ? svgEl('rect', { x: P[k].x - 2.5, y: P[k].y - 2.5, width: 5, height: 5 }, g) : svgEl('circle', { cx: P[k].x, cy: P[k].y, r: 2.6 }, g);
        c.setAttribute('style', `fill:var(${PALETTE[li % 6]});stroke:var(--bg);stroke-width:0.6`);
        c.addEventListener('mousemove', (ev) => showTip(ev as MouseEvent, `point ${k}  label ${labs[k]}`)); c.addEventListener('mouseleave', hideTip);
      }
    },
  });
  scene.render();
  const rows: [string, string][] = uniq.map((u, i) => [`label ${u} (${i === 1 ? 'square' : 'circle'})`, `${labs.filter((x) => x === u).length} points`]);
  const t = kvTable(scene.panel, [...rows, ['PC1 / PC2 / PC3 explained', ex.map((e) => (e * 100).toFixed(1) + '%').join(' / ')]]);
  uniq.forEach((_, i) => { const td = t.rows[i].cells[0]; const sw = document.createElement('span'); sw.className = 'sw-area'; sw.style.background = `var(${PALETTE[i % 6]})`; sw.style.marginRight = '6px'; td.prepend(sw); });
  return `${n} points of dimension ${d}, projected onto the top 3 principal components (wasm). ${labels ? 'Coloured by label.' : 'No labels logged.'}`;
}
export function drawVolume(box: Element, title: string, shape: number[], data: Float32Array, step: number, ti: number): string {
  const [nt, hh, ww] = shape;
  const [lo, hi] = finiteRange([data]);
  const sc = colorScale(lo, hi);
  const scene = new Scene(box, {
    title: `${title}: stacked slices, step ${step}`, el: 22,
    axes: { x: { label: 'x', ticks: axisTicks(0, ww - 1, 3) }, y: { label: 'y', ticks: axisTicks(0, hh - 1, 3) }, z: { label: 'tick', ticks: axisTicks(0, nt - 1, 4) } },
    draw: (s) => {
      const nx = (c: number) => (ww > 1 ? 2 * c / (ww - 1) - 1 : 0), ny = (r: number) => (hh > 1 ? 2 * r / (hh - 1) - 1 : 0), nz = (t: number) => (nt > 1 ? 2 * t / (nt - 1) - 1 : 0);
      const hx = ww > 1 ? 1 / (ww - 1) : 1, hy = hh > 1 ? 1 / (hh - 1) : 1;
      const planeD = s.proj(Array.from({ length: nt }, (_, t) => [0, 0, nz(t)]).flat()).map((p) => p.d);
      const order = Array.from(A.depth_order(Float64Array.from(planeD)));
      const g = svgEl('g', {}, s.g);
      for (const t of order) {
        const sel = t === ti;
        const corners = s.proj([-1, -1, nz(t), 1, -1, nz(t), 1, 1, nz(t), -1, 1, nz(t)]);
        svgEl('polygon', { points: corners.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' '), style: `fill:none;stroke:${sel ? 'var(--accent)' : 'var(--grid)'};stroke-width:${sel ? 1.2 : 0.6}` }, g);
        for (let r = 0; r < hh; r++) for (let c = 0; c < ww; c++) {
          const v = data[(t * hh + r) * ww + c];
          const a = hi > lo ? (v - lo) / (hi - lo) : 0;
          if (!(a > 0.15) && !sel) continue;
          const q = s.proj([nx(c) - hx, ny(r) - hy, nz(t), nx(c) + hx, ny(r) - hy, nz(t), nx(c) + hx, ny(r) + hy, nz(t), nx(c) - hx, ny(r) + hy, nz(t)]);
          svgEl('polygon', { points: q.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' '), style: `fill:${sc(v)};fill-opacity:${(sel ? 0.95 : 0.04 + 0.22 * a).toFixed(2)};stroke:none` }, g);
        }
      }
    },
  });
  scene.render();
  const plane = Array.from(data.slice(ti * hh * ww, (ti + 1) * hh * ww));
  // heatmap rows = y (top row first) -> flip so row 0 is at the top
  const z: number[] = []; for (let r = hh - 1; r >= 0; r--) for (let c = 0; c < ww; c++) z.push(plane[r * ww + c]);
  heatmap(box, { title: `${title}: tick ${ti}`, gx: Array.from({ length: hh }, (_, i) => hh - 1 - i), gy: Array.from({ length: ww }, (_, i) => i), z, xlabel: 'y', ylabel: 'x', zlabel: 'value', scale: sc, H: 300 });
  const per: number[] = []; for (let t = 0; t < nt; t++) { let m = 0; for (let k = 0; k < hh * ww; k++) m += data[t * hh * ww + k]; per.push(m / (hh * ww)); }
  chart2d(box, { title: `${title}: mean value per tick`, xlabel: 'tick', series: [{ name: 'mean over cells', color: '--c1', area: true, xs: per.map((_, t) => t), ys: per }], marker: [ti] });
  return `Volume ${nt} ticks x ${hh} x ${ww}: each tick is a plane, stacked bottom (tick 0) to top; cell opacity follows the value. The highlighted plane is the tick slider's; the heatmap shows it exactly.`;
}
export { surface3d, heatmap, chart2d };
