// Heatmap companion to the surfaces: rows = gx (vertical), cols = gy (horizontal). Missing = hatched.
// Ported unchanged from the vanilla dashboard.
import { h, svgEl, showTip, hideTip } from '@/lib/dom';
import { fmt, fmtTick, finiteRange } from '@/lib/fmt';
import { colorScale, type Scale } from '@/lib/color';
import { legendTable } from '@/charts/legend';
import { colorBar } from '@/charts/scene';

export interface HeatmapOpts {
  title: string; gx: number[]; gy: number[]; z: number[]; xlabel: string; ylabel: string; zlabel: string; H?: number;
  scale?: Scale; xfmt?: (v: number) => string; yfmt?: (v: number) => string; mark?: [number, number] | null;
  onpick?: (i: number, j: number) => void; tipText?: (i: number, j: number) => string;
}
export function heatmap(parent: Element, o: HeatmapOpts): HTMLDivElement {
  const nx = o.gx.length, ny = o.gy.length;
  const W = 540, H = o.H || 240, L = 64, R = 56, T = 10, B = 28, pw = W - L - R, ph = H - T - B;
  const panel = h('div', { class: 'panel' }, parent);
  h('div', { class: 'ptitle' }, panel, o.title);
  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': o.title }, panel);
  const defs = svgEl('defs', {}, svg);
  const pid = 'hatch' + Math.random().toString(36).slice(2, 8);
  const pat = svgEl('pattern', { id: pid, width: 5, height: 5, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, defs);
  svgEl('line', { x1: 0, y1: 0, x2: 0, y2: 5, style: 'stroke:var(--miss);stroke-width:1' }, pat);
  const [zl, zh] = finiteRange([o.z]);
  const sc = o.scale || colorScale(zl, zh);
  const cw = pw / ny, chh = ph / nx;
  const g = svgEl('g', { 'shape-rendering': 'crispEdges' }, svg);
  for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) {
    const k = i * ny + j, v = o.z[k];
    const r = svgEl('rect', { x: L + j * cw, y: T + (nx - 1 - i) * chh, width: cw + 0.3, height: chh + 0.3, style: Number.isFinite(v) ? `fill:${sc(v)}` : `fill:url(#${pid})` }, g);
    r.addEventListener('mousemove', (ev) => showTip(ev, o.tipText ? o.tipText(i, j) : `${o.xlabel} ${fmt(o.gx[i])}\n${o.ylabel} ${fmt(o.gy[j])}\n${o.zlabel} ${Number.isFinite(v) ? fmt(v) : 'no data'}`));
    r.addEventListener('mouseleave', hideTip);
    if (o.onpick) { const pick = o.onpick; r.style.cursor = 'pointer'; r.addEventListener('click', () => pick(i, j)); }
  }
  if (o.mark) { const [i, j] = o.mark; svgEl('rect', { x: L + j * cw, y: T + (nx - 1 - i) * chh, width: cw, height: chh, style: 'fill:none;stroke:var(--accent);stroke-width:1.5' }, svg); }
  svgEl('rect', { x: L + 0.5, y: T + 0.5, width: pw - 1, height: ph - 1, style: 'fill:none;stroke:var(--frame);stroke-width:1', 'shape-rendering': 'crispEdges' }, svg);
  const lab = svgEl('g', { style: 'font:10px var(--mono);fill:var(--ink)' }, svg);
  const every = (n: number, max: number) => Math.max(1, Math.ceil(n / max));
  const fx = o.xfmt || fmtTick, fy = o.yfmt || fmtTick;
  for (let i = 0; i < nx; i += every(nx, 8)) { const e = svgEl('text', { x: L - 4, y: T + (nx - 1 - i + 0.5) * chh + 3, 'text-anchor': 'end' }, lab); e.textContent = fx(o.gx[i]); }
  for (let j = 0; j < ny; j += every(ny, 8)) { const e = svgEl('text', { x: L + (j + 0.5) * cw, y: T + ph + 13, 'text-anchor': 'middle' }, lab); e.textContent = fy(o.gy[j]); }
  { const e = svgEl('text', { x: L + pw, y: H - 3, 'text-anchor': 'end', style: 'fill:var(--muted)' }, lab); e.textContent = o.ylabel; }
  { const e = svgEl('text', { x: 10, y: T + ph / 2, transform: `rotate(-90 10 ${T + ph / 2})`, 'text-anchor': 'middle', style: 'fill:var(--muted)' }, lab); e.textContent = o.xlabel; }
  colorBar(svg, W - R + 8, T + 12, 10, Math.min(120, ph - 16), sc, '');
  const fin = Array.from(o.z).filter(Number.isFinite);
  const missing = o.z.length - fin.length;
  legendTable(panel, [{ name: `${o.zlabel}`, values: fin }], '');
  if (missing) h('div', { class: 'sub' }, panel, `${missing} of ${o.z.length} cells have no data (hatched)`);
  return panel;
}
