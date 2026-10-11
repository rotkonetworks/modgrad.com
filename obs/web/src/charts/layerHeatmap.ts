// Layer x step heatmap (the query stage `heatmap over layer`): rows = layers (layer 0 at the
// bottom), columns = steps. The matrix comes from wasm (layers.rs); this only maps values to
// colours. Cells are painted into a canvas at one pixel per cell and shown as an SVG <image>
// (thousands of cells stay cheap); one pointer handler does tooltips and picks. Missing = hatched.
import { h, svgEl, showTip, hideTip } from '@/lib/dom';
import { fmt, fmtTick, finiteRange } from '@/lib/fmt';
import { colorScale, type Scale } from '@/lib/color';
import { A } from '@/lib/wasm';

export interface LayerHeatmapOpts {
  title: string; layers: number[]; cols: ArrayLike<number>; z: ArrayLike<number>;
  /** colour on log10 of the values (non-positive cells are drawn as missing) */
  log: boolean; zlabel: string; xlabel?: string;
  /** diverging around 1 (rownorm: 1 = typical for the layer); else a sequential scale over the range */
  centered?: boolean;
  mark?: [number, number] | null; onpick?: (row: number, col: number) => void;
}

/** sequential over [lo, hi] even when the range straddles 0 (colorScale would go diverging) */
function sequential(lo: number, hi: number): Scale {
  const base = colorScale(0, hi - lo || 1);
  const f = ((v: number) => base(v - lo)) as Scale;
  f.div = false; f.lo = lo; f.hi = hi;
  return f;
}
/** diverging around `mid` (0 in log space = a value of 1) */
function centeredScale(lo: number, hi: number, mid: number): Scale {
  const m = Math.max(Math.abs(lo - mid), Math.abs(hi - mid)) || 1;
  const base = colorScale(-m, m);
  const f = ((v: number) => base(v - mid)) as Scale;
  f.div = true; f.lo = mid - m; f.hi = mid + m;
  return f;
}

export function layerHeatmap(parent: Element, o: LayerHeatmapOpts): HTMLDivElement {
  const nr = o.layers.length, nc = o.cols.length;
  const W = 540, H = Math.max(150, Math.min(330, 11 * nr + 50)), L = 46, R = 62, T = 8, B = 28;
  const pw = W - L - R, ph = H - T - B;
  const panel = h('div', { class: 'panel lheat' }, parent);
  h('div', { class: 'ptitle' }, panel, o.title);
  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': o.title, tabindex: 0 }, panel);
  const tz = (v: number) => (o.log ? (v > 0 ? Math.log10(v) : NaN) : v);
  const t = Float64Array.from(o.z as ArrayLike<number>, tz);
  const [lo, hi] = finiteRange([t]);
  const a = Number.isFinite(lo) ? lo : 0, b = Number.isFinite(hi) ? hi : 1;
  const sc: Scale = o.centered ? centeredScale(a, b, o.log ? 0 : 1) : sequential(a, b);
  // missing cells: a hatch under the image shows through transparent pixels
  const defs = svgEl('defs', {}, svg);
  const pid = 'lhatch' + Math.random().toString(36).slice(2, 8);
  const pat = svgEl('pattern', { id: pid, width: 5, height: 5, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, defs);
  svgEl('line', { x1: 0, y1: 0, x2: 0, y2: 5, style: 'stroke:var(--miss);stroke-width:1' }, pat);
  svgEl('rect', { x: L, y: T, width: pw, height: ph, style: `fill:url(#${pid})` }, svg);
  if (nr && nc) {
    const cv = document.createElement('canvas'); cv.width = nc; cv.height = nr;
    const ctx = cv.getContext('2d');
    if (ctx) {
      const img = ctx.createImageData(nc, nr);
      const rgb = (s: string | null): [number, number, number] | null => { if (!s) return null; const m = s.match(/(\d+),\s*(\d+),\s*(\d+)/); return m ? [+m[1], +m[2], +m[3]] : null; };
      for (let i = 0; i < nr; i++) for (let j = 0; j < nc; j++) {
        const c = rgb(sc(t[i * nc + j]));
        const p = ((nr - 1 - i) * nc + j) * 4; // layer 0 at the bottom
        if (c) { img.data[p] = c[0]; img.data[p + 1] = c[1]; img.data[p + 2] = c[2]; img.data[p + 3] = 255; }
      }
      ctx.putImageData(img, 0, 0);
      svgEl('image', { href: cv.toDataURL(), x: L, y: T, width: pw, height: ph, preserveAspectRatio: 'none', style: 'image-rendering:pixelated;image-rendering:crisp-edges' }, svg);
    }
  }
  const cw = pw / Math.max(1, nc), ch = ph / Math.max(1, nr);
  if (o.mark) {
    const [i, j] = o.mark;
    svgEl('rect', { x: L + j * cw - 0.5, y: T + (nr - 1 - i) * ch - 0.5, width: Math.max(2, cw) + 1, height: ch + 1, style: 'fill:none;stroke:var(--accent);stroke-width:1.5;pointer-events:none' }, svg);
  }
  svgEl('rect', { x: L + 0.5, y: T + 0.5, width: pw - 1, height: ph - 1, style: 'fill:none;stroke:var(--frame);stroke-width:1', 'shape-rendering': 'crispEdges' }, svg);
  const lab = svgEl('g', { style: 'font:10px var(--mono);fill:var(--ink)' }, svg);
  const every = Math.max(1, Math.ceil(nr / Math.max(2, Math.floor(ph / 12))));
  for (let i = 0; i < nr; i += every) { const e = svgEl('text', { x: L - 4, y: T + (nr - 1 - i + 0.5) * ch + 3, 'text-anchor': 'end' }, lab); e.textContent = String(o.layers[i]); }
  // nice step values, each placed at the column holding it
  if (nc) {
    const c0 = o.cols[0], c1 = o.cols[nc - 1];
    const ticks = c1 > c0 ? Array.from(A.nice_ticks(c0, c1, 5)).filter((v) => v >= c0 && v <= c1) : [c0];
    for (const v of ticks) {
      let j = 0; while (j < nc - 1 && o.cols[j] < v) j++;
      const x = L + (j + 0.5) * cw;
      svgEl('line', { x1: x, x2: x, y1: T + ph, y2: T + ph + 3, style: 'stroke:var(--frame)' }, svg);
      const e = svgEl('text', { x, y: T + ph + 13, 'text-anchor': x < L + 20 ? 'start' : x > L + pw - 20 ? 'end' : 'middle' }, lab);
      e.textContent = fmtTick(v);
    }
  }
  { const e = svgEl('text', { x: L + pw, y: H - 3, 'text-anchor': 'end', style: 'fill:var(--muted)' }, lab); e.textContent = o.xlabel || 'step'; }
  { const e = svgEl('text', { x: 10, y: T + ph / 2, transform: `rotate(-90 10 ${T + ph / 2})`, 'text-anchor': 'middle', style: 'fill:var(--muted)' }, lab); e.textContent = 'layer'; }
  // colour key, labelled in data units (10^x on a log scale)
  const bx = W - R + 8, by = T + 10, bh = Math.min(120, ph - 20), N = 32;
  for (let k = 0; k < N; k++) { const v = sc.lo + (sc.hi - sc.lo) * (1 - k / (N - 1)); svgEl('rect', { x: bx, y: by + (k * bh) / N, width: 10, height: bh / N + 0.6, style: `fill:${sc(v)}` }, svg); }
  svgEl('rect', { x: bx + 0.5, y: by + 0.5, width: 9, height: bh - 1, style: 'fill:none;stroke:var(--frame);stroke-width:1' }, svg);
  const un = (v: number) => (o.log ? Math.pow(10, v) : v);
  for (const [v, y] of [[sc.hi, by + 4], [(sc.lo + sc.hi) / 2, by + bh / 2 + 3], [sc.lo, by + bh]] as [number, number][]) { const e = svgEl('text', { x: bx + 13, y }, lab); e.textContent = Number.isFinite(v) ? fmtTick(+un(v).toPrecision(3)) : ''; }
  if (o.log) { const e = svgEl('text', { x: bx + 5, y: by - 3, 'text-anchor': 'middle', style: 'fill:var(--muted)' }, lab); e.textContent = 'log'; }
  // one handler for every cell
  const cell = (ev: { clientX: number; clientY: number }): [number, number] | null => {
    const m = svg.getScreenCTM(); if (!m) return null;
    const p = new DOMPoint(ev.clientX, ev.clientY).matrixTransform(m.inverse());
    const j = Math.floor((p.x - L) / cw), r = Math.floor((p.y - T) / ch);
    if (j < 0 || j >= nc || r < 0 || r >= nr) return null;
    return [nr - 1 - r, j];
  };
  svg.addEventListener('pointermove', (ev) => {
    const c = cell(ev); if (!c) { hideTip(); return; }
    const [i, j] = c, v = o.z[i * nc + j];
    showTip(ev, `layer ${o.layers[i]}\nstep ${fmt(o.cols[j])}\n${o.zlabel} ${Number.isFinite(v) ? fmt(v) : 'no data'}${o.onpick ? '\nclick: this layer as a line' : ''}`);
  });
  svg.addEventListener('pointerleave', hideTip);
  if (o.onpick) {
    const pick = o.onpick;
    svg.style.cursor = 'pointer';
    svg.addEventListener('click', (ev) => { const c = cell(ev); if (c) pick(c[0], c[1]); });
  }
  const fin = Array.from(o.z as ArrayLike<number>).filter(Number.isFinite);
  const miss = nr * nc - fin.length;
  h('div', { class: 'sub' }, panel, `${nr} layers x ${nc} columns${miss ? `, ${miss} cells without data (hatched)` : ''}`);
  return panel;
}
