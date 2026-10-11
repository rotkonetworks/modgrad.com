// Drawing one board panel from its query result: cells (split), colours (by), forecasts with the
// accountability trace, follow, legend decoration (solo/hide, right axis, remove, live dot, drag
// handle) and the plot overlay (events, spikes, forecast horizon tab, fit-window bracket, target).
// Imperative on purpose: it wraps chart2d, whose SVG must stay pixel-identical. The Solid Panel
// component decides when to call it (only when this panel's query, view or data changed).
import { chart2d, zoomGroup, type OverlayApi, type Series as ChartSeries } from '@/charts/chart2d';
import { h, btn, svgEl } from '@/lib/dom';
import { fmt, fmtTick, fmtTime, fmtStep } from '@/lib/fmt';
import { PALETTE, cssv, hexToRgb, mixRgb } from '@/lib/color';
import * as Dt from '@/logic/data';
import { fitInWorker } from '@/logic/forecasts';
import type { FitForecast, Forecast, Panel, QueryResult, RunEvent, Trace } from '@/logic/types';
import { S } from '@/stores/runs';
import {
  fcCache, fcHist, fresh, hiddenTrace, settings, commit, setStage, placeTarget, soloTrace, toggleY2, removeTrace, pauseFollow,
  findPanel, type TraceRef,
} from '@/stores/board';
import { draggable } from '@/board/dnd';
import { layerHeatmap } from '@/charts/layerHeatmap';
import { getStage, quote } from '@/logic/board';
import { addToNew } from '@/stores/board';

export const MAX_POINTS = 1500;
/** chart width for a box: its CSS width rounded down to 60 px, so the SVG scales by at most ~1.2
 *  (text stays 10-12 px from a 400 px phone to a full-screen focus view) and a resize redraws
 *  only when the step changes */
export const chartWidth = (px: number): number => (px > 0 ? Math.max(300, Math.floor(px / 60) * 60) : 540);
export interface DrawOut { badge: { text: string; skill: boolean; title: string } | null }
export interface DrawCtx { panel: Panel; host: HTMLElement; foot: HTMLElement; res: QueryResult; events: Map<string, RunEvent[]>; inFocus: boolean; focusOpen: boolean; dataKey: string; title: string; redraw?: () => void }

/** stable across data updates (labels carry counts like n=3) */
const traceKey = (t: Trace) => `${t.letter}|${t.metric}|${t.runs[0]}`;
/** a cheap fingerprint of the points a forecast reads (length, ends, two weighted sums) */
function dataSig(t: { xs: ArrayLike<number>; raw: ArrayLike<number> }): string {
  let s1 = 0, s2 = 0; const n = t.raw.length;
  for (let i = 0; i < n; i++) { const v = t.raw[i]; if (Number.isFinite(v)) { s1 += v; s2 += v * ((i % 97) + 1); } }
  return `${n}:${t.xs[0]}:${t.xs[t.xs.length - 1]}:${s1}:${s2}`;
}
const lastX = (t: { xs: ArrayLike<number> }): number => { for (let i = t.xs.length - 1; i >= 0; i--) if (Number.isFinite(t.xs[i])) return t.xs[i]; return -Infinity; };
const FC_LIVE_MS = 2000;
const fcLast = new Map<string, { f: Forecast; at: number; n: number }>();
const fcTimers = new Map<string, ReturnType<typeof setTimeout>>();
/** one redraw per panel per frame when several worker fits land together */
const fcRedraws = new Set<string>();
function redrawSoon(id: string, redraw: () => void) {
  if (fcRedraws.has(id)) return;
  fcRedraws.add(id);
  requestAnimationFrame(() => { fcRedraws.delete(id); redraw(); });
}
const niceRound = (v: number) => { const e = Math.pow(10, Math.floor(Math.log10(v)) - 1); return Math.round(v / e) * e; };

export function drawPanel(c: DrawCtx): DrawOut {
  const { panel: p, host, foot, res } = c;
  host.textContent = ''; foot.textContent = '';
  const out: DrawOut = { badge: null };
  if (!res.ok) { h('div', { class: 'bempty' }, host, 'query error (see above)'); return out; }
  const meta = res.meta, traces = res.traces;
  if (res.heatmaps.length) return drawHeatmaps(c, res);
  if (!traces.length) {
    // with warnings the reason is known (e.g. no run logs the x-axis metric): say it in the panel, not under it
    if (!meta.warnings.length) { h('div', { class: 'bempty' }, host, 'nothing matches this query yet; it updates as runs arrive'); return out; }
    const box = h('div', { class: 'bempty' }, host, `nothing to draw: ${meta.warnings.join('; ')}`);
    const xs = getStage(p.q, 'x');
    if (xs) { h('br', {}, box); btn(box, 'plot against step instead', `plot against step instead (removes "${xs}")`, () => setStage(findPanel(p.id)!, 'x', null, 'x axis: step')); }
    return out;
  }
  const isStep = meta.x_label === 'step';
  const fc = meta.forecast && isStep ? meta.forecast : null;
  const el = host.closest('.bpanel') as HTMLElement | null;
  // colours
  let colour = (i: number) => PALETTE[i % PALETTE.length];
  let key: Dt.ByColours | null = null;
  if (meta.by) {
    const bc = Dt.byColours(traces, meta.by_numeric);
    const lo = hexToRgb(cssv('--c2')), hi = hexToRgb(cssv('--c3'));
    traces.forEach((_, i) => {
      const col = bc.numeric ? mixRgb(lo, hi, bc.pos[i]) : `var(${PALETTE[bc.idx[i] % PALETTE.length]})`;
      host.style.setProperty('--bk' + i, col);
    });
    colour = (i) => '--bk' + i;
    key = bc;
  }
  // forecasts (step axis), with the previous fit kept as a faint trace for accountability
  const fcs = new Map<string, Forecast>();
  if (fc) {
    const to = fc.to || Dt.defaultHorizon(traces, S.byId);
    for (const t of traces) {
      if (hiddenTrace(p, t)) continue;
      const pool = p.view.fc.pool === 'none' ? [] : traces.filter((o) => o !== t && (p.view.fc.pool === 'group' ? o.letter === t.letter : o.metric === t.metric));
      let f: Forecast | undefined;
      const fr = p.view.fc.frozen && p.view.fc.frozen[traceKey(t)];
      if (!p.view.fc.live && fr) f = fr;
      else {
        // keyed on what the fit reads (this trace's and the pooled traces' data), not on the query
        // text: a display stage (log y, ema, colour, another series) must not refit (~10-35 ms each)
        // (the default horizon moves with the data, so only a horizon the viewer set is identity)
        const id = JSON.stringify([p.id, traceKey(t), fc.to || null, fc.method, p.view.fc.window, p.view.fc.band, p.view.fc.pool]);
        const ck = JSON.stringify([id, to, dataSig(t), pool.map(dataSig)]);
        f = fcCache.get(ck);
        if (!f) {
          // while rows are being appended (a live run, or a live run in the pool) a trace refits at
          // most every FC_LIVE_MS; in between the last fit is drawn and one redraw is queued, so
          // the final rows always get a fit
          const prev = fcLast.get(id), now = Date.now();
          // rows were appended: the newest x moved on (the length is pinned once block-averaged to
          // MAX_POINTS, and a value-only change such as norm z must refit at once)
          const n = Math.max(lastX(t), ...pool.map(lastX));
          if (prev && c.redraw && n > prev.n && now - prev.at < FC_LIVE_MS) {
            f = prev.f;
            if (!fcTimers.has(p.id)) fcTimers.set(p.id, setTimeout(() => { fcTimers.delete(p.id); c.redraw!(); }, FC_LIVE_MS - (now - prev.at) + 20));
          } else {
            const opts = { method: fc.method, window: p.view.fc.window, band: p.view.fc.band, pool };
            const keep = (r: Forecast) => { if (fcCache.size > 400) fcCache.clear(); fcCache.set(ck, r); fcLast.set(id, { f: r, at: Date.now(), n }); };
            if (c.redraw) {
              // fit in the worker; meanwhile draw this trace's last fit (or a "fitting" badge)
              const redraw = c.redraw;
              fitInWorker(ck, t, to, opts, (r) => { keep(r); redrawSoon(p.id, redraw); });
              f = prev ? prev.f : { none: 'fitting…' };
            } else {
              f = Dt.forecastTrace(t, to, opts);
              keep(f);
            }
          }
        }
      }
      fcs.set(t.label, f);
      const hk = p.id + '\u0001' + traceKey(t), hist = fcHist.get(hk);
      if (f && f.none === undefined) {
        if (!hist) fcHist.set(hk, { cur: f, prev: null });
        else if (hist.cur.lastX !== f.lastX) fcHist.set(hk, { cur: f, prev: hist.cur });
        else hist.cur = f;
      }
    }
  }
  // cells (split -> small multiples with locked axes)
  const cells = meta.cells.length ? meta.cells : [''];
  const split = cells.length > 1 || cells[0] !== '';
  if (el) el.classList.toggle('wide', split && !c.focusOpen);
  let yrange: [number, number] | null = null;
  const lockY = split && new Set(traces.map((t) => t.metric)).size === 1; // same quantity in every cell
  if (lockY) {
    let lo = Infinity, hi = -Infinity;
    for (const t of traces) if (!hiddenTrace(p, t)) for (let i = 0; i < t.ys.length; i++) {
      const v = t.ys[i]; if (Number.isFinite(v) && (!meta.ylog || v > 0)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
      if (t.band) { if (Number.isFinite(t.lo[i]) && (!meta.ylog || t.lo[i] > 0)) lo = Math.min(lo, t.lo[i]); if (Number.isFinite(t.hi[i])) hi = Math.max(hi, t.hi[i]); }
    }
    if (Number.isFinite(lo)) yrange = [lo, hi];
  }
  const wrap = split ? h('div', { class: 'bcells' }, host) : host;
  const group = split ? 'bsplit:' + p.id : settings.linkX ? 'b:' + meta.x_label : `b:${meta.x_label}:${p.id}`;
  // follow: keep the zoom width, slide the window to the newest x
  if (p.view.follow) {
    const g = zoomGroup(group);
    let mx = -Infinity; for (const t of traces) for (const x of t.xs) if (Number.isFinite(x)) mx = Math.max(mx, x);
    if (g.view && Number.isFinite(mx)) { const span = g.view.x1 - g.view.x0; g.view = { x0: mx - span, x1: mx }; }
  }
  const nTr = traces.filter((t) => !hiddenTrace(p, t)).length;
  const metricsIn = (ts: Trace[]) => new Set(ts.map((t) => t.metric)).size;
  const shortName = (t: Trace, ts: Trace[]) => (metricsIn(ts) === 1 && !t.metric.includes(' = ') ? t.label.replace(':' + t.metric, '') : t.label);
  const anyBand = traces.some((t) => t.band);
  const now = Date.now();
  for (const cell of cells) {
    const ct = traces.map((t, i) => ({ t, i })).filter(({ t }) => (t.cell || '') === cell);
    const cbox = split ? h('div', { class: 'bcell' }, wrap) : wrap;
    if (split) h('div', { class: 'ctitle' }, cbox, cell || '(none)');
    const series = ct.map(({ t, i }) => {
      const f = fcs.get(t.label);
      return {
        name: shortName(t, ct.map((x) => x.t)) + (meta.by ? `  [${meta.by}=${t.color}]` : ''), trace: t, color: colour(i),
        // a wash under the line only when it is the one visible series: under an overlay it hides the others
        area: nTr === 1 && !hiddenTrace(p, t) && !meta.by && !anyBand && !split, xs: t.xs, ys: t.ys, raw: t.raw,
        band: t.band ? { lo: t.lo, hi: t.hi } : null, hidden: hiddenTrace(p, t), y2: p.view.y2 === t.label,
        ghost: f && f.none === undefined ? { xs: f.xs, ys: f.ys, lo: f.lo, hi: f.hi } : null,
        fresh: t.runs.some((r) => now - (fresh.get(p.id + '\u0001' + r) || 0) < 20000),
      };
    });
    if (!series.length) continue;
    const W = chartWidth(cbox.clientWidth);
    // the focus view fills the window's height (one chart); the board keeps the 540:210 shape
    const H = c.inFocus && !split ? Math.max(210, Math.min(Math.round(W * 0.6), innerHeight - 300)) : undefined;
    chart2d(cbox, {
      title: c.title + (cell ? ' / ' + cell : ''), series, ylog: meta.ylog, yrange: lockY ? yrange : null, width: W, height: H,
      xlabel: meta.x_label, xfmt: meta.x_label === 'time' ? fmtTime : fmtTick, group,
      legendHead: series.length > 6 ? `${series.length} series` : '',
      decorateLegend: (tbl, ss) => decorateLegend(p.id, tbl, ss, cbox),
      onview: () => { const mp = findPanel(p.id); if (mp && mp.view.follow) pauseFollow(mp); },
      overlay: (api) => drawOverlay(p, api, series as unknown as ChartSeries[], fcs, isStep, fc, c.events),
    });
  }
  if (key) {
    const k = h('div', { class: 'bkey' }, foot);
    k.append(`colour: ${meta.by} `);
    if (key.numeric) { k.append(fmt(key.lo)); const g = h('span', { class: 'grad' }, k); g.style.background = `linear-gradient(to right, ${cssv('--c2')}, ${cssv('--c3')})`; k.append(fmt(key.hi) + (key.log ? ' (log scale)' : '')); }
    else key.cats.forEach((cat, i) => { const s = h('span', { class: 'sw-line' }, k); s.style.borderTopColor = `var(${PALETTE[i % PALETTE.length]})`; k.append(cat + ' '); });
  }
  // forecast badge + ETA
  if (fc) {
    const first = traces.find((t) => !hiddenTrace(p, t) && fcs.get(t.label));
    const f = first && fcs.get(first.label);
    const bt = Dt.badgeText(f);
    out.badge = {
      text: bt.text + (fcs.size > 1 ? ` +${fcs.size - 1}` : ''), skill: bt.skill,
      title: [...fcs.entries()].map(([l, g]) => `${l}: ${Dt.badgeText(g).text}${g && g.none === undefined ? '  ' + g.cfg : ''}`).join('\n') + '\nclick for forecast settings',
    };
    if (p.view.fc.target !== null && p.view.fc.target !== undefined) {
      for (const t of traces.filter((t) => !hiddenTrace(p, t)).slice(0, 4)) {
        const e = Dt.etaFor(t, fcs.get(t.label), p.view.fc.target);
        h('div', { class: 'beta' }, foot, `${t.label}: ${Dt.etaText(e, p.view.fc.target, fmt)}`);
      }
    }
    if (p.view.fc.window) h('div', { class: 'bnote' }, foot, `forecast fitted on step ${Math.abs(p.view.fc.window[0]) < 1e300 ? fmtStep(p.view.fc.window[0]) : 'start'} to ${Math.abs(p.view.fc.window[1]) < 1e300 ? fmtStep(p.view.fc.window[1]) : 'end'}`);
  } else if (meta.forecast && !isStep) h('div', { class: 'bnote' }, foot, 'forecasts are drawn on the step axis only');
  if (meta.downsampled) h('div', { class: 'bnote' }, foot, `long series block-averaged to ${MAX_POINTS} points (longest ${meta.downsampled}); legend statistics are over block means`);
  for (const w of meta.warnings) h('div', { class: 'bnote' }, foot, w);
  return out;
}

/** picked heatmap cell per panel and (run, family): survives redraws (live rows) */
const heatPick = new Map<string, [number, number]>();
/** `heatmap over layer`: one layer x step matrix per (run, family); a click draws that layer's line */
function drawHeatmaps(c: DrawCtx, res: Extract<QueryResult, { ok: true }>): DrawOut {
  const { panel: p, host, foot } = c;
  const el = host.closest('.bpanel') as HTMLElement | null;
  if (el) el.classList.toggle('wide', res.heatmaps.length > 1 && !c.focusOpen);
  const rownorm = /rownorm/.test(getStage(p.q, 'heatmap') || '');
  const wrap = res.heatmaps.length > 1 ? h('div', { class: 'bcells' }, host) : host;
  for (const hm of res.heatmaps) {
    const key = p.id + '\u0001' + hm.run + '\u0001' + hm.family;
    const box = h('div', { class: 'bcell' }, wrap);
    const stat = hm.family.slice(hm.family.lastIndexOf('*') + 1).replace(/^[/._-]+/, '') || hm.family;
    const draw = (): void => {
      box.textContent = '';
      const mark = heatPick.get(key) || null;
      layerHeatmap(box, {
        title: `${hm.run} / ${hm.family}`, layers: hm.layers, cols: hm.cols, z: hm.z, log: res.meta.ylog, mark, centered: rownorm,
        zlabel: rownorm ? `${stat} / layer median` : stat,
        onpick: (i, j) => { heatPick.set(key, [i, j]); draw(); },
      });
      if (!mark) return;
      const [i, j] = mark;
      if (i >= hm.metrics.length || j >= hm.cols.length) { heatPick.delete(key); return; }
      const metric = hm.metrics[i];
      const bar = h('div', { class: 'bkey' }, box);
      h('span', {}, bar, `layer ${hm.layers[i]}, step ${fmt(hm.cols[j])}: ${metric}`);
      btn(bar, 'open as panel', `new panel with ${metric} of ${hm.run}`, () => addToNew(`${metric}{run=${quote(hm.run)}}`, `${hm.run}:${metric}`));
      btn(bar, 'close', 'hide this layer line', () => { heatPick.delete(key); draw(); });
      const tr = res.traces.find((t) => t.metric === metric && t.runs.length === 1 && t.runs[0] === hm.run);
      if (tr) chart2d(box, { title: `${hm.run}: ${metric}`, xlabel: 'step', ylog: res.meta.ylog, marker: [hm.cols[j]], series: [{ name: metric, color: '--c1', area: true, xs: tr.steps, ys: tr.raw, raw: tr.raw }] });
    };
    draw();
  }
  if (rownorm) h('div', { class: 'bnote' }, foot, 'rownorm: each layer divided by its own median, so 1 is typical for that layer and a spike in one layer stands out');
  h('div', { class: 'bnote' }, foot, 'click a cell for that layer as a line; columns are the union of logged steps, block-reduced past 400');
  for (const w of res.meta.warnings) h('div', { class: 'bnote' }, foot, w);
  return { badge: null };
}

type DSeries = ChartSeries & { trace: Trace; fresh: boolean };
function decorateLegend(pid: string, tbl: HTMLTableElement | null, ss0: ChartSeries[], cbox: Element): void {
  if (!tbl) return;
  const p = findPanel(pid); if (!p) return;
  const ss = ss0 as DSeries[];
  const compact = ss.length > 8;
  tbl.classList.toggle('compact', compact);
  const hr = tbl.rows[0];
  [...hr.cells].forEach((c, i) => { if (i >= 2 && i <= 4) c.classList.add('st'); });
  hr.insertBefore(h('th', {}), hr.cells[1]);
  h('th', {}, hr, '');
  const svgs = cbox.querySelectorAll('svg');
  const svg = svgs[svgs.length - 1];
  const dim = (si: number | null) => { if (!svg) return; svg.querySelectorAll<SVGElement>('[data-si]').forEach((e) => { e.style.opacity = si === null || +(e.dataset.si || -1) === si ? '' : '0.15'; }); };
  ss.forEach((s, i) => {
    const tr = tbl.rows[i + 1]; if (!tr) return;
    const t = s.trace;
    [...tr.cells].forEach((c, j) => { if (j >= 2 && j <= 4) c.classList.add('st'); });
    if (s.hidden) tr.classList.add('off');
    if (s.fresh) tr.classList.add('fresh');
    const sw = tr.cells[0]; sw.title = 'drag to another panel, onto the canvas, or out of the board to remove';
    const nm = tr.cells[1]; nm.textContent = '';
    // live dot: age of the newest row of the run(s)
    const ages = t.runs.map((r) => Dt.liveAge(S.byId.get(r)));
    const newest = ages.filter((a) => Number.isFinite(a.age)).sort((a, b) => a.age - b.age)[0];
    const nb = btn(nm, '', `${t.label}: click to solo, shift-click to hide or show`, (ev) => soloTrace(findPanel(pid)!, t.label, ev.shiftKey));
    nb.setAttribute('aria-pressed', String(p.view.solo === t.label));
    if (newest && !newest.finished && newest.age < 6 * 3600) {
      const d = h('span', { class: 'ldot' + (newest.stale ? ' stale' : ''), title: `last row ${fmtTime(newest.age)} ago${newest.stale ? ' (stale)' : ''}` }, nb);
      void d;
    }
    nb.append(s.name);
    if (s.fresh) h('span', { class: 'tagnew', title: 'run appeared since the last update' }, nb, 'new');
    tr.insertBefore(h('td', { class: 'al' }, null, t.letter), nm);
    const ctl = h('td', { class: 'ctl' }, tr);
    const r = btn(ctl, 'R', s.y2 ? 'back to the left axis' : 'plot on the right-hand axis (one series per panel)', () => toggleY2(findPanel(pid)!, t.label), s.y2 ? 'on' : '');
    r.setAttribute('aria-pressed', String(!!s.y2));
    const single = t.runs.length === 1 && !t.metric.includes(' = ');
    const ref: TraceRef = { letter: t.letter, run: t.runs[0], single, label: t.label, entries: single ? `${t.metric}{run="${t.runs[0].replace(/"/g, '')}"}` : t.metric.includes(' = ') ? null : `${t.metric}{run=~"${t.runs.map((x) => x.replace(/"/g, '')).join('|')}"}` };
    btn(ctl, 'x', `remove ${t.label} from this panel`, () => removeTrace(findPanel(pid)!, ref));
    tr.addEventListener('pointerenter', () => dim(s.si));
    tr.addEventListener('pointerleave', () => dim(null));
    draggable(sw, () => ({ type: 'trace', pid, ref, label: t.label }));
  });
}

/** overlay: event and spike markers, previous forecasts, and the forecast handles */
function drawOverlay(p: Panel, api: OverlayApi, series: ChartSeries[], fcs: Map<string, Forecast>, isStep: boolean, fc: { to?: number | null; method?: string } | null, events: Map<string, RunEvent[]>): void {
  const { g, X, Y, L, T, pw, ph, x0, x1, ty } = api;
  const el = <K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number>, parent: Element = g) => svgEl(tag, attrs, parent);
  const clipX = (px: number) => Math.min(L + pw, Math.max(L, px));
  const inPlot = (px: number) => px >= L && px <= L + pw;
  const dataX = (px: number) => x0 + ((px - L) / pw) * (x1 - x0);
  const dataY = (py: number) => { const v = api.y0 + ((T + ph - py) / ph) * (api.y1 - api.y0); return ty(10) === 1 ? Math.pow(10, v) : v; };
  const vis = (series as DSeries[]).filter((s) => !s.hidden);
  if (isStep && settings.events) {
    const evs = new Map<number, string[]>();
    for (const s of vis) for (const r of s.trace.runs) for (const e of events.get(r) || []) { const k = e.step; if (!evs.has(k)) evs.set(k, []); evs.get(k)!.push(`${r}: ${e.text}`); }
    for (const [step, texts] of evs) {
      const px = X(step); if (!inPlot(px)) continue;
      el('line', { x1: px, x2: px, y1: T, y2: T + ph, style: 'stroke:var(--c4);stroke-width:1;stroke-dasharray:1 3;pointer-events:none' });
      const m = el('path', { d: `M${px - 3},${T}L${px + 3},${T}L${px},${T + 5}Z`, style: 'fill:var(--c4);cursor:help' });
      el('title', {}, m).textContent = `step ${step}\n` + [...new Set(texts)].slice(0, 6).join('\n');
    }
  }
  if (settings.spikes) {
    for (const s of vis) {
      for (const i of Dt.spikesOf(s.trace)) {
        const px = X(s.trace.xs[i]), v = ty(s.trace.raw[i]); if (!inPlot(px) || !Number.isFinite(v)) continue;
        const py = Math.max(T + 6, Y(v) - 6);
        const m = el('path', { d: `M${px - 3.5},${py - 6}L${px + 3.5},${py - 6}L${px},${py}Z`, style: 'fill:var(--accent);stroke:var(--bg);stroke-width:0.5;cursor:help' });
        el('title', {}, m).textContent = `loss spike: ${s.trace.label}\nstep ${fmt(s.trace.steps[i])}, value ${fmt(s.trace.raw[i])} (above the median of the previous 10 points by > 6 robust sd)`;
      }
    }
  }
  if (!fc) return;
  // previous forecasts: faint dotted traces, so forecast vs reality stays visible
  for (const s of vis) {
    const hist = fcHist.get(p.id + '\u0001' + traceKey(s.trace));
    if (!hist || !hist.prev) continue;
    const f = hist.prev; const pts: string[] = [];
    for (let i = 0; i < f.xs.length; i++) { const v = ty(f.ys[i]); if (Number.isFinite(v)) pts.push(`${X(f.xs[i]).toFixed(1)},${Y(v).toFixed(1)}`); }
    if (pts.length > 1) el('polyline', { points: pts.join(' '), style: `fill:none;stroke:var(${s.color});stroke-width:1;stroke-opacity:0.35;stroke-dasharray:1 2;pointer-events:none` });
  }
  const fits = [...fcs.values()].filter((f): f is FitForecast => !!f && f.none === undefined);
  // horizon handle at the end of the forecast: drag right past the data to extend it
  const fx = fits[0];
  const lastX = fx ? fx.lastX : Math.max(...vis.flatMap((s) => Array.from(s.trace.xs).filter(Number.isFinite)));
  const hz = fc.to || Math.max(lastX, ...fits.map((f) => f.xs[f.xs.length - 1]));
  const hg = el('g', { style: 'cursor:ew-resize' });
  const hx = clipX(X(hz));
  const hl = el('line', { x1: hx, x2: hx, y1: T, y2: T + ph, style: 'stroke:var(--accent);stroke-width:1;stroke-dasharray:4 2' }, hg);
  const tab = el('rect', { x: hx - 5, y: T + ph - 18, width: 10, height: 18, style: 'fill:var(--accent);stroke:var(--bg)' }, hg);
  const hlab = el('text', { x: hx - 7, y: T + ph - 5, 'text-anchor': 'end', style: 'font:10px var(--mono);fill:var(--accent)' }, hg);
  hlab.textContent = 'to ' + fmtStep(hz);
  el('title', {}, tab).textContent = 'forecast horizon: drag right past the plot to extend, left to shorten';
  handle(tab, api, (pt) => { const v = Math.max(lastX * 1.001, dataX(pt.x)); const px = clipX(X(v)); hl.setAttribute('x1', String(px)); hl.setAttribute('x2', String(px)); tab.setAttribute('x', String(px - 5)); hlab.setAttribute('x', String(px - 7)); hlab.textContent = 'to ' + fmtStep(v); return v; },
    (v) => setStage(findPanel(p.id)!, 'forecast', `forecast to ${fmtStep(niceRound(v))}${fc.method && fc.method !== 'auto' ? ' ' + fc.method : ''}`, 'forecast horizon'));
  // fit-window bracket under the plot
  const xsAll = vis.flatMap((s) => Array.from(s.trace.xs).filter(Number.isFinite));
  const dmin = Math.min(...xsAll), dmax = Math.max(...xsAll);
  let [wa, wb] = p.view.fc.window || [dmin, dmax];
  wa = Math.max(wa, dmin); wb = Math.min(wb, dmax);
  const by = T + ph + 4;
  const bl = el('line', { x1: clipX(X(wa)), x2: clipX(X(wb)), y1: by, y2: by, style: 'stroke:var(--accent);stroke-width:2' });
  const mk = (v: number, which: 'start' | 'end') => {
    const e = el('rect', { x: clipX(X(v)) - 3, y: by - 4, width: 6, height: 8, style: 'fill:var(--bg);stroke:var(--accent);stroke-width:1.5;cursor:ew-resize' });
    el('title', {}, e).textContent = `forecast fit window ${which}: drag to exclude ${which === 'start' ? 'warmup' : 'the tail'} from the fit`;
    handle(e, api, (pt) => { let v2 = Math.min(dmax, Math.max(dmin, dataX(pt.x))); if (which === 'start') { v2 = Math.min(v2, wb); wa = v2; } else { v2 = Math.max(v2, wa); wb = v2; } e.setAttribute('x', String(clipX(X(v2)) - 3)); bl.setAttribute('x1', String(clipX(X(wa)))); bl.setAttribute('x2', String(clipX(X(wb)))); return v2; },
      () => commit('forecast fit window', () => { findPanel(p.id)!.view.fc.window = wa <= dmin && wb >= dmax ? null : [wa, wb]; }));
  };
  mk(wa, 'start'); mk(wb, 'end');
  // target line with ETA, or a tab to place one
  const tgt = p.view.fc.target;
  if (tgt !== null && tgt !== undefined && Number.isFinite(ty(tgt))) {
    const py = Math.min(T + ph, Math.max(T, Y(ty(tgt))));
    const tg = el('g', { style: 'cursor:ns-resize' });
    const tl = el('line', { x1: L, x2: L + pw, y1: py, y2: py, style: 'stroke:var(--accent);stroke-width:1;stroke-dasharray:6 3' }, tg);
    const trr = el('rect', { x: L + 2, y: py - 7, width: 58, height: 14, style: 'fill:var(--bg);stroke:var(--accent)' }, tg);
    const tt = el('text', { x: L + 6, y: py + 3, style: 'font:10px var(--mono);fill:var(--accent)' }, tg);
    tt.textContent = 'target ' + fmt(tgt);
    el('title', {}, trr).textContent = 'target: drag up or down; the ETA readout is under the plot';
    handle(trr, api, (pt) => { const yy = Math.min(T + ph, Math.max(T, pt.y)); const v = dataY(yy); tl.setAttribute('y1', String(yy)); tl.setAttribute('y2', String(yy)); trr.setAttribute('y', String(yy - 7)); tt.setAttribute('y', String(yy + 3)); tt.textContent = 'target ' + fmt(v); return v; },
      (v) => commit('forecast target', () => { findPanel(p.id)!.view.fc.target = +v.toPrecision(4); }));
  } else {
    const trr = el('rect', { x: L + 2, y: T + 2, width: 50, height: 13, style: 'fill:var(--bg);stroke:var(--accent);cursor:pointer' });
    const tt = el('text', { x: L + 6, y: T + 12, style: 'font:10px var(--mono);fill:var(--accent);pointer-events:none' }); tt.textContent = '+ target';
    el('title', {}, trr).textContent = 'place a target line for an ETA readout (also in the forecast settings)';
    trr.addEventListener('click', () => placeTarget(findPanel(p.id)!));
  }
}
/** drag an SVG handle with pointer capture; onMove(pt) returns the value, onEnd(value) commits */
function handle(e: SVGElement, api: OverlayApi, onMove: (pt: DOMPoint) => number, onEnd: (v: number) => void): void {
  e.addEventListener('pointerdown', (ev) => {
    if (ev.button !== 0) return;
    ev.stopPropagation(); ev.preventDefault();
    try { e.setPointerCapture(ev.pointerId); } catch { /* synthetic events */ }
    let val: number | null = null, moved = false;
    const mv = (m: PointerEvent) => { if (m.pointerId !== ev.pointerId) return; moved = true; val = onMove(api.local(m)); };
    const up = (m: PointerEvent) => { if (m.pointerId !== ev.pointerId) return; document.removeEventListener('pointermove', mv); document.removeEventListener('pointerup', up); if (moved && val !== null) onEnd(val); };
    document.addEventListener('pointermove', mv); document.addEventListener('pointerup', up);
  });
}
