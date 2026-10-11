// Series resolution, query evaluation and forecast/ETA computations: pure functions, no DOM. The
// wasm module does the numeric work; loaders are injected so this runs the same in a worker.
import { A } from '@/lib/wasm';
import type { FitForecast, Forecast, QueryMeta, QueryResult, Run, Trace } from '@/logic/types';

export const STATS = ['mean', 'min', 'max', 'rms'];
const clean = (s: unknown) => String(s).replace(/[\t\n\r]/g, ' ');

/** the catalog the query engine matches against (see query.rs Catalog) */
export function catalogText(runs: Run[]): string {
  const out: string[] = [];
  for (const r of runs) {
    out.push('R\t' + clean(r.id));
    for (const k of ['name', 'source', 'project'] as const) if (r[k] !== undefined && r[k] !== null) out.push(`I\t${k}\t${clean(r[k])}`);
    for (const [k, v] of Object.entries(r.config || {})) if (v !== null && typeof v !== 'object') out.push(`C\t${clean(k)}\t${clean(v)}`);
    for (const m of Object.keys(r.metrics || {})) out.push('M\t' + clean(m));
    for (const [k, a] of Object.entries(r.arrays || {})) if ((a.n_steps || 0) >= 2) for (const s of STATS) out.push(`M\t${clean(k)}.${s}`);
  }
  return out.join('\n');
}
export interface Stream { run: string; metric: string; kind: 'metric' | 'array'; n: number; shape?: number[] }
/** browsable streams: metrics, plus the mean of arrays logged over two or more steps */
export function streamList(runs: Run[]): Stream[] {
  const out: Stream[] = [];
  for (const r of runs) {
    for (const m of Object.keys(r.metrics || {})) out.push({ run: r.id, metric: m, kind: 'metric', n: r.metrics[m].n });
    for (const [k, a] of Object.entries(r.arrays || {})) if ((a.n_steps || 0) >= 2) out.push({ run: r.id, metric: k + '.mean', kind: 'array', n: a.n_steps, shape: a.shape });
  }
  return out;
}
export function metricCounts(runs: Run[]): Map<string, number> {
  const m = new Map<string, number>();
  for (const r of runs) for (const k of Object.keys(r.metrics || {})) m.set(k, (m.get(k) || 0) + 1);
  return m;
}
/** runs whose configs agree on everything (seed replicates without a seed field, or with one) */
export function hasReplicates(runs: Run[]): boolean {
  const key = (r: Run) => JSON.stringify(Object.entries(r.config || {}).filter(([k]) => k !== 'seed').sort());
  const seen = new Set<string>();
  for (const r of runs) { const k = key(r); if (seen.has(k)) return true; seen.add(k); }
  return false;
}
export function fieldsOf(runs: Run[]): string[] {
  const s = new Set<string>(); for (const r of runs) for (const [k, v] of Object.entries(r.config || {})) if (v !== null && typeof v !== 'object') s.add(k);
  return [...s].sort();
}
/** config fields that vary across these runs (useful for by / split) */
export function varyingFields(runs: Run[]): string[] {
  return fieldsOf(runs).filter((k) => new Set(runs.map((r) => String((r.config || {})[k]))).size > 1);
}

export type Loader = (runId: string, metric: string) => Promise<{ steps: Float64Array; values: Float64Array } | null>;

/** plan a query against the catalog: [[run, metric]] streams it reads (cheap, synchronous) */
export function planOf(q: string, catalog: string): [string, string][] {
  try { return A.query_plan(q, catalog).split('\n').filter(Boolean).map((l) => l.split('\t') as [string, string]); } catch { return []; }
}

/** Evaluate a panel query: plan (wasm) -> load streams -> eval (wasm). */
export async function evalQuery(q: string, catalog: string, load: Loader, maxPoints = 1500): Promise<QueryResult> {
  const parsed = A.query_parse(q);
  if (!parsed.ok()) { const e: QueryResult = { ok: false, error: parsed.error(), start: parsed.err_start(), end: parsed.err_end() }; parsed.free(); return e; }
  parsed.free();
  const plan = planOf(q, catalog);
  const loaded = await Promise.all(plan.map(([r, m]) => load(r, m).catch(() => null)));
  const off = new Uint32Array(plan.length + 1); let n = 0;
  loaded.forEach((d, i) => { off[i] = n; n += d ? d.steps.length : 0; }); off[plan.length] = n;
  const st = new Float64Array(n), va = new Float64Array(n);
  loaded.forEach((d, i) => { if (d) { st.set(d.steps, off[i]); va.set(d.values, off[i]); } });
  const out = A.query_eval(q, catalog, off, st, va, maxPoints);
  const meta = JSON.parse(out.meta()) as QueryMeta; const data = out.data(); out.free();
  const missing = plan.filter((_, i) => !loaded[i]).map(([r, m]) => `${r}:${m}`);
  if (missing.length) meta.warnings.push(`could not load ${missing.slice(0, 3).join(', ')}${missing.length > 3 ? ` and ${missing.length - 3} more` : ''}`);
  const traces: Trace[] = meta.traces.map((t) => {
    const s = (k: number) => data.subarray(t.off + k * t.n, t.off + (k + 1) * t.n);
    return { ...t, xs: s(0), ys: s(1), raw: s(2), lo: s(3), hi: s(4), steps: s(5) };
  });
  return { ok: true, meta, traces, streams: plan.length };
}

// ------------------------------------------------------------ forecasts
const Z: Record<number, number> = { 50: 0.6745, 80: 1.2816, 95: 1.96 };
type Arr = ArrayLike<number>;
export interface ForecastOpts { method?: string; window?: [number, number] | null; band?: number; pool?: { xs: Arr; raw: Arr }[] }
/** Forecast one trace to `to` (in x units, step axis only). The ridge band is the 10-90% residual
 *  band from the backtest; other levels rescale its half-width by the normal quantile ratio. */
export function forecastTrace(tr: { xs: Arr; raw: Arr }, to: number, opts: ForecastOpts = {}): Forecast {
  const fit = windowed(tr.xs, tr.raw, opts.window);
  if (fit.xs.length < 4) return { none: 'fewer than 4 points in the fit window' };
  const last = fit.xs[fit.xs.length - 1];
  if (!(to > last)) return { none: 'horizon is not past the last point' };
  const method = opts.method || 'auto';
  const scale = (Z[opts.band || 80] || Z[80]) / Z[80];
  const res: Omit<FitForecast, 'lastX'>[] = [];
  if (method === 'auto' || method === 'ridge' || method === 'last') {
    const off = [0], ps: number[] = [], pv: number[] = [];
    for (const p of opts.pool || []) { const w = windowed(p.xs, p.raw, opts.window); for (let i = 0; i < w.xs.length; i++) { ps.push(w.xs[i]); pv.push(w.raw[i]); } off.push(ps.length); }
    let f: ReturnType<typeof A.forecast_n> | null = null;
    try { f = A.forecast_n(Float64Array.from(fit.xs), Float64Array.from(fit.raw), Uint32Array.from(off), Float64Array.from(ps), Float64Array.from(pv), to, ps.length > 0, 1024); } catch { f = null; }
    if (f) {
      res.push({ method: f.skill() && method !== 'last' ? 'ridge' : 'last value', skill: f.skill() && method !== 'last', mae: f.mae_test(), naive: f.mae_test_naive(), nTest: f.n_test(), xs: f.steps(), ys: f.mean(), lo: f.lo(), hi: f.hi(), cfg: `L=${f.lookback()} ${f.scaler()}${f.pooled() ? ` pooled(${f.n_pool()})` : ''}` });
      f.free();
    }
  }
  if (method === 'auto' || method === 'powerlaw') {
    const v = A.powerlaw_forecast(Float64Array.from(fit.xs), Float64Array.from(fit.raw), to, 24);
    if (v.length) {
      const n = v[7], o = 8;
      res.push({ method: 'power law', skill: v[0] > 0, mae: v[1], naive: v[2], nTest: v[3], xs: v.slice(o, o + n), ys: v.slice(o + n, o + 2 * n), lo: v.slice(o + 2 * n, o + 3 * n), hi: v.slice(o + 3 * n, o + 4 * n), cfg: `y = ${fmt3(v[4])} + ${fmt3(v[5])}·x^-${fmt3(v[6])}` });
    }
  }
  if (!res.length) return { none: 'not enough points for this method' };
  // auto: the method with skill and the lowest test MAE; else the first (its last-value fallback)
  let best = res[0];
  if (method === 'auto') { const sk = res.filter((r) => r.skill).sort((a, b) => a.mae - b.mae); best = sk[0] || res.find((r) => r.method === 'last value') || res[0]; }
  else if (method === 'powerlaw') best = res.find((r) => r.method === 'power law') || res[0];
  if (scale !== 1) {
    const ys = best.ys;
    best = { ...best, lo: Array.from(best.lo, (l, i) => ys[i] - (ys[i] - l) * scale), hi: Array.from(best.hi, (hv, i) => ys[i] + (hv - ys[i]) * scale) };
  }
  return { ...best, lastX: last };
}
function windowed(xs: Arr, ys: Arr, w?: [number, number] | null): { xs: number[]; raw: number[] } {
  const ox: number[] = [], oy: number[] = [];
  for (let i = 0; i < xs.length; i++) if (Number.isFinite(xs[i]) && Number.isFinite(ys[i]) && (!w || (xs[i] >= w[0] && xs[i] <= w[1]))) { ox.push(xs[i]); oy.push(ys[i]); }
  return { xs: ox, raw: oy };
}
const fmt3 = (v: number) => (Math.abs(v) >= 100 || Math.abs(v) < 0.01 ? v.toExponential(2) : v.toFixed(3));
/** badge text: "forecast · ridge · 1.4× vs last-value (11 test pts)" */
export function badgeText(f: Forecast | null | undefined): { text: string; skill: boolean } {
  if (!f || f.none !== undefined) return { text: 'forecast · ' + (f ? f.none : 'off'), skill: false };
  if (!f.skill) return { text: `forecast · no skill · last value (${f.nTest} test pt${f.nTest === 1 ? '' : 's'})`, skill: false };
  const ratio = f.mae > 0 ? f.naive / f.mae : Infinity;
  return { text: `forecast · ${f.method} · ${Number.isFinite(ratio) ? ratio.toFixed(1) + '×' : 'exact'} vs last-value (${f.nTest} test pt${f.nTest === 1 ? '' : 's'})`, skill: true };
}
export type Eta = { reached: number } | { mean: number; early: number; late: number; horizon: number } | null;
/** ETA for a target from the forecast (and the observed data first: already reached?) */
export function etaFor(tr: { xs: Arr; raw: Arr }, f: Forecast | undefined, target: number): Eta {
  const raw = Float64Array.from(tr.raw);
  const seen = A.forecast_eta(Float64Array.from(tr.xs), raw, raw, raw, target)[0];
  if (Number.isFinite(seen)) return { reached: seen };
  if (!f || f.none !== undefined) return null;
  const e = A.forecast_eta(Float64Array.from(f.xs), Float64Array.from(f.ys), Float64Array.from(f.lo), Float64Array.from(f.hi), target);
  return { mean: e[0], early: e[1], late: e[2], horizon: f.xs[f.xs.length - 1] };
}
export function etaText(e: Eta, target: number, fmt: (v: number) => string, xname = 'step'): string {
  if (!e) return `target ${fmt(target)}: no forecast`;
  if ('reached' in e) return `reached ${fmt(target)} at ${xname} ${fmt(Math.round(e.reached))}`;
  if (!Number.isFinite(e.mean)) return `not reached ${fmt(target)} by ${fmt(Math.round(e.horizon))}` + (Number.isFinite(e.early) ? ` (band reaches it from ~${fmt(Math.round(e.early))})` : '');
  const rng = `${Number.isFinite(e.early) ? fmt(Math.round(e.early)) : '?'}–${Number.isFinite(e.late) ? fmt(Math.round(e.late)) : `>${fmt(Math.round(e.horizon))}`}`;
  return `reaches ${fmt(target)} at ~${xname} ${fmt(Math.round(e.mean))} (${rng})`;
}
export function spikesOf(tr: { raw: Arr }): number[] {
  if (tr.raw.length < 15) return [];
  return Array.from(A.spikes(Float64Array.from(tr.raw), 10, 6));
}
/** age of a run's newest row in seconds (NaN if unknown) and whether it looks stale */
export function liveAge(run: Run | undefined, now = Date.now() / 1000): { age: number; stale: boolean; finished: boolean } {
  const st = run && run.meta && run.meta.start_unix, lt = run && run.last_t;
  if (!(st && st > 0) || !(lt !== null && lt !== undefined && lt >= 0)) return { age: NaN, stale: false, finished: !!(run && run.finished) };
  const age = now - (st + lt);
  return { age, stale: !run!.finished && age > 600, finished: !!run!.finished };
}
export type ByColours = { numeric: true; lo: number; hi: number; log: boolean; pos: number[] } | { numeric: false; cats: string[]; idx: number[] };
/** colours for "by <field>": numeric -> gradient position t in [0, 1] (log when spanning > 20x), else category index */
export function byColours(traces: { color: string }[], numeric?: boolean): ByColours {
  const vals = traces.map((t) => t.color);
  if (numeric) {
    const ns = vals.map(Number);
    const lo = Math.min(...ns), hi = Math.max(...ns);
    const lg = lo > 0 && hi / lo > 20;
    const f = (v: number) => (hi > lo ? (lg ? (Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo)) : (v - lo) / (hi - lo)) : 0.5);
    return { numeric: true, lo, hi, log: lg, pos: ns.map(f) };
  }
  const cats = [...new Set(vals)];
  return { numeric: false, cats, idx: vals.map((v) => cats.indexOf(v)) };
}
/** default forecast horizon: a run's planned total steps when larger, else +50% of the span */
export function defaultHorizon(traces: { xs: Arr; runs: string[] }[], byId: Map<string, Run>): number {
  let lo = Infinity, hi = -Infinity, planned = 0;
  for (const t of traces) {
    for (let i = 0; i < t.xs.length; i++) { const x = t.xs[i]; if (Number.isFinite(x)) { lo = Math.min(lo, x); hi = Math.max(hi, x); } }
    for (const r of t.runs) { const run = byId.get(r); if (run) for (const k of ['steps', 'total_steps', 'max_steps', 'train_steps', 'num_steps']) { const v = run.config[k]; if (typeof v === 'number' && v > hi) planned = Math.max(planned, v); } }
  }
  return planned || hi + 0.5 * (hi - lo);
}
