// Spike and non-finite overview for pre-training post-mortems: pure, no DOM. Detection is the wasm
// kernel `anomalies` (curve.rs: > k robust sd above the trailing median, plus non-finite values);
// this groups nearby hits into one entry and folds per-layer metrics into their family.
import { A } from '@/lib/wasm';

/** metrics worth scanning: losses, gradient norms (incl. per-layer), RL reward / kl */
export const SCAN = /(^|[/_])(loss|nll)($|[/_])|loss$|grad_?norm|(^|[/_])(reward|kl)$/i;

export interface Hit { step: number; kind: 'spike' | 'nonfinite'; ratio: number; value: number }
/** hits in one series (values: NaN where a non-finite value was logged) */
export function scanSeries(steps: ArrayLike<number>, values: ArrayLike<number>, window = 10, k = 6): Hit[] {
  if (values.length < window + 5) {
    const out: Hit[] = [];
    for (let i = 0; i < values.length; i++) if (!Number.isFinite(values[i])) out.push({ step: steps[i], kind: 'nonfinite', ratio: NaN, value: values[i] });
    return out;
  }
  const r = A.anomalies(Float64Array.from(values), window, k);
  const out: Hit[] = [];
  for (let j = 0; j < r.length; j += 3) {
    const i = r[j];
    out.push({ step: steps[i], kind: r[j + 1] === 0 ? 'spike' : 'nonfinite', ratio: r[j + 2], value: values[i] });
  }
  return out;
}

export interface Anomaly {
  run: string; kind: 'spike' | 'nonfinite';
  /** the metric, or the family template for per-layer metrics (layers/x/grad_norm) */
  what: string; layer: boolean;
  from: number; to: number; at: number; ratio: number; value: number;
  /** per-layer: layers involved (and how many the family has) */
  layers: number[]; nLayers: number; count: number;
}

/** typical step spacing of a series (median difference) */
function spacing(steps: ArrayLike<number>): number {
  const d: number[] = [];
  for (let i = 1; i < steps.length; i++) d.push(steps[i] - steps[i - 1]);
  d.sort((a, b) => a - b);
  return d.length ? d[d.length >> 1] || 1 : 1;
}

export interface ScanIn { run: string; metric: string; steps: ArrayLike<number>; values: ArrayLike<number>; family?: { family: string; layer: number; n: number } }
/** Scan every input and merge: hits of one (run, metric-or-family, kind) closer than a few steps of
 *  that series' spacing are one entry; at = the step of the largest ratio (first non-finite). */
export function overview(inputs: ScanIn[]): Anomaly[] {
  interface Raw { run: string; what: string; layer: boolean; kind: Hit['kind']; step: number; ratio: number; value: number; li: number | null; nLayers: number; gap: number }
  const raw: Raw[] = [];
  for (const s of inputs) {
    const gap = 3 * spacing(s.steps);
    for (const h of scanSeries(s.steps, s.values)) raw.push({ run: s.run, what: s.family ? s.family.family : s.metric, layer: !!s.family, kind: h.kind, step: h.step, ratio: h.ratio, value: h.value, li: s.family ? s.family.layer : null, nLayers: s.family ? s.family.n : 0, gap });
  }
  raw.sort((a, b) => a.run.localeCompare(b.run) || a.what.localeCompare(b.what) || a.kind.localeCompare(b.kind) || a.step - b.step);
  const out: Anomaly[] = [];
  for (const r of raw) {
    const last = out[out.length - 1];
    if (last && last.run === r.run && last.what === r.what && last.kind === r.kind && r.step - last.to <= r.gap) {
      last.to = Math.max(last.to, r.step); last.count++;
      if (r.li !== null && !last.layers.includes(r.li)) last.layers.push(r.li);
      if (r.kind === 'spike' && r.ratio > last.ratio) { last.ratio = r.ratio; last.at = r.step; last.value = r.value; }
      continue;
    }
    out.push({ run: r.run, kind: r.kind, what: r.what, layer: r.layer, from: r.step, to: r.step, at: r.step, ratio: r.ratio, value: r.value, layers: r.li === null ? [] : [r.li], nLayers: r.nLayers, count: 1 });
  }
  for (const a of out) a.layers.sort((x, y) => x - y);
  return out.sort((a, b) => a.from - b.from || a.run.localeCompare(b.run) || a.what.localeCompare(b.what));
}

/** "18-23" style compact list of layer numbers */
export function layerList(ls: number[]): string {
  const parts: string[] = [];
  for (let i = 0; i < ls.length; i++) {
    let j = i; while (j + 1 < ls.length && ls[j + 1] === ls[j] + 1) j++;
    parts.push(j > i ? `${ls[i]}-${ls[j]}` : String(ls[i]));
    i = j;
  }
  return parts.join(',');
}
