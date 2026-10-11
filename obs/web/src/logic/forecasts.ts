// Main-thread side of the forecast worker (forecastWorker.ts). A fit of a 1500-point trace with a
// pooled search is ~10-35 ms of wasm; a board with several forecast panels would block a draw for
// all of them, so fits run in the worker and the panel redraws when one lands. Jobs are keyed by
// what the fit reads (the caller's cache key): a key already in flight is not posted again. If the
// worker cannot start, fits fall back to the main thread.
import { forecastTrace, type ForecastOpts } from '@/logic/data';
import type { Forecast } from '@/logic/types';

type Arr = ArrayLike<number>;
type Job = { done: (f: Forecast) => void; sync: () => Forecast };
let worker: Worker | null = null, broken = false, seq = 0;
const pending = new Map<number, Job>();
const inflight = new Set<string>();

function getWorker(): Worker | null {
  if (worker || broken) return worker;
  try {
    const w = new Worker(new URL('./forecastWorker.ts', import.meta.url), { type: 'module' });
    w.onmessage = (e: MessageEvent<{ id: number; f?: Forecast; error?: string }>) => {
      const m = e.data, j = pending.get(m.id);
      if (!j) return;
      pending.delete(m.id);
      j.done(m.f || { none: 'forecast failed: ' + (m.error || 'unknown error') });
    };
    w.onerror = () => {
      // the worker died (or never started): finish what was queued here, and stay on the main thread
      broken = true; w.terminate(); worker = null;
      const jobs = [...pending.values()]; pending.clear();
      for (const j of jobs) j.done(j.sync());
    };
    worker = w;
  } catch { broken = true; }
  return worker;
}

/** Fit `tr` to `to` in the worker and call `done` with the result; no-op while `key` is in flight. */
export function fitInWorker(key: string, tr: { xs: Arr; raw: Arr }, to: number, opts: ForecastOpts, done: (f: Forecast) => void): void {
  if (inflight.has(key)) return;
  inflight.add(key);
  const finish = (f: Forecast) => { inflight.delete(key); done(f); };
  const xs = Float64Array.from(tr.xs), raw = Float64Array.from(tr.raw);
  const pool = (opts.pool || []).map((p) => ({ xs: Float64Array.from(p.xs), raw: Float64Array.from(p.raw) }));
  const sync = () => forecastTrace({ xs, raw }, to, { ...opts, pool });
  const w = getWorker();
  if (!w) { finish(sync()); return; }
  const id = ++seq;
  pending.set(id, { done: finish, sync });
  w.postMessage({ id, xs, raw, to, opts: { ...opts, pool } });
}
