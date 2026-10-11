// The query engine's stream loader: metrics from the runs store, plus per-step statistics of
// logged arrays (`<array>.mean|min|max|rms`).
import { A } from '@/lib/wasm';
import { f32 } from '@/lib/fmt';
import { STATS, SAMPLES_METRIC, type Loader } from '@/logic/data';
import { S, metricSeries, arrayOf } from '@/stores/runs';

export const loader: Loader = async (runId, metric) => {
  const run = S.byId.get(runId); if (!run) return null;
  if (metric === SAMPLES_METRIC && !run.metrics[metric]) return { steps: new Float64Array(0), values: new Float64Array(0) }; // text, not a curve
  if (run.metrics[metric]) {
    const s = await metricSeries(run, metric);
    return s ? { steps: Float64Array.from(s.steps), values: Float64Array.from(s.values) } : null;
  }
  const d = metric.lastIndexOf('.');
  const arr = metric.slice(0, d), stat = STATS.indexOf(metric.slice(d + 1));
  if (d < 0 || stat < 0 || !run.arrays[arr]) return null;
  const j = await arrayOf(run, arr);
  const vals = new Float64Array(j.step.length);
  j.b64.forEach((b, i) => { vals[i] = A.array_stats(f32(b))[stat]; });
  return { steps: Float64Array.from(j.step), values: vals };
};
