// Samples of a run, fetched once per run version and cached. Live `samples` events append to the
// cached copy (appendSamples) instead of refetching; other bumps (live rows) refetch. Server: api/v1/runs/<id>/samples; static export:
// <run path>/samples.json; both have the shape {samples, steps, keys, n}.
import type { Sample, SamplesData } from '@/logic/samples';
import { S, runVersion, getJSON } from '@/stores/runs';
import { dataURL } from '@/lib/base';

const cache = new Map<string, { v: number; p: Promise<SamplesData> }>();
const EMPTY: SamplesData = { samples: [], steps: [], keys: [], n: 0 };

export const SAMPLES_LIMIT = 20000;
export function samplesURL(runId: string): string | null {
  const run = S.byId.get(runId);
  if (!run || run.local) return null;
  return S.api ? `api/v1/runs/${runId.split('/').map(encodeURIComponent).join('/')}/samples?limit=${SAMPLES_LIMIT}` : run.path ? run.path + '/samples.json' : null;
}
/** a run's samples (reactive on the run's version when called inside a tracking scope) */
export function samplesOf(runId: string): Promise<SamplesData> {
  const v = runVersion(runId);
  const c = cache.get(runId);
  if (c && c.v === v) return c.p;
  const url = samplesURL(runId);
  const run = S.byId.get(runId);
  const p = !url || !run || !run.n_samples
    ? Promise.resolve(EMPTY)
    : S.api
      ? fetch(dataURL(url), { cache: 'no-store' }).then((r) => (r.ok ? (r.json() as Promise<SamplesData>) : EMPTY)).catch(() => EMPTY)
      : getJSON<SamplesData>(url).catch(() => EMPTY); // static export / bundle.json snapshot
  cache.set(runId, { v, p });
  return p;
}
/** live: `xs` are the run's newest samples and `n` its total after them. Merged into the cached
 *  copy when it is exactly `xs.length` behind (a gap refetches; a fetch that already has them is
 *  kept). Stored under the run's next version: call it right before bumpRun. */
export function appendSamples(runId: string, xs: Sample[], n: number): void {
  const c = cache.get(runId);
  if (!c) return;
  const v = runVersion(runId) + 1;
  const p = c.p.then((d): SamplesData | Promise<SamplesData> => {
    if (d.n >= n) return d; // the fetch already saw them
    if (d.n + xs.length !== n) { cache.delete(runId); return samplesOf(runId); } // missed some: refetch
    const steps = new Set(d.steps), keys = new Set(d.keys);
    for (const s of xs) { steps.add(s.step); keys.add(s.key); }
    const samples = d.samples.concat(xs);
    return { samples: samples.length > SAMPLES_LIMIT ? samples.slice(samples.length - SAMPLES_LIMIT) : samples,
      steps: [...steps].sort((a, b) => a - b), keys: [...keys].sort(), n };
  });
  cache.set(runId, { v, p });
}
/** runs that logged samples */
export const sampleRuns = (): string[] => S.runs.filter((r) => (r.n_samples || 0) > 0).map((r) => r.id);
