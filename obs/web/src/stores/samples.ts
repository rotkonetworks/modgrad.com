// Samples of a run, fetched once per run version (live rows bump it, so an RL run that logs every
// step keeps its samples fresh) and cached. Server: api/v1/runs/<id>/samples; static export:
// <run path>/samples.json; both have the shape {samples, steps, keys, n}.
import type { SamplesData } from '@/logic/samples';
import { S, runVersion } from '@/stores/runs';
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
    : fetch(dataURL(url), { cache: 'no-store' }).then((r) => (r.ok ? (r.json() as Promise<SamplesData>) : EMPTY)).catch(() => EMPTY);
  cache.set(runId, { v, p });
  return p;
}
/** runs that logged samples */
export const sampleRuns = (): string[] => S.runs.filter((r) => (r.n_samples || 0) > 0).map((r) => r.id);
