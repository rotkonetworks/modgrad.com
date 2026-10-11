// Runs: the index (static index.json or the server's api/index), selection, and data access.
//
// Reactivity: run objects are plain (live updates mutate them in place); signals say what changed:
//   runsVersion()      the run list or any run's table fields changed (runs table, browser)
//   catalogVersion()   runs were added/removed or a run gained metric keys: every query re-plans
//   runVersion(id)     that run's rows changed: only panels reading it re-evaluate
// Series are cached per run and metric as mutable arrays; live rows are appended in place.
import { createSignal, type Accessor, type Setter } from 'solid-js';
import { fmt, nums } from '@/lib/fmt';
import type { ArrayFile, Index, Run, RunEvent } from '@/logic/types';
import { catalogText } from '@/logic/data';
import { dataURL } from '@/lib/base';

export const S = { api: false, dev: false, index: null as Index | null, runs: [] as Run[], byId: new Map<string, Run>() };

const [runsVersion, setRunsVersion] = createSignal(0);
const [catalogVersion, setCatalogVersion] = createSignal(0);
export { runsVersion, catalogVersion };
export const touchRuns = (): void => { setRunsVersion((v) => v + 1); };
let catalogCache: { v: number; text: string } = { v: -1, text: '' };
/** the query catalog (tab-separated), recomputed when the catalog version changes */
export function catalog(): string {
  const v = catalogVersion();
  if (catalogCache.v !== v) catalogCache = { v, text: catalogText(S.runs) };
  return catalogCache.text;
}
function bumpCatalog(): void { setCatalogVersion((v) => v + 1); setRunsVersion((v) => v + 1); }

const runSigs = new Map<string, [Accessor<number>, Setter<number>]>();
function runSig(id: string): [Accessor<number>, Setter<number>] {
  let s = runSigs.get(id);
  if (!s) { s = createSignal(0); runSigs.set(id, s); }
  return s;
}
export const runVersion = (id: string): number => runSig(id)[0]();
export const bumpRun = (id: string): void => { runSig(id)[1]((v) => v + 1); };

// ------------------------------------------------------------ selection (Runs table checkboxes)
const [selSig, setSelSig] = createSignal<Set<string>>(new Set(), { equals: false });
export const selection = selSig;
export function setSelection(fn: (s: Set<string>) => void): void { const s = selSig(); fn(s); setSelSig(s); }
export const selectedRuns = (): Run[] => { const s = selSig(); runsVersion(); return S.runs.filter((r) => s.has(r.id)); };

// ------------------------------------------------------------ errors
const [errs, setErrs] = createSignal<string[]>([]);
export const errors = errs;
export function err(msg: string): void { setErrs((e) => (e.includes(msg) ? e : [...e, msg])); }
export function dismissErrors(): void { setErrs([]); }

// ------------------------------------------------------------ fetch cache
const cache = new Map<string, Promise<unknown>>();
// Snapshot mode: a `bundle.json` next to the data ({path: json}) carries every static file in one
// request, so a shared snapshot is a single data file. Absent bundle → per-file fetches as usual.
let bundle: Promise<Record<string, unknown> | null> | null = null;
function loadBundle(): Promise<Record<string, unknown> | null> {
  bundle ??= fetch(dataURL('bundle.json'), { cache: 'no-store' })
    .then((r) => (r.ok ? (r.json() as Promise<Record<string, unknown>>) : null))
    .catch(() => null);
  return bundle;
}
function fetchJSON(path: string): Promise<unknown> {
  const direct = () => fetch(dataURL(path), { cache: 'no-store' }).then((r) => { if (!r.ok) throw new Error(path + ': HTTP ' + r.status); return r.json(); });
  if (path.startsWith('api/')) return direct();
  return loadBundle().then((b) => (b && path in b ? b[path] : direct()));
}
export function getJSON<T>(path: string): Promise<T> {
  let p = cache.get(path);
  if (!p) {
    p = fetchJSON(path);
    p.catch(() => cache.delete(path));
    cache.set(path, p);
  }
  return p as Promise<T>;
}
const enc = encodeURIComponent;
/** the one place that knows where data lives: static export files or the server API */
export const src = {
  metric: (run: Run, key: string) => (S.api ? `api/runs/${enc(run.id)}/metrics/${enc(key)}?max_points=4000` : run.path + '/' + run.metrics[key].file),
  array: (run: Run, key: string) => (S.api ? `api/runs/${enc(run.id)}/arrays/${enc(key)}?max_steps=200` : run.path + '/' + run.arrays[key].file),
  events: (run: Run) => (S.api ? `api/runs/${enc(run.id)}/events` : run.path + '/events.json'),
};

// ------------------------------------------------------------ series store
export interface SeriesData { steps: number[]; values: number[] }
const series = new Map<string, Promise<SeriesData | null>>();
const skey = (id: string, k: string) => id + '\u0001' + k;
// Local runs (dropped files) are served from memory; like the server's ?max_points, long series
// are strided (never averaged).
const LOCAL_MAX_POINTS = 4000;
function localSeries(m: { step: ArrayLike<number>; value: ArrayLike<number> }): SeriesData {
  const n = m.step.length;
  if (n <= LOCAL_MAX_POINTS) return { steps: Array.from(m.step), values: Array.from(m.value) };
  const k = Math.ceil(n / LOCAL_MAX_POINTS), steps: number[] = [], values: number[] = [];
  for (let i = 0; i < n; i++) if (i % k === 0 || i === n - 1) { steps.push(m.step[i]); values.push(m.value[i]); }
  return { steps, values };
}
async function fetchSeries(run: Run, key: string): Promise<SeriesData | null> {
  if (run.local) return run.local.metrics[key] ? localSeries(run.local.metrics[key]) : null;
  if (run.data && run.data[key]) return { steps: run.data[key].step, values: Array.from(nums(run.data[key].value)) };
  if (!S.api && run.has_small) {
    const sm = await getJSON<{ step: number[]; keys: Record<string, (number | null)[]> }>(run.path + '/metrics.small.json');
    return { steps: sm.step.slice(), values: Array.from(nums(sm.keys[key] || [])) };
  }
  if (!run.metrics[key]) return null;
  const j = await getJSON<{ step: number[]; value: (number | null)[] }>(src.metric(run, key));
  return { steps: j.step.slice(), values: Array.from(nums(j.value)) };
}
/** one metric of one run: cached, kept current by live rows */
export function metricSeries(run: Run, key: string): Promise<SeriesData | null> {
  const k = skey(run.id, key);
  let p = series.get(k);
  if (!p) { p = fetchSeries(run, key); p.catch(() => series.delete(k)); series.set(k, p); }
  return p;
}
function dropRunCaches(r: Run): void {
  for (const k of [...series.keys()]) if (k.startsWith(r.id + '\u0001')) series.delete(k);
  for (const k of [...cache.keys()]) if ((r.path && k.startsWith(r.path + '/')) || k.startsWith(`api/runs/${enc(r.id)}/`)) cache.delete(k);
  events.delete(r.id);
}

/** live rows: append to loaded series in place (a re-send of the last step overwrites it) */
export async function appendRows(runId: string, rows: { step: number; [k: string]: number | undefined }[]): Promise<{ newKeys: boolean }> {
  const r = S.byId.get(runId);
  if (!r) return { newKeys: false };
  let newKeys = false;
  const waits: Promise<unknown>[] = [];
  for (const row of rows) {
    if (r.last_step == null || row.step > r.last_step) r.last_step = row.step;
    if (typeof row.t === 'number') r.last_t = row.t;
    for (const [k, v] of Object.entries(row)) {
      if (k === 'step' || k === 't' || typeof v !== 'number') continue;
      if (!r.metrics[k]) { r.metrics[k] = { file: 'metrics/' + k + '.json', n: 0 }; newKeys = true; }
      r.metrics[k].n++;
      const p = series.get(skey(runId, k));
      if (p) waits.push(p.then((s) => {
        if (!s) return;
        const n = s.steps.length;
        if (!n || row.step > s.steps[n - 1]) { s.steps.push(row.step); s.values.push(v); }
        else if (row.step === s.steps[n - 1]) s.values[n - 1] = v;
      }).catch(() => {}));
    }
  }
  r.n_rows = (r.n_rows || 0) + rows.length;
  await Promise.all(waits);
  bumpRun(runId);
  // new metric keys change the catalog: queries re-plan and the default board may follow
  if (newKeys) changed({ added: [], updated: [] }); else touchRuns();
  return { newKeys };
}

// ------------------------------------------------------------ events and arrays
const events = new Map<string, Promise<RunEvent[]>>();
export function eventsOf(runId: string): Promise<RunEvent[]> {
  let p = events.get(runId);
  if (!p) {
    const run = S.byId.get(runId);
    p = !run ? Promise.resolve([]) : run.local ? Promise.resolve(run.local.events || []) : run.n_events ? getJSON<RunEvent[]>(src.events(run)).catch(() => []) : Promise.resolve([]);
    events.set(runId, p);
  }
  return p;
}
export function invalidateEvents(runId: string): void { events.delete(runId); }
export function arrayOf(run: Run, key: string): Promise<ArrayFile> {
  return run.local ? Promise.resolve(run.local.arrays[key]) : getJSON<ArrayFile>(src.array(run, key));
}

// ------------------------------------------------------------ run list changes
const DEFAULTS = (): Omit<Run, 'id'> => ({ source: 'added', project: '', config: {}, summary: {}, metrics: {}, arrays: {}, unparsed: [], meta: {}, n_rows: 0, n_events: 0 });
export interface RunsChanged { added: string[]; updated: string[] }
const listeners = new Set<(c: RunsChanged) => void>();
export function onRunsChanged(fn: (c: RunsChanged) => void): () => void { listeners.add(fn); return () => listeners.delete(fn); }
export function changed(c: RunsChanged): void {
  bumpCatalog();
  for (const id of c.updated) bumpRun(id);
  for (const fn of listeners) fn(c);
  document.dispatchEvent(new CustomEvent('obs:runs-changed', { detail: c }));
}
/** Register new runs or replace existing ones (by id), e.g. from a live feed or another module.
 *  Each run needs at least {id, metrics: {key: {file, n}}}; missing fields get defaults. Cached
 *  data of an updated run is dropped so it is re-read. */
export function upsertRuns(runs: Partial<Run>[] | Partial<Run>, opts: { select?: boolean } = {}): RunsChanged {
  const added: string[] = [], updated: string[] = [];
  const sel = selSig();
  for (const r0 of ([] as Partial<Run>[]).concat(runs)) {
    if (!r0 || !r0.id) continue;
    const r = { ...DEFAULTS(), ...r0 } as Run;
    const i = S.runs.findIndex((x) => x.id === r.id);
    if (i >= 0) { dropRunCaches(S.runs[i]); S.runs[i] = r; updated.push(r.id); }
    else { S.runs.push(r); added.push(r.id); }
    S.byId.set(r.id, r);
    if (opts.select !== false && !r.config.synthetic) sel.add(r.id);
  }
  setSelSig(sel);
  changed({ added, updated });
  return { added, updated };
}
/** add runs under unique ids (dropped files: a second drop of the same name becomes name~2) */
export function addRunsUnique(runs: Run[], { select = true } = {}): string[] {
  const sel = selSig(), ids: string[] = [];
  for (const r of runs) {
    let id = r.id, k = 2; while (S.byId.has(id)) id = `${r.id}~${k++}`; r.id = id;
    S.runs.push(r); S.byId.set(r.id, r); if (select) sel.add(r.id); ids.push(id);
  }
  setSelSig(sel);
  changed({ added: ids, updated: [] });
  return ids;
}
export function removeRuns(pred: (r: Run) => boolean): void {
  for (const r of S.runs.filter(pred)) dropRunCaches(r);
  S.runs = S.runs.filter((r) => !pred(r)); S.byId.clear(); for (const r of S.runs) S.byId.set(r.id, r);
  setSelection((s) => { for (const id of [...s]) if (!S.byId.has(id)) s.delete(id); });
  changed({ added: [], updated: [] });
}
export function indexURL(project: string | null, run: string | null): string {
  const p = project || (run && run.startsWith('anon-') ? run.split('/')[0] : null);
  return 'api/index' + (p ? '?project=' + enc(p) : '');
}
/** re-read the index (reload button, live resync); local runs stay */
export async function reloadIndex(idxURL: string): Promise<void> {
  cache.clear(); series.clear(); events.clear();
  try {
    S.index = S.api ? await (await fetch(dataURL(idxURL), { cache: 'no-store' })).json() : await getJSON<Index>('index.json');
  } catch (e) { err(String(e)); return; }
  const local = S.runs.filter((r) => r.local);
  S.runs = S.index!.runs.concat(local); S.byId.clear(); for (const r of S.runs) S.byId.set(r.id, r);
  setSelection((s) => { for (const id of [...s]) if (!S.byId.has(id)) s.delete(id); });
  changed({ added: [], updated: S.runs.map((r) => r.id) });
}
/** first load: server mode when the API answers, else static files next to the page */
export async function loadIndex(idxURL: string): Promise<void> {
  try { const r = await fetch(dataURL(idxURL), { cache: 'no-store' }); if (r.ok) { const j = await r.json() as Index; if (j.server) { S.api = true; S.dev = !!j.dev; S.index = j; } } } catch { /* static mode */ }
  if (!S.api) {
    try { S.index = await getJSON<Index>('index.json'); }
    catch (e) { err('could not load index.json (' + e + '). Put index.json and runs/ next to this page (scripts/obs_collect.sh writes them), or run obs-server. Dropped log files still work.'); S.index = { runs: [], generated_unix: Date.now() / 1000 }; }
  }
  S.runs = S.index!.runs; S.byId.clear(); for (const r of S.runs) S.byId.set(r.id, r);
  for (const e of S.index!.errors || []) err('export error: ' + JSON.stringify(e));
  setSelection((s) => { for (const r of S.runs) if (!r.config.synthetic) s.add(r.id); });
  bumpCatalog();
}
export function genMeta(): string {
  runsVersion();
  const n = new Set(S.runs.map((r) => r.source)).size;
  const when = S.index ? new Date(S.index.generated_unix * 1000).toISOString().slice(0, 16).replace('T', ' ') + ' UTC' : '';
  return `${S.api ? 'server mode' : 'static mode'}, ${S.runs.length} run${S.runs.length === 1 ? '' : 's'} from ${n} source${n === 1 ? '' : 's'}${when ? ', index built ' + when : ''}`;
}
export const fmtConfig = (v: unknown): string => (typeof v === 'number' ? fmt(v) : String(v));
