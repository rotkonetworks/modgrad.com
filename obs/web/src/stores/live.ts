// Live updates in server mode: turns LiveClient events (SSE, or polling fallback) into run-store
// changes. Rows are appended to the cached series of that run only, so only the panels reading it
// re-evaluate. Also: #new (anonymous project), the waiting panel, and the per-run live badges.
import { createSignal } from 'solid-js';
import { LiveClient, type HelloEv, type RowsEv, type RunEv, type SummaryEv, type EventsEv, type SamplesEv, type Row } from '@/logic/liveClient';
import { ssGet, ssSet } from '@/lib/storage';
import { hash, setHash } from '@/lib/hash';
import { ago } from '@/lib/fmt';
import { DATA, dataURL } from '@/lib/base';
import { S, appendRows, upsertRuns, reloadIndex, indexURL, touchRuns, runsVersion, invalidateEvents, bumpRun, setSelection, changed } from '@/stores/runs';
import { appendSamples } from '@/stores/samples';

const tokenKey = (p: string) => 'obs-anon-token:' + p;
export const recallToken = (project: string): string | null => ssGet(tokenKey(project));

/** #new: create an anonymous project and turn the URL into its board */
export async function prepareLive(): Promise<void> {
  if (!hash().isNew) return;
  try {
    const r = await fetch(dataURL('api/v1/projects'), { method: 'POST' });
    if (!r.ok) return;
    const j = await r.json() as { project: string; token: string };
    ssSet(tokenKey(j.project), j.token);
    setHash({ project: j.project, run: null });
  } catch { /* static mode or server away */ }
}

const [status, setStatus] = createSignal('');
export const liveStatus = status;
const [tick, setTick] = createSignal(Date.now() / 1000);
const updated = new Map<string, number>(); // run -> server unix time of its last row
let skew = 0; // server clock minus ours, seconds

export function wanted(id: string): boolean {
  const t = hash();
  return t.run ? id === t.run : t.project ? id.startsWith(t.project + '/') : true;
}
/** the waiting panel: a run/project target with nothing to show yet */
export function waitingFor(): string | null {
  runsVersion();
  const t = hash();
  if (!S.api || !(t.run || t.project) || S.runs.some((r) => wanted(r.id))) return null;
  return t.run ? `run ${t.run}` : `project ${t.project}`;
}
export function liveBadge(id: string): { text: string; on: boolean; title: string } {
  const now = tick() + skew;
  runsVersion();
  const r = S.byId.get(id), u = updated.get(id);
  if (!r || r.finished || !u) return { text: '', on: false, title: '' };
  const a = now - u, on = a < 30;
  return { text: on ? 'live ' + ago(a) : 'last row ' + ago(a) + ' ago', on, title: 'last row at ' + new Date(u * 1000).toISOString().replace('T', ' ').slice(0, 19) + ' UTC' };
}

let client: LiveClient | null = null;
export function attachLive(): LiveClient {
  const t = hash();
  const opts = t.run && t.run.startsWith('anon-') ? { runs: [t.run] } : t.project ? { project: t.project } : {};
  const c = client = new LiveClient({ ...opts, base: DATA });
  setStatus('live: connecting');
  setInterval(() => setTick(Date.now() / 1000), 1000);
  c.addEventListener('status', (e) => { const d = (e as CustomEvent).detail as { mode: string; reason?: string }; setStatus('live: ' + (d.mode === 'sse' ? 'streaming' : d.mode === 'poll' ? 'polling (' + (d.reason || 'fallback') + ')' : d.mode)); });

  // rows arrive per step; batch them per run so a panel re-evaluates at most twice a second
  const pending = new Map<string, Row[]>();
  let timer: ReturnType<typeof setTimeout> | null = null;
  const flush = () => {
    timer = null;
    const batch = [...pending]; pending.clear();
    for (const [run, rows] of batch) appendRows(run, rows);
  };

  c.addEventListener('hello', (e) => {
    const d = (e as CustomEvent).detail as HelloEv;
    skew = d.unix - Date.now() / 1000;
    for (const x of d.runs) if (x.updated_unix) updated.set(x.run, x.updated_unix);
    for (const r of S.runs) if (!r.finished && !r.local) c.watch(r.id, r.last_step ?? -1); // dropped runs live only in this browser
    touchRuns();
  });
  c.addEventListener('run', (e) => {
    const d = (e as CustomEvent).detail as RunEv, entry = d.entry;
    if (!entry || !entry.id) { reloadIndex(indexURL(hash().project, hash().run)); return; }
    updated.set(d.run, d.unix);
    const known = S.byId.get(entry.id);
    if (wanted(entry.id) && !entry.config.synthetic) setSelection((s) => { s.add(entry.id); });
    if (known) {
      // a re-sent entry (config changed, arrays landed): take its metadata; row counts and
      // series stay with the `rows` events so nothing is counted twice
      const { metrics: _m, n_rows: _n, last_step, last_t: _t, ...meta } = entry;
      Object.assign(known, meta);
      if (last_step != null && (known.last_step == null || last_step > known.last_step)) known.last_step = last_step;
      changed({ added: [], updated: [] });
    }
    else upsertRuns([entry], { select: wanted(entry.id) });
    c.watch(entry.id, entry.last_step ?? -1);
  });
  c.addEventListener('rows', (e) => {
    const { run, rows, unix } = (e as CustomEvent).detail as RowsEv;
    if (!S.byId.has(run)) return; // its `run` event (or the next resync) brings it in
    updated.set(run, unix);
    const q = pending.get(run); if (q) q.push(...rows); else pending.set(run, rows.slice());
    if (!timer) timer = setTimeout(flush, 500);
  });
  c.addEventListener('events', (e) => {
    const d = (e as CustomEvent).detail as EventsEv;
    const r = S.byId.get(d.run); if (!r) return;
    r.n_events = (r.n_events || 0) + d.events.length;
    invalidateEvents(d.run); bumpRun(d.run);
  });
  c.addEventListener('samples', (e) => {
    const d = (e as CustomEvent).detail as SamplesEv;
    const r = S.byId.get(d.run); if (!r) return;
    const first = !r.n_samples;
    r.n_samples = Math.max(r.n_samples || 0, d.n);
    appendSamples(d.run, d.samples, d.n); // cached under the next version, so the bump below finds it
    bumpRun(d.run);
    if (first) touchRuns(); // the run joins the Samples tab's run list
  });
  c.addEventListener('summary', (e) => {
    const d = (e as CustomEvent).detail as SummaryEv;
    const r = S.byId.get(d.run); if (!r) return;
    r.finished = !!d.finished;
    if (r.finished) c.unwatch(r.id);
    touchRuns(); bumpRun(r.id);
  });
  c.addEventListener('resync', () => { reloadIndex(indexURL(hash().project, hash().run)); });
  c.start();
  return c;
}
export const liveClient = (): LiveClient | null => client;
