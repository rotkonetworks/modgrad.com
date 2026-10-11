// modgrad-obs live client: no DOM access (ported from obs/live-client.js).
//
//   const c = new LiveClient({ base: '', project: 'anon-…' } | { runs: ['a/b'] });
//   c.addEventListener('rows', (e) => (e as CustomEvent).detail);
//   c.start();
//
// Events (CustomEvent, payload in .detail, shapes as the server's SSE data):
//   status   {mode: 'connecting' | 'sse' | 'poll' | 'idle', reason?}
//   hello    {runs: [{run, finished, updated_unix}], unix}
//   run      {run, entry, unix}           a run appeared
//   rows     {run, rows: [{step, t, key: value}], unix}
//   events   {run, events: [{step, t, text}], unix}
//   summary  {run, finished, summary, unix}
//   resync   {run?}                        state may be stale: refetch
//
// Transport: Server-Sent Events from GET api/v1/stream; when EventSource is missing or the stream
// fails to open (or errors repeatedly), it falls back to polling: api/index for the run list and
// api/v1/runs/<id>/history?from= for the runs passed to watch(). Both paths emit the same events.
import type { Run } from '@/logic/types';

const KINDS = ['hello', 'run', 'rows', 'events', 'summary', 'resync'] as const;
export type LiveMode = 'idle' | 'connecting' | 'sse' | 'poll';
export interface Row { step: number; t?: number; [k: string]: number | undefined }
export interface HelloEv { runs: { run: string; finished?: boolean; updated_unix?: number | null }[]; unix: number }
export interface RunEv { run: string; entry?: Run; unix: number }
export interface RowsEv { run: string; rows: Row[]; unix: number }
export interface EventsEv { run: string; events: { step: number; t?: number; text: string }[]; unix: number }
export interface SummaryEv { run: string; finished: boolean; summary: unknown; unix: number }
export interface LiveOpts { base?: string; project?: string | null; runs?: string[] | null; pollMs?: number; indexMs?: number; openTimeoutMs?: number; fetch?: typeof fetch; EventSource?: typeof EventSource | null }

export class LiveClient extends EventTarget {
  base: string; project: string | null; runs: string[] | null; pollMs: number; indexMs: number; openTimeoutMs: number;
  fetch: typeof fetch; ES: typeof EventSource | null | undefined;
  mode: LiveMode = 'idle';
  watched = new Map<string, number>(); // run -> last step seen (polling)
  known: Set<string> | null = null; // run ids seen (polling)
  lastFinished: Map<string, boolean> | null = null;
  helloSeen = false;
  timers: ReturnType<typeof setTimeout>[] = [];
  es: EventSource | null = null;

  constructor(o: LiveOpts = {}) {
    super();
    this.base = o.base || '';
    this.project = o.project || null;
    this.runs = o.runs || null;
    this.pollMs = o.pollMs || 3000;
    this.indexMs = o.indexMs || 10000;
    this.openTimeoutMs = o.openTimeoutMs || 5000;
    this.fetch = o.fetch || ((...a: Parameters<typeof fetch>) => globalThis.fetch(...a));
    this.ES = o.EventSource === undefined ? globalThis.EventSource : o.EventSource;
  }

  emit(kind: string, detail: unknown): void { this.dispatchEvent(new CustomEvent(kind, { detail })); }
  setMode(mode: LiveMode, reason?: string): void { if (this.mode !== mode) { this.mode = mode; this.emit('status', { mode, reason }); } }

  query(): string {
    if (this.runs) return '?runs=' + this.runs.map(encodeURIComponent).join(',');
    if (this.project) return '?project=' + encodeURIComponent(this.project);
    return '';
  }

  /** Runs to poll for rows (only used in polling mode); `lastStep` = what the caller already has. */
  watch(run: string, lastStep?: number | null): void { if (!this.watched.has(run)) this.watched.set(run, lastStep ?? -1); }
  unwatch(run: string): void { this.watched.delete(run); }

  start(): this {
    if (!this.ES) { this.startPolling('EventSource unavailable'); return this; }
    this.setMode('connecting');
    let opened = false, errors = 0;
    const es = new this.ES(this.base + 'api/v1/stream' + this.query());
    this.es = es;
    const giveUp = (why: string) => { if (this.es === es) { es.close(); this.es = null; this.startPolling(why); } };
    this.timers.push(setTimeout(() => { if (!opened) giveUp('stream did not open'); }, this.openTimeoutMs));
    es.onopen = () => { opened = true; errors = 0; this.setMode('sse'); };
    es.onerror = () => {
      // the browser reconnects by itself; repeated failures mean SSE is blocked (proxy, network)
      if (++errors >= 3 || es.readyState === 2) giveUp('stream failed');
      else this.setMode('connecting', 'reconnecting');
    };
    for (const k of KINDS) {
      es.addEventListener(k, (m) => {
        errors = 0;
        let d: unknown; try { d = JSON.parse((m as MessageEvent).data); } catch { return; }
        if (k === 'hello') {
          // a second hello means we reconnected: anything in between was missed
          if (this.helloSeen) this.emit('resync', {});
          this.helloSeen = true;
        }
        this.emit(k, d);
      });
    }
    return this;
  }

  stop(): void {
    if (this.es) { this.es.close(); this.es = null; }
    for (const t of this.timers) { clearTimeout(t); clearInterval(t); }
    this.timers = [];
    this.setMode('idle');
  }

  async getJSON<T>(path: string): Promise<T> {
    const r = await this.fetch(this.base + path, { cache: 'no-store' });
    if (!r.ok) throw new Error(path + ': HTTP ' + r.status);
    return r.json() as Promise<T>;
  }

  startPolling(reason: string): void {
    this.setMode('poll', reason);
    const tickIndex = async () => {
      try {
        const q = this.project ? '?project=' + encodeURIComponent(this.project) : '';
        const idx = await this.getJSON<{ runs: Run[] }>('api/index' + q);
        const unix = Date.now() / 1000;
        const ids = new Set(idx.runs.map((e) => e.id));
        if (this.known === null) {
          this.emit('hello', { unix, runs: idx.runs.filter((e) => this.keeps(e.id)).map((e) => ({ run: e.id, finished: e.finished, updated_unix: e.meta && e.meta.start_unix != null && e.last_t != null ? e.meta.start_unix + e.last_t : null })) });
        } else {
          for (const e of idx.runs) if (!this.known.has(e.id) && this.keeps(e.id)) this.emit('run', { run: e.id, entry: e, unix });
        }
        for (const e of idx.runs) {
          if (this.known && this.known.has(e.id) && this.lastFinished?.get(e.id) === false && e.finished) this.emit('summary', { run: e.id, finished: true, summary: null, unix });
        }
        this.lastFinished = new Map(idx.runs.map((e) => [e.id, !!e.finished]));
        this.known = ids;
      } catch { /* server away: try again next tick */ }
    };
    const tickRows = async () => {
      for (const [run, last] of this.watched) {
        if (this.lastFinished?.get(run)) continue;
        try {
          const enc = run.split('/').map(encodeURIComponent).join('/');
          const h = await this.getJSON<{ steps: number[]; t: number[]; columns: Record<string, (number | null)[]> }>(`api/v1/runs/${enc}/history?from=${last + 1}`);
          if (!h.steps.length) continue;
          const keys = Object.keys(h.columns);
          const rows = h.steps.map((step, i) => {
            const r: Row = { step, t: h.t[i] };
            for (const k of keys) { const v = h.columns[k][i]; if (v !== null) r[k] = v; }
            return r;
          });
          this.watched.set(run, h.steps[h.steps.length - 1]);
          this.emit('rows', { run, rows, unix: Date.now() / 1000 });
        } catch { /* next tick */ }
      }
    };
    tickIndex();
    this.timers.push(setInterval(tickIndex, this.indexMs));
    this.timers.push(setInterval(tickRows, this.pollMs));
  }

  keeps(run: string): boolean {
    if (this.runs) return this.runs.includes(run);
    if (this.project) return run.startsWith(this.project + '/');
    return true;
  }
}
