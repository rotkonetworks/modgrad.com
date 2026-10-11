// modgrad-obs live client: a pure module with no DOM access, reusable by the
// current vanilla dashboard and a later SolidJS port.
//
// const c = new LiveClient({ base: '', project: 'anon-…' | runs: ['a/b'] });
// c.addEventListener('rows', (e) => e.detail.run, e.detail.rows);
// c.start();
//
// Events (CustomEvent, payload in .detail, shapes as the server's SSE data):
//   status   {mode: 'connecting' | 'sse' | 'poll', reason?}
//   hello    {runs: [{run, finished, updated_unix}], unix}
//   run      {run, entry, unix}           a run appeared
//   rows     {run, rows: [{step, t, key: value}], unix}
//   events   {run, events: [{step, t, text}], unix}
//   summary  {run, finished, summary, unix}
//   resync   {run?}                        state may be stale: refetch
//
// Transport: Server-Sent Events from GET api/v1/stream; when EventSource is
// missing or the stream fails to open (or errors repeatedly), it falls back
// to polling: api/index for the run list and api/v1/runs/<id>/history?from=
// for the runs passed to watch(). Both paths emit the same events.

const KINDS = ['hello', 'run', 'rows', 'events', 'summary', 'resync'];

export class LiveClient extends EventTarget {
  constructor(o = {}) {
    super();
    this.base = o.base || '';
    this.project = o.project || null;
    this.runs = o.runs || null;
    this.pollMs = o.pollMs || 3000;
    this.indexMs = o.indexMs || 10000;
    this.openTimeoutMs = o.openTimeoutMs || 5000;
    this.fetch = o.fetch || ((...a) => globalThis.fetch(...a));
    this.ES = o.EventSource === undefined ? globalThis.EventSource : o.EventSource;
    this.mode = 'idle';
    this.watched = new Map(); // run -> last step seen (polling)
    this.known = null; // run ids seen (polling)
    this.helloSeen = false;
    this.timers = [];
  }

  emit(kind, detail) { this.dispatchEvent(new CustomEvent(kind, { detail })); }
  setMode(mode, reason) { if (this.mode !== mode) { this.mode = mode; this.emit('status', { mode, reason }); } }

  query() {
    if (this.runs) return '?runs=' + this.runs.map(encodeURIComponent).join(',');
    if (this.project) return '?project=' + encodeURIComponent(this.project);
    return '';
  }

  /// Runs to poll for rows (only used in polling mode); `lastStep` = what the caller already has.
  watch(run, lastStep) { if (!this.watched.has(run)) this.watched.set(run, lastStep ?? -1); }
  unwatch(run) { this.watched.delete(run); }

  start() {
    if (!this.ES) { this.startPolling('EventSource unavailable'); return this; }
    this.setMode('connecting');
    let opened = false, errors = 0;
    const es = new this.ES(this.base + 'api/v1/stream' + this.query());
    this.es = es;
    const giveUp = (why) => { if (this.es === es) { es.close(); this.es = null; this.startPolling(why); } };
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
        let d; try { d = JSON.parse(m.data); } catch (e) { return; }
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

  stop() {
    if (this.es) { this.es.close(); this.es = null; }
    for (const t of this.timers) { clearTimeout(t); clearInterval(t); }
    this.timers = [];
    this.setMode('idle');
  }

  async getJSON(path) {
    const r = await this.fetch(this.base + path, { cache: 'no-store' });
    if (!r.ok) throw new Error(path + ': HTTP ' + r.status);
    return r.json();
  }

  startPolling(reason) {
    this.setMode('poll', reason);
    const tickIndex = async () => {
      try {
        const q = this.project ? '?project=' + encodeURIComponent(this.project) : '';
        const idx = await this.getJSON('api/index' + q);
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
        this.lastFinished = new Map(idx.runs.map((e) => [e.id, e.finished]));
        this.known = ids;
      } catch (e) { /* server away: try again next tick */ }
    };
    const tickRows = async () => {
      for (const [run, last] of this.watched) {
        if (this.lastFinished?.get(run)) continue;
        try {
          const enc = run.split('/').map(encodeURIComponent).join('/');
          const h = await this.getJSON(`api/v1/runs/${enc}/history?from=${last + 1}`);
          if (!h.steps.length) continue;
          const keys = Object.keys(h.columns);
          const rows = h.steps.map((step, i) => {
            const r = { step, t: h.t[i] };
            for (const k of keys) if (h.columns[k][i] !== null) r[k] = h.columns[k][i];
            return r;
          });
          this.watched.set(run, h.steps[h.steps.length - 1]);
          this.emit('rows', { run, rows, unix: Date.now() / 1000 });
        } catch (e) { /* next tick */ }
      }
    };
    tickIndex();
    this.timers.push(setInterval(tickIndex, this.indexMs));
    this.timers.push(setInterval(tickRows, this.pollMs));
  }

  keeps(run) {
    if (this.runs) return this.runs.includes(run);
    if (this.project) return run.startsWith(this.project + '/');
    return true;
  }
}
