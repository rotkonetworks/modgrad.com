// Live updates for dashboard.html in server mode: the rendering side.
// The connection (SSE, reconnect, polling fallback) is live-client.js, which has
// no DOM access; this file turns its events into table badges and chart appends.
//
// URL hash:
//   #run=<id>        select that run (and wait for it if it does not exist yet)
//   #project=<p>     a board: only that project's runs, new ones appear live
//   #new             get an anonymous project token, show how to send data,
//                    then become #project=<p> and wait for the first row
//
// The dashboard calls prepareLive() before loading the index, indexURL() to
// load it, and attachLive(D) once it has rendered, passing its own helpers.

import { LiveClient } from './live-client.js';

const enc = encodeURIComponent;

export function liveTarget() {
  const p = new URLSearchParams(location.hash.replace(/^#/, ''));
  return { run: p.get('run'), project: p.get('project'), isNew: p.has('new') };
}

/// The index URL for the current target (a board lists only its project).
export function indexURL() {
  const t = liveTarget();
  const project = t.project || (t.run && t.run.startsWith('anon-') ? t.run.split('/')[0] : null);
  return 'api/index' + (project ? '?project=' + enc(project) : '');
}

const tokenKey = (p) => 'obs-anon-token:' + p;
function remember(project, token) { try { sessionStorage.setItem(tokenKey(project), token); } catch (e) { /* storage blocked */ } }
function recall(project) { try { return sessionStorage.getItem(tokenKey(project)); } catch (e) { return null; } }

/// #new: create an anonymous project and turn the URL into its board.
export async function prepareLive() {
  if (!liveTarget().isNew) return;
  try {
    const r = await fetch('api/v1/projects', { method: 'POST' });
    if (!r.ok) return;
    const j = await r.json();
    remember(j.project, j.token);
    history.replaceState(null, '', '#project=' + enc(j.project));
  } catch (e) { /* static mode or server away */ }
}

const CSS = `
.live-badge { font: 10px var(--mono); margin-left: 6px; color: var(--muted); white-space: nowrap; }
.live-badge.on { color: var(--c1); }
.live-badge.on::before { content: ''; display: inline-block; width: 6px; height: 6px; margin-right: 4px; background: var(--c1); vertical-align: 1px; animation: live-pulse 1.6s ease-in-out infinite; }
@keyframes live-pulse { 50% { opacity: 0.25; } }
@media (prefers-reduced-motion: reduce) { .live-badge.on::before { animation: none; } }
#live-status { font: 11px var(--mono); color: var(--muted); margin-left: 10px; }
.live-wait { border: 1px solid var(--frame); padding: 10px 12px; margin: 8px 0 12px; }
.live-wait b { font-weight: 600; }
.live-wait pre { font: 12px var(--mono); white-space: pre-wrap; word-break: break-all; margin: 4px 0 8px; padding: 6px 8px; border: 1px dashed var(--muted); }
.live-wait .dim { color: var(--muted); font-size: 11px; }
`;

function ago(s) {
  if (!(s >= 0)) return '';
  if (s < 90) return Math.round(s) + 's';
  if (s < 5400) return Math.round(s / 60) + 'm';
  if (s < 172800) return Math.round(s / 3600) + 'h';
  return Math.round(s / 86400) + 'd';
}

export function attachLive(D) {
  const { S, A, $ } = D;
  const target = liveTarget();
  const style = document.createElement('style'); style.textContent = CSS; document.head.appendChild(style);
  const updated = new Map(); // run -> server unix time of its last row
  let skew = 0; // server clock minus ours, seconds
  const now = () => Date.now() / 1000 + skew;

  const status = document.createElement('span'); status.id = 'live-status'; status.textContent = 'live: connecting';
  $('genmeta').after(status);

  // which runs to subscribe to
  const opts = target.run && target.run.startsWith('anon-') ? { runs: [target.run] } : target.project ? { project: target.project } : {};
  const client = new LiveClient(opts);
  client.addEventListener('status', (e) => { const d = e.detail; status.textContent = 'live: ' + (d.mode === 'sse' ? 'streaming' : d.mode === 'poll' ? 'polling (' + (d.reason || 'fallback') + ')' : d.mode); });

  // ---------------------------------------------------- selection and waiting
  const wanted = (id) => (target.run ? id === target.run : target.project ? id.startsWith(target.project + '/') : true);
  // with nothing chosen (the dashboard's preferred keys are ctmlm's), pick loss-like metrics
  function ensureMetrics() {
    const sel = $('metrics');
    if (sel.selectedOptions.length || !sel.options.length) return false;
    const opts = [...sel.options];
    const rank = (k) => (/(^|[/_])loss$/.test(k) ? 0 : /loss/.test(k) ? 1 : /acc|ppl|perplex|reward|score/.test(k) ? 2 : k.startsWith('sys/') ? 4 : 3);
    opts.sort((a, b) => rank(a.value) - rank(b.value)).slice(0, 4).forEach((o) => { o.selected = true; });
    return true;
  }
  if (target.run) { S.sel.clear(); if (S.byId.has(target.run)) S.sel.add(target.run); D.onSelection(); }
  if (ensureMetrics()) D.renderHistory();
  let wait = null;
  function showWaiting() {
    if (wait || !(target.run || target.project) || S.runs.some((r) => wanted(r.id))) return;
    wait = document.createElement('div'); wait.className = 'live-wait';
    const origin = location.origin;
    const what = target.run ? `run ${target.run}` : `project ${target.project}`;
    const token = target.project && recall(target.project);
    const dev = S.index && S.index.dev && !token; // a dev server takes tokenless writes
    const tok = token || '<token>';
    const run = target.run || 'my-run';
    const add = (label, text) => { const d = document.createElement('div'); d.innerHTML = '<b></b>'; d.firstChild.textContent = label; wait.appendChild(d); const p = document.createElement('pre'); p.textContent = text; wait.appendChild(p); };
    const head = document.createElement('div'); head.innerHTML = '<b>Listening</b> for <span></span>. Graphs appear here when the first row arrives.';
    head.querySelector('span').textContent = what; wait.appendChild(head);
    add('Ask your coding agent', `Add modgrad-obs logging to this repo. Server: ${origin}${dev ? '' : '  Token: ' + tok}  API reference: ${origin}/api/v1`);
    add('Or wrap your training command', `obs run --server ${origin}${dev ? '' : ' --token ' + tok} --name ${run} -- python train.py`);
    add('Or send one row with curl', `curl -X POST ${origin}/api/v1/runs/${run}/log${dev ? '' : ` -H "Authorization: Bearer ${tok}"`} -d '{"loss": 0.5}'`);
    const note = document.createElement('div'); note.className = 'dim';
    note.textContent = token ? 'This token writes to this board; keep it to yourself. Anyone with this page URL can read the board.' : dev ? 'Dev server: writes need no token.' : 'Use the token that came with this board (POST /api/v1/projects issues one).';
    wait.appendChild(note);
    $('errors').after(wait);
  }
  function hideWaiting() { if (wait) { wait.remove(); wait = null; } }
  showWaiting();

  // ---------------------------------------------------- badges
  function badges() {
    const t = now();
    for (const tr of document.querySelectorAll('#runtable tr[data-run]')) {
      const id = tr.dataset.run, r = S.byId.get(id), u = updated.get(id);
      let b = tr.querySelector('.live-badge');
      if (!b) { const cell = tr.children[1]; if (!cell) continue; b = document.createElement('span'); b.className = 'live-badge'; cell.appendChild(b); }
      let text = '', on = false;
      if (r && !r.finished && u) { const a = t - u; on = a < 30; text = on ? 'live ' + ago(a) : 'last row ' + ago(a) + ' ago'; }
      if (b.textContent !== text) b.textContent = text;
      b.classList.toggle('on', on);
      if (u) b.title = 'last row at ' + new Date(u * 1000).toISOString().replace('T', ' ').slice(0, 19) + ' UTC';
    }
  }
  setInterval(badges, 1000);

  // ---------------------------------------------------- charts
  // Cached metric JSON is patched in place, so any later re-render (EMA slider,
  // selection) sees the live points; open charts get their series replaced
  // through chart.update(), which keeps the zoom.
  function patch(j, step, v) {
    const n = j.step.length;
    if (!n || step > j.step[n - 1]) { j.step.push(step); j.value.push(v); }
    else if (step === j.step[n - 1]) j.value[n - 1] = v; // a re-send: last write wins
  }
  let chartTimer = null, fullTimer = null, lastFull = 0, tableTimer = null;
  // the run table (last step, metric count) is cheap to rebuild; do it at most every 2 s
  function scheduleTable() { if (!tableTimer) tableTimer = setTimeout(() => { tableTimer = null; D.renderRuns(); badges(); }, 2000); }
  // graph-builder panels re-evaluate on 'obs:runs-changed'; batch live rows into at most one per second
  const builderPending = new Set(); let builderTimer = null;
  function notifyBuilder(run) {
    builderPending.add(run);
    if (builderTimer) return;
    builderTimer = setTimeout(() => {
      builderTimer = null;
      const updated = [...builderPending]; builderPending.clear();
      document.dispatchEvent(new CustomEvent('obs:runs-changed', { detail: { added: [], updated } }));
    }, 1000);
  }
  function scheduleCharts() { if (!chartTimer) chartTimer = setTimeout(() => { chartTimer = null; updateCharts(); }, 300); }
  function scheduleFull(minGapMs = 0) {
    if (fullTimer) return;
    const wait = Math.max(300, lastFull + minGapMs - Date.now());
    fullTimer = setTimeout(() => { fullTimer = null; lastFull = Date.now(); D.renderHistory(); }, wait);
  }
  async function updateCharts() {
    // time axis and forecasts are derived per render; redraw those wholesale, at most every 3 s
    if ($('xaxis').value === 't' || $('fcast').checked) { scheduleFull(3000); return; }
    const w = +$('ema').value;
    for (const panel of document.querySelectorAll('#history .panel')) {
      const key = panel.dataset.key, chart = panel.chart;
      if (!key || !chart) continue;
      for (const r of S.runs) {
        if (!S.sel.has(r.id) || !r.metrics[key]) continue;
        const p = S.cache.get(D.src.metric(r, key));
        if (!p) { scheduleFull(); return; } // a series we never fetched: rebuild
        let j; try { j = await p; } catch (e) { continue; }
        const raw = Float64Array.from(j.value, (v) => (v === null ? NaN : v));
        const ys = w > 0 ? Array.from(A.ema(raw, w)) : Array.from(raw);
        if (!chart.update(r.id, j.step, ys, raw)) { scheduleFull(); return; }
      }
    }
  }

  // ---------------------------------------------------- events
  async function reloadIndex() {
    S.cache.clear();
    try { S.index = await (await fetch(indexURL(), { cache: 'no-store' })).json(); } catch (e) { return; }
    S.runs = S.index.runs.concat(S.runs.filter((r) => r.local)); S.byId.clear(); for (const r of S.runs) S.byId.set(r.id, r);
    D.renderRuns(); D.refreshMetricSelect(); scheduleFull();
  }

  client.addEventListener('hello', (e) => {
    const d = e.detail;
    skew = d.unix - Date.now() / 1000;
    for (const x of d.runs) if (x.updated_unix) updated.set(x.run, x.updated_unix);
    for (const r of S.runs) if (!r.finished) client.watch(r.id, r.last_step ?? -1);
    badges();
  });

  client.addEventListener('run', (e) => {
    const d = e.detail, entry = d.entry;
    if (!entry || !entry.id) { reloadIndex(); return; }
    updated.set(d.run, d.unix);
    const known = S.byId.get(entry.id);
    if (known) Object.assign(known, entry);
    else { S.runs.push(entry); S.byId.set(entry.id, entry); }
    if (wanted(entry.id) && !entry.config.synthetic) S.sel.add(entry.id);
    if (wanted(entry.id)) hideWaiting();
    client.watch(entry.id, entry.last_step ?? -1);
    D.renderRuns(); D.refreshMetricSelect(); ensureMetrics(); scheduleFull();
  });

  client.addEventListener('rows', (e) => {
    const { run, rows, unix } = e.detail;
    const r = S.byId.get(run);
    if (!r) return; // its `run` event (or the next resync) brings it in
    updated.set(run, unix);
    let newKeys = false;
    const patches = [];
    for (const row of rows) {
      if (r.last_step == null || row.step > r.last_step) r.last_step = row.step;
      for (const [k, v] of Object.entries(row)) {
        if (k === 'step' || k === 't') continue;
        if (!r.metrics[k]) { r.metrics[k] = { file: 'metrics/' + k + '.json', n: 0 }; newKeys = true; }
        r.metrics[k].n++;
        const p = S.cache.get(D.src.metric(r, k));
        if (p) patches.push(p.then((j) => patch(j, row.step, v)).catch(() => {}));
      }
    }
    r.n_rows = (r.n_rows || 0) + rows.length;
    scheduleTable();
    if (newKeys) { D.refreshMetricSelect(); if (ensureMetrics()) scheduleFull(); }
    Promise.all(patches).then(() => { if (S.sel.has(run)) scheduleCharts(); notifyBuilder(run); });
  });

  client.addEventListener('events', (e) => { const r = S.byId.get(e.detail.run); if (r) r.n_events = (r.n_events || 0) + e.detail.events.length; });

  client.addEventListener('summary', (e) => {
    const r = S.byId.get(e.detail.run);
    if (!r) return;
    r.finished = !!e.detail.finished;
    if (r.finished) client.unwatch(r.id);
    D.renderRuns(); badges();
  });

  client.addEventListener('resync', () => reloadIndex());

  addEventListener('hashchange', () => location.reload());
  client.start();
  return client;
}
