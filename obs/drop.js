// Drag and drop your training logs onto the dashboard: CSV/TSV, JSONL, plain
// text logs, TensorBoard event files, W&B offline .wandb files and modgrad-obs
// run folders. Parsed in a Web Worker by the Rust ingest compiled to wasm
// (drop-worker.js); nothing leaves the browser. Each dropped file, or each
// folder of files, becomes a run with source "local" that the rest of the
// dashboard reads through run.local (see metricSeries in dashboard.html).
//
//   initDrop({ addRuns, removeRuns, err })   called once from dashboard main()

const MAX_RUNS_STORED = 50;
const DB = 'modgrad-obs-local', STORE = 'runs';
// files worth trying when they arrive inside a folder (a single dropped file is always tried)
const FOLDER_OK = /(\.(csv|tsv|tab|jsonl|ndjson|json|log|txt|out|wandb)$)|^events\.out\.tfevents\./i;
const FOLDER_SKIP = /(^|\/)(\.|__pycache__\/|node_modules\/)|(^|\/)(requirements\.txt|debug(-internal)?\.log|wandb-metadata\.json|events\.json|metrics\.small\.json|index\.json)$/i;

let api = null, worker = null, seq = 0;
const pending = new Map(); // id -> {resolve, reject, onProgress}
let ui = null;

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const mb = (n) => (n >= 1 << 20 ? (n / (1 << 20)).toFixed(1) + ' MB' : n >= 1024 ? (n / 1024).toFixed(1) + ' kB' : n + ' B');

// ------------------------------------------------------------ styles + DOM
function injectUI() {
  const css = document.createElement('style');
  css.textContent = `
.dropzone { position: fixed; inset: 0; z-index: 50; display: none; align-items: center; justify-content: center; background: color-mix(in srgb, var(--bg) 88%, transparent); }
.dropzone.on { display: flex; }
.dropzone .box { border: 2px dashed var(--frame); background: var(--bg); padding: 28px 36px; max-width: min(560px, calc(100vw - 32px)); text-align: center; font: 13px/1.5 var(--mono); }
.dropzone .box b { display: block; font-size: 15px; margin-bottom: 6px; }
.dropzone .box span { color: var(--muted); font-size: 11px; }
.dropreport { border: 1px solid var(--rule); margin: 8px 0; padding: 4px 8px 6px; font: 11px var(--mono); }
.dropreport .hd { display: flex; gap: 10px; align-items: baseline; }
.dropreport .hd b { font-weight: 600; }
.dropreport .hd .sp { flex: 1; }
.dropreport .tw { overflow-x: auto; }
.dropreport table { border-collapse: collapse; width: 100%; margin-top: 4px; }
.dropreport td, .dropreport th { padding: 1px 8px 1px 0; text-align: left; white-space: nowrap; vertical-align: top; border-bottom: 1px dotted var(--rule); }
.dropreport th { color: var(--muted); font-weight: 600; }
.dropreport td.num { text-align: right; }
.dropreport td.notes { white-space: normal; color: var(--muted); min-width: 220px; }
.dropreport .bad { color: var(--c3); }
.dropreport progress { width: 160px; height: 8px; vertical-align: middle; accent-color: var(--c1); }
.droptools { display: inline-flex; gap: 6px; align-items: baseline; flex-wrap: wrap; }
.droptools .hint { font: 11px var(--mono); color: var(--muted); }
`;
  document.head.append(css);

  const header = document.querySelector('header');
  const tools = document.createElement('span');
  tools.className = 'droptools';
  tools.innerHTML = `<span class="hint" title="CSV/TSV, JSONL, text logs, TensorBoard event files, W&amp;B .wandb, modgrad-obs run folders. Parsed in your browser; nothing is uploaded.">drop logs anywhere, or</span>
<button id="dropopen" title="parse log files in this browser (nothing is uploaded)">open files</button>
<button id="dropdir" title="open a folder (a run directory, a TensorBoard logdir, ...)">open folder</button>
<button id="dropforget" title="remove the local runs from this page and from this browser's storage" style="display:none">forget local runs</button>
<input id="dropinput" type="file" multiple hidden>
<input id="dropdirinput" type="file" webkitdirectory multiple hidden>`;
  header.insertBefore(tools, document.getElementById('reload'));

  const zone = document.createElement('div');
  zone.className = 'dropzone';
  zone.innerHTML = `<div class="box"><b>drop logs to graph them</b>parsed in your browser, nothing is uploaded<br><span>CSV / TSV · JSONL · text logs · TensorBoard events.out.tfevents.* · W&amp;B .wandb · modgrad-obs run folders</span></div>`;
  document.body.append(zone);

  const report = document.createElement('div');
  report.className = 'dropreport';
  report.style.display = 'none';
  document.getElementById('errors').after(report);

  ui = { zone, report, forget: tools.querySelector('#dropforget') };
  const input = tools.querySelector('#dropinput'), dirInput = tools.querySelector('#dropdirinput');
  tools.querySelector('#dropopen').addEventListener('click', () => input.click());
  tools.querySelector('#dropdir').addEventListener('click', () => dirInput.click());
  for (const inp of [input, dirInput]) inp.addEventListener('change', () => { const files = [...inp.files].map((f) => ({ rel: f.webkitRelativePath || f.name, file: f })); inp.value = ''; ingest(files); });
  ui.forget.addEventListener('click', forgetLocal);
}

// ------------------------------------------------------------ drag and drop
function hasFiles(e) { return e.dataTransfer && [...e.dataTransfer.types].includes('Files'); }
function wireDrag() {
  let depth = 0;
  addEventListener('dragenter', (e) => { if (!hasFiles(e)) return; e.preventDefault(); depth++; ui.zone.classList.add('on'); });
  addEventListener('dragover', (e) => { if (!hasFiles(e)) return; e.preventDefault(); e.dataTransfer.dropEffect = 'copy'; });
  // enter/leave pairs are counted; a leave at the window edge (drag taken out or cancelled
  // there) always hides, in case the browser dropped a pair
  const outside = (e) => e.clientX <= 0 || e.clientY <= 0 || e.clientX >= innerWidth || e.clientY >= innerHeight;
  addEventListener('dragleave', (e) => { if (!hasFiles(e)) return; depth = Math.max(0, depth - 1); if (!depth || outside(e)) { depth = 0; ui.zone.classList.remove('on'); } });
  addEventListener('dragend', () => { depth = 0; ui.zone.classList.remove('on'); });
  addEventListener('drop', async (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault(); depth = 0; ui.zone.classList.remove('on');
    ingest(await filesFromDrop(e.dataTransfer));
  });
}

// folder drops: walk FileSystemEntry trees; plain file lists (synthetic drops, other browsers) as is
async function filesFromDrop(dt) {
  const entries = [...(dt.items || [])].filter((i) => i.kind === 'file').map((i) => (i.webkitGetAsEntry ? i.webkitGetAsEntry() : null));
  if (!entries.length || entries.some((x) => !x)) return [...dt.files].map((f) => ({ rel: f.webkitRelativePath || f.name, file: f }));
  const out = [];
  const walk = async (ent) => {
    if (ent.isFile) { const f = await new Promise((res, rej) => ent.file(res, rej)); out.push({ rel: ent.fullPath.replace(/^\//, ''), file: f }); return; }
    if (!ent.isDirectory) return;
    const rd = ent.createReader();
    for (;;) { // readEntries returns batches until empty
      const batch = await new Promise((res, rej) => rd.readEntries(res, rej));
      if (!batch.length) break;
      for (const c of batch) await walk(c);
    }
  };
  for (const ent of entries) await walk(ent);
  return out;
}

// [{rel, file}] -> [{name, files}]: a top-level file is its own run; files in a folder form one
// run per folder; a folder that is itself a run (metrics.jsonl, or a .wandb file) swallows its subfolders
export function groupFiles(list) {
  const dirOf = (rel) => (rel.includes('/') ? rel.slice(0, rel.lastIndexOf('/')) : '');
  const base = (rel) => rel.slice(rel.lastIndexOf('/') + 1);
  const runDirs = new Set(list.filter((x) => /(^|\/)metrics\.jsonl$|\.wandb$/i.test(x.rel)).map((x) => dirOf(x.rel)).filter((d) => d));
  const under = (d) => [...runDirs].some((r) => d !== r && d.startsWith(r + '/'));
  const groups = new Map(), skipped = [];
  for (const x of list) {
    const d = dirOf(x.rel);
    if (!d) { groups.set('\u0000' + x.rel, { name: base(x.rel), files: [x] }); continue; }
    if (under(d) || FOLDER_SKIP.test(x.rel) || !FOLDER_OK.test(base(x.rel))) { skipped.push(x.rel); continue; }
    if (!groups.has(d)) groups.set(d, { name: d, files: [] });
    groups.get(d).files.push(x);
  }
  for (const g of groups.values()) g.files.sort((a, b) => a.rel.localeCompare(b.rel));
  return { groups: [...groups.values()], skipped };
}

// ------------------------------------------------------------ worker
function getWorker() {
  if (worker) return worker;
  worker = new Worker(new URL('./drop-worker.js', import.meta.url), { type: 'module' });
  worker.onmessage = (e) => {
    const m = e.data, p = pending.get(m.id);
    if (!p) return;
    if (m.type === 'progress') p.onProgress(m);
    else { pending.delete(m.id); m.type === 'done' ? p.resolve(m) : p.reject(new Error(m.error)); }
  };
  worker.onerror = (e) => { for (const p of pending.values()) p.reject(new Error('worker: ' + (e.message || 'failed to start'))); pending.clear(); worker.terminate(); worker = null; };
  return worker;
}
function parseGroup(g, onProgress) {
  const id = ++seq;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject, onProgress });
    getWorker().postMessage({ id, name: g.name, files: g.files });
  }).catch((e) => { if (worker) { worker.terminate(); worker = null; } throw e; }); // a wasm trap poisons the instance
}

function toRun(name, m) {
  const meta = m.meta, metrics = {}, local = {};
  meta.keys.forEach((k, i) => { local[k] = { step: m.steps[i], value: m.values[i] }; metrics[k] = { n: m.steps[i].length }; });
  return {
    id: 'local:' + name, name, source: 'local', project: (meta.meta && meta.meta.project) || '', path: null,
    config: meta.config || {}, meta: meta.meta || {}, summary: meta.summary || {}, finished: !!meta.finished,
    last_step: meta.last_step, last_t: meta.last_t, n_rows: meta.n_rows, n_events: meta.n_events,
    skipped_lines: meta.skipped_lines, unparsed: [], has_small: false, metrics, arrays: meta.arrays || {},
    local: { metrics: local, t: m.ts.length ? { step: m.ts, value: m.tv } : null, arrays: meta.array_files || {}, events: meta.events || [], files: meta.files, added: Date.now() },
  };
}

// ------------------------------------------------------------ report
const state = { batches: [] }; // newest first
function renderReport() {
  const r = ui.report;
  if (!state.batches.length) { r.style.display = 'none'; r.innerHTML = ''; return; }
  r.style.display = '';
  let html = `<div class="hd"><b>local files</b><span>parsed in this browser, nothing uploaded</span><span class="sp"></span><button id="dropdismiss" title="hide this report (the runs stay)">dismiss</button></div>`;
  html += '<div class="tw"><table><tr><th>file</th><th>run</th><th>format</th><th>metrics</th><th>steps</th><th>points</th><th>skipped</th><th>notes</th></tr>';
  for (const b of state.batches) {
    for (const g of b.groups) {
      if (g.error) { html += `<tr><td>${esc(g.name)}</td><td></td><td colspan="5" class="bad">failed</td><td class="notes bad">${esc(g.error)}</td></tr>`; continue; }
      if (g.progress !== undefined && !g.files) {
        const pct = g.total ? Math.round((100 * g.progress) / g.total) : 0;
        html += `<tr><td>${esc(g.name)}</td><td></td><td colspan="5"><progress max="100" value="${pct}"></progress> ${pct}% of ${mb(g.total)}</td><td class="notes">${esc(g.file || '')}</td></tr>`;
        continue;
      }
      for (const f of g.files || []) {
        const notes = (f.error ? [f.error] : []).concat(f.notes || []).join('; ');
        html += `<tr><td title="${esc(mb(f.bytes))}">${esc(f.file)}</td><td>${esc(g.runId || '(nothing to graph)')}</td><td>${esc(f.format)}</td><td class="num">${f.metrics}</td><td class="num">${f.steps}</td><td class="num">${f.points}</td><td class="num">${f.skipped || ''}</td><td class="notes${f.error ? ' bad' : ''}">${esc(notes)}</td></tr>`;
      }
    }
    if (b.skipped.length) html += `<tr><td colspan="7" class="notes">not tried (${b.skipped.length}): ${esc(b.skipped.slice(0, 12).join(', '))}${b.skipped.length > 12 ? ', ...' : ''}</td><td></td></tr>`;
  }
  r.innerHTML = html + '</table></div>';
  r.querySelector('#dropdismiss').addEventListener('click', () => { state.batches = []; renderReport(); });
}

// ------------------------------------------------------------ ingest
const queue = [];
let busy = false;
export async function ingest(list) {
  if (!list || !list.length) return;
  const { groups, skipped } = groupFiles(list);
  const batch = { groups: groups.map((g) => ({ name: g.name, src: g, progress: 0, total: g.files.reduce((a, f) => a + f.file.size, 0) })), skipped };
  state.batches.unshift(batch);
  renderReport();
  ui.report.scrollIntoView({ block: 'nearest' });
  queue.push(batch);
  if (busy) return;
  busy = true;
  try {
    while (queue.length) {
      const b = queue.shift();
      const added = [];
      for (const g of b.groups) {
        let raf = 0;
        try {
          const m = await parseGroup(g.src, (p) => { g.progress = p.done; g.file = p.file; if (!raf) raf = requestAnimationFrame(() => { raf = 0; renderReport(); }); });
          g.files = m.meta.files;
          if (m.meta.keys.length || Object.keys(m.meta.arrays || {}).length) { const run = toRun(g.name, m); added.push(run); }
        } catch (e) {
          g.error = String(e.message || e);
        }
        if (raf) cancelAnimationFrame(raf);
        renderReport();
      }
      if (added.length) {
        api.addRuns(added);
        b.groups.forEach((g) => { const r = added.find((x) => x.name === g.name); if (r) g.runId = r.id; });
        renderReport();
        save(added);
      }
    }
  } finally {
    busy = false;
  }
  refreshForget();
}

// ------------------------------------------------------------ persistence (best effort)
let localIds = new Set();
function refreshForget() { ui.forget.style.display = localIds.size ? '' : 'none'; ui.forget.textContent = `forget local runs (${localIds.size})`; }
function openDB() {
  return new Promise((res, rej) => {
    const q = indexedDB.open(DB, 1);
    q.onupgradeneeded = () => q.result.createObjectStore(STORE, { keyPath: 'id' });
    q.onsuccess = () => res(q.result);
    q.onerror = () => rej(q.error);
  });
}
async function save(runs) {
  for (const r of runs) localIds.add(r.id);
  refreshForget();
  try {
    const db = await openDB();
    const tx = db.transaction(STORE, 'readwrite');
    for (const r of runs) tx.objectStore(STORE).put(r);
    await new Promise((res, rej) => { tx.oncomplete = res; tx.onerror = () => rej(tx.error); tx.onabort = () => rej(tx.error); });
    // keep the newest MAX_RUNS_STORED
    const all = await new Promise((res, rej) => { const q = db.transaction(STORE).objectStore(STORE).getAllKeys(); q.onsuccess = () => res(q.result); q.onerror = () => rej(q.error); });
    if (all.length > MAX_RUNS_STORED) {
      const t2 = db.transaction(STORE, 'readwrite');
      const olds = await new Promise((res) => { const q = t2.objectStore(STORE).getAll(); q.onsuccess = () => res(q.result); });
      olds.sort((a, b) => a.local.added - b.local.added).slice(0, all.length - MAX_RUNS_STORED).forEach((r) => t2.objectStore(STORE).delete(r.id));
    }
    db.close();
  } catch (e) { /* storage unavailable (private window, quota): runs stay for this page only */ }
}
async function restore() {
  try {
    const db = await openDB();
    const runs = await new Promise((res, rej) => { const q = db.transaction(STORE).objectStore(STORE).getAll(); q.onsuccess = () => res(q.result); q.onerror = () => rej(q.error); });
    db.close();
    if (!runs.length) return;
    runs.sort((a, b) => a.local.added - b.local.added);
    for (const r of runs) localIds.add(r.id);
    api.addRuns(runs, { select: false });
    refreshForget();
  } catch (e) { /* no storage */ }
}
async function forgetLocal() {
  api.removeRuns((r) => !!r.local);
  localIds = new Set();
  state.batches = []; renderReport(); refreshForget();
  try { const db = await openDB(); const tx = db.transaction(STORE, 'readwrite'); tx.objectStore(STORE).clear(); await new Promise((res) => { tx.oncomplete = res; tx.onerror = res; }); db.close(); } catch (e) { /* no storage */ }
}

export function initDrop(a) {
  api = a;
  injectUI();
  wireDrag();
  restore();
}
