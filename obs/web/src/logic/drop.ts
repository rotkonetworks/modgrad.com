// Local file drop, DOM-free parts: grouping files into runs, the worker protocol (parsing happens
// in dropWorker.ts with the Rust ingest compiled to wasm; nothing leaves the browser), turning a
// parse result into a run, and best-effort IndexedDB persistence of dropped runs.
import type { Run } from '@/logic/types';

export const MAX_RUNS_STORED = 50;
const DB = 'modgrad-obs-local', STORE = 'runs';
// files worth trying when they arrive inside a folder (a single dropped file is always tried)
const FOLDER_OK = /(\.(csv|tsv|tab|jsonl|ndjson|json|log|txt|out|wandb)$)|^events\.out\.tfevents\./i;
const FOLDER_SKIP = /(^|\/)(\.|__pycache__\/|node_modules\/)|(^|\/)(requirements\.txt|debug(-internal)?\.log|wandb-metadata\.json|events\.json|metrics\.small\.json|index\.json)$/i;

export interface DropFile { rel: string; file: File }
export interface Group { name: string; files: DropFile[] }

/** a top-level file is its own run; files in a folder form one run per folder; a folder that is
 *  itself a run (metrics.jsonl, or a .wandb file) swallows its subfolders */
export function groupFiles(list: DropFile[]): { groups: Group[]; skipped: string[] } {
  const dirOf = (rel: string) => (rel.includes('/') ? rel.slice(0, rel.lastIndexOf('/')) : '');
  const base = (rel: string) => rel.slice(rel.lastIndexOf('/') + 1);
  const runDirs = new Set(list.filter((x) => /(^|\/)metrics\.jsonl$|\.wandb$/i.test(x.rel)).map((x) => dirOf(x.rel)).filter((d) => d));
  const under = (d: string) => [...runDirs].some((r) => d !== r && d.startsWith(r + '/'));
  const groups = new Map<string, Group>(), skipped: string[] = [];
  for (const x of list) {
    const d = dirOf(x.rel);
    if (!d) { groups.set('\u0000' + x.rel, { name: base(x.rel), files: [x] }); continue; }
    if (under(d) || FOLDER_SKIP.test(x.rel) || !FOLDER_OK.test(base(x.rel))) { skipped.push(x.rel); continue; }
    if (!groups.has(d)) groups.set(d, { name: d, files: [] });
    groups.get(d)!.files.push(x);
  }
  for (const g of groups.values()) g.files.sort((a, b) => a.rel.localeCompare(b.rel));
  return { groups: [...groups.values()], skipped };
}

/** folder drops: walk FileSystemEntry trees; plain file lists (synthetic drops, other browsers) as is */
export async function filesFromDrop(dt: DataTransfer): Promise<DropFile[]> {
  const entries = [...(dt.items || [])].filter((i) => i.kind === 'file').map((i) => (i.webkitGetAsEntry ? i.webkitGetAsEntry() : null));
  if (!entries.length || entries.some((x) => !x)) return [...dt.files].map((f) => ({ rel: f.webkitRelativePath || f.name, file: f }));
  const out: DropFile[] = [];
  const walk = async (ent: FileSystemEntry): Promise<void> => {
    if (ent.isFile) { const f = await new Promise<File>((res, rej) => (ent as FileSystemFileEntry).file(res, rej)); out.push({ rel: ent.fullPath.replace(/^\//, ''), file: f }); return; }
    if (!ent.isDirectory) return;
    const rd = (ent as FileSystemDirectoryEntry).createReader();
    for (;;) { // readEntries returns batches until empty
      const batch = await new Promise<FileSystemEntry[]>((res, rej) => rd.readEntries(res, rej));
      if (!batch.length) break;
      for (const c of batch) await walk(c);
    }
  };
  for (const ent of entries) await walk(ent as FileSystemEntry);
  return out;
}

// ------------------------------------------------------------ worker
export interface FileReport { file: string; format: string; metrics: number; steps: number; points: number; skipped?: number; bytes: number; error?: string; notes?: string[] }
export interface ParseMeta {
  keys: string[]; files: FileReport[]; config?: Record<string, never>; meta?: Record<string, unknown>; summary?: Record<string, unknown>;
  finished?: boolean; last_step?: number; last_t?: number; n_rows?: number; n_events?: number; skipped_lines?: number;
  arrays?: Run['arrays']; array_files?: NonNullable<Run['local']>['arrays']; events?: NonNullable<Run['local']>['events'];
}
export interface ParseDone { meta: ParseMeta; steps: Float64Array[]; values: Float64Array[]; ts: Float64Array; tv: Float64Array }
export interface Progress { done: number; total: number; file: string }
type Pending = { resolve: (m: ParseDone) => void; reject: (e: Error) => void; onProgress: (p: Progress) => void };

let worker: Worker | null = null, seq = 0;
const pending = new Map<number, Pending>();
function getWorker(): Worker {
  if (worker) return worker;
  const w = new Worker(new URL('./dropWorker.ts', import.meta.url), { type: 'module' });
  w.onmessage = (e: MessageEvent) => {
    const m = e.data, p = pending.get(m.id);
    if (!p) return;
    if (m.type === 'progress') p.onProgress(m);
    else { pending.delete(m.id); if (m.type === 'done') p.resolve(m); else p.reject(new Error(m.error)); }
  };
  w.onerror = (e) => { for (const p of pending.values()) p.reject(new Error('worker: ' + (e.message || 'failed to start'))); pending.clear(); w.terminate(); worker = null; };
  worker = w;
  return w;
}
export function parseGroup(g: Group, onProgress: (p: Progress) => void): Promise<ParseDone> {
  const id = ++seq;
  return new Promise<ParseDone>((resolve, reject) => {
    pending.set(id, { resolve, reject, onProgress });
    getWorker().postMessage({ id, name: g.name, files: g.files });
  }).catch((e) => { if (worker) { worker.terminate(); worker = null; } throw e; }); // a wasm trap poisons the instance
}

export function toRun(name: string, m: ParseDone): Run {
  const meta = m.meta, metrics: Run['metrics'] = {}, local: NonNullable<Run['local']>['metrics'] = {};
  meta.keys.forEach((k, i) => { local[k] = { step: m.steps[i], value: m.values[i] }; metrics[k] = { n: m.steps[i].length }; });
  return {
    id: 'local:' + name, name, source: 'local', project: String((meta.meta && meta.meta.project) || ''), path: null,
    config: meta.config || {}, meta: meta.meta || {}, summary: meta.summary || {}, finished: !!meta.finished,
    last_step: meta.last_step, last_t: meta.last_t, n_rows: meta.n_rows, n_events: meta.n_events,
    skipped_lines: meta.skipped_lines, unparsed: [], has_small: false, metrics, arrays: meta.arrays || {},
    local: { metrics: local, t: m.ts.length ? { step: m.ts, value: m.tv } : null, arrays: meta.array_files || {}, events: meta.events || [], files: meta.files, added: Date.now() },
  };
}

// ------------------------------------------------------------ persistence (best effort)
function openDB(): Promise<IDBDatabase> {
  return new Promise((res, rej) => {
    const q = indexedDB.open(DB, 1);
    q.onupgradeneeded = () => q.result.createObjectStore(STORE, { keyPath: 'id' });
    q.onsuccess = () => res(q.result);
    q.onerror = () => rej(q.error);
  });
}
export async function saveLocal(runs: Run[]): Promise<void> {
  try {
    const db = await openDB();
    const tx = db.transaction(STORE, 'readwrite');
    for (const r of runs) tx.objectStore(STORE).put(r);
    await new Promise((res, rej) => { tx.oncomplete = res; tx.onerror = () => rej(tx.error); tx.onabort = () => rej(tx.error); });
    // keep the newest MAX_RUNS_STORED
    const all = await new Promise<IDBValidKey[]>((res, rej) => { const q = db.transaction(STORE).objectStore(STORE).getAllKeys(); q.onsuccess = () => res(q.result); q.onerror = () => rej(q.error); });
    if (all.length > MAX_RUNS_STORED) {
      const t2 = db.transaction(STORE, 'readwrite');
      const olds = await new Promise<Run[]>((res) => { const q = t2.objectStore(STORE).getAll(); q.onsuccess = () => res(q.result); });
      olds.sort((a, b) => a.local!.added - b.local!.added).slice(0, all.length - MAX_RUNS_STORED).forEach((r) => t2.objectStore(STORE).delete(r.id));
    }
    db.close();
  } catch { /* storage unavailable (private window, quota): runs stay for this page only */ }
}
export async function restoreLocal(): Promise<Run[]> {
  try {
    const db = await openDB();
    const runs = await new Promise<Run[]>((res, rej) => { const q = db.transaction(STORE).objectStore(STORE).getAll(); q.onsuccess = () => res(q.result); q.onerror = () => rej(q.error); });
    db.close();
    return runs.sort((a, b) => a.local!.added - b.local!.added);
  } catch { return []; }
}
export async function clearLocal(): Promise<void> {
  try { const db = await openDB(); const tx = db.transaction(STORE, 'readwrite'); tx.objectStore(STORE).clear(); await new Promise((res) => { tx.oncomplete = res; tx.onerror = res; }); db.close(); } catch { /* no storage */ }
}
