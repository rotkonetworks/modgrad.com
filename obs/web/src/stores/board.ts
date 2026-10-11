// The board: panels (query + view), undo, settings, selection/focus, presets, import/export.
//
// The model (M.board) is a plain object that actions mutate inside commit(); after every commit a
// Solid store mirror is reconciled by panel id, so a change re-renders only the panels it touched.
// Persistence: localStorage 'obs-builder-board-v2' (per data target: '…:run=<id>', '…:project=<p>'),
// settings under 'obs-builder-settings-v1'. A shared board (#board=<name>) is shown unsaved until
// "save as my board".
import { createSignal } from 'solid-js';
import { createStore, reconcile } from 'solid-js/store';
import * as Bd from '@/logic/board';
import { fieldsOf, hasReplicates, metricCounts, type Stream } from '@/logic/data';
import type { Board, FitForecast, Forecast, Panel, QueryResult, Run } from '@/logic/types';
import { lsGet, lsSet } from '@/lib/storage';
import { hash, setHash } from '@/lib/hash';
import { S, selectedRuns, getJSON, err, onRunsChanged } from '@/stores/runs';
import { showToast } from '@/stores/ui';

const LS_BOARD = 'obs-builder-board-v2';
const LS_SET = 'obs-builder-settings-v1';

export interface Settings { palette: boolean; events: boolean; spikes: boolean; linkX: boolean }
export const M = {
  board: { version: Bd.BOARD_VERSION, panels: [] } as Board,
  undo: Bd.makeUndo(),
  autoBoard: false,
  key: LS_BOARD,
  pushedFocus: false,
  scrollY: 0,
};
const [board, setBoardStore] = createStore<Board>({ version: Bd.BOARD_VERSION, panels: [] });
export { board };
const [settings, setSettings] = createStore<Settings>({ palette: true, events: true, spikes: true, linkX: true });
export { settings };
export function setSetting<K extends keyof Settings>(k: K, v: Settings[K]): void { setSettings(k, v); lsSet(LS_SET, JSON.stringify({ ...settings })); }

const [selPanel, setSelPanel] = createSignal<string | null>(null);
export { selPanel, setSelPanel };
const [editingSig, setEditingSig] = createSignal<Set<string>>(new Set(), { equals: false });
export const editing = editingSig;
export function toggleEditing(id: string, on?: boolean): void { const s = editingSig(); if (on ?? !s.has(id)) s.add(id); else s.delete(id); setEditingSig(s); }
const [inspector, setInspector] = createSignal(false);
export { inspector, setInspector };
const [undoTop, setUndoTop] = createSignal<string | null>(null);
export { undoTop };
const [shared, setShared] = createSignal<{ name: string; title: string; source: string } | null>(null);
export { shared };

/** focus view: the panel named in the hash, if it exists */
export const focusId = (): string | null => { const id = hash().panel; return id && board.panels.some((p) => p.id === id) ? id : null; };

// ------------------------------------------------------------ per-panel runtime state (not saved)
export const results = new Map<string, QueryResult>(); // last evaluation per panel
export const fcHist = new Map<string, { cur: FitForecast; prev: FitForecast | null }>(); // accountability traces
export const fcCache = new Map<string, Forecast>();
export const seen = new Map<string, Set<string>>(); // panel -> runs seen at the last evaluation
export const fresh = new Map<string, number>(); // panel \u0001 run -> time it newly matched

// ------------------------------------------------------------ commit / undo / persistence
const clone = <T>(x: T): T => JSON.parse(JSON.stringify(x)) as T;
function sync(): void {
  setBoardStore(reconcile(clone(M.board), { key: 'id' }));
  const t = M.undo.peek(); setUndoTop(t ? t.what : null);
}
/** an untouched default board is saved with auto: true, so it keeps following the data after a reload */
function save(): void {
  if (shared()) return;
  lsSet(M.key, M.autoBoard ? JSON.stringify({ version: Bd.BOARD_VERSION, auto: true, panels: M.board.panels }, null, 1) : Bd.serialize(M.board));
}
export const findPanel = (id: string | null | undefined): Panel | undefined => (id ? M.board.panels.find((p) => p.id === id) : undefined);
export const parseQ = Bd.parse;
export function titleOf(p: { title: string; q: string }): string {
  if (p.title) return p.title;
  const pq = Bd.parse(p.q); if (!pq.ok) return p.q;
  const ms = pq.chips.filter((c) => c.kind === 'metric' || c.kind === 'formula').map((c) => (c.text === '*' ? 'all metrics' : c.text));
  return ms.length ? ms.slice(0, 3).join(', ') + (ms.length > 3 ? ` +${ms.length - 3}` : '') : p.q;
}
/** every state change goes through here: optional undo snapshot, mutation, save, re-render */
export function commit(what: string, fn: () => void, opts: { toast?: string } = {}): boolean {
  if (what) M.autoBoard = false; // the viewer shaped the board: stop rebuilding it from conventions
  if (what) M.undo.push(what, M.board);
  try { fn(); } catch (e) { if (what) M.undo.pop(); showToast(String((e as Error).message || e)); return false; }
  save(); sync();
  if (what && opts.toast) showToast(opts.toast, true);
  return true;
}
export function undo(): void {
  const u = M.undo.pop(); if (!u) return;
  M.board = Bd.validBoard(u.board); save(); sync(); showToast(`undid: ${u.what}`);
}
export function setBoard(b: Board, what = 'set board'): void { commit(what, () => { M.board = Bd.validBoard(b); setSelPanel(null); }); }

// ------------------------------------------------------------ presets
const PREFERRED = ['brain_heldout_nll', 'train_loss_ema', 'brain_heldout_acc_pct', 'nat_brain_minus_none', 'brain_indist_nll', 's_per_step'];
const synthetic = (r: Run) => !!(r.config && r.config.synthetic);
let pickedMetrics: () => string[] = () => [];
/** the series browser registers its picked streams so the history preset can use them */
export function registerPicked(fn: () => Stream[] | string[]): void { pickedMetrics = () => [...new Set((fn() as (Stream | string)[]).map((s) => (typeof s === 'string' ? s.slice(s.lastIndexOf(':') + 1) : s.metric)))]; }
export function historyPreset(): Board {
  const t = hash();
  const runs = t.run ? S.runs.filter((r) => r.id === t.run) : t.project ? S.runs.filter((r) => r.id.startsWith(t.project + '/')) : selectedRuns();
  const names = [...metricCounts(runs).keys()];
  let metrics = pickedMetrics().filter((m) => names.includes(m));
  if (!metrics.length) metrics = PREFERRED.filter((k) => names.includes(k)).slice(0, 4);
  if (!metrics.length) metrics = Bd.rankMetrics(names, 4);
  return Bd.presetHistory({ runIds: runs.map((r) => r.id), metrics });
}
export function autoPreset(): Board | null {
  const t = hash();
  const sel = selectedRuns();
  // a single-run page builds from that run only (empty until it arrives, then it follows)
  const inProject = (r: Run) => !!t.project && r.id.startsWith(t.project + '/');
  const runs = t.run ? S.runs.filter((r) => r.id === t.run) : t.project ? S.runs.filter(inProject) : sel.length ? sel : S.runs;
  // a metric logged once per run (a fixed baseline, a final score) is no curve: keep it off the default board
  const counts = new Map([...metricCounts(runs)].filter(([m]) => runs.some((r) => (r.metrics?.[m]?.n ?? 0) >= 2)));
  // demo runs stay out of the default board without pinning it to today's run list
  let filter = S.runs.some(synthetic) && !runs.some(synthetic) ? '{synthetic!=true}' : '';
  if (t.run) filter = `{run=${Bd.quote(t.run)}}`;
  else if (t.project) filter = `{run=~${Bd.quote(t.project + '/*')}}`;
  const tok = [...counts.keys()].find((m) => /tokens?_seen|^tokens$|num_tokens|consumed_tokens/.test(m));
  return Bd.autoBoard(counts, { replicates: hasReplicates(runs), hasTokens: tok || null, filter });
}
export const defaultBoard = (): Board => autoPreset() || historyPreset();
export function applyPreset(name: string): void {
  const sel = selectedRuns();
  const b = name === 'auto' ? defaultBoard() : name === 'history' ? historyPreset() : name === 'all' ? Bd.presetAll([...metricCounts(sel).keys()].sort(), sel.map((r) => r.id)) : { version: Bd.BOARD_VERSION, panels: [] };
  commit(`apply preset ${name}`, () => { M.board = b; setSelPanel(null); }, { toast: `board: ${name}` });
}
export function resetDefault(): void { commit('reset to default', () => { M.board = defaultBoard(); setSelPanel(null); }, { toast: 'board reset to default' }); }

// ------------------------------------------------------------ import / export / shared boards
export function exportBoard(): void {
  const blob = new Blob([Bd.serialize(M.board)], { type: 'application/json' });
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = `obs-board-${new Date().toISOString().slice(0, 10)}.json`;
  document.body.appendChild(a); a.click(); setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
}
export function importBoard(file: File): void {
  file.text().then((txt) => {
    let b: Board; try { b = Bd.validBoard(fromAnyBoard(JSON.parse(txt))); } catch (e) { showToast('import failed: ' + ((e as Error).message || e)); return; }
    commit('import board', () => { M.board = b; setSelPanel(null); }, { toast: `imported ${b.panels.length} panels from ${file.name}` });
  });
}
/** a board file from the MCP make_board tool ({panels: [{query}]}) or an exported board */
export function fromAnyBoard(j: unknown): unknown {
  const b = j as { panels?: { query?: string; q?: string; title?: string }[] };
  if (!b || !Array.isArray(b.panels)) return j;
  return { ...b, panels: b.panels.map((p) => (typeof p.q === 'string' ? p : typeof p.query === 'string' ? { title: p.title || '', q: p.query } : p)) };
}
async function loadShared(name: string): Promise<{ board: Board; title: string; source: string } | null> {
  const tries = (S.api ? [`api/v1/boards/${name}`] : []).concat([`boards/${name}.json`]);
  for (const path of tries) {
    try {
      const j = await getJSON<{ title?: string }>(path);
      const board = Bd.validBoard(fromAnyBoard(j));
      return { board, title: String(j.title || name), source: path };
    } catch { /* next */ }
  }
  return null;
}
export function saveSharedAsMine(): void {
  if (!shared()) return;
  const name = shared()!.name;
  setShared(null);
  M.key = LS_BOARD; M.autoBoard = false;
  save();
  setHash({ board: null });
  showToast(`saved board "${name}" as your board`);
}

// ------------------------------------------------------------ focus view
export function openFocus(id: string): void {
  if (!findPanel(id)) return;
  if (!hash().panel) M.scrollY = window.scrollY;
  setSelPanel(id);
  if (hash().panel !== id) { setHash({ panel: id }, true); M.pushedFocus = true; }
}
export function closeFocus(): void {
  if (!hash().panel) return;
  if (M.pushedFocus) { M.pushedFocus = false; history.back(); }
  else setHash({ panel: null });
  const y = M.scrollY; requestAnimationFrame(() => window.scrollTo(0, y));
}

// ------------------------------------------------------------ init
export async function initBoard(): Promise<void> {
  try { Object.assign(settingsInit, JSON.parse(lsGet(LS_SET) || '{}')); } catch { /* defaults */ }
  setSettings({ ...settingsInit });
  const t = hash();
  M.key = t.run ? `${LS_BOARD}:run=${t.run}` : t.project ? `${LS_BOARD}:project=${t.project}` : LS_BOARD;
  let b: Board | null = null;
  if (t.board) {
    const sb = await loadShared(t.board);
    if (sb) { b = sb.board; setShared({ name: t.board, title: sb.title, source: sb.source }); }
    else err(`could not load board "${t.board}" (tried ${S.api ? 'api/v1/boards/' + t.board + ' and ' : ''}boards/${t.board}.json)`);
  }
  let savedAuto = false;
  if (!b) { try { const s = lsGet(M.key); if (s) { const j = JSON.parse(s); b = Bd.validBoard(j); savedAuto = !!j.auto; } } catch { b = null; } }
  if (!b && M.key === LS_BOARD) { try { const s = lsGet('obs-builder-layout-v1'); if (s) b = Bd.validBoard(JSON.parse(s)); } catch { b = null; } }
  M.autoBoard = !b || savedAuto; // the default tracks incoming runs until the first edit
  if (!b || savedAuto) {
    // rebuild the default from today's data; keep the saved auto board if nothing matches yet
    const d = defaultBoard();
    M.board = d.panels.length || !b ? d : b;
  } else M.board = b;
  sync();
  onRunsChanged(() => {
    // until the viewer edits it, the default board follows the data (e.g. a live run's first metrics)
    if (!M.autoBoard || shared()) return;
    const nb = defaultBoard(), qs = (x: Board) => x.panels.map((p) => p.q).join('\n');
    if (qs(nb) !== qs(M.board)) { M.board = nb; save(); sync(); }
  });
}
const settingsInit: Settings = { palette: true, events: true, spikes: true, linkX: true };

// ------------------------------------------------------------ panel actions
export function panelRuns(p: Panel): Run[] {
  const r = results.get(p.id);
  const ids = r && r.ok ? [...new Set(r.traces.flatMap((t) => t.runs))] : [];
  const runs = ids.map((id) => S.byId.get(id)).filter((x): x is Run => !!x);
  return runs.length ? runs : selectedRuns();
}
export function addToNew(q: string, label?: string): Panel | undefined {
  let p: Panel | undefined;
  commit('', () => { const pq = Bd.parse(q); p = Bd.newPanel(pq.ok ? pq.canonical : q); M.board.panels.push(p); setSelPanel(p.id); });
  if (p) showToast(`new panel: ${label || q}`);
  return p;
}
export function addTo(p: Panel, entries: string, label?: string): void {
  if (commit('', () => { p.q = Bd.addEntries(p.q, entries); setSelPanel(p.id); })) showToast(`overlaid ${label || entries} on "${titleOf(p)}"`);
}
export interface TraceRef { letter: string; run: string; single: boolean; label: string; entries?: string | null }
export function removeTrace(p: Panel, d: TraceRef): void {
  commit(`remove ${d.label}`, () => {
    if (d.run && d.single) p.q = Bd.excludeRun(p.q, d.letter, d.run);
    else p.view.hidden = [...new Set([...p.view.hidden, d.label])];
  }, { toast: `removed ${d.label} from "${titleOf(p)}"` });
}
export function moveTrace(src: Panel, d: TraceRef, dest: Panel | 'new'): void {
  const entries = d.entries; if (!entries) return;
  commit(`move ${d.label}`, () => {
    src.q = Bd.excludeRun(src.q, d.letter, d.run);
    if (dest === 'new') { const p = Bd.newPanel(entries); M.board.panels.push(p); setSelPanel(p.id); }
    else { dest.q = Bd.addEntries(dest.q, entries); setSelPanel(dest.id); }
  }, { toast: `moved ${d.label}` });
}
export function deletePanel(p: Panel): void {
  commit(`delete panel "${titleOf(p)}"`, () => {
    M.board.panels = M.board.panels.filter((q) => q !== p);
    if (selPanel() === p.id) setSelPanel(null);
    if (hash().panel === p.id) closeFocus();
  }, { toast: `deleted panel "${titleOf(p)}"` });
}
export function mergePanel(src: Panel, dst: Panel): void {
  commit(`merge "${titleOf(src)}" into "${titleOf(dst)}"`, () => {
    dst.q = Bd.mergeQueries(dst.q, src.q);
    M.board.panels = M.board.panels.filter((p) => p !== src);
    setSelPanel(dst.id);
  }, { toast: `merged into "${titleOf(dst)}"` });
}
export function movePanel(src: Panel, where: 'before' | 'after' | 'end' | 'up' | 'down', ref?: Panel): void {
  commit('', () => {
    const L = M.board.panels, i = L.indexOf(src);
    if (where === 'up' || where === 'down') { const j = where === 'up' ? i - 1 : i + 1; if (j < 0 || j >= L.length) return; L.splice(i, 1); L.splice(j, 0, src); return; }
    L.splice(i, 1);
    if (where === 'end' || !ref) L.push(src); else { const j = L.indexOf(ref); L.splice(where === 'before' ? j : j + 1, 0, src); }
    setSelPanel(src.id);
  });
}
export function duplicatePanel(p: Panel): void {
  commit('duplicate panel', () => { const c = clone(p); c.id = Bd.uid(); M.board.panels.splice(M.board.panels.indexOf(p) + 1, 0, c); });
}
export function setStage(p: Panel, kind: string, text: string | null, what?: string): void {
  commit(what || (text ? text : `remove ${kind}`), () => { p.q = Bd.setStage(p.q, kind, text); });
}
export function setQuery(p: Panel, q: string, what = 'edit query'): void { if (q !== p.q) commit(what, () => { p.q = q; }); }
export function renamePanel(p: Panel, v: string): void { commit('rename panel', () => { p.title = v === titleOf({ ...p, title: '' }) ? '' : v; }); }
export function toggleFollow(p: Panel): void { commit('', () => { p.view.follow = !p.view.follow; }); }
/** zooming pauses follow without an undo entry */
export function pauseFollow(p: Panel): void { if (p.view.follow) { p.view.follow = false; save(); sync(); } }
export function showAll(p: Panel): void { commit('', () => { p.view.hidden = []; p.view.solo = null; }); }
export function soloTrace(p: Panel, label: string, hide: boolean): void {
  commit('', () => {
    if (hide) { const s = new Set(p.view.hidden); if (s.has(label)) s.delete(label); else s.add(label); p.view.hidden = [...s]; p.view.solo = null; }
    else { p.view.solo = p.view.solo === label ? null : label; p.view.hidden = p.view.hidden.filter((x) => x !== label); }
  });
}
export function toggleY2(p: Panel, label: string): void { commit('', () => { p.view.y2 = p.view.y2 === label ? null : label; }); }
export function setForecastView(p: Panel, patch: Partial<Panel['view']['fc']>, what = ''): void { commit(what, () => { Object.assign(p.view.fc, patch); }); }
export function toggleLiveRefit(p: Panel): void {
  commit('', () => {
    p.view.fc.live = !p.view.fc.live;
    if (!p.view.fc.live) {
      const fr: Record<string, FitForecast> = {};
      for (const [k, v] of fcHist) if (k.startsWith(p.id + '\u0001')) { const f = v.cur; fr[k.slice(p.id.length + 1)] = { ...f, xs: Array.from(f.xs), ys: Array.from(f.ys), lo: Array.from(f.lo), hi: Array.from(f.hi) }; }
      p.view.fc.frozen = fr;
    } else p.view.fc.frozen = null;
  });
}
export const hiddenTrace = (p: { view: Panel['view'] }, t: { label: string }): boolean => p.view.hidden.includes(t.label) || (!!p.view.solo && p.view.solo !== t.label);
/** a target near the current value of the first visible trace */
export function placeTarget(p: Panel, v?: number): void {
  if (v === undefined) {
    const r = results.get(p.id); const t = r && r.ok ? r.traces.find((t) => !hiddenTrace(p, t)) : undefined;
    if (!t) return;
    const fin = Array.from(t.raw).filter(Number.isFinite); const last = fin[fin.length - 1], lo = Math.min(...fin), hi = Math.max(...fin);
    v = +(last - 0.1 * (hi - lo || Math.abs(last) || 1) * (fin[0] > last ? 1 : -1)).toPrecision(3);
  }
  commit('forecast target', () => { p.view.fc.target = v!; });
}
export const fields = (runs: Run[]): string[] => fieldsOf(runs);
