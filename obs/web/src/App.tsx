import { createSignal, For, Match, onMount, Show, Switch } from 'solid-js';
import { A, initWasm } from '@/lib/wasm';
import { initTheme } from '@/lib/theme';
import { hash, onHashChange, targetKey } from '@/lib/hash';
import { S, loadIndex, indexURL, setSelection, errors, err, dismissErrors, upsertRuns } from '@/stores/runs';
import { prepareLive, attachLive } from '@/stores/live';
import { restoreDropped } from '@/stores/drop';
import {
  initBoard, focusId, settings, undo, findPanel, deletePanel, selPanel, setSelPanel, setStage, addTo, addToNew, toggleFollow, toggleEditing,
  openFocus, closeFocus, applyPreset, exportBoard, setInspector, setBoard, M,
} from '@/stores/board';
import { setPaletteOpen, paletteOpen, showToast } from '@/stores/ui';
import { metricCounts, fieldsOf } from '@/logic/data';
import { parse } from '@/logic/board';
import type { Action } from '@/logic/commands';
import { setHash } from '@/lib/hash';
import { wireDnd } from '@/board/dnd';
import Header, { currentTab } from '@/components/Header';
import { DropZone, DropReport } from '@/components/Drop';
import Waiting from '@/components/Waiting';
import { Popover, ToastView, Palette } from '@/components/Overlays';
import { pickedEntries, pickedCount } from '@/components/SeriesBrowser';
import BoardView, { openImport } from '@/views/BoardView';
import FocusView from '@/views/FocusView';
import CrossView from '@/views/CrossView';
import ArraysView from '@/views/ArraysView';
import RunsView from '@/views/RunsView';
import SamplesView from '@/views/SamplesView';
import EvalsView from '@/views/EvalsView';
import { anomaliesOpen, setAnomaliesOpen } from '@/components/Anomalies';

/** palette actions: the same operations as the chips, menus and inspector */
function runAction(a: Action): void {
  const p = findPanel(focusId()) || findPanel(selPanel());
  const needP = () => { if (!p) { showToast('select a panel first'); return false; } return true; };
  switch (a.type) {
    case 'stage': if (needP()) setStage(p!, a.kind, a.text); break;
    case 'add': if (p) addTo(p, a.entries, a.entries); else addToNew(a.entries, a.entries); break;
    case 'new': {
      if (a.q) { const r = parse(a.q); if (!r.ok) { showToast(`query error: ${r.error}`); break; } addToNew(r.canonical); }
      else if (pickedCount()) addToNew(pickedEntries(), `${pickedCount()} series`);
      else showToast('pick series in the list first, or type "new panel <query>"');
      break;
    }
    case 'follow': if (needP()) toggleFollow(p!); break;
    case 'editText': if (needP()) toggleEditing(p!.id, true); break;
    case 'inspect': if (needP()) { setSelPanel(p!.id); setInspector(true); } break;
    case 'expand': if (needP()) openFocus(p!.id); break;
    case 'back': closeFocus(); break;
    case 'undo': undo(); break;
    case 'preset': applyPreset(a.name); break;
    case 'export': exportBoard(); break;
    case 'import': openImport(); break;
    case 'view': setHash({ view: a.name === 'board' ? null : a.name, panel: null }); break;
    case 'anomalies': setAnomaliesOpen(!anomaliesOpen()); if (anomaliesOpen()) setHash({ view: null, panel: null }); break;
  }
}

function keys(): void {
  // Ctrl/Cmd+K palette (opt-out), Ctrl/Cmd+Z inside the board, Delete on a focused panel. No single-letter keys.
  document.addEventListener('keydown', (ev) => {
    const a = document.activeElement as HTMLElement | null;
    const typing = !!a && (a.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(a.tagName));
    if ((ev.ctrlKey || ev.metaKey) && !ev.shiftKey && !ev.altKey && ev.key.toLowerCase() === 'k' && settings.palette) { setPaletteOpen(!paletteOpen()); ev.preventDefault(); return; }
    const inBoard = !!a && !!a.closest && !!a.closest('[data-board-scope]');
    if (!inBoard || typing) return;
    if ((ev.ctrlKey || ev.metaKey) && !ev.shiftKey && ev.key.toLowerCase() === 'z') { undo(); ev.preventDefault(); return; }
    if (ev.key === 'Delete' && !ev.ctrlKey && !ev.metaKey) { const pe = a!.closest<HTMLElement>('.bpanel'); const p = pe && findPanel(pe.dataset.pid); if (p) { deletePanel(p); ev.preventDefault(); } }
  });
}

export default function App() {
  const [ready, setReady] = createSignal(false);
  onMount(async () => {
    initTheme();
    try { await initWasm(); } catch (e) { err('could not load the analysis wasm (' + e + '). Build it with scripts/build_obs_wasm.sh, then rebuild obs/web.'); return; }
    await prepareLive();
    const t = hash();
    await loadIndex(indexURL(t.project, t.run));
    if (t.run) { const id = t.run; setSelection((s) => { s.clear(); if (S.byId.has(id)) s.add(id); }); }
    await initBoard();
    wireDnd(); keys();
    restoreDropped();
    if (S.api) attachLive();
    // a different data target (run / project / board) is a different page: reload
    onHashChange((prev, next) => { if (targetKey(prev) !== targetKey(next) || next.isNew) location.reload(); });
    // small API for other modules and tests
    const w = window as unknown as Record<string, unknown>;
    w.obsAddRuns = upsertRuns;
    w.OBS = { S, A, upsertRuns, builder: { board: () => M.board, setBoard, addPanel: (q: string) => addToNew(q), openFocus, closeFocus, runAction } };
    setReady(true);
  });
  const ctx = () => ({ metrics: [...metricCounts(S.runs).keys()].sort(), fields: fieldsOf(S.runs) });
  return (
    <>
      <Header />
      <Show when={errors().length}>
        <div class="errs" role="alert">
          <For each={errors()}>{(e) => <div class="err">{e}</div>}</For>
          <button type="button" title="hide these messages" aria-label="dismiss errors" onClick={dismissErrors}>dismiss</button>
        </div>
      </Show>
      <DropReport />
      <Waiting />
      <main>
        <Show when={ready()} fallback={<div class="note">loading…</div>}>
          <Switch>
            <Match when={currentTab() === 'board'}><BoardView /></Match>
            <Match when={currentTab() === 'samples'}><SamplesView /></Match>
            <Match when={currentTab() === 'evals'}><EvalsView /></Match>
            <Match when={currentTab() === 'cross'}><CrossView /></Match>
            <Match when={currentTab() === 'arrays'}><ArraysView /></Match>
            <Match when={currentTab() === 'runs'}><RunsView /></Match>
          </Switch>
        </Show>
      </main>
      <Show when={ready() && focusId()}><FocusView /></Show>
      <Popover />
      <ToastView />
      <Palette ctx={ctx} run={runAction} />
      <DropZone />
    </>
  );
}
