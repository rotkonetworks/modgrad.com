import { For, Show } from 'solid-js';
import {
  board, undo, undoTop, applyPreset, resetDefault, exportBoard, importBoard, inspector, setInspector, focusId, titleOf, settings, setSetting,
  shared, saveSharedAsMine,
} from '@/stores/board';
import { openPopover, setPaletteOpen, railOpen, setRailOpen } from '@/stores/ui';
import Panel from '@/components/Panel';
import Rail from '@/components/Rail';
import Inspector from '@/components/Inspector';
import Anomalies, { anomaliesOpen, setAnomaliesOpen } from '@/components/Anomalies';

let fileInput!: HTMLInputElement;
export const openImport = (): void => fileInput?.click();

export function SettingsPop() {
  const row = (k: 'palette' | 'events' | 'spikes' | 'linkX', label: string, title: string) => (
    <label class="row" title={title}><input type="checkbox" checked={settings[k]} onChange={(e) => setSetting(k, e.currentTarget.checked)} /> {label}</label>
  );
  return (
    <>
      <h3>settings</h3>
      {row('palette', 'Ctrl+K command palette', 'the only global shortcut; off leaves Ctrl+K to the browser')}
      {row('linkX', 'link x zoom across panels', 'panels with the same x axis share their zoom; off = each panel zooms alone')}
      {row('events', 'event markers', "vertical markers from the runs' event logs (step axis)")}
      {row('spikes', 'loss-spike markers', 'upward spikes detected in wasm (> 6 robust sd above the trailing median)')}
    </>
  );
}

export function BoardToolbar() {
  return (
    <div class="btool">
      <Show when={!railOpen()}>
        <button type="button" title="show the series and runs rail" aria-label="show the series and runs rail" onClick={() => setRailOpen(true)}>series &gt;&gt;</button>
      </Show>
      <label title="replace the board with a preset (undoable)">preset <select aria-label="board presets" onChange={(e) => { const v = e.currentTarget.value; if (v) { applyPreset(v); e.currentTarget.value = ''; } }}>
        <option value="">...</option>
        <option value="auto">auto (naming conventions)</option>
        <option value="history">history (selected runs)</option>
        <option value="all">all metrics</option>
        <option value="blank">blank</option>
      </select></label>
      <button type="button" title="rebuild the default board (auto, else history of the selected runs); undoable" aria-label="reset the board to default" onClick={resetDefault}>reset to default</button>
      <button type="button" disabled={!undoTop()} title={undoTop() ? `undo: ${undoTop()} (ctrl+z inside the board)` : 'nothing to undo'} aria-label={undoTop() ? `undo ${undoTop()}` : 'nothing to undo'} onClick={undo}>undo</button>
      <button type="button" title="download this board as JSON" aria-label="export the board as JSON" onClick={exportBoard}>export</button>
      <button type="button" title="load a board JSON file" aria-label="import a board JSON file" onClick={openImport}>import</button>
      <input ref={fileInput} type="file" accept=".json,application/json" style={{ display: 'none' }} aria-hidden="true" tabindex="-1"
        onChange={(e) => { const f = e.currentTarget.files && e.currentTarget.files[0]; if (f) importBoard(f); e.currentTarget.value = ''; }} />
      <button type="button" title="command palette (Ctrl+K)" aria-label="open the command palette" onClick={() => setPaletteOpen(true)}>commands</button>
      <button type="button" title="builder settings" aria-label="builder settings" onClick={(e) => openPopover(e.currentTarget, () => <SettingsPop />)}>settings</button>
      <button type="button" classList={{ on: inspector() }} aria-pressed={inspector()} title="panel settings for the selected panel" aria-label="toggle the inspector" onClick={() => setInspector(!inspector())}>inspector</button>
      <button type="button" classList={{ on: anomaliesOpen() }} aria-pressed={anomaliesOpen()} title="loss / grad-norm spikes and NaN/inf in the selected runs, with jump-to-step" aria-label="toggle the anomaly list" onClick={() => setAnomaliesOpen(!anomaliesOpen())}>anomalies</button>
      <span class="msg">{board.panels.length} panel{board.panels.length === 1 ? '' : 's'}</span>
    </div>
  );
}

export function SharedBanner() {
  return (
    <Show when={shared()}>{(s) => (
      <div class="shared" role="status">
        <span>Shared board <b>{s().name}</b>{s().title !== s().name ? `: ${s().title}` : ''} (from {s().source}). Edits are kept on this page only.</span>
        <button type="button" title="keep this board as your board in this browser" aria-label="save this shared board as my board" onClick={saveSharedAsMine}>save as my board</button>
      </div>
    )}</Show>
  );
}

export default function BoardView() {
  return (
    <div class="bview" classList={{ 'rail-off': !railOpen(), 'insp-on': inspector() }} data-board-scope>
      <Show when={railOpen()}><Rail onCollapse={() => setRailOpen(false)} /></Show>
      <div class="bmain">
        <SharedBanner />
        <BoardToolbar />
        <Anomalies />
        <div class="bcanvas" data-canvas="board" aria-label="board">
          <For each={board.panels}>{(p) => (
            <Show when={focusId() !== p.id} fallback={<div class="bnew">"{titleOf(p)}" is open in the focus view</div>}>
              <Panel panel={p} />
            </Show>
          )}</For>
          <div class="bnew">{board.panels.length ? 'drop series here for a new panel' : 'Drag series from the list onto this area to make a panel (or use their + buttons); drop more onto a panel to overlay them. Presets above build a whole board.'}</div>
        </div>
      </div>
      <Show when={inspector()}><Inspector /></Show>
    </div>
  );
}
