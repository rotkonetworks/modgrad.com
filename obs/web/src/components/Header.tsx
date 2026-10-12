import { For, Show } from 'solid-js';
import { genMeta, S, reloadIndex, indexURL } from '@/stores/runs';
import { liveStatus } from '@/stores/live';
import { localCount, ingest, forgetLocal } from '@/stores/drop';
import { themeMode, cycleTheme } from '@/lib/theme';
import { hash, setHash } from '@/lib/hash';
import { TABS, openMenu, openPopover, setPaletteOpen, type Tab, type MenuItem } from '@/stores/ui';
import { resetDefault, exportBoard, inspector, setInspector } from '@/stores/board';
import { openImport, SettingsPop } from '@/views/BoardView';
import { anomaliesOpen, setAnomaliesOpen } from '@/components/Anomalies';
import { accountId, accountLabel, available, goAccount, goLogin, portalView, signOut } from '@/stores/account';

export const currentTab = (): Tab => { const v = hash().view; return (TABS.some(([t]) => t === v) ? v : 'board') as Tab; };

export default function Header() {
  let files!: HTMLInputElement, dir!: HTMLInputElement;
  const pick = (inp: HTMLInputElement) => { const list = [...(inp.files || [])].map((f) => ({ rel: f.webkitRelativePath || f.name, file: f })); inp.value = ''; ingest(list); };
  // narrow screens: the header's and the board toolbar's secondary actions in one labelled menu
  const more = (anchor: HTMLElement) => {
    const items: MenuItem[] = [
      { label: 'open log files…', title: 'parse log files in this browser (nothing is uploaded)', run: () => files.click() },
      { label: 'open a log folder…', title: 'a run directory, a TensorBoard logdir, ...', run: () => dir.click() },
      ...(localCount() ? [{ label: `forget local runs (${localCount()})`, run: forgetLocal }] : []),
      { label: 'reload the index and data', run: () => reloadIndex(indexURL(hash().project, hash().run)) },
      { label: `theme: ${themeMode()} (cycle)`, run: cycleTheme },
    ];
    if (currentTab() === 'board') items.push(
      { label: 'board: reset to default', run: resetDefault },
      { label: 'board: export JSON', run: exportBoard },
      { label: 'board: import JSON…', run: openImport },
      { label: 'commands (Ctrl+K)', run: () => setPaletteOpen(true) },
      { label: 'builder settings', run: () => openPopover(anchor, () => <SettingsPop />) },
      { label: inspector() ? 'hide the inspector' : 'show the inspector', run: () => setInspector(!inspector()) },
      { label: anomaliesOpen() ? 'hide anomalies' : 'show anomalies (spikes, NaN)', run: () => setAnomaliesOpen(!anomaliesOpen()) },
    );
    openMenu(anchor, items, 'more actions');
  };
  return (
    <header class="top">
      <h1>modgrad</h1>
      <Show when={!portalView()}><span class="meta wideonly">{genMeta()}</span></Show>
      <Show when={liveStatus()}><span class="meta live-status">{liveStatus()}</span></Show>
      <span class="sp" />
      <button type="button" class="narrowonly" title="more actions: open files, reload, theme, board tools" aria-label="more actions" aria-haspopup="menu" onClick={(e) => more(e.currentTarget)}>more</button>
      <span class="droptools wideonly">
        <span class="hint" title="CSV/TSV, JSONL, text logs, TensorBoard event files, W&B .wandb, modgrad-obs run folders. Parsed in your browser; nothing is uploaded.">drop logs anywhere, or</span>
        <button type="button" title="parse log files in this browser (nothing is uploaded)" aria-label="open log files" onClick={() => files.click()}>open files</button>
        <button type="button" title="open a folder (a run directory, a TensorBoard logdir, ...)" aria-label="open a log folder" onClick={() => dir.click()}>open folder</button>
        <Show when={localCount()}>
          <button type="button" title="remove the local runs from this page and from this browser's storage" aria-label="forget local runs" onClick={forgetLocal}>forget local runs ({localCount()})</button>
        </Show>
        <input ref={files} type="file" multiple hidden aria-label="log files" onChange={() => pick(files)} />
        <input ref={dir} type="file" multiple hidden aria-label="log folder" {...{ webkitdirectory: '' }} onChange={() => pick(dir)} />
      </span>
      <button type="button" class="wideonly" title="re-fetch the index and data" aria-label="reload the index and data" onClick={() => reloadIndex(indexURL(hash().project, hash().run))}>reload</button>
      <button type="button" class="wideonly" title="cycle theme: auto / light / dark" aria-label={'theme: ' + themeMode() + ', click to cycle'} onClick={cycleTheme}>theme: {themeMode()}</button>
      <AccountButton />
      <nav class="tabs" aria-label="views">
        <For each={TABS}>{([t, label]) => (
          <a href={'#view=' + t} classList={{ on: !portalView() && currentTab() === t }} aria-current={!portalView() && currentTab() === t ? 'page' : undefined}
            onClick={(e) => { e.preventDefault(); setHash({ view: t === 'board' ? null : t }); }}>{label}</a>
        )}</For>
      </nav>
    </header>
  );
}

/** "sign in", or the account menu; nothing when the server has no accounts (static mode) */
function AccountButton() {
  const short = (id: string) => (id.length > 12 ? id.slice(0, 10) + '…' : id);
  const menu = (anchor: HTMLElement) => openMenu(anchor, [
    { label: 'account and credentials', title: 'credentials, projects, API tokens, sessions', run: goAccount },
    { label: 'sign out', title: 'end this session', run: () => void signOut() },
  ], 'account');
  return (
    <Show when={available()}>
      <Show when={accountId()} fallback={
        <button type="button" classList={{ acctbtn: true, on: portalView() === 'login' }} title="sign in with email, passkey, zafu or an SSH key" aria-label="sign in" onClick={goLogin}>sign in</button>
      }>{(id) => (
        <button type="button" classList={{ acctbtn: true, on: portalView() === 'account' }} title={'signed in as ' + id() + ': account page, sign out'} aria-label="account menu" aria-haspopup="menu" onClick={(e) => menu(e.currentTarget)}>
          <span class="mono">{accountLabel() ?? short(id())}</span>
        </button>
      )}</Show>
    </Show>
  );
}
