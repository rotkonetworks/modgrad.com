import { For, Show } from 'solid-js';
import { genMeta, S, reloadIndex, indexURL } from '@/stores/runs';
import { liveStatus } from '@/stores/live';
import { localCount, ingest, forgetLocal } from '@/stores/drop';
import { themeMode, cycleTheme } from '@/lib/theme';
import { hash, setHash } from '@/lib/hash';
import { TABS, type Tab } from '@/stores/ui';

export const currentTab = (): Tab => { const v = hash().view; return (TABS.some(([t]) => t === v) ? v : 'board') as Tab; };

export default function Header() {
  let files!: HTMLInputElement, dir!: HTMLInputElement;
  const pick = (inp: HTMLInputElement) => { const list = [...(inp.files || [])].map((f) => ({ rel: f.webkitRelativePath || f.name, file: f })); inp.value = ''; ingest(list); };
  return (
    <header class="top">
      <h1>modgrad obs</h1>
      <span class="meta">{genMeta()}</span>
      <Show when={liveStatus()}><span class="meta live-status">{liveStatus()}</span></Show>
      <span class="sp" />
      <span class="droptools">
        <span class="hint" title="CSV/TSV, JSONL, text logs, TensorBoard event files, W&B .wandb, modgrad-obs run folders. Parsed in your browser; nothing is uploaded.">drop logs anywhere, or</span>
        <button type="button" title="parse log files in this browser (nothing is uploaded)" aria-label="open log files" onClick={() => files.click()}>open files</button>
        <button type="button" title="open a folder (a run directory, a TensorBoard logdir, ...)" aria-label="open a log folder" onClick={() => dir.click()}>open folder</button>
        <Show when={localCount()}>
          <button type="button" title="remove the local runs from this page and from this browser's storage" aria-label="forget local runs" onClick={forgetLocal}>forget local runs ({localCount()})</button>
        </Show>
        <input ref={files} type="file" multiple hidden aria-label="log files" onChange={() => pick(files)} />
        <input ref={dir} type="file" multiple hidden aria-label="log folder" {...{ webkitdirectory: '' }} onChange={() => pick(dir)} />
      </span>
      <button type="button" title="re-fetch the index and data" aria-label="reload the index and data" onClick={() => reloadIndex(indexURL(hash().project, hash().run))}>reload</button>
      <button type="button" title="cycle theme: auto / light / dark" aria-label={'theme: ' + themeMode() + ', click to cycle'} onClick={cycleTheme}>theme: {themeMode()}</button>
      <nav class="tabs" aria-label="views">
        <For each={TABS}>{([t, label]) => (
          <a href={'#view=' + t} classList={{ on: currentTab() === t }} aria-current={currentTab() === t ? 'page' : undefined}
            onClick={(e) => { e.preventDefault(); setHash({ view: t === 'board' ? null : t }); }}>{label}</a>
        )}</For>
      </nav>
    </header>
  );
}
