import { createMemo, createSignal, For, Show } from 'solid-js';
import { S, runsVersion, selection, setSelection } from '@/stores/runs';
import { liveBadge } from '@/stores/live';
import { setHash } from '@/lib/hash';
import SeriesBrowser from '@/components/SeriesBrowser';

const [railTab, setRailTab] = createSignal<'series' | 'runs'>('series');

/** the left rail: series browser, and a compact run picker (the full table is the Runs tab) */
export default function Rail(props: { onCollapse?: () => void }) {
  return (
    <aside class="bside" aria-label="series and runs">
      <div class="rtabs" role="tablist" aria-label="rail">
        <button type="button" role="tab" aria-selected={railTab() === 'series'} classList={{ on: railTab() === 'series' }} title="series browser: drag or + to build panels" aria-label="series browser" onClick={() => setRailTab('series')}>series</button>
        <button type="button" role="tab" aria-selected={railTab() === 'runs'} classList={{ on: railTab() === 'runs' }} title="select runs" aria-label="run selection" onClick={() => setRailTab('runs')}>runs ({selection().size}/{(runsVersion(), S.runs.length)})</button>
        <span class="sp" />
        <Show when={props.onCollapse}>
          <button type="button" title="hide the rail" aria-label="hide the series and runs rail" onClick={() => props.onCollapse!()}>{'<<'}</button>
        </Show>
      </div>
      <Show when={railTab() === 'series'} fallback={<RunPicker />}>
        <SeriesBrowser />
      </Show>
    </aside>
  );
}

function RunPicker() {
  const [filter, setFilter] = createSignal('');
  const rows = createMemo(() => {
    runsVersion();
    const ws = filter().toLowerCase().split(/\s+/).filter(Boolean);
    return S.runs.filter((r) => { const hay = (r.id + ' ' + r.source + ' ' + (r.project || '') + ' ' + JSON.stringify(r.config)).toLowerCase(); return ws.every((w) => hay.includes(w)); });
  });
  return (
    <div class="rpick">
      <div class="bhead">
        <input type="search" placeholder="filter runs: name, source, config" aria-label="filter runs" value={filter()} onInput={(e) => setFilter(e.currentTarget.value)} />
        <div class="brow">
          <button type="button" title="select the runs shown" aria-label="select the runs shown" onClick={() => setSelection((s) => { for (const r of rows()) s.add(r.id); })}>select shown</button>
          <button type="button" title="clear the run selection" aria-label="clear the run selection" onClick={() => setSelection((s) => s.clear())}>clear</button>
          <a href="#view=runs" title="the full runs table with config and summary" onClick={(e) => { e.preventDefault(); setHash({ view: 'runs' }); }}>full table</a>
        </div>
      </div>
      <div class="blist">
        <For each={rows()}>{(r) => {
          const b = () => liveBadge(r.id);
          return (
            <label class="ritem" title={r.id}>
              <input type="checkbox" checked={selection().has(r.id)} aria-label={'select run ' + r.id} onChange={(e) => { const on = e.currentTarget.checked; setSelection((s) => { if (on) s.add(r.id); else s.delete(r.id); }); }} />
              <span class="lab">{r.id}</span>
              <Show when={r.config.synthetic}><span class="tag">synthetic</span></Show>
              <Show when={r.local}><span class="tag">local</span></Show>
              <Show when={b().text}><span class="live-badge" classList={{ on: b().on }} title={b().title}>{b().text}</span></Show>
            </label>
          );
        }}</For>
      </div>
      <div class="bfoot bnote">Selection feeds presets (history, auto, all) and the other tabs; panel queries stay as written.</div>
    </div>
  );
}
