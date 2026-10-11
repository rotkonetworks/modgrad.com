import { createMemo, createResource, createSignal, For, Show } from 'solid-js';
import { fmt } from '@/lib/fmt';
import { S, runsVersion, selection, setSelection, eventsOf, fmtConfig } from '@/stores/runs';
import { liveBadge } from '@/stores/live';
import type { Run } from '@/logic/types';

type Col = [string, ((r: Run) => string | number) | null];
const COLS: Col[] = [
  ['', null], ['id', (r) => r.id], ['source', (r) => r.source], ['project', (r) => r.project || ''],
  ['last step', (r) => r.last_step ?? -1], ['brain params', (r) => (typeof r.config.brain_params === 'number' ? r.config.brain_params : NaN)],
  ['metrics', (r) => Object.keys(r.metrics).length], ['arrays', (r) => Object.keys(r.arrays).length],
  ['state', (r) => (r.finished ? 'finished' : 'running/died')], ['unparsed', (r) => r.unparsed.length],
];
export function matches(r: Run, q: string): boolean {
  if (!q) return true;
  const hay = (r.id + ' ' + r.source + ' ' + (r.project || '') + ' ' + JSON.stringify(r.config)).toLowerCase();
  return q.toLowerCase().split(/\s+/).every((w) => hay.includes(w));
}

const [filter, setFilter] = createSignal('');
const [sort, setSort] = createSignal<{ k: string; dir: number }>({ k: 'id', dir: 1 });
const [detail, setDetail] = createSignal<string | null>(null);

/** the runs table: filter, sort, select (feeds presets and the other tabs), details */
export default function RunsView() {
  const rows = createMemo(() => {
    runsVersion(); selection();
    const s = sort(), col = COLS.find((c) => c[0] === s.k) || COLS[1];
    const f = col[1]!;
    return S.runs.filter((r) => matches(r, filter())).sort((a, b) => { const x = f(a), y = f(b); return (x > y ? 1 : x < y ? -1 : 0) * s.dir; });
  });
  return (
    <section class="runsview">
      <div class="controls">
        <label>filter <input placeholder="name, source, config" size="24" aria-label="filter runs" value={filter()} onInput={(e) => setFilter(e.currentTarget.value)} /></label>
        <button type="button" title="select the runs shown" aria-label="select the runs shown" onClick={() => setSelection((s) => { for (const r of rows()) s.add(r.id); })}>select shown</button>
        <button type="button" title="clear the selection" aria-label="clear the run selection" onClick={() => setSelection((s) => s.clear())}>clear</button>
        <span class="note">{selection().size} of {(runsVersion(), S.runs.length)} selected, {rows().length} shown</span>
      </div>
      <div class="tablewrap"><table class="runs">
        <thead><tr>
          <For each={COLS}>{([name, f]) => (
            <th aria-sort={sort().k === name ? (sort().dir > 0 ? 'ascending' : 'descending') : undefined}>
              <Show when={f} fallback={name}>
                <button type="button" class="th" title={'sort by ' + name} aria-label={'sort by ' + name} onClick={() => setSort((s) => ({ k: name, dir: s.k === name ? -s.dir : 1 }))}>{name + (sort().k === name ? (sort().dir > 0 ? ' +' : ' -') : '')}</button>
              </Show>
            </th>
          )}</For>
        </tr></thead>
        <tbody>
          <For each={rows()}>{(r) => {
            const b = () => liveBadge(r.id);
            return (
              <tr data-run={r.id}>
                <td><input type="checkbox" checked={selection().has(r.id)} aria-label={'select run ' + r.id} onChange={(e) => { const on = e.currentTarget.checked; setSelection((s) => { if (on) s.add(r.id); else s.delete(r.id); }); }} /></td>
                <td>
                  <a href={'#run=' + r.id} title={'details of ' + r.id} onClick={(e) => { e.preventDefault(); setDetail(detail() === r.id ? null : r.id); }}>{r.id}</a>
                  <Show when={r.config.synthetic}> <span class="tag">synthetic</span></Show>
                  <Show when={r.local}> <span class="tag" title="parsed in this browser from a dropped file; not uploaded">local</span></Show>
                  <Show when={b().text}><span class="live-badge" classList={{ on: b().on }} title={b().title}>{b().text}</span></Show>
                </td>
                <td>{r.source}</td><td>{r.project || ''}</td>
                <td class="num">{(runsVersion(), r.last_step ?? '-')}</td>
                <td class="num">{typeof r.config.brain_params === 'number' ? fmt(r.config.brain_params) : '-'}</td>
                <td class="num">{(runsVersion(), Object.keys(r.metrics).length)}</td>
                <td class="num">{Object.keys(r.arrays).length}</td>
                <td>{(runsVersion(), r.finished ? 'finished' : 'running/died')}</td>
                <td class="num">{r.unparsed.length || ''}</td>
              </tr>
            );
          }}</For>
        </tbody>
      </table></div>
      <Show when={detail() ? S.byId.get(detail()!) : undefined} keyed>{(r) => <Details run={r} />}</Show>
    </section>
  );
}

function Details(props: { run: Run }) {
  const r = props.run;
  const [ev] = createResource(() => r.id, (id) => eventsOf(id));
  const meta = r.meta || {};
  return (
    <details class="run" open>
      <summary>{`${r.id}  (step ${r.last_step ?? '-'}, ${r.n_rows ?? 0} rows, ${r.n_events ?? 0} events${r.skipped_lines ? ', ' + r.skipped_lines + ' skipped lines' : ''})`}</summary>
      <pre>{`git ${meta.git || '-'}  host ${meta.hostname || '-'}  start ${meta.start_unix ? new Date(meta.start_unix * 1000).toISOString().replace('.000', '') : '-'}${meta.legacy_log ? '\nimported from ' + meta.legacy_log : ''}`}</pre>
      <div class="dh">config</div>
      <div class="kv"><For each={Object.entries(r.config)}>{([k, v]) => <div><span>{k}</span><span>{fmtConfig(v)}</span></div>}</For></div>
      <div class="dh">summary</div>
      <div class="kv"><For each={Object.entries(r.summary)}>{([k, v]) => <div><span>{k}</span><span>{fmtConfig(v)}</span></div>}</For></div>
      <Show when={r.unparsed.length}><div class="dh">unparsed log lines</div><pre>{r.unparsed.join('\n')}</pre></Show>
      <Show when={ev() && ev()!.length}><div class="dh">events</div><pre>{ev()!.map((e) => `step ${e.step}: ${e.text}`).join('\n')}</pre></Show>
    </details>
  );
}
