import { createEffect, createMemo, createResource, createSignal, For, Show } from 'solid-js';
import { fmt } from '@/lib/fmt';
import { S, runsVersion, selection, setSelection, eventsOf, fmtConfig } from '@/stores/runs';
import { liveBadge } from '@/stores/live';
import { lineage, parentDraft, setParentDraft } from '@/stores/lineage';
import { openLineage, setRunParent } from '@/stores/board';
import { draggable } from '@/board/dnd';
import { lineageOf, resolveRun } from '@/logic/data';
import { STAGE_NAMES } from '@/logic/board';
import type { Lineage, Run } from '@/logic/types';

type Col = [string, ((r: Run) => string | number) | null];
const COLS: Col[] = [
  ['', null], ['id', (r) => r.id], ['continues from', (r) => lineageOf(r, lineage())?.parent || ''], ['source', (r) => r.source], ['project', (r) => r.project || ''],
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

/** "parent@step (stage)" for the lineage column */
const lineageText = (l: Lineage): string => `${l.parent}${typeof l.step === 'number' ? '@' + fmt(l.step) : ''}${l.stage ? ` (${l.stage})` : ''}`;

/** the runs table: filter, sort, select (feeds presets and the other tabs), lineage, details */
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
      <Show when={parentDraft()} keyed>{(d) => <ParentEditor run={d.run} parent={d.parent} />}</Show>
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
            const lin = () => lineageOf(r, lineage());
            // a run with a parent or with children has a lineage to show
            const inLineage = () => { runsVersion(); const l = lineage(); return !!lin() || S.runs.some((o) => { const p = lineageOf(o, l); return !!p && resolveRun(S.runs, p.parent) === r; }); };
            return (
              <tr data-run={r.id}>
                <td><input type="checkbox" checked={selection().has(r.id)} aria-label={'select run ' + r.id} onChange={(e) => { const on = e.currentTarget.checked; setSelection((s) => { if (on) s.add(r.id); else s.delete(r.id); }); }} /></td>
                <td>
                  <a href={'#run=' + r.id} title={'details of ' + r.id} onClick={(e) => { e.preventDefault(); setDetail(detail() === r.id ? null : r.id); }}>{r.id}</a>
                  <Show when={r.config.synthetic}> <span class="tag">synthetic</span></Show>
                  <Show when={r.local}> <span class="tag" title="parsed in this browser from a dropped file; not uploaded">local</span></Show>
                  <Show when={b().text}><span class="live-badge" classList={{ on: b().on }} title={b().title}>{b().text}</span></Show>
                </td>
                <td class="lin">
                  <Show when={lin()}>{(l) => <span title={lineage()[r.id] ? 'set in this browser (saved with the board)' : 'from the run config'}>{lineageText(l())}{lineage()[r.id] ? ' ' : ''}<Show when={lineage()[r.id]}><span class="tag">board</span></Show> </span>}</Show>
                  <Show when={inLineage()}>
                    <button type="button" class="mini" title={`new panel: the loss of ${r.id} and all its ancestors on one axis`} aria-label={'whole lineage of ' + r.id} onClick={() => openLineage(r.id)}>whole lineage</button>
                  </Show>
                  <button type="button" class="mini" ref={(el) => draggable(el, () => ({ type: 'run', run: r.id, label: r.id }))}
                    title={`set the run ${r.id} continues from; or drag this onto another run's row`} aria-label={'set parent of ' + r.id}
                    onClick={() => setParentDraft({ run: r.id, parent: lin()?.parent ?? null })}>set parent</button>
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

/** "<run> continues from <parent> at step <n>, stage <s>": the browser-side lineage of one run */
function ParentEditor(props: { run: string; parent: string | null }) {
  const cur = () => lineage()[props.run] || null;
  const own = () => { const r = S.byId.get(props.run); return r ? lineageOf(r) : null; };
  const [parent, setParent] = createSignal(props.parent || '');
  const [step, setStep] = createSignal('');
  const [stage, setStage] = createSignal('');
  createEffect(() => {
    // prefill from what the run has now (browser lineage over its config); a dragged parent wins
    const l = cur() || own();
    if (l && (!props.parent || props.parent === l.parent) && typeof l.step === 'number') setStep(String(l.step));
    setStage((l && l.stage) || '');
  });
  const pr = () => S.byId.get(parent());
  const save = () => {
    const p = parent(); if (!p) return;
    const st = step().trim() === '' ? null : Number(step());
    if (st !== null && !Number.isFinite(st)) return;
    setRunParent(props.run, { parent: p, ...(st !== null ? { step: st } : {}), ...(stage() ? { stage: stage() } : {}) });
    setParentDraft(null);
  };
  return (
    <div class="lineage-edit" role="group" aria-label={'set the parent of ' + props.run}>
      <span><b>{props.run}</b> continues from</span>
      <select aria-label="parent run" value={parent()} onChange={(e) => { setParent(e.currentTarget.value); setStep(''); }}>
        <option value="">(choose a run)</option>
        <For each={(runsVersion(), S.runs.filter((r) => r.id !== props.run))}>{(r) => <option value={r.id}>{r.id}</option>}</For>
      </select>
      <label>at step <input type="number" min="0" size="8" aria-label="parent step" placeholder={pr() && typeof pr()!.last_step === 'number' ? `last (${pr()!.last_step})` : 'last'} value={step()} onInput={(e) => setStep(e.currentTarget.value)} /></label>
      <label>stage <select aria-label="stage" value={stage()} onChange={(e) => setStage(e.currentTarget.value)}>
        <option value="">(none)</option>
        <For each={[...new Set([...STAGE_NAMES, ...(stage() ? [stage()] : [])])]}>{(s) => <option value={s}>{s}</option>}</For>
      </select></label>
      <button type="button" class="mini" disabled={!parent()} title="save: stored in the board (exported with it), not on the server" aria-label="save the parent" onClick={save}>save</button>
      <Show when={cur()}><button type="button" class="mini" title="remove the parent set in this browser" aria-label="clear the parent set in this browser" onClick={() => { setRunParent(props.run, null); setParentDraft(null); }}>clear</button></Show>
      <button type="button" class="mini" title="close without changes" aria-label="cancel" onClick={() => setParentDraft(null)}>cancel</button>
      <Show when={own() && !cur()}><span class="note">its config says {lineageText(own()!)}; saving overrides that on this board</span></Show>
    </div>
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
