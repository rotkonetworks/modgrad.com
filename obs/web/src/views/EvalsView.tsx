import { createMemo, createResource, createSignal, For, Show } from 'solid-js';
import { fmt, fmtStep } from '@/lib/fmt';
import { PALETTE } from '@/lib/color';
import { setHash } from '@/lib/hash';
import { S, selectedRuns, runsVersion, runVersion, metricSeries, catalogVersion, type SeriesData } from '@/stores/runs';
import { addToNew } from '@/stores/board';
import { openMenu, showToast } from '@/stores/ui';
import { quote, runsMatcher } from '@/logic/board';
import { evalColumns, evalRows, bestRows, sortRows, type EvalCol } from '@/logic/evals';
import { jumpToStep } from '@/components/Anomalies';

const [sortBy, setSortBy] = createSignal<{ by: number | 'run' | 'step'; dir: 1 | -1 }>({ by: 'step', dir: 1 });

function Spark(props: { col: EvalCol; runs: string[]; data: Map<string, SeriesData> }) {
  const W = 120, H = 26, P = 2;
  const lines = createMemo(() => {
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    const ss = props.runs.map((r) => props.data.get(r + '\u0001' + props.col.name)).map((s) => s || { steps: [], values: [] });
    for (const s of ss) for (let i = 0; i < s.steps.length; i++) if (Number.isFinite(s.values[i])) { x0 = Math.min(x0, s.steps[i]); x1 = Math.max(x1, s.steps[i]); y0 = Math.min(y0, s.values[i]); y1 = Math.max(y1, s.values[i]); }
    if (!Number.isFinite(x0)) return [];
    const X = (x: number) => P + ((x - x0) / (x1 - x0 || 1)) * (W - 2 * P), Y = (y: number) => H - P - ((y - y0) / (y1 - y0 || 1)) * (H - 2 * P);
    return ss.map((s, k) => {
      const pts: string[] = [];
      for (let i = 0; i < s.steps.length; i++) if (Number.isFinite(s.values[i])) pts.push(`${X(s.steps[i]).toFixed(1)},${Y(s.values[i]).toFixed(1)}`);
      return { pts: pts.join(' '), color: PALETTE[k % PALETTE.length] };
    });
  });
  return (
    <svg class="spark" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${props.col.name} over steps`}>
      <For each={lines()}>{(l) => <polyline points={l.pts} style={{ fill: 'none', stroke: `var(${l.color})`, 'stroke-width': 1.2 }} />}</For>
    </svg>
  );
}

/** Evals tab: checkpoints x benchmarks for metrics named eval/<benchmark>/<metric> */
export default function EvalsView() {
  // the selected runs with evals; when none of them has any (e.g. the selection leaves out the demo
  // runs that do), every run with evals, said so above the table rather than an empty tab
  const hasEval = (r: { metrics: Record<string, unknown> }) => Object.keys(r.metrics).some((m) => m.startsWith('eval/'));
  const scope = createMemo((): 'selected' | 'every' | 'unselected' => { runsVersion(); catalogVersion(); const sel = selectedRuns(); return !sel.length ? 'every' : sel.some(hasEval) ? 'selected' : 'unselected'; });
  const runs = createMemo(() => { runsVersion(); catalogVersion(); return (scope() === 'selected' ? selectedRuns() : S.runs).filter(hasEval); });
  const cols = createMemo(() => evalColumns(runs().flatMap((r) => Object.keys(r.metrics))));
  const runIds = () => runs().map((r) => r.id);
  const [data] = createResource(() => ({ ids: runIds(), cols: cols().map((c) => c.name), v: runIds().map((r) => runVersion(r)).join(',') }), async ({ ids, cols }) => {
    const m = new Map<string, SeriesData>();
    await Promise.all(ids.flatMap((r) => cols.map(async (c) => {
      const run = S.byId.get(r); if (!run || !run.metrics[c]) return;
      try { const s = await metricSeries(run, c); if (s) m.set(r + '\u0001' + c, s); } catch { /* shown as missing */ }
    })));
    return m;
  });
  const rows = createMemo(() => { const d = data(); return d ? evalRows(runIds(), cols(), (r, c) => d.get(r + '\u0001' + c)) : []; });
  const best = createMemo(() => bestRows(rows(), cols()));
  const sorted = createMemo(() => { const s = sortBy(); return sortRows(rows(), typeof s.by === 'number' && s.by >= cols().length ? 'step' : s.by, s.dir); });
  const bestSet = createMemo(() => new Set(best().map((b, ci) => (b >= 0 ? rows()[b].run + '\u0001' + rows()[b].step + '\u0001' + ci : ''))));
  const sortOn = (by: number | 'run' | 'step') => { const s = sortBy(); setSortBy({ by, dir: s.by === by ? (s.dir === 1 ? -1 : 1) : typeof by === 'number' && !cols()[by].lower ? -1 : 1 }); };
  const arrow = (by: number | 'run' | 'step') => (sortBy().by === by ? (sortBy().dir === 1 ? ' ^' : ' v') : '');
  const addCol = (c: EvalCol) => { addToNew(c.name + runsMatcher(runIds())); showToast(`added ${c.name} to the board`); };
  const cellMenu = (anchor: Element, run: string, step: number, c: EvalCol) => openMenu(anchor, [
    { label: `add ${c.name} of ${run} to the board`, run: () => { addToNew(`${c.name}{run=${quote(run)}}`); showToast(`added ${c.name} of ${run}`); } },
    { label: `add every ${c.bench} metric of ${run}`, run: () => { addToNew(`eval/${c.bench}/*{run=${quote(run)}}`); showToast(`added eval/${c.bench}/* of ${run}`); } },
    { label: `board: zoom step-axis panels to step ${fmtStep(step)}`, run: () => { setHash({ view: null }); requestAnimationFrame(() => jumpToStep(step)); } },
  ], 'eval cell');
  return (
    <section class="evalsview">
      <h2>Evals</h2>
      <Show when={cols().length} fallback={
        <div class="note">No eval metrics in any run. Log them as <code>eval/&lt;benchmark&gt;/&lt;metric&gt;</code> at each checkpoint step, e.g. <code>{'{'}"step": 2000, "eval/gsm8k/acc": 0.41, "eval/wikitext/ppl": 18.2{'}'}</code>.</div>
      }>
        <div class="controls">
          <span class="note">{rows().length} checkpoints from {runs().length} run{runs().length === 1 ? '' : 's'} ({scope() === 'selected' ? 'the selected runs' : scope() === 'every' ? 'every run' : 'none of the selected runs has evals, so these are runs outside the selection'}) x {cols().length} metrics. Best per column is marked; lower is better for loss, nll, ppl and error rates. Click a header to sort, a value for actions.</span>
          <button type="button" title="one board panel per benchmark (every metric of it, across these runs)" aria-label="add every benchmark to the board"
            onClick={() => { for (const b of [...new Set(cols().map((c) => c.bench))]) addToNew(`eval/${b}/*${runsMatcher(runIds())}`); showToast('added one panel per benchmark'); }}>add every benchmark to the board</button>
        </div>
        <Show when={data.loading}><div class="note">loading…</div></Show>
        <div class="evalwrap">
          <table class="evals">
            <thead><tr>
              <th><button type="button" class="th" title="sort by run" aria-label="sort by run" onClick={() => sortOn('run')}>run{arrow('run')}</button></th>
              <th><button type="button" class="th" title="sort by step" aria-label="sort by step" onClick={() => sortOn('step')}>step{arrow('step')}</button></th>
              <For each={cols()}>{(c, ci) => (
                <th class="ecol">
                  <button type="button" class="th" title={`sort by ${c.name} (${c.lower ? 'lower' : 'higher'} is better)`} aria-label={`sort by ${c.name}`} onClick={() => sortOn(ci())}>
                    <span class="eb">{c.bench}</span><span class="em">{c.metric} {c.lower ? '(low)' : '(high)'}{arrow(ci())}</span>
                  </button>
                  <Show when={data()}>{(d) => <Spark col={c} runs={runIds()} data={d()} />}</Show>
                  <button type="button" class="add" title={`add ${c.name} across these runs to the board`} aria-label={`add ${c.name} to the board`} onClick={() => addCol(c)}>+ board</button>
                </th>
              )}</For>
            </tr></thead>
            <tbody>
              <For each={sorted()}>{(r, i) => (
                <tr>
                  <td class="rn" classList={{ rep: sortBy().by === 'run' && i() > 0 && sorted()[i() - 1].run === r.run }}>{r.run}</td>
                  <td class="num">{r.step < 1e5 ? r.step : fmtStep(r.step)}</td>
                  <For each={cols()}>{(c, ci) => (
                    <td class="num">
                      <Show when={Number.isFinite(r.vals[ci()])} fallback={<span class="miss">-</span>}>
                        <button type="button" class="ev" classList={{ best: bestSet().has(r.run + '\u0001' + r.step + '\u0001' + ci()) }}
                          title={`${c.name} = ${r.vals[ci()]} at step ${r.step} of ${r.run}${bestSet().has(r.run + '\u0001' + r.step + '\u0001' + ci()) ? ' (best in column)' : ''}`}
                          aria-label={`${c.name} ${fmt(r.vals[ci()])} at step ${r.step} of ${r.run}: actions`} onClick={(e) => cellMenu(e.currentTarget, r.run, r.step, c)}>{fmt(r.vals[ci()])}</button>
                      </Show>
                    </td>
                  )}</For>
                </tr>
              )}</For>
            </tbody>
          </table>
        </div>
        <div class="bkey"><For each={runIds()}>{(r, k) => <span><span class="sw-line" style={{ 'border-top-color': `var(${PALETTE[k() % PALETTE.length]})` }} /> {r} </span>}</For></div>
      </Show>
    </section>
  );
}
