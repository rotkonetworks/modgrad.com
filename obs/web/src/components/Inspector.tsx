import { createSignal, For, Show } from 'solid-js';
import * as Bd from '@/logic/board';
import { fieldsOf, varyingFields } from '@/logic/data';
import { runsVersion } from '@/stores/runs';
import {
  board, selPanel, findPanel, titleOf, parseQ, setStage, setQuery, renamePanel, showAll, openFocus, deletePanel, panelRuns, setInspector,
} from '@/stores/board';
import ForecastControls from '@/components/ForecastControls';
import ErrLine from '@/components/ErrLine';

type Opt = [string, string];
const uniq = (a: Opt[]) => a.filter((v, i) => a.findIndex((w) => w[0] === v[0]) === i);
function Sel(props: { label: string; title: string; value: string; options: Opt[]; onChange: (v: string) => void }) {
  return (
    <label title={props.title}>{props.label} <select aria-label={props.label + ': ' + props.title} value={props.value} onChange={(e) => props.onChange(e.currentTarget.value)}>
      <For each={props.options}>{([v, t]) => <option value={v} selected={v === props.value}>{t}</option>}</For>
    </select></label>
  );
}

/** settings of the selected panel (on demand: the panel's "inspect" button, or the toolbar) */
export default function Inspector() {
  const pv = () => { const id = selPanel(); return id ? board.panels.find((p) => p.id === id) : undefined; };
  const model = () => findPanel(selPanel());
  const [qerr, setQerr] = createSignal<{ src: string; msg: string; a: number; b: number; suggest: string[] } | null>(null);
  const stage = (k: string) => { const p = pv(); return p ? Bd.getStage(p.q, k) : null; };
  const runs = () => { runsVersion(); const p = model(); return p ? panelRuns(p) : []; };
  const set = (k: string, t: string | null) => { const p = model(); if (p) setStage(p, k, t); };
  let qi!: HTMLInputElement;
  const applyQ = () => { const x = parseQ(qi.value); if (!x.ok) { setQerr({ src: qi.value, msg: x.error, a: x.start, b: x.end, suggest: x.suggest }); return; } setQerr(null); const p = model(); if (p) setQuery(p, x.canonical); };
  return (
    <aside class="binsp" aria-label="panel inspector" data-board-scope>
      <div class="ihead">
        <h3>inspector</h3>
        <button type="button" title="close the inspector" aria-label="close the inspector" onClick={() => setInspector(false)}>close</button>
      </div>
      <Show when={pv()} fallback={<div class="bnote">Select a panel (click it, or its "inspect" button) to see its settings here.</div>}>
        {(p) => {
          const vf = () => varyingFields(runs()), fl = () => fieldsOf(runs());
          const metrics = () => { const ms = new Set<string>(); for (const r of runs()) for (const k of Object.keys(r.metrics)) ms.add(k); return [...ms].sort(); };
          const baseline = () => { const bs = stage('baseline') || ''; const bm = bs.match(/^baseline "?([^"]*)"?( ratio)?$/); return { run: bm ? bm[1] : '', ratio: !!(bm && bm[2]) }; };
          return (
            <>
              <div class="ititle">panel: {titleOf(p())}</div>
              <div class="row"><label>title <input value={p().title} placeholder={titleOf({ ...p(), title: '' })} aria-label="panel title" onChange={(e) => { const m = model(); if (m) renamePanel(m, e.currentTarget.value.trim()); }} /></label></div>
              <div class="row">
                <input ref={qi} class="q" value={p().q} spellcheck={false} aria-label="query" title="query text; Enter applies" onKeyDown={(e) => { if (e.key === 'Enter') applyQ(); }} />
                <button type="button" title="apply the query" aria-label="apply the query" onClick={applyQ}>apply</button>
              </div>
              <Show when={qerr()}>{(e) => <ErrLine src={e().src} msg={e().msg} a={e().a} b={e().b} suggest={e().suggest} onFix={(t) => { qi.value = t; applyQ(); }} />}</Show>
              <div class="row">
                <button type="button" classList={{ on: !!stage('log') }} aria-pressed={!!stage('log')} title="y scale: linear or log10" aria-label="y scale: linear or log10" onClick={() => set('log', stage('log') ? null : 'log')}>{stage('log') ? 'log y' : 'lin y'}</button>
                <Sel label="x" title="step, time, tokens, or any metric joined on step" value={(stage('x') || 'x step').slice(2)}
                  options={uniq([['step', 'step'], ['time', 'time'], ['tokens', 'tokens'], ...metrics().map((m) => [m, m] as Opt)])} onChange={(v) => set('x', v === 'step' ? null : `x ${v}`)} />
                <label title="EMA smoothing in [0, 1); legend statistics stay raw">ema <input value={(stage('ema') || 'ema 0').slice(4)} size="4" aria-label="ema weight" onChange={(e) => { const v = +e.currentTarget.value; set('ema', v > 0 && v < 1 ? `ema ${e.currentTarget.value.trim()}` : null); }} /></label>
              </div>
              <div class="row">
                <Sel label="norm" title="normalise each series" value={(stage('norm') || 'norm ').slice(5)} options={[['', 'none'], ['z', 'z-score'], ['minmax', 'min-max'], ['first', '/ first']]} onChange={(v) => set('norm', v ? `norm ${v}` : null)} />
                <Sel label="align" title="step alignment for formulas, baselines, seeds and metric x axes; never extrapolates" value={(stage('align') || 'align intersection').slice(6)} options={[['intersection', 'overlap'], ['union', 'union'], ['exact', 'exact']]} onChange={(v) => set('align', v === 'intersection' ? null : `align ${v}`)} />
              </div>
              <div class="row">
                <Sel label="colour by" title="colour by a config value (gradient for numbers)" value={(stage('by') || 'by ').slice(3)}
                  options={[['', 'series'], ['run', 'run'], ['metric', 'metric'], ...vf().map((f) => [f, f] as Opt), ...fl().filter((f) => !vf().includes(f)).map((f) => [f, f + ' (constant)'] as Opt)]} onChange={(v) => set('by', v ? `by ${v}` : null)} />
              </div>
              <div class="row">
                <Sel label="split" title="small multiples with locked axes" value={(stage('split') || 'split ').slice(6)} options={[['', 'none'], ['metric', 'metric'], ['run', 'run'], ['letter', 'letter'], ...vf().map((f) => [f, f] as Opt)]} onChange={(v) => set('split', v ? `split ${v}` : null)} />
              </div>
              <div class="row">
                <Sel label="seeds" title="aggregate runs that differ only in this field into a line plus band" value={stage('agg') || ''}
                  options={uniq([['', 'separate'], ['mean±sd over seed', 'mean±sd over seed'], ['median over seed', 'median (min-max band)'], ...vf().filter((f) => f !== 'seed').map((f) => [`mean±sd over ${f}`, `mean±sd over ${f}`] as Opt)])} onChange={(v) => set('agg', v || null)} />
              </div>
              <div class="row">
                <Sel label="baseline" title="plot relative to one run" value={baseline().run} options={[['', 'none'], ...runs().map((x) => [x.id, x.id] as Opt)]} onChange={(v) => set('baseline', v ? `baseline ${Bd.quote(v)}${baseline().ratio ? ' ratio' : ''}` : null)} />
                <Show when={baseline().run}>
                  <Sel label="as" title="difference or ratio" value={baseline().ratio ? 'ratio' : 'diff'} options={[['diff', 'difference'], ['ratio', 'ratio']]} onChange={(v) => set('baseline', `baseline ${Bd.quote(baseline().run)}${v === 'ratio' ? ' ratio' : ''}`)} />
                </Show>
              </div>
              <h3>forecast</h3>
              <ForecastControls id={p().id} />
              <div class="row">
                <button type="button" title="clear solo and hidden series" aria-label="show all series" onClick={() => { const m = model(); if (m) showAll(m); }}>show all series</button>
                <button type="button" title="open in the focus view" aria-label="expand panel" onClick={() => openFocus(p().id)}>expand</button>
                <button type="button" title="delete this panel (undoable)" aria-label="delete panel" onClick={() => { const m = model(); if (m) deletePanel(m); }}>delete panel</button>
              </div>
            </>
          );
        }}
      </Show>
    </aside>
  );
}
