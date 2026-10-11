import { createEffect, createMemo, createSignal, For, on, onCleanup, Show } from 'solid-js';
import type { Chip, Panel as PanelT, QueryResult, RunEvent } from '@/logic/types';
import * as Bd from '@/logic/board';
import { evalQuery, planOf, varyingFields, SAMPLES_METRIC } from '@/logic/data';
import SamplesPanel from '@/components/SamplesPanel';
import { A } from '@/lib/wasm';
import { themeVersion } from '@/lib/theme';
import { catalog, catalogVersion, runVersion, eventsOf } from '@/stores/runs';
import {
  findPanel, titleOf, parseQ, selPanel, setSelPanel, editing, toggleEditing, settings, results, seen, fresh, commit,
  setStage, setQuery, renamePanel, toggleFollow, deletePanel, duplicatePanel, movePanel, mergePanel, showAll, openFocus,
  panelRuns, setInspector, focusId, board,
} from '@/stores/board';
import { openMenu, openPopover } from '@/stores/ui';
import { drawPanel, MAX_POINTS, type DrawOut } from '@/board/drawPanel';
import { chartBusy } from '@/charts/chart2d';
import { draggable } from '@/board/dnd';
import { loader } from '@/board/loader';
import ForecastControls from '@/components/ForecastControls';
import ErrLine from '@/components/ErrLine';

const CHIP_LABEL = (c: Chip): string => {
  if (c.kind === 'matcher') { const m = c.text.match(/^run(=~|=|!=|!~)"?(.*?)"?$/); if (m) return `${m[1].startsWith('!') ? 'not ' : ''}runs: ${m[2]}`; return c.text.replace(/"/g, ''); }
  if (c.kind === 'metric') return c.text === '*' ? 'all metrics' : c.text === SAMPLES_METRIC ? 'text samples' : c.text;
  return c.text;
};

/** one board panel: a query (chips / text) and its chart. Re-draws only when this panel's query,
 *  view, settings, theme, or the data of the runs its query reads change. */
export default function Panel(props: { panel: PanelT; inFocus?: boolean }) {
  const id = props.panel.id;
  const model = () => findPanel(id);
  const title = () => titleOf(props.panel);
  const parsed = createMemo(() => parseQ(props.panel.q));
  const [badge, setBadge] = createSignal<DrawOut['badge']>(null);
  const [chipEdit, setChipEdit] = createSignal<number | null>(null);
  const [qErr, setQErr] = createSignal<{ src: string | null; msg: string; a: number; b: number } | null>(null);
  const [pending, setPending] = createSignal(true);
  let host!: HTMLDivElement, foot!: HTMLDivElement, titleBtn!: HTMLButtonElement;

  // what the drawing depends on; run versions only for the runs this query reads
  const dataKey = createMemo(() => {
    const cat = catalog();
    const runs = [...new Set(planOf(props.panel.q, cat).map(([r]) => r))].sort();
    return catalogVersion() + ':' + runs.map((r) => r + '@' + runVersion(r)).join(',');
  });
  // follow is left out: pausing it (a zoom on the chart) must not rebuild the chart mid-gesture;
  // switching it on redraws once below so the window jumps to the newest data
  const viewSig = () => { const { follow: _f, ...rest } = props.panel.view; return rest; };
  const sig = createMemo(() => JSON.stringify([props.panel.q, props.panel.title, viewSig(), dataKey(), settings.events, settings.spikes, settings.linkX, themeVersion(), !!props.inFocus, !!focusId()]));

  // a query over the text-samples pseudo-metric only is a samples panel, not a chart
  const samplesQ = createMemo(() => {
    const pq = parsed(); if (!pq.ok) return false;
    const ms = pq.chips.filter((c) => c.kind === 'metric');
    return ms.length > 0 && ms.every((c) => c.text === SAMPLES_METRIC) && !pq.chips.some((c) => c.kind === 'formula');
  });
  const sampleRunIds = createMemo(() => (samplesQ() ? [...new Set(planOf(props.panel.q, catalog()).filter(([, m]) => m === SAMPLES_METRIC).map(([r]) => r))] : []));

  let tok = 0, cached: { key: string; res: QueryResult } | null = null;
  async function draw(): Promise<void> {
    const t = ++tok;
    const p = model(); if (!p) return;
    if (samplesQ()) { host.textContent = ''; foot.textContent = ''; setBadge(null); setPending(false); return; }
    const key = p.q + '\u0001' + dataKey();
    let res: QueryResult;
    if (cached && cached.key === key) res = cached.res;
    else {
      try { res = await evalQuery(p.q, catalog(), loader, MAX_POINTS); } catch (e) { res = { ok: false, error: String((e as Error).message || e) }; }
      if (t !== tok) return;
      cached = { key, res };
      // runs that newly match since the last evaluation of this panel
      if (res.ok) {
        const runs = new Set(res.traces.flatMap((tr) => tr.runs));
        const prev = seen.get(id);
        if (prev) for (const r of runs) if (!prev.has(r)) fresh.set(id + '\u0001' + r, Date.now());
        seen.set(id, runs);
      }
    }
    results.set(id, res);
    const events = new Map<string, RunEvent[]>();
    if (res.ok && settings.events && res.meta.x_label === 'step') {
      const ids = [...new Set(res.traces.flatMap((tr) => tr.runs))];
      const evs = await Promise.all(ids.map(eventsOf));
      ids.forEach((r, i) => events.set(r, evs[i]));
    }
    if (t !== tok || !host.isConnected) return;
    // never rebuild a chart under a drag (live rows land mid-gesture): retry when it ends
    if (chartBusy()) { setTimeout(() => { if (t === tok) draw(); }, 250); return; }
    const cur = model(); if (!cur) return;
    // keep keyboard/wheel focus on the same chart across a redraw (click-then-wheel zoom)
    const fi = [...host.querySelectorAll('svg')].indexOf(document.activeElement as SVGSVGElement);
    const out = drawPanel({ panel: cur, host, foot, res, events, inFocus: !!props.inFocus, focusOpen: !!focusId(), dataKey: dataKey(), title: title() });
    if (fi >= 0) host.querySelectorAll('svg')[fi]?.focus({ preventScroll: true });
    setBadge(out.badge);
    setPending(false);
  }
  createEffect(on(sig, () => { draw(); }));
  createEffect(on(() => props.panel.view.follow, (f) => { if (f) draw(); }, { defer: true }));
  onCleanup(() => { tok++; });

  const select = () => { if (selPanel() !== id) setSelPanel(id); };
  const multi = () => { const pq = parsed(); return pq.ok && new Set(pq.chips.filter((c) => c.letter).map((c) => c.letter)).size > 1; };
  const hasForecast = () => { const pq = parsed(); return pq.ok && pq.chips.some((c) => c.kind === 'forecast'); };

  function rename(): void {
    const t = titleBtn; const inp = document.createElement('input');
    inp.value = props.panel.title || title(); inp.setAttribute('aria-label', 'panel title');
    t.textContent = ''; t.appendChild(inp); inp.focus(); inp.select();
    let done = false;
    const fin = () => { if (done) return; done = true; const p = model(); const v = inp.value.trim(); inp.remove(); t.textContent = title(); if (p && v !== title()) renamePanel(p, v); };
    inp.addEventListener('keydown', (ev) => { if (ev.key === 'Enter') fin(); if (ev.key === 'Escape') { done = true; inp.remove(); t.textContent = title(); } });
    inp.addEventListener('blur', fin);
  }
  function panelMenu(anchor: Element): void {
    const p = model(); if (!p) return;
    openMenu(anchor, [
      { label: 'rename', run: rename },
      { label: 'inspect (settings)', run: () => { setSelPanel(id); setInspector(true); } },
      { label: 'duplicate', run: () => duplicatePanel(p) },
      { label: 'move up', run: () => movePanel(p, 'up') },
      { label: 'move down', run: () => movePanel(p, 'down') },
      ...board.panels.filter((q) => q.id !== id).map((q) => ({ label: `merge into "${titleOf(q)}"`, run: () => { const dst = findPanel(q.id); if (dst) mergePanel(p, dst); } })),
      { label: 'show all series', run: () => showAll(p) },
      ...(props.inFocus ? [] : [{ label: 'expand', run: () => openFocus(id) }]),
      { label: 'delete', run: () => deletePanel(p) },
    ], 'panel menu');
  }
  function stageMenu(anchor: Element): void {
    const p = model(); if (!p) return;
    const runs = panelRuns(p);
    const vf = varyingFields(runs);
    const lets = Bd.letters(p.q);
    const st = (kind: string, text: string) => () => setStage(p, kind, text);
    openMenu(anchor, [
      { label: 'ema .9 (smoothing)', run: st('ema', 'ema .9') },
      { label: 'mean±sd over seed', run: st('agg', 'mean±sd over seed') },
      ...vf.slice(0, 6).map((f) => ({ label: `colour by ${f}`, run: st('by', `by ${f}`) })),
      { label: 'split by metric', run: st('split', 'split metric') },
      { label: 'split by run', run: st('split', 'split run') },
      ...vf.slice(0, 4).map((f) => ({ label: `split by ${f}`, run: st('split', `split ${f}`) })),
      { label: 'x axis: time', run: st('x', 'x time') },
      { label: 'x axis: tokens', run: st('x', 'x tokens') },
      { label: 'forecast', run: st('forecast', 'forecast') },
      { label: 'log y', run: st('log', 'log') },
      { label: 'layer x step heatmap (layers/<i>/<stat>)', run: st('heatmap', 'heatmap over layer') },
      { label: 'layer heatmap, each layer / its median', run: st('heatmap', 'heatmap over layer rownorm') },
      { label: 'normalise: z-score', run: st('norm', 'norm z') },
      ...runs.slice(0, 8).map((r) => ({ label: `baseline ${r.id} (difference)`, run: st('baseline', `baseline ${Bd.quote(r.id)}`) })),
      ...(lets.length >= 2 ? [{ label: `formula: ${lets[0]} - ${lets[1]}`, run: () => commit('add formula', () => { p.q = Bd.addEntries(p.q, `${Bd.nextLetter(lets)} = ${lets[0]} - ${lets[1]}`); }) }] : []),
      { label: 'formula (type it)...', run: () => toggleEditing(id, true) },
      { label: 'filter: lr > ...', run: () => toggleEditing(id, true) },
    ], 'add a stage');
  }
  function applyChip(i: number, c: Chip, text: string): void {
    const r = A.query_replace_chip(props.panel.q, i, text);
    if (!r.ok()) { setQErr({ src: null, msg: `${text}: ${r.error()}`, a: 0, b: 0 }); r.free(); return; }
    const q = r.canonical(); r.free();
    setQErr(null); setChipEdit(null);
    const p = model(); if (p) commit(`edit ${c.text}`, () => { p.q = q; });
  }
  function applyText(v: string): void {
    const r = parseQ(v);
    if (!r.ok) { setQErr({ src: v, msg: r.error, a: r.start, b: r.end }); return; }
    setQErr(null);
    const p = model(); if (p) setQuery(p, r.canonical);
  }

  return (
    <div class="bpanel" classList={{ selp: selPanel() === id }} data-pid={id} tabindex="0" role="group" aria-label={'panel ' + title()}
      onPointerDown={select} onFocusIn={select}>
      <div class="bbar">
        <button type="button" class="bt" ref={(el) => { titleBtn = el; draggable(el, () => ({ type: 'panel', pid: id, label: `panel "${title()}"` })); }}
          title={`panel "${title()}": drag to reorder (drop on a panel edge) or merge (drop on its middle); double-click to rename`}
          aria-label={`panel ${title()}: drag to reorder or merge, double-click to rename`} onDblClick={rename}>{title()}</button>
        <button type="button" class="mini" classList={{ on: props.panel.view.follow }} aria-pressed={props.panel.view.follow}
          title={props.panel.view.follow ? 'following the newest data; click to pause' : 'follow the newest data (keeps the zoom width, slides to the end)'}
          aria-label="follow the newest data" onClick={() => { const p = model(); if (p) toggleFollow(p); }}>{props.panel.view.follow ? '|| follow' : '> follow'}</button>
        <button type="button" class="mini" classList={{ on: editing().has(id) }} aria-pressed={editing().has(id)} title="edit the query as text" aria-label="edit the query as text" onClick={() => toggleEditing(id)}>text</button>
        <button type="button" class="mini" title="panel settings in the inspector" aria-label="inspect panel settings" onClick={() => { setSelPanel(id); setInspector(true); }}>inspect</button>
        <button type="button" class="mini" title="panel menu" aria-label="panel menu" onClick={(e) => panelMenu(e.currentTarget)}>menu</button>
        <Show when={!props.inFocus}>
          <button type="button" class="mini" title="expand this panel into the focus view" aria-label="expand this panel" onClick={() => openFocus(id)}>expand</button>
        </Show>
        <button type="button" class="mini" title={`delete panel "${title()}"`} aria-label={`delete panel ${title()}`} onClick={() => { const p = model(); if (p) deletePanel(p); }}>x</button>
      </div>
      <div class="bchips">
        <For each={(() => { const pq = parsed(); return pq.ok ? pq.chips : []; })()}>{(c, i) => (
          <span class={'chipw k-' + c.kind}>
            <Show when={chipEdit() === i()} fallback={
              <button type="button" class="c" title={`edit "${c.text}"`} aria-label={`edit ${c.text}`} onClick={() => setChipEdit(i())}>
                {(multi() && c.kind === 'metric' ? c.letter + ': ' : '') + CHIP_LABEL(c)}
              </button>
            }>
              <ChipEditor text={c.text} onApply={(v) => applyChip(i(), c, v)} onCancel={() => setChipEdit(null)} />
            </Show>
            <button type="button" class="x" title={`remove "${c.text}"`} aria-label={`remove ${c.text}`} onClick={() => { const p = model(); if (p) commit(`remove ${c.text}`, () => { p.q = Bd.replaceChip(p.q, i(), ''); }); }}>x</button>
          </span>
        )}</For>
        <button type="button" class="mini" title="add a stage: smoothing, colour by, split, seeds, x axis, forecast, ..." aria-label="add a stage" onClick={(e) => stageMenu(e.currentTarget)}>+</button>
        <Show when={hasForecast()}>
          <button type="button" class="badge" classList={{ noskill: !!badge() && !badge()!.skill }} title={badge()?.title || 'forecast settings'} aria-label="forecast settings"
            onClick={(e) => openPopover(e.currentTarget, () => <ForecastPopover id={id} />, 'forecast')}>{badge()?.text || 'forecast ...'}</button>
        </Show>
      </div>
      <Show when={!parsed().ok}>
        {(() => { const pq = parsed(); return pq.ok ? null : <ErrLine src={props.panel.q} msg={pq.error} a={pq.start} b={pq.end} />; })()}
      </Show>
      <Show when={qErr()}>{(e) => <ErrLine src={e().src} msg={e().msg} a={e().a} b={e().b} />}</Show>
      <Show when={editing().has(id)}>
        <QueryText q={props.panel.q} onApply={applyText} />
      </Show>
      <Show when={pending()}><div class="bempty">evaluating…</div></Show>
      <Show when={samplesQ()}><SamplesPanel runs={sampleRunIds()} k={props.panel.view.samplesK ?? 4} /></Show>
      <div class="bchart" ref={host} />
      <div class="bfootn" ref={foot} />
    </div>
  );
}

function ChipEditor(props: { text: string; onApply: (v: string) => void; onCancel: () => void }) {
  let inp!: HTMLInputElement;
  let applied = false;
  const apply = () => { applied = true; props.onApply(inp.value); };
  return (
    <>
      <input ref={(el) => { inp = el; queueMicrotask(() => { el.focus(); el.select(); }); }} value={props.text} spellcheck={false} aria-label={'edit ' + props.text}
        onKeyDown={(e) => { if (e.key === 'Enter') apply(); if (e.key === 'Escape') props.onCancel(); }}
        onBlur={(e) => setTimeout(() => { if (!applied && !(e.target.parentElement && e.target.parentElement.contains(document.activeElement))) props.onCancel(); }, 200)} />
      <button type="button" class="c" title="apply" aria-label="apply" onPointerDown={(e) => e.preventDefault()} onClick={apply}>ok</button>
    </>
  );
}
function QueryText(props: { q: string; onApply: (v: string) => void }) {
  let inp!: HTMLInputElement;
  return (
    <div class="bq">
      <input ref={inp} value={props.q} spellcheck={false} aria-label="query text" title="query text: Enter applies" onKeyDown={(e) => { if (e.key === 'Enter') props.onApply(inp.value); }} />
      <button type="button" class="mini" title="apply the query text" aria-label="apply the query text" onClick={() => props.onApply(inp.value)}>apply</button>
    </div>
  );
}
function ForecastPopover(props: { id: string }) {
  return (
    <>
      <h3>forecast</h3>
      <ForecastControls id={props.id} />
      <div class="bnote">Ridge: one-step ridge on lag windows, hyperparameters chosen on a rolling-origin backtest (SearchCast style). Power law: y = a + b·x^-c, backtested on the last quarter. Auto uses the one with skill and the lower test error; with no skill the ghost is the last value. On the plot: drag the horizon tab, the bracket under the axis (fit window), and the target line.</div>
    </>
  );
}
