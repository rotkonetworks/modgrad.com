import { createMemo, createSignal, For, Show } from 'solid-js';
import { createStore } from 'solid-js/store';
import * as Bd from '@/logic/board';
import { streamList, layerFamilies, heatmapQuery, type Stream } from '@/logic/data';
import { S, runsVersion, selection } from '@/stores/runs';
import { board, findPanel, titleOf, addToNew, addTo, focusId, registerPicked } from '@/stores/board';
import { openMenu } from '@/stores/ui';
import { draggable } from '@/board/dnd';

// browser state lives at module level so it survives moving between the board and the focus view
const [st, setSt] = createStore({ filter: '', group: 'metric' as 'metric' | 'run', onlySel: false });
const [open, setOpen] = createSignal<Set<string>>(new Set(), { equals: false });
const [picks, setPicks] = createSignal<Set<string>>(new Set(), { equals: false });
let lastPick: string | null = null;
registerPicked(() => [...picks()]);

const splitKey = (k: string) => { const i = k.lastIndexOf(':'); return { run: k.slice(0, i), metric: k.slice(i + 1) }; };
export const pickedEntries = (): string => Bd.selectorFor([...picks()].map(splitKey));
export const pickedCount = (): number => picks().size;

/** "+ add to…" menu: new panel, or onto any panel (the focused one first) */
export function addMenu(anchor: Element, entries: string, label: string, q?: string): void {
  const f = focusId();
  const ids = f ? [f, ...board.panels.filter((p) => p.id !== f).map((p) => p.id)] : board.panels.map((p) => p.id);
  openMenu(anchor, [
    { label: 'new panel', title: `new panel with ${label}`, run: () => addToNew(q || entries, label) },
    ...ids.map((id) => { const p = findPanel(id)!; return { label: `onto "${titleOf(p)}"`, title: `overlay ${label} on ${titleOf(p)}`, run: () => { const m = findPanel(id); if (m) addTo(m, entries, label); } }; }),
  ], 'add to');
}

export default function SeriesBrowser() {
  const items = createMemo(() => {
    runsVersion();
    const sel = selection();
    const words = st.filter.trim().toLowerCase().split(/\s+/).filter(Boolean);
    const runs = S.runs.filter((r) => !st.onlySel || sel.has(r.id));
    return streamList(runs).filter((s) => words.every((w) => (s.run + ':' + s.metric).toLowerCase().includes(w)));
  });
  const visible = () => items().map((s) => s.run + ':' + s.metric);
  // per-layer metrics (layers/<i>/<stat>) collapse into one group per family, offered as a heatmap
  const famOf = createMemo(() => {
    const m = new Map<string, { family: string; n: number }>();
    for (const f of layerFamilies([...new Set(items().map((s) => s.metric))], 2)) for (const x of f.metrics) m.set(x, { family: f.family, n: f.layers.length });
    return m;
  });
  const groups = createMemo(() => {
    const g = new Map<string, Stream[]>();
    const fam = new Map<string, number>();
    for (const s of items()) {
      const f = st.group === 'metric' ? famOf().get(s.metric) : undefined;
      const k = st.group === 'run' ? s.run : f ? f.family : s.metric;
      if (f) fam.set(k, f.n);
      if (!g.has(k)) g.set(k, []); g.get(k)!.push(s);
    }
    return [...g.keys()].sort((a, b) => Number(fam.has(b)) - Number(fam.has(a)) || a.localeCompare(b)).map((k) => ({ name: k, items: g.get(k)!, layers: fam.get(k) || 0 }));
  });
  const searching = () => st.filter.trim().length > 0;
  const toggleGroup = (name: string) => { const k = st.group + ':' + name; const o = open(); if (o.has(k)) o.delete(k); else o.add(k); setOpen(o); };
  const pick = (key: string, shift: boolean) => {
    const s = picks();
    if (shift && lastPick) { const v = visible(); const a = v.indexOf(lastPick), b = v.indexOf(key); if (a >= 0 && b >= 0) for (let i = Math.min(a, b); i <= Math.max(a, b); i++) s.add(v[i]); }
    else if (s.has(key)) s.delete(key); else s.add(key);
    lastPick = key; setPicks(s);
  };
  return (
    <div class="sbrowse" aria-label="series browser">
      <div class="bhead">
        <input type="search" placeholder="filter series: run metric ..." aria-label="filter series" spellcheck={false} value={st.filter} onInput={(e) => setSt('filter', e.currentTarget.value)} />
        <div class="brow">
          <label>group <select aria-label="group the list" value={st.group} onChange={(e) => setSt('group', e.currentTarget.value as 'run')}>
            <option value="metric">by metric</option><option value="run">by run</option>
          </select></label>
          <label title="only list runs selected in the Runs list"><input type="checkbox" checked={st.onlySel} onChange={(e) => setSt('onlySel', e.currentTarget.checked)} /> selected runs</label>
        </div>
        {/* next to the list, not in the footer: the footer of a tall rail starts below the fold */}
        <Show when={picks().size}>
          <div class="brow picked" role="group" aria-label="picked series">
            <span>{picks().size} picked:</span>
            <button type="button" title="new panel from the picked series" aria-label="new panel from the picked series" onClick={() => { if (picks().size) addToNew(pickedEntries(), `${picks().size} series`); }}>new panel</button>
            <button type="button" title="add the picked series to a panel" aria-label="add the picked series to a panel" onClick={(e) => { if (picks().size) addMenu(e.currentTarget, pickedEntries(), `${picks().size} series`); }}>add to...</button>
            <button type="button" title="clear the picks" aria-label="clear the picked series" onClick={() => setPicks(new Set<string>())}>clear</button>
          </div>
        </Show>
      </div>
      <div class="blist">
        <For each={groups()}>{(g) => {
          const isOpen = () => searching() || open().has(st.group + ':' + g.name);
          const gq = () => (st.group === 'run' ? Bd.runQuery(g.name) : g.layers ? heatmapQuery(g.name) : Bd.metricQuery(g.name));
          const gEntries = () => (st.group === 'run' ? `{run=${Bd.quote(g.name)}}` : g.name);
          const glabel = () => (st.group === 'run' ? `run ${g.name} (split by metric)` : g.layers ? `${g.name}: layer x step heatmap (${g.layers} layers)` : `${g.name} across runs`);
          return (
            <div class="bgrp">
              <div class="bgh" ref={(el) => draggable(el, () => ({ type: 'series', entries: gEntries(), q: gq(), label: glabel() }))}>
                <span class="bgrip" title="drag onto the board or a panel" aria-hidden="true">::</span>
                <button type="button" class="tog" aria-expanded={isOpen()} title={`${isOpen() ? 'collapse' : 'expand'} ${g.name}`} aria-label={`${isOpen() ? 'collapse' : 'expand'} ${g.name}`} onClick={() => toggleGroup(g.name)}>
                  <span>{isOpen() ? '-' : '+'}</span><span class="lab">{g.name}</span>
                  <Show when={g.layers}><span class="tag" title="per-layer metrics: drag or + for a layer x step heatmap">{g.layers} layers</span></Show>
                  <span class="n">{g.items.length}</span>
                </button>
                <button type="button" class="add" title={`add ${glabel()} to...`} aria-label={`add ${glabel()} to a panel`} onClick={(e) => addMenu(e.currentTarget, gEntries(), glabel(), gq())}>+</button>
              </div>
              <Show when={isOpen()}>
                <For each={g.items}>{(s) => {
                  const key = s.run + ':' + s.metric;
                  const on = () => picks().has(key);
                  return (
                    <div class="bitem" classList={{ sel: on() }} ref={(el) => draggable(el, () => {
                      const ps = (picks().has(key) ? [...picks()] : [key]).map(splitKey);
                      return { type: 'series', entries: Bd.selectorFor(ps), label: ps.length > 1 ? `${ps.length} series` : key };
                    })}>
                      <span class="bgrip" title="drag onto the board or a panel" aria-hidden="true">::</span>
                      <button type="button" class="pick" aria-pressed={on()} title={`${on() ? 'deselect' : 'select'} ${key} (${s.n} points${s.kind === 'array' ? ', mean of a logged array' : ''}); shift for a range`}
                        aria-label={`${on() ? 'deselect' : 'select'} ${key}`} onClick={(e) => pick(key, e.shiftKey)}>
                        <span class="lab">{st.group === 'run' || g.layers ? (g.layers ? `${s.run}:${s.metric}` : s.metric) : s.run}</span><span class="k">{s.kind === 'array' ? 'array' : String(s.n)}</span>
                      </button>
                      <button type="button" class="add" title={`add ${key} to...`} aria-label={`add ${key} to a panel`} onClick={(e) => addMenu(e.currentTarget, Bd.selectorFor([{ run: s.run, metric: s.metric }]), key)}>+</button>
                    </div>
                  );
                }}</For>
              </Show>
            </div>
          );
        }}</For>
        <Show when={!items().length}><div class="bnote" style={{ padding: '6px' }}>No series match.</div></Show>
      </div>
      <div class="bfoot">
        <span>{items().length} shown{picks().size ? '' : '; click series to pick several for one panel'}</span>
      </div>
    </div>
  );
}
