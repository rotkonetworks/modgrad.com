// Legend table (RRD GPRINT style): swatch, name, Max/Avg/Min/Cur.
import { A } from '@/lib/wasm';
import { h } from '@/lib/dom';
import { fmt, nums } from '@/lib/fmt';

export interface LegendRow { name: string; color?: string | null; area?: boolean; stats?: number[]; values?: ArrayLike<number | null> }

export function legendTable(parent: Element, rows: LegendRow[], extraHead?: string): HTMLTableElement {
  const wrap = h('div', { class: 'legendwrap' }, parent);
  const t = h('table', { class: 'legend' }, wrap);
  const hr = h('tr', {}, t);
  h('th', {}, hr, ''); h('th', { style: 'text-align:left' }, hr, extraHead || '');
  for (const c of ['Max', 'Avg', 'Min', 'Cur']) h('th', {}, hr, c);
  for (const r of rows) {
    const tr = h('tr', {}, t);
    const sw = h('td', { class: 'sw' }, tr);
    if (r.color) {
      const s = h('span', { class: r.area ? 'sw-area' : 'sw-line' }, sw);
      if (r.area) s.style.background = `var(${r.color})`; else s.style.borderTopColor = `var(${r.color})`;
    }
    h('td', { class: 'name', title: r.name }, tr, r.name);
    const st = r.stats || Array.from(A.legend_stats(nums(r.values || [])));
    for (const v of st) h('td', {}, tr, Number.isFinite(v) ? fmt(v) : '-');
  }
  return t;
}
export function kvTable(parent: Element, pairs: [string, string][]): HTMLTableElement {
  const t = h('table', { class: 'legend' }, h('div', { class: 'legendwrap' }, parent));
  for (const [k, v] of pairs) { const tr = h('tr', {}, t); h('td', { class: 'name' }, tr, k); h('td', {}, tr, v); }
  return t;
}
