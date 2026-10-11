// Board and panel state: pure functions, no DOM. A board is {version, panels}; a panel is a query
// (text, parsed in wasm) plus view state. Query edits go through the wasm parser so chips and text
// always round-trip (crates/modgrad-obs-analysis/src/query.rs).
import { A } from '@/lib/wasm';
import type { Board, Chip, Lineage, Panel, PanelView, Parsed } from '@/logic/types';

export const BOARD_VERSION = 2;
export const STAGE_KINDS = ['ema', 'by', 'agg', 'split', 'x', 'forecast', 'log', 'norm', 'baseline', 'align', 'show', 'heatmap', 'stitch'];

export const uid = (): string => 'p' + Math.random().toString(36).slice(2, 10).replace(/[^a-z0-9]/g, '0');
export const quote = (s: unknown): string => '"' + String(s).replace(/"/g, '') + '"';

export function defaultView(): PanelView {
  return { follow: false, hidden: [], solo: null, y2: null, fc: { window: null, target: null, band: 80, pool: 'metric', live: true, frozen: null } };
}
export function newPanel(q: string, title = ''): Panel {
  return { id: uid(), title, q, view: defaultView() };
}

// ------------------------------------------------------------ validation and serialisation
interface LegacySeries { key?: string; expr?: string; alias?: string }
interface LegacyPanel { series: LegacySeries[]; ema?: number; x?: string; ylog?: boolean; norm?: string }
export function validBoard(j: unknown): Board {
  const b = j as { panels?: unknown };
  if (!b || !Array.isArray(b.panels)) throw new Error('not a board: expected {"panels": [...]}');
  const panels: Panel[] = [];
  for (const p0 of b.panels) {
    if (!p0 || typeof p0 !== 'object') continue;
    const p = p0 as Record<string, unknown>;
    let q = typeof p.q === 'string' ? p.q : null;
    if (q === null && Array.isArray(p.series)) q = legacyQuery(p as unknown as LegacyPanel); // version 1 layouts: series lists
    if (!q) continue;
    const pv = (p.view || {}) as Partial<PanelView>;
    const v: PanelView = { ...defaultView(), ...pv };
    v.fc = { ...defaultView().fc, ...(pv.fc || {}) };
    v.hidden = Array.isArray(v.hidden) ? v.hidden.map(String) : [];
    panels.push({ id: typeof p.id === 'string' && /^[A-Za-z0-9-]+$/.test(p.id) ? p.id : uid(), title: String(p.title || ''), q, view: v });
  }
  const lineage = validLineage((j as { lineage?: unknown }).lineage);
  return Object.keys(lineage).length ? { version: BOARD_VERSION, panels, lineage } : { version: BOARD_VERSION, panels };
}
/** {run id: {parent, step?, stage?}}; anything else is dropped */
export function validLineage(j: unknown): Record<string, Lineage> {
  const out: Record<string, Lineage> = {};
  if (!j || typeof j !== 'object') return out;
  for (const [id, v0] of Object.entries(j as Record<string, unknown>)) {
    const v = v0 as Partial<Lineage> | null;
    if (!v || typeof v.parent !== 'string' || !v.parent || v.parent === id) continue;
    const l: Lineage = { parent: v.parent };
    if (typeof v.step === 'number' && Number.isFinite(v.step)) l.step = v.step;
    if (typeof v.stage === 'string' && v.stage.trim()) l.stage = v.stage.trim();
    out[id] = l;
  }
  return out;
}
function legacyQuery(p: LegacyPanel): string {
  const ents: string[] = [];
  for (const s of p.series) {
    if (s.key) { const i = s.key.lastIndexOf(':'); ents.push(`${s.key.slice(i + 1)}{run=${quote(s.key.slice(0, i))}}`); }
    else if (s.expr && s.alias) ents.push(`${s.alias} = ${s.expr}`);
  }
  let q = ents.join('; ');
  if (q && p.ema) q += ` | ema ${p.ema}`;
  if (q && p.x && p.x !== 'step') q += ` | x ${p.x === 't' ? 'time' : p.x}`;
  if (q && p.ylog) q += ' | log';
  if (q && p.norm) q += ` | norm ${p.norm}`;
  return q;
}
export const serialize = (board: Board): string => JSON.stringify({ version: BOARD_VERSION, panels: board.panels, ...(board.lineage && Object.keys(board.lineage).length ? { lineage: board.lineage } : {}) }, null, 1);
/** the stitched loss of a run's whole lineage (Runs view, "whole lineage") */
export const lineageQuery = (metric: string, runId: string): string => `${metric}{run=${quote(runId)}} | stitch`;
/** stage names offered when setting a parent in the browser */
export const STAGE_NAMES = ['pretrain', 'midtrain', 'sft', 'rl', 'eval'];

// ------------------------------------------------------------ undo
export interface UndoEntry { what: string; board: Board }
export interface Undo { push(what: string, board: Board): void; pop(): UndoEntry | null; peek(): { what: string } | null; readonly size: number }
export function makeUndo(limit = 40): Undo {
  const stack: { what: string; json: string }[] = [];
  return {
    push(what, board) { stack.push({ what, json: JSON.stringify(board) }); if (stack.length > limit) stack.shift(); },
    pop() { const u = stack.pop(); return u ? { what: u.what, board: JSON.parse(u.json) as Board } : null; },
    peek() { return stack[stack.length - 1] || null; },
    get size() { return stack.length; },
  };
}

// ------------------------------------------------------------ query text operations (via wasm)
/** string index of UTF-8 byte offset `b` in `s` (the Rust parser reports byte offsets) */
export function byteIdx(s: string, b: number): number {
  let n = 0;
  for (let i = 0; i < s.length; i++) {
    if (n >= b) return i;
    const c = s.codePointAt(i)!;
    n += c < 0x80 ? 1 : c < 0x800 ? 2 : c < 0x10000 ? 3 : 4;
    if (c >= 0x10000) i++;
  }
  return s.length;
}
export function parse(q: string): Parsed {
  const r = A.query_parse(q);
  const out: Parsed = r.ok() ? { ok: true, canonical: r.canonical(), chips: JSON.parse(r.chips()) as Chip[] }
    : { ok: false, error: r.error(), start: byteIdx(q, r.err_start()), end: byteIdx(q, r.err_end()), suggest: r.suggest().split('\n').filter(Boolean) };
  r.free();
  return out;
}
/** matcher texts carried by every selector entry of every query (the board's default filter,
 *  e.g. `synthetic!=true` or `run=~"proj/*"`); [] when a query has no selector or fails to parse */
export function sharedMatchers(qs: string[]): string[] {
  let common: string[] | null = null;
  for (const q of qs) {
    const p = parse(q);
    if (!p.ok) return [];
    const per = new Map<number, Set<string>>();
    for (const c of p.chips) {
      if (c.kind === 'metric' && c.entry !== null) per.set(c.entry, per.get(c.entry) || new Set());
      if (c.kind === 'matcher' && c.entry !== null) (per.get(c.entry) || per.set(c.entry, new Set()).get(c.entry)!).add(c.text);
    }
    for (const ms of per.values()) common = (common ?? [...ms]).filter((t) => ms.has(t));
    if (common && !common.length) return [];
  }
  return common || [];
}
/** replace matcher `old` in every entry of `q` (`text` empty removes it) */
export function replaceMatcherAll(q: string, old: string, text: string): string {
  if (text.trim() === old) return q;
  for (;;) {
    const p = parse(q); if (!p.ok) return q;
    const i = p.chips.findIndex((c) => c.kind === 'matcher' && c.text === old);
    if (i < 0) return q;
    const next = replaceChip(q, i, text);
    if (next === q) return q;
    q = next;
  }
}
export function replaceChip(q: string, idx: number, text: string): string {
  const r = A.query_replace_chip(q, idx, text);
  const out = r.ok() ? r.canonical() : null;
  const er = r.ok() ? null : r.error();
  r.free();
  if (out === null) throw new Error(er || 'query error');
  return out;
}
/** index of the first top-level '|' (outside quotes, braces, parentheses), or -1 */
export function pipeAt(q: string): number {
  let d = 0, inq = false;
  for (let i = 0; i < q.length; i++) {
    const c = q[i];
    if (c === '"') inq = !inq;
    else if (!inq && (c === '{' || c === '(')) d++;
    else if (!inq && (c === '}' || c === ')')) d--;
    else if (!inq && d === 0 && c === '|') return i;
  }
  return -1;
}
function canon(q: string): string { const p = parse(q); if (!p.ok) throw new Error(p.error); return p.canonical; }
export function addEntries(q: string, entries: string): string {
  const i = pipeAt(q);
  return canon(i < 0 ? `${q}; ${entries}` : `${q.slice(0, i)}; ${entries} ${q.slice(i)}`);
}
export function getStage(q: string, kind: string): string | null {
  const p = parse(q); if (!p.ok) return null;
  const c = [...p.chips].reverse().find((c) => c.kind === kind);
  return c ? c.text : null;
}
/** set (text) or remove (null) the last stage of a kind */
export function setStage(q: string, kind: string, text: string | null): string {
  const p = parse(q); if (!p.ok) throw new Error(p.error);
  let idx = -1; p.chips.forEach((c, i) => { if (c.kind === kind) idx = i; });
  if (idx >= 0) return replaceChip(q, idx, text || '');
  return text ? canon(`${p.canonical} | ${text}`) : p.canonical;
}
export function letters(q: string): string[] {
  const p = parse(q); if (!p.ok) return [];
  return [...new Set(p.chips.filter((c) => c.letter).map((c) => c.letter as string))];
}
export function letterOf(i: number): string { let s = ''; i++; while (i > 0) { const m = (i - 1) % 26; s = String.fromCharCode(97 + m) + s; i = Math.floor((i - 1) / 26); } return s; }
export const nextLetter = (lets: string[]): string => { for (let i = 0; ; i++) { const l = letterOf(i); if (!lets.includes(l)) return l; } };

/** merge src's entries into dst (src stages are dropped; src letters are renamed so formulas keep working) */
export function mergeQueries(dst: string, src: string): string {
  const pd = parse(dst), ps = parse(src);
  if (!pd.ok) throw new Error(pd.error);
  if (!ps.ok) throw new Error(ps.error);
  interface Entry { letter: string; metric: string; matchers: string[]; formula: string | null }
  const entryTexts = (chips: Chip[]) => { // entry index -> {letter, text, formula}
    const m = new Map<number, Entry>();
    for (const c of chips) {
      if (c.entry === null) continue;
      const e = m.get(c.entry) || { letter: c.letter || '', metric: '', matchers: [], formula: null };
      if (c.kind === 'metric') e.metric = c.text; else if (c.kind === 'matcher') e.matchers.push(c.text); else if (c.kind === 'formula') e.formula = c.text;
      m.set(c.entry, e);
    }
    return [...m.values()].map((e) => ({ letter: e.letter, formula: e.formula, text: e.formula || (e.matchers.length ? `${e.metric === '*' ? '' : e.metric}{${e.matchers.join(', ')}}` : e.metric) }));
  };
  const de = entryTexts(pd.chips), se = entryTexts(ps.chips);
  const used = new Set(de.map((e) => e.letter));
  const nextFree = () => { for (let i = 0; ; i++) { const l = letterOf(i); if (!used.has(l)) { used.add(l); return l; } } };
  // selectors get positional letters: after merge, src selector k becomes the (nDstSel + k)-th selector letter
  const ren = new Map<string, string>();
  const sels = se.filter((e) => !e.formula);
  const formulas = se.filter((e) => e.formula);
  // formula names in dst and src are taken; positional letters skip them
  const taken = new Set([...de.filter((e) => e.formula), ...formulas].map((e) => e.letter));
  const nd = de.filter((e) => !e.formula).length;
  const posLetter = (n: number) => { let c = 0; for (let i = 0; ; i++) { const l = letterOf(i); if (taken.has(l)) continue; if (c === n) return l; c++; } };
  sels.forEach((e, j) => { ren.set(e.letter, posLetter(nd + j)); });
  const out = de.map((e) => e.text).concat(sels.map((e) => e.text));
  for (const f of formulas) {
    const [name, ...rest] = (f.formula as string).split('=');
    let nm = name.trim();
    if (de.some((e) => e.letter === nm) || [...ren.values()].includes(nm)) { const n2 = nextFree(); ren.set(nm, n2); nm = n2; }
    const body = rest.join('=').replace(/[A-Za-z0-9_./:\-]+/g, (t) => (ren.has(t) ? ren.get(t)! : t));
    out.push(`${nm} = ${body.trim()}`);
  }
  const i = pipeAt(pd.canonical);
  return canon(out.join('; ') + (i < 0 ? '' : ' ' + pd.canonical.slice(i)));
}
/** drop one run from a selector letter: removes the entry when it selects exactly that run (and no
 *  formula depends on positions), else adds or extends a run!~ exclusion */
export function excludeRun(q: string, letter: string, runId: string): string {
  const p = parse(q); if (!p.ok) throw new Error(p.error);
  const idxs = p.chips.map((c, i) => [c, i] as [Chip, number]).filter(([c]) => c.letter === letter && c.entry !== null);
  if (!idxs.length) throw new Error('no selector ' + letter);
  const [c0, i0] = idxs[0];
  if (c0.kind === 'formula') return replaceChip(q, i0, '');
  const hasFormula = p.chips.some((c) => c.kind === 'formula');
  const only = idxs.find(([c]) => c.kind === 'matcher' && (c.text === `run=${quote(runId)}` || c.text === `run=${runId}`));
  if (only && !hasFormula) return replaceChip(q, i0, '');
  const ex = idxs.find(([c]) => c.kind === 'matcher' && c.text.startsWith('run!~'));
  if (ex) { const v = ex[0].text.slice(5).replace(/^"|"$/g, ''); return replaceChip(q, ex[1], `run!~${quote(v + '|' + runId)}`); }
  const [cl, il] = idxs[idxs.length - 1];
  const add = `run!~${quote(runId)}`;
  return replaceChip(q, il, cl.kind === 'metric' ? `${cl.text}{${add}}` : `${cl.text}, ${add}`);
}

// ------------------------------------------------------------ selectors from browser picks
export interface Pick { run: string; metric: string }
/** picks -> selector entries text (one entry per metric) */
export function selectorFor(picks: Pick[]): string {
  const by = new Map<string, string[]>();
  for (const p of picks) { if (!by.has(p.metric)) by.set(p.metric, []); by.get(p.metric)!.push(p.run); }
  return [...by.entries()].map(([m, runs]) => (runs.length === 1 ? `${m}{run=${quote(runs[0])}}` : `${m}{run=~${quote(runs.join('|'))}}`)).join('; ');
}
export const metricQuery = (metric: string): string => metric;
export const runQuery = (runId: string): string => `{run=${quote(runId)}} | split metric`;
export const runsMatcher = (ids: string[]): string => (ids.length ? `{run=~${quote(ids.join('|'))}}` : '');

// ------------------------------------------------------------ presets and auto boards
/** history: one panel per metric over the given runs (the old History section, now a preset) */
export function presetHistory({ runIds, metrics, x = 'step', ema = 0 }: { runIds: string[]; metrics: string[]; x?: string; ema?: number }): Board {
  const sel = runsMatcher(runIds);
  const panels = metrics.map((m) => {
    let q = m + sel;
    if (ema > 0) q += ` | ema ${+ema.toFixed(2)}`;
    if (x === 't' || x === 'time') q += ' | x time';
    return newPanel(q);
  });
  return { version: BOARD_VERSION, panels };
}
export function presetAll(metrics: string[], runIds: string[]): Board {
  const sel = runsMatcher(runIds);
  return { version: BOARD_VERSION, panels: metrics.map((m) => newPanel(m + sel)) };
}
// Pre-training, SFT and RL naming conventions -> a default board. `metrics` maps name -> number of
// runs logging it; `replicates` is true when some runs differ only in seed.
const AUTO: [string, RegExp, string][] = [
  ['training loss', /^(train_?)?loss$|^train[_/]loss|^loss[_/]train/i, 'ema .6'],
  ['eval loss', /(eval|val|valid|heldout|held_out|test)[_/]?(loss|nll)|(loss|nll)[_/]?(eval|val|heldout)|heldout_nll$/i, ''],
  ['learning rate', /^(lr|learning_rate)$|[_/](lr|learning_rate)$/i, ''],
  ['grad norm', /grad_?norm/i, 'ema .6'],
  ['throughput', /tokens?_per_s|tok_s|throughput|samples_per_s|s_per_step|step_time/i, ''],
  ['tokens', /tokens?_seen|^tokens$|num_tokens|consumed_tokens/i, ''],
  ['accuracy', /acc(uracy)?(_pct)?$/i, ''],
  ['reward', /reward/i, 'ema .6'],
  ['kl', /(^|[_/])kl($|[_/])/i, ''],
  ['entropy', /entropy/i, ''],
  ['response length', /resp(onse)?_?len/i, ''],
];
export interface AutoLLM { families?: { family: string; metrics: string[] }[]; samples?: boolean; heatmapQuery?: (family: string, filter: string) => string; samplesMetric?: string; runsOf?: (metric: string) => string[] }
/** families that earn a heatmap on the default board, most useful first */
const HEATMAP_RANK = (f: string): number => (/grad_?norm/i.test(f) ? 0 : /act(ivation)?s?_?rms|rms/i.test(f) ? 1 : /update|weight_?norm|norm/i.test(f) ? 2 : 9);
export function autoBoard(metrics: Map<string, number>, { replicates = false, hasTokens = null as string | null, filter = '', families = [], samples = false, heatmapQuery, samplesMetric = 'samples:text', runsOf } = {} as { replicates?: boolean; hasTokens?: string | null; filter?: string } & AutoLLM): Board | null {
  // per-layer metrics belong to their family's heatmap, not to a single-metric panel
  const layered = new Set(families.flatMap((f) => f.metrics));
  const names = [...metrics.keys()].filter((m) => !layered.has(m));
  const panels: Panel[] = [];
  const used = new Set<string>();
  for (const [title, re, extra] of AUTO) {
    const hits = names.filter((m) => re.test(m) && !used.has(m)).sort((a, b) => metrics.get(b)! - metrics.get(a)! || a.localeCompare(b));
    if (!hits.length) continue;
    // one quantity under several names in different runs (loss / train/loss, val_loss / eval/loss,
    // a dropped CSV's loss next to train_loss_ema) is one panel: names whose runs do not overlap
    // the panel's runs join it (up to 3); accuracies are different benchmarks and stay apart
    const groups: string[][] = [];
    for (const m of hits) {
      const rs = runsOf ? runsOf(m) : [];
      const g = runsOf && title !== 'accuracy' ? groups.find((g) => g.length < 3 && !g.some((o) => runsOf(o).some((r) => rs.includes(r)))) : undefined;
      if (g) g.push(m); else groups.push([m]);
    }
    for (const g of groups.slice(0, title === 'eval loss' || title === 'accuracy' ? 2 : 1)) {
      for (const m of g) used.add(m);
      let q = g.map((m) => m + filter).join('; ');
      if (extra) q += ` | ${extra}`;
      if (replicates && /loss|nll|reward|acc/.test(title)) q += ' | mean±sd over seed';
      // tokens on x only when every run of the panel logs them (a run without would draw nothing)
      const tokOk = (m: string) => !runsOf || runsOf(m).every((r) => runsOf(hasTokens!).includes(r));
      if (hasTokens && /loss|nll/.test(title) && g.every(tokOk)) q += ` | x ${hasTokens}`;
      panels.push(newPanel(q));
    }
  }
  // per-layer heatmaps (pre-training post-mortems: which layer's gradients blew up, how activations drift with depth)
  if (heatmapQuery) {
    const fams = families.filter((f) => f.metrics.length >= 4 && HEATMAP_RANK(f.family) < 9).sort((a, b) => HEATMAP_RANK(a.family) - HEATMAP_RANK(b.family) || a.family.localeCompare(b.family));
    for (const f of fams.slice(0, 3)) panels.push(newPanel(heatmapQuery(f.family, filter)));
  }
  // generations of SFT / RL runs: the latest samples
  if (samples) panels.push(newPanel(samplesMetric + filter));
  return panels.length >= 2 ? { version: BOARD_VERSION, panels } : null;
}
/** headline metrics when nothing else applies: loss first, then acc/ppl/reward/score, sys/* last */
export function rankMetrics(names: string[], n = 4): string[] {
  const rank = (k: string) => (/(^|[/_])loss$/.test(k) ? 0 : /loss|nll/.test(k) ? 1 : /acc|ppl|perplex|reward|score/.test(k) ? 2 : k.startsWith('sys/') ? 4 : 3);
  return [...names].sort((a, b) => rank(a) - rank(b) || a.localeCompare(b)).slice(0, n);
}
