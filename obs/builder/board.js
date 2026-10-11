// Board and panel state: pure functions, no DOM. A board is {version, panels}; a panel is a query
// (text, parsed in wasm) plus view state. Query edits go through the wasm parser so chips and text
// always round-trip (crates/modgrad-obs-analysis/src/query.rs).

export const BOARD_VERSION = 2;
export const STAGE_KINDS = ['ema', 'by', 'agg', 'split', 'x', 'forecast', 'log', 'norm', 'baseline', 'align', 'show'];

export const uid = () => 'p' + Math.random().toString(36).slice(2, 10).replace(/[^a-z0-9]/g, '0');
export const quote = (s) => '"' + String(s).replace(/"/g, '') + '"';

export function defaultView() {
  return { follow: false, hidden: [], solo: null, y2: null, fc: { window: null, target: null, band: 80, pool: 'metric', live: true, frozen: null } };
}
export function newPanel(q, title = '') {
  return { id: uid(), title, q, view: defaultView() };
}

// ------------------------------------------------------------ validation and serialisation
export function validBoard(j) {
  if (!j || !Array.isArray(j.panels)) throw new Error('not a board: expected {"panels": [...]}');
  const panels = [];
  for (const p of j.panels) {
    if (!p || typeof p !== 'object') continue;
    let q = typeof p.q === 'string' ? p.q : null;
    if (q === null && Array.isArray(p.series)) q = legacyQuery(p); // version 1 layouts: series lists
    if (!q) continue;
    const v = { ...defaultView(), ...(p.view || {}) };
    v.fc = { ...defaultView().fc, ...((p.view && p.view.fc) || {}) };
    v.hidden = Array.isArray(v.hidden) ? v.hidden.map(String) : [];
    panels.push({ id: typeof p.id === 'string' && /^[A-Za-z0-9-]+$/.test(p.id) ? p.id : uid(), title: String(p.title || ''), q, view: v });
  }
  return { version: BOARD_VERSION, panels };
}
function legacyQuery(p) {
  const ents = [];
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
export const serialize = (board) => JSON.stringify({ version: BOARD_VERSION, panels: board.panels }, null, 1);

// ------------------------------------------------------------ undo
export function makeUndo(limit = 40) {
  const stack = [];
  return {
    push(what, board) { stack.push({ what, json: JSON.stringify(board) }); if (stack.length > limit) stack.shift(); },
    pop() { const u = stack.pop(); return u ? { what: u.what, board: JSON.parse(u.json) } : null; },
    peek() { return stack[stack.length - 1] || null; },
    get size() { return stack.length; },
  };
}

// ------------------------------------------------------------ query text operations (via wasm A)
export function parse(A, q) {
  const r = A.query_parse(q);
  const out = r.ok() ? { ok: true, canonical: r.canonical(), chips: JSON.parse(r.chips()) } : { ok: false, error: r.error(), start: r.err_start(), end: r.err_end() };
  r.free();
  return out;
}
function replace(A, q, idx, text) {
  const r = A.query_replace_chip(q, idx, text);
  const out = r.ok() ? r.canonical() : null;
  const er = r.ok() ? null : r.error();
  r.free();
  if (out === null) throw new Error(er);
  return out;
}
export function replaceChip(A, q, idx, text) { return replace(A, q, idx, text); }
// index of the first top-level '|' (outside quotes, braces, parentheses), or -1
export function pipeAt(q) {
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
function canon(A, q) { const p = parse(A, q); if (!p.ok) throw new Error(p.error); return p.canonical; }
export function addEntries(A, q, entries) {
  const i = pipeAt(q);
  return canon(A, i < 0 ? `${q}; ${entries}` : `${q.slice(0, i)}; ${entries} ${q.slice(i)}`);
}
export function getStage(A, q, kind) {
  const p = parse(A, q); if (!p.ok) return null;
  const c = [...p.chips].reverse().find((c) => c.kind === kind);
  return c ? c.text : null;
}
// set (text) or remove (null) the last stage of a kind
export function setStage(A, q, kind, text) {
  const p = parse(A, q); if (!p.ok) throw new Error(p.error);
  let idx = -1; p.chips.forEach((c, i) => { if (c.kind === kind) idx = i; });
  if (idx >= 0) return replace(A, q, idx, text || '');
  return text ? canon(A, `${p.canonical} | ${text}`) : p.canonical;
}
export function letters(A, q) {
  const p = parse(A, q); if (!p.ok) return [];
  return [...new Set(p.chips.filter((c) => c.letter).map((c) => c.letter))];
}
// merge src's entries into dst (src stages are dropped; src letters are renamed so formulas keep working)
export function mergeQueries(A, dst, src) {
  const pd = parse(A, dst), ps = parse(A, src);
  if (!pd.ok) throw new Error(pd.error);
  if (!ps.ok) throw new Error(ps.error);
  const entryTexts = (chips) => { // entry index -> {letter, text, formula}
    const m = new Map();
    for (const c of chips) {
      if (c.entry === null) continue;
      const e = m.get(c.entry) || { letter: c.letter, metric: '', matchers: [], formula: null };
      if (c.kind === 'metric') e.metric = c.text; else if (c.kind === 'matcher') e.matchers.push(c.text); else if (c.kind === 'formula') e.formula = c.text;
      m.set(c.entry, e);
    }
    return [...m.values()].map((e) => ({ letter: e.letter, formula: e.formula, text: e.formula || (e.matchers.length ? `${e.metric === '*' ? '' : e.metric}{${e.matchers.join(', ')}}` : e.metric) }));
  };
  const de = entryTexts(pd.chips), se = entryTexts(ps.chips);
  const used = new Set(de.map((e) => e.letter));
  const nextFree = () => { for (let i = 0; ; i++) { const l = letterOf(i); if (!used.has(l)) { used.add(l); return l; } } };
  // selectors get positional letters: after merge, src selector k becomes the (nDstSel + k)-th selector letter
  const ren = new Map();
  const sels = se.filter((e) => !e.formula);
  const formulas = se.filter((e) => e.formula);
  // formula names in dst and src are taken; positional letters skip them
  const taken = new Set([...de.filter((e) => e.formula), ...formulas].map((e) => e.letter));
  let k = 0; const nd = de.filter((e) => !e.formula).length;
  const posLetter = (n) => { let c = 0; for (let i = 0; ; i++) { const l = letterOf(i); if (taken.has(l)) continue; if (c === n) return l; c++; } };
  sels.forEach((e, j) => { ren.set(e.letter, posLetter(nd + j)); k++; });
  void k;
  const out = de.map((e) => e.text).concat(sels.map((e) => e.text));
  for (const f of formulas) {
    const [name, ...rest] = f.formula.split('=');
    let nm = name.trim();
    if (de.some((e) => e.letter === nm) || [...ren.values()].includes(nm)) { const n2 = nextFree(); ren.set(nm, n2); nm = n2; }
    const body = rest.join('=').replace(/[A-Za-z0-9_./:\-]+/g, (t) => (ren.has(t) ? ren.get(t) : t));
    out.push(`${nm} = ${body.trim()}`);
  }
  const i = pipeAt(pd.canonical);
  return canon(A, out.join('; ') + (i < 0 ? '' : ' ' + pd.canonical.slice(i)));
}
export function letterOf(i) { let s = ''; i++; while (i > 0) { const m = (i - 1) % 26; s = String.fromCharCode(97 + m) + s; i = Math.floor((i - 1) / 26); } return s; }
// drop one run from a selector letter: removes the entry when it selects exactly that run (and no
// formula depends on positions), else adds or extends a run!~ exclusion
export function excludeRun(A, q, letter, runId) {
  const p = parse(A, q); if (!p.ok) throw new Error(p.error);
  const idxs = p.chips.map((c, i) => [c, i]).filter(([c]) => c.letter === letter && c.entry !== null);
  if (!idxs.length) throw new Error('no selector ' + letter);
  const [c0, i0] = idxs[0];
  if (c0.kind === 'formula') return replace(A, q, i0, '');
  const hasFormula = p.chips.some((c) => c.kind === 'formula');
  const only = idxs.find(([c]) => c.kind === 'matcher' && (c.text === `run=${quote(runId)}` || c.text === `run=${runId}`));
  if (only && !hasFormula) return replace(A, q, i0, '');
  const ex = idxs.find(([c]) => c.kind === 'matcher' && c.text.startsWith('run!~'));
  if (ex) { const v = ex[0].text.slice(5).replace(/^"|"$/g, ''); return replace(A, q, ex[1], `run!~${quote(v + '|' + runId)}`); }
  const [cl, il] = idxs[idxs.length - 1];
  const add = `run!~${quote(runId)}`;
  return replace(A, q, il, cl.kind === 'metric' ? `${cl.text}{${add}}` : `${cl.text}, ${add}`);
}

// ------------------------------------------------------------ selectors from browser picks
// picks: [{run, metric}] -> selector entries text (one entry per metric)
export function selectorFor(picks) {
  const by = new Map();
  for (const p of picks) { if (!by.has(p.metric)) by.set(p.metric, []); by.get(p.metric).push(p.run); }
  return [...by.entries()].map(([m, runs]) => (runs.length === 1 ? `${m}{run=${quote(runs[0])}}` : `${m}{run=~${quote(runs.join('|'))}}`)).join('; ');
}
export const metricQuery = (metric) => metric;
export const runQuery = (runId) => `{run=${quote(runId)}} | split metric`;

// ------------------------------------------------------------ presets and auto boards
// history: one panel per metric chosen in the History section, over the selected runs
export function presetHistory({ runIds, metrics, x, ema }) {
  const sel = runIds.length ? `{run=~${quote(runIds.join('|'))}}` : '';
  const panels = metrics.map((m) => {
    let q = m + sel;
    if (ema > 0) q += ` | ema ${+ema.toFixed(2)}`;
    if (x === 't') q += ' | x time';
    return newPanel(q);
  });
  return { version: BOARD_VERSION, panels };
}
export function presetAll(metrics, runIds) {
  const sel = runIds.length ? `{run=~${quote(runIds.join('|'))}}` : '';
  return { version: BOARD_VERSION, panels: metrics.map((m) => newPanel(m + sel)) };
}
// Pre-training, SFT and RL naming conventions -> a default board. `metrics` maps name -> number of
// runs logging it; `replicates` is true when some runs differ only in seed.
const AUTO = [
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
export function autoBoard(metrics, { replicates = false, hasTokens = null, filter = '' } = {}) {
  const names = [...metrics.keys()];
  const panels = [];
  const used = new Set();
  for (const [title, re, extra] of AUTO) {
    const hits = names.filter((m) => re.test(m) && !used.has(m)).sort((a, b) => metrics.get(b) - metrics.get(a) || a.localeCompare(b));
    if (!hits.length) continue;
    for (const m of hits.slice(0, title === 'eval loss' || title === 'accuracy' ? 2 : 1)) {
      used.add(m);
      let q = m + filter;
      if (extra) q += ` | ${extra}`;
      if (replicates && /loss|nll|reward|acc/.test(title)) q += ' | mean±sd over seed';
      if (hasTokens && /loss|nll/.test(title)) q += ` | x ${hasTokens}`;
      panels.push(newPanel(q, hits.length > 1 && title !== 'eval loss' && title !== 'accuracy' ? '' : ''));
    }
  }
  return panels.length >= 2 ? { version: BOARD_VERSION, panels } : null;
}
