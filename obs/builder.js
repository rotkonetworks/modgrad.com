// modgrad obs: graph builder (the Workspace section). DOM rendering and input only; state, query
// text operations, data access, forecasts and palette commands live in pure modules:
//   builder/board.js     board/panel state, serialisation, query edits, presets, auto boards, undo
//   builder/data.js      catalog, stream list, loaders, query evaluation, forecasts, ETA, spikes
//   builder/commands.js  command palette commands (input text -> actions)
// The query engine itself is Rust/wasm (crates/modgrad-obs-analysis/src/query.rs).
//
// Mounted from dashboard.html as mount(OBS). Listens for document 'obs:runs-changed' (fired by
// window.obsAddRuns and by reload) and re-evaluates every panel's query, so new matching runs
// appear without editing anything.
//
// Keyboard: no single-letter shortcuts (Vimium-C users). Ctrl/Cmd+K opens the command palette
// (can be switched off in settings), Ctrl/Cmd+Z undoes while focus is inside the board, Delete
// removes the focused panel. Everything else is a real button, so link hints reach it.

import * as Bd from './builder/board.js';
import * as Dt from './builder/data.js';
import { commands } from './builder/commands.js';

const LS_BOARD = 'obs-builder-board-v2';
const LS_SET = 'obs-builder-settings-v1';
const MAX_POINTS = 1500;

let O, A;
const U = {
  board: null, undo: Bd.makeUndo(), sel: new Set(), lastPick: null, selPanel: null, focus: null,
  group: 'metric', onlySel: false, filter: '', open: new Set(), catalog: '', dataVersion: 0,
  results: new Map(), seen: new Map(), fresh: new Map(), fcHist: new Map(), events: new Map(),
  settings: { palette: true, events: true, spikes: true }, els: {}, panelEls: new Map(), editing: new Set(),
};

// ------------------------------------------------------------ styles
const CSS = `
#builder { display: flex; gap: 12px; align-items: flex-start; }
.bside { width: 280px; flex: none; border: 1px solid var(--rule); font: 11px var(--mono); position: sticky; top: 8px; max-height: calc(100vh - 16px); display: flex; flex-direction: column; background: var(--bg); }
.bside.dz-trash { outline: 2px dashed var(--c3); outline-offset: -2px; }
.bside .bhead { padding: 4px 6px; border-bottom: 1px solid var(--rule); display: flex; flex-direction: column; gap: 4px; }
.bside .bhead input[type=search] { width: 100%; font: 11px var(--mono); }
.bside .brow { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.bside .brow label { display: inline-flex; gap: 3px; align-items: center; }
.blist { overflow-y: auto; flex: 1 1 auto; min-height: 120px; }
.bgrp > .bgh { display: flex; background: var(--hl); border-bottom: 1px solid var(--rule); }
.bgrp > .bgh button.tog { flex: 1; text-align: left; border: 0; background: none; font: 11px var(--mono); padding: 2px 6px; display: flex; gap: 6px; min-width: 0; }
.bgrp > .bgh button.tog .n { color: var(--muted); margin-left: auto; }
.bitem { display: flex; align-items: center; border-bottom: 1px dotted var(--rule); }
.bitem button.pick { flex: 1; min-width: 0; border: 0; background: none; text-align: left; font: 11px var(--mono); padding: 1px 4px; display: flex; gap: 6px; white-space: nowrap; }
.bitem button.pick .lab { overflow: hidden; text-overflow: ellipsis; }
.bitem button.pick .k { color: var(--muted); margin-left: auto; }
.bitem.sel { background: var(--hl); box-shadow: inset 3px 0 0 var(--c2); }
.bside button.add, .bpanel button.mini, .binsp button, .bmenu button, .bpop button { font: 10px var(--mono); padding: 0 4px; line-height: 15px; }
.bgrip { cursor: grab; touch-action: none; color: var(--muted); padding: 0 3px; user-select: none; -webkit-user-select: none; }
.bfoot { border-top: 1px solid var(--rule); padding: 4px 6px; display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
.binsp { border-top: 1px solid var(--frame); padding: 4px 6px; overflow-y: auto; max-height: 48%; flex: none; }
.binsp h3, .bpop h3 { font: 600 11px var(--mono); margin: 2px 0 4px; text-transform: uppercase; letter-spacing: .05em; }
.binsp .row, .bpop .row { display: flex; gap: 4px 8px; align-items: center; flex-wrap: wrap; margin: 3px 0; }
.binsp select, .binsp input, .bpop select, .bpop input { font: 10px var(--mono); padding: 0 2px; max-width: 150px; }
.binsp input.q { width: 100%; max-width: none; }
.bmain { flex: 1; min-width: 0; }
.btool { display: flex; flex-wrap: wrap; gap: 6px 10px; align-items: center; margin: 0 0 8px; font-size: 12px; }
.btool .msg { font: 11px var(--mono); color: var(--muted); }
#bcanvas { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 540px), 540px)); gap: 14px; align-items: start; min-height: 140px; padding: 4px; border: 1px dashed transparent; }
#bcanvas.dz-new, .bnew.dz-new { border-color: var(--c2); background: var(--hl); }
.bpanel { width: 100%; max-width: 540px; border: 1px solid transparent; position: relative; outline: none; background: var(--bg); }
.bpanel.wide { grid-column: 1 / -1; max-width: none; }
.bpanel.selp { border-color: var(--frame); }
.bpanel .panel .ptitle { display: none; }
.bpanel .bbar { display: flex; gap: 3px; align-items: center; font: 600 12px var(--mono); border-bottom: 1px solid var(--rule); padding: 1px 2px; }
.bpanel .bbar .bt { flex: 1; text-align: center; overflow-wrap: anywhere; cursor: grab; touch-action: none; user-select: none; -webkit-user-select: none; border: 0; background: none; font: 600 12px var(--mono); padding: 1px 2px; }
.bpanel .bbar .bt input { font: 600 12px var(--mono); width: 100%; text-align: center; }
.bpanel button.on, .binsp button.on, .bpop button.on { background: var(--hl); border-color: var(--frame); color: var(--ink); font-weight: 600; }
.bchips { display: flex; flex-wrap: wrap; gap: 3px; padding: 3px 2px 1px; font: 10px var(--mono); align-items: center; }
.chipw { display: inline-flex; border: 1px solid var(--muted); }
.chipw button { border: 0; font: 10px var(--mono); padding: 0 4px; line-height: 15px; background: var(--bg); }
.chipw button.x { border-left: 1px solid var(--rule); color: var(--muted); }
.chipw.k-metric { border-color: var(--frame); }
.chipw.k-metric button.c { font-weight: 600; }
.chipw.k-forecast, .chipw.k-agg, .chipw.k-by, .chipw.k-split { border-style: dashed; }
.chipw input { font: 10px var(--mono); width: 16em; border: 0; }
.bchips .badge { border: 1px solid var(--c2); color: var(--ink); font: 10px var(--mono); padding: 0 4px; line-height: 15px; }
.bchips .badge.noskill { border-color: var(--muted); color: var(--muted); }
.bq { display: flex; gap: 4px; padding: 3px 2px; align-items: center; }
.bq input { flex: 1; min-width: 0; font: 11px var(--mono); }
.berr { font: 11px var(--mono); color: var(--c3); padding: 1px 2px; white-space: pre-wrap; overflow-wrap: anywhere; }
.berr u { text-decoration: underline wavy var(--c3); }
.bnote { font: 11px var(--mono); color: var(--muted); padding: 1px 2px; white-space: pre-wrap; overflow-wrap: anywhere; }
.beta { font: 11px var(--mono); padding: 1px 2px; }
.bcells { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 380px), 1fr)); gap: 8px; }
.bcell .ctitle { font: 600 11px var(--mono); text-align: center; color: var(--muted); }
.bkey { display: flex; gap: 6px; align-items: center; font: 10px var(--mono); color: var(--muted); padding: 2px; }
.bkey .grad { width: 120px; height: 8px; border: 1px solid var(--frame); }
.bpanel table.legend td.al { color: var(--muted); text-align: left; padding-right: 2px; }
.bpanel table.legend td.name button { border: 0; background: none; padding: 0; font: 11px var(--mono); text-align: left; max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; display: block; color: var(--ink); }
.bpanel table.legend tr.off td { opacity: .45; }
.bpanel table.legend tr.off td.name button { text-decoration: line-through; }
.bpanel table.legend td.sw { cursor: grab; touch-action: none; }
.bpanel table.legend td.ctl { white-space: nowrap; }
.bpanel table.legend.compact .st { display: none; }
.bpanel table.legend tr.fresh td { animation: bfresh 2.5s ease-out; }
@keyframes bfresh { from { background: var(--hl); opacity: 0; } to { opacity: 1; } }
.bpanel .tagnew { font: 9px var(--mono); border: 1px solid var(--c1); color: var(--c1); padding: 0 2px; margin-left: 4px; }
.ldot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 3px; vertical-align: 0; }
.bpanel.dz-add, .bpanel.dz-merge { box-shadow: 0 0 0 2px var(--c2); border-color: var(--c2); }
.bpanel.dz-before { box-shadow: -5px 0 0 0 var(--c2); }
.bpanel.dz-after { box-shadow: 5px 0 0 0 var(--c2); }
.bempty { font: 11px var(--mono); color: var(--muted); padding: 30px 8px; text-align: center; border: 1px dashed var(--rule); margin: 4px 0; }
.bnew { border: 1px dashed var(--muted); min-height: 120px; display: flex; align-items: center; justify-content: center; text-align: center; font: 11px var(--mono); color: var(--muted); padding: 10px; }
.bghost { position: fixed; z-index: 60; pointer-events: none; background: var(--bg); border: 1px solid var(--frame); font: 11px var(--mono); padding: 2px 6px; white-space: pre; box-shadow: 3px 3px 0 var(--frame); }
.btoast { position: fixed; left: 50%; bottom: 14px; transform: translateX(-50%); z-index: 70; background: var(--bg); border: 1px solid var(--frame); font: 11px var(--mono); padding: 4px 8px; display: flex; gap: 8px; align-items: center; box-shadow: 3px 3px 0 var(--frame); }
.bmenu, .bpop { position: fixed; z-index: 65; background: var(--bg); border: 1px solid var(--frame); box-shadow: 3px 3px 0 var(--frame); font: 11px var(--mono); padding: 4px; max-height: 60vh; overflow-y: auto; min-width: 180px; max-width: min(420px, calc(100vw - 16px)); }
.bmenu button { display: block; width: 100%; text-align: left; border: 0; padding: 2px 6px; font: 11px var(--mono); }
.bmenu button:hover, .bmenu button:focus { background: var(--hl); }
.bfocus { position: fixed; inset: 0; z-index: 30; background: var(--bg); display: flex; flex-direction: column; padding: 8px 16px; }
.bfocus .fbar { display: flex; gap: 8px; align-items: center; border-bottom: 1px solid var(--frame); padding-bottom: 6px; margin-bottom: 8px; font: 600 13px var(--mono); }
.bfocus .fbar .sp { flex: 1; }
.bfocus .fbody { display: flex; gap: 12px; flex: 1; min-height: 0; }
.bfocus .fbody .bside { position: static; max-height: 100%; }
.bfocus .fmain { flex: 1; min-width: 0; overflow-y: auto; }
.bfocus .bpanel { max-width: none; }
.bfocus .bpanel .panel { max-width: min(100%, calc((100vh - 230px) * 540 / 210)); margin: 0 auto; }
.bfocus .bpanel .bcells .panel { max-width: none; }
.bpal { position: fixed; inset: 0; z-index: 80; background: rgba(0,0,0,.25); display: flex; justify-content: center; align-items: flex-start; padding-top: 12vh; }
.bpal .box { background: var(--bg); border: 1px solid var(--frame); box-shadow: 4px 4px 0 var(--frame); width: min(560px, calc(100vw - 32px)); font: 12px var(--mono); }
.bpal .box .top { display: flex; gap: 6px; padding: 6px; border-bottom: 1px solid var(--rule); }
.bpal input { flex: 1; font: 13px var(--mono); }
.bpal .list { max-height: 50vh; overflow-y: auto; }
.bpal .list button { display: block; width: 100%; text-align: left; border: 0; padding: 3px 8px; font: 12px var(--mono); }
.bpal .list button.cur { background: var(--hl); box-shadow: inset 3px 0 0 var(--c2); }
body.bdragging, body.bdragging * { cursor: grabbing !important; }
body.bfocused { overflow: hidden; }
@media (max-width: 860px) {
  #builder, .bfocus .fbody { flex-direction: column; }
  .bside { width: 100%; position: static; max-height: 46vh; }
  .bfocus { overflow-y: auto; }
}
`;

// ------------------------------------------------------------ helpers
const h = (...a) => O.h(...a);
function btn(parent, label, title, on, cls = '') {
  const b = h('button', { type: 'button', title, 'aria-label': title, class: cls }, parent, label);
  if (on) b.addEventListener('click', on);
  return b;
}
function sel(parent, opts, val, on, title) {
  const s = h('select', { title, 'aria-label': title }, parent);
  for (const [v, t] of opts) { const o = h('option', { value: v }, s, t); if (v === val) o.selected = true; }
  s.addEventListener('change', () => on(s.value));
  return s;
}
const findPanel = (id) => U.board.panels.find((p) => p.id === id);
const parseQ = (q) => Bd.parse(A, q);
function save() { try { localStorage.setItem(LS_BOARD, Bd.serialize(U.board)); } catch (e) { /* storage unavailable: board lives for this page */ } }
function saveSettings() { try { localStorage.setItem(LS_SET, JSON.stringify(U.settings)); } catch (e) { /* ignore */ } }
// every state change goes through here: optional undo snapshot, mutation, save, re-render
function commit(what, fn, opts = {}) {
  if (what) U.undo.push(what, U.board);
  try { fn(); } catch (e) { if (what) U.undo.pop(); toast(String(e.message || e)); return false; }
  save(); render(); updateUndo();
  if (what && opts.toast) toast(opts.toast, true);
  return true;
}
function undo() {
  const u = U.undo.pop(); if (!u) return;
  U.board = Bd.validBoard(u.board); save(); render(); updateUndo(); toast(`undid: ${u.what}`);
}
function updateUndo() {
  const b = U.els.undo; if (!b) return;
  const top = U.undo.peek(); b.disabled = !top; b.title = top ? `undo: ${top.what} (ctrl+z)` : 'nothing to undo';
}
let toastTimer = null;
function toast(text, withUndo) {
  document.querySelectorAll('.btoast').forEach((t) => t.remove());
  const t = h('div', { class: 'btoast', role: 'status' }, document.body);
  h('span', {}, t, text);
  if (withUndo) btn(t, 'undo', 'undo the last change', () => { t.remove(); undo(); });
  clearTimeout(toastTimer); toastTimer = setTimeout(() => t.remove(), 6000);
}
function popover(anchor, cls = 'bmenu') {
  closePopovers();
  const m = h('div', { class: cls, role: 'dialog' }, document.body);
  const r = anchor.getBoundingClientRect();
  m.style.left = Math.max(4, Math.min(r.left, innerWidth - 430)) + 'px';
  m.style.top = Math.min(r.bottom + 2, innerHeight - 80) + 'px';
  setTimeout(() => document.addEventListener('pointerdown', function off(ev) { if (!m.contains(ev.target) && ev.target !== anchor) { m.remove(); document.removeEventListener('pointerdown', off, true); } }, true), 0);
  return m;
}
function closePopovers() { document.querySelectorAll('.bmenu,.bpop').forEach((m) => m.remove()); }
function menu(anchor, items) {
  const m = popover(anchor);
  for (const it of items) btn(m, it.label, it.title || it.label, () => { m.remove(); it.run(); });
  const f = m.querySelector('button'); if (f) f.focus();
  return m;
}
function errLine(parent, src, msg, a, b) {
  const d = h('div', { class: 'berr', role: 'alert' }, parent);
  if (src && b > a) { d.append(src.slice(0, a)); h('u', {}, d, src.slice(a, b) || ' '); d.append(src.slice(b) + '\n'); }
  d.append(msg);
  return d;
}
const fmtStep = (n) => (n >= 1e6 ? +(n / 1e6).toPrecision(3) + 'M' : n >= 1e3 ? +(n / 1e3).toPrecision(3) + 'k' : String(Math.round(n)));
const titleOf = (p) => {
  if (p.title) return p.title;
  const pq = parseQ(p.q); if (!pq.ok) return p.q;
  const ms = pq.chips.filter((c) => c.kind === 'metric' || c.kind === 'formula').map((c) => c.text);
  return ms.length ? ms.slice(0, 3).join(', ') + (ms.length > 3 ? ` +${ms.length - 3}` : '') : p.q;
};
function fieldsOf(runs) {
  const s = new Set(); for (const r of runs) for (const [k, v] of Object.entries(r.config || {})) if (v !== null && typeof v !== 'object') s.add(k);
  return [...s].sort();
}
// config fields that vary across these runs (useful for by / split)
function varyingFields(runs) {
  return fieldsOf(runs).filter((k) => new Set(runs.map((r) => String((r.config || {})[k]))).size > 1);
}

// ------------------------------------------------------------ evaluation (cached per query + data version)
async function evaluate(p) {
  const key = p.q + '\u0001' + U.dataVersion;
  const c = U.results.get(p.id);
  if (c && c.key === key) return c.res;
  const res = await Dt.evalQuery(A, p.q, U.catalog, U.load, MAX_POINTS);
  U.results.set(p.id, { key, res });
  // new matching runs since the last evaluation of this panel
  if (res.ok) {
    const runs = new Set(res.traces.flatMap((t) => t.runs));
    const prev = U.seen.get(p.id);
    if (prev) for (const r of runs) if (!prev.has(r)) U.fresh.set(p.id + '\u0001' + r, Date.now());
    U.seen.set(p.id, runs);
  }
  return res;
}
async function eventsOf(runId) {
  if (U.events.has(runId)) return U.events.get(runId);
  const run = O.S.byId.get(runId); let ev = [];
  if (run && run.n_events) { try { ev = await O.getJSON(O.src.events(run)); } catch (e) { ev = []; } }
  U.events.set(runId, ev);
  return ev;
}

// ------------------------------------------------------------ drag and drop (pointer events: mouse, pen, touch)
// Starts after 4 px (mouse/pen) or a 300 ms hold (touch, so lists still scroll). Every drop has a
// click path too: the + buttons in the browser, the x on legend rows and chips, the panel menu.
let DR = null;
function draggable(el, payload) {
  el.addEventListener('pointerdown', (ev) => {
    if (ev.button !== 0 || DR) return;
    if (ev.target.closest('input, select')) return;
    const touch = ev.pointerType === 'touch';
    const st = { x: ev.clientX, y: ev.clientY, id: ev.pointerId, active: false, payload, el, touch, timer: null };
    DR = st;
    if (touch) st.timer = setTimeout(() => { if (DR === st) activate(st, st.x, st.y); }, 300);
  });
}
function activate(st, x, y) {
  st.data = st.payload();
  if (!st.data) { DR = null; return; }
  st.active = true;
  st.ghost = h('div', { class: 'bghost' }, document.body, '');
  document.body.classList.add('bdragging');
  O.hideTip(); closePopovers();
  move(x, y);
}
function clearHl() { document.querySelectorAll('.dz-add,.dz-merge,.dz-before,.dz-after,.dz-new,.dz-trash').forEach((e) => e.classList.remove('dz-add', 'dz-merge', 'dz-before', 'dz-after', 'dz-new', 'dz-trash')); }
function target(x, y) {
  const d = DR.data;
  const el = document.elementFromPoint(x, y);
  const pel = el && el.closest('.bpanel');
  const canvas = el && (el.closest('#bcanvas') || el.closest('.fmain'));
  if (pel) {
    const p = findPanel(pel.dataset.pid); if (!p) return null;
    if (d.type === 'panel') {
      if (p.id === d.pid) return null;
      const r = pel.getBoundingClientRect(), fx = (x - r.left) / r.width;
      if (fx < 0.22) return { kind: 'before', el: pel, p, cls: 'dz-before', text: `move before "${titleOf(p)}"` };
      if (fx > 0.78) return { kind: 'after', el: pel, p, cls: 'dz-after', text: `move after "${titleOf(p)}"` };
      return { kind: 'merge', el: pel, p, cls: 'dz-merge', text: `merge into "${titleOf(p)}"` };
    }
    if (d.type === 'trace' && d.pid === p.id) return { kind: 'none', text: 'drop outside this panel to remove' };
    return { kind: 'add', el: pel, p, cls: 'dz-add', text: `${d.type === 'trace' ? 'move' : 'overlay'} onto "${titleOf(p)}"` };
  }
  if (canvas) {
    if (d.type === 'panel') return canvas.id === 'bcanvas' ? { kind: 'end', el: canvas, cls: 'dz-new', text: 'move to the end' } : null;
    const nw = el.closest('.bnew');
    return { kind: 'new', el: nw || canvas, cls: 'dz-new', text: d.type === 'trace' ? 'move to a new panel' : 'new panel' };
  }
  if (d.type === 'trace') return { kind: 'remove', el: el && el.closest('.bside'), cls: 'dz-trash', text: 'release to remove from the panel' };
  return null;
}
function move(x, y) {
  const st = DR; if (!st || !st.active) return;
  const t = target(x, y); st.t = t;
  clearHl(); if (t && t.el && t.cls) t.el.classList.add(t.cls);
  st.ghost.textContent = `${st.data.label}\n${t ? t.text : 'no drop target here'}`;
  st.ghost.style.left = Math.min(x + 14, innerWidth - st.ghost.offsetWidth - 4) + 'px'; st.ghost.style.top = (y + 14) + 'px';
}
function endDrag(cancel) {
  const st = DR; DR = null; if (!st) return;
  clearTimeout(st.timer);
  if (!st.active) return;
  st.ghost.remove(); document.body.classList.remove('bdragging'); clearHl();
  st.el.dataset.dragged = '1'; setTimeout(() => { delete st.el.dataset.dragged; }, 0); // swallow the click after a drag
  if (!cancel && st.t) drop(st.data, st.t);
}
document.addEventListener('pointermove', (ev) => {
  const st = DR; if (!st || ev.pointerId !== st.id) return;
  if (!st.active) {
    const dist = Math.hypot(ev.clientX - st.x, ev.clientY - st.y);
    if (st.touch) { if (dist > 8) { clearTimeout(st.timer); DR = null; } return; }
    if (dist > 4) activate(st, ev.clientX, ev.clientY);
    return;
  }
  ev.preventDefault();
  move(ev.clientX, ev.clientY);
});
document.addEventListener('pointerup', (ev) => { if (DR && ev.pointerId === DR.id) endDrag(false); });
document.addEventListener('pointercancel', (ev) => { if (DR && ev.pointerId === DR.id) endDrag(true); });
document.addEventListener('touchmove', (ev) => { if (DR && DR.active) ev.preventDefault(); }, { passive: false });
document.addEventListener('contextmenu', (ev) => { if (DR && DR.touch) ev.preventDefault(); });
document.addEventListener('click', (ev) => { if (ev.target.closest && ev.target.closest('[data-dragged]')) { ev.stopPropagation(); ev.preventDefault(); } }, true);

// payload kinds: {type:'series', entries, q?, label} | {type:'trace', pid, letter, run, label} | {type:'panel', pid, label}
function drop(d, t) {
  if (d.type === 'series') {
    if (t.kind === 'new') return addToNew(d.q || d.entries, d.label);
    if (t.kind === 'add') return addTo(t.p, d.entries, d.label);
    return;
  }
  if (d.type === 'trace') {
    const src = findPanel(d.pid); if (!src) return;
    if (t.kind === 'none') return;
    if (t.kind === 'remove') return removeTrace(src, d);
    const entries = d.entries;
    commit(`move ${d.label}`, () => {
      src.q = Bd.excludeRun(A, src.q, d.letter, d.run);
      if (t.kind === 'new') { const p = Bd.newPanel(entries); U.board.panels.push(p); U.selPanel = p.id; }
      else { t.p.q = Bd.addEntries(A, t.p.q, entries); U.selPanel = t.p.id; }
    }, { toast: `moved ${d.label}` });
    return;
  }
  if (d.type === 'panel') {
    const src = findPanel(d.pid); if (!src) return;
    if (t.kind === 'merge') {
      commit(`merge "${titleOf(src)}" into "${titleOf(t.p)}"`, () => {
        t.p.q = Bd.mergeQueries(A, t.p.q, src.q);
        U.board.panels = U.board.panels.filter((p) => p !== src);
        U.selPanel = t.p.id;
      }, { toast: `merged into "${titleOf(t.p)}"` });
      return;
    }
    commit('', () => {
      const L = U.board.panels; L.splice(L.indexOf(src), 1);
      if (t.kind === 'end') L.push(src); else { const j = L.indexOf(t.p); L.splice(t.kind === 'before' ? j : j + 1, 0, src); }
      U.selPanel = src.id;
    });
  }
}
function addToNew(q, label) {
  let p;
  commit('', () => { p = Bd.newPanel(parseQ(q).ok ? parseQ(q).canonical : q); U.board.panels.push(p); U.selPanel = p.id; });
  if (p) toast(`new panel: ${label || q}`);
  return p;
}
function addTo(p, entries, label) {
  commit('', () => { p.q = Bd.addEntries(A, p.q, entries); U.selPanel = p.id; });
  toast(`overlaid ${label || entries} on "${titleOf(p)}"`);
}
function removeTrace(p, d) {
  commit(`remove ${d.label}`, () => {
    if (d.run && d.single) p.q = Bd.excludeRun(A, p.q, d.letter, d.run);
    else p.view.hidden = [...new Set([...p.view.hidden, d.label])];
  }, { toast: `removed ${d.label} from "${titleOf(p)}"` });
}
function deletePanel(p) {
  commit(`delete panel "${titleOf(p)}"`, () => {
    U.board.panels = U.board.panels.filter((q) => q !== p);
    if (U.selPanel === p.id) U.selPanel = null;
    if (U.focus === p.id) closeFocus();
  }, { toast: `deleted panel "${titleOf(p)}"` });
}

// ------------------------------------------------------------ series browser
function addMenu(anchor, entries, label, q) {
  const items = [{ label: 'new panel', title: `new panel with ${label}`, run: () => addToNew(q || entries, label) }];
  const ps = U.focus ? [findPanel(U.focus), ...U.board.panels.filter((p) => p.id !== U.focus)] : U.board.panels;
  for (const p of ps) if (p) items.push({ label: `onto "${titleOf(p)}"`, title: `overlay ${label} on ${titleOf(p)}`, run: () => addTo(p, entries, label) });
  menu(anchor, items);
}
function renderBrowser() {
  const list = U.els.list; if (!list) return;
  const words = U.filter.trim().toLowerCase().split(/\s+/).filter(Boolean);
  const runs = O.S.runs.filter((r) => !U.onlySel || O.S.sel.has(r.id));
  const items = Dt.streamList(runs).filter((s) => words.every((w) => (s.run + ':' + s.metric).toLowerCase().includes(w)));
  U.visible = items.map((s) => s.run + ':' + s.metric);
  list.innerHTML = '';
  const groups = new Map();
  for (const s of items) { const g = U.group === 'run' ? s.run : s.metric; if (!groups.has(g)) groups.set(g, []); groups.get(g).push(s); }
  const frag = document.createDocumentFragment();
  for (const g of [...groups.keys()].sort((a, b) => a.localeCompare(b))) {
    const arr = groups.get(g);
    const box = h('div', { class: 'bgrp' }, frag);
    const open = words.length > 0 || U.open.has(U.group + ':' + g);
    const hd = h('div', { class: 'bgh' }, box);
    h('span', { class: 'bgrip', title: 'drag onto the canvas or a panel', 'aria-hidden': 'true' }, hd, '::');
    const tog = btn(hd, '', `${open ? 'collapse' : 'expand'} ${g}`, () => { const k = U.group + ':' + g; U.open.has(k) ? U.open.delete(k) : U.open.add(k); renderBrowser(); }, 'tog');
    tog.setAttribute('aria-expanded', open);
    h('span', {}, tog, open ? '-' : '+'); h('span', { class: 'lab' }, tog, g); h('span', { class: 'n' }, tog, String(arr.length));
    // a metric heading is that metric across all runs; a run heading is its metrics as small multiples
    const gq = U.group === 'run' ? Bd.runQuery(g) : Bd.metricQuery(g);
    const gEntries = U.group === 'run' ? `{run=${Bd.quote(g)}}` : g;
    const glabel = U.group === 'run' ? `run ${g} (split by metric)` : `${g} across runs`;
    btn(hd, '+', `add ${glabel} to...`, (ev) => addMenu(ev.currentTarget, gEntries, glabel, gq), 'add');
    draggable(hd, () => ({ type: 'series', entries: gEntries, q: gq, label: glabel }));
    if (!open) continue;
    for (const s of arr) {
      const key = s.run + ':' + s.metric;
      const it = h('div', { class: 'bitem' + (U.sel.has(key) ? ' sel' : '') }, box);
      h('span', { class: 'bgrip', title: 'drag onto the canvas or a panel', 'aria-hidden': 'true' }, it, '::');
      const pk = btn(it, '', `${U.sel.has(key) ? 'deselect' : 'select'} ${key} (${s.n} points${s.kind === 'array' ? ', mean of a logged array' : ''}); shift for a range`, (ev) => {
        if (ev.shiftKey && U.lastPick) { const a = U.visible.indexOf(U.lastPick), b = U.visible.indexOf(key); if (a >= 0 && b >= 0) for (let i = Math.min(a, b); i <= Math.max(a, b); i++) U.sel.add(U.visible[i]); }
        else if (U.sel.has(key)) U.sel.delete(key); else U.sel.add(key);
        U.lastPick = key; renderBrowser();
      }, 'pick');
      pk.setAttribute('aria-pressed', U.sel.has(key));
      h('span', { class: 'lab' }, pk, U.group === 'run' ? s.metric : s.run);
      h('span', { class: 'k' }, pk, s.kind === 'array' ? 'array' : String(s.n));
      const entries = Bd.selectorFor([{ run: s.run, metric: s.metric }]);
      btn(it, '+', `add ${key} to...`, (ev) => addMenu(ev.currentTarget, entries, key), 'add');
      draggable(it, () => {
        const picks = (U.sel.has(key) ? [...U.sel] : [key]).map((k) => { const i = k.lastIndexOf(':'); return { run: k.slice(0, i), metric: k.slice(i + 1) }; });
        const e = Bd.selectorFor(picks);
        return { type: 'series', entries: e, label: picks.length > 1 ? `${picks.length} series` : key };
      });
    }
  }
  list.appendChild(frag);
  if (!items.length) h('div', { class: 'bnote', style: 'padding:6px' }, list, 'No series match.');
  U.els.selinfo.textContent = `${items.length} shown, ${U.sel.size} selected`;
  U.els.selnew.disabled = U.els.seladd.disabled = U.els.selclr.disabled = !U.sel.size;
}
function selEntries() {
  return Bd.selectorFor([...U.sel].map((k) => { const i = k.lastIndexOf(':'); return { run: k.slice(0, i), metric: k.slice(i + 1) }; }));
}

// ------------------------------------------------------------ board and panels
function render() {
  renderInspector();
  const c = U.els.canvas; if (!c) return;
  const fresh = new Map();
  c.innerHTML = '';
  for (const p of U.board.panels) {
    if (U.focus === p.id) { const ph = h('div', { class: 'bnew' }, c, `"${titleOf(p)}" is open in the focus view`); void ph; continue; }
    const el = panelShell(p, false); c.appendChild(el); fresh.set(p.id, el);
  }
  h('div', { class: 'bnew' }, c, U.board.panels.length ? 'drop series here for a new panel' : 'Drag series from the list onto this area to make a panel (or use their + buttons); drop more onto a panel to overlay them. Presets above build a whole board.');
  if (U.focus) { const p = findPanel(U.focus); if (p) { U.els.fmain.innerHTML = ''; const el = panelShell(p, true); U.els.fmain.appendChild(el); fresh.set(p.id, el); U.els.ftitle.textContent = titleOf(p); } }
  U.panelEls = fresh;
  U.els.count.textContent = `${U.board.panels.length} panel${U.board.panels.length === 1 ? '' : 's'}`;
  const tok = (U.renderTok = (U.renderTok || 0) + 1);
  (async () => { for (const [id, el] of fresh) { if (tok !== U.renderTok) return; const p = findPanel(id); if (p) await fillPanel(p, el, tok); } })();
}
function panelShell(p, inFocus) {
  const el = h('div', { class: 'bpanel' + (U.selPanel === p.id ? ' selp' : ''), 'data-pid': p.id, tabindex: 0, role: 'group', 'aria-label': 'panel ' + titleOf(p) });
  el.addEventListener('pointerdown', () => selectPanel(p.id));
  el.addEventListener('focusin', () => selectPanel(p.id));
  const bar = h('div', { class: 'bbar' }, el);
  const t = btn(bar, titleOf(p), `panel "${titleOf(p)}": drag to reorder (drop on a panel edge) or merge (drop on its middle); double-click to rename`, null, 'bt');
  draggable(t, () => ({ type: 'panel', pid: p.id, label: `panel "${titleOf(p)}"` }));
  t.addEventListener('dblclick', () => renameInline(p, t));
  const fol = btn(bar, p.view.follow ? '|| follow' : '> follow', p.view.follow ? 'following the newest data; click to pause' : 'follow the newest data (keeps the zoom width, slides to the end)', () => commit('', () => { p.view.follow = !p.view.follow; }), 'mini' + (p.view.follow ? ' on' : ''));
  fol.setAttribute('aria-pressed', !!p.view.follow);
  const tx = btn(bar, 'text', 'edit the query as text', () => { U.editing.has(p.id) ? U.editing.delete(p.id) : U.editing.add(p.id); render(); }, 'mini' + (U.editing.has(p.id) ? ' on' : ''));
  tx.setAttribute('aria-pressed', U.editing.has(p.id));
  btn(bar, 'menu', 'panel menu', (ev) => panelMenu(ev.currentTarget, p), 'mini');
  if (!inFocus) btn(bar, 'expand', 'expand this panel into the focus view', () => openFocus(p.id), 'mini');
  btn(bar, 'x', `delete panel "${titleOf(p)}"`, () => deletePanel(p), 'mini');
  // chips
  const chips = h('div', { class: 'bchips' }, el);
  const errs = h('div', {}, el);
  renderChips(p, chips, errs);
  if (U.editing.has(p.id)) {
    const row = h('div', { class: 'bq' }, el);
    const inp = h('input', { value: p.q, spellcheck: 'false', 'aria-label': 'query text', title: 'query text: Enter applies' }, row);
    const apply = () => {
      errs.innerHTML = '';
      const r = parseQ(inp.value);
      if (!r.ok) { errLine(errs, inp.value, r.error, r.start, r.end); return; }
      if (r.canonical !== p.q) commit('edit query', () => { p.q = r.canonical; });
    };
    inp.addEventListener('keydown', (ev) => { if (ev.key === 'Enter') apply(); });
    btn(row, 'apply', 'apply the query text', apply, 'mini');
  }
  h('div', { class: 'bchart' }, el);
  h('div', { class: 'bfootn' }, el);
  return el;
}
function selectPanel(id) {
  if (U.selPanel === id) return;
  U.selPanel = id;
  document.querySelectorAll('.bpanel.selp').forEach((e) => e.classList.toggle('selp', e.dataset.pid === id));
  document.querySelectorAll(`.bpanel[data-pid="${id}"]`).forEach((e) => e.classList.add('selp'));
  renderInspector();
}
function renameInline(p, t) {
  const inp = h('input', { value: p.title || titleOf(p), 'aria-label': 'panel title' }); t.innerHTML = ''; t.appendChild(inp); inp.focus(); inp.select();
  let done = false;
  const fin = () => { if (done) return; done = true; const v = inp.value.trim(); commit('rename panel', () => { p.title = v === titleOf({ ...p, title: '' }) ? '' : v; }); };
  inp.addEventListener('keydown', (ev) => { if (ev.key === 'Enter') fin(); });
  inp.addEventListener('blur', fin);
}
function panelMenu(anchor, p) {
  menu(anchor, [
    { label: 'rename', run: () => { const t = document.querySelector(`.bpanel[data-pid="${p.id}"] .bt`); if (t) renameInline(p, t); } },
    { label: 'duplicate', run: () => commit('duplicate panel', () => { const c = JSON.parse(JSON.stringify(p)); c.id = Bd.uid(); U.board.panels.splice(U.board.panels.indexOf(p) + 1, 0, c); }) },
    { label: 'move up', run: () => commit('', () => { const L = U.board.panels, i = L.indexOf(p); if (i > 0) { L.splice(i, 1); L.splice(i - 1, 0, p); } }) },
    { label: 'move down', run: () => commit('', () => { const L = U.board.panels, i = L.indexOf(p); if (i < L.length - 1) { L.splice(i, 1); L.splice(i + 1, 0, p); } }) },
    ...U.board.panels.filter((q) => q !== p).map((q) => ({ label: `merge into "${titleOf(q)}"`, run: () => drop({ type: 'panel', pid: p.id }, { kind: 'merge', p: q }) })),
    { label: 'show all series', run: () => commit('', () => { p.view.hidden = []; p.view.solo = null; }) },
    { label: 'expand', run: () => openFocus(p.id) },
    { label: 'delete', run: () => deletePanel(p) },
  ]);
}
const CHIP_LABEL = (c) => {
  if (c.kind === 'matcher') { const m = c.text.match(/^run(=~|=|!=|!~)"?(.*?)"?$/); if (m) return `${m[1].startsWith('!') ? 'not ' : ''}runs: ${m[2]}`; return c.text.replace(/"/g, ''); }
  if (c.kind === 'metric') return (c.text === '*' ? 'all metrics' : c.text);
  return c.text;
};
function renderChips(p, box, errs) {
  const pq = parseQ(p.q);
  if (!pq.ok) { errLine(errs, p.q, pq.error, pq.start, pq.end); }
  const multi = pq.ok && new Set(pq.chips.filter((c) => c.letter).map((c) => c.letter)).size > 1;
  (pq.ok ? pq.chips : []).forEach((c, i) => {
    const w = h('span', { class: 'chipw k-' + c.kind }, box);
    const lab = (multi && (c.kind === 'metric' || c.kind === 'formula') ? (c.kind === 'metric' ? c.letter + ': ' : '') : '') + CHIP_LABEL(c);
    btn(w, lab, `edit "${c.text}"`, () => editChip(p, w, i, c, errs), 'c');
    btn(w, 'x', `remove "${c.text}"`, () => commit(`remove ${c.text}`, () => { p.q = Bd.replaceChip(A, p.q, i, ''); }), 'x');
  });
  btn(box, '+', 'add a stage: smoothing, colour by, split, seeds, x axis, forecast, ...', (ev) => stageMenu(ev.currentTarget, p), 'mini');
  if (pq.ok && pq.chips.some((c) => c.kind === 'forecast')) {
    const b = btn(box, 'forecast ...', 'forecast settings', (ev) => forecastPopover(ev.currentTarget, p), 'badge');
    b.dataset.badge = p.id;
  }
}
function editChip(p, w, i, c, errs) {
  w.innerHTML = '';
  const inp = h('input', { value: c.text, spellcheck: 'false', 'aria-label': 'edit ' + c.text }, w);
  let applied = false;
  const apply = () => {
    errs.innerHTML = '';
    const r = A.query_replace_chip(p.q, i, inp.value);
    if (!r.ok()) { errLine(errs, null, `${inp.value}: ${r.error()}`, 0, 0); r.free(); return; }
    const q = r.canonical(); r.free(); applied = true;
    commit(`edit ${c.text}`, () => { p.q = q; });
  };
  inp.addEventListener('keydown', (ev) => { if (ev.key === 'Enter') apply(); });
  const ok = btn(w, 'ok', 'apply', apply, 'c');
  ok.addEventListener('pointerdown', (ev) => ev.preventDefault());
  inp.addEventListener('blur', () => setTimeout(() => { if (!applied && w.isConnected && !w.contains(document.activeElement)) render(); }, 200));
  inp.focus(); inp.select();
}
function panelRuns(p) {
  const r = U.results.get(p.id);
  const ids = r && r.res.ok ? [...new Set(r.res.traces.flatMap((t) => t.runs))] : [];
  return ids.map((id) => O.S.byId.get(id)).filter(Boolean);
}
function setStage(p, kind, text, what) {
  commit(what || (text ? text : `remove ${kind}`), () => { p.q = Bd.setStage(A, p.q, kind, text); });
}
function stageMenu(anchor, p) {
  const runs = panelRuns(p).length ? panelRuns(p) : O.selectedRuns();
  const vf = varyingFields(runs);
  const lets = Bd.letters(A, p.q);
  const items = [
    { label: 'ema .9 (smoothing)', run: () => setStage(p, 'ema', 'ema .9') },
    { label: 'mean±sd over seed', run: () => setStage(p, 'agg', 'mean±sd over seed') },
    ...vf.slice(0, 6).map((f) => ({ label: `colour by ${f}`, run: () => setStage(p, 'by', `by ${f}`) })),
    { label: 'split by metric', run: () => setStage(p, 'split', 'split metric') },
    { label: 'split by run', run: () => setStage(p, 'split', 'split run') },
    ...vf.slice(0, 4).map((f) => ({ label: `split by ${f}`, run: () => setStage(p, 'split', `split ${f}`) })),
    { label: 'x axis: time', run: () => setStage(p, 'x', 'x time') },
    { label: 'x axis: tokens', run: () => setStage(p, 'x', 'x tokens') },
    { label: 'forecast', run: () => setStage(p, 'forecast', 'forecast') },
    { label: 'log y', run: () => setStage(p, 'log', 'log') },
    { label: 'normalise: z-score', run: () => setStage(p, 'norm', 'norm z') },
    ...runs.slice(0, 8).map((r) => ({ label: `baseline ${r.id} (difference)`, run: () => setStage(p, 'baseline', `baseline ${Bd.quote(r.id)}`) })),
    ...(lets.length >= 2 ? [{ label: `formula: ${lets[0]} - ${lets[1]}`, run: () => commit('add formula', () => { p.q = Bd.addEntries(A, p.q, `${nextLetter(lets)} = ${lets[0]} - ${lets[1]}`); }) }] : []),
    { label: 'formula (type it)...', run: () => { U.editing.add(p.id); render(); } },
    { label: 'filter: lr > ...', run: () => { U.editing.add(p.id); render(); } },
  ];
  menu(anchor, items);
}
const nextLetter = (lets) => { for (let i = 0; ; i++) { const l = Bd.letterOf(i); if (!lets.includes(l)) return l; } };

// ------------------------------------------------------------ drawing a panel
async function fillPanel(p, el, tok) {
  const host = el.querySelector('.bchart'), foot = el.querySelector('.bfootn');
  let res;
  try { res = await evaluate(p); } catch (e) { res = { ok: false, error: String(e.message || e) }; }
  if (tok !== U.renderTok || !el.isConnected) return;
  host.innerHTML = ''; foot.innerHTML = '';
  if (!res.ok) { h('div', { class: 'bempty' }, host, 'query error (see above)'); return; }
  const meta = res.meta;
  const traces = res.traces;
  if (!traces.length) { h('div', { class: 'bempty' }, host, 'nothing matches this query yet; it updates as runs arrive'); for (const w of meta.warnings) h('div', { class: 'bnote' }, foot, w); return; }
  const fc = meta.forecast && meta.x_label === 'step' ? meta.forecast : null;
  const isStep = meta.x_label === 'step';
  // colours
  const PAL = O.PALETTE;
  let colour = (i) => PAL[i % PAL.length];
  let key = null;
  if (meta.by) {
    const bc = Dt.byColours(traces, meta.by_numeric);
    const hex = (v) => O.cssv(v);
    const lo = hexRgb(hex('--c2')), hi = hexRgb(hex('--c3'));
    traces.forEach((t, i) => {
      const c = bc.numeric ? mixRgb(lo, hi, bc.pos[i]) : null;
      el.style.setProperty('--bk' + i, c || `var(${PAL[bc.idx[i] % PAL.length]})`);
    });
    colour = (i) => '--bk' + i;
    key = bc;
  }
  // forecasts (step axis), with the previous fit kept as a faint trace for accountability
  const fcs = new Map();
  if (fc) {
    const to = fc.to || defaultHorizon(traces);
    for (const t of traces) {
      if (hiddenTrace(p, t)) continue;
      const pool = p.view.fc.pool === 'none' ? [] : traces.filter((o) => o !== t && (p.view.fc.pool === 'group' ? o.letter === t.letter : o.metric === t.metric));
      let f;
      const fr = p.view.fc.frozen && p.view.fc.frozen[t.label];
      if (!p.view.fc.live && fr) f = fr;
      else f = Dt.forecastTrace(A, t, to, { method: fc.method, window: p.view.fc.window, band: p.view.fc.band, pool });
      fcs.set(t.label, f);
      const hk = p.id + '\u0001' + t.label, hist = U.fcHist.get(hk);
      if (f && !f.none) {
        if (!hist) U.fcHist.set(hk, { cur: f, prev: null });
        else if (hist.cur.lastX !== f.lastX) U.fcHist.set(hk, { cur: f, prev: hist.cur });
        else hist.cur = f;
      }
    }
  }
  // cells (split -> small multiples with locked axes)
  const cells = meta.cells.length ? meta.cells : [''];
  const split = cells.length > 1 || cells[0] !== '';
  el.classList.toggle('wide', split && !U.focus);
  let yrange = null;
  if (split) { let lo = Infinity, hi = -Infinity; for (const t of traces) if (!hiddenTrace(p, t)) for (let i = 0; i < t.ys.length; i++) { const v = t.ys[i]; if (Number.isFinite(v) && (!meta.ylog || v > 0)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } if (t.band) { if (Number.isFinite(t.lo[i]) && (!meta.ylog || t.lo[i] > 0)) lo = Math.min(lo, t.lo[i]); if (Number.isFinite(t.hi[i])) hi = Math.max(hi, t.hi[i]); } } if (Number.isFinite(lo)) yrange = [lo, hi]; }
  const wrap = split ? h('div', { class: 'bcells' }, host) : host;
  const group = split ? 'bsplit:' + p.id : 'b:' + meta.x_label;
  // follow: keep the zoom width, slide the window to the newest x
  if (p.view.follow) {
    const g = ensureGroup(group);
    let mx = -Infinity; for (const t of traces) for (const x of t.xs) if (Number.isFinite(x)) mx = Math.max(mx, x);
    if (g.view && Number.isFinite(mx)) { const span = g.view.x1 - g.view.x0; g.view = { x0: mx - span, x1: mx }; }
  }
  const nTr = traces.filter((t) => !hiddenTrace(p, t)).length;
  const oneMetric = new Set(traces.map((t) => t.metric)).size === 1;
  const shortName = (t) => (oneMetric && !t.metric.includes(' = ') ? t.label.replace(':' + t.metric, '') : t.label);
  const anyBand = traces.some((t) => t.band);
  const now = Date.now();
  for (const cell of cells) {
    const ct = traces.map((t, i) => ({ t, i })).filter(({ t }) => (t.cell || '') === cell);
    const cbox = split ? h('div', { class: 'bcell' }, wrap) : wrap;
    if (split) h('div', { class: 'ctitle' }, cbox, cell || '(none)');
    const series = ct.map(({ t, i }, k) => {
      const f = fcs.get(t.label);
      return {
        name: shortName(t) + (meta.by ? `  [${meta.by}=${t.color}]` : ''), trace: t, color: colour(i),
        area: k === 0 && !meta.by && !anyBand && nTr <= 6 && !split, xs: Array.from(t.xs), ys: Array.from(t.ys), raw: t.raw,
        band: t.band ? { lo: t.lo, hi: t.hi } : null, hidden: hiddenTrace(p, t), y2: p.view.y2 === t.label,
        ghost: f && !f.none ? { xs: f.xs, ys: f.ys, lo: f.lo, hi: f.hi } : null,
        fresh: t.runs.some((r) => now - (U.fresh.get(p.id + '\u0001' + r) || 0) < 20000),
      };
    });
    if (!series.length) continue;
    O.chart2d(cbox, {
      title: titleOf(p) + (cell ? ' / ' + cell : ''), series, ylog: meta.ylog, yrange: split ? yrange : null,
      xlabel: meta.x_label, xfmt: meta.x_label === 'time' ? O.fmtTime : O.fmtTick, group,
      legendHead: series.length > 6 ? `${series.length} series` : '',
      decorateLegend: (tbl, ss) => decorateLegend(p, tbl, ss, cbox),
      onview: () => { if (p.view.follow) { p.view.follow = false; save(); const b = el.querySelector('.bbar button[aria-pressed]'); if (b) { b.textContent = '> follow'; b.classList.remove('on'); b.setAttribute('aria-pressed', 'false'); b.title = 'follow paused by zooming; click to resume'; } } },
      overlay: (api) => drawOverlay(p, api, series, fcs, isStep, fc),
    });
  }
  if (key) {
    const k = h('div', { class: 'bkey' }, foot);
    k.append(`colour: ${meta.by} `);
    if (key.numeric) { k.append(O.fmt(key.lo)); const g = h('span', { class: 'grad' }, k); g.style.background = `linear-gradient(to right, ${O.cssv('--c2')}, ${O.cssv('--c3')})`; k.append(O.fmt(key.hi) + (key.log ? ' (log scale)' : '')); }
    else key.cats.forEach((c, i) => { const s = h('span', { class: 'sw-line' }, k); s.style.borderTopColor = `var(${PAL[i % PAL.length]})`; k.append(c + ' '); });
  }
  // forecast badge + ETA
  if (fc) {
    const first = traces.find((t) => !hiddenTrace(p, t) && fcs.get(t.label));
    const f = first && fcs.get(first.label);
    const bt = Dt.badgeText(f);
    const b = el.querySelector('button.badge');
    if (b) { b.textContent = bt.text + (fcs.size > 1 ? ` +${fcs.size - 1}` : ''); b.classList.toggle('noskill', !bt.skill); b.title = [...fcs.entries()].map(([l, g]) => `${l}: ${Dt.badgeText(g).text}${g && g.cfg ? '  ' + g.cfg : ''}`).join('\n') + '\nclick for forecast settings'; }
    if (p.view.fc.target !== null && p.view.fc.target !== undefined) {
      for (const t of traces.filter((t) => !hiddenTrace(p, t)).slice(0, 4)) {
        const e = Dt.etaFor(A, t, fcs.get(t.label), p.view.fc.target);
        h('div', { class: 'beta' }, foot, `${t.label}: ${Dt.etaText(e, p.view.fc.target, O.fmt)}`);
      }
    }
    if (p.view.fc.window) h('div', { class: 'bnote' }, foot, `forecast fitted on step ${fmtStep(p.view.fc.window[0])} to ${fmtStep(p.view.fc.window[1])}`);
  } else if (meta.forecast && !isStep) h('div', { class: 'bnote' }, foot, 'forecasts are drawn on the step axis only');
  if (meta.downsampled) h('div', { class: 'bnote' }, foot, `long series block-averaged to ${MAX_POINTS} points (longest ${meta.downsampled}); legend statistics are over block means`);
  for (const w of meta.warnings) h('div', { class: 'bnote' }, foot, w);
  // events for the overlay markers (fetched lazily, then redraw once)
  if (U.settings.events && isStep) {
    const ids = [...new Set(traces.flatMap((t) => t.runs))].filter((r) => !U.events.has(r));
    if (ids.length) { await Promise.all(ids.map(eventsOf)); if (tok === U.renderTok && el.isConnected) fillPanel(p, el, tok); }
  }
}
function ensureGroup(name) { return O.zoomGroup(name); }
const hiddenTrace = (p, t) => p.view.hidden.includes(t.label) || (p.view.solo && p.view.solo !== t.label);
function defaultHorizon(traces) {
  let lo = Infinity, hi = -Infinity, planned = 0;
  for (const t of traces) { for (const x of t.xs) if (Number.isFinite(x)) { lo = Math.min(lo, x); hi = Math.max(hi, x); } for (const r of t.runs) { const run = O.S.byId.get(r); if (run) for (const k of ['steps', 'total_steps', 'max_steps', 'train_steps', 'num_steps']) if (run.config[k] > hi) planned = Math.max(planned, run.config[k]); } }
  return planned || hi + 0.5 * (hi - lo);
}
function hexRgb(hx) { hx = hx.replace('#', ''); if (hx.length === 3) hx = hx.split('').map((c) => c + c).join(''); const n = parseInt(hx, 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
function mixRgb(a, b, t) { const c = a.map((x, i) => Math.round(x + (b[i] - x) * Math.min(1, Math.max(0, t)))); return `rgb(${c[0]},${c[1]},${c[2]})`; }

function decorateLegend(p, tbl, ss, cbox) {
  if (!tbl) return;
  const compact = ss.length > 8;
  tbl.classList.toggle('compact', compact);
  const hr = tbl.rows[0];
  [...hr.cells].forEach((c, i) => { if (i >= 2 && i <= 4) c.classList.add('st'); });
  hr.insertBefore(h('th', {}), hr.cells[1]);
  h('th', {}, hr, '');
  const svgs = cbox.querySelectorAll('svg');
  const svg = svgs[svgs.length - 1];
  const dim = (si) => { if (!svg) return; svg.querySelectorAll('[data-si]').forEach((e) => { e.style.opacity = si === null || +e.dataset.si === si ? '' : '0.15'; }); };
  ss.forEach((s, i) => {
    const tr = tbl.rows[i + 1]; if (!tr) return;
    const t = s.trace;
    [...tr.cells].forEach((c, j) => { if (j >= 2 && j <= 4) c.classList.add('st'); });
    if (s.hidden) tr.classList.add('off');
    if (s.fresh) tr.classList.add('fresh');
    const sw = tr.cells[0]; sw.title = 'drag to another panel, onto the canvas, or out of the board to remove';
    const nm = tr.cells[1]; nm.innerHTML = '';
    // live dot: age of the newest row of the run(s)
    const runs = t.runs.map((r) => O.S.byId.get(r)).filter(Boolean);
    const ages = runs.map((r) => Dt.liveAge(r));
    const newest = ages.filter((a) => Number.isFinite(a.age)).sort((a, b) => a.age - b.age)[0];
    const nb = btn(nm, '', `${t.label}: click to solo, shift-click to hide or show`, (ev) => {
      if (ev.shiftKey) commit('', () => { const hset = new Set(p.view.hidden); hset.has(t.label) ? hset.delete(t.label) : hset.add(t.label); p.view.hidden = [...hset]; p.view.solo = null; });
      else commit('', () => { p.view.solo = p.view.solo === t.label ? null : t.label; p.view.hidden = p.view.hidden.filter((x) => x !== t.label); });
    });
    nb.setAttribute('aria-pressed', p.view.solo === t.label);
    if (newest && !newest.finished && newest.age < 6 * 3600) {
      const d = h('span', { class: 'ldot', title: `last row ${O.fmtTime(newest.age)} ago${newest.stale ? ' (stale)' : ''}` }, nb);
      d.style.background = newest.stale ? 'var(--c3)' : 'var(--c1)';
    }
    nb.append(s.name);
    if (s.fresh) h('span', { class: 'tagnew', title: 'run appeared since the last update' }, nb, 'new');
    tr.insertBefore(h('td', { class: 'al' }, null, t.letter), nm);
    const ctl = h('td', { class: 'ctl' }, tr);
    const r = btn(ctl, 'R', s.y2 ? 'back to the left axis' : 'plot on the right-hand axis (one series per panel)', () => commit('', () => { p.view.y2 = p.view.y2 === t.label ? null : t.label; }), s.y2 ? 'on' : '');
    r.setAttribute('aria-pressed', !!s.y2);
    const single = t.runs.length === 1 && !t.metric.includes(' = ');
    btn(ctl, 'x', `remove ${t.label} from this panel`, () => removeTrace(p, { letter: t.letter, run: t.runs[0], single, label: t.label }));
    tr.addEventListener('pointerenter', () => dim(s.si));
    tr.addEventListener('pointerleave', () => dim(null));
    const pay = () => ({ type: 'trace', pid: p.id, letter: t.letter, run: t.runs[0], single, label: t.label, entries: single ? Bd.selectorFor([{ run: t.runs[0], metric: t.metric }]) : t.metric.includes(' = ') ? null : Bd.selectorFor(t.runs.map((x) => ({ run: x, metric: t.metric }))) });
    draggable(sw, pay);
  });
}

// overlay: event and spike markers, previous forecasts, and the forecast handles
function drawOverlay(p, api, series, fcs, isStep, fc) {
  const { g, X, Y, L, T, pw, ph, x0, x1, ty } = api;
  const NSv = 'http://www.w3.org/2000/svg';
  const el = (tag, attrs, parent = g) => { const e = document.createElementNS(NSv, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); parent.appendChild(e); return e; };
  const clipX = (px) => Math.min(L + pw, Math.max(L, px));
  const inPlot = (px) => px >= L && px <= L + pw;
  const dataX = (px) => x0 + ((px - L) / pw) * (x1 - x0);
  const dataY = (py) => { const v = api.y0 + ((T + ph - py) / ph) * (api.y1 - api.y0); return ty(10) === 1 ? Math.pow(10, v) : v; };
  const vis = series.filter((s) => !s.hidden);
  if (isStep && U.settings.events) {
    const evs = new Map();
    for (const s of vis) for (const r of s.trace.runs) for (const e of U.events.get(r) || []) { const k = e.step; if (!evs.has(k)) evs.set(k, []); evs.get(k).push(`${r}: ${e.text}`); }
    for (const [step, texts] of evs) {
      const px = X(step); if (!inPlot(px)) continue;
      const ln = el('line', { x1: px, x2: px, y1: T, y2: T + ph, style: 'stroke:var(--c4);stroke-width:1;stroke-dasharray:1 3;pointer-events:none' });
      void ln;
      const m = el('path', { d: `M${px - 3},${T}L${px + 3},${T}L${px},${T + 5}Z`, style: 'fill:var(--c4);cursor:help' });
      el('title', {}, m).textContent = `step ${step}\n` + [...new Set(texts)].slice(0, 6).join('\n');
    }
  }
  if (U.settings.spikes) {
    for (const s of vis) {
      const sp = Dt.spikesOf(A, s.trace);
      for (const i of sp) {
        const px = X(s.trace.xs[i]), v = ty(s.trace.raw[i]); if (!inPlot(px) || !Number.isFinite(v)) continue;
        const py = Math.max(T + 6, Y(v) - 6);
        const m = el('path', { d: `M${px - 3.5},${py - 6}L${px + 3.5},${py - 6}L${px},${py}Z`, style: `fill:var(--c3);stroke:var(--bg);stroke-width:0.5;cursor:help` });
        el('title', {}, m).textContent = `loss spike: ${s.trace.label}\nstep ${O.fmt(s.trace.steps[i])}, value ${O.fmt(s.trace.raw[i])} (above the median of the previous 10 points by > 6 robust sd)`;
      }
    }
  }
  if (!fc) return;
  // previous forecasts: faint dotted traces, so forecast vs reality stays visible
  for (const s of vis) {
    const hist = U.fcHist.get(p.id + '\u0001' + s.trace.label);
    if (!hist || !hist.prev) continue;
    const f = hist.prev; const pts = [];
    for (let i = 0; i < f.xs.length; i++) { const v = ty(f.ys[i]); if (Number.isFinite(v)) pts.push(`${X(f.xs[i]).toFixed(1)},${Y(v).toFixed(1)}`); }
    if (pts.length > 1) { const pl = el('polyline', { points: pts.join(' '), style: `fill:none;stroke:var(${s.color});stroke-width:1;stroke-opacity:0.35;stroke-dasharray:1 2;pointer-events:none` }); void pl; }
  }
  // horizon handle at the end of the forecast: drag right past the data to extend it
  const fx = [...fcs.values()].find((f) => f && !f.none);
  const lastX = fx ? fx.lastX : Math.max(...vis.flatMap((s) => Array.from(s.trace.xs).filter(Number.isFinite)));
  const hz = fx ? fx.xs[fx.xs.length - 1] : (fc.to || lastX);
  const hg = el('g', { style: 'cursor:ew-resize' });
  const hx = clipX(X(hz));
  const hl = el('line', { x1: hx, x2: hx, y1: T, y2: T + ph, style: 'stroke:var(--c2);stroke-width:1;stroke-dasharray:4 2' }, hg);
  const tab = el('rect', { x: hx - 5, y: T + ph - 18, width: 10, height: 18, style: 'fill:var(--c2);stroke:var(--bg)' }, hg);
  const hlab = el('text', { x: hx - 7, y: T + ph - 5, 'text-anchor': 'end', style: 'font:10px var(--mono);fill:var(--c2)' }, hg);
  hlab.textContent = 'to ' + fmtStep(hz);
  el('title', {}, tab).textContent = 'forecast horizon: drag right past the plot to extend, left to shorten';
  handle(tab, api, (pt) => { const v = Math.max(lastX * 1.001, dataX(pt.x)); const px = clipX(X(v)); hl.setAttribute('x1', px); hl.setAttribute('x2', px); tab.setAttribute('x', px - 5); hlab.setAttribute('x', px - 7); hlab.textContent = 'to ' + fmtStep(v); return v; },
    (v) => setStage(p, 'forecast', `forecast to ${fmtStep(niceRound(v))}${fc.method && fc.method !== 'auto' ? ' ' + fc.method : ''}`, 'forecast horizon'));
  // fit-window bracket under the plot
  const xsAll = vis.flatMap((s) => Array.from(s.trace.xs).filter(Number.isFinite));
  const dmin = Math.min(...xsAll), dmax = Math.max(...xsAll);
  let [wa, wb] = p.view.fc.window || [dmin, dmax];
  const by = T + ph + 4;
  const bl = el('line', { x1: clipX(X(wa)), x2: clipX(X(wb)), y1: by, y2: by, style: 'stroke:var(--c2);stroke-width:2' });
  const mk = (v, which) => {
    const e = el('rect', { x: clipX(X(v)) - 3, y: by - 4, width: 6, height: 8, style: 'fill:var(--bg);stroke:var(--c2);stroke-width:1.5;cursor:ew-resize' });
    el('title', {}, e).textContent = `forecast fit window ${which}: drag to exclude ${which === 'start' ? 'warmup' : 'the tail'} from the fit`;
    handle(e, api, (pt) => { let v2 = Math.min(dmax, Math.max(dmin, dataX(pt.x))); if (which === 'start') { v2 = Math.min(v2, wb); wa = v2; } else { v2 = Math.max(v2, wa); wb = v2; } e.setAttribute('x', clipX(X(v2)) - 3); bl.setAttribute('x1', clipX(X(wa))); bl.setAttribute('x2', clipX(X(wb))); return v2; },
      () => commit('forecast fit window', () => { p.view.fc.window = wa <= dmin && wb >= dmax ? null : [wa, wb]; }));
  };
  mk(wa, 'start'); mk(wb, 'end');
  // target line with ETA, or a tab to place one
  const tgt = p.view.fc.target;
  if (tgt !== null && tgt !== undefined && Number.isFinite(ty(tgt))) {
    const py = Math.min(T + ph, Math.max(T, Y(ty(tgt))));
    const tg = el('g', { style: 'cursor:ns-resize' });
    const tl = el('line', { x1: L, x2: L + pw, y1: py, y2: py, style: 'stroke:var(--c4);stroke-width:1;stroke-dasharray:6 3' }, tg);
    const tr = el('rect', { x: L + 2, y: py - 7, width: 58, height: 14, style: 'fill:var(--bg);stroke:var(--c4)' }, tg);
    const tt = el('text', { x: L + 6, y: py + 3, style: 'font:10px var(--mono);fill:var(--c4)' }, tg);
    tt.textContent = 'target ' + O.fmt(tgt);
    el('title', {}, tr).textContent = 'target: drag up or down; the ETA readout is under the plot';
    handle(tr, api, (pt) => { const yy = Math.min(T + ph, Math.max(T, pt.y)); const v = dataY(yy); for (const e of [tl]) { e.setAttribute('y1', yy); e.setAttribute('y2', yy); } tr.setAttribute('y', yy - 7); tt.setAttribute('y', yy + 3); tt.textContent = 'target ' + O.fmt(v); return v; },
      (v) => commit('forecast target', () => { p.view.fc.target = +v.toPrecision(4); }));
  } else {
    const tr = el('rect', { x: L + 2, y: T + 2, width: 50, height: 13, style: 'fill:var(--bg);stroke:var(--c4);cursor:pointer' });
    const tt = el('text', { x: L + 6, y: T + 12, style: 'font:10px var(--mono);fill:var(--c4);pointer-events:none' }); tt.textContent = '+ target';
    el('title', {}, tr).textContent = 'place a target line for an ETA readout (also in the forecast settings)';
    tr.addEventListener('click', () => setTarget(p));
  }
}
function niceRound(v) { const e = Math.pow(10, Math.floor(Math.log10(v)) - 1); return Math.round(v / e) * e; }
function setTarget(p, v) {
  if (v === undefined) {
    const r = U.results.get(p.id); const t = r && r.res.ok && r.res.traces.find((t) => !hiddenTrace(p, t));
    if (!t) return;
    const fin = Array.from(t.raw).filter(Number.isFinite); const last = fin[fin.length - 1], lo = Math.min(...fin), hi = Math.max(...fin);
    v = +(last - 0.1 * (hi - lo || Math.abs(last) || 1) * (fin[0] > last ? 1 : -1)).toPrecision(3);
  }
  commit('forecast target', () => { p.view.fc.target = v; });
}
// drag an SVG handle with pointer capture; onMove(pt) returns the value, onEnd(value) commits
function handle(e, api, onMove, onEnd) {
  e.addEventListener('pointerdown', (ev) => {
    if (ev.button !== 0) return;
    ev.stopPropagation(); ev.preventDefault();
    try { e.setPointerCapture(ev.pointerId); } catch (er) { /* synthetic events */ }
    let val = null, moved = false;
    const mv = (m) => { if (m.pointerId !== ev.pointerId) return; moved = true; val = onMove(api.local(m)); };
    const up = (m) => { if (m.pointerId !== ev.pointerId) return; document.removeEventListener('pointermove', mv); document.removeEventListener('pointerup', up); if (moved && val !== null) onEnd(val); };
    document.addEventListener('pointermove', mv); document.addEventListener('pointerup', up);
  });
}

// ------------------------------------------------------------ forecast settings (popover on the badge, and the inspector)
function forecastControls(box, p) {
  const st = Bd.getStage(A, p.q, 'forecast');
  const m = st ? st.match(/^forecast(?: to (\S+))?(?: (\S+))?$/) : null;
  const to = m && m[1] ? m[1] : '', method = m && m[2] ? m[2] : 'auto';
  const write = (t, me) => setStage(p, 'forecast', `forecast${t ? ' to ' + t : ''}${me && me !== 'auto' ? ' ' + me : ''}`, 'forecast settings');
  let r = h('div', { class: 'row' }, box);
  const on = btn(r, st ? 'on' : 'off', st ? 'turn the forecast off' : 'turn the forecast on', () => (st ? setStage(p, 'forecast', null) : write('', 'auto')), st ? 'on' : '');
  on.setAttribute('aria-pressed', !!st);
  if (!st) return;
  r.append('to');
  const ti = h('input', { value: to, size: 6, placeholder: 'auto', 'aria-label': 'forecast horizon step', title: 'forecast horizon (step, e.g. 10k); empty = planned steps or +50%' }, r);
  ti.addEventListener('change', () => write(ti.value.trim(), method));
  r = h('div', { class: 'row' }, box);
  r.append('method');
  sel(r, [['auto', 'auto'], ['ridge', 'ridge'], ['powerlaw', 'power law'], ['last', 'last value']], method, (v) => write(to, v), 'forecast method: auto picks the method with skill and the lowest backtest error');
  r.append('band');
  sel(r, [['50', '50%'], ['80', '80%'], ['95', '95%']], String(p.view.fc.band), (v) => commit('', () => { p.view.fc.band = +v; }), 'band level');
  r = h('div', { class: 'row' }, box);
  r.append('pool');
  sel(r, [['none', 'none'], ['metric', 'same metric'], ['group', 'same letter']], p.view.fc.pool, (v) => commit('', () => { p.view.fc.pool = v; }), 'pool the ridge fit over other series in the panel');
  const lv = btn(r, p.view.fc.live ? 'live refit: on' : 'live refit: off', 'refit when new rows arrive; off freezes the current forecast', () => commit('', () => {
    p.view.fc.live = !p.view.fc.live;
    if (!p.view.fc.live) { const fr = {}; for (const [k, v] of U.fcHist) if (k.startsWith(p.id + '\u0001')) fr[k.slice(p.id.length + 1)] = v.cur; p.view.fc.frozen = fr; } else p.view.fc.frozen = null;
  }), p.view.fc.live ? 'on' : '');
  lv.setAttribute('aria-pressed', !!p.view.fc.live);
  r = h('div', { class: 'row' }, box);
  r.append('target');
  const tg = h('input', { value: p.view.fc.target ?? '', size: 7, 'aria-label': 'target value', title: 'target value for the ETA readout' }, r);
  tg.addEventListener('change', () => { const v = tg.value.trim() === '' ? null : +tg.value; if (v === null || Number.isFinite(v)) commit('forecast target', () => { p.view.fc.target = v; }); });
  btn(r, 'place', 'place a target near the current value', () => setTarget(p));
  r = h('div', { class: 'row' }, box);
  r.append('fit window');
  const w = p.view.fc.window || ['', ''];
  const wa = h('input', { value: w[0] === '' ? '' : Math.round(w[0]), size: 6, placeholder: 'start', 'aria-label': 'fit window start step' }, r);
  const wb = h('input', { value: w[1] === '' ? '' : Math.round(w[1]), size: 6, placeholder: 'end', 'aria-label': 'fit window end step' }, r);
  const setw = () => commit('forecast fit window', () => { const a = wa.value.trim() === '' ? -Infinity : +wa.value, b = wb.value.trim() === '' ? Infinity : +wb.value; p.view.fc.window = a === -Infinity && b === Infinity ? null : [a, b]; });
  wa.addEventListener('change', setw); wb.addEventListener('change', setw);
  btn(r, 'all', 'fit on all points', () => commit('forecast fit window', () => { p.view.fc.window = null; }));
}
function forecastPopover(anchor, p) {
  const m = popover(anchor, 'bpop');
  h('h3', {}, m, 'forecast');
  forecastControls(m, p);
  h('div', { class: 'bnote' }, m, 'Ridge: one-step ridge on lag windows, hyperparameters chosen on a rolling-origin backtest (SearchCast style). Power law: y = a + b·x^-c, backtested on the last quarter. Auto uses the one with skill and the lower test error; with no skill the ghost is the last value. On the plot: drag the horizon tab, the bracket under the axis (fit window), and the target line.');
}

// ------------------------------------------------------------ inspector (selected panel only)
function renderInspector() {
  const box = U.els.insp; if (!box) return;
  box.innerHTML = '';
  const p = U.selPanel && findPanel(U.selPanel);
  if (!p) { h('div', { class: 'bnote' }, box, 'Select a panel to see its settings here.'); return; }
  h('h3', {}, box, 'panel: ' + titleOf(p));
  let r = h('div', { class: 'row' }, box);
  r.append('title');
  const ti = h('input', { value: p.title, placeholder: titleOf({ ...p, title: '' }), 'aria-label': 'panel title' }, r);
  ti.addEventListener('change', () => commit('rename panel', () => { p.title = ti.value.trim(); }));
  r = h('div', { class: 'row' }, box);
  const qi = h('input', { class: 'q', value: p.q, spellcheck: 'false', 'aria-label': 'query', title: 'query text; Enter applies' }, r);
  const qerr = h('div', {}, box);
  const applyQ = () => { qerr.innerHTML = ''; const x = parseQ(qi.value); if (!x.ok) { errLine(qerr, qi.value, x.error, x.start, x.end); return; } if (x.canonical !== p.q) commit('edit query', () => { p.q = x.canonical; }); };
  qi.addEventListener('keydown', (ev) => { if (ev.key === 'Enter') applyQ(); });
  btn(r, 'apply', 'apply the query', applyQ);
  const runs = panelRuns(p).length ? panelRuns(p) : O.selectedRuns();
  const fields = fieldsOf(runs);
  const vf = varyingFields(runs);
  const stageVal = (k) => Bd.getStage(A, p.q, k);
  r = h('div', { class: 'row' }, box);
  const lg = !!stageVal('log');
  const lb = btn(r, lg ? 'log y' : 'lin y', 'y scale: linear or log10', () => setStage(p, 'log', lg ? null : 'log'), lg ? 'on' : '');
  lb.setAttribute('aria-pressed', lg);
  r.append('x');
  const xs = (stageVal('x') || 'x step').slice(2);
  const ms = new Set(); for (const run of runs) for (const k of Object.keys(run.metrics)) ms.add(k);
  sel(r, [['step', 'step'], ['time', 'time'], ['tokens', 'tokens'], ...[...ms].sort().map((m) => [m, m])].filter((v, i, a) => a.findIndex((w) => w[0] === v[0]) === i), xs, (v) => setStage(p, 'x', v === 'step' ? null : `x ${v}`), 'x axis: step, time, tokens, or any metric joined on step');
  r.append('ema');
  const ema = (stageVal('ema') || 'ema 0').slice(4);
  const ei = h('input', { value: ema, size: 4, 'aria-label': 'ema weight', title: 'EMA smoothing in [0, 1); legend statistics stay raw' }, r);
  ei.addEventListener('change', () => { const v = +ei.value; setStage(p, 'ema', v > 0 && v < 1 ? `ema ${ei.value.trim()}` : null); });
  r = h('div', { class: 'row' }, box);
  r.append('norm');
  sel(r, [['', 'none'], ['z', 'z-score'], ['minmax', 'min-max'], ['first', '/ first']], (stageVal('norm') || 'norm ').slice(5), (v) => setStage(p, 'norm', v ? `norm ${v}` : null), 'normalise each series');
  r.append('align');
  sel(r, [['intersection', 'overlap'], ['union', 'union'], ['exact', 'exact']], (stageVal('align') || 'align intersection').slice(6), (v) => setStage(p, 'align', v === 'intersection' ? null : `align ${v}`), 'step alignment for formulas, baselines, seeds and metric x axes; never extrapolates');
  r = h('div', { class: 'row' }, box);
  r.append('colour by');
  sel(r, [['', 'series'], ['run', 'run'], ['metric', 'metric'], ...vf.map((f) => [f, f]), ...fields.filter((f) => !vf.includes(f)).map((f) => [f, f + ' (constant)'])], (stageVal('by') || 'by ').slice(3), (v) => setStage(p, 'by', v ? `by ${v}` : null), 'colour by a config value (gradient for numbers)');
  r = h('div', { class: 'row' }, box);
  r.append('split');
  sel(r, [['', 'none'], ['metric', 'metric'], ['run', 'run'], ['letter', 'letter'], ...vf.map((f) => [f, f])], (stageVal('split') || 'split ').slice(6), (v) => setStage(p, 'split', v ? `split ${v}` : null), 'small multiples with locked axes');
  r = h('div', { class: 'row' }, box);
  r.append('seeds');
  const agg = stageVal('agg') || '';
  sel(r, [['', 'separate'], ['mean±sd over seed', 'mean±sd over seed'], ['median over seed', 'median (min-max band)'], ...vf.filter((f) => f !== 'seed').map((f) => [`mean±sd over ${f}`, `mean±sd over ${f}`])].filter((v, i, a) => a.findIndex((w) => w[0] === v[0]) === i), agg, (v) => setStage(p, 'agg', v || null), 'aggregate runs that differ only in this field into a line plus band');
  r = h('div', { class: 'row' }, box);
  r.append('baseline');
  const bs = stageVal('baseline') || '';
  const bm = bs.match(/^baseline "?([^"]*)"?( ratio)?$/);
  const brun = bm ? bm[1] : '', bratio = !!(bm && bm[2]);
  sel(r, [['', 'none'], ...runs.map((x) => [x.id, x.id])], brun, (v) => setStage(p, 'baseline', v ? `baseline ${Bd.quote(v)}${bratio ? ' ratio' : ''}` : null), 'plot relative to one run');
  if (brun) sel(r, [['diff', 'difference'], ['ratio', 'ratio']], bratio ? 'ratio' : 'diff', (v) => setStage(p, 'baseline', `baseline ${Bd.quote(brun)}${v === 'ratio' ? ' ratio' : ''}`), 'difference or ratio');
  h('h3', {}, box, 'forecast');
  forecastControls(box, p);
  r = h('div', { class: 'row' }, box);
  btn(r, 'show all series', 'clear solo and hidden series', () => commit('', () => { p.view.hidden = []; p.view.solo = null; }));
  btn(r, 'expand', 'open in the focus view', () => openFocus(p.id));
  btn(r, 'delete panel', 'delete this panel (undoable)', () => deletePanel(p));
}

// ------------------------------------------------------------ focus view (#panel-<id>)
function openFocus(id, fromHash) {
  const p = findPanel(id); if (!p) return;
  if (!U.focus) U.scrollY = window.scrollY;
  U.focus = id; U.selPanel = id;
  const f = U.els.focus;
  f.style.display = 'flex'; document.body.classList.add('bfocused');
  U.els.fbody.insertBefore(U.els.side, U.els.fmain); // dock the browser alongside
  if (!fromHash && location.hash !== '#panel-' + id) { history.pushState({ obsFocus: id }, '', '#panel-' + id); U.pushed = true; }
  render();
  setTimeout(() => U.els.fback.focus(), 0);
}
function closeFocus(fromHash) {
  if (!U.focus) return;
  U.focus = null;
  U.els.focus.style.display = 'none'; document.body.classList.remove('bfocused');
  U.els.root.insertBefore(U.els.side, U.els.root.firstChild);
  if (!fromHash && location.hash.startsWith('#panel-')) { if (U.pushed) { U.pushed = false; history.back(); } else history.replaceState(null, '', location.pathname + location.search); }
  render();
  window.scrollTo(0, U.scrollY || 0);
}
function syncHash() {
  const m = location.hash.match(/^#panel-([A-Za-z0-9-]+)$/);
  if (m && findPanel(m[1])) { if (U.focus !== m[1]) openFocus(m[1], true); }
  else if (U.focus) { U.pushed = false; closeFocus(true); }
}

// ------------------------------------------------------------ command palette (Ctrl/Cmd+K)
function openPalette() {
  if (document.querySelector('.bpal')) { closePalette(); return; }
  const bg = h('div', { class: 'bpal' }, document.body);
  bg.addEventListener('pointerdown', (ev) => { if (ev.target === bg) closePalette(); });
  const box = h('div', { class: 'box', role: 'dialog', 'aria-label': 'command palette' }, bg);
  const top = h('div', { class: 'top' }, box);
  const inp = h('input', { placeholder: 'command: forecast to 10k, split by seed, add loss, follow, new panel ...', spellcheck: 'false', 'aria-label': 'command' }, top);
  btn(top, 'close', 'close the command palette', closePalette);
  const list = h('div', { class: 'list' }, box);
  let cur = 0, items = [];
  const ctx = () => ({ metrics: [...Dt.metricCounts(O.S.runs).keys()].sort(), fields: fieldsOf(O.S.runs) });
  const draw = () => {
    items = commands(inp.value, ctx()); list.innerHTML = '';
    cur = Math.min(cur, Math.max(0, items.length - 1));
    items.forEach((c, i) => btn(list, c.label, c.label, () => { closePalette(); runAction(c.action); }, i === cur ? 'cur' : ''));
    if (!items.length) h('div', { class: 'bnote', style: 'padding:6px' }, list, 'no matching command');
  };
  inp.addEventListener('input', () => { cur = 0; draw(); });
  inp.addEventListener('keydown', (ev) => {
    if (ev.key === 'ArrowDown') { cur = Math.min(cur + 1, items.length - 1); draw(); ev.preventDefault(); }
    else if (ev.key === 'ArrowUp') { cur = Math.max(cur - 1, 0); draw(); ev.preventDefault(); }
    else if (ev.key === 'Enter' && items[cur]) { const a = items[cur].action; closePalette(); runAction(a); }
  });
  draw(); inp.focus();
}
function closePalette() { document.querySelectorAll('.bpal').forEach((e) => e.remove()); }
function targetPanel() { return (U.focus && findPanel(U.focus)) || (U.selPanel && findPanel(U.selPanel)) || null; }
function runAction(a) {
  const p = targetPanel();
  const needP = () => { if (!p) { toast('select a panel first'); return false; } return true; };
  switch (a.type) {
    case 'stage': if (needP()) setStage(p, a.kind, a.text); break;
    case 'add': if (p) addTo(p, a.entries, a.entries); else addToNew(a.entries, a.entries); break;
    case 'new': { if (a.q) { const r = parseQ(a.q); if (!r.ok) { toast(`query error: ${r.error}`); break; } addToNew(r.canonical); } else if (U.sel.size) addToNew(selEntries(), `${U.sel.size} series`); else toast('pick series in the list first, or type "new panel <query>"'); break; }
    case 'follow': if (needP()) commit('', () => { p.view.follow = !p.view.follow; }); break;
    case 'editText': if (needP()) { U.editing.add(p.id); render(); } break;
    case 'expand': if (needP()) openFocus(p.id); break;
    case 'back': closeFocus(); break;
    case 'undo': undo(); break;
    case 'preset': applyPreset(a.name); break;
    case 'export': exportBoard(); break;
    case 'import': U.els.file.click(); break;
    default: break;
  }
}

// ------------------------------------------------------------ presets, import/export, settings
function historyPreset() {
  return Bd.presetHistory({ runIds: O.selectedRuns().map((r) => r.id), metrics: [...document.getElementById('metrics').selectedOptions].map((o) => o.value), x: document.getElementById('xaxis').value, ema: +document.getElementById('ema').value || 0 });
}
function autoPreset() {
  const runs = O.selectedRuns().length ? O.selectedRuns() : O.S.runs;
  const counts = Dt.metricCounts(runs);
  // demo runs stay out of the default board without pinning it to today's run list
  const filter = O.S.runs.some((r) => r.config && r.config.synthetic) && !runs.some((r) => r.config && r.config.synthetic) ? '{synthetic!=true}' : '';
  const tok = [...counts.keys()].find((m) => /tokens?_seen|^tokens$|num_tokens|consumed_tokens/.test(m));
  return Bd.autoBoard(counts, { replicates: Dt.hasReplicates(runs), hasTokens: tok || null, filter });
}
function defaultBoard() { return autoPreset() || historyPreset(); }
function applyPreset(name) {
  const b = name === 'auto' ? autoPreset() || historyPreset() : name === 'history' ? historyPreset() : name === 'all' ? Bd.presetAll([...Dt.metricCounts(O.selectedRuns()).keys()].sort(), O.selectedRuns().map((r) => r.id)) : { version: Bd.BOARD_VERSION, panels: [] };
  commit(`apply preset ${name}`, () => { U.board = b; U.selPanel = null; }, { toast: `board: ${name}` });
}
function exportBoard() {
  const blob = new Blob([Bd.serialize(U.board)], { type: 'application/json' });
  const a = h('a', { href: URL.createObjectURL(blob), download: `obs-board-${new Date().toISOString().slice(0, 10)}.json` }, document.body);
  a.click(); setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
}
function importBoard(file) {
  file.text().then((txt) => {
    let b; try { b = Bd.validBoard(JSON.parse(txt)); } catch (e) { toast('import failed: ' + (e.message || e)); return; }
    commit('import board', () => { U.board = b; U.selPanel = null; }, { toast: `imported ${b.panels.length} panels from ${file.name}` });
  });
}
function settingsPopover(anchor) {
  const m = popover(anchor, 'bpop');
  h('h3', {}, m, 'settings');
  const row = (key, label, title) => {
    const l = h('label', { class: 'row', title }, m);
    const c = h('input', { type: 'checkbox' }, l); c.checked = !!U.settings[key];
    l.append(label);
    c.addEventListener('change', () => { U.settings[key] = c.checked; saveSettings(); render(); });
  };
  row('palette', 'Ctrl+K command palette', 'the only global shortcut; off leaves Ctrl+K to the browser');
  row('events', 'event markers', 'vertical markers from the runs\' event logs (step axis)');
  row('spikes', 'loss-spike markers', 'upward spikes detected in wasm (> 6 robust sd above the trailing median)');
}

// ------------------------------------------------------------ mount
export function mount(obs) {
  O = obs; A = obs.A;
  const style = document.createElement('style'); style.textContent = CSS; document.head.appendChild(style);
  try { Object.assign(U.settings, JSON.parse(localStorage.getItem(LS_SET) || '{}')); } catch (e) { /* defaults */ }
  const root = U.els.root = document.getElementById('builder');
  root.innerHTML = '';
  // series browser
  const side = U.els.side = h('div', { class: 'bside', 'aria-label': 'series browser' }, root);
  const hd = h('div', { class: 'bhead' }, side);
  const f = h('input', { type: 'search', placeholder: 'filter series: run metric ...', 'aria-label': 'filter series', spellcheck: 'false' }, hd);
  f.addEventListener('input', () => { U.filter = f.value; renderBrowser(); });
  const r1 = h('div', { class: 'brow' }, hd);
  r1.append('group');
  sel(r1, [['metric', 'by metric'], ['run', 'by run']], U.group, (v) => { U.group = v; renderBrowser(); }, 'group the list');
  const ol = h('label', { title: 'only list runs selected in the Runs table' }, r1);
  const oc = h('input', { type: 'checkbox' }, ol); ol.append('selected runs');
  oc.addEventListener('change', () => { U.onlySel = oc.checked; renderBrowser(); });
  U.els.list = h('div', { class: 'blist' }, side);
  const ft = h('div', { class: 'bfoot' }, side);
  U.els.selinfo = h('span', {}, ft, '');
  U.els.selnew = btn(ft, 'new panel', 'new panel from the selected series', () => { if (U.sel.size) addToNew(selEntries(), `${U.sel.size} series`); });
  U.els.seladd = btn(ft, 'add to...', 'add the selected series to a panel', (ev) => { if (U.sel.size) addMenu(ev.currentTarget, selEntries(), `${U.sel.size} series`); });
  U.els.selclr = btn(ft, 'clear', 'clear the selection', () => { U.sel.clear(); renderBrowser(); });
  U.els.insp = h('div', { class: 'binsp', 'aria-label': 'panel inspector' }, side);
  // board
  const main = h('div', { class: 'bmain' }, root);
  const tool = h('div', { class: 'btool' }, main);
  const pl = h('label', { title: 'replace the board with a preset (undoable)' }, tool); pl.append('preset ');
  const ps = sel(pl, [['', '...'], ['auto', 'auto (naming conventions)'], ['history', 'history view'], ['all', 'all metrics'], ['blank', 'blank']], '', (v) => { if (v) { applyPreset(v); ps.value = ''; } }, 'board presets');
  btn(tool, 'reset to default', 'rebuild the default board (auto, else the History view of the selected runs); undoable', () => commit('reset to default', () => { U.board = defaultBoard(); U.selPanel = null; }, { toast: 'board reset to default' }));
  U.els.undo = btn(tool, 'undo', 'undo', undo);
  btn(tool, 'export', 'download this board as JSON', exportBoard);
  btn(tool, 'import', 'load a board JSON file', () => U.els.file.click());
  U.els.file = h('input', { type: 'file', accept: '.json,application/json', style: 'display:none', 'aria-hidden': 'true' }, tool);
  U.els.file.addEventListener('change', () => { if (U.els.file.files[0]) importBoard(U.els.file.files[0]); U.els.file.value = ''; });
  btn(tool, 'commands', 'command palette (Ctrl+K)', openPalette);
  btn(tool, 'settings', 'builder settings', (ev) => settingsPopover(ev.currentTarget));
  U.els.count = h('span', { class: 'msg' }, tool, '');
  U.els.canvas = h('div', { id: 'bcanvas', 'aria-label': 'board' }, main);
  // focus view
  const fv = U.els.focus = h('div', { class: 'bfocus', style: 'display:none', role: 'dialog', 'aria-label': 'panel focus view' }, document.body);
  const fbar = h('div', { class: 'fbar' }, fv);
  U.els.fback = btn(fbar, '< back to board', 'back to the board', () => closeFocus());
  U.els.ftitle = h('span', {}, fbar, '');
  h('span', { class: 'sp' }, fbar);
  btn(fbar, 'fullscreen', 'toggle browser fullscreen', () => {
    try { const pr = document.fullscreenElement ? document.exitFullscreen() : fv.requestFullscreen(); if (pr && pr.catch) pr.catch(() => toast('fullscreen not available here')); } catch (e) { toast('fullscreen not available here'); }
  });
  U.els.fbody = h('div', { class: 'fbody' }, fv);
  U.els.fmain = h('div', { class: 'fmain' }, U.els.fbody);
  fv.addEventListener('keydown', (ev) => { if (ev.key === 'Escape' && !ev.defaultPrevented && !/^(INPUT|SELECT|TEXTAREA)$/.test(ev.target.tagName)) { closeFocus(); ev.preventDefault(); } });
  // keys: Ctrl/Cmd+K palette (opt-out), Ctrl/Cmd+Z inside the board, Delete on a focused panel
  document.addEventListener('keydown', (ev) => {
    const a = document.activeElement;
    const typing = a && (a.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(a.tagName));
    if ((ev.ctrlKey || ev.metaKey) && !ev.shiftKey && !ev.altKey && ev.key.toLowerCase() === 'k' && U.settings.palette) { openPalette(); ev.preventDefault(); return; }
    const inBoard = a && a.closest && (a.closest('#builder') || a.closest('.bfocus'));
    if (!inBoard || typing) return;
    if ((ev.ctrlKey || ev.metaKey) && !ev.shiftKey && ev.key.toLowerCase() === 'z') { undo(); ev.preventDefault(); return; }
    if (ev.key === 'Delete' && !ev.ctrlKey && !ev.metaKey) { const pe = a.closest('.bpanel'); const p = pe && findPanel(pe.dataset.pid); if (p) { deletePanel(p); ev.preventDefault(); } }
  });
  // data
  U.load = Dt.makeLoader({ byId: O.S.byId, metricSeries: O.metricSeries, getJSON: O.getJSON, src: O.src, f32: O.f32 }, A);
  U.catalog = Dt.catalogText(O.S.runs);
  document.addEventListener('obs:runs-changed', (ev) => {
    U.catalog = Dt.catalogText(O.S.runs); U.dataVersion++;
    for (const id of (ev.detail && ev.detail.updated) || []) U.events.delete(id);
    renderBrowser(); render();
  });
  window.addEventListener('hashchange', syncHash);
  window.addEventListener('popstate', syncHash);
  let b = null;
  try { const s = localStorage.getItem(LS_BOARD); if (s) b = Bd.validBoard(JSON.parse(s)); } catch (e) { b = null; }
  if (!b) { try { const s = localStorage.getItem('obs-builder-layout-v1'); if (s) b = Bd.validBoard(JSON.parse(s)); } catch (e) { b = null; } }
  U.board = b || defaultBoard();
  renderBrowser(); render(); updateUndo();
  syncHash();
  // small API for other modules and tests
  return {
    board: () => U.board,
    setBoard: (x) => commit('set board', () => { U.board = Bd.validBoard(x); }),
    addPanel: (q) => addToNew(q),
    openFocus, closeFocus, render, runAction,
  };
}
