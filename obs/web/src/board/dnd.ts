// Builder drag and drop (pointer events: mouse, pen, touch). Starts after 4 px (mouse/pen) or a
// 300 ms hold (touch, so lists still scroll). Every drop has a click path too: the + buttons in the
// browser, the x on legend rows and chips, the panel menu. Targets are found by hit-testing:
// .bpanel[data-pid] (overlay / merge / reorder by edge), [data-canvas] (new panel / move to end),
// .bnew (new panel), anywhere else for a legend row = remove.
import { h } from '@/lib/dom';
import { hideTip } from '@/lib/dom';
import {
  findPanel, titleOf, addToNew, addTo, removeTrace, moveTrace, mergePanel, movePanel, type TraceRef,
} from '@/stores/board';
import { closePop, setDragging } from '@/stores/ui';

export type Payload =
  | { type: 'series'; entries: string; q?: string; label: string }
  | { type: 'trace'; pid: string; ref: TraceRef; label: string }
  | { type: 'panel'; pid: string; label: string };
interface Target { kind: 'add' | 'merge' | 'before' | 'after' | 'end' | 'new' | 'remove' | 'none'; el?: Element | null; pid?: string; cls?: string; text: string }
interface Drag { x: number; y: number; id: number; active: boolean; payload: () => Payload | null; data?: Payload; el: Element; touch: boolean; timer: ReturnType<typeof setTimeout> | null; ghost?: HTMLDivElement; t?: Target | null }

let DR: Drag | null = null;
const HL = ['dz-add', 'dz-merge', 'dz-before', 'dz-after', 'dz-new', 'dz-trash'];

export function draggable(el: Element, payload: () => Payload | null): void {
  el.addEventListener('pointerdown', (e) => {
    const ev = e as PointerEvent;
    if (ev.button !== 0 || DR) return;
    if ((ev.target as Element).closest('input, select')) return;
    const touch = ev.pointerType === 'touch';
    const st: Drag = { x: ev.clientX, y: ev.clientY, id: ev.pointerId, active: false, payload, el, touch, timer: null };
    DR = st;
    if (touch) st.timer = setTimeout(() => { if (DR === st) activate(st, st.x, st.y); }, 300);
  });
}
function activate(st: Drag, x: number, y: number): void {
  const d = st.payload();
  if (!d) { DR = null; return; }
  st.data = d; st.active = true;
  st.ghost = h('div', { class: 'bghost' }, document.body, '');
  document.body.classList.add('bdragging'); setDragging(true);
  hideTip(); closePop();
  move(x, y);
}
function clearHl(): void { document.querySelectorAll('.' + HL.join(',.')).forEach((e) => e.classList.remove(...HL)); }
function target(x: number, y: number): Target | null {
  const d = DR!.data!;
  const el = document.elementFromPoint(x, y);
  const pel = el && el.closest<HTMLElement>('.bpanel[data-pid]');
  const canvas = el && el.closest<HTMLElement>('[data-canvas]');
  if (pel) {
    const p = findPanel(pel.dataset.pid); if (!p) return null;
    if (d.type === 'panel') {
      if (p.id === d.pid) return null;
      const r = pel.getBoundingClientRect(), fx = (x - r.left) / r.width;
      if (fx < 0.22) return { kind: 'before', el: pel, pid: p.id, cls: 'dz-before', text: `move before "${titleOf(p)}"` };
      if (fx > 0.78) return { kind: 'after', el: pel, pid: p.id, cls: 'dz-after', text: `move after "${titleOf(p)}"` };
      return { kind: 'merge', el: pel, pid: p.id, cls: 'dz-merge', text: `merge into "${titleOf(p)}"` };
    }
    if (d.type === 'trace' && d.pid === p.id) return { kind: 'none', text: 'drop outside this panel to remove' };
    return { kind: 'add', el: pel, pid: p.id, cls: 'dz-add', text: `${d.type === 'trace' ? 'move' : 'overlay'} onto "${titleOf(p)}"` };
  }
  if (canvas) {
    if (d.type === 'panel') return canvas.dataset.canvas === 'board' ? { kind: 'end', el: canvas, cls: 'dz-new', text: 'move to the end' } : null;
    const nw = el!.closest('.bnew');
    return { kind: 'new', el: nw || canvas, cls: 'dz-new', text: d.type === 'trace' ? 'move to a new panel' : 'new panel' };
  }
  if (d.type === 'trace') return { kind: 'remove', el: el && el.closest('.bside'), cls: 'dz-trash', text: 'release to remove from the panel' };
  return null;
}
function move(x: number, y: number): void {
  const st = DR; if (!st || !st.active || !st.ghost) return;
  const t = target(x, y); st.t = t;
  clearHl(); if (t && t.el && t.cls) t.el.classList.add(t.cls);
  st.ghost.textContent = `${st.data!.label}\n${t ? t.text : 'no drop target here'}`;
  st.ghost.style.left = Math.min(x + 14, innerWidth - st.ghost.offsetWidth - 4) + 'px'; st.ghost.style.top = (y + 14) + 'px';
}
function endDrag(cancel: boolean): void {
  const st = DR; DR = null; if (!st) return;
  if (st.timer) clearTimeout(st.timer);
  if (!st.active) return;
  st.ghost?.remove(); document.body.classList.remove('bdragging'); setDragging(false); clearHl();
  (st.el as HTMLElement).dataset.dragged = '1'; setTimeout(() => { delete (st.el as HTMLElement).dataset.dragged; }, 0); // swallow the click after a drag
  if (!cancel && st.t) drop(st.data!, st.t);
}
let wired = false;
export function wireDnd(): void {
  if (wired) return; wired = true;
  document.addEventListener('pointermove', (ev) => {
    const st = DR; if (!st || ev.pointerId !== st.id) return;
    if (!st.active) {
      const dist = Math.hypot(ev.clientX - st.x, ev.clientY - st.y);
      if (st.touch) { if (dist > 8) { if (st.timer) clearTimeout(st.timer); DR = null; } return; }
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
  document.addEventListener('click', (ev) => { const t = ev.target as Element; if (t.closest && t.closest('[data-dragged]')) { ev.stopPropagation(); ev.preventDefault(); } }, true);
}

function drop(d: Payload, t: Target): void {
  const tp = findPanel(t.pid);
  if (d.type === 'series') {
    if (t.kind === 'new') addToNew(d.q || d.entries, d.label);
    else if (t.kind === 'add' && tp) addTo(tp, d.entries, d.label);
    return;
  }
  if (d.type === 'trace') {
    const src = findPanel(d.pid); if (!src || t.kind === 'none') return;
    if (t.kind === 'remove') removeTrace(src, d.ref);
    else if (t.kind === 'new') moveTrace(src, d.ref, 'new');
    else if (tp) moveTrace(src, d.ref, tp);
    return;
  }
  const src = findPanel(d.pid); if (!src) return;
  if (t.kind === 'merge' && tp) mergePanel(src, tp);
  else if (t.kind === 'end') movePanel(src, 'end');
  else if ((t.kind === 'before' || t.kind === 'after') && tp) movePanel(src, t.kind, tp);
}
