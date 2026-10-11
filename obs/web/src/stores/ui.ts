// Transient UI: toasts, one menu/popover at a time, the command palette, the current tab.
import { createSignal, type JSX } from 'solid-js';

export interface Toast { text: string; undo: boolean; id: number }
const [toastSig, setToast] = createSignal<Toast | null>(null);
export const toast = toastSig;
let toastTimer: ReturnType<typeof setTimeout> | undefined, toastSeq = 0;
export function showToast(text: string, undo = false): void {
  setToast({ text, undo, id: ++toastSeq });
  clearTimeout(toastTimer); toastTimer = setTimeout(() => setToast(null), 6000);
}
export const hideToast = (): void => { setToast(null); };

export interface MenuItem { label: string; title?: string; run: () => void }
export interface Pop { x: number; y: number; anchor: Element; kind: 'menu' | 'pop'; items?: MenuItem[]; render?: () => JSX.Element; label: string }
const [popSig, setPop] = createSignal<Pop | null>(null);
export const pop = popSig;
function place(anchor: Element): { x: number; y: number } {
  const r = anchor.getBoundingClientRect();
  return { x: Math.max(4, Math.min(r.left, innerWidth - 430)), y: Math.min(r.bottom + 2, innerHeight - 80) };
}
export function openMenu(anchor: Element, items: MenuItem[], label = 'menu'): void { setPop({ ...place(anchor), anchor, kind: 'menu', items, label }); }
export function openPopover(anchor: Element, render: () => JSX.Element, label = 'settings'): void { setPop({ ...place(anchor), anchor, kind: 'pop', render, label }); }
export const closePop = (): void => { setPop(null); };

const [paletteOpen, setPaletteOpen] = createSignal(false);
export { paletteOpen, setPaletteOpen };

export type Tab = 'board' | 'cross' | 'arrays' | 'runs';
export const TABS: [Tab, string][] = [['board', 'Board'], ['cross', 'Across runs'], ['arrays', 'Arrays / 3D'], ['runs', 'Runs']];

const [railOpen, setRailOpen] = createSignal(true);
export { railOpen, setRailOpen };

// file drop: the pointer DnD of the builder sets this so the page-wide file drop zone stays out
const [dragging, setDragging] = createSignal(false);
export { dragging, setDragging };
