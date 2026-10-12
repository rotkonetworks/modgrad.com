// Theme: auto (OS) / light / dark, persisted under 'obs-theme'. `themeVersion` bumps on any
// effective change so charts that bake colours (heatmaps, gradients) redraw.
import { createSignal } from 'solid-js';
import { lsGet, lsSet } from '@/lib/storage';

export type ThemeMode = 'auto' | 'light' | 'dark';
const MODES: ThemeMode[] = ['auto', 'light', 'dark'];
const saved = lsGet('obs-theme') as ThemeMode | null;
const [mode, setMode] = createSignal<ThemeMode>(saved && MODES.includes(saved) ? saved : 'auto');
const [version, setVersion] = createSignal(0);
export const themeMode = mode;
export const themeVersion = version;

function apply(): void {
  const m = mode();
  if (m === 'auto') delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme = m;
}
export function cycleTheme(): void {
  setMode(MODES[(MODES.indexOf(mode()) + 1) % MODES.length]);
  apply(); lsSet('obs-theme', mode()); setVersion((v) => v + 1);
}
export function initTheme(): void {
  apply();
  try { matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { if (mode() === 'auto') setVersion((v) => v + 1); }); } catch { /* old browsers */ }
}
