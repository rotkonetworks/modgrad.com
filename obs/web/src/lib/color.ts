// Colour helpers over the theme tokens (src/app.css). Chart colours are CSS variables so a theme
// change restyles lines; scales that bake hex values are redrawn on theme change (see theme.ts).
export const PALETTE = ['--c1', '--c2', '--c3', '--c4', '--c5', '--c6'];
export const cssv = (v: string): string => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
export type RGB = [number, number, number];
export function hexToRgb(hx: string): RGB {
  hx = hx.replace('#', ''); if (hx.length === 3) hx = hx.split('').map((c) => c + c).join('');
  const n = parseInt(hx, 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
export function mix(a: RGB, b: RGB, t: number): RGB { return a.map((x, i) => Math.round(x + (b[i] - x) * t)) as RGB; }
export const rgb = (c: RGB): string => `rgb(${c[0]},${c[1]},${c[2]})`;
export function mixRgb(a: RGB, b: RGB, t: number): string { return rgb(mix(a, b, Math.min(1, Math.max(0, t)))); }

export interface Scale { (v: number): string | null; div: boolean; lo: number; hi: number }
/** sequential (one hue) or diverging when the range straddles 0 */
export function colorScale(lo: number, hi: number): Scale {
  const div = lo < 0 && hi > 0;
  const L = hexToRgb(cssv('--seq-lo')), H = hexToRgb(cssv('--seq-hi'));
  const N = hexToRgb(cssv('--div-neg')), M = hexToRgb(cssv('--div-mid')), P = hexToRgb(cssv('--div-pos'));
  const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1;
  const f = ((v: number) => {
    if (!Number.isFinite(v)) return null;
    if (div) return rgb(v < 0 ? mix(M, N, Math.min(1, -v / m)) : mix(M, P, Math.min(1, v / m)));
    return rgb(mix(L, H, hi > lo ? Math.min(1, Math.max(0, (v - lo) / (hi - lo))) : 0.5));
  }) as Scale;
  f.div = div; f.lo = div ? -m : lo; f.hi = div ? m : hi;
  return f;
}
export function tickColor(t: number, n: number): string {
  const L = hexToRgb(cssv('--seq-lo')), H = hexToRgb(cssv('--seq-hi'));
  return rgb(mix(mix(L, H, 0.25), H, n > 1 ? t / (n - 1) : 1));
}
