// Number and time formatting shared by the charts, legends and tables (RRD GPRINT style).
export function fmt(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return 'nan';
  if (!Number.isFinite(v)) return v > 0 ? 'inf' : '-inf';
  const a = Math.abs(v);
  if (a >= 1e9) return (v / 1e9).toFixed(2) + 'G';
  if (a >= 1e6) return (v / 1e6).toFixed(2) + 'M';
  if (a >= 1e4) return (v / 1e3).toFixed(1) + 'k';
  if (Number.isInteger(v) && a < 1e4) return String(v);
  if (a >= 100) return v.toFixed(1);
  if (a >= 1) return v.toFixed(3);
  if (a === 0) return '0';
  if (a >= 1e-3) return v.toFixed(4);
  return v.toExponential(2);
}
export function fmtTick(v: number): string {
  const a = Math.abs(v);
  if (a >= 1e9) return +(v / 1e9).toFixed(2) + 'G';
  if (a >= 1e6) return +(v / 1e6).toFixed(2) + 'M';
  if (a >= 1e3) return +(v / 1e3).toFixed(2) + 'k';
  return +v.toPrecision(4) + '';
}
/** RRD-ish elapsed labels */
export function fmtTime(s: number): string {
  if (!Number.isFinite(s)) return '';
  if (s < 120) return Math.round(s) + 's';
  if (s < 7200) return Math.round(s / 60) + 'm';
  const hh = Math.floor(s / 3600), mm = Math.round((s % 3600) / 60);
  return hh + 'h' + (mm ? String(mm).padStart(2, '0') : '');
}
export const fmtStep = (n: number): string => (n >= 1e6 ? +(n / 1e6).toPrecision(3) + 'M' : n >= 1e3 ? +(n / 1e3).toPrecision(3) + 'k' : String(Math.round(n)));
export function ago(s: number): string {
  if (!(s >= 0)) return '';
  if (s < 90) return Math.round(s) + 's';
  if (s < 5400) return Math.round(s / 60) + 'm';
  if (s < 172800) return Math.round(s / 3600) + 'h';
  return Math.round(s / 86400) + 'd';
}
/** base64 little-endian f32 block (array files) */
export function f32(b64: string): Float32Array {
  const s = atob(b64); const u = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i);
  return new Float32Array(u.buffer);
}
/** JSON number arrays (null = gap) -> Float64Array with NaN gaps */
export const nums = (a: ArrayLike<number | null>): Float64Array => Float64Array.from(a as ArrayLike<number>, (v) => (v === null ? NaN : v));
export function finiteRange(arrs: Iterable<ArrayLike<number>>): [number, number] {
  let lo = Infinity, hi = -Infinity;
  for (const a of arrs) for (let i = 0; i < a.length; i++) { const v = a[i]; if (Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; } }
  return [lo, hi];
}
