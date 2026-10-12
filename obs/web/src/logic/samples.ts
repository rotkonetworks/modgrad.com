// Text samples (generations) of SFT / RL / eval runs: pure helpers, no DOM. Records come from
// samples.jsonl via GET api/v1/runs/<id>/samples (or samples.json in a static export); the word
// diff and the length histogram run in wasm (crates/modgrad-obs-analysis/src/text.rs).
import { A } from '@/lib/wasm';

export interface Sample {
  step: number; t?: number; key: string; prompt: string; completion: string;
  reward?: number; score?: number; meta?: Record<string, unknown>;
}
export interface SamplesData { samples: Sample[]; steps: number[]; keys: string[]; n: number }

/** whitespace-separated words: the length unit of the histogram and the sort (tokenizer-free) */
export const words = (s: string): number => { const m = s.match(/\S+/g); return m ? m.length : 0; };

export interface SampleFilter { key: string | null; step: number | null; search: string; rmin: number | null; rmax: number | null }
/** filter: key, step (null = every step), search (every word must appear in prompt, completion or key), reward range */
export function filterSamples(all: Sample[], f: SampleFilter): Sample[] {
  const ws = f.search.toLowerCase().split(/\s+/).filter(Boolean);
  return all.filter((s) => (f.key === null || s.key === f.key)
    && (f.step === null || s.step === f.step)
    && (f.rmin === null || (s.reward !== undefined && s.reward >= f.rmin))
    && (f.rmax === null || (s.reward !== undefined && s.reward <= f.rmax))
    && (!ws.length || ws.every((w) => s.completion.toLowerCase().includes(w) || s.prompt.toLowerCase().includes(w) || s.key.toLowerCase().includes(w))));
}

export type SortKey = 'log' | 'reward-desc' | 'reward-asc' | 'score-desc' | 'len-desc' | 'len-asc';
export const SORTS: [SortKey, string][] = [
  ['log', 'logged order'], ['reward-desc', 'reward, high first'], ['reward-asc', 'reward, low first'],
  ['score-desc', 'score, high first'], ['len-desc', 'length, longest first'], ['len-asc', 'length, shortest first'],
];
/** stable sort; samples without the sorted field go last */
export function sortSamples(xs: Sample[], by: SortKey): Sample[] {
  if (by === 'log') return xs.slice();
  const val = (s: Sample): number | undefined => (by.startsWith('reward') ? s.reward : by.startsWith('score') ? s.score : words(s.completion));
  const desc = by.endsWith('desc');
  return xs.map((s, i) => ({ s, i, v: val(s) })).sort((a, b) => {
    const am = a.v === undefined || !Number.isFinite(a.v), bm = b.v === undefined || !Number.isFinite(b.v);
    if (am !== bm) return am ? 1 : -1;
    if (!am && a.v !== b.v) return desc ? b.v! - a.v! : a.v! - b.v!;
    return a.i - b.i;
  }).map((x) => x.s);
}

/** the same (key, prompt) across steps, in step order: the first sample logged at each step */
export function acrossSteps(all: Sample[], key: string, prompt: string): Sample[] {
  const by = new Map<number, Sample>();
  for (const s of all) if (s.key === key && s.prompt === prompt && !by.has(s.step)) by.set(s.step, s);
  return [...by.values()].sort((a, b) => a.step - b.step);
}

/** word-level diff old -> new (wasm LCS): runs of [op, text], op 0 same, 1 inserted, 2 deleted */
export interface Diff { ops: [0 | 1 | 2, string][]; same: number }
export function diffWords(a: string, b: string): Diff {
  try { return JSON.parse(A.diff_words(a, b)) as Diff; } catch { return { ops: [[2, a], [1, b]], same: 0 }; }
}

/** histogram of completion lengths (words): {lo, hi, counts} from wasm, null when empty */
export function lengthHistogram(xs: Sample[], bins = 16): { lo: number; hi: number; counts: number[]; mean: number; median: number } | null {
  const v = Float64Array.from(xs, (s) => words(s.completion));
  if (!v.length) return null;
  const h = A.histogram(v, bins);
  if (!h.length) return null;
  const sorted = Array.from(v).sort((a, b) => a - b);
  const median = sorted.length % 2 ? sorted[(sorted.length - 1) / 2] : (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2;
  return { lo: h[0], hi: h[1], counts: Array.from(h.slice(2)), mean: sorted.reduce((a, b) => a + b, 0) / sorted.length, median };
}

/** plain-text form for the clipboard */
export function sampleText(s: Sample, run: string): string {
  const head = [`run ${run}`, `step ${s.step}`, `key ${s.key}`];
  if (s.reward !== undefined) head.push(`reward ${s.reward}`);
  if (s.score !== undefined) head.push(`score ${s.score}`);
  return `${head.join(', ')}\n\n## prompt\n${s.prompt}\n\n## completion\n${s.completion}${s.meta ? `\n\n## meta\n${JSON.stringify(s.meta)}` : ''}\n`;
}
