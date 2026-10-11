// Eval table: metrics named eval/<benchmark>/<metric> logged at checkpoints (steps). Pure helpers,
// no DOM: columns, rows (run x step), best per column, sorting.

export interface EvalCol { name: string; bench: string; metric: string; lower: boolean }
/** lower is better for losses, perplexities and error rates; higher for everything else */
export const lowerIsBetter = (metric: string): boolean => /loss|nll|ppl|perplex|err(or)?($|[_/])|(^|[_/])(wer|cer|bpb|mae|mse|rmse)($|[_/])/i.test(metric);
/** columns from metric names: eval/<benchmark>/<metric...> (three or more path segments) */
export function evalColumns(names: string[]): EvalCol[] {
  const out: EvalCol[] = [];
  for (const n of new Set(names)) {
    const m = n.match(/^eval\/([^/]+)\/(.+)$/);
    if (m) out.push({ name: n, bench: m[1], metric: m[2], lower: lowerIsBetter(m[2]) });
  }
  return out.sort((a, b) => a.bench.localeCompare(b.bench) || a.metric.localeCompare(b.metric));
}

export interface EvalRow { run: string; step: number; vals: number[] }
export interface Series { steps: ArrayLike<number>; values: ArrayLike<number> }
/** one row per (run, step) at which any column was logged; NaN where a column was not */
export function evalRows(runs: string[], cols: EvalCol[], get: (run: string, col: string) => Series | null | undefined): EvalRow[] {
  const rows: EvalRow[] = [];
  for (const r of runs) {
    const byStep = new Map<number, number[]>();
    cols.forEach((c, ci) => {
      const s = get(r, c.name); if (!s) return;
      for (let i = 0; i < s.steps.length; i++) {
        const st = s.steps[i];
        let row = byStep.get(st);
        if (!row) { row = cols.map(() => NaN); byStep.set(st, row); }
        row[ci] = s.values[i];
      }
    });
    for (const [step, vals] of [...byStep.entries()].sort((a, b) => a[0] - b[0])) rows.push({ run: r, step, vals });
  }
  return rows;
}
/** index of the best row per column (-1 when the column has no finite value); ties keep the earliest */
export function bestRows(rows: EvalRow[], cols: EvalCol[]): number[] {
  return cols.map((c, ci) => {
    let best = -1;
    rows.forEach((r, i) => {
      const v = r.vals[ci];
      if (!Number.isFinite(v)) return;
      if (best < 0 || (c.lower ? v < rows[best].vals[ci] : v > rows[best].vals[ci])) best = i;
    });
    return best;
  });
}
/** sort by a column (index), or 'run' / 'step'; missing values last in either direction */
export function sortRows(rows: EvalRow[], by: number | 'run' | 'step', dir: 1 | -1): EvalRow[] {
  return rows.map((r, i) => ({ r, i })).sort((a, b) => {
    if (by === 'run') return (a.r.run.localeCompare(b.r.run) || a.r.step - b.r.step) * dir || a.i - b.i;
    if (by === 'step') return (a.r.step - b.r.step || a.r.run.localeCompare(b.r.run)) * dir || a.i - b.i;
    const x = a.r.vals[by], y = b.r.vals[by];
    const xm = !Number.isFinite(x), ym = !Number.isFinite(y);
    if (xm !== ym) return xm ? 1 : -1;
    return (xm ? 0 : (x - y) * dir) || a.i - b.i;
  }).map((x) => x.r);
}
