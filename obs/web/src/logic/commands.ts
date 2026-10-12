// Command palette commands: pure. commands(input, ctx) returns [{label, action}] where action is
// plain data the UI applies; the same actions back the chips and the inspector.

export type Action =
  | { type: 'stage'; kind: string; text: string | null }
  | { type: 'add'; entries: string }
  | { type: 'new'; q: string | null }
  | { type: 'follow' | 'editText' | 'expand' | 'back' | 'undo' | 'export' | 'import' | 'inspect' | 'anomalies' }
  | { type: 'preset'; name: string }
  | { type: 'view'; name: string };
export interface Command { label: string; action: Action }
export interface Ctx { metrics: string[]; fields: string[] }

const norm = (s: string) => s.toLowerCase().replace(/\s+/g, ' ').trim();

export function commands(input: string, ctx: Ctx): Command[] {
  const t = norm(input || '');
  const out: (Command & { hay: string })[] = [];
  const push = (label: string, action: Action, keywords = '') => out.push({ label, action, hay: norm(label + ' ' + keywords) });
  let m: RegExpMatchArray | null;
  // parametric commands first, so typing "forecast to 10k" offers exactly that
  if ((m = t.match(/^forecast(?: to)? ?([0-9.]+[kmgb]?)?$/i))) push(m[1] ? `forecast to ${m[1]}` : 'forecast', { type: 'stage', kind: 'forecast', text: m[1] ? `forecast to ${m[1]}` : 'forecast' });
  if ((m = t.match(/^split(?: by)? (\S+)$/))) push(`split by ${m[1]}`, { type: 'stage', kind: 'split', text: `split ${m[1]}` });
  if ((m = t.match(/^(?:colou?r )?by (\S+)$/))) push(`colour by ${m[1]}`, { type: 'stage', kind: 'by', text: `by ${m[1]}` });
  if ((m = t.match(/^ema ?([0-9.]+)$/))) push(`ema ${m[1]}`, { type: 'stage', kind: 'ema', text: `ema ${m[1]}` });
  if ((m = t.match(/^x (\S+)$/))) push(`x axis: ${m[1]}`, { type: 'stage', kind: 'x', text: `x ${m[1]}` });
  if ((m = t.match(/^(mean±sd|mean\+-sd|mean|median) over (\S+)$/))) push(`${m[1]} over ${m[2]}`, { type: 'stage', kind: 'agg', text: `${m[1]} over ${m[2]}` });
  if ((m = t.match(/^baseline (\S+)(?: (diff|ratio))?$/))) push(`baseline ${m[1]}${m[2] ? ' ' + m[2] : ''}`, { type: 'stage', kind: 'baseline', text: `baseline "${m[1]}"${m[2] === 'ratio' ? ' ratio' : ''}` });
  if ((m = t.match(/^(?:add|plot) (.+)$/))) {
    const q = m[1];
    for (const name of ctx.metrics.filter((n) => n.toLowerCase().includes(q)).slice(0, 8)) push(`add ${name}`, { type: 'add', entries: name });
    if (/[{=~]/.test(q)) push(`add ${q}`, { type: 'add', entries: q });
  }
  if ((m = t.match(/^(?:new panel|query) (.+)$/))) push(`new panel: ${m[1]}`, { type: 'new', q: m[1] });
  // fixed commands
  const nParam = out.length;
  push('new panel', { type: 'new', q: null }, 'create add');
  push('follow newest data', { type: 'follow' }, 'live tail');
  push('split by seed', { type: 'stage', kind: 'split', text: 'split seed' });
  push('mean±sd over seed', { type: 'stage', kind: 'agg', text: 'mean±sd over seed' }, 'aggregate seeds band');
  push('split by metric', { type: 'stage', kind: 'split', text: 'split metric' }, 'small multiples');
  push('forecast', { type: 'stage', kind: 'forecast', text: 'forecast' }, 'predict ghost');
  push('remove forecast', { type: 'stage', kind: 'forecast', text: null });
  push('log y axis', { type: 'stage', kind: 'log', text: 'log' });
  push('linear y axis', { type: 'stage', kind: 'log', text: null });
  push('x axis: step', { type: 'stage', kind: 'x', text: null });
  push('x axis: time', { type: 'stage', kind: 'x', text: 'x time' });
  push('x axis: tokens', { type: 'stage', kind: 'x', text: 'x tokens' });
  push('layer heatmap', { type: 'stage', kind: 'heatmap', text: 'heatmap over layer' }, 'per layer heatmap grad norm activation');
  push('layer heatmap, per-layer normalised', { type: 'stage', kind: 'heatmap', text: 'heatmap over layer rownorm' }, 'per layer heatmap rownorm spike');
  push('lines instead of heatmap', { type: 'stage', kind: 'heatmap', text: null }, 'remove heatmap');
  push('edit query as text', { type: 'editText' });
  push('inspect panel', { type: 'inspect' }, 'inspector settings');
  push('expand panel', { type: 'expand' }, 'focus fullscreen');
  push('back to board', { type: 'back' }, 'close focus');
  push('undo', { type: 'undo' });
  push('auto board', { type: 'preset', name: 'auto' }, 'default conventions');
  push('history board', { type: 'preset', name: 'history' }, 'reset default selected runs');
  push('export board', { type: 'export' }, 'download json');
  push('import board', { type: 'import' }, 'load json');
  push('go to board', { type: 'view', name: 'board' }, 'view tab');
  push('go to across runs', { type: 'view', name: 'cross' }, 'view tab surface heatmap');
  push('go to arrays / 3D', { type: 'view', name: 'arrays' }, 'view tab pca trajectory volume');
  push('go to runs', { type: 'view', name: 'runs' }, 'view tab table');
  push('go to samples', { type: 'view', name: 'samples' }, 'view tab generations completions text sft rl');
  push('go to evals', { type: 'view', name: 'evals' }, 'view tab benchmarks table checkpoints');
  push('show or hide anomalies (spikes, NaN)', { type: 'anomalies' }, 'spikes nan inf loss grad norm post-mortem jump');
  for (const f of ctx.fields.slice(0, 40)) { push(`colour by ${f}`, { type: 'stage', kind: 'by', text: `by ${f}` }); push(`split by ${f}`, { type: 'stage', kind: 'split', text: `split ${f}` }); }
  for (const name of ctx.metrics.slice(0, 200)) push(`add ${name}`, { type: 'add', entries: name });
  // parametric hits stay on top; the rest must contain every typed word
  const words = t.split(' ').filter(Boolean);
  const seen = new Set<string>();
  return out.filter((c, i) => (i < nParam || words.every((w) => c.hay.includes(w))) && !seen.has(c.label) && seen.add(c.label))
    .slice(0, 30).map(({ label, action }) => ({ label, action }));
}
