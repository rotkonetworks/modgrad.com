import { createEffect, createMemo, createSignal, For, on, onCleanup } from 'solid-js';
import { A } from '@/lib/wasm';
import { fmt } from '@/lib/fmt';
import { themeVersion } from '@/lib/theme';
import { chart2d } from '@/charts/chart2d';
import { surface3d } from '@/charts/scene';
import { heatmap } from '@/charts/heatmap';
import { metricCounts } from '@/logic/data';
import { selectedRuns, metricSeries, err, runVersion, catalogVersion, type SeriesData } from '@/stores/runs';
import type { Run } from '@/logic/types';
import { h } from '@/lib/dom';

const [metric, setMetric] = createSignal('');
const [field, setField] = createSignal('');
const [logx, setLogx] = createSignal<boolean | null>(null); // null = auto
const [bin, setBin] = createSignal<number | null>(null); // null = auto

/** metric over step and a numeric config field, across the selected runs: surface, heatmap, slices */
export default function CrossView() {
  let box!: HTMLDivElement, slices!: HTMLDivElement;
  const metrics = createMemo(() => { catalogVersion(); return [...metricCounts(selectedRuns()).entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).map((e) => e[0]); });
  const fields = createMemo(() => { const s = new Set<string>(); for (const r of selectedRuns()) for (const [k, v] of Object.entries(r.config)) if (typeof v === 'number') s.add(k); return [...s].sort(); });
  const key = () => (metrics().includes(metric()) ? metric() : metrics().includes('brain_heldout_nll') ? 'brain_heldout_nll' : metrics()[0] || '');
  const fld = () => (fields().includes(field()) ? field() : fields().includes('brain_params') ? 'brain_params' : fields()[0] || '');
  const runs = createMemo(() => selectedRuns().filter((r) => typeof r.config[fld()] === 'number' && r.metrics[key()]));
  const sig = createMemo(() => JSON.stringify([key(), fld(), logx(), bin(), runs().map((r) => r.id + '@' + runVersion(r.id)), themeVersion()]));
  const [autoLog, setAutoLog] = createSignal(false);
  const [autoBin, setAutoBin] = createSignal(0);

  let tok = 0;
  async function draw(pick?: [number, number]): Promise<void> {
    const t = ++tok;
    const k = key(), f = fld();
    const data: { r: Run; s: SeriesData }[] = [];
    for (const r of runs()) { try { const s = await metricSeries(r, k); if (s) data.push({ r, s }); } catch (e) { err(String(e)); } }
    if (t !== tok) return;
    box.textContent = ''; if (!pick) slices.textContent = '';
    if (data.length < 1 || !k || !f) { h('div', { class: 'note' }, box, 'Select runs that share a metric and have a numeric config field.'); return; }
    const xs = data.map((d) => d.r.config[f] as number);
    const auto = logx() ?? (Math.min(...xs) > 0 && Math.max(...xs) / Math.min(...xs) > 20);
    setAutoLog(auto);
    // step bin: most common positive step difference (so evaluation steps align)
    let b = bin();
    if (b === null) {
      const cnt = new Map<number, number>();
      for (const d of data) for (let i = 1; i < d.s.steps.length; i++) { const df = d.s.steps[i] - d.s.steps[i - 1]; if (df > 0) cnt.set(df, (cnt.get(df) || 0) + 1); }
      const best = [...cnt.entries()].sort((x, y) => y[1] - x[1])[0];
      b = best ? best[0] : 0; setAutoBin(b);
    }
    const off = new Uint32Array(data.length + 1); let n = 0;
    data.forEach((d, i) => { off[i] = n; n += d.s.steps.length; }); off[data.length] = n;
    const steps = new Float64Array(n), vals = new Float64Array(n);
    data.forEach((d, i) => { steps.set(d.s.steps, off[i]); vals.set(d.s.values, off[i]); });
    const so = A.assemble_surface(Float64Array.from(xs), off, steps, vals, b || 0);
    const gx = Array.from(so.xs()), gy = Array.from(so.ys()), z = Array.from(so.z()), count = Array.from(so.count());
    so.free();
    const runsAt = (i: number) => data.filter((d) => d.r.config[f] === gx[i]).map((d) => d.r.id).join(', ');
    const tipText = (i: number, j: number) => `${f} ${fmt(gx[i])}  (${runsAt(i)})\nstep ${fmt(gy[j])}\n${k} ${Number.isFinite(z[i * gy.length + j]) ? fmt(z[i * gy.length + j]) + (count[i * gy.length + j] > 1 ? `  (mean of ${count[i * gy.length + j]})` : '') : 'no data'}`;
    const onpick = (i: number, j: number) => { draw([i, j]); };
    surface3d(box, { title: `${k} over step x ${f}`, gx, gy, z, logx: auto, xlabel: f + (auto ? ' (log)' : ''), ylabel: 'step', zlabel: k, onpick, tipText });
    heatmap(box, { title: `${k}: ${f} x step`, gx, gy, z, xlabel: f, ylabel: 'step', zlabel: k, onpick, tipText, mark: pick || null });
    if (pick) {
      const [i, j] = pick, ny = gy.length;
      slices.textContent = '';
      chart2d(slices, { title: `${k} vs step at ${f} = ${fmt(gx[i])}`, xlabel: 'step', series: [{ name: `${f}=${fmt(gx[i])}`, color: '--c1', area: true, xs: gy, ys: gy.map((_, jj) => z[i * ny + jj]) }], marker: [gy[j]] });
      chart2d(slices, { title: `${k} vs ${f} at step ${fmt(gy[j])}`, xlabel: f + (auto ? ' (log)' : ''), xlog: auto, series: [{ name: `step ${fmt(gy[j])}`, color: '--c2', area: false, xs: gx, ys: gx.map((_, ii) => z[ii * ny + j]) }], marker: [gx[i]] });
    }
  }
  createEffect(on(sig, () => { draw(); }));
  onCleanup(() => { tok++; });
  return (
    <section>
      <h2>Across runs: metric over step and config</h2>
      <div class="controls">
        <label>metric <select aria-label="metric" value={key()} onChange={(e) => setMetric(e.currentTarget.value)}><For each={metrics()}>{(m) => <option value={m} selected={m === key()}>{m}</option>}</For></select></label>
        <label>config field <select aria-label="config field" value={fld()} onChange={(e) => { setField(e.currentTarget.value); setLogx(null); }}><For each={fields()}>{(m) => <option value={m} selected={m === fld()}>{m}</option>}</For></select></label>
        <label><input type="checkbox" checked={logx() ?? autoLog()} onChange={(e) => setLogx(e.currentTarget.checked)} /> log10 config</label>
        <label>step bin <input size="6" aria-label="step bin" title="steps are snapped to the nearest multiple; 0 = exact" value={bin() ?? autoBin()} onChange={(e) => { const v = e.currentTarget.value.trim(); setBin(v === '' ? null : +v || 0); }} /></label>
      </div>
      <div class="note">Uses the runs selected in the Runs list. Cells where a run has no value at that step are gaps: hatched in the heatmap, marked x on the base of the surface. Nothing is interpolated. Runs sharing a config value are averaged (count in the tooltip). Click a cell or vertex for slices.</div>
      <div class="grid" ref={box} />
      <div class="grid" ref={slices} style={{ 'margin-top': '10px' }} />
    </section>
  );
}
