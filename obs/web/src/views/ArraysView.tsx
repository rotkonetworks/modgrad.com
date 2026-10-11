import { createEffect, createMemo, createSignal, For, on, onCleanup, Show } from 'solid-js';
import { f32 } from '@/lib/fmt';
import { themeVersion } from '@/lib/theme';
import { S, runsVersion, selection, arrayOf, err, runVersion } from '@/stores/runs';
import { drawTrajectory, drawPoints, drawVolume, surface3d, heatmap, chart2d } from '@/charts/arrays';
import type { ArrayFile } from '@/logic/types';

const [runId, setRunId] = createSignal('');
const [akey, setAkey] = createSignal('');
const [stepIdx, setStepIdx] = createSignal<number | null>(null); // null = newest
const [tick, setTick] = createSignal(0);

/** logged arrays: trajectories (PCA), point clouds, volumes, tensors and histograms over steps */
export default function ArraysView() {
  let box!: HTMLDivElement;
  const [note, setNote] = createSignal('');
  const [nrec, setNrec] = createSignal(0);
  const [stepLabel, setStepLabel] = createSignal('');
  const [ntick, setNtick] = createSignal(0);
  const withArr = createMemo(() => { runsVersion(); return S.runs.filter((r) => Object.keys(r.arrays).length); });
  const run = () => { const l = withArr(); const sel = selection(); return l.find((r) => r.id === runId()) || l.find((r) => sel.has(r.id)) || l[0]; };
  const keys = () => { const r = run(); return r ? Object.keys(r.arrays) : []; };
  const key = () => (keys().includes(akey()) ? akey() : keys()[0] || '');
  const sig = createMemo(() => JSON.stringify([run()?.id, key(), stepIdx(), tick(), run() ? runVersion(run()!.id) : 0, themeVersion()]));

  let tok = 0;
  async function draw(): Promise<void> {
    const t = ++tok;
    const r = run(), k = key();
    if (!r || !k) { box.textContent = ''; setNote('No run with logged arrays.'); setNrec(0); setNtick(0); return; }
    let j: ArrayFile; try { j = await arrayOf(r, k); } catch (e) { err(String(e)); return; }
    if (t !== tok) return;
    box.textContent = '';
    const n = j.step.length; setNrec(n);
    const ri = Math.min(stepIdx() ?? n - 1, n - 1);
    setStepLabel(`${j.step[ri]} (${ri + 1}/${n})`);
    const shape = j.shape[ri], data = f32(j.b64[ri]);
    const synth = r.config.synthetic ? ' [synthetic demo data]' : '';
    const kind = j.kind;
    if (kind !== 'volume') setNtick(0);
    if (kind === 'trajectory') { setNote(drawTrajectory(box, k + synth, shape, data, j.step[ri])); return; }
    if (kind === 'points') { const lb = j.labels_b64 && j.labels_b64[ri]; setNote(drawPoints(box, k + synth, shape, data, lb ? f32(lb) : null, j.step[ri])); return; }
    if (kind === 'volume') { const nt = shape[0]; setNtick(nt); const ti = Math.min(tick(), nt - 1); setNote(drawVolume(box, k + synth, shape, data, j.step[ri], ti)); return; }
    // 1-D tensors / histograms over steps: surface over step x index
    if (shape.length === 1) {
      const sameLen = j.shape.every((s) => s.length === 1 && s[0] === shape[0]);
      const recs = sameLen ? j.b64.map(f32) : [data];
      const steps = sameLen ? j.step : [j.step[ri]];
      const nIdx = shape[0];
      const edges = kind === 'histogram' && j.edges && j.edges[ri];
      const idx = edges ? Array.from({ length: nIdx }, (_, b) => (edges[b] + edges[b + 1]) / 2) : Array.from({ length: nIdx }, (_, b) => b);
      const z: number[] = []; for (let a = 0; a < nIdx; a++) for (let s = 0; s < steps.length; s++) z.push(recs[s][a]);
      setNote(`${kind} of length ${nIdx}, ${steps.length} logged step${steps.length > 1 ? 's' : ''}.` + (edges ? " Bin centres use the selected step's edges." : '') + (steps.length < 2 ? ' One step only: the surface is a single curve.' : ''));
      const ilab = edges ? 'bin centre' : 'index';
      if (steps.length > 1) {
        surface3d(box, { title: `${k}: step x ${ilab}${synth}`, gx: idx, gy: steps, z, xlabel: ilab, ylabel: 'step', zlabel: k, onpick: (_, s) => setStepIdx(s) });
        heatmap(box, { title: `${k}${synth}`, gx: idx, gy: steps, z, xlabel: ilab, ylabel: 'step', zlabel: k, onpick: (_, s) => setStepIdx(s) });
      }
      chart2d(box, { title: `${k} at step ${j.step[ri]}${synth}`, xlabel: ilab, series: [{ name: `step ${j.step[ri]}`, color: '--c1', area: true, xs: idx, ys: Array.from(data) }] });
      return;
    }
    if (shape.length === 2) {
      const [n0, n1] = shape;
      const gx = Array.from({ length: n0 }, (_, i) => i), gy = Array.from({ length: n1 }, (_, i) => i);
      const z = Array.from(data);
      const lab0 = /tick/.test(k) ? 'tick' : 'axis 0', lab1 = /neuron/.test(k) ? 'neuron' : 'axis 1';
      setNote(`2-D array ${n0}x${n1} at step ${j.step[ri]}; move the step slider to scrub through ${n} logged steps.`);
      heatmap(box, { title: `${k} at step ${j.step[ri]}${synth}`, gx, gy, z, xlabel: lab0, ylabel: lab1, zlabel: k, H: 260 });
      surface3d(box, { title: `${k} at step ${j.step[ri]}${synth}`, gx, gy, z, xlabel: lab0, ylabel: lab1, zlabel: 'value' });
      return;
    }
    setNote(`Array of shape ${shape.join('x')}: no view for rank ${shape.length} (log it as a volume or trajectory).`);
  }
  createEffect(on(sig, () => { draw(); }));
  onCleanup(() => { tok++; });
  return (
    <section>
      <h2>Arrays and 3-D views</h2>
      <div class="controls">
        <label>run <select aria-label="run with logged arrays" value={run()?.id || ''} onChange={(e) => { setRunId(e.currentTarget.value); setStepIdx(null); }}>
          <For each={withArr()}>{(r) => <option value={r.id} selected={r.id === run()?.id}>{r.id + (r.config.synthetic ? '  [synthetic]' : '')}</option>}</For>
        </select></label>
        <label>array <select aria-label="array" value={key()} onChange={(e) => { setAkey(e.currentTarget.value); setStepIdx(null); }}>
          <For each={keys()}>{(k) => { const a = run()!.arrays[k]; return <option value={k} selected={k === key()}>{`${k}  [${a.kind} ${a.shape.join('x')}, ${a.n_steps} step${a.n_steps > 1 ? 's' : ''}]`}</option>; }}</For>
        </select></label>
        <Show when={nrec() > 1}>
          <label>step <input type="range" min="0" max={nrec() - 1} value={stepIdx() ?? nrec() - 1} aria-label="logged step" onInput={(e) => setStepIdx(+e.currentTarget.value)} /><span class="mono">{stepLabel()}</span></label>
        </Show>
        <Show when={ntick() > 0}>
          <label>tick <input type="range" min="0" max={ntick() - 1} value={Math.min(tick(), ntick() - 1)} aria-label="volume tick" onInput={(e) => setTick(+e.currentTarget.value)} /><span class="mono">{Math.min(tick(), ntick() - 1)}/{ntick() - 1}</span></label>
        </Show>
      </div>
      <div class="note">{note()}</div>
      <div class="grid" ref={box} />
    </section>
  );
}
