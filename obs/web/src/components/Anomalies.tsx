import { createMemo, createResource, createSignal, For, Show } from 'solid-js';
import { fmt, fmtStep } from '@/lib/fmt';
import { lsGet, lsSet } from '@/lib/storage';
import { setGroupViews } from '@/charts/chart2d';
import { S, selectedRuns, runVersion, metricSeries, catalogVersion } from '@/stores/runs';
import { M, board, findPanel, pauseFollow, results, settings } from '@/stores/board';
import { showToast } from '@/stores/ui';
import { layerFamilies } from '@/logic/data';
import { overview, layerList, SCAN, type Anomaly, type ScanIn } from '@/logic/anomalies';

const [open, setOpenSig] = createSignal(lsGet('obs-anomalies') === '1');
export const anomaliesOpen = open;
export function setAnomaliesOpen(v: boolean): void { setOpenSig(v); lsSet('obs-anomalies', v ? '1' : '0'); }
const [kinds, setKinds] = createSignal<{ spike: boolean; nonfinite: boolean }>({ spike: true, nonfinite: true });
const [picked, setPicked] = createSignal<string | null>(null);
/** spikes smaller than this (value / trailing median) are left out: the kernel flags > 6 robust sd,
 *  which on a very smooth curve can be a few percent */
const [minRatio, setMinRatio] = createSignal(1.5);
const stepTxt = (n: number) => (Math.abs(n) < 1e5 ? String(Math.round(n)) : fmtStep(n));

/** Zoom every step-axis panel on the board to a window around `step` (w = half-width; default: 4%
 *  of the longest selected run, at least 25 steps). Following panels pause, so the next live row
 *  does not slide the window away. */
export function jumpToStep(step: number, w?: number): void {
  const span = Math.max(1, ...selectedRuns().map((r) => r.last_step || 0));
  const half = w ?? Math.max(25, Math.round(span * 0.04));
  for (const p of board.panels) { const m = findPanel(p.id); if (m && m.view.follow) pauseFollow(m); }
  let n = setGroupViews((g) => g.startsWith('b:step') || g.startsWith('bsplit:'), { x0: step - half, x1: step + half });
  // panels on a metric x axis (x tokens_seen): map the step window through the panel's own step -> x
  const done = new Set<string>();
  for (const p of board.panels) {
    const r = results.get(p.id);
    if (!r || !r.ok || r.meta.x_label === 'step' || r.meta.x_label === 'time') continue;
    const g = settings.linkX ? 'b:' + r.meta.x_label : `b:${r.meta.x_label}:${p.id}`;
    if (done.has(g)) continue;
    const tr = r.traces.find((t) => t.steps.length > 1);
    if (!tr) continue;
    // x at the last point whose step is <= s (the first point when s is before the data)
    const at = (s: number) => { let x = NaN; for (let i = 0; i < tr.steps.length; i++) { if (!Number.isFinite(tr.xs[i])) continue; if (tr.steps[i] > s && Number.isFinite(x)) break; x = tr.xs[i]; } return x; };
    const x0 = at(step - half), x1 = at(step + half);
    if (Number.isFinite(x0) && Number.isFinite(x1) && x1 > x0) { done.add(g); n += setGroupViews((name) => name === g, { x0, x1 }); }
  }
  showToast(n ? `step-axis panels zoomed to steps ${fmtStep(step - half)}-${fmtStep(step + half)}; double-click a chart to reset` : 'no step-axis panel on the board to zoom');
}

/** Compact list of loss / grad-norm spikes and non-finite values in the selected runs, on the board. */
export default function Anomalies() {
  const runs = createMemo(() => { catalogVersion(); const sel = selectedRuns(); return sel.length ? sel : S.runs; });
  const key = () => runs().map((r) => r.id + '@' + runVersion(r.id)).join(',');
  const [res] = createResource(() => (open() ? key() : null), async () => {
    const ins: ScanIn[] = [];
    for (const r of runs()) {
      const names = Object.keys(r.metrics).filter((m) => SCAN.test(m) && (r.metrics[m].n || 0) >= 2);
      const fam = new Map<string, { family: string; layer: number; n: number }>();
      for (const f of layerFamilies(names, 2)) f.metrics.forEach((m, i) => fam.set(m, { family: f.family, layer: f.layers[i], n: f.layers.length }));
      await Promise.all(names.map(async (m) => {
        try { const s = await metricSeries(r, m); if (s && s.steps.length) ins.push({ run: r.id, metric: m, steps: s.steps, values: s.values, family: fam.get(m) }); } catch { /* skipped */ }
      }));
    }
    return { list: overview(ins), scanned: ins.length };
  });
  const sized = createMemo(() => (res()?.list || []).filter((a) => a.kind === 'nonfinite' || !(a.ratio < minRatio())));
  const list = createMemo(() => sized().filter((a) => kinds()[a.kind]));
  const counts = createMemo(() => { const l = sized(); return { spike: l.filter((a) => a.kind === 'spike').length, nonfinite: l.filter((a) => a.kind === 'nonfinite').length }; });
  const id = (a: Anomaly) => `${a.run}|${a.what}|${a.kind}|${a.from}`;
  const what = (a: Anomaly) => (a.layer ? `${a.what}: ${a.layers.length === a.nLayers ? 'all' : a.layers.length} of ${a.nLayers} layers (${layerList(a.layers)})` : a.what);
  const size = (a: Anomaly) => (a.kind === 'nonfinite' ? `${a.count} value${a.count === 1 ? '' : 's'} NaN or inf` : Number.isFinite(a.ratio) ? `x${a.ratio.toPrecision(3)} the trailing median (value ${fmt(a.value)})` : fmt(a.value));
  return (
    <Show when={open()}>
      <section class="anoms" aria-label="spikes and non-finite values">
        <div class="ahead">
          <b>anomalies</b>
          <span class="note">{res.loading ? 'scanning…' : `${counts().spike} spike${counts().spike === 1 ? '' : 's'}, ${counts().nonfinite} non-finite in ${runs().length} run${runs().length === 1 ? '' : 's'} (${res()?.scanned ?? 0} loss / grad-norm / reward / kl series)`}</span>
          <label title="upward spikes: more than 6 robust sd above the median of the previous 10 points"><input type="checkbox" checked={kinds().spike} onChange={(e) => setKinds({ ...kinds(), spike: e.currentTarget.checked })} /> spikes</label>
          <label title="NaN or inf: the writer stores both as null, so they cannot be told apart"><input type="checkbox" checked={kinds().nonfinite} onChange={(e) => setKinds({ ...kinds(), nonfinite: e.currentTarget.checked })} /> NaN/inf</label>
          <label title="leave out spikes smaller than this multiple of the trailing median">min size <select aria-label="minimum spike size" value={String(minRatio())} onChange={(e) => setMinRatio(+e.currentTarget.value)}>
            <For each={[1, 1.25, 1.5, 2, 5, 10]}>{(v) => <option value={String(v)} selected={v === minRatio()}>{v === 1 ? 'any' : `x${v}`}</option>}</For>
          </select></label>
          <span class="sp" />
          <button type="button" title="reset the zoom of every step-axis panel" aria-label="reset the zoom of the board" onClick={() => { setGroupViews((g) => g.startsWith('b:step') || g.startsWith('bsplit:'), null); setPicked(null); }}>reset zoom</button>
          <button type="button" title="hide the anomaly list" aria-label="hide the anomaly list" onClick={() => setAnomaliesOpen(false)}>hide</button>
        </div>
        <Show when={list().length} fallback={<Show when={!res.loading}><div class="note">Nothing found{(res()?.list.length || 0) > list().length ? ' with these filters' : ''}.</div></Show>}>
          <ol class="alist">
            <For each={list().slice(0, 60)}>{(a) => (
              <li classList={{ on: picked() === id(a) }}>
                <span class={'akind ' + a.kind}>{a.kind === 'spike' ? 'spike' : 'NaN/inf'}</span>
                <span class="mono astep">{a.from === a.to ? `step ${stepTxt(a.from)}` : `steps ${stepTxt(a.from)}-${stepTxt(a.to)}`}</span>
                <span class="mono arun">{a.run}</span>
                <span class="mono awhat" title={a.what}>{what(a)}</span>
                <span class="mono asize">{size(a)}</span>
                <button type="button" title={`zoom every step-axis panel to step ${a.at}`} aria-label={`jump to step ${a.at} (${a.run} ${a.what})`}
                  onClick={() => { setPicked(id(a)); jumpToStep(a.at); }}>jump</button>
              </li>
            )}</For>
          </ol>
          <Show when={list().length > 60}><div class="note">{list().length - 60} more not shown; narrow the run selection.</div></Show>
        </Show>
        <div class="note">Series are scanned as loaded (server mode strides runs longer than 4000 points), so a one-step spike between strided points can be missed.{M.board.panels.length ? '' : ' Add step-axis panels for jump to zoom.'}</div>
      </section>
    </Show>
  );
}
