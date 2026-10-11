import { createEffect, createMemo, createResource, createSignal, For, on, Show } from 'solid-js';
import { fmt } from '@/lib/fmt';
import { setHash } from '@/lib/hash';
import { runsVersion } from '@/stores/runs';
import { showToast } from '@/stores/ui';
import { samplesOf, sampleRuns } from '@/stores/samples';
import {
  filterSamples, sortSamples, acrossSteps, diffWords, lengthHistogram, sampleText, words, SORTS,
  type Sample, type SortKey,
} from '@/logic/samples';

// view state lives at module level: it survives tab switches, and the board panel can open a run here
const [run, setRun] = createSignal<string | null>(null);
const [key, setKey] = createSignal<string | null>(null); // null = every key
const [step, setStep] = createSignal<number | null>(null); // null = latest
const [allSteps, setAllSteps] = createSignal(false);
const [search, setSearch] = createSignal('');
const [rmin, setRmin] = createSignal<number | null>(null);
const [rmax, setRmax] = createSignal<number | null>(null);
const [sort, setSort] = createSignal<SortKey>('log');
const [compare, setCompare] = createSignal<{ key: string; prompt: string } | null>(null);
const [shown, setShown] = createSignal(60);

/** open the Samples tab on a run (and optionally a key and step) */
export function openSamples(runId: string, k: string | null = null, st: number | null = null): void {
  setRun(runId); setKey(k); setStep(st); setAllSteps(false); setCompare(null);
  setHash({ view: 'samples', panel: null });
}

export async function copySample(s: Sample, runId: string): Promise<void> {
  const text = sampleText(s, runId);
  try { await navigator.clipboard.writeText(text); showToast('copied the sample'); return; } catch { /* insecure context: fall back */ }
  const ta = document.createElement('textarea'); ta.value = text; ta.setAttribute('readonly', ''); ta.style.position = 'fixed'; ta.style.opacity = '0';
  document.body.appendChild(ta); ta.select();
  try { document.execCommand('copy'); showToast('copied the sample'); } catch { showToast('copy failed: select the text by hand'); }
  ta.remove();
}

const num = (v: string): number | null => { const t = v.trim(); if (!t) return null; const n = Number(t); return Number.isFinite(n) ? n : null; };

/** one sample: header, prompt, completion, meta, copy / compare */
export function SampleCard(props: { s: Sample; run: string; compact?: boolean; onCompare?: () => void; diffWith?: Sample | null }) {
  const [open, setOpen] = createSignal(!props.compact);
  const long = () => props.compact && props.s.completion.length > 360;
  const head = () => {
    const s = props.s, out = [`step ${fmt(s.step)}`, s.key];
    if (s.reward !== undefined) out.push(`reward ${fmt(s.reward)}`);
    if (s.score !== undefined) out.push(`score ${fmt(s.score)}`);
    out.push(`${words(s.completion)} words`);
    return out.join('  ·  ');
  };
  const diff = createMemo(() => (props.diffWith ? diffWords(props.diffWith.completion, props.s.completion) : null));
  return (
    <article class="sample" aria-label={`sample at step ${props.s.step}`}>
      <div class="shead">
        <span class="mono">{head()}</span>
        <span class="sp" />
        <Show when={props.onCompare}>
          <button type="button" title="the same prompt across steps, with a word diff of the completions" aria-label="compare this prompt across steps" onClick={() => props.onCompare!()}>compare across steps</button>
        </Show>
        <button type="button" title="copy prompt, completion and fields as text" aria-label="copy this sample" onClick={() => copySample(props.s, props.run)}>copy</button>
      </div>
      <Show when={props.s.prompt}><pre class="sprompt">{props.s.prompt}</pre></Show>
      <Show when={diff()} fallback={
        <pre class="scomp">{long() && !open() ? props.s.completion.slice(0, 360) + ' …' : props.s.completion}</pre>
      }>{(d) => (
        <pre class="scomp" aria-label="completion, word diff against the previous step">
          <For each={d().ops}>{([op, t]) => (op === 0 ? <span>{t}</span> : op === 1 ? <ins class="d-ins">{t}</ins> : <del class="d-del">{t}</del>)}</For>
        </pre>
      )}</Show>
      <Show when={long()}>
        <button type="button" class="more" aria-expanded={open()} title={open() ? 'show less' : 'show the whole completion'} aria-label={open() ? 'show less of the completion' : 'show the whole completion'} onClick={() => setOpen(!open())}>{open() ? 'less' : 'more'}</button>
      </Show>
      <Show when={props.s.meta && !props.compact}><div class="smeta mono">{JSON.stringify(props.s.meta)}</div></Show>
    </article>
  );
}

function LengthHist(props: { xs: Sample[]; label: string }) {
  const h = createMemo(() => lengthHistogram(props.xs));
  const W = 540, H = 92, L = 34, R = 8, T = 6, B = 22;
  return (
    <Show when={h()} fallback={<div class="note">no samples for a length histogram</div>}>{(hh) => {
      const max = () => Math.max(1, ...hh().counts);
      const bw = () => (W - L - R) / hh().counts.length;
      return (
        <figure class="lhist">
          <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`completion length histogram, ${props.label}`}>
            <For each={hh().counts}>{(c, i) => {
              const ht = ((H - T - B) * c) / max();
              const a = hh().lo + ((hh().hi - hh().lo) * i()) / hh().counts.length, b = hh().lo + ((hh().hi - hh().lo) * (i() + 1)) / hh().counts.length;
              return <rect x={L + i() * bw() + 0.5} y={H - B - ht} width={Math.max(1, bw() - 1)} height={ht} style={{ fill: 'var(--c1)' }}><title>{`${Math.round(a)}-${Math.round(b)} words: ${c}`}</title></rect>;
            }}</For>
            <line x1={L} x2={W - R} y1={H - B + 0.5} y2={H - B + 0.5} style={{ stroke: 'var(--frame)' }} />
            <text x={L} y={H - 8} style={{ font: '10px var(--mono)', fill: 'var(--ink)' }}>{fmt(Math.round(hh().lo))}</text>
            <text x={W - R} y={H - 8} text-anchor="end" style={{ font: '10px var(--mono)', fill: 'var(--ink)' }}>{fmt(Math.round(hh().hi))} words</text>
            <text x={L - 4} y={T + 8} text-anchor="end" style={{ font: '10px var(--mono)', fill: 'var(--muted)' }}>{max()}</text>
          </svg>
          <figcaption class="note">completion length ({props.label}): n {props.xs.length}, mean {hh().mean.toFixed(1)}, median {fmt(hh().median)} words</figcaption>
        </figure>
      );
    }}</Show>
  );
}

function Compare(props: { all: Sample[]; run: string; c: { key: string; prompt: string } }) {
  const [diffOn, setDiffOn] = createSignal(true);
  const xs = createMemo(() => acrossSteps(props.all, props.c.key, props.c.prompt));
  return (
    <section class="scompare" aria-label="same prompt across steps">
      <div class="controls">
        <b class="mono">same prompt across {xs().length} step{xs().length === 1 ? '' : 's'} ({props.c.key})</b>
        <label title="mark words added (underlined) and removed (struck) since the previous step"><input type="checkbox" checked={diffOn()} onChange={(e) => setDiffOn(e.currentTarget.checked)} /> word diff vs previous step</label>
        <button type="button" title="back to the sample list" aria-label="close the comparison" onClick={() => setCompare(null)}>close comparison</button>
      </div>
      <pre class="sprompt">{props.c.prompt}</pre>
      <For each={xs()}>{(s, i) => <SampleCard s={{ ...s, prompt: '' }} run={props.run} diffWith={diffOn() && i() > 0 ? xs()[i() - 1] : null} />}</For>
    </section>
  );
}

/** Samples tab: generations of SFT / RL / eval runs */
export default function SamplesView() {
  const runs = createMemo(() => { runsVersion(); return sampleRuns(); });
  const cur = () => (run() && runs().includes(run()!) ? run()! : runs()[0] || null);
  const [data] = createResource(() => { const r = cur(); return r ? { r, v: runsVersion() } : null; }, ({ r }) => samplesOf(r));
  const all = () => data()?.samples || [];
  const keys = createMemo(() => [...new Set(all().map((s) => s.key))].sort());
  const steps = createMemo(() => { const k = key(); return [...new Set(all().filter((s) => k === null || s.key === k).map((s) => s.step))].sort((a, b) => a - b); });
  const curStep = () => { const st = steps(); const s = step(); return s !== null && st.includes(s) ? s : st[st.length - 1] ?? null; };
  const stepIdx = () => Math.max(0, steps().indexOf(curStep() ?? -1));
  const atStep = createMemo(() => filterSamples(all(), { key: key(), step: allSteps() ? null : curStep(), search: '', rmin: null, rmax: null }));
  const list = createMemo(() => sortSamples(filterSamples(all(), { key: key(), step: allSteps() ? null : curStep(), search: search(), rmin: rmin(), rmax: rmax() }), sort()));
  createEffect(on([cur, key, curStep, allSteps, search, rmin, rmax, sort], () => setShown(60), { defer: true }));
  const rewards = () => atStep().filter((s) => s.reward !== undefined).map((s) => s.reward!);
  return (
    <section class="samplesview">
      <h2>Samples</h2>
      <Show when={runs().length} fallback={
        <div class="note">No run has logged text samples yet. Log them with <code>run.log_sample(step, &Sample{'{'} prompt, completion, reward, .. {'}'})</code> (Rust) or <code>POST /api/v1/runs/&lt;id&gt;/samples</code> with <code>{'{'}"step", "key", "prompt", "completion", "reward"{'}'}</code>; see GET /api/v1.</div>
      }>
        <div class="controls">
          <label>run <select aria-label="run" value={cur() || ''} onChange={(e) => { setRun(e.currentTarget.value); setStep(null); setCompare(null); }}>
            <For each={runs()}>{(r) => <option value={r} selected={r === cur()}>{r}</option>}</For>
          </select></label>
          <label>key <select aria-label="sample key" value={key() ?? ''} onChange={(e) => { setKey(e.currentTarget.value || null); setStep(null); }}>
            <option value="">all keys</option>
            <For each={keys()}>{(k) => <option value={k} selected={k === key()}>{k}</option>}</For>
          </select></label>
          <span class="stepper">
            <button type="button" disabled={allSteps() || stepIdx() <= 0} title="previous step with samples" aria-label="previous step" onClick={() => setStep(steps()[stepIdx() - 1])}>&lt;</button>
            <input type="range" min="0" max={Math.max(0, steps().length - 1)} value={stepIdx()} disabled={allSteps() || steps().length < 2}
              aria-label="step" title="step with samples" onInput={(e) => setStep(steps()[+e.currentTarget.value])} />
            <button type="button" disabled={allSteps() || stepIdx() >= steps().length - 1} title="next step with samples" aria-label="next step" onClick={() => setStep(steps()[stepIdx() + 1])}>&gt;</button>
            <span class="mono">{allSteps() ? `all ${steps().length} steps` : `step ${curStep() === null ? '-' : fmt(curStep()!)} (${stepIdx() + 1}/${steps().length})`}</span>
          </span>
          <label title="list the samples of every step"><input type="checkbox" checked={allSteps()} onChange={(e) => setAllSteps(e.currentTarget.checked)} /> all steps</label>
        </div>
        <div class="controls">
          <label>search <input type="search" size="22" placeholder="words in prompt or completion" aria-label="search samples" value={search()} onInput={(e) => setSearch(e.currentTarget.value)} /></label>
          <label title="only samples with a reward at or above this">reward &ge; <input size="5" aria-label="minimum reward" value={rmin() ?? ''} onChange={(e) => setRmin(num(e.currentTarget.value))} /></label>
          <label title="only samples with a reward at or below this">reward &le; <input size="5" aria-label="maximum reward" value={rmax() ?? ''} onChange={(e) => setRmax(num(e.currentTarget.value))} /></label>
          <label>sort <select aria-label="sort samples" value={sort()} onChange={(e) => setSort(e.currentTarget.value as SortKey)}>
            <For each={SORTS}>{([k, l]) => <option value={k} selected={k === sort()}>{l}</option>}</For>
          </select></label>
          <span class="note">{list().length} of {atStep().length} shown{rewards().length ? `, mean reward here ${(rewards().reduce((a, b) => a + b, 0) / rewards().length).toFixed(3)}` : ''}{data()?.n ? `, ${data()!.n} in the run` : ''}</span>
        </div>
        <Show when={data.loading}><div class="note">loading samples…</div></Show>
        <LengthHist xs={atStep()} label={allSteps() ? 'all steps' : `step ${curStep() === null ? '-' : fmt(curStep()!)}`} />
        <Show when={compare()} fallback={
          <div class="slist">
            <For each={list().slice(0, shown())}>{(s) => <SampleCard s={s} run={cur()!} onCompare={() => setCompare({ key: s.key, prompt: s.prompt })} />}</For>
            <Show when={list().length > shown()}>
              <button type="button" title="show more samples" aria-label="show more samples" onClick={() => setShown(shown() + 100)}>show {Math.min(100, list().length - shown())} more of {list().length - shown()}</button>
            </Show>
            <Show when={!list().length && !data.loading}><div class="note">No samples match.</div></Show>
          </div>
        }>{(c) => <Compare all={all()} run={cur()!} c={c()} />}</Show>
      </Show>
    </section>
  );
}
