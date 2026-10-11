import { createResource, For, Show } from 'solid-js';
import { fmt } from '@/lib/fmt';
import { runVersion } from '@/stores/runs';
import { samplesOf } from '@/stores/samples';
import { SampleCard, openSamples } from '@/views/SamplesView';
import type { Sample } from '@/logic/samples';

/** Board panel for a `samples:text{...}` query: the latest `k` samples of each matched run (at
 *  most three runs), newest step first. Live rows bump the run version, which refetches. */
export default function SamplesPanel(props: { runs: string[]; k: number }) {
  const runs = () => props.runs.slice(0, 3);
  const [data] = createResource(() => ({ runs: runs(), v: runs().map((r) => runVersion(r)).join(',') }), async ({ runs }) => {
    const out: { run: string; step: number | null; xs: Sample[] }[] = [];
    for (const r of runs) {
      const d = await samplesOf(r);
      const last = d.samples.length ? Math.max(...d.samples.map((s) => s.step)) : null;
      out.push({ run: r, step: last, xs: d.samples.filter((s) => s.step === last).slice(0, props.k) });
    }
    return out;
  });
  return (
    <div class="spanel">
      <Show when={props.runs.length} fallback={<div class="bempty">no matched run has text samples yet</div>}>
        <For each={data() || []}>{(g) => (
          <div class="sgroup">
            <div class="bkey">
              <span class="mono">{g.run}{g.step !== null ? `: step ${fmt(g.step)}` : ''}, latest {g.xs.length}</span>
              <button type="button" title={`open ${g.run} in the Samples tab (slider, search, diff across steps)`} aria-label={`open ${g.run} in the Samples tab`} onClick={() => openSamples(g.run, null, g.step)}>open in Samples</button>
            </div>
            <For each={g.xs}>{(s) => <SampleCard s={s} run={g.run} compact />}</For>
          </div>
        )}</For>
        <Show when={props.runs.length > 3}><div class="bnote">{props.runs.length - 3} more runs match; narrow the query with run=...</div></Show>
      </Show>
    </div>
  );
}
