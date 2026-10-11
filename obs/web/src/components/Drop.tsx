import { createSignal, For, onCleanup, onMount, Show } from 'solid-js';
import { report, dismissReport, ingest } from '@/stores/drop';
import { filesFromDrop } from '@/logic/drop';

const mb = (n: number) => (n >= 1 << 20 ? (n / (1 << 20)).toFixed(1) + ' MB' : n >= 1024 ? (n / 1024).toFixed(1) + ' kB' : n + ' B');
const hasFiles = (e: DragEvent) => !!e.dataTransfer && [...e.dataTransfer.types].includes('Files');

/** page-wide file drop target (HTML5 drag of files from the OS; the builder's own drags are pointer events) */
export function DropZone() {
  const [on, setOn] = createSignal(false);
  let depth = 0;
  const outside = (e: DragEvent) => e.clientX <= 0 || e.clientY <= 0 || e.clientX >= innerWidth || e.clientY >= innerHeight;
  const enter = (e: DragEvent) => { if (!hasFiles(e)) return; e.preventDefault(); depth++; setOn(true); };
  const over = (e: DragEvent) => { if (!hasFiles(e)) return; e.preventDefault(); e.dataTransfer!.dropEffect = 'copy'; };
  // enter/leave pairs are counted; a leave at the window edge always hides
  const leave = (e: DragEvent) => { if (!hasFiles(e)) return; depth = Math.max(0, depth - 1); if (!depth || outside(e)) { depth = 0; setOn(false); } };
  const end = () => { depth = 0; setOn(false); };
  const drop = async (e: DragEvent) => { if (!hasFiles(e)) return; e.preventDefault(); depth = 0; setOn(false); ingest(await filesFromDrop(e.dataTransfer!)); };
  onMount(() => { addEventListener('dragenter', enter); addEventListener('dragover', over); addEventListener('dragleave', leave); addEventListener('dragend', end); addEventListener('drop', drop); });
  onCleanup(() => { removeEventListener('dragenter', enter); removeEventListener('dragover', over); removeEventListener('dragleave', leave); removeEventListener('dragend', end); removeEventListener('drop', drop); });
  return (
    <Show when={on()}>
      <div class="dropzone"><div class="box"><b>drop logs to graph them</b>parsed in your browser, nothing is uploaded<br /><span>CSV / TSV · JSONL · text logs · TensorBoard events.out.tfevents.* · W&amp;B .wandb · modgrad-obs run folders</span></div></div>
    </Show>
  );
}

/** per-file parse report for dropped logs */
export function DropReport() {
  return (
    <Show when={report.batches.length}>
      <div class="dropreport" role="status">
        <div class="hd"><b>local files</b><span>parsed in this browser, nothing uploaded</span><span class="sp" />
          <button type="button" title="hide this report (the runs stay)" aria-label="dismiss the file report" onClick={dismissReport}>dismiss</button></div>
        <div class="tw"><table>
          <thead><tr><th>file</th><th>run</th><th>format</th><th>metrics</th><th>steps</th><th>points</th><th>skipped</th><th>notes</th></tr></thead>
          <tbody>
            <For each={report.batches}>{(b) => (
              <>
                <For each={b.groups}>{(g) => (
                  <Show when={!g.error} fallback={<tr><td>{g.name}</td><td /><td colSpan={5} class="bad">failed</td><td class="notes bad">{g.error}</td></tr>}>
                    <Show when={g.files} fallback={
                      <tr><td>{g.name}</td><td /><td colSpan={5}><progress max="100" value={g.total ? Math.round((100 * g.progress) / g.total) : 0} /> {g.total ? Math.round((100 * g.progress) / g.total) : 0}% of {mb(g.total)}</td><td class="notes">{g.file || ''}</td></tr>
                    }>
                      <For each={g.files}>{(f) => (
                        <tr><td title={mb(f.bytes)}>{f.file}</td><td>{g.runId || '(nothing to graph)'}</td><td>{f.format}</td><td class="num">{f.metrics}</td><td class="num">{f.steps}</td><td class="num">{f.points}</td><td class="num">{f.skipped || ''}</td>
                          <td class="notes" classList={{ bad: !!f.error }}>{(f.error ? [f.error] : []).concat(f.notes || []).join('; ')}</td></tr>
                      )}</For>
                    </Show>
                  </Show>
                )}</For>
                <Show when={b.skipped.length}><tr><td colSpan={7} class="notes">not tried ({b.skipped.length}): {b.skipped.slice(0, 12).join(', ')}{b.skipped.length > 12 ? ', ...' : ''}</td><td /></tr></Show>
              </>
            )}</For>
          </tbody>
        </table></div>
      </div>
    </Show>
  );
}
