// Local file drop: parse in a worker, report per file, add runs, keep them in IndexedDB.
import { createSignal } from 'solid-js';
import { createStore, produce } from 'solid-js/store';
import { groupFiles, parseGroup, toRun, saveLocal, restoreLocal, clearLocal, type DropFile, type FileReport } from '@/logic/drop';
import { addRunsUnique, removeRuns } from '@/stores/runs';
import { boardIsAuto } from '@/stores/board';
import { showToast } from '@/stores/ui';
import type { Run } from '@/logic/types';

export interface GroupState { name: string; progress: number; total: number; file?: string; files?: FileReport[]; error?: string; runId?: string }
export interface Batch { id: number; groups: GroupState[]; skipped: string[]; runIds: string[] }
const [report, setReport] = createStore<{ batches: Batch[] }>({ batches: [] });
export { report };
const [localCount, setLocalCount] = createSignal(0);
export { localCount };
let localIds = new Set<string>();
const refresh = () => setLocalCount(localIds.size);

let batchSeq = 0;
const queue: { id: number; src: { name: string; files: DropFile[] }[] }[] = [];
let busy = false;
/** update one group of one batch (batches are addressed by id: newer batches are unshifted) */
function upd(id: number, gi: number, fn: (s: GroupState) => void): void {
  setReport(produce((s) => { const b = s.batches.find((x) => x.id === id); if (b && b.groups[gi]) fn(b.groups[gi]); }));
}
export async function ingest(list: DropFile[]): Promise<void> {
  if (!list || !list.length) return;
  const { groups, skipped } = groupFiles(list);
  const id = ++batchSeq;
  setReport(produce((s) => { s.batches.unshift({ id, groups: groups.map((g) => ({ name: g.name, progress: 0, total: g.files.reduce((a, f) => a + f.file.size, 0) })), skipped, runIds: [] }); }));
  queue.push({ id, src: groups });
  if (busy) return;
  busy = true;
  try {
    while (queue.length) {
      const job = queue.shift()!;
      const added: Run[] = [], addedGi: number[] = [];
      for (let gi = 0; gi < job.src.length; gi++) {
        const g = job.src[gi];
        let raf = 0, lastP: { done: number; file: string } | null = null;
        try {
          const m = await parseGroup(g, (p) => { lastP = p; if (!raf) raf = requestAnimationFrame(() => { raf = 0; const lp = lastP; if (lp) upd(job.id, gi, (s) => { s.progress = lp.done; s.file = lp.file; }); }); });
          if (raf) cancelAnimationFrame(raf);
          upd(job.id, gi, (s) => { s.files = m.meta.files; });
          if (m.meta.keys.length || Object.keys(m.meta.arrays || {}).length) { added.push(toRun(g.name, m)); addedGi.push(gi); }
        } catch (e) {
          if (raf) cancelAnimationFrame(raf);
          upd(job.id, gi, (s) => { s.error = String((e as Error).message || e); });
        }
      }
      if (added.length) {
        const ids = addRunsUnique(added);
        addedGi.forEach((gi, k) => upd(job.id, gi, (s) => { s.runId = ids[k]; }));
        setReport(produce((s) => { const b = s.batches.find((x) => x.id === job.id); if (b) b.runIds.push(...ids); }));
        // an untouched default board already follows the new runs; an edited one is the viewer's: say how to add them
        showToast(boardIsAuto()
          ? `${ids.length} local run${ids.length === 1 ? '' : 's'} added; the default board includes ${ids.length === 1 ? 'it' : 'them'}`
          : `${ids.length} local run${ids.length === 1 ? '' : 's'} added: "add to board" in the file report graphs ${ids.length === 1 ? 'it' : 'them'}`);
        for (const r of added) localIds.add(r.id);
        refresh();
        saveLocal(added);
      }
    }
  } finally { busy = false; }
}
export function dismissReport(): void { setReport('batches', []); }
export async function restoreDropped(): Promise<void> {
  const runs = await restoreLocal();
  if (!runs.length) return;
  for (const r of runs) localIds.add(r.id);
  addRunsUnique(runs, { select: false });
  refresh();
}
export async function forgetLocal(): Promise<void> {
  removeRuns((r) => !!r.local);
  localIds = new Set(); refresh(); dismissReport();
  await clearLocal();
}
