// Lineage set in the browser (Runs view: "set parent", or a run dragged onto another). It lives in
// the board (Board.lineage: saved, exported, undoable with it); this signal mirrors it so the query
// catalog (stores/runs.ts) can read it without importing the board store.
import { createSignal } from 'solid-js';
import type { Lineage } from '@/logic/types';

const [lineage, setLineageSig] = createSignal<Record<string, Lineage>>({});
const [lineageVersion, setVersion] = createSignal(0);
export { lineage, lineageVersion };
/** called by the board store after every commit; a change re-plans every query */
export function mirrorLineage(l: Record<string, Lineage> | undefined): void {
  const next = l || {};
  if (JSON.stringify(next) === JSON.stringify(lineage())) return;
  setLineageSig({ ...next }); setVersion((v) => v + 1);
}

/** the Runs view's "continues from" editor: which run, and a parent picked by dragging (prefill) */
export interface ParentDraft { run: string; parent: string | null }
const [parentDraft, setParentDraft] = createSignal<ParentDraft | null>(null);
export { parentDraft, setParentDraft };
