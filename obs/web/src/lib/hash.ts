// URL hash = the app's address. Parameters (URLSearchParams syntax after '#'):
//   run=<id>        only that run is selected; waits for it if it does not exist yet (server mode)
//   project=<p>     a project board: only that project's runs, new ones appear live
//   new             issue an anonymous project, then become project=<p>
//   board=<name>    a shared board file (boards/<name>.json, written by the MCP make_board tool)
//   view=<tab>      samples | evals | cross | arrays | runs (board is the default)
//   panel=<id>      focus view of one panel (the legacy form #panel-<id> is accepted too)
import { createSignal } from 'solid-js';

export const BOARD_NAME = /^[A-Za-z0-9._~-]+$/;
const PANEL_ID = /^[A-Za-z0-9-]+$/;

export interface HashState { run: string | null; project: string | null; isNew: boolean; board: string | null; view: string | null; panel: string | null }
export function parseHash(h = location.hash): HashState {
  const raw = h.replace(/^#/, '');
  const legacy = raw.match(/^panel-([A-Za-z0-9-]+)$/);
  if (legacy) return { run: null, project: null, isNew: false, board: null, view: null, panel: legacy[1] };
  const p = new URLSearchParams(raw);
  const board = p.get('board'), panel = p.get('panel');
  return {
    run: p.get('run'), project: p.get('project'), isNew: p.has('new'),
    board: board && BOARD_NAME.test(board) ? board : null,
    view: p.get('view'), panel: panel && PANEL_ID.test(panel) ? panel : null,
  };
}
function format(s: HashState): string {
  const parts: string[] = [];
  const enc = (v: string) => encodeURIComponent(v).replace(/%2F/g, '/');
  if (s.run) parts.push('run=' + enc(s.run));
  if (s.project) parts.push('project=' + enc(s.project));
  if (s.board) parts.push('board=' + s.board);
  if (s.view && s.view !== 'board') parts.push('view=' + s.view);
  if (s.panel) parts.push('panel=' + s.panel);
  return parts.length ? '#' + parts.join('&') : '';
}
const [hash, setHashSig] = createSignal<HashState>(parseHash());
export { hash };
/** the data target (what the page loads and subscribes to) as one comparable string */
export const targetKey = (s: HashState): string => JSON.stringify([s.run, s.project, s.board]);

/** update hash params; push = new history entry (focus view: browser Back closes it) */
export function setHash(patch: Partial<HashState>, push = false): void {
  const next = { ...hash(), ...patch, isNew: false };
  const url = location.pathname + location.search + format(next);
  if (push) history.pushState(null, '', url); else history.replaceState(null, '', url);
  setHashSig(next);
}
export function onHashChange(fn: (prev: HashState, next: HashState) => void): void {
  const handler = () => { const prev = hash(), next = parseHash(); setHashSig(next); fn(prev, next); };
  window.addEventListener('hashchange', handler);
  window.addEventListener('popstate', handler);
}
