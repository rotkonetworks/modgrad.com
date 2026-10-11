import { onCleanup, onMount, Show } from 'solid-js';
import { board, focusId, closeFocus, titleOf, inspector } from '@/stores/board';
import { showToast } from '@/stores/ui';
import Panel from '@/components/Panel';
import Rail from '@/components/Rail';
import Inspector from '@/components/Inspector';

/** one panel full-viewport, with the series browser and inspector docked alongside (#panel=<id>) */
export default function FocusView() {
  let root!: HTMLDivElement, back!: HTMLButtonElement;
  const p = () => board.panels.find((x) => x.id === focusId());
  onMount(() => { document.body.classList.add('bfocused'); setTimeout(() => back?.focus(), 0); });
  onCleanup(() => document.body.classList.remove('bfocused'));
  const fullscreen = () => {
    try {
      const pr = document.fullscreenElement ? document.exitFullscreen() : root.requestFullscreen();
      if (pr && pr.catch) pr.catch(() => showToast('fullscreen not available here'));
    } catch { showToast('fullscreen not available here'); }
  };
  return (
    <div class="bfocus" ref={root} role="dialog" aria-label="panel focus view" data-board-scope
      onKeyDown={(e) => { if (e.key === 'Escape' && !e.defaultPrevented && !/^(INPUT|SELECT|TEXTAREA)$/.test((e.target as Element).tagName)) { closeFocus(); e.preventDefault(); } }}>
      <div class="fbar">
        <button type="button" ref={back} title="back to the board" aria-label="back to the board" onClick={closeFocus}>&lt; back to board</button>
        <span class="ftitle">{p() ? titleOf(p()!) : ''}</span>
        <span class="sp" />
        <button type="button" title="toggle browser fullscreen" aria-label="toggle browser fullscreen" onClick={fullscreen}>fullscreen</button>
      </div>
      <div class="fbody">
        <Rail />
        <div class="fmain" data-canvas="focus">
          <Show when={p()} keyed>{(pp) => <Panel panel={pp} inFocus />}</Show>
        </div>
        <Show when={inspector()}><Inspector /></Show>
      </div>
    </div>
  );
}
