import { createMemo, createSignal, For, onCleanup, onMount, Show } from 'solid-js';
import { Portal } from 'solid-js/web';
import { pop, closePop, toast, hideToast, paletteOpen, setPaletteOpen } from '@/stores/ui';
import { undo } from '@/stores/board';
import { commands, type Action, type Ctx } from '@/logic/commands';

/** the one open menu or popover (anchored under its button; outside click or Esc closes it) */
export function Popover() {
  return (
    <Show when={pop()} keyed>{(p) => {
      let el!: HTMLDivElement;
      const off = (ev: PointerEvent) => { if (el && !el.contains(ev.target as Node) && ev.target !== p.anchor && !p.anchor.contains(ev.target as Node)) closePop(); };
      const key = (ev: KeyboardEvent) => { if (ev.key === 'Escape') { closePop(); (p.anchor as HTMLElement).focus?.(); } };
      onMount(() => { setTimeout(() => document.addEventListener('pointerdown', off, true), 0); document.addEventListener('keydown', key); if (p.kind === 'menu') el.querySelector('button')?.focus(); });
      onCleanup(() => { document.removeEventListener('pointerdown', off, true); document.removeEventListener('keydown', key); });
      return (
        <Portal>
          <div ref={el} class={p.kind === 'menu' ? 'bmenu' : 'bpop'} role="dialog" aria-label={p.label} style={{ left: p.x + 'px', top: p.y + 'px' }}>
            <Show when={p.kind === 'menu'} fallback={p.render!()}>
              <For each={p.items}>{(it) => <button type="button" title={it.title || it.label} aria-label={it.title || it.label} onClick={() => { closePop(); it.run(); }}>{it.label}</button>}</For>
            </Show>
          </div>
        </Portal>
      );
    }}</Show>
  );
}

export function ToastView() {
  return (
    <Show when={toast()} keyed>{(t) => (
      <div class="btoast" role="status">
        <span>{t.text}</span>
        <Show when={t.undo}><button type="button" title="undo the last change" aria-label="undo the last change" onClick={() => { hideToast(); undo(); }}>undo</button></Show>
      </div>
    )}</Show>
  );
}

/** Ctrl/Cmd+K command palette (toggleable in settings); also opened by the "commands" button */
export function Palette(props: { ctx: () => Ctx; run: (a: Action) => void }) {
  return (
    <Show when={paletteOpen()}>{(() => {
      const [text, setText] = createSignal('');
      const [cur, setCur] = createSignal(0);
      const items = createMemo(() => commands(text(), props.ctx()));
      const close = () => setPaletteOpen(false);
      const go = (a: Action) => { close(); props.run(a); };
      let inp!: HTMLInputElement;
      onMount(() => inp.focus());
      return (
        <div class="bpal" onPointerDown={(e) => { if (e.target === e.currentTarget) close(); }}>
          <div class="box" role="dialog" aria-label="command palette">
            <div class="top">
              <input ref={inp} placeholder="command: forecast to 10k, split by seed, add loss, follow, new panel ..." spellcheck={false} aria-label="command"
                value={text()} onInput={(e) => { setText(e.currentTarget.value); setCur(0); }}
                onKeyDown={(e) => {
                  const n = items().length;
                  if (e.key === 'ArrowDown') { setCur(Math.min(cur() + 1, n - 1)); e.preventDefault(); }
                  else if (e.key === 'ArrowUp') { setCur(Math.max(cur() - 1, 0)); e.preventDefault(); }
                  else if (e.key === 'Enter' && items()[cur()]) go(items()[cur()].action);
                  else if (e.key === 'Escape') { close(); e.preventDefault(); }
                }} />
              <button type="button" title="close the command palette" aria-label="close the command palette" onClick={close}>close</button>
            </div>
            <div class="list">
              <For each={items()} fallback={<div class="bnote" style={{ padding: '6px' }}>no matching command</div>}>{(c, i) => (
                <button type="button" classList={{ cur: i() === Math.min(cur(), items().length - 1) }} title={c.label} aria-label={c.label} onClick={() => go(c.action)}>{c.label}</button>
              )}</For>
            </div>
          </div>
        </div>
      );
    })()}</Show>
  );
}
