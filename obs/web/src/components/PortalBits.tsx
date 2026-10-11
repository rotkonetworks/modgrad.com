// Small pieces shared by the sign-in and account views.
import { createSignal, Show, type JSX } from 'solid-js';
import { ago } from '@/lib/fmt';

const iso = (u: number) => new Date(u * 1000).toISOString().replace('T', ' ').slice(0, 16) + ' UTC';
/** "3d ago" / "in 5d" with the exact time on hover; "never" for null */
export function When(props: { unix: number | null | undefined; never?: string }) {
  return (
    <Show when={props.unix} fallback={<span class="dim">{props.never ?? 'never'}</span>}>{(u) => {
      const d = () => Date.now() / 1000 - u();
      return <time datetime={new Date(u() * 1000).toISOString()} title={iso(u())}>{d() >= 0 ? ago(d()) + ' ago' : 'in ' + ago(-d())}</time>;
    }}</Show>
  );
}

export async function copyText(text: string): Promise<boolean> {
  try { await navigator.clipboard.writeText(text); return true; } catch { /* insecure context or denied */ }
  try {
    const ta = document.createElement('textarea'); ta.value = text; ta.setAttribute('readonly', ''); ta.style.position = 'fixed'; ta.style.opacity = '0';
    document.body.appendChild(ta); ta.select(); const ok = document.execCommand('copy'); ta.remove(); return ok;
  } catch { return false; }
}

/** a command or secret on one line with a labelled copy button */
export function CopyLine(props: { text: string; what: string }) {
  const [state, setState] = createSignal('');
  const copy = async () => { setState((await copyText(props.text)) ? 'copied' : 'select and copy by hand'); setTimeout(() => setState(''), 2500); };
  return (
    <div class="copyline">
      <code>{props.text}</code>
      <button type="button" title={'copy ' + props.what} aria-label={'copy ' + props.what} onClick={copy}>{state() || 'copy'}</button>
    </div>
  );
}

/** a destructive action confirmed in place: first click arms it, the second does it */
export function ConfirmButton(props: { label: string; confirm: string; what: string; run: () => Promise<unknown> | void; disabled?: boolean }) {
  const [armed, setArmed] = createSignal(false);
  const [busy, setBusy] = createSignal(false);
  const go = async () => { setBusy(true); try { await props.run(); } finally { setBusy(false); setArmed(false); } };
  return (
    <Show when={armed()} fallback={<button type="button" disabled={props.disabled} title={props.label + ' ' + props.what} aria-label={props.label + ' ' + props.what} onClick={() => setArmed(true)}>{props.label}</button>}>
      <span class="confirm">
        <button type="button" class="danger" disabled={busy()} title={props.confirm + ': ' + props.what} aria-label={props.confirm + ': ' + props.what} onClick={go}>{busy() ? 'working…' : props.confirm}</button>
        <button type="button" disabled={busy()} title="keep it" aria-label={'cancel: keep ' + props.what} onClick={() => setArmed(false)}>cancel</button>
      </span>
    </Show>
  );
}

/** an inline error or result line under a form */
export function Msg(props: { err?: string; ok?: string | JSX.Element }) {
  return (
    <>
      <Show when={props.err}><div class="err" role="alert">{props.err}</div></Show>
      <Show when={!props.err && props.ok}><div class="pok" role="status">{props.ok}</div></Show>
    </>
  );
}

/** run an async action with a busy flag and an error message */
export function useAction(): { busy: () => boolean; err: () => string; setErr: (s: string) => void; run: (f: () => Promise<unknown>) => Promise<boolean> } {
  const [busy, setBusy] = createSignal(false);
  const [err, setErr] = createSignal('');
  const run = async (f: () => Promise<unknown>) => {
    setBusy(true); setErr('');
    try { await f(); return true; } catch (e) { setErr((e as Error).message || String(e)); return false; } finally { setBusy(false); }
  };
  return { busy, err, setErr, run };
}
