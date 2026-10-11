// The four sign-in methods. The same controls sign in (signed out) or add a credential to the
// current account (signed in): the server links a new credential to the session's account.
import { createSignal, onMount, Show } from 'solid-js';
import { api, passkeyLogin, passkeyRegister, webauthn, zafu, zafuLogin } from '@/logic/account';
import type { ZafuHandle } from '@/logic/zafu';
import { accountId, methods, signedIn } from '@/stores/account';
import { CopyLine, Msg, useAction } from '@/components/PortalBits';

export function EmailMethod() {
  const [addr, setAddr] = createSignal('');
  const [sent, setSent] = createSignal('');
  const a = useAction();
  const send = (e: Event) => {
    e.preventDefault();
    const v = addr().trim();
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(v)) { a.setErr('Enter an email address.'); return; }
    void a.run(async () => { await api.emailRequest(v); setSent(v); });
  };
  return (
    <Show when={methods()?.email} fallback={<p class="note">Email sign-in is not configured on this server.</p>}>
      <Show when={!sent()} fallback={
        <div class="pok" role="status">
          <b>Check your inbox.</b> If {sent()} can receive mail, a link is on its way. It works once, for 15 minutes{accountId() ? ', and adds the address to this account' : ''}.
          <div class="prow"><button type="button" title="send a link to another address" aria-label="use a different email address" onClick={() => { setSent(''); }}>use a different address</button></div>
        </div>
      }>
        <form class="prow" onSubmit={send}>
          <input type="email" autocomplete="email" placeholder="you@example.org" name="email" aria-label="email address" value={addr()} onInput={(e) => setAddr(e.currentTarget.value)} />
          <button type="submit" class="pri" disabled={a.busy()} title="email a one-time sign-in link" aria-label="send sign-in link">{a.busy() ? 'sending…' : 'Send sign-in link'}</button>
        </form>
        <Msg err={a.err()} />
      </Show>
    </Show>
  );
}

export function PasskeyMethod(props: { mode: 'login' | 'add' }) {
  const a = useAction();
  const [label, setLabel] = createSignal('');
  const login = () => a.run(async () => signedIn(await passkeyLogin(), 'passkey'));
  const register = () => a.run(async () => signedIn(await passkeyRegister(label().trim() || undefined), 'passkey'));
  return (
    <Show when={webauthn().available()} fallback={<p class="note">This browser does not support passkeys.</p>}>
      <Show when={props.mode === 'login'} fallback={
        <div class="prow">
          <input type="text" placeholder="label, e.g. work laptop" name="passkey-label" aria-label="passkey label (optional)" value={label()} onInput={(e) => setLabel(e.currentTarget.value)} maxLength={60} />
          <button type="button" class="pri" disabled={a.busy()} title="create a passkey on this device and add it to the account" aria-label="add a passkey" onClick={register}>{a.busy() ? 'waiting for the browser…' : 'Add a passkey'}</button>
        </div>
      }>
        <div class="prow">
          <button type="button" class="pri" disabled={a.busy()} title="use a passkey saved on this device or a security key" aria-label="sign in with passkey" onClick={login}>{a.busy() ? 'waiting for the browser…' : 'Sign in with passkey'}</button>
          <button type="button" disabled={a.busy()} title="create a new account whose first credential is a passkey on this device" aria-label="create an account with a new passkey" onClick={register}>new account with a passkey</button>
        </div>
      </Show>
      <Msg err={a.err()} />
    </Show>
  );
}

// detection runs once per page; both views share it
const [zafuState, setZafuState] = createSignal<'checking' | 'none' | ZafuHandle>('checking');
let zafuProbe: Promise<void> | null = null;
function probeZafu(): void {
  zafuProbe ??= (async () => {
    let h = await zafu().detect();
    // the provider is injected at document_start; one retry covers a slow extension start
    if (!h) { await new Promise((r) => setTimeout(r, 600)); h = await zafu().detect(); }
    setZafuState(h ?? 'none');
  })();
}

export function ZafuMethod(props: { mode: 'login' | 'add' }) {
  const a = useAction();
  onMount(probeZafu);
  const go = (h: ZafuHandle) => a.run(async () => signedIn(await zafuLogin(h), 'zafu key'));
  const label = () => (props.mode === 'login' ? 'Sign in with zafu' : 'Add zafu');
  return (
    <>
      <Show when={zafuState() === 'checking'}><p class="note">looking for the zafu extension…</p></Show>
      <Show when={zafuState() === 'none'}>
        <div class="prow">
          <a class="btnlink" href="https://zafu.pro" target="_blank" rel="noopener" title="zafu.pro: the zafu wallet browser extension">Install zafu</a>
          <span class="note">not detected in this browser (Chromium-based browsers only)</span>
        </div>
      </Show>
      <Show when={typeof zafuState() === 'object' && zafuState() as ZafuHandle}>{(h) => (
        <div class="prow">
          <button type="button" class="pri" disabled={a.busy()} title="zafu signs a one-time challenge with its key for this site" aria-label={label()} onClick={() => go(h())}>{a.busy() ? 'approve in zafu…' : label()}</button>
        </div>
      )}</Show>
      <Msg err={a.err()} />
    </>
  );
}

export function SshMethod() {
  return (
    <Show when={methods()?.ssh} fallback={<p class="note">SSH sign-in is not enabled on this server.</p>}>{(s) => (
      <>
        <CopyLine text={s().command} what="the ssh sign-in command" />
        <p class="note">Prints a one-time link for the key you connect with; open it here within 10 minutes.{accountId() ? ' Signed in, it adds that key to this account.' : ''}</p>
      </>
    )}</Show>
  );
}
