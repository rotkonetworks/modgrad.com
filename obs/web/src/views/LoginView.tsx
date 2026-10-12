import { Show } from 'solid-js';
import { webauthn } from '@/logic/account';
import { accountId, available, consumeError, consuming, goAccount } from '@/stores/account';
import { EmailMethod, PasskeyMethod, SshMethod, ZafuMethod } from '@/components/SignInMethods';

/** #view=login: the four sign-in methods, and the landing for #email= / #ssh= links */
export default function LoginView() {
  return (
    <section class="portal">
      <h2>Sign in</h2>
      <Show when={consuming()}>{(k) => <div class="pok" role="status">Signing in with the {k() === 'email' ? 'email' : 'SSH'} link…</div>}</Show>
      <Show when={consumeError()}><div class="err" role="alert">{consumeError()}</div></Show>
      <Show when={available() === false}>
        <p class="note">This server does not run accounts (static mode, or a modgrad-server without the portal). Boards here are read from files and need no sign-in.</p>
      </Show>
      <Show when={available() === null}><p class="note">loading…</p></Show>
      <Show when={available()}>
        <Show when={accountId()} fallback={<p class="note">One account, any number of credentials: each method below works on its own, and you can add the others later as recovery.</p>}>
          <p class="note">
            Signed in as <span class="mono">{accountId()}</span>. A method used here is added to this account.{' '}
            <a href="#view=account" onClick={(e) => { e.preventDefault(); goAccount(); }}>Account page</a>
          </p>
        </Show>
        <div class="pgrid">
          <div class="pbox"><h3>Email</h3><EmailMethod /></div>
          <Show when={webauthn().available()}><div class="pbox"><h3>Passkey</h3><PasskeyMethod mode="login" /></div></Show>
          <div class="pbox"><h3>zafu wallet</h3><ZafuMethod mode="login" /></div>
          <div class="pbox"><h3>SSH key</h3><SshMethod /></div>
        </div>
      </Show>
    </section>
  );
}
