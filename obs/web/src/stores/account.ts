// Account state for the header and the sign-in / account views. The portal lives in the same hash
// scheme as the rest of the app: #view=login and #view=account. The paths /login and /account (the
// sign-in mail link is <public>/login#email=<token>, `ssh … login` prints <public>/login#ssh=<token>)
// are entry aliases only: on load they are rewritten to the parent path + #view=…, and a one-time
// token in the fragment is taken into memory and removed from the URL and history before anything
// else reads it.
import { createSignal } from 'solid-js';
import { api, onSignedOut, useLayers, ApiError, type Me, type Methods, type SignedIn } from '@/logic/account';
import { hash, setHash } from '@/lib/hash';
import { showToast } from '@/stores/ui';

export type PortalView = 'login' | 'account';
export const portalView = (): PortalView | null => { const v = hash().view; return v === 'login' || v === 'account' ? v : null; };
export const goLogin = (): void => setHash({ view: 'login', panel: null });
export const goAccount = (): void => setHash({ view: 'account', panel: null });

// ------------------------------------------------------------------ entry: /login, /account, #email=, #ssh=
let pendingToken: { kind: 'email' | 'ssh'; token: string } | null = null;
/** run once, synchronously, before the app reads the URL */
export function takeEntry(): void {
  const m = location.hash.match(/^#(email|ssh)=([A-Za-z0-9_-]{8,200})$/);
  if (m) pendingToken = { kind: m[1] as 'email' | 'ssh', token: m[2] };
  const p = location.pathname.match(/^(.*\/)(login|account)\/?$/);
  if (!m && !p) return;
  const view: PortalView = m ? 'login' : (p![2] as PortalView);
  const path = p ? p[1] : location.pathname;
  history.replaceState(null, '', path + location.search + '#view=' + view);
  setHash({ run: null, project: null, board: null, panel: null, view });
}

// ------------------------------------------------------------------ state
const [methods, setMethods] = createSignal<Methods | null>(null);
/** null while unknown; false = no portal on this server (static mode, old server) */
const [available, setAvailable] = createSignal<boolean | null>(null);
const [me, setMe] = createSignal<Me | null>(null);
const [meError, setMeError] = createSignal('');
const [consuming, setConsuming] = createSignal<'email' | 'ssh' | null>(null);
const [consumeError, setConsumeError] = createSignal('');
export { methods, available, me, meError, consuming, consumeError };
/** the signed-in account id (from /me, or from /auth/methods before /me has loaded) */
export const accountId = (): string | null => me()?.account.id ?? methods()?.account?.id ?? null;
/** what the header shows: the account's email (or first credential), else its id */
export const accountLabel = (): string | null => methods()?.account?.label || me()?.creds.find((c) => c.kind === 'email')?.label || accountId();

onSignedOut(() => {
  setMe(null); setMethods((m) => (m ? { ...m, account: null } : m));
  if (portalView() === 'account') { showToast('your session ended; sign in again'); goLogin(); }
});

export async function refreshMe(): Promise<void> {
  try { setMe(await api.me()); setMeError(''); }
  catch (e) { setMe(null); if (!(e instanceof ApiError && e.status === 401)) setMeError((e as Error).message); }
}
async function refreshMethods(): Promise<void> {
  try {
    const m = await api.methods();
    if (!m || typeof m !== 'object' || !('passkey' in m)) throw new Error('no portal');
    setMethods(m); setAvailable(true);
  } catch { setMethods(null); setAvailable(false); }
}

/** a sign-in (or link) succeeded: say what happened, reload the account, show it */
export async function signedIn(r: SignedIn, what: string): Promise<void> {
  showToast(r.linked ? `${what} added to this account` : r.created ? 'new account created' : 'signed in');
  await refreshMethods();
  await refreshMe();
  goAccount();
}

export async function signOut(): Promise<void> {
  try { await api.logout(); } catch { /* the cookie may already be gone */ }
  setMe(null); setMethods((m) => (m ? { ...m, account: null } : m));
  showToast('signed out');
  goLogin();
}

async function consumePending(): Promise<void> {
  const t = pendingToken; pendingToken = null;
  if (!t) return;
  setConsuming(t.kind); setConsumeError('');
  try {
    const r = t.kind === 'email' ? await api.emailConsume(t.token) : await api.sshConsume(t.token);
    setConsuming(null);
    await signedIn(r, t.kind === 'email' ? 'email address' : 'SSH key');
  } catch (e) {
    setConsuming(null);
    const err = e as ApiError;
    // the contract names no code for an expired or used token: any 4xx other than these is that
    const generic = t.kind === 'email'
      ? 'This sign-in link has expired or was already used. Links work once, for 15 minutes. Request a new one below.'
      : 'This SSH sign-in link has expired or was already used. Links work once, for 10 minutes. Run the ssh command again.';
    setConsumeError(err instanceof ApiError && err.status >= 400 && err.status < 500 && !['cred_taken', 'rate_limited'].includes(err.code) ? generic : err.message);
  }
}

/** the dev-only mock (?mock in the dev server) swaps the API layers before the first call */
async function maybeMock(): Promise<void> {
  if (!import.meta.env.DEV) return;
  const q = new URLSearchParams(location.search);
  if (!q.has('mock')) return;
  const m = await import('@/logic/accountMock');
  useLayers(m.mockLayers(q.get('mock') || ''));
}

export async function initAccount(): Promise<void> {
  await maybeMock();
  await refreshMethods();
  if (!available()) return;
  if (pendingToken) await consumePending();
  else if (methods()?.account) await refreshMe();
  if (!me() && portalView() === 'account' && !consuming()) goLogin();
}

// ------------------------------------------------------------------ anonymous project tokens held by this browser
const ANON_PREFIX = 'obs-anon-token:';
export function heldAnonTokens(): { project: string; token: string }[] {
  const out: { project: string; token: string }[] = [];
  try {
    for (let i = 0; i < sessionStorage.length; i++) {
      const k = sessionStorage.key(i);
      if (!k || !k.startsWith(ANON_PREFIX)) continue;
      const v = sessionStorage.getItem(k);
      if (v) out.push({ project: k.slice(ANON_PREFIX.length), token: v });
    }
  } catch { /* storage blocked */ }
  return out.sort((a, b) => a.project.localeCompare(b.project));
}
