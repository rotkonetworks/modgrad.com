// Portal API client: accounts, sign-in methods, projects, tokens (contract: PORTAL_SPEC.md at the
// repo root). No DOM, no Solid. The transport, WebAuthn and zafu layers are swappable so a mock
// (accountMock.ts, dev only) can drive every path, including the byte conversions, without a server,
// an authenticator or the extension.
import { dataURL } from '@/lib/base';
import { detectZafu, zafuSign, type ZafuHandle } from '@/logic/zafu';

// ------------------------------------------------------------------ wire types
export type CredKind = 'ssh' | 'email' | 'zafu' | 'passkey' | (string & {});
export interface AccountSummary { id: string; created: number; /** the email, else the first credential's label */ label?: string | null }
export interface Methods { email: boolean; zafu: boolean; passkey: boolean; ssh: { command: string } | null; account: AccountSummary | null }
export interface Cred { kind: CredKind; id: string; label: string; added: number; last_used: number | null }
export interface Project { name: string; visibility: 'private' | 'public'; runs: number; last_unix: number | null; anon: boolean; expires?: number | null }
export interface Session { id: string; created: number; last_seen: number; ua: string; current: boolean }
export interface Token { id: string; project: string; label: string; created: number; expires: number | null; last_used: number | null }
export interface Me { account: AccountSummary; creds: Cred[]; projects: Project[]; sessions: Session[]; tokens: Token[] }
export interface SignedIn { account: AccountSummary | string; linked?: boolean; created?: boolean }
export interface MintedToken { id: string; token: string }
export interface ZafuChallenge { challenge_hex: string; statement: string; expires: number }

// ------------------------------------------------------------------ errors
export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string) { super(message); }
}
const FRIENDLY: Record<string, string> = {
  signed_out: 'You are signed out. Sign in again.',
  cred_taken: 'That credential already belongs to another account. Sign in with it there and remove it first; accounts are never merged automatically.',
  name_taken: 'That project name is taken.',
  bad_name: 'Project names are 3 to 40 characters: lowercase letters, digits and hyphens, starting with a letter or digit.',
  last_cred: 'This is the only way into the account. Add another credential before removing it.',
};
function friendly(status: number, code: string, server: string, retryAfter: string | null): string {
  if (status === 429) {
    const s = retryAfter && /^\d+$/.test(retryAfter) ? +retryAfter : 0;
    return 'Too many attempts. ' + (s ? `Try again in ${s < 90 ? s + ' seconds' : Math.ceil(s / 60) + ' minutes'}.` : 'Wait a few minutes and try again.');
  }
  if (FRIENDLY[code]) return FRIENDLY[code];
  if (status === 0) return 'The server could not be reached.';
  return server || `Request failed (HTTP ${status}).`;
}

// ------------------------------------------------------------------ transport (swappable)
export interface Reply { status: number; body: unknown; retryAfter: string | null }
export type Transport = (method: string, path: string, body?: unknown) => Promise<Reply>;
const fetchTransport: Transport = async (method, path, body) => {
  let r: Response;
  try {
    r = await fetch(dataURL('api/v1/' + path), {
      method,
      headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
      credentials: 'same-origin',
    });
  } catch { return { status: 0, body: null, retryAfter: null }; }
  let parsed: unknown = null;
  if (r.status !== 204) { const t = await r.text(); try { parsed = t ? JSON.parse(t) : null; } catch { parsed = { error: '', code: 'not_json' }; } }
  return { status: r.status, body: parsed, retryAfter: r.headers.get('Retry-After') };
};

/** a WebAuthn layer: the browser's, or a mock's */
export interface WebAuthn { available(): boolean; create(o: CredentialCreationOptions): Promise<Credential | null>; get(o: CredentialRequestOptions): Promise<Credential | null> }
const browserWebAuthn: WebAuthn = {
  available: () => typeof window !== 'undefined' && !!window.PublicKeyCredential && !!navigator.credentials,
  create: (o) => navigator.credentials.create(o),
  get: (o) => navigator.credentials.get(o),
};
/** a zafu layer: the extension bridge (logic/zafu.ts), or a mock's */
export interface ZafuDriver { detect(): Promise<ZafuHandle | null>; sign(h: ZafuHandle, challengeHex: string, statement: string): Promise<{ signature: string; publicKey: string }> }
const extensionZafu: ZafuDriver = { detect: detectZafu, sign: zafuSign };

const impl = { transport: fetchTransport, webauthn: browserWebAuthn, zafu: extensionZafu };
/** swap layers (the dev mock does this before the first call) */
export function useLayers(l: Partial<typeof impl>): void { Object.assign(impl, l); }
export const webauthn = (): WebAuthn => impl.webauthn;
export const zafu = (): ZafuDriver => impl.zafu;

/** listeners for 401 signed_out from any call (the account store clears its state) */
const signedOutHooks: (() => void)[] = [];
export const onSignedOut = (fn: () => void): void => { signedOutHooks.push(fn); };

async function call<T>(method: string, path: string, body?: unknown): Promise<T> {
  const r = await impl.transport(method, path, body);
  if (r.status >= 200 && r.status < 300) return r.body as T;
  const b = (r.body && typeof r.body === 'object' ? r.body : {}) as { error?: string; code?: string };
  const code = b.code || (r.status === 401 ? 'signed_out' : r.status === 429 ? 'rate_limited' : '');
  if (r.status === 401) for (const f of signedOutHooks) f();
  throw new ApiError(r.status, code, friendly(r.status, code, b.error || '', r.retryAfter));
}

// ------------------------------------------------------------------ base64url <-> bytes
export function b64urlToBuf(s: string): ArrayBuffer {
  const b = atob(s.replace(/-/g, '+').replace(/_/g, '/') + '==='.slice((s.length + 3) % 4));
  const u = new Uint8Array(b.length);
  for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i);
  return u.buffer;
}
export function bufToB64url(buf: ArrayBuffer | ArrayBufferView): string {
  const u = buf instanceof ArrayBuffer ? new Uint8Array(buf) : new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  let s = '';
  for (let i = 0; i < u.length; i++) s += String.fromCharCode(u[i]);
  return btoa(s).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

// ------------------------------------------------------------------ project names (mirror of the server rule)
export const PROJECT_NAME = /^[a-z0-9][a-z0-9-]{2,39}$/;
export const RESERVED_NAMES = ['api', 'login', 'account', 'admin', 'lab', 'docs', 'new', 'me', 'static', 'assets'];
/** null when the name is acceptable, else why not */
export function projectNameProblem(n: string): string | null {
  if (!n) return 'Enter a name.';
  if (!PROJECT_NAME.test(n)) return FRIENDLY.bad_name;
  if (n.startsWith('anon-')) return 'The anon- prefix is reserved for anonymous projects.';
  if (RESERVED_NAMES.includes(n)) return `"${n}" is reserved.`;
  return null;
}

// ------------------------------------------------------------------ endpoints
export const api = {
  methods: () => call<Methods>('GET', 'auth/methods'),
  me: () => call<Me>('GET', 'me'),
  logout: () => call<void>('POST', 'auth/logout'),
  emailRequest: (email: string) => call<void>('POST', 'auth/email', { email }),
  emailConsume: (token: string) => call<SignedIn>('POST', 'auth/email/consume', { token }),
  sshConsume: (token: string) => call<SignedIn>('POST', 'auth/ssh/consume', { token }),
  zafuChallenge: () => call<ZafuChallenge>('POST', 'auth/zafu/challenge'),
  zafuVerify: (b: { challenge_hex: string; signature: string; public_key: string }) => call<SignedIn>('POST', 'auth/zafu/verify', b),
  passkeyRegisterOptions: () => call<unknown>('POST', 'auth/passkey/register/options'),
  passkeyRegisterVerify: (b: unknown) => call<SignedIn>('POST', 'auth/passkey/register/verify', b),
  passkeyLoginOptions: () => call<unknown>('POST', 'auth/passkey/login/options'),
  passkeyLoginVerify: (b: unknown) => call<SignedIn>('POST', 'auth/passkey/login/verify', b),
  createProject: (name: string, visibility: 'private' | 'public') => call<Project>('POST', 'me/projects', { name, visibility }),
  setVisibility: (name: string, visibility: 'private' | 'public') => call<unknown>('PATCH', 'me/projects/' + encodeURIComponent(name), { visibility }),
  claim: (project: string, token: string) => call<unknown>('POST', 'me/claim', { project, token }),
  mintToken: (project: string, label: string, expires_days?: number) => call<MintedToken>('POST', 'me/tokens', expires_days ? { project, label, expires_days } : { project, label }),
  revokeToken: (id: string) => call<void>('DELETE', 'me/tokens/' + encodeURIComponent(id)),
  addSshKey: (public_key: string) => call<unknown>('POST', 'me/creds/ssh', { public_key }),
  removeCred: (id: string) => call<void>('DELETE', 'me/creds/' + encodeURIComponent(id)),
  revokeSession: (id: string) => call<void>('DELETE', 'me/sessions/' + encodeURIComponent(id)),
  deleteAccount: (confirm: string) => call<void>('DELETE', 'me', { confirm }),
};

// ------------------------------------------------------------------ composite flows
/** options JSON (b64url fields; bare or wrapped in {publicKey}) -> WebAuthn creation options */
function creationOptions(j: unknown): CredentialCreationOptions {
  const o = ((j as { publicKey?: unknown }).publicKey ?? j) as Record<string, unknown> & { user: Record<string, unknown>; excludeCredentials?: { id: string }[] };
  return { publicKey: {
    ...(o as object),
    challenge: b64urlToBuf(o.challenge as string),
    user: { ...o.user, id: b64urlToBuf(o.user.id as string) },
    excludeCredentials: (o.excludeCredentials || []).map((c) => ({ ...c, id: b64urlToBuf(c.id) })),
  } as PublicKeyCredentialCreationOptions };
}
function requestOptions(j: unknown): CredentialRequestOptions {
  const o = ((j as { publicKey?: unknown }).publicKey ?? j) as Record<string, unknown> & { allowCredentials?: { id: string }[] };
  return { publicKey: {
    ...(o as object),
    challenge: b64urlToBuf(o.challenge as string),
    allowCredentials: (o.allowCredentials || []).map((c) => ({ ...c, id: b64urlToBuf(c.id) })),
  } as PublicKeyCredentialRequestOptions };
}
const NO_PASSKEY = 'No passkey was used (the browser dialog was closed or timed out).';
function webauthnError(e: unknown): Error {
  if (e instanceof ApiError) return e;
  const n = (e as { name?: string })?.name;
  if (n === 'NotAllowedError' || n === 'AbortError') return new Error(NO_PASSKEY);
  if (n === 'InvalidStateError') return new Error('This passkey is already registered.');
  return new Error('Passkey error: ' + ((e as Error)?.message || String(e)));
}

/** sign in with a discoverable passkey */
export async function passkeyLogin(): Promise<SignedIn> {
  const opts = requestOptions(await api.passkeyLoginOptions());
  let c: PublicKeyCredential | null;
  try { c = await impl.webauthn.get(opts) as PublicKeyCredential | null; } catch (e) { throw webauthnError(e); }
  if (!c) throw new Error(NO_PASSKEY);
  const r = c.response as AuthenticatorAssertionResponse;
  return api.passkeyLoginVerify({
    id: c.id, rawId: bufToB64url(c.rawId),
    response: {
      clientDataJSON: bufToB64url(r.clientDataJSON), authenticatorData: bufToB64url(r.authenticatorData),
      signature: bufToB64url(r.signature), userHandle: r.userHandle ? bufToB64url(r.userHandle) : null,
    },
  });
}
/** create a passkey: signed in -> added to the account; signed out -> a new account */
export async function passkeyRegister(label?: string): Promise<SignedIn> {
  const opts = creationOptions(await api.passkeyRegisterOptions());
  let c: PublicKeyCredential | null;
  try { c = await impl.webauthn.create(opts) as PublicKeyCredential | null; } catch (e) { throw webauthnError(e); }
  if (!c) throw new Error(NO_PASSKEY);
  const r = c.response as AuthenticatorAttestationResponse;
  const body: Record<string, unknown> = {
    id: c.id, rawId: bufToB64url(c.rawId),
    response: { clientDataJSON: bufToB64url(r.clientDataJSON), attestationObject: bufToB64url(r.attestationObject) },
  };
  if (label) body.label = label;
  return api.passkeyRegisterVerify(body);
}
/** challenge -> zafu signs it with this site's ZID key -> verify (signs in, or links when signed in) */
export async function zafuLogin(h: ZafuHandle): Promise<SignedIn> {
  const ch = await api.zafuChallenge();
  const s = await impl.zafu.sign(h, ch.challenge_hex, ch.statement);
  return api.zafuVerify({ challenge_hex: ch.challenge_hex, signature: s.signature, public_key: s.publicKey });
}
