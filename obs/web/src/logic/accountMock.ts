// Dev-only in-memory stand-in for the portal API (PORTAL_SPEC.md), loaded by stores/account.ts only
// when the dev server runs with ?mock[=flags]; production builds never import it. It also swaps in
// fake WebAuthn and zafu layers, so the byte conversions in logic/account.ts still run.
//
// flags (comma separated): in = start signed in, single = signed in with one credential (remove
// -> last_cred), nozafu = no wallet detected, noemail = email sign-in not configured.
// Trigger cases: email containing "limit" -> 429; project name "taken-name" -> 409 name_taken;
// link token starting "expired" (e.g. /login#email=expiredtoken123) -> 410; an SSH key containing
// "taken" -> 409 cred_taken.
import { bufToB64url, type Cred, type Me, type Project, type Reply, type Session, type Token, type Transport, type WebAuthn, type ZafuDriver } from '@/logic/account';

const now = () => Date.now() / 1000;
const rnd = (n: number) => { const u = new Uint8Array(n); crypto.getRandomValues(u); return u; };
const hex = (u: Uint8Array) => Array.from(u, (x) => x.toString(16).padStart(2, '0')).join('');
const id = (p: string) => p + '_' + bufToB64url(rnd(9));
const ok = (body: unknown = null, status = 200): Reply => ({ status, body, retryAfter: null });
const fail = (status: number, code: string, error: string, retryAfter: string | null = null): Reply => ({ status, body: { code, error }, retryAfter });

export function mockLayers(flagStr: string): { transport: Transport; webauthn: WebAuthn; zafu: ZafuDriver } {
  const flags = new Set(flagStr.split(',').filter(Boolean));
  const t0 = now();
  const seed = (): Me => ({
    account: { id: 'acct_' + 'q7m2xk4r9t3w8p5n6v1c0b'.toUpperCase(), created: t0 - 86400 * 40 },
    creds: flags.has('single')
      ? [{ kind: 'email', id: 'cred_e1', label: 'tommi@example.org', added: t0 - 86400 * 40, last_used: t0 - 3600 }]
      : [
        { kind: 'ssh', id: 'cred_s1', label: 'SHA256:Xw3zq8eB0mLr2vYQ1kH5o7sT9uFiA4dNcPgJ6bE  laptop', added: t0 - 86400 * 40, last_used: t0 - 7200 },
        { kind: 'email', id: 'cred_e1', label: 'tommi@example.org', added: t0 - 86400 * 12, last_used: t0 - 86400 * 2 },
        { kind: 'passkey', id: 'cred_p1', label: 'this laptop', added: t0 - 86400 * 3, last_used: null },
      ],
    projects: [
      { name: 'maze-ctm', visibility: 'private', runs: 14, last_unix: t0 - 600, anon: false },
      { name: 'sudoku-sheaf', visibility: 'public', runs: 3, last_unix: t0 - 86400 * 6, anon: false },
      { name: 'anon-7f3kq2', visibility: 'private', runs: 1, last_unix: t0 - 86400, anon: true, expires: t0 + 86400 * 5 },
    ],
    sessions: [
      { id: 'sess_a', created: t0 - 86400 * 3, last_seen: t0 - 20, ua: navigator.userAgent.slice(0, 80), current: true },
      { id: 'sess_b', created: t0 - 86400 * 9, last_seen: t0 - 86400 * 2, ua: 'Mozilla/5.0 (X11; Linux x86_64; rv:140.0) Gecko/20100101 Firefox/140.0', current: false },
    ],
    tokens: [{ id: 'tok_1', project: 'maze-ctm', label: 'atomman trainer', created: t0 - 86400 * 5, expires: t0 + 86400 * 85, last_used: t0 - 300 }],
  });
  let me: Me | null = flags.has('in') || flags.has('single') ? seed() : null;
  // an anonymous board this browser created earlier (what #new leaves in sessionStorage)
  try { if (!sessionStorage.getItem('obs-anon-token:anon-k3j9x2')) sessionStorage.setItem('obs-anon-token:anon-k3j9x2', 'obsp_mockanon' + hex(rnd(8))); } catch { /* ignore */ }

  const signIn = (kind: Cred['kind'], label: string) => {
    if (me) { me.creds.push({ kind, id: id('cred'), label, added: now(), last_used: now() }); return ok({ account: me.account, linked: true, created: false }); }
    me = seed(); me.creds = [{ kind, id: id('cred'), label, added: now(), last_used: now() }]; me.projects = []; me.tokens = []; me.sessions = me.sessions.slice(0, 1);
    return ok({ account: me.account, linked: false, created: true });
  };
  const summary = () => (me ? { id: me.account.id, created: me.account.created } : null);
  const needMe = (): Reply | null => (me ? null : fail(401, 'signed_out', 'signed out'));

  const route = (method: string, path: string, body: Record<string, unknown>): Reply => {
    const seg = path.split('/');
    if (method === 'GET' && path === 'auth/methods') return ok({ email: !flags.has('noemail'), zafu: true, passkey: true, ssh: { command: 'ssh modgrad.com login' }, account: summary() });
    if (method === 'POST' && path === 'auth/logout') { me = null; return ok(null, 204); }
    if (method === 'POST' && path === 'auth/email') return String(body.email).includes('limit') ? fail(429, 'rate_limited', 'slow down', '600') : ok(null, 202);
    if (method === 'POST' && (path === 'auth/email/consume' || path === 'auth/ssh/consume')) {
      if (String(body.token).startsWith('expired')) return fail(410, 'token_invalid', 'token expired or already used');
      return signIn(path.startsWith('auth/email') ? 'email' : 'ssh', path.startsWith('auth/email') ? 'linked@example.org' : 'SHA256:mockKeyFromLink');
    }
    if (method === 'POST' && path === 'auth/zafu/challenge') {
      const c = `modgrad-login-v1\n${location.origin}\n${bufToB64url(rnd(16))}\n${Math.floor(now() + 300)}`;
      return ok({ challenge_hex: hex(new TextEncoder().encode(c)), statement: 'Sign in to modgrad', expires: now() + 300 });
    }
    if (method === 'POST' && path === 'auth/zafu/verify') return signIn('zafu', String(body.public_key).slice(0, 16) + '…');
    if (method === 'POST' && (path === 'auth/passkey/register/options' || path === 'auth/passkey/login/options')) {
      const challenge = bufToB64url(rnd(32));
      return ok(path.includes('register')
        ? { challenge, rp: { name: 'modgrad', id: location.hostname }, user: { id: bufToB64url(rnd(16)), name: me?.account.id || 'new', displayName: 'modgrad' }, pubKeyCredParams: [-8, -7, -257].map((alg) => ({ type: 'public-key', alg })), authenticatorSelection: { residentKey: 'required', userVerification: 'preferred' }, attestation: 'none', excludeCredentials: [] }
        : { challenge, rpId: location.hostname, allowCredentials: [], userVerification: 'preferred' });
    }
    if (method === 'POST' && path === 'auth/passkey/register/verify') return signIn('passkey', String(body.label || 'passkey'));
    if (method === 'POST' && path === 'auth/passkey/login/verify') return signIn('passkey', 'passkey');

    if (seg[0] !== 'me') return fail(404, 'not_found', 'no such endpoint');
    const no = needMe(); if (no) return no;
    const m = me!;
    if (method === 'GET' && path === 'me') return ok(structuredClone(m));
    if (method === 'POST' && path === 'me/projects') {
      const name = String(body.name);
      if (name === 'taken-name' || m.projects.some((p) => p.name === name)) return fail(409, 'name_taken', 'name taken');
      if (!/^[a-z0-9][a-z0-9-]{2,39}$/.test(name)) return fail(400, 'bad_name', 'bad name');
      const p: Project = { name, visibility: (body.visibility as Project['visibility']) || 'private', runs: 0, last_unix: null, anon: false };
      m.projects.push(p); return ok(p, 201);
    }
    if (method === 'PATCH' && seg[1] === 'projects') { const p = m.projects.find((x) => x.name === decodeURIComponent(seg[2])); if (p) p.visibility = body.visibility as Project['visibility']; return ok({}); }
    if (method === 'POST' && path === 'me/claim') {
      const name = String(body.project);
      if (!m.projects.some((p) => p.name === name)) m.projects.push({ name, visibility: 'private', runs: 2, last_unix: now() - 4000, anon: false });
      else m.projects = m.projects.map((p) => (p.name === name ? { ...p, anon: false, expires: null } : p));
      return ok({});
    }
    if (method === 'POST' && path === 'me/tokens') {
      const tk: Token = { id: id('tok'), project: String(body.project), label: String(body.label || ''), created: now(), expires: body.expires_days ? now() + 86400 * Number(body.expires_days) : null, last_used: null };
      m.tokens.push(tk); return ok({ id: tk.id, token: 'obsp_' + bufToB64url(rnd(24)) }, 201);
    }
    if (method === 'DELETE' && seg[1] === 'tokens') { m.tokens = m.tokens.filter((x) => x.id !== decodeURIComponent(seg[2])); return ok(null, 204); }
    if (method === 'POST' && path === 'me/creds/ssh') {
      const k = String(body.public_key);
      if (k.includes('taken')) return fail(409, 'cred_taken', 'key belongs to another account');
      m.creds.push({ kind: 'ssh', id: id('cred'), label: 'SHA256:' + bufToB64url(rnd(32)) + '  ' + (k.split(/\s+/)[2] || ''), added: now(), last_used: null }); return ok({}, 201);
    }
    if (method === 'DELETE' && seg[1] === 'creds') {
      if (m.creds.length <= 1) return fail(409, 'last_cred', 'last credential');
      m.creds = m.creds.filter((x) => x.id !== decodeURIComponent(seg[2])); return ok(null, 204);
    }
    if (method === 'DELETE' && seg[1] === 'sessions') {
      const sid = decodeURIComponent(seg[2]); const s = m.sessions.find((x: Session) => x.id === sid);
      m.sessions = m.sessions.filter((x) => x.id !== sid); if (s?.current) me = null; return ok(null, 204);
    }
    if (method === 'DELETE' && path === 'me') { if (body.confirm !== m.account.id) return fail(400, 'confirm', 'confirm with the account id'); me = null; return ok(null, 204); }
    return fail(404, 'not_found', 'no such endpoint');
  };

  const transport: Transport = async (method, path, body) => {
    await new Promise((r) => setTimeout(r, 250));
    return route(method, path, (body || {}) as Record<string, unknown>);
  };

  // a credential shaped like the browser's PublicKeyCredential (bytes as ArrayBuffers)
  const fakeCred = (assert: boolean, opts: CredentialCreationOptions | CredentialRequestOptions) => {
    const pk = (opts as { publicKey: { challenge: ArrayBuffer } }).publicKey;
    const clientDataJSON = new TextEncoder().encode(JSON.stringify({ type: assert ? 'webauthn.get' : 'webauthn.create', challenge: bufToB64url(pk.challenge), origin: location.origin })).buffer;
    const raw = rnd(16).buffer;
    const response = assert
      ? { clientDataJSON, authenticatorData: rnd(37).buffer, signature: rnd(64).buffer, userHandle: rnd(16).buffer }
      : { clientDataJSON, attestationObject: rnd(120).buffer };
    return { id: bufToB64url(raw), rawId: raw, type: 'public-key', response } as unknown as Credential;
  };
  const webauthn: WebAuthn = {
    available: () => true,
    create: async (o) => { await new Promise((r) => setTimeout(r, 400)); return fakeCred(false, o); },
    get: async (o) => { await new Promise((r) => setTimeout(r, 400)); return fakeCred(true, o); },
  };
  const zafu: ZafuDriver = {
    detect: async () => (flags.has('nozafu') ? null : { origin: 'chrome-extension://mockzafu' }),
    sign: async (_h, challengeHex) => {
      await new Promise((r) => setTimeout(r, 400));
      if (!challengeHex.startsWith('6d')) throw new Error('mock zafu: challenge must start with "m"');
      return { signature: hex(rnd(64)), publicKey: hex(rnd(32)) };
    },
  };
  return { transport, webauthn, zafu };
}
