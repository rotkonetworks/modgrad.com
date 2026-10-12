// "Sign in with zafu": the minimum of @zafu/zid needed to detect the wallet and have it sign one
// challenge, vendored so the dashboard carries no wallet SDK dependency. Faithful to (zafu repo):
//   packages/zid/src/provider.ts:51   detectZafu: providers live on window[Symbol.for('penumbra')],
//                                     one per wallet (Prax may sit beside zafu); ping each, take the
//                                     one answering { zafu: true }
//   packages/zid/src/transport.ts:23  extension id = provider key minus "chrome-extension://" and "/"
//   packages/zid/src/transport.ts:54  chrome.runtime.sendMessage(extId, req, cb); runtime.lastError = failure
//   packages/zid/src/messaging.ts:108 detect / :138 requireWallet: protocolVersions ?? [protocolVersion]
//                                     must include ZAFU_PROTOCOL_VERSION (1, zafu-protocol/src/version.ts)
//   packages/zid/src/messaging.ts:181 sign: { type: 'zafu_sign', challengeHex, statement }
//   packages/zafu-protocol/src/methods.ts:56,90,97  ping / zafu_sign wire types
// How the page reaches the extension: its MAIN-world content script registers the provider under
// that symbol keyed by `chrome-extension://<id>` at document_start
// (apps/extension/src/content-scripts/injected-penumbra-global.ts:196), and the manifest's
// externally_connectable (apps/extension/public/manifest.json:89, https and localhost) lets the page
// call chrome.runtime.sendMessage on that id directly. No provider.connect() first: the sign handler
// has no connect gate, the per-request approval popup is the consent
// (apps/extension/src/message/listen/sign-request.ts:14).

/** a detected wallet: its extension origin (the provider record key) */
export interface ZafuHandle { origin: string }

const PROTOCOL = 1;
const PING_TIMEOUT_MS = 3000;

interface Runtime { sendMessage: (id: string, msg: unknown, cb: (r: unknown) => void) => void; lastError?: { message?: string } }
const runtime = (): Runtime | null => {
  const rt = (globalThis as { chrome?: { runtime?: Partial<Runtime> } }).chrome?.runtime;
  return rt && typeof rt.sendMessage === 'function' ? rt as Runtime : null;
};
const extId = (origin: string): string => origin.replace('chrome-extension://', '').replace(/\/$/, '');

function send(h: ZafuHandle, msg: unknown, timeoutMs = 0): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const rt = runtime();
    if (!rt) { reject(new Error('no zafu wallet reachable')); return; }
    let done = false;
    const t = timeoutMs ? setTimeout(() => { if (!done) { done = true; reject(new Error('zafu did not answer')); } }, timeoutMs) : undefined;
    try {
      rt.sendMessage(extId(h.origin), msg, (r) => {
        if (done) return;
        done = true; clearTimeout(t);
        if (rt.lastError) reject(new Error(rt.lastError.message || 'transport error')); else resolve(r);
      });
    } catch (e) { done = true; clearTimeout(t); reject(e instanceof Error ? e : new Error(String(e))); }
  });
}

/** the zafu wallet speaking protocol 1, or null (not installed, not Chromium, disabled, too old) */
export async function detectZafu(): Promise<ZafuHandle | null> {
  const providers = (globalThis as Record<symbol, unknown>)[Symbol.for('penumbra')];
  if (!providers || typeof providers !== 'object' || !runtime()) return null;
  for (const [origin, provider] of Object.entries(providers as Record<string, unknown>)) {
    if (!provider) continue;
    const h = { origin };
    try {
      const pong = await send(h, { type: 'ping' }, PING_TIMEOUT_MS) as { zafu?: unknown; protocolVersion?: number; protocolVersions?: number[] } | null;
      if (!pong || pong.zafu !== true) continue;
      const versions = pong.protocolVersions ?? [pong.protocolVersion];
      if (versions.includes(PROTOCOL)) return h;
    } catch { /* not zafu, or unreachable: try the next provider */ }
  }
  return null;
}

const WALLET_ERRORS: Record<string, string> = {
  denied: 'The request was declined in zafu.',
  locked: 'zafu is locked. Unlock it, then try again.',
  rate_limited: 'zafu is refusing requests from this site for a moment. Try again shortly.',
  not_available: 'Identity signing is turned off in zafu settings.',
  invalid_request: 'zafu rejected the challenge as malformed.',
};
/** zafu signs the challenge bytes (hex) with this site's ZID ed25519 key; waits while the user approves */
export async function zafuSign(h: ZafuHandle, challengeHex: string, statement: string): Promise<{ signature: string; publicKey: string }> {
  let r: { success?: boolean; signature?: string; publicKey?: string; error?: string; code?: string } | null;
  try { r = await send(h, { type: 'zafu_sign', challengeHex, statement }) as typeof r; }
  catch (e) { throw new Error('Could not reach zafu: ' + (e as Error).message); }
  if (r && r.success && r.signature && r.publicKey) return { signature: r.signature, publicKey: r.publicKey };
  const code = r?.code || '';
  throw new Error(WALLET_ERRORS[code] || 'zafu did not sign: ' + (r?.error || 'no signature returned'));
}
