# modgrad portal — shared contract (backend ↔ passkey ↔ web)

Goal: modgrad becomes a service portal. Sign in with **SSH key**, **email magic link**,
**passkey (WebAuthn)**, or **zafu wallet**. All four are credentials on one **Account**;
an account can hold several of each (recovery = another credential). Signing in with a
new credential while already signed in LINKS it to the current account.

## Model (server, `crates/modgrad-obs-server/src/auth.rs`, store `<root>/.obs-auth.json`)
- `Account { id: "acct_<22 base32>", created, creds: Vec<Credential>, projects }`
- `Credential` kinds (open enum, serde tag `kind`):
  - `ssh`  { fingerprint: "SHA256:…" — EXACTLY the format `ssh.rs::fingerprint` yields, public_key (openssh line), label }
  - `email` { address (lowercased, trimmed) }
  - `zafu` { public_key (hex, 32 bytes; the site-scoped ZID key) }
  - `passkey` { cred_id (b64url), public_key (COSE bytes b64url), alg (-7 | -8 | -257), sign_count, label }
  - each has `added`, `last_used` (unix f64)
- `Session { hash(sha256 of cookie value), account, created, last_seen, expires, ua }` — 30 d sliding, last_seen persisted at most every 10 min.
- Projects: flat namespace. Owned project = `{ name, owner: acct, visibility: "private"|"public", created }`.
  - name `^[a-z0-9][a-z0-9-]{2,39}$`, `anon-` prefix reserved, plus reserved words (api, login, account, admin, lab, docs, new, me, static, assets).
  - **Owned projects default private**: reads (index, run data, SSE, samples, boards, /mcp read tools) need the owner's session cookie OR a project token (bearer) for that project. Anon projects stay unlisted-by-random-id as today. Public = today's behaviour.
- Schema: bump the file version; a v1 file (today's) MUST still open and migrate (ssh fingerprint → anon project becomes an Account with that ssh cred owning that project). Test it.
- Every secret at rest is a SHA-256 hash (sessions, email tokens, ssh login tokens, challenges, api tokens). Atomic write (tmp + rename), 0600.

## Cookie / CSRF
- `mg_session=<32 random bytes b64url>; HttpOnly; SameSite=Lax; Path=/; Max-Age=2592000` + `Secure` when `--public-url` is https.
- Cookie-authed state-changing requests (POST/PUT/DELETE) must pass: `Sec-Fetch-Site` ∈ {same-origin, none} if present, else `Origin` == public origin (or the request Host when no public url). Bearer-token requests unaffected.
- The dashboard is served same-origin by modgrad-server; `fetch(..., {credentials: 'same-origin'})` (default) is enough.

## Endpoints (JSON; errors `{error, code}`)
### auth
- `GET  /api/v1/auth/methods` → `{ email: bool, zafu: true, passkey: true, ssh: { command: "ssh modgrad.com login" } | null, account: <me summary> | null }`
- `POST /api/v1/auth/logout` → 204, clears cookie.
- **email**: `POST /api/v1/auth/email { email }` → 202 ALWAYS (no enumeration). Rate limits: 3 / address / 15 min, 10 / IP / hour. Token 32 bytes, TTL 15 min, single use. Mail body link: `<public>/login#email=<token>` (fragment: never logged, never consumed by mail scanners — only a POST consumes).
  `POST /api/v1/auth/email/consume { token }` → sets cookie, `{ account, linked: bool, created: bool }`.
- **zafu**: `POST /api/v1/auth/zafu/challenge` → `{ challenge_hex, statement, expires }`. Challenge bytes =
  `"modgrad-login-v1\n" + origin + "\n" + nonce_b64url + "\n" + expires_unix` (UTF-8, hex-encoded). Starts with 'm' so zafu's reserved-prefix refusal never triggers; ≤ 1024 bytes. Stored hashed, TTL 5 min, single use.
  `POST /api/v1/auth/zafu/verify { challenge_hex, signature, public_key }` (hex) → ed25519 verify over the EXACT challenge bytes (use `ring::signature::ED25519`) → cookie. Note: zafu keys are per-origin — dogfood host ≠ modgrad.com account.
- **passkey** (WebAuthn, attestation "none" only, rpId = public host):
  `POST /api/v1/auth/passkey/register/options` → PublicKeyCredentialCreationOptions JSON (b64url fields), `residentKey: "required"`, `userVerification: "preferred"`, alg [-8, -7, -257]. Signed-in → adds to account; signed-out → creates a new account.
  `POST /api/v1/auth/passkey/register/verify { id, rawId, response: { clientDataJSON, attestationObject }, label? }`
  `POST /api/v1/auth/passkey/login/options` → request options (empty allowCredentials: discoverable).
  `POST /api/v1/auth/passkey/login/verify { id, rawId, response: { clientDataJSON, authenticatorData, signature, userHandle } }` → cookie.
- **ssh**: `ssh <host> login` (new verb in ssh.rs) prints `<public>/login#ssh=<token>` (TTL 10 min, single use, bound to the key's fingerprint + public key line).
  `POST /api/v1/auth/ssh/consume { token }` → cookie; if already signed in → links the key. The old fingerprint→anon project mapping is kept: that project belongs to the key's account.
- Linking rule (all kinds): signed-in + credential unknown → link to current account. Credential owned by ANOTHER account → 409 `cred_taken` (no silent merge). Signed-out + unknown → create account.

### me (session cookie required; 401 `{code:"signed_out"}` otherwise)
- `GET    /api/v1/me` → `{ account: {id, created}, creds: [{kind, id, label, added, last_used}], projects: [{name, visibility, runs, last_unix, anon, expires?}], sessions: [{id, created, last_seen, ua, current}], tokens: [{id, project, label, created, expires, last_used}] }`
- `POST   /api/v1/me/projects { name, visibility? }` → 201 | 409 `name_taken` | 400 `bad_name`
- `PATCH  /api/v1/me/projects/:name { visibility }`
- `POST   /api/v1/me/claim { project, token }` → an `anon-*` project whose project token you hold becomes yours (no expiry, keeps its name).
- `POST   /api/v1/me/tokens { project, label, expires_days? }` → `{ id, token }` shown ONCE (`obsp_…`)
- `DELETE /api/v1/me/tokens/:id`
- `POST   /api/v1/me/creds/ssh { public_key: "ssh-ed25519 AAAA… comment" }` → links (fingerprint must match `ssh.rs::fingerprint`)
- `DELETE /api/v1/me/creds/:id` → 409 `last_cred` if it is the only one
- `DELETE /api/v1/me/sessions/:id`
- `DELETE /api/v1/me { confirm: <account id> }` → deletes account + sessions + tokens; owned projects' data is removed.

## Mail (`Mailer` trait)
- `--mail log` (default; prints the link at info level — dev/tests), `--mail cmd:<path>` (pipe RFC 5322 to a sendmail-like command), `--mail jmap` (Stalwart, Basic auth, env `MODGRAD_JMAP_URL`, `MODGRAD_JMAP_USER`, `MODGRAD_JMAP_PASS`; From = JMAP user). Use `ureq` (already in Cargo.lock) behind a feature. NEVER send a real email in tests or by hand.
- Subject `Sign in to modgrad`; plain text; the link; "expires in 15 minutes"; "if you didn't ask, ignore this".

## Web (obs/web)
- Route `/login` (or `#login` — follow how the app routes today): four labelled buttons/forms: Email (input + "Send link"), Passkey ("Sign in with passkey"), zafu ("Sign in with zafu" — hidden-ish/“install zafu” link when not detected), SSH (shows `ssh modgrad.com login`, copy button). Handles `#email=` / `#ssh=` fragments on load → POST consume → clears the fragment.
- Header: "sign in" link, or account menu when signed in.
- `/account`: credentials (add another of each kind; remove), projects (create, visibility toggle, claim the anon project whose token this browser holds in sessionStorage `obs-anon-token:<p>`), API tokens (mint → shown once with copy + `MODGRAD_TOKEN=… modgrad run …` snippet; revoke), sessions (revoke), delete account (type the account id).
- No single-letter hotkeys (Vimium-C); every control a real labelled button. Washi/sumi theme, single shu accent, match existing components.

## Clarifications (agreed with web, implemented)
1. Consume endpoints (`email/consume`, `ssh/consume`): unknown, used or expired token → **410** `{code: "token_expired"}` (one answer for all).
2. Passkey options endpoints return the **bare** PublicKeyCredentialCreationOptions / RequestOptions object (no `{publicKey}` wrapper); challenge, user.id, credential ids are b64url strings.
3. `auth/methods.account` and the login responses' `account` are objects `{id, created}` (methods also carries `label`; null when signed out).
4. Nullable times (`last_used`, `last_unix`, `expires`) are JSON `null` when absent, never 0.
5. DELETE routes accept a JSON body (`DELETE /api/v1/me {confirm}`).
6. `/login` and `/account` (exactly, no trailing slash) serve the same page as `/`.
7. `GET /api/v1/me` lists anon projects owned via an SSH key (`anon: true`).

## Deviations (backend)
- **Visibility of anon projects.** Owned projects default private only when made with `POST /me/projects` or `POST /me/claim`. An SSH key's own `anon-*` project (made by the door, or by the v1 migration) is owned by the key's account but stays **public** (unlisted by random id, as today), so existing links and `ssh … runs` URLs keep working; the owner can PATCH it private.
- **Auto accounts.** Every new SSH key (first connect) and every migrated key gets an account flagged `auto` (never signed in on the web). Exception to 409 `cred_taken`: proving such a key (`ssh login` link) while signed in to another account folds the auto account (its one key and its projects) into the signed-in one. Whoever proves the key already controls everything in it. Once an account has had a web session it is never folded.
- **Web-added SSH keys are unconfirmed.** `POST /me/creds/ssh` stores the key with `verified: false` (anyone can paste a public key). The door serves an unconfirmed key only `login` and `help`; `ssh … login` from that key, consumed by the account that added it, confirms it. Consumed signed out or by anyone else, the paste is dropped and the holder's account wins. Without this a pasted public key would route its owner's training logs into the paster's private project. `/me` creds of kind ssh carry `fingerprint` and `verified`.
- **SSH `public_key` is optional** in storage: keys migrated from v1 only have a fingerprint; the line is filled on the key's next connect.
- **Model storage.** Ownership lives in a top-level `owned: {project: {owner, visibility, created}}` map; `Account` has no `projects` field (derived). `Account` gains `auto`. File `version: 2`; the v1 original is kept as `.obs-auth.json.v1`; a newer version is refused.
- **Challenges** (zafu, passkey) are held hashed in memory only (bounded to 10k, 5 min TTL), so a restart invalidates outstanding ones. Email and ssh-login tokens are on disk (hashed), pruned when expired.
- **Private reads** are refused with **401** `code: "signed_out"` (no session, no token) or **403** `code: "forbidden"`, in the v1 error shape (`{error, example, help, message, code}`). A nonexistent run inside a private project gets the same answer (no probing names). The unfiltered index silently drops private runs the viewer can't read; `?project=<private>` is 401/403. Raw store files (`/runs/<p>/…`, `/boards/<p>/…` when the obs root is the static dir) are 404.
- **Hosted `/mcp`**: private reads need the bearer token; the session cookie is not forwarded to its loopback store.
- **CSRF** applies to every state-changing `/api/v1/auth/*` and `/api/v1/me/*` request, signed in or not (login CSRF): `Sec-Fetch-Site` ∈ {same-origin, none} if present, else `Origin` must match, else (no browser headers) only allowed **without** the cookie, so curl works signed out.
- **Email**: malformed address → 400 `bad_email` (no account enumeration either way); well-formed → 202 even when rate-limited. Rate-limit IP = TCP peer, or `X-Real-IP` / last `X-Forwarded-For` hop when the peer is loopback (local reverse proxy).
- **`/me` shapes**: projects add `created` (and `expires: null`); `POST /me/tokens` replies `{id, token, project, note, use}` (`use` = the `MODGRAD_TOKEN=… modgrad run` line); `POST /me/creds/ssh` replies 201 `{id, fingerprint, verified: false, confirm}`.
- **Claim** accepts the project token or the anon project's `obsc_…` claim token; a project already owned (e.g. an SSH key's) is 409 `already_owned`. Claiming clears the expiry of that project's tokens.
- **Passkey register signed out**: `user.id` is a fresh random handle, not the id of the account created at verify (lookup is by credential id). Verification goes through `webauthn.rs`; the stub in this branch returns 501 `passkey_unsupported` after the challenge checks.
- **Error codes** used: `signed_out, csrf, bad_request, bad_email, bad_name, bad_key, bad_confirm, bad_token, bad_signature, bad_passkey, unknown_passkey, passkey_unsupported, token_expired, cred_taken, last_cred, name_taken, already_owned, not_found, forbidden, busy, store`.
- **Account deletion** also removes `boards/<project>/`, the projects' run tokens and the SSH routes of its keys.
