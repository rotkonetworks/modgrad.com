//! The account portal over HTTP: `/api/v1/auth/*` (sign in with an email
//! link, a zafu signature, a passkey, or an `ssh <server> login` link) and
//! `/api/v1/me/*` (credentials, projects, tokens, sessions). The contract
//! is `docs/design/portal-spec.md`; the account rules are in [`crate::account`].
//!
//! Sessions are a cookie (`mg_session`, HttpOnly, SameSite=Lax, Secure
//! under an https `--public-url`). Every state-changing request here passes
//! [`csrf`]: a browser's `Sec-Fetch-Site` must be `same-origin`/`none`,
//! else its `Origin` must be this site's; a request with neither header
//! (curl) may not carry the cookie. Errors are `{error, code}`.
//!
//! The origin is `--public-url`'s, else the request's own (Host /
//! X-Forwarded-*), which is also how a dev server works on localhost. The
//! WebAuthn rpId is the origin's host.
//!
//! [`access`] and [`viewer`] are the read side: what the rest of the API
//! asks before serving a private project.

use crate::account::{Access, AccountError, SignIn, Viewer, normalize_email};
use crate::auth::{Challenge, Cred, CredKind, Visibility};
use crate::http::AppState;
use crate::mail::{LogMailer, Mailer, sign_in_mail};
use axum::Router;
use axum::body::Bytes;
use axum::extract::{ConnectInfo, Path, Request, State as AxState};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, patch, post};
use base64::Engine;
use serde_json::{Value, json};
use std::collections::{HashMap, VecDeque};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

/// The session cookie's name.
pub const COOKIE: &str = "mg_session";
const B64: base64::engine::GeneralPurpose = base64::engine::general_purpose::URL_SAFE_NO_PAD;

/// Portal parts of [`AppState`]: the mailer, the email rate limits, and
/// the SSH sign-in command to show.
#[derive(Clone)]
pub struct Portal {
    pub mailer: Arc<dyn Mailer>,
    limits: Arc<Mutex<Limits>>,
    /// E.g. `ssh -p 2222 modgrad.com login`; None = no SSH door.
    pub ssh_login: Option<String>,
}

impl Default for Portal {
    fn default() -> Self {
        Portal { mailer: Arc::new(LogMailer), limits: Arc::default(), ssh_login: None }
    }
}

/// Sliding-window counters for sign-in emails.
#[derive(Default)]
struct Limits {
    by_address: HashMap<String, VecDeque<f64>>,
    by_ip: HashMap<String, VecDeque<f64>>,
}

impl Limits {
    /// Count one email to `address` from `ip`, if both are under their limits.
    fn allow(&mut self, address: &str, ip: &str, per_address: (usize, f64), per_ip: (usize, f64), now: f64) -> bool {
        fn trim(q: &mut VecDeque<f64>, window: f64, now: f64) {
            while q.front().is_some_and(|t| now - t >= window) {
                q.pop_front();
            }
        }
        if self.by_address.len() + self.by_ip.len() > 100_000 {
            self.by_address.retain(|_, q| {
                trim(q, per_address.1, now);
                !q.is_empty()
            });
            self.by_ip.retain(|_, q| {
                trim(q, per_ip.1, now);
                !q.is_empty()
            });
        }
        let a = self.by_address.entry(address.into()).or_default();
        trim(a, per_address.1, now);
        let full_a = a.len() >= per_address.0;
        let i = self.by_ip.entry(ip.into()).or_default();
        trim(i, per_ip.1, now);
        if full_a || i.len() >= per_ip.0 {
            return false;
        }
        i.push_back(now);
        self.by_address.get_mut(address).unwrap().push_back(now);
        true
    }
}

/// A portal error: `{error, code}` with a status.
#[derive(Debug)]
pub struct PortalError {
    pub status: StatusCode,
    pub code: &'static str,
    pub error: String,
}

fn perr(status: StatusCode, code: &'static str, error: impl Into<String>) -> PortalError {
    PortalError { status, code, error: error.into() }
}

impl From<AccountError> for PortalError {
    fn from(e: AccountError) -> Self {
        if let AccountError::Store(m) = &e {
            tracing::error!("auth store: {m}");
        }
        PortalError { status: StatusCode::from_u16(e.status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR), code: e.code(), error: e.to_string() }
    }
}

impl IntoResponse for PortalError {
    fn into_response(self) -> Response {
        (self.status, axum::Json(json!({ "error": self.error, "code": self.code }))).into_response()
    }
}

type PResult = Result<Response, PortalError>;

fn signed_out() -> PortalError {
    perr(StatusCode::UNAUTHORIZED, "signed_out", "sign in first")
}

fn bad(m: impl Into<String>) -> PortalError {
    perr(StatusCode::BAD_REQUEST, "bad_request", m)
}

// ------------------------------------------------------------ origin, cookie

/// `scheme://host[:port]` of a URL.
fn origin_of(url: &str) -> String {
    match url.split_once("://") {
        Some((scheme, rest)) => format!("{scheme}://{}", rest.split('/').next().unwrap_or(rest)),
        None => url.trim_end_matches('/').to_string(),
    }
}

/// This site's origin: `--public-url`'s, else the request's.
pub fn origin(st: &AppState, h: &HeaderMap) -> String {
    origin_of(&st.base_url(h))
}

/// The WebAuthn relying-party id: the origin's host, without port.
pub fn rp_id(origin: &str) -> String {
    let host = origin.split_once("://").map_or(origin, |(_, r)| r);
    match host.strip_prefix('[') {
        Some(v6) => v6.split(']').next().unwrap_or(v6).to_string(),
        None => host.rsplit_once(':').map_or(host, |(h, _)| h).to_string(),
    }
}

/// The `mg_session` cookie of a request, if well-formed.
pub fn cookie(h: &HeaderMap) -> Option<String> {
    h.get_all(header::COOKIE).iter().filter_map(|v| v.to_str().ok()).flat_map(|v| v.split(';')).find_map(|kv| {
        let v = kv.trim().strip_prefix(COOKIE)?.strip_prefix('=')?;
        (v.len() == 43 && v.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')).then(|| v.to_string())
    })
}

fn secure(st: &AppState) -> bool {
    st.public_url.as_deref().is_some_and(|u| u.starts_with("https://"))
}

/// The path of a URL (`/app` of `https://modgrad.com/app/`), `/` for none.
fn path_of(url: &str) -> String {
    let rest = url.split_once("://").map_or(url, |(_, r)| r);
    let path = rest.find('/').map_or("", |i| rest[i..].trim_end_matches('/'));
    if path.is_empty() { "/".into() } else { path.to_string() }
}

/// The cookie is scoped to `--public-url`'s path, so a server mounted under
/// a prefix (`https://modgrad.com/app`) does not hand its session to the
/// rest of the site.
fn set_cookie(st: &AppState, value: &str, max_age: u64) -> HeaderValue {
    let path = st.public_url.as_deref().map_or_else(|| "/".to_string(), path_of);
    let s = format!("{COOKIE}={value}; HttpOnly; SameSite=Lax; Path={path}; Max-Age={max_age}{}", if secure(st) { "; Secure" } else { "" });
    HeaderValue::from_str(&s).expect("cookie is ascii")
}

fn session_cookie(st: &AppState, value: &str) -> HeaderValue {
    set_cookie(st, value, (st.auth.portal.session_days * 86400.0) as u64)
}

/// The signed-in account and its cookie.
fn session(st: &AppState, h: &HeaderMap) -> Option<(String, String)> {
    let c = cookie(h)?;
    st.auth.session(&c).map(|a| (a, c))
}

fn need_session(st: &AppState, h: &HeaderMap) -> Result<(String, String), PortalError> {
    session(st, h).ok_or_else(signed_out)
}

/// Who is asking: session account and bearer credential (an unknown or
/// expired token counts as none, as reads always treated it).
pub fn viewer(st: &AppState, h: &HeaderMap) -> Viewer {
    Viewer { account: session(st, h).map(|(a, _)| a), cred: st.auth.cred(crate::v1::bearer(h).as_deref()).unwrap_or(Cred::None) }
}

/// May this request read `project`? Looks at the session and token only
/// when the project is private.
pub fn access(st: &AppState, h: &HeaderMap, project: &str) -> Access {
    match st.auth.owned(project) {
        Some(o) if o.visibility == Visibility::Private => st.auth.readable(project, &viewer(st, h)),
        _ => Access::Open,
    }
}

/// The CSRF rule for state-changing requests (see the module docs).
pub fn csrf(st: &AppState, h: &HeaderMap) -> Result<(), PortalError> {
    let refuse = |why: &str| Err(perr(StatusCode::FORBIDDEN, "csrf", format!("cross-site request refused: {why}")));
    if let Some(s) = h.get("sec-fetch-site") {
        return match s.to_str().unwrap_or("") {
            "same-origin" | "none" => Ok(()),
            other => refuse(&format!("Sec-Fetch-Site is {other:?}")),
        };
    }
    if let Some(o) = h.get(header::ORIGIN) {
        let want = origin(st, h);
        return if o.to_str().ok() == Some(want.as_str()) { Ok(()) } else { refuse(&format!("Origin is not {want}")) };
    }
    if cookie(h).is_some() {
        return refuse("a request with the session cookie needs an Origin header");
    }
    Ok(())
}

async fn csrf_layer(AxState(st): AxState<AppState>, req: Request, next: Next) -> Response {
    if !matches!(*req.method(), Method::GET | Method::HEAD | Method::OPTIONS)
        && let Err(e) = csrf(&st, req.headers())
    {
        return e.into_response();
    }
    next.run(req).await
}

/// Is `ip` where a reverse proxy in front of this server would connect
/// from: loopback, or a private network (a container network, e.g. nginx
/// and this server on one podman network)?
fn local_proxy(ip: std::net::IpAddr) -> bool {
    match ip {
        std::net::IpAddr::V4(v) => v.is_loopback() || v.is_private() || v.is_link_local(),
        std::net::IpAddr::V6(v) => v.is_loopback() || v.is_unique_local() || v.is_unicast_link_local() || v.to_ipv4_mapped().is_some_and(|v| v.is_loopback() || v.is_private()),
    }
}

/// The client address for rate limits: the peer, unless it is a local
/// reverse proxy (see [`local_proxy`]), in which case the address it forwarded.
fn client_ip(peer: Option<SocketAddr>, h: &HeaderMap) -> String {
    let forwarded = || {
        h.get("x-real-ip").and_then(|v| v.to_str().ok()).map(|s| s.trim().to_string()).or_else(|| h.get("x-forwarded-for").and_then(|v| v.to_str().ok()).and_then(|v| v.rsplit(',').next()).map(|s| s.trim().to_string()))
    };
    match peer {
        Some(p) if !local_proxy(p.ip()) => p.ip().to_string(),
        Some(p) => forwarded().unwrap_or_else(|| p.ip().to_string()),
        None => forwarded().unwrap_or_else(|| "unknown".into()),
    }
}

// ------------------------------------------------------------ bodies

fn body(b: &Bytes) -> Result<Value, PortalError> {
    let v = crate::v1::parse_body(b).map_err(|_| bad("the body must be a JSON object"))?;
    if v.is_null() { Ok(json!({})) } else if v.is_object() { Ok(v) } else { Err(bad("the body must be a JSON object")) }
}

fn field<'a>(v: &'a Value, k: &str) -> Result<&'a str, PortalError> {
    v.get(k).and_then(Value::as_str).ok_or_else(|| bad(format!("{k} (a string) is required")))
}

fn unhex(s: &str, what: &str) -> Result<Vec<u8>, PortalError> {
    let s = s.trim();
    if !s.len().is_multiple_of(2) || !s.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(bad(format!("{what} must be hex")));
    }
    Ok((0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect())
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn unb64(s: &str, what: &str) -> Result<Vec<u8>, PortalError> {
    B64.decode(s.trim_end_matches('=')).or_else(|_| base64::engine::general_purpose::STANDARD.decode(s)).map_err(|_| bad(format!("{what} must be base64url")))
}

fn ua(h: &HeaderMap) -> String {
    h.get(header::USER_AGENT).and_then(|v| v.to_str().ok()).unwrap_or("").to_string()
}

fn json_status(status: StatusCode, v: Value) -> Response {
    (status, axum::Json(v)).into_response()
}

/// After a proven credential: link or sign in, and start a session when
/// there was none. `{account: {id, created}, linked, created}`.
fn finish(st: &AppState, h: &HeaderMap, current: Option<(String, String)>, kind: CredKind) -> PResult {
    let SignIn { account, linked, created } = st.auth.sign_in(current.as_ref().map(|(a, _)| a.as_str()), kind)?;
    let created_at = st.auth.account(&account).map(|a| a.created);
    let mut r = json_status(StatusCode::OK, json!({ "account": { "id": account, "created": created_at }, "linked": linked, "created": created }));
    if current.is_none() {
        let c = st.auth.new_session(&account, &ua(h))?;
        r.headers_mut().insert(header::SET_COOKIE, session_cookie(st, &c));
    }
    Ok(r)
}

// ------------------------------------------------------------ auth routes

fn me_summary(st: &AppState, account: &str) -> Value {
    let Some(a) = st.auth.account(account) else { return Value::Null };
    let label = a.creds.iter().find(|c| c.kind.kind() == "email").or(a.creds.first()).map(|c| c.kind.label());
    json!({ "id": a.id, "created": a.created, "label": label })
}

async fn methods(AxState(st): AxState<AppState>, h: HeaderMap) -> Response {
    let me = session(&st, &h);
    let mut r = axum::Json(json!({
        "email": true,
        "zafu": true,
        "passkey": true,
        "ssh": st.portal.ssh_login.as_ref().map(|c| json!({ "command": c })),
        "account": me.as_ref().map(|(a, _)| me_summary(&st, a)),
    }))
    .into_response();
    // the browser's copy of the cookie slides with the server's
    if let Some((_, c)) = me {
        r.headers_mut().insert(header::SET_COOKIE, session_cookie(&st, &c));
    }
    r
}

async fn logout(AxState(st): AxState<AppState>, h: HeaderMap) -> PResult {
    if let Some(c) = cookie(&h) {
        st.auth.end_session(&c)?;
    }
    let mut r = StatusCode::NO_CONTENT.into_response();
    r.headers_mut().insert(header::SET_COOKIE, set_cookie(&st, "", 0));
    Ok(r)
}

/// `GET /api/v1/auth/whoami`: the account behind a request, for services
/// next to this one (the metered LLM proxy) that bill modgrad accounts.
/// A bearer token, when sent, decides: a project token stands for its
/// project's owner (`{account, via: "token", project}`); an unknown or
/// expired token, a run token, or a token of an unowned (anonymous)
/// project is 401. Without one, the session cookie (`{account, via:
/// "session"}`). Nothing is 401 `signed_out`. Reads only; no cookie is set.
async fn whoami(AxState(st): AxState<AppState>, h: HeaderMap) -> PResult {
    let no_store = |v: Value| {
        let mut r = axum::Json(v).into_response();
        r.headers_mut().insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
        r
    };
    let Some(token) = crate::v1::bearer(&h) else {
        let (account, _) = need_session(&st, &h)?;
        return Ok(no_store(json!({ "account": account, "via": "session" })));
    };
    let refuse = |code: &'static str, m: &str| perr(StatusCode::UNAUTHORIZED, code, m);
    match st.auth.cred(Some(&token)) {
        Ok(Cred::Project(p)) => match st.auth.owned(&p) {
            Some(o) if st.auth.account(&o.owner).is_some() => Ok(no_store(json!({ "account": o.owner, "via": "token", "project": p }))),
            _ => Err(refuse("unowned", "this token's project has no account; sign in and claim it (POST /api/v1/me/claim)")),
        },
        Ok(Cred::Run(_)) => Err(refuse("run_token", "a run token only writes its run; send a project token (obsp_…)")),
        Ok(Cred::None) | Err(_) => Err(refuse("bad_token", "unknown or expired token")),
    }
}

/// `POST /api/v1/auth/email {email}`: 202 whether or not a mail went out
/// (never says whether an account exists; rate limits are silent too).
async fn email(AxState(st): AxState<AppState>, peer: Option<ConnectInfo<SocketAddr>>, h: HeaderMap, b: Bytes) -> PResult {
    let v = body(&b)?;
    let address = normalize_email(field(&v, "email")?).ok_or_else(|| perr(StatusCode::BAD_REQUEST, "bad_email", "that does not look like an email address"))?;
    let ip = client_ip(peer.map(|c| c.0), &h);
    let p = st.auth.portal;
    let allowed = st.portal.limits.lock().unwrap().allow(&address, &ip, p.email_per_address, p.email_per_ip, modgrad_obs::now_unix());
    if allowed {
        let token = st.auth.email_token(&address)?;
        let mail = sign_in_mail(&address, &format!("{}/login#email={token}", st.base_url(&h)), (p.email_ttl / 60.0).round() as u64);
        let mailer = st.portal.mailer.clone();
        // off the request: delivery time says nothing, and a slow relay holds nothing up
        tokio::task::spawn_blocking(move || {
            if let Err(e) = mailer.send(&mail) {
                tracing::warn!("sign-in mail to {} failed: {e}", mail.to);
            }
        });
    } else {
        tracing::debug!("sign-in mail to {address} from {ip}: rate limited");
    }
    Ok(json_status(StatusCode::ACCEPTED, json!({ "status": "if that address can receive mail, a sign-in link is on its way" })))
}

/// A sign-in link that is unknown, used or expired: one answer for all.
fn gone(e: AccountError) -> PortalError {
    match e {
        AccountError::BadToken => perr(StatusCode::GONE, "token_expired", "this sign-in link was used or has expired; ask for a new one"),
        e => e.into(),
    }
}

async fn email_consume(AxState(st): AxState<AppState>, h: HeaderMap, b: Bytes) -> PResult {
    let v = body(&b)?;
    let address = st.auth.consume_email(field(&v, "token")?).map_err(gone)?;
    finish(&st, &h, session(&st, &h), CredKind::Email { address })
}

async fn ssh_consume(AxState(st): AxState<AppState>, h: HeaderMap, b: Bytes) -> PResult {
    let v = body(&b)?;
    let (fingerprint, line) = st.auth.consume_ssh_login(field(&v, "token")?).map_err(gone)?;
    finish(&st, &h, session(&st, &h), CredKind::Ssh { fingerprint, public_key: Some(line), label: None, verified: true })
}

/// The bytes a zafu wallet signs.
pub fn zafu_challenge(origin: &str, nonce: &str, expires: u64) -> String {
    format!("modgrad-login-v1\n{origin}\n{nonce}\n{expires}")
}

async fn zafu_challenge_route(AxState(st): AxState<AppState>, h: HeaderMap) -> PResult {
    let origin = origin(&st, &h);
    let nonce = B64.encode(crate::auth::Auth::challenge_bytes(32)?);
    let expires = (modgrad_obs::now_unix() + st.auth.portal.challenge_ttl).floor() as u64;
    let text = zafu_challenge(&origin, &nonce, expires);
    st.auth.put_challenge(text.as_bytes(), Challenge::Zafu)?;
    Ok(axum::Json(json!({ "challenge_hex": hex(text.as_bytes()), "statement": format!("Sign in to modgrad at {origin}"), "expires": expires })).into_response())
}

async fn zafu_verify(AxState(st): AxState<AppState>, h: HeaderMap, b: Bytes) -> PResult {
    let v = body(&b)?;
    let msg = unhex(field(&v, "challenge_hex")?, "challenge_hex")?;
    let sig = unhex(field(&v, "signature")?, "signature")?;
    let pk = unhex(field(&v, "public_key")?, "public_key")?;
    if sig.len() != 64 || pk.len() != 32 {
        return Err(bad("signature is 64 bytes and public_key 32 bytes, hex"));
    }
    let invalid = |why: &str| perr(StatusCode::UNAUTHORIZED, "bad_token", format!("challenge refused: {why}"));
    if st.auth.take_challenge(&msg) != Some(Challenge::Zafu) {
        return Err(invalid("unknown, used or expired; ask for a new one"));
    }
    let text = std::str::from_utf8(&msg).map_err(|_| invalid("not text"))?;
    let parts: Vec<&str> = text.split('\n').collect();
    let [tag, o, _nonce, expires] = parts[..] else { return Err(invalid("malformed")) };
    if tag != "modgrad-login-v1" || o != origin(&st, &h) {
        return Err(invalid("issued for another site"));
    }
    if expires.parse::<f64>().map_or(true, |e| e < modgrad_obs::now_unix()) {
        return Err(invalid("expired"));
    }
    ring::signature::UnparsedPublicKey::new(&ring::signature::ED25519, &pk).verify(&msg, &sig).map_err(|_| perr(StatusCode::UNAUTHORIZED, "bad_signature", "the signature does not verify for this key and challenge"))?;
    finish(&st, &h, session(&st, &h), CredKind::Zafu { public_key: hex(&pk) })
}

// ------------------------------------------------------------ passkeys

fn webauthn_err(e: crate::webauthn::WebauthnError) -> PortalError {
    match e {
        crate::webauthn::WebauthnError::Unsupported => perr(StatusCode::NOT_IMPLEMENTED, "passkey_unsupported", "passkeys are not available on this server yet"),
        crate::webauthn::WebauthnError::Invalid(m) => perr(StatusCode::BAD_REQUEST, "bad_passkey", m),
    }
}

/// The challenge inside a clientDataJSON (verified again by the verifier).
fn client_challenge(cdj: &[u8]) -> Result<Vec<u8>, PortalError> {
    let v: Value = serde_json::from_slice(cdj).map_err(|_| perr(StatusCode::BAD_REQUEST, "bad_passkey", "clientDataJSON is not JSON"))?;
    unb64(v["challenge"].as_str().ok_or_else(|| perr(StatusCode::BAD_REQUEST, "bad_passkey", "clientDataJSON has no challenge"))?, "challenge")
}

fn resp_field(v: &Value, k: &str) -> Result<Vec<u8>, PortalError> {
    unb64(v["response"][k].as_str().ok_or_else(|| bad(format!("response.{k} is required")))?, k)
}

async fn passkey_register_options(AxState(st): AxState<AppState>, h: HeaderMap) -> PResult {
    let me = session(&st, &h).map(|(a, _)| a);
    let challenge = crate::auth::Auth::challenge_bytes(32)?;
    let user = match &me {
        Some(a) => a.clone(),
        None => crate::auth::new_account_id().map_err(|e| AccountError::Store(e.to_string()))?,
    };
    let origin = origin(&st, &h);
    let (name, exclude) = match me.as_deref().and_then(|a| st.auth.account(a)) {
        Some(a) => (
            me_summary(&st, &a.id)["label"].as_str().unwrap_or(&a.id).to_string(),
            a.creds.iter().filter_map(|c| match &c.kind { CredKind::Passkey { cred_id, .. } => Some(json!({ "type": "public-key", "id": cred_id })), _ => None }).collect::<Vec<_>>(),
        ),
        None => ("modgrad account".to_string(), vec![]),
    };
    st.auth.put_challenge(&challenge, Challenge::PasskeyRegister { account: me, user: user.clone() })?;
    Ok(axum::Json(json!({
        "rp": { "id": rp_id(&origin), "name": "modgrad" },
        "user": { "id": B64.encode(user.as_bytes()), "name": name, "displayName": name },
        "challenge": B64.encode(&challenge),
        "pubKeyCredParams": [{ "type": "public-key", "alg": -8 }, { "type": "public-key", "alg": -7 }, { "type": "public-key", "alg": -257 }],
        "timeout": (st.auth.portal.challenge_ttl * 1000.0) as u64,
        "excludeCredentials": exclude,
        "authenticatorSelection": { "residentKey": "required", "requireResidentKey": true, "userVerification": "preferred" },
        "attestation": "none",
    }))
    .into_response())
}

async fn passkey_register_verify(AxState(st): AxState<AppState>, h: HeaderMap, b: Bytes) -> PResult {
    let v = body(&b)?;
    let cdj = resp_field(&v, "clientDataJSON")?;
    let att = resp_field(&v, "attestationObject")?;
    let challenge = client_challenge(&cdj)?;
    let me = session(&st, &h);
    match st.auth.take_challenge(&challenge) {
        Some(Challenge::PasskeyRegister { account, .. }) if account.is_none() || account == me.as_ref().map(|(a, _)| a.clone()) => {}
        Some(Challenge::PasskeyRegister { .. }) => return Err(perr(StatusCode::FORBIDDEN, "forbidden", "this registration was started by another session")),
        _ => return Err(perr(StatusCode::UNAUTHORIZED, "bad_token", "unknown, used or expired passkey challenge")),
    }
    let origin = origin(&st, &h);
    let key = crate::webauthn::verify_registration(&cdj, &att, &challenge, &rp_id(&origin), &origin).map_err(webauthn_err)?;
    let label = v.get("label").and_then(Value::as_str).map(|l| l.chars().filter(|c| !c.is_control()).take(100).collect());
    finish(&st, &h, me, CredKind::Passkey { cred_id: B64.encode(&key.cred_id), public_key: B64.encode(&key.public_key_cose), alg: key.alg, sign_count: key.sign_count, label })
}

async fn passkey_login_options(AxState(st): AxState<AppState>, h: HeaderMap) -> PResult {
    let challenge = crate::auth::Auth::challenge_bytes(32)?;
    st.auth.put_challenge(&challenge, Challenge::PasskeyLogin)?;
    Ok(axum::Json(json!({
        "challenge": B64.encode(&challenge),
        "rpId": rp_id(&origin(&st, &h)),
        "timeout": (st.auth.portal.challenge_ttl * 1000.0) as u64,
        "userVerification": "preferred",
        "allowCredentials": [],
    }))
    .into_response())
}

async fn passkey_login_verify(AxState(st): AxState<AppState>, h: HeaderMap, b: Bytes) -> PResult {
    let v = body(&b)?;
    let cdj = resp_field(&v, "clientDataJSON")?;
    let ad = resp_field(&v, "authenticatorData")?;
    let sig = resp_field(&v, "signature")?;
    let raw = unb64(v.get("rawId").or(v.get("id")).and_then(Value::as_str).ok_or_else(|| bad("rawId is required"))?, "rawId")?;
    let challenge = client_challenge(&cdj)?;
    if st.auth.take_challenge(&challenge) != Some(Challenge::PasskeyLogin) {
        return Err(perr(StatusCode::UNAUTHORIZED, "bad_token", "unknown, used or expired passkey challenge"));
    }
    let cred_id = B64.encode(&raw);
    let (_, cose, alg, count) = st.auth.passkey(&cred_id).ok_or_else(|| perr(StatusCode::UNAUTHORIZED, "unknown_passkey", "this passkey is not registered here"))?;
    let origin = origin(&st, &h);
    let n = crate::webauthn::verify_assertion(&unb64(&cose, "key")?, alg, count, &cdj, &ad, &sig, &challenge, &rp_id(&origin), &origin).map_err(webauthn_err)?;
    finish(&st, &h, session(&st, &h), CredKind::Passkey { cred_id, public_key: cose, alg, sign_count: n, label: None })
}

// ------------------------------------------------------------ me

/// Runs and the newest write of a project, from the current snapshot.
fn project_stats(st: &AppState, project: &str) -> (usize, Option<f64>) {
    let snap = st.read.snapshot();
    let prefix = format!("{project}/");
    let mut n = 0;
    let mut last: Option<f64> = None;
    for (_, c) in snap.runs.iter().filter(|(k, _)| k.starts_with(&prefix)) {
        n += 1;
        let t = ["metrics.jsonl", "config.json"].iter().filter_map(|f| std::fs::metadata(c.dir.join(f)).and_then(|m| m.modified()).ok()).map(crate::live::unix_of).fold(None, |a: Option<f64>, t| Some(a.map_or(t, |a| a.max(t))));
        last = match (last, t) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        };
    }
    (n, last)
}

async fn me(AxState(st): AxState<AppState>, h: HeaderMap) -> PResult {
    let (account, c) = need_session(&st, &h)?;
    let a = st.auth.account(&account).ok_or_else(signed_out)?;
    let creds: Vec<Value> = a
        .creds
        .iter()
        .map(|c| {
            let mut v = json!({ "kind": c.kind.kind(), "id": c.kind.id(), "label": c.kind.label(), "added": c.added, "last_used": c.last_used });
            if let CredKind::Ssh { fingerprint, verified, .. } = &c.kind {
                v["fingerprint"] = json!(fingerprint);
                v["verified"] = json!(verified);
            }
            v
        })
        .collect();
    let projects: Vec<Value> = st
        .auth
        .projects_of(&account)
        .into_iter()
        .map(|(name, o)| {
            let (runs, last) = project_stats(&st, &name);
            json!({ "name": name, "visibility": o.visibility.as_str(), "runs": runs, "last_unix": last, "anon": name.starts_with(crate::auth::ANON_PREFIX), "expires": Value::Null, "created": o.created })
        })
        .collect();
    let mut r = axum::Json(json!({
        "account": { "id": a.id, "created": a.created },
        "creds": creds,
        "projects": projects,
        "sessions": st.auth.sessions_json(&account, Some(&c)),
        "tokens": st.auth.tokens_json(&account),
    }))
    .into_response();
    r.headers_mut().insert(header::SET_COOKIE, session_cookie(&st, &c));
    Ok(r)
}

/// `DELETE /api/v1/me {confirm: <account id>}`: the account, its sessions
/// and tokens, and its projects' data.
async fn delete_me(AxState(st): AxState<AppState>, h: HeaderMap, b: Bytes) -> PResult {
    let (account, _) = need_session(&st, &h)?;
    let v = body(&b)?;
    if v.get("confirm").and_then(Value::as_str) != Some(account.as_str()) {
        return Err(perr(StatusCode::BAD_REQUEST, "bad_confirm", "send {\"confirm\": \"<your account id>\"} to delete the account"));
    }
    let projects = st.auth.delete_account(&account)?;
    for p in projects {
        use tower::ServiceExt;
        if let Err(e) = st.ingest.clone().oneshot(crate::ingest::IngestRequest::DeleteProject { project: p.clone() }).await {
            tracing::error!("deleting project {p}: {e}");
        }
        if let Some(dir) = &st.boards
            && let Err(e) = std::fs::remove_dir_all(dir.join(&p))
            && e.kind() != std::io::ErrorKind::NotFound
        {
            tracing::error!("deleting boards of {p}: {e}");
        }
    }
    let mut r = StatusCode::NO_CONTENT.into_response();
    r.headers_mut().insert(header::SET_COOKIE, set_cookie(&st, "", 0));
    Ok(r)
}

fn visibility(v: &Value, default: Visibility) -> Result<Visibility, PortalError> {
    match v.get("visibility") {
        None | Some(Value::Null) => Ok(default),
        Some(x) => x.as_str().and_then(Visibility::parse).ok_or_else(|| bad("visibility is \"private\" or \"public\"")),
    }
}

async fn create_project(AxState(st): AxState<AppState>, h: HeaderMap, b: Bytes) -> PResult {
    let (account, _) = need_session(&st, &h)?;
    let v = body(&b)?;
    let name = field(&v, "name")?;
    let vis = visibility(&v, Visibility::Private)?;
    let prefix = format!("{name}/");
    let in_use = st.read.snapshot().runs.keys().any(|r| r.starts_with(&prefix)) || st.boards.as_ref().is_some_and(|d| d.join(name).exists());
    st.auth.create_project(&account, name, vis, in_use)?;
    Ok(json_status(StatusCode::CREATED, json!({ "name": name, "visibility": vis.as_str() })))
}

async fn patch_project(AxState(st): AxState<AppState>, Path(name): Path<String>, h: HeaderMap, b: Bytes) -> PResult {
    let (account, _) = need_session(&st, &h)?;
    let v = body(&b)?;
    if v.get("visibility").is_none() {
        return Err(bad("visibility is required"));
    }
    let vis = visibility(&v, Visibility::Private)?;
    st.auth.set_visibility(&account, &name, vis)?;
    Ok(axum::Json(json!({ "name": name, "visibility": vis.as_str() })).into_response())
}

async fn claim(AxState(st): AxState<AppState>, h: HeaderMap, b: Bytes) -> PResult {
    let (account, _) = need_session(&st, &h)?;
    let v = body(&b)?;
    let project = field(&v, "project")?;
    st.auth.claim(&account, project, field(&v, "token")?)?;
    Ok(axum::Json(json!({ "project": project, "visibility": "private" })).into_response())
}

async fn mint_token(AxState(st): AxState<AppState>, h: HeaderMap, b: Bytes) -> PResult {
    let (account, _) = need_session(&st, &h)?;
    let v = body(&b)?;
    let project = field(&v, "project")?;
    let label = v.get("label").and_then(Value::as_str).map(String::from);
    let days = match v.get("expires_days") {
        None | Some(Value::Null) => None,
        Some(d) => Some(d.as_f64().filter(|d| *d > 0.0).ok_or_else(|| bad("expires_days is a positive number"))?),
    };
    let (id, token) = st.auth.mint_token(&account, project, label, days)?;
    let base = st.base_url(&h);
    Ok(json_status(
        StatusCode::CREATED,
        json!({ "id": id, "token": token, "project": project, "note": "shown once; store it now", "use": format!("MODGRAD_TOKEN={token} modgrad run --server {base} -- python train.py") }),
    ))
}

async fn revoke_token(AxState(st): AxState<AppState>, Path(id): Path<String>, h: HeaderMap) -> PResult {
    let (account, _) = need_session(&st, &h)?;
    st.auth.revoke_token(&account, &id)?;
    Ok(StatusCode::NO_CONTENT.into_response())
}

/// `POST /api/v1/me/creds/ssh {public_key, label?}`: the key is added
/// unconfirmed (see [`crate::account`]) until `ssh <server> login`.
async fn add_ssh(AxState(st): AxState<AppState>, h: HeaderMap, b: Bytes) -> PResult {
    let (account, _) = need_session(&st, &h)?;
    let v = body(&b)?;
    let line = field(&v, "public_key")?.trim();
    let fp = ssh_fingerprint(line)?;
    let label = v.get("label").and_then(Value::as_str).map(String::from).or_else(|| line.split_whitespace().nth(2).map(String::from)).map(|l| l.chars().filter(|c| !c.is_control()).take(100).collect());
    let id = st.auth.add_ssh_key(&account, &fp, line, label)?;
    let confirm = st.portal.ssh_login.clone().unwrap_or_else(|| "ssh <server> login".into());
    Ok(json_status(StatusCode::CREATED, json!({ "id": id, "fingerprint": fp, "verified": false, "confirm": format!("{confirm}  (from this key, then open the link while signed in)") })))
}

/// The `SHA256:…` fingerprint of an openssh public key line, exactly as
/// the SSH door computes it ([`crate::ssh::fingerprint`]).
#[cfg(feature = "ssh")]
pub fn ssh_fingerprint(line: &str) -> Result<String, PortalError> {
    let k = russh::keys::PublicKey::from_openssh(line).map_err(|e| perr(StatusCode::BAD_REQUEST, "bad_key", format!("not an openssh public key line (\"ssh-ed25519 AAAA… comment\"): {e}")))?;
    Ok(crate::ssh::fingerprint(&k))
}

#[cfg(not(feature = "ssh"))]
pub fn ssh_fingerprint(_line: &str) -> Result<String, PortalError> {
    Err(perr(StatusCode::NOT_IMPLEMENTED, "ssh_unsupported", "this server was built without SSH"))
}

async fn remove_cred(AxState(st): AxState<AppState>, Path(id): Path<String>, h: HeaderMap) -> PResult {
    let (account, _) = need_session(&st, &h)?;
    st.auth.remove_cred(&account, &id)?;
    Ok(StatusCode::NO_CONTENT.into_response())
}

async fn end_session(AxState(st): AxState<AppState>, Path(id): Path<String>, h: HeaderMap) -> PResult {
    let (account, _) = need_session(&st, &h)?;
    st.auth.end_session_id(&account, &id)?;
    Ok(StatusCode::NO_CONTENT.into_response())
}

/// The portal routes, CSRF-checked.
pub fn routes(st: AppState) -> Router<AppState> {
    Router::new()
        .route("/api/v1/auth/methods", get(methods))
        .route("/api/v1/auth/logout", post(logout))
        .route("/api/v1/auth/whoami", get(whoami))
        .route("/api/v1/auth/email", post(email))
        .route("/api/v1/auth/email/consume", post(email_consume))
        .route("/api/v1/auth/ssh/consume", post(ssh_consume))
        .route("/api/v1/auth/zafu/challenge", post(zafu_challenge_route))
        .route("/api/v1/auth/zafu/verify", post(zafu_verify))
        .route("/api/v1/auth/passkey/register/options", post(passkey_register_options))
        .route("/api/v1/auth/passkey/register/verify", post(passkey_register_verify))
        .route("/api/v1/auth/passkey/login/options", post(passkey_login_options))
        .route("/api/v1/auth/passkey/login/verify", post(passkey_login_verify))
        .route("/api/v1/me", get(me).delete(delete_me))
        .route("/api/v1/me/projects", post(create_project))
        .route("/api/v1/me/projects/:name", patch(patch_project))
        .route("/api/v1/me/claim", post(claim))
        .route("/api/v1/me/tokens", post(mint_token))
        .route("/api/v1/me/tokens/:id", delete(revoke_token))
        .route("/api/v1/me/creds/ssh", post(add_ssh))
        .route("/api/v1/me/creds/:id", delete(remove_cred))
        .route("/api/v1/me/sessions/:id", delete(end_session))
        .route_layer(axum::middleware::from_fn_with_state(st, csrf_layer))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn origins_and_cookies() {
        assert_eq!(origin_of("https://modgrad.com/x/y"), "https://modgrad.com");
        assert_eq!(origin_of("http://127.0.0.1:8790"), "http://127.0.0.1:8790");
        assert_eq!(rp_id("https://modgrad.com"), "modgrad.com");
        assert_eq!(rp_id("http://localhost:8790"), "localhost");
        assert_eq!(rp_id("http://[::1]:8790"), "::1");
        assert_eq!(path_of("https://modgrad.com/app/"), "/app");
        assert_eq!(path_of("https://modgrad.com/a/b"), "/a/b");
        assert_eq!(path_of("https://modgrad.com"), "/");
        assert_eq!(path_of("http://127.0.0.1:8790/"), "/");
        let v = "a".repeat(43);
        let mut h = HeaderMap::new();
        h.insert(header::COOKIE, HeaderValue::from_str(&format!("theme=dark; {COOKIE}={v}; x=1")).unwrap());
        assert_eq!(cookie(&h), Some(v));
        h.insert(header::COOKIE, HeaderValue::from_static("mg_session=short"));
        assert_eq!(cookie(&h), None);
    }

    #[test]
    fn email_rate_limits() {
        let mut l = Limits::default();
        let (pa, pi) = ((3, 900.0), (10, 3600.0));
        for _ in 0..3 {
            assert!(l.allow("a@x.org", "1.2.3.4", pa, pi, 0.0));
        }
        assert!(!l.allow("a@x.org", "5.6.7.8", pa, pi, 1.0), "per address");
        assert!(l.allow("a@x.org", "5.6.7.8", pa, pi, 901.0), "the window slides");
        for i in 0..7 {
            assert!(l.allow(&format!("b{i}@x.org"), "1.2.3.4", pa, pi, 2.0));
        }
        assert!(!l.allow("c@x.org", "1.2.3.4", pa, pi, 3.0), "per ip: 3 + 7 reached 10");
        assert!(l.allow("c@x.org", "9.9.9.9", pa, pi, 3.0));
    }

    #[test]
    fn client_ip_trusts_only_a_local_proxy() {
        let mut h = HeaderMap::new();
        h.insert("x-forwarded-for", HeaderValue::from_static("6.6.6.6, 1.2.3.4"));
        assert_eq!(client_ip(Some("8.8.8.8:1".parse().unwrap()), &h), "8.8.8.8");
        assert_eq!(client_ip(Some("127.0.0.1:1".parse().unwrap()), &h), "1.2.3.4");
        // nginx on the same container network
        assert_eq!(client_ip(Some("10.89.0.7:1".parse().unwrap()), &h), "1.2.3.4");
        assert_eq!(client_ip(Some("[fd00::7]:1".parse().unwrap()), &h), "1.2.3.4");
        assert_eq!(client_ip(Some("[2001:db8::1]:1".parse().unwrap()), &h), "2001:db8::1");
        assert_eq!(client_ip(None, &HeaderMap::new()), "unknown");
    }
}
