//! The sign-in portal over the router (tower `oneshot`, no sockets, no
//! mail leaves the process: a [`MemoryMailer`] catches it).

use axum::Router;
use axum::body::{Body, to_bytes};
use axum::http::{HeaderMap, Request, StatusCode};
use modgrad_obs_server::auth::{Auth, AuthConfig, PortalConfig};
use modgrad_obs_server::http::{AppState, router};
use modgrad_obs_server::mail::MemoryMailer;
use modgrad_obs_server::{IngestConfig, open};
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

const ORIGIN: &str = "http://obs.test";

struct T {
    app: Router,
    mail: MemoryMailer,
    auth: Auth,
    root: PathBuf,
}

fn tmp(tag: &str) -> PathBuf {
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let d = std::env::temp_dir().join(format!("obs-portal-{tag}-{}-{}", std::process::id(), N.fetch_add(1, std::sync::atomic::Ordering::SeqCst)));
    let _ = std::fs::remove_dir_all(&d);
    d
}

fn app_with(tag: &str, p: PortalConfig) -> T {
    let root = tmp(tag);
    let (ingest, read) = open(&root, IngestConfig::default()).unwrap();
    std::fs::write(root.join("dashboard.html"), "<html>dash</html>").unwrap();
    let auth = Auth::open(&root, AuthConfig::default()).unwrap().with_portal(p);
    let mail = MemoryMailer::default();
    let st = AppState::new(ingest, read, Some(root.clone()), auth.clone()).with_boards(root.join("boards")).with_mailer(Arc::new(mail.clone()));
    T { app: router(st), mail, auth, root }
}

fn app(tag: &str) -> T {
    app_with(tag, PortalConfig::default())
}

/// A request as a same-origin browser page sends it (Origin set), with an
/// optional session cookie and bearer token.
#[derive(Default, Clone)]
struct Req<'a> {
    cookie: Option<&'a str>,
    token: Option<&'a str>,
    origin: Option<&'a str>,
    fetch_site: Option<&'a str>,
    host: Option<&'a str>,
    no_origin: bool,
}

async fn call(t: &T, method: &str, uri: &str, r: Req<'_>, body: Value) -> (StatusCode, HeaderMap, Value) {
    let mut b = Request::builder().method(method).uri(uri).header("host", r.host.unwrap_or("obs.test"));
    if !r.no_origin && method != "GET" {
        b = b.header("origin", r.origin.unwrap_or(ORIGIN));
    }
    if let Some(c) = r.cookie {
        b = b.header("cookie", format!("mg_session={c}"));
    }
    if let Some(tk) = r.token {
        b = b.header("authorization", format!("Bearer {tk}"));
    }
    if let Some(s) = r.fetch_site {
        b = b.header("sec-fetch-site", s);
    }
    let body = if body.is_null() { Body::empty() } else { Body::from(body.to_string()) };
    let resp = t.app.clone().oneshot(b.body(body).unwrap()).await.unwrap();
    let (code, h) = (resp.status(), resp.headers().clone());
    let bytes = to_bytes(resp.into_body(), 1 << 24).await.unwrap();
    (code, h, serde_json::from_slice(&bytes).unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into())))
}

fn as_cookie(c: Option<&str>) -> Req<'_> {
    Req { cookie: c, ..Default::default() }
}

/// The `mg_session` value a response set.
fn set_cookie(h: &HeaderMap) -> Option<String> {
    let v = h.get("set-cookie")?.to_str().ok()?;
    let c = v.strip_prefix("mg_session=")?.split(';').next()?.to_string();
    (!c.is_empty()).then_some(c)
}

/// Wait for the `n`-th mail (sending runs off the request).
async fn nth_mail(t: &T, n: usize) -> modgrad_obs_server::mail::Mail {
    for _ in 0..200 {
        if let Some(m) = t.mail.0.lock().unwrap().get(n).cloned() {
            return m;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("no mail #{n}");
}

fn link_token(m: &modgrad_obs_server::mail::Mail, kind: &str) -> String {
    let at = m.body.find(&format!("#{kind}=")).expect("link in the body") + kind.len() + 2;
    m.body[at..].split_whitespace().next().unwrap().to_string()
}

/// Sign in with a fresh address; returns (cookie, account id).
async fn email_sign_in(t: &T, address: &str, cookie: Option<&str>) -> (StatusCode, Option<String>, Value) {
    let before = t.mail.0.lock().unwrap().len();
    let (c, _, v) = call(t, "POST", "/api/v1/auth/email", Req::default(), json!({ "email": address })).await;
    assert_eq!(c, StatusCode::ACCEPTED, "{v}");
    let m = nth_mail(t, before).await;
    assert_eq!(m.to, address.trim().to_lowercase());
    let (c, h, v) = call(t, "POST", "/api/v1/auth/email/consume", as_cookie(cookie), json!({ "token": link_token(&m, "email") })).await;
    (c, set_cookie(&h), v)
}

async fn signed_in(t: &T, address: &str) -> (String, String) {
    let (c, cookie, v) = email_sign_in(t, address, None).await;
    assert_eq!(c, StatusCode::OK, "{v}");
    (cookie.expect("a session cookie"), v["account"]["id"].as_str().unwrap().to_string())
}

#[tokio::test]
async fn email_magic_link() {
    let t = app("email");
    let (c, cookie, v) = email_sign_in(&t, "  Me@Example.org ", None).await;
    assert_eq!(c, StatusCode::OK, "{v}");
    assert_eq!((v["created"].as_bool(), v["linked"].as_bool()), (Some(true), Some(false)));
    assert!(v["account"]["id"].as_str().unwrap().starts_with("acct_") && v["account"]["created"].is_f64());
    let m = t.mail.0.lock().unwrap()[0].clone();
    assert_eq!(m.subject, "Sign in to modgrad");
    assert!(m.body.contains("http://obs.test/login#email=") && m.body.contains("expires in 15 minutes") && m.body.contains("didn't ask"));
    let cookie = cookie.unwrap();
    let (c, _, me) = call(&t, "GET", "/api/v1/me", as_cookie(Some(&cookie)), Value::Null).await;
    assert_eq!(c, StatusCode::OK, "{me}");
    assert_eq!(me["creds"][0]["kind"], "email");
    assert_eq!(me["creds"][0]["label"], "me@example.org");
    assert_eq!(me["sessions"][0]["current"], true);
    // the same address again: same account
    let (_, _, again) = email_sign_in(&t, "me@example.org", None).await;
    assert_eq!((again["account"]["id"].clone(), again["created"].as_bool()), (v["account"]["id"].clone(), Some(false)));
    // a used link: 410, as an unknown one
    let tok = link_token(&m, "email");
    let (c, _, e) = call(&t, "POST", "/api/v1/auth/email/consume", Req::default(), json!({ "token": tok })).await;
    assert_eq!((c, e["code"].as_str()), (StatusCode::GONE, Some("token_expired")));
    let (c, _, e) = call(&t, "POST", "/api/v1/auth/email/consume", Req::default(), json!({ "token": "nope" })).await;
    assert_eq!((c, e["code"].as_str()), (StatusCode::GONE, Some("token_expired")));
    let (c, _, e) = call(&t, "POST", "/api/v1/auth/email", Req::default(), json!({ "email": "not an address" })).await;
    assert_eq!((c, e["code"].as_str()), (StatusCode::BAD_REQUEST, Some("bad_email")));
    // nothing secret on disk
    let store = std::fs::read_to_string(t.root.join(".obs-auth.json")).unwrap();
    assert!(!store.contains(&cookie) && !store.contains(&tok));
}

#[tokio::test]
async fn email_rate_limit_is_silent() {
    let t = app("ratelimit");
    for _ in 0..5 {
        let (c, _, _) = call(&t, "POST", "/api/v1/auth/email", Req::default(), json!({ "email": "x@y.org" })).await;
        assert_eq!(c, StatusCode::ACCEPTED, "the same answer whether sent or not");
    }
    nth_mail(&t, 2).await;
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(t.mail.0.lock().unwrap().len(), 3, "3 per address per 15 minutes");
}

#[tokio::test]
async fn expired_email_link() {
    let t = app_with("email-exp", PortalConfig { email_ttl: 1e-9, ..Default::default() });
    call(&t, "POST", "/api/v1/auth/email", Req::default(), json!({ "email": "x@y.org" })).await;
    let m = nth_mail(&t, 0).await;
    let (c, _, e) = call(&t, "POST", "/api/v1/auth/email/consume", Req::default(), json!({ "token": link_token(&m, "email") })).await;
    assert_eq!((c, e["code"].as_str()), (StatusCode::GONE, Some("token_expired")));
}

fn zafu_sign(key: &ring::signature::Ed25519KeyPair, challenge_hex: &str) -> Value {
    use ring::signature::KeyPair;
    let msg: Vec<u8> = (0..challenge_hex.len()).step_by(2).map(|i| u8::from_str_radix(&challenge_hex[i..i + 2], 16).unwrap()).collect();
    let hex = |b: &[u8]| b.iter().map(|x| format!("{x:02x}")).collect::<String>();
    json!({ "challenge_hex": challenge_hex, "signature": hex(key.sign(&msg).as_ref()), "public_key": hex(key.public_key().as_ref()) })
}

fn zafu_key() -> ring::signature::Ed25519KeyPair {
    let rng = ring::rand::SystemRandom::new();
    let pkcs8 = ring::signature::Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
    ring::signature::Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap()
}

#[tokio::test]
async fn zafu_sign_in() {
    let t = app("zafu");
    let key = zafu_key();
    let (c, _, ch) = call(&t, "POST", "/api/v1/auth/zafu/challenge", Req::default(), Value::Null).await;
    assert_eq!(c, StatusCode::OK, "{ch}");
    let hexs = ch["challenge_hex"].as_str().unwrap();
    let text = String::from_utf8((0..hexs.len()).step_by(2).map(|i| u8::from_str_radix(&hexs[i..i + 2], 16).unwrap()).collect()).unwrap();
    let lines: Vec<&str> = text.split('\n').collect();
    assert_eq!((lines.len(), lines[0], lines[1]), (4, "modgrad-login-v1", ORIGIN));
    assert_eq!(lines[3], ch["expires"].as_u64().unwrap().to_string());
    assert!(text.starts_with('m') && text.len() <= 1024);
    let signed = zafu_sign(&key, hexs);
    let (c, h, v) = call(&t, "POST", "/api/v1/auth/zafu/verify", Req::default(), signed.clone()).await;
    assert_eq!(c, StatusCode::OK, "{v}");
    assert!(v["created"] == true && set_cookie(&h).is_some());
    // replay
    let (c, _, e) = call(&t, "POST", "/api/v1/auth/zafu/verify", Req::default(), signed).await;
    assert_eq!((c, e["code"].as_str()), (StatusCode::UNAUTHORIZED, Some("bad_token")));
    // a wrong signature (another key over the right challenge) burns the challenge
    let (_, _, ch) = call(&t, "POST", "/api/v1/auth/zafu/challenge", Req::default(), Value::Null).await;
    let mut forged = zafu_sign(&zafu_key(), ch["challenge_hex"].as_str().unwrap());
    forged["public_key"] = zafu_sign(&key, ch["challenge_hex"].as_str().unwrap())["public_key"].clone();
    let (c, _, e) = call(&t, "POST", "/api/v1/auth/zafu/verify", Req::default(), forged).await;
    assert_eq!((c, e["code"].as_str()), (StatusCode::UNAUTHORIZED, Some("bad_signature")));
    // a challenge issued for another origin (host a.test) does not sign in on b.test
    let a = Req { host: Some("a.test"), origin: Some("http://a.test"), ..Default::default() };
    let b = Req { host: Some("b.test"), origin: Some("http://b.test"), ..Default::default() };
    let (_, _, ch) = call(&t, "POST", "/api/v1/auth/zafu/challenge", a, Value::Null).await;
    let (c, _, e) = call(&t, "POST", "/api/v1/auth/zafu/verify", b, zafu_sign(&key, ch["challenge_hex"].as_str().unwrap())).await;
    assert_eq!(c, StatusCode::UNAUTHORIZED, "{e}");
    assert!(e["error"].as_str().unwrap().contains("another site"), "{e}");
}

#[tokio::test]
async fn zafu_challenge_expires() {
    let t = app_with("zafu-exp", PortalConfig { challenge_ttl: 1e-9, ..Default::default() });
    let (_, _, ch) = call(&t, "POST", "/api/v1/auth/zafu/challenge", Req::default(), Value::Null).await;
    tokio::time::sleep(Duration::from_millis(5)).await;
    let (c, _, e) = call(&t, "POST", "/api/v1/auth/zafu/verify", Req::default(), zafu_sign(&zafu_key(), ch["challenge_hex"].as_str().unwrap())).await;
    assert_eq!((c, e["code"].as_str()), (StatusCode::UNAUTHORIZED, Some("bad_token")));
}

#[tokio::test]
async fn linking_rule_and_last_cred() {
    let t = app("link");
    let (cookie, me) = signed_in(&t, "me@x.org").await;
    // a zafu key linked while signed in
    let key = zafu_key();
    let (_, _, ch) = call(&t, "POST", "/api/v1/auth/zafu/challenge", as_cookie(Some(&cookie)), Value::Null).await;
    let (c, h, v) = call(&t, "POST", "/api/v1/auth/zafu/verify", as_cookie(Some(&cookie)), zafu_sign(&key, ch["challenge_hex"].as_str().unwrap())).await;
    assert_eq!((c, v["linked"].as_bool(), v["account"]["id"].as_str()), (StatusCode::OK, Some(true), Some(me.as_str())), "{v}");
    assert!(set_cookie(&h).is_none(), "linking keeps the session");
    // another account's address: 409, nothing merged
    let (other_cookie, other) = signed_in(&t, "you@x.org").await;
    let (c, _, v) = email_sign_in(&t, "me@x.org", Some(&other_cookie)).await;
    assert_eq!((c, v["code"].as_str()), (StatusCode::CONFLICT, Some("cred_taken")));
    let (_, _, them) = call(&t, "GET", "/api/v1/me", as_cookie(Some(&other_cookie)), Value::Null).await;
    assert_eq!((them["account"]["id"].as_str(), them["creds"].as_array().unwrap().len()), (Some(other.as_str()), 1));
    // the last credential cannot go; one of two can
    let id = them["creds"][0]["id"].as_str().unwrap().to_string();
    let (c, _, v) = call(&t, "DELETE", &format!("/api/v1/me/creds/{id}"), as_cookie(Some(&other_cookie)), Value::Null).await;
    assert_eq!((c, v["code"].as_str()), (StatusCode::CONFLICT, Some("last_cred")));
    let (_, _, mine) = call(&t, "GET", "/api/v1/me", as_cookie(Some(&cookie)), Value::Null).await;
    let zid = mine["creds"].as_array().unwrap().iter().find(|c| c["kind"] == "zafu").unwrap()["id"].as_str().unwrap().to_string();
    let (c, _, _) = call(&t, "DELETE", &format!("/api/v1/me/creds/{zid}"), as_cookie(Some(&cookie)), Value::Null).await;
    assert_eq!(c, StatusCode::NO_CONTENT);
}

#[tokio::test]
async fn ssh_login_link() {
    let t = app("ssh");
    // what the door does: first connect, then `ssh <server> login`
    let (project, _) = t.auth.ssh_key_project_with("SHA256:abc", Some("ssh-ed25519 AAAA abc")).unwrap();
    let tok = t.auth.ssh_login_token("SHA256:abc", "ssh-ed25519 AAAA abc").unwrap();
    let (c, h, v) = call(&t, "POST", "/api/v1/auth/ssh/consume", Req::default(), json!({ "token": tok })).await;
    assert_eq!(c, StatusCode::OK, "{v}");
    let cookie = set_cookie(&h).unwrap();
    let (_, _, me) = call(&t, "GET", "/api/v1/me", as_cookie(Some(&cookie)), Value::Null).await;
    assert_eq!(me["projects"][0]["name"], project.as_str());
    assert_eq!((me["projects"][0]["anon"].as_bool(), me["projects"][0]["visibility"].as_str()), (Some(true), Some("public")));
    assert!(me["projects"][0]["last_unix"].is_null());
    let (c, _, e) = call(&t, "POST", "/api/v1/auth/ssh/consume", Req::default(), json!({ "token": tok })).await;
    assert_eq!((c, e["code"].as_str()), (StatusCode::GONE, Some("token_expired")), "single use");
    // a second key, never signed in on the web, joins an email account with its project
    let (cookie2, acct2) = signed_in(&t, "me@x.org").await;
    let (p2, _) = t.auth.ssh_key_project_with("SHA256:def", Some("ssh-ed25519 AAAA def")).unwrap();
    let tok = t.auth.ssh_login_token("SHA256:def", "ssh-ed25519 AAAA def").unwrap();
    let (c, _, v) = call(&t, "POST", "/api/v1/auth/ssh/consume", as_cookie(Some(&cookie2)), json!({ "token": tok })).await;
    assert_eq!((c, v["linked"].as_bool(), v["account"]["id"].as_str()), (StatusCode::OK, Some(true), Some(acct2.as_str())), "{v}");
    assert_eq!(t.auth.owned(&p2).unwrap().owner, acct2);
    // the first key's account has signed in on the web: linking it elsewhere is 409
    let tok = t.auth.ssh_login_token("SHA256:abc", "ssh-ed25519 AAAA abc").unwrap();
    let (c, _, v) = call(&t, "POST", "/api/v1/auth/ssh/consume", as_cookie(Some(&cookie2)), json!({ "token": tok })).await;
    assert_eq!((c, v["code"].as_str()), (StatusCode::CONFLICT, Some("cred_taken")));
}

#[cfg(feature = "ssh")]
#[tokio::test]
async fn pasted_ssh_key_has_the_doors_fingerprint() {
    use russh::keys::{PrivateKey, ssh_key};
    let t = app("sshfp");
    let (cookie, _) = signed_in(&t, "me@x.org").await;
    let key = PrivateKey::from(ssh_key::private::Ed25519Keypair::from_seed(&[7u8; 32]));
    let line = format!("{} me@laptop", key.public_key().to_openssh().unwrap());
    let (c, _, v) = call(&t, "POST", "/api/v1/me/creds/ssh", as_cookie(Some(&cookie)), json!({ "public_key": line })).await;
    assert_eq!(c, StatusCode::CREATED, "{v}");
    assert_eq!(v["fingerprint"].as_str().unwrap(), modgrad_obs_server::ssh::fingerprint(key.public_key()));
    assert_eq!(v["verified"], false);
    let (_, _, me) = call(&t, "GET", "/api/v1/me", as_cookie(Some(&cookie)), Value::Null).await;
    let ssh = me["creds"].as_array().unwrap().iter().find(|c| c["kind"] == "ssh").unwrap().clone();
    assert_eq!((ssh["label"].as_str(), ssh["verified"].as_bool()), (Some("me@laptop"), Some(false)));
    let (c, _, e) = call(&t, "POST", "/api/v1/me/creds/ssh", as_cookie(Some(&cookie)), json!({ "public_key": "ssh-ed25519 notbase64" })).await;
    assert_eq!((c, e["code"].as_str()), (StatusCode::BAD_REQUEST, Some("bad_key")));
}

#[tokio::test]
async fn private_projects_and_tokens() {
    let t = app("private");
    let (cookie, _) = signed_in(&t, "me@x.org").await;
    let (c, _, v) = call(&t, "POST", "/api/v1/me/projects", as_cookie(Some(&cookie)), json!({ "name": "secret-lab" })).await;
    assert_eq!((c, v["visibility"].as_str()), (StatusCode::CREATED, Some("private")), "{v}");
    for (name, code) in [("secret-lab", "name_taken"), ("anon-x", "bad_name"), ("api", "bad_name"), ("No", "bad_name")] {
        let (_, _, e) = call(&t, "POST", "/api/v1/me/projects", as_cookie(Some(&cookie)), json!({ "name": name })).await;
        assert_eq!(e["code"], code, "{name}");
    }
    // a token, shown once, writes as a bearer
    let (c, _, tk) = call(&t, "POST", "/api/v1/me/tokens", as_cookie(Some(&cookie)), json!({ "project": "secret-lab", "label": "ci" })).await;
    assert_eq!(c, StatusCode::CREATED, "{tk}");
    let token = tk["token"].as_str().unwrap().to_string();
    let (c, _, v) = call(&t, "POST", "/api/v1/runs/r1/log", Req { token: Some(&token), ..Default::default() }, json!({ "loss": 0.5 })).await;
    assert_eq!((c, v["run"].as_str()), (StatusCode::OK, Some("secret-lab/r1")), "{v}");
    let (_, _, me) = call(&t, "GET", "/api/v1/me", as_cookie(Some(&cookie)), Value::Null).await;
    assert!(!me.to_string().contains(&token), "listing never shows a token");
    assert_eq!((me["tokens"][0]["label"].as_str(), me["projects"][0]["runs"].as_u64()), (Some("ci"), Some(1)));
    assert!(me["tokens"][0]["expires"].is_null() && me["projects"][0]["last_unix"].is_f64());
    // reads: refused without the cookie or token, allowed with either
    let reads = ["/api/v1/runs/secret-lab/r1/history", "/api/v1/runs/secret-lab/r1/config", "/api/v1/runs/secret-lab/r1/samples", "/api/runs/secret-lab%2Fr1/metrics/loss", "/api/index?project=secret-lab"];
    let (other, _) = signed_in(&t, "you@x.org").await;
    for uri in reads {
        let (c, _, e) = call(&t, "GET", uri, Req::default(), Value::Null).await;
        assert_eq!((c, e["code"].as_str()), (StatusCode::UNAUTHORIZED, Some("signed_out")), "{uri}: {e}");
        let (c, _, _) = call(&t, "GET", uri, as_cookie(Some(&other)), Value::Null).await;
        assert_eq!(c, StatusCode::FORBIDDEN, "{uri}");
        let (c, _, _) = call(&t, "GET", uri, as_cookie(Some(&cookie)), Value::Null).await;
        assert_eq!(c, StatusCode::OK, "{uri}");
        let (c, _, _) = call(&t, "GET", uri, Req { token: Some(&token), ..Default::default() }, Value::Null).await;
        assert_eq!(c, StatusCode::OK, "{uri}");
    }
    // a run that does not exist is refused the same way (no probing names)
    let (c, _, _) = call(&t, "GET", "/api/v1/runs/secret-lab/nope/history", Req::default(), Value::Null).await;
    assert_eq!(c, StatusCode::UNAUTHORIZED);
    let (c, _, _) = call(&t, "GET", "/api/v1/runs/secret-lab/r1/stream", Req::default(), Value::Null).await;
    assert_eq!(c, StatusCode::UNAUTHORIZED);
    // the store's own files are not a way around it
    let (c, _, _) = call(&t, "GET", "/runs/secret-lab/r1/metrics.jsonl", Req::default(), Value::Null).await;
    assert_eq!(c, StatusCode::NOT_FOUND);
    // the unfiltered index lists it for the owner only
    let ids = |v: &Value| v["runs"].as_array().unwrap().iter().map(|r| r["id"].as_str().unwrap().to_string()).collect::<Vec<_>>();
    assert!(ids(&call(&t, "GET", "/api/index", Req::default(), Value::Null).await.2).is_empty());
    assert_eq!(ids(&call(&t, "GET", "/api/index", as_cookie(Some(&cookie)), Value::Null).await.2), vec!["secret-lab/r1"]);
    // public: today's behaviour
    let (c, _, _) = call(&t, "PATCH", "/api/v1/me/projects/secret-lab", as_cookie(Some(&cookie)), json!({ "visibility": "public" })).await;
    assert_eq!(c, StatusCode::OK);
    let (c, _, _) = call(&t, "GET", reads[0], Req::default(), Value::Null).await;
    assert_eq!(c, StatusCode::OK);
    let (c, _, _) = call(&t, "PATCH", "/api/v1/me/projects/secret-lab", as_cookie(Some(&other)), json!({ "visibility": "private" })).await;
    assert_eq!(c, StatusCode::NOT_FOUND, "not yours");
    // revoked: the token stops working
    let (c, _, _) = call(&t, "DELETE", &format!("/api/v1/me/tokens/{}", tk["id"].as_str().unwrap()), as_cookie(Some(&cookie)), Value::Null).await;
    assert_eq!(c, StatusCode::NO_CONTENT);
    let (c, _, _) = call(&t, "POST", "/api/v1/runs/r1/log", Req { token: Some(&token), ..Default::default() }, json!({ "loss": 0.4 })).await;
    assert_eq!(c, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn claim_an_anonymous_project() {
    let t = app("claim");
    let (_, _, p) = call(&t, "POST", "/api/v1/projects", Req { no_origin: true, ..Default::default() }, Value::Null).await;
    let (project, token) = (p["project"].as_str().unwrap().to_string(), p["token"].as_str().unwrap().to_string());
    call(&t, "POST", "/api/v1/runs/r/log", Req { token: Some(&token), ..Default::default() }, json!({ "loss": 1 })).await;
    let uri = format!("/api/v1/runs/{project}/r/history");
    assert_eq!(call(&t, "GET", &uri, Req::default(), Value::Null).await.0, StatusCode::OK, "anonymous: readable by id");
    let (cookie, _) = signed_in(&t, "me@x.org").await;
    let (c, _, e) = call(&t, "POST", "/api/v1/me/claim", as_cookie(Some(&cookie)), json!({ "project": project, "token": "obsp_wrong" })).await;
    assert_eq!((c, e["code"].as_str()), (StatusCode::FORBIDDEN, Some("forbidden")));
    let (c, _, v) = call(&t, "POST", "/api/v1/me/claim", as_cookie(Some(&cookie)), json!({ "project": project, "token": token })).await;
    assert_eq!(c, StatusCode::OK, "{v}");
    assert_eq!(call(&t, "GET", &uri, Req::default(), Value::Null).await.0, StatusCode::UNAUTHORIZED, "now private");
    assert_eq!(call(&t, "GET", &uri, Req { token: Some(&token), ..Default::default() }, Value::Null).await.0, StatusCode::OK);
    let (_, _, me) = call(&t, "GET", "/api/v1/me", as_cookie(Some(&cookie)), Value::Null).await;
    assert_eq!((me["projects"][0]["name"].as_str(), me["projects"][0]["anon"].as_bool()), (Some(project.as_str()), Some(true)));
    assert!(me["tokens"][0]["expires"].is_null(), "a claimed project does not expire");
    let (other, _) = signed_in(&t, "you@x.org").await;
    let (c, _, e) = call(&t, "POST", "/api/v1/me/claim", as_cookie(Some(&other)), json!({ "project": project, "token": token })).await;
    assert_eq!((c, e["code"].as_str()), (StatusCode::CONFLICT, Some("already_owned")));
}

#[tokio::test]
async fn csrf_logout_and_sessions() {
    let t = app("csrf");
    let (cookie, _) = signed_in(&t, "me@x.org").await;
    let new = json!({ "name": "proj-a" });
    let cross = [
        Req { cookie: Some(&cookie), origin: Some("https://evil.example"), ..Default::default() },
        Req { cookie: Some(&cookie), fetch_site: Some("cross-site"), ..Default::default() },
        Req { cookie: Some(&cookie), fetch_site: Some("same-site"), ..Default::default() },
        Req { cookie: Some(&cookie), no_origin: true, ..Default::default() },
    ];
    for r in cross {
        let (c, _, e) = call(&t, "POST", "/api/v1/me/projects", r, new.clone()).await;
        assert_eq!((c, e["code"].as_str()), (StatusCode::FORBIDDEN, Some("csrf")));
    }
    let (c, _, e) = call(&t, "POST", "/api/v1/auth/email", Req { origin: Some("https://evil.example"), ..Default::default() }, json!({ "email": "x@y.org" })).await;
    assert_eq!((c, e["code"].as_str()), (StatusCode::FORBIDDEN, Some("csrf")), "login CSRF too");
    let ok = Req { cookie: Some(&cookie), fetch_site: Some("same-origin"), no_origin: true, ..Default::default() };
    assert_eq!(call(&t, "POST", "/api/v1/me/projects", ok, new).await.0, StatusCode::CREATED);
    // bearer writes are not cookie requests
    let (_, _, p) = call(&t, "POST", "/api/v1/projects", Req { no_origin: true, ..Default::default() }, Value::Null).await;
    assert_eq!(call(&t, "POST", "/api/v1/runs/r/log", Req { token: p["token"].as_str(), no_origin: true, ..Default::default() }, json!({ "x": 1 })).await.0, StatusCode::OK);
    // a second session, revoked from the first
    email_sign_in(&t, "me@x.org", None).await;
    let (_, _, me) = call(&t, "GET", "/api/v1/me", as_cookie(Some(&cookie)), Value::Null).await;
    let sessions = me["sessions"].as_array().unwrap();
    assert_eq!(sessions.len(), 2, "{me}");
    let other = sessions.iter().find(|s| s["current"] == false).unwrap()["id"].as_str().unwrap().to_string();
    assert_eq!(call(&t, "DELETE", &format!("/api/v1/me/sessions/{other}"), as_cookie(Some(&cookie)), Value::Null).await.0, StatusCode::NO_CONTENT);
    // logout clears the cookie and the session
    let (c, h, _) = call(&t, "POST", "/api/v1/auth/logout", as_cookie(Some(&cookie)), Value::Null).await;
    assert_eq!(c, StatusCode::NO_CONTENT);
    let sc = h.get("set-cookie").unwrap().to_str().unwrap();
    assert!(sc.starts_with("mg_session=;") && sc.contains("Max-Age=0"), "{sc}");
    let (c, _, e) = call(&t, "GET", "/api/v1/me", as_cookie(Some(&cookie)), Value::Null).await;
    assert_eq!((c, e["code"].as_str()), (StatusCode::UNAUTHORIZED, Some("signed_out")));
}

#[tokio::test]
async fn methods_passkeys_and_app_paths() {
    let t = app("methods");
    let (c, _, m) = call(&t, "GET", "/api/v1/auth/methods", Req::default(), Value::Null).await;
    assert_eq!(c, StatusCode::OK);
    assert_eq!((m["email"].as_bool(), m["zafu"].as_bool(), m["passkey"].as_bool()), (Some(true), Some(true), Some(true)));
    assert!(m["ssh"].is_null() && m["account"].is_null());
    let (cookie, id) = signed_in(&t, "me@x.org").await;
    let (_, h, m) = call(&t, "GET", "/api/v1/auth/methods", as_cookie(Some(&cookie)), Value::Null).await;
    assert_eq!(m["account"]["id"].as_str(), Some(id.as_str()));
    let sc = h.get("set-cookie").unwrap().to_str().unwrap();
    assert!(sc.contains("HttpOnly") && sc.contains("SameSite=Lax") && sc.contains("Path=/") && sc.contains("Max-Age=2592000") && !sc.contains("Secure"), "{sc}");
    // passkey options: the bare WebAuthn options objects
    let (c, _, o) = call(&t, "POST", "/api/v1/auth/passkey/register/options", as_cookie(Some(&cookie)), Value::Null).await;
    assert_eq!(c, StatusCode::OK);
    assert_eq!((o["rp"]["id"].as_str(), o["authenticatorSelection"]["residentKey"].as_str(), o["attestation"].as_str()), (Some("obs.test"), Some("required"), Some("none")));
    assert_eq!(o["pubKeyCredParams"].as_array().unwrap().iter().map(|p| p["alg"].as_i64().unwrap()).collect::<Vec<_>>(), vec![-8, -7, -257]);
    let (_, _, l) = call(&t, "POST", "/api/v1/auth/passkey/login/options", Req::default(), Value::Null).await;
    assert_eq!((l["rpId"].as_str(), l["allowCredentials"].as_array().map(Vec::len)), (Some("obs.test"), Some(0)));
    // verification is not in this build: 501 after the challenge checks
    use base64::Engine;
    let b64 = |b: &[u8]| base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(b);
    let cdj = json!({ "type": "webauthn.create", "challenge": o["challenge"], "origin": ORIGIN }).to_string();
    let reg = json!({ "id": "x", "rawId": "x", "response": { "clientDataJSON": b64(cdj.as_bytes()), "attestationObject": b64(b"att") } });
    let (c, _, e) = call(&t, "POST", "/api/v1/auth/passkey/register/verify", as_cookie(Some(&cookie)), reg.clone()).await;
    assert_eq!((c, e["code"].as_str()), (StatusCode::NOT_IMPLEMENTED, Some("passkey_unsupported")));
    let (c, _, e) = call(&t, "POST", "/api/v1/auth/passkey/register/verify", as_cookie(Some(&cookie)), reg).await;
    assert_eq!((c, e["code"].as_str()), (StatusCode::UNAUTHORIZED, Some("bad_token")), "challenges are single use");
    // the app answers at /login and /account
    for p in ["/login", "/account"] {
        let (c, _, body) = call(&t, "GET", p, Req::default(), Value::Null).await;
        assert_eq!((c, body.as_str()), (StatusCode::OK, Some("<html>dash</html>")), "{p}");
    }
}

#[tokio::test]
async fn delete_account_removes_its_projects() {
    let t = app("delete");
    let (cookie, id) = signed_in(&t, "me@x.org").await;
    call(&t, "POST", "/api/v1/me/projects", as_cookie(Some(&cookie)), json!({ "name": "gone-soon" })).await;
    let (_, _, tk) = call(&t, "POST", "/api/v1/me/tokens", as_cookie(Some(&cookie)), json!({ "project": "gone-soon" })).await;
    let token = tk["token"].as_str().unwrap().to_string();
    call(&t, "POST", "/api/v1/runs/r/log", Req { token: Some(&token), ..Default::default() }, json!({ "loss": 1 })).await;
    assert!(t.root.join("runs/gone-soon/r").is_dir());
    let (c, _, e) = call(&t, "DELETE", "/api/v1/me", as_cookie(Some(&cookie)), json!({ "confirm": "acct_wrong" })).await;
    assert_eq!((c, e["code"].as_str()), (StatusCode::BAD_REQUEST, Some("bad_confirm")));
    let (c, _, _) = call(&t, "DELETE", "/api/v1/me", as_cookie(Some(&cookie)), json!({ "confirm": id })).await;
    assert_eq!(c, StatusCode::NO_CONTENT);
    assert!(!t.root.join("runs/gone-soon").exists());
    assert_eq!(call(&t, "GET", "/api/v1/me", as_cookie(Some(&cookie)), Value::Null).await.0, StatusCode::UNAUTHORIZED);
    assert_eq!(call(&t, "POST", "/api/v1/runs/r/log", Req { token: Some(&token), ..Default::default() }, json!({ "loss": 1 })).await.0, StatusCode::UNAUTHORIZED);
    assert!(call(&t, "GET", "/api/index", Req::default(), Value::Null).await.2["runs"].as_array().unwrap().is_empty());
}
