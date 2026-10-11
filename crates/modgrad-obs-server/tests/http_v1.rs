//! v1 HTTP API tests: the axum `Router` is a tower `Service<Request>`, so
//! every request goes through `oneshot`; no sockets.

use axum::Router;
use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use futures_util::StreamExt;
use modgrad_obs_server::auth::{Auth, AuthConfig};
use modgrad_obs_server::http::{AppState, router};
use modgrad_obs_server::{IngestConfig, open};
use serde_json::{Value, json};
use std::path::PathBuf;
use std::time::Duration;
use tower::ServiceExt;

fn tmp(tag: &str) -> PathBuf {
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let d = std::env::temp_dir().join(format!("obs-http-{tag}-{}-{}", std::process::id(), N.fetch_add(1, std::sync::atomic::Ordering::SeqCst)));
    let _ = std::fs::remove_dir_all(&d);
    d
}

fn app(tag: &str, dev: bool) -> (Router, PathBuf) {
    let root = tmp(tag);
    let (ingest, read) = open(&root, IngestConfig::default()).unwrap();
    std::fs::write(root.join("dashboard.html"), "<html>dash</html>").unwrap();
    let auth = Auth::open(&root, AuthConfig { dev, ..Default::default() }).unwrap();
    (router(AppState::new(ingest, read, Some(root.clone()), auth)), root)
}

async fn send(app: &Router, method: &str, uri: &str, token: Option<&str>, body: &str) -> (StatusCode, Value) {
    let mut b = Request::builder().method(method).uri(uri).header("host", "obs.test");
    if let Some(t) = token {
        b = b.header("authorization", format!("Bearer {t}"));
    }
    // no content-type on purpose: `curl -d` sends form-urlencoded
    let resp = app.clone().oneshot(b.body(Body::from(body.to_string())).unwrap()).await.unwrap();
    let code = resp.status();
    let bytes = to_bytes(resp.into_body(), 1 << 24).await.unwrap();
    (code, serde_json::from_slice(&bytes).unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into())))
}

async fn history(app: &Router, run: &str) -> Value {
    let (c, v) = send(app, "GET", &format!("/api/v1/runs/{run}/history"), None, "").await;
    assert_eq!(c, StatusCode::OK, "{v}");
    v
}

#[tokio::test]
async fn dev_mode_log_auto_creates_and_auto_steps() {
    let (app, _) = app("auto", true);
    let (c, v) = send(&app, "POST", "/api/v1/runs/exp/a/log", None, r#"{"loss": "3.5", "acc": "7.5%"}"#).await;
    assert_eq!(c, StatusCode::OK, "{v}");
    assert_eq!(v["run"], "exp/a");
    assert_eq!(v["created"], true);
    assert_eq!(v["dashboard"], "http://obs.test/#run=exp%2Fa");
    send(&app, "POST", "/api/v1/runs/exp/a/log", None, r#"[{"loss": 3.0}, {"loss": 2.5, "train": {"lr": 0.1}}]"#).await;
    send(&app, "POST", "/api/v1/runs/exp%2Fa/log", None, "{\"step\": 10, \"loss\": 1}\n{\"loss\": 0.5}\n").await;
    let h = history(&app, "exp/a").await;
    assert_eq!(h["steps"], json!([0, 1, 2, 10, 11]));
    assert_eq!(h["columns"]["loss"], json!([3.5, 3.0, 2.5, 1.0, 0.5]));
    assert_eq!(h["columns"]["train/lr"][2], json!(0.1));
    assert_eq!(h["columns"]["acc"][0], json!(7.5));
}

#[tokio::test]
async fn resending_a_row_is_last_write_wins() {
    let (app, _) = app("idem", true);
    for v in [1.0, 1.0, 2.0] {
        send(&app, "POST", "/api/v1/runs/r/log", None, &json!({"step": 5, "loss": v}).to_string()).await;
    }
    let h = history(&app, "r").await;
    assert_eq!((h["steps"].clone(), h["columns"]["loss"].clone()), (json!([5]), json!([2.0])));
    // "next" after a re-send of an old step is still one past the max
    send(&app, "POST", "/api/v1/runs/r/log", None, r#"{"step": 2, "loss": 9}"#).await;
    send(&app, "POST", "/api/v1/runs/r/log", None, r#"{"loss": 0}"#).await;
    assert_eq!(history(&app, "r").await["steps"], json!([2, 5, 6]));
}

#[tokio::test]
async fn anonymous_project_flow() {
    let (app, _) = app("anon", false);
    let (c, p) = send(&app, "POST", "/api/v1/projects", None, "").await;
    assert_eq!(c, StatusCode::CREATED, "{p}");
    let (project, token) = (p["project"].as_str().unwrap().to_string(), p["token"].as_str().unwrap().to_string());
    assert!(project.starts_with("anon-") && p["board"].as_str().unwrap().ends_with(&format!("#project={project}")));
    let (c, v) = send(&app, "POST", "/api/v1/runs/my-run/log", Some(&token), r#"{"loss": 0.5}"#).await;
    assert_eq!(c, StatusCode::OK, "{v}");
    assert_eq!(v["run"], format!("{project}/my-run"));
    // hidden from the public index, visible by project
    let (_, idx) = send(&app, "GET", "/api/index", None, "").await;
    assert_eq!(idx["runs"].as_array().unwrap().len(), 0);
    let (_, idx) = send(&app, "GET", &format!("/api/index?project={project}"), None, "").await;
    assert_eq!(idx["runs"][0]["id"], format!("{project}/my-run"));
    // the token does not reach other projects
    let (c, e) = send(&app, "POST", "/api/v1/runs/other/x/log", Some(&token), r#"{"loss": 1}"#).await;
    assert_eq!(c, StatusCode::FORBIDDEN, "{e}");
    let claim = p["claim"].as_str().unwrap().trim_start_matches("http://obs.test");
    let (c, cl) = send(&app, "GET", claim, None, "").await;
    assert_eq!((c, cl["project"].as_str()), (StatusCode::OK, Some(project.as_str())));
}

#[tokio::test]
async fn errors_teach_and_auth_is_enforced() {
    let (app, _) = app("auth", false);
    let (c, e) = send(&app, "POST", "/api/v1/runs/p/r/log", None, r#"{"loss": 1}"#).await;
    assert_eq!(c, StatusCode::UNAUTHORIZED);
    for k in ["error", "example", "help", "message"] {
        assert!(e[k].is_string(), "{k} missing: {e}");
    }
    assert!(e["example"].as_str().unwrap().contains("/api/v1/projects"));
    let (_, p) = send(&app, "POST", "/api/v1/projects", None, "").await;
    let tok = p["token"].as_str().unwrap();
    let (c, e) = send(&app, "POST", "/api/v1/runs/r/log", Some(tok), r#"{"step": -3, "loss": 1}"#).await;
    assert_eq!(c, StatusCode::BAD_REQUEST);
    assert!(e["error"].as_str().unwrap().contains("non-negative") && e["example"].as_str().unwrap().contains("curl"), "{e}");
    let (c, e) = send(&app, "POST", "/api/v1/runs/r/log", Some(tok), "{not json").await;
    assert_eq!(c, StatusCode::BAD_REQUEST);
    assert!(e["error"].as_str().unwrap().contains("/text"), "{e}");
    let (c, v) = send(&app, "POST", "/api/v1/runs/r/log", Some(tok), r#"{"loss": 1, "note": "hi"}"#).await;
    assert_eq!(c, StatusCode::OK);
    assert_eq!(v["ignored"], json!(["note"]));
    let (c, e) = send(&app, "POST", "/api/v1/runs/r/zap", Some(tok), "{}").await;
    assert_eq!(c, StatusCode::NOT_FOUND);
    assert!(e["error"].as_str().unwrap().contains(".../log"), "{e}");
    // legacy tokenless write routes are not mounted outside dev mode
    let (c, _) = send(&app, "POST", "/api/runs", None, r#"{"name": "x"}"#).await;
    assert!(c == StatusCode::NOT_FOUND || c == StatusCode::METHOD_NOT_ALLOWED, "{c}");
    // the token store is never served
    let (c, _) = send(&app, "GET", "/.obs-auth.json", None, "").await;
    assert_eq!(c, StatusCode::NOT_FOUND);
    let (c, t) = send(&app, "GET", "/api/v1", None, "").await;
    assert_eq!(c, StatusCode::OK);
    assert!(t.as_str().unwrap().contains("POST /api/v1/runs/<id>/log"));
}

#[tokio::test]
async fn explicit_create_mints_a_run_token() {
    let (app, _) = app("create", true);
    let (c, v) = send(&app, "POST", "/api/v1/runs", None, r#"{"name": "a", "project": "p", "config": {"lr": 0.1}, "meta": {"git": "abc123", "hostname": "box"}}"#).await;
    assert_eq!(c, StatusCode::CREATED, "{v}");
    assert_eq!(v["run"], "p/a");
    let rt = v["token"].as_str().unwrap().to_string();
    // protected now: tokenless writes conflict even in dev mode, the token works
    let (c, _) = send(&app, "POST", "/api/v1/runs/p/a/log", None, r#"{"x": 1}"#).await;
    assert_eq!(c, StatusCode::CONFLICT);
    let (c, _) = send(&app, "POST", "/api/v1/runs/p/a/log", Some(&rt), r#"{"x": 1}"#).await;
    assert_eq!(c, StatusCode::OK);
    let (c, v) = send(&app, "POST", "/api/v1/runs", Some(&rt), r#"{"name": "p/a"}"#).await;
    assert_eq!((c, v["created"].clone(), v.get("token").is_none()), (StatusCode::OK, json!(false), true));
    let (_, cfg) = send(&app, "GET", "/api/v1/runs/p/a/config", None, "").await;
    assert_eq!((cfg["config"]["lr"].clone(), cfg["meta"]["git"].clone(), cfg["meta"]["hostname"].clone()), (json!(0.1), json!("abc123"), json!("box")));
    // envelope: config merge, events, summary, finish
    let (c, v) = send(&app, "POST", "/api/v1/runs/p/a/log", Some(&rt), r#"{"rows": [{"x": 2}], "events": ["ckpt"], "config": {"bs": 8}, "finish": {"exit_code": 0}}"#).await;
    assert_eq!(c, StatusCode::OK, "{v}");
    let (_, cfg) = send(&app, "GET", "/api/v1/runs/p/a/config", None, "").await;
    assert_eq!((cfg["config"]["lr"].clone(), cfg["config"]["bs"].clone()), (json!(0.1), json!(8)));
    let (_, s) = send(&app, "GET", "/api/v1/runs/p/a/summary", None, "").await;
    assert_eq!((s["finished"].clone(), s["summary"]["exit_code"].clone()), (json!(true), json!(0)));
    let (_, ev) = send(&app, "GET", "/api/v1/runs/p/a/events", None, "").await;
    assert_eq!(ev[0]["text"], "ckpt");
}

#[tokio::test]
async fn text_ingest_parses_server_side() {
    let (app, _) = app("text", true);
    let log = "loading data\nstep 100: loss 2.5 acc 10%\nval_loss: 2.7\niter 200 | loss=2.1 | lr 3e-4\n";
    let (c, v) = send(&app, "POST", "/api/v1/runs/t/text?events=unparsed", None, log).await;
    assert_eq!(c, StatusCode::OK, "{v}");
    assert_eq!((v["lines"].clone(), v["rows"].clone(), v["unparsed"].clone()), (json!(4), json!(3), json!(1)));
    let h = history(&app, "t").await;
    assert_eq!(h["steps"], json!([100, 200]));
    assert_eq!(h["columns"]["val_loss"][0], json!(2.7));
    let (_, ev) = send(&app, "GET", "/api/v1/runs/t/events", None, "").await;
    assert_eq!(ev[0]["text"], "loading data");
}

/// Read SSE frames until one of `kind` arrives; returns its data.
async fn next_event(body: &mut (impl futures_util::Stream<Item = Result<axum::body::Bytes, axum::Error>> + Unpin), buf: &mut String, kind: &str) -> Value {
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            while let Some(i) = buf.find("\n\n") {
                let frame: String = buf.drain(..i + 2).collect();
                let ev = frame.lines().find_map(|l| l.strip_prefix("event: ")).unwrap_or("");
                let data = frame.lines().find_map(|l| l.strip_prefix("data: ")).unwrap_or("null");
                if ev == kind {
                    return serde_json::from_str(data).unwrap();
                }
            }
            let chunk = body.next().await.expect("stream ended").unwrap();
            buf.push_str(&String::from_utf8_lossy(&chunk));
        }
    })
    .await
    .unwrap_or_else(|_| panic!("no {kind} event within 5 s; buffer: {buf}"))
}

#[tokio::test]
async fn sse_pushes_new_runs_and_rows() {
    let (app, _) = app("sse", true);
    send(&app, "POST", "/api/v1/runs/s/old/log", None, r#"{"loss": 1}"#).await;
    let resp = app.clone().oneshot(Request::get("/api/v1/stream").body(Body::empty()).unwrap()).await.unwrap();
    assert_eq!(resp.headers()["content-type"], "text/event-stream");
    let mut body = resp.into_body().into_data_stream();
    let mut buf = String::new();
    let hello = next_event(&mut body, &mut buf, "hello").await;
    assert_eq!(hello["runs"][0]["run"], "s/old");
    assert!(hello["runs"][0]["updated_unix"].as_f64().unwrap() > 1e9);
    send(&app, "POST", "/api/v1/runs/s/new/log", None, r#"{"step": 0, "loss": 4}"#).await;
    let run = next_event(&mut body, &mut buf, "run").await;
    assert_eq!((run["run"].clone(), run["entry"]["id"].clone()), (json!("s/new"), json!("s/new")));
    send(&app, "POST", "/api/v1/runs/s/new/log", None, r#"[{"loss": 3}, {"loss": 2}]"#).await;
    // the create and the first append may be published separately: collect rows until step 2
    let mut steps: Vec<u64> = Vec::new();
    while !steps.contains(&2) {
        let rows = next_event(&mut body, &mut buf, "rows").await;
        assert_eq!(rows["run"], "s/new");
        assert!(rows["unix"].as_f64().unwrap() > 1e9);
        steps.extend(rows["rows"].as_array().unwrap().iter().map(|r| r["step"].as_u64().unwrap()));
    }
    assert!(steps.ends_with(&[1, 2]), "{steps:?}");
    send(&app, "POST", "/api/v1/runs/s/new/finish", None, r#"{"exit_code": 0}"#).await;
    let s = next_event(&mut body, &mut buf, "summary").await;
    assert_eq!((s["finished"].clone(), s["summary"]["summary"]["exit_code"].clone()), (json!(true), json!(0)));
}

#[tokio::test]
async fn sse_project_filter_hides_other_runs() {
    let (app, _) = app("ssef", true);
    let (_, p) = send(&app, "POST", "/api/v1/projects", None, "").await;
    let (project, tok) = (p["project"].as_str().unwrap().to_string(), p["token"].as_str().unwrap().to_string());
    let resp = app.clone().oneshot(Request::get(format!("/api/v1/stream?project={project}")).body(Body::empty()).unwrap()).await.unwrap();
    let mut body = resp.into_body().into_data_stream();
    let mut buf = String::new();
    next_event(&mut body, &mut buf, "hello").await;
    send(&app, "POST", "/api/v1/runs/public/x/log", None, r#"{"loss": 1}"#).await;
    send(&app, "POST", "/api/v1/runs/mine/log", Some(&tok), r#"{"loss": 1}"#).await;
    let run = next_event(&mut body, &mut buf, "run").await;
    assert_eq!(run["run"], format!("{project}/mine"), "the public run must not reach a project stream");
}

#[tokio::test]
async fn saved_boards_are_served_by_name_only() {
    let root = tmp("boards");
    let (ingest, read) = open(&root, IngestConfig::default()).unwrap();
    let auth = Auth::open(&root, AuthConfig::default()).unwrap();
    std::fs::create_dir_all(root.join("boards")).unwrap();
    std::fs::write(root.join("boards/loss-ab12.json"), r#"{"panels": [{"query": "loss{run=~\"a|b\"}"}]}"#).unwrap();
    std::fs::write(root.join("secret.json"), "{}").unwrap();
    let app = router(AppState::new(ingest, read, None, auth).with_boards(root.join("boards")));
    let (c, v) = send(&app, "GET", "/api/v1/boards/loss-ab12", None, "").await;
    assert_eq!(c, StatusCode::OK, "{v}");
    assert_eq!(v["panels"][0]["query"], "loss{run=~\"a|b\"}");
    for bad in ["/api/v1/boards/nope", "/api/v1/boards/..%2Fsecret", "/api/v1/boards/.hidden", "/api/v1/boards/a%20b"] {
        let (c, _) = send(&app, "GET", bad, None, "").await;
        assert_eq!(c, StatusCode::NOT_FOUND, "{bad}");
    }
}

#[tokio::test]
async fn samples_are_stored_and_served() {
    let (app, root) = app("samples", false);
    let (_, p) = send(&app, "POST", "/api/v1/projects", None, "").await;
    let tok = p["token"].as_str().unwrap().to_string();
    let proj = p["project"].as_str().unwrap().to_string();
    // no token: refused like any write
    let (c, _) = send(&app, "POST", "/api/v1/runs/rl/samples", None, r#"{"completion": "x"}"#).await;
    assert_eq!(c, StatusCode::UNAUTHORIZED);
    // the first write may be samples: the run is created; step defaults to the last history step (0)
    let (c, v) = send(&app, "POST", "/api/v1/runs/rl/samples", Some(&tok), r#"{"prompt": "2+2=", "completion": "5", "reward": 0}"#).await;
    assert_eq!(c, StatusCode::OK, "{v}");
    assert_eq!((v["samples"].as_u64(), v["created"].as_bool()), (Some(1), Some(true)));
    let run = v["run"].as_str().unwrap().to_string();
    send(&app, "POST", &format!("/api/v1/runs/{run}/log"), Some(&tok), r#"{"step": 10, "reward": 0.5}"#).await;
    let body = "{\"step\": 10, \"key\": \"eval\", \"prompt\": \"2+2=\", \"completion\": \"4\", \"reward\": 1, \"temperature\": 0.7}\n{\"prompt\": \"3+3=\", \"response\": \"6\"}\n";
    let (c, v) = send(&app, "POST", &format!("/api/v1/runs/{run}/samples"), Some(&tok), body).await;
    assert_eq!((c, v["samples"].as_u64()), (StatusCode::OK, Some(2)), "{v}");
    // a bad sample teaches
    let (c, v) = send(&app, "POST", &format!("/api/v1/runs/{run}/samples"), Some(&tok), r#"{"prompt": "no output"}"#).await;
    assert_eq!(c, StatusCode::BAD_REQUEST);
    assert!(v["error"].as_str().unwrap().contains("completion") && v["example"].as_str().unwrap().contains("/samples"), "{v}");
    // read back: all, by key, by step window, newest-limited
    let (c, v) = send(&app, "GET", &format!("/api/v1/runs/{run}/samples"), None, "").await;
    assert_eq!(c, StatusCode::OK, "{v}");
    assert_eq!((v["n"].as_u64(), v["steps"].clone(), v["keys"].clone()), (Some(3), json!([0, 10]), json!(["default", "eval"])));
    let s = v["samples"].as_array().unwrap();
    assert_eq!((s[1]["key"].as_str(), s[1]["meta"]["temperature"].as_f64(), s[2]["step"].as_u64(), s[2]["completion"].as_str()), (Some("eval"), Some(0.7), Some(10), Some("6")));
    let (_, v) = send(&app, "GET", &format!("/api/v1/runs/{run}/samples?key=eval"), None, "").await;
    assert_eq!(v["samples"].as_array().unwrap().len(), 1);
    let (_, v) = send(&app, "GET", &format!("/api/v1/runs/{run}/samples?from=5&limit=1"), None, "").await;
    assert_eq!(v["samples"].as_array().unwrap().iter().map(|x| x["completion"].clone()).collect::<Vec<_>>(), vec![json!("6")]);
    // the index advertises them; a run without samples has n_samples 0 and no file
    send(&app, "POST", "/api/v1/runs/plain/log", Some(&tok), r#"{"loss": 1}"#).await;
    let (_, idx) = send(&app, "GET", &format!("/api/index?project={proj}"), None, "").await;
    let n = |id: &str| idx["runs"].as_array().unwrap().iter().find(|r| r["id"] == id).map(|r| r["n_samples"].clone());
    assert_eq!(n(&run), Some(json!(3)));
    assert_eq!(n(&format!("{proj}/plain")), Some(json!(0)));
    assert!(!root.join("runs").join(&proj).join("plain").join("samples.jsonl").exists());
    let (c, _) = send(&app, "GET", "/api/v1/runs/nope/x/samples", None, "").await;
    assert_eq!(c, StatusCode::NOT_FOUND);
}
