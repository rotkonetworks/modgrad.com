//! `modgrad mcp --url`: the MCP tools with a `RemoteStore` against a real
//! modgrad-server on a loopback port. The store is blocking (std TcpStream), so
//! every tool call runs in `spawn_blocking` on a multi-thread runtime.

use modgrad_obs::mcp::Server;
use modgrad_obs::mcp::store::RemoteStore;
use modgrad_obs_server::auth::{Auth, AuthConfig};
use modgrad_obs_server::http::{AppState, router};
use modgrad_obs_server::{IngestConfig, open};
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::Arc;

fn tmp(tag: &str) -> PathBuf {
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let d = std::env::temp_dir().join(format!("obs-mcp-remote-{tag}-{}-{}", std::process::id(), N.fetch_add(1, std::sync::atomic::Ordering::SeqCst)));
    let _ = std::fs::remove_dir_all(&d);
    d
}

/// Serve a fresh store on 127.0.0.1:<free port>; returns (base URL, root, auth).
async fn serve(tag: &str, dev: bool) -> (String, PathBuf, Auth) {
    let root = tmp(tag);
    let (ingest, read) = open(&root, IngestConfig::default()).unwrap();
    let auth = Auth::open(&root, AuthConfig { dev, ..Default::default() }).unwrap();
    let app = router(AppState::new(ingest, read, None, auth.clone()).with_boards(root.join("boards")));
    let lis = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = lis.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(lis, app).await.unwrap() });
    (format!("http://{addr}"), root, auth)
}

async fn call(s: &Arc<Server>, tool: &str, args: Value) -> (String, bool) {
    let s = s.clone();
    let tool = tool.to_string();
    tokio::task::spawn_blocking(move || s.call_tool(&tool, &args)).await.unwrap()
}

async fn ok(s: &Arc<Server>, tool: &str, args: Value) -> String {
    let (t, e) = call(s, tool, args.clone()).await;
    assert!(!e, "{tool}({args}) failed: {t}");
    t
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dev_server_end_to_end() {
    let (url, root, _) = serve("dev", true).await;
    let s = Arc::new(Server::new(Box::new(RemoteStore::new(&url, None))));

    let t = ok(&s, "list_runs", json!({})).await;
    assert!(t.contains("No runs yet") && t.contains(&url), "{t}");

    let t = ok(&s, "log", json!({"run": "g/small", "metrics": {"loss": 2.5, "acc": "40%"}, "step": 0})).await;
    assert!(t.contains("Created new run g/small"), "{t}");
    let t = ok(&s, "log", json!({"run": "g/small", "metrics": {"loss": 2.4}})).await;
    assert!(t.contains("at step 1") && !t.contains("Created"), "{t}");

    let text: String = (2..40).map(|i| format!("step {} loss={:.4} lr=3e-4\n", i, 2.4 - 0.02 * i as f64)).collect();
    let t = ok(&s, "ingest_log", json!({"run": "g/small", "text": text})).await;
    assert!(t.contains("Ingested 38 rows"), "{t}");
    let text: String = (0..30).map(|i| format!("step {} loss={:.4}\n", i, 2.6 - 0.015 * i as f64)).collect();
    ok(&s, "ingest_log", json!({"run": "g/mid", "text": text})).await;

    let t = ok(&s, "list_runs", json!({})).await;
    assert!(t.starts_with("2 runs") && t.contains("g/small") && t.contains("g/mid"), "{t}");
    let t = ok(&s, "read_metric", json!({"run": "small", "metric": "LOSS"})).await;
    assert!(t.contains("last 1.62 at step 39"), "{t}");
    let t = ok(&s, "read_metric", json!({"run": "g/small", "metric": "accuracy"})).await;
    assert!(t.contains("40"), "{t}");
    let t = ok(&s, "summarize", json!({"run": "g/small"})).await;
    assert!(t.contains("improving"), "{t}");
    let t = ok(&s, "compare", json!({"runs": "g/*", "metric": "loss"})).await;
    assert!(t.contains("g/small") && t.contains("g/mid"), "{t}");
    let (t, e) = call(&s, "read_metric", json!({"run": "g/nope", "metric": "loss"})).await;
    assert!(e && t.contains("Closest"), "{t}");

    let t = ok(&s, "make_board", json!({"query": "loss{run=~\"g/small|g/mid\"}"})).await;
    let board_url = t.lines().find_map(|l| l.strip_prefix("URL: ")).unwrap().to_string();
    assert!(board_url.starts_with(&format!("{url}/#board=loss_")), "{t}");
    let id = board_url.split("#board=").nth(1).unwrap();
    let saved: Value = serde_json::from_slice(&std::fs::read(root.join("boards").join(format!("{id}.json"))).unwrap()).unwrap();
    assert_eq!(saved["panels"][0]["runs"], json!(["g/small", "g/mid"]));
    let r = modgrad_obs::net::request("GET", &format!("{url}/api/v1/boards/{id}"), None, None, "", std::time::Duration::from_secs(5)).unwrap();
    assert_eq!(r.json()["panels"][0]["metric"], "loss");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn project_token_scopes_writes_reads_and_boards() {
    let (url, root, auth) = serve("tok", false).await;
    let tok = auth.add_project_token("team").unwrap();
    let other = auth.add_project_token("other").unwrap();
    let s = Arc::new(Server::new(Box::new(RemoteStore::new(&url, Some(tok)))));
    let o = Arc::new(Server::new(Box::new(RemoteStore::new(&url, Some(other)))));

    let t = ok(&s, "log", json!({"run": "demo1", "metrics": {"loss": 2.0}, "step": 5})).await;
    assert!(t.contains("run team/demo1") && t.contains("Created"), "{t}");
    // a second write without a step continues the same run
    let t = ok(&s, "log", json!({"run": "demo1", "metrics": {"loss": 1.9}})).await;
    assert!(t.contains("team/demo1 at step 6") && !t.contains("Created"), "{t}");
    ok(&o, "log", json!({"run": "theirs", "metrics": {"loss": 9.0}, "step": 1})).await;

    let t = ok(&s, "list_runs", json!({})).await;
    assert!(t.starts_with("1 run") && t.contains("team/demo1") && !t.contains("theirs"), "{t}");
    let t = ok(&s, "read_metric", json!({"run": "demo1", "metric": "loss"})).await;
    assert!(t.contains("last 1.9 at step 6"), "{t}");

    let t = ok(&s, "make_board", json!({"query": "loss{run=~\"demo1\"}"})).await;
    assert!(t.contains("#board=team/loss_"), "{t}");
    let n = std::fs::read_dir(root.join("boards").join("team")).unwrap().count();
    assert_eq!(n, 1);

    // without a token a non-dev server refuses writes and says how to fix it
    let anon = Arc::new(Server::new(Box::new(RemoteStore::new(&url, None))));
    let (t, e) = call(&anon, "log", json!({"run": "x/y", "metrics": {"loss": 1.0}})).await;
    assert!(e && t.contains("--token") && t.contains("MODGRAD_TOKEN"), "{t}");
}

/// `modgrad-server --mcp`: POST /mcp runs the same tools server-side. The
/// request goes in through the router with a public Host header; the
/// tools reach the store over the server's real loopback socket.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hosted_mcp_endpoint() {
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use tower::ServiceExt;
    let root = tmp("hosted");
    let (ingest, read) = open(&root, IngestConfig::default()).unwrap();
    let auth = Auth::open(&root, AuthConfig { dev: false, ..Default::default() }).unwrap();
    let tok = auth.add_project_token("team").unwrap();
    let lis = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", lis.local_addr().unwrap());
    let app = router(AppState::new(ingest, read, None, auth).with_boards(root.join("boards")).with_mcp(url.clone()));
    let served = app.clone();
    tokio::spawn(async move { axum::serve(lis, served).await.unwrap() });

    let rpc = |body: Value, token: Option<&str>| {
        let mut b = Request::builder().method("POST").uri("/mcp").header("host", "obs.example.org").header("accept", "application/json, text/event-stream");
        if let Some(t) = token {
            b = b.header("authorization", format!("Bearer {t}"));
        }
        let app = app.clone();
        let req = b.body(Body::from(body.to_string())).unwrap();
        async move {
            let r = app.oneshot(req).await.unwrap();
            let code = r.status();
            let bytes = to_bytes(r.into_body(), 1 << 24).await.unwrap();
            (code, serde_json::from_slice::<Value>(&bytes).unwrap_or(Value::Null))
        }
    };
    let (c, v) = rpc(json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}}), None).await;
    assert_eq!(c, 200);
    assert_eq!(v["result"]["serverInfo"]["name"], "modgrad");
    let (c, _) = rpc(json!({"jsonrpc": "2.0", "method": "notifications/initialized"}), None).await;
    assert_eq!(c, 202);
    let (_, v) = rpc(json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}), None).await;
    assert_eq!(v["result"]["tools"].as_array().unwrap().len(), 7);
    let call = |id: i64, name: &str, args: Value| json!({"jsonrpc": "2.0", "id": id, "method": "tools/call", "params": {"name": name, "arguments": args}});
    let (_, v) = rpc(call(3, "log", json!({"run": "demo1", "metrics": {"loss": 2.0}, "step": 1})), Some(&tok)).await;
    assert_eq!(v["result"]["isError"], false, "{v}");
    assert!(v["result"]["content"][0]["text"].as_str().unwrap().contains("team/demo1"), "{v}");
    let (_, v) = rpc(call(4, "make_board", json!({"query": "loss{run=~\"demo1\"}"})), Some(&tok)).await;
    let t = v["result"]["content"][0]["text"].as_str().unwrap();
    assert!(t.contains("URL: http://obs.example.org/#board=team/loss_") && !t.contains(&url), "{t}");
    // without a token the write is refused, with the fix in the message
    let (_, v) = rpc(call(5, "log", json!({"run": "x/y", "metrics": {"loss": 1.0}})), None).await;
    assert_eq!(v["result"]["isError"], true);
}
