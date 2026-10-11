//! HTTP/3 listener over real sockets: a self-signed server on loopback,
//! rows pushed with the CLI's client (`modgrad_obs::net::push_request`),
//! read back over HTTP/3 and over TCP. Ports stay in 18950-18952.
#![cfg(feature = "h3")]

use modgrad_obs::net;
use modgrad_obs_server::auth::{Auth, AuthConfig};
use modgrad_obs_server::http::{AppState, router};
use modgrad_obs_server::{IngestConfig, h3, open};
use serde_json::{Value, json};
use std::path::PathBuf;
use std::time::{Duration, Instant};

fn tmp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("obs-h3-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// One self-signed certificate for every server in this process (the
/// client trusts one extra CA per process), installed as that CA.
fn cert() -> &'static (String, String) {
    static C: std::sync::OnceLock<(String, String)> = std::sync::OnceLock::new();
    C.get_or_init(|| {
        let c = rcgen::generate_simple_self_signed(vec!["127.0.0.1".to_string()]).unwrap();
        let (cert, key) = (c.cert.pem(), c.key_pair.serialize_pem());
        let p = tmp("ca").join("cert.pem");
        std::fs::write(&p, &cert).unwrap();
        net::set_tls_ca(p);
        (cert, key)
    })
}

/// A dev server on TCP `port` and HTTP/3 on UDP `port`, on its own runtime
/// thread for the rest of the test process.
fn serve(root: PathBuf, port: u16) {
    let (cert_pem, key_pem) = cert();
    std::fs::write(root.join("cert.pem"), cert_pem).unwrap();
    std::fs::write(root.join("key.pem"), key_pem).unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async move {
            let (ingest, read) = open(&root, IngestConfig::default()).unwrap();
            let auth = Auth::open(&root, AuthConfig { dev: true, ..Default::default() }).unwrap();
            let app = router(AppState::new(ingest, read, None, auth));
            let tls = h3::Tls::from_pem(&root.join("cert.pem"), &root.join("key.pem")).unwrap();
            let ep = h3::endpoint(([127, 0, 0, 1], port).into(), tls).unwrap();
            tokio::spawn(h3::serve(ep, app.clone()));
            let lis = tokio::net::TcpListener::bind(("127.0.0.1", port)).await.unwrap();
            tx.send(()).unwrap();
            axum::serve(lis, h3::advertise(app, port)).await.unwrap();
        });
    });
    rx.recv_timeout(Duration::from_secs(10)).unwrap()
}

#[test]
fn rows_pushed_over_h3_land_and_read_back() {
    let root = tmp("push");
    serve(root.clone(), 18950);
    let base = "http://127.0.0.1:18950";

    // TCP answers advertise the UDP listener
    let r = net::request("GET", &format!("{base}/api/index"), None, None, "application/json", Duration::from_secs(5)).unwrap();
    assert_eq!(r.alt_svc.as_deref().and_then(net::alt_svc_h3_port), Some(18950), "{:?}", r.alt_svc);

    // first push goes over TCP and decides; --h3 forces the attempt on an http:// URL
    let post = |path: &str, body: &Value, token: Option<&str>| net::push_request("POST", &format!("{base}{path}"), token, Some(body.to_string().as_bytes()), "application/json", Duration::from_secs(10), true);
    let (r, notice) = post("/api/v1/runs", &json!({"name": "h3/a"}), None);
    let r = r.unwrap();
    assert!(r.ok());
    assert!(r.alt_svc.is_some(), "the first request goes over TCP");
    let token = r.json()["token"].as_str().map(String::from);
    let n = notice.expect("transport notice");
    assert_eq!(n.transport, "h3", "{}", n.text);

    // then over HTTP/3: 5 posts of 1000 rows
    for b in 0..5u64 {
        let rows: Vec<Value> = (0..1000u64).map(|i| json!({"step": b * 1000 + i, "loss": 1.0 / (1 + b * 1000 + i) as f64})).collect();
        let (r, notice) = post("/api/v1/runs/h3/a/log", &json!({"rows": rows}), token.as_deref());
        let r = r.unwrap();
        assert!(notice.is_none());
        assert_eq!(r.status, 200, "{}", r.error_text());
        assert!(r.alt_svc.is_none(), "an h3 response, not TCP");
        // links built from :authority and the client's scheme point at the TCP origin
        assert_eq!(r.json()["dashboard"], json!("http://127.0.0.1:18950/#run=h3%2Fa"));
    }

    // read back over HTTP/3 and over TCP
    let (r, _) = net::push_request("GET", &format!("{base}/api/v1/runs/h3/a/history?keys=loss"), None, None, "application/json", Duration::from_secs(10), true);
    let h = r.unwrap().json();
    assert_eq!(h["steps"].as_array().unwrap().len(), 5000);
    assert_eq!(h["columns"]["loss"][4999], json!(1.0 / 5000.0));
    let t = net::request("GET", &format!("{base}/api/v1/runs/h3/a/history?keys=loss"), None, None, "application/json", Duration::from_secs(10)).unwrap().json();
    assert_eq!(t, h);
    // a run's token is enforced over HTTP/3 too
    let (r, _) = post("/api/v1/runs/h3/a/log", &json!({"step": 1, "loss": 0}), None);
    assert_eq!(r.unwrap().status, 409);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn bodies_over_the_cap_are_refused() {
    let root = tmp("cap");
    serve(root.clone(), 18951);
    let base = "http://127.0.0.1:18951";
    let (r, n) = net::push_request("POST", &format!("{base}/api/v1/runs"), None, Some(br#"{"name":"cap"}"#), "application/json", Duration::from_secs(10), true);
    assert!(r.unwrap().ok());
    assert_eq!(n.unwrap().transport, "h3");
    let big = vec![b' '; h3::MAX_BODY + 1];
    let t0 = Instant::now();
    let (r, _) = net::push_request("POST", &format!("{base}/api/v1/runs/cap/log"), None, Some(&big), "application/json", Duration::from_secs(20), true);
    assert_eq!(r.unwrap().status, 413);
    assert!(t0.elapsed() < Duration::from_secs(20));
    let _ = std::fs::remove_dir_all(&root);
}
