//! `modgrad run` / `push` transport: HTTP/3 to a real in-process server
//! (TCP + QUIC on loopback, self-signed), and the fallback to TCP when
//! UDP gets no answer. Ports stay in 18952-18956.
#![cfg(all(unix, feature = "h3"))]

use modgrad_obs_server::auth::{Auth, AuthConfig};
use modgrad_obs_server::http::{AppState, router};
use modgrad_obs_server::{IngestConfig, h3, open};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{Duration, Instant};

fn tmp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("modgrad-h3-cli-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A dev server on TCP `port`; with `h3 = Some((udp, advertised))` also
/// HTTP/3 on that UDP port and `Alt-Svc` naming `advertised`. Writes the
/// certificate to `<root>/cert.pem`. Runs for the rest of the test process.
fn serve(root: &Path, port: u16, h3: Option<(u16, u16)>) {
    let root = root.to_path_buf();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        tokio::runtime::Runtime::new().unwrap().block_on(async move {
            let (ingest, read) = open(&root.join("obs"), IngestConfig::default()).unwrap();
            let auth = Auth::open(&root.join("obs"), AuthConfig { dev: true, ..Default::default() }).unwrap();
            let mut app = router(AppState::new(ingest, read, None, auth));
            if let Some((udp, advertised)) = h3 {
                let (tls, pem) = h3::Tls::self_signed(vec!["127.0.0.1".into()]).unwrap();
                std::fs::write(root.join("cert.pem"), pem).unwrap();
                tokio::spawn(h3::serve(h3::endpoint(([127, 0, 0, 1], udp).into(), tls).unwrap(), app.clone()));
                app = h3::advertise(app, advertised);
            }
            let lis = tokio::net::TcpListener::bind(("127.0.0.1", port)).await.unwrap();
            tx.send(()).unwrap();
            axum::serve(lis, app).await.unwrap();
        });
    });
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
}

fn modgrad(cwd: &Path, args: &[&str]) -> (Output, Duration) {
    let mut c = Command::new(env!("CARGO_BIN_EXE_modgrad"));
    c.current_dir(cwd).args(args);
    for (k, _) in std::env::vars() {
        if k.starts_with("MODGRAD_") || k.starts_with("OBS_") {
            c.env_remove(k);
        }
    }
    c.env("XDG_CACHE_HOME", cwd.join(".cache"));
    let t0 = Instant::now();
    let o = c.output().unwrap();
    (o, t0.elapsed())
}

/// The `--json` events on stderr.
fn events(o: &Output) -> Vec<Value> {
    String::from_utf8_lossy(&o.stderr).lines().filter_map(|l| serde_json::from_str::<Value>(l).ok()).filter(|v| v["event"].is_string()).collect()
}

fn transport(o: &Output) -> Value {
    let t: Vec<Value> = events(o).into_iter().filter(|e| e["event"] == "transport").collect();
    assert_eq!(t.len(), 1, "one transport line: {}", String::from_utf8_lossy(&o.stderr));
    t[0].clone()
}

/// `loss` history of a run on the server, read over plain TCP.
fn losses(port: u16, run: &str) -> Vec<f64> {
    let r = modgrad_obs::net::request("GET", &format!("http://127.0.0.1:{port}/api/v1/runs/{run}/history?keys=loss"), None, None, "application/json", Duration::from_secs(5)).unwrap();
    r.json()["columns"]["loss"].as_array().map(|a| a.iter().filter_map(Value::as_f64).collect()).unwrap_or_default()
}

const SCRIPT: &str = "i=0; while [ $i -lt 300 ]; do echo \"step $i loss $i.5\"; i=$((i+1)); done";

#[test]
fn run_pushes_over_h3() {
    let d = tmp("run");
    serve(&d, 18952, Some((18952, 18952)));
    let ca = d.join("cert.pem");
    let (o, _) = modgrad(&d, &["run", "--server", "http://127.0.0.1:18952", "--h3", "--tls-ca", ca.to_str().unwrap(), "--name", "h3run", "--sys-interval", "0", "--json", "--", "sh", "-c", SCRIPT]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let t = transport(&o);
    assert_eq!((t["transport"].as_str(), t["fallback"].is_null()), (Some("h3"), true), "{t}");
    let l = losses(18952, "h3run");
    assert_eq!(l.len(), 300);
    assert_eq!(l[299], 299.5);
    assert!(events(&o).iter().any(|e| e["event"] == "end" && e["synced"] == true));
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn closed_udp_port_falls_back_to_tcp_quickly() {
    let d = tmp("closed");
    // TCP only: nothing listens on UDP 18953
    serve(&d, 18953, None);
    let (o, took) = modgrad(&d, &["run", "--server", "http://127.0.0.1:18953", "--h3", "--name", "fb", "--sys-interval", "0", "--json", "--", "sh", "-c", SCRIPT]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let t = transport(&o);
    assert_eq!(t["transport"], "http", "{t}");
    assert!(t["fallback"].as_str().unwrap().contains("udp 127.0.0.1:18953"), "{t}");
    assert_eq!(losses(18953, "fb").len(), 300);
    // the handshake timeout (1.5 s) plus a few local requests
    assert!(took < Duration::from_secs(5), "took {took:?}");
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn black_holed_advertised_port_falls_back_to_tcp_quickly() {
    let d = tmp("blackhole");
    // a firewall that drops UDP: the advertised port swallows packets, never answers
    let _hole = std::net::UdpSocket::bind("127.0.0.1:18955").unwrap();
    serve(&d, 18954, Some((18954, 18955)));
    let ca = d.join("cert.pem");
    let (o, took) = modgrad(&d, &["run", "--server", "http://127.0.0.1:18954", "--h3", "--tls-ca", ca.to_str().unwrap(), "--name", "bh", "--sys-interval", "0", "--json", "--", "sh", "-c", SCRIPT]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let t = transport(&o);
    // the Alt-Svc port was tried (not the URL's), timed out, and TCP took over
    let why = t["fallback"].as_str().unwrap();
    assert!(t["transport"] == "http" && why.contains("udp 127.0.0.1:18955") && why.contains("1.5s"), "{t}");
    assert_eq!(losses(18954, "bh").len(), 300);
    assert!(took < Duration::from_secs(5), "took {took:?}");
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn spool_from_a_down_server_is_pushed_over_h3() {
    let d = tmp("spool");
    let spool = d.join("spool");
    // nothing on 18956 yet: the run spools everything
    let (o, _) = modgrad(&d, &["run", "--server", "http://127.0.0.1:18956", "--h3", "--spool", spool.to_str().unwrap(), "--name", "late", "--sys-interval", "0", "--json", "--", "sh", "-c", SCRIPT]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert!(events(&o).iter().any(|e| e["event"] == "unsent"), "{}", String::from_utf8_lossy(&o.stderr));
    serve(&d, 18956, Some((18956, 18956)));
    let ca = d.join("cert.pem");
    let (o, _) = modgrad(&d, &["push", spool.join("late").to_str().unwrap(), "--h3", "--tls-ca", ca.to_str().unwrap(), "--json"]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert_eq!(transport(&o)["transport"], "h3");
    assert!(events(&o).iter().any(|e| e["event"] == "push" && e["synced"] == true));
    let l = losses(18956, "late");
    assert_eq!(l.len(), 300);
    assert!(!spool.join("late").exists(), "a fully sent spool is removed");
    let _ = std::fs::remove_dir_all(&d);
}
