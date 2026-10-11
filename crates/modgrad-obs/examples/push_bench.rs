//! Push N rows to a modgrad-server the way `modgrad run` does (batches of
//! up to 1000 rows through `net::push_request`) and report throughput and
//! per-POST latency. One JSON line on stdout.
//!
//! ```sh
//! modgrad-server --root /tmp/obs-bench --dev --listen 127.0.0.1:18950 --h3 127.0.0.1:18950 --h3-self-signed
//! cargo run --release -p modgrad-obs --example push_bench -- http://127.0.0.1:18950                # TCP (std TcpStream)
//! cargo run --release -p modgrad-obs --example push_bench -- http://127.0.0.1:18950 --h3 \
//!     --tls-ca /tmp/obs-bench/h3-self-signed.pem                                                   # HTTP/3
//! ```
//!
//! Flags: `--rows N` (default 100000), `--batch B` (default 1000), `--no-h3`
//! (stay on TCP even if the server advertises HTTP/3), `--gets N` (default
//! 500: small GETs afterwards, which skip the server's ingest batching, so
//! their latency is mostly transport).
//!
//! Note: the server groups writes (`BatchConfig`: 256 items or 20 ms), so a
//! POST of fewer than 256 rows waits up to 20 ms whatever the transport.

use modgrad_obs::net;
use serde_json::{Value, json};
use std::time::{Duration, Instant};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let flag = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1).cloned());
    let Some(base) = args.get(1).filter(|a| a.contains("://")).cloned() else {
        eprintln!("usage: push_bench <server url> [--h3 | --no-h3] [--tls-ca PEM] [--rows N] [--batch B]");
        std::process::exit(2)
    };
    let h3 = args.iter().any(|a| a == "--h3");
    let mode = if h3 { net::H3::Force } else if args.iter().any(|a| a == "--no-h3") { net::H3::Off } else { net::H3::Auto };
    if let Some(ca) = flag("--tls-ca") {
        net::set_tls_ca(ca.into());
    }
    let rows: u64 = flag("--rows").and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let batch: u64 = flag("--batch").and_then(|s| s.parse().ok()).unwrap_or(1000);
    let gets: usize = flag("--gets").and_then(|s| s.parse().ok()).unwrap_or(500);
    let timeout = Duration::from_secs(30);
    // no notice at all means TCP was never questioned
    let transport = std::cell::Cell::new(if base.starts_with("https") { "https" } else { "http" });
    let post = |path: &str, body: &Value, token: Option<&str>| {
        let (r, n) = net::push_request("POST", &format!("{base}{path}"), token, Some(body.to_string().as_bytes()), "application/json", timeout, mode);
        if let Some(n) = n {
            eprintln!("push_bench: {}", n.text);
            transport.set(n.transport);
        }
        r.expect("request")
    };

    // the create goes over TCP and decides the transport; not timed
    let name = format!("bench/{}-{}", if h3 { "h3" } else { base.split("://").next().unwrap_or("tcp") }, std::process::id());
    let r = post("/api/v1/runs", &json!({ "name": name }), None);
    assert!(r.ok(), "create: {}", r.error_text());
    let token = r.json()["token"].as_str().map(String::from);

    let mut lat = Vec::new();
    let t0 = Instant::now();
    let mut step = 0u64;
    while step < rows {
        let n = batch.min(rows - step);
        let body: Vec<Value> = (step..step + n)
            .map(|s| json!({ "step": s, "t": s as f64 * 0.01, "loss": 3.0 / (1.0 + s as f64 * 1e-4), "lr": 3e-4, "grad_norm": 1.0 + (s % 7) as f64 * 0.1, "tokens_seen": s * 4096 }))
            .collect();
        let t = Instant::now();
        let r = post(&format!("/api/v1/runs/{name}/log"), &json!({ "rows": body }), token.as_deref());
        lat.push(t.elapsed().as_secs_f64() * 1e3);
        assert!(r.ok(), "log: HTTP {} {}", r.status, r.error_text());
        step += n;
    }
    let total = t0.elapsed().as_secs_f64();
    lat.sort_by(f64::total_cmp);
    let pct = |p: f64| lat[((lat.len() - 1) as f64 * p).round() as usize];
    let mut glat = Vec::new();
    for _ in 0..gets {
        let t = Instant::now();
        let (r, _) = net::push_request("GET", &format!("{base}/api/v1/runs/{name}/summary"), None, None, "application/json", timeout, mode);
        glat.push(t.elapsed().as_secs_f64() * 1e3);
        assert!(r.expect("get").ok());
    }
    glat.sort_by(f64::total_cmp);
    let gpct = |p: f64| glat.get(((glat.len().max(1) - 1) as f64 * p).round() as usize).map_or(Value::Null, |x| json!((x * 1000.0).round() / 1000.0));
    // every row landed (read over plain TCP)
    let h = net::request("GET", &format!("{base}/api/v1/runs/{name}/history?keys=loss&stride=1"), None, None, "application/json", timeout).expect("history").json();
    let landed = h["steps"].as_array().map_or(0, |a| a.len());
    println!(
        "{}",
        json!({ "server": base, "transport": transport.get(), "rows": rows, "batch": batch, "posts": lat.len(), "total_s": (total * 1000.0).round() / 1000.0,
                "rows_per_s": (rows as f64 / total).round(), "p50_ms": (pct(0.5) * 100.0).round() / 100.0, "p90_ms": (pct(0.9) * 100.0).round() / 100.0,
                "p99_ms": (pct(0.99) * 100.0).round() / 100.0, "max_ms": (pct(1.0) * 100.0).round() / 100.0, "landed": landed,
                "gets": gets, "get_p50_ms": gpct(0.5), "get_p99_ms": gpct(0.99) })
    );
}
