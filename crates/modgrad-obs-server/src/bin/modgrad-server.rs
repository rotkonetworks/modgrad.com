//! ```text
//! modgrad-server [--root <obs dir>] [--listen 127.0.0.1:8790] [--static <dir> | --no-static]
//!            [--dev] [--no-anon] [--anon-expiry-days 30] [--mcp]
//!            [--h3 <udp addr> (--tls-cert <pem> --tls-key <pem> | --h3-self-signed) [--h3-advertise <port>]]
//! modgrad-server token add <project> [--root <obs dir>] [--json]
//! ```
//!
//! Serves the modgrad-obs store under `<root>/runs` over HTTP (the v1
//! ingest API, selection reads, SSE) and, by default, `<root>` itself as
//! static files so `/` opens dashboard.html in server mode (point `--static`
//! at obs/web/dist for the app). A root without dashboard.html (a project's
//! own `obs/`) gets the app from the checkout this binary was built in,
//! `obs/web/dist`, when that has been built. Saved boards under `<root>/boards` are served
//! at `/api/v1/boards/<name>` for the dashboard's `#board=<name>`.
//! `--dev` accepts writes without a token (local use only). `token add`
//! provisions a project token and prints it once. `--mcp` adds a hosted MCP
//! endpoint at `/mcp` for agents that cannot run `modgrad mcp` locally; its
//! tools analyse on this server's CPU (see `mcp_http.rs`). Logging:
//! `RUST_LOG=info`.
//!
//! `--h3 <udp addr>` (feature `h3`, on by default) also serves the same
//! API over HTTP/3 on that UDP address, and TCP responses then carry
//! `Alt-Svc: h3=":<port>"` (`--h3-advertise` overrides the port, e.g.
//! behind a port mapping). QUIC needs TLS: `--tls-cert`/`--tls-key` for
//! the public hostname, or `--h3-self-signed` for local testing, which
//! writes the certificate to `<root>/h3-self-signed.pem` for clients'
//! `--tls-ca`. The TCP listener stays plain HTTP (TLS there is the
//! reverse proxy's job).

use modgrad_obs_server::auth::{Auth, AuthConfig};
use modgrad_obs_server::{IngestConfig, http};
use std::path::PathBuf;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().with_env_filter(tracing_subscriber::EnvFilter::from_default_env()).init();
    let args: Vec<String> = std::env::args().collect();
    let flag = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1).cloned());
    let has = |k: &str| args.iter().any(|a| a == k);
    let root = PathBuf::from(flag("--root").unwrap_or_else(|| "obs".into()));
    let cfg = AuthConfig {
        dev: has("--dev"),
        anon_projects: !has("--no-anon"),
        anon_expiry_days: flag("--anon-expiry-days").and_then(|d| d.parse().ok()).unwrap_or(AuthConfig::default().anon_expiry_days),
    };
    if args.get(1).map(String::as_str) == Some("token") {
        let (Some("add"), Some(project)) = (args.get(2).map(String::as_str), args.get(3)) else {
            eprintln!("usage: modgrad-server token add <project> [--root <obs dir>] [--json]");
            std::process::exit(2)
        };
        std::fs::create_dir_all(&root).ok();
        let t = Auth::open(&root, cfg).and_then(|a| a.add_project_token(project)).unwrap_or_else(|e| {
            eprintln!("modgrad-server: token store under {}: {e}", root.display());
            std::process::exit(1)
        });
        if has("--json") {
            println!("{}", serde_json::json!({ "project": project, "token": t }));
        } else {
            println!("{t}");
        }
        return;
    }
    let listen = flag("--listen").unwrap_or_else(|| "127.0.0.1:8790".into());
    let static_dir = if has("--no-static") {
        None
    } else {
        Some(flag("--static").map(PathBuf::from).unwrap_or_else(|| {
            let dist = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../obs/web/dist"));
            if root.join("dashboard.html").is_file() || !dist.join("index.html").is_file() {
                if !root.join("dashboard.html").is_file() {
                    eprintln!("modgrad-server: no dashboard to serve at / (build obs/web/dist as obs/web/README.md says, or pass --static <dist>); the API works without it");
                }
                root.clone()
            } else {
                eprintln!("modgrad-server: dashboard from {}", dist.canonicalize().unwrap_or(dist.clone()).display());
                dist
            }
        }))
    };
    let (ingest, read) = match modgrad_obs_server::open(&root, IngestConfig::default()) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("modgrad-server: cannot open {}: {e}", root.display());
            std::process::exit(1);
        }
    };
    let auth = Auth::open(&root, cfg).unwrap_or_else(|e| {
        eprintln!("modgrad-server: token store under {}: {e}", root.display());
        std::process::exit(1)
    });
    let lis = tokio::net::TcpListener::bind(&listen).await.unwrap_or_else(|e| {
        eprintln!("modgrad-server: bind {listen}: {e}");
        std::process::exit(1)
    });
    let mut state = http::AppState::new(ingest, read, static_dir, auth).with_boards(root.join("boards"));
    if has("--mcp") {
        let mut a = lis.local_addr().expect("bound address");
        if a.ip().is_unspecified() {
            a.set_ip(if a.is_ipv4() { std::net::Ipv4Addr::LOCALHOST.into() } else { std::net::Ipv6Addr::LOCALHOST.into() });
        }
        state = state.with_mcp(format!("http://{a}"));
        eprintln!("modgrad-server: hosted MCP endpoint at /mcp (tool analysis runs on this server)");
    }
    let app = http::router(state);
    #[cfg(feature = "h3")]
    let (app, quic) = match h3_listener(&args, &root, &listen) {
        Some((ep, port)) => {
            tokio::spawn(modgrad_obs_server::h3::serve(ep.clone(), app.clone()));
            (modgrad_obs_server::h3::advertise(app, port), Some(ep))
        }
        None => (app, None),
    };
    #[cfg(not(feature = "h3"))]
    if has("--h3") {
        eprintln!("modgrad-server: built without the h3 feature; --h3 ignored (serving TCP only)");
    }
    eprintln!("modgrad-server: {} on http://{listen}/{}", root.display(), if cfg.dev { "  (dev mode: tokenless writes allowed)" } else { "" });
    axum::serve(lis, app).with_graceful_shutdown(async { let _ = tokio::signal::ctrl_c().await; }).await.unwrap();
    #[cfg(feature = "h3")]
    if let Some(ep) = quic {
        ep.close(0u32.into(), b"shutdown");
        let _ = tokio::time::timeout(std::time::Duration::from_secs(2), ep.wait_idle()).await;
    }
}

/// `--h3 <udp addr>`: bind the QUIC endpoint; returns it and the port to
/// advertise. Exits on a bad flag combination or bind error.
#[cfg(feature = "h3")]
fn h3_listener(args: &[String], root: &std::path::Path, listen: &str) -> Option<(quinn::Endpoint, u16)> {
    use modgrad_obs_server::h3::{Tls, endpoint};
    let flag = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1).cloned());
    let die = |m: String| -> ! {
        eprintln!("modgrad-server: {m}");
        std::process::exit(1)
    };
    let addr = flag("--h3")?;
    let addr: std::net::SocketAddr = addr.parse().unwrap_or_else(|_| die(format!("--h3 {addr:?}: want a UDP address like 0.0.0.0:8790")));
    let tls = match (flag("--tls-cert"), flag("--tls-key"), args.iter().any(|a| a == "--h3-self-signed")) {
        (Some(c), Some(k), _) => Tls::from_pem(c.as_ref(), k.as_ref()).unwrap_or_else(|e| die(format!("TLS: {e}"))),
        (None, None, true) => {
            let mut names = vec!["localhost".to_string(), "127.0.0.1".into(), "::1".into()];
            for h in [addr.ip().to_string(), listen.rsplit_once(':').map_or(listen, |(h, _)| h).trim_matches(['[', ']']).to_string()] {
                if !names.contains(&h) && h != "0.0.0.0" && h != "::" {
                    names.push(h);
                }
            }
            let (tls, pem) = Tls::self_signed(names.clone()).unwrap_or_else(|e| die(format!("self-signed certificate: {e}")));
            let p = root.join("h3-self-signed.pem");
            std::fs::write(&p, pem).unwrap_or_else(|e| die(format!("{}: {e}", p.display())));
            eprintln!("modgrad-server: DEV ONLY: HTTP/3 with a self-signed certificate for {}; clients must trust {} (modgrad run --tls-ca <it>)", names.join(", "), p.display());
            tls
        }
        _ => die("--h3 needs --tls-cert <pem> and --tls-key <pem> (or --h3-self-signed for local testing)".into()),
    };
    let ep = endpoint(addr, tls).unwrap_or_else(|e| die(format!("bind udp {addr}: {e}")));
    let bound = ep.local_addr().map(|a| a.port()).unwrap_or(addr.port());
    let port = flag("--h3-advertise").and_then(|p| p.parse().ok()).unwrap_or(bound);
    eprintln!("modgrad-server: HTTP/3 on udp {addr} (advertised as Alt-Svc h3=\":{port}\"; open this UDP port)");
    Some((ep, port))
}
