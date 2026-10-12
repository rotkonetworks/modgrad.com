//! ```text
//! modgrad-server [--root <obs dir>] [--listen 127.0.0.1:8790] [--static <dir> | --no-static]
//!            [--dev] [--no-anon] [--anon-expiry-days 30] [--mcp]
//!            [--h3 <udp addr> (--tls-cert <pem> --tls-key <pem> | --h3-self-signed) [--h3-advertise <port>]]
//!            [--ssh <tcp addr> [--ssh-idle <secs>] [--ssh-max-conns 256] [--ssh-max-per-key 8]]
//!            [--public-url <https://obs.example.org>] [--mail log | cmd:<path> | jmap] [--mail-from <addr>]
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
//!
//! `--ssh <tcp addr>` (feature `ssh`, on by default) adds the SSH door:
//! `python train.py 2>&1 | ssh -p <port> <host>` streams a log into a run,
//! the client's key being its account (see `ssh.rs`). The host key is
//! `<root>/ssh_host_ed25519`, made on first start. `--ssh-idle` closes a
//! connection silent that long (default 3600 s).
//!
//! `--public-url` is the base of links the server prints and returns
//! (dashboard URLs, board links); without it HTTP responses use the
//! request's Host / X-Forwarded-* headers and the SSH door uses
//! `http://<--listen>`. It is also the sign-in origin: cookies get
//! `Secure` under https, zafu challenges and passkeys are bound to it (rpId
//! = its host). Without it a request's own Host is the origin (dev). A
//! path in it (`https://modgrad.com/app`) is where a reverse proxy mounts
//! the server, stripping the path before it gets here: links keep the
//! path and the session cookie is scoped to it.
//!
//! Sign-in (`/api/v1/auth/*`, `/api/v1/me/*`, see `portal.rs`): `--mail`
//! picks how email sign-in links go out: `log` (default; nothing is sent,
//! the link is logged at info level: `RUST_LOG=info`), `cmd:<path>` (RFC
//! 5322 on the stdin of `<path> -t -i`, From `--mail-from`, default
//! `modgrad <noreply@<public host>>`), or `jmap` (feature `jmap`; env
//! `MODGRAD_JMAP_URL`, `MODGRAD_JMAP_USER`, `MODGRAD_JMAP_PASS`). With
//! `--ssh`, `ssh <host> login` prints a sign-in link.

use modgrad_obs_server::auth::{Auth, AuthConfig};
use modgrad_obs_server::{IngestConfig, http};
use std::path::PathBuf;

const USAGE: &str = "usage:
  modgrad-server [--root <obs dir>] [--listen 127.0.0.1:8790] [--static <dir> | --no-static]
             [--dev] [--no-anon] [--anon-expiry-days 30] [--mcp]
             [--h3 <udp addr> (--tls-cert <pem> --tls-key <pem> | --h3-self-signed) [--h3-advertise <port>]]
             [--ssh <tcp addr> [--ssh-idle <secs>] [--ssh-max-conns 256] [--ssh-max-per-key 8]]
             [--public-url <https://obs.example.org>]
             [--mail log | cmd:<path> | jmap] [--mail-from <addr>]
  modgrad-server token add <project> [--root <obs dir>] [--json]
--dev accepts writes without a token (local use only). Logging: RUST_LOG=info.
--public-url is the sign-in origin (cookies, zafu, passkeys); default: the request's Host.
--mail: how email sign-in links go out. log (default): not sent, logged at info level.
  cmd:<path>: the message on the stdin of `<path> -t -i` (sendmail, msmtp), From --mail-from.
  jmap (built with --features jmap): env MODGRAD_JMAP_URL, MODGRAD_JMAP_USER, MODGRAD_JMAP_PASS.";
/// flags that take a value, and flags that do not
const VALUED: &[&str] = &[
    "--root", "--listen", "--static", "--anon-expiry-days", "--h3", "--tls-cert", "--tls-key", "--h3-advertise",
    "--ssh", "--ssh-idle", "--ssh-max-conns", "--ssh-max-per-key", "--public-url", "--mail", "--mail-from",
];
const SWITCHES: &[&str] = &["--no-static", "--dev", "--no-anon", "--mcp", "--h3-self-signed", "--json"];

/// `--help` prints the usage; an unknown flag or a missing value is an error, not something to
/// ignore while starting a server on the default port
fn check_args(args: &[String]) {
    if args.iter().skip(1).any(|a| a == "-h" || a == "--help" || a == "help") {
        println!("{USAGE}");
        std::process::exit(0);
    }
    let mut i = if args.get(1).map(String::as_str) == Some("token") { 4 } else { 1 };
    while i < args.len() {
        let a = args[i].as_str();
        if VALUED.contains(&a) {
            if args.get(i + 1).is_none_or(|v| v.starts_with("--")) {
                eprintln!("modgrad-server: {a} needs a value\n{USAGE}");
                std::process::exit(2);
            }
            i += 2;
        } else if SWITCHES.contains(&a) {
            i += 1;
        } else {
            eprintln!("modgrad-server: unknown argument {a:?}\n{USAGE}");
            std::process::exit(2);
        }
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().with_env_filter(tracing_subscriber::EnvFilter::from_default_env()).init();
    let args: Vec<String> = std::env::args().collect();
    check_args(&args);
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
        state = state.with_mcp(loopback_url(&lis));
        eprintln!("modgrad-server: hosted MCP endpoint at /mcp (tool analysis runs on this server)");
    }
    let public_url = flag("--public-url").map(|u| u.trim_end_matches('/').to_string());
    if let Some(u) = &public_url {
        state = state.with_public_url(u.clone());
    }
    let mail_host = public_url.as_deref().and_then(|u| u.split_once("://")).map(|(_, r)| r.split(['/', ':']).next().unwrap_or(r).to_string()).unwrap_or_else(|| "localhost".into());
    let from = flag("--mail-from").unwrap_or_else(|| format!("modgrad <noreply@{mail_host}>"));
    let mailer = modgrad_obs_server::mail::from_spec(&flag("--mail").unwrap_or_else(|| "log".into()), &from).unwrap_or_else(|e| {
        eprintln!("modgrad-server: {e}");
        std::process::exit(2)
    });
    eprintln!("modgrad-server: sign-in mail: {}", mailer.describe());
    state = state.with_mailer(mailer);
    #[cfg(feature = "ssh")]
    if let Some(cmd) = ssh_listener(&args, &root, &state, public_url.clone().unwrap_or_else(|| loopback_url(&lis))).await {
        state = state.with_ssh_login(cmd);
    }
    #[cfg(not(feature = "ssh"))]
    if has("--ssh") {
        eprintln!("modgrad-server: built without the ssh feature; --ssh ignored");
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
    // the peer address feeds the sign-in email rate limit
    axum::serve(lis, app.into_make_service_with_connect_info::<std::net::SocketAddr>()).with_graceful_shutdown(async { let _ = tokio::signal::ctrl_c().await; }).await.unwrap();
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

/// `http://<bound addr>`, with an unspecified IP (0.0.0.0) as loopback.
fn loopback_url(lis: &tokio::net::TcpListener) -> String {
    let mut a = lis.local_addr().expect("bound address");
    if a.ip().is_unspecified() {
        a.set_ip(if a.is_ipv4() { std::net::Ipv4Addr::LOCALHOST.into() } else { std::net::Ipv6Addr::LOCALHOST.into() });
    }
    format!("http://{a}")
}

/// `--ssh <tcp addr>`: load or make the host key, bind, and serve in the
/// background. Exits on a bad flag or bind error. Returns the `ssh … login`
/// command for the sign-in page (None without `--ssh`).
#[cfg(feature = "ssh")]
async fn ssh_listener(args: &[String], root: &std::path::Path, state: &http::AppState, public: String) -> Option<String> {
    use modgrad_obs_server::ssh::{SshConfig, fingerprint, host_key, serve};
    let flag = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1).cloned());
    let die = |m: String| -> ! {
        eprintln!("modgrad-server: {m}");
        std::process::exit(1)
    };
    let addr = flag("--ssh")?;
    let num = |k: &str, d: f64| flag(k).map_or(d, |v| v.parse::<f64>().ok().filter(|x| *x > 0.0).unwrap_or_else(|| die(format!("{k} {v:?}: want a positive number"))));
    let d = SshConfig::default();
    let cfg = SshConfig {
        idle: std::time::Duration::from_secs_f64(num("--ssh-idle", d.idle.as_secs_f64())),
        max_conns: num("--ssh-max-conns", d.max_conns as f64) as usize,
        max_per_key: num("--ssh-max-per-key", d.max_per_key as f64) as usize,
        ..d
    };
    std::fs::create_dir_all(root).ok();
    let key = host_key(root).unwrap_or_else(|e| die(format!("ssh host key: {e}")));
    let lis = tokio::net::TcpListener::bind(&addr).await.unwrap_or_else(|e| die(format!("bind ssh {addr}: {e}")));
    eprintln!("modgrad-server: ssh on {addr} (host key {}; links to {public})", fingerprint(key.public_key()));
    if flag("--public-url").is_none() && lis.local_addr().is_ok_and(|a| !a.ip().is_loopback()) {
        eprintln!("modgrad-server: warning: ssh is reachable from other hosts but its links point at {public}; pass --public-url <https://your.host>");
    }
    let port = lis.local_addr().map(|a| a.port()).unwrap_or(22);
    let host = public.split_once("://").map_or(public.as_str(), |(_, r)| r).split(['/', ':']).next().unwrap_or("localhost").to_string();
    tokio::spawn(serve(lis, key, state.clone(), public, cfg));
    Some(if port == 22 { format!("ssh {host} login") } else { format!("ssh -p {port} {host} login") })
}
