//! ```text
//! obs-server [--root <obs dir>] [--listen 127.0.0.1:8790] [--static <dir> | --no-static]
//!            [--dev] [--no-anon] [--anon-expiry-days 30]
//! obs-server token add <project> [--root <obs dir>] [--json]
//! ```
//!
//! Serves the modgrad-obs store under `<root>/runs` over HTTP (the v1
//! ingest API, selection reads, SSE) and, by default, `<root>` itself as
//! static files so `/` opens dashboard.html in server mode.
//! `--dev` accepts writes without a token (local use only). `token add`
//! provisions a project token and prints it once. Logging: `RUST_LOG=info`.

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
            eprintln!("usage: obs-server token add <project> [--root <obs dir>] [--json]");
            std::process::exit(2)
        };
        std::fs::create_dir_all(&root).ok();
        let t = Auth::open(&root, cfg).and_then(|a| a.add_project_token(project)).unwrap_or_else(|e| {
            eprintln!("obs-server: token store under {}: {e}", root.display());
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
    let static_dir = if has("--no-static") { None } else { Some(flag("--static").map(PathBuf::from).unwrap_or_else(|| root.clone())) };
    let (ingest, read) = match modgrad_obs_server::open(&root, IngestConfig::default()) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("obs-server: cannot open {}: {e}", root.display());
            std::process::exit(1);
        }
    };
    let auth = Auth::open(&root, cfg).unwrap_or_else(|e| {
        eprintln!("obs-server: token store under {}: {e}", root.display());
        std::process::exit(1)
    });
    let app = http::router(http::AppState::new(ingest, read, static_dir, auth));
    let lis = tokio::net::TcpListener::bind(&listen).await.unwrap_or_else(|e| {
        eprintln!("obs-server: bind {listen}: {e}");
        std::process::exit(1)
    });
    eprintln!("obs-server: {} on http://{listen}/{}", root.display(), if cfg.dev { "  (dev mode: tokenless writes allowed)" } else { "" });
    axum::serve(lis, app).with_graceful_shutdown(async { let _ = tokio::signal::ctrl_c().await; }).await.unwrap();
}
