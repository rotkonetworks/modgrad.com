//! `obs-server --root <obs dir> [--listen 127.0.0.1:8790] [--static <dir>]`
//!
//! Serves the modgrad-obs store under `<root>/runs` over HTTP (ingest +
//! selection reads) and, by default, `<root>` itself as static files so
//! `/` opens dashboard.html in server mode. Logging: `RUST_LOG=info`.

use modgrad_obs_server::{IngestConfig, http};
use std::path::PathBuf;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().with_env_filter(tracing_subscriber::EnvFilter::from_default_env()).init();
    let args: Vec<String> = std::env::args().collect();
    let flag = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1).cloned());
    let root = PathBuf::from(flag("--root").unwrap_or_else(|| "obs".into()));
    let listen = flag("--listen").unwrap_or_else(|| "127.0.0.1:8790".into());
    let static_dir = if args.iter().any(|a| a == "--no-static") { None } else { Some(flag("--static").map(PathBuf::from).unwrap_or_else(|| root.clone())) };
    let (ingest, read) = match modgrad_obs_server::open(&root, IngestConfig::default()) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("obs-server: cannot open {}: {e}", root.display());
            std::process::exit(1);
        }
    };
    let app = http::router(http::AppState { ingest, read, static_dir });
    let lis = tokio::net::TcpListener::bind(&listen).await.unwrap_or_else(|e| {
        eprintln!("obs-server: bind {listen}: {e}");
        std::process::exit(1)
    });
    eprintln!("obs-server: {} on http://{listen}/", root.display());
    axum::serve(lis, app).with_graceful_shutdown(async { let _ = tokio::signal::ctrl_c().await; }).await.unwrap();
}
