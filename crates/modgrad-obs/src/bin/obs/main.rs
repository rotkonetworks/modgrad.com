//! `obs` — modgrad-obs command line.
//!
//! ```text
//! obs legacy <log.txt> <out_run_dir> [--name N] [--source S]
//!     Convert a ctmlm_pilot stdout log into the modgrad-obs format.
//!     Refuses to overwrite a native (non-legacy) run.
//! obs export <obs_dir>
//!     Write per-run plot files and <obs_dir>/index.json for the dashboard.
//! obs show <run_dir>
//!     Print keys, last step and skipped-line count of a run.
//! obs run [--server URL] [--token T] [--name N] [--project P] [--offline [--out DIR]]
//!         [--sys-interval S] [--no-gpu] [--spool DIR] [--json] -- <command...>
//!     Run a command, pass its output through unchanged, parse metric lines,
//!     sample RSS/CPU/GPU, and stream it all to the server (or a local run).
//! obs tail <run id | run dir> [--server URL] [--keys a,b] [--follow] [--interval S] [--json]
//!     Latest values and a sparkline per metric.
//! obs push <spool dir> [--json]
//!     Resend rows an earlier `obs run` could not deliver.
//! ```
//!
//! Environment: OBS_URL (server, default http://127.0.0.1:8790), OBS_TOKEN,
//! OBS_PROJECT. No command prompts; `--json` makes the CLI's own messages
//! one JSON object per line on stderr.

use std::path::{Path, PathBuf};

mod net;
mod run;
mod sys;
mod tail;

const USAGE: &str = "usage:
  obs run [--server URL] [--token T] [--name N] [--project P] [--offline [--out DIR]]
          [--sys-interval S] [--no-gpu] [--spool DIR] [--json] -- <command...>
  obs tail <run id | run dir> [--server URL] [--keys a,b] [--follow] [--interval S] [--json]
  obs push <spool dir> [--json]
  obs legacy <log.txt> <out_run_dir> [--name N] [--source S]
  obs export <obs_dir>
  obs show <run_dir>
  obs mcp [--dir <obs_dir>] [--base-url URL]   MCP server on stdio for coding agents
env: OBS_URL (default http://127.0.0.1:8790), OBS_TOKEN, OBS_PROJECT";

fn env(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|v| !v.is_empty())
}

fn spool_root() -> PathBuf {
    let base = env("XDG_CACHE_HOME").map(PathBuf::from).or_else(|| env("HOME").map(|h| PathBuf::from(h).join(".cache"))).unwrap_or_else(std::env::temp_dir);
    base.join("modgrad-obs").join("spool")
}

fn usage() -> ! {
    eprintln!("{USAGE}");
    std::process::exit(2)
}

fn flag(args: &[String], k: &str) -> Option<String> {
    args.iter().position(|a| a == k).and_then(|i| args.get(i + 1).cloned())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("run") => {
            // flags before `--`, the command after it
            let split = args.iter().position(|a| a == "--");
            let (flags, cmd) = match split {
                Some(i) => (&args[2..i], args[i + 1..].to_vec()),
                None => usage(),
            };
            let flags = flags.to_vec();
            let has = |k: &str| flags.iter().any(|a| a == k);
            let o = run::Opts {
                server: flag(&flags, "--server").or_else(|| env("OBS_URL")).unwrap_or_else(|| "http://127.0.0.1:8790".into()),
                token: flag(&flags, "--token").or_else(|| env("OBS_TOKEN")),
                name: flag(&flags, "--name"),
                project: flag(&flags, "--project").or_else(|| env("OBS_PROJECT")),
                offline: has("--offline"),
                out: PathBuf::from(flag(&flags, "--out").unwrap_or_else(|| "obs".into())),
                sys_interval: flag(&flags, "--sys-interval").and_then(|s| s.parse().ok()).unwrap_or(5.0),
                gpu: !has("--no-gpu"),
                json: has("--json"),
                spool_root: flag(&flags, "--spool").map(PathBuf::from).unwrap_or_else(spool_root),
                cmd,
            };
            std::process::exit(run::run(o));
        }
        Some("tail") => {
            let Some(target) = args.get(2).filter(|a| !a.starts_with("--")) else { usage() };
            let o = tail::Opts {
                target: target.clone(),
                server: flag(&args, "--server").or_else(|| env("OBS_URL")).unwrap_or_else(|| "http://127.0.0.1:8790".into()),
                keys: flag(&args, "--keys").map(|k| k.split(',').filter(|s| !s.is_empty()).map(String::from).collect()).unwrap_or_default(),
                follow: args.iter().any(|a| a == "--follow" || a == "-f"),
                interval: flag(&args, "--interval").and_then(|s| s.parse().ok()).unwrap_or(2.0),
                json: args.iter().any(|a| a == "--json"),
                width: flag(&args, "--width").and_then(|s| s.parse().ok()).unwrap_or(24),
            };
            std::process::exit(tail::tail(o));
        }
        Some("push") => {
            let Some(d) = args.get(2) else { usage() };
            std::process::exit(run::push(Path::new(d), args.iter().any(|a| a == "--json")));
        }
        Some("help" | "--help" | "-h") => {
            println!("{USAGE}");
        }
        Some("legacy") => {
            let (Some(log), Some(out)) = (args.get(2), args.get(3)) else { usage() };
            let out = PathBuf::from(out);
            if let Ok(b) = std::fs::read(out.join("config.json")) {
                let v: serde_json::Value = serde_json::from_slice(&b).unwrap_or_default();
                if v["meta"]["extra"]["legacy_log"].is_null() {
                    eprintln!("obs legacy: {} holds a native run; not overwriting", out.display());
                    std::process::exit(1);
                }
            }
            let text = std::fs::read_to_string(log).unwrap_or_else(|e| {
                eprintln!("obs legacy: {log}: {e}");
                std::process::exit(1)
            });
            let mtime = std::fs::metadata(log)
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs_f64())
                .unwrap_or_else(modgrad_obs::now_unix);
            let name = flag(&args, "--name").unwrap_or_else(|| {
                out.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "run".into())
            });
            let source = flag(&args, "--source").unwrap_or_else(|| "local".into());
            let log_ref = flag(&args, "--log-ref").unwrap_or_else(|| log.clone());
            match modgrad_obs::legacy::convert(&text, &out, &name, &source, &log_ref, mtime) {
                Ok(rep) => {
                    println!("{}: {} lines, {} step rows (last step {}), {} unparsed", out.display(), rep.lines, rep.steps, rep.last_step, rep.unparsed.len());
                    for u in &rep.unparsed {
                        println!("  unparsed: {u}");
                    }
                }
                Err(e) => {
                    eprintln!("obs legacy: {e}");
                    std::process::exit(1);
                }
            }
        }
        Some("export") => {
            let Some(obs) = args.get(2) else { usage() };
            match modgrad_obs::export::export_all(Path::new(obs)) {
                Ok(idx) => {
                    let runs = idx["runs"].as_array().map(|a| a.len()).unwrap_or(0);
                    println!("{obs}/index.json: {runs} runs");
                    for e in idx["errors"].as_array().into_iter().flatten() {
                        println!("  error: {e}");
                    }
                }
                Err(e) => {
                    eprintln!("obs export: {e}");
                    std::process::exit(1);
                }
            }
        }
        Some("show") => {
            let Some(d) = args.get(2) else { usage() };
            let rd = modgrad_obs::read_run(Path::new(d)).unwrap_or_else(|e| {
                eprintln!("obs show: {e}");
                std::process::exit(1)
            });
            println!("name {}  last step {:?}  rows {}  arrays {}  events {}  skipped {}", rd.name(), rd.last_step(), rd.metrics.len(), rd.arrays.len(), rd.events.len(), rd.skipped);
            println!("metrics: {}", rd.metric_keys().join(", "));
            println!("arrays:  {}", rd.array_keys().join(", "));
        }
        Some("mcp") => {
            // tools: log, ingest_log, list_runs, read_metric, summarize, compare, make_board.
            // --dir is the obs root holding runs/ (default $OBS_DIR, else ./obs); --url selects
            // remote mode (not implemented yet).
            use modgrad_obs::mcp::{Server, serve_stdio, store};
            let st: Box<dyn store::Store> = match flag(&args, "--url") {
                Some(url) => Box::new(store::RemoteStore { url, token: flag(&args, "--token") }),
                None => {
                    let dir = flag(&args, "--dir").or_else(|| env("OBS_DIR")).unwrap_or_else(|| "obs".into());
                    Box::new(store::LocalStore::open(Path::new(&dir), flag(&args, "--base-url")))
                }
            };
            if let Err(e) = serve_stdio(&Server::new(st)) {
                eprintln!("obs mcp: {e}");
                std::process::exit(1);
            }
        }
        _ => usage(),
    }
}
