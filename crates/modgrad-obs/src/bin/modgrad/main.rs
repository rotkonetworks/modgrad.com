//! `modgrad` — modgrad-obs command line.
//!
//! ```text
//! modgrad init [--server URL] [--token T] [--project P] [--local] [--data DIR] [--mcp claude|cursor] [--json]
//!     Set up this directory: write .modgrad.json (server, project, token; or
//!     the local data dir), add it to .gitignore, optionally register the MCP
//!     server, and print what to run next. Never prompts; safe to re-run.
//! modgrad legacy <log.txt> <out_run_dir> [--name N] [--source S]
//!     Convert a ctmlm_pilot stdout log into the modgrad-obs format.
//!     Refuses to overwrite a native (non-legacy) run.
//! modgrad export <obs_dir>
//!     Write per-run plot files and <obs_dir>/index.json for the dashboard.
//! modgrad show <run_dir>
//!     Print keys, last step and skipped-line count of a run.
//! modgrad run [--server URL] [--token T] [--name N] [--project P] [--offline [--out DIR]]
//!         [--sys-interval S] [--no-gpu] [--spool DIR] [--h3 | --no-h3] [--tls-ca PEM] [--json] -- <command...>
//!     Run a command, pass its output through unchanged, parse metric lines,
//!     sample RSS/CPU/GPU, and stream it all to the server (or a local run).
//!     Rows go over HTTP/3 when the server offers it (https + Alt-Svc) or
//!     with --h3, else TCP (always with --no-h3); a blocked UDP path falls
//!     back to TCP after 1.5 s.
//! modgrad tail <run id | run dir> [--server URL] [--keys a,b] [--follow] [--interval S] [--json]
//!     Latest values and a sparkline per metric.
//! modgrad push <spool dir> [--h3 | --no-h3] [--tls-ca PEM] [--json]
//!     Resend rows an earlier `modgrad run` could not deliver.
//! ```
//!
//! Environment: MODGRAD_URL (server, default http://127.0.0.1:8790), MODGRAD_TOKEN,
//! MODGRAD_PROJECT, MODGRAD_H3=1|0 (same as --h3 / --no-h3), MODGRAD_TLS_CA
//! (same as --tls-ca). Settings resolve as flag > environment > project file
//! (`.modgrad.json` here or in a parent directory) > default. No command
//! prompts; `--json` makes the CLI's own messages one JSON object per line
//! on stderr (`init --json` prints its one result object on stdout).

use std::path::{Path, PathBuf};

use modgrad_obs::net;
mod project;
mod run;
mod sys;
mod tail;

const USAGE: &str = "usage:
  modgrad init [--server URL] [--token T] [--project P] [--local] [--data DIR]
          [--mcp claude|cursor] [--json]                set up this directory (.modgrad.json)
  modgrad run [--server URL] [--token T] [--name N] [--project P] [--offline [--out DIR]]
          [--parent RUN[@STEP]] [--stage pretrain|midtrain|sft|rl|eval]
          [--sys-interval S] [--no-gpu] [--spool DIR] [--h3 | --no-h3] [--tls-ca PEM] [--json] -- <command...>
  modgrad tail <run id | run dir> [--server URL] [--keys a,b] [--follow] [--interval S] [--json]
  modgrad push <spool dir> [--h3 | --no-h3] [--tls-ca PEM] [--json]
  modgrad legacy <log.txt> <out_run_dir> [--name N] [--source S]
  modgrad export <obs_dir>
  modgrad show <run_dir>
  modgrad mcp [--dir <obs_dir>] [--base-url URL]   MCP server on stdio for coding agents (local data)
  modgrad mcp --url <server> [--token T]           same tools against a modgrad-server (analysis stays local)
env: MODGRAD_URL (default http://127.0.0.1:8790), MODGRAD_TOKEN, MODGRAD_PROJECT,
     MODGRAD_H3=1|0 (run/push: HTTP/3 even without Alt-Svc | never), MODGRAD_TLS_CA (extra CA, PEM),
     MODGRAD_DATA (local data dir: mcp, run --offline, tail), MODGRAD_MCP_URL (modgrad mcp server, same as --url);
     the pre-rename OBS_* names still work
project file: .modgrad.json (from modgrad init) in this or a parent directory;
     flag > env > project file > default";

/// `MODGRAD_*` variables also answer to their pre-rename `OBS_*` names.
fn env(k: &str) -> Option<String> {
    let get = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
    let old = match k {
        "MODGRAD_DATA" => "OBS_DIR".to_string(),
        _ => k.replacen("MODGRAD_", "OBS_", 1),
    };
    get(k).or_else(|| get(&old))
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

/// The `command` for an MCP config: plain `modgrad` when that is this
/// binary on PATH, else this binary's absolute path.
fn mcp_command() -> String {
    let me = std::env::current_exe().ok().and_then(|p| std::fs::canonicalize(p).ok());
    let on_path = std::env::var_os("PATH").and_then(|p| std::env::split_paths(&p).map(|d| d.join("modgrad")).find(|c| c.is_file())).and_then(|c| std::fs::canonicalize(c).ok());
    match (me, on_path) {
        (Some(a), Some(b)) if a == b => "modgrad".into(),
        (Some(a), _) => a.display().to_string(),
        _ => "modgrad".into(),
    }
}

/// The local data directory: MODGRAD_DATA > the project file's `data` > `obs`.
fn data_dir(pf: Option<&project::Found>) -> PathBuf {
    env("MODGRAD_DATA").map(PathBuf::from).or_else(|| pf.map(|f| f.data_dir())).unwrap_or_else(|| "obs".into())
}

/// `--h3` / `--no-h3` / `MODGRAD_H3` (1 or 0), and `--tls-ca` /
/// `MODGRAD_TLS_CA` (installed for curl and QUIC).
fn transport_flags(args: &[String]) -> net::H3 {
    if let Some(ca) = flag(args, "--tls-ca").or_else(|| env("MODGRAD_TLS_CA")) {
        net::set_tls_ca(PathBuf::from(ca));
    }
    let has = |k: &str| args.iter().any(|a| a == k);
    match (has("--h3"), has("--no-h3"), env("MODGRAD_H3").as_deref()) {
        (_, true, _) => net::H3::Off,
        (true, _, _) => net::H3::Force,
        (_, _, Some("0" | "off" | "false")) => net::H3::Off,
        (_, _, Some(_)) => net::H3::Force,
        _ => net::H3::Auto,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("--version" | "-V" | "version") => println!("modgrad {}", env!("CARGO_PKG_VERSION")),
        Some("init") => {
            let json = args.iter().any(|a| a == "--json");
            let o = project::InitOpts {
                server: flag(&args, "--server").or_else(|| env("MODGRAD_URL")),
                token: flag(&args, "--token"),
                project: flag(&args, "--project"),
                data: flag(&args, "--data").or_else(|| env("MODGRAD_DATA")),
                local: args.iter().any(|a| a == "--local"),
                mcp: flag(&args, "--mcp"),
                exe: mcp_command(),
            };
            let dir = std::env::current_dir().unwrap_or_else(|_| ".".into());
            match project::init(&dir, &o) {
                Ok(r) if json => println!("{}", project::report_json(&r)),
                Ok(r) => print!("{}", project::render(&r)),
                Err(e) => {
                    if json {
                        println!("{}", serde_json::json!({ "ok": false, "error": e }));
                    } else {
                        eprintln!("modgrad init: {e}");
                    }
                    std::process::exit(1);
                }
            }
        }
        Some("run") => {
            // flags before `--`, the command after it
            let split = args.iter().position(|a| a == "--");
            let (flags, cmd) = match split {
                Some(i) => (&args[2..i], args[i + 1..].to_vec()),
                None => usage(),
            };
            let flags = flags.to_vec();
            let has = |k: &str| flags.iter().any(|a| a == k);
            let pf = project::find_cwd();
            let named = flag(&flags, "--server").or_else(|| env("MODGRAD_URL"));
            // a local-mode project file (no server) makes `modgrad run` write files
            let local = named.is_none() && pf.as_ref().is_some_and(|f| f.p.local());
            let server = named.or_else(|| pf.as_ref().and_then(|f| f.p.server.clone())).unwrap_or_else(|| project::DEFAULT_SERVER.into());
            let file_token = pf.as_ref().and_then(|f| f.p.token_for(&server));
            let file_project = pf.as_ref().filter(|_| file_token.is_some()).and_then(|f| f.p.project.clone());
            let o = run::Opts {
                server,
                token: flag(&flags, "--token").or_else(|| env("MODGRAD_TOKEN")).or(file_token),
                name: flag(&flags, "--name"),
                project: flag(&flags, "--project").or_else(|| env("MODGRAD_PROJECT")).or(file_project),
                offline: has("--offline") || local,
                out: flag(&flags, "--out").map(PathBuf::from).unwrap_or_else(|| data_dir(pf.as_ref())),
                sys_interval: flag(&flags, "--sys-interval").and_then(|s| s.parse().ok()).unwrap_or(5.0),
                gpu: !has("--no-gpu"),
                json: has("--json"),
                spool_root: flag(&flags, "--spool").map(PathBuf::from).unwrap_or_else(spool_root),
                h3: transport_flags(&flags),
                config: match run::lineage_config(flag(&flags, "--parent").as_deref(), flag(&flags, "--stage").as_deref()) {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!("modgrad run: {e}");
                        std::process::exit(2);
                    }
                },
                cmd,
            };
            std::process::exit(run::run(o));
        }
        Some("tail") => {
            let Some(target) = args.get(2).filter(|a| !a.starts_with("--")) else { usage() };
            let pf = project::find_cwd();
            let named = flag(&args, "--server").or_else(|| env("MODGRAD_URL"));
            let server = named.clone().or_else(|| pf.as_ref().and_then(|f| f.p.server.clone())).unwrap_or_else(|| project::DEFAULT_SERVER.into());
            let mut target = target.clone();
            if !Path::new(&target).join("config.json").is_file() {
                match pf.as_ref() {
                    // local mode: a run id names <data>/runs/<id>
                    Some(f) if named.is_none() && f.p.local() => {
                        let d = data_dir(pf.as_ref()).join("runs").join(&target);
                        if !d.join("config.json").is_file() {
                            eprintln!("modgrad tail: no run {target} in {} (local mode, from {})", d.parent().unwrap_or(&d).display(), f.dir.join(project::FILE).display());
                            std::process::exit(1);
                        }
                        target = d.display().to_string();
                    }
                    // with a project token, `exp1` is stored as <project>/exp1
                    Some(f) if !target.contains('/') && f.p.token_for(&server).is_some() => {
                        if let Some(p) = &f.p.project {
                            target = format!("{p}/{target}");
                        }
                    }
                    _ => {}
                }
            }
            let o = tail::Opts {
                target,
                server,
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
            std::process::exit(run::push(Path::new(d), args.iter().any(|a| a == "--json"), transport_flags(&args)));
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
                    eprintln!("modgrad legacy: {} holds a native run; not overwriting", out.display());
                    std::process::exit(1);
                }
            }
            let text = std::fs::read_to_string(log).unwrap_or_else(|e| {
                eprintln!("modgrad legacy: {log}: {e}");
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
                    eprintln!("modgrad legacy: {e}");
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
                    eprintln!("modgrad export: {e}");
                    std::process::exit(1);
                }
            }
        }
        Some("show") => {
            let Some(d) = args.get(2) else { usage() };
            let rd = modgrad_obs::read_run(Path::new(d)).unwrap_or_else(|e| {
                eprintln!("modgrad show: {e}");
                std::process::exit(1)
            });
            println!("name {}  last step {:?}  rows {}  arrays {}  events {}  skipped {}", rd.name(), rd.last_step(), rd.metrics.len(), rd.arrays.len(), rd.events.len(), rd.skipped);
            println!("metrics: {}", rd.metric_keys().join(", "));
            println!("arrays:  {}", rd.array_keys().join(", "));
        }
        Some("mcp") => {
            // tools: log, ingest_log, list_runs, read_metric, summarize, compare, make_board.
            // --dir is the obs root holding runs/ (default $MODGRAD_DATA, else ./obs); --url
            // (or $MODGRAD_MCP_URL) selects remote mode against a modgrad-server, with --token /
            // $MODGRAD_TOKEN. Analysis always runs here, on fetched rows.
            // Without --url/--dir/MODGRAD_DATA the project file decides: its server
            // (remote, with its token) or its data dir (local).
            use modgrad_obs::mcp::{Server, serve_stdio, store};
            let pf = project::find_cwd();
            let token = |url: &str| flag(&args, "--token").or_else(|| env("MODGRAD_TOKEN")).or_else(|| pf.as_ref().and_then(|f| f.p.token_for(url)));
            // board links: --base-url, else the project file's server, else the local viewer
            let base_url = flag(&args, "--base-url").or_else(|| pf.as_ref().and_then(|f| f.p.server.clone()));
            let local = |dir: PathBuf| -> Box<dyn store::Store> { Box::new(store::LocalStore::open(&dir, base_url.clone())) };
            let url = flag(&args, "--url").or_else(|| env("MODGRAD_MCP_URL"));
            let st: Box<dyn store::Store> = match (url, flag(&args, "--dir"), pf.as_ref().and_then(|f| f.p.server.clone())) {
                (Some(url), _, _) => Box::new(store::RemoteStore::new(&url, token(&url))),
                (None, Some(dir), _) => local(dir.into()),
                (None, None, Some(url)) if env("MODGRAD_DATA").is_none() => Box::new(store::RemoteStore::new(&url, token(&url))),
                _ => local(data_dir(pf.as_ref())),
            };
            if let Err(e) = serve_stdio(&Server::new(st)) {
                eprintln!("modgrad mcp: {e}");
                std::process::exit(1);
            }
        }
        _ => usage(),
    }
}
