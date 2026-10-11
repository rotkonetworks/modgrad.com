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
//! ```

use std::path::{Path, PathBuf};

fn usage() -> ! {
    eprintln!("usage: obs legacy <log.txt> <out_run_dir> [--name N] [--source S] | obs export <obs_dir> | obs show <run_dir>");
    std::process::exit(2)
}

fn flag(args: &[String], k: &str) -> Option<String> {
    args.iter().position(|a| a == k).and_then(|i| args.get(i + 1).cloned())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
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
        _ => usage(),
    }
}
