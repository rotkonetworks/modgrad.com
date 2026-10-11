//! `obs-wandb` — W&B offline-run interop.
//!
//! ```text
//! obs-wandb export <run_dir> <out_dir> [--id <wandb run id>]
//!     -> <out_dir>/offline-run-<stamp>-<id>/run-<id>.wandb  (then: wandb sync <that dir>)
//! obs-wandb import <file.wandb> <obs_root> [--name <run name>]
//!     -> <obs_root>/runs/<name>/ (default name wandb/<display name>)
//! obs-wandb dump <file.wandb>
//!     print each record as one line (type, num, a short description)
//! ```

use modgrad_obs_wandb::proto::RecordType;
use std::path::Path;

fn flag(args: &[String], k: &str) -> Option<String> {
    args.iter().position(|a| a == k).and_then(|i| args.get(i + 1).cloned())
}

fn die(msg: String) -> ! {
    eprintln!("obs-wandb: {msg}");
    std::process::exit(1)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match (args.get(1).map(String::as_str), args.get(2), args.get(3)) {
        (Some("export"), Some(run), Some(out)) => {
            let r = modgrad_obs_wandb::export_run(Path::new(run), Path::new(out), flag(&args, "--id").as_deref()).unwrap_or_else(|e| die(e.to_string()));
            println!(
                "{}: {} records, {} history rows, {} histograms; not exported: {} arrays (no W&B tensor type), {} events",
                r.file.display(), r.records, r.history_rows, r.histograms, r.skipped_arrays, r.skipped_events
            );
        }
        (Some("import"), Some(file), Some(root)) => {
            let r = modgrad_obs_wandb::import_file(Path::new(file), Path::new(root), flag(&args, "--name").as_deref()).unwrap_or_else(|e| die(e.to_string()));
            println!("{}: {} records, {} history rows, {} histograms, {} skipped values, {} other records", r.run_dir.display(), r.records, r.history_rows, r.histograms, r.skipped_values, r.other_records);
        }
        (Some("dump"), Some(file), _) => {
            for r in modgrad_obs_wandb::read_records(Path::new(file)).unwrap_or_else(|e| die(e.to_string())) {
                let d = match &r.record_type {
                    Some(RecordType::Header(h)) => format!("header producer={:?}", h.version_info.as_ref().map(|v| v.producer.as_str())),
                    Some(RecordType::Run(x)) => format!("run id={} project={} name={} config_items={}", x.run_id, x.project, x.display_name, x.config.as_ref().map_or(0, |c| c.update.len())),
                    Some(RecordType::History(h)) => format!("history step={:?} items={}", h.step.as_ref().map(|s| s.num), h.item.len()),
                    Some(RecordType::Summary(s)) => format!("summary items={}", s.update.len()),
                    Some(RecordType::Config(c)) => format!("config items={}", c.update.len()),
                    Some(RecordType::Exit(e)) => format!("exit code={} runtime={}", e.exit_code, e.runtime),
                    Some(RecordType::Final(_)) => "final".into(),
                    None => "other (not modelled)".into(),
                };
                println!("{:5} {d}", r.num);
            }
        }
        _ => {
            eprintln!("usage: obs-wandb export <run_dir> <out_dir> [--id ID] | import <file.wandb> <obs_root> [--name N] | dump <file.wandb>");
            std::process::exit(2);
        }
    }
}
