//! Where the MCP tools get runs from and write runs to.
//!
//! [`Store`] is the seam between the tools and storage. [`LocalStore`]
//! works directly on a modgrad-obs data directory with the crate's own
//! writer and reader. [`RemoteStore`] is a placeholder for the HTTP API
//! (being redesigned on another branch: auto-created runs, a text-log
//! ingest endpoint, anonymous project tokens); it only carries the
//! connection settings and refuses every call until that API settles.
//! Analysis never goes through the store: the tools pull rows and compute
//! locally, also in remote mode.

use crate::{Meta, MetricRow, Run, RunData, export, now_unix, read_jsonl};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

/// Cheap per-run facts for `list_runs`.
#[derive(Debug, Clone)]
pub struct RunInfo {
    pub name: String,
    pub last_step: Option<u64>,
    pub n_metrics: usize,
    pub finished: bool,
    /// Seconds since the run's metrics file was last written.
    pub age_s: Option<f64>,
}

/// Rows to append: `(step, [(key, value)])`.
pub type Rows = Vec<(u64, Vec<(String, f64)>)>;

pub trait Store {
    /// One line for error messages and the server banner.
    fn describe(&self) -> String;
    /// Every run name (`bkk07/small`), sorted.
    fn run_names(&self) -> Result<Vec<String>, String>;
    fn info(&self, name: &str) -> Result<RunInfo, String>;
    /// Config, summary and scalar history (no arrays) of one run.
    fn load(&self, name: &str) -> Result<RunData, String>;
    /// Append rows, creating the run when it does not exist. Returns
    /// whether it was created.
    fn append(&self, name: &str, rows: &Rows) -> Result<bool, String>;
    /// Persist a board; returns `(where it was saved, URL that opens it)`.
    fn save_board(&self, slug: &str, board: &Value) -> Result<(String, String), String>;
}

// ------------------------------------------------------------------ local

pub struct LocalStore {
    /// The obs root (holds `runs/`, `boards/`, `dashboard.html`).
    pub root: PathBuf,
    /// Base URL of a served dashboard (`http://localhost:8080/`); boards
    /// open as `file://<root>/dashboard.html#…` when unset.
    pub base_url: Option<String>,
}

/// Seconds after which an unfinished run with no new rows counts as stale.
pub const RUNNING_WINDOW_S: f64 = 15.0 * 60.0;

impl LocalStore {
    /// `dir` may be the obs root (has `runs/`) or the `runs/` directory
    /// itself. A missing directory is created on the first write.
    pub fn open(dir: &Path, base_url: Option<String>) -> LocalStore {
        let root = if dir.file_name().is_some_and(|n| n == "runs") && !dir.join("runs").exists() {
            dir.parent().map(Path::to_path_buf).unwrap_or_else(|| dir.to_path_buf())
        } else {
            dir.to_path_buf()
        };
        LocalStore { root, base_url }
    }

    fn runs_dir(&self) -> PathBuf {
        self.root.join("runs")
    }

    fn run_dir(&self, name: &str) -> PathBuf {
        self.runs_dir().join(name)
    }
}

fn mtime_age(p: &Path) -> Option<f64> {
    let m = std::fs::metadata(p).ok()?.modified().ok()?;
    let t = m.duration_since(std::time::UNIX_EPOCH).ok()?.as_secs_f64();
    Some((now_unix() - t).max(0.0))
}

/// Config + summary + metrics only. `read_run` would also decode every
/// array record, which can be large and is never needed here.
fn load_scalars(dir: &Path) -> std::io::Result<RunData> {
    let mut rd = RunData { dir: dir.to_path_buf(), ..Default::default() };
    rd.config = std::fs::read(dir.join("config.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(Value::Null);
    rd.summary = std::fs::read(dir.join("summary.json")).ok().and_then(|b| serde_json::from_slice(&b).ok());
    let (lines, s) = read_jsonl(&dir.join("metrics.jsonl"))?;
    rd.skipped = s;
    for v in lines {
        let Some(step) = v.get("step").and_then(|s| s.as_u64()) else {
            rd.skipped += 1;
            continue;
        };
        let t = v.get("t").and_then(|x| x.as_f64()).unwrap_or(f64::NAN);
        let values = v
            .as_object()
            .into_iter()
            .flatten()
            .filter(|(k, _)| !crate::RESERVED_KEYS.contains(&k.as_str()))
            .filter_map(|(k, v)| if v.is_null() { Some((k.clone(), f64::NAN)) } else { v.as_f64().map(|x| (k.clone(), x)) })
            .collect();
        rd.metrics.push(MetricRow { step, t, values });
    }
    Ok(rd)
}

/// A run name is a relative path of `[A-Za-z0-9._-]` components.
pub fn valid_run_name(name: &str) -> bool {
    !name.is_empty()
        && !name.starts_with('/')
        && name.split('/').all(|c| !c.is_empty() && c != "." && c != ".." && c.chars().all(|ch| ch.is_ascii_alphanumeric() || "._-".contains(ch)))
}

impl Store for LocalStore {
    fn describe(&self) -> String {
        format!("local data directory {}", self.root.display())
    }

    fn run_names(&self) -> Result<Vec<String>, String> {
        let runs = self.runs_dir();
        let mut v: Vec<String> = export::find_runs(&runs)
            .into_iter()
            .filter_map(|d| d.strip_prefix(&runs).ok().map(|p| p.to_string_lossy().into_owned()))
            .filter(|n| !n.is_empty())
            .collect();
        v.sort();
        Ok(v)
    }

    fn info(&self, name: &str) -> Result<RunInfo, String> {
        let dir = self.run_dir(name);
        let rd = load_scalars(&dir).map_err(|e| format!("cannot read run {name}: {e}"))?;
        let finished = rd.summary.as_ref().and_then(|s| s.get("finished")).and_then(|f| f.as_bool()).unwrap_or(false);
        Ok(RunInfo { name: name.to_string(), last_step: rd.last_step(), n_metrics: rd.metric_keys().len(), finished, age_s: mtime_age(&dir.join("metrics.jsonl")) })
    }

    fn load(&self, name: &str) -> Result<RunData, String> {
        let dir = self.run_dir(name);
        if !dir.join("config.json").is_file() {
            return Err(format!("run {name} does not exist in {}", self.describe()));
        }
        load_scalars(&dir).map_err(|e| format!("cannot read run {name}: {e}"))
    }

    fn append(&self, name: &str, rows: &Rows) -> Result<bool, String> {
        if !valid_run_name(name) {
            return Err(format!("invalid run name {name:?}: use letters, digits, '.', '_', '-' and '/' between parts, e.g. \"bkk07/small\""));
        }
        let dir = self.run_dir(name);
        let exists = dir.join("config.json").is_file();
        // Never `Run::create*` an existing run: it rewrites config.json.
        let mut run = if exists {
            Run::open_existing(&dir).map_err(|e| format!("cannot open run {name}: {e}"))?
        } else {
            let project = name.split('/').next().filter(|_| name.contains('/')).map(String::from);
            let meta = Meta { project, git: None, hostname: None, start_unix: now_unix(), extra: Some(json!({"created_by": "obs mcp"})) };
            Run::create_in(&dir, name, &json!({}), meta).map_err(|e| format!("cannot create run {name}: {e}"))?
        };
        for (step, kv) in rows {
            let kv: Vec<(&str, f64)> = kv.iter().map(|(k, v)| (k.as_str(), *v)).collect();
            run.log_scalars(*step, &kv).map_err(|e| format!("cannot write to run {name}: {e}"))?;
        }
        run.flush_summary().map_err(|e| format!("cannot write summary of {name}: {e}"))?;
        Ok(!exists)
    }

    fn save_board(&self, slug: &str, board: &Value) -> Result<(String, String), String> {
        let dir = self.root.join("boards");
        std::fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
        let path = dir.join(format!("{slug}.json"));
        let tmp = dir.join(format!("{slug}.json.tmp"));
        std::fs::write(&tmp, serde_json::to_vec_pretty(board).unwrap_or_default()).and_then(|_| std::fs::rename(&tmp, &path)).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        let url = match &self.base_url {
            Some(b) => format!("{}#board={slug}", b.trim_end_matches('#')),
            None => {
                let root = std::fs::canonicalize(&self.root).unwrap_or_else(|_| self.root.clone());
                format!("file://{}/dashboard.html#board={slug}", root.display())
            }
        };
        Ok((path.display().to_string(), url))
    }
}

// ----------------------------------------------------------------- remote

/// Placeholder for `obs mcp --url <server> --token <project token>`.
///
/// Mapping planned once the HTTP API lands (see the obs-api branch):
/// `run_names`/`info` → run listing, `load` → metric history fetch (the
/// tools keep doing all analysis locally), `append` → auto-creating log
/// endpoint, `ingest_log` → text-log ingest endpoint (server-side parser),
/// `save_board` → board endpoint returning a URL with a `#token` fragment.
pub struct RemoteStore {
    pub url: String,
    pub token: Option<String>,
}

impl RemoteStore {
    fn todo<T>(&self) -> Result<T, String> {
        Err(format!("remote mode ({}) is not implemented yet. Restart the server in local mode: obs mcp --dir ./obs", self.url))
    }
}

impl Store for RemoteStore {
    fn describe(&self) -> String {
        format!("remote server {}", self.url)
    }
    fn run_names(&self) -> Result<Vec<String>, String> {
        self.todo()
    }
    fn info(&self, _: &str) -> Result<RunInfo, String> {
        self.todo()
    }
    fn load(&self, _: &str) -> Result<RunData, String> {
        self.todo()
    }
    fn append(&self, _: &str, _: &Rows) -> Result<bool, String> {
        self.todo()
    }
    fn save_board(&self, _: &str, _: &Value) -> Result<(String, String), String> {
        self.todo()
    }
}
