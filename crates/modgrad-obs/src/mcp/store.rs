//! Where the MCP tools get runs from and write runs to.
//!
//! [`Store`] is the seam between the tools and storage. [`LocalStore`]
//! works directly on a modgrad-obs data directory with the crate's own
//! writer and reader. [`RemoteStore`] talks to an obs-server over the v1
//! HTTP API (see `modgrad-obs-server/src/v1_api.txt`).
//! Analysis never goes through the store: the tools pull rows and compute
//! locally, also in remote mode. The server only stores and selects.

use crate::{EventRecord, Meta, MetricRow, Run, RunData, export, net, now_unix, read_jsonl};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

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

pub trait Store: Send + Sync {
    /// One line for error messages and the server banner.
    fn describe(&self) -> String;
    /// Every run name (`bkk07/small`), sorted.
    fn run_names(&self) -> Result<Vec<String>, String>;
    fn info(&self, name: &str) -> Result<RunInfo, String>;
    /// Config, summary and scalar history (no arrays) of one run.
    fn load(&self, name: &str) -> Result<RunData, String>;
    /// Append rows, creating the run when it does not exist. Returns
    /// whether it was created and the run's full name (a remote project
    /// token puts `demo1` into its project: `anon-…/demo1`).
    fn append(&self, name: &str, rows: &Rows) -> Result<(bool, String), String>;
    /// The full name a write to `name` lands in. Remote with a project
    /// token: `demo1` -> `<project>/demo1`. Local: unchanged.
    fn qualify(&self, name: &str) -> String {
        name.to_string()
    }
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

/// Config + summary + metrics + events. `read_run` would also decode every
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
    if let Ok((lines, _)) = read_jsonl(&dir.join("events.jsonl")) {
        for v in lines {
            if let (Some(step), Some(text)) = (v.get("step").and_then(|x| x.as_u64()), v.get("text").and_then(|x| x.as_str())) {
                rd.events.push(EventRecord { step, t: v.get("t").and_then(|x| x.as_f64()).unwrap_or(f64::NAN), text: text.into() });
            }
        }
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

    fn append(&self, name: &str, rows: &Rows) -> Result<(bool, String), String> {
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
        Ok((!exists, name.to_string()))
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

/// `obs mcp --url <server> [--token <token>]`: the same tools against an
/// obs-server.
///
/// | store call          | HTTP                                              |
/// |---------------------|---------------------------------------------------|
/// | `run_names`, `info` | `GET /api/index` (cached ~2 s; token scopes it)   |
/// | `load`              | `GET /api/v1/runs/<id>/history` + `/config`, `/summary`, `/events` |
/// | `append`            | `POST /api/v1/runs/<id>/log` (auto-creates)       |
/// | `save_board`        | `PUT /api/v1/boards/<name>`                       |
///
/// The token (`obsp_…` project or `obsr_…` run) goes on every request:
/// writes need it unless the server runs `--dev`, and on reads it scopes
/// the run list to the token's project.
pub struct RemoteStore {
    /// Server base URL, e.g. `http://127.0.0.1:8790` (no trailing slash).
    pub url: String,
    pub token: Option<String>,
    index: Mutex<Option<(Instant, Vec<Value>)>>,
    /// The token's project, as reported by `/api/index`.
    project: Mutex<Option<String>>,
}

const TIMEOUT: Duration = Duration::from_secs(30);
const INDEX_TTL: Duration = Duration::from_secs(2);

impl RemoteStore {
    pub fn new(url: &str, token: Option<String>) -> RemoteStore {
        let url = url.trim().trim_end_matches('/').to_string();
        let url = if url.contains("://") { url } else { format!("http://{url}") };
        RemoteStore { url, token: token.filter(|t| !t.is_empty()), index: Mutex::new(None), project: Mutex::new(None) }
    }

    fn req(&self, method: &str, path: &str, body: Option<&Value>) -> Result<Value, String> {
        let url = format!("{}{path}", self.url);
        let bytes = body.map(|b| serde_json::to_vec(b).unwrap_or_default());
        let r = net::request(method, &url, self.token.as_deref(), bytes.as_deref(), "application/json", TIMEOUT).map_err(|e| {
            format!("Cannot reach the obs server at {} ({e}). Check that it is running (obs-server --root obs --dev), or restart this tool in local mode: obs mcp --dir ./obs", self.url)
        })?;
        if r.ok() {
            return Ok(r.json());
        }
        let msg = r.error_text();
        Err(match r.status {
            401 => format!(
                "The obs server at {} refused the request: {msg}. Writes need a project token: start this tool as obs mcp --url {} --token obsp_… (or set OBS_TOKEN); for the hosted /mcp endpoint, send the header Authorization: Bearer obsp_…. Get a token with curl -X POST {}/api/v1/projects",
                self.url, self.url, self.url
            ),
            403 => format!("The token does not cover this: {msg}"),
            _ => format!("obs server {} answered {}: {msg}", self.url, r.status),
        })
    }

    /// Index entries (`{id, last_step, metrics, finished, last_t, meta}`).
    fn index(&self) -> Result<Vec<Value>, String> {
        let mut g = self.index.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((at, v)) = g.as_ref()
            && at.elapsed() < INDEX_TTL
        {
            return Ok(v.clone());
        }
        let v = self.req("GET", "/api/index", None)?;
        let runs = v.get("runs").and_then(|r| r.as_array()).cloned().unwrap_or_default();
        if let Some(p) = v.get("project").and_then(|p| p.as_str()) {
            *self.project.lock().unwrap_or_else(|e| e.into_inner()) = Some(p.to_string());
        }
        *g = Some((Instant::now(), runs.clone()));
        Ok(runs)
    }

    fn forget_index(&self) {
        *self.index.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }

    fn entry(&self, name: &str) -> Result<Value, String> {
        self.index()?.into_iter().find(|e| e["id"].as_str() == Some(name)).ok_or_else(|| format!("run {name} does not exist on {}", self.describe()))
    }
}

/// `{"steps", "t", "columns": {key: [...]}}` into metric rows; `null`
/// cells (key not logged at that step) are left out.
pub fn rows_from_history(h: &Value) -> Vec<MetricRow> {
    let steps: Vec<u64> = h["steps"].as_array().map(|a| a.iter().map(|s| s.as_u64().unwrap_or(0)).collect()).unwrap_or_default();
    let t = h["t"].as_array();
    let mut rows: Vec<MetricRow> = steps.iter().enumerate().map(|(i, &step)| MetricRow { step, t: t.and_then(|t| t.get(i)).and_then(|x| x.as_f64()).unwrap_or(f64::NAN), values: Default::default() }).collect();
    for (k, col) in h["columns"].as_object().into_iter().flatten() {
        for (i, v) in col.as_array().into_iter().flatten().enumerate() {
            if let (Some(x), Some(r)) = (v.as_f64(), rows.get_mut(i)) {
                r.values.insert(k.clone(), x);
            }
        }
    }
    rows
}

impl Store for RemoteStore {
    fn describe(&self) -> String {
        format!("obs server {}", self.url)
    }

    fn qualify(&self, name: &str) -> String {
        if name.contains('/') || self.token.is_none() {
            return name.to_string();
        }
        if self.project.lock().unwrap_or_else(|e| e.into_inner()).is_none() {
            let _ = self.index();
        }
        match self.project.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
            Some(p) => format!("{p}/{name}"),
            None => name.to_string(),
        }
    }

    fn run_names(&self) -> Result<Vec<String>, String> {
        let mut v: Vec<String> = self.index()?.iter().filter_map(|e| e["id"].as_str().map(String::from)).collect();
        v.sort();
        Ok(v)
    }

    fn info(&self, name: &str) -> Result<RunInfo, String> {
        let e = self.entry(name)?;
        let start = e["meta"]["start_unix"].as_f64();
        let last_t = e["last_t"].as_f64();
        let age_s = match (start, last_t) {
            (Some(s), Some(t)) => Some((now_unix() - (s + t)).max(0.0)),
            _ => None,
        };
        Ok(RunInfo {
            name: name.to_string(),
            last_step: e["last_step"].as_u64(),
            n_metrics: e["metrics"].as_object().map(|m| m.len()).unwrap_or(0),
            finished: e["finished"].as_bool().unwrap_or(false),
            age_s,
        })
    }

    fn load(&self, name: &str) -> Result<RunData, String> {
        let base = format!("/api/v1/runs/{name}");
        let h = self.req("GET", &format!("{base}/history"), None).map_err(|e| if e.contains("unknown run") { format!("run {name} does not exist on {}", self.describe()) } else { e })?;
        let config = self.req("GET", &format!("{base}/config"), None).unwrap_or(Value::Null);
        let summary = self.req("GET", &format!("{base}/summary"), None).ok().filter(|s| !s.is_null());
        let events = self
            .req("GET", &format!("{base}/events"), None)
            .ok()
            .and_then(|v| v.as_array().cloned())
            .unwrap_or_default()
            .into_iter()
            .filter_map(|v| Some(EventRecord { step: v["step"].as_u64()?, t: v["t"].as_f64().unwrap_or(f64::NAN), text: v["text"].as_str()?.to_string() }))
            .collect();
        Ok(RunData { dir: PathBuf::from(name), config, metrics: rows_from_history(&h), arrays: vec![], events, summary, skipped: 0 })
    }

    fn append(&self, name: &str, rows: &Rows) -> Result<(bool, String), String> {
        if !valid_run_name(name) {
            return Err(format!("invalid run name {name:?}: use letters, digits, '.', '_', '-' and '/' between parts, e.g. \"bkk07/small\""));
        }
        let body: Vec<Value> = rows
            .iter()
            .map(|(step, kv)| {
                let mut o = serde_json::Map::new();
                o.insert("step".into(), json!(step));
                for (k, v) in kv {
                    o.insert(k.clone(), if v.is_finite() { json!(v) } else { Value::Null });
                }
                Value::Object(o)
            })
            .collect();
        let mut created = false;
        let mut run = name.to_string();
        // keep request bodies modest; the server batches anyway
        for chunk in body.chunks(2000) {
            let r = self.req("POST", &format!("/api/v1/runs/{name}/log"), Some(&Value::Array(chunk.to_vec())))?;
            created |= r["created"].as_bool().unwrap_or(false);
            if let Some(n) = r["run"].as_str() {
                run = n.to_string();
            }
        }
        self.forget_index();
        Ok((created, run))
    }

    fn save_board(&self, slug: &str, board: &Value) -> Result<(String, String), String> {
        let r = self.req("PUT", &format!("/api/v1/boards/{slug}"), Some(board))?;
        let id = r["board"].as_str().unwrap_or(slug).to_string();
        let url = r["url"].as_str().map(String::from).unwrap_or_else(|| format!("{}/#board={id}", self.url));
        Ok((format!("{}/api/v1/boards/{id}", self.url), url))
    }
}
