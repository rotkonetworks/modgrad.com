//! Lightweight, local, append-only experiment observability.
//!
//! A run is a directory `<root>/runs/<name>/` holding:
//!
//! | file            | written by          | content                                         |
//! |-----------------|---------------------|-------------------------------------------------|
//! | `config.json`   | [`Run::init`]       | `{name, config, meta: {project, git, hostname, start_unix}}` |
//! | `metrics.jsonl` | [`Run::log`]        | history: `{"step":N,"t":secs,"key":value,...}` per line |
//! | `arrays.jsonl`  | [`Run::log`], [`Run::log_array`] | tensors and histograms: `{"step","t","key","kind","shape","dtype":"f32le","b64"[,"edges"]}` |
//! | `events.jsonl`  | [`Run::event`]      | `{"step","t","text"}` |
//! | `summary.json`  | [`Run::summary`], [`Run::finish`] | `{summary, elapsed_s, finished?, end_unix?}` |
//!
//! Config, history (metrics + arrays) and summary are separate concepts, as
//! in Weights & Biases, and the API mirrors `wandb.init` / `run.log` /
//! `run.summary[...]` / `run.finish`. The summary defaults to the last
//! value of every logged scalar; [`Run::summary`] overrides a key.
//!
//! `t` is wall-clock seconds since the run was created (an `f64`); the
//! absolute start time is `meta.start_unix` in `config.json`.
//!
//! Durability contract: every record is serialised to one `String`
//! ending in `\n` and handed to a single `write_all` on a file opened in
//! append mode (no `BufWriter`), then flushed. A crash therefore leaves at
//! most one partial final line. [`read_run`] ignores a final segment that
//! lacks its newline and skips any line that fails to parse, counting
//! them in [`RunData::skipped`].
//!
//! No network, no background threads. Dependencies: serde, serde_json,
//! base64, and the sibling no_std crate modgrad-obs-analysis (kernels).
//!

use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use serde::Serialize;
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Maximum number of f32 elements accepted by one [`Run::log_array`] call
/// (256 Ki elements = 1 MiB raw, ~1.4 MiB as base64). Larger arrays must be
/// reduced first, e.g. with [`downsample`].
pub const MAX_ARRAY_ELEMS: usize = 256 * 1024;

/// Reserved metric keys that [`Run::log`] refuses.
pub const RESERVED_KEYS: [&str; 2] = ["step", "t"];

#[derive(Debug)]
pub enum ObsError {
    Io(io::Error),
    /// `log_array` got more than [`MAX_ARRAY_ELEMS`] elements.
    TooLarge { key: String, elems: usize, cap: usize },
    /// `shape` product does not match `data.len()`.
    ShapeMismatch { key: String, shape: Vec<usize>, len: usize },
    /// A metric key collides with `step` or `t`.
    ReservedKey(String),
}

impl std::fmt::Display for ObsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ObsError::Io(e) => write!(f, "io: {e}"),
            ObsError::TooLarge { key, elems, cap } => {
                write!(f, "array {key}: {elems} elements exceeds cap {cap}; use modgrad_obs::downsample")
            }
            ObsError::ShapeMismatch { key, shape, len } => write!(f, "array {key}: shape {shape:?} != len {len}"),
            ObsError::ReservedKey(k) => write!(f, "metric key {k:?} is reserved"),
        }
    }
}
impl std::error::Error for ObsError {}
impl From<io::Error> for ObsError {
    fn from(e: io::Error) -> Self {
        ObsError::Io(e)
    }
}

/// Provenance written into `config.json` under `meta`.
#[derive(Debug, Clone, Serialize, Default)]
pub struct Meta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
    pub git: Option<String>,
    pub hostname: Option<String>,
    /// Seconds since the Unix epoch.
    pub start_unix: f64,
    /// Free-form extra provenance (e.g. `{"legacy_log": "..."}`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra: Option<Value>,
}

impl Meta {
    /// Detect git commit (of the current directory), hostname, and now.
    pub fn detect() -> Meta {
        Meta { project: None, git: git_commit(), hostname: hostname(), start_unix: now_unix(), extra: None }
    }
}

pub fn now_unix() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

fn git_commit() -> Option<String> {
    let out = std::process::Command::new("git").args(["rev-parse", "HEAD"]).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let mut s = String::from_utf8(out.stdout).ok()?.trim().to_string();
    let dirty = std::process::Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=no"])
        .output()
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(false);
    if dirty {
        s.push_str("-dirty");
    }
    Some(s)
}

fn hostname() -> Option<String> {
    std::fs::read_to_string("/etc/hostname")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| std::env::var("HOSTNAME").ok())
}

/// An open run. All writes are append-only single-`write_all` lines.
pub struct Run {
    dir: PathBuf,
    metrics: File,
    arrays: File,
    events: File,
    t0: Instant,
    start_unix: f64,
    /// Run summary (W&B semantics): last value of every logged scalar,
    /// overridable with [`Run::summary`]. Written to `summary.json`.
    summary: Map<String, Value>,
}

/// One logged value, mirroring W&B's value types: a scalar goes to the
/// history (`metrics.jsonl`), a histogram or tensor to `arrays.jsonl`.
#[derive(Debug, Clone, PartialEq)]
pub enum Item {
    Scalar(f64),
    Histogram(Histogram),
    Tensor(Tensor),
}

/// Row-major f32 tensor (W&B logs these as arrays / images / tables; here
/// it is shape + little-endian base64).
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn new(shape: &[usize], data: Vec<f32>) -> Tensor {
        Tensor { shape: shape.to_vec(), data }
    }
}

/// Histogram with explicit bin edges (`edges.len() == counts.len() + 1`),
/// the same shape as `wandb.Histogram(np_histogram=(counts, edges))`.
#[derive(Debug, Clone, PartialEq)]
pub struct Histogram {
    pub edges: Vec<f64>,
    pub counts: Vec<f32>,
}

/// W&B's histogram bin cap.
pub const MAX_HISTOGRAM_BINS: usize = 512;

impl Histogram {
    /// Equal-width histogram of `data` (non-finite values ignored) with
    /// `bins` bins (clamped to 1..=512).
    pub fn from_data(data: &[f32], bins: usize) -> Histogram {
        let bins = bins.clamp(1, MAX_HISTOGRAM_BINS);
        let fin: Vec<f64> = data.iter().filter(|x| x.is_finite()).map(|&x| x as f64).collect();
        let (lo, hi) = fin.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &x| (a.min(x), b.max(x)));
        let (lo, hi) = if fin.is_empty() { (0.0, 1.0) } else if lo == hi { (lo - 0.5, hi + 0.5) } else { (lo, hi) };
        let w = (hi - lo) / bins as f64;
        let mut counts = vec![0f32; bins];
        for x in fin {
            counts[(((x - lo) / w) as usize).min(bins - 1)] += 1.0;
        }
        Histogram { edges: (0..=bins).map(|i| lo + w * i as f64).collect(), counts }
    }
}

impl From<f64> for Item {
    fn from(x: f64) -> Item {
        Item::Scalar(x)
    }
}
impl From<f32> for Item {
    fn from(x: f32) -> Item {
        Item::Scalar(x as f64)
    }
}
impl From<usize> for Item {
    fn from(x: usize) -> Item {
        Item::Scalar(x as f64)
    }
}
impl From<Histogram> for Item {
    fn from(h: Histogram) -> Item {
        Item::Histogram(h)
    }
}
impl From<Tensor> for Item {
    fn from(t: Tensor) -> Item {
        Item::Tensor(t)
    }
}

fn open_append(p: &Path) -> io::Result<File> {
    OpenOptions::new().create(true).append(true).open(p)
}

fn write_line(f: &mut File, mut line: String) -> io::Result<()> {
    line.push('\n');
    f.write_all(line.as_bytes())?;
    f.flush()
}

/// If `p` ends in a partial line (a crash mid-write), truncate it back to
/// its last newline so the next append starts on a fresh line. Returns the
/// number of bytes dropped.
pub fn repair_tail(p: &Path) -> io::Result<u64> {
    let bytes = match std::fs::read(p) {
        Ok(b) => b,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(0),
        Err(e) => return Err(e),
    };
    if bytes.is_empty() || bytes.last() == Some(&b'\n') {
        return Ok(0);
    }
    let keep = bytes.iter().rposition(|&b| b == b'\n').map(|i| i + 1).unwrap_or(0);
    OpenOptions::new().write(true).open(p)?.set_len(keep as u64)?;
    Ok((bytes.len() - keep) as u64)
}

/// The JSONL files of a run, in the order used by [`Run::committed`] and
/// [`read_run_limited`].
pub const JSONL_FILES: [&str; 3] = ["metrics.jsonl", "arrays.jsonl", "events.jsonl"];

/// One history line. Errors on reserved keys.
pub fn metric_line(step: u64, t: f64, kv: &[(&str, f64)]) -> Result<String, ObsError> {
    let mut m = Map::new();
    m.insert("step".into(), json!(step));
    m.insert("t".into(), json!(round6(t)));
    for (k, v) in kv {
        if RESERVED_KEYS.contains(k) {
            return Err(ObsError::ReservedKey((*k).to_string()));
        }
        m.insert((*k).to_string(), if v.is_finite() { json!(v) } else { Value::Null });
    }
    Ok(Value::Object(m).to_string())
}

/// One array line (`kind`: tensor, trajectory, points, volume, histogram).
/// Validates shape, the size cap, labels and histogram edges.
#[allow(clippy::too_many_arguments)]
pub fn array_line(step: u64, t: f64, key: &str, kind: &str, shape: &[usize], data: &[f32], labels: Option<&[f32]>, edges: Option<&[f64]>) -> Result<String, ObsError> {
    let mismatch = || ObsError::ShapeMismatch { key: key.into(), shape: shape.to_vec(), len: data.len() };
    if shape.iter().product::<usize>() != data.len() {
        return Err(mismatch());
    }
    let cap = if kind == "histogram" { MAX_HISTOGRAM_BINS } else { MAX_ARRAY_ELEMS };
    if data.len() > cap {
        return Err(ObsError::TooLarge { key: key.into(), elems: data.len(), cap });
    }
    match kind {
        "trajectory" if shape.len() != 2 && shape.len() != 3 => return Err(mismatch()),
        "points" if shape.len() != 2 || labels.is_some_and(|l| l.len() != shape[0]) => return Err(mismatch()),
        "volume" if shape.len() != 3 => return Err(mismatch()),
        "histogram" if edges.is_none_or(|e| e.len() != data.len() + 1) => return Err(mismatch()),
        _ => {}
    }
    let mut line = json!({
        "step": step, "t": round6(t), "key": key, "kind": kind, "shape": shape,
        "dtype": "f32le", "b64": encode_f32(data),
    });
    if let Some(l) = labels {
        line["labels_b64"] = json!(encode_f32(l));
    }
    if let Some(e) = edges {
        line["edges"] = json!(e);
    }
    Ok(line.to_string())
}

pub fn event_line(step: u64, t: f64, text: &str) -> String {
    json!({ "step": step, "t": round6(t), "text": text }).to_string()
}

/// Pre-serialised records for one run, written by [`Run::append_batch`]
/// with a single `write_all` per file (the unit a server batches into).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Batch {
    pub metrics: Vec<String>,
    pub arrays: Vec<String>,
    pub events: Vec<String>,
    /// Last value per scalar key in this batch (feeds the auto-summary).
    pub last: Vec<(String, f64)>,
}

impl Batch {
    pub fn is_empty(&self) -> bool {
        self.metrics.is_empty() && self.arrays.is_empty() && self.events.is_empty()
    }
    pub fn len(&self) -> usize {
        self.metrics.len() + self.arrays.len() + self.events.len()
    }
    /// Add one W&B-style `log` call: scalars to one history line, the rest
    /// to array lines.
    pub fn push_log(&mut self, step: u64, t: f64, items: &[(String, Item)]) -> Result<(), ObsError> {
        let scalars: Vec<(&str, f64)> = items.iter().filter_map(|(k, v)| if let Item::Scalar(x) = v { Some((k.as_str(), *x)) } else { None }).collect();
        if !scalars.is_empty() {
            self.metrics.push(metric_line(step, t, &scalars)?);
            self.last.extend(scalars.iter().map(|(k, v)| (k.to_string(), *v)));
        }
        for (k, v) in items {
            match v {
                Item::Scalar(_) => {}
                Item::Tensor(x) => self.arrays.push(array_line(step, t, k, "tensor", &x.shape, &x.data, None, None)?),
                Item::Histogram(h) => self.arrays.push(array_line(step, t, k, "histogram", &[h.counts.len()], &h.counts, None, Some(&h.edges))?),
            }
        }
        Ok(())
    }
    pub fn push_event(&mut self, step: u64, t: f64, text: &str) {
        self.events.push(event_line(step, t, text));
    }
    pub fn extend(&mut self, other: Batch) {
        self.metrics.extend(other.metrics);
        self.arrays.extend(other.arrays);
        self.events.extend(other.events);
        self.last.extend(other.last);
    }
}

fn write_lines(f: &mut File, lines: &[String]) -> io::Result<()> {
    if lines.is_empty() {
        return Ok(());
    }
    let mut buf = String::with_capacity(lines.iter().map(|l| l.len() + 1).sum());
    for l in lines {
        buf.push_str(l);
        buf.push('\n');
    }
    f.write_all(buf.as_bytes())?;
    f.flush()
}

impl Run {
    /// W&B-style entry point (`wandb.init(project=…, name=…, config=…)`):
    /// creates `<root>/runs/<name>/` and records `project` in `config.json`.
    pub fn init(root: &Path, project: &str, name: &str, config: &impl Serialize) -> io::Result<Run> {
        let mut meta = Meta::detect();
        meta.project = Some(project.to_string());
        Run::create_with_meta(root, name, config, meta)
    }

    /// Create (or reopen for append) `<root>/runs/<name>/`, writing
    /// `config.json` with the given config and detected provenance.
    pub fn create(root: &Path, name: &str, config: &impl Serialize) -> io::Result<Run> {
        Run::create_with_meta(root, name, config, Meta::detect())
    }

    /// As [`Run::create`] but with explicit provenance (used when importing
    /// logs produced elsewhere, where local git/hostname would be wrong).
    pub fn create_with_meta(root: &Path, name: &str, config: &impl Serialize, meta: Meta) -> io::Result<Run> {
        Run::create_in(&root.join("runs").join(name), name, config, meta)
    }

    /// Create a run directly in `dir` (no `runs/<name>` nesting).
    pub fn create_in(dir: &Path, name: &str, config: &impl Serialize, meta: Meta) -> io::Result<Run> {
        std::fs::create_dir_all(dir)?;
        let cfg = json!({ "name": name, "config": config, "meta": meta });
        let tmp = dir.join("config.json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(&cfg)?)?;
        std::fs::rename(tmp, dir.join("config.json"))?;
        for f in JSONL_FILES {
            repair_tail(&dir.join(f))?;
        }
        Ok(Run {
            metrics: open_append(&dir.join("metrics.jsonl"))?,
            arrays: open_append(&dir.join("arrays.jsonl"))?,
            events: open_append(&dir.join("events.jsonl"))?,
            dir: dir.to_path_buf(),
            t0: Instant::now(),
            start_unix: meta.start_unix,
            summary: Map::new(),
        })
    }

    /// Reopen an existing run directory for append without touching its
    /// config: repairs torn tails, restores `start_unix` (so `t` continues
    /// from wall-clock start) and the summary.
    pub fn open_existing(dir: &Path) -> io::Result<Run> {
        let cfg: Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        let start_unix = cfg["meta"]["start_unix"].as_f64().unwrap_or_else(now_unix);
        for f in JSONL_FILES {
            repair_tail(&dir.join(f))?;
        }
        let summary = std::fs::read(dir.join("summary.json"))
            .ok()
            .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
            .and_then(|v| v["summary"].as_object().cloned())
            .unwrap_or_default();
        let behind = (now_unix() - start_unix).max(0.0);
        Ok(Run {
            metrics: open_append(&dir.join("metrics.jsonl"))?,
            arrays: open_append(&dir.join("arrays.jsonl"))?,
            events: open_append(&dir.join("events.jsonl"))?,
            dir: dir.to_path_buf(),
            t0: Instant::now().checked_sub(std::time::Duration::from_secs_f64(behind)).unwrap_or_else(Instant::now),
            start_unix,
            summary,
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Seconds since the run started (the `t` written with each record).
    pub fn elapsed(&self) -> f64 {
        self.t0.elapsed().as_secs_f64()
    }

    /// Write a [`Batch`]: one `write_all` + flush per non-empty file.
    pub fn append_batch(&mut self, b: &Batch) -> io::Result<()> {
        write_lines(&mut self.metrics, &b.metrics)?;
        write_lines(&mut self.arrays, &b.arrays)?;
        write_lines(&mut self.events, &b.events)?;
        for (k, v) in &b.last {
            self.summary.insert(k.clone(), if v.is_finite() { json!(v) } else { Value::Null });
        }
        Ok(())
    }

    /// Current byte length of each JSONL file ([`JSONL_FILES`] order). Every
    /// byte below these lengths belongs to a complete, flushed line.
    pub fn committed(&self) -> io::Result<[u64; 3]> {
        Ok([self.metrics.metadata()?.len(), self.arrays.metadata()?.len(), self.events.metadata()?.len()])
    }

    /// Merge `extra` into the summary and write the final `summary.json`
    /// without consuming the run (for long-lived owners such as a server).
    pub fn finish_in_place(&mut self, extra: &impl Serialize) -> io::Result<()> {
        self.merge_summary(extra)?;
        let e = self.elapsed();
        self.write_summary(Some(e))
    }

    /// Merge `extra` into the summary and rewrite `summary.json` (not
    /// finished). W&B `run.summary.update({...})` mid-run.
    pub fn update_summary(&mut self, extra: &impl Serialize) -> io::Result<()> {
        self.merge_summary(extra)?;
        self.write_summary(None)
    }

    /// Merge the keys of `patch` (a JSON object) into `config.json`'s
    /// `config`, atomically (temp file + rename). W&B `run.config.update`.
    pub fn update_config(&mut self, patch: &Value) -> io::Result<()> {
        let Value::Object(p) = patch else { return Ok(()) };
        let path = self.dir.join("config.json");
        let mut cfg: Value = serde_json::from_slice(&std::fs::read(&path)?)?;
        if !cfg["config"].is_object() {
            cfg["config"] = json!({});
        }
        let c = cfg["config"].as_object_mut().unwrap();
        for (k, v) in p {
            c.insert(k.clone(), v.clone());
        }
        let tmp = self.dir.join("config.json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(&cfg)?)?;
        std::fs::rename(tmp, path)
    }

    /// W&B-style `run.log({...}, step=…)`: scalars become one history line
    /// in `metrics.jsonl`; each histogram/tensor becomes one line in
    /// `arrays.jsonl`. Items: anything `Into<Item>` (`f64`, `f32`,
    /// [`Histogram`], [`Tensor`]).
    ///
    /// ```no_run
    /// # use modgrad_obs::{Run, Histogram, Tensor};
    /// # let mut run = Run::init(std::path::Path::new("obs"), "p", "r", &()).unwrap();
    /// run.log(10, [
    ///     ("loss", 0.42.into()),
    ///     ("w", Histogram::from_data(&[0.1, 0.2, 0.3], 32).into()),
    ///     ("recall_by_gap", Tensor::new(&[3], vec![0.9, 0.7, 0.4]).into()),
    /// ]).unwrap();
    /// ```
    pub fn log<'a>(&mut self, step: u64, items: impl IntoIterator<Item = (&'a str, Item)>) -> Result<(), ObsError> {
        let t = self.elapsed();
        let mut scalars: Vec<(&str, f64)> = Vec::new();
        let mut rest: Vec<(&str, Item)> = Vec::new();
        for (k, v) in items {
            match v {
                Item::Scalar(x) => scalars.push((k, x)),
                other => rest.push((k, other)),
            }
        }
        if !scalars.is_empty() {
            self.log_with_time(step, t, &scalars)?;
        }
        for (k, v) in rest {
            match v {
                Item::Tensor(x) => self.log_array_with_time(step, t, k, &x.shape, &x.data)?,
                Item::Histogram(h) => self.log_histogram_with_time(step, t, k, &h)?,
                Item::Scalar(_) => unreachable!(),
            }
        }
        Ok(())
    }

    /// Scalars only: one line `{"step":…, "t":…, "k":v, …}`. Non-finite
    /// values are written as `null` (JSON has no NaN) and read back as `NaN`.
    pub fn log_scalars(&mut self, step: u64, kv: &[(&str, f64)]) -> Result<(), ObsError> {
        let t = self.elapsed();
        self.log_with_time(step, t, kv)
    }

    /// As [`Run::log_scalars`] with an explicit `t` (seconds since run start).
    pub fn log_with_time(&mut self, step: u64, t: f64, kv: &[(&str, f64)]) -> Result<(), ObsError> {
        let line = metric_line(step, t, kv)?;
        for (k, v) in kv {
            self.summary.insert((*k).to_string(), if v.is_finite() { json!(v) } else { Value::Null });
        }
        write_line(&mut self.metrics, line)?;
        Ok(())
    }

    /// One array record: `shape` (row-major) plus base64 of little-endian
    /// f32 data. Fails above [`MAX_ARRAY_ELEMS`]; reduce with [`downsample`].
    pub fn log_array(&mut self, step: u64, key: &str, shape: &[usize], data: &[f32]) -> Result<(), ObsError> {
        let t = self.elapsed();
        self.log_array_with_time(step, t, key, shape, data)
    }

    pub fn log_array_with_time(&mut self, step: u64, t: f64, key: &str, shape: &[usize], data: &[f32]) -> Result<(), ObsError> {
        self.log_kind(step, t, key, "tensor", shape, data, None)
    }

    #[allow(clippy::too_many_arguments)]
    fn log_kind(&mut self, step: u64, t: f64, key: &str, kind: &str, shape: &[usize], data: &[f32], labels: Option<&[f32]>) -> Result<(), ObsError> {
        let line = array_line(step, t, key, kind, shape, data, labels, None)?;
        write_line(&mut self.arrays, line)?;
        Ok(())
    }

    /// Thinking trajectory: neuron state per tick for a few samples,
    /// `shape = [samples, ticks, neurons]` (or `[ticks, neurons]` for one
    /// sample). The exporter projects all states jointly onto their top 3
    /// principal components ([`pca3`]); keep `neurons` small (e.g. 256
    /// top-variance neurons, see [`top_variance_columns`]).
    pub fn log_trajectory(&mut self, step: u64, key: &str, shape: &[usize], data: &[f32]) -> Result<(), ObsError> {
        let t = self.elapsed();
        self.log_kind(step, t, key, "trajectory", shape, data, None)
    }

    /// Point cloud `[n, d]` (episode keys, slots …), optionally with one
    /// numeric label per point (e.g. 0 = stored, 1 = lure). Projected to 3D
    /// by PCA at export.
    pub fn log_points(&mut self, step: u64, key: &str, shape: &[usize], data: &[f32], labels: Option<&[f32]>) -> Result<(), ObsError> {
        let t = self.elapsed();
        self.log_kind(step, t, key, "points", shape, data, labels)
    }

    /// Volume `[ticks, h, w]` (e.g. maze path probability per tick), drawn
    /// as stacked slices.
    pub fn log_volume(&mut self, step: u64, key: &str, shape: &[usize], data: &[f32]) -> Result<(), ObsError> {
        let t = self.elapsed();
        self.log_kind(step, t, key, "volume", shape, data, None)
    }

    /// One histogram record (`kind: "histogram"`, counts as the f32 array
    /// of shape `[bins]`, plus `edges`).
    pub fn log_histogram(&mut self, step: u64, key: &str, h: &Histogram) -> Result<(), ObsError> {
        let t = self.elapsed();
        self.log_histogram_with_time(step, t, key, h)
    }

    pub fn log_histogram_with_time(&mut self, step: u64, t: f64, key: &str, h: &Histogram) -> Result<(), ObsError> {
        let line = array_line(step, t, key, "histogram", &[h.counts.len()], &h.counts, None, Some(&h.edges))?;
        write_line(&mut self.arrays, line)?;
        Ok(())
    }

    /// W&B-style `run.summary[key] = value`: overrides the automatic
    /// last-value summary for `key`. Persisted to `summary.json`
    /// immediately (atomic rename) and again at [`Run::finish`].
    pub fn summary(&mut self, key: &str, value: impl Serialize) -> io::Result<()> {
        self.summary.insert(key.to_string(), serde_json::to_value(value)?);
        self.write_summary(None)
    }

    /// Current in-memory summary.
    pub fn summary_map(&self) -> &Map<String, Value> {
        &self.summary
    }

    fn write_summary(&self, end: Option<f64>) -> io::Result<()> {
        let mut v = json!({ "summary": self.summary, "elapsed_s": round6(self.elapsed()) });
        if let Some(e) = end {
            v["end_unix"] = json!(self.start_unix + e);
            v["elapsed_s"] = json!(round6(e));
            v["finished"] = json!(true);
        }
        let tmp = self.dir.join("summary.json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(&v)?)?;
        std::fs::rename(tmp, self.dir.join("summary.json"))
    }

    /// Free-text event (phase change, checkpoint written, anomaly …).
    pub fn event(&mut self, step: u64, text: &str) -> Result<(), ObsError> {
        let t = self.elapsed();
        self.event_with_time(step, t, text)
    }

    pub fn event_with_time(&mut self, step: u64, t: f64, text: &str) -> Result<(), ObsError> {
        write_line(&mut self.events, event_line(step, t, text))?;
        Ok(())
    }

    /// W&B-style `run.finish()`: write the final `summary.json`
    /// (atomically: temp file + rename) and close the run.
    pub fn finish(self) -> io::Result<()> {
        let e = self.elapsed();
        self.write_summary(Some(e))
    }

    /// Merge `extra` (a JSON object; other values go under `"value"`) into
    /// the summary, then [`Run::finish`].
    pub fn finish_with(mut self, extra: &impl Serialize) -> io::Result<()> {
        self.merge_summary(extra)?;
        self.finish()
    }

    /// As [`Run::finish_with`] with an explicit elapsed time (imports).
    pub fn finish_with_time(mut self, extra: &impl Serialize, elapsed_s: f64) -> io::Result<()> {
        self.merge_summary(extra)?;
        self.write_summary(Some(elapsed_s))
    }

    fn merge_summary(&mut self, extra: &impl Serialize) -> io::Result<()> {
        match serde_json::to_value(extra)? {
            Value::Object(m) => self.summary.extend(m),
            Value::Null => {}
            v => {
                self.summary.insert("value".into(), v);
            }
        }
        Ok(())
    }
}

fn round6(x: f64) -> f64 {
    (x * 1e6).round() / 1e6
}

pub fn encode_f32(data: &[f32]) -> String {
    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
    B64.encode(bytes)
}

pub fn decode_f32(b64: &str) -> Option<Vec<f32>> {
    let bytes = B64.decode(b64).ok()?;
    if bytes.len() % 4 != 0 {
        return None;
    }
    Some(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

/// Analysis kernels shared with the wasm dashboard (re-exported).
pub use modgrad_obs_analysis::{downsample, pca3, select_columns, top_variance_columns};

// ---------------------------------------------------------------- reader

#[derive(Debug, Clone)]
pub struct MetricRow {
    pub step: u64,
    pub t: f64,
    pub values: BTreeMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct ArrayRecord {
    pub step: u64,
    pub t: f64,
    pub key: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
    /// `"tensor"` or `"histogram"`.
    pub kind: String,
    /// Histogram bin edges (`len == data.len() + 1`).
    pub edges: Option<Vec<f64>>,
    /// Per-point labels for `kind == "points"`.
    pub labels: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
pub struct EventRecord {
    pub step: u64,
    pub t: f64,
    pub text: String,
}

#[derive(Debug, Clone, Default)]
pub struct RunData {
    pub dir: PathBuf,
    /// Full `config.json` (`{name, config, meta}`), `Null` if absent.
    pub config: Value,
    pub metrics: Vec<MetricRow>,
    pub arrays: Vec<ArrayRecord>,
    pub events: Vec<EventRecord>,
    /// `summary.json` (`{summary, elapsed_s, finished?, end_unix?}`) if
    /// written; `finished` is set only by [`Run::finish`].
    pub summary: Option<Value>,
    /// Lines skipped: partial trailing line or unparsable.
    pub skipped: usize,
}

impl RunData {
    /// `(step, value)` for every row containing `key`, in file order.
    pub fn series(&self, key: &str) -> Vec<(u64, f64)> {
        self.metrics.iter().filter_map(|r| r.values.get(key).map(|v| (r.step, *v))).collect()
    }

    /// Sorted set of metric keys present.
    pub fn metric_keys(&self) -> Vec<String> {
        let mut s: std::collections::BTreeSet<String> = Default::default();
        for r in &self.metrics {
            s.extend(r.values.keys().cloned());
        }
        s.into_iter().collect()
    }

    /// All records of one array key, in file order.
    pub fn array_series(&self, key: &str) -> Vec<&ArrayRecord> {
        self.arrays.iter().filter(|a| a.key == key).collect()
    }

    pub fn array_keys(&self) -> Vec<String> {
        let s: std::collections::BTreeSet<String> = self.arrays.iter().map(|a| a.key.clone()).collect();
        s.into_iter().collect()
    }

    pub fn last_step(&self) -> Option<u64> {
        self.metrics.iter().map(|r| r.step).max()
    }

    pub fn name(&self) -> String {
        self.config.get("name").and_then(|v| v.as_str()).map(String::from).unwrap_or_else(|| {
            self.dir.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
        })
    }
}

/// Read complete JSON lines from `path`. A final segment without a
/// trailing newline (crash mid-write) and any unparsable line are skipped
/// and counted. A missing file yields no lines.
pub fn read_jsonl(path: &Path) -> io::Result<(Vec<Value>, usize)> {
    read_jsonl_limited(path, None)
}

/// As [`read_jsonl`] but reads at most the first `limit` bytes (a
/// committed length from [`Run::committed`]).
pub fn read_jsonl_limited(path: &Path, limit: Option<u64>) -> io::Result<(Vec<Value>, usize)> {
    let text = match std::fs::read(path) {
        Ok(mut b) => {
            if let Some(l) = limit {
                b.truncate(l as usize);
            }
            String::from_utf8_lossy(&b).into_owned()
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok((Vec::new(), 0)),
        Err(e) => return Err(e),
    };
    let mut out = Vec::new();
    let mut skipped = 0;
    let complete = match text.rfind('\n') {
        Some(i) => {
            if i + 1 < text.len() {
                skipped += 1; // partial trailing line
            }
            &text[..=i]
        }
        None => {
            if !text.is_empty() {
                skipped += 1;
            }
            ""
        }
    };
    for line in complete.lines() {
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<Value>(line) {
            Ok(v @ Value::Object(_)) => out.push(v),
            _ => skipped += 1,
        }
    }
    Ok((out, skipped))
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}

/// One parsed `arrays.jsonl` line (`None` if malformed).
pub fn array_record(v: &Value) -> Option<ArrayRecord> {
    let shape: Vec<usize> = v.get("shape")?.as_array()?.iter().map(|x| x.as_u64().map(|u| u as usize)).collect::<Option<_>>()?;
    let data = decode_f32(v.get("b64")?.as_str()?)?;
    if shape.iter().product::<usize>() != data.len() {
        return None;
    }
    Some(ArrayRecord {
        step: v.get("step")?.as_u64()?,
        t: v.get("t").map(num).unwrap_or(f64::NAN),
        key: v.get("key")?.as_str()?.to_string(),
        shape,
        data,
        kind: v.get("kind").and_then(|k| k.as_str()).unwrap_or("tensor").to_string(),
        edges: v.get("edges").and_then(|e| e.as_array()).map(|e| e.iter().map(num).collect()),
        labels: v.get("labels_b64").and_then(|b| b.as_str()).and_then(decode_f32),
    })
}

/// Read a run directory written by [`Run`].
pub fn read_run(dir: &Path) -> io::Result<RunData> {
    read_run_limited(dir, None)
}

/// Read a run, each JSONL file only up to the given committed byte length
/// ([`JSONL_FILES`] order) — a consistent snapshot while a writer appends.
pub fn read_run_limited(dir: &Path, limits: Option<[u64; 3]>) -> io::Result<RunData> {
    let lim = |i: usize| limits.map(|l| l[i]);
    let mut rd = RunData { dir: dir.to_path_buf(), ..Default::default() };
    rd.config = std::fs::read(dir.join("config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or(Value::Null);
    rd.summary = std::fs::read(dir.join("summary.json")).ok().and_then(|b| serde_json::from_slice(&b).ok());

    let (lines, s) = read_jsonl_limited(&dir.join("metrics.jsonl"), lim(0))?;
    rd.skipped += s;
    for v in lines {
        let Some(step) = v.get("step").and_then(|s| s.as_u64()) else {
            rd.skipped += 1;
            continue;
        };
        let t = v.get("t").map(num).unwrap_or(f64::NAN);
        let values = v
            .as_object()
            .unwrap()
            .iter()
            .filter(|(k, _)| !RESERVED_KEYS.contains(&k.as_str()))
            .filter(|(_, v)| v.is_number() || v.is_null())
            .map(|(k, v)| (k.clone(), num(v)))
            .collect();
        rd.metrics.push(MetricRow { step, t, values });
    }

    let (lines, s) = read_jsonl_limited(&dir.join("arrays.jsonl"), lim(1))?;
    rd.skipped += s;
    for v in lines {
        match array_record(&v) {
            Some(r) => rd.arrays.push(r),
            None => rd.skipped += 1,
        }
    }

    let (lines, s) = read_jsonl_limited(&dir.join("events.jsonl"), lim(2))?;
    rd.skipped += s;
    for v in lines {
        match (v.get("step").and_then(|x| x.as_u64()), v.get("text").and_then(|x| x.as_str())) {
            (Some(step), Some(text)) => {
                rd.events.push(EventRecord { step, t: v.get("t").map(num).unwrap_or(f64::NAN), text: text.into() })
            }
            _ => rd.skipped += 1,
        }
    }
    Ok(rd)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("modgrad-obs-test-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn lines_parse_and_series() {
        let root = tmpdir("parse");
        let mut run = Run::create(&root, "r1", &json!({"lr": 1e-3, "d": 256})).unwrap();
        for s in 0..5u64 {
            run.log(s, [("loss", (1.0 / (s as f64 + 1.0)).into()), ("acc", (s as f64 * 0.1).into())]).unwrap();
        }
        run.log_scalars(5, &[("loss", f64::NAN)]).unwrap();
        run.event(2, "checkpoint").unwrap();
        run.summary("best", 0.2).unwrap();
        run.finish().unwrap();

        let dir = root.join("runs/r1");
        for f in ["metrics.jsonl", "events.jsonl"] {
            for l in std::fs::read_to_string(dir.join(f)).unwrap().lines() {
                serde_json::from_str::<Value>(l).unwrap();
            }
        }
        let rd = read_run(&dir).unwrap();
        assert_eq!(rd.skipped, 0);
        assert_eq!(rd.name(), "r1");
        assert_eq!(rd.config["config"]["d"], 256);
        assert!(rd.config["meta"]["start_unix"].as_f64().unwrap() > 0.0);
        let loss = rd.series("loss");
        assert_eq!(loss.len(), 6);
        assert_eq!(loss[1], (1, 0.5));
        assert!(loss[5].1.is_nan());
        assert_eq!(rd.series("acc").len(), 5);
        assert_eq!(rd.metric_keys(), vec!["acc".to_string(), "loss".to_string()]);
        assert_eq!(rd.last_step(), Some(5));
        assert_eq!(rd.events[0].text, "checkpoint");
        let sm = rd.summary.unwrap();
        assert_eq!(sm["summary"]["best"], 0.2);
        assert_eq!(sm["summary"]["acc"], 0.4); // automatic last-value summary
        assert_eq!(sm["finished"], true);
        assert!(matches!(run_reserved(&root), Err(ObsError::ReservedKey(_))));
    }

    fn run_reserved(root: &Path) -> Result<(), ObsError> {
        let mut run = Run::create(root, "r2", &()).unwrap();
        run.log_scalars(0, &[("step", 1.0)])
    }

    #[test]
    fn partial_last_line_skipped() {
        let root = tmpdir("partial");
        let mut run = Run::create(&root, "r", &()).unwrap();
        for s in 0..3u64 {
            run.log_scalars(s, &[("x", s as f64)]).unwrap();
        }
        run.log_array(0, "a", &[2], &[1.0, 2.0]).unwrap();
        run.log_array(1, "a", &[2], &[3.0, 4.0]).unwrap();
        drop(run);
        let dir = root.join("runs/r");
        // simulate a crash mid-write: truncate inside the last line
        for f in ["metrics.jsonl", "arrays.jsonl"] {
            let p = dir.join(f);
            let len = std::fs::metadata(&p).unwrap().len();
            let file = OpenOptions::new().write(true).open(&p).unwrap();
            file.set_len(len - 7).unwrap();
        }
        let rd = read_run(&dir).unwrap();
        assert_eq!(rd.series("x"), vec![(0, 0.0), (1, 1.0)]);
        assert_eq!(rd.arrays.len(), 1);
        assert_eq!(rd.skipped, 2);
        // appending after a crash: the torn line stays torn but later lines parse
        let mut f = open_append(&dir.join("metrics.jsonl")).unwrap();
        write_line(&mut f, "garbage".into()).unwrap();
        let rd = read_run(&dir).unwrap();
        assert_eq!(rd.series("x").len(), 2);
    }

    #[test]
    fn torn_tail_repaired_on_reopen_and_batch_is_one_write() {
        let root = tmpdir("repair");
        let mut run = Run::create(&root, "r", &()).unwrap();
        let mut b = Batch::default();
        for s in 0..3u64 {
            b.push_log(s, 0.0, &[("x".into(), Item::Scalar(s as f64))]).unwrap();
        }
        b.push_event(2, 0.0, "e");
        run.append_batch(&b).unwrap();
        let before = run.committed().unwrap();
        drop(run);
        let dir = root.join("runs/r");
        // crash mid-batch: half a line at the end
        let mut f = open_append(&dir.join("metrics.jsonl")).unwrap();
        f.write_all(b"{\"step\":3,\"t\":0,\"x\"").unwrap();
        assert_eq!(read_run(&dir).unwrap().series("x").len(), 3);
        let mut run = Run::open_existing(&dir).unwrap();
        assert_eq!(run.committed().unwrap(), before);
        run.log_scalars(4, &[("x", 4.0)]).unwrap();
        let rd = read_run(&dir).unwrap();
        assert_eq!(rd.skipped, 0);
        assert_eq!(rd.series("x").iter().map(|p| p.0).collect::<Vec<_>>(), vec![0, 1, 2, 4]);
        assert_eq!(run.summary_map()["x"], 4.0);
        // limited read sees only the committed prefix
        let rd = read_run_limited(&dir, Some(before)).unwrap();
        assert_eq!(rd.series("x").len(), 3);
    }

    #[test]
    fn array_roundtrip() {
        let root = tmpdir("array");
        let mut run = Run::create(&root, "r", &()).unwrap();
        let data: Vec<f32> = (0..24).map(|i| (i as f32).sin() * 1e3 + f32::EPSILON).collect();
        let weird = [f32::MIN_POSITIVE, -0.0, f32::MAX, 1.0 / 3.0];
        run.log_array(7, "act", &[2, 3, 4], &data).unwrap();
        run.log_array(8, "weird", &[4], &weird).unwrap();
        assert!(matches!(run.log_array(9, "bad", &[3], &[1.0]), Err(ObsError::ShapeMismatch { .. })));
        let big = vec![0f32; MAX_ARRAY_ELEMS + 1];
        assert!(matches!(run.log_array(9, "big", &[big.len()], &big), Err(ObsError::TooLarge { .. })));
        let rd = read_run(&root.join("runs/r")).unwrap();
        let a = rd.array_series("act");
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].step, 7);
        assert_eq!(a[0].shape, vec![2, 3, 4]);
        assert!(a[0].data.iter().zip(&data).all(|(x, y)| x.to_bits() == y.to_bits()));
        let w = &rd.array_series("weird")[0].data;
        assert!(w.iter().zip(&weird).all(|(x, y)| x.to_bits() == y.to_bits()));
        assert_eq!(rd.array_keys(), vec!["act".to_string(), "weird".to_string()]);
    }

    #[test]
    fn wandb_style_items() {
        let root = tmpdir("items");
        let mut run = Run::init(&root, "proj", "r", &json!({"d": 8})).unwrap();
        let h = Histogram::from_data(&[0.0, 0.5, 1.0, 1.0, f32::NAN], 4);
        assert_eq!(h.counts, vec![1.0, 0.0, 1.0, 2.0]);
        run.log(3, [("loss", 1.5f32.into()), ("h", h.clone().into()), ("x", Tensor::new(&[2], vec![1.0, 2.0]).into())]).unwrap();
        run.finish().unwrap();
        let rd = read_run(&root.join("runs/r")).unwrap();
        assert_eq!(rd.config["meta"]["project"], "proj");
        assert_eq!(rd.series("loss"), vec![(3, 1.5)]);
        let hr = rd.array_series("h")[0];
        assert_eq!(hr.kind, "histogram");
        assert_eq!(hr.data, h.counts);
        assert_eq!(hr.edges.as_ref().unwrap(), &h.edges);
        assert_eq!(rd.array_series("x")[0].kind, "tensor");
    }

    #[test]
    fn downsample_caps_and_preserves_mean() {
        let data: Vec<f32> = (0..(7 * 33)).map(|i| i as f32).collect();
        let (shape, out) = downsample(&[7, 33], &data, 40);
        assert!(out.len() <= 40);
        assert_eq!(shape.iter().product::<usize>(), out.len());
        let (s1, d1) = downsample(&[5], &[1.0, 3.0, 5.0, 7.0, 9.0], 3);
        assert_eq!(s1, vec![3]);
        assert_eq!(d1, vec![2.0, 6.0, 9.0]);
        let (s2, d2) = downsample(&[3], &[1.0, 2.0, 3.0], 10);
        assert_eq!((s2, d2), (vec![3], vec![1.0, 2.0, 3.0]));
    }
}

pub mod export;
pub mod legacy;
pub mod sink;
pub use sink::{OpenSink, RunAt, Sink};
pub mod textlog;
