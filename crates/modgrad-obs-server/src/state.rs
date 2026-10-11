//! `State`: the single writer. Owns every run's file handles; meant to sit
//! behind `tower::buffer::Buffer` so all callers share one owner and no
//! locks are needed. After each successful write it publishes a new
//! [`Snapshot`] (committed byte lengths) for [`crate::ReadState`].

use crate::error::StateError;
use modgrad_obs::{Batch, Item, Meta, Run};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::future::{Ready, ready};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::sync::watch;

/// A `step` of `AUTO_STEP` means "next": for a history row, one past the
/// run's last logged step (0 for an empty run); for an array or event, the
/// last logged step. Resolved by [`State`], the only place that knows it.
pub const AUTO_STEP: u64 = u64::MAX;

/// One record to append, before serialisation. `t: None` means "now"
/// (seconds since the run started, stamped by the writer).
#[derive(Debug, Clone, PartialEq)]
pub enum Record {
    Log { step: u64, t: Option<f64>, items: Vec<(String, Item)> },
    Array { step: u64, t: Option<f64>, key: String, kind: String, shape: Vec<usize>, data: Vec<f32>, labels: Option<Vec<f32>> },
    Event { step: u64, t: Option<f64>, text: String },
    /// A text sample (generation), appended to `samples.jsonl`; its step
    /// resolves like an event's.
    Sample { step: u64, t: Option<f64>, sample: modgrad_obs::Sample },
}

#[derive(Debug, Clone)]
pub enum StateRequest {
    /// `meta`: client provenance (`git`, `hostname`, anything else goes to
    /// `meta.extra`). The server's own git checkout says nothing about the
    /// client, so `git` is only ever what the client sent.
    CreateRun { name: String, project: Option<String>, config: Value, meta: Option<Value> },
    /// All records for one run, written as one batch (one `write_all` per file).
    Append { run: String, records: Vec<Record> },
    Finish { run: String, summary: Value },
    /// Merge into the summary without finishing.
    Summary { run: String, summary: Value },
    /// Merge keys into `config.json`'s `config`.
    Config { run: String, config: Value },
}

#[derive(Debug, Clone, PartialEq)]
pub enum StateResponse {
    Created { run: String },
    Appended { run: String, records: usize, committed: [u64; 3] },
    Finished { run: String },
    Updated { run: String },
}

/// What a reader may see of one run.
#[derive(Debug, Clone, PartialEq)]
pub struct RunCommit {
    pub dir: PathBuf,
    /// Committed byte length of metrics/arrays/events.jsonl.
    pub committed: [u64; 3],
    pub finished: bool,
}

/// Immutable view published after every write.
#[derive(Debug, Clone, Default)]
pub struct Snapshot {
    pub root: PathBuf,
    pub runs: BTreeMap<String, RunCommit>,
    /// Bumped on every publish.
    pub version: u64,
}

pub struct State {
    root: PathBuf,
    runs: HashMap<String, Run>,
    /// Last history step per run (for [`AUTO_STEP`]).
    last_step: HashMap<String, u64>,
    snap: Snapshot,
    tx: watch::Sender<Arc<Snapshot>>,
}

/// Run names are paths under `runs/`: 1 to 3 segments of `[A-Za-z0-9_.-]`.
pub fn valid_run_name(name: &str) -> bool {
    let segs: Vec<&str> = name.split('/').collect();
    !name.is_empty()
        && segs.len() <= 3
        && segs.iter().all(|s| !s.is_empty() && *s != "." && *s != ".." && s.chars().all(|c| c.is_ascii_alphanumeric() || "_.-".contains(c)))
}

impl State {
    /// Open (or create) a store at `root`, reopening every existing run
    /// under `root/runs` (torn tails are repaired). Returns the writer and
    /// the snapshot channel a [`crate::ReadState`] is built from.
    pub fn open(root: &Path) -> Result<(State, watch::Receiver<Arc<Snapshot>>), StateError> {
        let runs_dir = root.join("runs");
        std::fs::create_dir_all(&runs_dir)?;
        let mut runs = HashMap::new();
        let mut last_step = HashMap::new();
        let mut snap = Snapshot { root: root.to_path_buf(), ..Default::default() };
        for dir in modgrad_obs::export::find_runs(&runs_dir) {
            let name = dir.strip_prefix(&runs_dir).unwrap().to_string_lossy().replace('\\', "/");
            if !valid_run_name(&name) {
                continue;
            }
            let run = Run::open_existing(&dir)?;
            if let Some(s) = last_logged_step(&dir) {
                last_step.insert(name.clone(), s);
            }
            snap.runs.insert(name.clone(), RunCommit { dir: dir.clone(), committed: run.committed()?, finished: finished(&dir) });
            runs.insert(name, run);
        }
        let (tx, rx) = watch::channel(Arc::new(snap.clone()));
        Ok((State { root: root.to_path_buf(), runs, last_step, snap, tx }, rx))
    }

    fn publish(&mut self) {
        self.snap.version += 1;
        // send_replace never fails, even with no receivers
        self.tx.send_replace(Arc::new(self.snap.clone()));
    }

    fn handle(&mut self, req: StateRequest) -> Result<StateResponse, StateError> {
        match req {
            StateRequest::CreateRun { name, project, config, meta: client } => {
                if !valid_run_name(&name) {
                    return Err(StateError::InvalidRunName(name));
                }
                let dir = self.root.join("runs").join(&name);
                let run = if dir.join("config.json").is_file() {
                    // idempotent: a client reconnecting resumes the run
                    Run::open_existing(&dir)?
                } else {
                    let mut meta = Meta::detect();
                    meta.project = project;
                    meta.git = None; // the server's checkout says nothing about the client
                    if let Some(Value::Object(mut c)) = client {
                        if let Some(g) = c.remove("git") {
                            meta.git = g.as_str().map(String::from);
                        }
                        if let Some(h) = c.remove("hostname").and_then(|h| h.as_str().map(String::from)) {
                            meta.hostname = Some(h);
                        }
                        if !c.is_empty() {
                            meta.extra = Some(Value::Object(c));
                        }
                    }
                    Run::create_in(&dir, &name, &config, meta)?
                };
                if let Some(s) = last_logged_step(&dir) {
                    self.last_step.entry(name.clone()).or_insert(s);
                }
                self.snap.runs.insert(name.clone(), RunCommit { dir, committed: run.committed()?, finished: false });
                self.runs.insert(name.clone(), run);
                self.publish();
                Ok(StateResponse::Created { run: name })
            }
            StateRequest::Append { run, records } => {
                let r = self.runs.get_mut(&run).ok_or_else(|| StateError::UnknownRun(run.clone()))?;
                // serialise and validate everything first: an invalid record
                // rejects the whole append before any byte is written
                let now = r.elapsed();
                let mut last = self.last_step.get(&run).copied();
                let mut b = Batch::default();
                for rec in &records {
                    match rec {
                        Record::Log { step, t, items } => {
                            let s = if *step == AUTO_STEP { last.map_or(0, |l| l + 1) } else { *step };
                            b.push_log(s, t.unwrap_or(now), items)?;
                            last = Some(last.map_or(s, |l| l.max(s)));
                        }
                        Record::Array { step, t, key, kind, shape, data, labels } => {
                            let s = if *step == AUTO_STEP { last.unwrap_or(0) } else { *step };
                            b.arrays.push(modgrad_obs::array_line(s, t.unwrap_or(now), key, kind, shape, data, labels.as_deref(), None)?)
                        }
                        Record::Event { step, t, text } => b.push_event(if *step == AUTO_STEP { last.unwrap_or(0) } else { *step }, t.unwrap_or(now), text),
                        Record::Sample { step, t, sample } => b.push_sample(if *step == AUTO_STEP { last.unwrap_or(0) } else { *step }, t.unwrap_or(now), sample),
                    }
                }
                r.append_batch(&b)?;
                if let Some(l) = last {
                    self.last_step.insert(run.clone(), l);
                }
                let committed = r.committed()?;
                if let Some(c) = self.snap.runs.get_mut(&run) {
                    c.committed = committed;
                }
                self.publish();
                Ok(StateResponse::Appended { run, records: records.len(), committed })
            }
            StateRequest::Finish { run, summary } => {
                let r = self.runs.get_mut(&run).ok_or_else(|| StateError::UnknownRun(run.clone()))?;
                r.finish_in_place(&summary)?;
                if let Some(c) = self.snap.runs.get_mut(&run) {
                    c.finished = true;
                }
                self.publish();
                Ok(StateResponse::Finished { run })
            }
            StateRequest::Summary { run, summary } => {
                self.runs.get_mut(&run).ok_or_else(|| StateError::UnknownRun(run.clone()))?.update_summary(&summary)?;
                self.publish();
                Ok(StateResponse::Updated { run })
            }
            StateRequest::Config { run, config } => {
                self.runs.get_mut(&run).ok_or_else(|| StateError::UnknownRun(run.clone()))?.update_config(&config)?;
                self.publish();
                Ok(StateResponse::Updated { run })
            }
        }
    }
}

/// Step of the last complete line of `metrics.jsonl` (reads the tail only).
fn last_logged_step(dir: &Path) -> Option<u64> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(dir.join("metrics.jsonl")).ok()?;
    let len = f.metadata().ok()?.len();
    let from = len.saturating_sub(64 * 1024);
    f.seek(SeekFrom::Start(from)).ok()?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).ok()?;
    buf.split(|&b| b == b'\n').rev().filter(|l| !l.is_empty()).find_map(|l| serde_json::from_slice::<Value>(l).ok()?["step"].as_u64())
}

fn finished(dir: &Path) -> bool {
    std::fs::read(dir.join("summary.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
        .is_some_and(|v| v["finished"] == true)
}

impl tower::Service<StateRequest> for State {
    type Response = StateResponse;
    type Error = StateError;
    type Future = Ready<Result<StateResponse, StateError>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), StateError>> {
        Poll::Ready(Ok(()))
    }

    /// Writes are small appends done synchronously by the single owner (the
    /// `Buffer` worker); the response future is already resolved.
    fn call(&mut self, req: StateRequest) -> Self::Future {
        let span = tracing::debug_span!("state", kind = req_kind(&req));
        let _g = span.enter();
        ready(self.handle(req))
    }
}

fn req_kind(r: &StateRequest) -> &'static str {
    match r {
        StateRequest::CreateRun { .. } => "create_run",
        StateRequest::Append { .. } => "append",
        StateRequest::Finish { .. } => "finish",
        StateRequest::Summary { .. } => "summary",
        StateRequest::Config { .. } => "config",
    }
}
