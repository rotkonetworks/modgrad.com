//! `ReadState`: clonable, read-only, selection-only access over immutable
//! snapshots. Each request reads JSONL files only up to the committed
//! lengths of the snapshot it started from, so it never observes a partial
//! line and needs no lock. No smoothing, aggregation or PCA here.

use crate::error::ReadError;
use crate::state::{RunCommit, Snapshot};
use modgrad_obs::RunData;
use serde_json::Value;
use std::future::Future;
use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::sync::watch;

#[derive(Debug, Clone)]
pub enum ReadRequest {
    /// Index entries (same schema as the static `index.json` runs), runs
    /// whose name contains `filter`.
    ListRuns { filter: Option<String> },
    Config(String),
    Summary(String),
    /// Rows with `range.start <= step < range.end`, every `stride`-th row
    /// (last kept), only `keys` (empty = all keys).
    History { run: String, keys: Vec<String>, range: Range<u64>, stride: usize },
    /// Records of `key` at `steps` (None = all), each cut to the flat
    /// element range `index_range` (None = whole record).
    Arrays { run: String, key: String, steps: Option<Vec<u64>>, index_range: Option<Range<usize>> },
    /// Last 200 events.
    Events(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct HistoryBlock {
    pub steps: Vec<u64>,
    pub t: Vec<f64>,
    /// One column per key, `NaN` where a row lacks the key.
    pub columns: Vec<(String, Vec<f64>)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ArrayBlock {
    pub step: u64,
    pub t: f64,
    pub kind: String,
    /// Shape of the full record.
    pub shape: Vec<usize>,
    /// Flat offset of `data[0]` within the record.
    pub offset: usize,
    pub data: Vec<f32>,
    pub edges: Option<Vec<f64>>,
    pub labels: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
pub enum ReadResponse {
    Runs(Vec<Value>),
    Config(Value),
    Summary(Option<Value>),
    History(HistoryBlock),
    Arrays { key: String, blocks: Vec<ArrayBlock> },
    Events(Vec<Value>),
}

#[derive(Clone)]
pub struct ReadState {
    rx: watch::Receiver<Arc<Snapshot>>,
}

impl ReadState {
    pub fn new(rx: watch::Receiver<Arc<Snapshot>>) -> ReadState {
        ReadState { rx }
    }
    pub fn snapshot(&self) -> Arc<Snapshot> {
        self.rx.borrow().clone()
    }
    /// A receiver that wakes on every published snapshot (for [`crate::Live`]).
    pub fn subscribe(&self) -> watch::Receiver<Arc<Snapshot>> {
        self.rx.clone()
    }
}

fn load(c: &RunCommit) -> Result<RunData, ReadError> {
    Ok(modgrad_obs::read_run_limited(&c.dir, Some(c.committed))?)
}

fn commit<'a>(snap: &'a Snapshot, run: &str) -> Result<&'a RunCommit, ReadError> {
    snap.runs.get(run).ok_or_else(|| ReadError::UnknownRun(run.to_string()))
}

fn serve(snap: &Snapshot, req: ReadRequest) -> Result<ReadResponse, ReadError> {
    match req {
        ReadRequest::ListRuns { filter } => {
            let mut out = Vec::new();
            for (name, c) in &snap.runs {
                if filter.as_deref().is_some_and(|f| !name.contains(f)) {
                    continue;
                }
                let rd = load(c)?;
                let rel = format!("runs/{name}");
                let source = modgrad_obs::export::source_of(&rd, Some(name));
                out.push(modgrad_obs::export::build_run(&rd, &rel, &source).0);
            }
            Ok(ReadResponse::Runs(out))
        }
        ReadRequest::Config(run) => {
            let c = commit(snap, &run)?;
            Ok(ReadResponse::Config(serde_json::from_slice(&std::fs::read(c.dir.join("config.json"))?).unwrap_or(Value::Null)))
        }
        ReadRequest::Summary(run) => {
            let c = commit(snap, &run)?;
            Ok(ReadResponse::Summary(std::fs::read(c.dir.join("summary.json")).ok().and_then(|b| serde_json::from_slice(&b).ok())))
        }
        ReadRequest::History { run, keys, range, stride } => {
            let rd = load(commit(snap, &run)?)?;
            let keys = if keys.is_empty() { rd.metric_keys() } else { keys };
            // merge rows by step (later wins), as the static export does
            let mut rows: std::collections::BTreeMap<u64, (f64, std::collections::BTreeMap<&str, f64>)> = Default::default();
            for r in &rd.metrics {
                if !range.contains(&r.step) {
                    continue;
                }
                let e = rows.entry(r.step).or_insert((r.t, Default::default()));
                e.0 = r.t;
                for k in &keys {
                    if let Some(v) = r.values.get(k) {
                        e.1.insert(k.as_str(), *v);
                    }
                }
            }
            let rows: Vec<_> = rows.into_iter().filter(|(_, (_, m))| !m.is_empty()).collect();
            let n = rows.len();
            let stride = stride.max(1);
            let pick: Vec<usize> = (0..n).filter(|i| i % stride == 0 || i + 1 == n).collect();
            let mut b = HistoryBlock { steps: Vec::new(), t: Vec::new(), columns: keys.iter().map(|k| (k.clone(), Vec::new())).collect() };
            for i in pick {
                let (s, (t, m)) = &rows[i];
                b.steps.push(*s);
                b.t.push(*t);
                for (k, col) in b.columns.iter_mut() {
                    col.push(m.get(k.as_str()).copied().unwrap_or(f64::NAN));
                }
            }
            Ok(ReadResponse::History(b))
        }
        ReadRequest::Arrays { run, key, steps, index_range } => {
            let rd = load(commit(snap, &run)?)?;
            let recs = rd.array_series(&key);
            if recs.is_empty() {
                return Err(ReadError::UnknownKey { run, key });
            }
            let blocks = recs
                .into_iter()
                .filter(|r| steps.as_ref().is_none_or(|s| s.contains(&r.step)))
                .map(|r| {
                    let (lo, hi) = match &index_range {
                        Some(ir) => (ir.start.min(r.data.len()), ir.end.min(r.data.len()).max(ir.start.min(r.data.len()))),
                        None => (0, r.data.len()),
                    };
                    ArrayBlock { step: r.step, t: r.t, kind: r.kind.clone(), shape: r.shape.clone(), offset: lo, data: r.data[lo..hi].to_vec(), edges: r.edges.clone(), labels: r.labels.clone() }
                })
                .collect();
            Ok(ReadResponse::Arrays { key, blocks })
        }
        ReadRequest::Events(run) => {
            let rd = load(commit(snap, &run)?)?;
            Ok(ReadResponse::Events(
                rd.events.iter().rev().take(200).rev().map(|e| serde_json::json!({"step": e.step, "t": if e.t.is_finite() { serde_json::json!(e.t) } else { Value::Null }, "text": e.text})).collect(),
            ))
        }
    }
}

impl tower::Service<ReadRequest> for ReadState {
    type Response = ReadResponse;
    type Error = ReadError;
    type Future = Pin<Box<dyn Future<Output = Result<ReadResponse, ReadError>> + Send>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), ReadError>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: ReadRequest) -> Self::Future {
        let snap = self.snapshot();
        let span = tracing::debug_span!("read", version = snap.version);
        Box::pin(async move {
            let _g = span.enter();
            tokio::task::spawn_blocking(move || serve(&snap, req)).await.map_err(|e| ReadError::Join(e.to_string()))?
        })
    }
}
