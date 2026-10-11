//! `Live`: change notifications for the dashboard (served as SSE).
//!
//! One task watches the snapshot channel that [`crate::State`] publishes
//! after every write. It keeps the committed byte lengths it last saw per
//! run and, on each new snapshot, reads only the bytes in between from
//! `metrics.jsonl` / `events.jsonl`. Those bytes are whole lines (the
//! committed-length invariant), and because the diff is against what this
//! task last saw, `watch` coalescing several writes into one wake-up loses
//! nothing. `State` is untouched; this is a reader like [`crate::ReadState`].
//!
//! Messages go out on a `broadcast` channel. A subscriber that falls
//! behind gets [`LiveKind::Resync`] and should refetch.

use crate::read::{ReadRequest, ReadResponse, ReadState};
use crate::state::Snapshot;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::broadcast;
use tower::ServiceExt;

/// Larger deltas are not shipped; subscribers get `Resync` for that run.
pub const MAX_DELTA_BYTES: u64 = 4 << 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiveKind {
    /// A run appeared; `data.entry` is its index entry (as in `api/index`).
    Run,
    /// New history rows: `data.rows` = `[{step, t, key: value, ...}]`.
    Rows,
    /// New events: `data.events` = `[{step, t, text}]`.
    Events,
    /// The run's summary changed (`data.finished` says whether it ended).
    Summary,
    /// Too much changed to ship; refetch `data.run` (or everything if absent).
    Resync,
}

impl LiveKind {
    pub fn name(self) -> &'static str {
        match self {
            LiveKind::Run => "run",
            LiveKind::Rows => "rows",
            LiveKind::Events => "events",
            LiveKind::Summary => "summary",
            LiveKind::Resync => "resync",
        }
    }
}

/// One change. `data` always holds `run` and `unix` (server time).
#[derive(Debug, Clone)]
pub struct LiveMsg {
    pub kind: LiveKind,
    pub run: String,
    pub data: Value,
}

#[derive(Clone)]
pub struct Live {
    tx: broadcast::Sender<Arc<LiveMsg>>,
    read: ReadState,
}

#[derive(Clone, Copy)]
struct Seen {
    committed: [u64; 3],
    finished: bool,
    summary_mtime: Option<std::time::SystemTime>,
    config_mtime: Option<std::time::SystemTime>,
}

/// Lines in `[from, to)` of a JSONL file, parsed (bad lines skipped).
fn read_lines(path: &Path, from: u64, to: u64) -> std::io::Result<Vec<Value>> {
    if to <= from {
        return Ok(Vec::new());
    }
    let mut f = std::fs::File::open(path)?;
    f.seek(SeekFrom::Start(from))?;
    let mut buf = vec![0u8; (to - from) as usize];
    f.read_exact(&mut buf)?;
    Ok(buf.split(|&b| b == b'\n').filter(|l| !l.is_empty()).filter_map(|l| serde_json::from_slice(l).ok()).collect())
}

fn mtime(p: &Path) -> Option<std::time::SystemTime> {
    std::fs::metadata(p).and_then(|m| m.modified()).ok()
}

pub fn unix_of(t: std::time::SystemTime) -> f64 {
    t.duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

impl Live {
    /// Spawn the diff task (needs a tokio runtime).
    pub fn spawn(read: ReadState) -> Live {
        let (tx, _) = broadcast::channel(1024);
        let live = Live { tx: tx.clone(), read: read.clone() };
        let mut rx = read.subscribe();
        let first = rx.borrow_and_update().clone();
        let mut seen: HashMap<String, Seen> = first
            .runs
            .iter()
            .map(|(k, c)| (k.clone(), Seen { committed: c.committed, finished: c.finished, summary_mtime: mtime(&c.dir.join("summary.json")), config_mtime: mtime(&c.dir.join("config.json")) }))
            .collect();
        let reader = live.clone();
        tokio::spawn(async move {
            while rx.changed().await.is_ok() {
                let snap = rx.borrow_and_update().clone();
                for m in reader.diff(&snap, &mut seen).await {
                    // no receivers is fine
                    let _ = tx.send(Arc::new(m));
                }
            }
        });
        live
    }

    pub fn subscribe(&self) -> broadcast::Receiver<Arc<LiveMsg>> {
        self.tx.subscribe()
    }

    /// Per-run state for a fresh subscriber: `{run, updated_unix,
    /// finished, last_step}` where `updated_unix` is the mtime of
    /// `metrics.jsonl` (when the last row landed).
    pub fn hello(&self, keep: &dyn Fn(&str) -> bool) -> Value {
        let snap = self.read.snapshot();
        let list: Vec<Value> = snap
            .runs
            .iter()
            .filter(|(k, _)| keep(k))
            .map(|(k, c)| json!({ "run": k, "finished": c.finished, "updated_unix": mtime(&c.dir.join("metrics.jsonl")).map(unix_of) }))
            .collect();
        json!({ "unix": modgrad_obs::now_unix(), "version": snap.version, "runs": list })
    }

    async fn entry(&self, run: &str) -> Option<Value> {
        match self.read.clone().oneshot(ReadRequest::ListRuns { filter: Some(run.to_string()) }).await {
            Ok(ReadResponse::Runs(v)) => v.into_iter().find(|e| e["id"] == run),
            _ => None,
        }
    }

    async fn diff(&self, snap: &Snapshot, seen: &mut HashMap<String, Seen>) -> Vec<LiveMsg> {
        let now = modgrad_obs::now_unix();
        let mut out = Vec::new();
        let msg = |kind, run: &str, mut data: Value| {
            data["run"] = json!(run);
            data["unix"] = json!(now);
            LiveMsg { kind, run: run.to_string(), data }
        };
        for (run, c) in &snap.runs {
            let fresh = !seen.contains_key(run);
            let prev = seen.get(run).copied().unwrap_or(Seen { committed: [0; 3], finished: false, summary_mtime: None, config_mtime: None });
            let cm = mtime(&c.dir.join("config.json"));
            // the entry carries config, the arrays index and last_step: re-send
            // it when config changes after the first rows or arrays land
            if fresh || cm != prev.config_mtime || c.committed[1] != prev.committed[1] {
                let entry = self.entry(run).await.unwrap_or(Value::Null);
                out.push(msg(LiveKind::Run, run, json!({ "entry": entry })));
            }
            let mut resync = false;
            for (i, kind, field) in [(0usize, LiveKind::Rows, "rows"), (2, LiveKind::Events, "events")] {
                let (from, to) = (prev.committed[i], c.committed[i]);
                if to <= from || fresh {
                    // a fresh run's existing lines are already in its entry
                    continue;
                }
                if to - from > MAX_DELTA_BYTES {
                    resync = true;
                    continue;
                }
                let path = c.dir.join(modgrad_obs::JSONL_FILES[i]);
                let p2 = path.clone();
                match tokio::task::spawn_blocking(move || read_lines(&p2, from, to)).await {
                    Ok(Ok(lines)) if !lines.is_empty() => out.push(msg(kind, run, json!({ field: lines }))),
                    Ok(Ok(_)) => {}
                    _ => resync = true,
                }
            }
            if resync {
                out.push(msg(LiveKind::Resync, run, json!({})));
            }
            let sm = mtime(&c.dir.join("summary.json"));
            if !fresh && (c.finished != prev.finished || sm != prev.summary_mtime) {
                let summary = std::fs::read(c.dir.join("summary.json")).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok()).unwrap_or(Value::Null);
                out.push(msg(LiveKind::Summary, run, json!({ "finished": c.finished, "summary": summary })));
            }
            seen.insert(run.clone(), Seen { committed: c.committed, finished: c.finished, summary_mtime: sm, config_mtime: cm });
        }
        out
    }
}
