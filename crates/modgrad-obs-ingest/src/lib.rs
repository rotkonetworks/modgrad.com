//! Drag-and-drop ingest for modgrad-obs: training logs from anywhere,
//! parsed into an in-memory run the dashboard can graph. Runs in the
//! browser (wasm, in a Web Worker); nothing is uploaded.
//!
//! Formats ([`detect`]):
//!
//! * CSV / TSV ([`csv`]): header row, a step-like column, numeric columns
//!   become metrics.
//! * JSONL ([`jsonl`]): one object per line, nested objects flattened with
//!   `/` (a top-level `metrics` wrapper is unwrapped).
//! * plain-text training logs ([`text`]): `key value` / `key=value` /
//!   `key: value` heuristics; the `ctmlm_pilot` log format is tried first
//!   (via `modgrad_obs::legacy`).
//! * TensorBoard event files ([`tfevents`]).
//! * W&B offline `.wandb` files (via `modgrad_obs_wandb`).
//! * modgrad-obs run directories: `config.json`, `summary.json`,
//!   `metrics.jsonl`, `events.jsonl`, `arrays.jsonl` dropped together.
//!
//! Text formats are streamed ([`Ingest::push`] per chunk, the partial last
//! line carried over) so a 100 MB log never sits in memory as one string.

pub mod csv;
pub mod jsonl;
pub mod num;
pub mod text;
pub mod tfevents;
#[cfg(feature = "wasm")]
pub mod wasm;

use modgrad_obs::{ArrayRecord, EventRecord, Histogram, Meta, ObsError, OpenSink, Sink};
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, HashMap};
use std::io;

/// One metric: points in arrival order until [`LocalRun::finalize`] sorts
/// them by step (duplicate steps keep the last value).
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Series {
    pub step: Vec<f64>,
    pub value: Vec<f64>,
}

/// What one file contributed.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct FileReport {
    pub file: String,
    pub format: String,
    pub bytes: u64,
    pub lines: usize,
    /// Lines / rows / records that were not used.
    pub skipped: usize,
    pub metrics: usize,
    pub points: usize,
    pub steps: usize,
    pub notes: Vec<String>,
    /// Set when the file could not be parsed at all.
    pub error: Option<String>,
}

impl FileReport {
    pub fn to_json(&self) -> Value {
        json!({
            "file": self.file, "format": self.format, "bytes": self.bytes, "lines": self.lines,
            "skipped": self.skipped, "metrics": self.metrics, "points": self.points, "steps": self.steps,
            "notes": self.notes, "error": self.error,
        })
    }
}

/// A run held in memory.
#[derive(Debug, Default, Clone)]
pub struct LocalRun {
    pub name: String,
    /// The run's config (flat `config` object of `config.json`).
    pub config: Map<String, Value>,
    pub meta: Map<String, Value>,
    pub summary: Map<String, Value>,
    pub finished: bool,
    pub series: BTreeMap<String, Series>,
    /// (step, seconds since start), when the source has a clock.
    pub time: Vec<(f64, f64)>,
    pub arrays: Vec<ArrayRecord>,
    pub events: Vec<EventRecord>,
    pub files: Vec<FileReport>,
    /// Metric rows (distinct step per file contribution, for the table).
    pub rows: usize,
}

fn clean(v: f64) -> f64 {
    if v.is_finite() { v } else { f64::NAN }
}

impl LocalRun {
    pub fn new(name: &str) -> LocalRun {
        LocalRun { name: name.to_string(), ..Default::default() }
    }

    /// Append one point. Non-finite values are stored as NaN (a gap, as in
    /// the run format where they are written as `null`).
    pub fn push(&mut self, key: &str, step: f64, v: f64) {
        let s = match self.series.get_mut(key) {
            Some(s) => s,
            None => self.series.entry(key.to_string()).or_default(),
        };
        s.step.push(step);
        s.value.push(clean(v));
    }

    pub fn push_time(&mut self, step: f64, t: f64) {
        if t.is_finite() {
            self.time.push((step, t));
        }
    }

    /// Sort every series by step and drop duplicate steps (last wins),
    /// fill the last-value summary, and return the number of dropped
    /// duplicates. Idempotent.
    pub fn finalize(&mut self) -> usize {
        let mut dups = 0;
        for s in self.series.values_mut() {
            dups += sort_dedup(&mut s.step, &mut s.value);
        }
        let (mut ts, mut tv): (Vec<f64>, Vec<f64>) = self.time.iter().copied().unzip();
        sort_dedup(&mut ts, &mut tv);
        self.time = ts.into_iter().zip(tv).collect();
        for (k, s) in &self.series {
            if !self.summary.contains_key(k) {
                if let Some(v) = s.value.iter().rev().find(|v| v.is_finite()) {
                    self.summary.insert(k.clone(), json!(v));
                }
            }
        }
        dups
    }

    pub fn last_step(&self) -> Option<f64> {
        self.series.values().filter_map(|s| s.step.last().copied()).fold(None, |a: Option<f64>, b| Some(a.map_or(b, |a| a.max(b))))
    }

    /// The dashboard's per-run metadata (everything except the metric
    /// arrays, which go out as typed arrays): index-entry fields, arrays
    /// index plus their plot JSON, events, and the per-file reports.
    pub fn meta_json(&self) -> Value {
        let metrics: Map<String, Value> = self.series.iter().map(|(k, s)| (k.clone(), json!({ "n": s.step.len() }))).collect();
        let rd = modgrad_obs::RunData { arrays: self.arrays.clone(), ..Default::default() };
        let mut arrays = Map::new();
        let mut array_files = Map::new();
        for k in rd.array_keys() {
            if let Some(v) = modgrad_obs::export::array_json(&rd, &k, 200) {
                let recs = rd.array_series(&k);
                arrays.insert(k.clone(), json!({ "kind": recs[0].kind, "shape": recs[recs.len() - 1].shape, "n_steps": recs.len() }));
                array_files.insert(k, v);
            }
        }
        let events: Vec<Value> = self.events.iter().rev().take(200).rev().map(|e| json!({ "step": e.step, "t": if e.t.is_finite() { json!(e.t) } else { Value::Null }, "text": e.text })).collect();
        json!({
            "name": self.name,
            "config": self.config,
            "meta": self.meta,
            "summary": self.summary,
            "finished": self.finished,
            "last_step": self.last_step(),
            "last_t": self.time.last().map(|p| p.1),
            "n_rows": self.rows,
            "n_events": self.events.len(),
            "metrics": metrics,
            "arrays": arrays,
            "array_files": array_files,
            "events": events,
            "skipped_lines": self.files.iter().map(|f| f.skipped).sum::<usize>(),
            "files": self.files.iter().map(FileReport::to_json).collect::<Vec<_>>(),
        })
    }
}

/// Stable sort by step, then keep the last value of each run of equal
/// steps. Returns how many points were dropped.
fn sort_dedup(step: &mut Vec<f64>, value: &mut Vec<f64>) -> usize {
    let n = step.len();
    let sorted = step.windows(2).all(|w| w[0] < w[1]);
    if sorted {
        return 0;
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| step[a].total_cmp(&step[b]));
    let (mut s2, mut v2) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for &i in &idx {
        if s2.last() == Some(&step[i]) {
            *v2.last_mut().unwrap() = value[i];
        } else {
            s2.push(step[i]);
            v2.push(value[i]);
        }
    }
    let dropped = n - s2.len();
    *step = s2;
    *value = v2;
    dropped
}

// ------------------------------------------------------------------ rows

/// Which column/token names mean "step". Order is priority.
pub const STEP_KEYS: &[&str] = &["_step", "step", "global_step", "globalstep", "trainer/global_step", "train/global_step", "iter", "iteration", "it", "steps", "iters", "num_steps", "update", "updates"];
pub const EPOCH_KEYS: &[&str] = &["epoch", "epochs", "trainer/epoch", "train/epoch"];

pub fn step_rank(key: &str) -> Option<usize> {
    let k = key.to_ascii_lowercase();
    STEP_KEYS.iter().position(|s| *s == k)
}
pub fn is_epoch(key: &str) -> bool {
    EPOCH_KEYS.contains(&key.to_ascii_lowercase().as_str())
}

/// Row-oriented accumulator shared by the JSONL and text parsers: the x
/// axis (explicit step, epoch, or line order) is chosen once all rows are
/// in.
#[derive(Default)]
pub struct Rows {
    names: Vec<String>,
    ids: HashMap<String, u32>,
    rows: Vec<Row>,
}

#[derive(Default)]
pub struct Row {
    pub step: Option<f64>,
    pub epoch: Option<f64>,
    pub t: Option<f64>,
    pub kv: Vec<(u32, f64)>,
}

/// What to do with rows that have no explicit step when others do.
#[derive(Clone, Copy, PartialEq)]
pub enum Missing {
    /// Use the previous row's step (JSONL: deliberate rows).
    CarryForward,
    /// Drop them (text: they are most likely not metric lines).
    Drop,
}

impl Rows {
    pub fn id(&mut self, key: &str) -> u32 {
        if let Some(&i) = self.ids.get(key) {
            return i;
        }
        let i = self.names.len() as u32;
        self.names.push(key.to_string());
        self.ids.insert(key.to_string(), i);
        i
    }
    pub fn push(&mut self, row: Row) {
        self.rows.push(row);
    }
    pub fn len(&self) -> usize {
        self.rows.len()
    }
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Emit into `run`. `min_count`: drop keys seen in fewer rows (noise
    /// filter for text logs). Returns the rows used.
    pub fn finish(self, run: &mut LocalRun, rep: &mut FileReport, missing: Missing, min_count: usize) -> usize {
        let n = self.rows.len();
        let explicit = self.rows.iter().filter(|r| r.step.is_some()).count();
        let with_epoch = self.rows.iter().filter(|r| r.epoch.is_some()).count();
        // epoch as x only when it nearly always distinguishes rows
        let epoch_x = explicit == 0 && with_epoch == n && n > 0 && {
            let mut e: Vec<f64> = self.rows.iter().filter_map(|r| r.epoch).collect();
            e.dedup();
            e.len() * 10 >= n * 9
        };
        // with explicit steps, the stepless (noise) lines are dropped anyway;
        // the rare-key filter is for logs without a step
        let min_count = if explicit > 0 { 1 } else { min_count };
        let mut count = vec![0usize; self.names.len()];
        for r in self.rows.iter().filter(|r| explicit == 0 || r.step.is_some() || missing == Missing::CarryForward) {
            for (k, _) in &r.kv {
                count[*k as usize] += 1;
            }
        }
        let keep: Vec<bool> = count.iter().map(|&c| c >= min_count).collect();
        let dropped_keys: Vec<&str> = self.names.iter().zip(&keep).filter(|(_, k)| !**k).map(|(n, _)| n.as_str()).collect();
        if !dropped_keys.is_empty() {
            let mut d: Vec<&str> = dropped_keys.iter().take(8).copied().collect();
            if dropped_keys.len() > 8 {
                d.push("...");
            }
            rep.notes.push(format!("ignored {} key(s) seen on fewer than {min_count} lines: {}", dropped_keys.len(), d.join(", ")));
        }
        if explicit > 0 {
            if explicit < n {
                rep.notes.push(match missing {
                    Missing::Drop => format!("{} line(s) without a step ignored", n - explicit),
                    Missing::CarryForward => format!("{} row(s) without a step took the previous row's step", n - explicit),
                });
            }
        } else if epoch_x {
            rep.notes.push("no step column; x = epoch".into());
        } else if n > 0 {
            rep.notes.push("no step column; x = line order".into());
        }
        let mut used = 0;
        let mut last: Option<f64> = None;
        let mut implicit = 0.0;
        let mut seen_step = std::collections::HashSet::new();
        for r in self.rows {
            let x = if explicit > 0 {
                match (r.step, missing) {
                    (Some(s), _) => s,
                    (None, Missing::CarryForward) => match last {
                        Some(s) => s,
                        None => {
                            rep.skipped += 1;
                            continue;
                        }
                    },
                    (None, Missing::Drop) => {
                        rep.skipped += 1;
                        continue;
                    }
                }
            } else if epoch_x {
                r.epoch.unwrap()
            } else {
                implicit += 1.0;
                implicit - 1.0
            };
            last = Some(x);
            let mut any = false;
            for (k, v) in r.kv {
                if keep[k as usize] {
                    run.push(&self.names[k as usize], x, v);
                    rep.points += 1;
                    any = true;
                }
            }
            if any {
                used += 1;
                seen_step.insert(x.to_bits());
                if let Some(t) = r.t {
                    run.push_time(x, t);
                }
            } else {
                rep.skipped += 1;
            }
        }
        rep.metrics = keep.iter().filter(|k| **k).count();
        rep.steps = seen_step.len();
        run.rows += used;
        used
    }
}

// ------------------------------------------------------------------ sink

/// [`Sink`] into a [`LocalRun`] (legacy pilot logs, W&B import).
pub struct MemSink<'a> {
    run: &'a mut LocalRun,
    rep: &'a mut FileReport,
    steps: std::collections::HashSet<u64>,
}

pub struct OpenMem<'a> {
    pub run: &'a mut LocalRun,
    pub rep: &'a mut FileReport,
}

impl<'a> OpenSink for OpenMem<'a> {
    type Sink = MemSink<'a>;
    fn open(self, _name: &str, config: &Value, meta: Meta) -> io::Result<MemSink<'a>> {
        if let Value::Object(m) = config {
            for (k, v) in m {
                self.run.config.insert(k.clone(), v.clone());
            }
        }
        if let Some(p) = meta.project {
            self.run.meta.insert("project".into(), json!(p));
        }
        if let Some(h) = meta.hostname {
            self.run.meta.insert("hostname".into(), json!(h));
        }
        if meta.start_unix > 0.0 {
            self.run.meta.insert("start_unix".into(), json!(meta.start_unix));
        }
        if let Some(Value::Object(e)) = meta.extra {
            for (k, v) in e {
                self.run.meta.insert(k, v);
            }
        }
        Ok(MemSink { run: self.run, rep: self.rep, steps: Default::default() })
    }
}

impl Sink for MemSink<'_> {
    type Output = ();
    fn log_with_time(&mut self, step: u64, t: f64, kv: &[(&str, f64)]) -> Result<(), ObsError> {
        for (k, v) in kv {
            self.run.push(k, step as f64, *v);
            self.rep.points += 1;
        }
        self.run.push_time(step as f64, t);
        if self.steps.insert(step) {
            self.run.rows += 1;
        }
        Ok(())
    }
    fn log_array_with_time(&mut self, step: u64, t: f64, key: &str, shape: &[usize], data: &[f32]) -> Result<(), ObsError> {
        self.run.arrays.push(ArrayRecord { step, t, key: key.into(), shape: shape.to_vec(), data: data.to_vec(), kind: "tensor".into(), edges: None, labels: None });
        Ok(())
    }
    fn log_histogram_with_time(&mut self, step: u64, t: f64, key: &str, h: &Histogram) -> Result<(), ObsError> {
        self.run.arrays.push(ArrayRecord { step, t, key: key.into(), shape: vec![h.counts.len()], data: h.counts.clone(), kind: "histogram".into(), edges: Some(h.edges.clone()), labels: None });
        Ok(())
    }
    fn event_with_time(&mut self, step: u64, t: f64, text: &str) -> Result<(), ObsError> {
        self.run.events.push(EventRecord { step, t, text: text.into() });
        Ok(())
    }
    fn summary_value(&mut self, key: &str, value: &Value) -> io::Result<()> {
        self.run.summary.insert(key.into(), value.clone());
        Ok(())
    }
    fn finish_with_time(self, extra: &Value, elapsed_s: f64) -> io::Result<()> {
        if let Value::Object(m) = extra {
            for (k, v) in m {
                self.run.summary.insert(k.clone(), v.clone());
            }
        }
        self.run.summary.insert("elapsed_s".into(), json!(elapsed_s));
        self.run.finished = true;
        self.close()
    }
    fn close(self) -> io::Result<()> {
        self.rep.steps = self.steps.len();
        Ok(())
    }
}

// ------------------------------------------------------------------ detect

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Csv(u8),
    Jsonl,
    Text,
    Legacy,
    TfEvents,
    Wandb,
    /// Single JSON document (`{key, step, value}` metric file, or an array
    /// of row objects).
    JsonDoc,
    RunConfig,
    RunSummary,
    RunMetrics,
    RunEvents,
    RunArrays,
    /// Known companion file with nothing to graph (exported `events.json`,
    /// `metrics.small.json`, ...).
    Ignore,
    Binary,
}

impl Format {
    pub fn name(self) -> &'static str {
        match self {
            Format::Csv(b'\t') => "tsv",
            Format::Csv(_) => "csv",
            Format::Jsonl => "jsonl",
            Format::Text => "text log",
            Format::Legacy => "ctmlm_pilot log",
            Format::TfEvents => "tensorboard",
            Format::Wandb => "wandb",
            Format::JsonDoc => "json",
            Format::RunConfig => "run config.json",
            Format::RunSummary => "run summary.json",
            Format::RunMetrics => "run metrics.jsonl",
            Format::RunEvents => "run events.jsonl",
            Format::RunArrays => "run arrays.jsonl",
            Format::Ignore => "ignored",
            Format::Binary => "binary",
        }
    }
    fn buffered(self) -> bool {
        matches!(self, Format::Legacy | Format::TfEvents | Format::Wandb | Format::JsonDoc | Format::RunConfig | Format::RunSummary | Format::Binary)
    }
}

fn basename(name: &str) -> &str {
    name.rsplit(['/', '\\']).next().unwrap_or(name)
}

/// Pick the parser from the file name and its first bytes.
pub fn detect(name: &str, head: &[u8]) -> Format {
    let base = basename(name);
    let lower = base.to_ascii_lowercase();
    let ext = lower.rsplit_once('.').map(|(_, e)| e).unwrap_or("");
    if head.starts_with(b":W&B") || ext == "wandb" {
        return Format::Wandb;
    }
    if lower.contains("tfevents") || tfevents::looks_like(head) {
        return Format::TfEvents;
    }
    match lower.as_str() {
        "config.json" => return Format::RunConfig,
        "summary.json" | "wandb-summary.json" => return Format::RunSummary,
        "metrics.jsonl" => return Format::RunMetrics,
        "events.jsonl" => return Format::RunEvents,
        "arrays.jsonl" => return Format::RunArrays,
        "events.json" | "metrics.small.json" | "index.json" | "wandb-metadata.json" => return Format::Ignore,
        _ => {}
    }
    let sample = &head[..head.len().min(1 << 16)];
    if sample.iter().take(4096).any(|&b| b == 0) {
        return Format::Binary;
    }
    let text = String::from_utf8_lossy(sample);
    match ext {
        "csv" => return Format::Csv(csv::sniff_delim(&text).unwrap_or(b',')),
        "tsv" | "tab" => return Format::Csv(b'\t'),
        "jsonl" | "ndjson" => return Format::Jsonl,
        "json" => {
            // JSON Lines saved as .json: several lines that each start with '{'
            let objs = text.lines().filter(|l| !l.trim().is_empty()).take(3).filter(|l| l.trim_start().starts_with('{') && l.trim_end().ends_with('}')).count();
            return if objs >= 2 { Format::Jsonl } else { Format::JsonDoc };
        }
        _ => {}
    }
    if modgrad_obs::legacy::looks_like(&text) {
        return Format::Legacy;
    }
    let first = text.lines().find(|l| !l.trim().is_empty()).unwrap_or("");
    if first.trim_start().starts_with('{') && serde_json::from_str::<Value>(first.trim()).is_ok() {
        return Format::Jsonl;
    }
    if let Some(d) = csv::sniff_table(&text) {
        return Format::Csv(d);
    }
    Format::Text
}

// ------------------------------------------------------------------ ingest

const SNIFF: usize = 1 << 16;

enum Parser {
    Csv(csv::Csv),
    Jsonl(jsonl::Jsonl),
    Text(text::Text),
    Events(Vec<EventRecord>),
    Arrays(Vec<ArrayRecord>),
    None,
}

/// One file being streamed in. `push` bytes in order, then `finish` into
/// a [`LocalRun`] (several files can finish into the same run: a dropped
/// folder).
pub struct Ingest {
    name: String,
    format: Option<Format>,
    parser: Parser,
    buf: Vec<u8>,
    head: Vec<u8>,
    partial: Vec<u8>,
    bytes: u64,
    lines: usize,
    skipped: usize,
    /// File mtime (unix seconds), used as the start-time estimate.
    pub mtime_unix: f64,
}

impl Ingest {
    pub fn new(name: &str) -> Ingest {
        Ingest { name: name.to_string(), format: None, parser: Parser::None, buf: Vec::new(), head: Vec::new(), partial: Vec::new(), bytes: 0, lines: 0, skipped: 0, mtime_unix: 0.0 }
    }

    pub fn format(&self) -> Option<Format> {
        self.format
    }

    pub fn push(&mut self, chunk: &[u8]) {
        self.bytes += chunk.len() as u64;
        if self.format.is_none() {
            // sniff on the first 64 KiB, however the caller chunks
            self.head.extend_from_slice(chunk);
            if self.head.len() >= SNIFF {
                self.start();
            }
            return;
        }
        self.feed(chunk);
    }

    fn start(&mut self) {
        let head = std::mem::take(&mut self.head);
        let f = detect(&self.name, &head);
        self.format = Some(f);
        self.parser = match f {
            Format::Csv(d) => Parser::Csv(csv::Csv::new(d)),
            Format::Jsonl => Parser::Jsonl(jsonl::Jsonl::new(false)),
            Format::RunMetrics => Parser::Jsonl(jsonl::Jsonl::new(true)),
            Format::Text => Parser::Text(text::Text::default()),
            Format::RunEvents => Parser::Events(Vec::new()),
            Format::RunArrays => Parser::Arrays(Vec::new()),
            _ => Parser::None,
        };
        self.feed(&head);
    }

    fn feed(&mut self, chunk: &[u8]) {
        let Some(f) = self.format else { return };
        if f == Format::Ignore {
            return;
        }
        if f.buffered() {
            self.buf.extend_from_slice(chunk);
            return;
        }
        // split on \n and \r (tqdm redraws with \r); carry the partial line
        let mut start = 0;
        for (i, &b) in chunk.iter().enumerate() {
            if b == b'\n' || b == b'\r' {
                if self.partial.is_empty() {
                    let line = String::from_utf8_lossy(&chunk[start..i]).into_owned();
                    self.line(&line);
                } else {
                    self.partial.extend_from_slice(&chunk[start..i]);
                    let p = std::mem::take(&mut self.partial);
                    let line = String::from_utf8_lossy(&p).into_owned();
                    self.line(&line);
                }
                start = i + 1;
            }
        }
        self.partial.extend_from_slice(&chunk[start..]);
    }

    fn line(&mut self, l: &str) {
        if l.trim().is_empty() {
            return;
        }
        self.lines += 1;
        let ok = match &mut self.parser {
            Parser::Csv(p) => p.line(l),
            Parser::Jsonl(p) => p.line(l),
            Parser::Text(p) => p.line(l),
            Parser::Events(ev) => match serde_json::from_str::<Value>(l) {
                Ok(v) => match (v.get("step").and_then(|x| x.as_u64()), v.get("text").and_then(|x| x.as_str())) {
                    (Some(step), Some(text)) => {
                        ev.push(EventRecord { step, t: v.get("t").and_then(|x| x.as_f64()).unwrap_or(f64::NAN), text: text.into() });
                        true
                    }
                    _ => false,
                },
                Err(_) => false,
            },
            Parser::Arrays(a) => match serde_json::from_str::<Value>(l).ok().as_ref().and_then(modgrad_obs::array_record) {
                Some(r) => {
                    a.push(r);
                    true
                }
                None => false,
            },
            Parser::None => false,
        };
        if !ok {
            self.skipped += 1;
        }
    }

    /// Parse whatever is left and merge into `run`.
    pub fn finish(mut self, run: &mut LocalRun) -> FileReport {
        if self.format.is_none() {
            self.start();
        }
        if !self.partial.is_empty() {
            let p = std::mem::take(&mut self.partial);
            let line = String::from_utf8_lossy(&p).into_owned();
            // a JSON object cut off mid-write would be skipped by the parser anyway
            self.line(&line);
        }
        let format = self.format.unwrap_or(Format::Text);
        let mut rep = FileReport { file: self.name.clone(), format: format.name().into(), bytes: self.bytes, lines: self.lines, skipped: self.skipped, ..Default::default() };
        let points0: usize = run.series.values().map(|s| s.step.len()).sum();
        let res: Result<(), String> = (|| {
            match std::mem::replace(&mut self.parser, Parser::None) {
                Parser::Csv(p) => p.finish(run, &mut rep),
                Parser::Jsonl(p) => p.finish(run, &mut rep),
                Parser::Text(p) => p.finish(run, &mut rep),
                Parser::Events(ev) => {
                    rep.notes.push(format!("{} event(s)", ev.len()));
                    run.events.extend(ev);
                }
                Parser::Arrays(a) => {
                    let keys: std::collections::BTreeSet<&str> = a.iter().map(|r| r.key.as_str()).collect();
                    rep.notes.push(format!("{} array record(s), {} key(s)", a.len(), keys.len()));
                    run.arrays.extend(a);
                }
                Parser::None => {}
            }
            let buf = std::mem::take(&mut self.buf);
            match format {
                Format::Legacy => {
                    let text = String::from_utf8_lossy(&buf);
                    let name = run.name.clone();
                    let (lr, ()) = modgrad_obs::legacy::convert_into(&text, OpenMem { run, rep: &mut rep }, &name, "local", &self.name, self.mtime_unix).map_err(|e| e.to_string())?;
                    rep.lines = lr.lines;
                    rep.skipped = lr.unparsed.len();
                    if !lr.unparsed.is_empty() {
                        rep.notes.push(format!("{} unparsed line(s)", lr.unparsed.len()));
                    }
                }
                Format::Wandb => {
                    let recs = modgrad_obs_wandb::read_records_from_bytes(&buf).map_err(|e| e.to_string())?;
                    let name = run.name.clone();
                    let (wr, ()) = modgrad_obs_wandb::import_records(&recs, &self.name, &name, OpenMem { run, rep: &mut rep }).map_err(|e| e.to_string())?;
                    rep.lines = wr.records;
                    rep.skipped = wr.skipped_values;
                    rep.notes.push(format!("{} records, {} history rows", wr.records, wr.history_rows));
                    if wr.histograms > 0 {
                        rep.notes.push(format!("{} histogram(s) kept as arrays", wr.histograms));
                    }
                    if wr.skipped_values > 0 {
                        rep.notes.push(format!("{} non-scalar history value(s) skipped (images, tables, ...)", wr.skipped_values));
                    }
                }
                Format::TfEvents => tfevents::ingest(&buf, run, &mut rep),
                Format::JsonDoc => json_doc(&buf, &self.name, run, &mut rep)?,
                Format::RunConfig => {
                    let v: Value = serde_json::from_slice(&buf).map_err(|e| format!("config.json: {e}"))?;
                    // our layout {name, config, meta}; anything else is the config itself
                    match (v.get("config"), v.get("meta")) {
                        (Some(Value::Object(c)), _) => {
                            run.config.extend(c.clone());
                            if let Some(Value::Object(m)) = v.get("meta") {
                                run.meta.extend(m.clone());
                            }
                        }
                        _ => {
                            if let Value::Object(c) = v {
                                run.config.extend(c);
                            }
                        }
                    }
                    rep.notes.push(format!("{} config field(s)", run.config.len()));
                }
                Format::RunSummary => {
                    let v: Value = serde_json::from_slice(&buf).map_err(|e| format!("summary.json: {e}"))?;
                    let finished = v.get("finished").and_then(|x| x.as_bool()).unwrap_or(false);
                    let m = match v.get("summary") {
                        Some(Value::Object(m)) => m.clone(),
                        _ => v.as_object().cloned().unwrap_or_default(),
                    };
                    rep.notes.push(format!("{} summary field(s){}", m.len(), if finished { ", finished" } else { "" }));
                    run.summary.extend(m.into_iter().filter(|(k, _)| !k.starts_with('_')));
                    run.finished |= finished;
                }
                Format::Binary => return Err("binary file, not a known log format".into()),
                Format::Ignore => rep.notes.push("companion file, nothing to graph".into()),
                _ => {}
            }
            Ok(())
        })();
        if let Err(e) = res {
            rep.error = Some(e);
        }
        if rep.metrics == 0 {
            // formats that report through the sink: count from the run
            let points1: usize = run.series.values().map(|s| s.step.len()).sum();
            if points1 > points0 && rep.points == 0 {
                rep.points = points1 - points0;
            }
            if matches!(format, Format::Legacy | Format::Wandb | Format::TfEvents) {
                rep.metrics = run.series.len();
            }
        }
        run.files.push(rep.clone());
        rep
    }
}

/// `{key, step:[], value:[]}` (an exported metric file) or `[{...}, ...]`
/// rows.
fn json_doc(buf: &[u8], name: &str, run: &mut LocalRun, rep: &mut FileReport) -> Result<(), String> {
    let v: Value = serde_json::from_slice(buf).map_err(|e| format!("not valid JSON: {e}"))?;
    match &v {
        Value::Object(o) if o.get("step").is_some_and(|s| s.is_array()) && o.get("value").is_some_and(|s| s.is_array()) => {
            let key = o.get("key").and_then(|k| k.as_str()).map(String::from).unwrap_or_else(|| basename(name).trim_end_matches(".json").to_string());
            let st = o["step"].as_array().unwrap();
            let va = o["value"].as_array().unwrap();
            for (s, x) in st.iter().zip(va) {
                if let Some(s) = s.as_f64() {
                    run.push(&key, s, x.as_f64().unwrap_or(f64::NAN));
                    rep.points += 1;
                }
            }
            rep.metrics = 1;
            rep.steps = st.len();
            run.rows += st.len();
            Ok(())
        }
        Value::Array(a) if a.iter().all(|x| x.is_object()) && !a.is_empty() => {
            let mut p = jsonl::Jsonl::new(false);
            for x in a {
                if !p.value(x) {
                    rep.skipped += 1;
                }
            }
            rep.lines = a.len();
            p.finish(run, rep);
            Ok(())
        }
        _ => Err("JSON document with no series (expected {step:[],value:[]} or an array of row objects)".into()),
    }
}

/// Parse a whole file in one go (tests, CLI).
pub fn ingest_bytes(run: &mut LocalRun, name: &str, bytes: &[u8], chunk: usize) -> FileReport {
    let mut ing = Ingest::new(name);
    for c in bytes.chunks(chunk.max(1)) {
        ing.push(c);
    }
    if bytes.is_empty() {
        ing.push(&[]);
    }
    ing.finish(run)
}

#[cfg(test)]
mod tests;
