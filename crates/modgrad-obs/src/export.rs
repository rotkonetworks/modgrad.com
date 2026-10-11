//! Static export for the dashboard: walk `<obs>/runs/**` for run
//! directories (anything with a `config.json`), and write next to each run
//! small, plot-ready JSON files plus a global `<obs>/index.json`.
//!
//! Per run (all paths relative to the run directory):
//!
//! * `metrics/<file>.json` — `{"key", "step":[…], "value":[…]}` for one
//!   metric, rows merged by step (later wins), sorted, NaN → `null`.
//! * `metrics.small.json` — only when the run has more than
//!   [`SMALL_THRESHOLD`] history rows: every key, block-mean downsampled to
//!   at most [`SMALL_ROWS`] rows (`{"step":[…], "keys":{k:[…]}}`).
//! * `arrays/<file>.json` — `{"key","kind","step":[…],"shape":[[…]],"edges":[[…]]?,"b64":[…]}`,
//!   each record downsampled to at most [`ARRAY_ELEMS`] elements and at
//!   most [`ARRAY_STEPS`] records (evenly strided, last always kept).
//!   `trajectory`, `points` and `volume` records are capped at
//!   [`STATE_ELEMS`] instead; no analysis (PCA, smoothing, surfaces) happens
//!   here — the dashboard does it in wasm (modgrad-obs-analysis).
//! * `events.json` — the last 200 events.
//!
//! The dashboard fetches only the keys it plots.

use crate::{RunData, downsample, encode_f32, read_run};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};

pub const SMALL_THRESHOLD: usize = 2000;
pub const SMALL_ROWS: usize = 1000;
pub const ARRAY_ELEMS: usize = 4096;
pub const ARRAY_STEPS: usize = 200;
/// Cap for `trajectory` / `points` / `volume` records (raw state goes to
/// the browser, which runs PCA in wasm).
pub const STATE_ELEMS: usize = 65536;

/// Filesystem-safe name for a key.
pub fn key_file(key: &str) -> String {
    key.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.' { c } else { '_' }).collect()
}

fn jnum(x: f64) -> Value {
    if x.is_finite() { json!(x) } else { Value::Null }
}

/// Find run directories (containing `config.json`) under `root`, up to 4 levels.
pub fn find_runs(root: &Path) -> Vec<PathBuf> {
    fn walk(d: &Path, depth: usize, out: &mut Vec<PathBuf>) {
        if d.join("config.json").is_file() {
            out.push(d.to_path_buf());
            return;
        }
        if depth == 0 {
            return;
        }
        if let Ok(rd) = std::fs::read_dir(d) {
            let mut v: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_dir()).collect();
            v.sort();
            for p in v {
                walk(&p, depth - 1, out);
            }
        }
    }
    let mut out = Vec::new();
    walk(root, 4, &mut out);
    out
}

/// Merge history rows by step: step → key → value.
fn merged(rd: &RunData) -> BTreeMap<u64, BTreeMap<String, f64>> {
    let mut m: BTreeMap<u64, BTreeMap<String, f64>> = BTreeMap::new();
    for r in &rd.metrics {
        let e = m.entry(r.step).or_default();
        for (k, v) in &r.values {
            e.insert(k.clone(), *v);
        }
    }
    m
}

/// Flatten a JSON object to `a.b` → scalar (numbers, bools, short strings).
fn flatten(prefix: &str, v: &Value, out: &mut Map<String, Value>, depth: usize) {
    match v {
        Value::Object(m) if depth > 0 => {
            for (k, x) in m {
                let key = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
                flatten(&key, x, out, depth - 1);
            }
        }
        Value::Number(_) | Value::Bool(_) => {
            out.insert(prefix.into(), v.clone());
        }
        Value::String(s) if s.len() <= 120 => {
            out.insert(prefix.into(), v.clone());
        }
        Value::Array(a) if a.len() <= 16 && a.iter().all(|x| x.is_number()) => {
            out.insert(prefix.into(), json!(a.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")));
        }
        _ => {}
    }
}

fn write_json(p: &Path, v: &Value) -> io::Result<()> {
    if let Some(d) = p.parent() {
        std::fs::create_dir_all(d)?;
    }
    let tmp = p.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec(v)?)?;
    std::fs::rename(tmp, p)
}

/// Source label of a run: explicit provenance, else the first path
/// component under `runs/`, else its project.
pub fn source_of(rd: &RunData, runs_rel: Option<&str>) -> String {
    rd.config["meta"]["extra"]["source"].as_str().map(String::from).unwrap_or_else(|| match runs_rel.map(|r| r.split('/').collect::<Vec<_>>()) {
        Some(parts) if parts.len() >= 2 => parts[0].to_string(),
        // a run directly under runs/: label it by its project if any
        _ => rd.config["meta"]["project"].as_str().unwrap_or("local").to_string(),
    })
}

/// `{"key","step":[…],"value":[…]}` for one metric over merged rows
/// (`range`/`stride` select rows; they never average).
pub fn metric_json(rd: &RunData, key: &str, range: std::ops::Range<u64>, stride: usize) -> Value {
    let (mut st, mut va) = (Vec::new(), Vec::new());
    for (s, r) in merged(rd).range(range) {
        if let Some(v) = r.get(key) {
            st.push(*s);
            va.push(jnum(*v));
        }
    }
    let stride = stride.max(1);
    if stride > 1 {
        let n = st.len();
        let keep: Vec<usize> = (0..n).filter(|i| i % stride == 0 || *i + 1 == n).collect();
        st = keep.iter().map(|&i| st[i]).collect();
        va = keep.iter().map(|&i| va[i].clone()).collect();
    }
    json!({ "key": key, "step": st, "value": va })
}

/// The dashboard JSON for one array key: records at evenly strided steps
/// (at most `max_steps`, last kept), each mean-pooled to its kind's cap.
pub fn array_json(rd: &RunData, key: &str, max_steps: usize) -> Option<Value> {
    let recs = rd.array_series(key);
    let n = recs.len();
    if n == 0 {
        return None;
    }
    let max_steps = max_steps.max(2);
    let pick: Vec<usize> = if n <= max_steps {
        (0..n).collect()
    } else {
        let mut v: Vec<usize> = (0..max_steps - 1).map(|i| i * (n - 1) / (max_steps - 1)).collect();
        v.push(n - 1);
        v.dedup();
        v
    };
    let kind = recs[0].kind.clone();
    let (mut steps, mut shapes, mut b64s, mut edges, mut labels) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for &i in &pick {
        let r = recs[i];
        let (shape, data) = match r.kind.as_str() {
            "histogram" => (r.shape.clone(), r.data.clone()),
            // high-dim states go out raw (PCA runs in the browser, wasm)
            "trajectory" | "points" | "volume" => downsample(&r.shape, &r.data, STATE_ELEMS),
            _ => downsample(&r.shape, &r.data, ARRAY_ELEMS),
        };
        steps.push(r.step);
        shapes.push(shape);
        b64s.push(encode_f32(&data));
        edges.push(r.edges.clone().map(|e| json!(e)).unwrap_or(Value::Null));
        labels.push(r.labels.as_ref().map(|l| json!(encode_f32(l))).unwrap_or(Value::Null));
    }
    let mut v = json!({ "key": key, "kind": kind, "step": steps, "shape": shapes, "b64": b64s });
    match kind.as_str() {
        "histogram" => v["edges"] = json!(edges),
        "trajectory" | "points" if labels.iter().any(|l| !l.is_null()) => v["labels_b64"] = json!(labels),
        _ => {}
    }
    Some(v)
}

/// The index entry and every plot file of one run, without touching disk.
/// `rel` is the run's path relative to the obs root (e.g. `runs/bkk07/mid`);
/// file names in the entry are relative to that path.
pub fn build_run(rd: &RunData, rel: &str, source: &str) -> (Value, Vec<(String, Value)>) {
    let mut files: Vec<(String, Value)> = Vec::new();
    let rows = merged(rd);
    let keys = rd.metric_keys();
    let mut metric_files = Map::new();
    for k in &keys {
        let v = metric_json(rd, k, 0..u64::MAX, 1);
        let file = format!("metrics/{}.json", key_file(k));
        metric_files.insert(k.clone(), json!({ "file": file, "n": v["step"].as_array().map_or(0, |a| a.len()) }));
        files.push((file, v));
    }
    let has_small = rows.len() > SMALL_THRESHOLD;
    if has_small {
        let all: Vec<(&u64, &BTreeMap<String, f64>)> = rows.iter().collect();
        let block = all.len().div_ceil(SMALL_ROWS);
        let mut steps = Vec::new();
        let mut cols: BTreeMap<&str, Vec<Value>> = keys.iter().map(|k| (k.as_str(), Vec::new())).collect();
        for ch in all.chunks(block) {
            steps.push(*ch.last().unwrap().0);
            for k in &keys {
                let vals: Vec<f64> = ch.iter().filter_map(|(_, r)| r.get(k)).copied().filter(|v| v.is_finite()).collect();
                let v = if vals.is_empty() { f64::NAN } else { vals.iter().sum::<f64>() / vals.len() as f64 };
                cols.get_mut(k.as_str()).unwrap().push(jnum(v));
            }
        }
        files.push(("metrics.small.json".into(), json!({ "step": steps, "keys": cols, "block": block })));
    }
    let mut array_index = Map::new();
    for k in rd.array_keys() {
        let recs = rd.array_series(&k);
        let file = format!("arrays/{}.json", key_file(&k));
        array_index.insert(k.clone(), json!({ "file": file, "kind": recs[0].kind, "shape": recs[recs.len() - 1].shape, "n_steps": recs.len() }));
        files.push((file, array_json(rd, &k, ARRAY_STEPS).unwrap()));
    }
    let ev: Vec<Value> = rd.events.iter().rev().take(200).rev().map(|e| json!({"step": e.step, "t": jnum(e.t), "text": e.text})).collect();
    files.push(("events.json".into(), json!(ev)));

    let mut cfg = Map::new();
    flatten("", &rd.config["config"], &mut cfg, 3);
    let mut summ = Map::new();
    let sum_obj = rd.summary.as_ref().map(|s| s["summary"].clone()).unwrap_or(Value::Null);
    flatten("", &sum_obj, &mut summ, 2);
    let unparsed: Vec<Value> = sum_obj["legacy_unparsed"].as_array().cloned().unwrap_or_default();
    let last = rows.iter().next_back();
    let index = json!({
        "id": rel.trim_start_matches("runs/"),
        "name": rd.name(),
        "source": source,
        "path": rel,
        "project": rd.config["meta"]["project"],
        "meta": {
            "git": rd.config["meta"]["git"], "hostname": rd.config["meta"]["hostname"],
            "start_unix": rd.config["meta"]["start_unix"],
            "legacy_log": rd.config["meta"]["extra"]["legacy_log"],
        },
        "config": cfg,
        "summary": summ,
        "finished": rd.summary.as_ref().map(|s| s["finished"] == true).unwrap_or(false),
        "last_step": last.map(|(s, _)| *s),
        "last_t": rd.metrics.iter().map(|r| r.t).filter(|t| t.is_finite()).fold(None, |a: Option<f64>, t| Some(a.map_or(t, |a| a.max(t)))),
        "n_rows": rows.len(),
        "metrics": metric_files,
        "has_small": has_small,
        "arrays": array_index,
        "n_events": rd.events.len(),
        "skipped_lines": rd.skipped,
        "unparsed": unparsed.into_iter().take(50).collect::<Vec<_>>(),
    });
    (index, files)
}

/// Export one run (write its plot files next to it); returns its index entry.
pub fn export_run(obs: &Path, dir: &Path) -> io::Result<Value> {
    let rd = read_run(dir)?;
    let rel = dir.strip_prefix(obs).unwrap_or(dir).to_string_lossy().replace('\\', "/");
    let runs_rel = dir.strip_prefix(obs.join("runs")).ok().map(|p| p.to_string_lossy().into_owned());
    let source = source_of(&rd, runs_rel.as_deref());
    for sub in ["metrics", "arrays"] {
        let _ = std::fs::remove_dir_all(dir.join(sub));
    }
    let _ = std::fs::remove_file(dir.join("metrics.small.json"));
    let (index, files) = build_run(&rd, &rel, &source);
    for (f, v) in files {
        write_json(&dir.join(f), &v)?;
    }
    Ok(index)
}

/// Export every run under `<obs>/runs` and write `<obs>/index.json`.
pub fn export_all(obs: &Path) -> io::Result<Value> {
    let mut runs = Vec::new();
    let mut errors = Vec::new();
    for dir in find_runs(&obs.join("runs")) {
        match export_run(obs, &dir) {
            Ok(v) => runs.push(v),
            Err(e) => errors.push(json!({ "dir": dir.to_string_lossy(), "error": e.to_string() })),
        }
    }
    let idx = json!({ "generated_unix": crate::now_unix(), "runs": runs, "errors": errors });
    write_json(&obs.join("index.json"), &idx)?;
    Ok(idx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Run, Tensor};

    #[test]
    fn export_index_and_small() {
        let obs = std::env::temp_dir().join(format!("modgrad-obs-export-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&obs);
        let mut run = Run::init(&obs.join("runs").join("local"), "p", "big", &json!({"brain_params": 1000})).unwrap();
        for s in 0..2500u64 {
            run.log_scalars(s, &[("loss", 1.0 / (1.0 + s as f64))]).unwrap();
        }
        run.log(10, [("act", Tensor::new(&[100, 100], vec![0.5; 10000]).into())]).unwrap();
        run.finish().unwrap();
        let idx = export_all(&obs).unwrap();
        let r = &idx["runs"][0];
        assert_eq!(r["id"], "local/runs/big");
        assert_eq!(r["last_step"], 2499);
        assert_eq!(r["has_small"], true);
        assert_eq!(r["config"]["brain_params"], 1000);
        assert_eq!(r["arrays"]["act"]["shape"], json!([100, 100]));
        let dir = obs.join(r["path"].as_str().unwrap());
        let small: Value = serde_json::from_slice(&std::fs::read(dir.join("metrics.small.json")).unwrap()).unwrap();
        assert!(small["step"].as_array().unwrap().len() <= SMALL_ROWS);
        let a: Value = serde_json::from_slice(&std::fs::read(dir.join("arrays/act.json")).unwrap()).unwrap();
        let sh: Vec<usize> = serde_json::from_value(a["shape"][0].clone()).unwrap();
        assert!(sh.iter().product::<usize>() <= ARRAY_ELEMS);
        let m: Value = serde_json::from_slice(&std::fs::read(dir.join("metrics/loss.json")).unwrap()).unwrap();
        assert_eq!(m["step"].as_array().unwrap().len(), 2500);
    }
}
