//! W&B offline-run interop for modgrad-obs, with no wandb SDK and no
//! `wandb-core`:
//!
//! * [`export_run`]: a modgrad-obs run directory becomes
//!   `offline-run-<YYYYMMDD_HHMMSS>-<id>/run-<id>.wandb`, the layout
//!   `wandb sync` uploads.
//! * [`import_file`]: a `.wandb` file becomes a modgrad-obs run directory
//!   that the dashboard can read.
//!
//! The mapping:
//!
//! | modgrad-obs                       | W&B record                                        |
//! |-----------------------------------|---------------------------------------------------|
//! | `config.json` `config`            | `RunRecord.config` (one `ConfigItem` per top-level key) |
//! | metrics row at step s             | `HistoryRecord{step: s}` with `_step`, `_runtime`, `_timestamp` plus one item per key (`value_json`) |
//! | histogram array at step s         | same step's history item `{"_type":"histogram","values":counts,"bins":edges}` |
//! | other arrays (tensor/trajectory/…)| not exported (W&B has no plain tensor type); counted in [`ExportReport::skipped_arrays`] |
//! | events                            | not exported; counted                             |
//! | `summary.json` `summary`          | `SummaryRecord` (one item per key)                |
//! | finished / not                    | `RunExitRecord{exit_code: 0 / 1}` then `FinalRecord` |
//!
//! Non-finite metric values are left out of the history row (W&B's JSON has
//! no NaN), so a gap stays a gap.

pub mod log;
pub mod proto;

use proto::*;
use prost::Message;
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};

pub const PRODUCER: &str = concat!("modgrad-obs-wandb ", env!("CARGO_PKG_VERSION"));

#[derive(Debug, Default, Clone, PartialEq)]
pub struct ExportReport {
    pub file: PathBuf,
    pub run_id: String,
    pub records: usize,
    pub history_rows: usize,
    pub histograms: usize,
    pub skipped_arrays: usize,
    pub skipped_events: usize,
}

fn info(run_id: &str) -> Option<RecordInfo> {
    Some(RecordInfo { stream_id: run_id.to_string() })
}

/// Deterministic 8-character W&B-style run id (`[a-z0-9]`) from a name.
pub fn run_id_for(name: &str) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in name.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    const A: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789";
    (0..8).map(|i| A[((h >> (i * 7)) % 36) as usize] as char).collect()
}

/// `YYYYMMDD_HHMMSS` (UTC) for a Unix time, without a date library.
pub fn stamp(unix: f64) -> String {
    let secs = unix.max(0.0) as i64;
    let (days, rem) = (secs.div_euclid(86400), secs.rem_euclid(86400));
    // civil-from-days (H. Hinnant)
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
    format!("{y:04}{m:02}{d:02}_{:02}{:02}{:02}", rem / 3600, (rem % 3600) / 60, rem % 60)
}

fn num_json(x: f64) -> String {
    serde_json::to_string(&json!(x)).unwrap()
}

/// Write `out_dir/offline-run-<stamp>-<id>/run-<id>.wandb` for the run at
/// `run_dir`. `run_id` defaults to [`run_id_for`] of the run name.
pub fn export_run(run_dir: &Path, out_dir: &Path, run_id: Option<&str>) -> io::Result<ExportReport> {
    let rd = modgrad_obs::read_run(run_dir)?;
    let name = rd.name();
    let id = run_id.map(String::from).unwrap_or_else(|| run_id_for(&name));
    let start = rd.config["meta"]["start_unix"].as_f64().unwrap_or_else(modgrad_obs::now_unix);
    let project = rd.config["meta"]["project"].as_str().unwrap_or("modgrad").to_string();
    let host = rd.config["meta"]["hostname"].as_str().unwrap_or("").to_string();

    let dir = out_dir.join(format!("offline-run-{}-{id}", stamp(start)));
    std::fs::create_dir_all(dir.join("files"))?;
    let file = dir.join(format!("run-{id}.wandb"));
    let mut w = log::LogWriter::new(io::BufWriter::new(std::fs::File::create(&file)?))?;
    let mut rep = ExportReport { file: file.clone(), run_id: id.clone(), ..Default::default() };
    let mut num = 0i64;
    let mut put = |w: &mut log::LogWriter<_>, rt: RecordType, rep: &mut ExportReport| -> io::Result<()> {
        num += 1;
        let rec = Record { num, record_type: Some(rt), uuid: String::new(), info: info(&id) };
        rep.records += 1;
        w.write(&rec.encode_to_vec())
    };

    put(&mut w, RecordType::Header(HeaderRecord { version_info: Some(VersionInfo { producer: PRODUCER.into(), min_consumer: String::new() }), info: info(&id) }), &mut rep)?;

    let config = ConfigRecord {
        update: rd.config["config"].as_object().map(|m| m.iter().map(|(k, v)| ConfigItem { key: k.clone(), nested_key: vec![], value_json: v.to_string() }).collect()).unwrap_or_default(),
        remove: vec![],
        info: None,
    };
    let run = RunRecord {
        run_id: id.clone(),
        project: project.clone(),
        display_name: name.clone(),
        host,
        config: Some(config),
        start_time: Some(Timestamp { seconds: start.floor() as i64, nanos: ((start.fract()) * 1e9) as i32 }),
        notes: format!("imported from modgrad-obs run {name}"),
        info: info(&id),
        ..Default::default()
    };
    put(&mut w, RecordType::Run(run), &mut rep)?;

    // history: merge metric rows by step (later wins), add histograms
    let mut rows: BTreeMap<u64, (f64, BTreeMap<String, String>)> = BTreeMap::new();
    for r in &rd.metrics {
        let e = rows.entry(r.step).or_insert((r.t, BTreeMap::new()));
        if r.t.is_finite() {
            e.0 = r.t;
        }
        for (k, v) in &r.values {
            if v.is_finite() {
                e.1.insert(k.clone(), num_json(*v));
            }
        }
    }
    for a in &rd.arrays {
        if a.kind == "histogram" {
            let e = rows.entry(a.step).or_insert((a.t, BTreeMap::new()));
            let v = json!({ "_type": "histogram", "values": a.data, "bins": a.edges.clone().unwrap_or_default() });
            e.1.insert(a.key.clone(), v.to_string());
            rep.histograms += 1;
        } else {
            rep.skipped_arrays += 1;
        }
    }
    rep.skipped_events = rd.events.len();
    for (step, (t, items)) in &rows {
        let t = if t.is_finite() { *t } else { 0.0 };
        let mut item = vec![
            HistoryItem { key: "_step".into(), value_json: step.to_string(), ..Default::default() },
            HistoryItem { key: "_runtime".into(), value_json: num_json(t), ..Default::default() },
            HistoryItem { key: "_timestamp".into(), value_json: num_json(start + t), ..Default::default() },
        ];
        item.extend(items.iter().map(|(k, v)| HistoryItem { key: k.clone(), value_json: v.clone(), ..Default::default() }));
        put(&mut w, RecordType::History(HistoryRecord { item, step: Some(HistoryStep { num: *step as i64 }), info: info(&id) }), &mut rep)?;
        rep.history_rows += 1;
    }

    let summary = rd.summary.as_ref().and_then(|s| s["summary"].as_object().cloned()).unwrap_or_default();
    if !summary.is_empty() {
        let update = summary.iter().filter(|(_, v)| !v.is_null()).map(|(k, v)| SummaryItem { key: k.clone(), nested_key: vec![], value_json: v.to_string() }).collect();
        put(&mut w, RecordType::Summary(SummaryRecord { update, remove: vec![], info: info(&id) }), &mut rep)?;
    }
    let finished = rd.summary.as_ref().is_some_and(|s| s["finished"] == true);
    let runtime = rd.summary.as_ref().and_then(|s| s["elapsed_s"].as_f64()).or_else(|| rows.values().map(|r| r.0).filter(|t| t.is_finite()).fold(None, |a: Option<f64>, t| Some(a.map_or(t, |a| a.max(t))))).unwrap_or(0.0);
    put(&mut w, RecordType::Exit(RunExitRecord { exit_code: if finished { 0 } else { 1 }, runtime: runtime as i32, not_complete: false, info: info(&id) }), &mut rep)?;
    put(&mut w, RecordType::Final(FinalRecord { info: info(&id) }), &mut rep)?;
    w.into_inner()?;
    Ok(rep)
}

/// Decode every record of a `.wandb` file.
pub fn read_records(path: &Path) -> io::Result<Vec<Record>> {
    let payloads = log::read_all(std::fs::File::open(path)?).map_err(io::Error::other)?;
    payloads.iter().map(|p| Record::decode(&p[..]).map_err(io::Error::other)).collect()
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct ImportReport {
    pub run_dir: PathBuf,
    pub records: usize,
    pub history_rows: usize,
    pub histograms: usize,
    /// History values that are neither numbers, booleans nor histograms
    /// (images, tables, text ...).
    pub skipped_values: usize,
    /// Records of types this importer does not model (telemetry, output ...).
    pub other_records: usize,
}

fn history_value(it: &HistoryItem) -> Option<Value> {
    match &it.value {
        Some(HistoryValue::None(_)) => Some(Value::Null),
        Some(HistoryValue::Boolean(b)) => Some(json!(b)),
        Some(HistoryValue::Integer(i)) => Some(json!(i)),
        Some(HistoryValue::Number(x)) => Some(json!(x)),
        Some(HistoryValue::Text(s)) => Some(json!(s)),
        Some(HistoryValue::Json(s)) => serde_json::from_str(s).ok(),
        None => serde_json::from_str(&it.value_json).ok().or_else(|| match it.value_json.as_str() {
            "NaN" => Some(json!(null)),
            _ => None,
        }),
    }
}

fn item_key(key: &str, nested: &[String]) -> String {
    if nested.is_empty() { key.to_string() } else { nested.join("/") }
}

/// Import a `.wandb` file as `<obs_root>/runs/<name>` (name defaults to
/// `wandb/<display_name or run_id>`).
pub fn import_file(path: &Path, obs_root: &Path, name: Option<&str>) -> io::Result<ImportReport> {
    let recs = read_records(path)?;
    let mut rep = ImportReport { records: recs.len(), ..Default::default() };
    let run = recs.iter().find_map(|r| if let Some(RecordType::Run(x)) = &r.record_type { Some(x.clone()) } else { None }).unwrap_or_default();
    let run_name = name.map(String::from).unwrap_or_else(|| format!("wandb/{}", if run.display_name.is_empty() { &run.run_id } else { &run.display_name }).replace(|c: char| !(c.is_ascii_alphanumeric() || "_-./".contains(c)), "_"));
    let mut config = Map::new();
    let apply_config = |c: &ConfigRecord, config: &mut Map<String, Value>| {
        for it in &c.update {
            config.insert(item_key(&it.key, &it.nested_key), serde_json::from_str(&it.value_json).unwrap_or(Value::String(it.value_json.clone())));
        }
        for it in &c.remove {
            config.remove(&item_key(&it.key, &it.nested_key));
        }
    };
    if let Some(c) = &run.config {
        apply_config(c, &mut config);
    }
    for r in &recs {
        if let Some(RecordType::Config(c)) = &r.record_type {
            apply_config(c, &mut config);
        }
    }
    let start = run.start_time.as_ref().map(|t| t.seconds as f64 + t.nanos as f64 * 1e-9).unwrap_or(0.0);
    let meta = modgrad_obs::Meta {
        project: (!run.project.is_empty()).then(|| run.project.clone()),
        git: None,
        hostname: (!run.host.is_empty()).then(|| run.host.clone()),
        start_unix: start,
        extra: Some(json!({ "source": "wandb", "wandb_run_id": run.run_id, "wandb_file": path.to_string_lossy() })),
    };
    let dir = obs_root.join("runs").join(&run_name);
    for f in ["metrics.jsonl", "arrays.jsonl", "events.jsonl", "summary.json"] {
        let _ = std::fs::remove_file(dir.join(f));
    }
    let mut out = modgrad_obs::Run::create_in(&dir, &run_name, &Value::Object(config), meta)?;
    let err = |e: modgrad_obs::ObsError| io::Error::other(e.to_string());
    let mut summary = Map::new();
    let mut exit: Option<RunExitRecord> = None;
    let mut last_t = 0.0;
    for r in &recs {
        match &r.record_type {
            Some(RecordType::History(h)) => {
                let mut step = h.step.as_ref().map(|s| s.num.max(0) as u64);
                let mut t = None;
                let mut scalars: Vec<(String, f64)> = Vec::new();
                let mut hists: Vec<(String, Vec<f32>, Vec<f64>)> = Vec::new();
                for it in &h.item {
                    let key = item_key(&it.key, &it.nested_key);
                    let Some(v) = history_value(it) else {
                        rep.skipped_values += 1;
                        continue;
                    };
                    match key.as_str() {
                        "_step" => step = step.or(v.as_u64()),
                        "_runtime" => t = v.as_f64(),
                        "_timestamp" => {}
                        _ => match &v {
                            Value::Number(n) => scalars.push((key, n.as_f64().unwrap_or(f64::NAN))),
                            Value::Bool(b) => scalars.push((key, if *b { 1.0 } else { 0.0 })),
                            Value::Null => scalars.push((key, f64::NAN)),
                            Value::Object(o) if o.get("_type").and_then(|x| x.as_str()) == Some("histogram") => {
                                let counts: Vec<f32> = o.get("values").and_then(|x| x.as_array()).map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect()).unwrap_or_default();
                                let edges: Vec<f64> = o.get("bins").and_then(|x| x.as_array()).map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect()).unwrap_or_default();
                                hists.push((key, counts, edges));
                            }
                            _ => rep.skipped_values += 1,
                        },
                    }
                }
                let step = step.unwrap_or(rep.history_rows as u64);
                let t = t.unwrap_or(last_t);
                last_t = t;
                if !scalars.is_empty() {
                    let kv: Vec<(&str, f64)> = scalars.iter().map(|(k, v)| (k.as_str(), *v)).collect();
                    out.log_with_time(step, t, &kv).map_err(err)?;
                }
                for (k, counts, edges) in hists {
                    if edges.len() == counts.len() + 1 {
                        out.log_histogram_with_time(step, t, &k, &modgrad_obs::Histogram { edges, counts }).map_err(err)?;
                        rep.histograms += 1;
                    } else {
                        rep.skipped_values += 1;
                    }
                }
                rep.history_rows += 1;
            }
            Some(RecordType::Summary(s)) => {
                for it in &s.update {
                    summary.insert(item_key(&it.key, &it.nested_key), serde_json::from_str(&it.value_json).unwrap_or(Value::String(it.value_json.clone())));
                }
                for it in &s.remove {
                    summary.remove(&item_key(&it.key, &it.nested_key));
                }
            }
            Some(RecordType::Exit(e)) => exit = Some(e.clone()),
            Some(RecordType::Run(_) | RecordType::Config(_) | RecordType::Header(_) | RecordType::Final(_)) => {}
            None => rep.other_records += 1,
        }
    }
    let summary = Value::Object(summary);
    match exit {
        Some(e) => {
            let mut s = summary;
            s["wandb_exit_code"] = json!(e.exit_code);
            out.finish_with_time(&s, e.runtime as f64)?;
        }
        None => {
            if let Value::Object(m) = summary {
                for (k, v) in m {
                    out.summary(&k, v)?;
                }
            }
        }
    }
    rep.run_dir = dir;
    Ok(rep)
}

#[cfg(test)]
mod tests {
    use super::*;
    use modgrad_obs::{Histogram, Run, Tensor};

    #[test]
    fn stamp_and_id() {
        assert_eq!(stamp(0.0), "19700101_000000");
        assert_eq!(stamp(1_760_158_800.0), "20251011_050000");
        let id = run_id_for("bkk07/mid");
        assert_eq!(id.len(), 8);
        assert_eq!(id, run_id_for("bkk07/mid"));
        assert!(id.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit()));
    }

    #[test]
    fn export_then_import_roundtrips() {
        let root = std::env::temp_dir().join(format!("obs-wandb-rt-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let mut run = Run::init(&root, "proj", "r1", &json!({"brain_params": 494325, "lr": 1e-3, "tag": "a"})).unwrap();
        for s in 0..40u64 {
            run.log_with_time(s * 10, s as f64, &[("loss", 3.0 - s as f64 * 0.05), ("acc", if s % 7 == 0 { f64::NAN } else { s as f64 })]).unwrap();
        }
        run.log_histogram_with_time(100, 10.0, "w", &Histogram { edges: vec![0.0, 0.5, 1.0], counts: vec![3.0, 4.0] }).unwrap();
        run.log_histogram_with_time(105, 10.5, "w", &Histogram { edges: vec![0.0, 0.5, 1.0], counts: vec![1.0, 2.0] }).unwrap();
        run.log(100, [("recall", Tensor::new(&[3], vec![1.0, 2.0, 3.0]).into())]).unwrap();
        run.event(5, "e").unwrap();
        run.summary("best", 0.25).unwrap();
        run.finish().unwrap();
        let src = root.join("runs/r1");

        let rep = export_run(&src, &root.join("wandb"), None).unwrap();
        assert_eq!(rep.history_rows, 41); // 40 metric steps + step 105 (histogram only)
        assert_eq!((rep.histograms, rep.skipped_arrays, rep.skipped_events), (2, 1, 1));
        assert!(rep.file.file_name().unwrap().to_string_lossy().starts_with("run-"));
        assert!(rep.file.parent().unwrap().file_name().unwrap().to_string_lossy().starts_with("offline-run-"));
        let recs = read_records(&rep.file).unwrap();
        assert!(matches!(recs[0].record_type, Some(RecordType::Header(_))));
        assert!(matches!(recs[1].record_type, Some(RecordType::Run(_))));
        assert!(matches!(recs.last().unwrap().record_type, Some(RecordType::Final(_))));
        assert!(recs.windows(2).all(|w| w[1].num == w[0].num + 1));

        let imp = import_file(&rep.file, &root.join("back"), Some("r1")).unwrap();
        assert_eq!(imp.skipped_values, 0);
        let a = modgrad_obs::read_run(&src).unwrap();
        let b = modgrad_obs::read_run(&imp.run_dir).unwrap();
        assert_eq!(a.series("loss"), b.series("loss"));
        let fin = |v: Vec<(u64, f64)>| v.into_iter().filter(|p| p.1.is_finite()).collect::<Vec<_>>();
        assert_eq!(fin(a.series("acc")), b.series("acc")); // NaN = gap = absent
        assert_eq!(b.config["config"]["brain_params"], 494325);
        assert_eq!(b.config["config"]["tag"], "a");
        assert_eq!(b.config["meta"]["project"], "proj");
        let h = b.array_series("w");
        assert_eq!(h.len(), 2);
        assert_eq!((h[1].step, h[1].data.clone(), h[1].edges.clone().unwrap()), (105, vec![1.0, 2.0], vec![0.0, 0.5, 1.0]));
        let s = &b.summary.unwrap()["summary"];
        assert_eq!(s["best"], 0.25);
        assert_eq!(s["wandb_exit_code"], 0);
        // runtime survives (t of each row)
        assert_eq!(b.metrics.iter().find(|r| r.step == 50).unwrap().t, 5.0);
    }
}
