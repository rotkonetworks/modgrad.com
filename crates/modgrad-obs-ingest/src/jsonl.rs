//! JSON Lines: one object per line (W&B `wandb-history.jsonl`, our
//! `metrics.jsonl`, most loggers' `--jsonl`). Nested objects flatten to
//! `a/b`; a top-level `metrics` (or `data`) wrapper is unwrapped. Numbers,
//! booleans (1/0), numeric strings (`"NaN"`, `"55%"`) and `null` (a gap
//! point, as in our run format) are kept; W&B media objects (`_type`),
//! arrays and text are skipped and counted.

use crate::num::parse_value;
use crate::{FileReport, LocalRun, Missing, Row, Rows, is_epoch, step_rank};
use serde_json::Value;

pub struct Jsonl {
    /// Our own `metrics.jsonl`: `step` and `t` are reserved.
    own: bool,
    rows: Rows,
    skipped_values: usize,
    t0: Option<f64>,
}

impl Jsonl {
    pub fn new(own: bool) -> Jsonl {
        Jsonl { own, rows: Rows::default(), skipped_values: 0, t0: None }
    }

    pub fn line(&mut self, l: &str) -> bool {
        match serde_json::from_str::<Value>(l.trim()) {
            Ok(v) => self.value(&v),
            Err(_) => false,
        }
    }

    pub fn value(&mut self, v: &Value) -> bool {
        let Value::Object(o) = v else { return false };
        let mut flat: Vec<(String, f64)> = Vec::new();
        for (k, x) in o {
            let wrapper = (k == "metrics" || k == "data") && x.is_object();
            flatten(if wrapper { "" } else { k }, x, &mut flat, &mut self.skipped_values, 0);
        }
        let mut row = Row::default();
        let mut best: Option<usize> = None;
        let mut rest = Vec::with_capacity(flat.len());
        for (k, x) in flat {
            if self.own {
                match k.as_str() {
                    "step" => row.step = Some(x),
                    "t" => row.t = Some(x),
                    _ => rest.push((k, x)),
                }
                continue;
            }
            if let Some(r) = step_rank(&k) {
                if x.is_finite() && best.is_none_or(|b| r < b) {
                    best = Some(r);
                    row.step = Some(x);
                }
                continue;
            }
            match k.as_str() {
                "_runtime" => row.t = Some(x),
                "_timestamp" | "timestamp" | "wall_time" => {
                    if row.t.is_none() && x.is_finite() {
                        let t0 = *self.t0.get_or_insert(x);
                        row.t = Some(x - t0);
                    }
                }
                _ => {
                    if is_epoch(&k) && x.is_finite() {
                        row.epoch = Some(x);
                    }
                    rest.push((k, x));
                }
            }
        }
        row.kv = rest.into_iter().map(|(k, x)| (self.rows.id(&k), x)).collect();
        if row.kv.is_empty() {
            return false;
        }
        self.rows.push(row);
        true
    }

    pub fn finish(self, run: &mut LocalRun, rep: &mut FileReport) {
        if self.skipped_values > 0 {
            rep.notes.push(format!("{} non-scalar value(s) skipped (media, arrays, text)", self.skipped_values));
        }
        self.rows.finish(run, rep, Missing::CarryForward, 1);
    }
}

fn flatten(prefix: &str, v: &Value, out: &mut Vec<(String, f64)>, skipped: &mut usize, depth: usize) {
    let key = |k: &str| if prefix.is_empty() { k.to_string() } else { format!("{prefix}/{k}") };
    match v {
        Value::Number(n) => out.push((prefix.to_string(), n.as_f64().unwrap_or(f64::NAN))),
        Value::Bool(b) => out.push((prefix.to_string(), if *b { 1.0 } else { 0.0 })),
        Value::Null => out.push((prefix.to_string(), f64::NAN)),
        Value::String(s) => match parse_value(s) {
            Some(x) => out.push((prefix.to_string(), x)),
            None => *skipped += 1,
        },
        Value::Object(o) if o.contains_key("_type") => *skipped += 1,
        Value::Object(o) if depth < 6 => {
            for (k, x) in o {
                flatten(&key(k), x, out, skipped, depth + 1);
            }
        }
        _ => *skipped += 1,
    }
}
