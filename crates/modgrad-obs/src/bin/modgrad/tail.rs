//! `modgrad tail <run | run dir>`: latest value, range and a braille sparkline
//! per metric, from a server (`/api/v1/runs/<id>/history`) or a local run
//! directory. `--follow` redraws every `--interval` seconds; `--json`
//! prints one JSON object per refresh instead.

use crate::net;
use modgrad_obs::spark::{stats, table};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::io::IsTerminal;
use std::path::Path;
use std::time::Duration;

pub struct Opts {
    pub target: String,
    pub server: String,
    pub keys: Vec<String>,
    pub follow: bool,
    pub interval: f64,
    pub json: bool,
    pub width: usize,
}

#[derive(Default)]
struct Series {
    steps: Vec<u64>,
    cols: BTreeMap<String, Vec<f64>>,
}

impl Series {
    /// Merge history rows (from `from` on) into the series.
    fn merge(&mut self, steps: &[u64], cols: &BTreeMap<String, Vec<f64>>) {
        let Some(&first) = steps.first() else { return };
        // drop our rows at or after the first incoming step (they may have changed)
        let keep = self.steps.partition_point(|&s| s < first);
        self.steps.truncate(keep);
        for c in self.cols.values_mut() {
            c.truncate(keep);
        }
        for (k, v) in cols {
            let c = self.cols.entry(k.clone()).or_insert_with(|| vec![f64::NAN; keep]);
            c.resize(keep, f64::NAN);
            c.extend_from_slice(v);
        }
        self.steps.extend_from_slice(steps);
        let n = self.steps.len();
        for c in self.cols.values_mut() {
            c.resize(n, f64::NAN);
        }
    }
}

fn fetch_remote(o: &Opts, from: u64) -> Result<(Vec<u64>, BTreeMap<String, Vec<f64>>), String> {
    let mut url = format!("{}/api/v1/runs/{}/history?from={from}", o.server.trim_end_matches('/'), o.target);
    if !o.keys.is_empty() {
        url.push_str(&format!("&keys={}", o.keys.join(",")));
    }
    let r = net::request("GET", &url, None, None, "application/json", Duration::from_secs(10)).map_err(|e| format!("{}: {e}", o.server))?;
    if !r.ok() {
        return Err(r.error_text());
    }
    let j = r.json();
    let steps = j["steps"].as_array().map(|a| a.iter().filter_map(|s| s.as_u64()).collect()).unwrap_or_default();
    let cols = j["columns"].as_object().map(|m| m.iter().map(|(k, v)| (k.clone(), v.as_array().map(|a| a.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect()).unwrap_or_default())).collect()).unwrap_or_default();
    Ok((steps, cols))
}

fn fetch_local(dir: &Path, keys: &[String]) -> Result<(Vec<u64>, BTreeMap<String, Vec<f64>>), String> {
    let rd = modgrad_obs::read_run(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let keys = if keys.is_empty() { rd.metric_keys() } else { keys.to_vec() };
    let mut rows: BTreeMap<u64, BTreeMap<String, f64>> = BTreeMap::new();
    for r in &rd.metrics {
        let e = rows.entry(r.step).or_default();
        for k in &keys {
            if let Some(v) = r.values.get(k) {
                e.insert(k.clone(), *v);
            }
        }
    }
    let steps: Vec<u64> = rows.keys().copied().collect();
    let cols = keys.iter().map(|k| (k.clone(), rows.values().map(|m| m.get(k).copied().unwrap_or(f64::NAN)).collect())).collect();
    Ok((steps, cols))
}

fn render(o: &Opts, s: &Series) -> String {
    let step = s.steps.last().copied();
    if o.json {
        let keys: serde_json::Map<String, Value> = s
            .cols
            .iter()
            .map(|(k, c)| {
                let (last, lo, hi, n) = stats(c);
                let f = |x: f64| if x.is_finite() { json!(x) } else { Value::Null };
                (k.clone(), json!({ "last": last, "min": f(lo), "max": f(hi), "n": n }))
            })
            .collect();
        return json!({ "run": o.target, "step": step, "rows": s.steps.len(), "keys": keys }).to_string();
    }
    table(&o.target, &s.steps, &s.cols, o.width)
}

pub fn tail(o: Opts) -> i32 {
    let local = Path::new(&o.target).join("config.json").is_file();
    let mut s = Series::default();
    let tty = std::io::stdout().is_terminal();
    loop {
        let from = s.steps.last().copied().unwrap_or(0);
        let got = if local { fetch_local(Path::new(&o.target), &o.keys) } else { fetch_remote(&o, from) };
        match got {
            Ok((steps, cols)) => {
                if local {
                    s = Series::default();
                }
                s.merge(&steps, &cols);
                // "check the run has rows" must be able to fail
                if s.steps.is_empty() && !o.follow {
                    eprintln!("modgrad tail: {} has no rows yet: nothing has been logged (did the training run under `modgrad run`?)", o.target);
                    return 2;
                }
                let text = render(&o, &s);
                if o.follow && tty && !o.json {
                    print!("\x1b[H\x1b[2J{text}");
                } else {
                    println!("{}", text.trim_end());
                }
            }
            Err(e) => {
                eprintln!("modgrad tail: {e}");
                if !o.follow {
                    return 1;
                }
            }
        }
        if !o.follow {
            return 0;
        }
        std::thread::sleep(Duration::from_secs_f64(o.interval.max(0.2)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_replaces_overlap() {
        let mut s = Series::default();
        let c = |v: Vec<f64>| BTreeMap::from([("loss".to_string(), v)]);
        s.merge(&[0, 1, 2], &c(vec![3.0, 2.0, 1.0]));
        s.merge(&[2, 3], &c(vec![1.5, 0.5]));
        assert_eq!(s.steps, vec![0, 1, 2, 3]);
        assert_eq!(s.cols["loss"], vec![3.0, 2.0, 1.5, 0.5]);
        // a key that appears later is padded
        s.merge(&[4], &BTreeMap::from([("acc".to_string(), vec![0.9]), ("loss".to_string(), vec![0.4])]));
        assert_eq!(s.cols["acc"].len(), 5);
        assert!(s.cols["acc"][0].is_nan());
    }
}
