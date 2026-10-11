//! wasm-bindgen surface for the dashboard's drop worker (obs/drop-worker.js).
//!
//! ```js
//! const b = new RunBuilder('my-run');
//! b.begin('logs/train.csv', file.lastModified / 1000);
//! for (chunk of file) b.push(chunk);       // Uint8Array, any size
//! const report = JSON.parse(b.end());      // per-file report
//! const meta = JSON.parse(b.finish());     // config, summary, arrays, events, keys ...
//! b.steps(i), b.values(i)                  // Float64Array per metric key i
//! ```

use crate::{Ingest, LocalRun};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct RunBuilder {
    run: LocalRun,
    cur: Option<Ingest>,
    keys: Vec<String>,
}

#[wasm_bindgen]
impl RunBuilder {
    #[wasm_bindgen(constructor)]
    pub fn new(name: &str) -> RunBuilder {
        RunBuilder { run: LocalRun::new(name), cur: None, keys: Vec::new() }
    }

    /// Start a file (`name` may be a relative path; its basename picks the
    /// parser together with the first bytes).
    pub fn begin(&mut self, name: &str, mtime_unix: f64) {
        let mut ing = Ingest::new(name);
        ing.mtime_unix = mtime_unix;
        self.cur = Some(ing);
    }

    pub fn push(&mut self, chunk: &[u8]) {
        if let Some(c) = self.cur.as_mut() {
            c.push(chunk);
        }
    }

    /// Finish the current file; returns its report as JSON.
    pub fn end(&mut self) -> String {
        match self.cur.take() {
            Some(c) => c.finish(&mut self.run).to_json().to_string(),
            None => "null".into(),
        }
    }

    /// Finish the run: sort/dedupe series, returns the run metadata JSON
    /// (`keys` lists the metric keys in the order of `steps(i)`/`values(i)`).
    pub fn finish(&mut self) -> String {
        let dups = self.run.finalize();
        self.keys = self.run.series.keys().cloned().collect();
        let mut m = self.run.meta_json();
        m["keys"] = serde_json::json!(self.keys);
        m["duplicate_steps"] = serde_json::json!(dups);
        m.to_string()
    }

    pub fn n_keys(&self) -> usize {
        self.keys.len()
    }
    pub fn steps(&self, i: usize) -> Vec<f64> {
        self.keys.get(i).and_then(|k| self.run.series.get(k)).map(|s| s.step.clone()).unwrap_or_default()
    }
    pub fn values(&self, i: usize) -> Vec<f64> {
        self.keys.get(i).and_then(|k| self.run.series.get(k)).map(|s| s.value.clone()).unwrap_or_default()
    }
    pub fn time_steps(&self) -> Vec<f64> {
        self.run.time.iter().map(|p| p.0).collect()
    }
    pub fn time_values(&self) -> Vec<f64> {
        self.run.time.iter().map(|p| p.1).collect()
    }
}

/// The format a file would be parsed as (`csv`, `jsonl`, `tensorboard`, ...).
#[wasm_bindgen]
pub fn detect_format(name: &str, head: &[u8]) -> String {
    crate::detect(name, head).name().into()
}
