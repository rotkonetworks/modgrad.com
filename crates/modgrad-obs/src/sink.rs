//! Where an importer writes: a [`Run`] on disk, or any in-memory collector
//! (the browser ingest in `modgrad-obs-ingest` parses logs into memory on
//! wasm32, where there is no filesystem). The importers (`legacy`, the
//! W&B import) are written against this trait so both paths share one
//! parser.

use crate::{Histogram, Meta, ObsError, Run};
use serde_json::Value;
use std::io;
use std::path::Path;

/// The subset of [`Run`]'s writer API that importers use (explicit times).
pub trait Sink {
    /// What closing the sink yields (`()` for a [`Run`] on disk).
    type Output;
    fn log_with_time(&mut self, step: u64, t: f64, kv: &[(&str, f64)]) -> Result<(), ObsError>;
    fn log_array_with_time(&mut self, step: u64, t: f64, key: &str, shape: &[usize], data: &[f32]) -> Result<(), ObsError>;
    fn log_histogram_with_time(&mut self, step: u64, t: f64, key: &str, h: &Histogram) -> Result<(), ObsError>;
    fn event_with_time(&mut self, step: u64, t: f64, text: &str) -> Result<(), ObsError>;
    /// `run.summary[key] = value`.
    fn summary_value(&mut self, key: &str, value: &Value) -> io::Result<()>;
    /// Merge `extra` into the summary, mark finished at `elapsed_s`, close.
    fn finish_with_time(self, extra: &Value, elapsed_s: f64) -> io::Result<Self::Output>;
    /// Close without marking the run finished (still running, or died).
    fn close(self) -> io::Result<Self::Output>;
}

impl Sink for Run {
    type Output = ();
    fn log_with_time(&mut self, step: u64, t: f64, kv: &[(&str, f64)]) -> Result<(), ObsError> {
        Run::log_with_time(self, step, t, kv)
    }
    fn log_array_with_time(&mut self, step: u64, t: f64, key: &str, shape: &[usize], data: &[f32]) -> Result<(), ObsError> {
        Run::log_array_with_time(self, step, t, key, shape, data)
    }
    fn log_histogram_with_time(&mut self, step: u64, t: f64, key: &str, h: &Histogram) -> Result<(), ObsError> {
        Run::log_histogram_with_time(self, step, t, key, h)
    }
    fn event_with_time(&mut self, step: u64, t: f64, text: &str) -> Result<(), ObsError> {
        Run::event_with_time(self, step, t, text)
    }
    fn summary_value(&mut self, key: &str, value: &Value) -> io::Result<()> {
        Run::summary(self, key, value)
    }
    fn finish_with_time(self, extra: &Value, elapsed_s: f64) -> io::Result<()> {
        Run::finish_with_time(self, extra, elapsed_s)
    }
    fn close(self) -> io::Result<()> {
        Ok(())
    }
}

/// A [`Sink`] factory: given the run name, the parsed config and [`Meta`],
/// open the sink (importers learn the config before the first metric).
pub trait OpenSink {
    type Sink: Sink;
    fn open(self, name: &str, config: &Value, meta: Meta) -> io::Result<Self::Sink>;
}

/// Opens a [`Run`] at a directory (the on-disk importers).
pub struct RunAt<'a>(pub &'a Path);

impl OpenSink for RunAt<'_> {
    type Sink = Run;
    fn open(self, name: &str, config: &Value, meta: Meta) -> io::Result<Run> {
        Run::create_in(self.0, name, config, meta)
    }
}
