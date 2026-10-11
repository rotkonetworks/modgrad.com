//! `Ingest`: the client-facing write API. [`Batcher`] translates
//! [`IngestRequest`]s into [`StateRequest`]s, grouping `Log`/`LogArray`/
//! `Event` per run into one `Append` (tower-batch style: flush at
//! `max_items` or `max_latency`). `CreateRun` and `Finish` are barriers:
//! everything queued before them is flushed first. [`ingest_service`]
//! stacks the middleware layers.

use crate::error::{BoxError, IngestError, StateError};
use crate::state::{Record, StateRequest, StateResponse};
use modgrad_obs::Item;
use serde_json::Value;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};
use tower::util::BoxCloneSyncService;
use tower::{Layer, Service, ServiceBuilder, ServiceExt};

#[derive(Debug, Clone)]
pub enum IngestRequest {
    CreateRun { name: String, project: Option<String>, config: Value, meta: Option<Value> },
    Log { run: String, step: u64, t: Option<f64>, items: Vec<(String, Item)> },
    LogArray { run: String, step: u64, t: Option<f64>, key: String, kind: String, shape: Vec<usize>, data: Vec<f32>, labels: Option<Vec<f32>> },
    Event { run: String, step: u64, t: Option<f64>, text: String },
    /// Several records for one run, kept in order and committed in one
    /// batch with one reply (what the v1 HTTP API sends; order matters for
    /// [`crate::state::AUTO_STEP`]).
    Rows { run: String, records: Vec<Record> },
    Finish { run: String, summary: Value },
    /// Mid-run summary merge (barrier, like `Finish`).
    Summary { run: String, summary: Value },
    /// Config merge (barrier).
    Config { run: String, config: Value },
    /// Remove every run of a project, files included (barrier). Only the
    /// account portal sends this (account deletion); [`crate::Authed`]
    /// refuses it.
    DeleteProject { project: String },
}

impl IngestRequest {
    pub fn kind(&self) -> &'static str {
        match self {
            IngestRequest::CreateRun { .. } => "create_run",
            IngestRequest::Log { .. } => "log",
            IngestRequest::LogArray { .. } => "log_array",
            IngestRequest::Event { .. } => "event",
            IngestRequest::Rows { .. } => "rows",
            IngestRequest::Finish { .. } => "finish",
            IngestRequest::Summary { .. } => "summary",
            IngestRequest::Config { .. } => "config",
            IngestRequest::DeleteProject { .. } => "delete_project",
        }
    }
    pub fn run(&self) -> &str {
        match self {
            IngestRequest::CreateRun { name, .. } => name,
            IngestRequest::Log { run, .. }
            | IngestRequest::LogArray { run, .. }
            | IngestRequest::Event { run, .. }
            | IngestRequest::Rows { run, .. }
            | IngestRequest::Finish { run, .. }
            | IngestRequest::Summary { run, .. }
            | IngestRequest::Config { run, .. } => run,
            IngestRequest::DeleteProject { project } => project,
        }
    }
    /// Split into (run, records) for batchable requests.
    fn into_records(self) -> Result<(String, Vec<Record>), IngestRequest> {
        match self {
            IngestRequest::Log { run, step, t, items } => Ok((run, vec![Record::Log { step, t, items }])),
            IngestRequest::LogArray { run, step, t, key, kind, shape, data, labels } => Ok((run, vec![Record::Array { step, t, key, kind, shape, data, labels }])),
            IngestRequest::Event { run, step, t, text } => Ok((run, vec![Record::Event { step, t, text }])),
            IngestRequest::Rows { run, records } => Ok((run, records)),
            other => Err(other),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum IngestResponse {
    Created { run: String },
    /// The record is on disk (its batch committed).
    Accepted,
    Finished { run: String },
    /// A summary or config merge committed.
    Updated { run: String },
}

#[derive(Debug, Clone, Copy)]
pub struct BatchConfig {
    pub max_items: usize,
    pub max_latency: Duration,
    /// Bound of the queue in front of the worker.
    pub queue: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        BatchConfig { max_items: 256, max_latency: Duration::from_millis(20), queue: 4096 }
    }
}

type Reply = oneshot::Sender<Result<IngestResponse, IngestError>>;

/// tower-batch style batcher over a `State`-like service.
#[derive(Clone)]
pub struct Batcher {
    tx: mpsc::Sender<(IngestRequest, Reply)>,
}

#[derive(Clone, Copy)]
pub struct BatchLayer(pub BatchConfig);

impl<S> Layer<S> for BatchLayer
where
    S: Service<StateRequest, Response = StateResponse> + Send + 'static,
    S::Error: Into<BoxError>,
    S::Future: Send,
{
    type Service = Batcher;
    fn layer(&self, inner: S) -> Batcher {
        Batcher::new(inner, self.0)
    }
}

fn state_err(e: BoxError) -> IngestError {
    match e.downcast::<StateError>() {
        Ok(s) => IngestError::State(*s),
        Err(e) => IngestError::Closed(e.to_string()),
    }
}

impl Batcher {
    /// Spawn the worker task (needs a tokio runtime).
    pub fn new<S>(state: S, cfg: BatchConfig) -> Batcher
    where
        S: Service<StateRequest, Response = StateResponse> + Send + 'static,
        S::Error: Into<BoxError>,
        S::Future: Send,
    {
        let (tx, rx) = mpsc::channel(cfg.queue.max(1));
        tokio::spawn(worker(state, rx, cfg));
        Batcher { tx }
    }
}

async fn call_state<S>(state: &mut S, req: StateRequest) -> Result<StateResponse, IngestError>
where
    S: Service<StateRequest, Response = StateResponse>,
    S::Error: Into<BoxError>,
{
    let svc = state.ready().await.map_err(|e| state_err(e.into()))?;
    svc.call(req).await.map_err(|e| state_err(e.into()))
}

/// Pending records grouped per run, in first-arrival order of runs.
#[derive(Default)]
struct Pending {
    order: Vec<String>,
    by_run: std::collections::HashMap<String, (Vec<Record>, Vec<Reply>)>,
    items: usize,
}

impl Pending {
    fn push(&mut self, run: String, recs: Vec<Record>, reply: Reply) {
        if !self.by_run.contains_key(&run) {
            self.order.push(run.clone());
        }
        let e = self.by_run.entry(run).or_default();
        self.items += recs.len().max(1);
        e.0.extend(recs);
        e.1.push(reply);
    }
    async fn flush<S>(&mut self, state: &mut S)
    where
        S: Service<StateRequest, Response = StateResponse>,
        S::Error: Into<BoxError>,
    {
        for run in std::mem::take(&mut self.order) {
            let (records, replies) = self.by_run.remove(&run).unwrap();
            let span = tracing::debug_span!("flush", run = %run, records = records.len());
            let _g = span.enter();
            let res = call_state(state, StateRequest::Append { run, records }).await;
            for r in replies {
                let _ = r.send(res.clone().map(|_| IngestResponse::Accepted));
            }
        }
        self.items = 0;
    }
}

async fn worker<S>(mut state: S, mut rx: mpsc::Receiver<(IngestRequest, Reply)>, cfg: BatchConfig)
where
    S: Service<StateRequest, Response = StateResponse>,
    S::Error: Into<BoxError>,
{
    let mut pending = Pending::default();
    while let Some(first) = rx.recv().await {
        let mut next = Some(first);
        let deadline = tokio::time::Instant::now() + cfg.max_latency;
        loop {
            let Some((req, reply)) = next.take() else { break };
            match req.into_records() {
                Ok((run, recs)) => pending.push(run, recs, reply),
                Err(barrier) => {
                    pending.flush(&mut state).await;
                    let res = match barrier {
                        IngestRequest::CreateRun { name, project, config, meta } => {
                            call_state(&mut state, StateRequest::CreateRun { name, project, config, meta }).await.map(|r| match r {
                                StateResponse::Created { run } => IngestResponse::Created { run },
                                _ => IngestResponse::Accepted,
                            })
                        }
                        IngestRequest::Finish { run, summary } => {
                            call_state(&mut state, StateRequest::Finish { run: run.clone(), summary }).await.map(|_| IngestResponse::Finished { run })
                        }
                        IngestRequest::Summary { run, summary } => {
                            call_state(&mut state, StateRequest::Summary { run: run.clone(), summary }).await.map(|_| IngestResponse::Updated { run })
                        }
                        IngestRequest::Config { run, config } => {
                            call_state(&mut state, StateRequest::Config { run: run.clone(), config }).await.map(|_| IngestResponse::Updated { run })
                        }
                        IngestRequest::DeleteProject { project } => {
                            call_state(&mut state, StateRequest::DeleteProject { project: project.clone() }).await.map(|_| IngestResponse::Updated { run: project })
                        }
                        _ => unreachable!(),
                    };
                    let _ = reply.send(res);
                    break;
                }
            }
            if pending.items >= cfg.max_items {
                break;
            }
            // keep collecting until the batch is full or the deadline passes
            match tokio::time::timeout_at(deadline, rx.recv()).await {
                Ok(Some(m)) => next = Some(m),
                Ok(None) | Err(_) => break,
            }
        }
        pending.flush(&mut state).await;
    }
}

impl Service<IngestRequest> for Batcher {
    type Response = IngestResponse;
    type Error = IngestError;
    type Future = Pin<Box<dyn Future<Output = Result<IngestResponse, IngestError>> + Send>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), IngestError>> {
        // back-pressure is applied by ConcurrencyLimit/LoadShed in front and
        // by the bounded queue (send awaits capacity)
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: IngestRequest) -> Self::Future {
        let tx = self.tx.clone();
        Box::pin(async move {
            let (rtx, rrx) = oneshot::channel();
            tx.send((req, rtx)).await.map_err(|_| IngestError::Closed("batch worker stopped".into()))?;
            rrx.await.map_err(|_| IngestError::Closed("batch worker dropped the request".into()))?
        })
    }
}

/// One `tracing` span per request.
#[derive(Clone, Copy, Default)]
pub struct TraceLayer;

#[derive(Clone)]
pub struct Trace<S>(S);

impl<S> Layer<S> for TraceLayer {
    type Service = Trace<S>;
    fn layer(&self, s: S) -> Trace<S> {
        Trace(s)
    }
}

impl<S> Service<IngestRequest> for Trace<S>
where
    S: Service<IngestRequest>,
    S::Future: Send + 'static,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<S::Response, S::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), S::Error>> {
        self.0.poll_ready(cx)
    }

    fn call(&mut self, req: IngestRequest) -> Self::Future {
        use tracing::Instrument;
        let span = tracing::info_span!("ingest", kind = req.kind(), run = %req.run());
        Box::pin(self.0.call(req).instrument(span))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct IngestConfig {
    pub batch: BatchConfig,
    pub concurrency: usize,
    pub timeout: Duration,
}

impl Default for IngestConfig {
    fn default() -> Self {
        IngestConfig { batch: BatchConfig::default(), concurrency: 1024, timeout: Duration::from_secs(10) }
    }
}

pub type IngestService = BoxCloneSyncService<IngestRequest, IngestResponse, BoxError>;

/// The full ingest stack over a (usually `Buffer`ed) `State`:
/// Trace -> LoadShed -> ConcurrencyLimit -> Timeout -> Batch -> State.
/// Errors come back boxed; [`IngestError::classify`] recovers the type.
pub fn ingest_service<S>(state: S, cfg: IngestConfig) -> IngestService
where
    S: Service<StateRequest, Response = StateResponse> + Send + 'static,
    S::Error: Into<BoxError>,
    S::Future: Send,
{
    let svc = ServiceBuilder::new()
        .layer(TraceLayer)
        .load_shed()
        .concurrency_limit(cfg.concurrency)
        .timeout(cfg.timeout)
        .layer(BatchLayer(cfg.batch))
        .service(state);
    BoxCloneSyncService::new(svc.map_err(|e: BoxError| e))
}
