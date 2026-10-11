//! Typed errors, one per service.

use std::sync::Arc;

pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

#[derive(Debug, Clone, thiserror::Error)]
pub enum StateError {
    #[error("unknown run {0:?}")]
    UnknownRun(String),
    #[error("invalid run name {0:?} (segments of [A-Za-z0-9_.-], at most 3, no '..')")]
    InvalidRunName(String),
    #[error("invalid record: {0}")]
    Obs(String),
    #[error("io: {0}")]
    Io(Arc<std::io::Error>),
}

impl From<std::io::Error> for StateError {
    fn from(e: std::io::Error) -> Self {
        StateError::Io(Arc::new(e))
    }
}
impl From<modgrad_obs::ObsError> for StateError {
    fn from(e: modgrad_obs::ObsError) -> Self {
        match e {
            modgrad_obs::ObsError::Io(io) => StateError::Io(Arc::new(io)),
            other => StateError::Obs(other.to_string()),
        }
    }
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum ReadError {
    #[error("unknown run {0:?}")]
    UnknownRun(String),
    #[error("run {run:?} has no key {key:?}")]
    UnknownKey { run: String, key: String },
    #[error("io: {0}")]
    Io(Arc<std::io::Error>),
    #[error("read task failed: {0}")]
    Join(String),
}

impl From<std::io::Error> for ReadError {
    fn from(e: std::io::Error) -> Self {
        ReadError::Io(Arc::new(e))
    }
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum IngestError {
    #[error(transparent)]
    State(#[from] StateError),
    #[error("overloaded: request shed")]
    Overloaded,
    #[error("timed out")]
    Timeout,
    #[error("ingest pipeline closed: {0}")]
    Closed(String),
}

impl IngestError {
    /// Recover a typed error from whatever the layer stack returned.
    pub fn classify(e: BoxError) -> IngestError {
        if e.is::<tower::load_shed::error::Overloaded>() {
            return IngestError::Overloaded;
        }
        if e.is::<tower::timeout::error::Elapsed>() {
            return IngestError::Timeout;
        }
        let e = match e.downcast::<IngestError>() {
            Ok(i) => return *i,
            Err(e) => e,
        };
        match e.downcast::<StateError>() {
            Ok(s) => IngestError::State(*s),
            Err(e) => IngestError::Closed(e.to_string()),
        }
    }
}
