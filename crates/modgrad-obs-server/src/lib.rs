//! modgrad-obs-server: Zebra-style tower services over the modgrad-obs
//! store. See `docs/design/obs-server.md`.
//!
//! * [`State`] — single writer (put it behind `tower::buffer::Buffer`).
//! * [`ReadState`] — clonable reads over immutable snapshots.
//! * [`ingest_service`] — the client write API with batching and load
//!   shedding layers.
//! * [`http`] — a thin axum adapter translating HTTP into these requests.
//!
//! The server stores and selects; all analysis runs in the browser (wasm).

pub mod error;
pub mod http;
pub mod ingest;
pub mod read;
pub mod state;
pub mod wire;

pub use error::{BoxError, IngestError, ReadError, StateError};
pub use ingest::{BatchConfig, BatchLayer, Batcher, IngestConfig, IngestRequest, IngestResponse, IngestService, TraceLayer, ingest_service};
pub use read::{ArrayBlock, HistoryBlock, ReadRequest, ReadResponse, ReadState};
pub use state::{Record, RunCommit, Snapshot, State, StateRequest, StateResponse, valid_run_name};

/// Open a store and wire the services: `State` behind a `Buffer`, the
/// ingest stack on top, and a `ReadState` on the snapshot channel.
pub fn open(root: &std::path::Path, cfg: IngestConfig) -> Result<(IngestService, ReadState), StateError> {
    let (state, rx) = State::open(root)?;
    let buffered = tower::buffer::Buffer::new(state, 1024);
    Ok((ingest_service(buffered, cfg), ReadState::new(rx)))
}
