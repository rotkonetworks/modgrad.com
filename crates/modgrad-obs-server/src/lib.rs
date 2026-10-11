//! modgrad-obs-server: Zebra-style tower services over the modgrad-obs
//! store. See `docs/design/modgrad-server.md`.
//!
//! * [`State`] — single writer (put it behind `tower::buffer::Buffer`).
//! * [`ReadState`] — clonable reads over immutable snapshots.
//! * [`ingest_service`] — the client write API with batching and load
//!   shedding layers.
//! * [`http`] — a thin axum adapter translating HTTP into these requests;
//!   [`v1`] is the documented, authenticated ingest API on top of it.
//! * [`Authed`] — write tokens in front of the ingest stack.
//! * [`Live`] — the change feed behind the SSE endpoints.
//! * `h3` (feature `h3`) — the same router over HTTP/3 (QUIC).
//! * `ssh` (feature `ssh`) — `python train.py | ssh <server>`: the key is the account.
//! * [`webauthn`] — passkey registration and assertion verification (pure).
//!
//! The server stores and selects; all analysis runs in the browser (wasm).

pub mod auth;
pub mod error;
#[cfg(feature = "h3")]
pub mod h3;
pub mod http;
pub mod ingest;
pub mod live;
pub mod mcp_http;
pub mod read;
#[cfg(feature = "ssh")]
pub mod ssh;
pub mod state;
pub mod v1;
pub mod webauthn;
pub mod wire;

pub use auth::{Auth, AuthError, Authed, AuthedRequest, AuthedResponse};
pub use live::{Live, LiveKind, LiveMsg};
pub use error::{BoxError, IngestError, ReadError, StateError};
pub use ingest::{BatchConfig, BatchLayer, Batcher, IngestConfig, IngestRequest, IngestResponse, IngestService, TraceLayer, ingest_service};
pub use read::{ArrayBlock, HistoryBlock, ReadRequest, ReadResponse, ReadState};
pub use state::{AUTO_STEP, Record, RunCommit, Snapshot, State, StateRequest, StateResponse, valid_run_name};

/// Open a store and wire the services: `State` behind a `Buffer`, the
/// ingest stack on top, and a `ReadState` on the snapshot channel.
pub fn open(root: &std::path::Path, cfg: IngestConfig) -> Result<(IngestService, ReadState), StateError> {
    let (state, rx) = State::open(root)?;
    let buffered = tower::buffer::Buffer::new(state, 1024);
    Ok((ingest_service(buffered, cfg), ReadState::new(rx)))
}
