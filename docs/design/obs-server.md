# modgrad-obs-server

A small observability server for modgrad runs, structured the way Zebra
structures its node (services as values, explicit request/response enums,
middleware as tower layers). The server stores and serves data. It does no
analysis: smoothing, PCA, surface assembly, projection and legend statistics
all run in the browser through `modgrad-obs-analysis` compiled to wasm. The
same `.wasm` is used by the static (offline) dashboard and in server mode.

## Store

The store is the `modgrad-obs` on-disk format, unchanged:

```
<root>/runs/<run>/config.json      written once at CreateRun
                  metrics.jsonl    history (scalars), append-only
                  arrays.jsonl     tensors / histograms / trajectories / points / volumes
                  events.jsonl     free-text events
                  summary.json     written by Finish (temp file + rename)
```

So `scripts/obs_collect.sh` and the static dashboard keep working on a
server's root directory, and a run written by `modgrad_obs::Run` directly can
be served without conversion.

## Services

| service     | request enum      | response enum      | owner / cloning                                  |
|-------------|-------------------|--------------------|--------------------------------------------------|
| `State`     | `StateRequest`    | `StateResponse`    | single owner of all file handles, behind `tower::buffer::Buffer` |
| `ReadState` | `ReadRequest`     | `ReadResponse`     | `Clone`; reads immutable snapshots, never writes |
| `Ingest`    | `IngestRequest`   | `IngestResponse`   | translates client calls into `StateRequest`s     |

### `StateRequest` (writes)

- `CreateRun { name, project, config }` creates the run directory (or
  reopens it, see "Recovery").
- `Append { run, batch }`: a `Batch` holds history rows, array records and
  events for one run. The whole batch is serialised and written with one
  `write_all` per file, then flushed.
- `Finish { run, summary }` merges the summary and writes `summary.json`.

After every successful write the state publishes a new `Snapshot` through a
`tokio::sync::watch` channel.

### `ReadRequest` (selection only)

- `ListRuns { filter }`
- `Config(run)`
- `Summary(run)`
- `History { run, keys, range, stride }`: rows with `range.start <= step <
  range.end`, every `stride`-th row, only the requested keys.
- `Arrays { run, key, steps, index_range }`: the records of one key at the
  requested steps, each optionally cut to a flat element range.

These are data access, not analysis. There is no smoothing, aggregation or
PCA on the server. `stride` and `index_range` select existing values; they
never average them.

### `IngestRequest` (client API)

`CreateRun { .. }`, `Log { run, step, items }`, `LogArray { .. }`,
`Event { .. }`, `Finish { .. }`. `items` uses the same value types as
`modgrad_obs::Item`: scalar, histogram, tensor.

## Snapshots and the read path

A `Snapshot` is `Arc<BTreeMap<RunName, RunCommit>>`, where `RunCommit` holds
the run's config and summary plus the committed byte length of each JSONL
file. The writer publishes a new snapshot only after `flush` returns. A reader
reads each file's prefix `[0, committed_len)`, which:

- never contains a partial line, even while the writer is mid-append;
- is immutable, because files are append-only, so readers need no lock and
  can run concurrently with the writer and with each other.

`ReadState` clones a `watch::Receiver` and parses prefixes in
`spawn_blocking`.

## Layers

On `Ingest`, outermost first:

1. `TraceLayer` (ours): one `tracing` span per request (`ingest`, `kind`,
   `run`).
2. `LoadShed`: refuses requests when the inner service is not ready, instead
   of queueing without bound.
3. `ConcurrencyLimit(n)`.
4. `Timeout(d)`.
5. `BatchLayer` (ours, tower-batch style): `Log`, `LogArray` and `Event`
   requests are queued on a bounded channel to a worker. The worker groups them
   per run and flushes when it holds `max_items` items or `max_latency` has
   passed since the first one. A flush sends one `StateRequest::Append` per run,
   so a whole group is one append and one flush. Each caller's future resolves
   after its batch commits. `CreateRun` and `Finish` act as barriers: the worker
   flushes everything pending before forwarding them, so ordering is preserved.

`State` sits behind `Buffer`, so any number of `Ingest` clones share one
writer.

## Transport

`http.rs` (axum) only translates between HTTP and requests:

| route                                   | request                                   |
|-----------------------------------------|-------------------------------------------|
| `POST /api/runs`                        | `IngestRequest::CreateRun`                |
| `POST /api/runs/{run}/log`              | `IngestRequest::Log`                      |
| `POST /api/runs/{run}/event`            | `IngestRequest::Event`                    |
| `POST /api/runs/{run}/finish`           | `IngestRequest::Finish`                   |
| `GET  /api/index`                       | `ReadRequest::ListRuns` (same schema as `index.json`) |
| `GET  /api/runs/{run}/metrics/{key}`    | `ReadRequest::History` (same schema as the static `metrics/<key>.json`) |
| `GET  /api/runs/{run}/arrays/{key}`     | `ReadRequest::Arrays` (same schema as the static `arrays/<key>.json`) |
| `GET  /api/runs/{run}/history?keys=..`  | `ReadRequest::History`                    |
| `GET  /` and other static paths         | `dashboard.html`, `pkg/`                  |

`history` and `arrays` return JSON by default and the binary frame format
when the client sends `Accept: application/x-obs-frames`:

```
"OBS1" | u32 LE header_len | header JSON | payload
header = {"frames":[{"name":..,"dtype":"f32"|"f64","shape":[..]}], ...}
payload = the frames' little-endian values, back to back, in header order
```

The dashboard picks server mode when `api/index` answers, and otherwise falls
back to static mode (`index.json` next to the page). Both modes use the same
JSON shapes, so the drawing code has one path. A W&B-compatible API would be
another adapter of this kind, translating W&B's file-stream and GraphQL
calls into these requests without touching the services.

## Errors

Each service has a typed error (`thiserror`):

- `StateError`: `UnknownRun`, `InvalidRunName`, `Obs(ObsError)`, `Io`.
- `ReadError`: `UnknownRun`, `UnknownKey`, `Io`.
- `IngestError`: `State(StateError)`, `Overloaded`, `Timeout`, `Closed`.

Layer errors (`BoxError`) are downcast into `IngestError` at the boundary.
HTTP maps them to 400, 404, 503 and 504.

## Invariants

1. Single writer: exactly one task holds file handles (the `Buffer` worker),
   and there are no locks.
2. Append-only: lines are never rewritten. `summary.json` and `config.json`
   are replaced atomically.
3. Whole-line visibility: a reader never observes a partial line, because it
   reads only up to the committed length.
4. Per-run order: for one run, records reach disk in the order `Ingest`
   accepted them (one queue per batcher, FIFO grouping). Concurrent ingests
   for different runs, interleaved arbitrarily, produce the same store as
   sequential ingest. A property test checks this.
5. Recovery: opening a run whose JSONL file ends in a partial line (a crash
   mid-batch) truncates that file back to its last newline before
   appending, so a torn batch costs only itself and never corrupts the next
   record. A test checks this.
6. No analysis on the server.

## Testing

- Every service is called directly with `tower::ServiceExt::oneshot`; no
  HTTP is involved.
- `tower-test` mocks `State` to check that the batch layer turns N `Log`s for
  one run into a single `Append`, and that barriers flush first.
- A property test (proptest) runs concurrent per-run producers with random
  interleavings and compares the store with sequential ingest.
- A crash test writes a torn batch and checks that the store is still readable
  and that the next append is intact.
