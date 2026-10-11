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
| `GET  /` and other static paths         | the `--static` dir: `obs/web/dist` (the app, `dashboard.html` = `index.html`) |
| `GET  /api/v1/boards/{name}`            | `<root>/boards/<name>.json` (MCP `make_board`), for `#board=<name>` |

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

## MCP server for coding agents (`obs mcp`)

`obs mcp` is a Model Context Protocol server on stdio, built into the `obs`
CLI (`crates/modgrad-obs/src/mcp/`). It is a client of the store, not part
of the server. Analysis (trend, spikes, plateau, forecast) runs in the
`obs mcp` process using `modgrad-obs-analysis`, the same code the dashboard
runs as wasm. Invariant 6 still holds: the server does no analysis.

```
obs mcp [--dir <obs root>] [--base-url <dashboard url>]
claude mcp add obs -- /path/to/obs mcp --dir /path/to/obs
```

`--dir` is the root that holds `runs/` (default `$OBS_DIR`, else `./obs`).
`--base-url` is the dashboard to link boards to. Without it, boards link to
`file://<root>/dashboard.html#board=…`. `--url <server> [--token T]` selects
remote mode. Remote mode is not implemented yet: every tool returns an error
that tells the agent to use `--dir`.

The protocol layer is hand-written JSON-RPC 2.0, one message per line. It
implements `initialize`, `ping`, `tools/list` and `tools/call`, and returns
empty lists for resources and prompts. The official SDK would pull tokio and
schemars into a crate that has neither.

| tool          | does                                                                 |
|---------------|----------------------------------------------------------------------|
| `log`         | append one row `{metric: number}` at a step (default: last step + 1); creates the run |
| `ingest_log`  | parse raw stdout (`step N loss=…`, `step N \| loss … \| acc …%`, `ctmlm_pilot` lines) into rows; reports skipped lines |
| `list_runs`   | table of runs: last step, metric count, finished / running / idle, last update |
| `read_metric` | last N values of one metric, plus min / max / last with steps        |
| `summarize`   | per key metric: latest value, best value, trend with ± and verdict, spikes, plateau, backtested forecast, ETA to a target, mean ± sd across seeds (`name`, `name_s2`, …) |
| `compare`     | one metric across runs: table at the steps they share, differences, best at each run's own last step and at the latest shared step |
| `make_board`  | save `boards/<slug>.json` from `metric{run=~"a\|b"}` queries; return its URL |

The tools are designed for weak local models:

- Arguments are flat strings and numbers. Argument names and tool names
  are matched case-insensitively and with aliases (`run_name`, `metric_name`,
  `listRuns`). Numbers may be strings (`"7.5%"`, `"3e-4"`). Arguments sent
  as a JSON string are parsed.
- Run names resolve in this order: exact, case-insensitive, last path
  component, punctuation-insensitive, unique substring. Metric names also
  match aliases (`lr` → `learning_rate`). On a miss, the error lists the
  closest names.
- Output is short plain text, kept under about 40 lines (the tests check this on fixtures; the code does not truncate). Each result ends with a
  `Next:` line that suggests the call to make next.
- A tool failure is returned as a result with `isError: true` and an example
  call, not as a JSON-RPC error, so the model can read it and retry.
- The forecast is a ridge model backtested against repeating the last value.
  If it does not beat the last value, the output says "no skill" and gives no
  ETA.

`bench/obs_agent_bench/` checks these claims against a local model; see its
README.
