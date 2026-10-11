# modgrad-obs as a product: onboarding, agents, transport

[`obs-server.md`](obs-server.md) is the server's internals (store, services,
invariants). This note covers the outside: how a person or a coding agent
gets from nothing to a live graph, what the API promises to callers that
cannot ask questions, and where the transport goes next. The API reference
itself is served as plain text at `GET /api/v1`
(`crates/modgrad-obs-server/src/v1_api.txt`). That file is the source of
truth, and this note does not repeat it.

## Onboarding

Goal: one command or one URL to a graph that updates live, with no account,
no create call and no config file.

Three entry points end in the same state, a run under a project that the
dashboard is already subscribed to:

| entry | what happens |
|---|---|
| open `$OBS/#new` | `live.js` calls `POST /api/v1/projects`, keeps the `obsp_` token in `sessionStorage`, rewrites the URL to `#project=anon-…` and shows the waiting panel |
| `obs run --server $OBS [--token T] --name N -- <cmd>` | wraps the training command, parses its stdout, prints the board link at start |
| `curl -X POST $OBS/api/v1/runs/N/log -d '{"loss":0.5}'` | the run is created on first write |

The waiting panel shows three ready-to-paste commands with the real server
URL and token filled in: a prompt for a coding agent, `obs run`, and `curl`.
On a dev server (`/api/index` reports `"dev": true`) the token is left out.
When the first `run` event for the target arrives, the panel goes away and
the charts render in the same tab. Nothing needs a reload.

URL targets (`obs/live.js`):

- `#run=<id>` selects only that run and waits for it if it does not exist yet.
- `#project=<p>` is a board: only that project's runs, and new ones appear live.
- `#new` issues an anonymous project, then behaves as `#project=`.

Anonymous projects (`anon-<16 hex>`) expire after 30 days unless claimed,
and they are left out of the unfiltered index. They can be reached by their
board URL, which cannot be guessed. Reads are open. The token holds write
access only. **Not built:** claiming. `GET /api/v1/claim/<c>` resolves the
project but cannot yet attach an SSH key or email.

Metric preselection: when no metric is selected, `live.js` picks up to four,
ranked in this order: `*loss`, then names containing loss, then
acc/ppl/reward/score, then other keys, with `sys/*` last. The dashboard's own
preferred list is specific to ctmlm, and a new user's run would otherwise
render with no charts.

Verified in a browser (screenshots in the session's scratch data): waiting
panel to live charts over SSE for `#run=`; `#new` to an empty board to live
charts with `EventSource` removed (polling path); zoom kept across appends;
no console errors.

## Agent-first

The main user of the write API is a coding agent that adds logging to a repo
it has never seen. It cannot ask questions, and a failed call costs a whole
loop iteration. The rules:

1. **Self-describing.** `GET /api/v1` returns the full reference as text, and
   the waiting panel's agent prompt points to it. An agent needs nothing else.
2. **Errors teach.** Every error is `{"error", "example", "help", "message"}`:
   what was wrong, a corrected request, and both as one paragraph. A 404 on an
   unknown action lists the valid actions.
3. **Forgiving input.** The body is parsed as JSON whatever the
   `Content-Type`. The API accepts one row, an array, NDJSON or an envelope;
   numeric strings such as `"7.5%"`; nested objects, which are flattened with
   `/`; and the W&B key `_step`. Text values are skipped, and their keys are
   reported under `ignored`, so the call does not fail.
4. **No mandatory create.** The first `log` creates the run. `step` is
   optional (default: last step + 1).
5. **Idempotent.** Rows merge by `(step, key)`, last write wins, so retries
   and resends are safe. `obs run` resolves steps on the client so that its
   resends are idempotent too.
6. **Raw text is a first-class input.** `POST …/text` takes stdout as it is
   and parses it with `textlog::parse_line`, the same parser `obs run` uses.
7. **CLI for scripts.** `--json` turns the CLI's own messages into JSON lines
   on stderr. Nothing prompts. The exit code is the child's (128+signal).
8. **Crash-safe.** `obs run` appends rows to a spool and fsyncs before
   sending, and records the acked offset after each POST. Anything not sent
   by exit stays on disk for `obs push <dir>`.

## Live transport

Server-Sent Events on `GET /api/v1/stream[?runs=|?project=]` and
`GET /api/v1/runs/<id>/stream`, with the event kinds `hello`, `run`, `rows`,
`events`, `summary` and `resync`. A client that falls behind is sent
`resync` instead of being buffered without limit.

`obs/live-client.js` is the browser side. It has no DOM access and extends
`EventTarget`, so the planned SolidJS port can reuse it unchanged. It falls
back to polling (`api/index` every 10 s, `history?from=` every 3 s for watched
runs) when `EventSource` is missing, the stream does not open within 5 s, or
three errors arrive in a row. Both paths emit the same events. A second
`hello` means a reconnect, and the client turns it into `resync`.

`obs/live.js` patches the cached metric JSON in place, so later re-renders
see the new points. It replaces open charts' series through `chart.update()`,
which redraws without rebuilding the panel and keeps the zoom. It does a full
redraw, throttled to one every 3 s, only for the time axis and forecasts,
which are derived per render. The run table is rebuilt at most every 2 s.

## QUIC (planned, not built)

Today the CLI talks HTTP/1.1 with one short connection per POST (`net.rs`,
std only, and `https://` through the `curl` binary so that no TLS stack is
linked). That is enough at the default rate. The costs show up at high log
rates and on lossy links: one handshake per batch, head-of-line blocking,
and a reconnect after every network change.

Plan:

- **One bidirectional QUIC stream per run.** The client sends spool frames
  (`offset, rows`). The server replies with the acked offset. The spool's
  acked offset becomes the stream's flow control, so resume after a drop
  means sending from that offset again. The idempotency rule above makes
  duplicates harmless.
- **Connection migration and 0-RTT**, so a laptop that changes networks or
  sleeps keeps streaming without a new handshake.
- **The same bearer tokens.** They are sent in the first frame, and auth goes
  through the same `Authed` service as HTTP. QUIC is another adapter in front
  of `IngestService`, like `http.rs`, and does not change the services.
- **Reuse, don't add:** `modgrad-io` already ships a `quinn` + `rustls`
  telemetry transport (`telemetry_transport.rs`, feature `telemetry`). The
  obs transport should share its endpoint setup instead of adding a second
  QUIC stack.
- **HTTP stays canonical.** The CLI tries QUIC and falls back to HTTP when
  UDP is blocked, and `curl` and the browser never need QUIC. The server
  read path (SSE) does not change.

Gate before building it: measure what one POST per batch costs in `obs run`
at its real rate. If HTTP keep-alive (one connection per run) closes the
gap, do that first; it is a much smaller change.
