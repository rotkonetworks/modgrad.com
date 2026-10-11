//! Thin HTTP adapter (axum): every handler only translates between HTTP
//! and an [`IngestRequest`] or [`ReadRequest`]. No logic lives here that a
//! W&B-compatible adapter would have to duplicate.

use crate::auth::{Auth, AuthError, Authed};
use crate::error::{IngestError, ReadError};
use crate::live::Live;
use crate::ingest::{IngestRequest, IngestResponse, IngestService};
use crate::read::{ReadRequest, ReadResponse, ReadState};
use crate::wire::{self, Frame, FrameData};
use axum::Router;
use axum::body::Body;
use axum::extract::{Path, Query, State as AxState};
use axum::http::{HeaderMap, StatusCode, Uri, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use base64::Engine;
use modgrad_obs::{Histogram, Item, Tensor};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::PathBuf;
use tower::ServiceExt;

#[derive(Clone)]
pub struct AppState {
    pub ingest: IngestService,
    pub read: ReadState,
    /// Directory served for non-API paths (dashboard.html, pkg/); None = API only.
    pub static_dir: Option<PathBuf>,
    /// Write tokens for the v1 API (and the dev-mode switch).
    pub auth: Auth,
    /// Change feed behind the SSE endpoints.
    pub live: Live,
    /// Where `PUT /api/v1/boards/<name>` stores board JSON; None = no boards.
    pub boards: Option<PathBuf>,
    /// `--mcp`: this server's own loopback base URL, which the hosted MCP
    /// endpoint's tools call; None = no `/mcp` route.
    pub mcp_self: Option<String>,
}

impl AppState {
    /// Wire the v1 extras around an ingest/read pair (spawns the [`Live`] task).
    pub fn new(ingest: IngestService, read: ReadState, static_dir: Option<PathBuf>, auth: Auth) -> AppState {
        let live = Live::spawn(read.clone());
        AppState { ingest, read, static_dir, auth, live, boards: None, mcp_self: None }
    }
    /// Store boards under `dir` (`<obs root>/boards`, the same place
    /// `modgrad mcp --dir` writes them).
    pub fn with_boards(mut self, dir: PathBuf) -> AppState {
        self.boards = Some(dir);
        self
    }
    /// Serve the hosted MCP endpoint at `/mcp`; `self_url` is how this
    /// process reaches its own HTTP API (e.g. `http://127.0.0.1:8790`).
    pub fn with_mcp(mut self, self_url: String) -> AppState {
        self.mcp_self = Some(self_url);
        self
    }
    /// The ingest service behind write authorisation.
    pub fn authed(&self) -> Authed {
        Authed { auth: self.auth.clone(), read: self.read.clone(), inner: self.ingest.clone() }
    }
}

#[derive(Debug)]
pub enum ApiError {
    /// An error that says what was wrong and shows a corrected request.
    Teach { status: StatusCode, error: String, example: &'static str },
    Auth(AuthError),
    Ingest(IngestError),
    Read(ReadError),
    Bad(String),
    NotFound,
}

impl IntoResponse for ApiError {
    /// Every error body is `{error, example, help, message}`: what was
    /// wrong, a corrected request, where the reference is, and both as one
    /// plain-text paragraph (for agents that only show text).
    fn into_response(self) -> Response {
        use crate::error::StateError as S;
        use crate::v1::{EX_CREATE, EX_LOG, EX_TOKEN};
        const EX_READ: &str = r#"curl "$OBS/api/index"   # lists runs; then GET /api/runs/<id with / as %2F>/metrics/<key>"#;
        let (code, msg, example) = match &self {
            ApiError::Teach { status, error, example } => (*status, error.clone(), *example),
            ApiError::Bad(m) => (StatusCode::BAD_REQUEST, m.clone(), EX_LOG),
            ApiError::NotFound => (StatusCode::NOT_FOUND, "not found".into(), EX_READ),
            ApiError::Auth(e @ (AuthError::Unauthorized | AuthError::Expired)) => (StatusCode::UNAUTHORIZED, e.to_string(), EX_TOKEN),
            ApiError::Auth(e @ AuthError::Forbidden(_)) => (StatusCode::FORBIDDEN, e.to_string(), EX_LOG),
            ApiError::Auth(e @ AuthError::Conflict(_)) => (StatusCode::CONFLICT, e.to_string(), EX_CREATE),
            ApiError::Auth(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string(), EX_LOG),
            ApiError::Read(e @ (ReadError::UnknownRun(_) | ReadError::UnknownKey { .. })) => (StatusCode::NOT_FOUND, e.to_string(), EX_READ),
            ApiError::Read(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string(), EX_READ),
            ApiError::Ingest(e @ IngestError::State(S::UnknownRun(_))) => (StatusCode::NOT_FOUND, e.to_string(), EX_LOG),
            ApiError::Ingest(e @ IngestError::State(S::InvalidRunName(_) | S::Obs(_))) => (StatusCode::BAD_REQUEST, e.to_string(), EX_LOG),
            ApiError::Ingest(e @ IngestError::Overloaded) => (StatusCode::SERVICE_UNAVAILABLE, format!("{e}; retry after a short backoff"), EX_LOG),
            ApiError::Ingest(e @ IngestError::Timeout) => (StatusCode::GATEWAY_TIMEOUT, format!("{e}; retry after a short backoff"), EX_LOG),
            ApiError::Ingest(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string(), EX_LOG),
        };
        let help = "GET /api/v1 for the API reference ($OBS is the server URL, $MODGRAD_TOKEN your token)";
        let body = json!({ "error": msg, "example": example, "help": help, "message": format!("{msg}\nExample: {example}\nHelp: {help}") });
        let mut r = (code, axum::Json(body)).into_response();
        if code == StatusCode::UNAUTHORIZED {
            r.headers_mut().insert(header::WWW_AUTHENTICATE, header::HeaderValue::from_static("Bearer"));
        }
        r
    }
}

impl ApiError {
    /// Recover the typed error from an [`Authed`] / ingest stack error.
    pub fn from_box(e: crate::BoxError) -> ApiError {
        match e.downcast::<AuthError>() {
            Ok(a) => ApiError::Auth(*a),
            Err(e) => ApiError::Ingest(IngestError::classify(e)),
        }
    }
}

type ApiResult = Result<Response, ApiError>;

async fn ingest(st: &AppState, req: IngestRequest) -> Result<IngestResponse, ApiError> {
    st.ingest.clone().oneshot(req).await.map_err(|e| ApiError::Ingest(IngestError::classify(e)))
}

async fn read(st: &AppState, req: ReadRequest) -> Result<ReadResponse, ApiError> {
    st.read.clone().oneshot(req).await.map_err(ApiError::Read)
}

fn ok_json(v: Value) -> ApiResult {
    Ok(axum::Json(v).into_response())
}

fn wants_frames(h: &HeaderMap) -> bool {
    h.get(header::ACCEPT).and_then(|v| v.to_str().ok()).is_some_and(|v| v.contains(wire::CONTENT_TYPE))
}

fn frames_response(bytes: Vec<u8>) -> Response {
    ([(header::CONTENT_TYPE, wire::CONTENT_TYPE)], bytes).into_response()
}

fn jnum(x: f64) -> Value {
    if x.is_finite() { json!(x) } else { Value::Null }
}

/// JSON item -> [`Item`]: a number, `{"histogram":{"edges","counts"}}`,
/// or `{"tensor":{"shape", "data" | "b64"}}`.
pub(crate) fn parse_item(v: &Value) -> Result<Item, ApiError> {
    if let Some(x) = v.as_f64() {
        return Ok(Item::Scalar(x));
    }
    if v.is_null() {
        return Ok(Item::Scalar(f64::NAN));
    }
    if let Some(h) = v.get("histogram") {
        let edges: Vec<f64> = serde_json::from_value(h["edges"].clone()).map_err(|e| ApiError::Bad(format!("histogram.edges: {e}")))?;
        let counts: Vec<f32> = serde_json::from_value(h["counts"].clone()).map_err(|e| ApiError::Bad(format!("histogram.counts: {e}")))?;
        return Ok(Item::Histogram(Histogram { edges, counts }));
    }
    if let Some(t) = v.get("tensor") {
        let shape: Vec<usize> = serde_json::from_value(t["shape"].clone()).map_err(|e| ApiError::Bad(format!("tensor.shape: {e}")))?;
        return Ok(Item::Tensor(Tensor { shape, data: parse_data(t)? }));
    }
    Err(ApiError::Bad(format!("unsupported item value: {v}")))
}

fn parse_data(v: &Value) -> Result<Vec<f32>, ApiError> {
    if let Some(b) = v.get("b64").and_then(|b| b.as_str()) {
        let bytes = base64::engine::general_purpose::STANDARD.decode(b).map_err(|e| ApiError::Bad(format!("b64: {e}")))?;
        if bytes.len() % 4 != 0 {
            return Err(ApiError::Bad("b64 length not a multiple of 4".into()));
        }
        return Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect());
    }
    serde_json::from_value(v["data"].clone()).map_err(|e| ApiError::Bad(format!("data: {e}")))
}

fn opt_t(v: &Value) -> Option<f64> {
    v.get("t").and_then(|t| t.as_f64())
}

fn step_of(v: &Value) -> Result<u64, ApiError> {
    v["step"].as_u64().ok_or_else(|| ApiError::Bad("step (non-negative integer) required".into()))
}

// ------------------------------------------------------------ ingest routes

async fn create_run(AxState(st): AxState<AppState>, axum::Json(b): axum::Json<Value>) -> ApiResult {
    let name = b["name"].as_str().ok_or_else(|| ApiError::Bad("name required".into()))?.to_string();
    let project = b["project"].as_str().map(String::from);
    match ingest(&st, IngestRequest::CreateRun { name, project, config: b.get("config").cloned().unwrap_or(Value::Null), meta: b.get("meta").cloned() }).await? {
        IngestResponse::Created { run } => Ok((StatusCode::CREATED, axum::Json(json!({ "run": run }))).into_response()),
        other => ok_json(json!({ "response": format!("{other:?}") })),
    }
}

fn log_request(run: &str, b: &Value) -> Result<IngestRequest, ApiError> {
    let items = b["items"].as_object().ok_or_else(|| ApiError::Bad("items object required".into()))?;
    let items = items.iter().map(|(k, v)| Ok((k.clone(), parse_item(v)?))).collect::<Result<Vec<_>, ApiError>>()?;
    Ok(IngestRequest::Log { run: run.to_string(), step: step_of(b)?, t: opt_t(b), items })
}

/// Body: one `{step, t?, items}` or an array of them. An array is sent as
/// concurrent requests so the batch layer can group them into one append.
async fn log(AxState(st): AxState<AppState>, Path(run): Path<String>, axum::Json(b): axum::Json<Value>) -> ApiResult {
    let bodies: Vec<Value> = match b {
        Value::Array(a) => a,
        one => vec![one],
    };
    let reqs = bodies.iter().map(|b| log_request(&run, b)).collect::<Result<Vec<_>, _>>()?;
    let n = reqs.len();
    let mut set = tokio::task::JoinSet::new();
    for (i, r) in reqs.into_iter().enumerate() {
        let st = st.clone();
        set.spawn(async move { (i, ingest(&st, r).await) });
    }
    while let Some(res) = set.join_next().await {
        let (_, r) = res.map_err(|e| ApiError::Bad(e.to_string()))?;
        r?;
    }
    ok_json(json!({ "accepted": n }))
}

async fn log_array(AxState(st): AxState<AppState>, Path(run): Path<String>, axum::Json(b): axum::Json<Value>) -> ApiResult {
    let shape: Vec<usize> = serde_json::from_value(b["shape"].clone()).map_err(|e| ApiError::Bad(format!("shape: {e}")))?;
    let labels = match b.get("labels") {
        Some(l) if !l.is_null() => Some(parse_data(l)?),
        _ => None,
    };
    let req = IngestRequest::LogArray {
        run,
        step: step_of(&b)?,
        t: opt_t(&b),
        key: b["key"].as_str().ok_or_else(|| ApiError::Bad("key required".into()))?.to_string(),
        kind: b["kind"].as_str().unwrap_or("tensor").to_string(),
        shape,
        data: parse_data(&b)?,
        labels,
    };
    ingest(&st, req).await?;
    ok_json(json!({ "accepted": 1 }))
}

async fn event(AxState(st): AxState<AppState>, Path(run): Path<String>, axum::Json(b): axum::Json<Value>) -> ApiResult {
    let text = b["text"].as_str().ok_or_else(|| ApiError::Bad("text required".into()))?.to_string();
    ingest(&st, IngestRequest::Event { run, step: step_of(&b)?, t: opt_t(&b), text }).await?;
    ok_json(json!({ "accepted": 1 }))
}

async fn finish(AxState(st): AxState<AppState>, Path(run): Path<String>, body: Option<axum::Json<Value>>) -> ApiResult {
    let summary = body.map(|b| b.0.get("summary").cloned().unwrap_or(Value::Null)).unwrap_or(Value::Null);
    ingest(&st, IngestRequest::Finish { run, summary }).await?;
    ok_json(json!({ "finished": true }))
}

// ------------------------------------------------------------ read routes

pub(crate) async fn index(AxState(st): AxState<AppState>, Query(q): Query<HashMap<String, String>>, h: HeaderMap) -> ApiResult {
    // anonymous projects are listed only when asked for by name (their board
    // URL) or by a project token, which scopes the list to its project
    let project = q.get("project").cloned().or_else(|| match st.auth.cred(crate::v1::bearer(&h).as_deref()) {
        Ok(crate::auth::Cred::Project(p)) => Some(p),
        _ => None,
    });
    let keep = |e: &Value| {
        let id = e["id"].as_str().unwrap_or("");
        match &project {
            Some(p) => id.strip_prefix(p.as_str()).is_some_and(|r| r.starts_with('/')),
            None => !id.starts_with(crate::auth::ANON_PREFIX),
        }
    };
    let filter = q.get("filter").cloned().or_else(|| project.as_ref().map(|p| format!("{p}/")));
    match read(&st, ReadRequest::ListRuns { filter }).await? {
        ReadResponse::Runs(runs) => {
            let runs: Vec<Value> = runs.into_iter().filter(keep).collect(); ok_json(json!({ "generated_unix": modgrad_obs::now_unix(), "runs": runs, "errors": [], "server": true, "project": project, "dev": st.auth.dev() }))
        }
        _ => Err(ApiError::NotFound),
    }
}

pub(crate) async fn config(AxState(st): AxState<AppState>, Path(run): Path<String>) -> ApiResult {
    match read(&st, ReadRequest::Config(run)).await? {
        ReadResponse::Config(v) => ok_json(v),
        _ => Err(ApiError::NotFound),
    }
}

pub(crate) async fn summary(AxState(st): AxState<AppState>, Path(run): Path<String>) -> ApiResult {
    match read(&st, ReadRequest::Summary(run)).await? {
        ReadResponse::Summary(v) => ok_json(v.unwrap_or(Value::Null)),
        _ => Err(ApiError::NotFound),
    }
}

pub(crate) async fn events(AxState(st): AxState<AppState>, Path(run): Path<String>) -> ApiResult {
    match read(&st, ReadRequest::Events(run)).await? {
        ReadResponse::Events(v) => ok_json(json!(v)),
        _ => Err(ApiError::NotFound),
    }
}

fn range_of(q: &HashMap<String, String>) -> std::ops::Range<u64> {
    let p = |k: &str, d: u64| q.get(k).and_then(|v| v.parse().ok()).unwrap_or(d);
    p("from", 0)..p("to", u64::MAX)
}

/// `?keys=a,b&from=&to=&stride=`; JSON `{steps,t,columns:{k:[…]}}` or frames.
pub(crate) async fn history(AxState(st): AxState<AppState>, Path(run): Path<String>, Query(q): Query<HashMap<String, String>>, h: HeaderMap) -> ApiResult {
    let keys: Vec<String> = q.get("keys").map(|k| k.split(',').filter(|s| !s.is_empty()).map(String::from).collect()).unwrap_or_default();
    let stride = q.get("stride").and_then(|s| s.parse().ok()).unwrap_or(1);
    let ReadResponse::History(b) = read(&st, ReadRequest::History { run: run.clone(), keys, range: range_of(&q), stride }).await? else { return Err(ApiError::NotFound) };
    if wants_frames(&h) {
        let n = b.steps.len();
        let mut frames = vec![
            Frame { name: "step".into(), shape: vec![n], data: FrameData::F64(b.steps.iter().map(|&s| s as f64).collect()) },
            Frame { name: "t".into(), shape: vec![n], data: FrameData::F64(b.t.clone()) },
        ];
        for (k, col) in &b.columns {
            frames.push(Frame { name: k.clone(), shape: vec![n], data: FrameData::F32(col.iter().map(|&v| v as f32).collect()) });
        }
        return Ok(frames_response(wire::encode(&frames, json!({ "run": run }))));
    }
    let cols: serde_json::Map<String, Value> = b.columns.iter().map(|(k, c)| (k.clone(), json!(c.iter().map(|&v| jnum(v)).collect::<Vec<_>>()))).collect();
    ok_json(json!({ "run": run, "steps": b.steps, "t": b.t.iter().map(|&v| jnum(v)).collect::<Vec<_>>(), "columns": cols }))
}

/// Same schema as the static `metrics/<key>.json`. `?max_points=N` picks
/// every k-th row (selection, not averaging) so at most ~N come back.
pub(crate) async fn metric(AxState(st): AxState<AppState>, Path((run, key)): Path<(String, String)>, Query(q): Query<HashMap<String, String>>) -> ApiResult {
    let ReadResponse::History(full) = read(&st, ReadRequest::History { run: run.clone(), keys: vec![key.clone()], range: range_of(&q), stride: 1 }).await? else { return Err(ApiError::NotFound) };
    let n = full.steps.len();
    let stride = q.get("max_points").and_then(|m| m.parse::<usize>().ok()).filter(|&m| m > 0).map(|m| n.div_ceil(m)).unwrap_or(1).max(1);
    let col = &full.columns[0].1;
    let pick: Vec<usize> = (0..n).filter(|i| i % stride == 0 || i + 1 == n).collect();
    ok_json(json!({
        "key": key,
        "step": pick.iter().map(|&i| full.steps[i]).collect::<Vec<_>>(),
        "value": pick.iter().map(|&i| jnum(col[i])).collect::<Vec<_>>(),
        "stride": stride,
    }))
}

/// Same schema as the static `arrays/<key>.json` (plus `offset`).
/// `?steps=1,2&lo=&hi=&max_steps=`; frames when asked.
pub(crate) async fn arrays(AxState(st): AxState<AppState>, Path((run, key)): Path<(String, String)>, Query(q): Query<HashMap<String, String>>, h: HeaderMap) -> ApiResult {
    let steps: Option<Vec<u64>> = q.get("steps").map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect());
    let index_range = match (q.get("lo").and_then(|v| v.parse().ok()), q.get("hi").and_then(|v| v.parse().ok())) {
        (None, None) => None,
        (lo, hi) => Some(lo.unwrap_or(0)..hi.unwrap_or(usize::MAX)),
    };
    let ReadResponse::Arrays { key, mut blocks } = read(&st, ReadRequest::Arrays { run: run.clone(), key, steps, index_range }).await? else { return Err(ApiError::NotFound) };
    if let Some(m) = q.get("max_steps").and_then(|m| m.parse::<usize>().ok()).filter(|&m| m >= 2) {
        let n = blocks.len();
        if n > m {
            let mut keep: Vec<usize> = (0..m - 1).map(|i| i * (n - 1) / (m - 1)).collect();
            keep.push(n - 1);
            keep.dedup();
            blocks = keep.into_iter().map(|i| blocks[i].clone()).collect();
        }
    }
    if wants_frames(&h) {
        let frames: Vec<Frame> = blocks.iter().map(|b| Frame { name: format!("{key}@{}", b.step), shape: if b.offset == 0 && b.data.len() == b.shape.iter().product::<usize>() { b.shape.clone() } else { vec![b.data.len()] }, data: FrameData::F32(b.data.clone()) }).collect();
        let meta = json!({ "run": run, "key": key, "kind": blocks.first().map(|b| b.kind.clone()), "steps": blocks.iter().map(|b| b.step).collect::<Vec<_>>(), "offsets": blocks.iter().map(|b| b.offset).collect::<Vec<_>>(), "shapes": blocks.iter().map(|b| b.shape.clone()).collect::<Vec<_>>() });
        return Ok(frames_response(wire::encode(&frames, meta)));
    }
    let kind = blocks.first().map(|b| b.kind.clone()).unwrap_or_default();
    let mut v = json!({
        "key": key, "kind": kind,
        "step": blocks.iter().map(|b| b.step).collect::<Vec<_>>(),
        "shape": blocks.iter().map(|b| b.shape.clone()).collect::<Vec<_>>(),
        "offset": blocks.iter().map(|b| b.offset).collect::<Vec<_>>(),
        "b64": blocks.iter().map(|b| modgrad_obs::encode_f32(&b.data)).collect::<Vec<_>>(),
    });
    if kind == "histogram" {
        v["edges"] = json!(blocks.iter().map(|b| b.edges.clone()).collect::<Vec<_>>());
    }
    if blocks.iter().any(|b| b.labels.is_some()) {
        v["labels_b64"] = json!(blocks.iter().map(|b| b.labels.as_ref().map(|l| modgrad_obs::encode_f32(l))).collect::<Vec<_>>());
    }
    ok_json(v)
}

// ------------------------------------------------------------ static files

fn content_type(p: &std::path::Path) -> &'static str {
    match p.extension().and_then(|e| e.to_str()).unwrap_or("") {
        "html" => "text/html; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "wasm" => "application/wasm",
        "json" => "application/json",
        "css" => "text/css; charset=utf-8",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        _ => "application/octet-stream",
    }
}

async fn static_file(AxState(st): AxState<AppState>, uri: Uri) -> ApiResult {
    let Some(root) = &st.static_dir else { return Err(ApiError::NotFound) };
    let rel = uri.path().trim_start_matches('/');
    let rel = if rel.is_empty() { "dashboard.html" } else { rel };
    // no traversal: every component must be a plain name
    let path = std::path::Path::new(rel);
    // and no dotfiles (the token store lives at <root>/.obs-auth.json)
    if path.components().any(|c| !matches!(c, std::path::Component::Normal(n) if !n.to_string_lossy().starts_with('.'))) {
        return Err(ApiError::NotFound);
    }
    let full = root.join(path);
    let bytes = tokio::task::spawn_blocking(move || std::fs::read(&full).map(|b| (b, full))).await.map_err(|e| ApiError::Bad(e.to_string()))?;
    match bytes {
        Ok((b, full)) => Ok(([(header::CONTENT_TYPE, content_type(&full))], Body::from(b)).into_response()),
        Err(_) => Err(ApiError::NotFound),
    }
}

/// The legacy unversioned write routes carry no credentials, so they
/// exist only in dev mode; the v1 routes (`crate::v1`) are authenticated.
pub fn router(st: AppState) -> Router {
    let mut r = Router::new().route("/api/index", get(index));
    if st.mcp_self.is_some() {
        r = r.route("/mcp", post(crate::mcp_http::post_mcp).get(crate::mcp_http::get_mcp));
    }
    if st.auth.dev() {
        r = r
            .route("/api/runs", post(create_run))
            .route("/api/runs/:run/log", post(log))
            .route("/api/runs/:run/array", post(log_array))
            .route("/api/runs/:run/event", post(event))
            .route("/api/runs/:run/finish", post(finish));
    }
    r.merge(crate::v1::routes())
        .route("/api/runs/:run/config", get(config))
        .route("/api/runs/:run/summary", get(summary))
        .route("/api/runs/:run/events", get(events))
        .route("/api/runs/:run/history", get(history))
        .route("/api/runs/:run/metrics/:key", get(metric))
        .route("/api/runs/:run/arrays/:key", get(arrays))
        .fallback(static_file)
        .with_state(st)
}
