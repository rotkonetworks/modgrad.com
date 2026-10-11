//! The v1 HTTP ingest API: plain JSON, curl-friendly, forgiving,
//! authenticated. `GET /api/v1` serves [`API_TEXT`], the reference that
//! humans and agents read.
//!
//! Translation from request bodies to [`IngestRequest`]s is pure
//! ([`parse_log`], [`parse_text`], [`row_record`]); handlers only call it
//! and the services, so another transport (HTTP/3, MCP, SSH) reuses it.

use crate::auth::{ANON_PREFIX, AuthedRequest, AuthedResponse};
use crate::http::{ApiError, AppState, parse_item};
use crate::ingest::IngestRequest;
use crate::state::{AUTO_STEP, Record};
use axum::Router;
use axum::body::Bytes;
use axum::extract::{Path, Query, State as AxState};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, put};
use modgrad_obs::Item;
use serde_json::{Map, Value, json};
use std::collections::HashMap;
use tower::ServiceExt;

type ApiResult = Result<Response, ApiError>;

/// The API reference, served as text/plain at `GET /api/v1`.
pub const API_TEXT: &str = include_str!("v1_api.txt");

/// A corrected example for log-shaped errors.
pub const EX_LOG: &str = r#"curl -X POST "$OBS/api/v1/runs/my-run/log" -H "Authorization: Bearer $OBS_TOKEN" -d '{"step": 1, "loss": 0.52, "acc": "71.5%"}'"#;
pub const EX_TOKEN: &str = r#"curl -X POST "$OBS/api/v1/projects"   # -> {"token": "obsp_..."}; then send -H "Authorization: Bearer obsp_...""#;
pub const EX_TEXT: &str = r#"python train.py 2>&1 | curl -X POST "$OBS/api/v1/runs/my-run/text" -H "Authorization: Bearer $OBS_TOKEN" --data-binary @-"#;
pub const EX_BOARD: &str = r#"curl -X PUT "$OBS/api/v1/boards/my-board" -H "Authorization: Bearer $OBS_TOKEN" -d '{"title": "loss", "panels": [{"query": "loss{run=~\"my-run\"}", "metric": "loss", "runs": ["my-run"]}]}'"#;
pub const EX_CREATE: &str = r#"curl -X POST "$OBS/api/v1/runs" -H "Authorization: Bearer $OBS_TOKEN" -d '{"name": "my-run", "config": {"lr": 0.0003}}'"#;

fn bad(m: impl Into<String>) -> ApiError {
    ApiError::Teach { status: StatusCode::BAD_REQUEST, error: m.into(), example: EX_LOG }
}

/// Everything one `log` body asks for, applied in this order: config,
/// records (rows then events), summary, finish.
#[derive(Debug, Clone, PartialEq)]
pub struct LogBody {
    pub config: Option<Value>,
    pub records: Vec<Record>,
    pub rows: usize,
    pub events: usize,
    pub summary: Option<Value>,
    /// `Some(summary)` to finish (summary may be `Null`).
    pub finish: Option<Value>,
    /// Keys whose values could not be read as numbers (dropped, reported).
    pub ignored: Vec<String>,
}

const ENVELOPE: [&str; 5] = ["rows", "events", "config", "summary", "finish"];

/// Parse a body: one JSON value, else NDJSON (one value per line).
pub fn parse_body(b: &[u8]) -> Result<Value, ApiError> {
    if b.iter().all(u8::is_ascii_whitespace) {
        return Ok(Value::Null);
    }
    if let Ok(v) = serde_json::from_slice::<Value>(b) {
        return Ok(v);
    }
    let text = std::str::from_utf8(b).map_err(|_| bad("the body is not UTF-8 text"))?;
    let mut rows = Vec::new();
    for (i, l) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
        rows.push(serde_json::from_str::<Value>(l).map_err(|e| bad(format!("the body is neither one JSON value nor NDJSON (line {}: {e}); for raw log text use POST /api/v1/runs/<id>/text", i + 1)))?);
    }
    Ok(Value::Array(rows))
}

/// A forgiving number: JSON numbers, bools (0/1), and strings like
/// `"3.4"`, `"7.5%"`, `"1e-3"`; `null`, `""`, `"nan"`, `"inf"` are gaps
/// (stored as null). `None` = not a number.
pub fn loose_number(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::Null => Some(f64::NAN),
        Value::String(s) => {
            let t = s.trim().trim_end_matches('%').trim();
            match t.to_ascii_lowercase().as_str() {
                "" | "nan" | "inf" | "+inf" | "-inf" | "infinity" | "-infinity" | "none" | "null" => Some(f64::NAN),
                _ => t.parse::<f64>().ok().filter(|x| !x.is_nan() || t.eq_ignore_ascii_case("nan")),
            }
        }
        _ => None,
    }
}

fn loose_step(v: &Value) -> Result<u64, ApiError> {
    match loose_number(v) {
        Some(x) if x.is_finite() && x >= 0.0 && x.fract() == 0.0 && x < 9e15 => Ok(x as u64),
        _ => Err(bad(format!("step must be a non-negative integer (or left out to mean \"next\"), got {v}"))),
    }
}

/// Flatten `{"train": {"loss": 1}}` into `train/loss`; `{"histogram"}` and
/// `{"tensor"}` objects are items, not namespaces.
fn flatten_into(prefix: &str, v: &Value, depth: usize, items: &mut Vec<(String, Item)>, ignored: &mut Vec<String>) -> Result<(), ApiError> {
    match v {
        Value::Object(o) if o.contains_key("histogram") || o.contains_key("tensor") => items.push((prefix.to_string(), parse_item(v)?)),
        Value::Object(o) if depth < 4 => {
            for (k, x) in o {
                flatten_into(&format!("{prefix}/{k}"), x, depth + 1, items, ignored)?;
            }
        }
        other => match loose_number(other) {
            Some(x) => items.push((prefix.to_string(), Item::Scalar(x))),
            None => ignored.push(prefix.to_string()),
        },
    }
    Ok(())
}

/// One row -> a `Log` record. `step` (or W&B's `_step`) is optional:
/// missing means "next" ([`AUTO_STEP`]). `t` is seconds since run start.
pub fn row_record(row: &Value, ignored: &mut Vec<String>) -> Result<Record, ApiError> {
    let o = row.as_object().ok_or_else(|| bad(format!("a row must be a JSON object like {{\"step\": 1, \"loss\": 0.5}}, got {row}")))?;
    let step = match o.get("step").or_else(|| o.get("_step")) {
        None | Some(Value::Null) => AUTO_STEP,
        Some(s) => loose_step(s)?,
    };
    let t = o.get("t").and_then(loose_number).filter(|x| x.is_finite());
    let mut items = Vec::new();
    let before = ignored.len();
    for (k, v) in o.iter().filter(|(k, _)| !matches!(k.as_str(), "step" | "_step" | "t" | "_timestamp" | "_runtime")) {
        flatten_into(k, v, 0, &mut items, ignored)?;
    }
    if items.is_empty() {
        let why = if ignored.len() > before { format!("none of its values are numbers ({})", ignored[before..].join(", ")) } else { "it has no metric keys".into() };
        return Err(bad(format!("row {row} has nothing to log: {why}")));
    }
    Ok(Record::Log { step, t, items })
}

fn event_record(e: &Value) -> Result<Record, ApiError> {
    let (step, text) = match e {
        Value::String(s) => (AUTO_STEP, s.clone()),
        Value::Object(o) => (
            match o.get("step") {
                None | Some(Value::Null) => AUTO_STEP,
                Some(s) => loose_step(s)?,
            },
            o.get("text").and_then(|t| t.as_str()).ok_or_else(|| bad("an event needs \"text\": {\"text\": \"checkpoint saved\", \"step\": 100}"))?.to_string(),
        ),
        other => return Err(bad(format!("an event is a string or {{\"step\"?, \"text\"}}, got {other}"))),
    };
    Ok(Record::Event { step, t: e.get("t").and_then(|t| t.as_f64()), text })
}

/// A `log` body -> what to do. Shapes: one row object, an array of rows
/// (or NDJSON), or an envelope with any of `rows`, `events`, `config`,
/// `summary`, `finish` (`true` or a summary object).
pub fn parse_log(body: &Value) -> Result<LogBody, ApiError> {
    let mut out = LogBody { config: None, records: Vec::new(), rows: 0, events: 0, summary: None, finish: None, ignored: Vec::new() };
    let is_env = |o: &Map<String, Value>| o.keys().any(|k| ENVELOPE.contains(&k.as_str())) && !o.contains_key("step");
    let (rows, env): (Vec<&Value>, Option<&Map<String, Value>>) = match body {
        Value::Array(a) => (a.iter().collect(), None),
        Value::Object(o) if is_env(o) => {
            if let Some(k) = o.keys().find(|k| !ENVELOPE.contains(&k.as_str())) {
                return Err(bad(format!("{k:?} is not an envelope key; an envelope has only rows, events, config, summary, finish. To log {k:?} as a metric put it in a row: {{\"rows\": [{{\"{k}\": 1.0}}]}}")));
            }
            let rows = match o.get("rows") {
                None | Some(Value::Null) => vec![],
                Some(Value::Array(a)) => a.iter().collect(),
                Some(r @ Value::Object(_)) => vec![r],
                Some(other) => return Err(bad(format!("rows must be an array of objects, got {other}"))),
            };
            (rows, Some(o))
        }
        Value::Object(_) => (vec![body], None),
        Value::Null => return Err(bad("the body is empty; send a row like {\"step\": 1, \"loss\": 0.5}")),
        other => return Err(bad(format!("expected a row like {{\"step\": 1, \"loss\": 0.5}}, an array of rows, or an envelope {{\"rows\": [...], \"finish\": true}}; got {other}"))),
    };
    for r in rows {
        out.records.push(row_record(r, &mut out.ignored)?);
        out.rows += 1;
    }
    let Some(env) = env else { return Ok(out) };
    match env.get("events") {
        None | Some(Value::Null) => {}
        Some(Value::Array(a)) => {
            for e in a {
                out.records.push(event_record(e)?);
                out.events += 1;
            }
        }
        Some(e) => {
            out.records.push(event_record(e)?);
            out.events += 1;
        }
    }
    for (k, slot) in [("config", &mut out.config), ("summary", &mut out.summary)] {
        match env.get(k) {
            None | Some(Value::Null) => {}
            Some(v @ Value::Object(_)) => *slot = Some(v.clone()),
            Some(other) => return Err(bad(format!("{k} must be an object like {{\"lr\": 0.0003}}, got {other}"))),
        }
    }
    out.finish = match env.get("finish") {
        None | Some(Value::Null) | Some(Value::Bool(false)) => None,
        Some(Value::Bool(true)) => Some(Value::Null),
        Some(v @ Value::Object(_)) => Some(v.clone()),
        Some(other) => return Err(bad(format!("finish must be true or a summary object like {{\"exit_code\": 0}}, got {other}"))),
    };
    Ok(out)
}

/// Raw log text -> rows via [`modgrad_obs::textlog::parse_line`]. A line
/// without a step takes the step of the last line in this body that had
/// one, else "next". Returns (records, lines, unparsed non-empty lines).
pub fn parse_text(text: &str) -> (Vec<Record>, usize, Vec<String>) {
    let mut recs = Vec::new();
    let mut unparsed = Vec::new();
    let mut carry: Option<u64> = None;
    let mut lines = 0;
    for l in text.lines() {
        lines += 1;
        match modgrad_obs::textlog::parse_line(l) {
            Some(r) => {
                if r.step.is_some() {
                    carry = r.step;
                }
                if r.values.is_empty() {
                    continue;
                }
                recs.push(Record::Log { step: r.step.or(carry).unwrap_or(AUTO_STEP), t: None, items: r.values.into_iter().map(|(k, v)| (k, Item::Scalar(v))).collect() });
            }
            None if !l.trim().is_empty() => unparsed.push(l.to_string()),
            None => {}
        }
    }
    (recs, lines, unparsed)
}

pub(crate) fn bearer(h: &HeaderMap) -> Option<String> {
    let v = h.get(header::AUTHORIZATION)?.to_str().ok()?;
    let t = v.strip_prefix("Bearer ").or_else(|| v.strip_prefix("bearer ")).unwrap_or(v).trim();
    (!t.is_empty()).then(|| t.to_string())
}

/// `scheme://host` the client used (honours X-Forwarded-*).
pub fn base_url(h: &HeaderMap) -> String {
    let get = |k: &str| h.get(k).and_then(|v| v.to_str().ok()).map(|s| s.split(',').next().unwrap_or("").trim().to_string());
    let host = get("x-forwarded-host").or_else(|| get("host")).unwrap_or_else(|| "localhost".into());
    let proto = get("x-forwarded-proto").unwrap_or_else(|| "http".into());
    format!("{proto}://{host}")
}

pub fn run_url(base: &str, run: &str) -> String {
    format!("{base}/#run={}", run.replace('/', "%2F"))
}

async fn call(st: &AppState, token: &Option<String>, req: IngestRequest) -> Result<AuthedResponse, ApiError> {
    st.authed().oneshot(AuthedRequest { token: token.clone(), req }).await.map_err(ApiError::from_box)
}

/// The id a create gets when no name is given: `run-<UTC stamp>-<4 hex>`.
pub fn auto_name(unix: f64, suffix: &str) -> String {
    let s = unix.max(0.0) as i64;
    let (days, rem) = (s.div_euclid(86400), s.rem_euclid(86400));
    // civil_from_days (H. Hinnant)
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
    format!("run-{y:04}{m:02}{d:02}-{:02}{:02}{:02}-{suffix}", rem / 3600, rem / 60 % 60, rem % 60)
}

fn suffix() -> String {
    let n = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.subsec_nanos()).unwrap_or(0);
    format!("{:04x}", (n ^ (n >> 16)) & 0xffff)
}

fn str_field(o: &Map<String, Value>, k: &str) -> Result<Option<String>, ApiError> {
    match o.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.clone())),
        Some(v) => Err(ApiError::Teach { status: StatusCode::BAD_REQUEST, error: format!("{k} must be a string, got {v}"), example: EX_CREATE }),
    }
}

fn written(base: &str, a: &AuthedResponse, extra: Value) -> Value {
    let mut v = json!({ "run": a.run, "created": a.created, "dashboard": run_url(base, &a.run) });
    if let Some(t) = &a.token {
        v["token"] = json!(t);
        v["token_note"] = json!("run token for this run only; shown once");
    }
    if let Value::Object(m) = extra {
        for (k, x) in m {
            v[k] = x;
        }
    }
    v
}

async fn create(AxState(st): AxState<AppState>, h: HeaderMap, body: Bytes) -> ApiResult {
    let b = parse_body(&body)?;
    let b = if b.is_null() { json!({}) } else { b };
    let o = b.as_object().ok_or_else(|| ApiError::Teach { status: StatusCode::BAD_REQUEST, error: format!("the create body must be an object {{name?, project?, config?, meta?}}, got {b}"), example: EX_CREATE })?;
    let (name, project) = (str_field(o, "name")?, str_field(o, "project")?);
    let name = name.unwrap_or_else(|| auto_name(modgrad_obs::now_unix(), &suffix()));
    let name = match (&project, name.contains('/')) {
        (Some(p), false) => format!("{p}/{name}"),
        _ => name,
    };
    let req = IngestRequest::CreateRun { name, project, config: o.get("config").cloned().unwrap_or(json!({})), meta: o.get("meta").cloned() };
    let a = call(&st, &bearer(&h), req).await?;
    let base = base_url(&h);
    let v = written(&base, &a, json!({ "log": format!("{base}/api/v1/runs/{}/log", a.run), "text": format!("{base}/api/v1/runs/{}/text", a.run) }));
    Ok((if a.created { StatusCode::CREATED } else { StatusCode::OK }, axum::Json(v)).into_response())
}

async fn new_project(AxState(st): AxState<AppState>, h: HeaderMap) -> ApiResult {
    let p = st.auth.new_anon_project().map_err(ApiError::Auth)?;
    let base = base_url(&h);
    let v = json!({
        "project": p.project,
        "token": p.token,
        "board": format!("{base}/#project={}", p.project),
        "claim": format!("{base}/api/v1/claim/{}", p.claim),
        "expires_unix": p.expires_unix,
        "next": format!("curl -X POST \"{base}/api/v1/runs/my-run/log\" -H \"Authorization: Bearer {}\" -d '{{\"loss\": 0.5}}'", p.token),
        "note": "keep the token secret: it writes to this project. The board URL is unguessable but not private; anyone with it can read.",
    });
    Ok((StatusCode::CREATED, axum::Json(v)).into_response())
}

async fn claim(AxState(st): AxState<AppState>, Path(c): Path<String>) -> ApiResult {
    let project = st.auth.claim_project(&c).ok_or(ApiError::NotFound)?;
    Ok(axum::Json(json!({ "project": project, "claimed": false, "how": "attaching an SSH key or email to a project is not implemented yet; keep this URL to claim it later" })).into_response())
}

/// Split `<run>/<action>` (run ids contain `/`).
fn split_action<'a>(path: &'a str, actions: &[&str]) -> Option<(&'a str, &'a str)> {
    let (run, action) = path.trim_matches('/').rsplit_once('/')?;
    (actions.contains(&action) && !run.is_empty()).then_some((run, action))
}

async fn post_dispatch(AxState(st): AxState<AppState>, Path(path): Path<String>, Query(q): Query<HashMap<String, String>>, h: HeaderMap, body: Bytes) -> ApiResult {
    let Some((run, action)) = split_action(&path, &["log", "text", "finish"]) else {
        return Err(ApiError::Teach { status: StatusCode::NOT_FOUND, error: format!("no POST endpoint /api/v1/runs/{path}; the run actions are .../log, .../text and .../finish"), example: EX_LOG });
    };
    let token = bearer(&h);
    let base = base_url(&h);
    let mut req_run = run.to_string();
    let mut last: Option<AuthedResponse> = None;
    let mut created = false;
    let mut minted = None;
    // every call goes through Authed; the first one auto-creates the run
    let mut send = async |req: IngestRequest, req_run: &mut String| -> Result<(), ApiError> {
        let a = call(&st, &token, req).await?;
        created |= a.created;
        if a.token.is_some() {
            minted = a.token.clone();
        }
        *req_run = a.run.clone();
        last = Some(a);
        Ok(())
    };
    let report = match action {
        "text" => {
            let text = std::str::from_utf8(&body).map_err(|_| ApiError::Teach { status: StatusCode::BAD_REQUEST, error: "the body is not UTF-8 text".into(), example: EX_TEXT })?;
            let (mut records, lines, unparsed) = parse_text(text);
            let rows = records.len();
            let keep_unparsed = q.get("events").is_some_and(|v| v == "unparsed");
            if keep_unparsed {
                records.extend(unparsed.iter().map(|l| Record::Event { step: AUTO_STEP, t: None, text: l.chars().take(2000).collect() }));
            }
            // always sent, so even a body with no metric lines creates the run
            send(IngestRequest::Rows { run: req_run.clone(), records }, &mut req_run).await?;
            json!({ "lines": lines, "rows": rows, "unparsed": unparsed.len(), "unparsed_sample": unparsed.iter().take(3).collect::<Vec<_>>(), "unparsed_stored_as_events": keep_unparsed })
        }
        "finish" => {
            let summary = match parse_body(&body)? {
                Value::Object(mut o) if o.len() == 1 && o.contains_key("summary") => o.remove("summary").unwrap(),
                v @ (Value::Object(_) | Value::Null) => v,
                other => return Err(bad(format!("a finish body is an optional object merged into the summary, like {{\"exit_code\": 0}}; got {other}"))),
            };
            send(IngestRequest::Finish { run: req_run.clone(), summary }, &mut req_run).await?;
            json!({ "finished": true })
        }
        _ => {
            let lb = parse_log(&parse_body(&body)?)?;
            if let Some(config) = lb.config {
                send(IngestRequest::Config { run: req_run.clone(), config }, &mut req_run).await?;
            }
            if !lb.records.is_empty() {
                send(IngestRequest::Rows { run: req_run.clone(), records: lb.records }, &mut req_run).await?;
            }
            if let Some(summary) = lb.summary {
                send(IngestRequest::Summary { run: req_run.clone(), summary }, &mut req_run).await?;
            }
            let finished = lb.finish.is_some();
            if let Some(summary) = lb.finish {
                send(IngestRequest::Finish { run: req_run.clone(), summary }, &mut req_run).await?;
            }
            let mut v = json!({ "rows": lb.rows, "events": lb.events, "finished": finished });
            if !lb.ignored.is_empty() {
                v["ignored"] = json!(lb.ignored);
                v["ignored_note"] = json!("these keys had values that are not numbers and were not stored");
            }
            v
        }
    };
    let Some(mut a) = last else { return Err(bad("nothing to do: the body held no rows, events, config, summary or finish")) };
    a.created = created;
    a.token = minted;
    Ok(axum::Json(written(&base, &a, report)).into_response())
}

/// Which messages a stream subscriber sees: explicit runs, one project,
/// or everything except anonymous projects.
#[derive(Clone)]
enum Filter {
    Runs(Vec<String>),
    Project(String),
    Public,
}

impl Filter {
    fn keeps(&self, run: &str) -> bool {
        match self {
            Filter::Runs(r) => r.iter().any(|x| x == run),
            Filter::Project(p) => run.strip_prefix(p.as_str()).is_some_and(|r| r.starts_with('/')),
            Filter::Public => !run.starts_with(ANON_PREFIX),
        }
    }
}

fn sse(st: &AppState, filter: Filter) -> Response {
    use tokio::sync::broadcast::error::RecvError;
    let mut rx = st.live.subscribe(); // before hello, so nothing falls in between
    let f2 = filter.clone();
    let hello = st.live.hello(&move |r: &str| f2.keeps(r));
    let stream = async_stream::stream! {
        yield Ok::<Event, std::convert::Infallible>(Event::default().event("hello").data(hello.to_string()));
        loop {
            match rx.recv().await {
                Ok(m) => {
                    if filter.keeps(&m.run) {
                        yield Ok(Event::default().event(m.kind.name()).data(m.data.to_string()));
                    }
                }
                Err(RecvError::Lagged(_)) => yield Ok(Event::default().event("resync").data(json!({ "unix": modgrad_obs::now_unix() }).to_string())),
                Err(RecvError::Closed) => break,
            }
        }
    };
    let mut r = Sse::new(stream).keep_alive(KeepAlive::default()).into_response();
    // tell buffering reverse proxies (nginx) to pass events through
    r.headers_mut().insert("x-accel-buffering", header::HeaderValue::from_static("no"));
    r
}

async fn stream_all(AxState(st): AxState<AppState>, Query(q): Query<HashMap<String, String>>) -> Response {
    let f = if let Some(r) = q.get("runs") {
        Filter::Runs(r.split(',').filter(|s| !s.is_empty()).map(String::from).collect())
    } else if let Some(p) = q.get("project") {
        Filter::Project(p.clone())
    } else {
        Filter::Public
    };
    sse(&st, f)
}

async fn get_dispatch(st: AxState<AppState>, Path(path): Path<String>, q: Query<HashMap<String, String>>, h: HeaderMap) -> ApiResult {
    use crate::http as hh;
    let Some((run, action)) = split_action(&path, &["stream", "history", "config", "summary", "events"]) else {
        return Err(ApiError::Teach { status: StatusCode::NOT_FOUND, error: format!("no GET endpoint /api/v1/runs/{path}; the run reads are .../history, .../config, .../summary, .../events and .../stream"), example: r#"curl "$OBS/api/v1/runs/my-project/my-run/history?keys=loss&from=0""# });
    };
    let run = Path(run.to_string());
    match action {
        "stream" => Ok(sse(&st.0, Filter::Runs(vec![run.0]))),
        "history" => hh::history(st, run, q, h).await,
        "config" => hh::config(st, run).await,
        "summary" => hh::summary(st, run).await,
        _ => hh::events(st, run).await,
    }
}

// ----------------------------------------------------------------- boards

/// `PUT /api/v1/boards/<name>`: store a board (any JSON object; the
/// dashboard reads `{title, panels: [{query, metric, runs}]}`). A project
/// token stores it as `<project>/<name>`; a tokenless write needs `--dev`.
/// The server only keeps the JSON; it never evaluates the queries.
async fn put_board(AxState(st): AxState<AppState>, Path(name): Path<String>, h: HeaderMap, body: Bytes) -> ApiResult {
    let teach = |status, error: String| ApiError::Teach { status, error, example: EX_BOARD };
    let Some(dir) = st.boards.clone() else { return Err(teach(StatusCode::NOT_FOUND, "this server does not store boards".into())) };
    if name.contains('/') || !crate::state::valid_run_name(&name) {
        return Err(teach(StatusCode::BAD_REQUEST, format!("board name {name:?} must be one segment of [A-Za-z0-9_.-]")));
    }
    let id = match st.auth.cred(bearer(&h).as_deref()).map_err(ApiError::Auth)? {
        crate::auth::Cred::Project(p) => format!("{p}/{name}"),
        crate::auth::Cred::None if st.auth.dev() => name.clone(),
        crate::auth::Cred::None => return Err(ApiError::Auth(crate::AuthError::Unauthorized)),
        crate::auth::Cred::Run(_) => return Err(ApiError::Auth(crate::AuthError::Forbidden("boards need a project token (obsp_…), not a run token".into()))),
    };
    let b = parse_body(&body)?;
    if !b.is_object() {
        return Err(teach(StatusCode::BAD_REQUEST, format!("a board body is a JSON object, got {}", if b.is_null() { "nothing".into() } else { b.to_string().chars().take(80).collect::<String>() })));
    }
    let path = dir.join(format!("{id}.json"));
    let write = || -> std::io::Result<()> {
        std::fs::create_dir_all(path.parent().unwrap_or(&dir))?;
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(&b).unwrap_or_default())?;
        std::fs::rename(&tmp, &path)
    };
    write().map_err(|e| ApiError::Bad(format!("cannot store board {id}: {e}")))?;
    let base = base_url(&h);
    Ok(axum::Json(json!({ "board": id, "url": format!("{base}/#board={id}"), "get": format!("{base}/api/v1/boards/{id}") })).into_response())
}

/// `GET /api/v1/boards/<name | project/name>`: the stored board JSON.
async fn get_board(AxState(st): AxState<AppState>, Path(id): Path<String>) -> ApiResult {
    let Some(dir) = st.boards.clone() else { return Err(ApiError::NotFound) };
    if id.split('/').count() > 2 || !crate::state::valid_run_name(&id) {
        return Err(ApiError::NotFound);
    }
    let b = std::fs::read(dir.join(format!("{id}.json"))).map_err(|_| ApiError::NotFound)?;
    let v: Value = serde_json::from_slice(&b).map_err(|_| ApiError::NotFound)?;
    Ok(axum::Json(v).into_response())
}

async fn api_text() -> Response {
    ([(header::CONTENT_TYPE, "text/plain; charset=utf-8")], API_TEXT).into_response()
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/api/v1", get(api_text))
        .route("/api/v1/", get(api_text))
        .route("/api/v1/projects", post(new_project))
        .route("/api/v1/claim/:claim", get(claim))
        .route("/api/v1/runs", post(create))
        .route("/api/v1/stream", get(stream_all))
        .route("/api/v1/runs/*path", post(post_dispatch).get(get_dispatch))
        .route("/api/v1/boards/*name", put(put_board).get(get_board))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn log_err(v: Value) -> String {
        match parse_log(&v) {
            Err(ApiError::Teach { error, .. }) => error,
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn names_and_split() {
        assert_eq!(auto_name(1_760_140_800.0, "beef"), "run-20251011-000000-beef");
        assert_eq!(auto_name(951_782_400.0 + 3661.0, "0"), "run-20000229-010101-0");
        assert_eq!(split_action("p/r/log", &["log"]), Some(("p/r", "log")));
        assert_eq!(split_action("log", &["log"]), None);
        assert_eq!(split_action("p/r/zap", &["log"]), None);
    }

    #[test]
    fn forgiving_rows() {
        let lb = parse_log(&json!({"loss": "3.4", "acc": "7.5%", "lr": 3e-4, "gap": "nan", "inf": "inf", "ok": true, "train": {"loss": 1, "sub": {"x": 2}}, "note": "hello"})).unwrap();
        let Record::Log { step, items, .. } = &lb.records[0] else { panic!() };
        assert_eq!(*step, AUTO_STEP, "missing step means next");
        let get = |k: &str| items.iter().find(|(x, _)| x == k).map(|(_, v)| if let Item::Scalar(s) = v { *s } else { panic!() });
        assert_eq!(get("loss"), Some(3.4));
        assert_eq!(get("acc"), Some(7.5));
        assert!(get("gap").unwrap().is_nan() && get("inf").unwrap().is_nan());
        assert_eq!(get("ok"), Some(1.0));
        assert_eq!(get("train/loss"), Some(1.0));
        assert_eq!(get("train/sub/x"), Some(2.0));
        assert_eq!(lb.ignored, vec!["note".to_string()]);
        // string steps, _step alias
        let Record::Log { step, .. } = &parse_log(&json!({"step": "12", "x": 1})).unwrap().records[0] else { panic!() };
        assert_eq!(*step, 12);
        let Record::Log { step, .. } = &parse_log(&json!({"_step": 5.0, "x": 1})).unwrap().records[0] else { panic!() };
        assert_eq!(*step, 5);
    }

    #[test]
    fn body_shapes() {
        assert_eq!(parse_log(&json!([{"step": 1, "x": 1}, {"x": 2}])).unwrap().rows, 2);
        let env = parse_log(&json!({"rows": [{"step": 7, "x": 1}], "events": ["hi", {"step": 2, "text": "ckpt"}], "config": {"lr": 1}, "summary": {"best": 1}, "finish": {"exit_code": 0}})).unwrap();
        assert_eq!((env.rows, env.events, env.records.len()), (1, 2, 3));
        assert_eq!(env.finish, Some(json!({"exit_code": 0})));
        assert_eq!(parse_log(&json!({"finish": true})).unwrap().finish, Some(Value::Null));
        // a row that happens to have a key named like an envelope key still works with a step
        assert_eq!(parse_log(&json!({"step": 1, "summary": 3})).unwrap().rows, 1);
    }

    #[test]
    fn errors_teach() {
        assert!(log_err(json!({"step": -1, "loss": 1})).contains("non-negative"));
        assert!(log_err(json!({"step": 1, "loss": "high"})).contains("loss"));
        assert!(log_err(json!({"step": 1})).contains("no metric keys"));
        assert!(log_err(json!({"rows": [], "bogus": 1})).contains("\"bogus\""));
        assert!(log_err(json!(5)).contains("expected a row"));
        assert!(log_err(Value::Null).contains("empty"));
    }

    #[test]
    fn text_bodies() {
        let (recs, lines, unparsed) = parse_text("starting\nstep 10: loss 2.5\nval_acc: 0.5\n[epoch 1/3] train_loss: 0.4\n");
        assert_eq!((lines, unparsed), (4, vec!["starting".to_string()]));
        let steps: Vec<u64> = recs.iter().map(|r| if let Record::Log { step, .. } = r { *step } else { panic!() }).collect();
        assert_eq!(steps, vec![10, 10, 10], "lines without a step carry the last step in the body");
        let (recs, ..) = parse_text("loss=1.0\n");
        assert!(matches!(&recs[0], Record::Log { step, .. } if *step == AUTO_STEP));
    }

    #[test]
    fn ndjson_and_loose_numbers() {
        assert_eq!(parse_body(b"{\"step\":1,\"x\":1}\n\n{\"step\":2,\"x\":2}\n").unwrap().as_array().unwrap().len(), 2);
        assert!(parse_body(b"{nope").is_err());
        assert_eq!(loose_number(&json!(" 1e-3 ")), Some(1e-3));
        assert_eq!(loose_number(&json!("abc")), None);
        assert_eq!(loose_number(&json!([1])), None);
    }
}
