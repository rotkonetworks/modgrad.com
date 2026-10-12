//! Hosted MCP endpoint, opt-in with `modgrad-server --mcp`: `POST /mcp`
//! speaks MCP's streamable-HTTP transport (one JSON-RPC message or batch
//! per request, a JSON response; no server-initiated stream) with the same
//! seven tools as `modgrad mcp`.
//!
//! This is for agents that cannot run a local binary. It bends the rule
//! that the server only stores and selects: the tools' analysis (trends,
//! forecasts, spike and seed-noise checks) runs here, on the server's CPU.
//! Prefer `modgrad mcp --url <server>` wherever a local process is possible.
//!
//! The tools reach the store through [`RemoteStore`] pointed at this
//! server's own loopback address with the caller's bearer token, so writes
//! go through the same authenticated ingest path as every other client
//! (the store keeps a single writer) and reads see the same snapshots.

use crate::http::AppState;
use axum::body::Bytes;
use axum::extract::State as AxState;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use modgrad_obs::mcp::Server;
use modgrad_obs::mcp::store::RemoteStore;
use serde_json::{Value, json};

pub async fn post_mcp(AxState(st): AxState<AppState>, h: HeaderMap, body: Bytes) -> Response {
    let Some(internal) = st.mcp_self.clone() else { return StatusCode::NOT_FOUND.into_response() };
    let msg: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return (StatusCode::BAD_REQUEST, axum::Json(json!({"jsonrpc": "2.0", "id": null, "error": {"code": -32700, "message": format!("parse error: {e}")}}))).into_response(),
    };
    let token = crate::v1::bearer(&h);
    let public = st.base_url(&h);
    let reply = tokio::task::spawn_blocking(move || {
        let server = Server::new(Box::new(RemoteStore::new(&internal, token)));
        // URLs in tool output (board links, the store description) name the
        // loopback address; show the caller the address it used instead
        let mut r = server.handle(&msg);
        if let Some(v) = r.as_mut() {
            fix_urls(v, &internal, &public);
        }
        r
    })
    .await;
    match reply {
        Ok(Some(v)) => ([(header::CONTENT_TYPE, "application/json")], v.to_string()).into_response(),
        Ok(None) => StatusCode::ACCEPTED.into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

/// `GET /mcp`: no server-to-client stream is offered (allowed by the spec).
pub async fn get_mcp() -> Response {
    (StatusCode::METHOD_NOT_ALLOWED, [(header::ALLOW, "POST")]).into_response()
}

fn fix_urls(v: &mut Value, from: &str, to: &str) {
    match v {
        Value::String(s) if s.contains(from) => *s = s.replace(from, to),
        Value::Array(a) => a.iter_mut().for_each(|x| fix_urls(x, from, to)),
        Value::Object(o) => o.values_mut().for_each(|x| fix_urls(x, from, to)),
        _ => {}
    }
}
