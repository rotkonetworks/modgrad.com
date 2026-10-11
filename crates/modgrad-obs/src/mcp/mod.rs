//! `obs mcp`: a Model Context Protocol server over stdio.
//!
//! Hand-rolled JSON-RPC 2.0 (one JSON message per line on stdin/stdout,
//! logs on stderr) implementing what MCP tool servers need: `initialize`,
//! `notifications/initialized`, `ping`, `tools/list`, `tools/call`. The
//! official `rmcp` SDK would add tokio and schemars to a crate that is
//! otherwise serde + base64 with no threads; the subset here is small and
//! [`Server::handle`] is a pure function, so the tests drive it directly.
//!
//! Tools are designed for weak local models: flat string/number arguments,
//! case- and alias-tolerant names, short plain-text output, and every
//! message ends with what to do next. Tool failures are returned as
//! `isError: true` results (not JSON-RPC errors) so the model sees the
//! explanation and the example call.

pub mod analysis;
pub mod store;
pub mod text;
pub mod tools;

use serde_json::{Value, json};
use std::io::{BufRead, Write};
use store::Store;
use tools::Args;

pub const SERVER_NAME: &str = "modgrad-obs";
const PROTOCOL_VERSIONS: [&str; 3] = ["2025-06-18", "2025-03-26", "2024-11-05"];

const INSTRUCTIONS: &str = "modgrad-obs: training-run metrics (loss curves, eval scores) stored locally. \
Start with list_runs() to see run names. Use summarize(run) for a plain-language status, \
read_metric(run, metric) for values, compare(runs, metric) across runs, log / ingest_log to record metrics, \
and make_board(query) for a dashboard link. Run and metric names are fuzzy-matched.";

/// Name, description, and flat JSON-schema properties of each tool.
fn tool_specs() -> Vec<Value> {
    let s = |d: &str| json!({"type": "string", "description": d});
    let n = |d: &str| json!({"type": "number", "description": d});
    let tool = |name: &str, desc: &str, props: Value, req: &[&str]| json!({"name": name, "description": desc, "inputSchema": {"type": "object", "properties": props, "required": req}});
    vec![
        tool(
            "log",
            "Record metric values for a training run at one step. Creates the run if it does not exist. \
metrics is a flat object of name -> number; strings like \"3.4\" or \"7.5%\" are accepted. \
If step is omitted, the run's last step + 1 is used.\n\
Example: log(run=\"demo1\", metrics={\"loss\": 2.31, \"acc\": \"71.2%\"}, step=100)",
            json!({
                "run": s("Run name, e.g. \"demo1\" or \"group/name\""),
                "metrics": {"type": "object", "description": "Flat object of metric name -> number, e.g. {\"loss\": 2.31, \"lr\": 0.0003}"},
                "step": n("Training step (whole number). Optional."),
            }),
            &["run", "metrics"],
        ),
        tool(
            "ingest_log",
            "Parse raw training stdout / log text and record every metric line in a run (created if needed). \
Understands lines like \"step 100 loss=2.31 lr=3e-4\", \"step 100 | loss 2.31 | acc 71.2%\", \"Epoch 3: val_loss=0.52\". \
Reports how many rows and metrics were found and shows skipped lines.\n\
Example: ingest_log(run=\"demo1\", text=\"step 100 loss=2.31\\nstep 200 loss=2.05\")",
            json!({
                "run": s("Run name to write into, e.g. \"demo1\""),
                "text": s("The raw log text, many lines in one string"),
            }),
            &["run", "text"],
        ),
        tool(
            "list_runs",
            "List training runs as a table: run name, last step, number of metrics, status (finished / running / idle) and last update. \
Optional filter: a substring (\"bkk07\"), a glob (\"bkk07/*\"), or a status (\"running\").\n\
Example: list_runs(filter=\"bkk07\")",
            json!({"filter": s("Optional substring, glob, or status to filter runs")}),
            &[],
        ),
        tool(
            "read_metric",
            "Show the recent values of one metric in one run as a small table, plus min / max / last. \
Metric names are fuzzy: \"lr\", \"learning_rate\" and \"LR\" all work; on a miss the closest names are suggested.\n\
Example: read_metric(run=\"bkk07/small\", metric=\"brain_heldout_nll\", last=10)",
            json!({
                "run": s("Run name, e.g. \"bkk07/small\""),
                "metric": s("Metric name, e.g. \"loss\" or \"brain_heldout_nll\""),
                "last": n("How many most recent points to show (default 15, max 40). Optional."),
            }),
            &["run", "metric"],
        ),
        tool(
            "summarize",
            "Plain-language status of a run: for each key metric the latest value, the recent trend (improving / getting worse / no clear trend) \
with slope per N steps, spikes, plateau, an honest forecast (says \"no skill\" when it cannot beat the last value), and mean ± sd across seeds. \
Give metric and target to get an ETA, e.g. when the loss will reach 4.5.\n\
Example: summarize(run=\"bkk07/small\")  or  summarize(run=\"bkk07/small\", metric=\"brain_heldout_nll\", target=4.5)",
            json!({
                "run": s("Run name, e.g. \"bkk07/small\""),
                "metric": s("Optional metric to focus on (default: the run's key metrics)"),
                "target": n("Optional target value for the metric; the answer includes an ETA or says it cannot be predicted"),
            }),
            &["run"],
        ),
        tool(
            "compare",
            "Compare one metric across two or more runs: aligned table at the steps all runs share, the differences, \
the best run at its latest value and at the latest shared step. runs is a comma-separated list, a glob (\"bkk07/*\") or a group name (\"bkk07\").\n\
Example: compare(runs=\"bkk07/small,bkk07/mid\", metric=\"brain_heldout_nll\")",
            json!({
                "runs": s("Comma-separated run names, a glob like \"bkk07/*\", or a group like \"bkk07\""),
                "metric": s("Metric name, e.g. \"loss\""),
            }),
            &["runs", "metric"],
        ),
        tool(
            "make_board",
            "Save a dashboard board and return a URL to open it. query is metric{run=~\"pattern|pattern\"}; several panels separated by ';'. \
Run patterns may be full names, globs (bkk07/*) or short names (small).\n\
Example: make_board(query='brain_heldout_nll{run=~\"bkk07/small|bkk07/mid\"}')",
            json!({"query": s("Board query, e.g. loss{run=~\"bkk07/*\"}")}),
            &["query"],
        ),
    ]
}

/// Map a model-typed tool name onto one of ours.
fn canonical_tool(name: &str) -> Option<&'static str> {
    let n = text::squash(name.rsplit(['.', '/', ':']).next().unwrap_or(name));
    Some(match n.as_str() {
        "log" | "logmetric" | "logmetrics" | "record" => "log",
        "ingestlog" | "ingest" | "parselog" | "ingestlogs" => "ingest_log",
        "listruns" | "list" | "runs" | "listrun" => "list_runs",
        "readmetric" | "read" | "getmetric" | "metric" | "readmetrics" => "read_metric",
        "summarize" | "summarise" | "summary" | "status" | "summarizerun" => "summarize",
        "compare" | "compareruns" | "comparemetric" => "compare",
        "makeboard" | "board" | "dashboard" | "createboard" => "make_board",
        _ => return None,
    })
}

pub struct Server {
    pub store: Box<dyn Store>,
}

impl Server {
    pub fn new(store: Box<dyn Store>) -> Server {
        Server { store }
    }

    /// Run one tool. Returns `(text, is_error)`.
    pub fn call_tool(&self, name: &str, args: &Value) -> (String, bool) {
        let a = Args::new(args);
        let st = self.store.as_ref();
        let r = match canonical_tool(name) {
            Some("log") => tools::log(st, &a),
            Some("ingest_log") => tools::ingest_log(st, &a),
            Some("list_runs") => tools::list_runs(st, &a),
            Some("read_metric") => tools::read_metric(st, &a),
            Some("summarize") => tools::summarize(st, &a),
            Some("compare") => tools::compare(st, &a),
            Some("make_board") => tools::make_board(st, &a),
            _ => Err(format!("Unknown tool {name:?}. Tools: log, ingest_log, list_runs, read_metric, summarize, compare, make_board. Start with list_runs().")),
        };
        match r {
            Ok(t) => (t, false),
            Err(e) => (e, true),
        }
    }

    /// Handle one JSON-RPC message; `None` for notifications.
    pub fn handle(&self, msg: &Value) -> Option<Value> {
        if let Value::Array(batch) = msg {
            let out: Vec<Value> = batch.iter().filter_map(|m| self.handle(m)).collect();
            return (!out.is_empty()).then_some(Value::Array(out));
        }
        let id = msg.get("id").cloned();
        let method = msg.get("method").and_then(|m| m.as_str()).unwrap_or("");
        let id = id?; // notifications (no id) get no response
        let params = msg.get("params").cloned().unwrap_or(Value::Null);
        let ok = |result: Value| json!({"jsonrpc": "2.0", "id": id, "result": result});
        let err = |code: i64, message: &str| json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}});
        Some(match method {
            "initialize" => {
                let asked = params.get("protocolVersion").and_then(|v| v.as_str()).unwrap_or("");
                let v = if PROTOCOL_VERSIONS.contains(&asked) { asked } else { "2024-11-05" };
                ok(json!({
                    "protocolVersion": v,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": SERVER_NAME, "version": env!("CARGO_PKG_VERSION")},
                    "instructions": INSTRUCTIONS,
                }))
            }
            "ping" => ok(json!({})),
            "tools/list" => ok(json!({"tools": tool_specs()})),
            "tools/call" => {
                let Some(name) = params.get("name").and_then(|n| n.as_str()) else {
                    return Some(err(-32602, "tools/call needs params.name"));
                };
                let args = params.get("arguments").cloned().unwrap_or(json!({}));
                let (text, is_error) = self.call_tool(name, &args);
                ok(json!({"content": [{"type": "text", "text": text}], "isError": is_error}))
            }
            "resources/list" => ok(json!({"resources": []})),
            "prompts/list" => ok(json!({"prompts": []})),
            _ => err(-32601, &format!("method not found: {method}")),
        })
    }
}

/// Serve on stdin/stdout until EOF.
pub fn serve_stdio(server: &Server) -> std::io::Result<()> {
    eprintln!("obs mcp: serving {} on stdio", server.store.describe());
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let reply = match serde_json::from_str::<Value>(&line) {
            Ok(msg) => server.handle(&msg),
            Err(e) => Some(json!({"jsonrpc": "2.0", "id": null, "error": {"code": -32700, "message": format!("parse error: {e}")}})),
        };
        if let Some(r) = reply {
            let mut s = serde_json::to_string(&r).unwrap_or_default();
            s.push('\n');
            stdout.write_all(s.as_bytes())?;
            stdout.flush()?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
