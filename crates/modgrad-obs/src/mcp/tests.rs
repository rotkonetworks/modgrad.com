//! Tool and protocol tests on synthetic fixture runs in a temp directory.

use super::store::LocalStore;
use super::*;
use crate::{Meta, Run};
use std::path::PathBuf;

fn scratch(name: &str) -> PathBuf {
    let base = std::env::var("OBS_MCP_TEST_DIR").map(PathBuf::from).unwrap_or_else(|_| std::env::temp_dir().join("obs-mcp-tests"));
    let d = base.join(format!("{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn mk(root: &std::path::Path, name: &str, rows: &[(u64, &[(&str, f64)])], finished: bool) {
    let meta = Meta { project: None, git: None, hostname: None, start_unix: crate::now_unix() - 3600.0, extra: None };
    let mut r = Run::create_in(&root.join("runs").join(name), name, &json!({"lr": 3e-4}), meta).unwrap();
    for (s, kv) in rows {
        r.log_scalars(*s, kv).unwrap();
    }
    if finished {
        r.finish().unwrap();
    } else {
        r.flush_summary().unwrap();
    }
}

/// g/small, g/small_s2, g/small_s3, g/mid: eval-like curves with seeds;
/// lin: 40-point clean linear decline; flat: drop then plateau; spiky:
/// smooth with one spike.
fn fixture(name: &str) -> (PathBuf, Server) {
    let root = scratch(name);
    let curve = |vals: &[f64]| -> Vec<(u64, Vec<(&'static str, f64)>)> { vals.iter().enumerate().map(|(i, &v)| ((i as u64) * 250 + 1, vec![("heldout_nll", v), ("learning_rate", 3e-4)])).collect() };
    let put = |n: &str, rows: Vec<(u64, Vec<(&'static str, f64)>)>, fin: bool| {
        let r: Vec<(u64, &[(&str, f64)])> = rows.iter().map(|(s, kv)| (*s, kv.as_slice())).collect();
        mk(&root, n, &r, fin);
    };
    put("g/small", curve(&[8.99, 5.44, 5.07, 5.44, 5.38, 5.17, 5.32, 5.74, 6.11, 5.16, 4.88]), false);
    put("g/small_s2", curve(&[8.98, 5.45, 5.12, 5.0, 4.93]), false);
    put("g/small_s3", curve(&[8.98, 5.32, 5.2, 5.12, 5.44]), false);
    put("g/mid", curve(&[9.0, 5.98, 5.65, 5.27, 5.52, 5.04, 5.58, 4.84]), true);
    let lin: Vec<(u64, Vec<(&'static str, f64)>)> = (0..40).map(|i| (i * 10, vec![("train_loss", 10.0 - 0.1 * i as f64 + 0.002 * ((i * 7) % 5) as f64)])).collect();
    put("lin", lin, true);
    let flat: Vec<(u64, Vec<(&'static str, f64)>)> = (0..30).map(|i| (i * 10, vec![("val_loss", if i < 10 { 5.0 - 0.3 * i as f64 } else { 2.3 + 0.01 * ((i * 3) % 4) as f64 })])).collect();
    put("flat", flat, true);
    let spiky: Vec<(u64, Vec<(&'static str, f64)>)> = (0..30).map(|i| (i * 10, vec![("loss", 4.0 - 0.05 * i as f64 + 0.01 * ((i * 7) % 3) as f64 + if i == 17 { 2.0 } else { 0.0 })])).collect();
    put("spiky", spiky, true);
    let srv = Server::new(Box::new(LocalStore::open(&root, None)));
    (root, srv)
}

fn call(s: &Server, tool: &str, args: Value) -> (String, bool) {
    let (t, e) = s.call_tool(tool, &args);
    assert!(t.lines().count() <= 40, "{tool} output too long ({} lines):\n{t}", t.lines().count());
    (t, e)
}

fn ok(s: &Server, tool: &str, args: Value) -> String {
    let (t, e) = call(s, tool, args);
    assert!(!e, "{tool} failed: {t}");
    t
}

fn fail(s: &Server, tool: &str, args: Value) -> String {
    let (t, e) = call(s, tool, args);
    assert!(e, "{tool} should fail: {t}");
    assert!(t.contains('(') && (t.contains("Example") || t.contains("e.g.") || t.contains("Retry") || t.contains("Pick")), "error without an example call: {t}");
    t
}

#[test]
fn protocol() {
    let (_r, s) = fixture("protocol");
    let init = s.handle(&json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"t","version":"0"}}})).unwrap();
    assert_eq!(init["result"]["protocolVersion"], "2025-03-26");
    assert!(init["result"]["capabilities"]["tools"].is_object());
    let old = s.handle(&json!({"jsonrpc":"2.0","id":2,"method":"initialize","params":{"protocolVersion":"1999-01-01"}})).unwrap();
    assert_eq!(old["result"]["protocolVersion"], "2024-11-05");
    assert!(s.handle(&json!({"jsonrpc":"2.0","method":"notifications/initialized"})).is_none());
    let list = s.handle(&json!({"jsonrpc":"2.0","id":3,"method":"tools/list"})).unwrap();
    let names: Vec<&str> = list["result"]["tools"].as_array().unwrap().iter().map(|t| t["name"].as_str().unwrap()).collect();
    assert_eq!(names, ["log", "ingest_log", "list_runs", "read_metric", "summarize", "compare", "make_board"]);
    for t in list["result"]["tools"].as_array().unwrap() {
        assert!(t["description"].as_str().unwrap().contains("Example"), "{t}");
        for (_, p) in t["inputSchema"]["properties"].as_object().unwrap() {
            assert!(p["type"] == "string" || p["type"] == "number" || p["type"] == "object");
        }
    }
    let unk = s.handle(&json!({"jsonrpc":"2.0","id":4,"method":"foo/bar"})).unwrap();
    assert_eq!(unk["error"]["code"], -32601);
    let bad = s.handle(&json!({"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"read_metric","arguments":{"run":"nope_zz"}}})).unwrap();
    assert_eq!(bad["result"]["isError"], true);
    assert!(bad["result"]["content"][0]["text"].as_str().unwrap().contains("list_runs"));
    let call = s.handle(&json!({"jsonrpc":"2.0","id":"x","method":"tools/call","params":{"name":"List_Runs","arguments":{}}})).unwrap();
    assert_eq!(call["id"], "x");
    assert_eq!(call["result"]["isError"], false);
    let ping = s.handle(&json!({"jsonrpc":"2.0","id":6,"method":"ping"})).unwrap();
    assert!(ping["result"].is_object());
}

#[test]
fn log_tool() {
    let (root, s) = fixture("log");
    let t = ok(&s, "log", json!({"run": "demo1", "metrics": {"loss": 3.4, "acc": "7.5%", "lr": "3e-4"}, "step": "5"}));
    assert!(t.contains("Created new run demo1") && t.contains("step 5") && t.contains("acc=7.5"), "{t}");
    assert!(t.contains("Next:"));
    // step omitted → last + 1; metrics as a string; top-level metric args
    let t = ok(&s, "log", json!({"run": "demo1", "metrics": "loss=3.1, acc: 8%"}));
    assert!(t.contains("at step 6"), "{t}");
    let t = ok(&s, "log", json!({"run": "DEMO1", "loss": 2.9}));
    assert!(t.contains("run demo1 at step 7") && !t.contains("Created"), "{t}");
    let t = ok(&s, "log", json!({"run": "demo1", "metrics": {"step": 20, "loss": 2.5}}));
    assert!(t.contains("at step 20"), "{t}");
    let rd = crate::read_run(&root.join("runs/demo1")).unwrap();
    assert_eq!(rd.series("loss"), vec![(5, 3.4), (6, 3.1), (7, 2.9), (20, 2.5)]);
    assert_eq!(rd.series("acc"), vec![(5, 7.5), (6, 8.0)]);
    let sum: Value = serde_json::from_slice(&std::fs::read(root.join("runs/demo1/summary.json")).unwrap()).unwrap();
    assert_eq!(sum["summary"]["loss"], 2.5);
    // appending to an existing run keeps its config
    let before = std::fs::read(root.join("runs/g/mid/config.json")).unwrap();
    ok(&s, "log", json!({"run": "g/mid", "metrics": {"heldout_nll": 4.7}, "step": 2000}));
    assert_eq!(before, std::fs::read(root.join("runs/g/mid/config.json")).unwrap());
    // errors explain and give an example
    let e = fail(&s, "log", json!({"metrics": {"loss": 1}}));
    assert!(e.contains("Missing run"));
    let e = fail(&s, "log", json!({"run": "demo1", "metrics": {"loss": "abc"}}));
    assert!(e.contains("not a number"), "{e}");
    fail(&s, "log", json!({"run": "../escape", "metrics": {"loss": 1}}));
}

#[test]
fn ingest_tool() {
    let (root, s) = fixture("ingest");
    let text = "Loading model...\nstep 100 | loss 2.31 | lr 3e-4 | acc 71.2%\nstep 200 loss=2.05 lr=0.0003 acc=73%\n  eval val_loss: 2.4\nsaved checkpoint to /tmp/x\nstep   250: train loss (ema) 3.614; BRAIN in-dist   0.0% nll 4.24 | held-out   5.0% nll 5.43  [1.95 s/step, 10 min, RSS 2392584 kB]\n";
    let t = ok(&s, "ingest_log", json!({"run": "ing", "text": text}));
    assert!(t.contains("Ingested 4 rows") && t.contains("(new run)"), "{t}");
    assert!(t.contains("2 skipped") && t.contains("Loading model"), "{t}");
    assert!(t.contains("1 ctmlm_pilot"), "{t}");
    let rd = crate::read_run(&root.join("runs/ing")).unwrap();
    assert_eq!(rd.series("loss"), vec![(100, 2.31), (200, 2.05)]);
    // textlog keeps a leading word as a prefix ("eval val_loss" -> eval_val_loss)
    let vk = rd.metric_keys().into_iter().find(|k| k.ends_with("val_loss")).expect("a val_loss key");
    assert_eq!(rd.series(&vk), vec![(200, 2.4)], "key {vk}");
    assert_eq!(rd.series("brain_heldout_nll"), vec![(250, 5.43)]);
    let e = fail(&s, "ingest_log", json!({"run": "ing", "text": "hello\nworld"}));
    assert!(e.contains("no metric lines") || e.contains("Found no metric"), "{e}");
    // array of lines and double-escaped newlines
    let t = ok(&s, "ingest_log", json!({"run": "ing2", "lines": ["step 1 loss 3", "step 2 loss 2.5"]}));
    assert!(t.contains("Ingested 2 rows"), "{t}");
    let t = ok(&s, "ingest_log", json!({"run": "ing3", "text": "step 1 loss 3\\nstep 2 loss 2.5"}));
    assert!(t.contains("Ingested 2 rows"), "{t}");
}

#[test]
fn list_tool() {
    let (_r, s) = fixture("list");
    let t = ok(&s, "list_runs", json!({}));
    assert!(t.starts_with("7 runs"), "{t}");
    for n in ["g/small", "g/small_s2", "g/mid", "lin", "flat", "spiky"] {
        assert!(t.contains(n), "{n} missing: {t}");
    }
    assert!(t.contains("finished") && t.contains("running"), "{t}");
    let t = ok(&s, "list_runs", json!({"filter": "g/*"}));
    assert!(t.starts_with("4 runs") && !t.contains("spiky"), "{t}");
    let t = ok(&s, "list_runs", json!({"filter": "finished"}));
    assert!(t.contains("g/mid") && !t.contains("g/small "), "{t}");
    fail(&s, "list_runs", json!({"filter": "zzz"}));
}

#[test]
fn read_metric_tool() {
    let (_r, s) = fixture("read");
    let t = ok(&s, "read_metric", json!({"run": "mid", "metric": "LR"}));
    assert!(t.contains("learning_rate in run g/mid"), "{t}");
    let t = ok(&s, "read_metric", json!({"run": "g/small", "metric": "heldout nll", "last": 3}));
    assert!(t.contains("showing the last 3 of 11"), "{t}");
    assert!(t.contains("min 4.88 at step 2501") && t.contains("max 8.99 at step 1"), "{t}");
    let e = fail(&s, "read_metric", json!({"run": "g/small", "metric": "accuracy"}));
    assert!(e.contains("Closest"), "{e}");
    let e = fail(&s, "read_metric", json!({"run": "sma", "metric": "loss"}));
    assert!(e.contains("ambiguous"), "{e}");
}

#[test]
fn summarize_tool() {
    let (_r, s) = fixture("summ");
    let t = ok(&s, "summarize", json!({"run": "lin"}));
    assert!(t.contains("train_loss (lower is better)") && t.contains("→ improving"), "{t}");
    assert!(t.contains("per 10 steps"), "{t}");
    let t = ok(&s, "summarize", json!({"run": "lin", "metric": "loss", "target": 5.0}));
    assert!(t.contains("Target 5"), "{t}");
    assert!(t.contains("ETA step") || t.contains("no skill"), "{t}");
    let t = ok(&s, "summarize", json!({"run": "flat"}));
    assert!(t.contains("Plateau: yes"), "{t}");
    let t = ok(&s, "summarize", json!({"run": "spiky"}));
    assert!(t.contains("Spikes: step 170"), "{t}");
    let t = ok(&s, "summarize", json!({"run": "g/small", "target": 4.5}));
    assert!(t.contains("Seeds: 3 runs") && t.contains("mean ± sd, n=3"), "{t}");
    assert!(t.contains("heldout_nll at step 1001"), "{t}");
    assert!(t.contains("Target 4.5"), "{t}");
    let t = ok(&s, "summarize", json!({"run": "g/small", "metric": "heldout_nll", "target": 5.0}));
    assert!(t.contains("already reached"), "{t}");
    let t = ok(&s, "summarize", json!({"run": "g/small_s3"}));
    assert!(t.contains("too few points to judge"), "{t}");
    fail(&s, "summarize", json!({"run": "lin", "target": "soon"}));
}

#[test]
fn honest_forecast_says_no_skill_on_noise() {
    let root = scratch("noise");
    // deterministic pseudo-noise around a constant: nothing beats last value reliably
    let mut x = 12345u64;
    let rows: Vec<(u64, Vec<(&str, f64)>)> = (0..40)
        .map(|i| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (i * 10, vec![("loss", 3.0 + ((x >> 33) as f64 / (1u64 << 31) as f64 - 0.5))])
        })
        .collect();
    let r: Vec<(u64, &[(&str, f64)])> = rows.iter().map(|(s, kv)| (*s, kv.as_slice())).collect();
    mk(&root, "noise", &r, true);
    let s = Server::new(Box::new(LocalStore::open(&root, None)));
    let t = ok(&s, "summarize", json!({"run": "noise", "target": 2.0}));
    assert!(t.contains("no clear trend"), "{t}");
    assert!(t.contains("no skill") || t.contains("Forecast (ridge"), "{t}");
    if t.contains("no skill") {
        assert!(t.contains("no reliable ETA"), "{t}");
    }
}

#[test]
fn compare_tool() {
    let (_r, s) = fixture("cmp");
    let t = ok(&s, "compare", json!({"runs": "g/small, g/mid", "metric": "nll"}));
    assert!(t.contains("mid-small"), "{t}");
    assert!(t.contains("lowest is g/mid = 4.84"), "{t}");
    assert!(t.contains("latest step all runs share (1751)"), "{t}");
    let t = ok(&s, "compare", json!({"runs": "g", "metric": "heldout_nll"}));
    assert!(t.contains("across 4 runs") && t.contains("best"), "{t}");
    let t = ok(&s, "compare", json!({"runs": ["g/small_s2", "g/small_s3"], "metric": "heldout_nll"}));
    assert!(t.contains("lowest is g/small_s2 = 4.93"), "{t}");
    let e = fail(&s, "compare", json!({"runs": "g/mid", "metric": "loss"}));
    assert!(e.contains("at least two"), "{e}");
}

#[test]
fn board_tool() {
    let (root, s) = fixture("board");
    let t = ok(&s, "make_board", json!({"query": "heldout_nll{run=~\"small|mid\"}"}));
    assert!(t.contains("URL: file://") && t.contains("#board="), "{t}");
    assert!(t.contains("g/small, g/mid"), "{t}");
    let boards: Vec<_> = std::fs::read_dir(root.join("boards")).unwrap().collect();
    assert_eq!(boards.len(), 1);
    let b: Value = serde_json::from_slice(&std::fs::read(boards[0].as_ref().unwrap().path()).unwrap()).unwrap();
    assert_eq!(b["panels"][0]["query"], "heldout_nll{run=~\"g/small|g/mid\"}");
    let t = ok(&s, "make_board", json!({"query": "heldout_nll{run=~\"g/.*\"}; learning_rate"}));
    assert!(t.contains("2 panels"), "{t}");
    let t = ok(&s, "make_board", json!({"query": "nll", "runs": "g/small,g/mid"}));
    assert!(t.contains("heldout_nll for 2 runs"), "{t}");
    fail(&s, "make_board", json!({"query": "nope_metric{run=~\"g/*\"}"}));
    // commas between panels (not inside the selector) split; same metric merges
    let t = ok(&s, "make_board", json!({"query": "heldout_nll{run=~\"g/small\"}, learning_rate{run=~\"g/small\"}, heldout_nll{run=~\"g/mid\"}"}));
    assert!(t.contains("2 panels") && t.contains("heldout_nll for 2 runs"), "{t}");
    let (t, e) = s.call_tool("make_board", &json!({"query": "heldout_nll{run=~\"small/*\"}"}));
    assert!(e && t.contains("g/small"), "{t}");
}

#[test]
fn remote_mode_unreachable_server_teaches() {
    // port 1 refuses connections; the full remote tests live in modgrad-obs-server
    let s = Server::new(Box::new(store::RemoteStore::new("http://127.0.0.1:1", None)));
    let (t, e) = s.call_tool("list_runs", &json!({}));
    assert!(e && t.contains("Cannot reach") && t.contains("obs mcp --dir"), "{t}");
}
