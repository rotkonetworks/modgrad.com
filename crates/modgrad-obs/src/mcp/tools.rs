//! The seven MCP tools. Each takes forgiving flat arguments and returns
//! short plain text: `Ok` text ends with what to do next, `Err` text says
//! what went wrong and gives an example call.

use super::analysis::{self, Dir};
use super::store::{Rows, Store, valid_run_name};
use super::text::{Resolved, closest, fmt, fmt_age, fmt_signed, glob, norm_key, num, parse_generic_line, resolve_metric, squash, table};
use crate::RunData;
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, BTreeSet};

pub type ToolResult = Result<String, String>;

// ------------------------------------------------------------- arguments

/// Tool arguments with case-, punctuation- and alias-tolerant lookup.
pub struct Args(Map<String, Value>);

impl Args {
    pub fn new(v: &Value) -> Args {
        match v {
            Value::Object(m) => Args(m.clone()),
            // some clients send the arguments object as a JSON string
            Value::String(s) => Args(serde_json::from_str::<Value>(s).ok().and_then(|v| v.as_object().cloned()).unwrap_or_default()),
            _ => Args(Map::new()),
        }
    }

    pub fn get(&self, names: &[&str]) -> Option<&Value> {
        for n in names {
            let n = squash(n);
            if let Some((_, v)) = self.0.iter().find(|(k, v)| squash(k) == n && !v.is_null()) {
                return Some(v);
            }
        }
        None
    }

    /// String argument; numbers are stringified, arrays joined with ", ".
    pub fn str(&self, names: &[&str]) -> Option<String> {
        let s = match self.get(names)? {
            Value::String(s) => s.trim().to_string(),
            Value::Number(n) => n.to_string(),
            Value::Array(a) => a.iter().map(|x| x.as_str().map(String::from).unwrap_or_else(|| x.to_string())).collect::<Vec<_>>().join(", "),
            other => other.to_string(),
        };
        (!s.is_empty()).then_some(s)
    }

    pub fn num(&self, names: &[&str]) -> Option<f64> {
        self.get(names).and_then(num)
    }

    fn keys(&self) -> impl Iterator<Item = (&String, &Value)> {
        self.0.iter()
    }
}

const RUN: &[&str] = &["run", "run_name", "runname", "name", "run_id", "id"];
const METRIC: &[&str] = &["metric", "metric_name", "key", "metrics_name", "series"];
const STEP: &[&str] = &["step", "global_step", "iteration", "iter"];

fn strip_quotes(s: &str) -> &str {
    s.trim().trim_matches(|c| c == '"' || c == '\'' || c == '`').trim()
}

// --------------------------------------------------------------- run names

/// Resolve a user-typed run name: exact → case-insensitive → last path
/// component → punctuation-insensitive → unique substring.
pub fn resolve_run(store: &dyn Store, q: &str) -> Result<String, String> {
    let names = store.run_names()?;
    let q = strip_quotes(q);
    let q = q.strip_prefix("runs/").unwrap_or(q).trim_matches('/');
    if names.is_empty() {
        return Err(format!("There are no runs yet in {}. Create one with log(run=\"demo1\", metrics={{\"loss\": 2.5}}, step=1).", store.describe()));
    }
    if let Some(n) = names.iter().find(|n| *n == q) {
        return Ok(n.clone());
    }
    let ql = q.to_lowercase();
    let pick = |hits: Vec<&String>| -> Option<Result<String, String>> {
        match hits.len() {
            0 => None,
            1 => Some(Ok(hits[0].clone())),
            _ => Some(Err(format!(
                "Run name {q:?} is ambiguous; it matches {}. Use the full name, e.g. summarize(run=\"{}\").",
                hits.iter().take(8).map(|s| s.as_str()).collect::<Vec<_>>().join(", "),
                hits[0]
            ))),
        }
    };
    let tries: [Box<dyn Fn(&String) -> bool>; 4] = [
        Box::new(|n: &String| n.to_lowercase() == ql),
        Box::new(|n: &String| n.rsplit('/').next().is_some_and(|l| l.to_lowercase() == ql)),
        Box::new(|n: &String| squash(n) == squash(q) || n.rsplit('/').next().is_some_and(|l| squash(l) == squash(q))),
        Box::new(|n: &String| n.to_lowercase().contains(&ql)),
    ];
    for t in tries.iter() {
        if let Some(r) = pick(names.iter().filter(|n| t(n)).collect()) {
            return r;
        }
    }
    let sug = closest(q, &names, 5);
    Err(format!("No run named {q:?}. Closest: {}. Call list_runs() to see all runs, then e.g. summarize(run=\"{}\").", sug.join(", "), sug.first().copied().unwrap_or("NAME")))
}

/// A run list: comma/space/`and`-separated names, globs (`bkk07/*`) or a
/// group prefix (`bkk07` = every run under `bkk07/`).
pub fn resolve_runs(store: &dyn Store, spec: &str) -> Result<Vec<String>, String> {
    let names = store.run_names()?;
    let spec = spec.replace(" and ", ",").replace(" vs ", ",").replace(" vs. ", ",");
    let mut out: Vec<String> = Vec::new();
    let push = |n: String, out: &mut Vec<String>| {
        if !out.contains(&n) {
            out.push(n)
        }
    };
    for item in spec.split([',', ' ', '|', ';', '\n']).map(strip_quotes).filter(|s| !s.is_empty()) {
        let item = item.trim_start_matches('[').trim_end_matches(']');
        let item = strip_quotes(item);
        if item.is_empty() {
            continue;
        }
        if item.contains('*') || item.contains('?') {
            let pat = item.strip_prefix("runs/").unwrap_or(item);
            let hits: Vec<&String> = names.iter().filter(|n| glob(pat, n)).collect();
            if hits.is_empty() {
                return Err(format!("Pattern {item:?} matches no run. Call list_runs() to see names; e.g. compare(runs=\"bkk07/*\", metric=\"loss\")."));
            }
            hits.into_iter().for_each(|h| push(h.clone(), &mut out));
            continue;
        }
        let group: Vec<&String> = names.iter().filter(|n| n.starts_with(&format!("{}/", item.trim_end_matches('/')))).collect();
        if !group.is_empty() && !names.iter().any(|n| n == item) {
            group.into_iter().for_each(|h| push(h.clone(), &mut out));
            continue;
        }
        push(resolve_run(store, item)?, &mut out);
    }
    Ok(out)
}

fn load(store: &dyn Store, run: &str) -> Result<RunData, String> {
    store.load(run)
}

/// Finite `(steps, values)` of one key, sorted by step, one value per step
/// (the last logged).
fn series(rd: &RunData, key: &str) -> (Vec<f64>, Vec<f64>) {
    let mut m: BTreeMap<u64, f64> = BTreeMap::new();
    for (s, v) in rd.series(key) {
        if v.is_finite() {
            m.insert(s, v);
        }
    }
    (m.keys().map(|&s| s as f64).collect(), m.values().copied().collect())
}

fn resolve_metric_in(q: &str, keys: &[String], run_label: &str, tool: &str, run_arg: &str) -> Result<(String, Option<String>), String> {
    match resolve_metric(q, keys) {
        Resolved::One { key, note } => Ok((key, note)),
        Resolved::Many(v) => Err(format!(
            "Metric {q:?} matches several metrics in {run_label}: {}. Pick one, e.g. {tool}({run_arg}, metric=\"{}\").",
            v.iter().take(10).cloned().collect::<Vec<_>>().join(", "),
            v[0]
        )),
        Resolved::None(s) => Err(format!(
            "No metric {q:?} in {run_label}. Closest: {}. Retry e.g. {tool}({run_arg}, metric=\"{}\").",
            if s.is_empty() { "(the run has no metrics)".to_string() } else { s.join(", ") },
            s.first().map(String::as_str).unwrap_or("loss")
        )),
    }
}

fn status_of(info: &super::store::RunInfo) -> (String, String) {
    let age = info.age_s.map(|a| format!("{} ago", fmt_age(a))).unwrap_or_else(|| "-".into());
    let st = if info.finished {
        "finished"
    } else if info.age_s.is_some_and(|a| a < super::store::RUNNING_WINDOW_S) {
        "running"
    } else {
        "idle"
    };
    (st.to_string(), age)
}

// ------------------------------------------------------------------- log

fn clean_key(k: &str) -> String {
    let k = k.trim();
    if !k.is_empty() && k.chars().all(|c| c.is_ascii_alphanumeric() || "_./-".contains(c)) { k.to_string() } else { norm_key(k) }
}

/// Metrics from an object, a JSON string, a `k=v, k2: v2` string or a
/// `[{name, value}]` list. Returns `(pairs, step found inside, rejected)`.
fn parse_metrics(v: &Value) -> (Vec<(String, f64)>, Option<u64>, Vec<String>) {
    let (mut kv, mut step, mut bad) = (Vec::new(), None, Vec::new());
    let take = |k: &str, x: &Value, kv: &mut Vec<(String, f64)>, step: &mut Option<u64>, bad: &mut Vec<String>| {
        let key = clean_key(k);
        match num(x) {
            Some(n) if matches!(key.as_str(), "step" | "global_step" | "iteration" | "iter") && n >= 0.0 => *step = Some(n as u64),
            Some(_) if key == "t" => bad.push("t (reserved for time)".into()),
            Some(n) if !key.is_empty() => kv.push((key, n)),
            _ => bad.push(format!("{k}={x} (not a number)")),
        }
    };
    match v {
        Value::Object(m) => m.iter().for_each(|(k, x)| take(k, x, &mut kv, &mut step, &mut bad)),
        Value::Array(a) => {
            for it in a {
                let name = it.get("name").or_else(|| it.get("metric")).or_else(|| it.get("key")).and_then(|n| n.as_str());
                match (name, it.get("value")) {
                    (Some(n), Some(x)) => take(n, x, &mut kv, &mut step, &mut bad),
                    _ => bad.push(format!("{it} (expected {{\"name\": ..., \"value\": ...}})")),
                }
            }
        }
        Value::String(s) => {
            if let Ok(j @ (Value::Object(_) | Value::Array(_))) = serde_json::from_str::<Value>(s) {
                return parse_metrics(&j);
            }
            match parse_generic_line(s) {
                Some((st, pairs)) => {
                    step = st;
                    kv = pairs;
                }
                None => bad.push(format!("{s:?} (no name=value pairs)")),
            }
        }
        other => bad.push(format!("{other} (expected an object like {{\"loss\": 2.5}})")),
    }
    (kv, step, bad)
}

pub fn log(store: &dyn Store, a: &Args) -> ToolResult {
    let ex = r#"log(run="demo1", metrics={"loss": 2.31, "acc": "71.2%"}, step=100)"#;
    let Some(run) = a.str(RUN) else {
        return Err(format!("Missing run name. Example: {ex}"));
    };
    let run = strip_quotes(&run).trim_start_matches("runs/").trim_matches('/').to_string();
    let mvals = a.get(&["metrics", "values", "data", "metric", "scalars", "kv", "items"]);
    let (mut kv, mut inner_step, mut bad) = match mvals {
        Some(v) => parse_metrics(v),
        None => (Vec::new(), None, Vec::new()),
    };
    // metrics passed as top-level arguments: log(run="x", loss=2.3)
    if mvals.is_none() {
        let skip: Vec<String> = RUN.iter().chain(STEP).map(|s| squash(s)).collect();
        let extra: Map<String, Value> = a.keys().filter(|(k, _)| !skip.contains(&squash(k))).map(|(k, v)| (k.clone(), v.clone())).collect();
        let (k2, s2, b2) = parse_metrics(&Value::Object(extra));
        kv = k2;
        inner_step = s2;
        bad = b2;
    }
    if kv.is_empty() {
        let why = if bad.is_empty() { String::new() } else { format!(" Rejected: {}.", bad.join("; ")) };
        return Err(format!("No numeric metrics given.{why} Pass metrics as a flat object of name → number, e.g. {ex}"));
    }
    // reuse the run if it exists under exactly this name (case-insensitive)
    let names = store.run_names().unwrap_or_default();
    let existing = names.iter().find(|n| n.eq_ignore_ascii_case(&run)).cloned();
    let run = existing.clone().unwrap_or(run);
    if !valid_run_name(&run) {
        return Err(format!("Invalid run name {run:?}: use letters, digits, '.', '_', '-' (and '/' for a group). Example: {ex}"));
    }
    let explicit = a.num(STEP);
    if let Some(s) = explicit {
        if s < 0.0 || s.fract() != 0.0 {
            return Err(format!("step must be a whole number ≥ 0, got {s}. Example: {ex}"));
        }
    }
    let prev_last = existing.as_ref().and_then(|n| store.info(n).ok()).and_then(|i| i.last_step);
    let (step, how) = match (explicit.map(|s| s as u64).or(inner_step), prev_last) {
        (Some(s), _) => (s, ""),
        (None, Some(l)) => (l + 1, " (step not given: used last step + 1)"),
        (None, None) => (0, " (step not given: first row is step 0)"),
    };
    let created = store.append(&run, &vec![(step, kv.clone())])?;
    let shown: Vec<String> = kv.iter().take(12).map(|(k, v)| format!("{k}={}", fmt(*v))).collect();
    let mut out = format!("Logged {} metric{} to run {run} at step {step}{how}: {}{}.", kv.len(), if kv.len() == 1 { "" } else { "s" }, shown.join(", "), if kv.len() > 12 { ", …" } else { "" });
    if created {
        out.push_str(&format!("\nCreated new run {run}."));
    }
    if prev_last.is_some_and(|l| step < l) {
        out.push_str(&format!("\nNote: step {step} is before the run's last step {}; rows are kept in logging order.", prev_last.unwrap()));
    }
    if !bad.is_empty() {
        out.push_str(&format!("\nSkipped: {}.", bad.join("; ")));
    }
    let k0 = &kv[0].0;
    out.push_str(&format!("\nNext: log the next step with log(run=\"{run}\", metrics={{\"{k0}\": …}}, step={}), or check it with read_metric(run=\"{run}\", metric=\"{k0}\").", step + 1));
    Ok(out)
}

// ------------------------------------------------------------ ingest_log

pub fn ingest_log(store: &dyn Store, a: &Args) -> ToolResult {
    let ex = r#"ingest_log(run="demo1", text="step 100 loss=2.31 lr=3e-4\nstep 200 loss=2.05 lr=3e-4")"#;
    let Some(run) = a.str(RUN) else {
        return Err(format!("Missing run name. Example: {ex}"));
    };
    let Some(text) = a.get(&["text", "log", "logs", "lines", "content", "stdout", "output", "data"]).map(|v| match v {
        Value::Array(x) => x.iter().map(|l| l.as_str().map(String::from).unwrap_or_else(|| l.to_string())).collect::<Vec<_>>().join("\n"),
        Value::String(s) => s.clone(),
        o => o.to_string(),
    }) else {
        return Err(format!("Missing text: pass the raw log lines as one string. Example: {ex}"));
    };
    let run = strip_quotes(&run).trim_start_matches("runs/").trim_matches('/').to_string();
    let names = store.run_names().unwrap_or_default();
    let existing = names.iter().find(|n| n.eq_ignore_ascii_case(&run)).cloned();
    let run = existing.clone().unwrap_or(run);
    if !valid_run_name(&run) {
        return Err(format!("Invalid run name {run:?}: use letters, digits, '.', '_', '-' (and '/' for a group). Example: {ex}"));
    }
    let prev_last = existing.as_ref().and_then(|n| store.info(n).ok()).and_then(|i| i.last_step);

    let mut rows: Rows = Vec::new();
    let (mut n_lines, mut n_legacy, mut n_generic, mut n_inferred) = (0usize, 0usize, 0usize, 0usize);
    let mut skipped: Vec<String> = Vec::new();
    let mut n_skipped = 0usize;
    let mut last_explicit: Option<u64> = None;
    let mut counter = prev_last.map(|l| l + 1).unwrap_or(0);
    // the text may arrive with literal "\n" when a client double-escapes
    let text = if !text.contains('\n') && text.contains("\\n") { text.replace("\\n", "\n") } else { text };
    for line in text.lines() {
        let l = line.trim();
        if l.is_empty() {
            continue;
        }
        n_lines += 1;
        if let Some((step, kv, _)) = crate::legacy::parse_step_line(l) {
            n_legacy += 1;
            last_explicit = Some(step);
            rows.push((step, kv.into_iter().map(|(k, v)| (k.to_string(), v)).collect()));
            continue;
        }
        // TODO(merge): swap for textlog::parse_line from the text-log ingest branch.
        match parse_generic_line(l) {
            Some((Some(step), kv)) => {
                n_generic += 1;
                last_explicit = Some(step);
                counter = counter.max(step + 1);
                rows.push((step, kv));
            }
            Some((None, kv)) => {
                n_generic += 1;
                n_inferred += 1;
                let step = last_explicit.unwrap_or_else(|| {
                    counter += 1;
                    counter - 1
                });
                rows.push((step, kv));
            }
            None => {
                n_skipped += 1;
                if skipped.len() < 3 {
                    let s: String = l.chars().take(80).collect();
                    skipped.push(format!("{s:?}"));
                }
            }
        }
    }
    if rows.is_empty() {
        return Err(format!(
            "Found no metric lines in {n_lines} line(s). Lines need name/number pairs, like \"step 100 loss=2.31 acc=71.2%\" or \"step 100 | loss 2.31\".{} Example: {ex}",
            if skipped.is_empty() { String::new() } else { format!(" Skipped e.g.: {}.", skipped.join(", ")) }
        ));
    }
    let created = store.append(&run, &rows)?;
    let n_vals: usize = rows.iter().map(|(_, kv)| kv.len()).sum();
    let keys: BTreeSet<&str> = rows.iter().flat_map(|(_, kv)| kv.iter().map(|(k, _)| k.as_str())).collect();
    let (lo, hi) = rows.iter().fold((u64::MAX, 0), |(a, b), (s, _)| (a.min(*s), b.max(*s)));
    let keys_v: Vec<&str> = keys.iter().copied().collect();
    let mut out = format!(
        "Ingested {} rows ({n_vals} values) into run {run}{}, steps {lo}–{hi}. {} metric{}: {}{}.",
        rows.len(),
        if created { " (new run)" } else { "" },
        keys.len(),
        if keys.len() == 1 { "" } else { "s" },
        keys_v.iter().take(12).copied().collect::<Vec<_>>().join(", "),
        if keys.len() > 12 { ", …" } else { "" }
    );
    out.push_str(&format!("\nRead {n_lines} non-empty lines: {n_generic} key/value lines, {n_legacy} ctmlm_pilot lines, {n_skipped} skipped."));
    if n_inferred > 0 {
        out.push_str(&format!("\n{n_inferred} line(s) had no step number; they got the previous line's step (or the next free step)."));
    }
    if !skipped.is_empty() {
        out.push_str(&format!("\nSkipped lines, e.g.: {}.", skipped.join(", ")));
    }
    let k0 = keys_v.iter().find(|k| k.contains("loss")).unwrap_or(&keys_v[0]);
    out.push_str(&format!("\nNext: read_metric(run=\"{run}\", metric=\"{k0}\") to see the values, or summarize(run=\"{run}\") for trends."));
    Ok(out)
}

// -------------------------------------------------------------- list_runs

pub fn list_runs(store: &dyn Store, a: &Args) -> ToolResult {
    let filter = a.str(&["filter", "pattern", "match", "query", "q", "prefix", "group", "project", "run", "status"]).map(|s| strip_quotes(&s).to_string());
    let names = store.run_names()?;
    if names.is_empty() {
        return Ok(format!("No runs yet in {}.\nNext: create one with log(run=\"demo1\", metrics={{\"loss\": 2.5}}, step=1) or ingest_log(run=\"demo1\", text=\"...\").", store.describe()));
    }
    let status_words = ["running", "finished", "idle"];
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut shown_names: Vec<String> = Vec::new();
    for n in &names {
        if let Some(f) = &filter {
            let fl = f.to_lowercase();
            if !status_words.contains(&fl.as_str()) {
                let ok = if f.contains('*') || f.contains('?') { glob(f, n) } else { n.to_lowercase().contains(&fl) };
                if !ok {
                    continue;
                }
            }
        }
        let info = match store.info(n) {
            Ok(i) => i,
            Err(e) => {
                rows.push(vec![n.clone(), "-".into(), "-".into(), format!("error: {e}"), "-".into()]);
                continue;
            }
        };
        let (st, age) = status_of(&info);
        if let Some(f) = &filter {
            if status_words.contains(&f.to_lowercase().as_str()) && st != f.to_lowercase() {
                continue;
            }
        }
        shown_names.push(n.clone());
        rows.push(vec![n.clone(), info.last_step.map(|s| s.to_string()).unwrap_or("-".into()), info.n_metrics.to_string(), st, age]);
    }
    if rows.is_empty() {
        let f = filter.unwrap_or_default();
        return Err(format!("No run matches filter {f:?} ({} runs exist, e.g. {}). Try list_runs() with no filter, or list_runs(filter=\"{}\").", names.len(), names.iter().take(3).cloned().collect::<Vec<_>>().join(", "), names[0].split('/').next().unwrap_or("")));
    }
    let total = rows.len();
    let cap = 30;
    let mut out = format!(
        "{total} run{}{} in {}:\n",
        if total == 1 { "" } else { "s" },
        filter.as_ref().map(|f| format!(" matching {f:?}")).unwrap_or_default(),
        store.describe()
    );
    rows.truncate(cap);
    out.push_str(&table(&["run".into(), "last step".into(), "metrics".into(), "status".into(), "updated".into()], &rows));
    if total > cap {
        out.push_str(&format!("\n… {} more not shown; narrow with list_runs(filter=\"group/*\").", total - cap));
    }
    out.push_str("\n(status: finished = run called finish; running = new rows in the last 15 min; idle = neither)");
    let r0 = &shown_names.first().cloned().unwrap_or_else(|| names[0].clone());
    out.push_str(&format!("\nNext: summarize(run=\"{r0}\") for a status report, read_metric(run=\"{r0}\", metric=\"loss\") for values, compare(runs=\"{}\", metric=\"loss\") across runs.", shown_names.iter().take(2).cloned().collect::<Vec<_>>().join(",")));
    Ok(out)
}

// ------------------------------------------------------------ read_metric

pub fn read_metric(store: &dyn Store, a: &Args) -> ToolResult {
    let ex = r#"read_metric(run="bkk07/small", metric="loss", last=10)"#;
    let Some(rq) = a.str(RUN) else {
        return Err(format!("Missing run name. Example: {ex}. list_runs() shows the names."));
    };
    let run = resolve_run(store, &rq)?;
    let rd = load(store, &run)?;
    let keys = rd.metric_keys();
    let Some(mq) = a.str(METRIC) else {
        return Err(format!("Missing metric name. Run {run} has: {}. Example: read_metric(run=\"{run}\", metric=\"{}\").", keys.iter().take(20).cloned().collect::<Vec<_>>().join(", "), keys.first().map(String::as_str).unwrap_or("loss")));
    };
    let (key, note) = resolve_metric_in(&mq, &keys, &format!("run {run}"), "read_metric", &format!("run=\"{run}\""))?;
    let (st, v) = series(&rd, &key);
    if v.is_empty() {
        return Err(format!("Metric {key} in run {run} has no finite values. Try another metric: {}.", keys.iter().take(10).cloned().collect::<Vec<_>>().join(", ")));
    }
    let want = a.num(&["last", "n", "limit", "count", "rows", "tail"]).map(|x| x.max(1.0) as usize).unwrap_or(15);
    let show = want.min(40).min(v.len());
    let mut out = String::new();
    if let Some(n) = note {
        out.push_str(&format!("({n})\n"));
    }
    out.push_str(&format!("{key} in run {run}, {} points, steps {}–{}:\n", v.len(), st[0], st[st.len() - 1]));
    let rows: Vec<Vec<String>> = (v.len() - show..v.len()).map(|i| vec![format!("{}", st[i]), fmt(v[i])]).collect();
    out.push_str(&table(&["step".into(), key.clone()], &rows));
    if show < v.len() {
        out.push_str(&format!("\n(showing the last {show} of {} points; pass last={} for more, max 40)", v.len(), v.len().min(40)));
    }
    let (imin, imax) = (0..v.len()).fold((0, 0), |(a, b), i| (if v[i] < v[a] { i } else { a }, if v[i] > v[b] { i } else { b }));
    out.push_str(&format!("\nmin {} at step {}, max {} at step {}, last {} at step {}.", fmt(v[imin]), st[imin], fmt(v[imax]), st[imax], fmt(v[v.len() - 1]), st[st.len() - 1]));
    out.push_str(&format!("\nNext: summarize(run=\"{run}\", metric=\"{key}\") for trend and forecast, or compare(runs=\"{run},OTHER\", metric=\"{key}\")."));
    Ok(out)
}

// -------------------------------------------------------------- summarize

/// Metrics worth a status line when the caller names none: loss / nll /
/// accuracy-like keys with ≥ 3 points, not baselines or derived stats.
fn key_metrics(rd: &RunData) -> Vec<String> {
    let keys = rd.metric_keys();
    let derived = |k: &str| {
        let k = k.to_lowercase();
        k.starts_with("none_") || k.starts_with("nat_") || k.starts_with("final_") || k.ends_with("_std") || k.ends_with("_sem") || k.contains("minus") || k.contains("baseline")
    };
    let rank = |k: &str| -> usize {
        let k = k.to_lowercase();
        let eval = ["val", "heldout", "eval", "test", "valid"].iter().any(|w| k.contains(w));
        match (analysis::direction(&k), eval) {
            (Dir::Lower, false) if k.contains("loss") => 0,
            (Dir::Lower, true) => 1,
            (Dir::Higher, true) => 2,
            (Dir::Lower, false) => 3,
            (Dir::Higher, false) => 4,
            _ => 9,
        }
    };
    let mut c: Vec<(usize, usize, String)> = keys
        .iter()
        .filter(|k| !derived(k) && analysis::direction(k) != Dir::Unknown)
        .map(|k| (rank(k), usize::MAX - series(rd, k).1.len(), k.clone()))
        .filter(|(_, n, _)| usize::MAX - n >= 3)
        .collect();
    c.sort();
    let mut out: Vec<String> = c.into_iter().map(|(_, _, k)| k).take(4).collect();
    if out.is_empty() {
        let mut all: Vec<(usize, String)> = keys.iter().map(|k| (usize::MAX - series(rd, k).1.len(), k.clone())).collect();
        all.sort();
        out = all.into_iter().take(3).map(|(_, k)| k).collect();
    }
    out
}

fn trend_words(t: &analysis::Trend, d: Dir) -> &'static str {
    if !t.clear() {
        return "no clear trend: the change is smaller than the noise";
    }
    match (d, t.slope < 0.0) {
        (Dir::Lower, true) | (Dir::Higher, false) => "improving",
        (Dir::Lower, false) | (Dir::Higher, true) => "getting worse",
        (Dir::Unknown, true) => "falling",
        (Dir::Unknown, false) => "rising",
    }
}

fn step_s(x: f64) -> String {
    format!("{}", x.round() as i64)
}

/// Status lines for one metric. `target` asks for an ETA.
fn metric_report(rd: &RunData, key: &str, target: Option<f64>, pool: &[(Vec<f64>, Vec<f64>)], horizon: usize) -> Vec<String> {
    let (st, v) = series(rd, key);
    let d = analysis::direction(key);
    let n = v.len();
    let mut out = Vec::new();
    if n == 0 {
        return vec![format!("- {key}: no finite values.")];
    }
    let last = v[n - 1];
    let ib = (0..n).fold(0, |b, i| if d.better(v[i], v[b]) { i } else { b });
    let best = if d == Dir::Unknown {
        String::new()
    } else if ib == n - 1 {
        "; the latest point is the best so far".to_string()
    } else {
        format!("; best {} at step {} ({} points ago)", fmt(v[ib]), step_s(st[ib]), n - 1 - ib)
    };
    out.push(format!("- {key} ({}): last {} at step {}{best}; {n} points.", d.phrase(), fmt(last), step_s(st[n - 1])));
    match analysis::trend(&st, &v) {
        Some(t) => {
            out.push(format!(
                "  Trend over the last {} points (steps {}–{}): {} per {} steps (± {}) → {}.",
                t.n,
                step_s(t.from_step),
                step_s(t.to_step),
                fmt_signed(t.slope * t.per),
                step_s(t.per),
                fmt(t.se * t.per),
                trend_words(&t, d)
            ));
            if n >= analysis::MIN_SHAPE_POINTS {
                let sp = analysis::spikes(&v);
                let sp_s = if sp.is_empty() {
                    "none".to_string()
                } else {
                    sp.iter().take(4).map(|(i, z)| format!("step {} ({}, robust z {})", step_s(st[*i]), fmt(v[*i]), fmt(*z))).collect::<Vec<_>>().join(", ")
                };
                let pl = if analysis::plateau(&v, &t) { "yes, the recent points moved less than 10% of the curve's range" } else { "no" };
                out.push(format!("  Spikes: {sp_s}. Plateau: {pl}."));
            } else {
                out.push(format!("  Spikes/plateau: too few points to judge ({n}, need {}).", analysis::MIN_SHAPE_POINTS));
            }
        }
        None => out.push(format!("  Trend: too few points ({n}, need 3).")),
    }
    // forecast
    match analysis::forecast_points(&st, &v, pool, horizon) {
        None => {
            let why = if n < modgrad_obs_analysis::forecast::MIN_POINTS {
                format!("too few points ({n}, need {})", modgrad_obs_analysis::forecast::MIN_POINTS)
            } else {
                "not enough history to backtest".to_string()
            };
            out.push(format!("  Forecast: none, {why}."));
            if let Some(y) = target {
                out.push(format!("  Target {}: no ETA possible ({why}).", fmt(y)));
            }
        }
        Some(f) => {
            let h = f.steps.len();
            let end = f.steps[h - 1];
            let extra = if f.max_backtest_h < h { format!("; band past {} intervals is extrapolated", f.max_backtest_h) } else { String::new() };
            if f.skill {
                out.push(format!(
                    "  Forecast (ridge; beat last-value on backtest, MAE {} vs {}): step {} → {} (80% band {}–{}{extra}).",
                    fmt(f.mae_test),
                    fmt(f.mae_test_naive),
                    step_s(end),
                    fmt(f.mean[h - 1]),
                    fmt(f.lo[h - 1]),
                    fmt(f.hi[h - 1])
                ));
            } else {
                out.push(format!(
                    "  Forecast: no skill. On backtest the ridge model (MAE {}) did not beat repeating the last value (MAE {}), so the honest forecast is the last value {} (80% band {}–{} by step {}{extra}).",
                    fmt(f.mae_test),
                    fmt(f.mae_test_naive),
                    fmt(last),
                    fmt(f.lo[h - 1]),
                    fmt(f.hi[h - 1]),
                    step_s(end)
                ));
            }
            if let Some(y) = target {
                // reached = some logged point is already at or past the target,
                // seen from where the curve started
                let reached = (0..n).find(|&i| if y <= v[0] { v[i] <= y } else { v[i] >= y });
                let down = y < last;
                // optimistic edge of the band moves towards the target first
                let (opt, pes) = if down { (&f.lo, &f.hi) } else { (&f.hi, &f.lo) };
                let early = analysis::first_reach(&f.steps, opt, last, y);
                let late = analysis::first_reach(&f.steps, pes, last, y);
                let mid = analysis::first_reach(&f.steps, &f.mean, last, y);
                let ys = fmt(y);
                let line = if let Some(i) = reached {
                    format!("Target {ys}: already reached at step {} (last value {}).", step_s(st[i]), fmt(last))
                } else if !f.skill {
                    match early {
                        Some(e) => format!("Target {ys}: no reliable ETA (the forecast has no skill). Only the optimistic edge of the band reaches it, by step {} at the earliest.", step_s(e)),
                        None => format!("Target {ys}: no reliable ETA (the forecast has no skill), and even the 80% band does not reach it by step {}.", step_s(end)),
                    }
                } else {
                    match mid {
                        Some(m) => format!(
                            "Target {ys}: ETA step ~{} (80% band: {} to {}).",
                            step_s(m),
                            early.map(step_s).unwrap_or_else(|| "?".into()),
                            late.map(step_s).unwrap_or_else(|| format!("not by step {}", step_s(end)))
                        ),
                        None => format!(
                            "Target {ys}: not reached by the forecast within step {}{}.",
                            step_s(end),
                            early.map(|e| format!("; the optimistic band edge reaches it by step {}", step_s(e))).unwrap_or_default()
                        ),
                    }
                };
                out.push(format!("  {line}"));
            }
        }
    }
    out
}

/// Runs in the same seed group as `run` (including itself), sorted.
fn seed_group(names: &[String], run: &str) -> Vec<String> {
    let base = analysis::seed_base(run);
    names.iter().filter(|n| analysis::seed_base(n) == base).cloned().collect()
}

pub fn summarize(store: &dyn Store, a: &Args) -> ToolResult {
    let ex = r#"summarize(run="bkk07/small")"#;
    let Some(rq) = a.str(RUN) else {
        return Err(format!("Missing run name. Example: {ex}, or with a target: summarize(run=\"bkk07/small\", metric=\"loss\", target=4.5). list_runs() shows the names."));
    };
    let run = resolve_run(store, &rq)?;
    let rd = load(store, &run)?;
    let keys = rd.metric_keys();
    if keys.is_empty() {
        return Err(format!("Run {run} has no metrics yet. Log some with log(run=\"{run}\", metrics={{\"loss\": 2.5}}, step=1)."));
    }
    let target = match a.get(&["target", "goal", "reach", "target_value", "threshold", "value"]) {
        None => None,
        Some(v) => Some(num(v).ok_or_else(|| format!("target must be a number, got {v}. Example: summarize(run=\"{run}\", metric=\"loss\", target=4.5)"))?),
    };
    let mut note = None;
    let metrics = match a.str(METRIC) {
        Some(mq) => {
            let mut ks = Vec::new();
            for m in mq.split(',').map(str::trim).filter(|s| !s.is_empty()).take(4) {
                let (k, n) = resolve_metric_in(m, &keys, &format!("run {run}"), "summarize", &format!("run=\"{run}\""))?;
                if n.is_some() {
                    note = n;
                }
                ks.push(k);
            }
            ks
        }
        None => key_metrics(&rd),
    };
    let horizon = a.num(&["horizon", "horizon_points", "ahead"]).map(|h| h.clamp(1.0, 64.0) as usize).unwrap_or(if target.is_some() { 64 } else { 8 });
    let info = store.info(&run)?;
    let (st, age) = status_of(&info);
    let names = store.run_names()?;
    let group = seed_group(&names, &run);
    let mut out: Vec<String> = Vec::new();
    out.push(format!(
        "Run {run}: {st} (last row {age}), last step {}, {} metrics.",
        info.last_step.map(|s| s.to_string()).unwrap_or("-".into()),
        keys.len()
    ));
    if let Some(n) = note {
        out.push(format!("({n})"));
    }
    if a.str(METRIC).is_none() {
        out.push(format!("Key metrics: {} (pass metric=\"…\" for another; all: {}{}).", metrics.join(", "), keys.iter().take(12).cloned().collect::<Vec<_>>().join(", "), if keys.len() > 12 { ", …" } else { "" }));
    }
    if target.is_some() && a.str(METRIC).is_none() {
        out.push(format!("(target applies to {})", metrics[0]));
    }
    let others: Vec<RunData> = group.iter().filter(|g| **g != run).filter_map(|g| store.load(g).ok()).collect();
    for (i, k) in metrics.iter().enumerate() {
        let pool: Vec<(Vec<f64>, Vec<f64>)> = others.iter().map(|o| series(o, k)).filter(|(s, _)| s.len() >= 2).collect();
        out.extend(metric_report(&rd, k, if i == 0 { target } else { None }, &pool, horizon));
    }
    if group.len() >= 2 {
        let all: Vec<(String, RunData)> = group.iter().filter_map(|g| if *g == run { Some((g.clone(), rd.clone())) } else { store.load(g).ok().map(|d| (g.clone(), d)) }).collect();
        out.push(format!("Seeds: {} runs in this group ({}).", all.len(), all.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(", ")));
        for k in &metrics {
            let ss: Vec<BTreeMap<u64, f64>> = all.iter().map(|(_, d)| d.series(k).into_iter().filter(|(_, v)| v.is_finite()).collect()).collect();
            let common = ss.iter().skip(1).fold(ss[0].keys().copied().collect::<BTreeSet<u64>>(), |acc, s| acc.intersection(&s.keys().copied().collect()).copied().collect());
            match common.iter().next_back() {
                Some(&cs) => {
                    let vals: Vec<f64> = ss.iter().map(|s| s[&cs]).collect();
                    let (m, sd) = analysis::mean_sd(&vals);
                    out.push(format!("  {k} at step {cs} (latest step all seeds share): {} ± {} (mean ± sd, n={}).", fmt(m), fmt(sd), vals.len()));
                }
                None => out.push(format!("  {k}: the seeds share no step.")),
            }
        }
    }
    out.push(format!(
        "Next: read_metric(run=\"{run}\", metric=\"{}\") for the raw values; summarize(run=\"{run}\", metric=\"{}\", target=…) for an ETA; compare(runs=\"{}\", metric=\"{}\").",
        metrics[0],
        metrics[0],
        if group.len() >= 2 { group.join(",") } else { format!("{run},OTHER") },
        metrics[0]
    ));
    Ok(out.join("\n"))
}

// ---------------------------------------------------------------- compare

fn short_names(runs: &[String]) -> Vec<String> {
    let first = runs[0].rsplit_once('/').map(|(p, _)| format!("{p}/"));
    match first {
        Some(p) if runs.iter().all(|r| r.starts_with(&p)) => runs.iter().map(|r| r[p.len()..].to_string()).collect(),
        _ => runs.to_vec(),
    }
}

pub fn compare(store: &dyn Store, a: &Args) -> ToolResult {
    let ex = r#"compare(runs="bkk07/small,bkk07/mid", metric="brain_heldout_nll")"#;
    let Some(spec) = a.str(&["runs", "run_names", "runnames", "run_list", "names", "run", "ids"]) else {
        return Err(format!("Missing runs: give two or more names, a glob or a group. Example: {ex} or compare(runs=\"bkk07/*\", metric=\"loss\")."));
    };
    let mut runs = resolve_runs(store, &spec)?;
    if runs.len() < 2 {
        let r = runs.first().cloned().unwrap_or_default();
        let sib = seed_group(&store.run_names()?, &r);
        let hint = if sib.len() >= 2 { format!("compare(runs=\"{}\", metric=\"loss\")", sib.join(",")) } else { ex.to_string() };
        return Err(format!("compare needs at least two runs; {spec:?} gave {}. For one run use read_metric(run=\"{r}\", metric=\"loss\"). Example: {hint}", runs.len()));
    }
    let datas: Vec<RunData> = runs.iter().map(|r| load(store, r)).collect::<Result<_, _>>()?;
    let union: Vec<String> = datas.iter().flat_map(|d| d.metric_keys()).collect::<BTreeSet<_>>().into_iter().collect();
    let Some(mq) = a.str(METRIC) else {
        return Err(format!("Missing metric. These runs have: {}. Example: compare(runs=\"{}\", metric=\"{}\").", union.iter().take(20).cloned().collect::<Vec<_>>().join(", "), spec, union.first().map(String::as_str).unwrap_or("loss")));
    };
    let (key, note) = resolve_metric_in(&mq, &union, "these runs", "compare", &format!("runs=\"{spec}\""))?;
    let missing: Vec<String> = runs.iter().zip(&datas).filter(|(_, d)| series(d, &key).1.is_empty()).map(|(r, _)| r.clone()).collect();
    let keep: Vec<usize> = (0..runs.len()).filter(|&i| !missing.contains(&runs[i])).collect();
    if keep.len() < 2 {
        return Err(format!("Only {} of these runs log {key} ({}). Pick runs that have it; list_runs() shows names.", keep.len(), keep.iter().map(|&i| runs[i].as_str()).collect::<Vec<_>>().join(", ")));
    }
    let cap_runs = 8;
    let truncated_runs = keep.len() > cap_runs;
    let keep: Vec<usize> = keep.into_iter().take(cap_runs).collect();
    runs = keep.iter().map(|&i| runs[i].clone()).collect();
    let ser: Vec<BTreeMap<u64, f64>> = keep.iter().map(|&i| datas[i].series(&key).into_iter().filter(|(_, v)| v.is_finite()).collect()).collect();
    let d = analysis::direction(&key);
    let short = short_names(&runs);
    let common: Vec<u64> = ser.iter().skip(1).fold(ser[0].keys().copied().collect::<BTreeSet<u64>>(), |acc, s| acc.intersection(&s.keys().copied().collect()).copied().collect()).into_iter().collect();
    let mut out: Vec<String> = Vec::new();
    if let Some(n) = note {
        out.push(format!("({n})"));
    }
    out.push(format!("{key} ({}) across {} runs{}:", d.phrase(), runs.len(), if truncated_runs { format!(" (first {cap_runs} shown)") } else { String::new() }));
    let (steps, label): (Vec<u64>, &str) = if common.is_empty() {
        (ser.iter().flat_map(|s| s.keys().copied()).collect::<BTreeSet<_>>().into_iter().collect(), "no step is shared by all runs; '-' = not logged")
    } else {
        (common.clone(), "steps all runs share")
    };
    let cap = 12;
    let shown = &steps[steps.len().saturating_sub(cap)..];
    let mut header = vec!["step".to_string()];
    header.extend(short.iter().cloned());
    if runs.len() == 2 {
        header.push(format!("{}-{}", short[1], short[0]));
    } else {
        header.push("best".into());
    }
    let rows: Vec<Vec<String>> = shown
        .iter()
        .map(|s| {
            let vals: Vec<Option<f64>> = ser.iter().map(|m| m.get(s).copied()).collect();
            let mut r = vec![s.to_string()];
            r.extend(vals.iter().map(|v| v.map(fmt).unwrap_or("-".into())));
            if runs.len() == 2 {
                r.push(match (vals[0], vals[1]) {
                    (Some(x), Some(y)) => fmt_signed(y - x),
                    _ => "-".into(),
                });
            } else {
                let b = (0..vals.len()).filter(|&i| vals[i].is_some()).fold(None, |b: Option<usize>, i| match b {
                    Some(j) if !d.better(vals[i].unwrap(), vals[j].unwrap()) => Some(j),
                    _ => Some(i),
                });
                r.push(b.map(|i| short[i].clone()).unwrap_or("-".into()));
            }
            r
        })
        .collect();
    out.push(format!("({label}{})", if steps.len() > cap { format!("; last {cap} of {}", steps.len()) } else { String::new() }));
    out.push(table(&header, &rows));
    let pick_best = |vals: &[(usize, f64)]| -> usize { vals.iter().fold(vals[0], |b, &x| if d.better(x.1, b.1) { x } else { b }).0 };
    // each run's own latest value
    let lasts: Vec<(usize, f64)> = ser.iter().enumerate().map(|(i, m)| (i, *m.values().last().unwrap())).collect();
    let bl = pick_best(&lasts);
    let lstep = |i: usize| *ser[i].keys().last().unwrap();
    let qual = if d == Dir::Higher { "highest" } else { "lowest" };
    out.push(format!(
        "Latest value of each run (its own last step): {qual} is {} = {} at step {}; {}.",
        runs[bl],
        fmt(lasts[bl].1),
        lstep(bl),
        lasts.iter().filter(|(i, _)| *i != bl).map(|(i, v)| format!("{} {} at step {}", runs[*i], fmt(*v), lstep(*i))).collect::<Vec<_>>().join(", ")
    ));
    if let Some(&cs) = common.last() {
        let at: Vec<(usize, f64)> = ser.iter().enumerate().map(|(i, m)| (i, m[&cs])).collect();
        let b = pick_best(&at);
        out.push(format!(
            "At the latest step all runs share ({cs}): {qual} is {} = {}; others {}.",
            runs[b],
            fmt(at[b].1),
            at.iter().filter(|(i, _)| *i != b).map(|(i, v)| format!("{} {} ({})", short[*i], fmt(*v), fmt_signed(v - at[b].1))).collect::<Vec<_>>().join(", ")
        ));
    }
    let bests: Vec<(usize, f64)> = ser.iter().enumerate().map(|(i, m)| (i, m.values().copied().fold(m.values().next().copied().unwrap(), |a, x| if d.better(x, a) { x } else { a }))).collect();
    let bb = pick_best(&bests);
    if d != Dir::Unknown && bb != bl {
        out.push(format!("Best value ever: {} = {}.", runs[bb], fmt(bests[bb].1)));
    }
    if !missing.is_empty() {
        out.push(format!("Not compared (no {key}): {}.", missing.join(", ")));
    }
    if d == Dir::Unknown {
        out.push(format!("({key}: direction unknown from its name; 'lowest'/'best' assume lower is better.)"));
    }
    out.push(format!("Next: make_board(query='{key}{{run=~\"{}\"}}') to chart this, or summarize(run=\"{}\") for one run's trend.", runs.join("|"), runs[bl]));
    Ok(out.join("\n"))
}

// ------------------------------------------------------------- make_board

struct Panel {
    metric: String,
    runs: Vec<String>,
}

/// `metric{run=~"a|b*"}`, `metric{run="x"}`, `metric run=~a|b`, or `metric`.
fn parse_panel(store: &dyn Store, q: &str) -> Result<Panel, String> {
    let q = q.trim();
    let (mpart, sel) = match q.find('{') {
        Some(i) => (q[..i].trim(), Some(q[i + 1..].trim_end().trim_end_matches('}').to_string())),
        None => match q.find(" run") {
            Some(i) => (q[..i].trim(), Some(q[i..].trim().to_string())),
            None => (q, None),
        },
    };
    let mpart = strip_quotes(mpart);
    let all = store.run_names()?;
    let runs: Vec<String> = match sel {
        None => all.clone(),
        Some(s) => {
            // run=~"a|b", run="a", runs=~'a', or a bare pattern
            let s = s.trim();
            let val = match s.find(['=', '~']) {
                Some(i) => s[i..].trim_start_matches(['=', '~', '!']).trim(),
                None => s,
            };
            let val = strip_quotes(val).trim_start_matches('^').trim_end_matches('$');
            let mut v = Vec::new();
            for alt in val.split('|').map(strip_quotes).filter(|x| !x.is_empty()) {
                let alt = alt.replace(".*", "*").replace(".+", "*");
                for r in resolve_runs(store, &alt)? {
                    if !v.contains(&r) {
                        v.push(r);
                    }
                }
            }
            v
        }
    };
    if mpart.is_empty() {
        return Err("The query has no metric name. Example: make_board(query='loss{run=~\"bkk07/*\"}').".into());
    }
    let datas: Vec<(String, RunData)> = runs.iter().filter_map(|r| store.load(r).ok().map(|d| (r.clone(), d))).collect();
    let union: Vec<String> = datas.iter().flat_map(|(_, d)| d.metric_keys()).collect::<BTreeSet<_>>().into_iter().collect();
    let metric = match resolve_metric(mpart, &union) {
        Resolved::One { key, .. } => key,
        Resolved::Many(v) => return Err(format!("Metric {mpart:?} matches several metrics: {}. Pick one, e.g. make_board(query='{}{{run=~\"…\"}}').", v.iter().take(10).cloned().collect::<Vec<_>>().join(", "), v[0])),
        Resolved::None(v) => return Err(format!("No metric {mpart:?} in the selected runs. Closest: {}. Retry e.g. make_board(query='{}{{run=~\"…\"}}').", v.join(", "), v.first().map(String::as_str).unwrap_or("loss"))),
    };
    let runs: Vec<String> = datas.into_iter().filter(|(_, d)| d.metric_keys().contains(&metric)).map(|(r, _)| r).collect();
    if runs.is_empty() {
        return Err(format!("No selected run logs {metric}. Example: make_board(query='{metric}{{run=~\"bkk07/*\"}}')."));
    }
    Ok(Panel { metric, runs })
}

fn fnv(s: &str) -> u64 {
    s.bytes().fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3))
}

pub fn make_board(store: &dyn Store, a: &Args) -> ToolResult {
    let ex = r#"make_board(query='brain_heldout_nll{run=~"bkk07/small|bkk07/mid"}')"#;
    let Some(q) = a.str(&["query", "queries", "q", "board", "promql", "expr", "metric"]) else {
        return Err(format!("Missing query. Example: {ex}. Several panels: separate queries with ';'."));
    };
    // a weak model may pass runs separately: make_board(query="loss", runs="a,b")
    let extra_runs = a.str(&["runs", "run"]);
    let mut panels = Vec::new();
    for part in q.split([';', '\n']).map(str::trim).filter(|s| !s.is_empty()).take(8) {
        let part = match (&extra_runs, part.contains('{')) {
            (Some(r), false) => format!("{part}{{run=~\"{}\"}}", r.replace([',', ' '], "|")),
            _ => part.to_string(),
        };
        panels.push(parse_panel(store, &part)?);
    }
    if panels.is_empty() {
        return Err(format!("Empty query. Example: {ex}"));
    }
    let canon: Vec<String> = panels.iter().map(|p| format!("{}{{run=~\"{}\"}}", p.metric, p.runs.join("|"))).collect();
    let title = canon.join("; ");
    let mut slug = norm_key(&format!("{} {}", panels[0].metric, short_names(&panels[0].runs).join(" ")));
    slug.truncate(48);
    let slug = format!("{}-{:06x}", slug.trim_end_matches('_'), fnv(&title) & 0xffffff);
    let board = json!({
        "version": 0,
        "title": title,
        "query": q,
        "panels": panels.iter().zip(&canon).map(|(p, c)| json!({"query": c, "metric": p.metric, "runs": p.runs})).collect::<Vec<_>>(),
        "created_unix": crate::now_unix(),
    });
    let (path, url) = store.save_board(&slug, &board)?;
    let mut out = format!("Board saved with {} panel{}:", panels.len(), if panels.len() == 1 { "" } else { "s" });
    for p in &panels {
        out.push_str(&format!("\n- {} for {} run{}: {}", p.metric, p.runs.len(), if p.runs.len() == 1 { "" } else { "s" }, p.runs.iter().take(10).cloned().collect::<Vec<_>>().join(", ")));
    }
    out.push_str(&format!("\nURL: {url}\nFile: {path}"));
    out.push_str("\nGive the URL to the user. Next: add panels with make_board(query='metric_a{run=~\"…\"}; metric_b{run=~\"…\"}'), or compare(runs=\"…\", metric=\"…\") for the numbers.");
    Ok(out)
}
