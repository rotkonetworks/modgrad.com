//! Back-compat: convert a `ctmlm_pilot` stdout log (`log.txt`) into a run
//! directory in the modgrad-obs format, by replaying it through [`Run`] so
//! the two paths cannot drift apart.
//!
//! Recognised line kinds (everything else is counted as unparsed and
//! listed in the summary under `legacy_unparsed`):
//!
//! * header `CTM-LM PILOT: <model>, [persistent gated slots B, ]tap L, read heads [..], M m; C c, n n, facts f, gap (a, b), W w, K (a, b), batch b, <replay|natural/replay> P%, brain N + read M params` → config
//! * `filler: …`, `natural text: N train / M held-out paragraphs` → config
//! * `NONE (fixed): in-dist A% nll X, held-out B% nll Y` → step 0 metrics
//! * `step N: train loss (ema) L; BRAIN in-dist A% nll X | held-out B% nll Y  [S s/step, M min, RSS K kB]` → metrics
//! * `  natural text (…, D docs × C chunks): BRAIN m ± s (sem e) | NONE … | FULL … | BRAIN−NONE … | FULL−NONE …` → metrics at the last step (or `final_*` summary after `FINAL natural text`)
//! * `    chunk j: BRAIN … | NONE … | FULL … | BRAIN−NONE …` → per-chunk arrays at the last step
//! * `FINAL (…)` block: `all gaps …` and `metamemory ARM: ECE …, AUROC …, mean conf …; [lo,hi) acc P% n K; …` → summary + reliability arrays; `gap g (n= K) BRAIN … | NONE … | FULL …` → per-gap arrays
//! * `LM weights hash-identical: yes; peak RSS K kB` → summary
//! * `LlmWeights …`, `hf_llama …`, `loaded checkpoint …` → events

use crate::Meta;
use crate::sink::{OpenSink, RunAt, Sink};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::io;
use std::path::Path;

/// Parse result: what was recognised and what was not.
#[derive(Debug, Default)]
pub struct LegacyReport {
    pub lines: usize,
    pub steps: usize,
    pub last_step: u64,
    pub unparsed: Vec<String>,
}

fn f(s: &str) -> Option<f64> {
    s.trim().trim_end_matches('%').trim_start_matches('+').parse().ok()
}

/// `"  5.0% nll 7.47"` → (5.0, 7.47)
fn acc_nll(s: &str) -> Option<(f64, f64)> {
    let s = s.trim();
    let (a, rest) = s.split_once('%')?;
    let n = rest.trim().strip_prefix("nll")?;
    Some((f(a)?, f(n.split_whitespace().next()?)?))
}

/// `"+3.5125 ± 0.7164 (sem 0.0414)"` → (mean, std, sem)
fn mean_std_sem(s: &str) -> Option<(f64, f64, f64)> {
    let (m, rest) = s.trim().split_once('±')?;
    let (sd, rest) = rest.split_once('(')?;
    let sem = rest.trim().strip_prefix("sem")?.trim().trim_end_matches(')');
    Some((f(m)?, f(sd)?, f(sem)?))
}

/// ASCII key for a natural-text label: `BRAIN−NONE` → `brain_minus_none`.
fn label_key(l: &str) -> String {
    l.trim().replace('−', "_minus_").replace('-', "_minus_").to_lowercase()
}

/// `"BRAIN x | NONE y"` → [("brain", "x"), ("none", "y")]
fn labelled(s: &str) -> Vec<(String, String)> {
    s.split(" | ")
        .filter_map(|seg| {
            let seg = seg.trim();
            let (l, v) = seg.split_once(' ')?;
            Some((label_key(l), v.trim().to_string()))
        })
        .collect()
}

/// One `ctmlm_pilot` progress line,
/// `step N: train loss (ema) L; BRAIN in-dist A% nll X | held-out B% nll Y  [S s/step, M min, RSS K kB]`,
/// as `(step, [(key, value)], elapsed minutes)`. The `NONE (fixed): …`
/// baseline line comes back as step 0. Shared by [`convert`] and the MCP
/// `ingest_log` tool so both produce the same metric keys.
pub fn parse_step_line(l: &str) -> Option<(u64, Vec<(&'static str, f64)>, Option<f64>)> {
    let l = l.trim();
    if let Some(rest) = l.strip_prefix("NONE (fixed):") {
        let (a, b) = rest.split_once(", held-out")?;
        let (ia, inl) = acc_nll(a.trim().strip_prefix("in-dist")?)?;
        let (ha, hn) = acc_nll(b)?;
        return Some((0, vec![("none_indist_acc_pct", ia), ("none_indist_nll", inl), ("none_heldout_acc_pct", ha), ("none_heldout_nll", hn)], None));
    }
    let rest = l.strip_prefix("step ")?;
    let (n, rest) = rest.split_once(':')?;
    let step: u64 = n.trim().parse().ok()?;
    let (loss, rest) = rest.trim().strip_prefix("train loss (ema)")?.split_once(';')?;
    let (main, bracket) = rest.split_once('[').unwrap_or((rest, ""));
    let (ind, held) = main.split_once('|')?;
    let (ia, inl) = acc_nll(ind.trim().strip_prefix("BRAIN")?.trim().strip_prefix("in-dist")?)?;
    let (ha, hn) = acc_nll(held.trim().strip_prefix("held-out")?)?;
    let mut kv = vec![("train_loss_ema", f(loss)?), ("brain_indist_acc_pct", ia), ("brain_indist_nll", inl), ("brain_heldout_acc_pct", ha), ("brain_heldout_nll", hn)];
    let b: Vec<&str> = bracket.trim_end_matches(']').split(',').map(|x| x.trim()).collect();
    let mut minutes = None;
    for x in &b {
        if let Some(v) = x.strip_suffix("s/step") {
            kv.push(("s_per_step", f(v)?));
        } else if let Some(v) = x.strip_suffix("min") {
            minutes = f(v);
            kv.push(("elapsed_min", minutes?));
        } else if let Some(v) = x.strip_prefix("RSS").and_then(|v| v.trim().strip_suffix("kB")) {
            kv.push(("rss_kb", f(v)?));
        }
    }
    Some((step, kv, minutes))
}

fn parse_header(h: &str, cfg: &mut Map<String, Value>) -> bool {
    let Some(body) = h.strip_prefix("CTM-LM PILOT:") else { return false };
    let (a, b) = body.split_once(';').unwrap_or((body, ""));
    // split on ", " but not inside [...] or (...)
    let split = |s: &str| -> Vec<String> {
        let (mut out, mut cur, mut depth) = (Vec::new(), String::new(), 0i32);
        for ch in s.chars() {
            match ch {
                '[' | '(' => depth += 1,
                ']' | ')' => depth -= 1,
                _ => {}
            }
            if ch == ',' && depth == 0 {
                out.push(cur.trim().to_string());
                cur.clear();
            } else {
                cur.push(ch);
            }
        }
        if !cur.trim().is_empty() {
            out.push(cur.trim().to_string());
        }
        out
    };
    let pair = |s: &str| -> Option<(f64, f64)> {
        let s = s.trim().trim_start_matches('(').trim_end_matches(')');
        let (x, y) = s.split_once(',')?;
        Some((f(x)?, f(y)?))
    };
    let mut parts = split(a);
    if !parts.is_empty() {
        cfg.insert("model".into(), json!(parts.remove(0)));
    }
    for p in parts.into_iter().chain(split(b)) {
        if let Some(v) = p.strip_prefix("persistent gated slots ") {
            cfg.insert("persistent".into(), json!(v.trim() == "true"));
        } else if let Some(v) = p.strip_prefix("tap ") {
            cfg.insert("tap".into(), json!(f(v)));
        } else if let Some(v) = p.strip_prefix("read heads ") {
            let hs: Vec<f64> = v.trim_matches(|c| c == '[' || c == ']').split(',').filter_map(f).collect();
            cfg.insert("read_heads".into(), json!(hs));
        } else if let Some((lo, hi)) = p.strip_prefix("gap ").and_then(pair) {
            cfg.insert("gap_lo".into(), json!(lo));
            cfg.insert("gap_hi".into(), json!(hi));
        } else if let Some((lo, hi)) = p.strip_prefix("K ").and_then(pair) {
            cfg.insert("ticks_lo".into(), json!(lo));
            cfg.insert("ticks_hi".into(), json!(hi));
        } else if let Some(v) = p.strip_prefix("natural/replay ") {
            cfg.insert("natural_replay_pct".into(), json!(f(v)));
        } else if let Some(v) = p.strip_prefix("replay ") {
            cfg.insert("replay_pct".into(), json!(f(v)));
        } else if let Some(v) = p.strip_prefix("brain ") {
            // "494053 + read 983044 params"
            let w: Vec<&str> = v.split_whitespace().collect();
            if let Some(n) = w.first().and_then(|x| f(x)) {
                cfg.insert("brain_params".into(), json!(n));
            }
            if let Some(i) = w.iter().position(|x| *x == "read") {
                if let Some(n) = w.get(i + 1).and_then(|x| f(x)) {
                    cfg.insert("read_params".into(), json!(n));
                }
            }
        } else if let Some((k, v)) = p.split_once(' ') {
            let key = match k {
                "M" => "slots_m",
                "C" => "c",
                "n" => "n",
                "facts" => "facts",
                "W" => "window",
                "batch" => "batch",
                other => other,
            };
            cfg.insert(key.into(), f(v).map(|x| json!(x)).unwrap_or_else(|| json!(v)));
        }
    }
    true
}

#[derive(Default)]
struct State {
    step: u64,
    t: f64,
    cfg: Map<String, Value>,
    summary: Map<String, Value>,
    in_final_nat: bool,
    in_final_acc: bool,
    prefix: String,
    /// per-chunk label → values
    chunks: BTreeMap<String, Vec<f32>>,
    /// (prefix, arm) → (gaps, acc, nll, n)
    gaps: BTreeMap<(String, String), (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)>,
    reliability: BTreeMap<(String, String), Vec<f32>>,
}

/// Convert `log` text into a run at `out_dir` (overwriting any previous
/// conversion there). `source` is recorded as provenance; `name` is the
/// run name (normally the directory name). `start_unix` is the best
/// estimate of the wall-clock start (e.g. log mtime minus elapsed).
pub fn convert(text: &str, out_dir: &Path, name: &str, source: &str, log_path: &str, mtime_unix: f64) -> io::Result<LegacyReport> {
    for f in ["metrics.jsonl", "arrays.jsonl", "events.jsonl", "summary.json"] {
        let _ = std::fs::remove_file(out_dir.join(f));
    }
    convert_into(text, RunAt(out_dir), name, source, log_path, mtime_unix).map(|(rep, ())| rep)
}

/// Cheap check: does `text` look like a `ctmlm_pilot` log?
pub fn looks_like(text: &str) -> bool {
    text.lines().take(40).any(|l| l.trim_start().starts_with("CTM-LM PILOT:"))
        || text.lines().take(40).any(|l| l.trim_start().starts_with("step ") && l.contains("train loss (ema)") && l.contains("BRAIN"))
}

/// As [`convert`], into any [`Sink`] (e.g. an in-memory run in the browser).
pub fn convert_into<O: OpenSink>(text: &str, open: O, name: &str, source: &str, log_path: &str, mtime_unix: f64) -> io::Result<(LegacyReport, <O::Sink as Sink>::Output)> {
    // First pass for config + elapsed (config.json is written at create).
    let mut st = State::default();
    let mut last_min = 0.0;
    for line in text.lines() {
        let l = line.trim();
        if parse_header(l, &mut st.cfg) {
            continue;
        }
        if let Some(rest) = l.strip_prefix("filler: ") {
            st.cfg.insert("filler".into(), json!(rest));
        } else if let Some(rest) = l.strip_prefix("natural text: ") {
            let w: Vec<&str> = rest.split_whitespace().collect();
            if let (Some(a), Some(b)) = (w.first().and_then(|x| f(x)), w.get(3).and_then(|x| f(x))) {
                st.cfg.insert("text_train_paragraphs".into(), json!(a));
                st.cfg.insert("text_heldout_paragraphs".into(), json!(b));
            }
        } else if let Some(rest) = l.strip_prefix("loaded checkpoint ") {
            st.cfg.insert("loaded_from".into(), json!(rest));
        } else if l.starts_with("step ") {
            if let Some(m) = l.split('[').nth(1).and_then(|b| b.split(',').nth(1)).and_then(|x| f(x.trim().trim_end_matches("min"))) {
                last_min = m;
            }
        }
    }
    let meta = Meta {
        project: Some("ctmlm".into()),
        git: None,
        hostname: Some(source.to_string()),
        start_unix: mtime_unix - last_min * 60.0,
        extra: Some(json!({ "legacy_log": log_path, "source": source, "converter": "modgrad-obs legacy ctmlm_pilot" })),
    };
    let cfg = Value::Object(std::mem::take(&mut st.cfg));
    let mut run = open.open(name, &cfg, meta)?;
    let mut rep = LegacyReport::default();
    let err = |e: crate::ObsError| io::Error::other(e.to_string());

    for line in text.lines() {
        rep.lines += 1;
        let l = line.trim();
        if l.is_empty() {
            continue;
        }
        let ok = (|| -> Option<()> {
            if l.starts_with("CTM-LM PILOT:") || l.starts_with("filler: ") || l.starts_with("natural text: ") {
                return Some(());
            }
            if l.starts_with("LlmWeights") || l.starts_with("hf_llama") || l.starts_with("loaded checkpoint") {
                run.event_with_time(st.step, st.t, l).ok()?;
                return Some(());
            }
            if let Some(rest) = l.strip_prefix("NONE (fixed):") {
                let (a, b) = rest.split_once(", held-out")?;
                let (ia, inl) = acc_nll(a.trim().strip_prefix("in-dist")?)?;
                let (ha, hn) = acc_nll(b)?;
                run.log_with_time(0, 0.0, &[("none_indist_acc_pct", ia), ("none_indist_nll", inl), ("none_heldout_acc_pct", ha), ("none_heldout_nll", hn)]).ok()?;
                return Some(());
            }
            if let Some((step, kv, minutes)) = parse_step_line(l) {
                let kv: Vec<(&str, f64)> = kv.iter().map(|(k, v)| (*k, *v)).collect();
                st.step = step;
                st.t = minutes.map(|m| m * 60.0).unwrap_or(st.t);
                run.log_with_time(step, st.t, &kv).ok()?;
                rep.steps += 1;
                rep.last_step = step;
                return Some(());
            }
            if l.starts_with("FINAL natural text") {
                st.in_final_nat = true;
                return Some(());
            }
            if l.starts_with("FINAL (") {
                st.in_final_nat = false;
                st.in_final_acc = true;
                return Some(());
            }
            if let Some(rest) = l.strip_prefix("natural text (") {
                let (desc, vals) = rest.split_once("):")?;
                // "... 60 docs × 5 chunks"
                let w: Vec<&str> = desc.split_whitespace().collect();
                if let Some(i) = w.iter().position(|x| x.starts_with("docs")) {
                    if let Some(d) = w.get(i.wrapping_sub(1)).and_then(|x| f(x.trim_start_matches(','))) {
                        st.summary.insert("nat_docs".into(), json!(d));
                    }
                }
                let mut kv: Vec<(String, f64)> = Vec::new();
                for (lab, v) in labelled(vals) {
                    let (m, sd, sem) = mean_std_sem(&v)?;
                    let base = if st.in_final_nat { format!("final_nat_{lab}") } else { format!("nat_{lab}") };
                    kv.push((base.clone(), m));
                    kv.push((format!("{base}_std"), sd));
                    kv.push((format!("{base}_sem"), sem));
                }
                if st.in_final_nat {
                    for (k, v) in kv {
                        st.summary.insert(k, json!(v));
                    }
                } else {
                    let r: Vec<(&str, f64)> = kv.iter().map(|(k, v)| (k.as_str(), *v)).collect();
                    run.log_with_time(st.step, st.t, &r).ok()?;
                }
                return Some(());
            }
            if let Some(rest) = l.strip_prefix("chunk ") {
                let (_, vals) = rest.split_once(':')?;
                for (lab, v) in labelled(vals) {
                    let (m, _, _) = mean_std_sem(&v)?;
                    st.chunks.entry(lab).or_default().push(m as f32);
                }
                return Some(());
            }
            if st.in_final_acc && l.ends_with(':') && !l.contains('|') {
                let lower = l.to_lowercase();
                st.prefix = if lower.starts_with("held-out") { "heldout".into() } else if lower.starts_with("in-dist") { "indist".into() } else { label_key(l.trim_end_matches(':')).replace(' ', "_") };
                return Some(());
            }
            if let Some(rest) = l.strip_prefix("all gaps") {
                for (arm, v) in labelled(rest) {
                    let (a, n) = acc_nll(&v)?;
                    st.summary.insert(format!("final_{}_{arm}_acc_pct", st.prefix), json!(a));
                    st.summary.insert(format!("final_{}_{arm}_nll", st.prefix), json!(n));
                }
                return Some(());
            }
            if let Some(rest) = l.strip_prefix("metamemory") {
                let (arm, body) = rest.split_once(':')?;
                let arm = label_key(arm);
                let p = format!("final_{}_{arm}", st.prefix);
                let body = body.trim();
                if body == "-" {
                    return Some(());
                }
                let (head, table) = body.split_once(';').unwrap_or((body, ""));
                for part in head.split(',') {
                    let part = part.trim();
                    for (lab, key) in [("ECE ", "ece"), ("AUROC ", "auroc"), ("mean conf ", "mean_conf")] {
                        if let Some(v) = part.strip_prefix(lab) {
                            st.summary.insert(format!("{p}_{key}"), json!(f(v)));
                        }
                    }
                }
                let mut rel = vec![f32::NAN; 10];
                for bin in table.split(';') {
                    // "[0.1,0.2) acc 12% n 34"
                    let bin = bin.trim();
                    if bin.is_empty() {
                        continue;
                    }
                    let lo = f(bin.trim_start_matches('[').split(',').next()?)?;
                    let acc = f(bin.split("acc").nth(1)?.split_whitespace().next()?)?;
                    rel[((lo * 10.0).round() as usize).min(9)] = acc as f32;
                }
                st.reliability.insert((st.prefix.clone(), arm), rel);
                return Some(());
            }
            if let Some(rest) = l.strip_prefix("gap ") {
                // "2 (n= 49) BRAIN ... | NONE ... | FULL ..."
                let (g, rest) = rest.split_once('(')?;
                let g = f(g)?;
                let (nstr, vals) = rest.split_once(')')?;
                let n = f(nstr.trim().trim_start_matches("n=").trim())?;
                for (arm, v) in labelled(vals) {
                    let (a, nl) = acc_nll(&v)?;
                    let e = st.gaps.entry((st.prefix.clone(), arm)).or_default();
                    e.0.push(g as f32);
                    e.1.push(a as f32);
                    e.2.push(nl as f32);
                    e.3.push(n as f32);
                }
                return Some(());
            }
            if let Some(rest) = l.strip_prefix("LM weights hash-identical:") {
                let (yes, rss) = rest.split_once(';').unwrap_or((rest, ""));
                st.summary.insert("lm_hash_identical".into(), json!(yes.trim() == "yes"));
                if let Some(v) = rss.trim().strip_prefix("peak RSS").and_then(|v| v.trim().strip_suffix("kB")) {
                    st.summary.insert("peak_rss_kb".into(), json!(f(v)));
                }
                return Some(());
            }
            None
        })();
        if ok.is_none() {
            rep.unparsed.push(line.to_string());
        }
    }

    let fs = rep.last_step;
    let t = st.t;
    for (lab, v) in &st.chunks {
        run.log_array_with_time(fs, t, &format!("final_nat_chunk_{lab}"), &[v.len()], v).map_err(err)?;
    }
    let mut gap_labels: Option<Vec<f32>> = None;
    for ((p, arm), (g, acc, nll, _n)) in &st.gaps {
        run.log_array_with_time(fs, t, &format!("final_{p}_gap_acc_{arm}"), &[acc.len()], acc).map_err(err)?;
        run.log_array_with_time(fs, t, &format!("final_{p}_gap_nll_{arm}"), &[nll.len()], nll).map_err(err)?;
        gap_labels.get_or_insert_with(|| g.clone());
    }
    for ((p, arm), rel) in &st.reliability {
        run.log_array_with_time(fs, t, &format!("final_{p}_reliability_{arm}"), &[10], rel).map_err(err)?;
    }
    if let Some(g) = gap_labels {
        st.summary.insert("gap_labels".into(), json!(g));
    }
    st.summary.insert("legacy_lines".into(), json!(rep.lines));
    st.summary.insert("legacy_unparsed".into(), json!(rep.unparsed));
    let finished = st.summary.contains_key("lm_hash_identical");
    let summary = Value::Object(std::mem::take(&mut st.summary));
    let out = if finished {
        run.finish_with_time(&summary, t)?
    } else {
        // still running (or died): write the summary but don't mark finished
        for (k, v) in summary.as_object().unwrap() {
            run.summary_value(k, v)?;
        }
        run.close()?
    };
    Ok((rep, out))
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOG: &str = "filler: natural text from /root/ctmlm/wikitext (65229 train / 7996 held-out sentences, disjoint)
natural text: 17556 train / 2183 held-out paragraphs
LlmWeights (safetensors): loaded 290 tensors, 1308 MB host RAM
CTM-LM PILOT: SmolLM2-360M, persistent gated slots true, tap 22, read heads [9, 12, 14, 4], M 16; C 32, n 6, facts 2, gap (2, 4), W 5, K (8, 16), batch 2, natural/replay 25%, brain 58911157 + read 983044 params
NONE (fixed): in-dist   5.0% nll 7.49, held-out   7.5% nll 8.99
step     1: train loss (ema) 4.386; BRAIN in-dist   5.0% nll 7.48 | held-out   7.5% nll 8.99  [6.09 s/step, 1 min, RSS 3071268 kB]
  natural text (held-out contiguous wikitext, 60 docs × 5 chunks): BRAIN +3.5125 ± 0.7164 (sem 0.0414) | NONE +3.5129 ± 0.7168 (sem 0.0414) | FULL +2.3985 ± 0.7032 (sem 0.0406) | BRAIN−NONE -0.0004 ± 0.0023 (sem 0.0001) | FULL−NONE -1.1145 ± 0.4549 (sem 0.0263)
step   250: train loss (ema) 3.614; BRAIN in-dist   0.0% nll 4.24 | held-out   5.0% nll 5.43  [1.95 s/step, 10 min, RSS 2392584 kB]

FINAL natural text (mean chunk NLL, ± std over documents):
  natural text (held-out contiguous wikitext, 60 docs × 2 chunks): BRAIN +3.5 ± 0.7 (sem 0.04) | NONE +3.6 ± 0.7 (sem 0.04) | FULL +2.4 ± 0.7 (sem 0.04) | BRAIN−NONE -0.1 ± 0.002 (sem 0.0001) | FULL−NONE -1.1 ± 0.45 (sem 0.02)
    chunk 1: BRAIN +3.4 ± 0.7 (sem 0.04) | NONE +3.5 ± 0.7 (sem 0.04) | FULL +2.4 ± 0.7 (sem 0.04) | BRAIN−NONE -0.1 ± 0.002 (sem 0.0001)
    chunk 2: BRAIN +3.6 ± 0.7 (sem 0.04) | NONE +3.7 ± 0.7 (sem 0.04) | FULL +2.4 ± 0.7 (sem 0.04) | BRAIN−NONE -0.1 ± 0.002 (sem 0.0001)

FINAL (150 documents per set; 10-way accuracy, chance 10%; answer NLL per token)
held-out relations and names:
  all gaps   BRAIN  10.7% nll 4.99 | NONE   8.7% nll 8.26 | FULL  45.3% nll 5.25
  metamemory BRAIN: ECE 0.123, AUROC 0.611, mean conf 0.200; [0.1,0.2) acc 10% n 100; [0.2,0.3) acc 15% n 50
  metamemory NONE : -
  gap 2 (n= 49) BRAIN  12.2% nll 4.61 | NONE  10.2% nll 7.79 | FULL  49.0% nll 4.60
  gap 3 (n= 46) BRAIN  15.2% nll 4.69 | NONE  10.9% nll 7.46 | FULL  43.5% nll 4.72
in-distribution (fresh docs):
  all gaps   BRAIN   8.0% nll 3.07 | NONE  12.0% nll 7.57 | FULL  54.7% nll 4.76
  gap 2 (n= 50) BRAIN   8.0% nll 3.37 | NONE  14.0% nll 7.38 | FULL  60.0% nll 4.46
something unexpected here
LM weights hash-identical: yes; peak RSS 3474780 kB
";

    #[test]
    fn converts_pilot_log() {
        let d = std::env::temp_dir().join(format!("modgrad-obs-legacy-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        let rep = convert(LOG, &d, "large", "bkk07", "/root/ctmlm/runs/large/log.txt", 1e9).unwrap();
        assert_eq!(rep.unparsed, vec!["something unexpected here".to_string()]);
        assert_eq!(rep.steps, 2);
        let rd = crate::read_run(&d).unwrap();
        assert_eq!(rd.skipped, 0);
        let c = &rd.config["config"];
        assert_eq!(c["brain_params"], 58911157.0);
        assert_eq!(c["read_params"], 983044.0);
        assert_eq!(c["persistent"], true);
        assert_eq!(c["gap_hi"], 4.0);
        assert_eq!(c["model"], "SmolLM2-360M");
        assert_eq!(rd.series("brain_heldout_nll"), vec![(1, 8.99), (250, 5.43)]);
        assert_eq!(rd.series("none_heldout_nll"), vec![(0, 8.99)]);
        assert_eq!(rd.series("nat_brain_minus_none"), vec![(1, -0.0004)]);
        assert_eq!(rd.series("rss_kb")[1].1, 2392584.0);
        assert_eq!(rd.array_series("final_heldout_gap_acc_brain")[0].data, vec![12.2, 15.2]);
        assert_eq!(rd.array_series("final_heldout_gap_acc_brain")[0].step, 250);
        assert_eq!(rd.array_series("final_nat_chunk_brain")[0].data, vec![3.4, 3.6]);
        let rel = &rd.array_series("final_heldout_reliability_brain")[0].data;
        assert_eq!(rel[1], 10.0);
        assert!(rel[0].is_nan());
        let s = &rd.summary.as_ref().unwrap()["summary"];
        assert_eq!(s["final_indist_full_acc_pct"], 54.7);
        assert_eq!(s["final_heldout_brain_ece"], 0.123);
        assert_eq!(s["final_nat_brain_minus_none"], -0.1);
        assert_eq!(s["peak_rss_kb"], 3474780.0);
        assert_eq!(rd.summary.as_ref().unwrap()["finished"], true);
        assert_eq!(rd.events.len(), 1);
    }

    #[test]
    fn header_without_persistent() {
        let mut m = Map::new();
        assert!(parse_header("CTM-LM PILOT: SmolLM2-360M, tap 22, read heads [9, 12, 14, 4], M 16; C 32, n 6, facts 2, gap (2, 4), W 5, K (8, 16), batch 2, replay 25%, brain 494053 + read 983044 params", &mut m));
        assert_eq!(m["brain_params"], 494053.0);
        assert_eq!(m["replay_pct"], 25.0);
        assert_eq!(m["window"], 5.0);
        assert_eq!(m["read_heads"], json!([9.0, 12.0, 14.0, 4.0]));
        assert!(!m.contains_key("persistent"));
    }
}
