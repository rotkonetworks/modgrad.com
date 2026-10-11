//! Forgiving input parsing, fuzzy name matching and compact number
//! formatting for the MCP tools.

use serde_json::Value;

/// A number from JSON: `3.4`, `"3.4"`, `"7.5%"` (→ 7.5), `"3e-4"`,
/// `"1,234"` (→ 1234). `None` for anything else.
pub fn num(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => num_str(s),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

pub fn num_str(s: &str) -> Option<f64> {
    let t = s.trim().trim_end_matches('%').trim().replace(['_', ','], "");
    let t = t.strip_prefix('+').unwrap_or(&t);
    match t.to_ascii_lowercase().as_str() {
        "nan" | "inf" | "-inf" | "infinity" | "-infinity" => return None,
        _ => {}
    }
    t.parse::<f64>().ok().filter(|x| x.is_finite())
}

/// Lowercase, every run of non-alphanumerics → `_`, trimmed.
/// `"Train Loss (EMA)"` → `train_loss_ema`.
pub fn norm_key(s: &str) -> String {
    let mut out = String::new();
    for ch in s.trim().chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        } else if !out.ends_with('_') {
            out.push('_');
        }
    }
    out.trim_matches('_').to_string()
}

/// Comparison form: lowercase alphanumerics only.
pub fn squash(s: &str) -> String {
    s.chars().filter(|c| c.is_ascii_alphanumeric()).map(|c| c.to_ascii_lowercase()).collect()
}

pub fn edit_distance(a: &str, b: &str) -> usize {
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for i in 1..=a.len() {
        let mut cur = vec![i; b.len() + 1];
        for j in 1..=b.len() {
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + usize::from(a[i - 1] != b[j - 1]));
        }
        prev = cur;
    }
    prev[b.len()]
}

/// The `n` candidates closest to `q` (substring hits first, then edit
/// distance on the squashed forms).
pub fn closest<'a>(q: &str, cands: &'a [String], n: usize) -> Vec<&'a str> {
    let qs = squash(q);
    let mut scored: Vec<(usize, &str)> = cands
        .iter()
        .map(|c| {
            let cs = squash(c);
            let d = if !qs.is_empty() && (cs.contains(&qs) || qs.contains(&cs)) { 0 } else { edit_distance(&qs, &cs) + 1 };
            (d, c.as_str())
        })
        .collect();
    scored.sort();
    scored.into_iter().take(n).map(|(_, c)| c).collect()
}

/// Metric-name families: a query equal to any member also matches the
/// others. Order inside a family is preference order.
const ALIASES: &[&[&str]] = &[
    &["lr", "learning_rate", "learningrate", "lr_scheduler", "step_size"],
    &["loss", "train_loss", "training_loss", "train_loss_ema", "loss_ema", "trainloss"],
    &["val_loss", "valid_loss", "validation_loss", "eval_loss", "test_loss", "heldout_loss"],
    &["acc", "accuracy", "train_acc", "train_accuracy", "acc_pct"],
    &["val_acc", "valid_acc", "validation_accuracy", "eval_acc", "eval_accuracy", "test_acc", "heldout_acc"],
    &["ppl", "perplexity"],
    &["grad_norm", "gradnorm", "gnorm", "gradient_norm"],
    &["throughput", "tokens_per_sec", "tok_s", "samples_per_sec", "it_s"],
    &["time_per_step", "s_per_step", "step_time", "sec_per_step"],
    &["memory", "rss", "rss_kb", "mem", "peak_mem"],
];

/// Outcome of resolving a user-typed metric name against a run's keys.
pub enum Resolved {
    /// One clear match; `note` says how it was matched when not exact.
    One { key: String, note: Option<String> },
    /// Several plausible keys; the caller lists them.
    Many(Vec<String>),
    /// Nothing close; suggestions attached.
    None(Vec<String>),
}

/// exact → case/punctuation-insensitive → alias family → unique
/// substring / token match → suggestions.
pub fn resolve_metric(q: &str, keys: &[String]) -> Resolved {
    let q = q.trim();
    if let Some(k) = keys.iter().find(|k| *k == q) {
        return Resolved::One { key: k.clone(), note: None };
    }
    let qs = squash(q);
    let hits: Vec<&String> = keys.iter().filter(|k| squash(k) == qs).collect();
    if hits.len() == 1 {
        return Resolved::One { key: hits[0].clone(), note: None };
    }
    // alias family: keys that are members, or end with a member
    let qn = norm_key(q);
    if let Some(fam) = ALIASES.iter().find(|f| f.iter().any(|m| squash(m) == qs)) {
        for m in fam.iter() {
            if let Some(k) = keys.iter().find(|k| norm_key(k) == *m) {
                return Resolved::One { key: k.clone(), note: Some(format!("{q:?} matched {k:?} (alias)")) };
            }
        }
        let suf: Vec<String> = keys.iter().filter(|k| fam.iter().any(|m| norm_key(k).ends_with(&format!("_{m}")))).cloned().collect();
        if suf.len() == 1 {
            return Resolved::One { key: suf[0].clone(), note: Some(format!("{q:?} matched {:?} (alias)", suf[0])) };
        }
        if suf.len() > 1 {
            return Resolved::Many(suf);
        }
    }
    // every underscore token of the query appears in the key
    let toks: Vec<&str> = qn.split('_').filter(|t| !t.is_empty()).collect();
    let sub: Vec<String> = keys
        .iter()
        .filter(|k| {
            let kn = norm_key(k);
            let kt: Vec<&str> = kn.split('_').collect();
            !toks.is_empty() && (toks.iter().all(|t| kt.contains(t)) || kn.contains(&qn))
        })
        .cloned()
        .collect();
    match sub.len() {
        1 => Resolved::One { key: sub[0].clone(), note: Some(format!("{q:?} matched {:?}", sub[0])) },
        0 => Resolved::None(closest(q, keys, 5).into_iter().map(String::from).collect()),
        _ => Resolved::Many(sub),
    }
}

/// Compact number: 4 significant digits, scientific outside [1e-3, 1e6).
pub fn fmt(x: f64) -> String {
    if !x.is_finite() {
        return "-".into();
    }
    if x == 0.0 {
        return "0".into();
    }
    let a = x.abs();
    if !(1e-3..1e6).contains(&a) {
        return format!("{x:.3e}");
    }
    let decimals = (3 - a.log10().floor() as i32).max(0) as usize;
    let s = format!("{x:.decimals$}");
    if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s }
}

/// Signed compact number (`+0.12`, `-3.4`).
pub fn fmt_signed(x: f64) -> String {
    if x.is_finite() && x >= 0.0 { format!("+{}", fmt(x)) } else { fmt(x) }
}

/// `3d`, `5h`, `12m`, `40s`.
pub fn fmt_age(s: f64) -> String {
    if s >= 86400.0 {
        format!("{}d", (s / 86400.0) as u64)
    } else if s >= 3600.0 {
        format!("{}h", (s / 3600.0) as u64)
    } else if s >= 60.0 {
        format!("{}m", (s / 60.0) as u64)
    } else {
        format!("{}s", s as u64)
    }
}

/// Left-aligned first column, right-aligned others, two-space gaps.
pub fn table(header: &[String], rows: &[Vec<String>]) -> String {
    let ncol = header.len();
    let mut w = vec![0usize; ncol];
    for r in std::iter::once(header).chain(rows.iter().map(|r| r.as_slice())) {
        for (i, c) in r.iter().enumerate().take(ncol) {
            w[i] = w[i].max(c.chars().count());
        }
    }
    let line = |r: &[String]| -> String {
        let cells: Vec<String> = (0..ncol)
            .map(|i| {
                let c = r.get(i).map(String::as_str).unwrap_or("");
                if i == 0 { format!("{c:<width$}", width = w[i]) } else { format!("{c:>width$}", width = w[i]) }
            })
            .collect();
        cells.join("  ").trim_end().to_string()
    };
    let mut out = vec![line(header)];
    out.extend(rows.iter().map(|r| line(r)));
    out.join("\n")
}

/// Shell-style glob with `*` and `?`, case-insensitive.
pub fn glob(pat: &str, s: &str) -> bool {
    fn go(p: &[char], s: &[char]) -> bool {
        match p.first() {
            None => s.is_empty(),
            Some('*') => go(&p[1..], s) || (!s.is_empty() && go(p, &s[1..])),
            Some('?') => !s.is_empty() && go(&p[1..], &s[1..]),
            Some(c) => s.first().is_some_and(|x| x == c) && go(&p[1..], &s[1..]),
        }
    }
    let p: Vec<char> = pat.to_lowercase().chars().collect();
    let s: Vec<char> = s.to_lowercase().chars().collect();
    go(&p, &s)
}

// ------------------------------------------------- generic log lines
//
// TODO(merge): this whole section is a stopgap. When the text-log ingest
// branch lands, replace `parse_generic_line` with `textlog::parse_line`
// (same input, same `(step?, [(key, value)])` output) and delete the rest.

const STEP_WORDS: &[&str] = &["step", "steps", "global_step", "iter", "iteration", "it", "update", "batch_idx"];

fn is_key_token(t: &str) -> bool {
    let t = t.trim_end_matches([':', '=']);
    t.chars().next().is_some_and(|c| c.is_ascii_alphabetic()) && t.chars().all(|c| c.is_ascii_alphanumeric() || "_-./()".contains(c)) && t.len() <= 48
}

/// Value token: a number with optional `%`, trailing `,;|` stripped, and a
/// `N/M` progress form read as `N`.
fn value_token(t: &str) -> Option<f64> {
    let t = t.trim_end_matches([',', ';', '|', ']', ')', ':']);
    if let Some((a, b)) = t.split_once('/') {
        if num_str(b).is_some() {
            return num_str(a);
        }
        return None;
    }
    num_str(t)
}

/// Generic `key value` / `key=value` / `key: value` scanner for one log
/// line. Returns `(step, metrics)`; `step` comes from a step-like key
/// (`step`, `iter`, `it`, …; `epoch` when there is no step). Keys are
/// normalised with [`norm_key`]. `None` when the line holds no metric.
pub fn parse_generic_line(line: &str) -> Option<(Option<u64>, Vec<(String, f64)>)> {
    // split `k=v` and `k:v` glued forms, and separators, into tokens
    let mut toks: Vec<String> = Vec::new();
    for raw in line.split(|c: char| c.is_whitespace() || c == '|' || c == ';' || c == ',' || c == '[' || c == ']' || c == '{' || c == '}') {
        let raw = raw.trim_matches(|c: char| c == '"' || c == '\'');
        if raw.is_empty() {
            continue;
        }
        if let Some((k, v)) = raw.split_once('=').or_else(|| raw.split_once(':').filter(|(k, v)| is_key_token(k) && !v.is_empty())) {
            toks.push(format!("{k}="));
            if !v.is_empty() {
                toks.push(v.to_string());
            }
        } else {
            toks.push(raw.to_string());
        }
    }
    let mut kv: Vec<(String, f64)> = Vec::new();
    let mut i = 0;
    while i + 1 < toks.len() {
        let k = &toks[i];
        if is_key_token(k) {
            if let Some(v) = value_token(&toks[i + 1]) {
                let key = norm_key(k.trim_end_matches(['=', ':']));
                if !key.is_empty() && !kv.iter().any(|(x, _)| *x == key) {
                    kv.push((key, v));
                }
                i += 2;
                continue;
            }
        }
        i += 1;
    }
    if kv.is_empty() {
        return None;
    }
    let mut step = None;
    if let Some(pos) = kv.iter().position(|(k, v)| STEP_WORDS.contains(&k.as_str()) && *v >= 0.0 && v.fract() == 0.0) {
        step = Some(kv.remove(pos).1 as u64);
    } else if let Some((_, v)) = kv.iter().find(|(k, v)| k == "epoch" && *v >= 0.0 && v.fract() == 0.0) {
        step = Some(*v as u64);
    }
    kv.retain(|(k, _)| k != "t");
    if kv.is_empty() { None } else { Some((step, kv)) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn numbers_are_forgiving() {
        assert_eq!(num(&json!(3.4)), Some(3.4));
        assert_eq!(num(&json!("3.4")), Some(3.4));
        assert_eq!(num(&json!("7.5%")), Some(7.5));
        assert_eq!(num(&json!(" 3e-4 ")), Some(3e-4));
        assert_eq!(num(&json!("1,234")), Some(1234.0));
        assert_eq!(num(&json!("abc")), None);
        assert_eq!(num(&json!("nan")), None);
    }

    #[test]
    fn formatting() {
        assert_eq!(fmt(4.88), "4.88");
        assert_eq!(fmt(1234.5678), "1235");
        assert_eq!(fmt(0.000123), "1.230e-4");
        assert_eq!(fmt(0.5), "0.5");
        assert_eq!(fmt_signed(0.25), "+0.25");
        assert_eq!(fmt_age(3.0 * 86400.0 + 5.0), "3d");
    }

    #[test]
    fn metric_resolution() {
        let keys: Vec<String> = ["learning_rate", "train_loss_ema", "brain_heldout_nll", "brain_indist_nll", "acc_pct"].iter().map(|s| s.to_string()).collect();
        let one = |q: &str| match resolve_metric(q, &keys) {
            Resolved::One { key, .. } => key,
            Resolved::Many(v) => format!("many:{}", v.join(",")),
            Resolved::None(v) => format!("none:{}", v.join(",")),
        };
        assert_eq!(one("lr"), "learning_rate");
        assert_eq!(one("LR"), "learning_rate");
        assert_eq!(one("Learning Rate"), "learning_rate");
        assert_eq!(one("loss"), "train_loss_ema");
        assert_eq!(one("heldout_nll"), "brain_heldout_nll");
        assert_eq!(one("heldout nll"), "brain_heldout_nll");
        assert!(one("nll").starts_with("many:"));
        assert!(one("zzz_qqq").starts_with("none:"));
    }

    #[test]
    fn glob_matches() {
        assert!(glob("bkk07/*", "bkk07/small"));
        assert!(glob("*small*", "bkk07/small_s2"));
        assert!(!glob("bkk07/mid", "bkk07/mid_s2"));
    }

    #[test]
    fn generic_lines() {
        let (s, kv) = parse_generic_line("step 100 | loss 2.31 | lr 3e-4 | acc 71.2%").unwrap();
        assert_eq!(s, Some(100));
        assert_eq!(kv, vec![("loss".into(), 2.31), ("lr".into(), 3e-4), ("acc".into(), 71.2)]);
        let (s, kv) = parse_generic_line("2026-10-11 12:00:01 INFO step=5 loss=1.25, val_loss: 1.5").unwrap();
        assert_eq!(s, Some(5));
        assert_eq!(kv, vec![("loss".into(), 1.25), ("val_loss".into(), 1.5)]);
        let (s, kv) = parse_generic_line("Epoch 3: train_loss=0.45 val_loss=0.52").unwrap();
        assert_eq!(s, Some(3));
        assert!(kv.contains(&("val_loss".into(), 0.52)));
        let (s, _) = parse_generic_line("iter 40/1000 loss 0.9").unwrap();
        assert_eq!(s, Some(40));
        assert!(parse_generic_line("Loading checkpoint from /tmp/x").is_none());
        assert!(parse_generic_line("").is_none());
    }
}
