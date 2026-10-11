//! Generic metric extraction from free-form training log lines.
//!
//! [`parse_line`] is a pure function: one line in, the metrics it holds
//! out. It is what `obs run` applies to a wrapped command's stdout and
//! stderr, so it must accept the common shapes training scripts print and
//! reject prose:
//!
//! ```text
//! step 1250: loss 3.48 acc 7.5%            -> step 1250, loss 3.48, acc 7.5
//! iter 100 | loss=2.31 | lr 3e-4           -> step 100, loss 2.31, lr 0.0003
//! [epoch 2/10] train_loss: 0.45            -> epoch 2, epoch_total 10, train_loss 0.45
//! {'loss': 0.41, 'learning_rate': 5e-05}   -> loss 0.41, learning_rate 5e-05
//! step 1250: train loss (ema) 3.48; BRAIN in-dist 5.0% nll 7.47 | held-out 4.0% nll 7.55  [1.23 s/step, ...]
//!                                          -> step 1250, train_loss_ema 3.48, brain_in_dist 5.0,
//!                                             brain_in_dist_nll 7.47, held_out 4.0, held_out_nll 7.55, s/step 1.23, ...
//! ```
//!
//! Rules, in order:
//!
//! 1. ANSI escapes are removed and only the text after the last `\r` is
//!    kept (progress bars redraw with `\r`).
//! 2. Whitespace tokens that are dates or clock times (`2026-10-11`,
//!    `12:00:03`, `[00:12<00:15,`) are dropped.
//! 3. `| , ; [ ] { }` split the line into segments. `= :` quotes and
//!    parentheses are ignored, so `loss=2`, `loss: 2` and `'loss': 2` read
//!    the same, and `train loss (ema) 3.4` keys as `train_loss_ema`.
//! 4. Within a segment, the words before a number name it (the last four
//!    words, snake_cased and lowercased; `/` is kept, so `train/loss`
//!    stays). A segment that starts with a number takes the words after
//!    it as the name (`1.23 s/step`); words trailing a named value are
//!    units and are dropped (`RSS 123456 kB`).
//! 5. Numbers: integers, decimals, exponents, `nan`/`inf`, a trailing `%`
//!    (stripped, value kept as printed) and `a/b` fractions (`epoch 2/10`
//!    gives `epoch = 2` and `epoch_total = 10`).
//! 6. Any other token (`±`, `36.5it/s`, `█`) ends the current name.
//! 7. A value named `step`, `iter`, `iteration`, `it` or `global_step`
//!    (by its last word) is the row's step when it is a non-negative
//!    integer; it is not also returned as a metric.
//! 8. A key that occurs twice in one line is qualified by its segment's
//!    first key (`nll` under `held-out` becomes `held_out_nll`).
//! 9. Prose guard: a line is accepted only if it has a step, or at least
//!    two values, or one value written with `=`/`:` and not followed by
//!    more words (`loading dataset: 12000 examples` is prose), or one value whose
//!    name contains a common metric word (loss, acc, lr, ...). So
//!    `Loaded 3 files` is rejected while `loss 0.3` is kept.

/// The metrics found on one line.
#[derive(Debug, Clone, PartialEq)]
pub struct TextRow {
    /// The step, when the line names one (rule 7).
    pub step: Option<u64>,
    /// `(key, value)` in line order. Keys are never `step` or `t`.
    pub values: Vec<(String, f64)>,
}

const STEP_WORDS: [&str; 5] = ["step", "iter", "iteration", "it", "global_step"];
const LEVEL_WORDS: [&str; 7] = ["info", "debug", "warn", "warning", "error", "trace", "critical"];
const METRIC_WORDS: [&str; 18] =
    ["loss", "acc", "accuracy", "lr", "learning_rate", "ppl", "perplexity", "nll", "grad", "norm", "reward", "score", "bleu", "f1", "auc", "epoch", "tokens", "throughput"];
const MAX_KEY_WORDS: usize = 4;

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Word(String),
    Num { v: f64, total: Option<f64> },
    /// Segment break.
    Sep,
    /// `=` or `:` (marks an explicit key/value pair).
    Join,
    /// Anything else: ends the pending name.
    Junk,
}

fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut it = s.chars().peekable();
    while let Some(c) = it.next() {
        if c == '\u{1b}' {
            if it.peek() == Some(&'[') {
                it.next();
                // CSI: parameters then one final byte in @..~
                for d in it.by_ref() {
                    if ('@'..='~').contains(&d) {
                        break;
                    }
                }
            }
            continue;
        }
        out.push(c);
    }
    out
}

/// `12:00:03`, `00:12<00:15`, `2026-10-11`, `2026-10-11T12:00:03Z`.
fn is_clock_or_date(tok: &str) -> bool {
    let b = tok.as_bytes();
    let digit_colon_digit = b.windows(3).any(|w| w[0].is_ascii_digit() && w[1] == b':' && w[2].is_ascii_digit());
    let date = b.len() >= 10
        && b[..4].iter().all(u8::is_ascii_digit)
        && (b[4] == b'-' || b[4] == b'/')
        && b[5..7].iter().all(u8::is_ascii_digit)
        && b[7] == b[4]
        && b[8..10].iter().all(u8::is_ascii_digit);
    let trimmed = tok.trim_start_matches(['[', '(']);
    digit_colon_digit || date || (trimmed.len() != tok.len() && is_clock_or_date(trimmed))
}

fn parse_num(s: &str) -> Option<f64> {
    let s = s.strip_suffix('%').unwrap_or(s);
    let l = s.to_ascii_lowercase();
    match l.trim_start_matches(['+', '-']) {
        "nan" => return Some(f64::NAN),
        "inf" | "infinity" => return Some(if l.starts_with('-') { f64::NEG_INFINITY } else { f64::INFINITY }),
        _ => {}
    }
    // only plain decimal syntax: Rust's parser also takes "infinity" etc.
    let ok = !s.is_empty() && s.chars().all(|c| c.is_ascii_digit() || "+-.eE".contains(c)) && s.chars().any(|c| c.is_ascii_digit());
    if !ok {
        return None;
    }
    s.parse::<f64>().ok()
}

fn classify(raw: &str) -> Tok {
    if raw.is_empty() {
        return Tok::Junk;
    }
    if let Some(v) = parse_num(raw) {
        return Tok::Num { v, total: None };
    }
    if let Some((a, b)) = raw.split_once('/')
        && let (Some(a), Some(b)) = (parse_num(a), parse_num(b))
        && !raw.ends_with('%')
    {
        return Tok::Num { v: a, total: Some(b) };
    }
    let wordish = raw.chars().all(|c| c.is_ascii_alphanumeric() || "_-./".contains(c));
    let starts_alpha = raw.chars().next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_');
    if wordish && starts_alpha { Tok::Word(raw.to_string()) } else { Tok::Junk }
}

fn tokenize(line: &str) -> Vec<Tok> {
    let mut out = Vec::new();
    for ws in line.split_whitespace() {
        if is_clock_or_date(ws) {
            // a dropped clock token like `[00:12<00:15,` may carry a separator
            if ws.contains(['|', ',', ';', '[', ']', '{', '}']) {
                out.push(Tok::Sep);
            }
            continue;
        }
        let mut cur = String::new();
        let flush = |cur: &mut String, out: &mut Vec<Tok>| {
            if !cur.is_empty() {
                out.push(classify(cur));
                cur.clear();
            }
        };
        for c in ws.chars() {
            match c {
                '|' | ',' | ';' | '[' | ']' | '{' | '}' => {
                    flush(&mut cur, &mut out);
                    out.push(Tok::Sep);
                }
                '=' | ':' => {
                    flush(&mut cur, &mut out);
                    out.push(Tok::Join);
                }
                '(' | ')' | '\'' | '"' | '`' => flush(&mut cur, &mut out),
                _ => cur.push(c),
            }
        }
        flush(&mut cur, &mut out);
    }
    out
}

fn snake(words: &[String]) -> String {
    let mut k = String::new();
    for w in words {
        let w = w.to_ascii_lowercase().replace(['-', '.'], "_");
        let w = w.trim_matches('_');
        if w.is_empty() {
            continue;
        }
        if !k.is_empty() && !k.ends_with('/') && !w.starts_with('/') {
            k.push('_');
        }
        k.push_str(w);
    }
    k.trim_matches(['_', '/']).to_string()
}

struct Emit {
    key: String,
    seg_head: String,
    v: f64,
    joined: bool,
}

/// Extract the metrics on one log line; `None` when it holds none.
pub fn parse_line(line: &str) -> Option<TextRow> {
    let clean = strip_ansi(line);
    let clean = clean.rsplit('\r').next().unwrap_or("");
    let toks = tokenize(clean);

    let mut emits: Vec<Emit> = Vec::new();
    let mut step: Option<u64> = None;
    let mut words: Vec<String> = Vec::new();
    let mut joined = false;
    // value that opened the segment, waiting for its trailing name
    let mut leading: Option<f64> = None;
    let mut seg_head = String::new();
    let mut seg_named = false;
    // words left after a named value at a segment end (units, or prose like `12000 examples`)
    let mut trailing = false;

    let end_segment = |words: &mut Vec<String>, leading: &mut Option<f64>, seg_head: &mut String, seg_named: &mut bool, emits: &mut Vec<Emit>| {
        if let Some(v) = leading.take() {
            let key = snake(words);
            if !key.is_empty() {
                emits.push(Emit { key, seg_head: String::new(), v, joined: false });
            }
        }
        words.clear();
        seg_head.clear();
        *seg_named = false;
    };

    for t in toks {
        match t {
            Tok::Word(w) => {
                if LEVEL_WORDS.contains(&w.to_ascii_lowercase().as_str()) {
                    continue;
                }
                words.push(w);
            }
            Tok::Join => joined |= !words.is_empty(),
            Tok::Junk => {
                words.clear();
                joined = false;
            }
            Tok::Sep => {
                trailing |= leading.is_none() && seg_named && !words.is_empty();
                end_segment(&mut words, &mut leading, &mut seg_head, &mut seg_named, &mut emits);
                joined = false;
            }
            Tok::Num { v, total } => {
                if words.is_empty() {
                    if !seg_named && leading.is_none() {
                        leading = Some(v);
                    }
                    joined = false;
                    continue;
                }
                // a leading value whose name is these words (`1.23 s/step 4`)
                if let Some(lv) = leading.take() {
                    let key = snake(&words);
                    if !key.is_empty() {
                        emits.push(Emit { key, seg_head: String::new(), v: lv, joined: false });
                    }
                    words.clear();
                    continue;
                }
                let last = words.last().unwrap().to_ascii_lowercase();
                let name = &words[words.len().saturating_sub(MAX_KEY_WORDS)..];
                let key = snake(name);
                words.clear();
                seg_named = true;
                if STEP_WORDS.contains(&last.as_str()) {
                    // a step word never becomes a metric; only a valid one is the step
                    if step.is_none() && v >= 0.0 && v.fract() == 0.0 && v < 1e18 {
                        step = Some(v as u64);
                    }
                    joined = false;
                    continue;
                }
                if key.is_empty() || key == "t" {
                    joined = false;
                    continue;
                }
                if seg_head.is_empty() {
                    seg_head = key.clone();
                }
                emits.push(Emit { key: key.clone(), seg_head: seg_head.clone(), v, joined });
                if let Some(tv) = total {
                    emits.push(Emit { key: format!("{key}_total"), seg_head: seg_head.clone(), v: tv, joined });
                }
                joined = false;
            }
        }
    }
    trailing |= leading.is_none() && seg_named && !words.is_empty();
    end_segment(&mut words, &mut leading, &mut seg_head, &mut seg_named, &mut emits);

    // rule 8: qualify keys that repeat
    let mut counts = std::collections::HashMap::<String, usize>::new();
    for e in &emits {
        *counts.entry(e.key.clone()).or_default() += 1;
    }
    for e in emits.iter_mut() {
        if counts[&e.key] > 1 && !e.seg_head.is_empty() && e.seg_head != e.key {
            e.key = format!("{}_{}", e.seg_head, e.key);
        }
    }
    let mut seen = std::collections::HashMap::<String, usize>::new();
    let mut values = Vec::with_capacity(emits.len());
    for e in emits.iter().filter(|e| e.key != "step" && e.key != "t") {
        let n = seen.entry(e.key.clone()).or_default();
        *n += 1;
        let key = if *n > 1 { format!("{}_{}", e.key, n) } else { e.key.clone() };
        values.push((key, e.v));
    }

    let metricish = |k: &str| METRIC_WORDS.iter().any(|m| k.split(['_', '/']).any(|p| p == *m));
    let accept = step.is_some() || values.len() >= 2 || (emits.iter().any(|e| e.joined) && !trailing) || values.iter().any(|(k, _)| metricish(k));
    if !accept || (values.is_empty() && step.is_none()) {
        return None;
    }
    Some(TextRow { step, values })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kv(line: &str) -> (Option<u64>, Vec<(String, f64)>) {
        let r = parse_line(line).unwrap_or_else(|| panic!("no metrics in {line:?}"));
        (r.step, r.values)
    }
    fn get(vals: &[(String, f64)], k: &str) -> f64 {
        vals.iter().find(|(x, _)| x == k).unwrap_or_else(|| panic!("no {k} in {vals:?}")).1
    }

    #[test]
    fn colon_step_and_percent() {
        let (s, v) = kv("step 1250: loss 3.48 acc 7.5%");
        assert_eq!(s, Some(1250));
        assert_eq!(v, vec![("loss".into(), 3.48), ("acc".into(), 7.5)]);
    }

    #[test]
    fn pipes_equals_and_exponents() {
        let (s, v) = kv("iter 100 | loss=2.31 | lr 3e-4");
        assert_eq!(s, Some(100));
        assert_eq!(v, vec![("loss".into(), 2.31), ("lr".into(), 3e-4)]);
    }

    #[test]
    fn bracketed_epoch_fraction() {
        let (s, v) = kv("[epoch 2/10] train_loss: 0.45");
        assert_eq!(s, None);
        assert_eq!(v, vec![("epoch".into(), 2.0), ("epoch_total".into(), 10.0), ("train_loss".into(), 0.45)]);
    }

    #[test]
    fn legacy_ctmlm_line() {
        let (s, v) = kv("step 1250: train loss (ema) 3.4812; BRAIN in-dist 5.0% nll 7.47 | held-out 4.0% nll 7.55  [1.23 s/step, 12.3 min, RSS 123456 kB]");
        assert_eq!(s, Some(1250));
        assert_eq!(get(&v, "train_loss_ema"), 3.4812);
        assert_eq!(get(&v, "brain_in_dist"), 5.0);
        assert_eq!(get(&v, "brain_in_dist_nll"), 7.47);
        assert_eq!(get(&v, "held_out"), 4.0);
        assert_eq!(get(&v, "held_out_nll"), 7.55);
        assert_eq!(get(&v, "s/step"), 1.23);
        assert_eq!(get(&v, "min"), 12.3);
        assert_eq!(get(&v, "rss"), 123456.0);
        assert!(!v.iter().any(|(k, _)| k == "kb" || k == "nll"), "{v:?}");
    }

    #[test]
    fn python_dict_print() {
        let (s, v) = kv("{'loss': 0.4123, 'grad_norm': 1.25, 'learning_rate': 5e-05, 'epoch': 0.5}");
        assert_eq!(s, None);
        assert_eq!(v, vec![("loss".into(), 0.4123), ("grad_norm".into(), 1.25), ("learning_rate".into(), 5e-5), ("epoch".into(), 0.5)]);
    }

    #[test]
    fn timestamps_levels_and_logger_prefixes() {
        let (s, v) = kv("2026-10-11 12:00:03,512 INFO step 5 loss 3.25");
        assert_eq!(s, Some(5));
        assert_eq!(v, vec![("loss".into(), 3.25)]);
        let (s, v) = kv("[2026-10-11T12:00:03Z] it 7, val/loss 1.5, val/acc 81.2%");
        assert_eq!(s, Some(7));
        assert_eq!(v, vec![("val/loss".into(), 1.5), ("val/acc".into(), 81.2)]);
    }

    #[test]
    fn tqdm_progress_bar_keeps_postfix_only() {
        let line = "\r 45%|████▌     | 450/1000 [00:12<00:15, 36.5it/s, loss=0.234]";
        let (s, v) = kv(line);
        assert_eq!(s, None);
        assert_eq!(v, vec![("loss".into(), 0.234)]);
        // with an earlier redraw on the same line, only the last one counts
        let (_, v) = kv("10%|# | loss=0.9\r 20%|## | loss=0.8");
        assert_eq!(v, vec![("loss".into(), 0.8)]);
    }

    #[test]
    fn ansi_colours_are_ignored() {
        let (s, v) = kv("\u{1b}[32mstep 3\u{1b}[0m loss \u{1b}[1m2.5\u{1b}[0m");
        assert_eq!((s, v), (Some(3), vec![("loss".into(), 2.5)]));
    }

    #[test]
    fn nan_and_inf_are_values() {
        let (_, v) = kv("step 9 loss nan grad_norm inf");
        assert!(get(&v, "loss").is_nan());
        assert_eq!(get(&v, "grad_norm"), f64::INFINITY);
    }

    #[test]
    fn negative_and_signed_values() {
        let (_, v) = kv("step 2 reward -1.5 delta +0.25");
        assert_eq!(v, vec![("reward".into(), -1.5), ("delta".into(), 0.25)]);
    }

    #[test]
    fn step_with_total() {
        let (s, v) = kv("Epoch 3/10, Step 500/2000, loss 0.2");
        assert_eq!(s, Some(500));
        assert_eq!(get(&v, "epoch"), 3.0);
        assert_eq!(get(&v, "loss"), 0.2);
    }

    #[test]
    fn prose_is_rejected() {
        for l in [
            "",
            "Loaded 3 files from disk",
            "Saving checkpoint to ckpt_100.pt",
            "Downloading model (1.2 GB) ...",
            "Traceback (most recent call last):",
            "  File \"train.py\", line 42, in <module>",
            "12:00:03",
            "2026-10-11 12:00:03 starting",
            "using 8 workers",
            "hello world",
            "loading dataset: 12000 examples",
            "found: 3 GPUs",
        ] {
            assert_eq!(parse_line(l), None, "{l:?}");
        }
    }

    #[test]
    fn single_metric_word_value_is_kept() {
        assert_eq!(kv("loss 0.3"), (None, vec![("loss".into(), 0.3)]));
        assert_eq!(kv("tokens/s 12345"), (None, vec![("tokens/s".into(), 12345.0)]));
    }

    #[test]
    fn duplicate_keys_without_segments_get_suffixes() {
        let (_, v) = kv("loss 1 loss 2");
        assert_eq!(v, vec![("loss".into(), 1.0), ("loss_2".into(), 2.0)]);
    }

    #[test]
    fn no_reserved_keys() {
        for l in ["t 5 loss 1", "step 3 t 1.5 loss 2", "time t=3 loss=1"] {
            let r = parse_line(l).unwrap();
            assert!(r.values.iter().all(|(k, _)| k != "step" && k != "t"), "{l}: {r:?}");
        }
    }

    #[test]
    fn version_strings_and_glued_units_are_not_values() {
        assert_eq!(parse_line("torch 2.4.1+cu121 cuda 12.1"), None);
        assert_eq!(parse_line("throughput 36.5it/s"), None);
    }

    #[test]
    fn never_panics_on_odd_input() {
        for l in ["::::", "|||", "= = =", "step", "step -1 loss 2", "1e999 x", "a/b/c 1/0", "\u{1b}[", "\u{1b}[31", "[[[1]]]", "%%%", "1/ 2/", "loss 1/0"] {
            let _ = parse_line(l);
        }
        assert_eq!(parse_line("step -1 loss 2").unwrap().step, None);
    }
}
