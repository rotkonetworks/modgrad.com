//! Plain-text training logs: a `key value` extractor that copes with the
//! usual shapes,
//!
//! ```text
//! step 1250: loss 3.48 acc 7.5%
//! 2026-10-11 12:00:01 INFO iter 100 | loss=2.31 | lr 3e-4 | 1200 tok/s
//! [epoch 2/10] train_loss: 0.45
//! {'loss': 2.31, 'learning_rate': 5e-05, 'epoch': 0.1}      (HF Trainer)
//! Epoch 1: 45%|████▌ | 450/1000 [00:12<00:15, 36.2it/s, loss=0.234]  (tqdm: only k=v)
//! ```
//!
//! A value is a whole token that parses as a number (units such as `%`,
//! `s`, `k`, `it/s` allowed); its key is the word just before it (two
//! lowercase words join: `train loss 0.4` → `train_loss`), or the token's
//! own `key=value` / `key:value`, or a rate unit right after it (`1200
//! tok/s` → `tok/s`). Dates, clock times, paths and `n/total` progress
//! never become values (`epoch 2/10` → epoch 2). The step comes from a
//! `step`/`iter`/`it`/`global_step` key; when any line has one, lines
//! without one are ignored. Otherwise x is the epoch (if it distinguishes
//! lines) or line order. Keys seen on only one line of a longer log are
//! dropped as noise.

use crate::num::{is_rate_unit, parse_value};
use crate::{FileReport, LocalRun, Missing, Row, Rows, is_epoch, step_rank};

#[derive(Default)]
pub struct Text {
    rows: Rows,
}

const MAX_VALUES_PER_LINE: usize = 64;

fn is_sep(c: char) -> bool {
    c.is_whitespace() || "|,;[](){}<>".contains(c)
}

/// `train_loss`, `in-dist`, `loss/val`, `tok/s`: starts with a letter or
/// underscore, then letters, digits, `_ . / -`.
fn is_word(s: &str) -> bool {
    let mut c = s.chars();
    matches!(c.next(), Some(ch) if ch.is_alphabetic() || ch == '_') && s.len() <= 64 && s.chars().all(|ch| ch.is_alphanumeric() || "_./-".contains(ch)) && !s.contains("//")
}

fn strip_quotes(s: &str) -> &str {
    s.trim_matches(|c| c == '"' || c == '\'' || c == '`')
}

/// `loss:` → (`loss`, true); `'loss':` → (`loss`, true); `loss` → (`loss`, false)
fn clean_word(tok: &str) -> (&str, bool) {
    let a = strip_quotes(tok);
    let b = a.trim_end_matches([':', '=']);
    let labelled = b.len() < a.len();
    (strip_quotes(b), labelled)
}

fn value_of(tok: &str) -> Option<f64> {
    parse_value(strip_quotes(tok.trim_end_matches([':', '.'])).trim_end_matches(':'))
}

/// `2/10` → 2 (progress); `450/1000` → 450.
fn progress(tok: &str) -> Option<f64> {
    let (a, b) = tok.trim_end_matches(':').split_once('/')?;
    if a.is_empty() || b.is_empty() || !a.bytes().all(|c| c.is_ascii_digit()) || !b.bytes().all(|c| c.is_ascii_digit()) {
        return None;
    }
    a.parse().ok()
}

/// `key=value` / `key:value` inside one token.
fn inline_kv(tok: &str) -> Option<(&str, f64)> {
    let (k, v) = match tok.split_once('=') {
        Some(p) => p,
        None => {
            let (k, v) = tok.split_once(':')?;
            if v.is_empty() {
                return None;
            }
            (k, v)
        }
    };
    let k = strip_quotes(k);
    if !is_word(k) {
        return None;
    }
    let v = value_of(v).or_else(|| progress(v))?;
    Some((k, v))
}

const STOP: &[&str] = &["at", "on", "in", "of", "to", "for", "the", "and", "with", "from", "is", "was", "by", "a", "an"];

fn simple_lower(w: &str) -> bool {
    !w.is_empty() && w.chars().all(|c| c.is_ascii_lowercase() || c == '_')
}

/// Key from the words seen since the previous value.
fn make_key(words: &[(String, bool)]) -> Option<String> {
    let (w, _) = words.last()?;
    if step_rank(w).is_some() || is_epoch(w) {
        return Some(w.clone());
    }
    if words.len() >= 2 {
        let (p, p_label) = &words[words.len() - 2];
        if !p_label && simple_lower(p) && simple_lower(w) && !STOP.contains(&p.as_str()) && step_rank(p).is_none() && !is_epoch(p) {
            return Some(format!("{p}_{w}"));
        }
    }
    Some(w.clone())
}

fn is_tqdm(l: &str) -> bool {
    l.contains("it/s]") || l.contains("s/it]") || l.contains("it/s,") || l.contains('█') || l.contains("%|")
}

/// Extract (key, value) pairs from one line.
pub fn extract(l: &str) -> Vec<(String, f64)> {
    let mut out: Vec<(String, f64)> = Vec::new();
    let toks: Vec<&str> = l.split(is_sep).filter(|t| !t.is_empty()).collect();
    if is_tqdm(l) {
        for t in &toks {
            if t.contains('=') {
                if let Some((k, v)) = inline_kv(t) {
                    out.push((k.to_string(), v));
                }
            }
        }
        return out;
    }
    let mut words: Vec<(String, bool)> = Vec::new();
    let mut i = 0;
    while i < toks.len() {
        let t = toks[i];
        i += 1;
        if let Some((k, v)) = inline_kv(t) {
            out.push((k.to_string(), v));
            words.clear();
            continue;
        }
        let v = value_of(t).or_else(|| progress(t));
        if let Some(v) = v {
            let key = match make_key(&words) {
                Some(k) => Some(k),
                None => match toks.get(i) {
                    Some(u) if is_rate_unit(u) => {
                        i += 1;
                        Some(u.trim_end_matches([',', ';']).to_string())
                    }
                    _ => None,
                },
            };
            if let Some(k) = key {
                out.push((k, v));
            }
            words.clear();
            continue;
        }
        let (w, labelled) = clean_word(t);
        if is_word(w) {
            words.push((w.to_string(), labelled));
            if words.len() > 4 {
                words.remove(0);
            }
        } else {
            words.clear();
        }
    }
    out
}

impl Text {
    pub fn line(&mut self, l: &str) -> bool {
        let kv = extract(l);
        if kv.is_empty() || kv.len() > MAX_VALUES_PER_LINE {
            return false;
        }
        let mut row = Row::default();
        let mut best: Option<usize> = None;
        let mut rest = Vec::new();
        for (k, v) in kv {
            if let Some(r) = step_rank(&k) {
                if v.is_finite() && best.is_none_or(|b| r < b) {
                    best = Some(r);
                    row.step = Some(v);
                }
                continue;
            }
            if is_epoch(&k) && v.is_finite() {
                row.epoch = Some(v);
            }
            rest.push((k, v));
        }
        if rest.is_empty() {
            return false;
        }
        row.kv = rest.into_iter().map(|(k, v)| (self.rows.id(&k), v)).collect();
        self.rows.push(row);
        true
    }

    pub fn finish(self, run: &mut LocalRun, rep: &mut FileReport) {
        let n = self.rows.len();
        if n == 0 {
            rep.error = Some("no `key value` pairs found".into());
            return;
        }
        let min = if n >= 5 { 2 } else { 1 };
        self.rows.finish(run, rep, Missing::Drop, min);
    }
}
