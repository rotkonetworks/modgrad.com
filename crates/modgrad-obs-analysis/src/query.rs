//! Panel queries for the dashboard's graph builder. A panel is a query plus a
//! view; the query is text that round-trips with the chips shown in the
//! panel header.
//!
//! ```text
//! loss{run=~"bkk07/*", lr>1e-4}; nat_none{run="bkk07/small"}; d = a - b
//!     | ema .9 | by size | mean±sd over seed | x tokens | forecast to 10k
//! ```
//!
//! # Grammar
//!
//! ```text
//! query    := entry (';' entry)* ('|' stages)*
//! entry    := name '=' expr                     formula over letters (see crate::expr)
//!           | [metric] ['{' matcher (',' matcher)* '}']
//! metric   := glob over metric names ('*', '?', 'a|b' alternatives); omitted = '*'
//! matcher  := field op value
//! op       := '=' | '!=' | '=~' | '!~' | '<' | '<=' | '>' | '>='
//! value    := "quoted" | number[k|M|G] | bare-word
//! stages   := stage+                            several stages may share one '|' segment
//! stage    := 'ema' w | 'by' field | ('mean±sd'|'mean+-sd'|'mean'|'median'|'min'|'max') 'over' field
//!           | 'split' field | 'x' axis | 'forecast' ['to' n] [method] | 'log' | 'norm' (z|minmax|first)
//!           | 'baseline' run ['diff'|'ratio'] | 'align' (intersection|union|exact) | 'show' letters
//!           | 'heatmap' ['over' 'layer'] ['rownorm'] ['max']
//! ```
//!
//! Selector entries are lettered `a, b, c, …` in order (skipping names
//! taken by formulas); each matches a *set* of streams (run × metric).
//! Fields: `run` (id, also `id`), `name`, `source`, `project`, else a
//! config key. `=~` / `!~` take anchored globs (`*`, `?`, `|` between
//! alternatives), not full regular expressions. Numeric comparisons
//! (`< <= > >=`, and `=`/`!=` when both sides are numbers) need a numeric
//! config value; a run without the field does not match them. A missing
//! field reads as the empty string for `=`, `!=`, `=~`, `!~`.
//!
//! # Evaluation ([`eval`])
//!
//! Per panel, in this order: selectors → formulas (per run; a letter
//! matching one stream is broadcast to every run) → `show` → `baseline`
//! (per run, against the baseline run's stream of the same letter and
//! metric, aligned; the baseline itself is dropped) → `norm` (per run) →
//! aggregation (runs whose config agrees on every field except the
//! `over` field are one group; a field none of them has is ignored, so
//! replicate runs without a seed field still group) → x mapping (x is a
//! metric of the same run joined on step via [`crate::expr::align`];
//! `time` is `elapsed_min × 60`; an unknown name falls back to the first
//! metric containing it, so `x tokens` finds `tokens_seen`; aggregates use
//! their first member's x) → `ema` (display only; `raw` stays unsmoothed)
//! → block-mean downsampling when x is step or time.
//!
//! `heatmap over layer` additionally groups the selector streams of each
//! run into layer families ([`layer_index`]: `layers/<i>/<stat>` and the
//! other recognised spellings) and builds one layer × step matrix per
//! (run, family) from the raw values, after `norm` and before aggregation
//! ([`heatmaps`]). The traces stay in the output (the dashboard draws one
//! layer's line from them on click).

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use libm::sqrt;

use crate::expr::{self, Align};

// ------------------------------------------------------------------ catalog

/// What the evaluator knows about the runs. Line format (tab-separated):
/// `R id` starts a run, `I key value` an identity field (name, source,
/// project), `C key value` a config field, `M metric` a metric.
#[derive(Clone, Debug, Default)]
pub struct Catalog {
    pub runs: Vec<RunInfo>,
}

#[derive(Clone, Debug, Default)]
pub struct RunInfo {
    pub id: String,
    pub ident: Vec<(String, String)>,
    pub config: Vec<(String, String)>,
    pub metrics: Vec<String>,
}

impl RunInfo {
    pub fn field(&self, f: &str) -> Option<&str> {
        if f == "run" || f == "id" {
            return Some(&self.id);
        }
        self.config.iter().chain(self.ident.iter()).find(|(k, _)| k == f).map(|(_, v)| v.as_str())
    }
}

impl Catalog {
    pub fn parse(s: &str) -> Catalog {
        let mut runs: Vec<RunInfo> = Vec::new();
        for line in s.split('\n') {
            let mut it = line.splitn(3, '\t');
            let (Some(tag), Some(a)) = (it.next(), it.next()) else { continue };
            let b = it.next().unwrap_or("");
            match tag {
                "R" => runs.push(RunInfo { id: a.to_string(), ..Default::default() }),
                "I" => {
                    if let Some(r) = runs.last_mut() {
                        r.ident.push((a.to_string(), b.to_string()))
                    }
                }
                "C" => {
                    if let Some(r) = runs.last_mut() {
                        r.config.push((a.to_string(), b.to_string()))
                    }
                }
                "M" => {
                    if let Some(r) = runs.last_mut() {
                        r.metrics.push(a.to_string())
                    }
                }
                _ => {}
            }
        }
        Catalog { runs }
    }
}

// ------------------------------------------------------------------ query model

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Eq,
    Ne,
    Glob,
    NotGlob,
    Lt,
    Le,
    Gt,
    Ge,
}

impl Op {
    fn text(self) -> &'static str {
        match self {
            Op::Eq => "=",
            Op::Ne => "!=",
            Op::Glob => "=~",
            Op::NotGlob => "!~",
            Op::Lt => "<",
            Op::Le => "<=",
            Op::Gt => ">",
            Op::Ge => ">=",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Matcher {
    pub field: String,
    pub op: Op,
    pub value: String,
    pub quoted: bool,
}

impl Matcher {
    pub fn text(&self) -> String {
        let v = if self.quoted || self.value.is_empty() || !self.value.bytes().all(bare_char) { format!("\"{}\"", self.value) } else { self.value.clone() };
        format!("{}{}{}", self.field, self.op.text(), v)
    }
    pub fn matches(&self, r: &RunInfo) -> bool {
        let got = r.field(&self.field);
        let num = |s: &str| parse_num(s);
        match self.op {
            Op::Lt | Op::Le | Op::Gt | Op::Ge => {
                let (Some(g), Some(w)) = (got.and_then(num), num(&self.value)) else { return false };
                match self.op {
                    Op::Lt => g < w,
                    Op::Le => g <= w,
                    Op::Gt => g > w,
                    _ => g >= w,
                }
            }
            Op::Eq | Op::Ne => {
                let g = got.unwrap_or("");
                let eq = match (num(g), num(&self.value)) {
                    (Some(a), Some(b)) if !self.quoted => a == b,
                    _ => g == self.value,
                };
                eq == (self.op == Op::Eq)
            }
            Op::Glob | Op::NotGlob => glob(&self.value, got.unwrap_or("")) == (self.op == Op::Glob),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Entry {
    Sel { metric: String, matchers: Vec<Matcher> },
    Formula { expr: String },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stat {
    MeanSd,
    Mean,
    Median,
    Min,
    Max,
}

impl Stat {
    fn text(self) -> &'static str {
        match self {
            Stat::MeanSd => "mean±sd",
            Stat::Mean => "mean",
            Stat::Median => "median",
            Stat::Min => "min",
            Stat::Max => "max",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Stage {
    Ema(String),
    By(String),
    Agg(Stat, String),
    Split(String),
    X(String),
    Forecast { to: Option<String>, method: Option<String> },
    Log,
    Norm(String),
    Baseline { run: String, ratio: bool },
    Align(String),
    Show(Vec<String>),
    /// Layer × step matrix per (run, layer family): `over` is `layer`;
    /// `rownorm` divides each row by its median; `max` reduces a column bin
    /// by its maximum instead of the mean.
    Heatmap { over: String, rownorm: bool, max: bool },
}

impl Stage {
    pub fn text(&self) -> String {
        match self {
            Stage::Ema(w) => format!("ema {w}"),
            Stage::By(f) => format!("by {f}"),
            Stage::Agg(s, f) => format!("{} over {f}", s.text()),
            Stage::Split(f) => format!("split {f}"),
            Stage::X(f) => format!("x {f}"),
            Stage::Forecast { to, method } => {
                let mut s = String::from("forecast");
                if let Some(t) = to {
                    s += " to ";
                    s += t;
                }
                if let Some(m) = method {
                    s += " ";
                    s += m;
                }
                s
            }
            Stage::Log => "log".into(),
            Stage::Norm(m) => format!("norm {m}"),
            Stage::Baseline { run, ratio } => {
                let r = if run.bytes().all(bare_char) && !run.is_empty() { run.clone() } else { format!("\"{run}\"") };
                format!("baseline {r}{}", if *ratio { " ratio" } else { "" })
            }
            Stage::Align(m) => format!("align {m}"),
            Stage::Show(l) => format!("show {}", l.join(",")),
            Stage::Heatmap { over, rownorm, max } => format!("heatmap over {over}{}{}", if *rownorm { " rownorm" } else { "" }, if *max { " max" } else { "" }),
        }
    }
    fn kind(&self) -> &'static str {
        match self {
            Stage::Ema(_) => "ema",
            Stage::By(_) => "by",
            Stage::Agg(..) => "agg",
            Stage::Split(_) => "split",
            Stage::X(_) => "x",
            Stage::Forecast { .. } => "forecast",
            Stage::Log => "log",
            Stage::Norm(_) => "norm",
            Stage::Baseline { .. } => "baseline",
            Stage::Align(_) => "align",
            Stage::Show(_) => "show",
            Stage::Heatmap { .. } => "heatmap",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Query {
    /// `(letter, entry)` in order.
    pub entries: Vec<(String, Entry)>,
    pub stages: Vec<Stage>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct QueryError {
    pub start: usize,
    pub end: usize,
    pub msg: String,
}

fn qe<T>(start: usize, end: usize, msg: impl Into<String>) -> Result<T, QueryError> {
    Err(QueryError { start, end, msg: msg.into() })
}

const fn bare_char(c: u8) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, b'_' | b'.' | b'/' | b':' | b'-' | b'*' | b'?' | b'+')
}

/// Number with an optional `k`, `M`, `G`/`B` suffix (`10k` = 10000).
pub fn parse_num(s: &str) -> Option<f64> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    let (body, mul) = match s.as_bytes()[s.len() - 1] {
        b'k' | b'K' => (&s[..s.len() - 1], 1e3),
        b'M' => (&s[..s.len() - 1], 1e6),
        b'G' | b'B' => (&s[..s.len() - 1], 1e9),
        _ => (s, 1.0),
    };
    let v: f64 = body.parse().ok()?;
    let v = v * mul;
    v.is_finite().then_some(v)
}

/// Anchored glob: `*` any run, `?` one character, `|` separates alternatives.
pub fn glob(pat: &str, s: &str) -> bool {
    fn one(p: &[u8], s: &[u8]) -> bool {
        let (mut pi, mut si, mut star, mut mark) = (0, 0, usize::MAX, 0);
        while si < s.len() {
            if pi < p.len() && (p[pi] == b'?' || p[pi] == s[si]) {
                pi += 1;
                si += 1;
            } else if pi < p.len() && p[pi] == b'*' {
                star = pi;
                mark = si;
                pi += 1;
            } else if star != usize::MAX {
                pi = star + 1;
                mark += 1;
                si = mark;
            } else {
                return false;
            }
        }
        while pi < p.len() && p[pi] == b'*' {
            pi += 1;
        }
        pi == p.len()
    }
    pat.split('|').any(|p| one(p.as_bytes(), s.as_bytes()))
}

// ------------------------------------------------------------------ parser

struct Lx<'a> {
    s: &'a str,
    i: usize,
}

impl<'a> Lx<'a> {
    fn b(&self) -> &'a [u8] {
        self.s.as_bytes()
    }
    fn ws(&mut self) {
        while self.i < self.s.len() && self.b()[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }
    fn peek(&mut self) -> Option<u8> {
        self.ws();
        self.b().get(self.i).copied()
    }
    fn eat(&mut self, c: u8) -> bool {
        if self.peek() == Some(c) {
            self.i += 1;
            true
        } else {
            false
        }
    }
    fn word(&mut self) -> (String, usize, usize) {
        self.ws();
        let st = self.i;
        while self.i < self.s.len() && bare_char(self.b()[self.i]) {
            self.i += 1;
        }
        (self.s[st..self.i].to_string(), st, self.i)
    }
    /// A value: quoted string or bare word. Returns (text, quoted, start, end).
    fn value(&mut self) -> Result<(String, bool, usize, usize), QueryError> {
        self.ws();
        let st = self.i;
        if self.peek() == Some(b'"') {
            let rest = &self.s[st + 1..];
            let Some(p) = rest.find('"') else { return qe(st, self.s.len(), "unterminated quote") };
            self.i = st + 1 + p + 1;
            return Ok((rest[..p].to_string(), true, st, self.i));
        }
        let (w, a, b) = self.word();
        if w.is_empty() {
            let end = (a + 1).min(self.s.len()).max(a);
            return qe(a, end, "expected a value");
        }
        Ok((w, false, a, b))
    }
}

fn parse_matcher(lx: &mut Lx) -> Result<Matcher, QueryError> {
    lx.ws();
    let st = lx.i;
    while lx.i < lx.s.len() && (lx.b()[lx.i].is_ascii_alphanumeric() || matches!(lx.b()[lx.i], b'_' | b'.' | b'-')) {
        lx.i += 1;
    }
    let field = lx.s[st..lx.i].to_string();
    if field.is_empty() {
        return qe(st, (st + 1).min(lx.s.len()), "expected a field name (run, source, or a config key)");
    }
    lx.ws();
    let rest = &lx.s[lx.i..];
    let (op, n) = if rest.starts_with("=~") {
        (Op::Glob, 2)
    } else if rest.starts_with("!~") {
        (Op::NotGlob, 2)
    } else if rest.starts_with("!=") {
        (Op::Ne, 2)
    } else if rest.starts_with("<=") {
        (Op::Le, 2)
    } else if rest.starts_with(">=") {
        (Op::Ge, 2)
    } else if rest.starts_with("==") {
        (Op::Eq, 2)
    } else if rest.starts_with('=') {
        (Op::Eq, 1)
    } else if rest.starts_with('<') {
        (Op::Lt, 1)
    } else if rest.starts_with('>') {
        (Op::Gt, 1)
    } else {
        return qe(lx.i, (lx.i + 1).min(lx.s.len()), format!("expected an operator after '{field}' (= != =~ !~ < <= > >=)"));
    };
    lx.i += n;
    let (value, quoted, a, b) = lx.value()?;
    if matches!(op, Op::Lt | Op::Le | Op::Gt | Op::Ge) && parse_num(&value).is_none() {
        return qe(a, b, format!("'{value}' is not a number"));
    }
    Ok(Matcher { field, op, value, quoted })
}

fn split_top(s: &str, sep: u8) -> Vec<(usize, usize)> {
    // split on `sep` outside quotes, braces and parentheses
    let (mut out, mut depth, mut q, mut st) = (Vec::new(), 0i32, false, 0);
    for (i, &c) in s.as_bytes().iter().enumerate() {
        match c {
            b'"' => q = !q,
            b'{' | b'(' if !q => depth += 1,
            b'}' | b')' if !q => depth -= 1,
            _ if c == sep && !q && depth == 0 => {
                out.push((st, i));
                st = i + 1;
            }
            _ => {}
        }
    }
    out.push((st, s.len()));
    out
}

fn parse_entry(src: &str, a: usize, b: usize) -> Result<Option<(Option<String>, Entry)>, QueryError> {
    let text = &src[a..b];
    let t = text.trim();
    if t.is_empty() {
        return Ok(None);
    }
    let off = a + (text.len() - text.trim_start().len());
    // formula: name '=' expr (but not '=~' / '==')
    let nb = t.bytes().take_while(|c| c.is_ascii_alphanumeric() || *c == b'_').count();
    if nb > 0 && t.as_bytes()[0].is_ascii_alphabetic() {
        let rest = t[nb..].trim_start();
        if rest.starts_with('=') && !rest.starts_with("=~") && !rest.starts_with("==") {
            let expr = rest[1..].trim();
            if expr.is_empty() {
                return qe(off, off + t.len(), "formula needs an expression after '='");
            }
            return Ok(Some((Some(t[..nb].to_string()), Entry::Formula { expr: expr.to_string() })));
        }
    }
    let mut lx = Lx { s: src, i: off };
    let (metric, ms, me) = lx.word();
    let mut matchers = Vec::new();
    if lx.eat(b'{') {
        if !lx.eat(b'}') {
            loop {
                matchers.push(parse_matcher(&mut lx)?);
                if lx.eat(b',') {
                    continue;
                }
                if lx.eat(b'}') {
                    break;
                }
                let p = lx.i.min(b);
                return qe(p, (p + 1).min(src.len()), "expected ',' or '}'");
            }
        }
    } else if metric.is_empty() {
        return qe(ms, (me + 1).min(b), "expected a metric name or {matchers}");
    }
    lx.ws();
    if lx.i < b {
        return qe(lx.i, b, "unexpected text in selector (separate selectors with ';')");
    }
    Ok(Some((None, Entry::Sel { metric: if metric.is_empty() { "*".into() } else { metric }, matchers })))
}

fn parse_stages(src: &str, a: usize, b: usize, out: &mut Vec<Stage>) -> Result<(), QueryError> {
    let mut lx = Lx { s: &src[..b], i: a };
    loop {
        lx.ws();
        if lx.i >= b {
            return Ok(());
        }
        let (kw, ka, kb) = lx.word();
        // "mean±sd": '±' is not a bare char; take it by hand
        let kw = if kw == "mean" && src[lx.i..b].starts_with('±') {
            lx.i += '±'.len_utf8();
            let (w, _, _) = lx.word();
            if w != "sd" {
                return qe(ka, lx.i, "expected mean±sd");
            }
            "mean±sd".to_string()
        } else {
            kw
        };
        if kw.is_empty() {
            let ch = src[lx.i..].chars().next().map_or(1, |c| c.len_utf8());
            return qe(lx.i, lx.i + ch, "expected a stage (ema, by, split, mean±sd over, x, forecast, log, norm, baseline, align, show, heatmap)");
        }
        let need = |lx: &mut Lx, what: &str| -> Result<String, QueryError> {
            let (w, a, _) = lx.word();
            if w.is_empty() {
                return qe(ka, a.max(kb), format!("{kw} needs {what}"));
            }
            Ok(w)
        };
        let st = match kw.as_str() {
            "ema" => {
                let (w, a, e) = lx.word();
                match parse_num(&w) {
                    Some(v) if (0.0..1.0).contains(&v) => Stage::Ema(w),
                    _ => return qe(a.min(ka), e.max(kb), "ema needs a weight in [0, 1), e.g. ema .9"),
                }
            }
            "by" => Stage::By(need(&mut lx, "a field")?),
            "split" => Stage::Split(need(&mut lx, "a field (metric, run, or a config key)")?),
            "x" => Stage::X(need(&mut lx, "an axis: step, time, or a metric")?),
            "log" => Stage::Log,
            "norm" => {
                let w = need(&mut lx, "z, minmax or first")?;
                if !matches!(w.as_str(), "z" | "minmax" | "first") {
                    return qe(ka, lx.i, "norm takes z, minmax or first");
                }
                Stage::Norm(w)
            }
            "align" => {
                let w = need(&mut lx, "intersection, union or exact")?;
                if !matches!(w.as_str(), "intersection" | "union" | "exact") {
                    return qe(ka, lx.i, "align takes intersection, union or exact");
                }
                Stage::Align(w)
            }
            "mean±sd" | "mean+-sd" | "mean" | "median" | "min" | "max" => {
                let stat = match kw.as_str() {
                    "mean±sd" | "mean+-sd" => Stat::MeanSd,
                    "mean" => Stat::Mean,
                    "median" => Stat::Median,
                    "min" => Stat::Min,
                    _ => Stat::Max,
                };
                let (o, _, _) = lx.word();
                if o != "over" {
                    return qe(ka, lx.i, format!("expected '{kw} over <field>', e.g. mean±sd over seed"));
                }
                Stage::Agg(stat, need(&mut lx, "a field")?)
            }
            "forecast" => {
                let mut to = None;
                let mut method = None;
                let save = lx.i;
                let (w, _, _) = lx.word();
                if w == "to" {
                    let (n, a, e) = lx.word();
                    if parse_num(&n).is_none() {
                        return qe(a, e.max(a + 1), "forecast to needs a step, e.g. forecast to 10k");
                    }
                    to = Some(n);
                } else {
                    lx.i = save;
                }
                let save = lx.i;
                let (w, _, _) = lx.word();
                if matches!(w.as_str(), "auto" | "ridge" | "powerlaw" | "last") {
                    method = Some(w);
                } else {
                    lx.i = save;
                }
                Stage::Forecast { to, method }
            }
            "baseline" => {
                let (v, _, _, _) = lx.value()?;
                let save = lx.i;
                let (w, _, _) = lx.word();
                let ratio = match w.as_str() {
                    "ratio" => true,
                    "diff" => false,
                    _ => {
                        lx.i = save;
                        false
                    }
                };
                Stage::Baseline { run: v, ratio }
            }
            "heatmap" => {
                let (mut over, mut rownorm, mut max) = (String::from("layer"), false, false);
                loop {
                    let save = lx.i;
                    let (w, a, e) = lx.word();
                    match w.as_str() {
                        "over" => {
                            let (f, fa, fe) = lx.word();
                            if f != "layer" {
                                return qe(fa.min(a), fe.max(e), "heatmap over takes layer (per-layer metrics named layers/<i>/<stat>)");
                            }
                            over = f;
                        }
                        "rownorm" => rownorm = true,
                        "max" => max = true,
                        _ => {
                            lx.i = save;
                            break;
                        }
                    }
                }
                Stage::Heatmap { over, rownorm, max }
            }
            "show" => {
                let mut l = Vec::new();
                loop {
                    let (w, _, _) = lx.word();
                    if w.is_empty() {
                        break;
                    }
                    l.extend(w.split(',').filter(|s| !s.is_empty()).map(String::from));
                    if !lx.eat(b',') {
                        break;
                    }
                }
                if l.is_empty() {
                    return qe(ka, kb, "show needs letters, e.g. show c");
                }
                Stage::Show(l)
            }
            _ => return qe(ka, kb, format!("unknown stage '{kw}'")),
        };
        out.push(st);
    }
}

/// Parse a panel query.
pub fn parse(src: &str) -> Result<Query, QueryError> {
    let segs = split_top(src, b'|');
    let (ea, eb) = segs[0];
    let mut raw = Vec::new();
    for (a, b) in split_top(&src[ea..eb], b';') {
        if let Some(e) = parse_entry(src, ea + a, ea + b)? {
            raw.push(e);
        }
    }
    if raw.is_empty() {
        return qe(0, src.len(), "empty query: give a metric, e.g. loss or loss{run=\"x\"}");
    }
    let taken: Vec<String> = raw.iter().filter_map(|(n, _)| n.clone()).collect();
    let mut entries = Vec::new();
    let mut k = 0;
    for (n, e) in raw {
        let letter = match n {
            Some(n) => n,
            None => loop {
                let l = letter(k);
                k += 1;
                if !taken.contains(&l) {
                    break l;
                }
            },
        };
        entries.push((letter, e));
    }
    let mut stages = Vec::new();
    for &(a, b) in &segs[1..] {
        if src[a..b].trim().is_empty() {
            return qe(a.saturating_sub(1), b, "empty stage after '|'");
        }
        parse_stages(src, a, b, &mut stages)?;
    }
    Ok(Query { entries, stages })
}

pub fn letter(i: usize) -> String {
    let mut s = Vec::new();
    let mut i = i + 1;
    while i > 0 {
        s.push(b'a' + ((i - 1) % 26) as u8);
        i = (i - 1) / 26;
    }
    s.reverse();
    String::from_utf8(s).unwrap()
}

// ------------------------------------------------------------------ chips and canonical text

#[derive(Clone, Debug, PartialEq)]
pub struct Chip {
    /// metric | matcher | formula | stage kind (ema, by, agg, ...)
    pub kind: String,
    pub text: String,
    pub entry: Option<usize>,
    pub letter: String,
}

pub fn chips(q: &Query) -> Vec<Chip> {
    let mut out = Vec::new();
    for (i, (l, e)) in q.entries.iter().enumerate() {
        match e {
            Entry::Sel { metric, matchers } => {
                out.push(Chip { kind: "metric".into(), text: metric.clone(), entry: Some(i), letter: l.clone() });
                for m in matchers {
                    out.push(Chip { kind: "matcher".into(), text: m.text(), entry: Some(i), letter: l.clone() });
                }
            }
            Entry::Formula { expr } => out.push(Chip { kind: "formula".into(), text: format!("{l} = {expr}"), entry: Some(i), letter: l.clone() }),
        }
    }
    for s in &q.stages {
        out.push(Chip { kind: s.kind().into(), text: s.text(), entry: None, letter: String::new() });
    }
    out
}

/// Text from chip texts (an empty chip text drops the chip; a selector
/// whose metric chip is dropped is dropped with its matchers).
pub fn compose(chips: &[Chip]) -> String {
    let mut ents: Vec<String> = Vec::new();
    let mut cur: Option<(usize, String, Vec<String>, bool)> = None;
    let flush = |cur: &mut Option<(usize, String, Vec<String>, bool)>, ents: &mut Vec<String>| {
        if let Some((_, m, ms, keep)) = cur.take() {
            if keep {
                ents.push(if ms.is_empty() { m } else { format!("{}{{{}}}", if m == "*" { "" } else { &m }, ms.join(", ")) });
            }
        }
    };
    let mut stages = Vec::new();
    for c in chips {
        match (c.kind.as_str(), c.entry) {
            ("metric", Some(i)) => {
                flush(&mut cur, &mut ents);
                cur = Some((i, c.text.trim().to_string(), Vec::new(), !c.text.trim().is_empty()));
            }
            ("matcher", Some(i)) => {
                if let Some((ci, _, ms, _)) = cur.as_mut() {
                    if *ci == i && !c.text.trim().is_empty() {
                        ms.push(c.text.trim().to_string());
                    }
                }
            }
            ("formula", _) => {
                flush(&mut cur, &mut ents);
                if !c.text.trim().is_empty() {
                    ents.push(c.text.trim().to_string());
                }
            }
            _ => {
                if !c.text.trim().is_empty() {
                    stages.push(c.text.trim().to_string());
                }
            }
        }
    }
    flush(&mut cur, &mut ents);
    let mut s = ents.join("; ");
    for st in stages {
        s += " | ";
        s += &st;
    }
    s
}

/// Canonical text of a query (parse → chips → compose).
pub fn canonical(q: &Query) -> String {
    compose(&chips(q))
}

/// Replace chip `idx` of `src` with `text` (empty removes it) and return
/// the new canonical query, or the parse error of the result.
pub fn replace_chip(src: &str, idx: usize, text: &str) -> Result<String, QueryError> {
    let q = parse(src)?;
    let mut cs = chips(&q);
    if idx >= cs.len() {
        return qe(0, 0, "no such chip");
    }
    cs[idx].text = text.to_string();
    let s = compose(&cs);
    let q2 = parse(&s)?;
    Ok(canonical(&q2))
}

// ------------------------------------------------------------------ resolution

/// One stream to load: run index into the catalog and a metric name.
#[derive(Clone, Debug, PartialEq)]
pub struct Stream {
    pub run: usize,
    pub metric: String,
}

#[derive(Clone, Debug, Default)]
pub struct Plan {
    /// Streams in the order the evaluator expects their data.
    pub streams: Vec<Stream>,
    /// Per selector entry: indices into `streams`.
    pub sel: Vec<(usize, Vec<usize>)>,
    /// Per run: index into `streams` of its x stream.
    pub x: Vec<Option<usize>>,
    pub x_kind: XKind,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub enum XKind {
    #[default]
    Step,
    Time,
    Metric(String),
}

fn stage<'a, T>(q: &'a Query, f: impl Fn(&'a Stage) -> Option<T>) -> Option<T> {
    q.stages.iter().rev().find_map(f)
}

fn x_metric(r: &RunInfo, want: &str) -> Option<String> {
    if r.metrics.iter().any(|m| m == want) {
        return Some(want.to_string());
    }
    let w = want.trim_end_matches('s');
    r.metrics.iter().find(|m| m.contains(w)).cloned()
}

/// Which streams a query needs.
pub fn plan(q: &Query, cat: &Catalog) -> Plan {
    let mut p = Plan { x: vec![None; cat.runs.len()], ..Default::default() };
    let idx = |p: &mut Plan, run: usize, metric: &str| -> usize {
        if let Some(i) = p.streams.iter().position(|s| s.run == run && s.metric == metric) {
            return i;
        }
        p.streams.push(Stream { run, metric: metric.to_string() });
        p.streams.len() - 1
    };
    for (ei, (l, e)) in q.entries.iter().enumerate() {
        let Entry::Sel { metric, matchers } = e else { continue };
        let mut v = Vec::new();
        for (ri, r) in cat.runs.iter().enumerate() {
            if !matchers.iter().all(|m| m.matches(r)) {
                continue;
            }
            for m in &r.metrics {
                if glob(metric, m) {
                    v.push(idx(&mut p, ri, m));
                }
            }
        }
        if v.is_empty() {
            p.warnings.push(format!("{l}: nothing matches {}", chips(&Query { entries: vec![(l.clone(), e.clone())], stages: vec![] }).iter().map(|c| c.text.clone()).collect::<Vec<_>>().join(" ")));
        }
        p.sel.push((ei, v));
    }
    let xk = stage(q, |s| if let Stage::X(f) = s { Some(f.clone()) } else { None }).unwrap_or_else(|| "step".into());
    p.x_kind = match xk.as_str() {
        "step" => XKind::Step,
        "time" | "t" => XKind::Time,
        _ => XKind::Metric(xk.clone()),
    };
    if p.x_kind != XKind::Step {
        let runs: Vec<usize> = {
            let mut r: Vec<usize> = p.streams.iter().map(|s| s.run).collect();
            r.sort();
            r.dedup();
            r
        };
        for ri in runs {
            let want = if p.x_kind == XKind::Time { Some("elapsed_min".to_string()) } else { x_metric(&cat.runs[ri], &xk) };
            match want.filter(|m| cat.runs[ri].metrics.contains(m)) {
                Some(m) => p.x[ri] = Some(idx(&mut p, ri, &m)),
                None => p.warnings.push(format!("{} has no {}", cat.runs[ri].id, if p.x_kind == XKind::Time { "elapsed_min (time axis)".to_string() } else { xk.clone() })),
            }
        }
    }
    p
}

// ------------------------------------------------------------------ evaluation

#[derive(Clone, Debug, Default)]
pub struct Trace {
    pub letter: String,
    pub label: String,
    pub metric: String,
    /// Catalog run indices (one, or the members of an aggregate).
    pub runs: Vec<usize>,
    pub steps: Vec<f64>,
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,
    pub raw: Vec<f64>,
    pub lo: Vec<f64>,
    pub hi: Vec<f64>,
    pub band: bool,
    pub cell: String,
    pub color: String,
}

#[derive(Clone, Debug, Default)]
pub struct Output {
    pub traces: Vec<Trace>,
    pub cells: Vec<String>,
    pub warnings: Vec<String>,
    pub by: Option<String>,
    pub by_numeric: bool,
    pub ylog: bool,
    pub forecast: Option<(Option<f64>, String)>,
    pub x_label: String,
    pub downsampled: usize,
    /// `heatmap over layer`: one matrix per (run, layer family).
    pub heatmaps: Vec<LayerHeatmap>,
}

/// Columns per heatmap: the union of steps, block-reduced past this.
pub const HEATMAP_COLS: usize = 400;

#[derive(Clone, Debug, Default)]
pub struct LayerHeatmap {
    /// Catalog run index.
    pub run: usize,
    /// Family template (`layers/*/grad_norm`).
    pub family: String,
    /// Row `r` is layer `layers[r]`, logged as metric `metrics[r]`.
    pub layers: Vec<u32>,
    pub metrics: Vec<String>,
    pub m: crate::layers::Matrix,
}

/// Group single-run traces into layer families per run and build their
/// matrices ([`crate::layers::matrix`]); families need 2 or more layers.
pub fn heatmaps(traces: &[Trace], rownorm: bool, max: bool, max_cols: usize) -> Vec<LayerHeatmap> {
    let mut groups: Vec<(usize, String, Vec<(u32, usize)>)> = Vec::new();
    for (ti, t) in traces.iter().enumerate() {
        let (1, Some((fam, layer))) = (t.runs.len(), crate::layers::layer_index(&t.metric)) else { continue };
        match groups.iter_mut().find(|g| g.0 == t.runs[0] && g.1 == fam) {
            Some(g) => g.2.push((layer, ti)),
            None => groups.push((t.runs[0], fam, vec![(layer, ti)])),
        }
    }
    let mut out = Vec::new();
    for (run, family, mut rows) in groups {
        rows.sort_by_key(|r| r.0);
        rows.dedup_by_key(|r| r.0);
        if rows.len() < 2 {
            continue;
        }
        let data: Vec<(&[f64], &[f64])> = rows.iter().map(|&(_, ti)| (&traces[ti].steps[..], &traces[ti].raw[..])).collect();
        let m = crate::layers::matrix(&data, max_cols, rownorm, max);
        out.push(LayerHeatmap { run, family, layers: rows.iter().map(|r| r.0).collect(), metrics: rows.iter().map(|&(_, ti)| traces[ti].metric.clone()).collect(), m });
    }
    out
}

fn run_label(cat: &Catalog, runs: &[usize]) -> String {
    if runs.len() == 1 {
        cat.runs[runs[0]].id.clone()
    } else {
        let ids: Vec<&str> = runs.iter().map(|&r| cat.runs[r].id.as_str()).collect();
        format!("{} (+{})", ids[0], ids.len() - 1)
    }
}

/// Aggregate member series (already per-run) on one grid.
pub fn aggregate(members: &[(&[f64], &[f64])], stat: Stat, mode: Align) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let (grid, vals) = expr::align(members, mode);
    let n = grid.len();
    let (mut c, mut lo, mut hi) = (vec![f64::NAN; n], vec![f64::NAN; n], vec![f64::NAN; n]);
    for i in 0..n {
        let mut col: Vec<f64> = vals.iter().map(|v| v[i]).filter(|x| x.is_finite()).collect();
        if col.is_empty() {
            continue;
        }
        let k = col.len() as f64;
        let mean = col.iter().sum::<f64>() / k;
        match stat {
            Stat::MeanSd | Stat::Mean => {
                c[i] = mean;
                if stat == Stat::MeanSd && col.len() > 1 {
                    let sd = sqrt(col.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (k - 1.0));
                    lo[i] = mean - sd;
                    hi[i] = mean + sd;
                }
            }
            Stat::Median => {
                col.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let m = col.len();
                c[i] = if m % 2 == 1 { col[m / 2] } else { (col[m / 2 - 1] + col[m / 2]) / 2.0 };
                lo[i] = col[0];
                hi[i] = col[m - 1];
            }
            Stat::Min => c[i] = col.iter().copied().fold(f64::INFINITY, f64::min),
            Stat::Max => c[i] = col.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        }
    }
    (grid, c, lo, hi)
}

/// Evaluate a query. `data[i]` is `(steps, values)` of `plan.streams[i]`.
pub fn eval(q: &Query, cat: &Catalog, plan: &Plan, data: &[(&[f64], &[f64])], max_points: usize) -> Output {
    let mut out = Output { warnings: plan.warnings.clone(), ..Default::default() };
    let mode = Align::parse(&stage(q, |s| if let Stage::Align(m) = s { Some(m.clone()) } else { None }).unwrap_or_default());
    let get = |i: usize| -> (Vec<f64>, Vec<f64>) { if i < data.len() { expr::clean(data[i].0, data[i].1) } else { (Vec::new(), Vec::new()) } };
    // selectors
    let mut traces: Vec<Trace> = Vec::new();
    for (ei, v) in &plan.sel {
        let l = &q.entries[*ei].0;
        for &si in v {
            let s = &plan.streams[si];
            let (st, va) = get(si);
            traces.push(Trace { letter: l.clone(), label: format!("{}:{}", cat.runs[s.run].id, s.metric), metric: s.metric.clone(), runs: vec![s.run], steps: st, raw: va, ..Default::default() });
        }
    }
    // formulas, per run
    for (l, e) in &q.entries {
        let Entry::Formula { expr: src } = e else { continue };
        let letters: Vec<String> = traces.iter().map(|t| t.letter.clone()).fold(Vec::new(), |mut a, x| {
            if !a.contains(&x) {
                a.push(x)
            }
            a
        });
        let names: Vec<&str> = letters.iter().map(|s| s.as_str()).collect();
        let refs = match expr::refs(src, &names) {
            Ok(r) => r,
            Err(er) => {
                out.warnings.push(format!("{l} = {src}: {} (at '{}')", er.msg, src.get(er.start..er.end).unwrap_or("")));
                continue;
            }
        };
        if refs.is_empty() {
            out.warnings.push(format!("{l} = {src}: refers to no letter"));
            continue;
        }
        // per letter: traces by run
        let sets: Vec<Vec<&Trace>> = refs.iter().map(|r| traces.iter().filter(|t| &t.letter == r).collect()).collect();
        if let Some(i) = sets.iter().position(|s| {
            let mut runs: Vec<usize> = s.iter().map(|t| t.runs[0]).collect();
            let n = runs.len();
            runs.sort();
            runs.dedup();
            runs.len() != n
        }) {
            out.warnings.push(format!("{l} = {src}: {} matches several metrics per run; give it one metric", refs[i]));
            continue;
        }
        let mut runs: Vec<usize> = sets.iter().filter(|s| s.len() > 1).flat_map(|s| s.iter().map(|t| t.runs[0])).collect();
        runs.sort();
        runs.dedup();
        if runs.is_empty() {
            runs = sets.iter().flat_map(|s| s.iter().map(|t| t.runs[0])).take(1).collect();
        }
        let mut made = Vec::new();
        let mut first_err = None;
        for &r in &runs {
            let pick: Option<Vec<&Trace>> = sets.iter().map(|s| if s.len() == 1 { Some(s[0]) } else { s.iter().copied().find(|t| t.runs[0] == r) }).collect();
            let Some(pick) = pick else { continue };
            let d: Vec<(&[f64], &[f64])> = pick.iter().map(|t| (&t.steps[..], &t.raw[..])).collect();
            let rn: Vec<&str> = refs.iter().map(|s| s.as_str()).collect();
            match expr::eval(src, &rn, &d, mode) {
                Ok((st, va)) => made.push(Trace { letter: l.clone(), label: format!("{}: {l} = {src}", cat.runs[r].id), metric: format!("{l} = {src}"), runs: vec![r], steps: st, raw: va, ..Default::default() }),
                Err(er) => {
                    first_err.get_or_insert(format!("{l} = {src} on {}: {}", cat.runs[r].id, er.msg));
                }
            }
        }
        if made.is_empty() {
            out.warnings.push(first_err.unwrap_or(format!("{l} = {src}: no run has all of {}", refs.join(", "))));
        } else if let Some(e) = first_err {
            out.warnings.push(e);
        }
        traces.extend(made);
    }
    // show
    if let Some(show) = stage(q, |s| if let Stage::Show(l) = s { Some(l.clone()) } else { None }) {
        traces.retain(|t| show.contains(&t.letter));
    }
    // baseline
    if let Some((bg, ratio)) = stage(q, |s| if let Stage::Baseline { run, ratio } = s { Some((run.clone(), *ratio)) } else { None }) {
        let is_base = |r: usize| glob(&bg, &cat.runs[r].id) || cat.runs[r].field("name").is_some_and(|n| glob(&bg, n));
        let (base, rest): (Vec<Trace>, Vec<Trace>) = traces.into_iter().partition(|t| is_base(t.runs[0]));
        if base.is_empty() {
            out.warnings.push(format!("baseline {bg}: no plotted run matches"));
            traces = rest;
        } else {
            traces = Vec::new();
            for mut t in rest {
                let Some(b) = base.iter().find(|b| b.letter == t.letter && b.metric == t.metric) else {
                    out.warnings.push(format!("baseline: no {} for {}", t.metric, t.label));
                    continue;
                };
                let (g, v) = expr::align(&[(&t.steps, &t.raw), (&b.steps, &b.raw)], mode);
                if g.is_empty() {
                    out.warnings.push(format!("baseline: {} does not overlap the baseline", t.label));
                    continue;
                }
                t.raw = v[0].iter().zip(&v[1]).map(|(&x, &y)| if ratio { if y == 0.0 { f64::NAN } else { x / y } } else { x - y }).collect();
                t.steps = g;
                t.label = format!("{} {} {}", t.label, if ratio { "/" } else { "-" }, cat.runs[b.runs[0]].id);
                traces.push(t);
            }
        }
    }
    // norm
    if let Some(m) = stage(q, |s| if let Stage::Norm(m) = s { Some(m.clone()) } else { None }) {
        for t in &mut traces {
            t.raw = expr::normalize(&t.raw, &m);
        }
    }
    // layer heatmaps (from the per-run raw values, before aggregation and x mapping)
    if let Some((rownorm, max)) = stage(q, |s| if let Stage::Heatmap { rownorm, max, .. } = s { Some((*rownorm, *max)) } else { None }) {
        out.heatmaps = heatmaps(&traces, rownorm, max, HEATMAP_COLS);
        if out.heatmaps.is_empty() {
            out.warnings.push("heatmap: no family of per-layer metrics here (name them layers/<i>/<stat>, e.g. layers/0/grad_norm)".into());
        }
    }
    // aggregation
    if let Some((stat, over)) = stage(q, |s| if let Stage::Agg(st, f) = s { Some((*st, f.clone())) } else { None }) {
        let mut groups: Vec<(String, Vec<Trace>)> = Vec::new();
        for t in traces {
            let r = &cat.runs[t.runs[0]];
            let mut key: Vec<String> = r.config.iter().filter(|(k, _)| *k != over).map(|(k, v)| format!("{k}={v}")).collect();
            key.sort();
            let k = format!("{}\u{1}{}\u{1}{}", t.letter, t.metric, key.join("\u{2}"));
            match groups.iter_mut().find(|g| g.0 == k) {
                Some(g) => g.1.push(t),
                None => groups.push((k, vec![t])),
            }
        }
        traces = Vec::new();
        for (_, g) in groups {
            let runs: Vec<usize> = g.iter().map(|t| t.runs[0]).collect();
            let members: Vec<(&[f64], &[f64])> = g.iter().map(|t| (&t.steps[..], &t.raw[..])).collect();
            let (grid, c, lo, hi) = aggregate(&members, stat, mode);
            let n = g.len();
            if grid.is_empty() {
                out.warnings.push(format!("{} over {over}: members do not overlap", run_label(cat, &runs)));
                continue;
            }
            let t0 = &g[0];
            let what = t0.metric.clone();
            traces.push(Trace {
                letter: t0.letter.clone(),
                label: if n > 1 { format!("{}:{what} · {} over {over} (n={n})", run_label(cat, &runs), stat.text()) } else { t0.label.clone() },
                metric: t0.metric.clone(),
                runs,
                steps: grid,
                raw: c,
                band: n > 1 && matches!(stat, Stat::MeanSd | Stat::Median),
                lo,
                hi,
                ..Default::default()
            });
        }
    }
    // x
    for t in &mut traces {
        match &plan.x_kind {
            XKind::Step => t.xs = t.steps.clone(),
            k => {
                let Some(xi) = plan.x[t.runs[0]] else {
                    t.xs = vec![f64::NAN; t.steps.len()];
                    continue;
                };
                let (xs, xv) = get(xi);
                let scale = if *k == XKind::Time { 60.0 } else { 1.0 };
                let xv: Vec<f64> = xv.iter().map(|v| v * scale).collect();
                // x value at each of the trace's steps (aligned, never extrapolated)
                let (g, v) = expr::align(&[(&xs, &xv), (&t.steps, &t.raw)], mode);
                let band = |v: &[f64]| -> Vec<f64> { if t.band { g.iter().map(|&s| expr::interp_at(&t.steps, v, s)).collect() } else { Vec::new() } };
                let (lo, hi) = (band(&t.lo), band(&t.hi));
                t.xs = v[0].clone();
                t.raw = v[1].clone();
                t.steps = g;
                if t.band {
                    t.lo = lo;
                    t.hi = hi;
                }
            }
        }
        if !t.band {
            t.lo = vec![f64::NAN; t.raw.len()];
            t.hi = vec![f64::NAN; t.raw.len()];
        }
    }
    traces.retain(|t| t.xs.iter().zip(&t.raw).any(|(x, y)| x.is_finite() && y.is_finite()));
    // ema (display only)
    let w = stage(q, |s| if let Stage::Ema(w) = s { parse_num(w) } else { None }).unwrap_or(0.0);
    for t in &mut traces {
        t.ys = if w > 0.0 { crate::ema(&t.raw, w) } else { t.raw.clone() };
    }
    // downsample
    if max_points > 0 && matches!(plan.x_kind, XKind::Step | XKind::Time) {
        for t in &mut traces {
            if t.xs.len() > max_points {
                out.downsampled = out.downsampled.max(t.xs.len());
                let (xs, ys, _, _) = crate::downsample_series(&t.xs, &t.ys, max_points);
                let raw = crate::downsample_series(&t.xs, &t.raw, max_points).1;
                let lo = crate::downsample_series(&t.xs, &t.lo, max_points).1;
                let hi = crate::downsample_series(&t.xs, &t.hi, max_points).1;
                let st = crate::downsample_series(&t.xs, &t.steps, max_points).1;
                t.xs = xs;
                t.ys = ys;
                t.raw = raw;
                t.lo = lo;
                t.hi = hi;
                t.steps = st;
            }
        }
    }
    // split cells and colour values
    let fieldv = |t: &Trace, f: &str| -> String {
        match f {
            "metric" => t.metric.clone(),
            "letter" => t.letter.clone(),
            _ => {
                let vals: Vec<&str> = t.runs.iter().map(|&r| cat.runs[r].field(f).unwrap_or("")).collect();
                if vals.iter().all(|v| *v == vals[0]) { vals[0].to_string() } else { String::from("(mixed)") }
            }
        }
    };
    if let Some(f) = stage(q, |s| if let Stage::Split(f) = s { Some(f.clone()) } else { None }) {
        for t in &mut traces {
            t.cell = fieldv(t, &f);
        }
    }
    for t in &traces {
        if !out.cells.contains(&t.cell) {
            out.cells.push(t.cell.clone());
        }
    }
    if let Some(f) = stage(q, |s| if let Stage::By(f) = s { Some(f.clone()) } else { None }) {
        for t in &mut traces {
            t.color = fieldv(t, &f);
        }
        out.by_numeric = traces.iter().all(|t| parse_num(&t.color).is_some());
        out.by = Some(f);
    }
    out.ylog = q.stages.iter().any(|s| *s == Stage::Log);
    out.forecast = stage(q, |s| if let Stage::Forecast { to, method } = s { Some((to.as_deref().and_then(parse_num), method.clone().unwrap_or_else(|| "auto".into()))) } else { None });
    out.x_label = match &plan.x_kind {
        XKind::Step => "step".into(),
        XKind::Time => "time".into(),
        XKind::Metric(m) => m.clone(),
    };
    out.traces = traces;
    out
}

// ------------------------------------------------------------------ JSON (for the wasm boundary)

pub fn js_str(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn js_list(v: &[String]) -> String {
    format!("[{}]", v.iter().map(|s| js_str(s)).collect::<Vec<_>>().join(","))
}

pub fn chips_json(q: &Query) -> String {
    let cs = chips(q);
    let items: Vec<String> = cs
        .iter()
        .map(|c| format!("{{\"kind\":{},\"text\":{},\"entry\":{},\"letter\":{}}}", js_str(&c.kind), js_str(&c.text), c.entry.map_or("null".to_string(), |e| e.to_string()), js_str(&c.letter)))
        .collect();
    format!("[{}]", items.join(","))
}

/// Metadata of an [`Output`]; the numbers travel separately as one flat
/// array where trace `i` occupies `off .. off + 6n`: xs, ys, raw, lo, hi, steps.
pub fn output_json(o: &Output, cat: &Catalog) -> (String, Vec<f64>) {
    let mut flat = Vec::new();
    let mut tr = Vec::new();
    for t in &o.traces {
        let off = flat.len();
        let n = t.xs.len();
        for v in [&t.xs, &t.ys, &t.raw, &t.lo, &t.hi, &t.steps] {
            flat.extend(v.iter().copied().chain(core::iter::repeat(f64::NAN)).take(n));
        }
        let runs: Vec<String> = t.runs.iter().map(|&r| cat.runs[r].id.clone()).collect();
        tr.push(format!(
            "{{\"letter\":{},\"label\":{},\"metric\":{},\"runs\":{},\"band\":{},\"cell\":{},\"color\":{},\"off\":{off},\"n\":{n}}}",
            js_str(&t.letter),
            js_str(&t.label),
            js_str(&t.metric),
            js_list(&runs),
            t.band,
            js_str(&t.cell),
            js_str(&t.color)
        ));
    }
    // heatmaps after the traces: cols (ncols) then z (rows × ncols, row-major)
    let mut hm = Vec::new();
    for h in &o.heatmaps {
        let off = flat.len();
        flat.extend_from_slice(&h.m.cols);
        flat.extend_from_slice(&h.m.z);
        let layers: Vec<String> = h.layers.iter().map(|l| format!("{l}")).collect();
        hm.push(format!(
            "{{\"run\":{},\"family\":{},\"layers\":[{}],\"metrics\":{},\"ncols\":{},\"off\":{off}}}",
            js_str(&cat.runs[h.run].id),
            js_str(&h.family),
            layers.join(","),
            js_list(&h.metrics),
            h.m.cols.len()
        ));
    }
    let fc = match &o.forecast {
        None => "null".to_string(),
        Some((to, m)) => format!("{{\"to\":{},\"method\":{}}}", to.map_or("null".into(), |v| format!("{v}")), js_str(m)),
    };
    let meta = format!(
        "{{\"traces\":[{}],\"heatmaps\":[{}],\"cells\":{},\"warnings\":{},\"by\":{},\"by_numeric\":{},\"ylog\":{},\"forecast\":{fc},\"x_label\":{},\"downsampled\":{}}}",
        tr.join(","),
        hm.join(","),
        js_list(&o.cells),
        js_list(&o.warnings),
        o.by.as_ref().map_or("null".into(), |b| js_str(b)),
        o.by_numeric,
        o.ylog,
        js_str(&o.x_label),
        o.downsampled
    );
    (meta, flat)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CAT: &str = "R\tbkk07/small\nI\tname\tsmall\nC\tsize\t494325\nC\tlr\t0.001\nM\tloss\nM\tnat_none\nM\ttokens_seen\nM\telapsed_min\n\
R\tbkk07/small_s2\nI\tname\tsmall_s2\nC\tsize\t494325\nC\tlr\t0.001\nM\tloss\nM\tnat_none\n\
R\tbkk07/mid\nI\tname\tmid\nC\tsize\t10106037\nC\tlr\t0.0001\nM\tloss\nM\tnat_none\n\
R\tlocal/pilot1\nI\tname\tpilot1\nC\tsize\t494053\nM\tloss\n";

    #[test]
    fn parses_the_design_example_and_round_trips() {
        let q = parse("loss{run=~\"bkk07/*\", lr>1e-4} | ema .9 | by size mean±sd over seed | x tokens | forecast to 10k").unwrap();
        assert_eq!(q.entries.len(), 1);
        assert_eq!(q.stages, vec![
            Stage::Ema(".9".into()),
            Stage::By("size".into()),
            Stage::Agg(Stat::MeanSd, "seed".into()),
            Stage::X("tokens".into()),
            Stage::Forecast { to: Some("10k".into()), method: None },
        ]);
        let c = canonical(&q);
        assert_eq!(c, "loss{run=~\"bkk07/*\", lr>1e-4} | ema .9 | by size | mean±sd over seed | x tokens | forecast to 10k");
        assert_eq!(canonical(&parse(&c).unwrap()), c);
        let kinds: Vec<String> = chips(&q).into_iter().map(|c| c.kind).collect();
        assert_eq!(kinds, ["metric", "matcher", "matcher", "ema", "by", "agg", "x", "forecast"]);
    }

    #[test]
    fn chip_edits() {
        let src = "loss{run=\"bkk07/small\"} | ema .9";
        assert_eq!(replace_chip(src, 1, "").unwrap(), "loss | ema .9");
        assert_eq!(replace_chip(src, 2, "ema .5").unwrap(), "loss{run=\"bkk07/small\"} | ema .5");
        assert_eq!(replace_chip(src, 1, "run=bkk07/small, lr>1").unwrap(), "loss{run=bkk07/small, lr>1} | ema .9");
        assert_eq!(replace_chip(src, 0, "nat_*").unwrap(), "nat_*{run=\"bkk07/small\"} | ema .9");
        assert!(replace_chip(src, 2, "ema 3").is_err());
        assert_eq!(replace_chip("loss | ema .9", 0, "loss{run=x}").unwrap(), "loss{run=x} | ema .9");
    }

    #[test]
    fn letters_formulas_and_errors() {
        let q = parse("loss{run=\"a\"}; nat_none; d = a - b | show d").unwrap();
        let l: Vec<&str> = q.entries.iter().map(|e| e.0.as_str()).collect();
        assert_eq!(l, ["a", "b", "d"]);
        let q = parse("c = a - b; loss; nat_none").unwrap();
        let l: Vec<&str> = q.entries.iter().map(|e| e.0.as_str()).collect();
        assert_eq!(l, ["c", "a", "b"]);
        let e = parse("loss{lr>abc}").unwrap_err();
        assert!(e.msg.contains("not a number") && &"loss{lr>abc}"[e.start..e.end] == "abc");
        let e = parse("loss | frobnicate").unwrap_err();
        assert!(e.msg.contains("unknown stage") && e.start == 7);
        assert!(parse("loss{run=x").unwrap_err().msg.contains("expected ','"));
        assert!(parse("loss | ema 2").is_err());
        assert!(parse("").is_err());
        assert!(parse("loss |").is_err());
        assert!(parse("loss | mean over").is_err());
        assert_eq!(parse("{run=x}").unwrap().entries[0].1, Entry::Sel { metric: "*".into(), matchers: vec![Matcher { field: "run".into(), op: Op::Eq, value: "x".into(), quoted: false }] });
    }

    #[test]
    fn matching_globs_and_numbers() {
        assert!(glob("bkk07/*", "bkk07/small") && !glob("bkk07/*", "local/x"));
        assert!(glob("*small*|mid", "mid") && glob("s?all", "small") && !glob("small", "small_s2"));
        assert_eq!(parse_num("10k"), Some(10000.0));
        assert_eq!(parse_num("1e-4"), Some(1e-4));
        let cat = Catalog::parse(CAT);
        let names = |src: &str| {
            let q = parse(src).unwrap();
            let p = plan(&q, &cat);
            p.streams.iter().map(|s| format!("{}:{}", cat.runs[s.run].id, s.metric)).collect::<Vec<_>>()
        };
        assert_eq!(names("loss{lr>1e-4}"), ["bkk07/small:loss", "bkk07/small_s2:loss"]);
        assert_eq!(names("loss{lr=0.0001}"), ["bkk07/mid:loss"]);
        assert_eq!(names("loss{run!~bkk07/*}"), ["local/pilot1:loss"]);
        assert_eq!(names("loss{lr!=0.001}"), ["bkk07/mid:loss", "local/pilot1:loss"]); // missing lr reads as ""
        assert_eq!(names("loss{lr<1}"), ["bkk07/small:loss", "bkk07/small_s2:loss", "bkk07/mid:loss"]); // missing lr never matches <
        assert_eq!(names("nat_*{name=small}"), ["bkk07/small:nat_none"]);
        // x tokens resolves to tokens_seen where present, warns elsewhere
        let q = parse("loss{run=~bkk07/small*} | x tokens").unwrap();
        let p = plan(&q, &cat);
        assert_eq!(p.streams.len(), 3);
        assert!(p.warnings.iter().any(|w| w.contains("small_s2 has no tokens")));
    }

    fn run_eval(src: &str, data: impl Fn(&str, &str) -> (Vec<f64>, Vec<f64>)) -> Output {
        let cat = Catalog::parse(CAT);
        let q = parse(src).unwrap();
        let p = plan(&q, &cat);
        let owned: Vec<(Vec<f64>, Vec<f64>)> = p.streams.iter().map(|s| data(&cat.runs[s.run].id, &s.metric)).collect();
        let d: Vec<(&[f64], &[f64])> = owned.iter().map(|(a, b)| (&a[..], &b[..])).collect();
        eval(&q, &cat, &p, &d, 0)
    }

    fn sample(run: &str, metric: &str) -> (Vec<f64>, Vec<f64>) {
        let base = match run {
            "bkk07/small" => 5.0,
            "bkk07/small_s2" => 7.0,
            "bkk07/mid" => 4.0,
            _ => 3.0,
        };
        match metric {
            "loss" => (vec![0.0, 100.0, 200.0], vec![base, base - 1.0, base - 2.0]),
            "nat_none" => (vec![0.0, 200.0], vec![1.0, 3.0]),
            "tokens_seen" => (vec![0.0, 100.0, 200.0], vec![0.0, 1000.0, 2000.0]),
            _ => (vec![0.0, 100.0, 200.0], vec![0.0, 1.0, 2.0]),
        }
    }

    #[test]
    fn seeds_aggregate_into_a_band() {
        let o = run_eval("loss{run=~bkk07/*} | mean±sd over seed", sample);
        assert_eq!(o.traces.len(), 2); // small + small_s2 grouped (config equal), mid alone
        let g = o.traces.iter().find(|t| t.runs.len() == 2).unwrap();
        assert!(g.band && g.label.contains("n=2"));
        assert_eq!(g.ys, vec![6.0, 5.0, 4.0]);
        let sd = libm::sqrt(2.0);
        assert!((g.hi[0] - (6.0 + sd)).abs() < 1e-12 && (g.lo[2] - (4.0 - sd)).abs() < 1e-12);
        let m = o.traces.iter().find(|t| t.runs.len() == 1).unwrap();
        assert!(!m.band && m.lo[0].is_nan());
    }

    #[test]
    fn formulas_per_run_with_alignment() {
        let o = run_eval("loss{run=~bkk07/small*}; nat_none{run=~bkk07/small*}; c = a - b | show c", sample);
        assert_eq!(o.traces.len(), 2);
        // small: loss 5,4,3 at 0,100,200; nat_none 1..3 interpolated at 100 -> 2
        assert_eq!(o.traces[0].ys, vec![4.0, 2.0, 0.0]);
        assert!(o.traces[0].label.starts_with("bkk07/small: c = a - b"));
        // a single-stream letter broadcasts
        let o = run_eval("loss{run=~bkk07/*}; nat_none{run=bkk07/mid}; c = a / b | show c", sample);
        assert_eq!(o.traces.len(), 3);
        let o = run_eval("loss; c = a - zz", sample);
        assert!(o.warnings.iter().any(|w| w.contains("unknown series 'zz'")));
    }

    #[test]
    fn heatmap_stage_parses_and_builds_matrices() {
        let q = parse("layers/*/grad_norm | heatmap over layer rownorm | log").unwrap();
        assert_eq!(q.stages[0], Stage::Heatmap { over: "layer".into(), rownorm: true, max: false });
        assert_eq!(canonical(&q), "layers/*/grad_norm | heatmap over layer rownorm | log");
        assert_eq!(canonical(&parse("x | heatmap").unwrap()), "x | heatmap over layer");
        assert_eq!(canonical(&parse("x | heatmap max ema .5").unwrap()), "x | heatmap over layer max | ema .5");
        assert!(parse("x | heatmap over seed").unwrap_err().msg.contains("layer"));
        assert_eq!(chips(&q)[1].kind, "heatmap");
        // two runs x 3 layers; rows sorted numerically (10 after 2)
        let cat = Catalog::parse("R\tpt/a\nM\tlayers/0/grad_norm\nM\tlayers/10/grad_norm\nM\tlayers/2/grad_norm\nM\tloss\nR\tpt/b\nM\tlayers/0/grad_norm\nM\tlayers/1/grad_norm\n");
        let q = parse("layers/*/grad_norm | heatmap over layer").unwrap();
        let p = plan(&q, &cat);
        let owned: Vec<(Vec<f64>, Vec<f64>)> = p.streams.iter().map(|s| {
            let l = crate::layers::layer_index(&s.metric).unwrap().1 as f64;
            ((0..5).map(|i| i as f64 * 10.0).collect(), (0..5).map(|i| l + if i == 3 && l == 10.0 { 100.0 } else { 0.0 }).collect())
        }).collect();
        let d: Vec<(&[f64], &[f64])> = owned.iter().map(|(a, b)| (&a[..], &b[..])).collect();
        let o = eval(&q, &cat, &p, &d, 0);
        assert_eq!(o.heatmaps.len(), 2);
        let h = &o.heatmaps[0];
        assert_eq!((h.run, h.family.as_str(), h.layers.clone()), (0, "layers/*/grad_norm", vec![0, 2, 10]));
        assert_eq!(h.metrics[2], "layers/10/grad_norm");
        assert_eq!(h.m.cols, vec![0.0, 10.0, 20.0, 30.0, 40.0]);
        assert_eq!(h.m.z[2 * 5 + 3], 110.0);
        assert_eq!(o.traces.len(), 5, "traces stay for click-to-line");
        let (meta, flat) = output_json(&o, &cat);
        assert!(meta.contains("\"heatmaps\":[{\"run\":\"pt/a\",\"family\":\"layers/*/grad_norm\",\"layers\":[0,2,10]"), "{meta}");
        assert_eq!(flat.len(), 5 * 6 * 5 + (5 + 15) + (5 + 10));
        // nothing layered: a warning, no heatmap
        let o = run_eval("loss | heatmap", sample);
        assert!(o.heatmaps.is_empty() && o.warnings.iter().any(|w| w.contains("layers/<i>/<stat>")));
    }

    #[test]
    fn baseline_x_split_by_and_ema() {
        let o = run_eval("loss{run=~bkk07/*} | baseline bkk07/mid", sample);
        assert_eq!(o.traces.len(), 2);
        assert_eq!(o.traces[0].ys, vec![1.0, 1.0, 1.0]);
        let o = run_eval("loss{run=bkk07/small} | x tokens", sample);
        assert_eq!(o.traces[0].xs, vec![0.0, 1000.0, 2000.0]);
        assert_eq!(o.x_label, "tokens");
        let o = run_eval("loss; nat_none | split metric | by size", sample);
        assert_eq!(o.cells, vec!["loss".to_string(), "nat_none".to_string()]);
        assert!(o.by_numeric && o.traces[0].color == "494325");
        let o = run_eval("loss{run=\"bkk07/small\"} | ema .5", sample);
        assert_eq!(o.traces[0].raw, vec![5.0, 4.0, 3.0]);
        assert!(o.traces[0].ys[1] != 4.0);
        let o = run_eval("loss{run=bkk07/small} | x time", sample);
        assert_eq!(o.traces[0].xs, vec![0.0, 60.0, 120.0]);
        let (meta, flat) = output_json(&o, &Catalog::parse(CAT));
        assert!(meta.contains("\"x_label\":\"time\"") && flat.len() == 18);
    }
}
