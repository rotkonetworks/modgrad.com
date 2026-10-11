//! Derived series for the dashboard's graph builder: step-grid alignment,
//! per-panel normalisation, and a small expression language over named
//! series (`a - b`, `ema(x, 0.9)`, `bkk07/small:brain_heldout_nll - bkk07/small:nat_none`).
//!
//! # Alignment
//!
//! Series logged at different cadences are put on one step grid before any
//! arithmetic ([`align`]). Values are read off each series by linear
//! interpolation between its two neighbouring samples ([`interp_at`]); a
//! point outside a series' own `[first, last]` step range is `NaN`, never
//! extrapolated, and interpolating next to a `NaN` sample yields `NaN` (the
//! crate's no-filling convention).
//!
//! * [`Align::Intersection`] (default): the overlap range
//!   `[max first, min last]`, with every step any series has inside it.
//! * [`Align::Union`]: every step of every series; points outside a
//!   series' range are `NaN` for that series.
//! * [`Align::Exact`]: only steps that every series logged; no
//!   interpolation at all.
//!
//! # Expression language
//!
//! ```text
//! expr  := term (('+' | '-') term)*
//! term  := unary (('*' | '/') unary)*
//! unary := '-' unary | power
//! power := atom ('^' unary)?
//! atom  := number | name | "quoted name" | func '(' expr (',' expr)* ')' | '(' expr ')'
//! ```
//!
//! Names are matched against the caller's list of known series, longest
//! match first, so names may contain `/ : . -` (`bkk07/small:nat_none`).
//! Where a name would swallow an operator (`a-b` when a series `a-b`
//! exists), the longest known name wins; spaces or `"…"` disambiguate.
//!
//! Functions (series `x`, scalars `w n lo hi`):
//!
//! | function | meaning |
//! |---|---|
//! | `ema(x, w)` | debiased EMA, [`crate::ema`], `w` in `[0, 1)` |
//! | `diff(x)` | `x[i] - x[i-1]`, first point `NaN` |
//! | `pct_change(x)` | fraction, `x[i] / x[i-1] - 1`; `NaN` when `x[i-1]` is 0 or `NaN` |
//! | `rollmean(x, n)` | trailing mean over the last `n` samples' finite values; partial windows at the start |
//! | `cummin(x)`, `cummax(x)` | running min / max over finite values |
//! | `log(x)`, `log10(x)` | `NaN` for `x <= 0` |
//! | `exp(x)`, `sqrt(x)`, `abs(x)` | elementwise; `sqrt` of negative is `NaN` |
//! | `ratio(a, b)` | `a / b`, `NaN` where `b == 0` (same as `/`) |
//! | `clip(x, lo, hi)` | clamp |
//! | `zscore(x)`, `minmax(x)`, `rebase(x)` | [`normalize`] |
//!
//! `NaN` inputs give `NaN` outputs at that position (a visible gap); the
//! running functions skip them.

use alloc::collections::BTreeSet;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use libm::{exp, fabs, log, log10, pow, sqrt};

// ------------------------------------------------------------------ alignment

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Align {
    Intersection,
    Union,
    Exact,
}

impl Align {
    /// `"union"`, `"exact"`, anything else is intersection.
    pub fn parse(s: &str) -> Align {
        match s {
            "union" => Align::Union,
            "exact" => Align::Exact,
            _ => Align::Intersection,
        }
    }
}

/// Sort by step, drop non-finite steps, and keep the last value of
/// duplicate steps.
pub fn clean(steps: &[f64], vals: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = steps.len().min(vals.len());
    let mut idx: Vec<usize> = (0..n).filter(|&i| steps[i].is_finite()).collect();
    idx.sort_by(|&a, &b| steps[a].partial_cmp(&steps[b]).unwrap().then(a.cmp(&b)));
    let (mut s, mut v): (Vec<f64>, Vec<f64>) = (Vec::with_capacity(idx.len()), Vec::with_capacity(idx.len()));
    for i in idx {
        if s.last() == Some(&steps[i]) {
            *v.last_mut().unwrap() = vals[i];
        } else {
            s.push(steps[i]);
            v.push(vals[i]);
        }
    }
    (s, v)
}

/// Value of a sorted, deduplicated series at step `x`: the sample itself
/// on an exact hit, linear interpolation between neighbours, `NaN` outside
/// `[first, last]` or next to a `NaN` sample.
pub fn interp_at(steps: &[f64], vals: &[f64], x: f64) -> f64 {
    let n = steps.len();
    if n == 0 || !(x >= steps[0] && x <= steps[n - 1]) {
        return f64::NAN;
    }
    // first index with steps[i] >= x
    let i = steps.partition_point(|&s| s < x);
    if steps[i] == x {
        return vals[i];
    }
    let (s0, s1, v0, v1) = (steps[i - 1], steps[i], vals[i - 1], vals[i]);
    if !(v0.is_finite() && v1.is_finite()) {
        return f64::NAN;
    }
    v0 + (v1 - v0) * (x - s0) / (s1 - s0)
}

/// Put several series on one step grid (see the module docs for the modes).
/// Returns the grid and one value vector per series. An empty grid means
/// the series do not overlap.
pub fn align(series: &[(&[f64], &[f64])], mode: Align) -> (Vec<f64>, Vec<Vec<f64>>) {
    let cleaned: Vec<(Vec<f64>, Vec<f64>)> = series.iter().map(|(s, v)| clean(s, v)).collect();
    if cleaned.is_empty() {
        return (Vec::new(), Vec::new());
    }
    if cleaned.len() == 1 {
        let (s, v) = cleaned.into_iter().next().unwrap();
        return (s, vec![v]);
    }
    let mut all: Vec<f64> = cleaned.iter().flat_map(|(s, _)| s.iter().copied()).collect();
    all.sort_by(|a, b| a.partial_cmp(b).unwrap());
    all.dedup();
    let grid: Vec<f64> = match mode {
        Align::Union => all,
        Align::Intersection => {
            if cleaned.iter().any(|(s, _)| s.is_empty()) {
                Vec::new()
            } else {
                let lo = cleaned.iter().map(|(s, _)| s[0]).fold(f64::NEG_INFINITY, f64::max);
                let hi = cleaned.iter().map(|(s, _)| s[s.len() - 1]).fold(f64::INFINITY, f64::min);
                all.into_iter().filter(|&x| x >= lo && x <= hi).collect()
            }
        }
        Align::Exact => all
            .into_iter()
            .filter(|x| cleaned.iter().all(|(s, _)| s.binary_search_by(|p| p.partial_cmp(x).unwrap()).is_ok()))
            .collect(),
    };
    let vals = cleaned.iter().map(|(s, v)| grid.iter().map(|&x| interp_at(s, v, x)).collect()).collect();
    (grid, vals)
}

// ------------------------------------------------------------------ normalisation

/// Per-panel normalisation over the finite values:
///
/// * `"z"`: `(x - mean) / std` (population std); a flat series maps to 0.
/// * `"minmax"`: `(x - min) / (max - min)`; a flat series maps to 0.
/// * `"first"`: `x / first finite value`; all `NaN` when that value is 0.
/// * anything else: identity.
///
/// `NaN` stays `NaN`.
pub fn normalize(v: &[f64], mode: &str) -> Vec<f64> {
    let fin: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if fin.is_empty() {
        return v.to_vec();
    }
    let map = |f: &dyn Fn(f64) -> f64| v.iter().map(|&x| if x.is_finite() { f(x) } else { f64::NAN }).collect();
    match mode {
        "z" => {
            let n = fin.len() as f64;
            let m = fin.iter().sum::<f64>() / n;
            let sd = sqrt(fin.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n);
            map(&|x| if sd > 0.0 { (x - m) / sd } else { 0.0 })
        }
        "minmax" => {
            let lo = fin.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = fin.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            map(&|x| if hi > lo { (x - lo) / (hi - lo) } else { 0.0 })
        }
        "first" => {
            let f0 = fin[0];
            map(&|x| if f0 != 0.0 { x / f0 } else { f64::NAN })
        }
        _ => v.to_vec(),
    }
}

// ------------------------------------------------------------------ series functions

pub fn diff(v: &[f64]) -> Vec<f64> {
    (0..v.len()).map(|i| if i == 0 { f64::NAN } else { v[i] - v[i - 1] }).collect()
}

pub fn pct_change(v: &[f64]) -> Vec<f64> {
    (0..v.len()).map(|i| if i == 0 || v[i - 1] == 0.0 { f64::NAN } else { v[i] / v[i - 1] - 1.0 }).collect()
}

/// Trailing mean of the finite values among the last `n` samples (partial
/// windows at the start); `NaN` where the input is `NaN`.
pub fn rollmean(v: &[f64], n: usize) -> Vec<f64> {
    let n = n.max(1);
    let (mut sum, mut cnt) = (0.0, 0usize);
    let mut out = Vec::with_capacity(v.len());
    for i in 0..v.len() {
        if v[i].is_finite() {
            sum += v[i];
            cnt += 1;
        }
        if i >= n && v[i - n].is_finite() {
            sum -= v[i - n];
            cnt -= 1;
        }
        out.push(if v[i].is_finite() && cnt > 0 { sum / cnt as f64 } else { f64::NAN });
    }
    out
}

/// Running min (`max = false`) or max over finite values; `NaN` stays `NaN`.
pub fn cum(v: &[f64], max: bool) -> Vec<f64> {
    let mut acc = f64::NAN;
    v.iter()
        .map(|&x| {
            if !x.is_finite() {
                return f64::NAN;
            }
            acc = if acc.is_nan() { x } else if max { acc.max(x) } else { acc.min(x) };
            acc
        })
        .collect()
}

// ------------------------------------------------------------------ lexer

#[derive(Clone, Debug, PartialEq)]
pub struct ExprError {
    /// Byte range of the offending text in the expression.
    pub start: usize,
    pub end: usize,
    pub msg: String,
}

fn e<T>(start: usize, end: usize, msg: impl Into<String>) -> Result<T, ExprError> {
    Err(ExprError { start, end, msg: msg.into() })
}

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Num(f64),
    Name(String),
    Ident(String),
    Op(u8),
    End,
}

const fn name_char(c: u8) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, b'_' | b'.' | b'/' | b':' | b'-')
}

fn lex(src: &str, names: &BTreeSet<&str>) -> Result<Vec<(Tok, usize, usize)>, ExprError> {
    let b = src.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if c == b'"' || c == b'`' {
            let close = b[i + 1..].iter().position(|&x| x == c).map(|p| i + 1 + p);
            let Some(j) = close else { return e(i, b.len(), "unterminated quote") };
            out.push((Tok::Name(src[i + 1..j].to_string()), i, j + 1));
            i = j + 1;
            continue;
        }
        if c.is_ascii_alphanumeric() || c == b'_' {
            // longest known name starting here
            let mut j = i;
            while j < b.len() && name_char(b[j]) {
                j += 1;
            }
            if let Some(k) = (i + 1..=j).rev().find(|&k| names.contains(&src[i..k])) {
                out.push((Tok::Name(src[i..k].to_string()), i, k));
                i = k;
                continue;
            }
            if c.is_ascii_digit() || c == b'.' {
                // number: digits [. digits] [e [+-] digits]
                let mut k = i;
                while k < b.len() && (b[k].is_ascii_digit() || b[k] == b'.') {
                    k += 1;
                }
                if k < b.len() && (b[k] == b'e' || b[k] == b'E') {
                    let mut m = k + 1;
                    if m < b.len() && (b[m] == b'+' || b[m] == b'-') {
                        m += 1;
                    }
                    if m < b.len() && b[m].is_ascii_digit() {
                        while m < b.len() && b[m].is_ascii_digit() {
                            m += 1;
                        }
                        k = m;
                    }
                }
                let Ok(v) = src[i..k].parse::<f64>() else { return e(i, k, format!("bad number '{}'", &src[i..k])) };
                if !v.is_finite() {
                    return e(i, k, format!("number '{}' is not finite", &src[i..k]));
                }
                out.push((Tok::Num(v), i, k));
                i = k;
                continue;
            }
            if src[i..j].contains(':') {
                return e(i, j, format!("unknown series '{}'", &src[i..j]));
            }
            let mut k = i;
            while k < b.len() && (b[k].is_ascii_alphanumeric() || b[k] == b'_') {
                k += 1;
            }
            out.push((Tok::Ident(src[i..k].to_string()), i, k));
            i = k;
            continue;
        }
        if c == b'.' && i + 1 < b.len() && b[i + 1].is_ascii_digit() {
            let mut k = i + 1;
            while k < b.len() && b[k].is_ascii_digit() {
                k += 1;
            }
            out.push((Tok::Num(src[i..k].parse().unwrap()), i, k));
            i = k;
            continue;
        }
        if matches!(c, b'+' | b'-' | b'*' | b'/' | b'^' | b'(' | b')' | b',') {
            out.push((Tok::Op(c), i, i + 1));
            i += 1;
            continue;
        }
        let ch = src[i..].chars().next().unwrap();
        return e(i, i + ch.len_utf8(), format!("unexpected '{ch}'"));
    }
    out.push((Tok::End, b.len(), b.len()));
    Ok(out)
}

// ------------------------------------------------------------------ parser

#[derive(Clone, Debug, PartialEq)]
pub enum Ast {
    Num(f64),
    Ref(String, usize, usize),
    Neg(Box<Ast>),
    Bin(u8, Box<Ast>, Box<Ast>, usize),
    Call(String, Vec<Ast>, usize, usize),
}
use alloc::boxed::Box;

struct Parser {
    toks: Vec<(Tok, usize, usize)>,
    i: usize,
}

impl Parser {
    fn peek(&self) -> &(Tok, usize, usize) {
        &self.toks[self.i]
    }
    fn op(&self, c: u8) -> bool {
        self.peek().0 == Tok::Op(c)
    }
    fn expr(&mut self) -> Result<Ast, ExprError> {
        let mut l = self.term()?;
        while self.op(b'+') || self.op(b'-') {
            let (Tok::Op(c), p, _) = self.toks[self.i].clone() else { unreachable!() };
            self.i += 1;
            l = Ast::Bin(c, Box::new(l), Box::new(self.term()?), p);
        }
        Ok(l)
    }
    fn term(&mut self) -> Result<Ast, ExprError> {
        let mut l = self.unary()?;
        while self.op(b'*') || self.op(b'/') {
            let (Tok::Op(c), p, _) = self.toks[self.i].clone() else { unreachable!() };
            self.i += 1;
            l = Ast::Bin(c, Box::new(l), Box::new(self.unary()?), p);
        }
        Ok(l)
    }
    fn unary(&mut self) -> Result<Ast, ExprError> {
        if self.op(b'-') {
            self.i += 1;
            return Ok(Ast::Neg(Box::new(self.unary()?)));
        }
        if self.op(b'+') {
            self.i += 1;
            return self.unary();
        }
        let a = self.atom()?;
        if self.op(b'^') {
            let p = self.peek().1;
            self.i += 1;
            return Ok(Ast::Bin(b'^', Box::new(a), Box::new(self.unary()?), p));
        }
        Ok(a)
    }
    fn atom(&mut self) -> Result<Ast, ExprError> {
        let (t, s, en) = self.peek().clone();
        self.i += 1;
        match t {
            Tok::Num(v) => Ok(Ast::Num(v)),
            Tok::Name(n) => Ok(Ast::Ref(n, s, en)),
            Tok::Ident(n) => {
                if !self.op(b'(') {
                    return e(s, en, format!("unknown series '{n}'"));
                }
                self.i += 1;
                let mut args = Vec::new();
                if !self.op(b')') {
                    loop {
                        args.push(self.expr()?);
                        if self.op(b',') {
                            self.i += 1;
                            continue;
                        }
                        break;
                    }
                }
                if !self.op(b')') {
                    let (_, a, b) = *self.peek();
                    return e(a, b.max(a + 1), format!("expected ')' to close {n}("));
                }
                let close = self.peek().2;
                self.i += 1;
                Ok(Ast::Call(n, args, s, close))
            }
            Tok::Op(b'(') => {
                let inner = self.expr()?;
                if !self.op(b')') {
                    let (_, a, b) = *self.peek();
                    return e(a, b.max(a + 1), "expected ')'");
                }
                self.i += 1;
                Ok(inner)
            }
            Tok::End => e(s, en, "unexpected end of expression"),
            Tok::Op(c) => e(s, en, format!("unexpected '{}'", c as char)),
        }
    }
}

/// Parse `src`, matching names against `names`.
pub fn parse(src: &str, names: &[&str]) -> Result<Ast, ExprError> {
    let set: BTreeSet<&str> = names.iter().copied().collect();
    let toks = lex(src, &set)?;
    let mut p = Parser { toks, i: 0 };
    if p.peek().0 == Tok::End {
        return e(0, src.len(), "empty expression");
    }
    let a = p.expr()?;
    let (t, s, en) = p.peek().clone();
    if t != Tok::End {
        return e(s, en, "unexpected text after the expression");
    }
    Ok(a)
}

/// Distinct series names referenced by `src`, in order of first use.
pub fn refs(src: &str, names: &[&str]) -> Result<Vec<String>, ExprError> {
    fn walk(a: &Ast, out: &mut Vec<String>) {
        match a {
            Ast::Ref(n, ..) => {
                if !out.contains(n) {
                    out.push(n.clone())
                }
            }
            Ast::Neg(x) => walk(x, out),
            Ast::Bin(_, l, r, _) => {
                walk(l, out);
                walk(r, out)
            }
            Ast::Call(_, args, ..) => args.iter().for_each(|x| walk(x, out)),
            Ast::Num(_) => {}
        }
    }
    let a = parse(src, names)?;
    let mut out = Vec::new();
    walk(&a, &mut out);
    Ok(out)
}

// ------------------------------------------------------------------ evaluator

#[derive(Clone, Debug)]
enum Val {
    Scalar(f64),
    Series(Vec<f64>, Vec<f64>),
}

fn span(a: &Ast) -> (usize, usize) {
    match a {
        Ast::Num(_) => (0, 0),
        Ast::Ref(_, s, e) => (*s, *e),
        Ast::Neg(x) => span(x),
        Ast::Bin(_, l, r, _) => {
            let (a, _) = span(l);
            let (_, b) = span(r);
            (a, b)
        }
        Ast::Call(_, _, s, e) => (*s, *e),
    }
}

fn binop(c: u8, x: f64, y: f64) -> f64 {
    match c {
        b'+' => x + y,
        b'-' => x - y,
        b'*' => x * y,
        b'/' => {
            if y == 0.0 {
                f64::NAN
            } else {
                x / y
            }
        }
        _ => pow(x, y),
    }
}

fn map1(v: Val, f: impl Fn(f64) -> f64) -> Val {
    match v {
        Val::Scalar(x) => Val::Scalar(f(x)),
        Val::Series(s, v) => Val::Series(s, v.into_iter().map(f).collect()),
    }
}

struct Env<'a> {
    names: &'a [&'a str],
    data: &'a [(&'a [f64], &'a [f64])],
    mode: Align,
}

fn combine(env: &Env, c: u8, l: Val, r: Val, at: (usize, usize)) -> Result<Val, ExprError> {
    Ok(match (l, r) {
        (Val::Scalar(x), Val::Scalar(y)) => Val::Scalar(binop(c, x, y)),
        (Val::Series(s, v), Val::Scalar(y)) => Val::Series(s, v.into_iter().map(|x| binop(c, x, y)).collect()),
        (Val::Scalar(x), Val::Series(s, v)) => Val::Series(s, v.into_iter().map(|y| binop(c, x, y)).collect()),
        (Val::Series(s1, v1), Val::Series(s2, v2)) => {
            let (grid, vs) = align(&[(&s1, &v1), (&s2, &v2)], env.mode);
            if grid.is_empty() {
                let what = match env.mode {
                    Align::Exact => "no step in common",
                    _ => "step ranges do not overlap",
                };
                return e(at.0, at.1, format!("cannot combine: {what} (try union alignment)"));
            }
            let v = vs[0].iter().zip(&vs[1]).map(|(&x, &y)| binop(c, x, y)).collect();
            Val::Series(grid, v)
        }
    })
}

fn eval_ast(env: &Env, a: &Ast) -> Result<Val, ExprError> {
    match a {
        Ast::Num(v) => Ok(Val::Scalar(*v)),
        Ast::Ref(n, s, en) => match env.names.iter().position(|x| x == n) {
            Some(i) => {
                let (st, v) = clean(env.data[i].0, env.data[i].1);
                Ok(Val::Series(st, v))
            }
            None => e(*s, *en, format!("unknown series '{n}'")),
        },
        Ast::Neg(x) => Ok(map1(eval_ast(env, x)?, |v| -v)),
        Ast::Bin(c, l, r, _) => {
            let (lv, rv) = (eval_ast(env, l)?, eval_ast(env, r)?);
            combine(env, *c, lv, rv, span(a))
        }
        Ast::Call(f, args, s, en) => {
            let at = (*s, *en);
            let arity = |n: usize| -> Result<(), ExprError> {
                if args.len() != n {
                    return e(at.0, at.1, format!("{f}() takes {n} argument{}, got {}", if n == 1 { "" } else { "s" }, args.len()));
                }
                Ok(())
            };
            let series = |i: usize| -> Result<(Vec<f64>, Vec<f64>), ExprError> {
                match eval_ast(env, &args[i])? {
                    Val::Series(s, v) => Ok((s, v)),
                    Val::Scalar(_) => {
                        let (a, b) = span(&args[i]);
                        let (a, b) = if b > a { (a, b) } else { at };
                        e(a, b, format!("{f}() needs a series as argument {}", i + 1))
                    }
                }
            };
            let scalar = |i: usize| -> Result<f64, ExprError> {
                match eval_ast(env, &args[i])? {
                    Val::Scalar(x) => Ok(x),
                    Val::Series(..) => {
                        let (a, b) = span(&args[i]);
                        e(a, b, format!("{f}() needs a number as argument {}", i + 1))
                    }
                }
            };
            let elem = |g: fn(f64) -> f64| -> Result<Val, ExprError> {
                arity(1)?;
                Ok(map1(eval_ast(env, &args[0])?, g))
            };
            match f.as_str() {
                "log" => elem(|x| if x > 0.0 { log(x) } else { f64::NAN }),
                "log10" => elem(|x| if x > 0.0 { log10(x) } else { f64::NAN }),
                "exp" => elem(exp),
                "sqrt" => elem(|x| if x >= 0.0 { sqrt(x) } else { f64::NAN }),
                "abs" => elem(fabs),
                "ema" => {
                    arity(2)?;
                    let (s, v) = series(0)?;
                    let w = scalar(1)?;
                    if !(0.0..1.0).contains(&w) {
                        let (a, b) = span(&args[1]);
                        let (a, b) = if b > a { (a, b) } else { at };
                        return e(a, b, "ema weight must be in [0, 1)");
                    }
                    Ok(Val::Series(s, crate::ema(&v, w)))
                }
                "diff" | "pct_change" | "cummin" | "cummax" | "zscore" | "minmax" | "rebase" => {
                    arity(1)?;
                    let (s, v) = series(0)?;
                    let out = match f.as_str() {
                        "diff" => diff(&v),
                        "pct_change" => pct_change(&v),
                        "cummin" => cum(&v, false),
                        "cummax" => cum(&v, true),
                        "zscore" => normalize(&v, "z"),
                        "minmax" => normalize(&v, "minmax"),
                        _ => normalize(&v, "first"),
                    };
                    Ok(Val::Series(s, out))
                }
                "rollmean" => {
                    arity(2)?;
                    let (s, v) = series(0)?;
                    let n = scalar(1)?;
                    if !(n >= 1.0 && n == libm::floor(n)) {
                        return e(at.0, at.1, "rollmean window must be a whole number >= 1");
                    }
                    Ok(Val::Series(s, rollmean(&v, n as usize)))
                }
                "ratio" => {
                    arity(2)?;
                    let (l, r) = (eval_ast(env, &args[0])?, eval_ast(env, &args[1])?);
                    combine(env, b'/', l, r, at)
                }
                "clip" => {
                    arity(3)?;
                    let (lo, hi) = (scalar(1)?, scalar(2)?);
                    if lo > hi {
                        return e(at.0, at.1, "clip: lo is greater than hi");
                    }
                    Ok(map1(eval_ast(env, &args[0])?, |x| if x.is_nan() { x } else { x.clamp(lo, hi) }))
                }
                _ => e(at.0, at.0 + f.len(), format!("unknown function '{f}'")),
            }
        }
    }
}

/// Evaluate `src` over the named series `data[i] = (steps, values)` for
/// `names[i]`. The result is a series (a constant expression is an error).
pub fn eval(src: &str, names: &[&str], data: &[(&[f64], &[f64])], mode: Align) -> Result<(Vec<f64>, Vec<f64>), ExprError> {
    assert_eq!(names.len(), data.len(), "names/data mismatch");
    let a = parse(src, names)?;
    let env = Env { names, data, mode };
    match eval_ast(&env, &a)? {
        Val::Series(s, v) => Ok((s, v)),
        Val::Scalar(_) => e(0, src.len(), "expression is a constant: reference at least one series"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nan_eq(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x.is_nan() && y.is_nan()) || (x - y).abs() < 1e-12)
    }

    #[test]
    fn align_intersection_interpolates_inside_overlap_only() {
        // a every 100 from 0..400, b at 50, 250, 600: no shared step
        let (sa, va) = ([0.0, 100.0, 200.0, 300.0, 400.0], [0.0, 1.0, 2.0, 3.0, 4.0]);
        let (sb, vb) = ([50.0, 250.0, 600.0], [10.0, 30.0, 60.0]);
        let (g, v) = align(&[(&sa, &va), (&sb, &vb)], Align::Intersection);
        assert_eq!(g, vec![50.0, 100.0, 200.0, 250.0, 300.0, 400.0]);
        assert!(nan_eq(&v[0], &[0.5, 1.0, 2.0, 2.5, 3.0, 4.0]));
        assert!(nan_eq(&v[1], &[10.0, 15.0, 25.0, 30.0, 30.0 + 30.0 * 50.0 / 350.0, 30.0 + 30.0 * 150.0 / 350.0]));
        // exact: nothing in common
        assert!(align(&[(&sa, &va), (&sb, &vb)], Align::Exact).0.is_empty());
    }

    #[test]
    fn align_union_never_extrapolates() {
        let (sa, va) = ([0.0, 10.0], [1.0, 2.0]);
        let (sb, vb) = ([20.0, 30.0], [5.0, 6.0]);
        assert!(align(&[(&sa, &va), (&sb, &vb)], Align::Intersection).0.is_empty());
        let (g, v) = align(&[(&sa, &va), (&sb, &vb)], Align::Union);
        assert_eq!(g, vec![0.0, 10.0, 20.0, 30.0]);
        assert!(nan_eq(&v[0], &[1.0, 2.0, f64::NAN, f64::NAN]));
        assert!(nan_eq(&v[1], &[f64::NAN, f64::NAN, 5.0, 6.0]));
    }

    #[test]
    fn align_sorts_dedupes_and_keeps_gaps() {
        let (s, v) = clean(&[30.0, 10.0, 20.0, 10.0, f64::NAN], &[3.0, 1.0, 2.0, 1.5, 9.0]);
        assert_eq!(s, vec![10.0, 20.0, 30.0]);
        assert_eq!(v, vec![1.5, 2.0, 3.0]);
        // interpolation next to a NaN sample is NaN, exact hits are not
        let (st, vt) = ([0.0, 10.0, 20.0], [1.0, f64::NAN, 3.0]);
        assert!(interp_at(&st, &vt, 5.0).is_nan());
        assert_eq!(interp_at(&st, &vt, 20.0), 3.0);
        assert!(interp_at(&st, &vt, 21.0).is_nan());
    }

    #[test]
    fn normalisations() {
        assert!(nan_eq(&normalize(&[1.0, 3.0, f64::NAN], "z"), &[-1.0, 1.0, f64::NAN]));
        assert!(nan_eq(&normalize(&[2.0, 2.0], "z"), &[0.0, 0.0]));
        assert!(nan_eq(&normalize(&[2.0, 4.0, 3.0], "minmax"), &[0.0, 1.0, 0.5]));
        assert!(nan_eq(&normalize(&[5.0, 5.0], "minmax"), &[0.0, 0.0]));
        assert!(nan_eq(&normalize(&[f64::NAN, 2.0, 4.0], "first"), &[f64::NAN, 1.0, 2.0]));
        assert!(nan_eq(&normalize(&[0.0, 4.0], "first"), &[f64::NAN, f64::NAN]));
    }

    #[test]
    fn running_functions() {
        let n = f64::NAN;
        assert!(nan_eq(&diff(&[1.0, 4.0, n, 5.0]), &[n, 3.0, n, n]));
        assert!(nan_eq(&pct_change(&[2.0, 3.0, 0.0, 1.0]), &[n, 0.5, -1.0, n]));
        assert!(nan_eq(&rollmean(&[1.0, 2.0, 3.0, 4.0], 2), &[1.0, 1.5, 2.5, 3.5]));
        assert!(nan_eq(&rollmean(&[1.0, n, 3.0, 5.0], 3), &[1.0, n, 2.0, 4.0]));
        assert!(nan_eq(&cum(&[3.0, 1.0, n, 2.0, 0.5], false), &[3.0, 1.0, n, 1.0, 0.5]));
        assert!(nan_eq(&cum(&[3.0, 1.0, n, 4.0], true), &[3.0, 3.0, n, 4.0]));
    }

    const RUN_A: &str = "bkk07/small:brain_heldout_nll";
    const RUN_B: &str = "bkk07/small:nat_none";

    #[test]
    fn full_names_with_slashes_and_colons() {
        let names = [RUN_A, RUN_B];
        let (s1, v1) = ([1.0, 250.0, 500.0], [9.0, 6.0, 5.0]);
        let (s2, v2) = ([1.0, 250.0, 500.0], [8.0, 5.5, 4.0]);
        let data = [(&s1[..], &v1[..]), (&s2[..], &v2[..])];
        let (s, v) = eval("bkk07/small:brain_heldout_nll - bkk07/small:nat_none", &names, &data, Align::Intersection).unwrap();
        assert_eq!(s, vec![1.0, 250.0, 500.0]);
        assert!(nan_eq(&v, &[1.0, 0.5, 1.0]));
        // without spaces, the longest known name still wins
        let (_, v) = eval("bkk07/small:brain_heldout_nll/bkk07/small:nat_none", &names, &data, Align::Intersection).unwrap();
        assert!((v[0] - 9.0 / 8.0).abs() < 1e-12);
        assert_eq!(refs("ema(bkk07/small:nat_none, 0.5) - bkk07/small:nat_none", &names).unwrap(), vec![RUN_B.to_string()]);
        let err = eval("bkk07/small:nope - 1", &names, &data, Align::Intersection).unwrap_err();
        assert_eq!((err.start, err.end), (0, 16));
        assert!(err.msg.contains("unknown series"));
    }

    #[test]
    fn aliases_and_operators() {
        let names = ["a", "b"];
        let (s1, v1) = ([0.0, 1.0, 2.0], [2.0, 4.0, 8.0]);
        let (s2, v2) = ([0.0, 1.0, 2.0], [1.0, 2.0, 0.0]);
        let data = [(&s1[..], &v1[..]), (&s2[..], &v2[..])];
        let ev = |src| eval(src, &names, &data, Align::Intersection).unwrap().1;
        assert!(nan_eq(&ev("a-b"), &[1.0, 2.0, 8.0]));
        assert!(nan_eq(&ev("a/b"), &[2.0, 2.0, f64::NAN])); // zero denominator
        assert!(nan_eq(&ev("ratio(a, b)"), &[2.0, 2.0, f64::NAN]));
        assert!(nan_eq(&ev("-a + 2 * b ^ 2"), &[0.0, 4.0, -8.0]));
        assert!(nan_eq(&ev("(a + b) * .5"), &[1.5, 3.0, 4.0]));
        assert!(nan_eq(&ev("log(b)"), &[0.0, libm::log(2.0), f64::NAN]));
        assert!(nan_eq(&ev("clip(a, 3, 5)"), &[3.0, 4.0, 5.0]));
        assert!(nan_eq(&ev("cummax(diff(a))"), &[f64::NAN, 2.0, 4.0]));
        assert!(nan_eq(&ev("\"a\" * 1e1"), &[20.0, 40.0, 80.0]));
        assert!(nan_eq(&ev("ema(a, 0)"), &[2.0, 4.0, 8.0]));
    }

    #[test]
    fn errors_carry_positions() {
        let names = ["a"];
        let (s, v) = ([0.0, 1.0], [1.0, 2.0]);
        let data = [(&s[..], &v[..])];
        let er = |src| eval(src, &names, &data, Align::Intersection).unwrap_err();
        let x = er("a + zz");
        assert_eq!((x.start, x.end, x.msg.as_str()), (4, 6, "unknown series 'zz'"));
        let x = er("ema(a)");
        assert!(x.msg.contains("takes 2 arguments") && x.start == 0);
        assert!(er("ema(a, 1.5)").msg.contains("[0, 1)"));
        assert!(er("diff(3)").msg.contains("needs a series"));
        assert!(er("rollmean(a, a)").msg.contains("needs a number"));
        assert!(er("rollmean(a, 2.5)").msg.contains("whole number"));
        let x = er("a * 1e999");
        assert_eq!((x.start, x.end), (4, 9));
        assert!(x.msg.contains("not finite"));
        assert!(er("frob(a)").msg.contains("unknown function"));
        assert!(er("1 + 2").msg.contains("constant"));
        assert!(er("").msg.contains("empty"));
        assert!(er("(a + 1").msg.contains("expected ')'"));
        assert_eq!(er("a $").start, 2);
        assert!(er("a a").msg.contains("after the expression"));
        // non-overlapping series
        let names = ["a", "b"];
        let (s2, v2) = ([5.0, 6.0], [1.0, 1.0]);
        let data = [(&s[..], &v[..]), (&s2[..], &v2[..])];
        let x = eval("a - b", &names, &data, Align::Intersection).unwrap_err();
        assert!(x.msg.contains("do not overlap") && (x.start, x.end) == (0, 5));
        let (g, v) = eval("a - b", &names, &data, Align::Union).unwrap();
        assert_eq!(g.len(), 4);
        assert!(v.iter().all(|x| x.is_nan()));
    }
}
