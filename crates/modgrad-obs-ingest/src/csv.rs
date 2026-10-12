//! CSV / TSV: a header row, then rows. Quotes per RFC 4180 (doubled
//! quotes, delimiters and newlines inside quotes). Empty / NA cells are
//! gaps; `NaN`/`inf` are kept as NaN; `12.5%`, `1.2k`, `1,234.5` parse.
//!
//! The x axis: the first column named like a step (`step`, `_step`,
//! `global_step`, `iter`, ...), else `epoch` if it rarely repeats, else
//! the first numeric column that is non-decreasing integers with no gaps
//! (a pandas index), else row order. `_runtime` / `wall_time` /
//! `_timestamp` become the run's clock, not metrics.

use crate::num::{is_missing, parse_cell};
use crate::{FileReport, LocalRun, is_epoch, step_rank};

/// Delimiter of a header-ish line: the most frequent of `\t , ; |`
/// outside quotes.
pub fn sniff_delim(text: &str) -> Option<u8> {
    let first = text.lines().find(|l| !l.trim().is_empty() && !l.starts_with('#'))?;
    let mut best = None;
    let mut best_n = 0;
    for d in [b'\t', b',', b';', b'|'] {
        let n = split(first, d).len() - 1;
        if n > best_n {
            best_n = n;
            best = Some(d);
        }
    }
    best
}

/// Is this text a table? The first line is a header (mostly non-numeric
/// fields) and the next lines split into the same number of fields, mostly
/// numeric.
pub fn sniff_table(text: &str) -> Option<u8> {
    let d = sniff_delim(text)?;
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty() && !l.starts_with('#')).take(6).collect();
    if lines.len() < 2 {
        return None;
    }
    let head = split(lines[0], d);
    if head.len() < 2 {
        return None;
    }
    let numeric_head = head.iter().filter(|c| parse_cell(c).is_some()).count();
    if numeric_head * 2 > head.len() {
        return None;
    }
    let body = &lines[1..lines.len().saturating_sub(1).max(1)];
    let ok = body.iter().all(|l| {
        let f = split(l, d);
        f.len() == head.len() && f.iter().filter(|c| parse_cell(c).is_some() || is_missing(c)).count() * 2 >= f.len()
    });
    ok.then_some(d)
}

/// Split one complete record (quotes closed).
pub fn split(line: &str, d: u8) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut q = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        if q {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    cur.push('"');
                    chars.next();
                } else {
                    q = false;
                }
            } else {
                cur.push(c);
            }
        } else if c == '"' && cur.trim().is_empty() {
            cur.clear();
            q = true;
        } else if c as u32 == d as u32 {
            out.push(std::mem::take(&mut cur));
        } else {
            cur.push(c);
        }
    }
    out.push(cur);
    out
}

/// An empty cell: a NaN with its own payload, so a `NaN` written in the
/// file (kept, drawn as a break) is told apart from no value (no point).
const GAP: f64 = f64::from_bits(0x7ff8_0000_0000_0bad);
fn is_gap(v: f64) -> bool {
    v.to_bits() == GAP.to_bits()
}

fn quotes_open(s: &str) -> bool {
    s.bytes().filter(|&b| b == b'"').count() % 2 == 1
}

pub struct Csv {
    d: u8,
    header: Option<Vec<String>>,
    cols: Vec<Vec<f64>>,
    numeric: Vec<usize>,
    text: Vec<usize>,
    rows: usize,
    /// A record whose quoted field spans lines.
    pending: String,
}

impl Csv {
    pub fn new(d: u8) -> Csv {
        Csv { d, header: None, cols: Vec::new(), numeric: Vec::new(), text: Vec::new(), rows: 0, pending: String::new() }
    }

    /// One physical line. Returns false when the line was not used.
    pub fn line(&mut self, l: &str) -> bool {
        let rec = if self.pending.is_empty() {
            if quotes_open(l) {
                self.pending = l.to_string();
                return true;
            }
            l.to_string()
        } else {
            self.pending.push('\n');
            self.pending.push_str(l);
            if quotes_open(&self.pending) {
                return true;
            }
            std::mem::take(&mut self.pending)
        };
        if self.header.is_none() && rec.starts_with('#') {
            return false;
        }
        let f = split(&rec, self.d);
        let Some(h) = &self.header else {
            // a header row of numbers means there is no header
            if f.iter().filter(|c| parse_cell(c).is_some()).count() * 2 > f.len() {
                self.header = Some((1..=f.len()).map(|i| format!("col{i}")).collect());
                self.cols = vec![Vec::new(); f.len()];
                self.numeric = vec![0; f.len()];
                self.text = vec![0; f.len()];
                return self.row(&f);
            }
            let mut names: Vec<String> = Vec::new();
            for (i, n) in f.iter().enumerate() {
                let mut n = n.trim().to_string();
                if n.is_empty() {
                    n = if i == 0 { "index".into() } else { format!("col{}", i + 1) };
                }
                while names.contains(&n) {
                    n.push('_');
                }
                names.push(n);
            }
            self.cols = vec![Vec::new(); names.len()];
            self.numeric = vec![0; names.len()];
            self.text = vec![0; names.len()];
            self.header = Some(names);
            return true;
        };
        if f.len() > h.len() {
            return false;
        }
        self.row(&f)
    }

    fn row(&mut self, f: &[String]) -> bool {
        for (i, col) in self.cols.iter_mut().enumerate() {
            let c = f.get(i).map(String::as_str).unwrap_or("");
            let v = if is_missing(c) && !c.trim().eq_ignore_ascii_case("nan") {
                GAP
            } else {
                match parse_cell(c) {
                    Some(v) => {
                        self.numeric[i] += 1;
                        v
                    }
                    None => {
                        self.text[i] += 1;
                        GAP
                    }
                }
            };
            col.push(v);
        }
        self.rows += 1;
        true
    }

    pub fn finish(mut self, run: &mut LocalRun, rep: &mut FileReport) {
        if !self.pending.is_empty() {
            rep.skipped += 1;
            rep.notes.push("unterminated quoted field at end of file".into());
        }
        let Some(names) = self.header.take() else {
            rep.error = Some("empty table".into());
            return;
        };
        let n = self.rows;
        let numeric: Vec<bool> = (0..names.len()).map(|i| self.numeric[i] > 0 && self.numeric[i] >= self.text[i]).collect();
        let dropped: Vec<&str> = names.iter().zip(&numeric).filter(|(_, n)| !**n).map(|(s, _)| s.as_str()).collect();
        if !dropped.is_empty() {
            rep.notes.push(format!("non-numeric column(s) ignored: {}", dropped.join(", ")));
        }
        let mixed: usize = (0..names.len()).filter(|&i| numeric[i]).map(|i| self.text[i]).sum();
        if mixed > 0 {
            rep.notes.push(format!("{mixed} non-numeric cell(s) in numeric columns treated as gaps"));
        }
        let full_nondecreasing = |c: &Vec<f64>| c.iter().all(|v| v.is_finite()) && c.windows(2).all(|w| w[0] <= w[1]);
        let distinct = |c: &Vec<f64>| {
            let mut v: Vec<u64> = c.iter().map(|x| x.to_bits()).collect();
            v.dedup();
            v.len()
        };
        // step column
        let mut cands: Vec<(usize, usize)> = names.iter().enumerate().filter(|(i, _)| numeric[*i]).filter_map(|(i, n)| step_rank(n).map(|r| (r, i))).collect();
        cands.sort();
        let mut step_col = cands.first().map(|c| c.1);
        let mut why = step_col.map(|i| format!("x = column \"{}\"", names[i]));
        if step_col.is_none() {
            if let Some(i) = names.iter().position(|n| is_epoch(n)).filter(|&i| numeric[i] && full_nondecreasing(&self.cols[i]) && distinct(&self.cols[i]) * 10 >= n * 9) {
                step_col = Some(i);
                why = Some(format!("x = column \"{}\"", names[i]));
            }
        }
        if step_col.is_none() {
            if let Some(i) = (0..names.len()).find(|&i| numeric[i] && !is_epoch(&names[i]) && n >= 2 && full_nondecreasing(&self.cols[i]) && self.cols[i].iter().all(|v| v.fract() == 0.0) && distinct(&self.cols[i]) == n) {
                step_col = Some(i);
                why = Some(format!("no step column; x = first increasing integer column \"{}\"", names[i]));
            }
        }
        rep.notes.push(why.unwrap_or_else(|| "no step column; x = row order".into()));
        // clock
        let lname = |i: usize| names[i].to_ascii_lowercase();
        let t_col = (0..names.len()).find(|&i| numeric[i] && matches!(lname(i).as_str(), "_runtime" | "runtime" | "elapsed_s" | "wall_time" | "_timestamp" | "timestamp" | "walltime"));
        let t0 = t_col.and_then(|i| {
            let first = self.cols[i].iter().copied().find(|v| v.is_finite())?;
            Some(if first > 1e8 { first } else { 0.0 })
        });
        let mut keys = 0;
        let mut steps_seen = std::collections::HashSet::new();
        let xs: Vec<f64> = match step_col {
            Some(i) => self.cols[i].clone(),
            None => (0..n).map(|r| r as f64).collect(),
        };
        let gap_x = xs.iter().filter(|x| !x.is_finite()).count();
        if gap_x > 0 {
            rep.notes.push(format!("{gap_x} row(s) without a step ignored"));
            rep.skipped += gap_x;
        }
        for (i, name) in names.iter().enumerate() {
            if !numeric[i] || Some(i) == step_col || Some(i) == t_col || lname(i) == "_timestamp" {
                continue;
            }
            let col = &self.cols[i];
            let mut any = false;
            for (r, v) in col.iter().enumerate() {
                let x = xs[r];
                // empty cells are gaps: no point. NaN text is a NaN point.
                if !x.is_finite() || is_gap(*v) {
                    continue;
                }
                run.push(name, x, *v);
                rep.points += 1;
                steps_seen.insert(x.to_bits());
                any = true;
            }
            if any {
                keys += 1;
            }
        }
        if let (Some(i), Some(t0)) = (t_col, t0) {
            for (r, v) in self.cols[i].iter().enumerate() {
                if xs[r].is_finite() {
                    run.push_time(xs[r], v - t0);
                }
            }
        }
        rep.metrics = keys;
        rep.steps = steps_seen.len();
        run.rows += n - gap_x;
    }
}
