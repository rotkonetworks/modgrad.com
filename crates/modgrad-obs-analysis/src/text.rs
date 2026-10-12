//! Text helpers for the samples viewer: a word-level diff of two
//! completions and a histogram of lengths.

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

/// Diff operation on a run of words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Same,
    /// Only in the new text.
    Ins,
    /// Only in the old text.
    Del,
}

/// Words with their trailing whitespace: `"a  b\nc"` → `["a  ", "b\n", "c"]`;
/// leading whitespace is its own token.
pub fn tokens(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let b = s.as_bytes();
    let (mut i, n) = (0, b.len());
    if i < n && b[i].is_ascii_whitespace() {
        while i < n && b[i].is_ascii_whitespace() {
            i += 1;
        }
        out.push(&s[..i]);
    }
    while i < n {
        let st = i;
        while i < n && !b[i].is_ascii_whitespace() {
            i += 1;
        }
        while i < n && b[i].is_ascii_whitespace() {
            i += 1;
        }
        out.push(&s[st..i]);
    }
    out
}

/// Largest LCS table (cells) before the diff falls back to "replace the
/// differing middle" (common prefix and suffix are always trimmed first).
pub const MAX_CELLS: usize = 4_000_000;

/// Word-level diff of `a` (old) → `b` (new): runs of `(op, text)`, adjacent
/// runs of the same op merged. Words compare without their whitespace; the
/// text of a `Same` run is taken from `b`. Longest common subsequence over
/// words after trimming the common prefix and suffix. Also returns the
/// number of words in common.
pub fn diff_words(a: &str, b: &str) -> (Vec<(Op, String)>, usize) {
    let (ta, tb) = (tokens(a), tokens(b));
    let w = |t: &str| -> String { String::from(t.trim_end()) };
    let (wa, wb): (Vec<String>, Vec<String>) = (ta.iter().map(|t| w(t)).collect(), tb.iter().map(|t| w(t)).collect());
    let mut pre = 0;
    while pre < wa.len() && pre < wb.len() && wa[pre] == wb[pre] {
        pre += 1;
    }
    let mut suf = 0;
    while suf < wa.len() - pre && suf < wb.len() - pre && wa[wa.len() - 1 - suf] == wb[wb.len() - 1 - suf] {
        suf += 1;
    }
    let (ma, mb) = (&wa[pre..wa.len() - suf], &wb[pre..wb.len() - suf]);
    let (n, m) = (ma.len(), mb.len());
    // middle: ops in order over indices into ta/tb
    let mut mid: Vec<(Op, usize)> = Vec::new();
    if n > 0 && m > 0 && (n + 1) * (m + 1) <= MAX_CELLS {
        // dp[i][j] = LCS of ma[i..], mb[j..]
        let cols = m + 1;
        let mut dp = vec![0u32; (n + 1) * cols];
        for i in (0..n).rev() {
            for j in (0..m).rev() {
                dp[i * cols + j] = if ma[i] == mb[j] { dp[(i + 1) * cols + j + 1] + 1 } else { dp[(i + 1) * cols + j].max(dp[i * cols + j + 1]) };
            }
        }
        let (mut i, mut j) = (0, 0);
        while i < n || j < m {
            if i < n && j < m && ma[i] == mb[j] {
                mid.push((Op::Same, pre + j));
                i += 1;
                j += 1;
            } else if i < n && (j == m || dp[(i + 1) * cols + j] >= dp[i * cols + j + 1]) {
                mid.push((Op::Del, pre + i));
                i += 1;
            } else {
                mid.push((Op::Ins, pre + j));
                j += 1;
            }
        }
    } else {
        mid.extend((0..n).map(|i| (Op::Del, pre + i)));
        mid.extend((0..m).map(|j| (Op::Ins, pre + j)));
    }
    let mut out: Vec<(Op, String)> = Vec::new();
    let mut same = pre + suf;
    let mut push = |op: Op, t: &str| match out.last_mut() {
        Some((o, s)) if *o == op => s.push_str(t),
        _ => out.push((op, String::from(t))),
    };
    for t in &tb[..pre] {
        push(Op::Same, t);
    }
    for (op, k) in mid {
        match op {
            Op::Same => {
                same += 1;
                push(Op::Same, tb[k])
            }
            Op::Ins => push(Op::Ins, tb[k]),
            Op::Del => push(Op::Del, ta[k]),
        }
    }
    for t in &tb[tb.len() - suf..] {
        push(Op::Same, t);
    }
    (out, same)
}

/// Histogram of the finite `vals` in `bins` equal bins over `[lo, hi]`
/// (their min and max; a single value gets a unit-wide bin). Returns
/// `(lo, hi, counts)`; empty counts when there is no finite value.
pub fn histogram(vals: &[f64], bins: usize) -> (f64, f64, Vec<u32>) {
    let f: Vec<f64> = vals.iter().copied().filter(|v| v.is_finite()).collect();
    if f.is_empty() {
        return (f64::NAN, f64::NAN, Vec::new());
    }
    let bins = bins.max(1);
    let (mut lo, mut hi) = f.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
    if hi == lo {
        lo -= 0.5;
        hi += 0.5;
    }
    let mut c = vec![0u32; bins];
    for v in f {
        let k = (((v - lo) / (hi - lo)) * bins as f64) as usize;
        c[k.min(bins - 1)] += 1;
    }
    (lo, hi, c)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(ops: &[(Op, String)]) -> String {
        let mut s = String::new();
        for (o, t) in ops {
            match o {
                Op::Same => s.push_str(t),
                Op::Ins => {
                    s.push('+');
                    s.push_str(t);
                }
                Op::Del => {
                    s.push('-');
                    s.push_str(t);
                }
            }
        }
        s
    }

    #[test]
    fn tokens_keep_whitespace() {
        assert_eq!(tokens("a  b\nc"), ["a  ", "b\n", "c"]);
        assert_eq!(tokens("  x"), ["  ", "x"]);
        assert!(tokens("").is_empty());
    }

    #[test]
    fn word_diff() {
        let (ops, same) = diff_words("the cat sat on the mat", "the cat stood on a mat");
        assert_eq!(render(&ops), "the cat -sat +stood on -the +a mat");
        assert_eq!(same, 4);
        // identical, insert-only, delete-only, empty sides
        assert_eq!(diff_words("a b", "a b").0, vec![(Op::Same, "a b".into())]);
        assert_eq!(render(&diff_words("a c", "a b c").0), "a +b c");
        assert_eq!(render(&diff_words("a b c", "a c").0), "a -b c");
        assert_eq!(render(&diff_words("", "x y").0), "+x y");
        assert_eq!(render(&diff_words("x y", "").0), "-x y");
        // whitespace-only changes are not word changes; the new text's spacing is shown
        assert_eq!(diff_words("a  b", "a b").0, vec![(Op::Same, "a b".into())]);
    }

    #[test]
    fn histogram_bins() {
        let (lo, hi, c) = histogram(&[0.0, 1.0, 2.0, 3.0, f64::NAN, 4.0], 4);
        assert_eq!((lo, hi, c), (0.0, 4.0, vec![1, 1, 1, 2]));
        assert_eq!(histogram(&[5.0, 5.0], 3).2, vec![0, 2, 0]);
        assert!(histogram(&[f64::NAN], 3).2.is_empty());
    }
}
