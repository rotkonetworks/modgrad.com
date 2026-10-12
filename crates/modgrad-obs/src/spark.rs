//! Plain-text run summaries for terminals: the table `modgrad tail` and
//! `ssh <server> tail <run>` print (latest value, range and a braille
//! sparkline per metric).

use std::collections::BTreeMap;

/// Braille sparkline: two values per character, four dot rows of height.
/// NaN/inf values are drawn as gaps.
pub fn sparkline(vals: &[f64], width: usize) -> String {
    let v: Vec<f64> = vals.iter().rev().take(width * 2).rev().copied().collect();
    let fin: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if fin.is_empty() {
        return String::new();
    }
    let (lo, hi) = fin.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &x| (a.min(x), b.max(x)));
    let level = |x: f64| -> usize {
        if !x.is_finite() {
            0
        } else if hi > lo {
            1 + (((x - lo) / (hi - lo)) * 3.0).round() as usize
        } else {
            2
        }
    };
    // dot bits from the bottom row up, for the left and right columns
    const LEFT: [u32; 4] = [0x40, 0x04, 0x02, 0x01];
    const RIGHT: [u32; 4] = [0x80, 0x20, 0x10, 0x08];
    v.chunks(2)
        .map(|c| {
            let mut bits = 0u32;
            for (i, &x) in c.iter().enumerate() {
                let col = if i == 0 { &LEFT } else { &RIGHT };
                for b in col.iter().take(level(x)) {
                    bits |= b;
                }
            }
            char::from_u32(0x2800 + bits).unwrap_or(' ')
        })
        .collect()
}

/// (last finite value, min, max, finite count) of a column.
pub fn stats(c: &[f64]) -> (Option<f64>, f64, f64, usize) {
    let fin: Vec<f64> = c.iter().copied().filter(|x| x.is_finite()).collect();
    let last = c.iter().rev().find(|x| x.is_finite()).copied();
    let (lo, hi) = fin.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &x| (a.min(x), b.max(x)));
    (last, lo, hi, fin.len())
}

/// A short number: 4 decimals, or 3-digit scientific when tiny or huge.
pub fn num(x: f64) -> String {
    if !x.is_finite() {
        "-".into()
    } else if x != 0.0 && (x.abs() >= 1e5 || x.abs() < 1e-3) {
        format!("{x:.3e}")
    } else {
        format!("{:.4}", x).trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

/// `<name>  step <last>  rows <n>`, then one line per key: last, min,
/// max and a `width`-character sparkline.
pub fn table(name: &str, steps: &[u64], cols: &BTreeMap<String, Vec<f64>>, width: usize) -> String {
    let kw = cols.keys().map(|k| k.chars().count()).max().unwrap_or(3).clamp(3, 40);
    let mut out = format!("{name}  step {}  rows {}\n", steps.last().map_or("-".into(), |s| s.to_string()), steps.len());
    out.push_str(&format!("{:kw$}  {:>11}  {:>11}  {:>11}  trend\n", "key", "last", "min", "max"));
    for (k, c) in cols {
        let (last, lo, hi, _) = stats(c);
        out.push_str(&format!("{:kw$}  {:>11}  {:>11}  {:>11}  {}\n", k, last.map_or("-".into(), num), num(lo), num(hi), sparkline(c, width)));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparkline_shape() {
        let s = sparkline(&[0.0, 1.0, 2.0, 3.0], 10);
        assert_eq!(s.chars().count(), 2);
        // first char: left column 1 dot, right 2; last: 3 and 4 dots
        assert_eq!(s, "\u{28e0}\u{28fe}");
        assert_eq!(sparkline(&[f64::NAN], 4), "");
        assert_eq!(sparkline(&(0..100).map(f64::from).collect::<Vec<_>>(), 5).chars().count(), 5);
    }

    #[test]
    fn table_lines() {
        let cols = BTreeMap::from([("loss".to_string(), vec![2.0, 1.0, f64::NAN])]);
        let t = table("r", &[0, 1, 2], &cols, 10);
        assert!(t.starts_with("r  step 2  rows 3\n"), "{t}");
        assert!(t.lines().nth(2).unwrap().starts_with("loss"), "{t}");
    }
}
