//! Numbers as they appear in logs: `3e-4`, `12.5%`, `1.2k`, `0.8s`,
//! `NaN`, `inf`, `"2.0"`.

/// Unit suffixes accepted after a number. The magnitude ones scale the
/// value; the rest are dropped (a percentage stays in percent, so `acc
/// 7.5%` plots as 7.5, matching the `*_pct` metrics elsewhere).
const UNITS: &[(&str, f64)] = &[
    ("%", 1.0),
    ("k", 1e3),
    ("K", 1e3),
    ("M", 1e6),
    ("G", 1e9),
    ("T", 1e12),
    ("s", 1.0),
    ("sec", 1.0),
    ("secs", 1.0),
    ("ms", 1.0),
    ("us", 1.0),
    ("µs", 1.0),
    ("ns", 1.0),
    ("m", 1.0),
    ("min", 1.0),
    ("mins", 1.0),
    ("h", 1.0),
    ("x", 1.0),
    ("B", 1.0),
    ("KB", 1.0),
    ("kB", 1.0),
    ("MB", 1.0),
    ("GB", 1.0),
    ("TB", 1.0),
    ("KiB", 1.0),
    ("MiB", 1.0),
    ("GiB", 1.0),
    ("TiB", 1.0),
    ("W", 1.0),
    ("C", 1.0),
    ("°C", 1.0),
];

/// Throughput-style units (`it/s`, `tok/s`, `s/it`, `samples/sec`, ...).
pub fn is_rate_unit(u: &str) -> bool {
    let u = u.trim_end_matches(|c: char| c == ',' || c == ';');
    let Some((a, b)) = u.split_once('/') else { return false };
    let word = |s: &str| !s.is_empty() && s.len() <= 10 && s.chars().all(|c| c.is_ascii_alphabetic());
    (a.is_empty() || word(a)) && word(b) && (["s", "sec", "it", "step", "batch", "iter"].contains(&b) || ["s", "sec", "it"].contains(&a))
}

/// Length of the longest decimal-number prefix of `s` (sign, digits,
/// fraction, exponent); 0 if there is none.
pub fn number_prefix(s: &str) -> usize {
    let b = s.as_bytes();
    let mut i = 0;
    if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
        i += 1;
    }
    let d0 = i;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    let mut digits = i - d0;
    if i < b.len() && b[i] == b'.' {
        let f0 = i + 1;
        let mut j = f0;
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
        }
        if digits > 0 || j > f0 {
            digits += j - f0;
            i = j;
        }
    }
    if digits == 0 {
        return 0;
    }
    if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
        let mut j = i + 1;
        if j < b.len() && (b[j] == b'+' || b[j] == b'-') {
            j += 1;
        }
        let e0 = j;
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
        }
        if j > e0 {
            i = j;
        }
    }
    i
}

fn special(s: &str) -> Option<f64> {
    match s.to_ascii_lowercase().as_str() {
        "nan" | "+nan" | "-nan" => Some(f64::NAN),
        "inf" | "+inf" | "infinity" | "+infinity" => Some(f64::INFINITY),
        "-inf" | "-infinity" => Some(f64::NEG_INFINITY),
        _ => None,
    }
}

/// A missing value written as text (a gap, not a parse failure).
pub fn is_missing(s: &str) -> bool {
    matches!(s.trim().to_ascii_lowercase().as_str(), "" | "na" | "n/a" | "nan" | "none" | "null" | "-" | "--" | "?")
}

/// Parse one whole token as a number with an optional unit. `NaN`/`inf`
/// parse (to NaN / ±inf); anything left over that is not a known unit makes
/// the token not a number.
pub fn parse_value(tok: &str) -> Option<f64> {
    let s = tok.trim().trim_matches(|c| c == '"' || c == '\'');
    if s.is_empty() {
        return None;
    }
    if let Some(v) = special(s) {
        return Some(v);
    }
    let n = number_prefix(s);
    if n == 0 {
        return None;
    }
    let (num, unit) = s.split_at(n);
    let num = num.trim_start_matches('+');
    let v: f64 = num.parse().ok()?;
    let unit = unit.trim_start();
    if unit.is_empty() {
        return Some(v);
    }
    if is_rate_unit(unit) {
        return Some(v);
    }
    UNITS.iter().find(|(u, _)| *u == unit).map(|(_, m)| v * m)
}

/// CSV-cell flavour: also `True`/`False` and thousands separators
/// (`"1,234.5"`).
pub fn parse_cell(cell: &str) -> Option<f64> {
    let s = cell.trim();
    match s {
        "true" | "True" | "TRUE" => return Some(1.0),
        "false" | "False" | "FALSE" => return Some(0.0),
        _ => {}
    }
    if let Some(v) = parse_value(s) {
        return Some(v);
    }
    // 1,234,567.8
    let (int, frac) = s.split_once('.').unwrap_or((s, ""));
    let groups: Vec<&str> = int.trim_start_matches('-').split(',').collect();
    if groups.len() > 1 && !groups[0].is_empty() && groups[0].len() <= 3 && groups[1..].iter().all(|g| g.len() == 3) && groups.iter().all(|g| g.bytes().all(|b| b.is_ascii_digit())) {
        let joined = if frac.is_empty() { int.replace(',', "") } else { format!("{}.{frac}", int.replace(',', "")) };
        return parse_value(&joined);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_and_units() {
        assert_eq!(parse_value("3e-4"), Some(3e-4));
        assert_eq!(parse_value("12.5%"), Some(12.5));
        assert_eq!(parse_value("-.5"), Some(-0.5));
        assert_eq!(parse_value("+2"), Some(2.0));
        assert_eq!(parse_value("1.2k"), Some(1200.0));
        assert_eq!(parse_value("0.8s"), Some(0.8));
        assert_eq!(parse_value("36.2it/s"), Some(36.2));
        assert_eq!(parse_value("1308 MB"), Some(1308.0));
        assert!(parse_value("NaN").unwrap().is_nan());
        assert_eq!(parse_value("-inf"), Some(f64::NEG_INFINITY));
        assert_eq!(parse_value("\"2.0\""), Some(2.0));
        for bad in ["", "abc", "12:30", "2/10", "1e", "v2", "3.4.5", "10px", "2026-10-11", "."] {
            assert_eq!(parse_value(bad), None, "{bad}");
        }
        assert_eq!(parse_value("1e"), None);
    }

    #[test]
    fn cells() {
        assert_eq!(parse_cell("True"), Some(1.0));
        assert_eq!(parse_cell("1,234.5"), Some(1234.5));
        assert_eq!(parse_cell("12,34"), None);
        assert!(is_missing(" NA "));
        assert!(!is_missing("0"));
    }
}
