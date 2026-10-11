//! Per-layer metrics: the layer index in a metric name, families of metrics
//! that differ only by that index, and the layer × step matrix behind the
//! query stage `heatmap over layer` (see `docs/design/obs-llm-views.md`).
//!
//! Canonical name: `layers/<i>/<stat>` (`layers/0/grad_norm`). Also
//! recognised, so existing logs work: a numeric path component right after
//! `layers`, `layer`, `blocks`, `block` or `h` with `/` or `.` separators
//! (`grad_norm/layer.3`, `model.layers.3.mlp`, `transformer.h.3.attn`), and
//! a component `layer_<i>` / `layer-<i>` / `layer<i>`.

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

/// Path components (after lowercasing) that announce a layer index in the next component.
const LAYER_WORDS: [&str; 5] = ["layers", "layer", "blocks", "block", "h"];

fn digits(s: &str) -> Option<u32> {
    if s.is_empty() || s.len() > 6 || !s.bytes().all(|c| c.is_ascii_digit()) {
        return None;
    }
    s.parse().ok()
}

/// The layer index in a metric name and its family template (the name with
/// the index replaced by `*`), or `None` when the name has no layer index.
/// The first (outermost) index wins: `layers/3/heads/5/x` is layer 3 of the
/// family `layers/*/heads/5/x`.
pub fn layer_index(name: &str) -> Option<(String, u32)> {
    let mut comps: Vec<(usize, usize)> = Vec::new();
    let mut st = 0;
    for (i, c) in name.bytes().enumerate() {
        if c == b'/' || c == b'.' {
            comps.push((st, i));
            st = i + 1;
        }
    }
    comps.push((st, name.len()));
    let template = |a: usize, e: usize| -> String {
        let mut t = String::with_capacity(name.len());
        t.push_str(&name[..a]);
        t.push('*');
        t.push_str(&name[e..]);
        t
    };
    for k in 0..comps.len() {
        let (a, e) = comps[k];
        let c = &name[a..e];
        if k > 0 {
            let (pa, pe) = comps[k - 1];
            let prev = name[pa..pe].to_ascii_lowercase();
            if let Some(i) = digits(c).filter(|_| LAYER_WORDS.contains(&prev.as_str())) {
                return Some((template(a, e), i));
            }
        }
        let lc = c.to_ascii_lowercase();
        if let Some(rest) = lc.strip_prefix("layer") {
            let rest = rest.strip_prefix('_').or_else(|| rest.strip_prefix('-')).unwrap_or(rest);
            if let Some(i) = digits(rest) {
                return Some((template(e - rest.len(), e), i));
            }
        }
    }
    None
}

/// A set of metrics that differ only by their layer index.
#[derive(Clone, Debug, PartialEq)]
pub struct Family {
    /// The name with the index replaced by `*` (also a query glob for the family).
    pub template: String,
    /// `(layer, metric name)`, sorted by layer.
    pub members: Vec<(u32, String)>,
}

/// Families among `names` with at least `min_layers` distinct layers, sorted
/// by template. A name that appears twice is counted once.
pub fn families(names: &[&str], min_layers: usize) -> Vec<Family> {
    let mut out: Vec<Family> = Vec::new();
    for n in names {
        let Some((t, i)) = layer_index(n) else { continue };
        let f = match out.iter_mut().find(|f| f.template == t) {
            Some(f) => f,
            None => {
                out.push(Family { template: t, members: Vec::new() });
                out.last_mut().unwrap()
            }
        };
        if !f.members.iter().any(|(_, m)| m == n) {
            f.members.push((i, String::from(*n)));
        }
    }
    for f in &mut out {
        f.members.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    }
    out.retain(|f| {
        let mut l: Vec<u32> = f.members.iter().map(|m| m.0).collect();
        l.dedup();
        l.len() >= min_layers.max(1)
    });
    out.sort_by(|a, b| a.template.cmp(&b.template));
    out
}

/// One layer × step matrix: `z[r * cols.len() + c]` is row `r` (a layer)
/// at column `c`; `cols[c]` is the last step in that column.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Matrix {
    pub cols: Vec<f64>,
    pub z: Vec<f64>,
}

/// Build the matrix from one `(steps, values)` series per row on one shared
/// column grid: the union of the rows' steps when it has at most `max_cols`
/// steps (a cell is the row's value at that step, `NaN` when the row did not
/// log it), else `max_cols` equal-width step bins, empty bins dropped, each
/// cell the mean (or, with `max`, the maximum) of the row's finite values
/// in the bin. Every row uses the same columns, so cells line up.
///
/// `rownorm`: each row is divided by the median of its finite cells (a
/// non-positive median is subtracted instead), so 1 means "typical for this
/// layer" and one layer's spike stands out even when layers differ in scale.
pub fn matrix(rows: &[(&[f64], &[f64])], max_cols: usize, rownorm: bool, max: bool) -> Matrix {
    let mut u: Vec<f64> = rows.iter().flat_map(|(s, v)| s.iter().zip(v.iter()).filter(|(s, _)| s.is_finite()).map(|(s, _)| *s)).collect();
    u.sort_by(|a, b| a.partial_cmp(b).unwrap());
    u.dedup();
    if u.is_empty() {
        return Matrix::default();
    }
    let max_cols = max_cols.max(1);
    // column of a step
    let (lo, hi) = (u[0], u[u.len() - 1]);
    let exact = u.len() <= max_cols;
    let width = (hi - lo) / max_cols as f64;
    let bin = |s: f64| -> usize {
        if exact {
            u.partition_point(|x| *x < s)
        } else if width > 0.0 {
            (((s - lo) / width) as usize).min(max_cols - 1)
        } else {
            0
        }
    };
    let nb = if exact { u.len() } else { max_cols };
    // compact the non-empty bins
    let mut col_of = vec![usize::MAX; nb];
    let mut cols: Vec<f64> = Vec::new();
    for &s in &u {
        let b = bin(s);
        if col_of[b] == usize::MAX {
            col_of[b] = cols.len();
            cols.push(s);
        } else {
            cols[col_of[b]] = s;
        }
    }
    let nc = cols.len();
    let mut z = vec![f64::NAN; rows.len() * nc];
    let mut cnt = vec![0u32; rows.len() * nc];
    for (r, (st, va)) in rows.iter().enumerate() {
        for (s, v) in st.iter().zip(va.iter()) {
            if !s.is_finite() || !v.is_finite() {
                continue;
            }
            let k = r * nc + col_of[bin(*s)];
            if cnt[k] == 0 {
                z[k] = *v;
            } else if max {
                z[k] = z[k].max(*v);
            } else {
                z[k] += *v;
            }
            cnt[k] += 1;
        }
    }
    if !max {
        for (x, n) in z.iter_mut().zip(&cnt) {
            if *n > 1 {
                *x /= *n as f64;
            }
        }
    }
    if rownorm {
        for r in 0..rows.len() {
            let row = &mut z[r * nc..(r + 1) * nc];
            let mut f: Vec<f64> = row.iter().copied().filter(|x| x.is_finite()).collect();
            if f.is_empty() {
                continue;
            }
            f.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let m = f.len();
            let med = if m % 2 == 1 { f[m / 2] } else { (f[m / 2 - 1] + f[m / 2]) / 2.0 };
            for x in row.iter_mut() {
                *x = if med > 0.0 { *x / med } else { *x - med };
            }
        }
    }
    Matrix { cols, z }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    fn li(s: &str) -> Option<(String, u32)> {
        layer_index(s)
    }

    #[test]
    fn layer_names() {
        assert_eq!(li("layers/0/grad_norm"), Some(("layers/*/grad_norm".into(), 0)));
        assert_eq!(li("layers/23/attn/entropy"), Some(("layers/*/attn/entropy".into(), 23)));
        assert_eq!(li("actor/layers/7/act_rms"), Some(("actor/layers/*/act_rms".into(), 7)));
        assert_eq!(li("grad_norm/layer.3"), Some(("grad_norm/layer.*".into(), 3)));
        assert_eq!(li("model.layers.12.mlp.grad_norm"), Some(("model.layers.*.mlp.grad_norm".into(), 12)));
        assert_eq!(li("transformer.h.3.attn"), Some(("transformer.h.*.attn".into(), 3)));
        assert_eq!(li("act_rms/layer_5"), Some(("act_rms/layer_*".into(), 5)));
        assert_eq!(li("Layer3/norm"), Some(("Layer*/norm".into(), 3)));
        assert_eq!(li("blocks/02/x"), Some(("blocks/*/x".into(), 2)));
        // the outermost index wins
        assert_eq!(li("layers/3/heads/5/x"), Some(("layers/*/heads/5/x".into(), 3)));
        for n in ["loss", "grad_norm", "layer_norm/weight", "eval/gsm8k/acc", "layers/x/grad_norm", "lr/2", "layers", "h"] {
            assert_eq!(li(n), None, "{n}");
        }
    }

    #[test]
    fn families_group_and_sort_numerically() {
        let names = ["layers/10/g", "layers/2/g", "layers/0/g", "layers/2/g", "layers/0/a", "loss", "layers/1/a", "solo/layer.4"];
        let f = families(&names, 2);
        assert_eq!(f.iter().map(|f| f.template.as_str()).collect::<Vec<_>>(), ["layers/*/a", "layers/*/g"]);
        assert_eq!(f[1].members.iter().map(|m| m.0).collect::<Vec<_>>(), [0, 2, 10]);
        assert_eq!(f[1].members[2].1, "layers/10/g".to_string());
        assert_eq!(families(&names, 4).len(), 0);
    }

    #[test]
    fn matrix_exact_binned_rownorm() {
        // exact: union grid, NaN where a row did not log the step
        let a = ([0.0, 10.0, 20.0], [1.0, 2.0, 3.0]);
        let b = ([10.0, 30.0], [5.0, 7.0]);
        let m = matrix(&[(&a.0, &a.1), (&b.0, &b.1)], 100, false, false);
        assert_eq!(m.cols, vec![0.0, 10.0, 20.0, 30.0]);
        assert_eq!(&m.z[..3], &[1.0, 2.0, 3.0]);
        assert!(m.z[3].is_nan() && m.z[4].is_nan() && m.z[5] == 5.0 && m.z[7] == 7.0);
        // binned: 100 steps into 10 columns, same columns for every row, mean vs max
        let st: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let mut v = vec![1.0; 100];
        v[55] = 11.0;
        let flat = vec![2.0; 100];
        let m = matrix(&[(&st, &v), (&st, &flat)], 10, false, false);
        assert_eq!(m.cols.len(), 10);
        assert_eq!(m.cols[0], 9.0);
        assert!((m.z[5] - 2.0).abs() < 1e-12, "mean of 9 ones and one 11 = 2");
        assert_eq!(m.z[10 + 5], 2.0);
        let mx = matrix(&[(&st, &v), (&st, &flat)], 10, false, true);
        assert_eq!(mx.z[5], 11.0);
        // rownorm: divide by the row median -> 1 where typical, the spike stands out
        let r = matrix(&[(&st, &v), (&st, &flat)], 10, true, true);
        assert_eq!((r.z[0], r.z[5], r.z[10]), (1.0, 11.0, 1.0));
        assert_eq!(matrix(&[], 10, false, false), Matrix::default());
    }
}
