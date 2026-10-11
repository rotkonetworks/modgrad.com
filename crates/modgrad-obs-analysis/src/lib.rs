//! Analysis kernels for modgrad-obs. Pure functions over flat slices so the
//! same code runs natively (the collector, tests) and in the browser as
//! wasm (feature `wasm`, see [`wasm`]). `no_std` + `alloc` unless `wasm`.
//!
//! Conventions: missing values are `NaN` and are never interpolated —
//! every function either propagates them as gaps or skips them, as
//! documented per function.

#![cfg_attr(not(feature = "wasm"), no_std)]
extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use libm::{cos, floor, log10, pow, round, sin, sqrt};

pub mod forecast;

#[cfg(feature = "wasm")]
pub mod wasm;

/// Debiased exponential moving average, the W&B/TensorBoard smoother:
/// `s ← w·s + (1−w)·x`, output `s / (1 − w^k)` after `k` finite inputs.
/// `w` in `[0, 1)`; `w = 0` is identity. A `NaN` input yields `NaN` at that
/// position (a visible gap) and leaves the state untouched.
pub fn ema(v: &[f64], w: f64) -> Vec<f64> {
    let w = w.clamp(0.0, 0.9999);
    let (mut s, mut k) = (0.0, 0i32);
    v.iter()
        .map(|&x| {
            if !x.is_finite() {
                return f64::NAN;
            }
            s = w * s + (1.0 - w) * x;
            k += 1;
            s / (1.0 - pow(w, k as f64))
        })
        .collect()
}

/// RRD `GPRINT`-style legend statistics over the finite values:
/// `[max, avg, min, current]` where current is the last finite value.
/// All `NaN` when there is no finite value.
pub fn legend_stats(v: &[f64]) -> [f64; 4] {
    let (mut mx, mut mn, mut sum, mut n, mut cur) = (f64::NEG_INFINITY, f64::INFINITY, 0.0, 0usize, f64::NAN);
    for &x in v {
        if x.is_finite() {
            mx = mx.max(x);
            mn = mn.min(x);
            sum += x;
            n += 1;
            cur = x;
        }
    }
    if n == 0 { [f64::NAN; 4] } else { [mx, sum / n as f64, mn, cur] }
}

/// Block-reduce a series to at most `max_points` points. Each block reports
/// its last step, the mean of its finite values, and their min/max (an
/// envelope, as RRD consolidation keeps). A block with no finite value is
/// `NaN` (gap). Returns `(steps, mean, min, max)`.
pub fn downsample_series(steps: &[f64], vals: &[f64], max_points: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = steps.len().min(vals.len());
    let block = if max_points == 0 || n <= max_points { 1 } else { n.div_ceil(max_points) };
    let (mut s, mut me, mut lo, mut hi) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut i = 0;
    while i < n {
        let j = (i + block).min(n);
        let fin: Vec<f64> = vals[i..j].iter().copied().filter(|x| x.is_finite()).collect();
        s.push(steps[j - 1]);
        if fin.is_empty() {
            me.push(f64::NAN);
            lo.push(f64::NAN);
            hi.push(f64::NAN);
        } else {
            me.push(fin.iter().sum::<f64>() / fin.len() as f64);
            lo.push(fin.iter().copied().fold(f64::INFINITY, f64::min));
            hi.push(fin.iter().copied().fold(f64::NEG_INFINITY, f64::max));
        }
        i = j;
    }
    (s, me, lo, hi)
}

/// Reduce a row-major N-d array to at most `max_elems` elements by
/// repeatedly mean-pooling the currently longest axis by a factor of 2
/// (the last odd element is kept as its own bin). Mean-pooling, not
/// striding, keeps sparse spikes visible at reduced amplitude.
pub fn downsample(shape: &[usize], data: &[f32], max_elems: usize) -> (Vec<usize>, Vec<f32>) {
    assert_eq!(shape.iter().product::<usize>(), data.len(), "shape/data mismatch");
    let max_elems = max_elems.max(1);
    let mut shape = shape.to_vec();
    let mut data = data.to_vec();
    while data.len() > max_elems {
        let (ax, &len) = shape.iter().enumerate().max_by_key(|&(_, l)| *l).unwrap();
        if len <= 1 {
            break;
        }
        let outer: usize = shape[..ax].iter().product();
        let inner: usize = shape[ax + 1..].iter().product();
        let new_len = len.div_ceil(2);
        let mut out = vec![0f32; outer * new_len * inner];
        for o in 0..outer {
            for j in 0..new_len {
                let (a, b) = (2 * j, (2 * j + 1).min(len - 1));
                for i in 0..inner {
                    let va = data[(o * len + a) * inner + i];
                    out[(o * new_len + j) * inner + i] = if a == b { va } else { (va + data[(o * len + b) * inner + i]) / 2.0 };
                }
            }
        }
        shape[ax] = new_len;
        data = out;
    }
    (shape, data)
}

/// Top-3 principal components of `n` rows of dimension `d` (row-major).
/// Centres the data, finds components by power iteration with deflation
/// (deterministic start, 200 iterations, deterministic sign: the largest
/// loading is positive), and returns the projected coordinates (`n × 3`,
/// row-major) and each component's explained-variance ratio. Components
/// beyond the data's rank come out as zeros. Non-finite inputs count as 0.
pub fn pca3(data: &[f32], n: usize, d: usize) -> (Vec<f32>, [f64; 3]) {
    assert_eq!(data.len(), n * d);
    let mut x: Vec<f64> = data.iter().map(|&v| if v.is_finite() { v as f64 } else { 0.0 }).collect();
    for j in 0..d {
        let m = (0..n).map(|i| x[i * d + j]).sum::<f64>() / n.max(1) as f64;
        for i in 0..n {
            x[i * d + j] -= m;
        }
    }
    let total: f64 = x.iter().map(|v| v * v).sum();
    let mut coords = vec![0f32; n * 3];
    let mut ratio = [0f64; 3];
    let mut xv = vec![0f64; n];
    for c in 0..3.min(d) {
        let mut v: Vec<f64> = (0..d).map(|j| 1.0 + ((j * 7919 + c * 104_729) % 1000) as f64 / 1000.0).collect();
        for _ in 0..200 {
            for (i, o) in xv.iter_mut().enumerate() {
                *o = x[i * d..(i + 1) * d].iter().zip(&v).map(|(a, b)| a * b).sum();
            }
            let mut nv = vec![0f64; d];
            for (i, &p) in xv.iter().enumerate() {
                for (o, a) in nv.iter_mut().zip(&x[i * d..(i + 1) * d]) {
                    *o += a * p;
                }
            }
            let norm = sqrt(nv.iter().map(|a| a * a).sum::<f64>());
            if norm < 1e-12 {
                v.iter_mut().for_each(|a| *a = 0.0);
                break;
            }
            v = nv.into_iter().map(|a| a / norm).collect();
        }
        let big = v.iter().copied().fold(0.0f64, |a, b| if b.abs() > a.abs() { b } else { a });
        if big < 0.0 {
            v.iter_mut().for_each(|a| *a = -*a);
        }
        let mut var = 0.0;
        for i in 0..n {
            let row = &mut x[i * d..(i + 1) * d];
            let p: f64 = row.iter().zip(&v).map(|(a, b)| a * b).sum();
            coords[i * 3 + c] = p as f32;
            var += p * p;
            for (a, b) in row.iter_mut().zip(&v) {
                *a -= p * b;
            }
        }
        ratio[c] = if total > 0.0 { var / total } else { 0.0 };
    }
    (coords, ratio)
}

/// Indices of the `k` highest-variance columns of an `n × d` matrix, in
/// ascending index order (cut neuron state to a loggable width).
pub fn top_variance_columns(data: &[f32], n: usize, d: usize, k: usize) -> Vec<usize> {
    let mut var: Vec<(usize, f64)> = (0..d)
        .map(|j| {
            let m = (0..n).map(|i| data[i * d + j] as f64).sum::<f64>() / n.max(1) as f64;
            (j, (0..n).map(|i| (data[i * d + j] as f64 - m) * (data[i * d + j] as f64 - m)).sum::<f64>())
        })
        .collect();
    var.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut idx: Vec<usize> = var.into_iter().take(k).map(|(j, _)| j).collect();
    idx.sort_unstable();
    idx
}

/// Select columns `cols` from an `n × d` row-major matrix.
pub fn select_columns(data: &[f32], n: usize, d: usize, cols: &[usize]) -> Vec<f32> {
    (0..n).flat_map(|i| cols.iter().map(move |&j| data[i * d + j])).collect()
}

/// A metric over (config value × step) assembled from several runs.
#[derive(Debug, Clone, PartialEq)]
pub struct Surface {
    /// Distinct config values, ascending (grid rows).
    pub xs: Vec<f64>,
    /// Distinct (snapped) steps, ascending (grid columns).
    pub ys: Vec<f64>,
    /// `z[xi * ys.len() + yi]`: mean of the runs' values in that cell, `NaN`
    /// where no run logged that step (a gap, never interpolated).
    pub z: Vec<f64>,
    /// Number of runs contributing to each cell.
    pub count: Vec<u32>,
}

/// Assemble a cross-run surface. Run `r` has config value `run_x[r]` and
/// its series is `steps[off[r]..off[r+1]]`, `vals[…]`. Steps are snapped to
/// the nearest multiple of `step_bin` when `step_bin > 0` (so runs that
/// evaluate every 250 steps align even when one starts at step 1). Runs
/// with a non-finite `run_x` are skipped. Runs sharing a config value are
/// averaged per cell, and `count` says how many.
pub fn assemble_surface(run_x: &[f64], off: &[usize], steps: &[f64], vals: &[f64], step_bin: f64) -> Surface {
    let snap = |s: f64| if step_bin > 0.0 { round(s / step_bin) * step_bin } else { s };
    let mut xs: Vec<f64> = run_x.iter().copied().filter(|x| x.is_finite()).collect();
    xs.sort_by(f64::total_cmp);
    xs.dedup();
    let mut ys: Vec<f64> = Vec::new();
    for r in 0..run_x.len() {
        if run_x[r].is_finite() {
            for (s, v) in steps[off[r]..off[r + 1]].iter().zip(&vals[off[r]..off[r + 1]]) {
                if v.is_finite() {
                    ys.push(snap(*s));
                }
            }
        }
    }
    ys.sort_by(f64::total_cmp);
    ys.dedup();
    let (nx, ny) = (xs.len(), ys.len());
    let mut sum = vec![0f64; nx * ny];
    let mut count = vec![0u32; nx * ny];
    for r in 0..run_x.len() {
        let Ok(xi) = xs.binary_search_by(|a| a.total_cmp(&run_x[r])) else { continue };
        // one value per run per cell: the last logged one
        let mut last: Vec<Option<f64>> = vec![None; ny];
        for (s, v) in steps[off[r]..off[r + 1]].iter().zip(&vals[off[r]..off[r + 1]]) {
            if v.is_finite() {
                if let Ok(yi) = ys.binary_search_by(|a| a.total_cmp(&snap(*s))) {
                    last[yi] = Some(*v);
                }
            }
        }
        for (yi, v) in last.into_iter().enumerate() {
            if let Some(v) = v {
                sum[xi * ny + yi] += v;
                count[xi * ny + yi] += 1;
            }
        }
    }
    let z = sum.iter().zip(&count).map(|(s, &c)| if c == 0 { f64::NAN } else { s / c as f64 }).collect();
    Surface { xs, ys, z, count }
}

/// Orthographic view of points already scaled into the unit box
/// `[-1, 1]³` (x, y horizontal, z up): rotate by azimuth `az` about z,
/// then tilt by elevation `el` (degrees; gnuplot's `set view` analogue;
/// the viewer sits on the −y side, raised by `el`, so far points rise).
/// Returns `[sx, sy, depth]` per point, `sy` up, `depth` larger = farther.
pub fn project(xyz: &[f64], az_deg: f64, el_deg: f64) -> Vec<f64> {
    let (az, el) = (az_deg.to_radians(), el_deg.to_radians());
    let (ca, sa, ce, se) = (cos(az), sin(az), cos(el), sin(el));
    let mut out = Vec::with_capacity(xyz.len());
    for p in xyz.chunks_exact(3) {
        let (x1, y1, z1) = (p[0] * ca - p[1] * sa, p[0] * sa + p[1] * ca, p[2]);
        out.push(x1);
        out.push(z1 * ce + y1 * se);
        out.push(y1 * ce - z1 * se);
    }
    out
}

/// Scale points into `[-1, 1]³` per axis (an axis with zero range maps to
/// 0). Non-finite coordinates stay `NaN`. Returns the scaled points and
/// `[min_x, max_x, min_y, max_y, min_z, max_z]` for axis labels.
pub fn normalize_box(xyz: &[f64]) -> (Vec<f64>, [f64; 6]) {
    let mut b = [f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY];
    for p in xyz.chunks_exact(3) {
        for a in 0..3 {
            if p[a].is_finite() {
                b[2 * a] = b[2 * a].min(p[a]);
                b[2 * a + 1] = b[2 * a + 1].max(p[a]);
            }
        }
    }
    let out = xyz
        .chunks_exact(3)
        .flat_map(|p| {
            (0..3).map(move |a| {
                let (lo, hi) = (b[2 * a], b[2 * a + 1]);
                if !p[a].is_finite() { f64::NAN } else if hi > lo { 2.0 * (p[a] - lo) / (hi - lo) - 1.0 } else { 0.0 }
            })
        })
        .collect();
    (out, b)
}

/// Painter's order for a `nx × ny` grid surface (gnuplot `hidden3d`):
/// `depth[i * ny + j]` per vertex (from [`project`]). Returns the indices
/// `i * (ny-1) + j` of the quads whose four corners are finite, sorted
/// far to near. Quads touching a `NaN` vertex are omitted (drawn by the
/// caller as gaps).
pub fn painter_order(nx: usize, ny: usize, depth: &[f64]) -> Vec<u32> {
    if nx < 2 || ny < 2 {
        return Vec::new();
    }
    let mut q: Vec<(f64, u32)> = Vec::new();
    for i in 0..nx - 1 {
        for j in 0..ny - 1 {
            let c = [depth[i * ny + j], depth[i * ny + j + 1], depth[(i + 1) * ny + j], depth[(i + 1) * ny + j + 1]];
            if c.iter().all(|x| x.is_finite()) {
                q.push(((c[0] + c[1] + c[2] + c[3]) / 4.0, (i * (ny - 1) + j) as u32));
            }
        }
    }
    q.sort_by(|a, b| b.0.total_cmp(&a.0));
    q.into_iter().map(|(_, i)| i).collect()
}

/// Indices of `items` sorted far to near by `depth` (for points/lines).
pub fn depth_order(depth: &[f64]) -> Vec<u32> {
    let mut idx: Vec<u32> = (0..depth.len() as u32).collect();
    idx.sort_by(|&a, &b| depth[b as usize].total_cmp(&depth[a as usize]));
    idx
}

/// "Nice" axis ticks (1-2-5 steps) covering `[lo, hi]`, about `n` of them.
pub fn nice_ticks(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    if !(lo.is_finite() && hi.is_finite()) {
        return Vec::new();
    }
    let (lo, hi) = if hi > lo { (lo, hi) } else { (lo - 0.5, lo + 0.5) };
    let raw = (hi - lo) / n.max(1) as f64;
    let mag = pow(10.0, floor(log10(raw)));
    let step = [1.0, 2.0, 2.5, 5.0, 10.0].iter().map(|m| m * mag).find(|s| *s >= raw).unwrap_or(10.0 * mag);
    let mut t = libm::ceil(lo / step) * step;
    let mut out = Vec::new();
    while t <= hi + step * 1e-9 && out.len() < 100 {
        out.push(if t.abs() < step * 1e-9 { 0.0 } else { t });
        t += step;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ema_identity_debias_and_gaps() {
        assert_eq!(ema(&[1.0, 2.0, 3.0], 0.0), vec![1.0, 2.0, 3.0]);
        let e = ema(&[5.0, 5.0, f64::NAN, 5.0], 0.9);
        assert!((e[0] - 5.0).abs() < 1e-12 && (e[1] - 5.0).abs() < 1e-12 && (e[3] - 5.0).abs() < 1e-12);
        assert!(e[2].is_nan());
    }

    #[test]
    fn legend() {
        assert_eq!(legend_stats(&[1.0, f64::NAN, 3.0, 2.0]), [3.0, 2.0, 1.0, 2.0]);
        assert!(legend_stats(&[f64::NAN]).iter().all(|x| x.is_nan()));
    }

    #[test]
    fn series_downsample_keeps_gaps() {
        let s: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let v = [1.0, 3.0, f64::NAN, f64::NAN, 5.0, 7.0, 1.0, 1.0, 2.0, 4.0];
        let (st, me, lo, hi) = downsample_series(&s, &v, 5);
        assert_eq!(st, vec![1.0, 3.0, 5.0, 7.0, 9.0]);
        assert_eq!(me[0], 2.0);
        assert!(me[1].is_nan());
        assert_eq!((lo[2], hi[2]), (5.0, 7.0));
    }

    #[test]
    fn nd_downsample() {
        let (shape, data) = downsample(&[4, 1000], &vec![1.0; 4000], 1000);
        assert_eq!(shape, vec![4, 250]);
        assert!(data.iter().all(|&x| x == 1.0));
        assert_eq!(downsample(&[5], &[1.0, 3.0, 5.0, 7.0, 9.0], 3), (vec![3], vec![2.0, 6.0, 9.0]));
    }

    #[test]
    fn pca_plane() {
        let n = 50;
        let mut data = Vec::new();
        for i in 0..n {
            let (a, b) = (libm::sinf(i as f32 * 0.37) * 3.0, libm::cosf(i as f32 * 0.11));
            data.extend([a, b, a + b, a - b, 0.0]);
        }
        let (c, r) = pca3(&data, n, 5);
        assert_eq!(c.len(), n * 3);
        assert!(r[0] + r[1] > 0.999 && r[2] < 1e-6 && r[0] >= r[1]);
        assert_eq!(top_variance_columns(&[0.0, 1.0, 9.0, 1.0, 0.0, -9.0], 2, 3, 1), vec![2]);
        assert_eq!(select_columns(&[1.0, 2.0, 3.0, 4.0], 2, 2, &[1]), vec![2.0, 4.0]);
    }

    #[test]
    fn surface_marks_gaps_and_averages() {
        // run A (x=1): steps 1,250,500 ; run B (x=10): steps 1,250 ; run C (x=1): step 250
        let run_x = [1.0, 10.0, 1.0];
        let off = [0, 3, 5, 6];
        let steps = [1.0, 250.0, 500.0, 1.0, 250.0, 250.0];
        let vals = [5.0, 4.0, 3.0, 6.0, 5.0, 2.0];
        let s = assemble_surface(&run_x, &off, &steps, &vals, 250.0);
        assert_eq!(s.xs, vec![1.0, 10.0]);
        assert_eq!(s.ys, vec![0.0, 250.0, 500.0]);
        assert_eq!(s.z[1], 3.0); // (4 + 2) / 2
        assert_eq!(s.count[1], 2);
        assert!(s.z[5].is_nan()); // x=10 has no step 500
        assert_eq!(s.count[5], 0);
    }

    #[test]
    fn projection_and_painter() {
        // top view: depth = -z
        let p = project(&[0.0, 0.0, 1.0, 0.0, 0.0, -1.0], 0.0, 90.0);
        assert!(p[2] < p[5]);
        // elevated viewer in front (-y): the far point (y=1) is farther and higher on screen
        let p = project(&[0.0, 1.0, 0.0, 0.0, -1.0, 0.0], 0.0, 30.0);
        assert!(p[2] > p[5] && p[1] > p[4]);
        // 3x3 grid, one NaN vertex removes the quads touching it
        let mut depth = vec![0.0; 9];
        depth[0] = f64::NAN;
        let o = painter_order(3, 3, &depth);
        assert_eq!(o.len(), 3);
        assert!(!o.contains(&0));
        let (n, b) = normalize_box(&[0.0, 0.0, 0.0, 2.0, 4.0, 0.0]);
        assert_eq!(&n[3..6], &[1.0, 1.0, 0.0]);
        assert_eq!((b[1], b[3]), (2.0, 4.0));
        assert_eq!(nice_ticks(0.0, 10.0, 5), vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);
        assert_eq!(depth_order(&[1.0, 3.0, 2.0]), vec![1, 2, 0]);
    }
}
