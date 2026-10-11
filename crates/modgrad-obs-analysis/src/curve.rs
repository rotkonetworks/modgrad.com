//! Curve helpers for the builder's forecast and live views: a saturating
//! power-law forecast, target ETA readout from a forecast band, and loss
//! spike detection.

use alloc::vec::Vec;
use libm::{fabs, pow};

/// Fit `y = a + b · x^(−c)` over the finite points with `x > 0`: `c` by a
/// grid over `[0.05, 2]`, `(a, b)` by least squares for each `c`. Returns
/// `(a, b, c, sse)`, or `None` with fewer than 3 usable points.
pub fn powerlaw_fit(xs: &[f64], ys: &[f64]) -> Option<(f64, f64, f64, f64)> {
    let pts: Vec<(f64, f64)> = xs.iter().zip(ys).filter(|(x, y)| **x > 0.0 && x.is_finite() && y.is_finite()).map(|(x, y)| (*x, *y)).collect();
    if pts.len() < 3 {
        return None;
    }
    let mut best: Option<(f64, f64, f64, f64)> = None;
    for k in 0..=78 {
        let c = 0.05 + 0.025 * k as f64;
        let n = pts.len() as f64;
        let (mut su, mut suu, mut sy, mut suy) = (0.0, 0.0, 0.0, 0.0);
        for &(x, y) in &pts {
            let u = pow(x, -c);
            su += u;
            suu += u * u;
            sy += y;
            suy += u * y;
        }
        let det = n * suu - su * su;
        if fabs(det) < 1e-300 {
            continue;
        }
        let b = (n * suy - su * sy) / det;
        let a = (sy - b * su) / n;
        let sse: f64 = pts.iter().map(|&(x, y)| { let d = a + b * pow(x, -c) - y; d * d }).sum();
        if best.is_none_or(|bb| sse < bb.3) {
            best = Some((a, b, c, sse));
        }
    }
    best
}

#[derive(Clone, Debug, Default)]
pub struct PowerForecast {
    pub steps: Vec<f64>,
    pub mean: Vec<f64>,
    pub lo: Vec<f64>,
    pub hi: Vec<f64>,
    /// Mean absolute error on the held-out last quarter (fit on the rest),
    /// and the same for a last-value forecast.
    pub mae_test: f64,
    pub mae_test_naive: f64,
    pub n_test: usize,
    pub skill: bool,
    pub params: [f64; 3],
}

/// Power-law forecast from the last point to `horizon` (`n_out` points).
/// Honesty: the fit is backtested on the last quarter of points (at least
/// 2) against last-value; the band is ± the largest held-out absolute error,
/// widened by `sqrt(h / h_test)` beyond the backtest span.
pub fn powerlaw_forecast(xs: &[f64], ys: &[f64], horizon: f64, n_out: usize) -> Option<PowerForecast> {
    let pts: Vec<(f64, f64)> = xs.iter().zip(ys).filter(|(x, y)| x.is_finite() && y.is_finite()).map(|(x, y)| (*x, *y)).collect();
    if pts.len() < 6 {
        return None;
    }
    let last = pts[pts.len() - 1].0;
    if !(horizon > last) {
        return None;
    }
    let nt = (pts.len() / 4).max(2);
    let (fit, test) = pts.split_at(pts.len() - nt);
    let (fx, fy): (Vec<f64>, Vec<f64>) = fit.iter().copied().unzip();
    let (a, b, c, _) = powerlaw_fit(&fx, &fy)?;
    let naive = fit[fit.len() - 1].1;
    let (mut m, mut mn, mut worst) = (0.0, 0.0, 0.0f64);
    for &(x, y) in test {
        let e = fabs(a + b * pow(x, -c) - y);
        m += e;
        mn += fabs(naive - y);
        worst = worst.max(e);
    }
    let (m, mn) = (m / nt as f64, mn / nt as f64);
    let (a, b, c, _) = powerlaw_fit(&pts.iter().map(|p| p.0).collect::<Vec<_>>(), &pts.iter().map(|p| p.1).collect::<Vec<_>>())?;
    let span = test[nt - 1].0 - fit[fit.len() - 1].0;
    let n_out = n_out.max(2);
    let mut f = PowerForecast { mae_test: m, mae_test_naive: mn, n_test: nt, skill: m < mn, params: [a, b, c], ..Default::default() };
    for i in 1..=n_out {
        let x = last + (horizon - last) * i as f64 / n_out as f64;
        let y = a + b * pow(x, -c);
        let wide = if span > 0.0 && x - last > span { libm::sqrt((x - last) / span) } else { 1.0 };
        f.steps.push(x);
        f.mean.push(y);
        f.lo.push(y - worst * wide);
        f.hi.push(y + worst * wide);
    }
    Some(f)
}

/// First x at which `ys` reaches `target`, crossing in the direction set by
/// the first finite value (falling when it starts above the target), with
/// linear interpolation between points. `NaN` when it never does.
pub fn first_crossing(xs: &[f64], ys: &[f64], target: f64) -> f64 {
    let mut prev: Option<(f64, f64)> = None;
    let mut down = None;
    for (&x, &y) in xs.iter().zip(ys) {
        if !(x.is_finite() && y.is_finite()) {
            continue;
        }
        let d = *down.get_or_insert(y > target);
        let reached = if d { y <= target } else { y >= target };
        if reached {
            return match prev {
                Some((px, py)) if py != y => px + (x - px) * (target - py) / (y - py),
                _ => x,
            };
        }
        prev = Some((x, y));
    }
    f64::NAN
}

/// ETA readout: `[mean, early, late]` crossing of `target` for a forecast
/// with band `lo..hi`. For a falling curve the lower band reaches the target
/// first (early) and the upper band last (late); reversed when rising.
pub fn eta(xs: &[f64], mean: &[f64], lo: &[f64], hi: &[f64], target: f64) -> [f64; 3] {
    let first = mean.iter().copied().find(|v| v.is_finite()).unwrap_or(f64::NAN);
    let falling = first > target;
    let m = first_crossing(xs, mean, target);
    let (e, l) = if falling { (first_crossing(xs, lo, target), first_crossing(xs, hi, target)) } else { (first_crossing(xs, hi, target), first_crossing(xs, lo, target)) };
    [m, e, l]
}

/// Indices of upward spikes: points that exceed the median of the previous
/// `window` finite points by more than `k` robust standard deviations
/// (1.4826 × MAD, floored at 1e-3 of the median's magnitude so flat curves do
/// not flag noise). Needs a full window before the first detection.
pub fn spikes(vals: &[f64], window: usize, k: f64) -> Vec<u32> {
    let window = window.max(3);
    let mut hist: Vec<f64> = Vec::new();
    let mut out = Vec::new();
    for (i, &v) in vals.iter().enumerate() {
        if !v.is_finite() {
            continue;
        }
        if hist.len() >= window {
            let mut w: Vec<f64> = hist[hist.len() - window..].to_vec();
            w.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = w[window / 2];
            let mut dev: Vec<f64> = w.iter().map(|x| fabs(x - med)).collect();
            dev.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let sd = (1.4826 * dev[window / 2]).max(1e-3 * fabs(med)).max(1e-12);
            if v - med > k * sd {
                out.push(i as u32);
            }
        }
        hist.push(v);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn powerlaw_recovers_parameters_and_forecasts_with_skill() {
        let xs: Vec<f64> = (1..=40).map(|i| i as f64 * 100.0).collect();
        let ys: Vec<f64> = xs.iter().map(|x| 2.0 + 30.0 * pow(*x, -0.5)).collect();
        let (a, b, c, sse) = powerlaw_fit(&xs, &ys).unwrap();
        assert!((a - 2.0).abs() < 1e-6 && (b - 30.0).abs() < 1e-4 && (c - 0.5).abs() < 1e-9 && sse < 1e-12);
        let f = powerlaw_forecast(&xs, &ys, 8000.0, 10).unwrap();
        assert!(f.skill && f.mae_test < 1e-6 && f.n_test == 10);
        assert!((f.mean[9] - (2.0 + 30.0 * pow(8000.0, -0.5))).abs() < 1e-6);
        assert!(powerlaw_forecast(&xs[..5], &ys[..5], 9000.0, 4).is_none());
        assert!(powerlaw_forecast(&xs, &ys, 100.0, 4).is_none());
    }

    #[test]
    fn eta_reads_band_crossings() {
        let xs = [0.0, 10.0, 20.0, 30.0];
        let mean = [5.0, 4.0, 3.0, 2.0];
        let lo = [5.0, 3.5, 2.5, 1.5];
        let hi = [5.0, 4.5, 3.5, 2.5];
        let e = eta(&xs, &mean, &lo, &hi, 3.2);
        assert!((e[0] - 18.0).abs() < 1e-9);
        assert!((e[1] - 13.0).abs() < 1e-9 && (e[2] - 23.0).abs() < 1e-9);
        assert!(eta(&xs, &mean, &lo, &hi, 0.5)[0].is_nan());
        // rising
        assert!((first_crossing(&xs, &[0.0, 1.0, 2.0, 3.0], 1.5) - 15.0).abs() < 1e-9);
    }

    #[test]
    fn spikes_found_and_flat_noise_ignored() {
        let mut v: Vec<f64> = (0..50).map(|i| 3.0 - i as f64 * 0.01 + if i % 2 == 0 { 0.002 } else { -0.002 }).collect();
        v[30] += 0.5;
        assert_eq!(spikes(&v, 10, 6.0), vec![30]);
        assert!(spikes(&[1.0; 40], 10, 6.0).is_empty());
        assert!(spikes(&[1.0, 9.0], 10, 6.0).is_empty());
    }
}
