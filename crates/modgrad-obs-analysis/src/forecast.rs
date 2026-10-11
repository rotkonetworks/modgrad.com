//! Ghost forecasts for training curves, in the style of SearchCast (Sakana AI,
//! arXiv 2606.27282): ridge regression on lag windows, closed-form via
//! Cholesky, with preprocessing and hyperparameters chosen by a small search
//! on a rolling-origin validation split.
//!
//! Adapted to what modgrad logs actually look like (eval curves of 5–15
//! points, occasionally a few hundred):
//!
//! * **Recursive one-step model.** A direct multi-output model needs `H`
//!   targets per row and has no rows at all on a 7-point curve; one-step
//!   ridge rolled out `H` times does. Bands are per horizon from the
//!   residuals of that same rollout.
//! * **Index space.** Lag windows are over samples, not steps. Future
//!   points sit at `last + k * spacing` with `spacing` the median step
//!   difference. Pooled series must share that spacing (within 10%).
//! * **Optional pooling.** Rows from other runs of the same metric (passed
//!   in by the caller) can be added to the fit, each run scaled by its own
//!   statistics; an optional position feature `ln(1+step)/ln(1+ref)` lets a
//!   pooled model learn the shape of a typical curve.
//! * **Noise augmentation in its expected form.** Gaussian input noise of
//!   std `σ` on the (scaled) lag features, in the infinite-copy limit, adds
//!   `N σ²` to the lag block of the Gram matrix (Bishop 1995). It is a second
//!   ridge term on the lag weights only (bias and position are not shrunk),
//!   so it is cheap and deterministic.
//!
//! Honesty rules: hyperparameters are selected on the earlier origins
//! (validation), counting only targets that lie before the first test
//! origin; MAE is reported for model and last-value naive on the same
//! (origin, horizon) pairs of the later test origins. The ridge forecast is
//! only returned as the drawn forecast when it beats naive on both splits;
//! otherwise the naive forecast and its own residual band are returned and
//! `skill` is false. Bands beyond the largest horizon with ≥ 3 backtest
//! residuals are extrapolated by `sqrt(h / h_max)`; `max_backtest_h` says
//! where that starts.

use alloc::vec;
use alloc::vec::Vec;
use libm::{ceil, log1p, sqrt};

/// How each lag window is scaled before the ridge fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scaler {
    /// Per-series mean / std of the known values.
    Standard,
    /// Per-series median / (IQR / 1.349).
    Robust,
    /// Centre on the window's last value, per-series std as scale
    /// (predicts the next change).
    LocalLast,
    /// Centre on the window mean, per-series std as scale.
    LocalMean,
    /// One mean / std over the pooled other runs, applied to every series
    /// (pooled models only).
    Global,
}

impl Scaler {
    pub fn name(self) -> &'static str {
        match self {
            Scaler::Standard => "standard",
            Scaler::Robust => "robust",
            Scaler::LocalLast => "local-last",
            Scaler::LocalMean => "local-mean",
            Scaler::Global => "global",
        }
    }
}

/// One point of the search space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    pub lookback: usize,
    /// Ridge strength, multiplied by the row count.
    pub lambda: f64,
    /// Input-noise std in scaled units (0 = none).
    pub noise: f64,
    pub scaler: Scaler,
    /// Rows from the other runs are included in the fit.
    pub pooled: bool,
    /// The `ln(1+step)` position feature is included.
    pub position: bool,
}

/// Best configuration of one family (single-series or pooled) with its
/// validation and test MAE.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FamilyBest {
    pub config: Config,
    pub mae_val: f64,
    pub mae_test: f64,
}

#[derive(Clone, Debug)]
pub struct Forecast {
    /// Future steps, `last + k * spacing` up to the requested horizon.
    pub steps: Vec<f64>,
    /// The drawn forecast: ridge when `skill`, else the last value.
    pub mean: Vec<f64>,
    pub lo: Vec<f64>,
    pub hi: Vec<f64>,
    /// Ridge beat last-value naive on both validation and test.
    pub skill: bool,
    /// Selected ridge configuration (whether or not it is drawn).
    pub config: Config,
    pub mae_val: f64,
    pub mae_val_naive: f64,
    pub mae_test: f64,
    pub mae_test_naive: f64,
    /// Number of (origin, horizon) error pairs behind each MAE.
    pub n_val: usize,
    pub n_test: usize,
    pub single: Option<FamilyBest>,
    pub pooled: Option<FamilyBest>,
    /// Largest horizon (in points) whose band comes from ≥ 3 residuals.
    pub max_backtest_h: usize,
    /// Other runs that were usable for pooling.
    pub n_pool: usize,
    pub spacing: f64,
    pub n_configs: usize,
}

/// Search and output settings.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Forecast to this step (rounded to the nearest multiple of the spacing past the last point).
    pub horizon_step: f64,
    pub max_points: usize,
    pub q_lo: f64,
    pub q_hi: f64,
    pub allow_pool: bool,
    pub max_origins: usize,
}

impl Options {
    pub fn to_step(horizon_step: f64) -> Self {
        Options { horizon_step, max_points: 64, q_lo: 0.1, q_hi: 0.9, allow_pool: true, max_origins: 8 }
    }
}

/// Below this many finite points the target gets no forecast.
pub const MIN_POINTS: usize = 4;
const LOOKBACKS: [usize; 6] = [1, 2, 4, 8, 16, 32];
const LAMBDAS: [f64; 5] = [1e-4, 1e-3, 1e-2, 1e-1, 1.0];
const NOISES: [f64; 3] = [0.0, 0.05, 0.2];

struct Ser {
    st: Vec<f64>,
    v: Vec<f64>,
}

fn clean(steps: &[f64], vals: &[f64]) -> Ser {
    let (mut st, mut v) = (Vec::new(), Vec::new());
    for (&s, &x) in steps.iter().zip(vals) {
        if s.is_finite() && x.is_finite() {
            st.push(s);
            v.push(x);
        }
    }
    Ser { st, v }
}

fn median(xs: &mut [f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = xs.len();
    if n % 2 == 1 { xs[n / 2] } else { 0.5 * (xs[n / 2 - 1] + xs[n / 2]) }
}

/// Linear-interpolated quantile of an unsorted slice (sorted in place).
fn quantile(xs: &mut [f64], q: f64) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p = q.clamp(0.0, 1.0) * (xs.len() - 1) as f64;
    let i = p as usize;
    let f = p - i as f64;
    if i + 1 < xs.len() { xs[i] * (1.0 - f) + xs[i + 1] * f } else { xs[i] }
}

fn spacing(st: &[f64]) -> f64 {
    let mut d: Vec<f64> = st.windows(2).map(|w| w[1] - w[0]).filter(|d| *d > 0.0).collect();
    median(&mut d)
}

fn mean_sd(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let mu = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - mu) * (x - mu)).sum::<f64>() / n;
    (mu, sqrt(var))
}

fn safe_scale(sd: f64, mu: f64) -> f64 {
    if sd.is_finite() && sd > 1e-12 * (1.0 + mu.abs()) { sd } else { 1.0 }
}

/// (centre, scale) for a series from its known values.
fn series_stats(v: &[f64], scaler: Scaler, global: (f64, f64)) -> (f64, f64) {
    match scaler {
        Scaler::Global => global,
        Scaler::Robust => {
            let mut s = v.to_vec();
            let med = median(&mut s);
            let iqr = (quantile(&mut s, 0.75) - quantile(&mut s, 0.25)) / 1.349;
            let iqr = if iqr > 1e-12 * (1.0 + med.abs()) { iqr } else { mean_sd(v).1 };
            (med, safe_scale(iqr, med))
        }
        _ => {
            let (mu, sd) = mean_sd(v);
            (mu, safe_scale(sd, mu))
        }
    }
}

/// Feature layout: `lookback` lag values, then position (optional), then bias.
struct Feat {
    lookback: usize,
    scaler: Scaler,
    position: bool,
    ref_ln: f64,
}

impl Feat {
    fn dim(&self) -> usize {
        self.lookback + self.position as usize + 1
    }
    fn centre(&self, win: &[f64], mu: f64) -> f64 {
        match self.scaler {
            Scaler::LocalLast => win[win.len() - 1],
            Scaler::LocalMean => win.iter().sum::<f64>() / win.len() as f64,
            _ => mu,
        }
    }
    /// Fill `x` for a window predicting the value at `step`; returns the centre.
    fn row(&self, win: &[f64], step: f64, (mu, sd): (f64, f64), x: &mut [f64]) -> f64 {
        let c = self.centre(win, mu);
        for (j, &w) in win.iter().enumerate() {
            x[j] = (w - c) / sd;
        }
        let mut k = self.lookback;
        if self.position {
            x[k] = log1p(step.max(0.0)) / self.ref_ln;
            k += 1;
        }
        x[k] = 1.0;
        c
    }
}

/// Gram matrix `XᵀX`, `Xᵀy` and row count.
#[derive(Clone)]
struct Gram {
    d: usize,
    g: Vec<f64>,
    b: Vec<f64>,
    n: usize,
}

impl Gram {
    fn new(d: usize) -> Self {
        Gram { d, g: vec![0.0; d * d], b: vec![0.0; d], n: 0 }
    }
    /// Add every window of `v[..m]` (scaled by `stats`) as a row.
    fn add_series(&mut self, f: &Feat, st: &[f64], v: &[f64], m: usize, stats: (f64, f64)) {
        let d = self.d;
        let mut x = vec![0.0; d];
        for i in f.lookback..m {
            let c = f.row(&v[i - f.lookback..i], st[i], stats, &mut x);
            let y = (v[i] - c) / stats.1;
            for a in 0..d {
                self.b[a] += x[a] * y;
                for bb in a..d {
                    self.g[a * d + bb] += x[a] * x[bb];
                }
            }
            self.n += 1;
        }
    }
    fn merged(&self, o: &Gram) -> Gram {
        let mut r = self.clone();
        for (a, b) in r.g.iter_mut().zip(&o.g) {
            *a += b;
        }
        for (a, b) in r.b.iter_mut().zip(&o.b) {
            *a += b;
        }
        r.n += o.n;
        r
    }
    /// Ridge solve: `(G + λN·I_nonbias + σ²N·I_lag) w = b`, by Cholesky.
    fn solve(&self, lookback: usize, lambda: f64, noise: f64) -> Option<Vec<f64>> {
        let d = self.d;
        let n = self.n as f64;
        let mut a = vec![0.0; d * d];
        for i in 0..d {
            for j in i..d {
                a[i * d + j] = self.g[i * d + j];
                a[j * d + i] = self.g[i * d + j];
            }
        }
        for i in 0..d - 1 {
            a[i * d + i] += lambda * n;
        }
        for i in 0..lookback {
            a[i * d + i] += noise * noise * n;
        }
        a[(d - 1) * d + d - 1] += 1e-9 * n; // bias: numerically safe, effectively unpenalised
        cholesky_solve(&mut a, &self.b, d)
    }
}

/// Solve `A x = b` for symmetric positive-definite `A` (row-major, overwritten
/// with its Cholesky factor). `None` when `A` is not positive definite.
pub fn cholesky_solve(a: &mut [f64], b: &[f64], d: usize) -> Option<Vec<f64>> {
    for j in 0..d {
        let mut s = a[j * d + j];
        for k in 0..j {
            s -= a[j * d + k] * a[j * d + k];
        }
        if !(s > 0.0) || !s.is_finite() {
            return None;
        }
        let l = sqrt(s);
        a[j * d + j] = l;
        for i in j + 1..d {
            let mut t = a[i * d + j];
            for k in 0..j {
                t -= a[i * d + k] * a[j * d + k];
            }
            a[i * d + j] = t / l;
        }
    }
    let mut y = vec![0.0; d];
    for i in 0..d {
        let mut t = b[i];
        for k in 0..i {
            t -= a[i * d + k] * y[k];
        }
        y[i] = t / a[i * d + i];
    }
    for i in (0..d).rev() {
        let mut t = y[i];
        for k in i + 1..d {
            t -= a[k * d + i] * y[k];
        }
        y[i] = t / a[i * d + i];
    }
    Some(y)
}

/// Roll the one-step model forward from the window `v[m-L..m]`.
fn rollout(f: &Feat, w: &[f64], v: &[f64], m: usize, stats: (f64, f64), at: &[f64]) -> Vec<f64> {
    let mut win: Vec<f64> = v[m - f.lookback..m].to_vec();
    let mut x = vec![0.0; f.dim()];
    let mut out = Vec::with_capacity(at.len());
    for &s in at {
        let c = f.row(&win, s, stats, &mut x);
        let z: f64 = w.iter().zip(&x).map(|(a, b)| a * b).sum();
        let y = c + z * stats.1;
        out.push(y);
        win.remove(0);
        win.push(y);
    }
    out
}

/// Forecast `steps`/`vals` up to `opt.horizon_step`. `pool` holds other runs
/// of the same metric (`(steps, vals)`); runs with a different median step
/// spacing are ignored. `None` when the target has fewer than
/// [`MIN_POINTS`] finite points, the horizon is not past the last step, or no
/// configuration can be backtested.
pub fn forecast(steps: &[f64], vals: &[f64], pool: &[(&[f64], &[f64])], opt: &Options) -> Option<Forecast> {
    let t = clean(steps, vals);
    let n = t.v.len();
    if n < MIN_POINTS {
        return None;
    }
    let sp = spacing(&t.st);
    let last = t.st[n - 1];
    if !(sp > 0.0) || !(opt.horizon_step > last) {
        return None;
    }
    // points at last + k*spacing, k up to the nearest whole number of spacings to the horizon
    let h_out = (libm::round((opt.horizon_step - last) / sp) as usize).clamp(1, opt.max_points.max(1));
    let future: Vec<f64> = (1..=h_out).map(|k| last + k as f64 * sp).collect();

    let pool: Vec<Ser> = if opt.allow_pool {
        pool.iter()
            .map(|(s, v)| clean(s, v))
            .filter(|p| p.v.len() >= 2 && (spacing(&p.st) - sp).abs() <= 0.1 * sp)
            .collect()
    } else {
        Vec::new()
    };
    let gstats = if pool.is_empty() {
        (0.0, 1.0)
    } else {
        let all: Vec<f64> = pool.iter().flat_map(|p| p.v.iter().copied()).collect();
        let (mu, sd) = mean_sd(&all);
        (mu, safe_scale(sd, mu))
    };
    let max_step = pool.iter().flat_map(|p| p.st.iter()).chain(t.st.iter()).fold(1.0f64, |a, &b| a.max(b));
    let ref_ln = log1p(max_step).max(1e-9);

    // rolling origins: forecast from the prefix v[..o]
    let o_min = (ceil(n as f64 / 3.0) as usize).max(1);
    let mut origins: Vec<usize> = (o_min..n).collect();
    if origins.len() > opt.max_origins.max(2) {
        let k = opt.max_origins.max(2);
        let span = origins.len() - 1;
        origins = (0..k).map(|i| origins[(i * span + (k - 1) / 2) / (k - 1)]).collect();
        origins.dedup();
    }
    if origins.len() < 2 {
        return None;
    }
    let n_test_o = origins.len().div_ceil(3);
    let first_test = origins[origins.len() - n_test_o];
    let h_bt = h_out.max(1);
    // (origin index, horizon) pairs and which split they score in
    let mut pairs: Vec<(usize, usize, u8)> = Vec::new(); // split: 0 val, 1 test, 2 band-only
    for (oi, &o) in origins.iter().enumerate() {
        for h in 1..=h_bt.min(n - o) {
            let split = if o >= first_test {
                1
            } else if o + h - 1 < first_test {
                0
            } else {
                2
            };
            pairs.push((oi, h, split));
        }
    }
    let n_val = pairs.iter().filter(|p| p.2 == 0).count();
    let n_test = pairs.iter().filter(|p| p.2 == 1).count();
    if n_val == 0 || n_test == 0 {
        return None;
    }
    let naive_err = |oi: usize, h: usize| (t.v[origins[oi] + h - 1] - t.v[origins[oi] - 1]).abs();
    let mae_of = |errs: &dyn Fn(usize, usize) -> f64, split: u8| {
        let (mut s, mut c) = (0.0, 0usize);
        for &(oi, h, split_p) in &pairs {
            if split_p == split {
                s += errs(oi, h);
                c += 1;
            }
        }
        s / c as f64
    };
    let mae_val_naive = mae_of(&naive_err, 0);
    let mae_test_naive = mae_of(&naive_err, 1);

    // search: one Gram per (family, lookback, scaler, position, origin), all (λ, σ) solved on it
    let mut n_configs = 0usize;
    let mut best: [Option<(Config, f64, Vec<f64>)>; 2] = [None, None]; // (cfg, val mae, residuals per pair)
    for pooled in [false, true] {
        if pooled && pool.is_empty() {
            continue;
        }
        let scalers: &[Scaler] = if pooled {
            &[Scaler::Standard, Scaler::Robust, Scaler::LocalLast, Scaler::LocalMean, Scaler::Global]
        } else {
            &[Scaler::Standard, Scaler::Robust, Scaler::LocalLast, Scaler::LocalMean]
        };
        for &lookback in LOOKBACKS.iter() {
            // the rollout needs a full window of the target's own prefix at every origin
            if lookback > origins[0] {
                continue;
            }
            if !pooled && origins[0] < lookback + 2 {
                continue;
            }
            for &scaler in scalers {
                for position in [false, true] {
                    let f = Feat { lookback, scaler, position, ref_ln };
                    let mut pg = Gram::new(f.dim());
                    if pooled {
                        for p in &pool {
                            pg.add_series(&f, &p.st, &p.v, p.v.len(), series_stats(&p.v, scaler, gstats));
                        }
                        if pg.n == 0 {
                            continue;
                        }
                    }
                    let nc = LAMBDAS.len() * NOISES.len();
                    let mut preds: Vec<Vec<Option<Vec<f64>>>> = vec![Vec::new(); nc];
                    for &o in &origins {
                        let stats = series_stats(&t.v[..o], scaler, gstats);
                        let mut tg = Gram::new(f.dim());
                        tg.add_series(&f, &t.st, &t.v, o, stats);
                        let g = if pooled { pg.merged(&tg) } else { tg };
                        let hm = h_bt.min(n - o);
                        for (li, &lambda) in LAMBDAS.iter().enumerate() {
                            for (ni, &noise) in NOISES.iter().enumerate() {
                                let p = if g.n >= 2 {
                                    g.solve(lookback, lambda, noise).map(|w| rollout(&f, &w, &t.v, o, stats, &t.st[o..o + hm]))
                                } else {
                                    None
                                };
                                preds[li * NOISES.len() + ni].push(p.filter(|p| p.iter().all(|x| x.is_finite())));
                            }
                        }
                    }
                    for (ci, pr) in preds.iter().enumerate() {
                        if pr.iter().any(|p| p.is_none()) {
                            continue;
                        }
                        n_configs += 1;
                        let res: Vec<f64> = pairs.iter().map(|&(oi, h, _)| t.v[origins[oi] + h - 1] - pr[oi].as_ref().unwrap()[h - 1]).collect();
                        let (mut s, mut c) = (0.0, 0usize);
                        for (r, p) in res.iter().zip(&pairs) {
                            if p.2 == 0 {
                                s += r.abs();
                                c += 1;
                            }
                        }
                        let val = s / c as f64;
                        let slot = &mut best[pooled as usize];
                        if slot.as_ref().is_none_or(|b| val < b.1) {
                            let cfg = Config { lookback, lambda: LAMBDAS[ci / NOISES.len()], noise: NOISES[ci % NOISES.len()], scaler, pooled, position };
                            *slot = Some((cfg, val, res));
                        }
                    }
                }
            }
        }
    }
    let test_mae = |res: &[f64]| {
        let (mut s, mut c) = (0.0, 0usize);
        for (r, p) in res.iter().zip(&pairs) {
            if p.2 == 1 {
                s += r.abs();
                c += 1;
            }
        }
        s / c as f64
    };
    let fam = |b: &Option<(Config, f64, Vec<f64>)>| b.as_ref().map(|b| FamilyBest { config: b.0, mae_val: b.1, mae_test: test_mae(&b.2) });
    let single = fam(&best[0]);
    let pooled_best = fam(&best[1]);
    let winner = match (&best[0], &best[1]) {
        (Some(a), Some(b)) => if b.1 < a.1 { b } else { a },
        (Some(a), None) => a,
        (None, Some(b)) => b,
        (None, None) => return None,
    };
    let (cfg, mae_val, res) = (winner.0, winner.1, winner.2.clone());
    let mae_test = test_mae(&res);
    let skill = mae_val < mae_val_naive && mae_test < mae_test_naive;

    // final point forecast
    let mean: Vec<f64> = if skill {
        let f = Feat { lookback: cfg.lookback, scaler: cfg.scaler, position: cfg.position, ref_ln };
        let stats = series_stats(&t.v, cfg.scaler, gstats);
        let mut g = Gram::new(f.dim());
        if cfg.pooled {
            for p in &pool {
                g.add_series(&f, &p.st, &p.v, p.v.len(), series_stats(&p.v, cfg.scaler, gstats));
            }
        }
        g.add_series(&f, &t.st, &t.v, n, stats);
        let w = g.solve(cfg.lookback, cfg.lambda, cfg.noise)?;
        let m = rollout(&f, &w, &t.v, n, stats, &future);
        if m.iter().any(|x| !x.is_finite()) {
            return None;
        }
        m
    } else {
        vec![t.v[n - 1]; h_out]
    };

    // conformal-style band from the drawn method's backtest residuals, per horizon
    let resid: Vec<f64> = if skill { res } else { pairs.iter().map(|&(oi, h, _)| t.v[origins[oi] + h - 1] - t.v[origins[oi] - 1]).collect() };
    let by_h = |h: usize| -> Vec<f64> { resid.iter().zip(&pairs).filter(|(_, p)| p.1 == h).map(|(r, _)| *r).collect() };
    let max_backtest_h = (1..=h_bt).filter(|&h| by_h(h).len() >= 3).max().unwrap_or(0);
    let (mut lo, mut hi) = (Vec::with_capacity(h_out), Vec::with_capacity(h_out));
    for h in 1..=h_out {
        let (mut r, scale) = if h <= max_backtest_h {
            (by_h(h), 1.0)
        } else if max_backtest_h > 0 {
            (by_h(max_backtest_h), sqrt(h as f64 / max_backtest_h as f64))
        } else {
            let hm = pairs.iter().map(|p| p.1 as f64).sum::<f64>() / pairs.len() as f64;
            (resid.clone(), sqrt((h as f64 / hm).max(1.0)))
        };
        let (ql, qh) = (quantile(&mut r, opt.q_lo) * scale, quantile(&mut r, opt.q_hi) * scale);
        let m = mean[h - 1];
        lo.push((m + ql).min(m));
        hi.push((m + qh).max(m));
    }

    Some(Forecast {
        steps: future,
        mean,
        lo,
        hi,
        skill,
        config: cfg,
        mae_val,
        mae_val_naive,
        mae_test,
        mae_test_naive,
        n_val,
        n_test,
        single,
        pooled: pooled_best,
        max_backtest_h,
        n_pool: pool.len(),
        spacing: sp,
        n_configs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cholesky_matches_known_solution() {
        // A = [[4,2],[2,3]], x = [1,-1] -> b = [2,-1]
        let mut a = [4.0, 2.0, 2.0, 3.0];
        let x = cholesky_solve(&mut a, &[2.0, -1.0], 2).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-12 && (x[1] + 1.0).abs() < 1e-12);
        let mut bad = [1.0, 2.0, 2.0, 1.0];
        assert!(cholesky_solve(&mut bad, &[1.0, 1.0], 2).is_none());
    }

    #[test]
    fn tiny_series_and_bad_horizon_give_none() {
        let o = Options::to_step(100.0);
        assert!(forecast(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], &[], &o).is_none());
        let s: Vec<f64> = (0..20).map(|i| i as f64).collect();
        assert!(forecast(&s, &s, &[], &Options::to_step(19.0)).is_none());
        // NaNs are dropped before the length check
        assert!(forecast(&[0.0, 1.0, 2.0, 3.0], &[1.0, f64::NAN, 2.0, 3.0], &[], &o).is_none());
    }

    #[test]
    fn linear_trend_beats_naive_and_extrapolates() {
        let s: Vec<f64> = (0..40).map(|i| (i * 10) as f64).collect();
        let v: Vec<f64> = (0..40).map(|i| 5.0 + 0.5 * i as f64).collect();
        let f = forecast(&s, &v, &[], &Options::to_step(490.0)).unwrap();
        assert!(f.skill, "{f:?}");
        assert_eq!(f.steps.len(), 10);
        assert_eq!(f.steps[0], 400.0);
        assert!(f.mae_test < 0.1 * f.mae_test_naive);
        for (k, m) in f.mean.iter().enumerate() {
            assert!((m - (5.0 + 0.5 * (40 + k) as f64)).abs() < 0.5, "k={k} m={m}");
            assert!(f.lo[k] <= *m && *m <= f.hi[k]);
        }
    }

    #[test]
    fn random_walk_band_widens_and_skill_gate_is_not_a_significance_test() {
        // deterministic pseudo-random walk: no structure for ridge to find
        let mut x = 1u64;
        let mut v = vec![0.0];
        for _ in 0..59 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let e = (x % 2001) as f64 / 1000.0 - 1.0;
            v.push(v[v.len() - 1] + e);
        }
        let s: Vec<f64> = (0..60).map(|i| i as f64).collect();
        let f = forecast(&s, &v, &[], &Options::to_step(70.0)).unwrap();
        // This walk passes the gate by chance (test MAE ~2.19 vs naive ~2.29): "skill" means
        // "beat last-value on these backtest points", not a significance test. Pinned so a
        // change in that behaviour is noticed.
        assert!(f.skill && f.mae_test > 0.9 * f.mae_test_naive, "{f:?}");
        // the band widens (weakly) with horizon
        assert!(f.hi[9] - f.lo[9] >= f.hi[0] - f.lo[0]);
    }

    #[test]
    fn pooling_learns_curve_shape_from_other_runs() {
        // exponential-decay curves with different floors; target is short
        let curve = |a: f64, c: f64, n: usize| -> (Vec<f64>, Vec<f64>) {
            let s: Vec<f64> = (0..n).map(|i| (i * 250) as f64).collect();
            let v: Vec<f64> = (0..n).map(|i| c + a * libm::exp(-(i as f64) / 3.0)).collect();
            (s, v)
        };
        let (ps1, pv1) = curve(4.0, 3.0, 16);
        let (ps2, pv2) = curve(5.0, 2.5, 16);
        let (ps3, pv3) = curve(3.5, 3.2, 16);
        let (ts, tv) = curve(4.5, 2.8, 7);
        let pool: [(&[f64], &[f64]); 3] = [(&ps1, &pv1), (&ps2, &pv2), (&ps3, &pv3)];
        let f = forecast(&ts, &tv, &pool, &Options::to_step(2500.0)).unwrap();
        assert_eq!(f.n_pool, 3);
        let p = f.pooled.unwrap();
        assert!(p.mae_test < f.mae_test_naive, "{f:?}");
        // a run with a different step spacing is not pooled
        let (qs, qv) = curve(4.0, 3.0, 16);
        let qs: Vec<f64> = qs.iter().map(|s| s / 25.0).collect();
        let f2 = forecast(&ts, &tv, &[(&qs, &qv)], &Options::to_step(2500.0)).unwrap();
        assert_eq!(f2.n_pool, 0);
        assert!(f2.pooled.is_none());
    }
}
