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
    /// Configurations searched whose fits succeeded and whose validation
    /// rollouts stayed finite as far as they were run (a config dropped by
    /// the branch-and-bound still counts; test-origin rollouts are only run
    /// for the selected config).
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

/// How a window is centred before scaling (see [`Scaler`]); `Const` holds
/// the series centre in original units.
#[derive(Clone, Copy)]
enum Centre {
    Const(f64),
    Last,
    Mean,
}

fn centre_of(s: Scaler, mu: f64) -> Centre {
    match s {
        Scaler::LocalLast => Centre::Last,
        Scaler::LocalMean => Centre::Mean,
        _ => Centre::Const(mu),
    }
}

/// Prefix sums of `z zᵀ` (upper triangle, row-major, `D = L + 4`) over the
/// rows of one series for one lookback,
/// `z = [w_j − w_last (L), v − w_last, w_last − shift, position, 1]`:
/// the window and target relative to the window's last value, plus that
/// value relative to a constant per-series `shift`. Every scaler's Gram is an
/// affine image of it (see [`Gram::add_raw`]) without cancelling large
/// offsets, so a series is walked once per lookback instead of once per
/// (scaler, position, origin).
struct Raw {
    r: Vec<f64>,
    n: usize,
    shift: f64,
}

/// [`Raw`] for the prefixes `v[..e]`, `e` in `ends` (ascending).
fn raw_prefixes(v: &[f64], pos: &[f64], shift: f64, l: usize, ends: &[usize]) -> Vec<Raw> {
    let dd = l + 4;
    let mut r = vec![0.0; dd * dd];
    let mut z = vec![0.0; dd];
    let mut out = Vec::with_capacity(ends.len());
    let mut i = l;
    for &e in ends {
        while i < e {
            let win = &v[i - l..i];
            let c = win[l - 1];
            for j in 0..l {
                z[j] = win[j] - c;
            }
            z[l] = v[i] - c;
            z[l + 1] = c - shift;
            z[l + 2] = pos[i];
            z[l + 3] = 1.0;
            for a in 0..dd {
                let za = z[a];
                for (x, &zb) in r[a * dd + a..(a + 1) * dd].iter_mut().zip(&z[a..]) {
                    *x += za * zb;
                }
            }
            i += 1;
        }
        out.push(Raw { r: r.clone(), n: e.saturating_sub(l), shift });
    }
    out
}

/// Gram matrix `XᵀX` (upper triangle), `Xᵀy` and row count of the scaled
/// features: `lookback` lags, position (optional), bias.
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
    /// Add the rows of `raw` scaled as `x_j = (w_j − c(w)) / sd`,
    /// `y = (v − c(w)) / sd`. With `c` a linear functional `αᵀz` of the raw
    /// row, `E[(z_a − c)(z_b − c)] = R_ab − (Rα)_a − (Rα)_b + αᵀRα` on the
    /// centred coordinates (lags, target) and `R_ab − (Rα)_b` against the
    /// uncentred ones (position, bias).
    fn add_raw(&mut self, raw: &Raw, l: usize, centre: Centre, sd: f64, position: bool) {
        let dd = l + 4;
        let (q, bias) = (l + 1, l + 3);
        let r = &raw.r;
        let at = |a: usize, b: usize| if a <= b { r[a * dd + b] } else { r[b * dd + a] };
        // (Rα)_a = E[c z_a] for the centre c = αᵀz in raw coordinates; the
        // centred lags and target are z_a − c (a ≤ l)
        let mut ra = [0.0f64; 36];
        let cc = match centre {
            Centre::Const(mu) => {
                // c = (mu − shift) − (w_last − shift), so z_a − c = w_a − mu
                let c = mu - raw.shift;
                for (a, x) in ra.iter_mut().enumerate().take(dd) {
                    *x = c * at(a, bias) - at(a, q);
                }
                c * ra[bias] - ra[q]
            }
            Centre::Mean => {
                let inv = 1.0 / l as f64;
                for (a, x) in ra.iter_mut().enumerate().take(dd) {
                    *x = (0..l).map(|k| at(a, k)).sum::<f64>() * inv;
                }
                ra[..l].iter().sum::<f64>() * inv
            }
            Centre::Last => 0.0,
        };
        let inv_sd = 1.0 / sd;
        let entry = |a: usize, b: usize| {
            let (ca, cb) = (a <= l, b <= l);
            let x = at(a, b)
                - match (ca, cb) {
                    (true, true) => ra[a] + ra[b] - cc,
                    (true, false) => ra[b],
                    (false, true) => ra[a],
                    (false, false) => 0.0,
                };
            x * if ca { inv_sd } else { 1.0 } * if cb { inv_sd } else { 1.0 }
        };
        let d = self.d;
        let coord = |j: usize| if j < l { j } else if position && j == l { l + 2 } else { bias };
        for i in 0..d {
            let ci = coord(i);
            for j in i..d {
                self.g[i * d + j] += entry(ci, coord(j));
            }
            self.b[i] += entry(ci, l);
        }
        self.n += raw.n;
    }
    /// Ridge solve: `(G + λN·I_nonbias + σ²N·I_lag) w = b`, by Cholesky
    /// (`a` is scratch).
    fn solve(&self, lookback: usize, lambda: f64, noise: f64, a: &mut Vec<f64>) -> Option<Vec<f64>> {
        let d = self.d;
        let n = self.n as f64;
        a.clear();
        a.resize(d * d, 0.0);
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
        cholesky_solve(a, &self.b, d)
    }
}

/// Solve `A x = b` for symmetric positive-definite `A` (row-major, overwritten
/// with its Cholesky factor). `None` when `A` is not positive definite.
pub fn cholesky_solve(a: &mut [f64], b: &[f64], d: usize) -> Option<Vec<f64>> {
    for j in 0..d {
        let s = a[j * d + j] - dot(&a[j * d..j * d + j], &a[j * d..j * d + j]);
        if !(s > 0.0) || !s.is_finite() {
            return None;
        }
        let l = sqrt(s);
        a[j * d + j] = l;
        for i in j + 1..d {
            let t = a[i * d + j] - dot(&a[i * d..i * d + j], &a[j * d..j * d + j]);
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

/// The fitted one-step model in original units:
/// `y_t = Σ_j a_j y_{t−L+j} + k0 + k1·pos_t`. Every scaler is affine in the
/// window: `y = c + Σ w_j (y_j − c) + sd·(w_pos·pos + w_bias)` with `c` a
/// constant, the last value or the window mean.
struct Lin {
    a: [f64; 32],
    l: usize,
    k0: f64,
    k1: f64,
}

fn linear(w: &[f64], l: usize, centre: Centre, sd: f64, position: bool) -> Lin {
    let mut a = [0.0; 32];
    a[..l].copy_from_slice(&w[..l]);
    let rest = 1.0 - w[..l].iter().sum::<f64>();
    let wb = sd * w[w.len() - 1];
    let k0 = match centre {
        Centre::Const(mu) => mu * rest + wb,
        Centre::Last => {
            a[l - 1] += rest;
            wb
        }
        Centre::Mean => {
            let r = rest / l as f64;
            for x in &mut a[..l] {
                *x += r;
            }
            wb
        }
    };
    Lin { a, l, k0, k1: if position { sd * w[l] } else { 0.0 } }
}

/// Dot product with four partial sums (no bounds checks in the loop).
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let (ac, bc) = (a.chunks_exact(4), b.chunks_exact(4));
    let (ar, br) = (ac.remainder(), bc.remainder());
    let mut s = [0.0f64; 4];
    for (x, y) in ac.zip(bc) {
        s[0] += x[0] * y[0];
        s[1] += x[1] * y[1];
        s[2] += x[2] * y[2];
        s[3] += x[3] * y[3];
    }
    let mut t = (s[0] + s[1]) + (s[2] + s[3]);
    for (x, y) in ar.iter().zip(br) {
        t += x * y;
    }
    t
}

/// Roll the model forward from the window `init` (length `L`) at the
/// position features `pos`; the forecasts are `buf[L..]`.
fn rollout(m: &Lin, init: &[f64], pos: &[f64], buf: &mut Vec<f64>) {
    let l = m.l;
    buf.clear();
    buf.extend_from_slice(init);
    for (t, &p) in pos.iter().enumerate() {
        let y = dot(&m.a[..l], &buf[t..t + l]) + m.k0 + m.k1 * p;
        buf.push(y);
    }
}

/// One scored point of the search, in the original iteration order.
struct Cand {
    li: usize,
    scaler: Scaler,
    position: bool,
    ci: usize,
    /// Position in the search order (lookback, scaler, position, λ, σ).
    order: usize,
    val: f64,
}

/// Forecast `steps`/`vals` up to `opt.horizon_step`. `pool` holds other runs
/// of the same metric (`(steps, vals)`); runs with a different median step
/// spacing are ignored. `None` when the target has fewer than
/// [`MIN_POINTS`] finite points, the horizon is not past the last step, or no
/// configuration can be backtested.
pub fn forecast(steps: &[f64], vals: &[f64], pool: &[(&[f64], &[f64])], opt: &Options) -> Option<Forecast> {
    search(steps, vals, pool, opt, true)
}

/// [`forecast`]; `prune = false` scores every configuration in full on
/// validation (same result, kept for the equivalence test).
fn search(steps: &[f64], vals: &[f64], pool: &[(&[f64], &[f64])], opt: &Options, prune: bool) -> Option<Forecast> {
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

    // Search. Each series is walked once per lookback into prefix Grams
    // (raw_prefixes); every (family, scaler, position, origin) Gram is an
    // O(d²) transform of those, and all (λ, σ) are solved on it.
    // Phase 1 rolls out only the validation origins, up to the first test
    // origin, which is all that selection reads. Phase 2 walks each family's
    // configs by validation MAE (ties: search order, as a strict `<` scan)
    // and keeps the first whose rollouts are finite at every origin and
    // horizon, the same winner as scoring every config in full.
    const SCALERS: [Scaler; 5] = [Scaler::Standard, Scaler::Robust, Scaler::LocalLast, Scaler::LocalMean, Scaler::Global];
    let n_o = origins.len();
    let tpos: Vec<f64> = t.st.iter().map(|&s| log1p(s.max(0.0)) / ref_ln).collect();
    let mean_of = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let t_shift = mean_of(&t.v);
    // stats per (scaler, origin); index n_o is the full series
    let tstats: Vec<Vec<(f64, f64)>> = SCALERS
        .iter()
        .map(|&s| origins.iter().map(|&o| series_stats(&t.v[..o], s, gstats)).chain(core::iter::once(series_stats(&t.v, s, gstats))).collect())
        .collect();
    let pstats: Vec<Vec<(f64, f64)>> = SCALERS.iter().map(|&s| pool.iter().map(|p| series_stats(&p.v, s, gstats)).collect()).collect();
    let mut ends = origins.clone();
    ends.push(n);
    let usable = |l: usize| l <= origins[0];
    let traw: Vec<Vec<Raw>> = LOOKBACKS.iter().map(|&l| if usable(l) { raw_prefixes(&t.v, &tpos, t_shift, l, &ends) } else { Vec::new() }).collect();
    let ppos: Vec<Vec<f64>> = pool.iter().map(|p| p.st.iter().map(|&s| log1p(s.max(0.0)) / ref_ln).collect()).collect();
    let praw: Vec<Vec<Raw>> = LOOKBACKS
        .iter()
        .map(|&l| {
            if !usable(l) {
                return Vec::new();
            }
            pool.iter().zip(&ppos).map(|(p, pos)| raw_prefixes(&p.v, pos, mean_of(&p.v), l, &[p.v.len()]).pop().unwrap()).collect()
        })
        .collect();
    // Gram of one (lookback, scaler, position, family) at origin index k (n_o = full series)
    let gram = |li: usize, scaler: Scaler, position: bool, pooled: bool, k: usize| -> Gram {
        let l = LOOKBACKS[li];
        let mut g = Gram::new(l + position as usize + 1);
        let s = scaler as usize;
        if pooled {
            for (p, r) in praw[li].iter().enumerate() {
                let (mu, sd) = pstats[s][p];
                g.add_raw(r, l, centre_of(scaler, mu), sd, position);
            }
        }
        let (mu, sd) = tstats[s][k];
        g.add_raw(&traw[li][k], l, centre_of(scaler, mu), sd, position);
        g
    };
    let nc = LAMBDAS.len() * NOISES.len();
    let lam_noise = |ci: usize| (LAMBDAS[ci / NOISES.len()], NOISES[ci % NOISES.len()]);
    // Phase 1. Validation error is a running sum of non-negative terms
    // (accumulated in pair order, so a partial sum never exceeds the final
    // one), and a config whose partial sum already exceeds the best complete
    // sum of its family cannot be selected: with `prune` it is dropped there
    // (branch and bound). The 15 (λ, σ) chains of one origin step in
    // lockstep, which hides the latency of each recurrence.
    let phase1 = |prune: bool| -> ([Vec<Cand>; 2], usize) {
        const BLOCK: usize = 32;
        let mut n_configs = 0usize;
        let mut cands: [Vec<Cand>; 2] = [Vec::new(), Vec::new()];
        let (mut sums, mut ok, mut cut) = (vec![0.0; nc], vec![true; nc], vec![false; nc]);
        let (mut models, mut live, mut act): (Vec<Lin>, Vec<usize>, Vec<usize>) = (Vec::with_capacity(nc), Vec::with_capacity(nc), Vec::with_capacity(nc));
        let (mut abuf, mut buf) = (Vec::new(), Vec::new());
        for pooled in [false, true] {
            if pooled && pool.is_empty() {
                continue;
            }
            let mut bound = f64::INFINITY;
            let scalers: &[Scaler] = if pooled { &SCALERS } else { &SCALERS[..4] };
            for (li, &lookback) in LOOKBACKS.iter().enumerate() {
                // the rollout needs a full window of the target's own prefix at every origin
                if !usable(lookback) {
                    continue;
                }
                if !pooled && origins[0] < lookback + 2 {
                    continue;
                }
                if pooled && praw[li].iter().all(|r| r.n == 0) {
                    continue;
                }
                for &scaler in scalers {
                    for position in [false, true] {
                        sums.fill(0.0);
                        ok.fill(true);
                        cut.fill(false);
                        for (k, &o) in origins.iter().enumerate() {
                            if o >= first_test {
                                break;
                            }
                            let g = gram(li, scaler, position, pooled, k);
                            let (mu, sd) = tstats[scaler as usize][k];
                            let hv = h_bt.min(n - o).min(first_test - o);
                            models.clear();
                            live.clear();
                            for ci in 0..nc {
                                if !ok[ci] || cut[ci] {
                                    continue;
                                }
                                let (lambda, noise) = lam_noise(ci);
                                match if g.n >= 2 { g.solve(lookback, lambda, noise, &mut abuf) } else { None } {
                                    Some(w) => {
                                        models.push(linear(&w, lookback, centre_of(scaler, mu), sd, position));
                                        live.push(ci);
                                    }
                                    None => ok[ci] = false,
                                }
                            }
                            let stride = lookback + hv;
                            buf.clear();
                            buf.resize(live.len() * stride, 0.0);
                            for c in 0..live.len() {
                                buf[c * stride..c * stride + lookback].copy_from_slice(&t.v[o - lookback..o]);
                            }
                            act.clear();
                            act.extend(0..live.len());
                            let mut t0 = 0;
                            while t0 < hv && !act.is_empty() {
                                let t1 = (t0 + BLOCK).min(hv);
                                for i in t0..t1 {
                                    let p = tpos[o + i];
                                    for &c in &act {
                                        let m = &models[c];
                                        let base = c * stride + i;
                                        buf[base + lookback] = dot(&m.a[..lookback], &buf[base..base + lookback]) + m.k0 + m.k1 * p;
                                    }
                                }
                                for i in t0..t1 {
                                    let y0 = t.v[o + i];
                                    for &c in &act {
                                        sums[live[c]] += (y0 - buf[c * stride + lookback + i]).abs();
                                    }
                                }
                                act.retain(|&c| {
                                    let ci = live[c];
                                    if !buf[c * stride + lookback + t0..c * stride + lookback + t1].iter().all(|y| y.is_finite()) {
                                        ok[ci] = false;
                                        return false;
                                    }
                                    if prune && sums[ci] > bound {
                                        cut[ci] = true;
                                        return false;
                                    }
                                    true
                                });
                                t0 = t1;
                            }
                        }
                        for ci in 0..nc {
                            if ok[ci] {
                                n_configs += 1;
                                if !cut[ci] {
                                    bound = bound.min(sums[ci]);
                                    let order = ((li * SCALERS.len() + scaler as usize) * 2 + position as usize) * nc + ci;
                                    cands[pooled as usize].push(Cand { li, scaler, position, ci, order, val: sums[ci] / n_val as f64 });
                                }
                            }
                        }
                    }
                }
            }
        }
        (cands, n_configs)
    };
    let (mut abuf, mut rbuf) = (Vec::new(), Vec::new());
    // full backtest of one config: residuals per pair, None if any fit fails or any rollout is non-finite
    let mut full = |c: &Cand, pooled: bool| -> Option<Vec<f64>> {
        let l = LOOKBACKS[c.li];
        let (lambda, noise) = lam_noise(c.ci);
        let mut preds: Vec<Vec<f64>> = Vec::with_capacity(n_o);
        for (k, &o) in origins.iter().enumerate() {
            let g = gram(c.li, c.scaler, c.position, pooled, k);
            if g.n < 2 {
                return None;
            }
            let w = g.solve(l, lambda, noise, &mut abuf)?;
            let (mu, sd) = tstats[c.scaler as usize][k];
            let hm = h_bt.min(n - o);
            rollout(&linear(&w, l, centre_of(c.scaler, mu), sd, c.position), &t.v[o - l..o], &tpos[o..o + hm], &mut rbuf);
            if rbuf.iter().any(|y| !y.is_finite()) {
                return None;
            }
            preds.push(rbuf[l..].to_vec());
        }
        Some(pairs.iter().map(|&(oi, h, _)| t.v[origins[oi] + h - 1] - preds[oi][h - 1]).collect())
    };
    let (mut cands, n_configs) = phase1(prune);
    let mut best: [Option<(Config, f64, Vec<f64>)>; 2] = [None, None]; // (cfg, val mae, residuals per pair)
    for fam in 0..2 {
        let mut pass = 0;
        while best[fam].is_none() && pass < 2 {
            if pass == 1 {
                // every unpruned config failed in full: rescore without pruning (rare)
                if cands[fam].is_empty() {
                    break;
                }
                cands[fam] = phase1(false).0[fam].drain(..).collect();
            }
            pass += 1;
        let cs = &mut cands[fam];
        cs.sort_by(|a, b| a.val.partial_cmp(&b.val).unwrap_or(core::cmp::Ordering::Equal).then(a.order.cmp(&b.order))); // ties: search order, as a strict `<` scan
        for c in cs.iter() {
            if let Some(res) = full(c, fam == 1) {
                let (lambda, noise) = lam_noise(c.ci);
                let cfg = Config { lookback: LOOKBACKS[c.li], lambda, noise, scaler: c.scaler, pooled: fam == 1, position: c.position };
                best[fam] = Some((cfg, c.val, res));
                break;
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
        let li = LOOKBACKS.iter().position(|&l| l == cfg.lookback)?;
        let g = gram(li, cfg.scaler, cfg.position, cfg.pooled, n_o);
        let w = g.solve(cfg.lookback, cfg.lambda, cfg.noise, &mut abuf)?;
        let (mu, sd) = tstats[cfg.scaler as usize][n_o];
        let fpos: Vec<f64> = future.iter().map(|&s| log1p(s.max(0.0)) / ref_ln).collect();
        rollout(&linear(&w, cfg.lookback, centre_of(cfg.scaler, mu), sd, cfg.position), &t.v[n - cfg.lookback..], &fpos, &mut rbuf);
        let m = rbuf[cfg.lookback..].to_vec();
        if m.iter().any(|x| !x.is_finite()) {
            return None;
        }
        m
    } else {
        vec![t.v[n - 1]; h_out]
    };

    // conformal-style band from the drawn method's backtest residuals, per horizon
    let resid: Vec<f64> = if skill { res } else { pairs.iter().map(|&(oi, h, _)| t.v[origins[oi] + h - 1] - t.v[origins[oi] - 1]).collect() };
    let mut grouped: Vec<Vec<f64>> = vec![Vec::new(); h_bt + 1]; // residuals by horizon, in pair order
    for (r, p) in resid.iter().zip(&pairs) {
        grouped[p.1].push(*r);
    }
    let by_h = |h: usize| -> Vec<f64> { grouped[h].clone() };
    let max_backtest_h = (1..=h_bt).filter(|&h| grouped[h].len() >= 3).max().unwrap_or(0);
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
    fn pruned_search_selects_exactly_what_the_full_search_does() {
        let mut x = 7u64;
        let mut rnd = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % 20001) as f64 / 10000.0 - 1.0
        };
        for case in 0..24 {
            let n = [6, 9, 15, 40, 120, 400][case % 6];
            let s: Vec<f64> = (0..n).map(|i| (i * 10) as f64).collect();
            let kind = case / 6;
            let mut walk = 0.0;
            let mut series = |off: f64| -> Vec<f64> {
                (0..n)
                    .map(|i| {
                        let e = rnd();
                        walk += e;
                        let t = i as f64 + 1.0;
                        match kind {
                            0 => 2.0 + off + 5.0 * libm::pow(t, -0.5) + 0.05 * e,
                            1 => walk,
                            2 => 1e4 + off + libm::sin(t / 3.0) + 0.01 * e,
                            _ => off + 0.3 * t + 2.0 * e,
                        }
                    })
                    .collect()
            };
            let v = series(0.0);
            let pools: Vec<Vec<f64>> = (0..(case % 3)).map(|k| series(0.1 * k as f64)).collect();
            let pool: Vec<(&[f64], &[f64])> = pools.iter().map(|p| (s.as_slice(), p.as_slice())).collect();
            let o = Options::to_step(s[n - 1] * 1.6);
            let (a, b) = (search(&s, &v, &pool, &o, true), search(&s, &v, &pool, &o, false));
            match (a, b) {
                (None, None) => {}
                (Some(a), Some(b)) => {
                    assert_eq!(a.config, b.config, "case {case}");
                    assert_eq!((a.skill, a.single, a.pooled), (b.skill, b.single, b.pooled), "case {case}");
                    assert_eq!((&a.mean, &a.lo, &a.hi), (&b.mean, &b.lo, &b.hi), "case {case}");
                    assert_eq!(a.n_configs, b.n_configs, "case {case}");
                }
                (a, b) => panic!("case {case}: {a:?} vs {b:?}"),
            }
        }
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
