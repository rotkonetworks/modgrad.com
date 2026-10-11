//! wasm-bindgen exports. Slices cross as typed arrays (`Float64Array`,
//! `Float32Array`, `Uint32Array`); multi-output results are small structs
//! with getter methods.

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn ema(vals: &[f64], weight: f64) -> Vec<f64> {
    crate::ema(vals, weight)
}

/// `[max, avg, min, current]`.
#[wasm_bindgen]
pub fn legend_stats(vals: &[f64]) -> Vec<f64> {
    crate::legend_stats(vals).to_vec()
}

#[wasm_bindgen]
pub struct Series {
    steps: Vec<f64>,
    mean: Vec<f64>,
    min: Vec<f64>,
    max: Vec<f64>,
}

#[wasm_bindgen]
impl Series {
    pub fn steps(&self) -> Vec<f64> {
        self.steps.clone()
    }
    pub fn mean(&self) -> Vec<f64> {
        self.mean.clone()
    }
    pub fn min(&self) -> Vec<f64> {
        self.min.clone()
    }
    pub fn max(&self) -> Vec<f64> {
        self.max.clone()
    }
}

#[wasm_bindgen]
pub fn downsample_series(steps: &[f64], vals: &[f64], max_points: usize) -> Series {
    let (steps, mean, min, max) = crate::downsample_series(steps, vals, max_points);
    Series { steps, mean, min, max }
}

#[wasm_bindgen]
pub struct Pca {
    coords: Vec<f32>,
    explained: Vec<f64>,
}

#[wasm_bindgen]
impl Pca {
    /// `n × 3` row-major.
    pub fn coords(&self) -> Vec<f32> {
        self.coords.clone()
    }
    /// Explained-variance ratio of PC1..PC3.
    pub fn explained(&self) -> Vec<f64> {
        self.explained.clone()
    }
}

#[wasm_bindgen]
pub fn pca3(data: &[f32], n: usize, d: usize) -> Pca {
    let (coords, r) = crate::pca3(data, n, d);
    Pca { coords, explained: r.to_vec() }
}

#[wasm_bindgen]
pub struct SurfaceOut {
    inner: crate::Surface,
}

#[wasm_bindgen]
impl SurfaceOut {
    pub fn xs(&self) -> Vec<f64> {
        self.inner.xs.clone()
    }
    pub fn ys(&self) -> Vec<f64> {
        self.inner.ys.clone()
    }
    pub fn z(&self) -> Vec<f64> {
        self.inner.z.clone()
    }
    pub fn count(&self) -> Vec<u32> {
        self.inner.count.clone()
    }
}

/// `off` has `runs + 1` entries (as `Uint32Array`).
#[wasm_bindgen]
pub fn assemble_surface(run_x: &[f64], off: &[u32], steps: &[f64], vals: &[f64], step_bin: f64) -> SurfaceOut {
    let off: Vec<usize> = off.iter().map(|&o| o as usize).collect();
    SurfaceOut { inner: crate::assemble_surface(run_x, &off, steps, vals, step_bin) }
}

/// `[sx, sy, depth]` per point.
#[wasm_bindgen]
pub fn project(xyz: &[f64], az_deg: f64, el_deg: f64) -> Vec<f64> {
    crate::project(xyz, az_deg, el_deg)
}

/// Scaled points followed by 6 bounds: `[…3n…, minx, maxx, miny, maxy, minz, maxz]`.
#[wasm_bindgen]
pub fn normalize_box(xyz: &[f64]) -> Vec<f64> {
    let (mut v, b) = crate::normalize_box(xyz);
    v.extend_from_slice(&b);
    v
}

#[wasm_bindgen]
pub fn painter_order(nx: usize, ny: usize, depth: &[f64]) -> Vec<u32> {
    crate::painter_order(nx, ny, depth)
}

#[wasm_bindgen]
pub fn depth_order(depth: &[f64]) -> Vec<u32> {
    crate::depth_order(depth)
}

#[wasm_bindgen]
pub fn nice_ticks(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    crate::nice_ticks(lo, hi, n)
}

/// Mean-pool an N-d array to at most `max_elems`; returns data, and the new
/// shape via `downsample_shape` (same arguments).
#[wasm_bindgen]
pub fn downsample(shape: &[u32], data: &[f32], max_elems: usize) -> Vec<f32> {
    let s: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
    crate::downsample(&s, data, max_elems).1
}

#[wasm_bindgen]
pub fn downsample_shape(shape: &[u32], max_elems: usize) -> Vec<u32> {
    let mut s: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
    let mut n: usize = s.iter().product();
    while n > max_elems.max(1) {
        let (ax, &len) = s.iter().enumerate().max_by_key(|&(_, l)| *l).unwrap();
        if len <= 1 {
            break;
        }
        s[ax] = len.div_ceil(2);
        n = s.iter().product();
    }
    s.into_iter().map(|x| x as u32).collect()
}

/// Ghost forecast (see [`crate::forecast`]). Getters return `NaN` for a
/// family that had no eligible configuration.
#[wasm_bindgen]
pub struct ForecastOut {
    inner: crate::forecast::Forecast,
}

#[wasm_bindgen]
impl ForecastOut {
    pub fn steps(&self) -> Vec<f64> {
        self.inner.steps.clone()
    }
    pub fn mean(&self) -> Vec<f64> {
        self.inner.mean.clone()
    }
    pub fn lo(&self) -> Vec<f64> {
        self.inner.lo.clone()
    }
    pub fn hi(&self) -> Vec<f64> {
        self.inner.hi.clone()
    }
    /// Ridge beat last-value naive on validation and test; when false the
    /// drawn forecast is the last value.
    pub fn skill(&self) -> bool {
        self.inner.skill
    }
    pub fn lookback(&self) -> usize {
        self.inner.config.lookback
    }
    pub fn lambda(&self) -> f64 {
        self.inner.config.lambda
    }
    pub fn noise(&self) -> f64 {
        self.inner.config.noise
    }
    pub fn scaler(&self) -> String {
        self.inner.config.scaler.name().into()
    }
    pub fn pooled(&self) -> bool {
        self.inner.config.pooled
    }
    pub fn position(&self) -> bool {
        self.inner.config.position
    }
    pub fn mae_val(&self) -> f64 {
        self.inner.mae_val
    }
    pub fn mae_val_naive(&self) -> f64 {
        self.inner.mae_val_naive
    }
    pub fn mae_test(&self) -> f64 {
        self.inner.mae_test
    }
    pub fn mae_test_naive(&self) -> f64 {
        self.inner.mae_test_naive
    }
    pub fn n_val(&self) -> usize {
        self.inner.n_val
    }
    pub fn n_test(&self) -> usize {
        self.inner.n_test
    }
    pub fn single_mae_val(&self) -> f64 {
        self.inner.single.map_or(f64::NAN, |b| b.mae_val)
    }
    pub fn single_mae_test(&self) -> f64 {
        self.inner.single.map_or(f64::NAN, |b| b.mae_test)
    }
    pub fn pooled_mae_val(&self) -> f64 {
        self.inner.pooled.map_or(f64::NAN, |b| b.mae_val)
    }
    pub fn pooled_mae_test(&self) -> f64 {
        self.inner.pooled.map_or(f64::NAN, |b| b.mae_test)
    }
    pub fn max_backtest_h(&self) -> usize {
        self.inner.max_backtest_h
    }
    pub fn n_pool(&self) -> usize {
        self.inner.n_pool
    }
    pub fn spacing(&self) -> f64 {
        self.inner.spacing
    }
    pub fn n_configs(&self) -> usize {
        self.inner.n_configs
    }
}

/// Forecast `vals` over `steps` up to `horizon_step`. The pool (other runs of
/// the same metric) is flattened: run `i` is `pool_steps/vals[off[i]..off[i+1]]`.
/// `None` (undefined) for series shorter than `forecast::MIN_POINTS`.
#[wasm_bindgen]
pub fn forecast(steps: &[f64], vals: &[f64], pool_off: &[u32], pool_steps: &[f64], pool_vals: &[f64], horizon_step: f64, allow_pool: bool) -> Option<ForecastOut> {
    let pool: Vec<(&[f64], &[f64])> = pool_off
        .windows(2)
        .map(|w| (w[0] as usize, w[1] as usize))
        .filter(|&(a, b)| a <= b && b <= pool_steps.len() && b <= pool_vals.len())
        .map(|(a, b)| (&pool_steps[a..b], &pool_vals[a..b]))
        .collect();
    let mut opt = crate::forecast::Options::to_step(horizon_step);
    opt.allow_pool = allow_pool;
    crate::forecast::forecast(steps, vals, &pool, &opt).map(|inner| ForecastOut { inner })
}

/// `[mean, min, max, rms]` of one array record.
#[wasm_bindgen]
pub fn array_stats(data: &[f32]) -> Vec<f64> {
    crate::array_stats(data).to_vec()
}

// ------------------------------------------------------------------ graph builder

/// Per-panel normalisation: `"z"`, `"minmax"`, `"first"`, else identity
/// (see [`crate::expr::normalize`]).
#[wasm_bindgen]
pub fn normalize(vals: &[f64], mode: &str) -> Vec<f64> {
    crate::expr::normalize(vals, mode)
}

/// Two series on one step grid (`"intersection"`, `"union"`, `"exact"`),
/// e.g. to plot metric B against metric A joined on step.
#[wasm_bindgen]
pub struct Aligned {
    steps: Vec<f64>,
    a: Vec<f64>,
    b: Vec<f64>,
}

#[wasm_bindgen]
impl Aligned {
    pub fn steps(&self) -> Vec<f64> {
        self.steps.clone()
    }
    pub fn a(&self) -> Vec<f64> {
        self.a.clone()
    }
    pub fn b(&self) -> Vec<f64> {
        self.b.clone()
    }
}

#[wasm_bindgen]
pub fn align2(a_steps: &[f64], a_vals: &[f64], b_steps: &[f64], b_vals: &[f64], mode: &str) -> Aligned {
    let (steps, mut v) = crate::expr::align(&[(a_steps, a_vals), (b_steps, b_vals)], crate::expr::Align::parse(mode));
    let b = v.pop().unwrap_or_default();
    let a = v.pop().unwrap_or_default();
    Aligned { steps, a, b }
}

/// Result of [`expr_refs`] / [`eval_expr`]. On error `ok()` is false and
/// `error()`, `err_start()`, `err_end()` (byte offsets into the
/// expression) describe it.
#[wasm_bindgen]
pub struct ExprOut {
    steps: Vec<f64>,
    values: Vec<f64>,
    refs: String,
    error: Option<crate::expr::ExprError>,
}

#[wasm_bindgen]
impl ExprOut {
    pub fn ok(&self) -> bool {
        self.error.is_none()
    }
    pub fn steps(&self) -> Vec<f64> {
        self.steps.clone()
    }
    pub fn values(&self) -> Vec<f64> {
        self.values.clone()
    }
    /// Referenced names, newline-joined.
    pub fn refs(&self) -> String {
        self.refs.clone()
    }
    pub fn error(&self) -> String {
        self.error.as_ref().map(|e| e.msg.clone()).unwrap_or_default()
    }
    pub fn err_start(&self) -> usize {
        self.error.as_ref().map_or(0, |e| e.start)
    }
    pub fn err_end(&self) -> usize {
        self.error.as_ref().map_or(0, |e| e.end)
    }
}

fn split_names(names: &str) -> Vec<&str> {
    names.split('\n').filter(|s| !s.is_empty()).collect()
}

/// Parse only: which of the newline-joined `names` does `expr` use?
#[wasm_bindgen]
pub fn expr_refs(expr: &str, names: &str) -> ExprOut {
    match crate::expr::refs(expr, &split_names(names)) {
        Ok(r) => ExprOut { steps: Vec::new(), values: Vec::new(), refs: r.join("\n"), error: None },
        Err(e) => ExprOut { steps: Vec::new(), values: Vec::new(), refs: String::new(), error: Some(e) },
    }
}

/// Evaluate `expr` over the series named by the newline-joined `names`;
/// series `i` is `steps/vals[off[i]..off[i+1]]`.
#[wasm_bindgen]
pub fn eval_expr(expr: &str, names: &str, off: &[u32], steps: &[f64], vals: &[f64], mode: &str) -> ExprOut {
    let names = split_names(names);
    let data: Vec<(&[f64], &[f64])> = off
        .windows(2)
        .map(|w| (w[0] as usize, w[1] as usize))
        .map(|(a, b)| if a <= b && b <= steps.len() && b <= vals.len() { (&steps[a..b], &vals[a..b]) } else { (&steps[..0], &vals[..0]) })
        .collect();
    if data.len() != names.len() {
        let msg = format!("internal: {} names but {} series", names.len(), data.len());
        return ExprOut { steps: Vec::new(), values: Vec::new(), refs: String::new(), error: Some(crate::expr::ExprError { start: 0, end: 0, msg }) };
    }
    match crate::expr::eval(expr, &names, &data, crate::expr::Align::parse(mode)) {
        Ok((s, v)) => ExprOut { steps: s, values: v, refs: String::new(), error: None },
        Err(e) => ExprOut { steps: Vec::new(), values: Vec::new(), refs: String::new(), error: Some(e) },
    }
}
