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

// ------------------------------------------------------------------ panel queries

/// Parse a panel query: `ok()`, `error()`/`err_start()`/`err_end()` on
/// failure; `canonical()` text and `chips()` JSON
/// (`[{kind, text, entry, letter}]`) on success.
#[wasm_bindgen]
pub struct QueryParse {
    canonical: String,
    chips: String,
    error: Option<crate::query::QueryError>,
}

#[wasm_bindgen]
impl QueryParse {
    pub fn ok(&self) -> bool {
        self.error.is_none()
    }
    pub fn canonical(&self) -> String {
        self.canonical.clone()
    }
    pub fn chips(&self) -> String {
        self.chips.clone()
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
    /// Newline-joined replacements for `err_start..err_end` ("did you mean"), best first.
    pub fn suggest(&self) -> String {
        self.error.as_ref().map(|e| e.suggest.join("\n")).unwrap_or_default()
    }
}

fn qp(r: Result<crate::query::Query, crate::query::QueryError>) -> QueryParse {
    match r {
        Ok(q) => QueryParse { canonical: crate::query::canonical(&q), chips: crate::query::chips_json(&q), error: None },
        Err(e) => QueryParse { canonical: String::new(), chips: "[]".into(), error: Some(e) },
    }
}

#[wasm_bindgen]
pub fn query_parse(src: &str) -> QueryParse {
    qp(crate::query::parse(src))
}

/// Replace chip `idx` with `text` (empty removes it); the result parses as
/// the new query (its `canonical()`), or carries the error.
#[wasm_bindgen]
pub fn query_replace_chip(src: &str, idx: usize, text: &str) -> QueryParse {
    match crate::query::replace_chip(src, idx, text) {
        Ok(s) => qp(crate::query::parse(&s)),
        Err(e) => QueryParse { canonical: String::new(), chips: "[]".into(), error: Some(e) },
    }
}

/// Streams a query needs, as `run\tmetric` lines in the order `query_eval`
/// expects their data. Empty when the query does not parse.
#[wasm_bindgen]
pub fn query_plan(src: &str, catalog: &str) -> String {
    let Ok(q) = crate::query::parse(src) else { return String::new() };
    let cat = crate::query::Catalog::parse(catalog);
    let p = crate::query::plan(&q, &cat);
    p.streams.iter().map(|s| format!("{}\t{}", cat.runs[s.run].id, s.metric)).collect::<Vec<_>>().join("\n")
}

#[wasm_bindgen]
pub struct QueryOut {
    meta: String,
    data: Vec<f64>,
}

#[wasm_bindgen]
impl QueryOut {
    /// JSON: `{traces: [{letter, label, metric, runs, band, cell, color, off, n}], cells, warnings, by, by_numeric, ylog, forecast, x_label, downsampled}`.
    pub fn meta(&self) -> String {
        self.meta.clone()
    }
    /// Trace `i`: `data[off .. off + 6n]` = xs, ys, raw, lo, hi, steps.
    pub fn data(&self) -> Vec<f64> {
        self.data.clone()
    }
}

/// Evaluate a query; stream `i` of `query_plan` is `steps/vals[off[i]..off[i+1]]`.
#[wasm_bindgen]
pub fn query_eval(src: &str, catalog: &str, off: &[u32], steps: &[f64], vals: &[f64], max_points: usize) -> QueryOut {
    let Ok(q) = crate::query::parse(src) else { return QueryOut { meta: "{\"traces\":[],\"cells\":[],\"warnings\":[\"query does not parse\"]}".into(), data: Vec::new() } };
    let cat = crate::query::Catalog::parse(catalog);
    let p = crate::query::plan(&q, &cat);
    let data: Vec<(&[f64], &[f64])> = off
        .windows(2)
        .map(|w| (w[0] as usize, w[1] as usize))
        .map(|(a, b)| if a <= b && b <= steps.len() && b <= vals.len() { (&steps[a..b], &vals[a..b]) } else { (&steps[..0], &vals[..0]) })
        .collect();
    let mut o = crate::query::eval(&q, &cat, &p, &data, max_points);
    o.fixes = crate::query::metric_fixes(src, &cat);
    let (meta, data) = crate::query::output_json(&o, &cat);
    QueryOut { meta, data }
}

// ------------------------------------------------------------------ curve helpers

/// Power-law forecast; flat result `[skill, mae_test, mae_naive, n_test, a, b, c, n, steps…, mean…, lo…, hi…]`,
/// empty when there are too few points or the horizon is not past the data.
#[wasm_bindgen]
pub fn powerlaw_forecast(xs: &[f64], ys: &[f64], horizon: f64, n_out: usize) -> Vec<f64> {
    let Some(f) = crate::curve::powerlaw_forecast(xs, ys, horizon, n_out) else { return Vec::new() };
    let mut v = vec![f.skill as u8 as f64, f.mae_test, f.mae_test_naive, f.n_test as f64, f.params[0], f.params[1], f.params[2], f.steps.len() as f64];
    v.extend(f.steps);
    v.extend(f.mean);
    v.extend(f.lo);
    v.extend(f.hi);
    v
}

/// `[mean, early, late]` x at which a forecast reaches `target` (`NaN` = not reached).
#[wasm_bindgen]
pub fn forecast_eta(xs: &[f64], mean: &[f64], lo: &[f64], hi: &[f64], target: f64) -> Vec<f64> {
    crate::curve::eta(xs, mean, lo, hi, target).to_vec()
}

#[wasm_bindgen]
pub fn spikes(vals: &[f64], window: usize, k: f64) -> Vec<u32> {
    crate::curve::spikes(vals, window, k)
}

/// [`forecast`] with a cap on the number of rolled-out points (`forecast`
/// uses 64); the builder's horizon handle can reach far past the data.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn forecast_n(steps: &[f64], vals: &[f64], pool_off: &[u32], pool_steps: &[f64], pool_vals: &[f64], horizon_step: f64, allow_pool: bool, max_points: usize) -> Option<ForecastOut> {
    let pool: Vec<(&[f64], &[f64])> = pool_off
        .windows(2)
        .map(|w| (w[0] as usize, w[1] as usize))
        .filter(|&(a, b)| a <= b && b <= pool_steps.len() && b <= pool_vals.len())
        .map(|(a, b)| (&pool_steps[a..b], &pool_vals[a..b]))
        .collect();
    let mut opt = crate::forecast::Options::to_step(horizon_step);
    opt.allow_pool = allow_pool;
    opt.max_points = max_points.clamp(1, 1024);
    crate::forecast::forecast(steps, vals, &pool, &opt).map(|inner| ForecastOut { inner })
}

// ------------------------------------------------------------------ LLM views

/// Families of per-layer metrics among the newline-joined `names` with at
/// least `min_layers` distinct layers, as JSON
/// `[{"family": <template>, "layers": [0, 1, …], "metrics": [...]}]`, where the
/// template is the metric name with the layer index replaced by a star.
#[wasm_bindgen]
pub fn layer_families(names: &str, min_layers: usize) -> String {
    let n: Vec<&str> = split_names(names);
    let fams = crate::layers::families(&n, min_layers);
    let items: Vec<String> = fams
        .iter()
        .map(|f| {
            let layers: Vec<String> = f.members.iter().map(|m| format!("{}", m.0)).collect();
            let metrics: Vec<String> = f.members.iter().map(|m| crate::query::js_str(&m.1)).collect();
            format!("{{\"family\":{},\"layers\":[{}],\"metrics\":[{}]}}", crate::query::js_str(&f.template), layers.join(","), metrics.join(","))
        })
        .collect();
    format!("[{}]", items.join(","))
}

/// `"<family>\t<layer>"` for a per-layer metric name, empty otherwise.
#[wasm_bindgen]
pub fn layer_index(name: &str) -> String {
    crate::layers::layer_index(name).map(|(t, i)| format!("{t}\t{i}")).unwrap_or_default()
}

/// Word-level diff of two texts as JSON `{"ops": [[op, text], …], "same": n}`
/// where op 0 = same, 1 = inserted (only in `b`), 2 = deleted (only in `a`).
#[wasm_bindgen]
pub fn diff_words(a: &str, b: &str) -> String {
    let (ops, same) = crate::text::diff_words(a, b);
    let items: Vec<String> = ops
        .iter()
        .map(|(o, t)| format!("[{},{}]", match o { crate::text::Op::Same => 0, crate::text::Op::Ins => 1, crate::text::Op::Del => 2 }, crate::query::js_str(t)))
        .collect();
    format!("{{\"ops\":[{}],\"same\":{same}}}", items.join(","))
}

/// `[lo, hi, count_0, …, count_{bins-1}]` over the finite values; empty when none.
#[wasm_bindgen]
pub fn histogram(vals: &[f64], bins: usize) -> Vec<f64> {
    let (lo, hi, c) = crate::text::histogram(vals, bins);
    if c.is_empty() {
        return Vec::new();
    }
    let mut v = vec![lo, hi];
    v.extend(c.iter().map(|&x| x as f64));
    v
}

/// Spikes and non-finite values as flat triples `[index, kind, ratio]`:
/// kind 0 = spike (ratio = value / trailing median), 1 = non-finite (ratio NaN).
#[wasm_bindgen]
pub fn anomalies(vals: &[f64], window: usize, k: f64) -> Vec<f64> {
    let mut out = Vec::new();
    for (i, a) in crate::curve::anomalies(vals, window, k) {
        let (kind, r) = match a {
            crate::curve::Anomaly::Spike { ratio } => (0.0, ratio),
            crate::curve::Anomaly::NonFinite => (1.0, f64::NAN),
        };
        out.extend([i as f64, kind, r]);
    }
    out
}
