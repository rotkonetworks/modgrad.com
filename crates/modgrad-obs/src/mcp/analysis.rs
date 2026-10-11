//! Series statistics behind `summarize` and `compare`. All local: the MCP
//! server pulls rows and computes here, whatever the storage backend.

use modgrad_obs_analysis::forecast::{Forecast, Options, forecast};

/// Which way is good for a metric, guessed from its name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dir {
    Lower,
    Higher,
    Unknown,
}

pub fn direction(key: &str) -> Dir {
    let k = key.to_ascii_lowercase();
    let has = |ws: &[&str]| ws.iter().any(|w| k.contains(w));
    if has(&["acc", "reward", "score", "auc", "f1", "precision", "recall", "bleu", "return", "success", "solve", "win"]) {
        Dir::Higher
    } else if has(&["loss", "nll", "err", "perplex", "ppl", "mse", "mae", "rmse", "cost", "regret", "bpb", "bpc"]) {
        Dir::Lower
    } else {
        Dir::Unknown
    }
}

impl Dir {
    pub fn phrase(self) -> &'static str {
        match self {
            Dir::Lower => "lower is better",
            Dir::Higher => "higher is better",
            Dir::Unknown => "direction unknown",
        }
    }
    /// `a` is better than `b`.
    pub fn better(self, a: f64, b: f64) -> bool {
        match self {
            Dir::Higher => a > b,
            _ => a < b,
        }
    }
}

pub fn median(v: &[f64]) -> f64 {
    let mut s: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if s.is_empty() {
        return f64::NAN;
    }
    s.sort_by(|a, b| a.total_cmp(b));
    let n = s.len();
    if n % 2 == 1 { s[n / 2] } else { 0.5 * (s[n / 2 - 1] + s[n / 2]) }
}

pub fn mean_sd(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    if v.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let m = v.iter().sum::<f64>() / n;
    let sd = if v.len() > 1 { (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0)).sqrt() } else { 0.0 };
    (m, sd)
}

/// Median step difference (the logging interval).
pub fn spacing(steps: &[f64]) -> f64 {
    let d: Vec<f64> = steps.windows(2).map(|w| w[1] - w[0]).filter(|d| *d > 0.0).collect();
    median(&d)
}

/// Least-squares trend over the most recent points.
#[derive(Debug, Clone)]
pub struct Trend {
    pub n: usize,
    pub from_step: f64,
    pub to_step: f64,
    /// Change per step.
    pub slope: f64,
    /// Standard error of the slope.
    pub se: f64,
    /// Typical step interval, for "per N steps".
    pub per: f64,
}

impl Trend {
    /// |slope| / se ≥ 2: the recent change is larger than its noise.
    pub fn clear(&self) -> bool {
        self.se.is_finite() && self.slope.abs() > 2.0 * self.se || (self.se == 0.0 && self.slope != 0.0)
    }
}

/// Points in the trend window: the last third, at least 5 (or all when fewer).
pub fn trend_window(n: usize) -> usize {
    n.min((n.div_ceil(3)).max(5))
}

pub fn trend(steps: &[f64], vals: &[f64]) -> Option<Trend> {
    let n = steps.len();
    let k = trend_window(n);
    if k < 3 {
        return None;
    }
    let (x, y) = (&steps[n - k..], &vals[n - k..]);
    let mx = x.iter().sum::<f64>() / k as f64;
    let my = y.iter().sum::<f64>() / k as f64;
    let sxx: f64 = x.iter().map(|a| (a - mx).powi(2)).sum();
    if sxx <= 0.0 {
        return None;
    }
    let slope = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum::<f64>() / sxx;
    let rss: f64 = x.iter().zip(y).map(|(a, b)| (b - my - slope * (a - mx)).powi(2)).sum();
    let se = (rss / (k as f64 - 2.0) / sxx).sqrt();
    let per = spacing(x);
    Some(Trend { n: k, from_step: x[0], to_step: x[k - 1], slope, se, per: if per.is_finite() && per > 0.0 { per } else { 1.0 } })
}

/// Below this many points spikes and plateaus are not judged.
pub const MIN_SHAPE_POINTS: usize = 7;

/// Spikes: robust z-score (0.6745·r/MAD) of residuals from a centred
/// rolling median of width 5; |z| > 3.5. Returns `(index, z)`.
pub fn spikes(vals: &[f64]) -> Vec<(usize, f64)> {
    let n = vals.len();
    if n < MIN_SHAPE_POINTS {
        return Vec::new();
    }
    let r: Vec<f64> = (0..n)
        .map(|i| {
            let lo = i.saturating_sub(2);
            let hi = (i + 3).min(n);
            vals[i] - median(&vals[lo..hi])
        })
        .collect();
    let mr = median(&r);
    let dev: Vec<f64> = r.iter().map(|x| (x - mr).abs()).collect();
    let mut mad = median(&dev);
    if !(mad > 0.0) {
        // more than half the residuals are exactly zero: fall back to the mean deviation
        mad = dev.iter().sum::<f64>() / n as f64 * 0.7979;
    }
    if !(mad > 0.0) {
        return Vec::new();
    }
    r.iter().enumerate().map(|(i, x)| (i, 0.6745 * (x - mr) / mad)).filter(|(_, z)| z.abs() > 3.5).collect()
}

/// Plateau: enough points, no clear recent trend, and the recent window
/// moved less than 10% of the whole curve's range.
pub fn plateau(vals: &[f64], t: &Trend) -> bool {
    if vals.len() < MIN_SHAPE_POINTS || t.clear() {
        return false;
    }
    let (lo, hi) = vals.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &x| (a.min(x), b.max(x)));
    let range = hi - lo;
    range > 0.0 && (t.slope * (t.to_step - t.from_step)).abs() < 0.1 * range
}

/// Forecast out to `horizon_points` logging intervals past the last step.
pub fn forecast_points(steps: &[f64], vals: &[f64], pool: &[(Vec<f64>, Vec<f64>)], horizon_points: usize) -> Option<Forecast> {
    let sp = spacing(steps);
    let last = *steps.last()?;
    if !(sp > 0.0) {
        return None;
    }
    let pool: Vec<(&[f64], &[f64])> = pool.iter().map(|(s, v)| (s.as_slice(), v.as_slice())).collect();
    let mut opt = Options::to_step(last + sp * horizon_points as f64);
    opt.max_points = horizon_points.max(1);
    forecast(steps, vals, &pool, &opt)
}

/// First forecast step at which `curve` reaches `target` moving from
/// `last` towards it.
pub fn first_reach(steps: &[f64], curve: &[f64], last: f64, target: f64) -> Option<f64> {
    let down = target < last;
    steps.iter().zip(curve).find(|(_, v)| if down { **v <= target } else { **v >= target }).map(|(s, _)| *s)
}

/// Seed-group base name: `bkk07/small_s2` → `bkk07/small`. Recognises
/// `_s3`, `-s3`, `_seed3`, `-seed3`, `.s3` suffixes.
pub fn seed_base(name: &str) -> String {
    let b = name.trim_end_matches(|c: char| c.is_ascii_digit());
    if b.len() == name.len() {
        return name.to_string();
    }
    for suf in ["_seed", "-seed", ".seed", "_s", "-s", ".s"] {
        if let Some(base) = b.strip_suffix(suf) {
            if !base.is_empty() && !base.ends_with('/') {
                return base.to_string();
            }
        }
    }
    name.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trend_and_plateau() {
        let st: Vec<f64> = (0..20).map(|i| i as f64 * 10.0).collect();
        let down: Vec<f64> = (0..20).map(|i| 10.0 - i as f64 * 0.3 + if i % 2 == 0 { 0.01 } else { -0.01 }).collect();
        let t = trend(&st, &down).unwrap();
        assert!(t.clear() && t.slope < 0.0);
        assert!(!plateau(&down, &t));
        let flat: Vec<f64> = (0..20).map(|i| if i < 8 { 10.0 - i as f64 } else { 3.0 + if i % 2 == 0 { 0.01 } else { -0.01 } }).collect();
        let t = trend(&st, &flat).unwrap();
        assert!(!t.clear());
        assert!(plateau(&flat, &t));
    }

    #[test]
    fn spike_found() {
        let mut v: Vec<f64> = (0..30).map(|i| 5.0 - 0.05 * i as f64 + 0.01 * ((i * 7) % 3) as f64).collect();
        v[17] += 2.0;
        let s = spikes(&v);
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].0, 17);
        assert!(spikes(&v[..5]).is_empty());
    }

    #[test]
    fn seeds() {
        assert_eq!(seed_base("bkk07/small_s2"), "bkk07/small");
        assert_eq!(seed_base("bkk07/small"), "bkk07/small");
        assert_eq!(seed_base("run-seed12"), "run");
        assert_eq!(seed_base("pilot1"), "pilot1");
        assert_eq!(seed_base("x/s2"), "x/s2");
    }

    #[test]
    fn directions() {
        assert_eq!(direction("brain_heldout_nll"), Dir::Lower);
        assert_eq!(direction("val_acc"), Dir::Higher);
        assert_eq!(direction("lr"), Dir::Unknown);
    }

    #[test]
    fn reach() {
        let s = [1.0, 2.0, 3.0];
        assert_eq!(first_reach(&s, &[5.0, 4.0, 3.0], 6.0, 4.0), Some(2.0));
        assert_eq!(first_reach(&s, &[5.0, 4.5, 4.2], 6.0, 4.0), None);
    }
}
