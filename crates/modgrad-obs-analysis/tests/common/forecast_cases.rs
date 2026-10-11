// Deterministic forecast inputs shared by the golden test
// (tests/forecast_golden.rs) and the bench (examples/forecast_bench.rs).
// Shapes follow what the dashboard sends: a query trace downsampled to at
// most 1500 points, a +50% horizon with up to 1024 rolled-out points, and
// optionally other runs of the same metric as the pool.

#![allow(dead_code)]

pub struct Case {
    pub name: &'static str,
    pub steps: Vec<f64>,
    pub vals: Vec<f64>,
    pub pool: Vec<(Vec<f64>, Vec<f64>)>,
    pub horizon: f64,
    pub max_points: usize,
    /// ETA target for the golden readout.
    pub target: f64,
}

struct Rng(u64);
impl Rng {
    fn unif(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        // sum of 4 uniforms, unit variance
        (self.unif() + self.unif() + self.unif() + self.unif() - 2.0) * 3f64.sqrt()
    }
}

/// A pre-training loss curve: power law to a floor, multiplicative noise.
fn loss_curve(seed: u64, n: usize, every: f64, a: f64, b: f64, c: f64, noise: f64) -> (Vec<f64>, Vec<f64>) {
    let mut r = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let steps: Vec<f64> = (0..n).map(|i| (i + 1) as f64 * every).collect();
    let vals = steps.iter().map(|&s| (a + b * (s / every).powf(-c)) * (1.0 + noise * r.gauss())).collect();
    (steps, vals)
}

fn case(name: &'static str, n: usize, every: f64, pool_n: usize, seed: u64, target: f64) -> Case {
    let (steps, vals) = loss_curve(seed, n, every, 2.1, 6.0, 0.45, 0.01);
    let pool = (0..pool_n).map(|k| loss_curve(seed + 100 + k as u64, n + 200 * k, every, 2.0 + 0.1 * k as f64, 5.5 + 0.5 * k as f64, 0.4 + 0.05 * k as f64, 0.01)).collect();
    let last = steps[n - 1];
    Case { name, steps, vals, pool, horizon: last * 1.5, max_points: 1024, target }
}

pub fn cases() -> Vec<Case> {
    vec![
        case("single-1500", 1500, 10.0, 0, 1, 2.3),
        case("pooled-1500", 1500, 10.0, 3, 2, 2.3),
        case("single-300", 300, 50.0, 0, 3, 2.5),
        case("pooled-eval-9", 9, 500.0, 3, 4, 3.9),
        case("single-eval-12", 12, 500.0, 0, 5, 3.7),
    ]
}

pub fn run(c: &Case) -> Option<modgrad_obs_analysis::forecast::Forecast> {
    let pool: Vec<(&[f64], &[f64])> = c.pool.iter().map(|(s, v)| (s.as_slice(), v.as_slice())).collect();
    let mut opt = modgrad_obs_analysis::forecast::Options::to_step(c.horizon);
    opt.allow_pool = !pool.is_empty();
    opt.max_points = c.max_points;
    modgrad_obs_analysis::forecast::forecast(&c.steps, &c.vals, &pool, &opt)
}
