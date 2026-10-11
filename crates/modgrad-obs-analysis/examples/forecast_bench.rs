//! Time `forecast::forecast` on the dashboard-shaped cases of
//! tests/common/forecast_cases.rs.
//!
//!     cargo run --release -p modgrad-obs-analysis --example forecast_bench
//!     cargo run --release -p modgrad-obs-analysis --example forecast_bench -- --golden
//!
//! `--golden` prints the outputs in the form tests/forecast_golden.rs pins.

#[path = "../tests/common/forecast_cases.rs"]
mod forecast_cases;

use std::time::Instant;

fn main() {
    let golden = std::env::args().any(|a| a == "--golden");
    for c in forecast_cases::cases() {
        if golden {
            let f = forecast_cases::run(&c).expect(c.name);
            let eta = modgrad_obs_analysis::curve::eta(&f.steps, &f.mean, &f.lo, &f.hi, c.target);
            let cfg = f.config;
            println!("// {}", c.name);
            println!(
                "G {{ name: {:?}, skill: {}, lookback: {}, scaler: {:?}, pooled: {}, position: {}, lambda: {:e}, noise: {:e}, n_configs: {}, mae_val: {:e}, mae_test: {:e}, mae_test_naive: {:e}, len: {}, eta: [{:e}, {:e}, {:e}],",
                c.name, f.skill, cfg.lookback, cfg.scaler.name(), cfg.pooled, cfg.position, cfg.lambda, cfg.noise, f.n_configs, f.mae_val, f.mae_test, f.mae_test_naive, f.mean.len(), eta[0], eta[1], eta[2]
            );
            let idx = sample_idx(f.mean.len());
            let pick = |v: &[f64]| idx.iter().map(|&i| format!("({i}, {:e})", v[i])).collect::<Vec<_>>().join(", ");
            let fam = |b: Option<modgrad_obs_analysis::forecast::FamilyBest>| match b {
                Some(b) => format!("Some(({}, {:?}, {}, {:e}, {:e}))", b.config.lookback, b.config.scaler.name(), b.config.position, b.mae_val, b.mae_test),
                None => "None".into(),
            };
            println!("    single: {}, pooled_best: {}, max_backtest_h: {}, n_pool: {},", fam(f.single), fam(f.pooled), f.max_backtest_h, f.n_pool);
            println!("    mean: &[{}],", pick(&f.mean));
            println!("    lo: &[{}],", pick(&f.lo));
            println!("    hi: &[{}] }},", pick(&f.hi));
            continue;
        }
        let reps = if c.steps.len() > 100 { 5 } else { 50 };
        let _ = forecast_cases::run(&c); // warm
        let t = Instant::now();
        for _ in 0..reps {
            std::hint::black_box(forecast_cases::run(&c));
        }
        let ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
        let f = forecast_cases::run(&c).unwrap();
        println!("{:<16} n={:<5} pool={} h={:<4} {:>9.2} ms  skill={} L={} {} pooled={} configs={}", c.name, c.steps.len(), c.pool.len(), f.mean.len(), ms, f.skill, f.config.lookback, f.config.scaler.name(), f.config.pooled, f.n_configs);
    }
}

/// Indices pinned by the golden test: first, last and ~10 between.
pub fn sample_idx(n: usize) -> Vec<usize> {
    let mut v: Vec<usize> = (0..10).map(|k| k * (n - 1) / 9).collect();
    v.dedup();
    v
}
