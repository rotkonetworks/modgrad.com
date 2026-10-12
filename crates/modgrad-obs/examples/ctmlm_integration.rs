//! How `ctmlm_pilot` (crates/modgrad-lm-memory/src/bin/ctmlm_pilot.rs on the
//! `ctm-lm` branch) adopts modgrad-obs behind `--obs <dir>`. This example runs
//! on SYNTHETIC numbers only (no model is loaded); every value it logs is
//! fake and the run is named `demo_synthetic` so it is never mistaken for a
//! measurement. It exists to (1) show the integration and (2) give the
//! dashboard's array views real-shaped data.
//!
//! ```text
//! cargo run -p modgrad-obs --example ctmlm_integration -- --obs obs   # → obs/runs/demo_synthetic_*
//! ```
//!
//! ## The diff for ctmlm_pilot
//!
//! Cargo.toml of modgrad-lm-memory: `modgrad-obs = { path = "../modgrad-obs" }`.
//!
//! ```text
//! // after `let out_dir = …` in main():
//! let obs_dir = args.iter().position(|a| a == "--obs").map(|i| std::path::PathBuf::from(&args[i + 1]));
//!
//! // after the CTM-LM PILOT header println (config is the same fields):
//! let mut obs = obs_dir.as_ref().map(|d| modgrad_obs::Run::init(d, "ctmlm",
//!     &out_dir.clone().unwrap_or_else(|| "pilot".into()).rsplit('/').next().unwrap().to_string(),
//!     &serde_json::json!({ "c": spec.c, "n": spec.n, "facts": spec.facts, "gap_lo": spec.gap.0, "gap_hi": spec.gap.1,
//!         "window": window, "ticks_lo": tc.ticks.0, "ticks_hi": tc.ticks.1, "batch": BATCH,
//!         "natural_replay_pct": natural_frac * 100.0, "persistent": cfg_persist,
//!         "brain_params": store.len(), "read_params": read.wq.len() * 4 + read.b.len() })).expect("modgrad init"));
//!
//! // next to the NONE (fixed) println:
//! if let Some(o) = obs.as_mut() { o.log_scalars(0, &[("none_indist_acc_pct", none_i.pct()), ("none_indist_nll", none_i.mean_nll()), …]).ok(); }
//!
//! // next to the per-EVAL `step …` println:
//! if let Some(o) = obs.as_mut() { o.log_scalars(step as u64, &[("train_loss_ema", ema as f64),
//!     ("brain_indist_acc_pct", ei.pct()), ("brain_indist_nll", ei.mean_nll()),
//!     ("brain_heldout_acc_pct", eh.pct()), ("brain_heldout_nll", eh.mean_nll()),
//!     ("s_per_step", step_time / step as f64)]).ok(); }
//!
//! // in report_natural (pass `modgrad` and `step` through): nat_brain, nat_none,
//! // nat_full, nat_brain_minus_none (+ _std, _sem), and per chunk:
//! o.log_array(step, "nat_chunk_brain_minus_none", &[chunks], &per_chunk_delta).ok();
//!
//! // FINAL block: per-gap recall is an array, one per arm:
//! o.log_array(step, "final_heldout_gap_acc_brain", &[gaps], &acc_by_gap).ok();
//! o.summary("final_heldout_brain_acc_pct", e.brain[0].pct()).ok();
//! // and at the very end:
//! if let Some(o) = obs { o.finish().ok(); }
//! ```
//!
//! ## Thinking trajectories, point clouds, volumes (3-D views)
//!
//! Log raw state; the dashboard runs PCA in the browser (wasm). Keep it
//! small: a few samples, and cut neurons to <= 256 by variance.
//!
//! ```text
//! // ctmlm_pilot, at each EVAL step, for the first 4 eval documents' last chunk:
//! //   states: Vec<f32> of [samples=4][ticks=K][d=256] from the CTM tick loop
//! let (s, k, d) = (4, ticks, 256);
//! let cols = modgrad_obs::top_variance_columns(&states, s * k, d, 128);
//! let small = modgrad_obs::select_columns(&states, s * k, d, &cols);
//! o.log_trajectory(step, "state_traj", &[s, k, cols.len()], &small).ok();
//!
//! // episode keys (stored vs lure), [n][d] with one label per row:
//! o.log_points(step, "episode_keys", &[n, d], &keys, Some(&is_lure)).ok();
//!
//! // maze_spatial (examples/mazes, the spatial brain), per eval maze:
//! //   path probability per tick: [ticks][h][w]
//! o.log_volume(step, "path_prob", &[ticks, h, w], &prob).ok();
//! //   and the brain state per tick as a trajectory, one sample per maze:
//! o.log_trajectory(step, "maze_state_traj", &[mazes, ticks, cols.len()], &small).ok();
//! ```
//!
//! Key names match the legacy importer (`modgrad_obs::legacy`), so imported
//! old logs and new native runs plot on the same axes. `Acc::pct()` /
//! `Acc::mean_nll()` are two-line helpers next to `Acc::show()`.

use modgrad_obs::{Histogram, Run, Tensor, downsample, select_columns, top_variance_columns};
use serde_json::json;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let obs = args
        .iter()
        .position(|a| a == "--obs")
        .map(|i| std::path::PathBuf::from(&args[i + 1]))
        .unwrap_or_else(|| "obs".into());
    let steps: u64 = args.iter().position(|a| a == "--steps").map(|i| args[i + 1].parse().unwrap()).unwrap_or(3000);

    // Deterministic pseudo-random numbers (xorshift) — no rand dependency.
    let mut s = 0x9E3779B97F4A7C15u64;
    let mut rnd = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };

    for (name, brain_params, rate) in [("demo_synthetic_small", 494_325.0, 1.0), ("demo_synthetic_mid", 10_106_037.0, 1.4), ("demo_synthetic_large", 58_911_157.0, 1.8)] {
        let _ = std::fs::remove_dir_all(obs.join("runs").join(name));
        let cfg = json!({ "synthetic": true, "c": 32, "n": 6, "facts": 2, "gap_lo": 2, "gap_hi": 6, "window": 5,
            "ticks_lo": 8, "ticks_hi": 16, "batch": 2, "natural_replay_pct": 25, "brain_params": brain_params, "read_params": 983_044 });
        let mut run = Run::init(&obs, "ctmlm-demo", name, &cfg).expect("init");
        run.event(0, "SYNTHETIC demo data — not a measurement").unwrap();
        run.log_scalars(0, &[("none_indist_acc_pct", 5.0), ("none_indist_nll", 7.49), ("none_heldout_acc_pct", 7.5), ("none_heldout_nll", 8.99)]).unwrap();
        let mut ema = 3.9;
        let (gaps, ticks, neurons) = (5usize, 16usize, 64usize);
        for step in 1..=steps {
            let p = step as f64 / steps as f64;
            let loss = 3.9 - 0.6 * rate * (1.0 - (-4.0 * p).exp()) + 0.15 * (rnd() - 0.5);
            ema = if step == 1 { loss } else { 0.98 * ema + 0.02 * loss };
            if step % 10 == 0 || step == 1 {
                run.log_scalars(step, &[("train_loss_ema", ema), ("train_loss", loss)]).unwrap();
            }
            if step % 250 == 0 || step == 1 {
                let learn = rate * (1.0 - (-3.0 * p).exp());
                let nll_h = 9.0 - 4.0 * learn / 1.8 + 0.2 * rnd();
                let nll_i = 7.5 - 4.5 * learn / 1.8 + 0.2 * rnd();
                run.log(step, [
                    ("brain_indist_acc_pct", (5.0 + 20.0 * learn + 5.0 * rnd()).into()),
                    ("brain_indist_nll", nll_i.into()),
                    ("brain_heldout_acc_pct", (5.0 + 12.0 * learn + 5.0 * rnd()).into()),
                    ("brain_heldout_nll", nll_h.into()),
                    ("s_per_step", (1.5 * rate).into()),
                    ("nat_brain_minus_none", (-0.01 * learn + 0.002 * (rnd() - 0.5)).into()),
                ]).unwrap();
                // per-gap recall (%): longer gaps recalled worse, improving with training
                let recall: Vec<f32> = (0..gaps).map(|g| (10.0 + 60.0 * learn / 1.8 * (-(g as f64) / 3.0).exp() + 3.0 * rnd()) as f32).collect();
                run.log(step, [("recall_by_gap", Tensor::new(&[gaps], recall).into())]).unwrap();
                // per-tick certainty: rises over ticks, faster later in training
                let cert: Vec<f32> = (0..ticks).map(|t| (1.0 - (-(t as f64 + 1.0) * 0.1 * (1.0 + 2.0 * learn)).exp()) as f32).collect();
                run.log(step, [("certainty_by_tick", Tensor::new(&[ticks], cert).into())]).unwrap();
                // neuron × tick activations (logged as tick × neuron), travelling waves
                let act: Vec<f32> = (0..ticks)
                    .flat_map(|t| (0..neurons).map(move |n| (t, n)))
                    .map(|(t, n)| ((n as f64 * 0.3 + t as f64 * (0.4 + learn)).sin() * (-(t as f64) * 0.05).exp()) as f32)
                    .collect();
                let (sh, act) = downsample(&[ticks, neurons], &act, 4096);
                run.log_array(step, "activation_tick_neuron", &sh, &act).unwrap();
                // a weight histogram widening over training
                let w: Vec<f32> = (0..2000).map(|_| ((rnd() + rnd() + rnd() - 1.5) * (0.2 + learn)) as f32).collect();
                run.log(step, [("weights_hist", Histogram::from_data(&w, 48).into())]).unwrap();
            }
            if step % 1000 == 0 || step == 1 {
                let learn = rate * (1.0 - (-3.0 * p).exp());
                // thinking trajectories: 4 samples × 16 ticks × 512 neurons, state
                // spirals into a sample-specific attractor (faster when trained)
                let (ns, nt, nd) = (4usize, 16usize, 512usize);
                let mut st = Vec::with_capacity(ns * nt * nd);
                for smp in 0..ns {
                    for t in 0..nt {
                        let decay = (-(t as f64) * 0.15 * (1.0 + learn)).exp();
                        let ang = t as f64 * 0.6;
                        for n in 0..nd {
                            let basis = |k: usize| ((n * (k + 3) + smp * 17) as f64 * 0.013).sin();
                            let v = (smp as f64 - 1.5) * basis(0) + decay * (ang.cos() * basis(1) + ang.sin() * basis(2)) + 0.05 * (rnd() - 0.5);
                            st.push(v as f32);
                        }
                    }
                }
                let cols = top_variance_columns(&st, ns * nt, nd, 256);
                let small = select_columns(&st, ns * nt, nd, &cols);
                run.log_trajectory(step, "state_traj", &[ns, nt, cols.len()], &small).unwrap();
                // point cloud: 120 stored keys + 60 lures in 64-d; lures drift off
                let (np, dd) = (180usize, 64usize);
                let mut pts = Vec::with_capacity(np * dd);
                let mut lab = Vec::with_capacity(np);
                for i in 0..np {
                    let lure = i >= 120;
                    lab.push(if lure { 1.0 } else { 0.0 });
                    let c = (i % 3) as f64;
                    for j in 0..dd {
                        let centre = ((j as f64 + 1.0) * (c + 1.0) * 0.7).sin() * (1.0 + learn);
                        let off = if lure { 0.8 * ((j * 5) as f64).cos() } else { 0.0 };
                        pts.push((centre + off + 0.6 * (rnd() - 0.5)) as f32);
                    }
                }
                run.log_points(step, "episode_keys", &[np, dd], &pts, Some(&lab)).unwrap();
                // volume: path probability over a 15×15 maze spreading per tick
                let (vt, h, w) = (12usize, 15usize, 15usize);
                let vol: Vec<f32> = (0..vt)
                    .flat_map(|t| (0..h * w).map(move |c| (t, c / w, c % w)))
                    .map(|(t, y, x)| {
                        let front = (t as f64 + 1.0) * 1.3 * (0.5 + learn);
                        let dist = (x + y) as f64;
                        (1.0 / (1.0 + (dist - front).exp())) as f32 * if (x * 7 + y * 3) % 5 == 0 { 0.2 } else { 1.0 }
                    })
                    .collect();
                run.log_volume(step, "path_prob", &[vt, h, w], &vol).unwrap();
            }
        }
        run.summary("note", "synthetic demo").unwrap();
        run.finish().unwrap();
        println!("wrote {}", obs.join("runs").join(name).display());
    }
}
