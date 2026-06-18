//! MILESTONE 1 acceptance gate: the dep-free engine's full 8-region brain
//! (WITH the visual retina) must match the SDK `brain_oracle` bit-exact.
//!
//! Loads `brain_weights.json` (VisualCortex + RegionalWeights minus
//! embeddings) and `brain_reference.json` (per-maze input pixels, the
//! flat observation the SDK fed to `regional_forward`, and per-outer-tick
//! oracle traces). For each maze:
//!   1. re-derive the observation from the dumped pixels via the engine's
//!      own `VisualCortex::spatial_tokens` and assert it matches the
//!      dumped observation (exercises the retina path), then
//!   2. run the engine brain and assert per-tick predictions AND
//!      per-region activations AND global_sync match the oracle within
//!      1e-3 abs.

use modgrad_mini::brain::{regional_forward, RegionalWeights, VisualCortex};
use serde::Deserialize;
use std::fs;

const TOL: f32 = 1e-3;

#[derive(Deserialize)]
struct BrainWeights {
    cortex: VisualCortex,
    regional: RegionalWeights,
}

#[derive(Deserialize)]
struct TickTrace {
    prediction: Vec<f32>,
    region_activations: Vec<Vec<f32>>,
    global_sync: Vec<f32>,
    #[allow(dead_code)]
    exit_lambda: Option<f32>,
}

#[derive(Deserialize)]
struct MazeSample {
    #[allow(dead_code)]
    grid: Vec<u8>,
    pixels: Vec<f32>,
    observation: Vec<f32>,
    n_tokens: usize,
    token_dim: usize,
    ticks_used: usize,
    ticks: Vec<TickTrace>,
}

#[derive(Deserialize)]
struct Reference {
    size: usize,
    samples: Vec<MazeSample>,
}

fn dir() -> String {
    // tests run with CWD = crate root (engine/)
    "../public/models".to_string()
}

#[test]
fn brain_oracle_matches() {
    let d = dir();
    let bw: BrainWeights = serde_json::from_str(
        &fs::read_to_string(format!("{d}/brain_weights.json")).expect("brain_weights.json"),
    )
    .expect("parse brain_weights");
    let r: Reference = serde_json::from_str(
        &fs::read_to_string(format!("{d}/brain_reference.json")).expect("brain_reference.json"),
    )
    .expect("parse brain_reference");

    assert!(!r.samples.is_empty(), "no oracle samples");
    eprintln!("brain: {} mazes, size {}", r.samples.len(), r.size);

    let mut max_err = 0.0f32;

    for (si, s) in r.samples.iter().enumerate() {
        // 1. Retina path: rebuild the observation from the raw pixels and
        //    confirm it matches what the SDK fed to regional_forward.
        let (obs, n_tokens, token_dim) = bw.cortex.spatial_tokens(&s.pixels);
        assert_eq!(n_tokens, s.n_tokens, "sample {si}: n_tokens");
        assert_eq!(token_dim, s.token_dim, "sample {si}: token_dim");
        assert_eq!(obs.len(), s.observation.len(), "sample {si}: obs len");
        for (j, (&a, &b)) in obs.iter().zip(&s.observation).enumerate() {
            let e = (a - b).abs();
            if e > max_err {
                max_err = e;
            }
            assert!(
                e <= TOL,
                "sample {si} retina obs[{j}]: engine {a} vs oracle {b} (|d|={e})"
            );
        }

        // 2. Brain forward on the engine-derived observation.
        let out = regional_forward(&bw.regional, &obs);
        assert_eq!(
            out.ticks_used, s.ticks_used,
            "sample {si}: ticks_used engine {} vs oracle {}",
            out.ticks_used, s.ticks_used
        );
        assert_eq!(out.ticks.len(), s.ticks.len(), "sample {si}: tick count");

        for (t, (et, ot)) in out.ticks.iter().zip(&s.ticks).enumerate() {
            // predictions
            assert_eq!(et.prediction.len(), ot.prediction.len(), "s{si} t{t} pred len");
            for (j, (&a, &b)) in et.prediction.iter().zip(&ot.prediction).enumerate() {
                let e = (a - b).abs();
                if e > max_err {
                    max_err = e;
                }
                assert!(e <= TOL, "s{si} t{t} pred[{j}]: {a} vs {b} (|d|={e})");
            }
            // per-region activations
            assert_eq!(
                et.region_activations.len(),
                ot.region_activations.len(),
                "s{si} t{t} n_regions"
            );
            for (rg, (ea, oa)) in et
                .region_activations
                .iter()
                .zip(&ot.region_activations)
                .enumerate()
            {
                assert_eq!(ea.len(), oa.len(), "s{si} t{t} region {rg} d_model");
                for (j, (&a, &b)) in ea.iter().zip(oa).enumerate() {
                    let e = (a - b).abs();
                    if e > max_err {
                        max_err = e;
                    }
                    assert!(e <= TOL, "s{si} t{t} region {rg} act[{j}]: {a} vs {b} (|d|={e})");
                }
            }
            // global sync
            assert_eq!(et.global_sync.len(), ot.global_sync.len(), "s{si} t{t} gs len");
            for (j, (&a, &b)) in et.global_sync.iter().zip(&ot.global_sync).enumerate() {
                let e = (a - b).abs();
                if e > max_err {
                    max_err = e;
                }
                assert!(e <= TOL, "s{si} t{t} gs[{j}]: {a} vs {b} (|d|={e})");
            }
        }
    }

    eprintln!(
        "brain oracle: {} mazes matched within {TOL} (max abs err {max_err:.3e})",
        r.samples.len()
    );
}
