//! Acceptance gate: match the SDK oracle exactly.
//!
//! Loads real trained weights (`parity_weights.json`) and a reference of real
//! per-tick outputs (`parity_reference.json`), runs `modgrad_mini::forward` on
//! each of the 24 samples, and asserts per-tick prediction/certainty parity
//! within 1e-3, plus argmax-at-max-certainty == oracle `predicted`.

use modgrad_mini::{forward, CtmWeights};
use serde::Deserialize;

const WEIGHTS_PATH: &str = "/steam/rotko/modgrad.com/public/models/parity_weights.json";
const REF_PATH: &str = "/steam/rotko/modgrad.com/public/models/parity_reference.json";
const TOL: f32 = 1e-3;

#[derive(Debug, Deserialize)]
struct Reference {
    samples: Vec<Sample>,
}

#[derive(Debug, Deserialize)]
struct Sample {
    bits: Vec<f32>,
    #[allow(dead_code)]
    target: i64,
    predicted: i64,
    #[allow(dead_code)]
    commit_tick: usize,
    ticks: Vec<Tick>,
}

#[derive(Debug, Deserialize)]
struct Tick {
    prediction: Vec<f32>,
    certainty: Vec<f32>,
    #[serde(default)]
    activations: Vec<f32>,
}

fn load_weights() -> CtmWeights {
    let s = std::fs::read_to_string(WEIGHTS_PATH).expect("read parity_weights.json");
    serde_json::from_str(&s).expect("deserialize CtmWeights")
}

fn load_reference() -> Reference {
    let s = std::fs::read_to_string(REF_PATH).expect("read parity_reference.json");
    serde_json::from_str(&s).expect("deserialize Reference")
}

#[test]
fn weights_deserialize_and_reserialize() {
    // Schema sanity: round-trip through our structs and back to JSON.
    let w = load_weights();
    let json = serde_json::to_string(&serde_json::json!({
        "config_d_model": w.config.d_model,
        "iterations": w.config.iterations,
        "synapse_widths": w.synapse.widths,
        "n_sync_out_left": w.sync_out_left.len(),
    }))
    .unwrap();
    assert!(json.contains("d_model") || json.contains("config_d_model"));
    assert_eq!(w.config.out_dims, 2);
    assert_eq!(w.kv_proj.in_dim, 1);
}

#[test]
fn parity_all_samples() {
    let w = load_weights();
    let reference = load_reference();
    assert_eq!(reference.samples.len(), 24, "expected 24 samples");

    let mut failures: Vec<String> = Vec::new();

    for (si, sample) in reference.samples.iter().enumerate() {
        let n_tokens = sample.bits.len(); // 8
        let raw_dim = 1usize;
        let out = forward(&w, &sample.bits, n_tokens, raw_dim);

        assert_eq!(
            out.predictions.len(),
            sample.ticks.len(),
            "sample {si}: tick count mismatch ({} vs {})",
            out.predictions.len(),
            sample.ticks.len()
        );

        // Per-tick prediction + certainty parity.
        for (ti, tick) in sample.ticks.iter().enumerate() {
            for c in 0..tick.prediction.len() {
                let got = out.predictions[ti][c];
                let exp = tick.prediction[c];
                if (got - exp).abs() > TOL {
                    failures.push(format!(
                        "sample {si} tick {ti} pred[{c}]: got {got} exp {exp} (|d|={})",
                        (got - exp).abs()
                    ));
                }
            }
            for c in 0..tick.certainty.len() {
                let got = out.certainties[ti][c];
                let exp = tick.certainty[c];
                if (got - exp).abs() > TOL {
                    failures.push(format!(
                        "sample {si} tick {ti} cert[{c}]: got {got} exp {exp} (|d|={})",
                        (got - exp).abs()
                    ));
                }
            }
            // Activations parity (debug aid — also asserted within TOL).
            if !tick.activations.is_empty() {
                for n in 0..tick.activations.len() {
                    let got = out.activations[ti][n];
                    let exp = tick.activations[n];
                    if (got - exp).abs() > TOL {
                        failures.push(format!(
                            "sample {si} tick {ti} act[{n}]: got {got} exp {exp} (|d|={})",
                            (got - exp).abs()
                        ));
                    }
                }
            }
        }

        // Argmax of the prediction at the max-certainty tick == oracle predicted.
        let max_cert_tick = (0..out.certainties.len())
            .max_by(|&a, &b| {
                out.certainties[a][1]
                    .partial_cmp(&out.certainties[b][1])
                    .unwrap()
            })
            .unwrap();
        let pred_vec = &out.predictions[max_cert_tick];
        let argmax = (0..pred_vec.len())
            .max_by(|&a, &b| pred_vec[a].partial_cmp(&pred_vec[b]).unwrap())
            .unwrap() as i64;
        if argmax != sample.predicted {
            failures.push(format!(
                "sample {si}: argmax@maxcert(tick {max_cert_tick}) = {argmax}, oracle predicted = {}",
                sample.predicted
            ));
        }
    }

    if !failures.is_empty() {
        // Report only the first handful so the failure stays readable.
        let shown: Vec<_> = failures.iter().take(20).cloned().collect();
        panic!(
            "{} parity failures (showing first {}):\n{}",
            failures.len(),
            shown.len(),
            shown.join("\n")
        );
    }
}
