//! Validates the engine against the MAZE oracle (different config than parity:
//! out_dims=4, d_model=128, raw_dim=9, 16 ticks) AND, unlike the parity test,
//! checks the per-tick ATTENTION over cells. Reconstructs each oracle sample's
//! observation via the engine's own `encode_maze`, so this also exercises the
//! browser's encoding path.

use modgrad_mini::{encode_maze, forward, CtmWeights};
use serde::Deserialize;
use std::fs;

#[derive(Deserialize)]
struct TickTrace {
    prediction: Vec<f32>,
    certainty: [f32; 2],
    activations: Vec<f32>,
    attention: Vec<Vec<f32>>, // [heads][n_tokens]
}
#[derive(Deserialize)]
struct OracleSample {
    grid: Vec<u8>,
    end: [usize; 2],
    agent: [usize; 2],
    predicted: usize,
    commit_tick: usize,
    ticks: Vec<TickTrace>,
}
#[derive(Deserialize)]
struct Reference {
    size: usize,
    oracle: Vec<OracleSample>,
}

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1e-3
}

#[test]
fn maze_oracle_matches() {
    let dir = "../public/models";
    let w: CtmWeights = serde_json::from_str(
        &fs::read_to_string(format!("{dir}/maze_weights.json")).expect("maze_weights.json"),
    )
    .expect("parse weights");
    let r: Reference = serde_json::from_str(
        &fs::read_to_string(format!("{dir}/maze_reference.json")).expect("maze_reference.json"),
    )
    .expect("parse reference");

    let size = r.size;
    let n = size * size;
    assert!(!r.oracle.is_empty(), "no oracle samples");

    for (si, s) in r.oracle.iter().enumerate() {
        let obs = encode_maze(&s.grid, size, (s.agent[0], s.agent[1]), (s.end[0], s.end[1]));
        let out = forward(&w, &obs, n, 9);
        assert_eq!(out.predictions.len(), s.ticks.len(), "sample {si}: tick count");

        for (t, tk) in s.ticks.iter().enumerate() {
            for (j, &p) in tk.prediction.iter().enumerate() {
                assert!(close(out.predictions[t][j], p),
                    "s{si} t{t} pred[{j}]: {} vs {p}", out.predictions[t][j]);
            }
            for j in 0..2 {
                assert!(close(out.certainties[t][j], tk.certainty[j]),
                    "s{si} t{t} cert[{j}]: {} vs {}", out.certainties[t][j], tk.certainty[j]);
            }
            for (j, &a) in tk.activations.iter().enumerate() {
                assert!(close(out.activations[t][j], a),
                    "s{si} t{t} act[{j}]: {} vs {a}", out.activations[t][j]);
            }
            for (h, head) in tk.attention.iter().enumerate() {
                for (j, &av) in head.iter().enumerate() {
                    assert!(close(out.attention[t][h][j], av),
                        "s{si} t{t} head{h} attn[{j}]: {} vs {av}", out.attention[t][h][j]);
                }
            }
        }

        let ct = (0..out.certainties.len())
            .max_by(|&a, &b| out.certainties[a][1].partial_cmp(&out.certainties[b][1]).unwrap())
            .unwrap();
        let pred = (0..out.predictions[ct].len())
            .max_by(|&a, &b| out.predictions[ct][a].partial_cmp(&out.predictions[ct][b]).unwrap())
            .unwrap();
        assert_eq!(ct, s.commit_tick, "sample {si}: commit tick");
        assert_eq!(pred, s.predicted, "sample {si}: predicted move");
    }
    eprintln!("maze oracle: {} samples matched within 1e-3", r.oracle.len());
}
