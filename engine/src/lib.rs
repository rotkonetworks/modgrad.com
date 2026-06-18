//! `modgrad-mini` — a standalone, faithful scalar-f32 reimplementation of the
//! modgrad CTM (Continuous Thought Machine) forward pass for in-browser demos.
//!
//! This crate has ZERO dependency on any `modgrad-*` crate. It mirrors the
//! relevant weight structs (so the SDK's exported `parity_weights.json`
//! deserializes cleanly) and replicates the scalar forward path
//! operation-for-operation from `crates/modgrad-ctm/src/forward.rs`,
//! `crates/modgrad-compute/src/neuron.rs` and
//! `crates/modgrad-ctm/src/synapse.rs`.
//!
//! Faithfulness is proven by `tests/parity.rs`, which matches a real oracle
//! (`parity_reference.json`) of per-tick predictions/certainties/activations.
//!
//! Only `ExitStrategy::None` (fixed ticks, no early exit, no exit_gate) is
//! exercised. `state.episodic` is always `None`.

use serde::Deserialize;

pub mod brain;

// ═══════════════════════════════════════════════════════════════════════
//  Weight structs — mirror modgrad with EXACT field names/order.
// ═══════════════════════════════════════════════════════════════════════

/// Mirrors `modgrad_ctm::config::ExitStrategy`.
///
/// Serde uses the default externally-tagged representation. `parity_weights`
/// stores `"exit_strategy": "None"` (a unit variant => bare string).
#[derive(Debug, Clone, Deserialize)]
pub enum ExitStrategy {
    None,
    Certainty { threshold: f32 },
    AdaptiveGate { beta: f32, threshold: f32 },
}

impl Default for ExitStrategy {
    fn default() -> Self {
        ExitStrategy::None
    }
}

/// Mirrors `modgrad_ctm::config::CtmConfig`.
#[derive(Debug, Clone, Deserialize)]
pub struct CtmConfig {
    pub iterations: usize,
    pub d_model: usize,
    pub d_input: usize,
    pub heads: usize,
    pub n_synch_out: usize,
    pub n_synch_action: usize,
    pub synapse_depth: usize,
    pub memory_length: usize,
    pub deep_nlms: bool,
    pub memory_hidden_dims: usize,
    pub out_dims: usize,
    pub n_random_pairing_self: usize,
    pub min_width: usize,
    #[serde(default)]
    pub exit_strategy: ExitStrategy,
    #[serde(default)]
    pub collect_trajectories: bool,
}

/// Mirrors `modgrad_compute::neuron::Linear`.
/// `weight` is `[out_dim × in_dim]` row-major; `y[i] = bias[i] + dot(row_i, x)`.
#[derive(Debug, Clone, Deserialize)]
pub struct Linear {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub in_dim: usize,
    pub out_dim: usize,
}

impl Linear {
    /// Faithful to `Linear::forward` (CPU matvec): row-major `[out_dim × in_dim]`.
    pub(crate) fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut y = vec![0.0f32; self.out_dim];
        for i in 0..self.out_dim {
            let row = &self.weight[i * self.in_dim..(i + 1) * self.in_dim];
            let mut acc = self.bias[i];
            for j in 0..self.in_dim {
                acc += row[j] * x[j];
            }
            y[i] = acc;
        }
        y
    }
}

/// Mirrors `modgrad_compute::neuron::SuperLinear` (per-neuron parallel MLP).
/// `weights` is `[n_neurons × out_per × in_per]`; `biases` is `[n_neurons × out_per]`.
#[derive(Debug, Clone, Deserialize)]
pub struct SuperLinear {
    pub weights: Vec<f32>,
    pub biases: Vec<f32>,
    pub n_neurons: usize,
    pub in_per: usize,
    pub out_per: usize,
}

impl SuperLinear {
    /// Faithful to `SuperLinear::forward_cpu`: per-neuron, sequential.
    pub(crate) fn forward(&self, trace: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.n_neurons * self.out_per];
        for n in 0..self.n_neurons {
            let t = &trace[n * self.in_per..(n + 1) * self.in_per];
            let w_base = n * self.out_per * self.in_per;
            let o_base = n * self.out_per;
            for o in 0..self.out_per {
                let w = &self.weights[w_base + o * self.in_per..w_base + (o + 1) * self.in_per];
                let mut acc = self.biases[o_base + o];
                for k in 0..self.in_per {
                    acc += w[k] * t[k];
                }
                out[o_base + o] = acc;
            }
        }
        out
    }
}

/// Mirrors `modgrad_ctm::synapse::SynapseBlock`: Linear → LayerNorm(affine) → SiLU.
#[derive(Debug, Clone, Deserialize)]
pub struct SynapseBlock {
    pub linear: Linear,
    pub ln_gamma: Vec<f32>,
    pub ln_beta: Vec<f32>,
}

impl SynapseBlock {
    /// Faithful to `SynapseBlock::forward`: `linear.forward` then `ln_silu_fwd`
    /// (= LayerNorm(affine, eps 1e-5) followed by SiLU).
    pub(crate) fn forward(&self, x: &[f32]) -> Vec<f32> {
        let y = self.linear.forward(x);
        let mut out = vec![0.0f32; y.len()];
        ln_silu_fwd(&y, &self.ln_gamma, &self.ln_beta, &mut out);
        out
    }
}

/// Mirrors `modgrad_ctm::synapse::SynapseUNet`.
#[derive(Debug, Clone, Deserialize)]
pub struct SynapseUNet {
    pub widths: Vec<usize>,
    pub first_projection: SynapseBlock,
    pub down_blocks: Vec<SynapseBlock>,
    pub up_blocks: Vec<SynapseBlock>,
    pub skip_ln_gamma: Vec<Vec<f32>>,
    pub skip_ln_beta: Vec<Vec<f32>>,
}

impl SynapseUNet {
    /// Faithful to `SynapseUNet::forward`.
    pub(crate) fn forward(&self, x: &[f32]) -> Vec<f32> {
        let n_blocks = self.down_blocks.len();

        // Initial projection
        let out_first = self.first_projection.forward(x);

        // Down path — store each level for skip connections
        let mut outs_down: Vec<Vec<f32>> = Vec::with_capacity(n_blocks + 1);
        outs_down.push(out_first);
        for i in 0..n_blocks {
            let next = self.down_blocks[i].forward(outs_down.last().unwrap());
            outs_down.push(next);
        }

        // Up path — start from bottleneck, apply in reverse order
        let mut current = outs_down[n_blocks].clone();
        for i in 0..n_blocks {
            let up_idx = n_blocks - 1 - i;
            let mut up_out = self.up_blocks[up_idx].forward(&current);
            let skip = &outs_down[up_idx];
            for j in 0..up_out.len() {
                up_out[j] += skip[j];
            }
            affine_ln(&mut up_out, &self.skip_ln_gamma[up_idx], &self.skip_ln_beta[up_idx]);
            current = up_out;
        }

        current
    }
}

/// Mirrors `modgrad_ctm::weights::CtmWeights`.
#[derive(Debug, Clone, Deserialize)]
pub struct CtmWeights {
    pub config: CtmConfig,
    pub synapse: SynapseUNet,
    pub nlm_stage1: SuperLinear,
    pub nlm_stage2: Option<SuperLinear>,
    pub start_activated: Vec<f32>,
    pub start_trace: Vec<f32>,
    pub kv_proj: Linear,
    pub kv_ln_gamma: Vec<f32>,
    pub kv_ln_beta: Vec<f32>,
    pub q_proj: Linear,
    pub mha_in_proj: Linear,
    pub mha_out_proj: Linear,
    pub sync_out_left: Vec<usize>,
    pub sync_out_right: Vec<usize>,
    pub sync_action_left: Vec<usize>,
    pub sync_action_right: Vec<usize>,
    pub decay_params_out: Vec<f32>,
    pub decay_params_action: Vec<f32>,
    pub output_proj: Linear,
    #[serde(default)]
    pub exit_gate: Option<Linear>,
}

// ═══════════════════════════════════════════════════════════════════════
//  Scalar primitives — mirror the modgrad CPU backend exactly.
// ═══════════════════════════════════════════════════════════════════════

/// Mirrors `cpu::sigmoid` (with the same clamp at ±12).
#[inline]
pub(crate) fn sigmoid(x: f32) -> f32 {
    if x > 12.0 {
        1.0
    } else if x < -12.0 {
        0.0
    } else {
        1.0 / (1.0 + (-x).exp())
    }
}

/// Mirrors `cpu::ln_silu_fwd`: LayerNorm(affine, eps 1e-5) then SiLU.
pub(crate) fn ln_silu_fwd(x: &[f32], gamma: &[f32], beta: &[f32], out: &mut [f32]) {
    let n = x.len();
    let nf = n as f32;
    let mean: f32 = x.iter().sum::<f32>() / nf;
    let var: f32 = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / nf;
    let rstd = 1.0 / (var + 1e-5).sqrt();
    for c in 0..n {
        out[c] = (x[c] - mean) * rstd * gamma[c] + beta[c];
    }
    for v in out.iter_mut() {
        let s = sigmoid(*v);
        *v *= s;
    }
}

/// Mirrors `forward::affine_ln` (the synapse skip-conn LN and kv LN).
pub(crate) fn affine_ln(x: &mut [f32], gamma: &[f32], beta: &[f32]) {
    let n = x.len();
    if n == 0 {
        return;
    }
    let nf = n as f32;
    let mean: f32 = x.iter().sum::<f32>() / nf;
    let var: f32 = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / nf;
    let inv_std = 1.0 / (var + 1e-5).sqrt();
    for i in 0..n {
        x[i] = gamma[i] * (x[i] - mean) * inv_std + beta[i];
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Sync (random-pairing) — mirror forward::sync_*.
// ═══════════════════════════════════════════════════════════════════════

pub(crate) fn sync_init(
    activated: &[f32],
    left: &[usize],
    right: &[usize],
    alpha: &mut Vec<f32>,
    beta: &mut Vec<f32>,
) {
    let n = left.len();
    alpha.clear();
    alpha.reserve(n);
    beta.clear();
    beta.resize(n, 1.0);
    for i in 0..n {
        alpha.push(activated[left[i]] * activated[right[i]]);
    }
}

pub(crate) fn sync_update(
    activated: &[f32],
    left: &[usize],
    right: &[usize],
    alpha: &mut [f32],
    beta: &mut [f32],
    r: &[f32],
) -> Vec<f32> {
    let n = left.len();
    for i in 0..n {
        let pw = activated[left[i]] * activated[right[i]];
        alpha[i] = r[i] * alpha[i] + pw;
        beta[i] = r[i] * beta[i] + 1.0;
    }
    sync_read(alpha, beta)
}

pub(crate) fn sync_read(alpha: &[f32], beta: &[f32]) -> Vec<f32> {
    alpha
        .iter()
        .zip(beta)
        .map(|(&a, &b)| a / b.sqrt().max(1e-8))
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════
//  NLM (trace processor) — mirror forward::nlm_forward + per_neuron_glu.
// ═══════════════════════════════════════════════════════════════════════

/// Mirrors `forward::per_neuron_glu`: `[n_neurons × 2k] → [n_neurons × k]`.
pub(crate) fn per_neuron_glu(x: &[f32], n_neurons: usize, out_per: usize) -> Vec<f32> {
    let half = out_per / 2;
    let mut result = Vec::with_capacity(n_neurons * half);
    for n in 0..n_neurons {
        let base = n * out_per;
        for j in 0..half {
            let val = x[base + j];
            let gate = 1.0 / (1.0 + (-x[base + half + j]).exp());
            result.push(val * gate);
        }
    }
    result
}

/// Mirrors `forward::nlm_forward`.
pub(crate) fn nlm_forward(
    trace: &[f32],
    stage1: &SuperLinear,
    stage2: Option<&SuperLinear>,
    d_model: usize,
) -> Vec<f32> {
    let s1 = stage1.forward(trace);
    let s1_glu = per_neuron_glu(&s1, d_model, stage1.out_per);

    if let Some(s2) = stage2 {
        let s2_out = s2.forward(&s1_glu);
        per_neuron_glu(&s2_out, d_model, s2.out_per)
    } else {
        s1_glu
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Multihead attention — mirror forward::multihead_attention_with_attn.
// ═══════════════════════════════════════════════════════════════════════

/// Mirrors `forward::linear_slice`: partial matvec over rows `[row_start, row_end)`.
pub(crate) fn linear_slice(x: &[f32], linear: &Linear, row_start: usize, row_end: usize) -> Vec<f32> {
    let in_dim = linear.in_dim;
    let out_dim = row_end - row_start;
    let mut out = vec![0.0f32; out_dim];
    for i in 0..out_dim {
        let row_idx = row_start + i;
        let row = &linear.weight[row_idx * in_dim..(row_idx + 1) * in_dim];
        let mut acc = linear.bias[row_idx];
        for j in 0..in_dim {
            acc += row[j] * x[j];
        }
        out[i] = acc;
    }
    out
}

/// Mirrors `forward::multihead_attention_with_attn`. Returns `(out, per_head_weights)`.
/// `per_head_weights` is `[n_heads][n_tokens]` softmax weights.
pub(crate) fn multihead_attention(
    q_in: &[f32],
    kv_flat: &[f32],
    n_tokens: usize,
    d_input: usize,
    n_heads: usize,
    in_proj: &Linear,
    out_proj: &Linear,
) -> (Vec<f32>, Vec<Vec<f32>>) {
    let d_head = d_input / n_heads;
    let scale = 1.0 / (d_head as f32).sqrt();

    // Project query: rows [0, d_input)
    let q_full = linear_slice(q_in, in_proj, 0, d_input);

    // Project all KV tokens: K rows [d_input, 2*d_input), V rows [2*d_input, 3*d_input)
    let mut k_all = Vec::with_capacity(n_tokens * d_input);
    let mut v_all = Vec::with_capacity(n_tokens * d_input);
    for t in 0..n_tokens {
        let tok = &kv_flat[t * d_input..(t + 1) * d_input];
        k_all.extend_from_slice(&linear_slice(tok, in_proj, d_input, 2 * d_input));
        v_all.extend_from_slice(&linear_slice(tok, in_proj, 2 * d_input, 3 * d_input));
    }

    let mut concat_heads = Vec::with_capacity(d_input);
    let mut per_head_weights: Vec<Vec<f32>> = Vec::with_capacity(n_heads);
    for h in 0..n_heads {
        let q_h = &q_full[h * d_head..(h + 1) * d_head];

        // Scores
        let mut scores = Vec::with_capacity(n_tokens);
        for t in 0..n_tokens {
            let k_h = &k_all[t * d_input + h * d_head..t * d_input + (h + 1) * d_head];
            let dot: f32 = q_h.iter().zip(k_h).map(|(&a, &b)| a * b).sum();
            scores.push(dot * scale);
        }

        // Softmax
        let max_s = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_s: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
        let sum_s: f32 = exp_s.iter().sum();
        let inv_sum = 1.0 / sum_s;

        // Weighted sum of values
        let mut head_out = vec![0.0f32; d_head];
        let mut head_weights: Vec<f32> = Vec::with_capacity(n_tokens);
        for t in 0..n_tokens {
            let w = exp_s[t] * inv_sum;
            head_weights.push(w);
            let v_h = &v_all[t * d_input + h * d_head..t * d_input + (h + 1) * d_head];
            for j in 0..d_head {
                head_out[j] += w * v_h[j];
            }
        }
        per_head_weights.push(head_weights);
        concat_heads.extend_from_slice(&head_out);
    }

    (out_proj.forward(&concat_heads), per_head_weights)
}

// ═══════════════════════════════════════════════════════════════════════
//  Certainty — mirror forward::compute_certainty.
// ═══════════════════════════════════════════════════════════════════════

/// Mirrors `forward::compute_certainty`: `[normalized_entropy, 1 - normalized_entropy]`.
pub(crate) fn compute_certainty(prediction: &[f32]) -> [f32; 2] {
    let n = prediction.len();
    if n <= 1 {
        return [0.0, 1.0];
    }
    let max_p = prediction.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_s: Vec<f32> = prediction.iter().map(|&x| (x - max_p).exp()).collect();
    let sum: f32 = exp_s.iter().sum();
    let max_ent = (n as f32).ln();
    let ent: f32 = exp_s
        .iter()
        .map(|&e| {
            let p = e / sum;
            if p > 1e-10 {
                -p * p.ln()
            } else {
                0.0
            }
        })
        .sum();
    let ne = if max_ent > 0.0 {
        (ent / max_ent).clamp(0.0, 1.0)
    } else {
        0.0
    };
    [ne, 1.0 - ne]
}

// ═══════════════════════════════════════════════════════════════════════
//  Public forward API.
// ═══════════════════════════════════════════════════════════════════════

/// Per-tick observable output of the CTM forward pass.
#[cfg_attr(feature = "wasm", derive(serde::Serialize))]
#[derive(Debug, Clone)]
pub struct ForwardOut {
    /// `[ticks][out_dims]`.
    pub predictions: Vec<Vec<f32>>,
    /// `[ticks]` of `[normalized_entropy, 1 - normalized_entropy]`.
    pub certainties: Vec<[f32; 2]>,
    /// `[ticks][d_model]` — the activated neuron pool each tick.
    pub activations: Vec<Vec<f32>>,
    /// `[ticks][heads][n_tokens]` — softmax attention weights.
    pub attention: Vec<Vec<Vec<f32>>>,
}

/// Faithful CTM forward pass over a raw observation.
///
/// `obs`: `[n_tokens × raw_dim]` flat (raw_dim must equal `weights.kv_proj.in_dim`).
///
/// Replicates `ctm_forward` → `ctm_forward_with_kv_inner` (the tick loop) with
/// attention trace, for `ExitStrategy::None`, no episodic memory. State
/// (`activated`, `trace`, sync accumulators) is local to this call — a fresh
/// `CtmState`, matching the oracle generation (one fresh state per sample).
pub fn forward(weights: &CtmWeights, obs: &[f32], n_tokens: usize, raw_dim: usize) -> ForwardOut {
    let cfg = &weights.config;
    let d = cfg.d_model;
    let d_in = cfg.d_input;
    let k = cfg.iterations;
    let m = cfg.memory_length;

    // ── Step 1: project + LayerNorm each observation token into KV space. ──
    let mut kv: Vec<f32> = Vec::with_capacity(n_tokens * d_in);
    for t in 0..n_tokens {
        let tok = &obs[t * raw_dim..(t + 1) * raw_dim];
        let mut projected = weights.kv_proj.forward(tok);
        affine_ln(&mut projected, &weights.kv_ln_gamma, &weights.kv_ln_beta);
        kv.extend_from_slice(&projected);
    }

    // ── Decay factors r = exp(-clamp(p, 0, 15)). ──
    let r_out: Vec<f32> = weights
        .decay_params_out
        .iter()
        .map(|&p| (-p.clamp(0.0, 15.0)).exp())
        .collect();
    let r_action: Vec<f32> = weights
        .decay_params_action
        .iter()
        .map(|&p| (-p.clamp(0.0, 15.0)).exp())
        .collect();

    // ── Fresh state (matches the oracle: a new CtmState per sample). ──
    let mut activated: Vec<f32> = weights.start_activated.clone();
    let mut trace_state: Vec<f32> = weights.start_trace.clone();

    let mut alpha_out: Vec<f32> = vec![0.0; cfg.n_synch_out];
    let mut beta_out: Vec<f32> = vec![0.0; cfg.n_synch_out];
    sync_init(
        &activated,
        &weights.sync_out_left,
        &weights.sync_out_right,
        &mut alpha_out,
        &mut beta_out,
    );

    let mut alpha_action: Vec<f32> = Vec::new();
    let mut beta_action: Vec<f32> = Vec::new();
    let mut action_initialized = false;

    let mut predictions: Vec<Vec<f32>> = Vec::with_capacity(k);
    let mut certainties: Vec<[f32; 2]> = Vec::with_capacity(k);
    let mut activations: Vec<Vec<f32>> = Vec::with_capacity(k);
    let mut attention: Vec<Vec<Vec<f32>>> = Vec::with_capacity(k);

    // ── Tick loop. ──
    for _tick in 0..k {
        let sync_action = if !action_initialized {
            sync_init(
                &activated,
                &weights.sync_action_left,
                &weights.sync_action_right,
                &mut alpha_action,
                &mut beta_action,
            );
            action_initialized = true;
            sync_read(&alpha_action, &beta_action)
        } else {
            sync_update(
                &activated,
                &weights.sync_action_left,
                &weights.sync_action_right,
                &mut alpha_action,
                &mut beta_action,
                &r_action,
            )
        };

        let q = weights.q_proj.forward(&sync_action);
        let (attn_out, tick_attn) = multihead_attention(
            &q,
            &kv,
            n_tokens,
            d_in,
            cfg.heads,
            &weights.mha_in_proj,
            &weights.mha_out_proj,
        );
        attention.push(tick_attn);

        // pre_syn = concat(attn_out, activated)
        let mut pre_syn = Vec::with_capacity(d_in + d);
        pre_syn.extend_from_slice(&attn_out);
        pre_syn.extend_from_slice(&activated);
        let pre_act = weights.synapse.forward(&pre_syn);

        // Shift trace history left by one, append new pre-activation.
        for n in 0..d {
            let base = n * m;
            trace_state.copy_within(base + 1..base + m, base);
            trace_state[base + m - 1] = pre_act[n];
        }

        activated = nlm_forward(&trace_state, &weights.nlm_stage1, weights.nlm_stage2.as_ref(), d);
        activations.push(activated.clone());

        let sync_out = sync_update(
            &activated,
            &weights.sync_out_left,
            &weights.sync_out_right,
            &mut alpha_out,
            &mut beta_out,
            &r_out,
        );

        let pred = weights.output_proj.forward(&sync_out);
        let cert = compute_certainty(&pred);
        predictions.push(pred);
        certainties.push(cert);
        // ExitStrategy::None — never break.
    }

    ForwardOut {
        predictions,
        certainties,
        activations,
        attention,
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  WASM bindings (feature "wasm").
// ═══════════════════════════════════════════════════════════════════════

/// Build the maze observation the maze CTM was trained on: one token per cell,
/// `RAW_DIM = 9` features = `[is_wall, is_agent, is_goal, wall_up, wall_down,
/// wall_left, wall_right, x_norm, y_norm]`. The 4 neighbor-wall bits give the
/// CTM the local connectivity it needs to navigate. Out-of-bounds counts as
/// wall. `grid` is row-major, `1 = wall`, `0 = open`, length `size*size`.
/// This is the single source of truth for the encoding (training, validation,
/// and browser all call it), so the model always sees identical inputs.
pub fn encode_maze(grid: &[u8], size: usize, agent: (usize, usize), goal: (usize, usize)) -> Vec<f32> {
    const RD: usize = 9;
    let mut t = vec![0.0f32; size * size * RD];
    let denom = (size - 1) as f32;
    let wall = |r: i32, c: i32| -> f32 {
        if r < 0 || c < 0 || r >= size as i32 || c >= size as i32 { 1.0 }
        else if grid[r as usize * size + c as usize] != 0 { 1.0 } else { 0.0 }
    };
    for r in 0..size {
        for c in 0..size {
            let i = (r * size + c) * RD;
            let (ri, ci) = (r as i32, c as i32);
            t[i] = wall(ri, ci);
            t[i + 1] = if (r, c) == agent { 1.0 } else { 0.0 };
            t[i + 2] = if (r, c) == goal { 1.0 } else { 0.0 };
            t[i + 3] = wall(ri - 1, ci);
            t[i + 4] = wall(ri + 1, ci);
            t[i + 5] = wall(ri, ci - 1);
            t[i + 6] = wall(ri, ci + 1);
            t[i + 7] = c as f32 / denom;
            t[i + 8] = r as f32 / denom;
        }
    }
    t
}

#[cfg(feature = "wasm")]
mod wasm_bindings {
    use super::*;
    use std::cell::RefCell;
    use wasm_bindgen::prelude::*;

    thread_local! {
        static WEIGHTS: RefCell<Option<CtmWeights>> = const { RefCell::new(None) };
    }

    /// Parse weights JSON and stash them in a thread-local for subsequent `run`s.
    /// Returns an error string on parse failure.
    #[wasm_bindgen]
    pub fn load_weights(json: &str) -> Result<(), JsValue> {
        let w: CtmWeights = serde_json::from_str(json)
            .map_err(|e| JsValue::from_str(&format!("load_weights: {e}")))?;
        WEIGHTS.with(|cell| *cell.borrow_mut() = Some(w));
        Ok(())
    }

    /// Run the forward pass over `obs` (`[n_tokens × raw_dim]` flat) and return
    /// the per-tick observable state (`ForwardOut`) as a JS value.
    /// Errors if `load_weights` has not been called yet.
    #[wasm_bindgen]
    pub fn run(obs: &[f32], n_tokens: usize, raw_dim: usize) -> Result<JsValue, JsValue> {
        WEIGHTS.with(|cell| {
            let borrowed = cell.borrow();
            let w = borrowed
                .as_ref()
                .ok_or_else(|| JsValue::from_str("run: weights not loaded; call load_weights first"))?;
            let out = forward(w, obs, n_tokens, raw_dim);
            serde_wasm_bindgen::to_value(&out)
                .map_err(|e| JsValue::from_str(&format!("run serialize: {e}")))
        })
    }

    /// Browser-facing maze encoder. `grid` is row-major (1 = wall), returns the
    /// flat `[size*size * 9]` observation to pass straight into `run`.
    #[wasm_bindgen]
    pub fn encode_maze_js(
        grid: &[u8], size: usize,
        agent_r: usize, agent_c: usize,
        goal_r: usize, goal_c: usize,
    ) -> Vec<f32> {
        super::encode_maze(grid, size, (agent_r, agent_c), (goal_r, goal_c))
    }
}
