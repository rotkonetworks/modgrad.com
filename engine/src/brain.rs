//! Full 8-region "brain" forward WITH the visual retina — dep-free.
//!
//! Mirrors, operation-for-operation (inference only):
//!   - `modgrad_codec::retina::{Conv2d, VisualCortex}` (`spatial_tokens`)
//!   - `modgrad_compute::kv_buffer::EpisodicMemory`
//!   - `modgrad_ctm::graph::{RegionalWeights, RegionalConfig, regional_forward}`
//!   - `modgrad_ctm::forward::ctm_forward` with `ExitStrategy::AdaptiveGate`
//!     inner exit and episodic-memory KV (hippocampus).
//!
//! Reuses the single-CTM scalar primitives from the crate root. Proven
//! bit-exact against `brain_oracle` via `tests/brain.rs`.

use serde::Deserialize;

use crate::{
    affine_ln, compute_certainty, multihead_attention, nlm_forward, sync_init, sync_read,
    sync_update, CtmWeights, Linear,
};

// ═══════════════════════════════════════════════════════════════════════
//  VisualCortex (retina) — mirrors modgrad_codec::retina.
// ═══════════════════════════════════════════════════════════════════════

/// Mirrors `modgrad_codec::retina::Conv2d`.
/// `weight`: `[out_channels × in_channels × kh × kw]` row-major.
#[derive(Debug, Clone, Deserialize)]
pub struct Conv2d {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
}

/// Mirrors `retina::im2col` exactly. Returns `[patch_size × (n · n_patches)]`.
fn im2col(
    input: &[f32], n: usize, in_ch: usize, h: usize, w: usize,
    k: usize, s: usize, p: usize, out_h: usize, out_w: usize,
) -> Vec<f32> {
    let patch_size = in_ch * k * k;
    let n_patches = out_h * out_w;
    let cols_per_img = n_patches;
    let total_cols = n * cols_per_img;
    let mut col = vec![0.0f32; patch_size * total_cols];
    let img_stride = in_ch * h * w;
    for b in 0..n {
        let img = &input[b * img_stride..(b + 1) * img_stride];
        let col_off = b * cols_per_img;
        for oh in 0..out_h {
            for ow in 0..out_w {
                let pidx = col_off + oh * out_w + ow;
                for ic in 0..in_ch {
                    for kh in 0..k {
                        for kw in 0..k {
                            let ih = (oh * s + kh) as isize - p as isize;
                            let iw = (ow * s + kw) as isize - p as isize;
                            if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                let in_idx = ic * h * w + ih as usize * w + iw as usize;
                                let row_idx = ic * k * k + kh * k + kw;
                                col[row_idx * total_cols + pidx] = img[in_idx];
                            }
                        }
                    }
                }
            }
        }
    }
    col
}

impl Conv2d {
    /// Faithful to `Conv2d::forward` for `n = 1`: im2col + matmul + bias.
    /// Returns `(output [out_ch × out_h × out_w] CHW, out_h, out_w)`.
    fn forward1(&self, input: &[f32], h: usize, w: usize) -> (Vec<f32>, usize, usize) {
        let k = self.kernel_size;
        let s = self.stride;
        let p = self.padding;
        let out_h = (h + 2 * p - k) / s + 1;
        let out_w = (w + 2 * p - k) / s + 1;
        let patch_size = self.in_channels * k * k;
        let total_cols = out_h * out_w; // n = 1

        let col = im2col(input, 1, self.in_channels, h, w, k, s, p, out_h, out_w);

        // y[oc][col] = bias[oc] + Σ_k W[oc][k] · col[k][col].
        let mut y = vec![0.0f32; self.out_channels * total_cols];
        for oc in 0..self.out_channels {
            let w_row = &self.weight[oc * patch_size..(oc + 1) * patch_size];
            let y_row = &mut y[oc * total_cols..(oc + 1) * total_cols];
            let b = self.bias[oc];
            for c in 0..total_cols {
                let mut acc = b;
                for kk in 0..patch_size {
                    acc += w_row[kk] * col[kk * total_cols + c];
                }
                y_row[c] = acc;
            }
        }
        (y, out_h, out_w)
    }
}

/// Mirrors `retina::leaky_relu` (slope 0.1).
fn leaky_relu(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v *= 0.1;
        }
    }
}

/// Mirrors `modgrad_codec::retina::VisualCortex` (only the fields the
/// `spatial_tokens` path reads; the rest are skipped on deserialize).
#[derive(Debug, Clone, Deserialize)]
pub struct VisualCortex {
    pub retina: Conv2d,
    pub v1: Conv2d,
    pub v2: Conv2d,
    pub v4: Conv2d,
    pub input_h: usize,
    pub input_w: usize,
    #[serde(default)]
    pub per_token_ln_v4: bool,
}

impl VisualCortex {
    /// Faithful to `VisualCortex::spatial_tokens`: retina → v1 → v2 → v4,
    /// leaky_relu between, then CHW → tokens `[n_tokens × channels]`.
    /// `raw`: `[3 × input_h × input_w]` CHW. (per_token_ln_v4 = false here.)
    pub fn spatial_tokens(&self, raw: &[f32]) -> (Vec<f32>, usize, usize) {
        let h = self.input_h;
        let w = self.input_w;

        let (mut r_out, rh, rw) = self.retina.forward1(raw, h, w);
        leaky_relu(&mut r_out);
        let (mut v1_out, h1, w1) = self.v1.forward1(&r_out, rh, rw);
        leaky_relu(&mut v1_out);
        let (mut v2_out, h2, w2) = self.v2.forward1(&v1_out, h1, w1);
        leaky_relu(&mut v2_out);
        let (mut v4_out, h4, w4) = self.v4.forward1(&v2_out, h2, w2);
        leaky_relu(&mut v4_out);

        let channels = self.v4.out_channels;
        let n_tokens = h4 * w4;
        let mut tokens = vec![0.0f32; n_tokens * channels];
        for y in 0..h4 {
            for x in 0..w4 {
                let token_idx = y * w4 + x;
                for c in 0..channels {
                    tokens[token_idx * channels + c] = v4_out[c * h4 * w4 + y * w4 + x];
                }
            }
        }
        // per_token_ln_v4 defaults false for this brain — skip the LN.
        (tokens, n_tokens, channels)
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  EpisodicMemory — mirrors modgrad_compute::kv_buffer::EpisodicMemory.
// ═══════════════════════════════════════════════════════════════════════

/// Faithful scalar replica of the hierarchical bounded KV buffer used by
/// the hippocampus region. Only the methods the forward path exercises
/// (`push`, `as_kv`, `n_tokens`, `is_empty`) are implemented — but the
/// push cascade (short→mid via Otsu TAS, mid→long via merge) is mirrored
/// exactly so the per-tick KV stream matches the SDK bit-for-bit.
struct EpisodicMemory {
    dim: usize,
    short: Vec<f32>,
    mid: Vec<f32>,
    long: Vec<f32>,
    cap_short: usize,
    cap_mid: usize,
    cap_long: usize,
    recent_distances: Vec<f32>,
}

impl EpisodicMemory {
    fn new(dim: usize, short: usize, mid: usize, long: usize) -> Self {
        Self {
            dim,
            short: Vec::new(),
            mid: Vec::new(),
            long: Vec::new(),
            cap_short: short,
            cap_mid: mid,
            cap_long: long,
            recent_distances: Vec::new(),
        }
    }

    fn n_short(&self) -> usize { self.short.len() / self.dim.max(1) }
    fn n_mid(&self) -> usize { self.mid.len() / self.dim.max(1) }
    fn n_long(&self) -> usize { self.long.len() / self.dim.max(1) }
    fn n_tokens(&self) -> usize { self.n_short() + self.n_mid() + self.n_long() }
    fn is_empty(&self) -> bool {
        self.short.is_empty() && self.mid.is_empty() && self.long.is_empty()
    }

    /// Order: long (oldest) → mid → short (newest).
    fn as_kv(&self) -> Vec<f32> {
        let mut kv = Vec::with_capacity(self.long.len() + self.mid.len() + self.short.len());
        kv.extend_from_slice(&self.long);
        kv.extend_from_slice(&self.mid);
        kv.extend_from_slice(&self.short);
        kv
    }

    fn push(&mut self, entry: &[f32]) {
        self.short.extend_from_slice(entry);
        if self.n_short() > self.cap_short {
            let d = self.dim;
            let evicted: Vec<f32> = self.short.drain(..d).collect();
            self.push_mid(evicted);
        }
    }

    fn push_mid(&mut self, entry: Vec<f32>) {
        if self.mid.len() >= self.dim {
            let recent = &self.mid[self.mid.len() - self.dim..];
            let dist = cosine_distance(&entry, recent);
            self.recent_distances.push(dist);
            if self.recent_distances.len() > 64 {
                self.recent_distances.remove(0);
            }
            let threshold = if self.recent_distances.len() >= 4 {
                otsu_threshold(&self.recent_distances)
            } else {
                0.05
            };
            if dist <= threshold {
                if self.n_mid() > self.cap_mid {
                    self.compress_mid_to_long();
                }
                return;
            }
        }
        self.mid.extend_from_slice(&entry);
        if self.n_mid() > self.cap_mid {
            self.compress_mid_to_long();
        }
    }

    fn compress_mid_to_long(&mut self) {
        let n_evict = self.cap_mid / 2;
        let evict_bytes = n_evict * self.dim;
        if evict_bytes > self.mid.len() {
            return;
        }
        let evicted: Vec<f32> = self.mid.drain(..evict_bytes).collect();
        let entries: Vec<&[f32]> = evicted.chunks_exact(self.dim).collect();
        let merged = merge_similar(&entries, self.dim);
        for proto in &merged {
            self.long.extend_from_slice(proto);
        }
        let overflow = self.n_long().saturating_sub(self.cap_long);
        if overflow > 0 {
            self.long.drain(..overflow * self.dim);
        }
    }
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-10 {
        return 1.0;
    }
    1.0 - dot / denom
}

fn otsu_threshold(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let min_v = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_v = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let range = max_v - min_v;
    if range < 1e-10 {
        return min_v;
    }
    const BINS: usize = 64;
    let mut hist = [0u32; BINS];
    for &v in values {
        let bin = (((v - min_v) / range) * (BINS - 1) as f32) as usize;
        hist[bin.min(BINS - 1)] += 1;
    }
    let total = values.len() as f32;
    let sum_total: f32 = hist.iter().enumerate().map(|(i, &c)| i as f32 * c as f32).sum();
    let (mut best_bin, mut best_var) = (0usize, 0.0f32);
    let (mut w0, mut sum0) = (0.0f32, 0.0f32);
    for (i, &count) in hist.iter().enumerate() {
        w0 += count as f32;
        if w0 == 0.0 {
            continue;
        }
        let w1 = total - w0;
        if w1 == 0.0 {
            break;
        }
        sum0 += i as f32 * count as f32;
        let diff = sum0 / w0 - (sum_total - sum0) / w1;
        let var = w0 * w1 * diff * diff;
        if var > best_var {
            best_var = var;
            best_bin = i;
        }
    }
    min_v + (best_bin as f32 / (BINS - 1) as f32) * range
}

fn merge_similar(entries: &[&[f32]], dim: usize) -> Vec<Vec<f32>> {
    let n = entries.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![entries[0].to_vec()];
    }
    let mut distances = Vec::with_capacity(n * 3);
    for i in 0..n {
        for j in (i + 1)..n.min(i + 4) {
            distances.push((i, j, cosine_distance(entries[i], entries[j])));
        }
    }
    let dist_values: Vec<f32> = distances.iter().map(|d| d.2).collect();
    let threshold = otsu_threshold(&dist_values);
    let mut parent: Vec<usize> = (0..n).collect();
    for &(i, j, d) in &distances {
        if d <= threshold {
            union(&mut parent, i, j);
        }
    }
    let mut groups: std::collections::HashMap<usize, Vec<usize>> = Default::default();
    for i in 0..n {
        let root = find(&mut parent, i);
        groups.entry(root).or_default().push(i);
    }
    groups
        .values()
        .map(|members| {
            let mut proto = vec![0.0f32; dim];
            let wt = 1.0 / members.len() as f32;
            for &idx in members {
                for d in 0..dim {
                    proto[d] += wt * entries[idx][d];
                }
            }
            proto
        })
        .collect()
}

fn find(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}

fn union(parent: &mut [usize], a: usize, b: usize) {
    let ra = find(parent, a);
    let rb = find(parent, b);
    if ra != rb {
        parent[ra] = rb;
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  RegionalWeights — mirrors modgrad_ctm::graph.
// ═══════════════════════════════════════════════════════════════════════

/// Mirrors `modgrad_ctm::graph::Connection`.
#[derive(Debug, Clone, Deserialize)]
pub struct Connection {
    pub from: Vec<usize>,
    pub to: usize,
    #[serde(default)]
    pub receives_observation: bool,
    #[serde(default)]
    pub observation_scale: usize,
}

/// Mirrors the subset of `modgrad_ctm::graph::RegionalConfig` the forward
/// path reads. Unknown fields (router, aux_losses, cereb_mode, …) are
/// ignored on deserialize.
#[derive(Debug, Clone, Deserialize)]
pub struct RegionalConfig {
    pub regions: Vec<crate::CtmConfig>,
    pub region_names: Vec<String>,
    pub connections: Vec<Connection>,
    pub outer_ticks: usize,
    #[serde(default)]
    pub exit_strategy: crate::ExitStrategy,
    pub n_global_sync: usize,
    pub out_dims: usize,
    pub raw_obs_dim: usize,
    #[serde(default)]
    pub obs_scale_dims: Vec<usize>,
}

/// Mirrors `modgrad_ctm::graph::RegionalWeights` (forward-relevant fields).
/// `embeddings` is marked default + skipped — the oracle strips it.
#[derive(Debug, Clone, Deserialize)]
pub struct RegionalWeights {
    pub config: RegionalConfig,
    #[serde(default)]
    pub embeddings: Vec<f32>,
    pub regions: Vec<CtmWeights>,
    pub connection_synapses: Vec<Linear>,
    pub obs_proj: Linear,
    pub global_sync_left: Vec<usize>,
    pub global_sync_right: Vec<usize>,
    pub global_decay: Vec<f32>,
    pub output_proj: Linear,
    #[serde(default)]
    pub outer_exit_gate: Option<Linear>,
}

// ═══════════════════════════════════════════════════════════════════════
//  Per-region CTM state + forward (AdaptiveGate inner exit + episodic).
// ═══════════════════════════════════════════════════════════════════════

/// Per-region mutable state, persisted across outer ticks (mirrors
/// `modgrad_ctm::weights::CtmState`). `activated` / `trace` carry the
/// continuous-thinking state; `episodic` is `Some` only for hippocampus.
struct RegionState {
    trace: Vec<f32>,
    activated: Vec<f32>,
    alpha_out: Vec<f32>,
    beta_out: Vec<f32>,
    episodic: Option<EpisodicMemory>,
}

impl RegionState {
    fn new(w: &CtmWeights, is_hippocampus: bool) -> Self {
        let episodic = if is_hippocampus {
            Some(EpisodicMemory::new(w.config.d_input, 4, 16, 64))
        } else {
            None
        };
        Self {
            trace: w.start_trace.clone(),
            activated: w.start_activated.clone(),
            alpha_out: vec![0.0; w.config.n_synch_out],
            beta_out: vec![0.0; w.config.n_synch_out],
            episodic,
        }
    }
}

/// One region's `ctm_forward(CtmInput::Raw{ n_tokens: 1 })`, faithful to
/// the SDK: project the raw obs to KV, prepend episodic KV (if any),
/// run the inner tick loop with `AdaptiveGate` early exit, then push this
/// call's KV into episodic memory. Mutates `st.activated`/`st.trace`/
/// sync accumulators in place (continuous-thinking across outer ticks).
fn ctm_forward_region(w: &CtmWeights, st: &mut RegionState, obs: &[f32], raw_dim: usize) {
    let cfg = &w.config;
    let d = cfg.d_model;
    let d_in = cfg.d_input;
    let k = cfg.iterations;
    let m = cfg.memory_length;

    // Step 1: project this call's single token to KV (+ LN).
    let mut new_kv = Vec::with_capacity(d_in);
    {
        let tok = &obs[0..raw_dim];
        let mut projected = w.kv_proj.forward(tok);
        affine_ln(&mut projected, &w.kv_ln_gamma, &w.kv_ln_beta);
        new_kv.extend_from_slice(&projected);
    }

    // Step 2: prepend episodic entries (order: long → mid → short).
    let has_ep = st.episodic.as_ref().map_or(false, |e| !e.is_empty());
    let (kv_used, n_total) = if has_ep {
        let ep = st.episodic.as_ref().unwrap();
        let ep_n = ep.n_tokens();
        let mut combined = Vec::with_capacity((ep_n + 1) * d_in);
        combined.extend_from_slice(&ep.as_kv());
        combined.extend_from_slice(&new_kv);
        (combined, ep_n + 1)
    } else {
        (new_kv.clone(), 1)
    };

    // Step 3: inner tick loop.
    let r_out: Vec<f32> = w
        .decay_params_out
        .iter()
        .map(|&p| (-p.clamp(0.0, 15.0)).exp())
        .collect();
    let r_action: Vec<f32> = w
        .decay_params_action
        .iter()
        .map(|&p| (-p.clamp(0.0, 15.0)).exp())
        .collect();

    sync_init(
        &st.activated,
        &w.sync_out_left,
        &w.sync_out_right,
        &mut st.alpha_out,
        &mut st.beta_out,
    );

    let mut alpha_action: Vec<f32> = Vec::new();
    let mut beta_action: Vec<f32> = Vec::new();
    let mut action_initialized = false;

    let mut exit_cdf = 0.0f32;
    let mut survival = 1.0f32;

    for _tick in 0..k {
        let sync_action = if !action_initialized {
            sync_init(
                &st.activated,
                &w.sync_action_left,
                &w.sync_action_right,
                &mut alpha_action,
                &mut beta_action,
            );
            action_initialized = true;
            sync_read(&alpha_action, &beta_action)
        } else {
            sync_update(
                &st.activated,
                &w.sync_action_left,
                &w.sync_action_right,
                &mut alpha_action,
                &mut beta_action,
                &r_action,
            )
        };

        let q = w.q_proj.forward(&sync_action);
        let (attn_out, _attn) = multihead_attention(
            &q, &kv_used, n_total, d_in, cfg.heads, &w.mha_in_proj, &w.mha_out_proj,
        );

        let mut pre_syn = Vec::with_capacity(d_in + d);
        pre_syn.extend_from_slice(&attn_out);
        pre_syn.extend_from_slice(&st.activated);
        let pre_act = w.synapse.forward(&pre_syn);

        for n in 0..d {
            let base = n * m;
            st.trace.copy_within(base + 1..base + m, base);
            st.trace[base + m - 1] = pre_act[n];
        }

        st.activated = nlm_forward(&st.trace, &w.nlm_stage1, w.nlm_stage2.as_ref(), d);

        let sync_out = sync_update(
            &st.activated,
            &w.sync_out_left,
            &w.sync_out_right,
            &mut st.alpha_out,
            &mut st.beta_out,
            &r_out,
        );

        let pred = w.output_proj.forward(&sync_out);
        let _cert = compute_certainty(&pred);

        // Inner exit: every region in this brain uses AdaptiveGate.
        match &cfg.exit_strategy {
            crate::ExitStrategy::AdaptiveGate { threshold, .. } => {
                if let Some(ref gate) = w.exit_gate {
                    let gate_logit = gate.forward(&sync_out);
                    let lambda = 1.0 / (1.0 + (-gate_logit[0]).exp());
                    let p_exit = lambda * survival;
                    exit_cdf += p_exit;
                    survival *= 1.0 - lambda;
                    if exit_cdf > *threshold {
                        break;
                    }
                }
            }
            crate::ExitStrategy::Certainty { threshold } => {
                if _cert[1] > *threshold {
                    break;
                }
            }
            crate::ExitStrategy::None => {}
        }
    }

    // Step 4: persist this call's KV into episodic memory.
    if let Some(ref mut mem) = st.episodic {
        mem.push(&new_kv);
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Regional forward — mirrors modgrad_ctm::graph::regional_forward.
// ═══════════════════════════════════════════════════════════════════════

/// Per-outer-tick observable trace of the brain forward pass.
#[cfg_attr(feature = "wasm", derive(serde::Serialize))]
#[derive(Debug, Clone)]
pub struct BrainTick {
    /// `[out_dims]` next-step prediction.
    pub prediction: Vec<f32>,
    /// `[n_regions][d_model]` per-region activated state this tick.
    pub region_activations: Vec<Vec<f32>>,
    /// `[n_global_sync]` global sync vector.
    pub global_sync: Vec<f32>,
    /// Outer-level exit lambda (None if no exit gate).
    pub exit_lambda: Option<f32>,
}

/// Output of the brain forward pass.
#[cfg_attr(feature = "wasm", derive(serde::Serialize))]
#[derive(Debug, Clone)]
pub struct BrainOut {
    pub ticks: Vec<BrainTick>,
    pub ticks_used: usize,
}

/// Faithful brain forward over a flat observation buffer (`observation`
/// is the retina's `[n_tokens × token_dim]` spatial-token stream; the
/// SDK's `obs_proj` / connection synapses read only their `in_dim`
/// prefix of it, exactly as on the device). Sequential — no rayon — so
/// the floating-point accumulation order is deterministic.
pub fn regional_forward(w: &RegionalWeights, observation: &[f32]) -> BrainOut {
    let cfg = &w.config;
    let n_regions = cfg.regions.len();
    let total_neurons: usize = cfg.regions.iter().map(|r| r.d_model).sum();
    let n_sync = cfg.n_global_sync;

    // Per-region persistent state. Hippocampus gets episodic memory.
    let mut states: Vec<RegionState> = (0..n_regions)
        .map(|r| {
            let is_hippo = cfg
                .region_names
                .get(r)
                .map_or(false, |n| n.contains("hippocampus"));
            RegionState::new(&w.regions[r], is_hippo)
        })
        .collect();

    // region_outputs / prev_outputs both start at each region's start_activated.
    let mut region_outputs: Vec<Vec<f32>> =
        w.regions.iter().map(|rw| rw.start_activated.clone()).collect();
    let mut prev_outputs: Vec<Vec<f32>> = region_outputs.clone();

    let mut global_alpha = vec![0.0f32; n_sync];
    let mut global_beta = vec![1.0f32; n_sync];

    let obs_projected = w.obs_proj.forward(observation);

    let mut out_ticks: Vec<BrainTick> = Vec::with_capacity(cfg.outer_ticks);
    let mut exit_cdf = 0.0f32;
    let mut survival = 1.0f32;

    for _outer_tick in 0..cfg.outer_ticks {
        // Phase A: build each region's observation via fixed connections.
        let region_obs: Vec<Vec<f32>> = (0..n_regions)
            .map(|r| {
                let mut slot: Vec<f32> = Vec::new();
                for (ci, conn) in cfg.connections.iter().enumerate() {
                    if conn.to == r {
                        let mut src = Vec::new();
                        for &from_idx in &conn.from {
                            src.extend_from_slice(&prev_outputs[from_idx]);
                        }
                        if conn.receives_observation {
                            src.extend_from_slice(observation);
                        }
                        let projected = w.connection_synapses[ci].forward(&src);
                        // merge_into_region_obs: assign on empty, else add.
                        if slot.is_empty() {
                            slot = projected;
                        } else {
                            let nmin = slot.len().min(projected.len());
                            for i in 0..nmin {
                                slot[i] += projected[i];
                            }
                        }
                    }
                }
                if slot.is_empty() {
                    obs_projected.clone()
                } else {
                    slot
                }
            })
            .collect();

        // Phase B: run each region's CTM (sequential).
        for r in 0..n_regions {
            let d_input = w.regions[r].config.d_input;
            ctm_forward_region(&w.regions[r], &mut states[r], &region_obs[r], d_input);
        }

        // Phase C: commit outputs.
        for r in 0..n_regions {
            region_outputs[r] = states[r].activated.clone();
        }
        prev_outputs = region_outputs.clone();

        // Phase 3: global sync.
        let mut all_act = vec![0.0f32; total_neurons];
        {
            let mut offset = 0;
            for r in 0..n_regions {
                let dr = region_outputs[r].len();
                all_act[offset..offset + dr].copy_from_slice(&region_outputs[r]);
                offset += dr;
            }
        }
        for i in 0..n_sync {
            let l = w.global_sync_left[i];
            let rr = w.global_sync_right[i];
            if l < all_act.len() && rr < all_act.len() {
                let pw = all_act[l] * all_act[rr];
                let decay = (-w.global_decay[i].clamp(0.0, 15.0)).exp();
                global_alpha[i] = decay * global_alpha[i] + pw;
                global_beta[i] = decay * global_beta[i] + 1.0;
            }
        }
        let mut gs_buf = vec![0.0f32; n_sync];
        for i in 0..n_sync {
            gs_buf[i] = global_alpha[i] / global_beta[i].sqrt().max(1e-8);
        }

        // Phase 4: output prediction.
        let prediction = w.output_proj.forward(&gs_buf);

        // Phase 5: AdaptiveGate outer exit.
        let mut exit_lambda = None;
        let mut do_break = false;
        if let crate::ExitStrategy::AdaptiveGate { threshold, .. } = &cfg.exit_strategy {
            if let Some(ref gate) = w.outer_exit_gate {
                let gate_logit = gate.forward(&gs_buf);
                let lambda = 1.0 / (1.0 + (-gate_logit[0]).exp());
                exit_lambda = Some(lambda);
                let p_exit = lambda * survival;
                exit_cdf += p_exit;
                survival *= 1.0 - lambda;
                if exit_cdf > *threshold {
                    do_break = true;
                }
            }
        }

        out_ticks.push(BrainTick {
            prediction,
            region_activations: region_outputs.clone(),
            global_sync: gs_buf,
            exit_lambda,
        });

        if do_break {
            break;
        }
    }

    let ticks_used = out_ticks.len();
    BrainOut { ticks: out_ticks, ticks_used }
}

// ═══════════════════════════════════════════════════════════════════════
//  WASM bindings (feature "wasm").
// ═══════════════════════════════════════════════════════════════════════

#[cfg(feature = "wasm")]
mod wasm_bindings {
    use super::*;
    use std::cell::RefCell;
    use wasm_bindgen::prelude::*;

    thread_local! {
        static CORTEX: RefCell<Option<VisualCortex>> = const { RefCell::new(None) };
        static BRAIN: RefCell<Option<RegionalWeights>> = const { RefCell::new(None) };
    }

    /// Parse `brain_weights.json` (`{ cortex, regional }`) and stash both
    /// the retina and the regional weights for subsequent `run_brain`s.
    #[wasm_bindgen]
    pub fn load_brain_weights(json: &str) -> Result<(), JsValue> {
        #[derive(Deserialize)]
        struct BrainWeights {
            cortex: VisualCortex,
            regional: RegionalWeights,
        }
        let bw: BrainWeights = serde_json::from_str(json)
            .map_err(|e| JsValue::from_str(&format!("load_brain_weights: {e}")))?;
        CORTEX.with(|c| *c.borrow_mut() = Some(bw.cortex));
        BRAIN.with(|b| *b.borrow_mut() = Some(bw.regional));
        Ok(())
    }

    /// Run the brain on raw RGB pixels `[3 × H × W]` CHW: retina →
    /// spatial tokens → `regional_forward`. Returns the per-tick
    /// `BrainOut` as a JS value. Requires `load_brain_weights` first.
    #[wasm_bindgen]
    pub fn run_brain_pixels(pixels: &[f32]) -> Result<JsValue, JsValue> {
        CORTEX.with(|c| {
            let cortex_ref = c.borrow();
            let cortex = cortex_ref
                .as_ref()
                .ok_or_else(|| JsValue::from_str("run_brain: weights not loaded"))?;
            let (obs, _n, _d) = cortex.spatial_tokens(pixels);
            BRAIN.with(|b| {
                let brain_ref = b.borrow();
                let brain = brain_ref
                    .as_ref()
                    .ok_or_else(|| JsValue::from_str("run_brain: weights not loaded"))?;
                let out = regional_forward(brain, &obs);
                serde_wasm_bindgen::to_value(&out)
                    .map_err(|e| JsValue::from_str(&format!("run_brain serialize: {e}")))
            })
        })
    }

    /// Run the brain directly on a pre-computed flat observation
    /// (`[n_tokens × token_dim]`), bypassing the retina. Returns the
    /// per-tick `BrainOut` as a JS value.
    #[wasm_bindgen]
    pub fn run_brain_obs(obs: &[f32]) -> Result<JsValue, JsValue> {
        BRAIN.with(|b| {
            let brain_ref = b.borrow();
            let brain = brain_ref
                .as_ref()
                .ok_or_else(|| JsValue::from_str("run_brain: weights not loaded"))?;
            let out = regional_forward(brain, obs);
            serde_wasm_bindgen::to_value(&out)
                .map_err(|e| JsValue::from_str(&format!("run_brain serialize: {e}")))
        })
    }
}
