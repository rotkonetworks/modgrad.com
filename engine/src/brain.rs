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
    /// Infer the square input dims from a raw CHW `[3 × s × s]` buffer.
    /// If the buffer is exactly a 3-channel square (3·s·s == len) we use that
    /// `s` (arbitrary-resolution, zero-shot); otherwise fall back to the stored
    /// `input_h/input_w`. A 9×9 buffer is 243 floats → n=81 → s=9 → 3·81==243,
    /// so this is BIT-EXACT for the trained 9×9 brain.
    #[inline]
    fn dims_for(&self, raw: &[f32]) -> (usize, usize) {
        let n = raw.len() / 3;
        let s = (n as f64).sqrt().round() as usize;
        if s >= 1 && 3 * s * s == raw.len() {
            (s, s)
        } else {
            (self.input_h, self.input_w)
        }
    }

    /// Faithful to `VisualCortex::spatial_tokens`: retina → v1 → v2 → v4,
    /// leaky_relu between, then CHW → tokens `[n_tokens × channels]`.
    /// `raw`: `[3 × input_h × input_w]` CHW. (per_token_ln_v4 = false here.)
    pub fn spatial_tokens(&self, raw: &[f32]) -> (Vec<f32>, usize, usize) {
        let (h, w) = self.dims_for(raw);

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

    /// Per-layer feature maps (CHW), for visualisation: the same retina →
    /// v1 → v2 → v4 cascade `spatial_tokens` runs, but returns every layer's
    /// activation. Each entry: (name, data [ch×h×w], channels, h, w).
    pub fn feature_maps(
        &self,
        raw: &[f32],
    ) -> Vec<(&'static str, Vec<f32>, usize, usize, usize)> {
        let (h, w) = self.dims_for(raw);
        let (mut r, rh, rw) = self.retina.forward1(raw, h, w);
        leaky_relu(&mut r);
        let (mut v1, h1, w1) = self.v1.forward1(&r, rh, rw);
        leaky_relu(&mut v1);
        let (mut v2, h2, w2) = self.v2.forward1(&v1, h1, w1);
        leaky_relu(&mut v2);
        let (mut v4, h4, w4) = self.v4.forward1(&v2, h2, w2);
        leaky_relu(&mut v4);
        vec![
            ("retina", r, self.retina.out_channels, rh, rw),
            ("v1", v1, self.v1.out_channels, h1, w1),
            ("v2", v2, self.v2.out_channels, h2, w2),
            ("v4", v4, self.v4.out_channels, h4, w4),
        ]
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
//  Telemetry — derived, read-only views over a completed BrainOut.
//
//  These are NOT new signals invented out of thin air: every field below is
//  computed directly from quantities the forward already produces
//  (per-region activations, per-tick global sync, per-tick outer exit λ,
//  the readout prediction). Names are deliberately literal so the frontend
//  never claims the engine computes something it does not. This
//  reimplementation has no neuromodulator / homeostasis state, so none is
//  fabricated — we surface the activation statistics that ARE computable.
// ═══════════════════════════════════════════════════════════════════════

/// Per-region activation statistics for a single outer tick.
#[cfg_attr(feature = "wasm", derive(serde::Serialize))]
#[derive(Debug, Clone)]
pub struct RegionStat {
    /// Region index (matches `config.region_names`).
    pub region: usize,
    /// Region name from `config.region_names` (empty if unnamed).
    pub name: String,
    /// `d_model` (number of neurons) for this region.
    pub d_model: usize,
    /// RMS of the region's activated state this tick: sqrt(mean(a^2)).
    pub activation_rms: f32,
    /// Peak |activation| this tick.
    pub activation_peak: f32,
    /// Mean activation this tick (signed).
    pub activation_mean: f32,
}

/// One outer tick's worth of per-region telemetry plus the global scalars
/// that are genuinely computed by the forward at that tick.
#[cfg_attr(feature = "wasm", derive(serde::Serialize))]
#[derive(Debug, Clone)]
pub struct TickTelemetry {
    /// Outer-tick index (0-based).
    pub tick: usize,
    /// Per-region activation stats.
    pub regions: Vec<RegionStat>,
    /// RMS of the global sync vector this tick.
    pub global_sync_rms: f32,
    /// Peak |global sync| this tick.
    pub global_sync_peak: f32,
    /// Outer adaptive-gate exit λ this tick (None if no outer gate).
    pub exit_lambda: Option<f32>,
}

/// Compute per-tick / per-region telemetry from a finished `BrainOut`.
pub fn region_telemetry(w: &RegionalWeights, out: &BrainOut) -> Vec<TickTelemetry> {
    let names = &w.config.region_names;
    out.ticks
        .iter()
        .enumerate()
        .map(|(t, tick)| {
            let regions = tick
                .region_activations
                .iter()
                .enumerate()
                .map(|(r, act)| {
                    let d = act.len();
                    let (mut sumsq, mut sum, mut peak) = (0.0f32, 0.0f32, 0.0f32);
                    for &v in act {
                        sumsq += v * v;
                        sum += v;
                        let a = v.abs();
                        if a > peak {
                            peak = a;
                        }
                    }
                    let denom = d.max(1) as f32;
                    RegionStat {
                        region: r,
                        name: names.get(r).cloned().unwrap_or_default(),
                        d_model: d,
                        activation_rms: (sumsq / denom).sqrt(),
                        activation_peak: peak,
                        activation_mean: sum / denom,
                    }
                })
                .collect();
            let (mut gs_sumsq, mut gs_peak) = (0.0f32, 0.0f32);
            for &v in &tick.global_sync {
                gs_sumsq += v * v;
                let a = v.abs();
                if a > gs_peak {
                    gs_peak = a;
                }
            }
            let gs_denom = tick.global_sync.len().max(1) as f32;
            TickTelemetry {
                tick: t,
                regions,
                global_sync_rms: (gs_sumsq / gs_denom).sqrt(),
                global_sync_peak: gs_peak,
                exit_lambda: tick.exit_lambda,
            }
        })
        .collect()
}

/// "What drives the decision" — honest readout introspection.
///
/// The readout is `prediction = output_proj.forward(global_sync)`, i.e.
/// `pred[j] = bias[j] + Σ_i W[j,i] · gs[i]`. For the LAST tick we expose the
/// pre-activation (the global-sync vector that feeds the readout) and, for
/// each of the first-5 move logits, the per-input contribution `W[j,i]·gs[i]`
/// so the frontend can show which sync channels push each move up or down.
/// No attention/eligibility is fabricated — this brain's forward does not
/// surface those at the outer level.
#[cfg_attr(feature = "wasm", derive(serde::Serialize))]
#[derive(Debug, Clone)]
pub struct DecisionDrivers {
    /// Index of the tick these drivers were read from (the last tick).
    pub tick: usize,
    /// The readout pre-activation = last tick's global-sync vector `[n_global_sync]`.
    pub pre: Vec<f32>,
    /// The full readout output `[out_dims]` (== last tick `prediction`).
    pub logits: Vec<f32>,
    /// Softmax over the first `n_moves` logits (the move distribution).
    pub move_softmax: Vec<f32>,
    /// Per-move, per-sync-channel contribution `W[move,i]·pre[i]` `[n_moves][n_global_sync]`.
    pub move_contributions: Vec<Vec<f32>>,
    /// Per-move readout bias `[n_moves]`.
    pub move_bias: Vec<f32>,
    /// Number of move rows surfaced (min(5, out_dims)).
    pub n_moves: usize,
}

/// Softmax helper, numerically stable (matches `compute_certainty`'s style).
fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 {
        return vec![0.0; logits.len()];
    }
    exps.into_iter().map(|e| e / sum).collect()
}

/// Build `DecisionDrivers` from the readout weights and the last tick.
fn decision_drivers(proj: &Linear, out: &BrainOut) -> Option<DecisionDrivers> {
    let last = out.ticks.last()?;
    let pre = last.global_sync.clone();
    let logits = last.prediction.clone();
    let n_in = proj.in_dim;
    let n_moves = proj.out_dim.min(5);
    let mut move_contributions = Vec::with_capacity(n_moves);
    let mut move_bias = Vec::with_capacity(n_moves);
    for j in 0..n_moves {
        let row = &proj.weight[j * n_in..(j + 1) * n_in];
        let mut contrib = vec![0.0f32; n_in];
        for i in 0..n_in.min(pre.len()) {
            contrib[i] = row[i] * pre[i];
        }
        move_contributions.push(contrib);
        move_bias.push(proj.bias.get(j).copied().unwrap_or(0.0));
    }
    let move_logits: Vec<f32> = logits.iter().take(n_moves).copied().collect();
    Some(DecisionDrivers {
        tick: out.ticks.len() - 1,
        pre,
        logits,
        move_softmax: softmax(&move_logits),
        move_contributions,
        move_bias,
        n_moves,
    })
}

/// Outer adaptive-compute summary: the exit-gate λ trajectory and the
/// derived survival / cumulative-exit / certainty traces, one entry per
/// outer tick actually run. Mirrors the exact recurrence in
/// `regional_forward`'s Phase 5 (`exit_cdf += λ·survival; survival *= 1−λ`).
#[cfg_attr(feature = "wasm", derive(serde::Serialize))]
#[derive(Debug, Clone)]
pub struct AdaptiveComputeSummary {
    /// Number of outer ticks the brain actually used.
    pub ticks_used: usize,
    /// Outer exit λ per tick (0.0 placeholder if no outer gate that tick).
    pub lambda_trajectory: Vec<f32>,
    /// Survival probability Π(1−λ) entering each tick.
    pub survival: Vec<f32>,
    /// Cumulative exit probability Σ λ·survival after each tick.
    pub exit_cdf: Vec<f32>,
    /// Readout certainty = 1 − normalized_entropy(prediction) per tick.
    pub certainty: Vec<f32>,
}

/// Recompute the outer adaptive-compute traces from a finished `BrainOut`.
/// Read-only: replays the same λ recurrence Phase 5 uses, so values match
/// the forward exactly without touching it.
pub fn adaptive_compute_summary(out: &BrainOut) -> AdaptiveComputeSummary {
    let n = out.ticks.len();
    let mut lambda_trajectory = Vec::with_capacity(n);
    let mut survival_trace = Vec::with_capacity(n);
    let mut cdf_trace = Vec::with_capacity(n);
    let mut certainty = Vec::with_capacity(n);
    let mut survival = 1.0f32;
    let mut exit_cdf = 0.0f32;
    for tick in &out.ticks {
        let lambda = tick.exit_lambda.unwrap_or(0.0);
        survival_trace.push(survival);
        exit_cdf += lambda * survival;
        survival *= 1.0 - lambda;
        lambda_trajectory.push(lambda);
        cdf_trace.push(exit_cdf);
        certainty.push(compute_certainty(&tick.prediction)[1]);
    }
    AdaptiveComputeSummary {
        ticks_used: out.ticks_used,
        lambda_trajectory,
        survival: survival_trace,
        exit_cdf: cdf_trace,
        certainty,
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  VIN — Value-Iteration-Network ego-centric readout (in-browser trainable).
//
//  This VIN runs DIRECTLY on the RAW maze pixels (the 9×9 grid), NOT on the
//  retina's downsampled V4 tokens. Each maze cell ↔ one grid cell, so the
//  agent and walls localise 1:1 (no blur, no 5×5→9×9 remap). Everything that
//  defines the maze geometry is READ FROM THE PIXELS — not learned:
//
//    • gate[cell]   = 0 (wall) | 1 (traversable)      — from black vs. not.
//    • reward[cell] = 1 (goal) | 0 (otherwise)        — from green.
//    • agent cell   = the red pixel                   — from red.
//
//  EXACT (tabular) value iteration then floods scalar value backward from the
//  goal through open corridors; walls block it. Because this is fixed DP (not
//  a learned settler), more iterations is strictly better until converged, so
//  we run K = SIZE·SIZE iterations — enough to guarantee full propagation on a
//  SIZE×SIZE grid. The result is a real goal-compass: value[cell] ≈
//  GAMMA^(shortest-path-distance-to-goal) along open cells, 0 inside walls.
//
//  Only the small ego-centric MOVE HEAD learns in-browser (momentum-free
//  three-factor rule, no backprop graph). With a correct value field it only
//  has to learn "pick the open neighbour with the highest value." The
//  value/gate/reward are correct by construction and never learn.
//
//  Pixel scheme (CHW `[3 × SIZE × SIZE]`, SIZE = round(sqrt(len/3))):
//  for cell i, R = px[i], G = px[N+i], B = px[2N+i], N = SIZE².
//    wall  = black  (0,0,0)   open = white (1,1,1)
//    agent = red    (1,0,0)   goal = green (0,1,0)
//
//  Independent of the brain forward — existing exports are byte-for-byte
//  unchanged.
// ═══════════════════════════════════════════════════════════════════════

/// Neighbour offsets in canonical direction order: Up, Down, Left, Right.
/// The move-head readout gathers neighbours in THIS order, and move-head
/// output index `d` (for d ∈ 0..4) corresponds to direction `d` here. Index 4
/// is WAIT/Stay. So the convention the worker reads/teaches is:
///   0 = Up (row−1), 1 = Down (row+1), 2 = Left (col−1), 3 = Right (col+1),
///   4 = Wait. argmax over the 5 logits = chosen move; teach the same index.
pub const VIN_DIR_OFFSETS: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

/// Number of move directions emitted (U/D/L/R + Wait = 5). Index 4 = Wait.
pub const VIN_N_DIRECTIONS: usize = 5;

/// Discount factor for value iteration. value[cell] ≈ GAMMA^(dist-to-goal).
const VIN_GAMMA: f32 = 0.9;
/// Very-negative sentinel for an off-grid / wall neighbour in the ego readout
/// (so the move head sees "you cannot/should not go there").
/// Superseded by the scale-stable advantage readout (kept for back-compat).
#[allow(dead_code)]
const VIN_BLOCKED_VALUE: f32 = -1.0;
/// Wall / off-grid neighbour feature in the scale-stable advantage readout.
const VIN_WALL_FEAT: f32 = -1.0;
/// Floor for the L∞ advantage scale (avoid divide-by-zero on flat values).
const VIN_NORM_EPS: f32 = 1e-3;
/// Constant bias channel value in the readout (open-gate marker).
const VIN_OPEN_GATE: f32 = 1.0;
/// Fixed sharpening temperature applied when forming move logits. Smaller =
/// sharper; logits = raw / VIN_TEMP. Stashed so forward/argmax/learn agree.
const VIN_TEMP: f32 = 0.5;
/// Deterministic seed for the VIN move-head init.
const VIN_SEED: u64 = 0x5144_4956_4e5f_3031; // "VIN_01"-ish, fixed.

/// SplitMix64 — deterministic PRNG for the seeded VIN init.
struct SplitMix64(u64);
impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform f32 in [-1, 1).
    fn next_pm1(&mut self) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32; // [0,1)
        2.0 * u - 1.0
    }
}

/// A deterministically-seeded `Linear` (the engine's `Linear` has no `new`).
/// Weights ~ U(-scale, scale) with `scale = 1/sqrt(in_dim)` (Xavier-ish);
/// biases zero. Same `(in_dim, out_dim, rng-stream)` ⇒ identical weights.
fn vin_linear(in_dim: usize, out_dim: usize, rng: &mut SplitMix64) -> Linear {
    let scale = 1.0 / (in_dim.max(1) as f32).sqrt();
    let mut weight = vec![0.0f32; out_dim * in_dim];
    for w in weight.iter_mut() {
        *w = rng.next_pm1() * scale;
    }
    Linear {
        weight,
        bias: vec![0.0f32; out_dim],
        in_dim,
        out_dim,
    }
}

/// Move-head input width: `[agent_value, agent_gate, 4×neighbour_value]`.
const VIN_READOUT_DIM: usize = 6;

/// Self-contained VIN readout. The value field is computed by EXACT value
/// iteration over per-cell wall/goal derived from the raw maze pixels — only
/// the small ego-centric move head is learned. `raw_dim` is kept for the
/// rebuild-on-width-change check at the binding layer (the move head itself
/// is pixel-width-independent).
#[derive(Clone)]
pub struct VinReadout {
    /// Per-image channel layout is fixed CHW (3 channels); `raw_dim` is the
    /// stored channel count so the binding can detect a layout change.
    pub raw_dim: usize,

    // ── LEARNABLE ego-centric move head ──
    // Consumes [agent_value | agent_gate | 4×neighbour_value] (VIN_READOUT_DIM)
    // and emits VIN_N_DIRECTIONS move logits (U,D,L,R,Wait).
    move_head: Linear,
}

/// Per-cell maze semantics derived directly from the raw pixels (no learning).
struct MazeGrid {
    size: usize,
    /// Traversability: 0.0 = wall, 1.0 = open/goal/agent cell.
    gate: Vec<f32>,
    /// Reward: 1.0 = goal cell, 0.0 otherwise.
    reward: Vec<f32>,
    /// Agent cell located from the red pixel, if any.
    agent: Option<(usize, usize)>,
}

impl MazeGrid {
    /// Decode CHW `[3 × size × size]` pixels into per-cell gate/reward/agent.
    /// R = px[i], G = px[N+i], B = px[2N+i], N = size².
    ///   wall  = R<0.5 && G<0.5 && B<0.5   → gate 0
    ///   goal  = G>0.5 && R<0.5            → reward 1
    ///   agent = R>0.5 && G<0.5 && B<0.5   → agent cell
    fn from_pixels(pixels: &[f32], size: usize) -> Self {
        let n = size * size;
        let mut gate = vec![1.0f32; n];
        let mut reward = vec![0.0f32; n];
        let mut agent = None;
        for cell in 0..n {
            let r = pixels[cell];
            let g = pixels[n + cell];
            let b = pixels[2 * n + cell];
            let is_wall = r < 0.5 && g < 0.5 && b < 0.5;
            let is_goal = g > 0.5 && r < 0.5;
            let is_agent = r > 0.5 && g < 0.5 && b < 0.5;
            gate[cell] = if is_wall { 0.0 } else { 1.0 };
            if is_goal {
                reward[cell] = 1.0;
            }
            if is_agent && agent.is_none() {
                agent = Some((cell / size, cell % size));
            }
        }
        Self { size, gate, reward, agent }
    }

    /// Exact value iteration on the SIZE×SIZE grid. value init 0; for K = N
    /// sweeps (guarantees full backward propagation):
    ///   value_new[cell] = is_wall ? 0
    ///                   : reward[cell] + GAMMA·max_{traversable nbr} value[nbr]
    /// Off-grid / wall neighbours contribute 0 (never raise the max above a
    /// reachable open neighbour). Floods value backward from the goal.
    fn value_iteration(&self) -> Vec<f32> {
        let n = self.size * self.size;
        let mut value = vec![0.0f32; n];
        let mut next = vec![0.0f32; n];
        let k = n; // SIZE² iterations — fixed DP, strictly better until converged.
        for _ in 0..k {
            for r in 0..self.size {
                for c in 0..self.size {
                    let cell = r * self.size + c;
                    if self.gate[cell] < 0.5 {
                        next[cell] = 0.0; // wall: no value.
                        continue;
                    }
                    let mut best_nbr = 0.0f32;
                    for (dr, dc) in VIN_DIR_OFFSETS {
                        let nr = r as i32 + dr;
                        let nc = c as i32 + dc;
                        if nr < 0 || nc < 0 || nr >= self.size as i32 || nc >= self.size as i32 {
                            continue;
                        }
                        let ncell = nr as usize * self.size + nc as usize;
                        if self.gate[ncell] < 0.5 {
                            continue; // wall neighbour is not traversable.
                        }
                        if value[ncell] > best_nbr {
                            best_nbr = value[ncell];
                        }
                    }
                    next[cell] = self.reward[cell] + VIN_GAMMA * best_nbr;
                }
            }
            std::mem::swap(&mut value, &mut next);
        }
        value
    }
}

impl VinReadout {
    /// Build with the fixed VIN seed. Only the move head is learnable.
    pub fn seeded(raw_dim: usize) -> Self {
        let mut rng = SplitMix64::new(VIN_SEED);
        let move_head = vin_linear(VIN_READOUT_DIM, VIN_N_DIRECTIONS, &mut rng);
        Self { raw_dim, move_head }
    }

    /// Forward over raw CHW `[3 × size × size]` pixels. Derives wall/goal/agent
    /// from the pixels, runs exact value iteration, then reads ego-centric move
    /// logits at the agent's cell. `agent_fallback` (TRUE maze coords) is used
    /// only if no red agent pixel is found. Returns the full [`VinOutput`].
    pub fn forward_pixels(
        &self,
        pixels: &[f32],
        size: usize,
        agent_fallback: (usize, usize),
    ) -> VinOutput {
        let maze = MazeGrid::from_pixels(pixels, size);
        let value = maze.value_iteration();

        // Agent cell: from the red pixel, else the passed fallback (clamped).
        let (ar, ac) = maze
            .agent
            .unwrap_or((agent_fallback.0.min(size - 1), agent_fallback.1.min(size - 1)));
        let (ar, ac) = (ar.min(size - 1), ac.min(size - 1));
        let acell = ar * size + ac;

        // ── Scale-stable ADVANTAGE readout (layout [agent_slot, gate, U,D,L,R]).
        // The move head's input is the per-neighbour advantage value[nbr]−value[acell]
        // normalised by the L∞ scale over OPEN advantages, so the inter-neighbour
        // gap stays distance-invariant (≈ constant magnitude near/far from goal)
        // instead of shrinking with the raw 0.9^dist values. Wall / off-grid
        // neighbours read VIN_WALL_FEAT. Neighbour order is U,D,L,R == move idx d.
        let av = value[acell];
        // pass 1: raw advantage for OPEN neighbours; track openness.
        let mut adv = [0.0f32; 4];
        let mut open = [false; 4];
        for (k, (dr, dc)) in VIN_DIR_OFFSETS.iter().enumerate() {
            let nr = ar as i32 + dr;
            let nc = ac as i32 + dc;
            if nr < 0 || nc < 0 || nr >= size as i32 || nc >= size as i32 {
                continue;
            }
            let ncell = nr as usize * size + nc as usize;
            if maze.gate[ncell] < 0.5 {
                continue;
            }
            adv[k] = value[ncell] - av;
            open[k] = true;
        }
        // pass 2: L∞ scale over OPEN advantages → distance-invariant gap.
        let mut scale = VIN_NORM_EPS;
        for k in 0..4 {
            if open[k] {
                scale = scale.max(adv[k].abs());
            }
        }
        let mut readout = Vec::with_capacity(VIN_READOUT_DIM);
        readout.push(0.0); // agent-value slot carries no scale info → constant 0.
        readout.push(VIN_OPEN_GATE); // bias channel.
        for k in 0..4 {
            readout.push(if open[k] { adv[k] / scale } else { VIN_WALL_FEAT });
        }

        // Fixed sharpening temperature; STASH the tempered logits so the worker's
        // argmax and `learn`'s softmax operate on the same policy.
        let raw = self.move_head.forward(&readout);
        let move_logits: Vec<f32> = raw.iter().map(|z| z / VIN_TEMP).collect();

        VinOutput {
            move_logits,
            pre: readout,
            agent_cell: (ar, ac),
            value_grid: value,
            gate: maze.gate,
        }
    }

    /// THREE-FACTOR local update to the move head ONLY. No backprop graph,
    /// momentum-free. Given the stashed agent-cell pre-activation `pre`
    /// (== last forward's readout) and `logits`:
    ///
    ///   eligibility[d,j] = (onehot(target)[d] − softmax(logits)[d]) · pre[j]
    ///   ΔW[d,j]          = θ · reward_signal · eligibility[d,j]
    ///
    /// where `reward_signal = teach − pain`: `teach = 1` when `target_move ≥ 0`
    /// (push the agent toward the taught move), and `pain` is the wall-hit
    /// penalty (subtracted, so a painful move is pushed DOWN). Bounded:
    /// `|ΔW| ≤ DW_CLAMP`, post-update `|W| ≤ W_CLAMP`, NaN/Inf guard.
    /// Returns ‖ΔW‖ actually applied.
    fn learn(&mut self, pre: &[f32], logits: &[f32], target_move: i32, pain: f32) -> f32 {
        const THETA: f32 = 0.1;
        const DW_CLAMP: f32 = 0.25;
        const W_CLAMP: f32 = 6.0;

        let n_dirs = self.move_head.out_dim;
        let n_in = self.move_head.in_dim;
        let probs = vin_softmax(&logits[..n_dirs.min(logits.len())]);

        // Blended reward signal: positive teaching toward target, minus pain.
        let teach = if target_move >= 0 { 1.0f32 } else { 0.0f32 };
        let reward_signal = teach - pain;

        let mut norm_sq = 0.0f32;
        for d in 0..n_dirs {
            let onehot = if target_move >= 0 && d as i32 == target_move {
                1.0
            } else {
                0.0
            };
            let post = probs.get(d).copied().unwrap_or(0.0);
            let err = onehot - post; // (target − prediction)
            let row = &mut self.move_head.weight[d * n_in..(d + 1) * n_in];
            for j in 0..n_in.min(pre.len()) {
                let mut dw = THETA * reward_signal * err * pre[j];
                if !dw.is_finite() {
                    continue;
                }
                dw = dw.clamp(-DW_CLAMP, DW_CLAMP);
                let updated = (row[j] + dw).clamp(-W_CLAMP, W_CLAMP);
                let applied = updated - row[j];
                row[j] = updated;
                norm_sq += applied * applied;
            }
        }
        norm_sq.sqrt()
    }
}

/// Output of one VIN forward pass.
pub struct VinOutput {
    /// Ego-centric move logits `[VIN_N_DIRECTIONS]` from the agent cell.
    pub move_logits: Vec<f32>,
    /// The exact ego-centric readout (move-head input) — stashed for `learn`.
    pub pre: Vec<f32>,
    /// The `(row, col)` agent cell used — TRUE maze coords (clamped in-bounds).
    pub agent_cell: (usize, usize),
    /// Per-cell scalar value `[SIZE²]` (row-major) from exact value iteration,
    /// ≈ GAMMA^(dist-to-goal) on open cells, 0 inside walls — the maze heat.
    pub value_grid: Vec<f32>,
    /// Per-cell traversability gate `[SIZE²]` (1.0 free, 0.0 wall).
    pub gate: Vec<f32>,
}

/// Numerically-stable softmax (shared by VIN forward/learn).
fn vin_softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 {
        return vec![0.0; logits.len()];
    }
    exps.into_iter().map(|e| e / sum).collect()
}

// ════════════════════════════════════════════════════════════════════════
//  LEARNED Value-Iteration-Network forward pass (ported VERBATIM from the
//  SDK `crates/modgrad-ctm/src/vin.rs`). This replaces the hardcoded
//  value-iteration "cheat" (`VinReadout` above) with the TRAINED planner:
//  every cell's reward / traversability gate / value features are LEARNED
//  projections of the per-cell token, and value propagates via the learned
//  highway-gated soft Bellman backup. Deserialized from the SDK's serde JSON
//  export of a `VinReadout` (field names match the SDK exactly). Forward only.
//
//  IMPORTANT: the f32 op order here mirrors the SDK so the forward is
//  bit-identical (verified by `learned_vin_golden_matches_sdk`).
// ════════════════════════════════════════════════════════════════════════

/// Neighbour offsets in canonical direction order: Up, Down, Left, Right.
const LEARNED_DIR_OFFSETS: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

#[inline]
fn learned_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Dense linear layer matching the SDK's serde shape. `in_dim`/`out_dim` are
/// present in the JSON but unused here (derived from `weight`/`bias`); serde
/// ignores them. `forward` is byte-for-byte the SDK's
/// `y[o] = bias[o] + Σ_i weight[o*in+i] * x[i]`.
#[derive(Debug, Clone, Deserialize)]
pub struct LearnedLinear {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

impl LearnedLinear {
    #[inline]
    fn out_dim(&self) -> usize {
        self.bias.len()
    }
    #[inline]
    fn in_dim(&self) -> usize {
        if self.bias.is_empty() {
            0
        } else {
            self.weight.len() / self.bias.len()
        }
    }
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let out_dim = self.out_dim();
        let in_dim = self.in_dim();
        let mut y = vec![0.0f32; out_dim];
        for o in 0..out_dim {
            let mut acc = self.bias[o];
            let row = o * in_dim;
            for i in 0..in_dim {
                acc += self.weight[row + i] * x[i];
            }
            y[o] = acc;
        }
        y
    }
}

/// VIN configuration. Field names match the SDK's `VinConfig` serde output.
#[derive(Debug, Clone, Deserialize)]
pub struct LearnedVinConfig {
    pub value_dim: usize,
    pub iters: usize,
    pub max_iters: usize,
    pub softmax_temp: f32,
    pub highway_gate: bool,
    #[allow(dead_code)]
    pub n_dirs: usize,
}

/// Learned VIN readout deserialized from the SDK's `VinReadout` JSON export.
/// Field names (and order) match the SDK serde output exactly:
/// `config, raw_dim, reward_proj, gate_proj, value_proj, agent_proj,
///  highway_proj, move_head`.
#[derive(Debug, Clone, Deserialize)]
pub struct LearnedVin {
    pub config: LearnedVinConfig,
    pub raw_dim: usize,
    pub reward_proj: LearnedLinear,
    pub gate_proj: LearnedLinear,
    pub value_proj: LearnedLinear,
    pub agent_proj: LearnedLinear,
    pub highway_proj: LearnedLinear,
    pub move_head: LearnedLinear,
}

impl LearnedVin {
    #[inline]
    fn effective_iters(&self) -> usize {
        self.config.iters.min(self.config.max_iters.max(1))
    }

    /// FORWARD ONLY — ported VERBATIM from SDK `VinReadout::forward`. Runs the
    /// learned value propagation over the grid and returns the `n_dirs` move
    /// logits read ego-centrically at `agent`. `tokens` is flat
    /// `[n_cells × raw_dim]`, row-major; `agent` is an explicit `(row, col)`.
    pub fn forward(
        &self,
        tokens: &[f32],
        grid_h: usize,
        grid_w: usize,
        agent: (usize, usize),
    ) -> Vec<f32> {
        let v = self.config.value_dim.max(1);
        let (value, value_init, gate) = self.run_value(tokens, grid_h, grid_w);
        let (ar, ac) = agent;
        let readout =
            self.gather_agent_readout(ar, ac, grid_h, grid_w, &value, &value_init, &gate, v);
        self.move_head.forward(&readout)
    }

    /// Like `forward`, but ALSO returns the planner's own per-cell scalar value
    /// map (`sum_k |value[cell,k]|` — the same magnitude the backup uses to rank
    /// neighbours). This is the model's OWN estimate of proximity-to-goal along
    /// feasible routes (value floods backward from the goal through open cells),
    /// so it can drive an honest "am I getting closer?" signal with NO solver in
    /// the loop. Returns `(move_logits, value_map[n_cells])`.
    pub fn forward_compass(
        &self,
        tokens: &[f32],
        grid_h: usize,
        grid_w: usize,
        agent: (usize, usize),
    ) -> (Vec<f32>, Vec<f32>) {
        let n_cells = grid_h * grid_w;
        let v = self.config.value_dim.max(1);
        let (value, value_init, gate) = self.run_value(tokens, grid_h, grid_w);

        // per-cell scalar = L1 magnitude of the value vector (the planner's
        // flooded "closeness" field; higher = closer to the goal by route).
        let mut vmap = vec![0.0f32; n_cells];
        for cell in 0..n_cells {
            vmap[cell] = value[cell * v..(cell + 1) * v]
                .iter()
                .map(|x| x.abs())
                .sum::<f32>();
        }

        let (ar, ac) = agent;
        let readout =
            self.gather_agent_readout(ar, ac, grid_h, grid_w, &value, &value_init, &gate, v);
        (self.move_head.forward(&readout), vmap)
    }

    /// Shared value propagation: per-cell reward/gate/value projections, then K
    /// capped Bellman backups (with the optional Highway gate). Returns the final
    /// `value` field, the pre-iteration `value_init`, and the per-cell `gate`.
    fn run_value(
        &self,
        tokens: &[f32],
        grid_h: usize,
        grid_w: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let n_cells = grid_h * grid_w;
        let v = self.config.value_dim.max(1);

        // ── 1. Per-cell projections: reward, traversability gate, value ──
        let mut reward = vec![0.0f32; n_cells];
        let mut gate = vec![0.0f32; n_cells];
        let mut value = vec![0.0f32; n_cells * v];

        for cell in 0..n_cells {
            let tok = &tokens[cell * self.raw_dim..(cell + 1) * self.raw_dim];
            reward[cell] = self.reward_proj.forward(tok)[0];
            gate[cell] = learned_sigmoid(self.gate_proj.forward(tok)[0]);
            let vproj = self.value_proj.forward(tok);
            value[cell * v..(cell + 1) * v].copy_from_slice(&vproj);
        }

        let value_init = value.clone();

        // ── 2. K capped Bellman backups over the 4-neighbour grid ────────
        let iters = self.effective_iters();
        let mut next = value.clone();
        for _ in 0..iters {
            for r in 0..grid_h {
                for c in 0..grid_w {
                    let cell = r * grid_w + c;
                    let cand = self.backup_cell(
                        r, c, grid_h, grid_w, &value, &gate, reward[cell], v,
                    );
                    let dst = &mut next[cell * v..(cell + 1) * v];
                    if self.config.highway_gate {
                        let prev = &value[cell * v..(cell + 1) * v];
                        let mut hin = Vec::with_capacity(2 * v);
                        hin.extend_from_slice(prev);
                        hin.extend_from_slice(&cand);
                        let g = self.highway_proj.forward(&hin);
                        for k in 0..v {
                            let gk = learned_sigmoid(g[k]);
                            dst[k] = gk * cand[k] + (1.0 - gk) * prev[k];
                        }
                    } else {
                        dst.copy_from_slice(&cand);
                    }
                }
            }
            std::mem::swap(&mut value, &mut next);
        }

        (value, value_init, gate)
    }

    /// DREAM CONSOLIDATION — one supervised SGD step on the move head (the
    /// durable "cortex" readout) toward `target_move`, using cross-entropy on the
    /// move logits. The value-iteration core stays frozen; only the readout that
    /// turns the planner's value field into a move is fine-tuned. This is what a
    /// sleep/replay pass calls per remembered maze (target = its BFS-optimal move,
    /// the sanctioned "dream" answer). Mutates `self.move_head`; returns the
    /// cross-entropy loss BEFORE the update (so callers can show it dropping).
    pub fn consolidate_move(
        &mut self,
        tokens: &[f32],
        grid_h: usize,
        grid_w: usize,
        agent: (usize, usize),
        target_move: usize,
        lr: f32,
    ) -> f32 {
        let v = self.config.value_dim.max(1);
        let (value, value_init, gate) = self.run_value(tokens, grid_h, grid_w);
        let (ar, ac) = agent;
        let readout =
            self.gather_agent_readout(ar, ac, grid_h, grid_w, &value, &value_init, &gate, v);
        let logits = self.move_head.forward(&readout);
        let n = logits.len();
        if n == 0 || target_move >= n {
            return 0.0;
        }
        // softmax over the move logits
        let maxl = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut probs = vec![0.0f32; n];
        let mut sum = 0.0f32;
        for o in 0..n {
            let e = (logits[o] - maxl).exp();
            probs[o] = e;
            sum += e;
        }
        let inv = 1.0 / sum.max(1e-9);
        for p in probs.iter_mut() {
            *p *= inv;
        }
        let loss = -(probs[target_move].max(1e-9)).ln();

        // gradient of cross-entropy wrt logits: p - onehot(target). NORMALISED
        // (NLMS-style) step: divide by ‖readout‖² so the update is scale-invariant
        // — the value-iteration readout can be large, and a plain SGD step blows
        // up. This keeps each consolidation bounded and the loss DROPS.
        let in_dim = readout.len();
        let mut rn2 = 0.0f32;
        for &x in &readout {
            rn2 += x * x;
        }
        let scale = lr / (rn2 + 1.0);
        for o in 0..n {
            let g = probs[o] - if o == target_move { 1.0 } else { 0.0 };
            let row = o * in_dim;
            for i in 0..in_dim {
                self.move_head.weight[row + i] -= scale * g * readout[i];
            }
            self.move_head.bias[o] -= lr * 0.1 * g; // bias: small fixed step
        }
        loss
    }

    /// One cell's Bellman backup — ported VERBATIM from SDK.
    #[allow(clippy::too_many_arguments)]
    fn backup_cell(
        &self,
        r: usize,
        c: usize,
        grid_h: usize,
        grid_w: usize,
        value: &[f32],
        gate: &[f32],
        local_reward: f32,
        v: usize,
    ) -> Vec<f32> {
        let mut nbr_idx: Vec<usize> = Vec::with_capacity(4);
        let mut nbr_pref: Vec<f32> = Vec::with_capacity(4);
        for (dr, dc) in LEARNED_DIR_OFFSETS {
            let nr = r as i32 + dr;
            let nc = c as i32 + dc;
            if nr < 0 || nc < 0 || nr >= grid_h as i32 || nc >= grid_w as i32 {
                continue;
            }
            let ncell = nr as usize * grid_w + nc as usize;
            let g = gate[ncell];
            let mag: f32 = value[ncell * v..(ncell + 1) * v]
                .iter()
                .map(|x| x.abs())
                .sum::<f32>();
            nbr_idx.push(ncell);
            nbr_pref.push(g * mag);
        }

        let mut out = vec![0.0f32; v];
        if nbr_idx.is_empty() {
            for k in 0..v {
                out[k] = local_reward;
            }
            return out;
        }

        if self.config.softmax_temp <= 0.0 {
            let mut best = 0usize;
            let mut best_p = f32::NEG_INFINITY;
            for (i, &p) in nbr_pref.iter().enumerate() {
                if p > best_p {
                    best_p = p;
                    best = i;
                }
            }
            let ncell = nbr_idx[best];
            let g = gate[ncell];
            for k in 0..v {
                out[k] = local_reward + g * value[ncell * v + k];
            }
        } else {
            let t = self.config.softmax_temp;
            let maxp = nbr_pref.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut wsum = 0.0f32;
            let mut w = vec![0.0f32; nbr_idx.len()];
            for (i, &p) in nbr_pref.iter().enumerate() {
                let e = ((p - maxp) / t).exp();
                w[i] = e;
                wsum += e;
            }
            let inv = 1.0 / wsum.max(1e-8);
            for k in 0..v {
                let mut acc = 0.0f32;
                for (i, &ncell) in nbr_idx.iter().enumerate() {
                    let g = gate[ncell];
                    acc += w[i] * inv * g * value[ncell * v + k];
                }
                out[k] = local_reward + acc;
            }
        }
        out
    }

    /// Gather the agent-cell ego-centric readout — ported VERBATIM from SDK.
    /// Layout: `[value_init[acell] | value[acell] | 4 nbrs gate*value]`,
    /// zero block for off-grid neighbours. Length == `value_dim * 6`.
    #[allow(clippy::too_many_arguments)]
    fn gather_agent_readout(
        &self,
        ar: usize,
        ac: usize,
        grid_h: usize,
        grid_w: usize,
        value: &[f32],
        value_init: &[f32],
        gate: &[f32],
        v: usize,
    ) -> Vec<f32> {
        let mut out = Vec::with_capacity(v * 6);
        let acell = ar * grid_w + ac;
        out.extend_from_slice(&value_init[acell * v..(acell + 1) * v]);
        out.extend_from_slice(&value[acell * v..(acell + 1) * v]);
        for (dr, dc) in LEARNED_DIR_OFFSETS {
            let nr = ar as i32 + dr;
            let nc = ac as i32 + dc;
            if nr < 0 || nc < 0 || nr >= grid_h as i32 || nc >= grid_w as i32 {
                out.extend(std::iter::repeat(0.0f32).take(v));
                continue;
            }
            let ncell = nr as usize * grid_w + nc as usize;
            let g = gate[ncell];
            for k in 0..v {
                out.push(g * value[ncell * v + k]);
            }
        }
        out
    }
}

#[cfg(test)]
mod vin_tests {
    use super::*;

    /// Build CHW `[3 × size × size]` pixels from a textual maze.
    /// '#'=wall(black) '.'=open(white) 'A'=agent(red) 'G'=goal(green).
    fn maze_pixels(rows: &[&str]) -> (Vec<f32>, usize) {
        let size = rows.len();
        let n = size * size;
        let mut px = vec![0.0f32; 3 * n];
        for (r, row) in rows.iter().enumerate() {
            assert_eq!(row.chars().count(), size, "maze must be square");
            for (c, ch) in row.chars().enumerate() {
                let cell = r * size + c;
                let (rr, gg, bb) = match ch {
                    '#' => (0.0, 0.0, 0.0),
                    '.' => (1.0, 1.0, 1.0),
                    'A' => (1.0, 0.0, 0.0),
                    'G' => (0.0, 1.0, 0.0),
                    _ => panic!("bad maze char {ch}"),
                };
                px[cell] = rr;
                px[n + cell] = gg;
                px[2 * n + cell] = bb;
            }
        }
        (px, size)
    }

    #[test]
    fn seeded_is_deterministic() {
        let a = VinReadout::seeded(3);
        let b = VinReadout::seeded(3);
        assert_eq!(a.move_head.weight, b.move_head.weight);
    }

    #[test]
    fn forward_shapes() {
        let vin = VinReadout::seeded(3);
        let (px, size) = maze_pixels(&["A..", "...", "..G"]);
        let out = vin.forward_pixels(&px, size, (0, 0));
        assert_eq!(out.move_logits.len(), VIN_N_DIRECTIONS);
        assert_eq!(out.pre.len(), VIN_READOUT_DIM);
        assert_eq!(out.value_grid.len(), size * size);
        assert_eq!(out.gate.len(), size * size);
        assert!(out.value_grid.iter().all(|x| x.is_finite()));
        assert!(out.gate.iter().all(|g| *g == 0.0 || *g == 1.0));
    }

    #[test]
    fn agent_located_from_red_pixel() {
        let vin = VinReadout::seeded(3);
        // Agent at (1,2); fallback deliberately wrong.
        let (px, size) = maze_pixels(&["..G", "..A", "..."]);
        let out = vin.forward_pixels(&px, size, (0, 0));
        assert_eq!(out.agent_cell, (1, 2));
    }

    #[test]
    fn value_higher_nearer_goal() {
        // Open 3×3, goal at bottom-right (2,2).
        let (px, size) = maze_pixels(&["...", "...", "..G"]);
        let maze = MazeGrid::from_pixels(&px, size);
        let v = maze.value_iteration();
        let at = |r: usize, c: usize| v[r * size + c];
        // Goal cell highest; value strictly decreases with distance.
        assert!(at(2, 2) > at(1, 2));
        assert!(at(1, 2) > at(0, 2));
        assert!(at(2, 2) > at(2, 1));
        assert!(at(0, 0) < at(1, 1)); // farther corner < nearer centre.
        // Goal value ≈ 1/(1-GAMMA) bound; immediate value at least reward.
        assert!(at(2, 2) >= 1.0);
    }

    #[test]
    fn walls_block_value_propagation() {
        // A corridor maze where a wall row forces the long way around.
        //  A . #
        //  # . #
        //  # . G
        // Cell (0,2) is walled off from the goal by '#'s; its value must be
        // the around-the-corridor value, and the wall cells must be 0.
        let (px, size) = maze_pixels(&["A.#", "#.#", "#.G"]);
        let maze = MazeGrid::from_pixels(&px, size);
        let v = maze.value_iteration();
        // Walls carry zero value.
        assert_eq!(v[0 * size + 2], 0.0); // (0,2) is '#'
        assert_eq!(v[1 * size + 0], 0.0); // (1,0) is '#'
        // The open corridor column 1 floods from the goal.
        assert!(v[2 * size + 1] > 0.0);
        assert!(v[1 * size + 1] > 0.0);
        assert!(v[0 * size + 1] > 0.0);
    }

    #[test]
    fn greedy_max_value_neighbour_points_along_shortest_path() {
        // Straight corridor: agent top, goal bottom of column 1.
        let (px, size) = maze_pixels(&["#A#", "#.#", "#G#"]);
        let maze = MazeGrid::from_pixels(&px, size);
        let v = maze.value_iteration();
        let (ar, ac) = maze.agent.expect("agent");
        assert_eq!((ar, ac), (0, 1));
        // Among traversable neighbours, the max-value one is DOWN (toward goal).
        let mut best_dir = usize::MAX;
        let mut best_val = f32::NEG_INFINITY;
        for (d, (dr, dc)) in VIN_DIR_OFFSETS.iter().enumerate() {
            let nr = ar as i32 + dr;
            let nc = ac as i32 + dc;
            if nr < 0 || nc < 0 || nr >= size as i32 || nc >= size as i32 {
                continue;
            }
            let ncell = nr as usize * size + nc as usize;
            if maze.gate[ncell] < 0.5 {
                continue;
            }
            if v[ncell] > best_val {
                best_val = v[ncell];
                best_dir = d;
            }
        }
        assert_eq!(best_dir, 1, "shortest-path move should be Down (index 1)");
    }

    #[test]
    fn learn_zero_signal_is_noop_when_teach_cancels_pain() {
        // target=-1 (no teach) and pain=0 ⇒ reward_signal=0 ⇒ ΔW=0.
        let mut vin = VinReadout::seeded(8);
        let before = vin.move_head.weight.clone();
        let pre = vec![0.3f32; vin.move_head.in_dim];
        let logits = vec![0.1f32; VIN_N_DIRECTIONS];
        let norm = vin.learn(&pre, &logits, -1, 0.0);
        assert_eq!(norm, 0.0);
        assert_eq!(before, vin.move_head.weight);
    }

    #[test]
    fn learn_moves_weights_toward_target() {
        let mut vin = VinReadout::seeded(8);
        let before = vin.move_head.weight.clone();
        let pre = vec![0.5f32; vin.move_head.in_dim];
        let logits = vec![0.0f32; VIN_N_DIRECTIONS];
        let norm = vin.learn(&pre, &logits, 2, 0.0);
        assert!(norm > 0.0);
        assert_ne!(before, vin.move_head.weight);
        assert!(vin.move_head.weight.iter().all(|w| w.is_finite()));
    }

    /// Deterministic LCG (Numerical Recipes constants), no extra deps.
    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Self {
            Lcg(seed)
        }
        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }
        fn below(&mut self, n: u32) -> u32 {
            self.next_u32() % n
        }
        fn chance(&mut self, num: u32, den: u32) -> bool {
            self.below(den) < num
        }
    }

    /// Build a random SOLVABLE maze of the given size with a single agent (red)
    /// and a single goal (green), guaranteeing a path agent→goal exists. Returns
    /// the CHW pixels. Walls are sprinkled but never on the agent/goal cells, and
    /// a flood-fill check ensures reachability (retries until solvable).
    fn random_solvable_maze(size: usize, rng: &mut Lcg) -> (Vec<f32>, usize) {
        loop {
            let n = size * size;
            let mut wall = vec![false; n];
            for w in wall.iter_mut() {
                *w = rng.chance(1, 4); // ~25% walls.
            }
            let acell = rng.below(n as u32) as usize;
            let mut gcell = rng.below(n as u32) as usize;
            if gcell == acell {
                gcell = (gcell + 1) % n;
            }
            wall[acell] = false;
            wall[gcell] = false;
            // BFS reachability agent→goal.
            let mut seen = vec![false; n];
            let mut stack = vec![acell];
            seen[acell] = true;
            while let Some(cell) = stack.pop() {
                let r = cell / size;
                let c = cell % size;
                for (dr, dc) in VIN_DIR_OFFSETS {
                    let nr = r as i32 + dr;
                    let nc = c as i32 + dc;
                    if nr < 0 || nc < 0 || nr >= size as i32 || nc >= size as i32 {
                        continue;
                    }
                    let ncell = nr as usize * size + nc as usize;
                    if !seen[ncell] && !wall[ncell] {
                        seen[ncell] = true;
                        stack.push(ncell);
                    }
                }
            }
            if !seen[gcell] {
                continue; // unsolvable — retry.
            }
            // Render pixels.
            let mut px = vec![0.0f32; 3 * n];
            for cell in 0..n {
                let (rr, gg, bb) = if cell == acell {
                    (1.0, 0.0, 0.0)
                } else if cell == gcell {
                    (0.0, 1.0, 0.0)
                } else if wall[cell] {
                    (0.0, 0.0, 0.0)
                } else {
                    (1.0, 1.0, 1.0)
                };
                px[cell] = rr;
                px[n + cell] = gg;
                px[2 * n + cell] = bb;
            }
            return (px, size);
        }
    }

    /// The value-greedy OPTIMAL move at the agent cell: argmax over OPEN
    /// neighbours of the value grid. Returns None if the agent has no open
    /// neighbour (shouldn't happen on a solvable maze with the agent != goal).
    fn optimal_move(out: &VinOutput, size: usize) -> Option<i32> {
        let (ar, ac) = out.agent_cell;
        let mut best_dir: Option<i32> = None;
        let mut best_val = f32::NEG_INFINITY;
        for (d, (dr, dc)) in VIN_DIR_OFFSETS.iter().enumerate() {
            let nr = ar as i32 + dr;
            let nc = ac as i32 + dc;
            if nr < 0 || nc < 0 || nr >= size as i32 || nc >= size as i32 {
                continue;
            }
            let ncell = nr as usize * size + nc as usize;
            if out.gate[ncell] < 0.5 {
                continue;
            }
            let v = out.value_grid[ncell];
            if v > best_val {
                best_val = v;
                best_dir = Some(d as i32);
            }
        }
        best_dir
    }

    /// CHANGE-1 regression: with the scale-stable advantage readout + tempered
    /// logits + θ=0.1, the move head's argmax must LEARN the value-greedy optimal
    /// move online via `learn` (pain=0), climbing from chance to ≥85% agreement.
    #[test]
    fn move_head_agreement_climbs_to_85pct() {
        let size = 9;
        let window = 200usize;

        // Argmax of the (tempered) move logits = the head's chosen move.
        let argmax = |logits: &[f32]| -> i32 {
            logits
                .iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                    if v > bv {
                        (i, v)
                    } else {
                        (bi, bv)
                    }
                })
                .0 as i32
        };

        // Measure argmax-vs-optimal agreement over `n` fresh solvable mazes,
        // WITHOUT learning (frozen head). The maze stream is deterministic.
        let measure = |vin: &VinReadout, rng: &mut Lcg, n: usize| -> f32 {
            let mut correct = 0usize;
            let mut count = 0usize;
            for _ in 0..n {
                let (px, sz) = random_solvable_maze(size, rng);
                let out = vin.forward_pixels(&px, sz, (0, 0));
                let target = match optimal_move(&out, sz) {
                    Some(t) => t,
                    None => continue,
                };
                count += 1;
                if argmax(&out.move_logits) == target {
                    correct += 1;
                }
            }
            correct as f32 / count.max(1) as f32
        };

        let mut vin = VinReadout::seeded(3);
        // Start from a BLANK slate so agreement begins at chance and must be
        // LEARNED — the lucky seed already aligns with the advantage features.
        for w in vin.move_head.weight.iter_mut() {
            *w = 0.0;
        }
        for b in vin.move_head.bias.iter_mut() {
            *b = 0.0;
        }

        // ── PHASE 1: baseline agreement on the untrained head (frozen). ──
        let mut rng = Lcg::new(0xC0FFEE_1234_5678);
        let first = measure(&vin, &mut rng, window);

        // ── PHASE 2: online training toward the value-greedy optimal. ──
        // Forward returns TEMPERED logits in out.move_logits — the SAME vector
        // fed to learn(), so forward/argmax/learn all agree (pain = 0).
        for _ in 0..4000 {
            let (px, sz) = random_solvable_maze(size, &mut rng);
            let out = vin.forward_pixels(&px, sz, (0, 0));
            let target = match optimal_move(&out, sz) {
                Some(t) => t,
                None => continue,
            };
            vin.learn(&out.pre, &out.move_logits, target, 0.0);
        }

        // ── PHASE 3: post-training agreement on the trained head (frozen). ──
        let last = measure(&vin, &mut rng, window);

        assert!(
            first < 0.55,
            "early agreement should start near chance, got {first:.3}"
        );
        assert!(
            last >= 0.85,
            "late agreement should reach ≥0.85, got {last:.3}"
        );
        assert!(
            last - first > 0.30,
            "agreement should improve by >0.30 (first {first:.3} → last {last:.3})"
        );
    }

    /// GOLDEN-VECTOR BIT-EXACT GATE. The JSON + tokens + move_logits below are
    /// produced by the SDK's `vin::tests::print_golden_vector`
    /// (`crates/modgrad-ctm/src/vin.rs`): a deterministic `VinReadout`
    /// (value_dim=4, iters=5, temp=0.5, highway=true, n_dirs=4, raw_dim=3) on a
    /// 3×3 grid, agent (1,1), with `weight[i]=((i*2654435761)%1000)/1000-0.5`
    /// (bias likewise) and `tokens[i]=((i*40503)%100)/100`. We deserialize the
    /// SAME JSON into `LearnedVin`, run the ported forward, and assert each move
    /// logit matches the SDK golden to <1e-5. This proves the web port
    /// reproduces the SDK learned forward exactly.
    #[test]
    fn learned_vin_golden_matches_sdk() {
        const GOLDEN_JSON: &str = r#"{"config":{"value_dim":4,"iters":5,"max_iters":20,"softmax_temp":0.5,"highway_gate":true,"n_dirs":4},"raw_dim":3,"reward_proj":{"weight":[-0.5,0.26099998,0.022000015],"bias":[-0.5],"in_dim":3,"out_dim":1},"gate_proj":{"weight":[-0.5,0.26099998,0.022000015],"bias":[-0.5],"in_dim":3,"out_dim":1},"value_proj":{"weight":[-0.5,0.26099998,0.022000015,-0.21700001,-0.456,0.305,0.065999985,-0.17300001,-0.412,0.34899998,0.110000014,-0.12900001],"bias":[-0.5,0.26099998,0.022000015,-0.21700001],"in_dim":3,"out_dim":4},"agent_proj":{"weight":[-0.5,0.26099998,0.022000015],"bias":[-0.5],"in_dim":3,"out_dim":1},"highway_proj":{"weight":[-0.5,0.26099998,0.022000015,-0.21700001,-0.456,0.305,0.065999985,-0.17300001,-0.412,0.34899998,0.110000014,-0.12900001,-0.368,0.393,0.15399998,-0.08500001,-0.324,0.43699998,0.19800001,-0.04100001,-0.28,0.481,0.24199998,0.003000021,-0.236,-0.475,0.286,0.04699999,-0.192,-0.431,0.32999998,0.09100002],"bias":[-0.5,0.26099998,0.022000015,-0.21700001],"in_dim":8,"out_dim":4},"move_head":{"weight":[-0.5,0.26099998,0.022000015,-0.21700001,-0.456,0.305,0.065999985,-0.17300001,-0.412,0.34899998,0.110000014,-0.12900001,-0.368,0.393,0.15399998,-0.08500001,-0.324,0.43699998,0.19800001,-0.04100001,-0.28,0.481,0.24199998,0.003000021,-0.236,-0.475,0.286,0.04699999,-0.192,-0.431,0.32999998,0.09100002,-0.148,-0.387,0.374,0.13499999,-0.104,-0.343,0.41799998,0.17900002,-0.060000002,-0.299,0.462,0.22299999,-0.016000003,-0.255,-0.494,0.26700002,0.027999997,-0.211,-0.45,0.311,0.07200003,-0.167,-0.40600002,0.35500002,0.116,-0.122999996,-0.362,0.399,0.16000003,-0.078999996,-0.31800002,0.44300002,0.204,-0.034999996,-0.274,0.487,0.24800003,0.009000003,-0.22999999,-0.469,0.292,0.052999973,-0.18599999,-0.425,0.33600003,0.097,-0.14199999,-0.38099998,0.38,0.14099997,-0.09799999,-0.337,0.42400002,0.185,-0.05399999,-0.29299998,0.468,0.22899997,-0.00999999,-0.24900001,-0.488,0.273,0.03399998,-0.20500001],"bias":[-0.5,0.26099998,0.022000015,-0.21700001],"in_dim":24,"out_dim":4}}"#;

        // tokens[i] = ((i*40503)%100)/100, 3x3 grid × raw_dim 3 = 27 elems.
        let tokens: Vec<f32> = (0..27)
            .map(|i| ((i as u64).wrapping_mul(40503) % 100) as f32 / 100.0)
            .collect();

        // SDK golden move_logits for grid 3×3, agent (1,1).
        const GOLDEN_LOGITS: [f32; 4] = [-0.10739076, 0.49520034, 0.04578919, -0.38186702];

        let vin: LearnedVin = serde_json::from_str(GOLDEN_JSON).expect("deserialize LearnedVin");
        let logits = vin.forward(&tokens, 3, 3, (1, 1));
        assert_eq!(logits.len(), 4);

        let mut max_abs_diff = 0.0f32;
        for (i, (&got, &want)) in logits.iter().zip(GOLDEN_LOGITS.iter()).enumerate() {
            let d = (got - want).abs();
            if d > max_abs_diff {
                max_abs_diff = d;
            }
            assert!(
                d < 1e-5,
                "move_logit[{i}] web={got} sdk={want} diff={d:e} exceeds 1e-5"
            );
        }
        println!("learned_vin golden: max abs diff = {max_abs_diff:e}");
    }
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

        // ── Live-plasticity state ────────────────────────────────────────
        // Last forward's readout pre-activation (== last tick global_sync)
        // and logits (== last tick prediction). Stashed for free at the end
        // of every run so `apply_plasticity` can use the three-factor rule
        // WITHOUT re-running the forward.
        static LAST_PRE: RefCell<Option<Vec<f32>>> = const { RefCell::new(None) };
        static LAST_LOGITS: RefCell<Option<Vec<f32>>> = const { RefCell::new(None) };
        // Pristine snapshot of output_proj.{weight,bias}, taken lazily the
        // first time plasticity touches the readout, so `reset_plasticity`
        // can restore the as-loaded readout exactly.
        static PRISTINE_READOUT: RefCell<Option<(Vec<f32>, Vec<f32>)>> =
            const { RefCell::new(None) };

        // ── VIN (Value-Iteration-Network) readout state ──────────────────
        // Lazily built for the cortex's V4 channel width on the first
        // `vin_forward`. Its move head learns in-browser via `vin_learn`;
        // `vin_reset` re-seeds it. The last forward's agent-cell pre-
        // activation and move logits are stashed so `vin_learn` applies the
        // three-factor rule WITHOUT re-running the forward.
        static VIN: RefCell<Option<VinReadout>> = const { RefCell::new(None) };
        static VIN_LAST_PRE: RefCell<Option<Vec<f32>>> = const { RefCell::new(None) };
        static VIN_LAST_LOGITS: RefCell<Option<Vec<f32>>> = const { RefCell::new(None) };

        // ── LEARNED VIN (trained planner) ────────────────────────────────
        // The bit-exact port of the SDK `VinReadout::forward`, deserialized
        // from the SDK's serde JSON export via `load_learned_vin` and run by
        // `learned_vin_forward`. Distinct from the hardcoded `VIN` above.
        static LEARNED_VIN: RefCell<Option<super::LearnedVin>> = const { RefCell::new(None) };

        // Pristine (as-loaded) move-head weights, snapshotted lazily before the
        // first sleep consolidation so `learned_vin_reset` can restore the
        // planner to its trained-offline state — undoing every dream pass.
        static LEARNED_VIN_PRISTINE: RefCell<Option<(Vec<f32>, Vec<f32>)>> =
            const { RefCell::new(None) };
    }

    /// Stash the readout pre/logits from a finished forward for plasticity.
    /// Free (one clone of the last tick's two small vectors) and never
    /// touches the forward numerics — it only reads `out`.
    fn stash_plasticity_inputs(out: &BrainOut) {
        if let Some(last) = out.ticks.last() {
            LAST_PRE.with(|p| *p.borrow_mut() = Some(last.global_sync.clone()));
            LAST_LOGITS.with(|l| *l.borrow_mut() = Some(last.prediction.clone()));
        }
    }

    /// Parse `brain_weights.json` and stash the regional weights (and the
    /// retina, if present) for subsequent `run_brain`s.
    ///
    /// `cortex` is OPTIONAL: the visual-retina brain ships `{ cortex, regional }`,
    /// while the flat-encoding brain ships `{ regional }` with no retina at all.
    /// When `cortex` is absent, CORTEX stays `None` and only `run_brain_obs`
    /// (which takes a flat observation directly) is usable.
    #[wasm_bindgen]
    pub fn load_brain_weights(json: &str) -> Result<(), JsValue> {
        #[derive(Deserialize)]
        struct BrainWeights {
            #[serde(default)]
            cortex: Option<VisualCortex>,
            regional: RegionalWeights,
        }
        let bw: BrainWeights = serde_json::from_str(json)
            .map_err(|e| JsValue::from_str(&format!("load_brain_weights: {e}")))?;
        CORTEX.with(|c| *c.borrow_mut() = bw.cortex);
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
                .ok_or_else(|| JsValue::from_str(
                    "run_brain_pixels: no visual cortex loaded (this brain has no retina — use run_brain_obs with a flat observation)"))?;
            let (obs, _n, _d) = cortex.spatial_tokens(pixels);
            BRAIN.with(|b| {
                let brain_ref = b.borrow();
                let brain = brain_ref
                    .as_ref()
                    .ok_or_else(|| JsValue::from_str("run_brain: weights not loaded"))?;
                let out = regional_forward(brain, &obs);
                stash_plasticity_inputs(&out);
                serde_wasm_bindgen::to_value(&out)
                    .map_err(|e| JsValue::from_str(&format!("run_brain serialize: {e}")))
            })
        })
    }

    /// Retina feature maps for the given pixels: retina → v1 → v2 → v4
    /// activations (CHW), for the vision panel. Returns `[{name, channels,
    /// h, w, data}]`. Requires `load_brain_weights` first.
    #[wasm_bindgen]
    pub fn retina_maps(pixels: &[f32]) -> Result<JsValue, JsValue> {
        #[derive(serde::Serialize)]
        struct Map {
            name: String,
            channels: usize,
            h: usize,
            w: usize,
            data: Vec<f32>,
        }
        CORTEX.with(|c| {
            let cortex_ref = c.borrow();
            let cortex = cortex_ref
                .as_ref()
                .ok_or_else(|| JsValue::from_str("retina_maps: no visual cortex loaded"))?;
            let maps: Vec<Map> = cortex
                .feature_maps(pixels)
                .into_iter()
                .map(|(name, data, channels, h, w)| Map {
                    name: name.to_string(),
                    channels,
                    h,
                    w,
                    data,
                })
                .collect();
            serde_wasm_bindgen::to_value(&maps)
                .map_err(|e| JsValue::from_str(&format!("retina_maps serialize: {e}")))
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
            stash_plasticity_inputs(&out);
            serde_wasm_bindgen::to_value(&out)
                .map_err(|e| JsValue::from_str(&format!("run_brain serialize: {e}")))
        })
    }

    // ── Task 5: per-region telemetry ────────────────────────────────────

    /// Per-tick, per-region telemetry derived from a fresh forward over raw
    /// RGB pixels `[3 × H × W]` CHW. Returns `[TickTelemetry]`: for each
    /// outer tick, an array of per-region `{region, name, d_model,
    /// activation_rms, activation_peak, activation_mean}` plus
    /// `{global_sync_rms, global_sync_peak, exit_lambda}`.
    ///
    /// All values are computed from the forward's own outputs; no
    /// neuromodulator/homeostasis state exists in this reimplementation, so
    /// none is invented. Re-runs the forward (does not stash plasticity
    /// inputs — use `run_brain_pixels` if you also want to plasticise).
    #[wasm_bindgen]
    pub fn brain_telemetry_pixels(pixels: &[f32]) -> Result<JsValue, JsValue> {
        CORTEX.with(|c| {
            let cortex_ref = c.borrow();
            let cortex = cortex_ref
                .as_ref()
                .ok_or_else(|| JsValue::from_str("brain_telemetry_pixels: no visual cortex loaded"))?;
            let (obs, _n, _d) = cortex.spatial_tokens(pixels);
            BRAIN.with(|b| {
                let brain_ref = b.borrow();
                let brain = brain_ref
                    .as_ref()
                    .ok_or_else(|| JsValue::from_str("brain_telemetry: weights not loaded"))?;
                let out = regional_forward(brain, &obs);
                let telem = super::region_telemetry(brain, &out);
                serde_wasm_bindgen::to_value(&telem)
                    .map_err(|e| JsValue::from_str(&format!("brain_telemetry serialize: {e}")))
            })
        })
    }

    /// Same as `brain_telemetry_pixels` but over a pre-computed flat
    /// observation `[n_tokens × token_dim]` (bypasses the retina).
    #[wasm_bindgen]
    pub fn brain_telemetry_obs(obs: &[f32]) -> Result<JsValue, JsValue> {
        BRAIN.with(|b| {
            let brain_ref = b.borrow();
            let brain = brain_ref
                .as_ref()
                .ok_or_else(|| JsValue::from_str("brain_telemetry: weights not loaded"))?;
            let out = regional_forward(brain, obs);
            let telem = super::region_telemetry(brain, &out);
            serde_wasm_bindgen::to_value(&telem)
                .map_err(|e| JsValue::from_str(&format!("brain_telemetry serialize: {e}")))
        })
    }

    // ── Task 6: decision drivers (readout introspection) ────────────────

    /// "What drives the decision" for raw RGB pixels `[3 × H × W]` CHW.
    /// Runs the forward, then decomposes the LAST tick's readout
    /// `pred = output_proj(global_sync)`. Returns `DecisionDrivers`:
    /// `{tick, pre[n_global_sync], logits[out_dims], move_softmax[n_moves],
    ///   move_contributions[n_moves][n_global_sync], move_bias[n_moves],
    ///   n_moves}` where `move_contributions[j][i] = W[j,i]·pre[i]`.
    ///
    /// This brain's outer forward exposes no attention/eligibility, so none
    /// is fabricated; `pre` and the per-channel readout contributions are the
    /// honest "drivers." Also stashes plasticity inputs (free).
    #[wasm_bindgen]
    pub fn decision_drivers_pixels(pixels: &[f32]) -> Result<JsValue, JsValue> {
        CORTEX.with(|c| {
            let cortex_ref = c.borrow();
            let cortex = cortex_ref
                .as_ref()
                .ok_or_else(|| JsValue::from_str("decision_drivers_pixels: no visual cortex loaded"))?;
            let (obs, _n, _d) = cortex.spatial_tokens(pixels);
            BRAIN.with(|b| {
                let brain_ref = b.borrow();
                let brain = brain_ref
                    .as_ref()
                    .ok_or_else(|| JsValue::from_str("decision_drivers: weights not loaded"))?;
                let out = regional_forward(brain, &obs);
                stash_plasticity_inputs(&out);
                let drivers = super::decision_drivers(&brain.output_proj, &out)
                    .ok_or_else(|| JsValue::from_str("decision_drivers: empty forward (no ticks)"))?;
                serde_wasm_bindgen::to_value(&drivers)
                    .map_err(|e| JsValue::from_str(&format!("decision_drivers serialize: {e}")))
            })
        })
    }

    /// Same as `decision_drivers_pixels` but over a pre-computed flat
    /// observation `[n_tokens × token_dim]` (bypasses the retina).
    #[wasm_bindgen]
    pub fn decision_drivers_obs(obs: &[f32]) -> Result<JsValue, JsValue> {
        BRAIN.with(|b| {
            let brain_ref = b.borrow();
            let brain = brain_ref
                .as_ref()
                .ok_or_else(|| JsValue::from_str("decision_drivers: weights not loaded"))?;
            let out = regional_forward(brain, obs);
            stash_plasticity_inputs(&out);
            let drivers = super::decision_drivers(&brain.output_proj, &out)
                .ok_or_else(|| JsValue::from_str("decision_drivers: empty forward (no ticks)"))?;
            serde_wasm_bindgen::to_value(&drivers)
                .map_err(|e| JsValue::from_str(&format!("decision_drivers serialize: {e}")))
        })
    }

    // ── Task 7: live plasticity (bounded three-factor local update) ─────

    /// Apply ONE bounded three-factor local update to the readout's first-5
    /// (move) rows, using the LAST forward's stashed `(pre, logits)`:
    ///
    ///   ΔW[d,j] = θ · signal · (onehot(chosen)[d] − softmax(logits[0..5])[d]) · pre[j]
    ///
    /// with learning rate θ = 0.02, per-step `|ΔW| ≤ 0.25`, post-update
    /// `|W| ≤ 6.0`, and a NaN/Inf guard (non-finite ΔW elements are skipped).
    /// `chosen` is the move index in `0..5`. `signal` is the reward/error
    /// scalar; with `signal == 0` NOTHING changes (Δ is exactly 0), so the
    /// forward stays bit-exact between updates. The deltas are ADDED to
    /// `output_proj.weight` in place. Returns the L2 norm ‖ΔW‖ actually
    /// applied. The pristine readout is snapshotted lazily on first call so
    /// `reset_plasticity` can restore it.
    ///
    /// Requires `load_brain_weights` + at least one run/decision call first
    /// (so `pre`/`logits` are stashed).
    #[wasm_bindgen]
    pub fn apply_plasticity(chosen: usize, signal: f32) -> Result<f32, JsValue> {
        const THETA: f32 = 0.02;
        const DW_CLAMP: f32 = 0.25;
        const W_CLAMP: f32 = 6.0;

        let pre = LAST_PRE
            .with(|p| p.borrow().clone())
            .ok_or_else(|| JsValue::from_str("apply_plasticity: no stashed pre — run the brain first"))?;
        let logits = LAST_LOGITS
            .with(|l| l.borrow().clone())
            .ok_or_else(|| JsValue::from_str("apply_plasticity: no stashed logits — run the brain first"))?;

        BRAIN.with(|b| {
            let mut brain_ref = b.borrow_mut();
            let brain = brain_ref
                .as_mut()
                .ok_or_else(|| JsValue::from_str("apply_plasticity: weights not loaded"))?;
            let proj = &mut brain.output_proj;
            let n_in = proj.in_dim;
            let n_moves = proj.out_dim.min(5);

            if chosen >= n_moves {
                return Err(JsValue::from_str(&format!(
                    "apply_plasticity: chosen {chosen} out of range (n_moves = {n_moves})"
                )));
            }
            if pre.len() < n_in {
                return Err(JsValue::from_str(
                    "apply_plasticity: stashed pre shorter than readout in_dim",
                ));
            }

            // Lazily snapshot the pristine readout BEFORE the first mutation.
            PRISTINE_READOUT.with(|s| {
                if s.borrow().is_none() {
                    *s.borrow_mut() = Some((proj.weight.clone(), proj.bias.clone()));
                }
            });

            // Three-factor error term over the first-5 move logits.
            let move_logits: Vec<f32> = logits.iter().take(n_moves).copied().collect();
            let probs = super::softmax(&move_logits);

            let mut norm_sq = 0.0f32;
            for d in 0..n_moves {
                let onehot = if d == chosen { 1.0 } else { 0.0 };
                let post = probs.get(d).copied().unwrap_or(0.0);
                let err = onehot - post; // (target − prediction)
                let row = &mut proj.weight[d * n_in..(d + 1) * n_in];
                for j in 0..n_in {
                    let mut dw = THETA * signal * err * pre[j];
                    if !dw.is_finite() {
                        continue; // NaN/Inf guard — skip this element.
                    }
                    dw = dw.clamp(-DW_CLAMP, DW_CLAMP);
                    let updated = (row[j] + dw).clamp(-W_CLAMP, W_CLAMP);
                    let applied = updated - row[j]; // post-clamp effective Δ.
                    row[j] = updated;
                    norm_sq += applied * applied;
                }
            }
            Ok(norm_sq.sqrt())
        })
    }

    /// Restore `output_proj` to its pristine (as-loaded) state, undoing all
    /// `apply_plasticity` updates. No-op if plasticity was never applied
    /// (no snapshot taken yet). The snapshot itself is retained so repeated
    /// reset → plasticise → reset cycles all restore the same baseline.
    #[wasm_bindgen]
    pub fn reset_plasticity() -> Result<(), JsValue> {
        let snap = PRISTINE_READOUT.with(|s| s.borrow().clone());
        let (weight, bias) = match snap {
            Some(wb) => wb,
            None => return Ok(()), // nothing was ever changed.
        };
        BRAIN.with(|b| {
            let mut brain_ref = b.borrow_mut();
            let brain = brain_ref
                .as_mut()
                .ok_or_else(|| JsValue::from_str("reset_plasticity: weights not loaded"))?;
            brain.output_proj.weight = weight;
            brain.output_proj.bias = bias;
            Ok(())
        })
    }

    // ── Task 8: adaptive-compute summary (exit-gate λ trajectory) ───────

    /// Per-tick adaptive-compute summary for raw RGB pixels `[3 × H × W]`
    /// CHW. The brain's outer exit gate is global (not per-region), so the
    /// honest unit of "adaptive compute" is the OUTER tick. Returns:
    /// `{ticks_used, lambda_trajectory[ticks_used], survival[ticks_used],
    ///   exit_cdf[ticks_used], certainty[ticks_used]}` where, per tick,
    /// `lambda` is the outer gate λ, `survival` is Π(1−λ) up to (excluding)
    /// that tick, `exit_cdf` is the cumulative exit probability Σ λ·survival,
    /// and `certainty` is `1 − normalized_entropy(prediction)`.
    ///
    /// No per-region exit λ is fabricated: the inner per-region gates are not
    /// exported by the forward, so this summarises what genuinely gates the
    /// brain's compute — the outer adaptive gate.
    #[wasm_bindgen]
    pub fn adaptive_compute_pixels(pixels: &[f32]) -> Result<JsValue, JsValue> {
        CORTEX.with(|c| {
            let cortex_ref = c.borrow();
            let cortex = cortex_ref
                .as_ref()
                .ok_or_else(|| JsValue::from_str("adaptive_compute_pixels: no visual cortex loaded"))?;
            let (obs, _n, _d) = cortex.spatial_tokens(pixels);
            BRAIN.with(|b| {
                let brain_ref = b.borrow();
                let brain = brain_ref
                    .as_ref()
                    .ok_or_else(|| JsValue::from_str("adaptive_compute: weights not loaded"))?;
                let out = regional_forward(brain, &obs);
                serde_wasm_bindgen::to_value(&super::adaptive_compute_summary(&out))
                    .map_err(|e| JsValue::from_str(&format!("adaptive_compute serialize: {e}")))
            })
        })
    }

    /// Same as `adaptive_compute_pixels` but over a pre-computed flat
    /// observation `[n_tokens × token_dim]` (bypasses the retina).
    #[wasm_bindgen]
    pub fn adaptive_compute_obs(obs: &[f32]) -> Result<JsValue, JsValue> {
        BRAIN.with(|b| {
            let brain_ref = b.borrow();
            let brain = brain_ref
                .as_ref()
                .ok_or_else(|| JsValue::from_str("adaptive_compute: weights not loaded"))?;
            let out = regional_forward(brain, obs);
            serde_wasm_bindgen::to_value(&super::adaptive_compute_summary(&out))
                .map_err(|e| JsValue::from_str(&format!("adaptive_compute serialize: {e}")))
        })
    }

    // ── VIN (Value-Iteration-Network) ego-centric trainable readout ─────

    /// Serialisable VIN forward result returned to JS.
    #[derive(serde::Serialize)]
    struct VinForwardOut {
        /// Ego-centric move logits `[5]` (U/D/L/R/Wait) from the agent cell.
        move_logits: Vec<f32>,
        /// The agent cell actually used, `[row, col]` — TRUE 9×9 maze coords.
        agent_cell: [usize; 2],
        /// Grid dims `[SIZE, SIZE]` (the true maze, e.g. 9×9).
        grid: [usize; 2],
        /// Per-cell scalar value `[SIZE²]` (row-major) from exact value
        /// iteration, ≈ GAMMA^(dist-to-goal) on open cells, 0 in walls —
        /// the "how good is it to be here" heat field for the maze overlay.
        value_grid: Vec<f32>,
        /// Per-cell traversability gate `[SIZE²]` (1.0 free, 0.0 wall).
        gate: Vec<f32>,
    }

    /// Run the trainable VIN readout DIRECTLY over the raw maze pixels
    /// `[3 × SIZE × SIZE]` CHW (NOT the retina). Derives wall/goal/agent from
    /// the pixels (wall=black, goal=green, agent=red), runs EXACT value
    /// iteration on the SIZE×SIZE grid so goal-value floods backward around
    /// walls, then reads 5 move logits from the AGENT'S OWN cell (its value +
    /// gate + 4 neighbour values) — no global pooling, no learned geometry.
    ///
    /// `agent_row`/`agent_col` are the agent's TRUE maze coords, used only as a
    /// fallback if no red agent pixel is present. SIZE = round(sqrt(len/3)).
    /// The VIN move head is lazily seeded (deterministic) on first call.
    ///
    /// Returns `{ move_logits: [5], agent_cell: [r,c], grid: [SIZE,SIZE],
    ///            value_grid: [SIZE²], gate: [SIZE²] }`. The agent-cell
    /// readout and move logits are stashed for `vin_learn`. No cortex needed.
    #[wasm_bindgen]
    pub fn vin_forward(
        pixels: &[f32],
        agent_row: usize,
        agent_col: usize,
    ) -> Result<JsValue, JsValue> {
        if pixels.len() % 3 != 0 {
            return Err(JsValue::from_str(&format!(
                "vin_forward: pixel buffer not divisible by 3 channels (len = {})",
                pixels.len()
            )));
        }
        let n = pixels.len() / 3;
        let size = (n as f64).sqrt().round() as usize;
        if size == 0 || size * size != n {
            return Err(JsValue::from_str(&format!(
                "vin_forward: maze grid not square (channel size = {n})"
            )));
        }

        VIN.with(|vcell| {
            {
                let mut vref = vcell.borrow_mut();
                if vref.is_none() {
                    *vref = Some(VinReadout::seeded(3));
                }
            }
            let vref = vcell.borrow();
            let vin = vref.as_ref().unwrap();

            let out = vin.forward_pixels(pixels, size, (agent_row, agent_col));

            // Stash for the learning step.
            VIN_LAST_PRE.with(|p| *p.borrow_mut() = Some(out.pre.clone()));
            VIN_LAST_LOGITS.with(|l| *l.borrow_mut() = Some(out.move_logits.clone()));

            let payload = VinForwardOut {
                move_logits: out.move_logits,
                agent_cell: [out.agent_cell.0, out.agent_cell.1],
                grid: [size, size],
                value_grid: out.value_grid,
                gate: out.gate,
            };
            serde_wasm_bindgen::to_value(&payload)
                .map_err(|e| JsValue::from_str(&format!("vin_forward serialize: {e}")))
        })
    }

    /// Apply ONE three-factor local update to the VIN move head only, using
    /// the last `vin_forward`'s stashed `(pre, logits)`:
    ///
    ///   eligibility[d,j] = (onehot(target_move)[d] − softmax(logits)[d])·pre[j]
    ///   ΔW[d,j]          = θ · (teach − pain) · eligibility[d,j]
    ///
    /// `target_move` ∈ `0..5` teaches that move (teach = 1); `target_move < 0`
    /// applies only the `−pain` (wall-hit) penalty. θ = 0.05, bounded
    /// `|ΔW| ≤ 0.25`, `|W| ≤ 6.0`, NaN/Inf-guarded. Momentum-free, no backprop
    /// graph — only the move head changes (value-propagation weights are
    /// FIXED). Returns ‖ΔW‖ applied. Requires a prior `vin_forward`.
    #[wasm_bindgen]
    pub fn vin_learn(target_move: i32, pain: f32) -> Result<f32, JsValue> {
        let pre = VIN_LAST_PRE
            .with(|p| p.borrow().clone())
            .ok_or_else(|| JsValue::from_str("vin_learn: no stashed pre — run vin_forward first"))?;
        let logits = VIN_LAST_LOGITS.with(|l| l.borrow().clone()).ok_or_else(|| {
            JsValue::from_str("vin_learn: no stashed logits — run vin_forward first")
        })?;
        VIN.with(|vcell| {
            let mut vref = vcell.borrow_mut();
            let vin = vref
                .as_mut()
                .ok_or_else(|| JsValue::from_str("vin_learn: VIN not initialised"))?;
            Ok(vin.learn(&pre, &logits, target_move, pain))
        })
    }

    /// Re-seed the VIN readout to its deterministic init (the "Reset learning"
    /// path), discarding all `vin_learn` updates. Rebuilt for the current
    /// per-cell width on the next `vin_forward` if the width changes; if a VIN
    /// already exists it is re-seeded in place for the same width.
    #[wasm_bindgen]
    pub fn vin_reset() {
        VIN.with(|vcell| {
            let mut vref = vcell.borrow_mut();
            if let Some(existing) = vref.as_ref() {
                *vref = Some(VinReadout::seeded(existing.raw_dim));
            } else {
                *vref = None;
            }
        });
        VIN_LAST_PRE.with(|p| *p.borrow_mut() = None);
        VIN_LAST_LOGITS.with(|l| *l.borrow_mut() = None);
    }

    /// Load the TRAINED VIN planner from the SDK's serde JSON export of a
    /// `VinReadout` and stash it for `learned_vin_forward`. Field names match
    /// the SDK exactly (`config, raw_dim, reward_proj, gate_proj, value_proj,
    /// agent_proj, highway_proj, move_head`). This is the real learned forward
    /// — distinct from the hardcoded `vin_forward` value-iteration path.
    #[wasm_bindgen]
    pub fn load_learned_vin(json: &str) -> Result<(), JsValue> {
        let vin: super::LearnedVin = serde_json::from_str(json)
            .map_err(|e| JsValue::from_str(&format!("load_learned_vin: {e}")))?;
        LEARNED_VIN.with(|v| *v.borrow_mut() = Some(vin));
        Ok(())
    }

    /// Run the loaded learned VIN forward over `tokens` (flat
    /// `[grid_h*grid_w × raw_dim]`, row-major) with the agent at
    /// `(agent_r, agent_c)`. Returns the `n_dirs` move logits. Errors if no
    /// learned VIN has been loaded via `load_learned_vin`.
    #[wasm_bindgen]
    pub fn learned_vin_forward(
        tokens: &[f32],
        grid_h: usize,
        grid_w: usize,
        agent_r: usize,
        agent_c: usize,
    ) -> Result<Vec<f32>, JsValue> {
        LEARNED_VIN.with(|v| {
            let vref = v.borrow();
            let vin = vref.as_ref().ok_or_else(|| {
                JsValue::from_str("learned_vin_forward: no learned VIN loaded — call load_learned_vin first")
            })?;
            Ok(vin.forward(tokens, grid_h, grid_w, (agent_r, agent_c)))
        })
    }

    /// Like `learned_vin_forward`, but appends the planner's per-cell scalar
    /// value map after the move logits: `[n_dirs logits | grid_h*grid_w value]`.
    /// The value map is the model's OWN proximity-to-goal estimate (no solver),
    /// so the caller can derive an honest "getting closer?" signal from it.
    #[wasm_bindgen]
    pub fn learned_vin_forward_compass(
        tokens: &[f32],
        grid_h: usize,
        grid_w: usize,
        agent_r: usize,
        agent_c: usize,
    ) -> Result<Vec<f32>, JsValue> {
        LEARNED_VIN.with(|v| {
            let vref = v.borrow();
            let vin = vref.as_ref().ok_or_else(|| {
                JsValue::from_str("learned_vin_forward_compass: no learned VIN loaded — call load_learned_vin first")
            })?;
            let (logits, vmap) = vin.forward_compass(tokens, grid_h, grid_w, (agent_r, agent_c));
            let mut out = Vec::with_capacity(logits.len() + vmap.len());
            out.extend_from_slice(&logits);
            out.extend_from_slice(&vmap);
            Ok(out)
        })
    }

    /// DREAM CONSOLIDATION step on the LOADED learned VIN: one SGD update of the
    /// move head toward `target_move` for the maze in `tokens`/`agent`. Mutates
    /// the resident planner (persists for the session) and returns the
    /// cross-entropy loss before the step. A sleep/replay pass calls this per
    /// remembered maze (target = its BFS-optimal move).
    #[wasm_bindgen]
    pub fn learned_vin_train(
        tokens: &[f32],
        grid_h: usize,
        grid_w: usize,
        agent_r: usize,
        agent_c: usize,
        target_move: i32,
        lr: f32,
    ) -> Result<f32, JsValue> {
        if target_move < 0 {
            return Ok(0.0);
        }
        LEARNED_VIN.with(|v| {
            let mut vref = v.borrow_mut();
            let vin = vref.as_mut().ok_or_else(|| {
                JsValue::from_str("learned_vin_train: no learned VIN loaded — call load_learned_vin first")
            })?;
            // Lazily snapshot the pristine move head BEFORE the first mutation.
            LEARNED_VIN_PRISTINE.with(|s| {
                if s.borrow().is_none() {
                    *s.borrow_mut() =
                        Some((vin.move_head.weight.clone(), vin.move_head.bias.clone()));
                }
            });
            Ok(vin.consolidate_move(
                tokens,
                grid_h,
                grid_w,
                (agent_r, agent_c),
                target_move as usize,
                lr,
            ))
        })
    }

    /// Restore the learned VIN's move head to its pristine (as-loaded) weights,
    /// undoing every sleep-consolidation pass. No-op if it was never trained.
    #[wasm_bindgen]
    pub fn learned_vin_reset() {
        LEARNED_VIN_PRISTINE.with(|s| {
            if let Some((w, b)) = s.borrow().as_ref() {
                LEARNED_VIN.with(|v| {
                    if let Some(vin) = v.borrow_mut().as_mut() {
                        vin.move_head.weight = w.clone();
                        vin.move_head.bias = b.clone();
                    }
                });
            }
        });
    }
}
