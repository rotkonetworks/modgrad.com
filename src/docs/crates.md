---
title: SDK crates
section: Reference
order: 90
description: Fourteen composable Rust crates. Use any of them independently.
---

# SDK crates

modgrad is a workspace of independent crates. Each one is useful on its own:
`modgrad-compute` doesn't know about `modgrad-ctm`, and `modgrad-training` doesn't
know about `modgrad-codec`. Import one, or import all fourteen.

## The crates

| crate | what it gives you |
|-------|-------------------|
| **modgrad-ctm** | single CTM (NLM, sync, MHA, U-Net synapse, full BPTT) + graph composition, `NeuralComputer`, the bio modules, plural & organism, the `VinReadout` value-iteration planner |
| **modgrad-compute** | `Linear`, tensor ops, batched dispatch, resident GPU buffers |
| **modgrad-device** | CPU / CUDA / ROCm / KFD / Vulkan backends; resident kernels (matvec, AdamW, RoPE, RMSNorm) |
| **modgrad-transformer** | transformer blocks, MHA, RoPE, KV cache, `GptModelResident`, Qwen-class loader |
| **modgrad-blt** | byte-latent transformer: entropy patcher, local encoder/decoder, byteify recipe |
| **modgrad-substrate** | foundation-model substrate: Q4_K residency, streaming loaders, 7B-on-8 GB target |
| **modgrad-codec** | `VisualRetina` (V1→V4), VQ-VAE, audio codec, FSQ, byte n-gram hash |
| **modgrad-ffn** | SwiGLU language prior + `FrozenCerebellum` trait |
| **modgrad-data** | type-safe multimodal tokenization, mixed-modality streaming, lazy loading |
| **modgrad-training** | AdamW / Adam / SGD, warmup/cosine schedulers, gradient accumulation, dream replay (sleep consolidation), reverse-replay credit assignment |
| **modgrad-memory** | episodic memory with valence, content-addressable recall, surprise-gated replay |
| **modgrad-io** | telemetry, wincode serialization, safetensors + ONNX + GGUF backends |
| **modgrad-persist** | wincode / JSON save-load, f32/f16/i8 quantization |
| **modgrad-traits** | the core traits: `Brain`, `Encoder`, `LossFn`, `TokenInput` |

## The trait foundation

Everything rests on `modgrad-traits`. The central one is `Brain`:

```rust
// (Weights, State, Input) -> (Output, State)
trait Brain {
    type Weights;
    type State;
    type Input;
    // forward, backward, and gradient application are pure functions
}
```

State is **explicit** and threaded through, never hidden behind `&mut self`. The
type system enforces the brain/host boundary at compile time: a brain literally
cannot do I/O. The paired `Vjp` trait requires a `backward` alongside every
`forward`, so a new op can't be merged half-differentiable.

This is the "pure functions over magic" design: no YAML, no framework defaults, no
hidden global state. Optimizer moments, NLM history, and sync buffers are all
serializable structs you can inspect and checkpoint.
