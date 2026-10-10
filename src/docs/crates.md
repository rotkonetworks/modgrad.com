---
title: SDK crates
section: Reference
order: 90
description: The crates in the modgrad workspace and what each one contains.
---

# SDK crates

modgrad is a Cargo workspace of separate crates. Lower layers don't depend on
higher ones: `modgrad-compute` doesn't know about `modgrad-ctm`, and
`modgrad-training` doesn't know about `modgrad-codec`, so you can take only the
parts you need.

## Core

| crate | contents |
|-------|----------|
| **modgrad-traits** | core traits: `Brain`, `Encoder`, `LossFn`, `Vjp`, `FrozenCerebellum`; `TokenInput` |
| **modgrad-ctm** | single CTM (NLM, sync, MHA, U-Net synapse, BPTT), the CMM, graph composition, `NeuralComputer`, bio modules, plural & organism, the `VinReadout` value-iteration planner |
| **modgrad-compute** | `Linear`, tensor ops, batched dispatch, resident GPU buffers |
| **modgrad-device** | CPU / CUDA / ROCm / KFD / Vulkan backends and resident kernels (matvec, AdamW, RoPE, RMSNorm, quantized dequant-matvec) |
| **modgrad-autograd** | a small reverse-mode autodiff tape that composes existing modgrad kernels |
| **modgrad-training** | AdamW / Adam / SGD, warmup and cosine schedules, gradient accumulation, checkpointing, dream replay, reverse-replay credit assignment |
| **modgrad-persist** | save / load, f32 / f16 / i8 quantization |

## Models and data

| crate | contents |
|-------|----------|
| **modgrad-transformer** | transformer blocks, attention, RoPE, KV cache, `GptModelResident`, Qwen-class loading |
| **modgrad-blt** | byte-latent transformer: entropy patcher, local encoder / decoder, byteify |
| **modgrad-ffn** | SwiGLU MLP language model |
| **modgrad-codec** | `VisualCortex` (retina → V1 → V2 → V4), VQ-VAE, FSQ, audio codec, byte n-gram hash |
| **modgrad-data** | multimodal tokenization and streaming |
| **modgrad-io** | telemetry, safetensors and ONNX backends, Qwen / Llama weight loading |
| **modgrad-memory** | episodic memory and gated-delta fast weights |
| **modgrad-lm-memory** | memory for a frozen LLM: episodic store and read paths |
| **modgrad-mouth** | gated cross-attention bridge from an expert encoder into a frozen decoder |

## Research and tooling

| crate | contents |
|-------|----------|
| **modgrad-pc** | predictive coding and augmented-Lagrangian PC for residual MLPs, compared against backprop |
| **modgrad-sheaf** | Sheaf-ADMM, a learned unrolled convex optimizer |
| **modgrad-attribution** | gradient-based data attribution (per-sample influence ranking) |
| **modgrad-tasks** | multi-task harness: chess, mazes, sudoku, addition, with a parameter-matched transformer baseline |
| **modgrad-chess** | chess environment, encoding, UCI wrapper and Elo harness |
| **modgrad-substrate** | host observability: CPU frequency, throttle counters, thermal sensors |
| **modgrad-serve** | local inference server (REST, Anthropic Messages API, MCP) over a `LanguageModel` trait |
| **modgrad-gemma** | Gemma GGUF inference and server on `modgrad-device` |
| **modgrad-prompts** | shared system and user prompts |
| **modgrad-wasm** | WebAssembly bindings (used by the [/play demo](/play)) |

## The trait foundation

Most model code implements `modgrad-traits::Brain`:

```rust
// (Weights, State, Input) -> (Output, State)
trait Brain {
    type Input;
    type Weights;
    type State;
    type Cache;
    type Gradients;
    // forward, forward_cached, backward: associated functions, no &mut self
}
```

State is passed in and returned rather than hidden behind `&mut self`, so it can be
inspected, checkpointed or reset explicitly. The `Vjp` trait pairs every forward op
with its hand-written backward, so an op can't be merged with only the forward
half. Optimizer moments, NLM history and sync buffers are serializable structs.
