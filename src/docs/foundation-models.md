---
title: Foundation models
section: Systems
order: 80
description: The exact transformer stack, native Qwen2.5 inference, the lm_validate training proof, the byte-latent transformer pipeline, the quantized substrate, and the cerebellum-as-LLM design.
---

# Foundation models

modgrad's transformer stack lets you bring a pretrained model into a brain. The
design thesis: make the **cerebellum an LLM**. In biology the cerebellum holds
~80% of the brain's neurons; in modgrad's target preset a frozen language model
takes ~82% of the parameter budget while a small, trainable cortex orchestrates
it.

## Running today

- **Qwen2.5-0.5B inference** (`qwen_chat`). `modgrad-transformer` loads
  safetensors, tokenizes with an HF BPE tokenizer, and decodes coherent text with
  argmax / temperature / top-k sampling and a resident KV cache. RoPE stays
  host-side (sub-microsecond); per-head scores use strided KV access.
- **End-to-end LM training** (`lm_validate`). A small 2-layer, 4-head, `d_model`
  128 byte-level transformer drives cross-entropy from **5.72 → 0.74 in 10 steps**
  on real text, tracks a held-out validation curve, and reports bits-per-byte
  (random-uniform = 8.0). It exists to prove the full gradient chain trains on
  real data.
- **Resident GPT** (`GptModelResident`). Weights, gradients, and optimizer moments
  stay on the GPU across steps; matmul, softmax, RoPE, layer norm, and AdamW all
  dispatch to the device. QK RMSNorm and the attention scale are baked into the
  weight buffer, and per-head softmax is batched into a single MIOpen call.
- **FFN cerebellum** (`modgrad-ffn`). A standalone SwiGLU language prior
  (`vocab → 1024 → 4096 → 1024 → vocab`) with a `FrozenCerebellum` trait, trainable
  on next-token prediction and mountable as a frozen module the cortex blends
  across layers. CPU and zero-copy VRAM paths.

## The byte-latent path

`modgrad-blt` implements a Byte-Latent Transformer
([Pagnoni et al. 2024](https://arxiv.org/abs/2412.09871)): an **entropy patcher**
segments the byte stream, a **local encoder/decoder** maps bytes ↔ patches, and a
heavier **latent transformer** runs over patches (a ~4–8× shorter sequence than
raw bytes). The `ByteifyRecipe::from_qwen2` recipe initializes the latent
transformer from a pretrained Qwen2.5 and trains the local stack at one-tenth the
learning rate. The forward path lands today; resident backward through the
patch-aware cross-attention is the next slice. A `BltCerebellum` adapter already
wraps a BLT model as a `FrozenCerebellum`, exposing per-layer hidden states keyed
by **patch** count rather than byte count.

## The substrate (roadmap)

`modgrad-substrate` targets 7B-class models on 8 GB of VRAM via **Q4_K residency**
and streaming weight loaders. The quantized dequant-matvec kernels are committed
and the CPU dequant path is proven; full GPU residency is in progress.

## The cerebellum-mounted brain (roadmap)

`eight_region_v2` is the cerebellum-dominant preset: a frozen Qwen2.5-0.5B
(~494M params) as the cerebellum is ~82% of a ~600M-param brain, leaving the
cortex around 50M and trainable. The mounting path (`FrozenCerebellum` →
`forward_cached_frozen`) works; the connections that route the cerebellum's
hidden states *back* into the cortex are the remaining wiring. The full plan lives
in `docs/BRAIN_ARCHITECTURE.md` in the repo.

The transformer inference and LM-training pieces are real and tested today; the
substrate and the cerebellum-mounted brain are the active build, and this page
marks which is which.

Next: the [SDK crate reference →](/docs/crates).
