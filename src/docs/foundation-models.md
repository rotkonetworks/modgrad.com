---
title: Foundation models
section: Systems
order: 80
description: The transformer stack, native Qwen2.5 inference, the lm_validate training check, the byte-latent transformer pipeline, quantized Gemma inference, and the cerebellum-as-LLM design.
---

# Foundation models

modgrad's transformer stack lets a pretrained model sit inside a brain. The design
goal is to use a frozen LLM as the **cerebellum**: in the target preset a frozen
language model holds most of the parameters while a small trainable cortex reads
from it. That preset is not built yet; the pieces below are.

## Running today

- **Qwen2.5-0.5B inference** (`qwen_chat`). `modgrad-transformer` loads
  safetensors, tokenizes with an HF BPE tokenizer, and decodes text with argmax /
  temperature / top-k sampling and a resident KV cache.
- **LM training check** (`lm_validate`). An earlier smoke test showed the training
  chain is connected (cross-entropy 5.72 → 0.74 on a single repeated 17-token
  sequence, i.e. memorization). `lm_validate` is the real check: a 2-layer, 4-head,
  `d_model` 128 byte-level transformer trained on an English corpus with a held-out
  validation split, reporting bits-per-byte (random-uniform = 8.0) and a
  PASS / WARN / FAIL verdict on whether it trains rather than memorizes.
- **Resident GPT** (`GptModelResident`). Weights, gradients and optimizer moments
  stay on the GPU across steps; matmul, softmax, RoPE, layer norm and AdamW dispatch
  to the device.
- **FFN cerebellum** (`modgrad-ffn`). A standalone SwiGLU language prior (default
  `d_model` 1024, hidden 5120, 12 layers) behind the `FrozenCerebellum` trait,
  trainable on next-token prediction and mountable as a frozen module the cortex
  reads across layers.
- **Quantized inference** (`modgrad-gemma`). An application crate that runs
  Gemma-4 GGUF models on ROCm, using the K-quant matvec kernels and GGUF loader in
  `modgrad-device`.

## The byte-latent transformer

`modgrad-blt` implements a Byte-Latent Transformer
([Pagnoni et al. 2024](https://arxiv.org/abs/2412.09871)): a transformer on raw
bytes without a tokenizer, spending compute where bytes are hard to predict. The
pipeline is hierarchical:

```
bytes → entropy patcher → local encoder → latent transformer → local decoder → bytes
           (segment)       (bytes→patches)   (the heavy model)   (patches→bytes)
```

**The entropy patcher.** A small byte-level model estimates next-byte entropy at
every position. A patch boundary fires where the entropy (or its jump from the
previous byte) crosses a threshold, so predictable runs become one patch and
surprising bytes get their own. The context resets on newlines. Patch sequences come
out roughly 4–8× shorter than the bytes, and the large model runs once per patch.

**Local encoder and decoder.** The encoder embeds each byte (with rolling 3–8-byte
n-gram hashes) and pools bytes into patch representations through patch-aware
cross-attention. The decoder runs the inverse: byte queries cross-attend over patch
keys/values, causally up to the containing patch, to produce logits over the 256
byte values.

**The latent transformer.** The large "global" model runs over patches. It is a
standard transformer with its embedding and LM head bypassed: patch representations
feed straight into the blocks, and its KV cache is indexed by patch count, not byte
count. That is what lets a pretrained tokenizer model be reused on bytes.

**The byteify recipe.** `ByteifyRecipe::from_qwen2` initializes the latent from a
pretrained Qwen2.5, then trains the new local encoder/decoder at the full learning
rate while the latent moves at one-tenth, to adapt a tokenizer-based LLM to bytes
while changing it as little as possible.

**As a cerebellum.** `BltCerebellum` wraps a trained BLT as a `FrozenCerebellum`,
exposing per-layer latent hidden states (one row per patch) for the cortex to read.

**Status.** The forward pipeline (entropy model, patcher, encoder, latent, decoder,
cerebellum wrapper) is implemented and tested. Backward passes for encoder, latent
and decoder are implemented; their gradient-check status is tracked in
`docs/BLT_BACKWARD.md` in the SDK repo. End-to-end byteify training has not been
demonstrated.

## The cerebellum-mounted brain (not built yet)

`eight_region_v2` is the cerebellum-dominant preset. The design: a frozen
Qwen2.5-0.5B (~494M parameters) as the cerebellum, about 80% of a ~600M-parameter
brain, with a cortex of a few tens of millions of parameters that trains. In the
code today the preset's cerebellum region is a small placeholder; with
`forward_cached_frozen` the frozen model's output replaces that region's output each
outer tick, so it enters the global sync. Routing its hidden states into the
cortical regions directly is not wired. The
plan is in `docs/BRAIN_ARCHITECTURE.md` in the SDK repo.

Next: the [SDK crate reference →](/docs/crates).
