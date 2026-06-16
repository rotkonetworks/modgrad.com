---
title: Foundation models
section: Systems
order: 80
description: The exact transformer stack, native Qwen2.5 inference, the lm_validate training proof, the byte-latent transformer pipeline, the quantized substrate, and the cerebellum-as-LLM design.
---

# Foundation models

modgrad's transformer stack lets you bring a pretrained model into a brain. The
design thesis is to make the **cerebellum an LLM**. In biology the cerebellum holds
~80% of the brain's neurons. In modgrad's target preset a frozen language model
takes ~82% of the parameter budget while a small, trainable cortex orchestrates
it.

## Running today

- **Qwen2.5-0.5B inference** (`qwen_chat`). `modgrad-transformer` loads
  safetensors, tokenizes with an HF BPE tokenizer, and decodes coherent text with
  argmax / temperature / top-k sampling and a resident KV cache. RoPE stays
  host-side (sub-microsecond), and per-head scores use strided KV access.
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

## The byte-latent transformer

`modgrad-blt` implements a Byte-Latent Transformer
([Pagnoni et al. 2024](https://arxiv.org/abs/2412.09871)), a way to run a
transformer on raw bytes without a tokenizer, by spending compute where the bytes
are actually hard to predict. The pipeline is hierarchical:

```
bytes → entropy patcher → local encoder → latent transformer → local decoder → bytes
           (segment)       (bytes→patches)   (the heavy model)   (patches→bytes)
```

**The entropy patcher.** A small byte-level model estimates the next-byte entropy at
every position. A patch boundary fires where the entropy (or its jump from the
previous byte) crosses a threshold, so predictable runs collapse into one patch and
surprising bytes get their own. The context resets on newlines to avoid baseline
drift. The patch sequence ends up roughly **4–8× shorter** than the raw bytes, and
the expensive model runs only once per patch.

**Local encoder and decoder.** The encoder embeds each byte (augmented with rolling
3–8-byte n-gram hashes, so a pure-byte model matches tokenizer-level features) and
pools bytes into patch representations through patch-aware cross-attention. The
decoder runs the inverse: byte queries cross-attend over patch keys/values, causally
up to the containing patch, to produce logits over the 256-value byte vocabulary.

**The latent transformer.** The heavy "global" model runs over patches. It's a
standard transformer with its embedding and LM head **bypassed**. Patch
representations feed straight into the blocks, and its KV cache is keyed by **patch
count, not byte count**. That alignment is what lets a byte stream and a much shorter
patch stream share one model, and it's why a pretrained tokenizer model can be
repurposed for bytes at all.

**The byteify recipe.** `ByteifyRecipe::from_qwen2` initializes the latent from a
pretrained **Qwen2.5**, then trains the freshly-initialized local encoder/decoder at
the full learning rate while the latent moves at **one-tenth**. This adapts a
tokenizer-based LLM into a byte-level one without forgetting what it already knows.

**As a cerebellum.** `BltCerebellum` wraps a trained BLT as a `FrozenCerebellum`,
exposing its per-layer latent hidden states (one row per patch) for the cortex to
read. It's the byte-native path to mounting an LLM as the brain's cerebellum.

**Status.** The full forward pipeline works and is tested: entropy model, patcher,
encoder, latent, decoder, and the cerebellum wrapper. Encoder, latent, and decoder
backward are all implemented. The one remaining gap is an upstream attention detail
(propagating gradients through cached keys/values during full-sequence training,
tracked in `docs/BLT_BACKWARD.md`) before end-to-end byteify training is numerically
exact.

## The substrate (roadmap)

`modgrad-substrate` targets 7B-class models on 8 GB of VRAM via **Q4_K residency**
and streaming weight loaders. The quantized dequant-matvec kernels are committed
and the CPU dequant path is proven; full GPU residency is in progress.

## The cerebellum-mounted brain (roadmap)

`eight_region_v2` is the cerebellum-dominant preset: a frozen Qwen2.5-0.5B
(~494M params) as the cerebellum is ~82% of a ~600M-param brain, leaving the
cortex around 50M and trainable. The mounting path (`FrozenCerebellum` →
`forward_cached_frozen`) works. The connections that route the cerebellum's
hidden states *back* into the cortex are the remaining wiring. The full plan lives
in `docs/BRAIN_ARCHITECTURE.md` in the repo.

The transformer inference and LM-training pieces are real and tested today. The
substrate and the cerebellum-mounted brain are the active build, and this page
marks which is which.

Next: the [SDK crate reference →](/docs/crates).
