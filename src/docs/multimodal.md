---
title: Multimodal & token space
section: Systems
order: 70
description: A fixed retina feeding a learnable V1→V2→V4 visual cortex, Hebbian and global-guided learning rules, the VQ-VAE / audio / FSQ codecs, byte n-gram hashing, and one shared token space.
---

# Multimodal & token space

modgrad puts every modality into **one vocabulary**, with a single embedding table
and one next-token objective. Text is raw bytes; images, audio, timestamps and
actions are codec outputs placed at reserved offsets. Vision also has its own
convolutional cortex, described first.

## The visual cortex

`modgrad-codec`'s `VisualCortex` is a four-stage hierarchy loosely modeled on the
primate ventral stream: a **fixed** retina feeding three **learnable** stages.

| stage | in → out | learns | what it computes |
|-------|----------|--------|------------------|
| **retina** | 3 → 12 ch | no (fixed) | difference-of-Gaussian + color-opponent filters |
| **V1** | 12 → 32 ch | yes | oriented edges (Gabor-initialized) |
| **V2** | 32 → 64 ch | yes | mid-level features |
| **V4** | 64 → 128 ch | yes | spatial tokens for the CTM to attend over |
| **V4Ctm** (optional) | tokens → summary | yes | a small CTM run for a configurable number of ticks over the V4 tokens |

- **The retina is fixed**: a 12-channel bank of DoG and red/green, blue/yellow
  color-opponent filters that respond to contrast rather than absolute brightness.
- **V1 starts from Gabor priors** (σ = 0.8, λ = 2.5): 8 orientations × 2 phases ×
  2 source channels = 32 filters, with the priors on one source channel and random
  weights on the rest.
- **V4Ctm** is an alternative to pooling: a small CTM attends over the V4 tokens and
  emits one summary vector.

The [/play](/play) brain uses this cortex with a fixed 9×9 input; see
[visual cortex](/docs/visual-cortex).

### Learning rules

Besides backprop, two local rules are implemented for the cortex, applied with a
cascade of rates (V1 fast, V2 ¾, V4 ½):

- **Hebbian sparse coding**: encode patches, keep the top-K channels per position,
  reconstruct, and move weights to reduce reconstruction error.
- **GHL (global-guided Hebbian learning)**: a three-factor rule combining a local
  Oja update with the **sign of the task gradient** (`ghl_update`). The GHL paper
  reports it coming within ~3% of backprop on ResNet-50 / ImageNet; we have not
  reproduced that number.

**Per-token LayerNorm at V4** (`per_token_ln_v4`): z-scoring each V4 token across
channels raised the feature effective rank from 1.7 to 11.2 (of 128) and gave a
+10-point unseen-class k-NN gain on our CIFAR-10 probe. It is off by default for
compatibility with saved cortexes.

### Top-down generation

The cortex can also run **top-down**: seed V4 with sparse noise and project back
through transposed convolutions (V4ᵀ → V2ᵀ → V1ᵀ → pixels) to generate images.
`lsd()` adds a model of 5-HT2A receptor desensitization (availability drops with
dose and recovers over time), and an `integration` parameter sets how much of a
dream's weight change is kept. Measured on mazes: `integration = 0.7` keeps
out-of-distribution accuracy stable; at `≥ 0.95` the cortex drifts toward a
synthetic attractor and OOD accuracy collapses (the code warns on that setting).
The range in between has not been mapped.

## The codecs

The other modalities are discretized by codecs in `modgrad-codec`:

- **VQ-VAE** (image): a conv encoder maps a 32×32 image to an 8×8 grid of codes
  (64 codes per image) from a 4096-entry codebook, with EMA updates and dead-code
  revival, decoded by a transposed-conv stack.
- **Audio codec** (WavTokenizer-style): a 1-D conv stack downsamples 24 kHz audio by
  2×4×5×8 = 320× into a 4096-code book, about 75 codes per second.
- **FSQ**: finite scalar quantization. No codebook; each dimension is rounded to a
  few fixed levels with a straight-through estimator, so there is no codebook to
  collapse.
- **Byte n-gram hashing**: instead of BPE, each byte embedding is augmented with
  rolling polynomial hashes of the preceding 3–8 bytes (about 500k entries per
  table).

`modgrad-data` adds typed tokenization and lazy, interleaved, weighted streaming
(e.g. 70% text / 20% image / 10% audio) that holds at most one chunk in memory.

## The token space

Every codec output is placed at a fixed offset between the top of Qwen2.5's dense
vocabulary and its special tokens (151643 and up), so neither is disturbed. The
layout is checked at compile time in `unified_tokenizer`:

| range | modality |
|-------|----------|
| `0 – 255` | text (raw bytes) |
| `140000 – 140008` | delimiters `<img> </img> <aud> </aud> <vid> </vid>` |
| `140008 – 144104` | image VQ codes (4096) |
| `144104 – 148200` | audio codes (4096) |
| `148200 – 148600` | timestamps (400 ticks, 0.5 s resolution) |
| `148600 – 148878` | actions (mouse, keyboard, coordinates) |

With one vocabulary and one embedding table, a single autoregressive model sees all
modalities in the same sequence, with no separate encoders or cross-attention
bridges.

Next: [mounting a foundation model →](/docs/foundation-models).
