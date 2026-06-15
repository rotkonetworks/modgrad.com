---
title: Multimodal & token space
section: Systems
order: 70
description: A fixed retina feeding a learnable V1→V2→V4 visual cortex, Hebbian and global-guided learning, the VQ-VAE / audio / FSQ codecs, byte n-gram hashing, and one unified token space.
---

# Multimodal & token space

modgrad treats every modality as tokens in **one vocabulary**, sharing a single
embedding table and one next-token objective. Text is raw bytes; images, audio,
timestamps, and actions are codec outputs parked at reserved offsets. Vision gets
special treatment: a biologically-structured cortex that *learns its own features*.

## The visual cortex

`modgrad-codec`'s `VisualRetina` is a four-stage hierarchy modeled on the primate
ventral stream — a **fixed** retina feeding three **learnable** cortical stages:

| stage | in → out | learns | what it computes |
|-------|----------|--------|------------------|
| **retina** | 3 → 12 ch | no (fixed) | difference-of-Gaussian + color-opponent ganglion cells |
| **V1** | 12 → 32 ch | yes | oriented edges (Gabor-initialized) |
| **V2** | 32 → 64 ch | yes | contours, figure-ground |
| **V4** | 64 → 128 ch | yes | object-part spatial tokens |
| **V4-CTM** | tokens → 128-d | yes | a CTM runs 4 attention ticks over the spatial tokens |

- **The retina is evolutionarily fixed** — a 12-channel bank of DoG and
  red/green/blue-yellow color-opponent filters that respond to *contrast, not
  brightness*. It never learns, like the real retina.
- **V1 starts from Gabor priors** (σ = 0.8, λ = 2.5) laid out as 8 orientations × 2
  phases × 2 source channels = 32 filters, with the priors on one source channel and
  random weights on the rest for feature mixing.
- **V4 ends in a CTM**: rather than pooling, a small Continuous Thought Machine runs
  4 ticks of attention over the V4 spatial tokens and emits a single 128-dim
  observation vector — V4 is the brain's strongest attentional hub, so it gets to
  *think* about what it sees.

### How it learns

Two unsupervised/semi-supervised rules drive the cortex, applied with a cascade of
rates (V1 fast, V2 ¾, V4 ½):

- **Hebbian sparse coding** — encode patches, keep the top-K channels per position,
  reconstruct, and push weights toward minimizing reconstruction error.
- **GHL (global-guided Hebbian learning)** — a three-factor rule combining a local
  Oja update with the **sign of the task gradient**. On ResNet-50 / ImageNet this
  closes the gap with backprop to roughly **3%**, where pure local Hebbian collapses
  (>30% gap).

A simple but striking result: adding a **per-token LayerNorm** at V4 (stripping each
token's magnitude) lifts unseen-class k-NN generalization on CIFAR-10 by **~10 points**
and raises the feature effective-rank from 1.7 to 11.2 of 128 — the cortex stops
collapsing everything onto one direction.

### Dreaming

The cortex can run **top-down**: seed V4 with sparse noise and project back through
transposed convolutions (V4ᵀ → V2ᵀ → V1ᵀ → pixels) to hallucinate images — Hoel's
"overfitted brain" hypothesis as code. An `lsd()` mode models **5-HT2A receptor
desensitization** (availability drops with dose, recovers over time) and an
*integration* parameter controls how much of a dream's weight change is kept. The
measured sweet spot is `0.7` (out-of-distribution accuracy stays stable); above `0.95`
the cortex drifts onto a synthetic attractor and OOD collapses — a documented failure
mode, not a metaphor.

## The codecs

The other modalities are discretized by codecs in `modgrad-codec`:

- **VQ-VAE** (image) — a conv encoder maps a 32×32 image to an 8×8 grid of codes
  (**64 codes/image**) from a **4096-entry** codebook, with EMA updates and dead-code
  revival, decoded by a transposed-conv stack.
- **Audio codec** (WavTokenizer-style) — a 1-D conv stack downsamples 24 kHz audio by
  2×4×5×8 = **320×** into a 4096-code book, i.e. **~75 codes/second**.
- **FSQ** — finite scalar quantization: codebook-free, each dimension rounded to a few
  fixed levels with a straight-through estimator, immune to codebook collapse (used
  for both audio and image bottlenecks).
- **Byte n-gram hashing** — instead of BPE, each byte embedding is augmented with
  rolling polynomial hashes of the preceding 3–8 bytes (per-table vocab ~500k),
  normalized so a pure-byte model reaches tokenizer-level features.

`modgrad-data` adds type-safe tokenization and lazy, interleaved, weighted streaming
(e.g. 70% text / 20% image / 10% audio) that never holds more than a chunk in memory.

## The token space

Every codec output is parked at a fixed offset below the base LLM's special tokens
(the Qwen2.5-aligned layout, asserted at compile time in `unified_tokenizer`):

| range | modality |
|-------|----------|
| `0 – 255` | text (raw bytes) |
| `140000 – 140008` | delimiters `<img> </img> <aud> </aud> <vid> </vid>` |
| `140008 – 144104` | image VQ codes (4096) |
| `144104 – 148200` | audio codes (4096) |
| `148200 – 148600` | timestamps (400 ticks, 0.5 s resolution) |
| `148600 – 148878` | actions (mouse, keyboard, coordinates) |

Because it's one vocabulary with one embedding table, a single autoregressive model
learns the dependencies *between* modalities directly — no separate encoders, no
cross-attention bridges.

Next: [mounting a foundation model →](/docs/foundation-models).
