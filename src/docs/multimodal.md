---
title: Multimodal & token space
section: Systems
order: 70
description: The exact unified vocabulary, the fixed-retina → V1 → V2 → V4 learnable visual cortex, and the VQ-VAE / audio / FSQ / byte-hash codecs.
---

# Multimodal & token space

modgrad treats every modality as tokens in **one vocabulary**, sharing one
embedding table and one next-token objective. There are no per-modality encoders
or cross-attention bridges feeding the brain — just tokens. Text is raw bytes; no
BPE.

## The token space

Each codec's output is parked at a fixed reserved offset, all kept below the base
language model's special tokens (the layout below is Qwen2.5-aligned, and is
asserted at compile time in `unified_tokenizer`):

| range | modality |
|-------|----------|
| `0 – 255` | text (raw bytes) |
| `140000 – 140008` | delimiters: `<img> </img> <aud> </aud> <vid> </vid>` |
| `140008 – 144104` | image VQ codes (4096) |
| `144104 – 148200` | audio codes (4096) |
| `148200 – 148600` | timestamps (400 ticks, 0.5 s resolution) |
| `148600 – 148878` | actions (mouse, keyboard, screen coordinates) |

Everything ends below the Qwen2.5 special-token block (`151643+`). Because it is
one vocabulary, a single autoregressive model learns the dependencies *between*
modalities directly.

## The visual cortex

`VisualRetina` is a biologically-structured front end with a **fixed** retina and
a **learnable** cortex:

- **Retina (fixed)** — difference-of-Gaussian and color-opponent ganglion filters.
- **V1** — oriented edges from Gabor priors: 8 orientations × 2 phases × 2 sources
  (≈32 channels). Learnable.
- **V2** — collinear contours and figure-ground (≈64 channels). Learnable.
- **V4** — 128-dimensional spatial tokens fed to a CTM. Learnable.

V1–V4 train with Hebbian and **global-guided Hebbian learning** (GHL), a
three-factor rule that combines a local Oja update with the global task gradient.
With a Gabor V1 front end, the audit measured roughly +10 points of unseen-class
k-NN generalization on CIFAR-10. (The cortex also supports a "dream synthesis"
mode with modeled receptor desensitization, used for offline augmentation.)

## The codecs

`modgrad-codec` turns raw signals into those tokens:

- **VQ-VAE** — image tokenizer, 4096-entry codebook, an 8×8 grid of codes per
  32×32 patch, EMA codebook updates with dead-code revival.
- **Audio codec** — a WavTokenizer-style convolutional stack with 2×4×5×8 = **320×**
  downsampling and a single 4096-entry codebook, producing **~75 codes/second** at
  24 kHz.
- **FSQ** — finite scalar quantization: codebook-free, each dimension quantized to
  fixed levels with a straight-through estimator, immune to codebook collapse.
- **Byte n-gram hashing** — a BLT-style augmentation that adds rolling polynomial
  hashes of the preceding 3–8 bytes to each byte embedding, so a pure-byte model
  reaches tokenizer-level performance with no BPE.

`modgrad-data` adds type-safe multimodal tokenization, mixed-modality streaming,
and lazy loading on top.

## Status

The codecs, the learnable cortex, and the unified token space are implemented and
exercised by examples (`multimodal_smoke`, `cifar10_probe`, `ocr_smoke`). Combined
end-to-end multimodal *training* is nascent — the pieces exist; wiring them into a
single run is ongoing.

Next: [mounting a foundation model →](/docs/foundation-models).
