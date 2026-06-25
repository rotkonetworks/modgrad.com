---
title: Visual cortex
section: Architecture
order: 42
description: The retina that turns raw pixels into spatial tokens — a retina → V1 → V2 → V4 convolutional cascade — and how the /play demo renders what the brain actually sees at each maze cell.
---

# Visual cortex

The 8-region brain in the [/play demo](/play) does not read the maze as a grid of
integers. It reads it as an **image**, the same way the trained model was fed during
training, and a small convolutional **visual cortex** turns that image into the
spatial tokens the cortical regions attend over. The component is a faithful port of
`modgrad_codec::retina::VisualCortex`, and the demo's wasm reimplementation matches
it bit-for-bit.

## What it is

The cortex is a four-stage convolutional cascade, named after the early
visual-processing hierarchy:

```
raw pixels [3 × H × W]  →  retina  →  V1  →  V2  →  V4  →  spatial tokens
```

Each stage is a `Conv2d` (im2col + matmul + bias) followed by a leaky ReLU
(slope 0.1), exactly mirroring `retina::leaky_relu`. The final V4 feature map is
flattened into a `[n_tokens × token_dim]` stream — one token per surviving spatial
location, `token_dim` channels deep. That token stream is the **observation** the
brain's `input` region embeds; the inter-region synapses read only their own
`in_dim` prefix of it, exactly as on the device.

## Why it matters

Feeding the model pixels rather than a hand-built feature vector is what makes the
demo an honest test of the architecture. Nothing pre-parses the maze into "wall
here, opening there"; the brain has to **see** structure the way it would have to
see any visual input. The same retina front-end is what lets the architecture
generalise beyond mazes — the cortex is a reusable perception stage, not a
maze-specific encoder.

## How the demo uses it

At every cell the agent occupies, the worker renders the maze to RGB pixels in the
exact training scheme (`renderPixels` in `src/play/worker.ts`):

| cell | colour |
|------|--------|
| wall | black `(0,0,0)` |
| open | white `(1,1,1)` |
| agent | red `(1,0,0)` |
| goal | green `(0,1,0)` |

The wasm `retina_maps(pixels)` entry point runs the full cascade and returns every
layer's feature map (CHW). The demo prepends the literal RGB it fed in — labelled
`sight` — so the panel shows the whole pathway:

- **what it sees** — the framed RGB image (the eye's "screen"), front-most;
- **retina → V1 → V2 → V4** — each learned feature map as a cyan response field,
  brightness = mean absolute channel response at that location;
- a feedforward beam wiring those layers into the `input` region of the 3D brain,
  so you watch perception flow into cognition.

The same feature maps are also collapsed into a single saliency grid
(`visionSaliency`) and upsampled back onto the maze — a coarse "where the visual
system is responding" overlay, distinct from the occlusion-attention heatmap.

## Limitations

- The cortex used here is the **small** maze brain's, trained on 9×9 mazes; the
  feature maps are correspondingly low-resolution and read best as activity fields,
  not crisp edge detectors.
- The cyan response fields are a **visualisation** — channels are averaged per
  location for display. They show *that* a layer responds, not the per-channel
  filters individually.
- The saliency overlay is nearest-neighbour upsampled from a smaller feature map,
  so it localises to roughly the right region of the maze, not to the exact cell.

Next: how the eight regions consume these tokens and decide a move —
[the 8-region brain →](/docs/brain-composition).
