---
title: Visual cortex
section: Architecture
order: 42
description: The retina → V1 → V2 → V4 convolutional cascade that turns raw pixels into spatial tokens, and how the /play demo renders what the brain sees.
---

# Visual cortex

The 8-region brain in the [/play demo](/play) does not read the maze as a grid of
integers. It reads it as an **image**, as it did in training, and a small
convolutional **visual cortex** turns that image into the spatial tokens the
cortical regions attend over. This is `modgrad_codec::retina::VisualCortex`; the
browser runs the SDK's own code compiled to WebAssembly.

## What it is

A four-stage convolutional cascade, named after the early visual hierarchy:

```
raw pixels [3 × H × W]  →  retina  →  V1  →  V2  →  V4  →  spatial tokens
```

Each stage is a `Conv2d` (im2col + matmul + bias) followed by a leaky ReLU
(slope 0.1). The final V4 feature map is flattened into a `[n_tokens × token_dim]`
stream, one token per remaining spatial location, `token_dim` channels deep. That
token stream is the **observation** the brain's `input` region embeds.

## Why it matters

The model gets pixels, not a hand-built feature vector. Nothing pre-parses the maze
into walls and openings; whatever structure the brain uses, it had to learn from the
image. The same front-end is not maze-specific.

## How the demo uses it

For each maze, the worker renders the start position to RGB pixels in the training
scheme (`renderPixels` in `src/play/worker.ts`):

| cell | colour |
|------|--------|
| wall | black `(0,0,0)` |
| open | white `(1,1,1)` |
| start (agent) | red `(1,0,0)` |
| goal | green `(0,1,0)` |

The wasm `retina_maps(pixels)` entry point runs the cascade and returns every
layer's feature map (CHW). The demo prepends the literal RGB it fed in, labelled
`sight`, and draws the pathway in the 3D view:

- **what it sees**: the RGB image, in front of the eyes;
- **retina**: the first feature map, drawn in both eyes;
- **V1 → V2 → V4**: each feature map as a cyan response field (brightness = mean
  absolute channel response at that location), placed at the occipital pole and
  along the ventral stream, wired on into the `input` region.

The optic chiasm and LGN drawn along that path are anatomical landmarks only; the
model has no relay stage between retina and V1.

## Limitations

- The cortex takes a **fixed 9×9 input** (`input_h = input_w = 9` in every brain
  export) and reads exactly 3·9·9 pixel values, so the demo runs 9×9 only. A
  smaller image would be read out of bounds, a larger one as the wrong pixels.
- The feature maps are low-resolution; read them as activity fields, not edge
  detectors.
- Channels are averaged per location for display. The fields show *that* a layer
  responds, not what individual filters encode.

Next: how the eight regions consume these tokens and predict a route:
[the 8-region brain →](/docs/brain-composition).
