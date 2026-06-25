---
title: Learning at inference
section: Architecture
order: 46
description: Three-factor plasticity — a local eligibility × neuromodulator rule that updates the readout while the model runs, with no backward pass. What the rule is, how the /play wasm exposes it, and an honest note on what the shipped demo does and does not do with it.
---

# Learning at inference

[Bio-inspired learning](/docs/bio-inspired) describes modgrad's full
three-factor plasticity module as a training-time component. This page is about its
narrower, runtime-only cousin: a **local plasticity rule that can update the brain
while it runs**, with no backward pass — and an honest account of how the
[/play demo](/play) exposes it.

## What it is

Backpropagation needs a backward pass and a global gradient. A **three-factor rule**
needs neither. It updates a weight from three locally available quantities:

1. **pre** — the readout's input activations (the eligibility trace);
2. an **error / post** term — here `(onehot − softmax)` over the move logits;
3. a scalar **neuromodulator** — a global `signal`, negative for pain, positive
   for reward.

The demo's wasm exposes exactly this on the brain's output read-out
(`apply_plasticity(chosen, signal)` in `engine/src/brain.rs`):

```
Δwₐⱼ = θ · signal · (onehotₐ − softmaxₐ) · preⱼ
```

for each move logit `a` and readout input `j`, with `θ = 0.02`, a per-element clamp
on `Δw` (±0.25), and a weight clamp (±6.0). The first call snapshots the pristine
readout, so `reset_plasticity()` can restore the as-loaded weights exactly. The
return value is the L2 magnitude of the applied update.

This is the same shape as the training-time rule — eligibility × neuromodulator ×
advantage — collapsed to the single layer (the global-sync read-out) that it is safe
and cheap to nudge online.

## Why it matters

It is the mechanism behind "learn at inference": a brain that, on a surprising or
painful outcome, can shift its own decision boundary *during* a run rather than only
between training epochs. Because the update is local to one layer and gradient-free,
it is bounded, fast, and reversible — properties you want before letting a model
edit its own weights in the field.

## How the demo uses it

Honestly: **the shipped /play demo does not currently learn while playing.** The
wasm exports `apply_plasticity` and `reset_plasticity`, and the worker's message
schema reserves fields for it (`plasticDelta`, `signal` on the step result), but the
worker's engine binding wires up only `run_brain_pixels` and `retina_maps`. No
`signal` is computed from the brain's misses and no plasticity call is made on a
step, so the readout weights are static across a run.

What the demo *does* surface is the raw material such a rule would consume: the
move distribution, and a per-step **verdict** (`ok` / `wall` / `astray` / `wait`)
that classifies the brain's prediction against the true maze. That verdict is
exactly the kind of pain/reward signal you would feed into `apply_plasticity` — a
wall-bound or away-from-goal prediction is negative `signal`, an on-path one is
positive. Wiring that loop in (compute the signal from the verdict, call
`apply_plasticity(chosen, signal)`, surface `plasticDelta`) is the natural next step
for the demo, not something it claims to do today.

## Limitations

- **Readout-only.** The rule here nudges the single global-sync read-out, not the
  regions or the visual cortex. It can re-weight an existing decision; it cannot
  teach the brain a new feature.
- **Not the bio stack.** This is the bare three-factor update. The full system —
  pain as z-scored surprise, the four-neuromodulator state machine, salience gating,
  consolidation during sleep — lives in the SDK ([bio-inspired](/docs/bio-inspired))
  and is **not** present in the browser reimplementation.
- **Off by default in the demo.** As shipped, /play runs a fixed, trained readout.
  Treat learn-at-inference here as an exposed capability of the engine, not an
  active behaviour of the page.

Next: how the brain decides it has thought long enough —
[adaptive compute →](/docs/adaptive-compute).
