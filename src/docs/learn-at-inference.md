---
title: Learning at inference
section: Architecture
order: 46
description: A three-factor plasticity rule (eligibility × neuromodulator) that updates a readout while the model runs, without a backward pass. What the rule is, where it is implemented, and the fact that the /play demo does not use it.
---

# Learning at inference

[Bio-inspired learning](/docs/bio-inspired) describes modgrad's three-factor
plasticity as a training-time component. This page covers a narrower runtime
variant: a **local rule that updates one layer while the model runs**, with no
backward pass. The [/play demo](/play) does not run it; see below.

## What it is

Backpropagation needs a backward pass and a global gradient. A **three-factor
rule** updates a weight from three locally available quantities:

1. **pre**: the readout's input activations (the eligibility trace);
2. an **error / post** term: `(onehot − softmax)` over the move logits;
3. a scalar **neuromodulator** `signal`: negative for pain, positive for reward.

A version of this on the brain's global-sync readout is implemented as
`apply_plasticity(chosen, signal)` in this site repo's earlier browser engine
(`engine/src/brain.rs`):

```
Δwₐⱼ = θ · signal · (onehotₐ − softmaxₐ) · preⱼ
```

for each move logit `a` and readout input `j`, with `θ = 0.02`, a per-element clamp
on `Δw` of ±0.25 and a weight clamp of ±6.0. The first call snapshots the readout,
so `reset_plasticity()` restores the loaded weights exactly. It returns the L2 norm
of the applied update.

Because the update touches one layer, needs no gradient through the network and is
clamped, it is bounded and reversible.

## What the demo does

**The shipped /play demo does not learn.** It runs the brain once per maze on fixed,
trained weights and draws the route it predicts. The worker binds only
`load_brain_weights`, `run_brain_pixels` and `retina_maps`. The engine it loads is
the SDK's `modgrad-wasm` crate, where `apply_plasticity` / `reset_plasticity` are
currently no-ops kept for compatibility; the rule above lives in the older
`engine/` crate, which the page no longer uses.

What the page does show is what such a rule would consume: the brain's first-step
distribution at every tick, and where its predicted route leaves the shortest route
or runs into a wall. Turning that into a `signal` is not done.

## Limitations

- **Readout-only.** The rule re-weights an existing decision. It cannot teach the
  regions or the visual cortex a new feature.
- **Not the bio stack.** The SDK's fuller system (pain as z-scored surprise,
  neuromodulator state, salience gating, sleep consolidation; see
  [bio-inspired](/docs/bio-inspired)) is not in the browser engine.

Next: how the brain decides it has thought long enough:
[adaptive compute →](/docs/adaptive-compute).
