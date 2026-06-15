---
title: Bio-inspired learning
section: Architecture
order: 50
description: Exact mechanics of each module — pain as a z-scored surprise, the neuromodulator state machine, salience gating, dream consolidation, SPSA sleep, episodic valence, plural alters, and the organism orchestrator.
---

# Bio-inspired learning

Beyond plain gradient descent, modgrad ships optional modules that let a brain
regulate its own learning. They live in `modgrad-ctm` (`bio::*`, `plural`,
`organism`) and `modgrad-memory`. The core CTM trains without any of them; you
opt in per module.

> The throughline: learn from **relative surprise** rather than absolute loss,
> gate updates by **salience**, and consolidate offline during **sleep**.

## Pain — relative surprise

`bio::pain` keeps an exponential-moving-average **loss baseline** (mean and
variance). A step's "surprise" is the loss **z-scored against that baseline**, so
a loss only hurts when it is worse than expected, and beating the baseline
produces **relief** scaled by how much pain preceded it. The module exposes an
adaptive learning-rate multiplier, `lr_scale()`, bounded to roughly `[0.5, 2.0]`
and centered at 1.0 — a sigmoid of dopamine minus baseline, adjusted by curiosity
and anxiety. High novel surprise speeds learning; thrashing slows it.

## Neuromodulators

`bio::neuromod` is a dopamine / serotonin / norepinephrine / acetylcholine state
machine. Dopamine tracks surprise (reward-prediction error), serotonin tracks
learning progress, norepinephrine tracks arousal. Curiosity and anxiety are
derived in an active-inference style. These levels are what the other modules
read to decide how much, and how fast, to learn.

## Salience gating

`bio::salience` decides **who** learns. It multiplies the reward-prediction error
(`|dopamine − baseline|`) by **motor conflict**: high surprise *and* an unsettled
decision means learn hard; a confident decision in a steady state means skip the
update. This is what turns a uniform gradient step into a targeted one.

## Three-factor plasticity

`bio::three_factor` is REINFORCE with **Titans-style adaptive eligibility
traces**: the trace accumulates a post-synaptic × pre-synaptic product with a
data-dependent decay, the advantage comes from a reward baseline, and the salience
signal gates the effective learning rate.

## Cerebellar forward model

`bio::cerebellar` learns to predict the next state. It measures the RMSE between
its prediction and the observed activations and applies **delta-rule** updates
scaled by the dopamine burst — a fast, local error-correcting loop for timing and
forward modeling.

## Homeostasis and sleep

`bio::homeostasis` folds several signals — activation energy, synchronization
divergence, Hebbian drift, buffer fill, emotional pressure, and a surprise EMA —
into a single **pressure** scalar with GREEN / YELLOW / RED / FORCED zones. Cross
a threshold and the organism sleeps. Sleep runs two offline processes:

- **`bio::dream`** replays the most painful episodes. If performance on the replay
  improves past a margin, it fires an "overcoming" dopamine burst and
  **reappraises** the stored memory's valence; if it regresses, the pain deepens.
  `prime_state` blends a retrieved trajectory into a region's state *before* the
  forward pass, so computation starts from an episodically-informed prior.
- **`bio::consolidation`** runs gradient-free **SPSA "spindle-ripple"** cycles on
  the host: Rademacher weight perturbations, evaluate loss on both sides, step a
  momentum term toward the improvement. The brain itself never computes gradients
  here — it wakes up with better weights.

## Episodic memory

`modgrad-memory` stores trajectories (the per-tick states, certainties, and exit
lambdas) tagged with metadata: the loss at storage time, a confidence, and an
emotional **valence**. Retrieval is cosine similarity on the final-tick state,
blending the trajectories and valences of the top-K matches
(`hippocampus`: O(1) store, soft-attention-weighted recall, strength decay per
read). A surprise-gated `replay` buffer only admits moments above a surprise
threshold and evicts the least surprising first. Valence carries
consolidation dynamics — fearful memories decay far slower than positive ones,
with a reconsolidation window after recall.

## Plural — many minds, one brain

`plural` runs multiple **alters** over one set of weights, each with its own
episodic memory, neuromodulator baselines, and routing biases. A permeability
parameter ranges from `0.0` (amnesic barrier) through `0.3` (co-conscious) to
`1.0` (full blending). Switching is decided by an `evaluate_claims` score over
salience, curiosity, calm, energy, and routing affinity, under Salience,
Negotiated, or Handler policies.

## Organism — the orchestrator

`organism` ties the rest into one training lifecycle, with hooks at each stage:
prime from memory **before** a sample, compute pain **per position**, store the
episode **after** a sample, then **after** a batch scale the learning rate, check
sleep pressure, and split off a new alter if pressure stays in the red. It is
wired into the isis runtime as `OrganismNC`.

## What's integrated vs. experimental

The cerebellar, salience, neuromodulator, and three-factor modules are integrated
and tested. Pain, dream, homeostasis, consolidation, episodic memory, plural, and
the organism compile, run, and are wired into the runtime, but are best treated
as research surface rather than settled defaults.

Next: how it all [runs on the GPU →](/docs/compute-gpu).
