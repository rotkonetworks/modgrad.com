---
title: Bio-inspired learning
section: Architecture
order: 50
description: The exact mechanics: pain as z-scored surprise, a four-neuromodulator state machine, salience gating, cerebellar forward models, homeostatic sleep pressure, dream replay, and gradient-free spindle-ripple consolidation.
---

# Bio-inspired learning

Beyond plain gradient descent, modgrad ships optional modules that let a brain
**regulate its own learning**. They learn from relative surprise rather than
absolute loss, gate updates by salience, and consolidate offline during sleep.
They live in `modgrad-ctm` (`bio::*`). The core CTM trains without any of them; you
opt in per module. (For episodic memory, multiple selves, and the orchestrator that
ties everything together, see [Memory & multiplicity](/docs/memory).)

## Pain: relative surprise

`bio::pain` keeps an exponential-moving-average **loss baseline** (`α = 0.95`) of
the mean and variance of recent loss. A step's surprise is the loss **z-scored**
against that baseline:

```
surprise = (loss − ema) / sqrt(variance_ema)
```

So loss only hurts when it's *worse than expected*, and beating the baseline
produces relief. Pain is **amplified by confidence**, so being certain and wrong
hurts more (`pain = surprise · scale · (1 + confidence·1.5)`). That is exactly
the signal you want a learner to attend to. A retrieved memory with negative
valence below `−0.3` triggers an **avoidance** response that biases the exit gate
to deliberate longer. The module exposes an adaptive learning-rate multiplier
bounded to **`[0.5, 2.0]`**: a sigmoid of dopamine, curiosity, anxiety, and
serotonin that speeds learning when the brain is curious and slows it when anxious.

## Neuromodulators

`bio::neuromod` is a four-signal state machine, each bounded and decaying toward a
resting baseline so nothing saturates:

| signal | range | resting | driven by |
|--------|-------|---------|-----------|
| **dopamine** | 0.1–3.0 | 1.0 | prediction error / surprise |
| **serotonin** | 0.1–2.0 | 1.0 | learning progress (mood) |
| **norepinephrine** | 0.1–2.0 | 0.5 | arousal / urgency |
| **acetylcholine** | n/a | n/a | attention gain |

**Curiosity** and **anxiety** are derived in an active-inference style: prediction
error × dopamine × (calm) becomes curiosity, while prediction error × dopamine ×
(arousal) becomes anxiety. The same surprise pushes the brain to *explore* when
calm and to *freeze* when aroused, and these levels are what the other modules read
to decide how fast, and whether, to learn.

## Salience gating

`bio::salience` decides **who** learns. It multiplies the reward-prediction error
(`|dopamine − baseline|`) by **motor conflict**, a logistic on the gap between the
top two motor activations, so a close call (ambiguous decision) raises the gate and
a clear winner lowers it:

```
value = rpe · (0.3 + 0.7 · motor_conflict),   gate = value > 0.05 ? value : 0
```

High surprise *and* an unsettled decision means learn hard; a confident decision in
a steady state means skip the update entirely. The gate scales the effective
learning rate in the three-factor rule.

## Three-factor plasticity

`bio::three_factor` is REINFORCE with **Titans-style adaptive eligibility traces**.
The trace accumulates a post × pre product whose decay `η_t` is itself data-dependent
(high salience speeds accumulation, low salience speeds decay); the advantage comes
from a reward baseline; and the salience gate sets the learning rate. Below a
threshold, it returns exactly zero updates, so it never learns from noise.

## Cerebellar forward model

`bio::cerebellar` learns to predict the next state with a **delta rule**, measuring
RMSE between prediction and observation and updating weights scaled by the dopamine
burst (skipping rows whose error is negligible). High prediction error fires a
dopamine burst (surprise); low error lets dopamine decay back toward 1.0.

## Homeostasis and sleep

`bio::homeostasis` folds six signals into a single **sleep-pressure** scalar
(clamped to `[0, 1.5]`):

| component | weight | what it tracks |
|-----------|--------|----------------|
| activation energy | 0.25 | firing intensity (L2 of activations) |
| sync divergence | 0.15 | CTM struggling to reach a decision |
| Hebbian drift | 0.20 | output-weight creep from baseline |
| buffer fill | 0.20 | sensory + replay buffers nearing full |
| emotional pressure | 0.10 | unprocessed fearful memories |
| surprise EMA | 0.10 | sustained high surprise |

The total maps to four zones: **GREEN** (< 0.5, work indefinitely), **YELLOW**
(0.5–0.8), **RED** (0.8–1.0, sleep soon), and **FORCED** (> 1.0, sleep now). Output
quality degrades as pressure rises. Sleep runs two phases that clear
different components: **NREM** drains activation, drift, and buffers; **REM** drains
sync divergence, emotional pressure, and surprise.

## Dream replay

During sleep, `bio::dream` replays the **most painful episodes**. The caller
re-evaluates each with the current weights. If performance improved past +10% and
the answer is now correct it counts as *overcoming* (firing a relief burst); if it
regressed past −10% the pain deepens. Then every painful memory is **reappraised**,
its valence faded in proportion to how much the brain has since improved ("was the
pain justified, given what I know now?"). `prime_state` blends a retrieved trajectory
into a region's state *before* the forward pass, so computation starts from an
episodically-informed prior rather than zero. An adaptive focus weights positions the
brain keeps failing on up to 6× more heavily.

## Consolidation: spindle-ripple

`bio::consolidation` runs **gradient-free SPSA** cycles on the host, mapped onto the
sleep spindle-ripple rhythm: a Rademacher weight perturbation (the spindle), evaluate
loss at `±amplitude` (the ripples), and step a momentum term toward the improvement
(synaptic capture). It is **Pareto-constrained**: the change is capped at ~1% per
sleep and accepted only if the mean improves *and* no stored trace regresses beyond a
1% slack. The brain itself never computes a gradient here; it wakes up with better
weights.

## What's integrated vs. experimental

The cerebellar, salience, neuromodulator, and three-factor modules are integrated and
tested. Pain, dream, homeostasis, and consolidation compile, run, and are wired into
the runtime, but are best treated as research surface rather than settled defaults.

Next: how it all [runs on the GPU →](/docs/compute-gpu), or the
[memory & multiplicity →](/docs/memory) systems built on top.
