---
title: Memory & multiplicity
section: Architecture
order: 55
description: Episodic memory with emotional valence, content-addressable recall, multiple selves sharing one brain, the organism that orchestrates them, emotional-health vital signs, and red-team conditioning defenses.
---

# Memory & multiplicity

On top of the [bio-inspired learning](/docs/bio-inspired) modules, modgrad builds a
genuinely unusual set of systems: a brain that remembers episodes with how they
*felt*, that can host **several selves on one set of weights**, and that monitors its
own emotional health. These live in `modgrad-memory` and `modgrad-ctm` (`plural`,
`organism`, `monarch`, `bio::autonomic`).

## Episodic memory

`modgrad-memory` stores trajectories — the full per-tick path `h₀ → h₁ → … → hₖ`,
not just the answer — in a ring buffer (default 256 slots), each keyed by the
L2-normalized final-tick state and tagged with a **valence receipt** (valence, the
loss at storage time, confidence, whether it was correct).

- **Storage is gated.** An episode is stored only if it ran long enough
  (`≥ 2` ticks) or was surprising enough (`surprise ≥ 0.5`) — trivial moments are
  dropped.
- **Retrieval is content-addressable.** A query is matched by cosine similarity
  against all live keys; entries above a `0.7` threshold are combined by a sharp
  soft-attention (`temperature 0.1`) into a **blended trajectory and blended
  valence**. That blend can [prime a region's state](/docs/bio-inspired#dream-replay)
  before the forward pass.
- **Consolidation** runs during sleep: strengths decay (`×0.95`), weak episodes
  (`< 0.01`) are evicted, and near-duplicate keys (`cosine > 0.95`) are merged into
  the stronger one. After ~10 retrievals an episode undergoes **semantic collapse** —
  the path is discarded and only the summary kept.
- **Reappraisal** fades the valence of painful memories in proportion to how much
  the brain has since improved — sleep as therapy.

A faster **hippocampal CAM** (`memory::hippocampus`) gives O(1) store / O(capacity)
recall with the same cosine soft-attention for single-shot binding, and a
surprise-gated **replay buffer** (`memory::replay`) keeps only high-surprise
experiences, evicting the least surprising first so sleep replays what's worth
revisiting.

## Valence and emotional decay

Memories carry an emotional **valence** with type-dependent persistence — fearful
memories decay far slower than positive ones, and a reconsolidation window after
recall makes a memory briefly labile (re-writable). `bio::autonomic` turns this into
**emotional vital signs**:

| diagnosis | trigger |
|-----------|---------|
| PTSD risk | > 30% of memories fear-valenced |
| depressive risk | > 50% negative/fear |
| hate risk | avoidance patterns over-generalize (mean pairwise similarity > 0.7) |
| hypervigilance | > 20 active avoidance patterns |

A health score in `[0, 1]` deducts for each pathology, and a **subconscious REM**
process auto-tunes plasticity to the brain's state (aggressive when fearful, gentle
when healthy), replays fear/negative episodes to shift their valence toward neutral,
and prunes over-generalized avoidances before they snowball. The brain is
"safe to deploy" only when its health score clears 0.5 with no active diagnoses.

## Plural — many selves, one brain

The `plural` system runs **multiple alters** over one set of weights. Each alter has
its own private episodic memory, its own neuromodulator baselines (temperament), its
own sleep pressure, and additive routing/exit biases — a distinct personality on a
shared substrate.

- **Permeability** `[0, 1]` controls how much alters share memory: `0.0` is a hard
  amnesic barrier (alters invisible to each other), `0.3` is co-conscious (the
  default — others are dimly perceptible), `1.0` is full integration.
- **Switching** is decided by `evaluate_claims`: each alter scores a claim from its
  own salience, curiosity, calm, serotonin, and routing affinity, discounted by
  fatigue. Policies are **Salience** (best fitness fronts), **Negotiated** (a claim
  must beat the active one by a threshold), or **Handler** (external control). Every
  switch is logged with its trigger and the pressure at the time.
- **Partitions** install hard amnesic barriers between groups of alters, overriding
  permeability.

## The organism — orchestrator

`organism` ties the learning modules, memory, and plurality into one training
lifecycle, with a hook at each stage:

| hook | what it does |
|------|--------------|
| `begin_step` | check salience-based switching, apply a plural switch |
| `before_sample` | retrieve from the active alter's memory, prime, compute retrieval pain |
| `after_position` | update the loss baseline + adaptive focus, fire pain or relief |
| `after_sample` | store the episode with its blended valence receipt |
| `after_batch` | tick homeostasis, scale the learning rate, sleep if pressured, **split a new alter** if pressure stays in the red |

That last step is the striking one: under sustained red-zone pressure the organism
**spontaneously forks a new alter** with an inverted neuromodulator profile, up to a
configured maximum — multiplicity as a stress response.

## Monarch — red-team and defenses

`monarch` exists to test the above against adversarial control. It implements attack
primitives — **forced switching** (spiking an alter's arousal to seize the front),
**forced partitions** (amnesic walls), **conditioned-reflex injection** (logit
biases that fire when the hidden state matches a trigger), and **token suppression** —
each paired with a detector: a forced switch shows as high arousal without curiosity,
a partition as memory-isolation drift, a reflex as a sudden entropy drop. A
`verify_erosion` check measures whether sleep consolidation *weakens* a conditioned
response, and `deprogram` clears injected reflexes and partitions. It's a defensive
research surface, not a default training path.

## Status

Episodic memory, the hippocampal CAM, and the replay buffer are working and tested.
Plural, organism, autonomic, and monarch compile and run and are wired into the
runtime (`OrganismNC`), but are explicitly research surface — the interesting,
exploratory edge of the project rather than its settled core.

Next: how it all [runs on the GPU →](/docs/compute-gpu).
