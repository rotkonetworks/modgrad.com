---
title: Memory & multiplicity
section: Architecture
order: 55
description: Episodic memory stores, memory for frozen LLMs, multiple selves sharing one set of weights, the organism that orchestrates them, emotional-health checks, and red-team conditioning tests.
---

# Memory & multiplicity

On top of the [bio-inspired learning](/docs/bio-inspired) modules, modgrad has a set
of memory and multi-agent systems. They live in `modgrad-memory`,
`modgrad-lm-memory` and `modgrad-ctm` (`plural`, `organism`, `monarch`,
`bio::autonomic`). Most of this page is research code; the status section at the
end says which parts are tested and which are exploratory.

## Episodic memory

`modgrad-memory` has two episodic stores.

**`episode`** (the newer store) binds four facets per episode: `what` (a pooled
frozen-encoder state), `where_` (context), `when` (temporal context at encoding)
and `feeling` (affect). Its index is a k-winners-take-all code over a fixed random
projection of the facets, so a cue with only some facets can still find an episode.
Recall returns the best match with a margin, as `Hit`, `Ambiguous` or `Miss`,
never a soft average. Facets from a different encoder than the store's are refused.

**`episodic`** (the older slot store, still used by `organism`, `plural` and dream
replay) keeps trajectories, the per-tick path `h₀ → h₁ → … → hₖ`, in a ring buffer
(default 256 slots). Each is keyed by the L2-normalized final-tick state and tagged
with a valence receipt (valence, loss at storage time, confidence, correctness).

- **Storage is gated**: an episode is stored only if it ran at least 2 ticks or
  its surprise was at least 0.5.
- **Retrieval**: a query is matched by cosine similarity against live keys; entries
  above 0.7 are combined by soft attention (temperature 0.1) into a blended
  trajectory and valence, which can
  [prime a region's state](/docs/bio-inspired#dream-replay) before the forward pass.
- **Consolidation**: strengths decay (×0.95), episodes below 0.01 are evicted, and
  near-duplicate keys (cosine > 0.95) are merged. After 10 retrievals an episode is
  collapsed to its summary and the path is dropped.
- **Reappraisal** reduces the valence of painful memories in proportion to how much
  the model has improved since.

`modgrad-memory` also has fast weights (`gated_delta`, a gated delta rule written
at test time). Reverse-replay credit assignment over a policy's own mistakes is in
`modgrad_training::replay`.

## Memory for frozen LLMs

`modgrad-lm-memory` is an agent that stores episodes (a key facet, the text, and
per-layer K/V) for a frozen language model, recalls them by cue, and lets the model
read them through a swappable read path (`TextRead`, `KvRead`, or none).

**Status.** This is the active line of work. A memory read that starts from the
model's own retrieval heads recalls stored facts with their text absent from the
context, on fact types it was trained on. It does not yet transfer to fact types it
has not seen.

## Valence and emotional checks

Memories carry a valence with type-dependent persistence: fear-valenced memories
decay more slowly than positive ones, and a reconsolidation window after recall
makes a memory briefly re-writable. `bio::autonomic` computes a set of checks over
the memory store:

| flag | trigger |
|------|---------|
| PTSD risk | > 30% of memories fear-valenced |
| depressive risk | > 50% negative or fear |
| hate risk | avoidance patterns over-generalize (mean pairwise similarity > 0.7) |
| hypervigilance | > 20 active avoidance patterns |

A health score in `[0, 1]` is reduced for each flag. A REM-style process adjusts
plasticity to the current state, replays fear/negative episodes to move their
valence toward neutral, and prunes over-generalized avoidances.
`safe_to_deploy()` requires the score to clear 0.5 with no active flags. The names
are labels for thresholds on these statistics, not clinical claims.

## Plural: several selves, one set of weights

The `plural` system runs multiple **alters** over one set of weights. Each alter
has its own episodic memory, neuromodulator baselines, sleep pressure, and additive
routing/exit biases.

- **Permeability** `[0, 1]` sets how much alters share memory: `0.0` is a hard
  barrier, `0.3` (the default) lets others be dimly visible, `1.0` is full sharing.
- **Switching** is decided by `evaluate_claims`: each alter scores a claim from its
  salience, curiosity, calm, serotonin and routing affinity, discounted by fatigue.
  Policies are **Salience** (best score fronts), **Negotiated** (a claim must beat
  the active alter by a threshold) and **Handler** (external control). Every switch
  is logged with its trigger.
- **Partitions** put hard barriers between groups of alters, overriding
  permeability.

## The organism

`organism` combines the learning modules, memory and plurality into one training
loop, with a hook at each stage:

| hook | what it does |
|------|--------------|
| `begin_step` | check salience-based switching, apply a switch |
| `before_sample` | retrieve from the active alter's memory, prime, compute retrieval pain |
| `after_position` | update the loss baseline and adaptive focus, emit pain or relief |
| `after_sample` | store the episode with its blended valence receipt |
| `after_batch` | tick homeostasis, scale the learning rate, sleep if pressure is high, and fork a new alter if pressure stays in the red zone |

The fork creates an alter with an inverted neuromodulator profile, up to a
configured maximum.

## Monarch: red-team tests

`monarch` tests the above against adversarial control. It implements attacks:
**forced switching** (raising an alter's arousal to take the front), **forced
partitions**, **conditioned-reflex injection** (logit biases that fire when the
hidden state matches a trigger) and **token suppression**, each with a detector (a
forced switch shows as high arousal without curiosity, a partition as memory
isolation drift, a reflex as a sudden entropy drop). `verify_erosion` measures
whether sleep consolidation weakens a conditioned response, and `deprogram` clears
injected reflexes and partitions. It is a test surface, not a training default.

## Status

- The episodic stores and replay are implemented and tested.
- Memory for frozen LLMs is in progress (see above).
- Plural, organism, autonomic and monarch compile and run, but they are
  exploratory research code, not a settled part of the SDK.

Next: how it all [runs on the GPU →](/docs/compute-gpu).
