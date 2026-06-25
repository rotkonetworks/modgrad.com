---
title: Global-sync readout
section: Architecture
order: 44
description: How the 8-region brain pools its regions into a single decision through a global synchronization read-out — and the honest reason that pooling lets the /play brain predict ~84.5% per move without reliably solving a maze end-to-end.
---

# Global-sync readout

[Brain composition](/docs/brain-composition) covers how the eight regions are wired
and what each one does. This page is about the step that turns all of them into a
**single prediction** each tick — the **global synchronization read-out** — because
it is also the honest explanation for the headline result on [/play](/play): the
brain is right about **~84.5% of individual moves** yet does not reliably solve a
9×9 maze end-to-end.

## What it is

A single [CTM](/docs/continuous-thought-machine) reads out by pairing its own
neurons and accumulating a decaying multiplicative synchronization across ticks.
The brain does the same thing one level up, across **all regions at once**.

Each outer tick, the forward (`regional_forward`) concatenates every region's
activations into one long vector, then for `n_global_sync` learned left/right index
pairs accumulates a decaying product:

```
αᵢ ← decayᵢ · αᵢ + act[leftᵢ] · act[rightᵢ]
βᵢ ← decayᵢ · βᵢ + 1
syncᵢ = αᵢ / sqrt(βᵢ)
```

That global-sync vector is the brain's pooled state. A **single** linear readout
(`output_proj`) maps it to the prediction:

```
prediction = output_proj(global_sync)
```

The first five entries of the prediction are the move logits
(UP / DOWN / LEFT / RIGHT / WAIT); the argmax is the move. The whole prediction is
`route_len × 5` wide — the brain's intended **multi-step route**, which the demo
decodes into the teal target crosshairs.

## Why it matters

The read-out is what makes the brain a brain and not eight independent networks:
the decision is a function of how regions **co-fire**, accumulated over ticks, not
of any one region in isolation. The decaying accumulation is what lets evidence
build across the tick budget before the brain commits.

## How the demo shows it

The right-hand 3D panel and the telemetry card are direct views of this machinery:

- **global sync** meter — the RMS magnitude of the `global_sync` vector at the
  current tick, normalised across the run. It is the pooled signal feeding the
  readout, ticking up as the regions synchronise.
- **region telemetry** — per-region activation RMS and peak at the current tick.
  These are the components that get concatenated and paired into global sync; the
  bars are normalised to the loudest region so they stay comparable.
- **connectome edges** brighten when two regions co-spike — a literal picture of
  the products that feed the global-sync accumulators.

Every number on that card is **derived from the forward**, not invented: region
activations, per-tick global sync, the readout prediction.

## The honest limitation: pooled readout vs. end-to-end solving

A single global readout is a **bottleneck by design**, and on a sequential task it
shows. The brain pools its entire state into one vector and predicts the next move
from it. That is enough to be right about most *individual* moves — hence
**~84.5% per-move** on held-out 9×9 mazes — but per-move accuracy compounds badly
over a full path: one wrong step into a wall stalls the whole rollout, so the
end-to-end solve rate is near zero on 9×9.

This is why the demo does **not** let the brain drive the agent. The agent walks the
true shortest path (BFS, computed in the worker), and the brain **predicts alongside**
it. The move bars show the brain's distribution; the maze flags where its call would
miss (`✕ would hit a wall`, `→ away from goal`); and the running tally reports how
often the brain agreed with the optimal step. Nothing is faked to make the brain look
like it solves the maze — see the [/play framing](/play) for why honest prediction,
not a staged solve, is the point.

A larger preset with wider regions and a learned router (see the presets table in
[brain composition](/docs/brain-composition)) raises these numbers; the small 187k
maze brain is deliberately the one small enough to run in your browser.

Next: how a brain could keep learning while it runs —
[learning at inference →](/docs/learn-at-inference).
