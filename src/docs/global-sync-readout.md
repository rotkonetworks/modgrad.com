---
title: Global-sync readout
section: Architecture
order: 44
description: How the 8-region brain pools its regions into one prediction through a global synchronization read-out, and why getting the first move right is not the same as solving the maze.
---

# Global-sync readout

[Brain composition](/docs/brain-composition) covers how the eight regions are wired
and what each one does. This page covers the step that turns all of them into a
**single prediction** each tick, the **global synchronization read-out**, and what
its numbers on [/play](/play) do and don't mean: the brain usually predicts the
first step of a 9×9 maze correctly, but its full predicted route is almost never
right.

**Correction (2026-07).** An earlier version of this page cited ~84.5% per-move
accuracy and attributed the gap mostly to the pooled readout. A training-loop audit
then found the dominant cause: the batch gradient accumulator silently dropped the
synapse U-Net, NLM stage-2, start-state, decay and gate gradients, so most of the
brain's parameters were never trained. Fixing the accumulator (plus batch averaging
and full-coverage gradient clipping) lifted held-out first-move accuracy from
73.5% to 94.0% (cortex 16) and from 41.5% to 98.0% (cortex 32) at the same step
budget. The pooled-readout bottleneck described below is real, but it is the
second-order effect.

## What it is

A single [CTM](/docs/continuous-thought-machine) reads out by pairing its own
neurons and accumulating a decaying multiplicative synchronization across ticks.
The brain does the same one level up, across all regions at once.

Each outer tick, `regional_forward` concatenates every region's activations into
one vector. For `n_global_sync` index pairs (sampled at init, spanning all regions)
it accumulates a decaying product with a learned per-pair decay:

```
αᵢ ← decayᵢ · αᵢ + act[leftᵢ] · act[rightᵢ]
βᵢ ← decayᵢ · βᵢ + 1
syncᵢ = αᵢ / sqrt(βᵢ)
```

That global-sync vector is the brain's pooled state. One linear readout
(`output_proj`) maps it to the prediction:

```
prediction = output_proj(global_sync)
```

For the maze brain the prediction is `route_len × 5` wide (20 × 5): one group of
UP / DOWN / LEFT / RIGHT / WAIT logits per predicted step. The first group is the
first move.

## Why it matters

The readout makes the decision a function of how regions co-fire, accumulated over
ticks, rather than of any one region. The decay lets evidence build up across the
tick budget before the output is read.

## How the demo shows it

- **global sync** meter: RMS of the `global_sync` vector at the current tick,
  normalised across the run.
- **region telemetry**: per-region activation RMS and peak at the current tick.
  These are the activations that get paired into global sync; bars are normalised
  to the loudest region.
- **connectome edges** brighten when neurons at both ends of a wired region edge
  are active together.

All of these are read from the forward pass: region activations, per-tick global
sync, the readout prediction.

## Limitation: one pooled readout vs. a whole route

A single global readout is a bottleneck, and on a sequential output it shows. The
brain pools its state into one vector and predicts a 20-step route from it.

The 94–98% figures above are first-move accuracy on the training run's held-out
split. Treat them with care: the SDK's 9×9 maze generator produces only a few
hundred distinct mazes, and its held-out 9×9 sets were later found to overlap
training.

Measured on the live demo's path (the shipped wasm and weights, 200 random 9×9
mazes from the page's generator, the final tick's route read from the start image):

| cortex width | first step right | routes correct to the goal |
|---|---|---|
| 16 | 98.5% | 0 / 200 |
| 32 (default) | 199 / 200 | 0 / 200 |
| 64 | 91.5% | 12 / 200 |
| 128 | 87% | 0 / 200 |

For the default brain, on average the first ~4.3 of ~17.4 steps match the shortest
route. Errors compound over the route, and most predictions end in a step into a
wall.

The demo shows exactly that. The brain sees the maze image once, and the page draws
its predicted route tick by tick next to the true shortest route (BFS, computed in
the worker for reference only), marking where the prediction first leaves it and
where it predicts a step into a wall. No planner or agent stands in for the brain.

Next: how a brain could keep learning while it runs:
[learning at inference →](/docs/learn-at-inference).
