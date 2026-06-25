---
title: Adaptive compute
section: Architecture
order: 48
description: The CTM early-exit gate — a learned halt on the synchronization read-out that lets variable-difficulty inputs take variable compute. The exit-λ and ticks-used signals, and how the /play demo surfaces them.
---

# Adaptive compute

A [CTM](/docs/continuous-thought-machine)'s depth is its **tick budget**: it iterates
a single neuron pool over `T` internal ticks before it answers. Adaptive compute is
the brain deciding, per input, **how many of those ticks it actually needs** — easy
inputs halt early, hard ones run longer. The 8-region brain in the
[/play demo](/play) uses the `AdaptiveGate` strategy at both the per-region and the
whole-brain level.

## What it is

`AdaptiveGate` is a small learned linear **halt gate** on the synchronization
read-out. Each tick it produces a logit, squashed to a halting probability:

```
λ = sigmoid(gate(sync))           # per-tick halt probability
p_exit = λ · survival             # probability of stopping exactly now
exit_cdf += p_exit
survival  *= (1 − λ)
```

When the cumulative exit probability crosses the configured `threshold` (near 0.99),
the loop stops and the brain commits. The gate is trained with a KL regulariser and a
per-region β (on the order of 0.05–0.15), so peripheral regions (`input`, `motor`)
are tuned to exit fast while memory-heavy ones (`hippocampus`) deliberate longer —
see the per-region β column in [brain composition](/docs/brain-composition).

The same mechanism runs at the **outer** brain level in `regional_forward`: after the
global-sync read-out each tick, an `outer_exit_gate` produces an exit-λ, accumulates
the same survival CDF, and can break the outer tick loop early. `BrainOut` carries
both `exit_lambda` per tick and `ticks_used` — the count of ticks the brain actually
ran.

## Why it matters

It is how one model spends **variable compute on variable difficulty** without
changing its parameter count. Deeper thought costs more ticks, not more weights, and
the gate makes that cost adaptive instead of fixed. It is also legible: the exit-λ is
a directly inspectable measure of how committed the brain is at each tick.

## How the demo uses it

The demo animates the brain's thinking **tick by tick**, and the telemetry card reads
the gate straight from the trace:

- **tick m/N** — the current animation tick over the brain's tick count.
- **exit λ** — the outer-level `exit_lambda` at the current tick, shown when the
  engine reports one (`curBrainTick().exit`). It is the per-tick halt probability
  described above.
- **ticks used** — `brainTrace().ticksUsed` over the budget — how many ticks the
  brain ran before the gate halted it.

The pulse ring around the agent also scales with `tickIdx / ticksTotal`, so the
on-maze animation paces with the same tick clock.

## Limitations

- On the small 9×9 maze brain, inputs are mostly easy, so the gate typically lets the
  brain run close to its full tick budget; you will not always see a dramatic early
  exit. The signal is real, but the variation is modest at this scale.
- The demo's animation cadence (`TICK_MS`) is a **display** pace chosen for
  watchability, independent of the actual wall-clock cost of a forward pass.
- `exit λ` only appears when the loaded weights include an outer exit gate; without
  one, the field is simply omitted rather than faked.

Next: how the brain pools those ticks into one decision —
[the global-sync read-out →](/docs/global-sync-readout).
