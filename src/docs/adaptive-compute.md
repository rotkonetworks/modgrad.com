---
title: Adaptive compute
section: Architecture
order: 48
description: The CTM early-exit gate, a learned halt on the synchronization read-out that lets an input use fewer ticks than the budget. The exit-λ and ticks-used signals, and how /play shows them.
---

# Adaptive compute

A [CTM](/docs/continuous-thought-machine)'s depth is its **tick budget**: it iterates
a single neuron pool over `T` internal ticks before it answers. Adaptive compute lets
the model stop before the budget on a given input. The 8-region brains served on
[/play](/play) are exported with an `AdaptiveGate` exit strategy on their regions and
an outer, whole-brain exit gate.

## What it is

`AdaptiveGate` is a small learned linear **halt gate** on the synchronization
read-out. Each tick it produces a logit, squashed to a halting probability:

```
λ = sigmoid(gate(sync))           # per-tick halt probability
p_exit = λ · survival             # probability of stopping exactly now
exit_cdf += p_exit
survival  *= (1 − λ)
```

When the cumulative exit probability crosses the configured `threshold` (0.99 in
the 8-region presets), the loop stops. The gate is trained with a KL regulariser
and a per-region β (0.05–0.15 in those presets); see the per-region β column in
[brain composition](/docs/brain-composition).

The same mechanism runs at the **outer** brain level in `regional_forward`: after the
global-sync read-out each tick, an `outer_exit_gate` produces an exit-λ, accumulates
the same survival CDF, and can end the outer tick loop early. `BrainOut` carries
`exit_lambda` per tick and `ticks_used`, the number of ticks the brain actually ran.

## Why it is there

The cost of thinking is ticks, not parameters, and the gate makes that cost depend
on the input instead of being fixed. The exit-λ is also directly inspectable at
every tick.

## What /play shows

/play reveals the brain's ticks one at a time, and the telemetry reads the gate
from the trace:

- **tick m/N**: the tick shown, over the ticks the brain actually ran for this
  maze. The tick slider scrubs through them.
- **exit λ**: the outer-level `exit_lambda` at the tick shown, when the trace
  reports one.
- **ticks used**: `ticksUsed` against the 16-tick budget.

In the exported reference traces, ticks used vary from 2 to 16 of 16 depending on
the input and the brain width, so early exit does happen on these mazes.

## Limitations

- The 9×9 mazes are a narrow input distribution; whether ticks used track how hard
  a maze is has not been measured.
- The reveal pace on /play is a display choice, unrelated to the wall-clock cost of
  a forward pass.
- `exit λ` only appears when the loaded weights include an outer exit gate;
  otherwise the field is omitted.

Next: how the brain pools its regions into one prediction:
[the global-sync read-out →](/docs/global-sync-readout).
