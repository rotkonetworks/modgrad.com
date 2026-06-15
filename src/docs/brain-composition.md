---
title: Brain composition
section: Architecture
order: 40
description: The exact graph — eight regions, the connection and routing mechanism, episodic propagation, auxiliary losses, the NeuralComputer API, and presets from 187k to a billion params.
---

# Brain composition

A single CTM is one region. Graph composition (`RegionalConfig`,
`RegionalWeights`, `regional_train_token`) wires many CTMs into a directed brain.
Each region is its own CTM with its own width, memory depth, and tick budget;
regions exchange signals over learned inter-region synapses.

## The eight regions

The default brain is a cortical loop plus subcortical support. Cortical regions
are wide; subcortical regions are deliberately small.

| region | tier | role |
|--------|------|------|
| **input** | cortex | perception + motor feedback |
| **attention** | cortex | gating, routing |
| **output** | cortex | evidence accumulation |
| **motor** | cortex | action selection (drift-diffusion output) |
| **cerebellum** | subcortex | forward model, timing prediction |
| **basal ganglia** | subcortex | value estimation |
| **insula** | subcortex | interoception, salience |
| **hippocampus** | subcortex | episodic binding |

In the full `eight_region` preset the cortical regions run at `d_model = 512`
with 64 memory slots and the subcortical regions at `d_model = 64`; the
`eight_region_small` preset shrinks these to 32 and 8 respectively, which is the
187k-parameter brain used in the benchmark below.

## Connections and routing

A `Connection` is a directed edge. It concatenates the outputs of its source
regions and applies a learned projection into the destination. The default
topology is a cortical loop (motor → input → attention → output → motor) plus
subcortical edges for value, salience, and an episodic-memory path. You either
hand-write that topology — choosing which regions connect and which receive raw
observations — or enable the learned router.

The **`RegionalRouter`** is a mixture-of-selection mechanism: every tick it
projects each region into a shared routing space, scores affinities, and selects
the top-k sources per destination. On the 8-region brain it adds roughly 40k
parameters, under 0.3% of the model, and replaces fixed wiring with learned,
tick-conditioned routing.

Connections can also select an **observation scale** — V1, V2, or V4 from the
[visual cortex](/docs/multimodal) — so different regions can read different
levels of the visual hierarchy directly.

## Episodic memory and auxiliary losses

The hippocampus output feeds back into the attention region, which makes a
prediction causally dependent on past observations (covered by a dedicated
propagation test). On top of the main objective, an `AuxLossConfig` can switch on
neuroscience-shaped auxiliary losses: cerebellar prediction error, a hippocampal
contrastive term, and a basal-ganglia temporal-difference value loss.

## The NeuralComputer

`NeuralComputer` wraps a trained brain for interactive use. It exposes a
`generate → observe → step` loop and a text `chat(prompt, max_tokens,
temperature)` helper, holding persistent state across turns. A
`forward_cached_frozen` path lets you swap the cerebellum's output for a frozen
LLM while keeping the cortex synapses in the loop — the basis of the
[cerebellum-as-LLM](/docs/foundation-models) design. Run it as a daemon and the
[3D debugger](/docs/runtime-cli) attaches over TCP.

## Presets

All presets live in `RegionalConfig`:

| preset | params | shape | notes |
|--------|--------|-------|-------|
| `four_region` | ~450k | cortical loop only | minimal |
| `eight_region_small` | **187k** | cortex 32 / sub 8, no router | the maze brain |
| `eight_region` | ~81M | cortex 512 / sub 64, router | full |
| `eight_region_medium` | ~55M | cortex 256 / sub 32 | |
| `eight_region_large` | ~200M | cortex 512 / sub 64 | |
| `eight_region_billion` | ~1B | cortex 1024 / sub 128 | needs ~19 GB RAM |

The resident GPU forward becomes profitable around `d_model ≥ 1024`; below that,
dispatch overhead means small brains often run faster on CPU.

## The result that motivated this

On 21×21 maze routing (5000 steps, 3 seeds, held-out 200-maze eval, a fresh
random maze every batch), the 8-region brain beats a single CTM on every metric
with **2.4× fewer parameters**:

| config | params | first-step acc | per-step acc | correct prefix (of 20) |
|--------|--------|---------------|--------------|------------------------|
| single CTM | 450k | 51.3% | 27.4% | 1.2 |
| **8-region brain** | **187k** | **79.2%** | **38.1%** | **2.1** |

The brain's worst seed beats the single CTM's best seed; the ranges don't
overlap. Reproduce:

```bash
cargo run -p mazes --release -- --size 21 --steps 5000 --seed 42
cargo run -p mazes --release -- --brain --size 21 --steps 5000 --seed 42
```

Next: how a brain [learns from its own surprise →](/docs/bio-inspired).
