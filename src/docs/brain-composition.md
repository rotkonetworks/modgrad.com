---
title: Brain composition
section: Architecture
order: 40
description: Wiring CTMs into a directed graph of eight brain regions. The closed cortical loop, each region, inter-region synapses, learned routing, presets, and where this design stands now.
---

# Brain composition

A single [CTM](/docs/continuous-thought-machine) is one region. Graph composition
(`RegionalConfig`, `RegionalWeights`, `regional_train_token`) wires many of them
into a directed brain. Each region is its own CTM, with its own neuron pool, memory
depth, tick budget, and exit gate, and they exchange signals over learned
inter-region synapses.

## The closed cortical loop

The cortical regions form a **loop**, not a feedforward stack:

```
input → attention → output → motor
  ↑                            │
  └────────────────────────────┘   (motor feeds back into input)
```

The motor region's action is fed back into the input region alongside the next
observation, so perception is **action-conditioned**: the brain can predict the
next observation given its own action. This mirrors the efference-copy motif in
biology.

Around that loop sit the subcortical regions and an episodic memory:

```
motor ──(+obs)──→ cerebellum            output ──→ basal ganglia
{input,attention,output,motor} ──→ hippocampus ──→ insula
hippocampus ──→ attention               (memory-guided attention)
```

## The eight regions

Cortical regions are wide; subcortical regions are deliberately small. Memory
depth and the exit-gate β are tuned per region: fast peripheral regions exit
early, memory-heavy regions deliberate longer.

| region | tier | width (full / small) | memory | β (exit) |
|--------|------|----------------------|--------|----------|
| input | cortex | 512 / 32 | 64 / 8 | 0.05 (fast) |
| attention | cortex | 512 / 32 | 64 / 8 | 0.10 |
| output | cortex | 512 / 32 | 64 / 8 | 0.10 |
| motor | cortex | 512 / 32 | 64 / 8 | 0.05 (fast) |
| cerebellum | subcortex | 64 / 8 | 32 / 4 | 0.05 |
| basal ganglia | subcortex | 64 / 8 | 32 / 4 | 0.10 |
| insula | subcortex | 64 / 8 | 32 / 4 | 0.05 |
| hippocampus | memory | 64 / 8 | 64 / 8 | 0.15 (slowest) |

### input

Embeds the raw observation and receives motor's action feedback, closing the loop.
A fast, peripheral region. **In:** motor (+ observation). **Out:** attention.

### attention

Selects what's task-relevant and routes it toward a decision. In the full brain it
also reads the hippocampus, so **past context shapes what the cortex attends to**.
**In:** input (+ hippocampus). **Out:** output.

### output

Integrates attention's routing over many ticks to pick the next action or token.
It is the longest-deliberating cortical region. **In:** attention. **Out:** motor, basal
ganglia.

### motor

Emits the action and closes the cortical loop by feeding it back to input. It also
carries the action plus the observation to the cerebellum, the seam where a forward
model, or a mounted LLM, sees both the action and the world. **In:** output.
**Out:** input (+ obs), cerebellum (+ obs).

### cerebellum

In the small brain, a tiny CTM trained as a forward model that predicts the next
observation. In the [target architecture](/docs/foundation-models) this is where a
**frozen LLM mounts** (`FrozenCerebellum`), taking ~82% of the parameter budget.
**In:** motor (+ obs).

### basal ganglia

A value head: estimates the expected future loss of the current state and gates the
output action. **In:** output.

### insula

Monitors internal state, biased by hippocampal memory, to weight what matters now.
**In:** hippocampus.

### hippocampus

Reads the activations of all four cortical regions with the longest memory in the
brain, and feeds back into attention, so stored context can influence later
predictions. **In:** input, attention, output, motor. **Out:** insula (and attention).

## Planning, distributed across regions

The SDK can also run an exact grid planner inside the region graph, split the way
model-based planning is often described in the brain (hippocampal map and replay,
striatal value, motor readout). Its value iteration is divided across three regions
(`region_plugins`, keyed by region index):

- **basal ganglia**: the value head, a per-cell reward and value estimate.
- **hippocampus**: a per-cell traversability gate and the propagation of value
  across it, `K` rounds of a 3×3 Bellman backup.
- **motor**: the readout, gathered at the agent's own cell rather than pooled. In
  our probes, wall information that was linearly decodable (~94%) from the agent's
  own grid-cell token fell to near chance after pooling, so the move is read where
  the agent stands.

The three heads (`BgValueHead` / `HippoGateHead` / `MotorHead` in `modgrad-ctm`)
are initialised by splitting a trained standalone Value Iteration Network
(Tamar et al., 2016) with `split_vin`, so they reproduce it bit-for-bit. Each is a small bundle of `Linear`s; the generic per-region weights
stay task-agnostic, with the planning circuit owned at the composition layer
beside `connections`/`router`. Today `plan()` composes the three heads to run
the sweeps outside the regions' tick dynamics; running them *as* the hippocampus
region's own ticks — value flowing through a basal-ganglia↔hippocampus edge each
tick — is not done. This planner is a fixed algorithm with learned parts, not
something the CTM learned to do. The [/play demo](/play) does **not** use it: there,
the brain's own output head predicts the route from the maze image and nothing
else plans or decides.

## Inter-region synapses

A `Connection` concatenates the activations of its source regions (optionally
appending the raw observation), projects them through a learned linear "synapse"
into the destination's input width, and, when several connections target one
region, **sums** the projected inputs element-wise. So the hippocampus can
receive from four cortical regions without changing its neuron count, and each edge
trains independently.

## Routing: fixed or learned

You either hand-write the topology above, or switch on the **`RegionalRouter`**: a
thalamus-style mechanism that, every tick, scores region-to-region affinity from
the global sync state plus a tick embedding and selects the **top-k sources per
destination** (k = 3, with a little ε-greedy exploration). On the 8-region brain
it adds roughly 40k parameters, under 0.3%, and replaces fixed wiring with
learned, tick-conditioned routing. A bug fixed in 2026-10 made router brains ignore
their observation entirely; results obtained with the router before that fix should
be re-measured before they are relied on.

## Auxiliary losses

Optional neuroscience-shaped objectives (`AuxLossConfig`, off by default) teach the
subcortical regions their jobs: **cerebellar prediction error** (MSE of the
cerebellum's predicted next observation), **hippocampal contrastive** (a variance
term that stops episodic codes from collapsing), and **basal-ganglia temporal
difference** (a value head trained against realized loss).

## Presets

| preset | params | shape | notes |
|--------|--------|-------|-------|
| `four_region` | ~450k | cortical loop only | minimal |
| `eight_region_small` | **187k** | cortex 32 / sub 8, no router | the maze brain |
| `eight_region` | ~81M | cortex 512 / sub 64, router | full |
| `eight_region_medium` | ~55M | cortex 256 / sub 32 | |
| `eight_region_large` | ~200M | cortex 512 / sub 64 | |
| `eight_region_billion` | ~1B | cortex 1024 / sub 128 | ~19 GB RAM |

## The result that motivated this, and where it stands

On 21×21 maze routing (5000 steps, 3 seeds, held-out 200-maze eval, a fresh random
maze every batch), the 8-region brain beats a single CTM on every metric with
**2.4× fewer parameters**:

| config | params | first-step acc | per-step acc | correct prefix (of 20) |
|--------|--------|---------------|--------------|------------------------|
| single CTM | 450k | 51.3% | 27.4% | 1.2 |
| **8-region brain** | **187k** | **79.2%** | **38.1%** | **2.1** |

The brain's worst seed beat the single CTM's best seed.

Two later findings qualify this. First, until a 2026-10 fix, every region's
synchronization pairs were self-pairs (a random-number bug), so these runs had no
cross-neuron synchronization; they have not been repeated since. Second, a
pre-registered, parameter-matched comparison on a two-stream composition task (3
seeds) found **no difference** between a 3-block graph and a single block (96.5 vs
99.8 and 23.4 vs 20.5 on two versions). New work therefore uses a single CTM (or
CMM) block, and a multi-region graph is added only if it wins a new pre-registered
comparison. The /play brain is the older 8-region graph.

```bash
cargo run -p mazes --release -- --size 21 --steps 5000 --seed 42
cargo run -p mazes --release -- --brain --size 21 --steps 5000 --seed 42
```

## The NeuralComputer

`NeuralComputer` wraps a trained brain for interactive use: a `generate →
observe → step` loop and a text `chat(prompt, max_tokens, temperature)` helper that
holds state across turns. Run it as a daemon and the
[3D debugger](/docs/runtime-cli) attaches over TCP.

Next: how a brain [learns from its own surprise →](/docs/bio-inspired).
