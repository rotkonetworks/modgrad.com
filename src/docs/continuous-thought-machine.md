---
title: Continuous Thought Machine
section: Architecture
order: 30
description: The mechanics of a CTM in modgrad. The per-tick loop, U-Net synapse, non-local memory, synchronization read-out, learned early exit, the loss blend, and backpropagation through time.
---

# Continuous Thought Machine

The Continuous Thought Machine (CTM) is the core of modgrad: a recurrent
architecture whose depth comes from **internal ticks**, not stacked layers. A
single neuron pool of width `d_model` is iterated over `T` ticks before it
commits to an answer. It is a Rust port of the
[Sakana AI CTM](https://arxiv.org/abs/2505.05522), following the reference
`ContinuousThoughtMachine.forward()` and `SynapseUNET`, with gradients through
every tick.

## The tick loop

Each forward pass runs the same body `T` times. At tick `t`, the pool holds a
post-activation vector; the model then:

1. **Attends.** A query is projected from the current activations and runs
   multi-head attention over the (layer-normed) input tokens, returning context.
2. **Synapses.** The context is concatenated with the activations and passed
   through the **U-Net synapse**, producing the next pre-activations for every
   neuron.
3. **Remembers.** Each neuron appends its new pre-activation to an `M`-length
   **non-local memory** trace, and a per-neuron network maps that trace to the
   neuron's next post-activation.
4. **Synchronizes.** A **synchronization** read-out pairs neurons and accumulates
   a decaying dot-product over ticks, yielding the vectors used for output and
   for action/attention queries.
5. **Predicts.** The output synchronization is projected to a prediction and a
   certainty for this tick.

Because the loop is recurrent over a single pool, more thinking costs more
ticks, not more parameters.

## The U-Net synapse

Inter-neuron mixing is not a single linear layer. The synapse (`SynapseUNet`) is
a skip-connected U-Net whose intermediate widths follow a `linspace` bottleneck,
so the signal is compressed and re-expanded with a residual skip at each level.

## Non-local memory (NLM)

Every neuron keeps a sliding window of its last `memory_length` pre-activations.
A small per-neuron network turns that history into the neuron's next state. Two
variants ship:

- **deep**: a two-stage SuperLinear map `M → H → 2` with GLU gating;
- **shallow**: a single `M → 2` map.

So a neuron's output depends on its own recent trajectory, not only on the
current input.

## Synchronization read-out

The CTM's output is not the raw activations. Neurons are paired by random left/
right index sets, and for each pair the model accumulates a **decaying
multiplicative synchronization** across ticks (an `α ⊙ β` term with a learnable
per-pair decay). Two synchronization vectors are produced: one drives the output
projection, one drives the action/attention query. The 8-region brains on
[/play](/play) read out 96 to 256 global sync pairs, depending on width.

## Adaptive early exit

A CTM can stop before its tick budget. `CtmConfig::exit_strategy` selects one of
three modes:

| strategy | behavior |
|----------|----------|
| **None** | run all `T` ticks |
| **Certainty** | halt when an entropy-derived certainty exceeds a threshold |
| **AdaptiveGate** | a learned linear halt gate on the output synchronization, regularized with a KL term; per-region β of 0.05–0.15 and a threshold of 0.99 in the 8-region presets |

Details in [adaptive compute](/docs/adaptive-compute).

## Loss

The default `CtmLoss` does not take the last tick. It blends the **lowest-loss
tick** with the **most-certain tick** (50/50), so the model is trained both to
be right at some tick and to be confident when it is right. Two optional losses
extend this:

- **ThinkingLoss** rewards improvement across ticks;
- an **imagination** loss splits the tick budget into a silent "imagination"
  phase and a graded "commit" phase.

## Data structures

A CTM is a handful of explicit, serializable structs:

| type | role |
|------|------|
| `CtmConfig` | hyperparameters: `iterations`/ticks, `d_model`, `memory_length`, synapse depth, exit strategy |
| `CtmWeights` | all trainable parameters: synapse U-Net blocks, NLM stages, query/KV projections + LayerNorm, MHA in/out, sync decays, output projection, optional exit gate |
| `CtmState` | per-forward state: activations, NLM history, sync accumulators, optional episodic ring buffer |
| `CtmCache` | tick-by-tick intermediates kept for the backward pass |
| `CtmGradients` | accumulated gradients, mirroring `CtmWeights` |

For inspection, `ctm_forward_with_attn_trace` returns the attention weights for
every tick and head (`[ticks][heads][tokens]`).

## Backpropagation through time

Training unrolls all `T` ticks and computes gradients for every weight: the
U-Net blocks, both NLM stages, the sync decays, the attention projections, the
output projection, and the exit gate. The unrolled backward is checked against
finite differences in the test suite. Hot-path operations (matmul, layer norm,
SiLU/GLU backward, outer products) go through `modgrad-device`, so the same
forward and backward run on CPU or a GPU backend.

## Related: the CMM

`modgrad-ctm` also contains a CTM core with a pluggable working memory, after
the Continuous Memory Machine (CMM; Regan et al., Sakana, 2026), in
`modgrad_ctm::cmm`.

Next: compose many CTMs into a [multi-region brain →](/docs/brain-composition).
