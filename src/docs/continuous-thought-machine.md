---
title: Continuous Thought Machine
section: Architecture
order: 30
description: The exact mechanics of a CTM. The per-tick loop, U-Net synapse, non-local memory, synchronization read-out, learned early-exit, the loss blend, and full BPTT.
---

# Continuous Thought Machine

The Continuous Thought Machine (CTM) is the core of modgrad: a recurrent
architecture whose depth comes from **thinking time**, not stacked layers. A
single shared neuron pool of width `d_model` is iterated over `T` internal ticks,
accumulating evidence before it commits to an answer. It is a faithful Rust port
of the [Sakana AI CTM](https://arxiv.org/abs/2505.05522). The forward pass
matches `ContinuousThoughtMachine.forward()` and the synapse matches `SynapseUNET`
from the reference implementation, with full backpropagation through time.

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

Because the loop is recurrent over a single pool, a deeper "thought" costs more
ticks, not more parameters.

## The U-Net synapse

Inter-neuron mixing is not a single linear layer. The synapse is a
skip-connected **U-Net** whose intermediate widths are laid out with a `linspace`
bottleneck (`SynapseUNet`), so information is compressed and re-expanded with
residual skips at each level. This component gives a CTM its
expressivity per tick.

## Non-local memory (NLM)

Every neuron keeps a sliding window of its last `memory_length` pre-activations.
A small per-neuron network turns that history into the neuron's next state. Two
variants ship:

- **deep**: a two-stage SuperLinear map `M → H → 2` with GLU gating;
- **shallow**: a single `M → 2` map.

NLM lets a neuron's behavior depend on its own recent trajectory rather
than only the instantaneous input.

## Synchronization read-out

The CTM's output is not the raw activations. Neurons are paired by random left/
right index sets, and for each pair the model accumulates a **decaying
multiplicative synchronization** across ticks (an `α ⊙ β` term with a learnable
per-pair decay). Two synchronization vectors are produced: one drives the output
projection, one drives the action/attention query. On the 8-region brain this
scales to on the order of a thousand sync pairs.

## Adaptive early-exit

A CTM can stop thinking when it is confident. `CtmConfig::exit_strategy` selects
one of three modes:

| strategy | behavior |
|----------|----------|
| **None** | run all `T` ticks |
| **Certainty** | halt when the prediction's entropy drops below a threshold |
| **AdaptiveGate** | a learned linear halt gate on the output synchronization, regularized with a KL term and tuned per region (β on the order of 0.05–0.15, threshold near 0.99) |

This is how variable-difficulty inputs receive variable compute.

## Loss

The default `CtmLoss` does not take the last tick. It blends the **lowest-loss
tick** with the **most-certain tick** (a 50/50 combination), so the model is
trained both to be right at some point and to know when it is right. Two optional
losses extend this:

- **ThinkingLoss** rewards monotonic improvement across ticks;
- an **imagination** loss splits the tick budget into a silent "imagination"
  phase and a graded "commit" phase.

## Data structures

A CTM is a handful of explicit, serializable structs, with no hidden state:

| type | role |
|------|------|
| `CtmConfig` | hyperparameters: `iterations`/ticks, `d_model`, `memory_length`, synapse depth, exit strategy |
| `CtmWeights` | all trainable parameters: synapse U-Net blocks, NLM stages, query/KV projections + LayerNorm, MHA in/out, sync decays, output projection, optional exit gate |
| `CtmState` | per-forward ephemeral state: activations, NLM history, sync accumulators, optional episodic ring buffer |
| `CtmCache` | tick-by-tick intermediates retained for the backward pass |
| `CtmGradients` | accumulated gradients, mirroring `CtmWeights` |

For inspection, `ctm_forward_with_attn_trace` returns the attention weights for
every tick and head (`[ticks][heads][tokens]`).

## Full backpropagation through time

Training unrolls all `T` ticks and computes gradients for **every** weight: the
U-Net blocks, both NLM stages, the sync decays, the attention projections, the
output projection, and the exit gate. There are no stop-gradient shortcuts, and a
dedicated test (`small_scale_bptt`) checks the unrolled backward against a
reference. Every hot-path operation (matmul, layer norm, SiLU/GLU backward,
outer products) routes through `modgrad-device`, so the identical forward and
backward run on CPU or any GPU backend.

## Fidelity

The implementation tracks the published Sakana AI CTM rather than re-inventing
it: the forward pass, the U-Net synapse, and the synchronization mechanism each
correspond to the reference code, and the config presets are covered by tests.
That faithfulness is the point. modgrad is the architecture you can read,
re-derive, and trust.

Next: compose many CTMs into a [multi-region brain →](/docs/brain-composition).
