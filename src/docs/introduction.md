---
title: Introduction
section: Getting started
order: 10
description: modgrad is a set of composable Rust crates for building brains — pure functions, no framework, no YAML.
---

# Introduction

**modgrad** is a set of composable Rust crates for building general intelligence —
not a framework. You assemble neural runtimes out of primitives you can actually
reason about: Continuous Thought Machines, multi-region brains, bio-inspired
learning, multimodal codecs, and full-residency GPU compute. (The name is short
for *modular gradient*.)

You pick the architecture. modgrad provides the parts.

```toml
[dependencies]
modgrad-ctm = { git = "https://github.com/rotkonetworks/modgrad" }
modgrad-compute = { git = "https://github.com/rotkonetworks/modgrad" }
modgrad-training = { git = "https://github.com/rotkonetworks/modgrad" }
```

## The idea

The frontier scaled a single architecture — the transformer — to the limits of
capital. modgrad takes the other road. Brains are made of specialized regions
that **think recurrently**, route information between each other, remember what
happened, and learn from their own surprise. Each of those is a building block,
and modgrad hands you the blocks as plain crates.

The same SDK trains a 187k-parameter router on a laptop and mounts a frozen LLM
as a cerebellum. There is no YAML, no config-file framework, and no hidden global
state — just Rust functions you compose however you want.

## What's inside

- **A Continuous Thought Machine** — a neuron pool that iterates over internal
  "ticks", accumulating evidence before it commits to an answer. A faithful Rust
  port of the [Sakana AI CTM](https://arxiv.org/abs/2505.05522), with full
  backpropagation through time.
- **Graph composition** — wire many CTMs into a directed graph of cortical and
  subcortical brain regions, with learned routing and episodic memory.
- **Bio-inspired learning** — pain, neuromodulators, dream consolidation,
  experience replay, and even multiple personalities sharing one set of weights.
- **A compute layer** — CPU (AVX-512), CUDA, AMD ROCm, KFD, and Vulkan behind one
  backend trait, with resident kernels that keep weights on the GPU.
- **Multimodal sensing** — a unified token space where bytes, images, audio, and
  actions share one embedding table, fed by a learnable visual cortex.

## Honesty about status

modgrad is under active development and built in the open. The CTM core, graph
composition, full BPTT training, AdamW, checkpointing, multimodal tokenization,
the GPU backends, and the live debugger all **work today** and are tested. The
larger ambitions — mounting a 7B-class cerebellum, full multimodal training, the
byte-latent transformer path — are **in progress**, and each doc page marks what
is shipping versus what is on the roadmap.

Next: [Quickstart →](/docs/quickstart)
