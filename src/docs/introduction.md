---
title: Introduction
section: Getting started
order: 10
description: modgrad is a Rust workspace of crates for building recurrent, multi-region neural models. Plain functions, no framework, no config files.
---

# Introduction

**modgrad** is a Rust workspace of crates for building recurrent, multi-region
neural models. It is a library, not a framework: you compose Continuous Thought
Machines, region graphs, codecs and compute backends yourself, in ordinary Rust.
(The name is short for *modular gradient*.)

```toml
[dependencies]
modgrad-ctm = { git = "https://github.com/rotkonetworks/modgrad" }
modgrad-compute = { git = "https://github.com/rotkonetworks/modgrad" }
modgrad-training = { git = "https://github.com/rotkonetworks/modgrad" }
```

## The approach

Instead of one large feed-forward stack, a modgrad model is a set of smaller
regions that iterate over internal ticks, pass activations to each other over
learned connections, and can keep episodic memory. Each of those pieces is a
separate crate or module you can use on its own. The same code trains small
brains on a laptop CPU and can attach a frozen pretrained LLM (Qwen-class) as a
cerebellum. There is no config-file layer and no hidden global state; optimizer
moments, traces and sync buffers are ordinary structs you can inspect and save.

## What's inside

- **Continuous Thought Machine**: a neuron pool that runs for several internal
  ticks before answering, ported from the
  [Sakana AI CTM](https://arxiv.org/abs/2505.05522) with backpropagation through
  time. The SDK also has the Continuous Memory Machine (CMM, Sakana 2026).
- **Graph composition**: many CTMs wired as a directed graph of regions, with
  learned routing between them.
- **Bio-inspired modules**: surprise-based "pain", neuromodulators, dream replay,
  and several personalities sharing one set of weights. All opt-in.
- **Compute backends**: CPU, CUDA, AMD ROCm, AMD KFD and Vulkan behind one dispatch
  interface, with GPU-resident weights.
- **Multimodal tokens**: bytes, images, audio and actions in one token space, with
  a convolutional visual cortex for images.

## Status

Under active development, in the open. Tested and in use: the CTM core, graph
composition, BPTT training, AdamW, checkpointing, multimodal tokenization, the CPU
and ROCm backends, and the live debugger. Not done: mounting a 7B-class model as a
cerebellum within 8 GB, full multimodal training, and the byte-latent transformer
path.

Current research: episodic memory for frozen LLMs. A memory read seeded from the
model's own retrieval heads recalls stored facts with their text absent, on fact
types it was trained on; it does not yet transfer to unseen fact types. On mazes,
a paper-style supervised CTM solves 45% of 11×11 mazes and fails on larger boards.

Next: [Quickstart →](/docs/quickstart)
