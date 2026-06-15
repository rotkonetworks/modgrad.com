---
title: FAQ
section: Reference
order: 110
description: Common questions about CTMs, why Rust, the modular design, and how the pieces fit.
---

# FAQ

## What is a CTM?

A Continuous Thought Machine ([arXiv 2505.05522](https://arxiv.org/abs/2505.05522))
is a neuron pool that iterates internally over multiple ticks before producing
output. Each tick generates a prediction; the loss combines the best tick with the
most-certain tick. More ticks means more deliberation, and the architecture
naturally spends more compute on harder inputs.

## Why Rust instead of PyTorch?

No garbage-collector pauses during training, no Python in the inner loop, and
explicit memory layout. The type system enforces the brain/host boundary at
compile time — a brain module literally cannot do I/O. You get fearless
concurrency for parallel multi-region computation, and hand-written AVX-512
kernels where they matter.

## What does "modular gradient" mean?

The SDK is a set of crates, not a framework. `modgrad-compute` doesn't know about
`modgrad-ctm`; `modgrad-training` doesn't know about `modgrad-codec`. You compose
what you need. Want just the CTM forward pass? Import one crate. Want the full
8-region brain with multimodal codecs and a 3D debugger? Import a dozen.

## Do I have to use the brain regions, or the bio modules?

No. The core CTM trains fine on its own, and every bio-inspired module is opt-in.
Start with a single CTM, add graph composition when you want multiple regions, and
reach for pain / dream / episodic memory only if your problem benefits from them.

## What's actually working versus aspirational?

Working and tested today: the CTM forward/backward, graph composition, full BPTT,
AdamW and schedulers, checkpointing, multimodal tokenization, the CPU and ROCm
backends, Qwen2.5 inference, `lm_validate` LM training, and the live debugger.
In progress: the 7B-class quantized substrate, the byte-latent backward pass, the
cerebellum-mounted brain preset, and full end-to-end multimodal training. Each doc
page says which is which.

## How do I get involved or get in touch?

modgrad is open source under MIT. The code lives on
[GitHub](https://github.com/rotkonetworks/modgrad), and technical questions are
best asked in [Discussions](https://github.com/rotkonetworks/modgrad/discussions).
For private inquiries — collaboration or investment — use the
[contact form](/contact).
