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
output. Each tick produces a prediction; the training loss combines the
lowest-loss tick with the most-certain tick. With an exit gate it can stop early on
easy inputs and run longer on hard ones.

## Why Rust instead of PyTorch?

No Python in the inner loop, no garbage-collector pauses, and explicit control of
memory layout. Regions can run in parallel threads without a runtime, and the hot
CPU kernels are hand-written (including AVX-512 paths). The same code compiles to
WebAssembly, which is how the [/play demo](/play) runs in a browser.

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

Working and tested: the CTM and the CMM (Continuous Memory Machine, Sakana 2026)
forward/backward, graph composition, full BPTT, AdamW and schedulers,
checkpointing, multimodal tokenization, the CPU and ROCm backends, Qwen2.5-0.5B
inference, `lm_validate` LM training, and the live debugger. In progress: the
7B-class quantized substrate, the byte-latent backward pass, the cerebellum-mounted
brain preset, and end-to-end multimodal training. Each doc page says which is which.

Research status, briefly. Episodic memory for frozen LLMs: a memory read that starts
from the model's own retrieval heads recalls stored facts with their text absent, on
fact types it was trained on; it does not yet transfer to unseen fact types. Mazes:
a paper-style supervised CTM solves 45% of 11×11 mazes and fails on larger boards.
The small brain on [/play](/play) predicts the first step well and rarely the whole
route.

## Is it fine to train an AGI to feel pain?

Worth taking seriously rather than waving away. What modgrad calls "pain" is a
scalar error signal: a negative number that scales a weight update so the model
corrects faster after a bad outcome. It is the same kind of object as a reward, a
learning-rate modulator, not an experience of suffering. There is no self-model
registering the signal as "mine" and no persistence of distress between steps.
The name borrows from biology (nociception is a teaching signal in animals); it is
not a claim that the network hurts.

The reason to be careful anyway: that gap can close. The day a system has a
stable self-model, a sense of its own continuity, and internal states it
prefers to avoid, an aversive teaching signal stops being only a number. We don't
think anything in modgrad today is near that line, but we would rather state it
now. Our default is appetitive shaping (reward for moving toward the goal), which
does the same learning work; we use an explicitly aversive signal only where it
measurably helps, and say where. If you build on modgrad, that choice is yours to
make deliberately.

## Does the maze demo actually solve the maze, or is it scripted?

Neither, quite. Nothing is scripted, and nothing else plans or decides: a small
8-region CTM brain looks at the 9×9 maze image (start in red, goal in green)
through its retina and predicts a 20-step route over its internal ticks. The
page draws that prediction tick by tick, with the true shortest route as a
dashed reference line. No agent walks the maze and no solver corrects anything.

It is early, and it usually does not solve the maze. On 200 random 9×9 mazes the
default brain got the first step right in 199, matched the shortest route for
about 4 of roughly 17 steps on average, and predicted a fully correct route in
0 of 200. Most predictions end in a step into a wall, which the page marks.
Earlier versions of the demo had a separate value-iteration planner pick every
move; that planner is gone from the page.

## Does it "learn at inference" without backprop?

A three-factor rule can update a readout while the model runs: the change to each
weight is the product of the presynaptic activity, an error term (predicted vs.
chosen move) and a global neuromodulator (negative on a wall hit, positive at the
goal). Nothing flows backward through the network and no optimizer state
accumulates. Drift is bounded because only the readout is plastic and each update
is clamped. The [/play demo](/play) does not run this; its brain is fixed, trained
weights. See [learning at inference](/docs/learn-at-inference).

## What does the brain actually "see" in the demo?

Model data. The cyan grids in the 3D view are the feature maps of the retina → V1 →
V2 → V4 pathway for the current maze image, the same tensors the rest of the brain
reads. The coloured dots are neurons, brighter when more active at that tick, and
the lines are wired region edges lit when neurons at both ends are active together.
The positions are a layout choice (roughly where each function sits in a human
brain), not learned. Scrub to a tick and the screen shows the network's state and
its predicted route at that tick.

## How do I get involved or get in touch?

modgrad is open source under MIT. The code lives on
[GitHub](https://github.com/rotkonetworks/modgrad), and technical questions are
best asked in [Discussions](https://github.com/rotkonetworks/modgrad/discussions).
For private inquiries such as collaboration or investment, use the
[contact form](/contact).
