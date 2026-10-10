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
compile time, so a brain module literally cannot do I/O. You get fearless
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

## Is it fine to train an AGI to feel pain?

Worth taking seriously rather than waving away. What modgrad calls "pain" is a
scalar error signal: a negative number that, on a wrong move, increases the
size of the local weight update so the readout corrects faster. It is the same
kind of object as a reward — a learning-rate modulator — not an experience of
suffering. There is no self-model registering the signal as "mine," no
persistence of distress between steps, no welfare being harmed. Calling it pain
is an honest nod to the biology (nociception is a teaching signal in animals),
not a claim that the network hurts.

The reason to be careful anyway: that gap can close. The day a system has a
stable self-model, a sense of its own continuity, and internal states it
prefers to avoid, an aversive teaching signal stops being only a number. We do
not think today's maze readout is anywhere near that line, but we would rather
state the line now than discover we crossed it. So our defaults are
appetitive — reward-forward shaping (move *toward* the goal) does the same
learning work as aversive shaping, and we reach for an explicitly aversive
signal only when it measurably helps and document where. If you are building on
modgrad, the same choice is yours to make deliberately. The honest answer is:
it is fine for what this is, and the question stops being rhetorical the moment
the system is more than this.

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

## How does it "learn at inference" without backprop or forgetting?

The motor readout updates with a three-factor rule: the change to each weight is
the product of the presynaptic activity, the postsynaptic error (predicted move
vs. the move actually taken), and a global neuromodulator (pain on a wall-hit,
reward at the goal). No gradients flow backward through the network, no
optimizer state accumulates — it is a local, momentum-free update applied as the
model acts. Forgetting is bounded two ways: only the small readout is plastic
(the larger substrate stays fixed, the way a human cerebellum changes slowly
while cortex adapts fast), and the weight change is clamped each step so the rule
cannot blow up or drift far from the trained weights. This lives in the SDK; the
/play demo does not run it — the brain there is fixed, trained weights.

## What does the brain actually "see" in the demo?

Real model data, not a decorative render. The cyan grids in the 3D view are the
literal feature maps coming out of the visual retina → V1 → V2 → V4 pathway for
the current maze image — the same tensors the rest of the brain reads. The coloured
dots are neurons, brightened when they spike that tick, and the lines are
connectome edges lit when two regions fire together. If you scrub to a tick, what
is on screen is the network's state and its predicted route at that tick.

## How do I get involved or get in touch?

modgrad is open source under MIT. The code lives on
[GitHub](https://github.com/rotkonetworks/modgrad), and technical questions are
best asked in [Discussions](https://github.com/rotkonetworks/modgrad/discussions).
For private inquiries such as collaboration or investment, use the
[contact form](/contact).
