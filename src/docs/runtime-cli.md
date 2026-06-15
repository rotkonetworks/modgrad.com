---
title: CLI & runtime
section: Reference
order: 100
description: isis, the runtime built on the SDK; minictm, the minimal example; and the live 3D brain debugger.
---

# CLI & runtime

## isis

`isis` is the runtime built on the SDK — an 8-region brain with multimodal input
and a neural-computer mode. You don't need isis to use modgrad; it's just one
composition of the crates.

```bash
# train on a byte curriculum (or your own text/images/audio)
isis train model.bin
isis train model.bin --multimodal --images cifar.bin --audio clips/

# interactive neural computer
isis nc model.bin
isis nc model.bin --audio mic.wav --camera frames/ --debug-port 4747

# generate
isis generate model.bin --prompt "the cat "

# run as a TCP service
isis daemon model.bin --port 4747
isis send "hello world" --addr 127.0.0.1:4747

# show available compute devices
isis devices
```

There are also `isis learn` / `learn-ffn` for raw-byte and standalone FFN
pretraining, and `isis eval` for validation. Every command accepts `--debug-port`
to attach the live debugger.

## minictm

`minictm` is the minimal example — nanoGPT for CTMs. It uses the SDK directly with
no runtime, and is the best place to read how a training loop fits together.

```bash
cargo run -p minictm --release -- --data train.txt --steps 5000
cargo run -p minictm --release -- --data train.txt --steps 5000 --chat
```

## The 3D debugger

`modgrad-debugger` is a live brain visualizer (egui) that connects to any running
model over TCP. It renders:

- **3D neuron particles** placed by region and colored per region, sized by
  activation;
- a **token stream** color-coded by modality (text, image, audio, video, action,
  timestamp);
- **NLM trace heatmaps** per region and a **global sync** visualization;
- a **command center** to pause / resume / single-step, inject tokens, and inspect
  state.

```bash
modgrad-debugger 127.0.0.1:4747
```

## Examples

The repo ships ~36 examples covering the surface: `mazes` and `parity`
(benchmarks), `cifar10_probe` / `filter_viz` / `babyai_probe` (vision),
`qwen_chat` / `qwen_load_smoke` (foundation-model inference), `lm_validate`
(LM training proof), `blt_*` (byte-latent transformer), `dream_gallery` and
`brain_nas_*` (research). Each is a self-contained proof of one feature.
