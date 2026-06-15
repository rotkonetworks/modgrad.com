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

The debugger speaks a small bincode protocol over TCP — `GetMeta`, `GetState`,
`GetHistory`, `GetTrace(region)`, `Step`, `Resume`, `InjectToken` — so any tool can
drive a running brain, not just the GUI.

## Examples

The repo ships ~36 self-contained examples — each a proof of one feature:

| group | examples |
|-------|----------|
| **benchmarks** | `mazes` / `maze_viz` (brain vs single CTM), `parity` (CPU/GPU numerical parity), `bench_brain_crossover` (GPU speedup vs `d_model`) |
| **language models** | `lm_validate` (the 5.72→0.74 training proof), `minictm`, `qwen_chat` / `qwen_load_smoke` (Qwen2.5 inference), `blt_generate` / `blt_train_real_text` / `blt_cerebellum_smoke` (byte-latent transformer) |
| **vision** | `cifar10_probe`, `v4ctm_classifier`, `ocr_smoke`, `babyai_probe`, `filter_viz` / `retina_viz` (learned vs fixed filters), `pretrain_retina`, `attention_viz` |
| **multimodal** | `multimodal_smoke` (text + image + audio in one token space) |
| **research** | `dream_bench` / `dream_gallery` (offline replay & hallucination), `hebbian_sanity`, `brain_nas_*` (architecture search), `penumbra_arena` (plural alters), `eight_region_v2_brain_smoke`, `zec_mm_homeostatic` (sleep/consolidation) |
