---
title: CLI & runtime
section: Reference
order: 100
description: isis, the runtime built on the SDK; minictm, the minimal example; the 3D brain debugger; and the example binaries.
---

# CLI & runtime

## isis

`isis` is a runtime built on the SDK: an 8-region brain with multimodal input and
a neural-computer mode. You don't need it to use modgrad; it is one composition of
the crates.

```bash
# train on the byte curriculum, optionally with images / audio
isis train model.bin
isis train model.bin --multimodal --images cifar.bin --audio clips/

# interactive neural computer
isis nc model.bin
isis nc model.bin --audio mic.wav --camera frames/ --debug-port 4747

# generate from a prompt
isis generate model.bin "the cat "

# run as a TCP service, and send it text
isis daemon model.bin --port 4747
isis send "hello world" --addr 127.0.0.1:4747

# list compute devices
isis devices
```

Also: `isis learn` / `isis learn-ffn` for raw-byte and standalone FFN pretraining,
and `isis eval` for validation. `train`, `nc` and `learn` accept `--debug-port` to
attach the debugger.

## minictm

`minictm` is the minimal example: a single-file trainer that uses the SDK directly,
with no runtime. It is the shortest path to reading how a training loop fits
together.

```bash
cargo run -p minictm --release -- --data train.txt --steps 5000
cargo run -p minictm --release -- --data train.txt --steps 5000 --chat
```

## The 3D debugger

`modgrad-debugger` (egui) connects to a running model over TCP and shows:

- neurons as 3D particles, placed and coloured by region, sized by activation;
- the token stream, coloured by modality;
- per-region NLM trace heatmaps and global sync;
- controls to pause, resume, single-step, inject tokens and inspect state.

```bash
modgrad-debugger 127.0.0.1:4747
```

The protocol is length-prefixed bincode over TCP (`GetMeta`, `GetState`,
`GetTrace`, `GetHistory`, `Inject`, `InjectText`, `Pause`, `Resume`, `Step`,
`Subscribe`), so other tools can drive a running brain too.

## Examples

The repo has about 40 example binaries, each exercising one feature:

| group | examples |
|-------|----------|
| **benchmarks** | `mazes` / `maze_viz` (brain vs single CTM), `parity` (CPU/GPU numerical parity), `bench_brain_crossover` (GPU speedup vs `d_model`) |
| **language models** | `lm_validate` (byte-level transformer training with held-out validation), `minictm`, `qwen_chat` / `qwen_load_smoke` (Qwen2.5 inference), `blt_generate` / `blt_train_real_text` / `blt_cerebellum_smoke` (byte-latent transformer) |
| **vision** | `cifar10_probe`, `v4ctm_classifier`, `ocr_smoke`, `babyai_probe`, `filter_viz` / `retina_viz` (learned vs fixed filters), `pretrain_retina`, `attention_viz` |
| **multimodal** | `multimodal_smoke` (text, image and audio in one token space) |
| **research** | `dream_bench` / `dream_gallery` (offline replay), `hebbian_sanity`, `brain_nas_*` (architecture search), `penumbra_arena` (plural alters), `eight_region_v2_brain_smoke`, `zec_mm_homeostatic` (sleep / consolidation) |
