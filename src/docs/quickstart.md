---
title: Quickstart
section: Getting started
order: 20
description: Build, train and run an 8-region brain in about twenty lines of Rust.
---

# Quickstart

## Build a brain

Pick a preset, train it on your data, and run it as an interactive neural
computer. Presets are constructors on `modgrad_ctm::graph::RegionalConfig`, for
example `four_region`, `eight_region_small`, `eight_region`,
`eight_region_medium`, `eight_region_large` and `eight_region_billion`.

```rust
use modgrad_ctm::graph::{RegionalConfig, RegionalWeights, RegionalAdamW,
    RegionalGradients, regional_train_token, NeuralComputer};

// obs_dim, out_dims, ticks
let cfg = RegionalConfig::eight_region_small(128, 256, 16);

let mut w   = RegionalWeights::new(cfg);
let mut opt = RegionalAdamW::new(&w).with_lr(3e-4);

// train on your own data
let mut grads = RegionalGradients::zeros(&w);
for (token, target) in your_data {
    let (loss, _pred) = regional_train_token(&w, &mut grads, token, target);
    opt.step(&mut w, &mut grads);
    grads.zero();
}

// run it as a neural computer
let mut nc = NeuralComputer::new(w);
let response = nc.chat("hello", 100, 0.8);
```

You write the training loop. `regional_train_token` returns the loss and
prediction and accumulates gradients; `opt.step` applies them.

## Build flags

```bash
cargo build --release                    # default features: CPU + CUDA (CUDA loaded at runtime)
cargo build --release --features rocm    # adds AMD ROCm (needs HIP/hipBLAS at link time)
cargo test  --release                    # run the test suite
```

CUDA is in the default feature set because `cudarc` loads `libcuda.so` at runtime,
so the build works on machines without CUDA installed. ROCm is opt-in because it
links the system HIP and hipBLAS libraries at build time. Requires the Rust 2024
edition.

## Minimal example

`minictm` is a single-file trainer that uses the SDK directly, in the spirit of
nanoGPT.

```bash
cargo run -p minictm --release -- --data train.txt --steps 5000
cargo run -p minictm --release -- --data train.txt --steps 5000 --chat
```

Next: how the [Continuous Thought Machine →](/docs/continuous-thought-machine)
works, or the [SDK crate reference →](/docs/crates).
