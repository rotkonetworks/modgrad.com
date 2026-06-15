---
title: Quickstart
section: Getting started
order: 20
description: Build, train, and run an 8-region brain in about twenty lines of Rust.
---

# Quickstart

## Build a brain

Pick a preset, train it on your data, and run it as an interactive neural
computer. The presets — `four_region`, `eight_region_small` (187k params),
`eight_region` (~81M), `eight_region_medium`, `eight_region_large`, and
`eight_region_billion` — all live in `modgrad_ctm::graph::RegionalConfig`.

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
    opt.step(&mut w, &grads);
    grads.zero();
}

// run it as a neural computer
let mut nc = NeuralComputer::new(w);
let response = nc.chat("hello", 100, 0.8);
```

You own the training loop. modgrad gives you pure functions —
`regional_train_token` returns a loss and accumulates gradients; `opt.step`
applies them — and stays out of your way.

## Build flags

```bash
cargo build --release                   # CPU only (default)
cargo build --release --features cuda    # NVIDIA, via cudarc (no nvcc needed)
cargo build --release --features rocm    # AMD, via rocBLAS + HIP
cargo test  --release                    # run the test suite
```

CUDA stays in the default feature set because `cudarc` dynamic-loads `libcuda.so`
at runtime. ROCm is opt-in: it hard-links the system HIP/hipBLAS libraries at
build time. modgrad requires the Rust 2024 edition.

## Try the minimal example

`minictm` is nanoGPT for CTMs — a single-file trainer that uses the SDK directly,
with no runtime around it.

```bash
cargo run -p minictm --release -- --data train.txt --steps 5000
cargo run -p minictm --release -- --data train.txt --steps 5000 --chat
```

From here, read how the [Continuous Thought Machine →](/docs/continuous-thought-machine)
actually works, or jump to the [SDK crate reference →](/docs/crates).
