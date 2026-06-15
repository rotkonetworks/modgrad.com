---
title: Compute & GPU
section: Systems
order: 60
description: The backend matrix, the residency model and its resident Linear variants, the custom kernel set, and the measured benchmark numbers with their caveats.
---

# Compute & GPU

modgrad separates *what* you compute from *where* it runs. Every hot-path
operation routes through `modgrad-device` as an immutable op variant dispatched at
runtime by a preference-ordered registry. The CPU backend is always present as a
fallback; a new backend is one implementation of the backend trait.

## Backends

| backend | implementation |
|---------|----------------|
| **CPU** | rayon across rows; explicit **AVX-512 / AVX2** matmul (16-wide unrolled FMA, four accumulators, L1 cache-tiling); parallel only above a size threshold |
| **AMD ROCm** | hipBLAS (`sgemv`, `sgemm`, bf16 `gemmEx`, strided-batched), MIOpen (layernorm, softmax, activation, GLU), and custom hipcc kernels: RMSNorm, AdamW, RoPE backward, Q4_K/Q5_K/Q6_K dequant-matvec, SDPA decode |
| **NVIDIA CUDA** | cuBLAS via `cudarc`, dynamic-loading `libcuda.so` at runtime — no `nvcc` or toolkit needed to build |
| **AMD KFD** | hand-written RDNA3 assembly kernels with BAR-mapped zero-copy VRAM; opt-in (`MODGRAD_ENABLE_KFD=1`) |
| **Vulkan** | a cross-vendor matvec compute shader via `ash` (Steam Deck, Intel Arc, Apple Silicon via MoltenVK) |

## Residency

The performance idea is **residency**: stop bouncing weights across PCIe on every
call. Resident buffers upload once and dispatch kernels against device pointers,
so the inner loop never re-transfers. The resident `Linear` family covers the
common cases:

- `LinearResident` — fp32
- `LinearResidentBf16` — bf16 mixed precision
- `LinearResidentQuantized` — Q4_K weights, fused dequant-matvec
- `LinearResidentStreaming` — caches activations for the backward pass

To keep thousands of tiny dispatches per step from overflowing the command ring,
a `HipBatch` guard flushes the queue periodically (by default every 256 ops). On
ROCm, matvec/matmul are gated to shapes of at least 64×64 so small ops skip
dispatch overhead; resident ops drop that gate, since on-device shapes are free.

## What's measured

The honest end-to-end numbers, from `docs/GPU_BENCHMARKS.md`, training mazes on an
AMD Radeon RX 7600M XT (gfx1102):

| `d_model` | CPU | ROCm | result |
|-----------|-----|------|--------|
| 384 | 465 s | 393 s | **+15%** |
| 256 | 657 s | 603 s | **+8%** |
| 128 | 140 s | 144 s | −3% |

Two things to understand about these:

- **The win is orchestration-bound, not FLOPS.** A maze step issues on the order
  of 23,000 GPU dispatches; keeping weights resident removes per-dispatch
  allocation and transfer. At small `d_model`, dispatch overhead isn't amortized
  and the GPU loses.
- **Per-call** the residency gain is much larger — the repo reports up to 55× on
  matmul and 5.6× on AdamW at 1024×512 versus the host-bounce path.

The caveats are documented rather than hidden: the backward weight gradient
(`MatvecT`) is still dispatch-bound and currently *slower* than CPU at these
sizes (a resident activation buffer is the next step); single-shot timings vary
up to ~1.5× under CPU thermal throttling; KFD is reliable only for
single-workgroup matvec today; and the CUDA path is plumbed but was not validated
on NVIDIA hardware during the last audit.

Next: how the brain [senses the world →](/docs/multimodal).
