---
title: Compute & GPU
section: Systems
order: 60
description: A JAX-style op-dispatch model across CPU/CUDA/ROCm/KFD/Vulkan, custom HIP kernels, weight residency, and the honest end-to-end benchmark numbers with their caveats.
---

# Compute & GPU

modgrad separates *what* you compute from *where* it runs. Operations are expressed
as logical `Op` variants (matvec, matmul, layer-norm fwd/bwd, SiLU/GLU, AdamW, sync
update, super-linear, RoPE, dequant-matvec — 22+ in all) and a stateless
**registry** dispatches each to the first backend that supports it, in preference
order **KFD → ROCm → CUDA → Vulkan → CPU**. It's the JAX/PJRT plugin model, not
hardware-specific wrappers: a new backend is one implementation; `MODGRAD_BACKEND=cpu`
forces the reference path for debugging.

## Backends

| backend | implementation |
|---------|----------------|
| **CPU** | rayon-parallel; explicit **AVX-512 / AVX2** matmul; the numerical ground truth every other backend is checked against |
| **AMD ROCm** | dynamically-linked hipBLAS (`sgemv`, `sgemm`, strided-batched, bf16 `gemmEx`) + MIOpen (layer-norm, softmax, activation, GLU) + custom hipcc kernels: RMSNorm, RoPE, and Q4_K/Q5_K/Q6_K dequant-matvec |
| **NVIDIA CUDA** | cuBLAS via `cudarc`, dynamic-loaded at runtime — no `nvcc` or toolkit needed to build |
| **AMD KFD** | 25 hand-written RDNA3 assembly kernels with BAR-mapped zero-copy VRAM; opt-in (`MODGRAD_ENABLE_KFD=1`) with conservative shape gates while a ring-stall bug is investigated |
| **Vulkan** | a cross-vendor compute-shader fallback (Steam Deck, Intel Arc, Apple Silicon via MoltenVK) |

Dispatch has size gates — ROCm only claims matvec/matmul at ≥ 64×64, below which the
CPU wins on launch overhead — and a `HipBatch` guard flushes the command queue
periodically so thousands of tiny dispatches per step don't overflow the ring.

## Residency

The performance idea is **residency**: stop bouncing weights across PCIe every call.
A `GpuVec` keeps the buffer on-device, and the resident `Linear` family covers the
cases:

- `LinearResident` (fp32) and `LinearResidentBf16` (half the memory)
- `LinearResidentQuantized` — Q4_K/Q5_K/Q6_K weights with fused dequant-matvec
- `LinearResidentStreaming` — weights fetched from a VRAM pool per call, for
  7B-class models that don't fit resident on 8 GB

## What's measured

The honest end-to-end numbers, from `docs/GPU_BENCHMARKS.md`, training mazes on an AMD
Radeon RX 7600M XT (gfx1102):

| `d_model` | CPU | ROCm | result |
|-----------|-----|------|--------|
| 128 | 140 s | 144 s | −3% (dispatch overhead dominates) |
| 256 | 657 s | 603 s | **+8%** (crossover) |
| 384 | 465 s | 393 s | **+15%** (largest win) |
| 512 | 696 s | 661 s | +5% (fixed dispatch cost shrinks relative to compute) |

Two things to understand:

- **The win is orchestration-bound, not FLOPS.** A maze step issues ~23,000
  dispatches; keeping weights resident removes per-dispatch allocation and transfer.
  Under thermal load the host CPU clock-caps to ~52%, so the GPU's value is partly in
  *not being the bottleneck*.
- **The numbers are honest about the rough edges.** The backward weight gradient
  (`MatvecT`) was 2.3× *slower* than CPU when first tried and reverted — its fix
  (resident activations, one fused Linear-backward dispatch) is roadmap. Single-shot
  timings vary up to ~1.5× from throttling, KFD is reliable only for single-workgroup
  matvec today, and the CUDA path is plumbed but wasn't validated on NVIDIA hardware
  during the last audit.

Next: how the brain [senses the world →](/docs/multimodal).
