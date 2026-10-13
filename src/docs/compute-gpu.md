---
title: Compute & GPU
section: Systems
order: 60
description: How ops are dispatched across CPU, CUDA, ROCm, KFD and Vulkan backends, GPU weight residency, and the measured CPU-vs-ROCm training times with their caveats.
---

# Compute & GPU

modgrad separates *what* is computed from *where* it runs. Operations are logical
`Op` variants (matvec, matmul, layer-norm forward/backward, SiLU/GLU, AdamW, sync
update, super-linear, RoPE, dequant-matvec, and more), and a registry dispatches
each one to the first backend that supports it, in the order
**KFD → ROCm → CUDA → Vulkan → CPU**. Adding a backend means implementing one
trait. `MODGRAD_BACKEND=cpu` forces the CPU reference path for debugging.

## Backends

| backend | implementation |
|---------|----------------|
| **CPU** | rayon-parallel, with AVX-512 / AVX2 paths; the reference every other backend is checked against |
| **AMD ROCm** | hipBLAS (`sgemv`, `sgemm`, strided-batched, bf16 `gemmEx`), MIOpen (layer-norm, softmax, activations, GLU), and custom HIP kernels for RMSNorm, RoPE and Q4_K/Q5_K/Q6_K dequant-matvec |
| **NVIDIA CUDA** | cuBLAS via `cudarc`, loaded at runtime; no `nvcc` or toolkit needed to build |
| **AMD KFD** | hand-written RDNA3 assembly kernels with BAR-mapped VRAM. Opt-in (`--features kfd` plus `MODGRAD_ENABLE_KFD=1`) while a compute-ring stall is unresolved |
| **Vulkan** | compute-shader fallback for any Vulkan 1.3 driver (e.g. Steam Deck, Intel Arc, Apple Silicon via MoltenVK) |

Dispatch has size gates: ROCm only takes matvec/matmul at about 64×64 and larger,
below which the CPU is faster because of launch overhead, and a `HipBatch` guard
flushes the queue periodically so thousands of small dispatches per step don't
overflow it.

## Residency

The main performance lever is keeping weights on the GPU instead of copying them
across PCIe on every call. A `GpuVec` keeps a buffer on the device, and the
resident `Linear` types cover the common cases:

- `LinearResident` (fp32) and `LinearResidentBf16` (half the memory)
- `LinearResidentQuantized`: Q4_K/Q5_K/Q6_K weights with fused dequant-matvec
- `LinearResidentStreaming`: weights fetched from a VRAM pool per call, for models
  too large to keep resident in 8 GB

## Measured

End-to-end maze training times from `docs/GPU_BENCHMARKS.md` in the SDK repo, on an
AMD Radeon RX 7600M XT (gfx1102), single runs:

| `d_model` | CPU | ROCm | difference |
|-----------|-----|------|------------|
| 128 | 140 s | 144 s | −3% (dispatch overhead dominates) |
| 256 | 657 s | 603 s | +8% |
| 384 | 465 s | 393 s | +15% |
| 512 | 696 s | 661 s | +5% |

Eval results were bit-identical between CPU and ROCm in every row.

Caveats:

- **The workload is orchestration-bound, not FLOP-bound.** A maze training step
  issues about 23,000 dispatches. On this host the CPU ran at roughly half its boost
  clock under load, so much of the GPU's benefit is taking work off the CPU.
- **Single-shot timings vary.** A repeat of the d_model 384 ROCm run took 594 s
  instead of 393 s with identical results; the spread is up to ~1.5× from CPU clock
  capping.
- **Not everything helped.** Routing the backward weight-gradient matvec
  (`MatvecT`) through hipBLAS was 2.3× slower than the CPU at d_model 384 because
  of per-dispatch allocation and copies, and was reverted. KFD is reliable only for
  single-workgroup matvec, and the CUDA path has not been validated on NVIDIA
  hardware.

Next: how the brain [senses the world →](/docs/multimodal).
