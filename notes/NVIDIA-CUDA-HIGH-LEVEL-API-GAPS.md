# NVIDIA CUDA High-Level API Gaps for MLX Inference

This catalog records where the retained MLX CUDA work needed custom kernels or
backend logic even though a high-level NVIDIA API was the first choice. It
separates missing capability from APIs that express the math but did not meet
the measured latency, graph, memory, or numerical contract.

## Summary

| Area | Closest NVIDIA API | Gap type | Priority |
| --- | --- | --- | --- |
| Weight-only FP4 QMV | cuBLASLt block-scaled matmul | Missing W4A16 contract | High |
| Gathered/grouped FP expert matmul | cuBLASLt + cuBLAS grouped GEMM | Missing combined grouping and block scales | High |
| Mostly-changing CUDA graphs | CUDA Graph update APIs | Missing efficient mixed/opaque-node update | High |
| D256/D512 decode attention | cuDNN SDPA frontend | Capability/performance/workspace gap | High |
| Small depthwise conv graph update | cuDNN frontend graph population | Plan rejects native graph population | Medium |
| Fixed small-k expert routing | CUB block/warp sort | Missing partial top-k primitive | Medium |
| BF16 decode projections | cuBLASLt matmul | Existing path too slow at key M=1 shapes | Medium |
| Dual projection plus GeGLU | cuBLASLt epilogues | Missing two-matrix fused epilogue | Medium |

## Weight-Only FP4 Matrix-Vector Multiply

Closest API: cuBLASLt block-scaled matmul, with CUTLASS/CuTe as the lower-level
template fallback.

Required contract: E2M1 weights, per-group E4M3 scales at group size 16, BF16
activations, FP32 accumulation, BF16 output, and no activation quantization.
Some checkpoints also require a direct FP32 tensor output scale with a defined
BF16 rounding boundary.

Current gap: native Blackwell FP4 MMA paths are W4A4 and require activation
quantization. That changes the operation and measured model numerics. The
cuBLASLt block-scaled path also underperforms the scalar/vector QMV at M=1:

| Probe | Retained QMV | NVIDIA library path |
| --- | ---: | ---: |
| NVFP4 production shape | 38.00 us | 92.38 us |
| MXFP8 large-N head | 449.843 us | 682.660 us |

The current custom QMV uses native Blackwell E2M1/E4M3 conversions and
vectorized loads while preserving W4A16 numerics.

Smallest useful extension: a cuBLASLt weight-only block-scaled matmul/QMV
descriptor accepting compressed E2M1 weights and grouped E4M3 scales while
keeping BF16 activations and FP32 accumulation. M=1 must be a first-class
latency target rather than a degenerate GEMM case.

Evidence:

- `notes/MLX-CUDA-UM-QMM-EXPERIMENTS.md`, "cuBLASLt Native FP4 M=1 Probe"
- `.cache/agent-notes/mlx-cuda-moe-kernel-tuning-20260726.md`
- `.cache/patches/mlx-qqmm-cublas-m1-probe.patch`

## Gathered and Grouped Block-Scaled Expert Matmul

Closest APIs: cuBLASLt block-scaled matmul and cuBLAS grouped GEMM.

Required contract: one activation table, a runtime expert index per output
assignment, independently packed expert weights/scales, optional sorted expert
runs, and weight-only NVFP4/MXFP8 arithmetic. Prefill needs tensor-core tiles;
decode needs gathered M=1 QMV without materializing repeated activations.

Current gap: cuBLASLt exposes the required block-scale descriptors but no
grouped/gathered GEMM interface. cuBLAS exposes grouped GEMM but not the
block-scaled FP4/FP8 contract. Issuing one cuBLASLt operation per expert loses
graph and launch efficiency. The retained MLX CUDA path therefore extends its
CuTe QMM and QMV kernels with RHS gather/sorted-run handling.

Smallest useful extension: add grouped problem arrays to cuBLASLt block-scaled
matmul, including per-problem weight, scale, M/N/K, and output pointers. An
optional index indirection for repeated activation rows would avoid host-side
problem-list construction for sparse MoE dispatch.

Evidence:

- `notes/MLX-CUDA-UM-QMM-EXPERIMENTS.md`, grouped cuBLASLt investigation near line 1814
- retained `mlx/backend/cuda/quantized/qmm` GatherQMM changes

## Mostly-Changing CUDA Graphs with Library Child Graphs

Closest APIs: `cudaGraphExecUpdate`, per-node `cudaGraphExec*NodeSetParams`,
and device-updatable kernel nodes.

Required contract: update hundreds of mostly-changing kernel parameters in one
operation while retaining cuDNN and other opaque library child graphs. MLX
needs to reuse graph topology without paying one host API call per changed
node.

Current gap: 95-98% of direct node parameters changed in representative Qwen
decode graphs, so sparse per-node updates lose to whole-graph update.
Device-updatable nodes are mutually exclusive with `cudaGraphExecUpdate` and
cannot update opaque nodes inside cuDNN child graphs.

Smallest useful extension: a batched host-side parameter update API for an
existing graph executable, plus a compatible way for library child graphs to
expose/update their dynamic pointer slots without rebuilding or stream
capture.

Evidence:

- `.cache/diagnostics/qwen36-graph-param-stats-5090-20260801`
- `.cache/diagnostics/qwen36-graph-param-reuse-20260802`
- `notes/MLX-CUDA-UM-QMM-EXPERIMENTS.md`, "Device-updatable CUDA graph nodes"

## Wide-Head Decode Attention

Closest API: cuDNN frontend SDPA.

Required contract: BF16 GQA decode with head dimensions 256 and 512, causal or
additive masks, bounded temporary memory, and competitive single-query
latency.

Current gap: MLX's established cuDNN route is limited to D <= 128. A focused
backend experiment allowed cuDNN to claim D256/D512 decode, but D256 consumed
39.16 ms across 2,600 calls and D512 consumed 7.49 ms across 650 calls. The
combined result was about 11.8% slower than the retained two-pass MLX CUDA
kernel and prior prefill attempts created unacceptable workspace pressure.

Smallest useful extension: optimized, bounded-workspace D256/D512 decode plans
in the same cuDNN SDPA frontend contract, including GQA and additive masks.

Evidence:

- `.cache/patches/mlx-cudnn-wide-decode-blackwell-rejected.patch`
- `.cache/profiles/gemma4-12b-cudnn-wide-decode-v1-p2048g64-20260801`
- retained `python/tests/test_fast_sdpa.py` D256/D512 coverage

## Native Graph Population for Small Depthwise Convolution

Closest API: cuDNN frontend graph `populate_cuda_graph`.

Required contract: repeatedly execute a BF16 width-4 depthwise convolution in
an outer inference graph while updating pointers without stream capture.

Current gap: the selected cuDNN plan executes efficiently, but rejects native
graph population with `CUDNN_STATUS_NOT_SUPPORTED_CUDA_GRAPH_NATIVE_API`.
Direct replacement kernels were neutral at the endpoint, so MLX retains cuDNN
and pays the child-capture boundary.

Smallest useful extension: support native CUDA-graph population and pointer
updates for this existing depthwise-convolution plan. This asks for graph
compatibility, not a new convolution semantic.

Evidence:

- `.cache/bench/qwen36-cudnn-conv-graph-v1-p16g128-20260802`
- `.cache/bench/qwen36-depthwise-conv-child-v2-p16g128-20260802`

## Fixed Small-K Expert Selection

Closest API: CUB warp/block radix or merge sort.

Required contract: stable top 8 from 128 or 256 BF16/FP32 expert scores with
defined NaN and index tie-breaking, followed by normalization.

Current gap: sorting all candidates performs much more work than selecting
eight and adds register/shared-memory pressure. Replacing repeated fixed top-8
selection with `cub::WarpMergeSort` regressed Laguna p2048/g128 from
7203.37/157.27 to 6311.89/142.14 prompt/generate tok/s.

Smallest useful extension: a CUB warp/block partial-selection primitive for
small compile-time K, preserving caller-defined value/index comparison and
returning only the selected entries.

Evidence:

- `.cache/bench/laguna-5090-cub-router-p2048-20260731`
- `.cache/correctness/laguna-cub-router-5090-20260731`

## BF16 M=1 Projection and Dual-Projection Epilogues

Closest API: cuBLASLt matmul and epilogues.

Required contracts:

- efficient BF16 M=1 GEMV at skinny `N=32,K=2048` and very wide MLP shapes;
- two independent projections sharing one activation, followed by GeGLU.

Current gap: cuBLASLt expresses the single projection but regressed the skinny
shape from 36.53 us to 53.03 us. At large dense MLP shapes, custom BF16 kernels
match llama-server's memory-bandwidth-bound kernels. cuBLASLt has no epilogue
that consumes two independently weighted projection results to compute GeGLU,
so the library route requires two launches plus a separate activation.

Smallest useful extensions: improve latency-oriented M=1 BF16 algorithms for
small N, and consider a dual-matmul epilogue interface where two operations
share A and feed a supported binary activation. The latter is useful only if it
retains graph composability and does not duplicate weight traffic.

Evidence:

- `.cache/patches/mlx-cublaslt-bf16-n32-k2048.patch`
- `.cache/bench/gemv-cublaslt-n32-5090-20260801`
- `notes/MLX-CUDA-UM-QMM-EXPERIMENTS.md`, "Dense Gemma4 BF16 Generation"

## Custom Kernels Without a Narrow API Ask

Mamba2 selective scan, gated-delta recurrence, grouped gated RMSNorm, and
model-specific residual/router fusions do not map to a small extension of an
existing cuBLAS, cuDNN, or CUB operation. They should continue to follow shared
Metal/CUDA kernel semantics and use high-level libraries for their internal
GEMM/convolution pieces where applicable, but they are not good candidates for
the focused NVIDIA API feedback list.
