# MLX CUDA Unified-Memory QMM Experiments

This is a worktree-specific experiment log for the NVIDIA unified-memory
optimization effort. It is not general MLX profiling guidance.

## Status Reporting Protocol

When reporting current performance, use the latest correctness-passing,
accepted/current-best artifact for each host, model, quantization, and benchmark
cohort. Do not replace a best row with the output of an active experiment,
especially when that candidate regresses. Report an active candidate only in a
separately labeled experiment-delta section with its exact control and artifact.

Treat a best row as monotonic: it changes only after a candidate passes the same
correctness gates and preserves or improves the relevant p2048/g128 metrics
under the same protocol. If the prompt corpus, token rotation, scrub policy,
backend version, model artifact, or target baseline changes, retain the old row
and present the new cohort separately until both sides have matched artifacts.
Never silently recalculate or replace a headline percentage with an unmatched
baseline.

## Test Shape

- Host: `tater50` (DGX Spark / GB10 / SM12.1)
- Model: `dhiltgen/gemma4:26b-nvfp4`
- Benchmark: p2048/g128, one discarded full benchmark scrub, then three epochs
- llama-server target: 2730 prompt tok/s, 60.6 generate tok/s
- Correctness gates: full API/cache/logprobs suite plus 2432-token natural prompt

## QMM Results

| Experiment | Prompt | Generate | Representative QMM | Result |
|---|---:|---:|---:|---|
| Upstream scale loads | - | - | 7.41 ms | Baseline NCU |
| Coalesced shared-scale lanes (earlier) | - | - | 7.48 ms | No improvement |
| Vector shared-scale loads (earlier) | - | - | 7.50 ms | No improvement |
| Per-thread scale bundle | - | - | 7.23 ms | Compiler promoted the thread-private shared intermediate to local storage |
| Safe full NVFP4 scale slab | 2581.47 | 64.72 | 5.94 ms | Best validated path; 94.5% / 106.8% of target |
| Full slab plus wide MMA | 2578.25 | 64.21 | - | No improvement; removed |
| Full slab extended to MXFP8 | 2526.32 | 64.44 | - | MXFP8 QMM rose from 65.3 ms to 68.4 ms total; removed |
| Adjacent-lane scale remap (v2) | 2238.75 | 64.92 | 8.04 ms | Regression; removed |

The `2663.56 / 64.47` adjacent-lane result is invalid. The remote host dispatch
had not been synchronized and reused the cached full-slab JIT kernel. The
corrected experiment used the distinct `gather_qmm_rhs_sm80_cs` module key and
produced `2238.75 / 64.92`.

## Sorted MoE Prefill

The initial model path created two large generic gathers per MoE layer:

- Sorted dispatch duplicated each BF16 activation row once per selected expert.
- Sorted combine gathered the down-projection rows back into token order before
  scaling and reducing the top-eight experts.

`FastSortedMoECombine` initially did not run because MLX `Argsort` returns
`uint32` indices while the fast path required `int64`. Changing the guarded
kernel configuration and its equivalence test to `uint32` produced:

| Experiment | Prompt | Generate | Prompt target | Generate target |
|---|---:|---:|---:|---:|
| Full scale slab | 2581.47 | 64.72 | 94.5% | 106.8% |
| Fused sorted combine | 2898.75 | 65.47 | 106.2% | 108.0% |
| Fused combine + vectorized sorted dispatch | 2941.27 | 64.20 | 107.7% | 105.9% |

The final profile confirms 58 launches each of
`custom_kernel_fused_sorted_moe_combine` and
`custom_kernel_sorted_moe_dispatch`. The generic gather total fell from
`45.4 ms` to `0.5 ms`; the replacement dispatch kernels cost `23.9 ms` and
move the 58 layers' activation traffic at approximately `221 GB/s`.

The remaining indexed activation materialization can only be removed by adding
row-map support to the sorted RHS QMM mainloop. That is a broader change because
the existing CuTe `cp.async` copy assumes contiguous activation rows. Treat it
as a separate experiment, not a small extension of the dispatch kernel.

## Laguna Sorted MoE and N1x Load

Laguna's CUDA prefill now uses vectorized row copies for both expert-sorted
dispatch and the inverse output permutation. On tater50, the accepted
p2048/g128 median is 3021.30 prompt and 69.82 generate tok/s, or 104.7% and
145.0% of the fixed 2886.23 / 48.16 official llama-server target.

An Nsight A/B on the same tater50 build isolated the effect of packed
quantized gate/up expert weights. The accepted packed path spent 320.60 ms in
152 sorted RHS QMM launches. Keeping gate/up separate spent 337.09 ms in 228
launches, a 16.49 ms increase. Packing is a useful optimization, but it is not
large enough to explain the N1x's broader execution gap.

The 64 GiB N1x cannot create the packed gate/up stacks during load because the
concatenation temporarily retains both source stacks and the full destination.
It also OOMs when the Windows materializer evaluates a 30-array lazy batch.
The validated load-safe policy keeps Laguna gate/up separate only on
`JMJWOA-Generic-GPU` and caps Windows materialization to one array there. A
coherent direct p2048/g128 run then completed at 85.54 prompt and 3.18 generate
tok/s. Do not profile the N1x over SSH; use direct correctness and benchmark
runs there, and investigate analogous kernel behavior with Nsight on tater50.

Disabling CUDA graphs produced a coherent three-epoch Laguna median of 86.67
prompt and 4.26 generate tok/s, a 1.3% prompt and 34.0% generation improvement.
That is not a safe global N1x policy: the same control reduced dense Qwen3.6
27B from 360.50 / 8.04 to 211.42 / 2.77 tok/s. Graph policy must remain
model- or graph-shape-specific until the underlying launch behavior is
understood.

The same direct control on Qwen3.6 35B-A3B passed semantic correctness and
improved the p2048/g128 median from 80.56 / 4.42 to 81.22 / 6.38 tok/s. The
three graph-disabled epochs were 81.22 / 6.21, 81.15 / 6.38, and 81.41 / 6.47.
That is a 44.3% generation improvement, but still only 10.2% / 18.4% of the
794.07 / 34.65 llama-server target. Do not treat the graph toggle as the root
fix or enable it globally: dense Qwen3.6 27B strongly prefers graphs on this
device.

Qwen3.6 35B-A3B MXFP8 originally failed during `loadSwitchMLP`: cloning every
separate quantized routed-expert gate/up stack into a packed tensor exceeded the
N1x's usable memory. The load-safe path now follows Laguna's established policy:
only `JMJWOA-Generic-GPU` retains separate quantized gate/up stacks and executes
two equivalent `GatherQMM` operations. A focused fused-versus-separate numerical
test matched, and the real `TestBasic/blue-sky` integration test returned a
coherent Rayleigh-scattering answer before timing. The valid graph-enabled
p2048/g128 epochs were 71.26 / 3.55, 70.81 / 3.42, and 66.55 / 3.45 tok/s
(median 70.81 / 3.45). No matching N1x Q8_0 target is available yet.

The optimized Qwen loader also used to create decode expert-index arrays
whenever CUDA was active. Dense Qwen has `NumExpertsPerTok == 0`, so that path
attempted an empty `mlx.FromValues` and panicked during load. Decode expert
maps are now cached only for quantized MoE layers with a positive top-K. The
guard has focused policy coverage and the dense Qwen N1x correctness and
p2048/g128 runs pass.

## Current Best

The active MLX worktree should contain these patches over the staged tuning
baseline:

1. `mlx-qmm-rhs-sm121-nvfp4-full-scale-slab-safe.patch`
2. `mlx-qmm-rhs-full-scale-slab-finalize-dispatch.patch`

Do not reapply the wide-MMA or MXFP8 full-slab patches. The current slab path is
SM12.1-only, NVFP4-only, unbiased, and guarded by shared-memory capacity.

## Iteration Guardrail

QMM JIT source changes require synchronizing both:

- `mlx/backend/cuda/device/qmm_sm80.cuh`
- `mlx/backend/cuda/quantized/qmm/qmm_sm80.cu`

Use a distinct module-name suffix for experiments so a cached NVRTC artifact
cannot masquerade as the new kernel. Confirm the resulting demangled kernel
signature in Nsight Systems before accepting benchmark data.

## Qwen Direct-QMV Follow-Up

Qwen3.6 35B-A3B generation on GB10 remains below llama-server after the
accepted router and gated-delta fusions: 70.51 versus 77.03 tok/s. NCU shows
the largest direct NVFP4 QMV shape is dependency-latency bound on packed-weight
loads rather than occupancy limited.

The following direct-QMV experiments are rejected and must not be retried
without a materially different design:

| Experiment | Outcome |
| --- | --- |
| Shared activation vector | Correct, but generation fell to 68.68 tok/s |
| Packed weight/scale software prefetch | Correct, but generation fell to 68.45 tok/s |
| `rows_per_block=16` | Incorrect; the semantic integration response was empty |
| Direct-only `rows_per_block=16` | Correct, but generation fell to 68.18 tok/s |
| Reuse the two-rows-per-warp gathered-QMV body for direct QMV | Correct; prompt rose from 2,450.72 to 2,469.84 tok/s, but generation fell from 77.23 to 75.74 tok/s |

The accepted `rows_per_block=8` implementation was rebuilt and passed the
Qwen semantic integration gate after the failed rows-16 candidate was removed.
Detailed profiles and the broader rejected-candidate inventory are in
`.cache/agent-notes/mlx-cuda-moe-kernel-tuning-20260726.md`.

The global rows-16 failure was a gathered-QMV grid mismatch, not a direct-QMV
arithmetic failure. A corrected direct-only follow-up retained eight-row
gathered QMV, passed correctness, and still regressed, so rows 8 remains the
validated direct launch geometry.

Qwen's two decode GatherQMM calls now reuse exact load-time LHS maps instead of
building an `Arange` subgraph for every call. A contiguous top-K gate/up map
also avoids materializing its broadcast. Together these remove 15,480 CUDA
graph kernel nodes per 128-token trace and improve matched generation from
69.27 to 71.99 tok/s. The maps are CUDA-only and single-token-only; sorted
prefill and Metal dispatch are unchanged.

The shared expert now fuses its compatible quantized gate/up projections on
known-good CUDA devices and uses a decode-only packed SwiGLU kernel. Metal,
unsupported quantization layouts, and the pre-release N1x CUDA device retain
the original path. Focused projection and activation equivalence tests plus
the full semantic integration gate pass. The final device-guarded tater50
p2048/g128 median is 2,450.72 prompt and 77.23 generate tok/s, or 110.1% and
100.3% of the 2,225.52 / 77.03 llama-server target.

Direct N1x A/B runs showed that the fused shared projection reduced generation
throughput on `JMJWOA-Generic-GPU`; a device-name guard therefore keeps its
separate gate/up projections. The final guarded direct run produced coherent
output and measured 80.18 prompt / 4.16 generate tok/s. Individual N1x epochs
remain variable, so use that result as a correctness and dispatch check rather
than profiling evidence.

Do not retry row-concatenating the independent recurrent QKV and Z
projections. Although summed Nsight kernel duration falls, MLX overlaps the
split graph branches; combining them serializes the critical path, adds about
425 MB of duplicate weights, and regresses the p32/g256 median from 77.75 to
75.59 tok/s. A compiled shared-result sigmoid/scale/add also failed the
p2048/g128 gate despite looking positive at short context.

## Dense Gemma4 BF16 Generation

The current tater50 Gemma4 31B BF16 p2048/g128 median is 984.26 prompt and
3.75 generate tok/s, or 136.6% and 98.2% of the fixed 720.71 / 3.82
llama-server target. Full semantic correctness passes. The generation profile
is at:

`/home/daniel/code/ollama-upstream-um/.cache/profiles/gemma4-bf16-generation/gemma4-31b-bf16-generation-current`

The two Ollama fused BF16 MLP kernels account for 65% of GPU kernel time:
14.96 s in the `5376 -> 21504` Gate+Up+GeGLU kernel and 7.57 s in the
`21504 -> 5376` Down kernel across the 128-token trace. They are
memory-bandwidth-bound and already outperform decomposing the MLP into MLX
GEMVs.

Rejected correctness-passing controls:

| Control | Prompt tok/s | Generate tok/s | Result |
| --- | ---: | ---: | --- |
| Eight-warps-per-row fused kernels | 981.34 | 3.71 | Superseded |
| Accepted 16-byte BF16 vector loads | 984.26 | 3.75 | Keep |
| Bypass fused kernels for expanding dense MLP | 992.36 | 3.66 | Rejected |
| MLX-style eight rows/block, one warp/row | 947.75 | 3.65 | Rejected |
| Two rows/block with shared input staging | 986.07 | 3.70 | Rejected |
| Custom BF16 Q/O attention projections | 928.61 | 3.71 | Rejected |

The accepted kernel loads eight BF16 values with an aligned 16-byte vector.
NCU measured the Gate+Up+GeGLU kernel at 2.01 ms and the Down kernel at
1.05 ms. Relative to the BF16x2 path, executed instructions fell by 20.5% and
26.7%, respectively. Reports:

- `.../gemma4-bf16-ncu/geglu-bf16x8/capture.ncu-rep`
- `.../gemma4-bf16-ncu/projection-bf16x8/capture.ncu-rep`

A BF16x4 midpoint was also tested after comparing llama-server's fused BF16
matvec. It passed `TestBasic/blue-sky`, raised achieved occupancy from 82.3%
to 98.6%, and raised eligible warps per scheduler from 0.06 to 0.07, but
regressed the focused kernel from 2.01 to 2.09 ms. Do not retry narrower
transactions without a new pipelining design that offsets their added
instructions.

The matching llama-server kernel uses the same `21504 x 256` launch and
measured 2.00 ms with graphs disabled under Nsight Compute, versus 2.01 ms for
the accepted MLX custom kernel. llama executed 54.1 million instructions
versus MLX's 45.7 million, with 0.08 versus 0.06 eligible warps and 6.58%
versus 5.61% issue-active. The two implementations are therefore effectively
tied at the kernel level; llama's BF16x2 load stream exposes slightly more
memory-level parallelism but spends substantially more instructions to do so.

Profiling llama-server on GB10 requires application replay. Kernel replay
cannot checkpoint this roughly 60 GB unified-memory model. Launch
`llama-server` directly with `GGML_BACKEND_PATH` set to `libggml-cuda.so`,
disable CUDA graphs only for profiling, use `--no-warmup`, and make the target
script issue its own deterministic request so every application-replay pass
executes the same launch. The ignored helpers are:

- `.cache/scripts/profile_llama_ncu_tater50.sh`
- `.cache/scripts/run_llama_profile_target_tater50.sh`

The full comparative report is:

`.../dgx-gemma4-llama-ncu/llama-gemma4-31b-bf16-geglu-app-full/capture.ncu-rep`

Do not route Gemma4 BF16 decode Q/O projections through
`FastBF16Projection`. Isolated `Eval` microbenchmarks looked favorable:
`5376 -> 8192` was 0.775 vs 0.851 ms, `5376 -> 16384` was 1.086 vs
1.461 ms, and O projections improved by 4-9%. The full p2048/g128 endpoint
instead regressed to a 928.61 / 3.71 median. Nsight showed why: the candidate
spent 7.05 s in custom Q/O kernels plus 4.25 s in remaining MLX GEMV, almost
identical to the control's 11.33 s total GEMV duration, while the custom graph
nodes reduced scheduling overlap. The rejected trace is:

`/home/daniel/code/ollama-upstream-um/.cache/profiles/gemma4-bf16-qo-projection/gemma4-31b-bf16-qo-projection-g128`

Do not retry these launch-layout controls. The remaining 1.8% generation gap
is not explained by a model graph error; closing it requires a genuinely
better fused bandwidth kernel or an upstream BF16 GEMV improvement.

Additional correctness-passing controls rejected on 2026-07-28:

| Control | Prompt tok/s | Generate tok/s | Reason |
| --- | ---: | ---: | --- |
| Gemma-only sliding OProj custom kernel | 981.47 | 3.74 | Dispatch confirmed, but MLX's matching K=8192 GEMV was already about 0.38 ms |
| MLX GEMV BF16 vector width 8 | 928.83 | 3.74 | Mixed shape behavior and 42 registers versus 29 |
| MLX cooperative K=8192 GEMV | 947.17 | 3.73 | Same kernel time as existing GEMV; no endpoint gain |
| Two rows/block, no shared input, down only | 915.43 | 3.70 | Isolated down kernel improved 1.05 to 1.03 ms, but halving the grid reduced full-graph throughput |
| Read-only activation loads in both MLP kernels | 986.39 | 3.75 | Median decode time regressed from 266.47 to 266.64 ms/token |
| Streaming Gate/Up weight loads | 966.68 | 3.75 | Median decode time regressed to 266.85 ms/token |
| GB10 graph limits 40 ops / 100 MB | 934.00 | 3.76 | Decode changed only from 266.47 to 266.15 ms/token while prompt regressed |
| GB10 graph limits 20 ops / 100 MB | 959.02 | 3.74 | Isolated byte-cap increase regressed decode to 267.26 ms/token |

The full-model OProj trace showed why the isolated microbenchmark was
misleading. The old `grid.x=672` GEMV bucket mixed 50 K=8192 sliding
projections with 10 slower K=16384 global projections. Once split by kernel,
the existing K=8192 path and the custom path were both about 0.38 ms. The MLX
GEMV experiments and cache hints must not be retried without new evidence.

The accepted NCU reports show that the generic throughput percentages are not
a useful LPDDR speed-of-light metric on GB10. The fused Gate+Up kernel reads
about 462 MB of BF16 weights in 2.01 ms and the Down kernel reads about 231 MB
in 1.05 ms, approximately 230 and 220 GB/s. That is already close to the
platform's practical memory bandwidth. Prior two-vector unrolling,
dual-accumulator chains, four-warp launch geometry, and cuBLASLt M=1 controls
were slower.

The accepted full-model GEMV trace grouped by launch geometry:

| Grid | Launches | Total | Average | Projection |
| --- | ---: | ---: | ---: | --- |
| 1024 | 12,900 | 4,979.4 ms | 386.0 us | Sliding Q and fused KV, 8192x5376 |
| 672 | 7,740 | 3,435.2 ms | 443.8 us | OProj, 5376x8192 or 5376x16384 |
| 32768 | 129 | 1,560.5 ms | 12.10 ms | LM head, 262144x5376 |
| 2048 | 1,290 | 1,091.1 ms | 845.8 us | Global Q, 16384x5376 |
| 256 | 1,290 | 261.0 ms | 202.3 us | Global K=V, 2048x5376 |

A correctness-first generation-heavy graph-limit sweep also crossed the large
MLP weight thresholds that the earlier 100 MB controls never reached. The
`20/25` control produced 3.80 and 3.79 tok/s at p32/g256. At p32/g128,
`40/100`, `40/512`, `40/800`, `80/800`, and `100/1000` produced 3.80, 3.78,
3.81, 3.80, and 3.78 tok/s respectively. Every setting passed
`TestBasic/blue-sky` first. Even allowing a single graph to hold the roughly
693 MB GeGLU-plus-down weight set did not produce a material gain, so do not
retry larger graph limits without new evidence.

CUDA 13 programmatic dependent launch (PDL) was also prototyped for the fused
GeGLU-to-down edge. The experiment added an opt-in MLX custom-kernel property,
propagated it through MLX-C, emitted a programmatic CUDA graph edge on CC 9.0+,
and placed the CUDA device wait before the first load and the release after the
last global load in both kernels. A focused numerical chain test and
`TestBasic/blue-sky` on `dhiltgen/gemma4:31b-mlx-bf16` both passed.

Nsight Systems showed that the launch geometry makes PDL ineffective here.
Across 1,020 GeGLU-to-down pairs, 984 technically overlapped, but their mean
overlap was only 0.367 us; including the 36 positive gaps, the downstream
launch started just 0.072 us before upstream completion on average. The
GeGLU grid has 21,504 blocks, so the edge cannot fire until the last block
reaches its trigger, effectively at kernel completion. That is less than
0.02% of the roughly 2 ms GeGLU duration and cannot materially move endpoint
throughput. The candidate was rejected before a full p2048/g128 run rather
than retaining a broad MLX/MLX-C API for noise-sized overlap. Trace:

`/home/daniel/code/ollama-upstream-um/.cache/profiles/dgx-gemma4-pdl/gemma4-31b-bf16-pdl-decode-g16`

Fusing sliding Q with its already-fused KV projection is not promising. It
would replace two 8192x5376 projections of about 386 us each with the observed
16384x5376 geometry at about 846 us, and would serialize two graph branches
that MLX can currently overlap.

## Qwen3.6 MXFP8

The first current-tuned Qwen3.6 35B-A3B MXFP8 tater50 run passed
`TestBasic/blue-sky` and measured a p2048/g128 median of 2261.55 prompt and
56.35 generate tok/s. Against the fixed 1972.06 / 55.62 Q8_0 target, that is
114.7% / 101.3%. The three epochs were 2095.59 / 57.13,
2276.36 / 56.30, and 2261.55 / 56.35.

## Qwen3.6 Gated-Delta Vector Loads

The Qwen recurrent kernel's Dk=128 shape assigns four contiguous Q/K and
recurrent-state elements to each lane. The original scalar loads generated
millions of excess sectors. On a representative p2048 gated-delta launch,
Nsight Compute reported 84.36% of cycles with no eligible warp, 0.42 eligible
warps per scheduler, 64.86 cycles of issue latency, and 36.66% memory/SM
throughput. Long-scoreboard stalls dominated at 34.64 cycles per issue.

The accepted CUDA-only candidate uses aligned `uint2` transactions for each
lane's four BF16 Q/K elements and `uint4` transactions for its four FP32 state
elements. It follows the union-based vector-load idiom already used by the
custom CUDA projection kernels. The compile-time generic path remains intact
for other Dk/dtype combinations, and the Metal kernel source is unchanged.

The representative Nsight Compute launch fell from 10.26 to 3.17 ms. Memory/SM
throughput rose from 36.66% to 87.31%, L1 throughput from 38.50% to 84.84%,
eligible warps from 0.42 to 1.04, and issue latency fell to 23.87 cycles.
Long-scoreboard stalls fell from 34.64 to 11.13 cycles per issue. In the
matched p2048 Nsight Systems capture, aggregate gated-delta time fell from
199.46 to 131.63 ms.

End-to-end tater50 results after `TestBasic/blue-sky`:

| Model | Control p/g | Candidate p/g | Fixed llama p/g | Candidate target |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.6 35B-A3B MXFP8 | 2261.55 / 56.35 | 2346.15 / 56.46 | 1972.06 / 55.62 | 119.0% / 101.5% |
| Qwen3.6 35B-A3B NVFP4 | 2450.72 / 77.23 | 2636.37 / 76.95 | 2225.52 / 77.03 | 118.5% / 99.9% |
| Qwen3.6 27B NVFP4 | 1439 / 12.82 | 1618.88 / 13.16 | 805.2 / 12.30 | 201.1% / 107.0% |

The MoE NVFP4 standard-shape decode delta is noise-sized; a decode-focused
p32/g256 run improved from 77.75 to 78.66 tok/s. Do not replace the vector
transactions with scalar loads or retry the local-pointer/lambda draft. The
union form has the same transactions, clearer compile-time selection, and
avoids questionable local pointer aliasing.

The same vector transactions regress the pre-release N1x. A direct MXFP8 A/B
with semantic correctness before each benchmark measured:

| N1x path | Prompt tok/s | Generate tok/s |
| --- | ---: | ---: |
| Vector loads | 68.25 | 3.30 |
| Scalar loads | 69.71 | 3.48 |

The final kernel therefore takes a device-policy template argument:
`JMJWOA-Generic-GPU` compiles the original scalar loops, while GB10, RTX 5090,
and other CUDA devices retain the vector path. Focused Dk=128 numerical tests
pass on both GB10's vector branch and N1x's scalar branch against the
backend-agnostic MLX recurrence.

Do not use the subsequent N1x full-model load failures as kernel evidence.
After the final rebuild, both dense Qwen 27B NVFP4 and Qwen 35B-A3B MXFP8
failed while `materializeModelWeights` evaluated lazy weights, before the
gated-delta kernel was created or invoked. The failures repeated once on an
idle host. The immediately preceding scalar and vector A/B runs both loaded
the MXFP8 model successfully, so this remains a separate pre-release
Windows/CUDA allocator instability.
### Rejected: route Gemma4 global fused K/V through `FastBF16Projection`

The Gemma4 BF16 decode profile suggested that the ten global-attention fused
K/V projections (`5376 -> 2048`) might benefit from the accepted custom BF16
projection kernel. A focused production-shape benchmark on tater50 rejected
that route before any model code was changed:

| Shape | Custom BF16 projection | cuBLAS GEMV |
| --- | ---: | ---: |
| `5376 -> 2048` | 111.965 us | 66.209 us |
| `5376 -> 1024` | 44.000 us | 27.140 us |

The custom kernel remains useful for the much larger MLP projections, but it is
under-occupied at these narrower output sizes. Keep the fused K/V projection on
the existing MLX matmul/GEMV path.

### Rejected: one gathered-QMV output row per warp

The accepted gathered FP QMV kernel uses two output rows per warp and four
warps per block. Because the tater50 gate/up profile spent 84.7% of cycles with
no eligible warp, an exact-math control used one row per warp and eight warps
per block. A direct 40-layer Qwen3.6 MoE chain used eight distinct weight sets
to exceed cache capacity and was run without profiling on N1x:

| Host | Two rows/warp | One row/warp | Change |
| --- | ---: | ---: | ---: |
| tater50 | 3.735542 ms | 3.545261 ms | 5.1% faster |
| N1x | 26.357660 ms | 26.734425 ms | 1.4% slower |

The candidate helps GB10 by exposing more independent warps, but it does not
address the N1x gap and slightly regresses the target platform. Keep two rows
per warp. The reusable direct cross-host probe is
`.cache/scripts/qwen_qmv_platform_test.go`.

### Rejected: memory-safe gate/up packing on N1x

N1x normally retains Qwen3.6's routed gate and up expert stacks separately.
The packed form used on other CUDA devices previously exceeded the load-time
memory budget because consumed source arrays remained registered until the
post-load sweep. A candidate added targeted handle release after each packed
replacement was fully evaluated. Focused lifetime and packed-vs-separate
numerical tests passed on tater50, and the real Qwen3.6 35B-A3B NVFP4
`TestBasic/blue-sky` test passed on both hosts. The N1x packed model loaded at
the expected roughly 20.8 GB, so the targeted release solved the load spike.

Runtime performance rejected the packed representation on N1x:

| N1x Qwen3.6 35B-A3B NVFP4 | Prompt tok/s | Generate tok/s |
| --- | ---: | ---: |
| Separate gate/up control | 80.18 | 4.16 |
| Packed gate/up candidate, median | 78.68 | 3.55 |

The candidate reduced generation by about 15%. On this WoA stack, one wider
gathered QMV is slower than two half-width gathered QMVs despite removing a
launch. Keep the N1x separate-stack policy. Do not retry load-time release as
a performance optimization unless the wide gathered-QMV behavior changes.

## RTX 5090 Qwen Direct-QMV Async Staging

Qwen3.6 35B-A3B NVFP4 decode remained far below its fixed RTX 5090
llama-server target after the accepted model-graph improvements. The
known-good p2048/g32 profile attributed 179.2 ms across 7,169 direct NVFP4 QMV
launches. Nsight Compute had previously shown 80.95% no-eligible-warp cycles
and a packed-weight load dependency feeding the native
`F2FP.F16.E2M1.UNPACK_B` conversion.

The accepted experiment adds a Blackwell-only direct-QMV specialization for
BF16 activations, NVFP4 group-16 weights, `K=2048`, and output rows divisible
by eight. It follows MLX Steel's existing `cp_async` helpers and
double-buffered pipeline pattern. Each warp asynchronously stages two
512-byte packed-weight tiles in 8 KB of block shared memory while retaining
the existing native FP4 conversion and FP32 accumulation order. Batched,
gathered, Metal, non-Blackwell, non-BF16, non-NVFP4, and other-K paths are
unchanged.

Correctness and primitive evidence:

- A focused comparison used identical quantized rows and forced the existing
  kernel with a ninth control row. The candidate's BF16 outputs were bit-exact.
- `TestBasic/blue-sky` passed for every candidate and control model trial.
- The `2048 -> 8192` primitive candidate measured 36.65, 37.62, and 38.18 us
  (37.62 us median). The existing kernel measured 73.20, 46.89, and 46.12 us
  (46.89 us median), a 19.8% median reduction.
- A p2048/g32 Nsight capture reduced direct FP4 QMV GPU time from roughly
  190 ms to 113 ms and improved profiled generation from 38.97 to 43.10 tok/s,
  or 10.6%.

Fresh-process p2048/g128 trials were noisy but agreed with the profile:

| RTX 5090 path | Prompt tok/s | Generate tok/s |
| --- | ---: | ---: |
| Existing kernel | 5,324.79 / 4,025.41 / 4,276.81 | 92.65 / 62.00 / 107.42 |
| Async-staged candidate | 4,017.25 / 4,278.28 / 9,282.21 | 107.02 / 76.43 / 101.36 |
| Median change | 4,276.81 to 4,278.28 | 92.65 to 101.36 (+9.4%) |

The first candidate median was 4,278.28 prompt and 101.36 generate tok/s.
A later matched retained-state control, after removing unrelated experiments
and rebuilding the installed payload, measured 5,326.96 prompt and 104.62
generate tok/s. This is 91.4% and 47.6% of the fixed 5,830 / 219.7
llama-server target. Continue using fresh-process matched controls while this
model exhibits large request-to-request variance.

Rejected related experiments:

- A separate activation-quantization plus DP4A direct-QMV path was coherent
  but reduced generation to a 62.00 tok/s median. A useful integer path would
  have to fuse activation quantization rather than add another graph node.
- A routed packed-SwiGLU kernel passed focused and full correctness, but its
  fresh-process median was 99.96 tok/s versus the accepted 105.71 tok/s
  same-process reference. As with the rejected QKVZ fusion, summed copy and
  activation kernel time overstated the endpoint opportunity because MLX
  overlaps independent graph branches.
- Staging only the second QMV tile reduced shared memory to 4 KB and remained
  bit-exact, but primitive latency rose to a 41.35 us median. Keep both tiles
  staged.

Artifacts:

- Candidate patch:
  `.cache/patches/mlx-fp-qmv-cp-async-k2048.patch`
- Rejected second-tile refinement:
  `.cache/patches/mlx-fp-qmv-cp-async-second-tile.patch`
- Focused exact-output test:
  `.cache/scripts/qmv_cp_async_test.go`
- Candidate profile:
  `/home/daniel/.codex/profiles/ollama-4784/rtx5090/qwen36-35b-a3b-nvfp4-cp-async-k2048-p2048g32`
- Endpoint trials:
  `.cache/bench/qwen36-5090-qmm-old-ab-20260729`

### Rejected follow-up experiments

All experiments below passed focused numerical checks and three independent
`TestBasic/blue-sky` semantic checks before endpoint benchmarking. Each was
then removed and the installed payload was rebuilt. The retained state is the
direct `K=2048` async-staged QMV candidate described above, the original
two-rows-per-warp gathered QMV, four warps per gated-delta block, and no
shared-gate fusion.

| Experiment | Candidate p2048/g128 median | Matched retained control | Result |
| --- | ---: | ---: | --- |
| Gathered QMV, one row/warp and eight warps/block | 6,536.00 / 70.52 | 5,326.96 / 104.62 | Reject: 32.6% decode regression |
| Gathered `K=2048` async staging | 4,618.32 / 97.21 | 5,326.96 / 104.62 | Reject: 7.1% decode regression |
| Fused BF16 shared-expert gate dot and sigmoid | 4,108.45 / 103.26 | 5,326.96 / 104.62 | Reject: no endpoint gain |

The one-row gathered kernel improved isolated gate/up and a synthetic MoE
chain, but it reduced full-model generation sharply. The gathered async kernel
was bit-exact and improved the isolated gate/up primitive, yet still lost at
the endpoint. The shared-gate kernel exactly matched eager MLX output, but
eliminating its visible scalar sigmoid launch did not improve throughput.
These results reinforce that summed kernel time overstates fusion opportunity:
MLX overlaps independent graph branches, so serializing them can lose more
than the removed launches save.

The production gated-delta shape (`B=1`, `T=1`, `Hk=16`, `Hv=32`, `Dk=128`,
`Dv=128`) was also tested with eight and sixteen warps per block. Both passed
correctness, but neither beat the retained four-warp launch consistently:

| Gated-delta launch | Median probe time |
| --- | ---: |
| Four warps/block | 217.6 us |
| Eight warps/block | 211.9 us |
| Sixteen warps/block | 216.5 us |

The apparent 2.6% eight-warp primitive improvement was within run variance and
did not justify increasing resource pressure. Further gated-delta work should
target data movement or computation inside the kernel, not launch geometry.

Reusable artifacts:

- Gathered one-row patch:
  `.cache/patches/mlx-fp-gather-qmv-rpw1-experiment.patch`
- Gathered async-staging patch:
  `.cache/patches/mlx-fp-gather-qmv-cp-async-k2048.patch`
- Direct and gathered exact-output tests:
  `.cache/scripts/qmv_cp_async_test.go`
- Gated-delta shape probe:
  `.cache/scripts/qwen_gated_delta_platform_test.go`
- Retained-state fresh trials:
  `.cache/bench/qwen36-5090-qmm-old-ab-20260729`

### RTX 5090 follow-up: exhausted exact-math scheduling refinements

Four additional exact-math refinements were tested after the retained-state
control. Invalid microbenchmark samples caused by sweeping unpinned base
arrays were discarded; the corrected harness pins input weights and scales
across every timed shape.

Gated-delta refinements:

| Candidate | Production-shape probe | Result |
| --- | ---: | --- |
| Retained four-warp kernel | 221.72 us | Control |
| Stage shared Q/K once per block | 223.74 us | Reject: L1 already hides reuse and the barrier costs more |
| Two value rows per warp | 240.84 us | Reject: register and shuffle pressure dominate |

Both candidates matched the MLX fallback at the production
`B=1,T=1,Hk=16,Hv=32,Dk=128,Dv=128` shape before timing.

Direct QMV refinements:

| Candidate | Isolated evidence | p2048/g128 endpoint | Result |
| --- | --- | ---: | --- |
| Stage the BF16 activation once in an eight-warp block | 21.4% faster than the matched micro-harness control | 5,063.55 / 97.52 | Reject |
| Stage the activation with four warps and strided full-vector copies | Similar isolated median to the eight-warp form | 3,860.78 / 96.23 | Reject |
| Use async staging only for `N >= 4096` | `2048x512` was 42.75 us versus 41.52 us retained | Not run | Reject before endpoint |
| Hoist paired E2M1 conversions | `2048x8192` improved 40.23 to 38.98 us, but dominant `2048x512` regressed 41.52 to 44.21 us | Not run | Reject: sub-1% theoretical model gain |

Every endpoint candidate passed three independent
`TestBasic/blue-sky` checks. Activation staging increased isolated QMV
throughput but reduced full-model throughput because its resource and
scheduling effects interact poorly with MLX's concurrent graph. The four-warp
variant retained the original 8 KB shared-memory footprint but still
regressed, ruling out shared-memory capacity as the only cause.

The retained Nsight trace maps direct `K=2048` QMV cost as follows:

| Output rows | Calls | Total | Average |
| ---: | ---: | ---: | ---: |
| 512 | 3,300 | 47.473 ms | 14.386 us |
| 8,192 | 1,320 | 14.999 ms | 11.363 us |
| 4,096 | 990 | 12.298 ms | 12.422 us |

The `N=512` group is the separate Qwen shared-expert gate/up projections on
RTX 5090. Packing those projections is intentionally disabled on that device:
prior endpoint testing found that MLX's overlap of the independent branches
beats the packed graph. Do not infer a fusion opportunity from summed QMV
duration.

These results close the low-risk scheduling axes around the current W4A16
kernel. llama.cpp's fast Q4 decode instead quantizes activations and uses
integer dot-product kernels. Prior MLX Q8/DP4A and native FP4 tensor-core
experiments were slower and changed the numerical contract. A material next
step therefore requires a new exact W4A16 kernel architecture or an explicit
model-quality decision, not another launch-size, cache, prefetch, or simple
fusion tweak.

Additional artifacts:

- Activation staging:
  `.cache/patches/mlx-fp-qmv-stage-vector-experiment.patch`
- Four-warp staging:
  `.cache/patches/mlx-fp-qmv-stage-vector-rows4-experiment.patch`
- Four-warp copy correction:
  `.cache/patches/mlx-fp-qmv-stage-vector-strided-copy-fix.patch`
- Async dispatch gate:
  `.cache/patches/mlx-fp-qmv-async-min-n4096-experiment.patch`
- Paired E2M1 conversion:
  `.cache/patches/mlx-fp-qmv-hoist-e2m1-pair-experiment.patch`
- Shape summary:
  `.cache/scripts/summarize_qwen_qmv_trace.py`

### Rejected 5090 four-way split-K direct QMV

A four-way split-K direct NVFP4 QMV used four warps per output row for
`K=2048` and `N >= 4096`. The intent was to shorten the packed-weight load to
E2M1 conversion dependency chain without changing the BF16 activation, FP32
accumulation, or quantization contract.

The standalone varied-weight CUDA probe was bit exact and looked strong:

| Shape | Retained scalar | Split-K4 | Primitive change |
| --- | ---: | ---: | ---: |
| `N=512,K=2048` | 9.349 us | 7.713 us | 1.21x |
| `N=4096,K=2048` | 14.885 us | 10.371 us | 1.44x |
| `N=8192,K=2048` | 22.926 us | 12.586 us | 1.82x |

Production-dispatch tests forced an odd-row control through the retained
kernel. The candidate matched all 4,096 outputs at `M=1` and all 16,384
outputs at `M=2` exactly. `TestBasic/blue-sky` and a natural long-prompt probe
also returned coherent answers.

The full graph rejected the candidate. The p2048/g128 three-epoch generation
mean fell to 73.4 tok/s from the retained 104.6 tok/s result. A short Nsight
Systems decode A/B explained why the isolated result did not transfer:

| Production shape | Retained | Split-K4 |
| --- | ---: | ---: |
| `N=4096,K=2048` | 8.95 us | 9.17 us |
| `N=8192,K=2048` | 18.86 us | 14.77 us |
| Profiled endpoint | 32.63 tok/s | 32.71 tok/s |

The extra warps improve only the widest projection and consume graph-overlap
capacity elsewhere. Reject and restore the retained async direct QMV. Do not
retry split-K without a design that preserves row-level parallelism and MLX
graph concurrency.

Artifacts:

- `.cache/patches/mlx-fp-qmv-splitk4-k2048-experiment.patch`
- `.cache/scripts/nvfp4_splitk_qmv_probe.cu`
- `.cache/profiles/qwen36-35b-a3b-nvfp4-splitk4-p16g32`
- `.cache/bench/qwen36-5090-splitk4-20260730`

### Rejected 5090 graph-limit retuning

The retained consumer-Blackwell graph limits are 800 operations and 8000 MB.
A correctness-gated split sweep showed that smaller graphs can recover prompt
throughput, but do not close the decode gap and reduce the retained generation
median.

| Graph limits | Prompt tok/s | Generate tok/s |
| --- | ---: | ---: |
| 400 ops / 8000 MB | 5355.58 | 91.69 |
| 800 ops / 4000 MB | 4333.36 | 88.91 |
| 400 ops / 4000 MB, three-run median | 5643.09 | 94.83 |
| Retained matched control | 5326.96 | 104.62 |

The 400/4000 median is 96.8% prompt and 43.2% generation of the fixed
5830 / 219.7 llama-server target. Lowering either threshold affects graph
execution, but the operation cap is the stronger prompt lever and neither
threshold is the missing decode optimization. Keep the existing 800/8000
consumer-Blackwell policy while the retained graph supports better generation.
The sweep artifacts are under
`.cache/bench/qwen36-5090-graph-limits-20260729` and
`.cache/bench/qwen36-5090-graph-split-20260729`.

### Rejected 5090 exact M=1 CuTe QMM

MLX's existing `qmm_sm80` path already implements the desired quality-neutral
math: it dequantizes NVFP4 weights to BF16 fragments and uses BF16 tensor-core
MMA with FP32 accumulation. An environment-gated probe forced direct M=1
matmuls through that path and then tested smaller N tiles. Every output was
bit-exact to the retained QMV for constant production-shape inputs.

| M=1 geometry | N=512 | N=4096 | N=8192 |
| --- | ---: | ---: | ---: |
| Retained QMV, representative repeat | 54.54 us | 58.60 us | 47.52 us |
| QMM tile N=128 | 92.91 us | 73.26 us | 70.51 us |
| QMM tile N=64 | 66.35 us | 72.95 us | 62.74 us |
| QMM tile N=32, repeat | 68.33 us | 59.98 us | 62.16 us |

Smaller N tiles substantially improve the generic QMM and tile N=32 reaches
QMV parity at N=4096, but it still loses the other dominant shapes. The
hardware MMA's 16-row M tile wastes fifteen rows for decode, so retuning the
generic CuTe CTA cannot provide the required model-level gain. A useful
follow-up must be an M=1-native W4A16 architecture rather than dispatching the
existing QMM. Preserved artifacts:

- `.cache/patches/mlx-direct-qmm-dispatch-probe.patch`
- `.cache/patches/mlx-sm120-qmm-m1-tile-n64-experiment.patch`
- `.cache/patches/mlx-sm120-qmm-m1-tile-n32-followup.patch`
- `.cache/scripts/qmm_m1_ab_test.go`

### Rejected 5090 two-stage FP8-by-FP4 MMA QMV

The native SM120 follow-up removed the earlier block-scaled kernel's largest
structural flaw: activation conversion was moved to one separate launch and
reused by every output tile. The QMV used the canonical CUTLASS
`SM120_16x8x32_TN<E4M3,E2M1,F32>` register layouts. To retain every NVFP4
weight's E4M3 group-16 scale, each K16 group was evaluated independently in
one half of a K32 MMA and its partial accumulator was scaled before summation.

The kernel was exact relative to an E4M3-quantized activation reference, but
the activation conversion introduced 2.3-3.0% normalized RMS error relative
to the BF16 activation contract. It was also much slower than the retained
exact W4A16 QMV:

| K2048 output rows | Activation conversion | QMV | Total |
| ---: | ---: | ---: | ---: |
| 512 | 7.44 us | 61.52 us | 68.96 us |
| 4,096 | 7.50 us | 61.67 us | 69.17 us |
| 8,192 | 8.16 us | 61.56 us | 69.72 us |
| 248,320 | 7.80 us | 661.10 us | 668.90 us |

The retained scalar path is roughly 5-18 us for the direct model shapes, and
the existing MXFP8 output head is roughly 645 us. Separating activation
conversion therefore does not rescue the tensor-core approach. The required
K16 scale fidelity forces half-utilized K32 MMA instructions, while the
separate conversion adds graph work and changes numerical quality. Do not
integrate this path or retry it with launch/CTA tuning.

Artifact:

- `.cache/scripts/sm120_fp8_fp4_qmv_probe.cu`

## N1x Sorted-RHS QMM Bring-Up

The staged MLX CUDA tuning baseline had not been deployed to the N1x control
mirror. A fresh `sm_121a` build added the staged sorted-RHS GatherQMM path and
specialized gathered FP QMV wiring, plus an active N1x QMM experiment that:

- uses `tile_n=64` on compute capability 12.1;
- loads quantization scales and biases directly from global memory instead of
  staging them in shared memory.

CUDA 13.4 WoA exposed a host-compiler portability issue in the staged softmax
optimization. Its nested kernel-selection lambda captured the dispatched
`block_dim` and `N_READS` integral constants, which the WoA NVCC/MSVC path did
not preserve as constant expressions. The performance-neutral fix uses
`decltype(block_dim)::value` and explicit `if constexpr` launch branches.

The first full-model attempt was invalid: the remote Ollama source mirror still
contained the rejected packed gate/up load experiment. Its model-size
fingerprint was 20,837,299,264 bytes and it OOMed while materializing weights.
The accepted separate-stack source is 20,837,332,032 bytes with the current
decode-index arrays. Restoring the local accepted `qwen3_5.go` made the model
load normally through the one-array N1x materialization policy. The stale
remote file was preserved as
`qwen3_5.gateup-release-rejected-20260729.go`.

The focused direct/gathered NVFP4 probe passed finite-output checks:

| Shape | Average |
| --- | ---: |
| Direct 2048x8192 | 615.907 us |
| Gather gate/up 2048x1024 top-8 | 563.504 us |
| Gather down 512x2048 top-8 | 461.910 us |
| 40-layer MoE chain | 16.615 ms |

`TestBasic/blue-sky` then passed with a coherent Rayleigh-scattering response.
The graph-disabled p2048/g128 epochs were:

| Epoch | Prompt tok/s | Generate tok/s |
| ---: | ---: | ---: |
| 1 | 462.89 | 6.64 |
| 2 | 488.09 | 6.86 |
| 3 | 483.94 | 6.74 |
| Median | 483.94 | 6.74 |

Against the fixed 794.07 / 34.65 llama-server target, this is 60.9% prompt and
19.5% generation. Relative to the prior valid 81.22 / 6.38 MLX control, prompt
improved by 495.8% and generation by 5.6%. The next experiment must separate
the active `tile_n=64` direct-scale change from the staged sorted-RHS baseline
before attributing the prompt gain.

The clean staged control restored `tile_n=128` and shared scale/bias staging
while retaining sorted RHS and specialized gathered QMV. Its focused numerical
probe passed; the 40-layer MoE chain improved from 16.615 ms to 7.548 ms. The
first full load hit the N1x's intermittent `cudaMallocAsync` OOM with no stale
process or memory pressure. The required clean retry loaded normally and
passed `TestBasic/blue-sky` with a coherent response.

Staged-control p2048/g128 epochs:

| Epoch | Prompt tok/s | Generate tok/s |
| ---: | ---: | ---: |
| 1 | 893.74 | 20.69 |
| 2 | 905.08 | 20.99 |
| 3 | 887.49 | 17.95 |
| Median | 893.74 | 20.69 |

This is 112.5% prompt and 59.7% generation of the 794.07 / 34.65 target. The
sorted-RHS baseline has therefore achieved prompt parity on N1x. Direct global
scale loads plus `tile_n=64` reduced prompt by 45.9% and generation by 67.4%
relative to the staged control. Reject that experiment and retain shared scale
staging with `tile_n=128`. Its preserved patch is
`.cache/patches/n1x-qmm-tn64-direct-scale-rejected.patch`.

### Rejected: Blackwell direct-QMV async staging on N1x

There is one MLX optimization worktree and branch for all three Blackwell
targets: `mlx-nvidia-um` on `cuda-unified-memory-kernels`. Platform validation
uses separate installed artifacts, not separate source branches. Unstaged
kernel candidates are accepted into that shared branch only after correctness
and endpoint checks on the affected platforms.

The bit-exact `K=2048` `cp.async` direct-QMV candidate previously validated on
RTX 5090 was cross-tested unchanged on N1x. Nsight Compute confirmed dispatch
to `fp_qmv_async_k2048_single<bf16, 8>`, and the focused exact-output test
passed. The isolated `2048x8192` probe measured 0.519 ms, versus roughly
1.03 ms in the earlier clean-control probe, but the full graph regressed
severely despite coherent Blue Sky output:

| N1x Qwen3.6 35B-A3B NVFP4 | Prompt tok/s | Generate tok/s |
| --- | ---: | ---: |
| Shared staged baseline | 893.74 | 20.69 |
| Direct-QMV async candidate | 396.95 | 7.17 |

Only the two complete 128-token candidate epochs were used; the third stopped
at ten tokens. The candidate reduced endpoint prompt by 55.6% and generation
by 65.3%. This is another case where an isolated kernel win damages MLX graph
overlap/resource scheduling. Do not enable this kernel for all Blackwell
devices or add a host-name special case. The experiment remains preserved at
`.cache/patches/mlx-fp-qmv-cp-async-k2048.patch` while a common Blackwell
kernel design is pursued. The N1x source and installed payload were restored
to the shared staged baseline after rejection.

## CUDA graph sizing on compute capability 12.1

MLX CUDA currently defaults compute capability 12.1 to 20 operations or
25 MB per graph. The limit was introduced as an initial device-specific
starting point and can be overridden with `MLX_MAX_OPS_PER_BUFFER` and
`MLX_MAX_MB_PER_BUFFER`.

An artifact mix-up initially invalidated several graph trials: the tater50
`current-best` install was older and materially slower than the newer
`current` install. The verified candidate is:

```text
/home/daniel/.codex/dist/ollama-upstream-um-current-mlx-cuda13-sm121
```

The two artifacts also reported different Qwen model residency, approximately
21.16 GB for the stale install versus 20.67 GB for the verified candidate.
All results below used the verified candidate, passed
`TestBasic/blue-sky`, and contain three p2048/g128 epochs.

Qwen3.6 35B-A3B NVFP4 generation medians on DGX Spark:

| Ops | MB | Prompt tok/s | Generate tok/s |
| ---: | ---: | ---: | ---: |
| 20 | 25 | 2,669.50 | 72.00 |
| 20 | 256 | 2,668.34 | 73.23 |
| 100 | 25 | 2,652.79 | 75.11 |
| 40 | 100 | 2,479.24 | 83.30 |
| 100 | 64 | 2,624.19 | 82.77 |
| 100 | 128 | 2,672.40 | 81.92 |
| 100 | 256 | 2,664.29 | 84.09 |
| 100 | 512 | 2,670.00 | 80.58 |
| 200 | 256 | 2,505.45 | 84.92 |
| 200 | 512 | 2,610.36 | 86.31 |
| 200 | 1,024 | 2,630.28 | 77.76 |
| 400 | 512 | 2,673.05 | 86.49 |
| 400 | 1,024 | 2,665.45 | 77.72 |
| 800 | 2,048 | 2,433.26 | 76.91 |

The fixed llama-server target is 2,225.52 prompt and 77.03 generate tok/s.
The 400/512 result reaches 120.1% prompt and 112.3% generation, but larger
graphs fall back toward the target. The graph-size response is non-monotonic
and has a clear workload-specific knee.

Dense Gemma4 31B NVFP4 does not share the same optimum:

| Ops | MB | Prompt tok/s | Generate tok/s |
| ---: | ---: | ---: | ---: |
| 20 | 25 | 1,157.64 | 10.11 |
| 40 | 100 | 1,094.01 | 9.95 |
| 100 | 64 | 1,036.23 | 10.19 |
| 100 | 128 | 1,127.95 | 10.23 |
| 400 | 512 | 1,110.98 | 9.97 |

A blanket 400/512 default would therefore improve Qwen MoE while regressing
Gemma. Do not hard-code that static value. The next CUDA-backend experiment
should inspect graph composition and commit boundaries so prefill and decode
can use different effective graph sizes based on activation/workload
properties, without model-name or host-name checks.

The N1x has not rebooted since its profiler-induced allocator failures. Its
remote source/install still contains the rejected four-row direct-QMV staging
candidate and should not be used for endpoint conclusions until reboot,
restoration to the shared source, rebuild/install, correctness, and a fresh
20/25 control.

### Adaptive sm121 graph limits

Static graph limits are not a safe device default. A CUDA-backend experiment
instead classifies each graph from its outputs:

- wide 2D/3D activations with more than 64 logical rows retain sm121's
  conservative 20-op/25-MB limits;
- positively identified small-batch activations use 400 ops/512 MB;
- higher-rank recurrent states and narrow metadata arrays are not treated as
  batch-size evidence;
- the sm121 graph cache defaults to 800 entries to prevent prefill topologies
  from evicting decode graphs;
- explicit graph-limit and cache environment variables remain authoritative;
- compute capability 12.0 and all other architectures keep their existing
  limits and 400-entry cache.

The first classifier assumed small batch until it observed a large output.
That preserved most prompt performance but allowed metadata/state operations
at the beginning of prefill to open a decode-sized graph. It also thrashed the
400-entry graph cache, causing Qwen's third generation epoch to fall from
about 86 to 75 tok/s. Requiring positive small-activation evidence and using
800 cache entries removed the collapse.

Final no-override p2048/g128 results from the exact shared Ollama and MLX
source mirror on DGX Spark:

| Model | Configuration | Prompt tok/s | Generate tok/s |
| --- | --- | ---: | ---: |
| Qwen3.6 35B-A3B NVFP4 | 20/25 control | 2,291.40 | 72.22 |
| Qwen3.6 35B-A3B NVFP4 | Adaptive default | 2,295.73 | 89.63 |
| Gemma4 31B NVFP4 | 20/25 control | 1,242.51 | 10.10 |
| Gemma4 31B NVFP4 | Adaptive default | 1,261.75 | 10.27 |
| Qwen3.6 35B-A3B MXFP8 | Adaptive default | 2,410.47 | 57.03 |

Every row passed `TestBasic/blue-sky`; each performance row is the median of
three p2048/g128 epochs after one warmup. Against Qwen's fixed
2,225.52/77.03 llama-server target, the adaptive NVFP4 result is 103.2%
prompt and 116.4% generation. Relative to the shared 20/25 control, it
improves generation by 24.1% with effectively unchanged prompt. Gemma improves
prompt by 1.5% and generation by 1.7%, so the adaptive policy avoids the dense
model regression seen with a blanket 400/512 default.

The tater50 source divergence discovered during this work was preserved before
replacement:

- `.cache/patches/mlx-tater50-current-{staged,unstaged,full}.patch`
- `.cache/patches/ollama-tater50-current-{staged,unstaged,full}.patch`

The old remote MLX delta contains an earlier sorted-RHS implementation plus
sm121-only wide MMA and scale-slab staging. It was not replayed verbatim onto
the newer shared QMM design. Its prompt-side benefit remains a future
reconciliation candidate; it was not required for the adaptive graph
generation gain.

### Native SM120 NVFP4 direct QMV

A fused SM120 block-scaled MMA prototype quantized one BF16 activation vector
to NVFP4 in shared memory and used
`mma.sync.aligned.kind::mxf4nvf4.block_scale` against the existing NVFP4
weights. The standalone CUDA probe was correct within the expected activation
quantization error and looked excellent:

| Shape | Average latency | Max relative error |
| --- | ---: | ---: |
| K2048 x N512 | 10.14 us | 0.00338 |
| K2048 x N4096 | 10.37 us | 0.00360 |
| K2048 x N8192 | 9.41-11.25 us | 0.00355 |
| K2048 x N16384 | 12.59 us | 0.00333 |

The required CUDA 13 compile target is the architecture-specific form
`--generate-code=arch=compute_120a,code=[compute_120a,sm_120a]`. Plain
`-arch=sm_120` cannot assemble the block-scaled MMA instruction.

The result did not survive an end-to-end Qwen3.6 35B-A3B NVFP4 test. Blue-sky
correctness passed, but p2048/g128 measured only 3,489 prompt and 58.02
generate tok/s. The p2048/g32 Nsight trace showed the native kernel as the top
GPU hotspot:

| Output shape | Calls | Average latency |
| --- | ---: | ---: |
| N512 | 3,300 | 39.17 us |
| N4096 | 990 | 30.80 us |
| N8192 | 1,320 | 19.61 us |

Across all shapes it consumed 185.6 ms over 5,610 launches. Qwen schedules
direct projections concurrently with tensor-core MoE work. Moving the direct
QMV from CUDA-core unpack/FMA work onto the same tensor-core resource creates
contention, while each output-row block also repeats activation quantization.
The isolated microbenchmark therefore overstates the model-level value.

The production-source experiment was reverted. Its reproducible artifacts are
retained at:

- `.cache/scripts/nvfp4_mma_qmv_probe.cu`
- `.cache/patches/mlx-sm120-native-nvfp4-qmv-experiment.patch`
- `.cache/bench/qwen36-5090-native-nvfp4-mma-20260729`
- `/home/daniel/.codex/profiles/ollama-4784/rtx5090/qwen36-native-nvfp4-mma-p2048g32`

Do not retry a wholesale native-MMA replacement for direct QMV. A future
version would need workload-aware dispatch and activation-quantization reuse,
and must prove a win under the real concurrent graph rather than only in a
single-kernel benchmark.

### FP8 K2048 output-head async staging

The decode-only Qwen3.6 trace identified its 248,320-row FP8 output projection
as a distinct hotspot: the generic BF16 x MXFP8 QMV took 645.43 us/token,
83.26 ms over 129 calls, or 9.5% of traced GPU time.

An exact-math experiment extended the retained Blackwell `cp_async` QMV
pattern to FP8/group-32/K2048. It used four double-buffered 512-value tiles
and preserved the generic kernel's scale addressing and FP32 accumulation
order. A focused eight-row candidate versus nine-row generic control was
bit-exact, and full Qwen Blue Sky correctness passed.

The real output head regressed:

| Path | Head latency | Profiled generation |
| --- | ---: | ---: |
| Generic FP8 QMV | 645.43 us/token | 37.24 tok/s |
| Async-staged FP8 QMV | 662.05 us/token | 36.39 tok/s |

The experiment was removed. The output head is limited by the scalar
FP8-unpack/FMA architecture rather than ordinary global-load latency. A future
attempt should use a different compute architecture, likely a narrowly
dispatched native FP8 tensor-core path for the isolated large-N head, rather
than more scalar-kernel staging.

Artifacts:

- `.cache/patches/mlx-sm120-fp8-qmv-cp-async-k2048-experiment.patch`
- `.cache/scripts/qmv_cp_async_test.go`
- `.cache/bench/qwen36-5090-fp8-qmv-async-20260729`
- `/home/daniel/.codex/profiles/ollama-4784/rtx5090/qwen36-fp8-qmv-async-p16g128`

### Rejected Qwen gated-delta BA projection fusion

Qwen3.6 35B-A3B stores separate unquantized BF16 beta and alpha projection
weights in the tested checkpoint. A load-time experiment interleaved them into
the checkpoint-native combined `in_proj_ba` layout, replacing two `2048x32`
GEMVs with one `2048x64` GEMV in each of the 30 linear-attention layers. The
blue-sky integration test passed with coherent output, but a matched fresh
process p16/g128 comparison on the RTX 5090 regressed generation from 75.45 to
64.63 tok/s (14.3%). Prompt also fell from 92.48 to 77.74 tok/s.

Standard GEMV launch reduction is therefore not enough for this shape. The
wider GEMV and split-view graph cost more than the 60 eliminated decode
launches save. Do not retry load-time BA packing without a purpose-built fused
consumer or new evidence that MLX's `N=64` BF16 GEMV path has improved.

Artifacts:

- `.cache/bench/qwen36-5090-fused-ba-20260729/control-p16g128.csv`
- `.cache/bench/qwen36-5090-fused-ba-20260729/fused-ba-p16g128.csv`

### Rejected Blackwell M=8 gathered-QMM dispatch

MLX normally prefers gathered QMV through eight rows on post-Hopper devices.
A one-line CUDA-backend experiment changed the boundary so the eight selected
MoE experts used the existing SM80+ QMM path without changing arithmetic.

The RTX 5090 focused Qwen NVFP4 probe measured:

| Primitive | Gathered QMV | M=8 QMM |
| --- | ---: | ---: |
| Gate/up, `2048 -> 1024`, top-8 | 80.069 us | 79.986 us |
| Down, `512 -> 2048`, top-8 | 68.752 us | 53.736 us |
| 40-layer MoE chain | 2.741853 ms | 3.165097 ms |

The tensor-core path improves the narrow down operation in isolation but
regresses the complete chain by 15.4%, matching the model-level contention
seen with native-MMA direct QMV. Retain MLX's Blackwell `M <= 8` gathered-QMV
preference. Do not retry an existing-QMM dispatch change without a different
M=8 kernel architecture and a full-chain win.

Artifact:

- `.cache/patches/mlx-blackwell-gather-qmm-m8-experiment.patch`

### Rejected gathered NVFP4 gate/up plus SwiGLU fusion

A generic CUDA custom kernel fused a decode-only, top-8 NVFP4 gate/up
projection with SwiGLU. The kernel was bit-exact against a focused
GatherQMM-plus-SwiGLU reference and took 60.509 us versus 187.115 us for that
reference. A p16/g128 model run also improved generation from 75.45 to
93.04 tok/s.

That focused reference was not representative of the production graph. Qwen
already stores gate and up as one packed projection, so the retained model
performs one gathered QMV and then SwiGLU rather than two independent gathered
projections. A matched p2048/g32 Nsight trace showed the custom kernel
dispatching exactly 1,320 times across 40 layers and 33 evaluations:

| Path | Calls | GPU time |
| --- | ---: | ---: |
| Retained packed gate/up GatherQMV | 1,320 | about 33.8 ms |
| Custom gathered gate/up plus SwiGLU | 1,320 | 55.9 ms |

The candidate's p2048/g128 median was 3,908.12 prompt tok/s and 104.50
generation tok/s, versus the retained 5,326.96 and 104.62 tok/s. It therefore
adds complexity without improving the standard workload and materially
regresses prompt processing. The production experiment was removed.

Artifacts:

- `.cache/bench/qwen36-5090-fused-gather-swiglu-20260729`
- `/home/daniel/.codex/profiles/ollama-4784/rtx5090/qwen36-fused-gather-swiglu-p2048g32`

### Rejected generic top-8 weighted-sum fusion

A model-independent CUDA custom kernel replaced the decode-only BF16
`expert_output * scores` followed by the top-8 reduction. It preserved MLX's
BF16 rounding after every multiply and add and matched the eager expression
bit-for-bit in the focused CUDA test. The real Qwen blue-sky integration test
also passed with a coherent response.

The candidate improved the matched p16/g128 generation check from 75.45 to
78.24 tok/s, but failed the standard workload:

| RTX 5090 Qwen NVFP4 | Prompt tok/s | Generate tok/s |
| --- | ---: | ---: |
| Retained p2048/g128 | 5,326.96 | 104.62 |
| Weighted-sum candidate median | 6,146.04 | 93.99 |

The function cannot affect prefill because its fast path accepts only
`B=L=1`; the prompt difference is run variance. The 10.2% generation
regression shows that serializing the multiply and reduction into one graph
node harms full-graph overlap despite saving an intermediate and a launch.
The production experiment was removed.

Artifact:

- `.cache/bench/qwen36-5090-weighted-sum-20260729`

## N1x accepted-source deployment

The Ollama and MLX working trees intentionally use the Git index as the
accepted experiment checkpoint. Unstaged tracked changes and untracked source
files are active experiments and must not be sent to tater62 unless the run is
explicitly labeled `WorkingTree`.

Use:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .cache\scripts\stage_tater62_current_best.ps1 `
  -SourceState Accepted
```

`Accepted` is not a raw index archive. The staged interfaces currently require
these narrowly defined working-tree overlays:

- `x/mlxrunner/mlx/ops_extra.go`: supplies the fast QQMM implementation and
  backend-cache reset used by staged callers.
- `x/mlxrunner/mlx/mlx.go`: supplies `CUDAIsAvailable`, used by the staged
  fast-path callers and focused CUDA tests.
- `x/mlxrunner/mlx/moe_router.go`: supplies the normalized 256-expert router
  implementation called by the staged Qwen model.
- `x/models/nn/nn.go`: supplies the fast-matmul state carried by staged Qwen
  quantized projections.
- `x/mlxrunner/runner.go`: materializes one lazy weight array at a time on
  `JMJWOA-Generic-GPU`; without it, Qwen load intermittently exhausts the
  N1x's usable unified-memory budget.
- `x/models/qwen3_5/qwen3_5.go`: generated from the staged file plus
  `.cache/patches/qwen3_5-n1x-memory-safe-staged.patch`; it retains separate
  quantized expert gate/up stacks on `JMJWOA-Generic-GPU` so the 64 GiB N1x
  can load the model without materializing a second full packed copy. Do not
  overlay the complete working file because it also contains active router and
  projection experiments.
- `mlx/backend/cuda/softmax.cu`: preserves the staged softmax optimization
  while applying the performance-neutral CUDA 13.4/MSVC constant-expression
  portability fix.

The staging helper creates four archives and the prepare helper resets both
generated source mirrors before extraction. Audit the overlays before every
deployment. The accepted control must have the 128-wide QMM tile, no async QMV
candidate, and no CUDA profiler hooks in `pipeline.go`.

Do not infer remote build state from a local SSH timeout. A foreground
`tater62_build_current_best.ps1` process can survive after its SSH client is
terminated. Before retrying, inspect the remote process tree for the wrapper,
`cmake`, `ninja`, `nvcc`, and `cl`; a second configure against a live build
will fail Ninja recompaction with `Permission denied`. Do not use detached
`Start-Process` over SSH because the child may instead die when the SSH
session closes.

The earlier documented Qwen3.6 35B-A3B NVFP4 N1x result of 893.74 prompt
tok/s and 20.69 generation tok/s is historical but is not reproducible from a
fresh build of the source state described above. Do not use it as the active
baseline or as evidence for an optimization.

The source-provenance-checked control was rebuilt from scratch with CUDA 13.4
for `sm_121a`, passed `TestBasic/blue-sky`, and then ran three p2048/g128
epochs:

| Epoch | Prompt tok/s | Generate tok/s |
| --- | ---: | ---: |
| 1 | 333.78 | 15.11 |
| 2 | 397.95 | 14.91 |
| 3 | 387.94 | 14.91 |
| Median | 387.94 | 14.91 |

Against the retained llama-server target of 794.07 prompt tok/s and 34.65
generation tok/s, this is 48.9% and 43.0% of target respectively. This is the
current trusted N1x baseline. The local artifacts are in
`.cache/bench/qwen36-35b-accepted-complete-20260729`.

This gap is not explained by an idle or thermally throttled GPU. During a
correct measured request, the N1x reported roughly 79-93% GPU utilization,
1.24-1.39 GHz SM clocks, 5-7 W GPU power, and 52-55 C temperature with no
thermal or power throttle reason.

Performance runs must clear inherited MLX graph-limit variables and must not
use `OLLAMA_DEBUG=2`; that setting enables trace-level tensor logging and can
distort endpoint timing. `tater62_run_bench.ps1` defaults to no debug logging
and accepts `-Debug 2` only for an explicitly diagnostic run.

### Rejected: 32-bit C++ scale staging

Nsight Compute on the source-provenance-checked N1x control identified four
adjacent FP8 scale loads and stores as a major QMM pathology:

- four `LDG.E.U8` instructions generated 32.5M of the reported 34.54M
  excessive global sectors;
- the corresponding `STS.U8` operations contributed to 7.80M excessive
  shared-memory wavefronts;
- the profiled `qmm_sm80_rhs_kernel<16,bf16,e2m1,e4m3,64x128x64>` took
  12.98 ms, reached 54.83% SM throughput, and spent 49.64% of scheduler
  cycles with no eligible warp;
- occupancy was limited to 16.48% achieved by 254 registers/thread and
  41.47 KiB dynamic shared memory.

An experiment reinterpreted the four adjacent scale bytes as one `uint32_t`
global load and one `uint32_t` shared store, with the scalar path retained for
quantizations with bias or other scale-group sizes. Focused numerical coverage
passed on RTX 5090 and N1x, but NVCC scalarized the expression back into the
same four byte loads and four byte stores. The candidate N1x profile remained
effectively identical:

| Metric | Accepted | C++ `uint32_t` candidate |
| --- | ---: | ---: |
| Kernel duration | 12.98 ms | 12.97 ms |
| Excessive global sectors | 34.54M | 34.54M |
| Excessive shared wavefronts | 7.80M | 7.80M |
| Scheduler no eligible warp | 49.64% | 49.60% |
| Achieved occupancy | 16.48% | 16.47% |

The standard RTX 5090 endpoint also showed no benefit and approximately a 3%
steady-state prompt regression. The N1x full-model candidate passed repeated
`blue-sky` and natural p2048 semantic checks, but its final endpoint benchmark
was blocked by two consecutive clean-process `cudaMallocAsync` load OOMs with
no stale Ollama process and only 259 MiB reported GPU residency. Do not infer
an endpoint gain from the focused timings.

Reject the C++ reinterpret-cast experiment. Any retry must use an existing
MLX/CuTe vector-copy primitive, `cp.async`, or narrowly scoped inline PTX that
provably emits one 32-bit global transaction and one 32-bit shared transaction.
Confirm the SASS and NCU sector counters before running full-model benchmarks.

Artifacts:

- `.cache/patches/mlx-qmm-vector-scale-staging.patch`
- `.cache/profiles/ncu/tater62-qwen-qmm-p2048-accepted-rhs`
- `.cache/profiles/ncu/tater62-qwen-qmm-p2048-vector-scale`
- `.cache/bench/qwen36-5090-vector-scale-20260729`

### N1x CMake install sharp edge

The Windows ARM64 MLX build can compile and link successfully but fail the
`MLX_VENDOR` install component while resolving `cudnn64_9.dll`. The generated
subbuild install rule currently searches cuDNN's `bin/13.4` directory but can
miss the actual `bin/13.4/arm64` runtime directory. Adding the runtime directory
to process `PATH` alone does not affect CMake's generated
`file(GET_RUNTIME_DEPENDENCIES)` `DIRECTORIES` list.

When native objects are already complete, do not rebuild or hand-copy the new
MLX DLLs. The safe recovery used here was:

1. `cmake --install <mlx-subbuild> --config Release --component MLX`
2. `cmake --install <outer-build> --config Release --component ollama-local`

This installs the freshly linked `mlx.dll` and `mlxc.dll` through CMake while
retaining the previously validated, unchanged vendor runtime payload. A future
build-system cleanup should make the cuDNN ARM64 bin directory explicit in
`MLX_RUNTIME_DIRS`; until then, do not treat the vendor-component failure as a
native compile failure.

### Rejected: 32-bit `cp.async` scale staging

A follow-up to the scalarized C++ `uint32_t` experiment issued one explicit
four-byte `cp.async` per output row for the four adjacent NVFP4 scale bytes.
The idea was to replace the four `LDG.E.U8` and four `STS.U8` instructions
identified by Nsight Compute without changing the QMM arithmetic.

The first profile was invalid because `qmm_sm80.cuh` is embedded in
`cuda_jit_sources.h` and compiled at runtime. MLX's PTX disk cache is keyed by
the stable module name under the MLX version directory, not by embedded source
content. The test therefore reused the accepted scalar cubin even after the
native library was rebuilt. A second issue made the first genuinely fresh
candidate return zeros: `MLX_CUDA_SM_80_ENABLED` is defined through
`steel/defines.cuh`, which this JIT header does not include, so the guarded
inline assembly compiled out while the scalar copy had already been replaced.

After changing the experimental guard to the JIT-compatible
`__CUDA_ARCH__ >= 800` check, a deterministic group-16/K64 NVFP4 test compared
the sorted GatherQMM output with an independent
dequantize-plus-gathered-matmul reference. The candidate was numerically exact
but decisively slower on the RTX 5090:

| Focused RTX 5090 primitive | Accepted scalar | `cp.async` candidate | Change |
| --- | ---: | ---: | ---: |
| Gate/up p2048 top-8 | 6.39 ms | 8.82 ms | 38% slower |
| Down p2048 top-8 | 1.17 ms | 5.42 ms | 364% slower |

Reject this experiment without an endpoint benchmark. Four-byte `cp.async`
operations are a poor fit here even though they reduce the apparent scalar
instruction count. The accepted source was restored on both RTX 5090 and N1x.
The strengthened numerical reference passes exactly on both; the restored N1x
focused timings are 15.48 ms gate/up and 8.98 ms down.

Local JIT-kernel experiments must now:

- rebuild the embedded source after a deployed tar mirror changes, because tar
  can preserve an mtime older than the generated header;
- set `MLX_PTX_CACHE_DIR` to a path derived from the installed `mlx` DLL/SO
  hash, so a changed embedded source cannot reuse an old module;
- pass the deterministic numerical reference before any timing or NCU run;
- confirm changed SASS/counters before attributing a profile to the candidate.

The guardrails are implemented in the 5090 and N1x focused-test/profile
helpers. The rejected source remains reproducible in:

- `.cache/patches/mlx-qmm-cp-async-scale-staging.patch`
- `.cache/patches/mlx-qmm-cp-async-jit-guard.patch`
- `.cache/profiles/ncu/tater62-qwen-qmm-p2048-cp-async-scale`

The retained NCU artifact above is explicitly the stale accepted scalar cubin,
not evidence about the real `cp.async` candidate.

### Rejected: sm121-only synchronous 32-bit scale staging

A synchronous inline-PTX variant emits one `ld.global.u32` and one
`st.shared.u32` for the four adjacent NVFP4 scale bytes. The vector path is
restricted to `__CUDA_ARCH__ >= 1210`; RTX 5090 (`sm120`) retains the exact
accepted scalar implementation.

The deterministic group-16/K64 sorted GatherQMM test remains bit exact on all
tested hosts. Nsight Compute on N1x confirms that the candidate reached the
runtime-JIT module:

| N1x QMM metric | Accepted scalar | sm121 scale word | Change |
| --- | ---: | ---: | ---: |
| Kernel duration | 12.98 ms | 11.74 ms | 9.6% faster |
| Excessive global sectors | 34.54M | 7.80M | 77% lower |
| Total sectors | 63.96M | 37.22M | 41.8% lower |
| Excessive shared wavefronts | 7.80M | 6.68M | 14.4% lower |
| Scheduler no eligible warp | 49.64% | 45.15% | 4.49 points lower |

After rebooting N1x and restoring a source-provenance-checked current control,
the first focused direct A/B appeared to contradict the earlier single-kernel
profile:

| N1x focused primitive | Accepted scalar | sm121 scale word | Change |
| --- | ---: | ---: | ---: |
| Gate/up p2048 top-8 | 5.878 ms | 17.850 ms | 203.7% slower |
| Down p2048 top-8 | 3.066 ms | 12.089 ms | 294.3% slower |

The candidate remained bit exact (`max_abs=0`) and the slow timing repeated on
an idle host. However, the closing accepted-control leg invalidated that A/B:
source, generated JIT embedding, and fresh PTX contained no scale-word code,
yet the restored scalar path measured 18.754/12.202 ms. Its full correctness-
gated p2048/g128 median was only 282.83 prompt and 11.71 generation tok/s.

The opening accepted control had measured 5.878/3.066 ms and, after passing
`TestBasic/blue-sky` plus a natural 1,963-token prompt, 788.99 prompt and 29.18
generation tok/s. The same accepted source therefore lost roughly 2.8x
endpoint throughput during the A/B sequence. This runtime-state drift is
larger than the candidate effect and makes the N1x comparison inconclusive.
Do not accept or reject the scale-word path from these endpoint runs. It remains
shelved because its DGX Spark endpoint was neutral and N1x needs a stable
reset/measurement protocol before another A/B.

The host was rebooted again and the accepted source was rebuilt from an empty
build and install tree. Fresh CUTLASS v4.4.2 required the preserved
`cutlass-msvc-arm64-compact-seq.patch`; previous incremental trees had already
carried that compatibility fix. The resulting `mlx.dll` hash was
`f620a61466e53f7c2610b9555e2f2e8437c5bae1368d64328c9709387e466a05`.
The deterministic QMM reference remained exact, but a warm focused repeat was
still slow at 22.128/11.448 ms. The full Qwen control passed blue-sky and
natural long-prompt correctness, then measured a p2048/g128 median of 294.97
prompt and 11.09 generation tok/s. The first load attempt hit a transient
`cudaMallocAsync` OOM with no resident candidate process; the one allowed clean
retry succeeded.

This clean result disproves the runtime-reset theory. Treat the isolated
5.878/3.066 ms and 788.99/29.18 opening control as non-reproducible and exclude
it from performance conclusions. The current clean result is 37.1% prompt and
32.0% generation of the fixed 794.07/34.65 llama-server target. Continue N1x
work from the reproducible slow QMM path.

The sm121-only candidate was then rebuilt from that clean control using an
audited one-header overlay. The numerical reference remained bit exact. Warm
focused timings were:

| N1x focused primitive | Accepted scalar | sm121 scale word | Change |
| --- | ---: | ---: | ---: |
| Gate/up p2048 top-8 | 20.830 ms | 18.035 ms | 13.4% faster |
| Down p2048 top-8 | 11.549 ms | 12.106 ms | 4.8% slower |

The full graph rejected the candidate before benchmarking. Its first model
load hit the same transient post-build OOM seen by the clean control. The one
allowed retry loaded, but the 11-token preload took 50.44 seconds and the
23-token blue-sky completion took 29.91 seconds, causing the integration test
to time out before generation visibly started. This is a real small-batch/full-
graph regression, not a correctness mismatch or runner crash.

Reject the synchronous scale-word path. It improves the isolated large-prompt
gate/up kernel but harms the shapes and scheduling needed by full inference.
The shared MLX working tree and N1x install were restored to the accepted scalar
path. The restored deterministic reference passed exactly, with a warm
20.830/11.549 ms focused result.

A narrower follow-up specialized only sorted-RHS NVFP4/group-16 QMM on sm121
when `M >= 1024` and `K == 2048`. Direct QMM, short prompts, down projections,
decode, and sm120 all retained the scalar path. A new independent
dequantize-plus-matmul reference at the exact `M=1024,K=2048` dispatch boundary
passed with `max_abs=0`, as did blue-sky and a natural 1,963-token prompt.
Focused gate/up improved from 20.830 ms to 17.025 ms on a warm run, but the
three-epoch p2048/g128 endpoint median was only 295.03 prompt and 12.76
generation tok/s. Prompt was effectively identical to the clean accepted
294.97 tok/s control. The apparent generation change is not attributable to
this prefill-only dispatch and is benchmark-state variance. Reject and restore
the accepted scalar source.

Artifacts:

- `.cache/patches/mlx-sm121-qmm-large-prefill-scale-word-experiment.patch`
- `.cache/bench/qwen36-35b-sm121-large-prefill-scale-word-retry-20260730`
- `.cache/bench/qwen36-35b-sm121-large-prefill-scale-word-3e-20260730`

Cross-host guards:

| Host | Result |
| --- | --- |
| RTX 5090 / sm120 | Scalar path retained; correctness passed; endpoint neutral within run variance |
| DGX Spark / sm121 | QMM reference exact; focused 5.65/3.32 ms; endpoint effectively neutral |
| N1x / sm121 | QMM reference exact; A/B invalidated by 2.8x accepted-control drift |

The final DGX Spark candidate steady median was 2,250.47 prompt and 89.61
generation tok/s. The retained accepted result was 2,295.73 and 89.63 tok/s,
so this candidate does not move the endpoint on GB10. Against the fixed
2,225.52/77.03 llama-server target, the candidate is 101.1% prompt and 116.3%
generation.

The earlier DGX result of 2,493.40/56.93 is invalid. It accidentally combined
the stale `current-best` Ollama artifact with a one-header overlay on a
divergent MLX source mirror. A second intermediate 2,250.47/72.15 result is
also invalid for candidate comparison because the accepted archive omitted
the validated adaptive sm121 graph policy that was still unstaged. The policy
has now been promoted to the shared MLX index in `device.cpp` and `device.h`.

DGX Spark source/install provenance:

- validated install:
  `/home/daniel/.codex/dist/ollama-upstream-um-current-mlx-cuda13-sm121`;
- `current-best` is historical/stale and must not be used for conclusions;
- deploy the complete accepted shared MLX archive before a remote source mirror
  has diverged; do not overlay one header across mismatched QMM source files;
- after replacing a complete mirror, clean the MLX sub-build once because tar
  mtimes may leave incompatible objects falsely current;
- subsequent one-header experiments use the incremental build with generated
  JIT-header invalidation and installed-library-hash PTX caches.
- stage candidate/control swaps with `stage_tater62_current_best.ps1
  -MlxQmmOverlayOnly`; regenerating the base `git archive` changes every source
  timestamp and turns a 9-target QMM rebuild into a 231-target full rebuild.

Artifacts:

- `.cache/patches/mlx-qmm-inline-ptx-scale-word-experiment.patch`
- `.cache/patches/mlx-qmm-inline-ptx-scale-word-sm121-dispatch.patch`
- `.cache/profiles/ncu/tater62-qwen-qmm-p2048-inline-ptx-scale-word`
- `.cache/bench/qwen36-5090-sm121-scale-dispatch-steady-20260729`
- tater50:
  `.cache/bench/qwen36-sm121-scale-word-adaptive-20260729`

### Rejected: sm121 QMM tile N 64

An isolated experiment reduced only the sm121 QMM CTA N tile from 128 to 64.
The module name included the tile width so the runtime JIT cache could not
reuse the 128-column kernel. sm120 and all other architectures retained the
accepted 128-column tile.

The deterministic DGX Spark NVFP4 reference remained exact, but the dominant
gate/up primitive regressed sharply:

| DGX Spark focused primitive | Tile N 128 | Tile N 64 | Change |
| --- | ---: | ---: | ---: |
| Gate/up p2048 top-8 | 5.65 ms | 8.08 ms | 43.0% slower |
| Down p2048 top-8 | 3.32 ms | 3.21 ms | 3.2% faster |

Reject without a full-model benchmark because gate/up dominates this model's
MoE QMM work. The 128-column tile was restored in the shared MLX source and
the DGX Spark install. Preserve the patch only to avoid repeating the test:

- `.cache/patches/mlx-sm121-qmm-tile-n64-isolated-experiment.patch`

### Rejected: sm121 QMM tile M 32

The N1x QMM profile reports two-block limits from both registers and shared
memory. Reducing the three-stage shared-memory pipeline alone therefore cannot
raise occupancy while the kernel still uses 254 registers/thread. An isolated
experiment instead capped only sm121's QMM CTA M tile at 32 to reduce
accumulator and live-fragment pressure; all other devices retained M 64.

The deterministic DGX Spark NVFP4 reference remained exact, but both focused
operations regressed:

| DGX Spark focused primitive | Tile M 64 | Tile M 32 | Change |
| --- | ---: | ---: | ---: |
| Gate/up p2048 top-8 | 5.66 ms | 5.88 ms | 3.8% slower |
| Down p2048 top-8 | 3.23 ms | 3.27 ms | 1.3% slower |

Reject without an endpoint benchmark. The additional CTAs and repeated setup
cost more than any benefit from the smaller tile. The 64-row tile was restored
in the shared source and DGX Spark install.

- `.cache/patches/mlx-sm121-qmm-tile-m32-isolated-experiment.patch`

### Rejected: sm121 adaptive small-batch graphs at 800/1024

The N1x focused Qwen primitive probe showed that CUDA graph/runtime overhead
is much larger than the individual math-kernel time:

- gathered NVFP4 QMV: about 90 us in NCU versus 0.6-0.9 ms at the synchronized
  operation boundary;
- gated-delta custom kernel: about 95 us in NCU versus 0.89 ms at the
  synchronized operation boundary;
- the 40-layer MoE chain improved from 14.37 ms at the default 20/25 limits to
  12.19 ms with 400/512 and 11.51 ms with 800/1024.

The microbenchmark suggested increasing only the adaptive sm121 small-batch
limits from 400/512 to 800/1024. The full DGX Spark Qwen row rejected it:

| Adaptive limit | Prompt tok/s | Generate tok/s |
| --- | ---: | ---: |
| Accepted 400/512 | 2,250.47 | 89.61 |
| Candidate 800/1024 | 2,274.39 | 79.20 |

Both semantic gates passed, but generation regressed 11.6%. The natural
long-prompt first-use check also fell from 717.21 to 136.84 prompt tok/s,
showing that graph composition/JIT behavior outside the isolated MoE chain is
material. Restore and retain 400/512. Do not promote graph limits based only
on a focused chain benchmark.

Artifacts:

- `.cache/patches/mlx-sm121-adaptive-graph-800-1024-experiment.patch`
- tater50:
  `.cache/bench/qwen36-sm121-adaptive-800-1024-20260729`
- N1x NCU:
  `.cache/profiles/ncu/qwen-qmv-baseline-sm121` and
  `.cache/profiles/ncu/qwen-gated-delta-baseline-sm121`

### Rejected: sleeping CUDA completion worker

MLX's CUDA completion worker declares `current_batch` inside its loop. After
the first completion signal, the wait predicate therefore remains true and
the worker hot-polls. A tater50 thread sample confirmed one idle MLX worker
thread remained runnable at about 99% of one CPU core.

Moving the cursor outside the loop stopped the spin, but condition-variable
wake latency became visible at every synchronized operation. The exact QMM
reference passed, while focused timings rose from roughly 5.7/3.2 ms to
6.4/3.6 ms. The correctness-gated Qwen p2048/g128 row produced:

| CUDA worker | Prompt tok/s | Generate tok/s |
| --- | ---: | ---: |
| Accepted hot-polling worker | 2,250.47 | 89.61 |
| Sleeping-worker candidate | 2,267.40 | 86.10 |

Prompt was within run variance and generation regressed about 3.9%. The
candidate runner also faulted while being forcibly unloaded after the
benchmark. Reject the simple cursor-lifetime fix and restore the accepted
worker. A future cleanup needs a low-latency wait strategy rather than a
straight condition-variable sleep.

Artifacts:

- `.cache/patches/mlx-cuda-worker-batch-cursor-experiment.patch`
- tater50:
  `.cache/bench/qwen36-worker-cursor-candidate-20260730`

### Accepted: adaptive expert-run QMM tiles

A natural Qwen3.6 35B-A3B p2048 router diagnostic exposed a defect hidden by
the original balanced primitive benchmark:

| Router property | Value |
| --- | ---: |
| Sorted rows | 15,600 |
| Active experts | 253 |
| Fixed 64-row tiles | 244 |
| Tiles spanning multiple experts | 164 |
| Expert runs executed by fixed tiles | 495 |
| Effective rows of QMM work | 31,648 |
| Work amplification | 2.029x |

The sorted-RHS kernel previously invoked its complete K mainloop once per
expert run within each fixed 64-row tile, masking stores outside the run but
still loading and multiplying all 64 rows. The original focused benchmark
assigned exactly 64 rows to every expert and could not reveal this.

The accepted fix keeps the existing grid and arithmetic, offsets A/C to each
contiguous expert run, and selects a 16-, 32-, or 64-row CuTe CTA tile for that
run. It introduces no activation requantization or approximation and leaves
direct QMM, unsorted gather, and small/decode dispatch unchanged. Independent
NVFP4 dequantize-plus-matmul references passed for K=64, M=1024/K=2048, and an
uneven sorted M=1024 distribution.

Matched focused results:

| Host / shape | Accepted fixed tile | Adaptive run tile | Change |
| --- | ---: | ---: | ---: |
| RTX 5090 balanced gate/up | 0.638 ms | 0.636 ms | 0.4% faster |
| RTX 5090 balanced down | 1.115 ms | 1.149 ms | 3.0% slower |
| RTX 5090 uneven gate/up | 1.148 ms | 0.966 ms | 15.9% faster |
| RTX 5090 uneven down | 1.177 ms | 1.132 ms | 3.8% faster |
| DGX Spark balanced gate/up | 5.730 ms | 5.698 ms | 0.6% faster |
| DGX Spark balanced down | 3.013 ms | 3.073 ms | 2.0% slower |
| DGX Spark uneven gate/up | 9.473 ms | 6.808 ms | 28.1% faster |
| DGX Spark uneven down | 3.123 ms | 3.078 ms | 1.4% faster |
| N1x balanced gate/up | 20.830 ms | 19.915 ms | 4.4% faster |
| N1x balanced down | 11.549 ms | 10.501 ms | 9.1% faster |

Every endpoint passed `TestBasic/blue-sky` and a natural long-prompt semantic
check before benchmarking:

| Host | Accepted p2048/g128 | Adaptive p2048/g128 | llama target | Target |
| --- | ---: | ---: | ---: | ---: |
| RTX 5090 | 7,075 / 70.97 | 8,129 / 89.46 | 5,830 / 219.7 | 139.4% / 40.7% |
| DGX Spark | 2,250 / 89.61 | 2,250 / 90.06 | 2,226 / 77.03 | 101.1% / 116.9% |
| N1x | 294.97 / 11.09 | 322.37 / 12.16 | 794.07 / 34.65 | 40.6% / 35.1% |

The RTX 5090 generation values varied with graph/runtime warmup and are not
attributable to this prefill kernel. The prompt improvement repeated at
14-17% across matched epoch positions after both PTX caches were populated.
DGX endpoint behavior is neutral despite its large isolated uneven-routing
gain. N1x improves but remains far below target and is the next profiling
priority.

Artifacts:

- `.cache/patches/mlx-qmm-rhs-adaptive-run-tiles-experiment.patch`
- `.cache/scripts/qwen_qmm_platform_test.go`
- `.cache/bench/qwen36-5090-adaptive-run-tiles-cached-20260730`
- `.cache/bench/qwen36-5090-adaptive-run-tiles-control-20260730`
- tater50:
  `.cache/bench/qwen36-adaptive-run-tiles-20260731`
- tater62:
  `.cache/bench/qwen36-35b-adaptive-run-tiles-20260731`

### Rejected: expert-aligned QMM grid

An exact alternative launched one CTA per expert/N tile, found each expert's
sorted row range with binary search, and processed its 64-row chunks. This
removed expert-boundary replay but made the binary search and per-expert block
scheduling visible on short-K and balanced shapes:

| RTX 5090 shape | Fixed tile | Expert grid | Change |
| --- | ---: | ---: | ---: |
| Balanced gate/up | 0.638 ms | 0.706 ms | 10.6% slower |
| Balanced down | 1.115 ms | 1.640 ms | 47.1% slower |
| Uneven gate/up | 1.148 ms | 0.957 ms | 16.7% faster |
| Uneven down | 1.177 ms | 1.363 ms | 15.8% slower |

Gate/up savings and down regression approximately cancel in Qwen's MoE block.
Reject in favor of adaptive run tiles.

- `.cache/patches/mlx-sm12-qmm-rhs-expert-aligned-experiment.patch`
- `.cache/patches/mlx-sm12-qmm-rhs-expert-aligned-fmt-fix.patch`

### Rejected: sm120 QMM tile M 128

An isolated sm120-only M128 sorted-RHS experiment retained M64 on sm121,
direct QMM, and small batches. The exact reference passed, but RTX 5090
gate/up regressed from 0.626 to 1.224 ms and down from 1.181 to 1.674 ms.
The larger accumulator footprint and lower scheduling flexibility outweigh
the reduced block count.

- `.cache/patches/mlx-sm120-qmm-rhs-tile-m128-experiment.patch`

### Rejected: long-K expert-aligned hybrid

The previously rejected expert-aligned grid improved fragmented gate/up but
regressed short-K down projection enough to cancel the gain. A narrower hybrid
therefore selected expert alignment only for sorted-RHS `M >= 1024, K >= 1024`
on compute capability 12.x, while the accepted adaptive row grid handled
short-K down projection.

All three independent NVFP4 references passed, as did RTX 5090 blue-sky and
natural long-prompt correctness. The focused fragmented gate/up shape improved
from about 0.966 ms to 0.939 ms, but the full p2048/g128 median fell from the
accepted 8,129/89.46 tok/s run to 5,702/40.62 tok/s. The expert CTAs serialize
work that MLX's graph otherwise overlaps; a small isolated gate/up win does not
survive endpoint scheduling. Reject and retain the adaptive row grid for both
projections.

- `.cache/patches/mlx-qmm-rhs-long-k-expert-hybrid-rejected.patch`
- `.cache/bench/qwen36-5090-hybrid-expert-gate-20260730`

### Rejected: sm121 two-stage, three-CTA QMM

The adaptive N1x fragmented gate/up profile remained limited to two resident
CTAs by both 224 registers/thread and 41.47 KiB shared memory. A CC 12.1
variant reduced the `cp.async` pipeline from three stages to two and used
`launch_bounds(..., 3)` to constrain register allocation. RTX 5090 retained
the accepted three-stage path.

Nsight Compute confirmed that the intended occupancy change was real:

| N1x fragmented gate/up metric | Adaptive p3 | Candidate p2/r3 |
| --- | ---: | ---: |
| Kernel duration | 32.12 ms | 28.81 ms |
| Registers/thread | 224 | 168 |
| Dynamic shared memory | 41.47 KiB | 27.65 KiB |
| Achieved occupancy | 16.54% | 24.64% |
| Scheduler no eligible | 49.80% | 40.92% |
| Executed instructions | 625.7M | 663.1M |

The unrestricted candidate made 11- and 23-token requests take 65 and 30
seconds because the shallow pipeline is a poor fit for small routed batches.
Restricting it to `M >= 1024` restored short-prompt correctness. All three
independent NVFP4 references, blue-sky, and the natural long prompt then
passed. The correctness-gated p2048/g128 median was 321.67/12.61 tok/s versus
the accepted adaptive 322.37/12.16 tok/s. Generation movement is not
attributable to this prefill-only path; prompt is neutral.

The 10% isolated QMM improvement is too small a fraction of the endpoint and
the two-stage pipeline executes 6% more instructions. Reject the added kernel
surface and retain the accepted adaptive three-stage mainloop.

- `.cache/patches/mlx-sm121-qmm-p2r3-low-resource-rejected.patch`
- `.cache/profiles/ncu/tater62-qwen-qmm-p2r3-fragmented-gate-sm121`
- tater62:
  `.cache/bench/qwen36-35b-p2r3-large-retry-20260730`

### CUTLASS grouped NVFP4 audit

CUDA 13 cuBLASLt exposes NVFP4 block-scale descriptors but no grouped GEMM
API; cuBLAS exposes grouped GEMM but not the required block-scale contract.
CUTLASS 4.4.2 includes an sm120 GeForce NVFP4 grouped example, but the stock
example produces FP4 output rather than MLX's BF16 output contract and its own
verification failed for both 128-cubed and 64x1024x2048 test shapes. Its
unverified timing was physically implausible and must not be used as a
performance ceiling. A native FP4 path would also require activation
requantization and change arithmetic, so the exact adaptive scheduler was the
appropriate next step.

Nsight Compute counters remain unavailable on the RTX 5090 due to
`ERR_NVGPUCTRPERM`; do not infer counters from that failed capture. N1x has
working Nsight Compute CLI support and should be used for the next QMM pass.

### Rejected: CUTLASS grouped native NVFP4

The CUTLASS 4.4.2 `79d` grouped NVFP4 example can be adapted to BF16 output
with the standard collective epilogue. The adapted standalone kernel passed
host references for fixed and uneven eight-group shapes. Its ping-pong
schedule was promising in isolation on the RTX 5090:

| Qwen-like shape | CUTLASS ping-pong | Accepted MLX QMM |
| --- | ---: | ---: |
| Balanced gate/up, M64 N1024 K2048 x256 | 0.229 ms | 0.636 ms |
| Uneven gate/up | 0.244 ms | 0.966 ms |
| Balanced down, M64 N2048 K512 x256 | 0.295 ms | 1.149 ms |
| Uneven down | 0.171 ms | 1.132 ms |

The MLX integration quantized BF16 activations to native NVFP4, packed each
sorted expert run into a 128-row scale tile, and launched the grouped
ping-pong kernel. It preserved the accepted W4A16 path as a fallback. Uneven
long-K routing with two zero-row experts matched independently partitioned
dense QQMM calls bit-for-bit. The grouped and dense native QQMM paths also
matched each other bit-for-bit on M1024/N128/K2048.

Native QQMM is not numerically identical to the existing W4A16 GatherQMM:
both grouped and dense QQMM had max/mean absolute error 0.0686/0.0247 versus
dequantize-plus-BF16-matmul on the deterministic large-K reference. This is
the established A-side FP4 quantization contract, not a grouped-layout defect.

The integrated short-K down projection was slower because activation
quantization and scale repacking dominated, so the final candidate used native
grouped QQMM only for K>=1024. It passed `TestBasic/blue-sky`, produced a
coherent natural long-prompt answer, and passed all primitive routing checks.

The first endpoint comparison was invalid because the worktree still contained
the previously rejected `fp_qmv_async_k2048` experiment. It was archived and
removed before rebuilding both sides from the exact staged index. Cold and
single-epoch results were also discarded because both paths accelerated across
successive epochs as MLX's graph/runtime state settled.

The final matched six-epoch warm comparison was:

| RTX 5090 p2048/g128 | Prompt | Generate |
| --- | ---: | ---: |
| Accepted adaptive W4A16 median | 7,164.76 tok/s | 105.41 tok/s |
| Grouped native NVFP4, long-K median | 6,391.02 tok/s | 100.42 tok/s |
| llama-server target | 5,830 tok/s | 219.7 tok/s |

The candidate was 10.8% slower on prompt. Excluding the first epoch from each
run produced the same conclusion: 7,344.46 tok/s for the control versus
6,500.40 tok/s for the candidate, an 11.5% regression. One candidate
generation epoch stalled at 29.53 tok/s; generation is not attributable to
this prefill-only K>=1024 dispatch and is included only for completeness.

The isolated GEMM speed does not survive MLX graph execution. Activation
quantization, scale repacking, temporary metadata, and capture boundaries
more than erase the tensor-core gain. Reject until those operations can be
amortized or fused without changing the model/runtime architecture.

- `.cache/patches/mlx-cutlass-grouped-nvfp4-rejected/`
- `.cache/cutlass-grouped-nvfp4/`
- `.cache/bench/qwen36-5090-grouped-nvfp4-clean-ab-20260731`

### Rejected: BF16 GEMV two-row activation reuse

The clean Qwen3.6 35B-A3B decode trace exposed 130.36 ms of unquantized BF16
`gemv_single` work. The dominant `grid.x=4` shape was the pair of 32-output,
K=2048 B/A projections in each recurrent layer: 7,779 launches at 11.45 us
average. The existing kernel assigns one output row to each warp, so every row
loads the same activation vector.

A Blackwell-only experiment kept the existing eight-row block but used four
warps with two rows per warp. Each row retained the same FP32 accumulation
order, while each warp loaded the activation vector once for both rows.
Deterministic BF16 outputs were bit-exact at the three production shapes:
N=8, N=32, and N=256 with K=2048.

The model-level gate rejected the candidate. `TestBasic/blue-sky` and natural
p2048 correctness passed, but the p2048/g128 generation epochs were 104.84,
80.18, and 87.12 tok/s, for an 87.12 median. The exact retained same-binary
control median is 105.41 tok/s, so the candidate regressed generation 17.4%.
The reduced activation loads do not compensate for extra per-warp state and
the resulting loss of concurrent CUDA-graph throughput.

- `.cache/patches/mlx-bf16-gemv-row-reuse-rejected.patch`
- `.cache/bench/gemv-row-reuse-5090-20260731`
- `.cache/bench/qwen36-5090-gemv-row-reuse-20260731`

### Rejected: projected B/A consumer fusion on RTX 5090

The earlier projected B/A gated-delta experiment was replayed against the
clean accepted 5090 baseline because the current decode trace attributed
130.36 ms to BF16 GEMV, including 7,779 `N=32,K=2048` B/A launches. The
decode-only custom consumer computes both projections while producing the
gated-delta inputs, so it removes both generic GEMVs without packing weights.

Focused CUDA fusion coverage passed, as did `TestBasic/blue-sky` and the
natural p2048 semantic check. The three-epoch p2048/g128 generation median was
106.53 tok/s. A matched same-binary disabled control measured 101.85 tok/s,
but the retained clean control is 105.41 tok/s. The candidate is therefore
only about 1% above the best valid control, while the same design previously
regressed DGX Spark from 70.14 to 69.77 tok/s.

Reject the cross-platform model-path complexity. Eliminating the visible B/A
GEMV time does not materially shorten the concurrent graph.

- `.cache/patches/rejected-qwen36-projected-ba-20260728.patch`
- `.cache/bench/qwen36-5090-projected-ba-20260731`

### Rejected: M=1-native W4A16 tensor-core QMV

A standalone CuTe prototype used
`SM80_16x8x16_F32BF16BF16F32_TN` to compute eight output rows per warp.
Unlike the rejected native NVFP4 path, it retained BF16 activations and exact
E2M1 weights. Each K16 group was accumulated in FP32 before applying its
original E4M3 group scale, so there was no activation requantization.

The first one-warp-per-eight-rows schedule created a 128-MMA dependency chain
and took about 88 us at every production shape. A corrected split-K8 design
restored the scalar path's warp-level parallelism and reduced the chain to 16
MMAs per warp. Its BF16 outputs were bit-exact to the varied-data scalar
reference across all rows.

| K=2048 output rows | Scalar W4A16 | Split-K8 MMA | Result |
| ---: | ---: | ---: | --- |
| 512 | 12.55 us | 14.72 us | 14.7% slower |
| 4,096 | 17.24 us | 18.39 us | 6.6% slower |
| 8,192 | 26.96 us | 29.57 us | 9.7% slower |

The K16 scale boundary forces a separate MMA result and scale accumulation for
every group; E2M1 conversion remains on the critical path. Tensor cores do not
offset that overhead. Reject before MLX integration.

- `.cache/scripts/w4a16_mma_qmv_probe.cu`

The same probe also tested Blackwell bulk asynchronous copies while retaining
the scalar row-per-warp arithmetic. A whole eight-row packed-weight/scale slab
was loaded with two CuTe `SM90_BULK_COPY_G2S` transactions. It was bit-exact,
but only improved N=512 by 2-7% across repeats and was neutral at N=4,096 and
N=8,192. A two-stage K1024 pipeline overlapped the second transfer with first
stage computation, but the required row-wise transactions and barriers
regressed N=4,096/8,192 by 8-15%. Neither path has enough isolated signal to
justify MLX integration.

### Rejected: wider MXFP8 output-head conversion

The accepted Qwen3.6 decode profile attributed 80.98 ms across 129 launches to
the `248,320 x 2,048` MXFP8 output projection. A standalone probe reproduced
MLX's BF16-activation, E4M3-weight, UE8M0-scale arithmetic and compared the
current four-element CUTLASS converter against:

- an eight-element converter with the current K tile;
- an eight-wide per-thread K tile;
- four, eight, and sixteen output-row block geometries.

All variants except the wider K tile were BF16 bit-exact. The wider K tile
changed 20 of 248,320 BF16 results with maximum absolute error 0.25. At the
production shape, the retained kernel took 312.05 us and every candidate was
within 0.2%:

| RTX 5090 `N=248,320 K=2,048` | Time | Speedup |
| --- | ---: | ---: |
| Retained converter4 / rows8 | 312.05 us | 1.000x |
| Converter8 / rows8 | 311.60 us | 1.001x |
| Converter8 / rows4 | 311.55 us | 1.002x |
| Converter8 / rows16 | 311.80 us | 1.001x |
| K tile 8 / converter8 | 311.60 us | 1.001x |

The retained kernel streams roughly 524 MB of weights and scales per launch,
or about 1.68 TB/s in the standalone run. It is already bandwidth-saturated
on the RTX 5090; the roughly 628 us average in the full Nsight trace is
profiler/graph context, not evidence of an isolated conversion bottleneck.
Reject without MLX integration.

- `.cache/scripts/mxfp8_qmv_probe.cu`

### Rejected: sorted local heads in the normalized 256-expert router

The accepted decode profile attributed 33.80 ms across 5,160 launches to
Qwen3.6's CUDA-only normalized top-8 router. The retained kernel scans each
lane's eight local logits for every selected rank. A candidate stably sorted
the eight local values once, then merged the 32 lane-local heads through the
same warp reductions.

The standalone probe matched every BF16 weight and index for rows 1, 3, and 8,
including tied logits and injected NaNs. Normal finite inputs improved by
1.18x at the decode shape and by up to 1.33x at small batches. The focused
MLX CUDA reference test and Qwen gated-delta fusion test passed, as did
`TestBasic/blue-sky` and the natural long-prompt semantic check.

The p2048/g128 endpoint gate rejected the candidate:

| RTX 5090 Qwen3.6 35B-A3B NVFP4 | Prompt | Generate |
| --- | ---: | ---: |
| Accepted retained median | 7,164.76 tok/s | 105.41 tok/s |
| Sorted router median | 6,373.05 tok/s | 93.46 tok/s |

Generation regressed 11.3%. The extra live values and insertion-sort state
reduce graph concurrency enough to erase the isolated instruction savings.
Do not retry local sorting without evidence from a different low-register
selection architecture.

- `.cache/scripts/router256_probe.cu`
- `.cache/bench/qwen36-5090-router-local-sort-20260731`

### Rejected: CUTLASS block-scaled FP4 GEMV/native-MMA follow-up

CUTLASS 4.4.2 includes a dedicated `91_fp4_gemv` example with signed E4M3
scale factors and a 16-element scale vector, matching MLX NVFP4's stored
weight-scale contract. Source inspection closed two possible
misinterpretations:

- `GemvBlockScaled` is not a native block-scaled MMA kernel. It converts both
  FP4 operands to packed FP16 and performs explicit `fma.rn.f16x2`.
- SM120 `mma.sync...kind::mxf4nvf4` requires E2M1 inputs on both sides. It
  cannot consume MLX's BF16 activation directly.

Using the native instruction therefore still requires activation
requantization. That is the same arithmetic and scheduling tradeoff already
tested by the rejected native SM120 NVFP4 direct-QMV experiment: it was fast
in isolation, changed W4A16 arithmetic, contended with QMM tensor-core work,
and regressed the Qwen endpoint. The CUTLASS example does not expose a new
quality-neutral W4A16 path. Do not retry it as a wholesale QMV replacement.

### Rejected: 1,200-operation CUDA graphs on RTX 5090 Qwen

The accepted Qwen decode trace contained 286 `cudaGraphExecUpdate` calls over
128 generated tokens, or 2.23 updates/token, and approximately 963 CUDA
kernels/token. A 1,200-operation graph cap was the only untested point between
the accepted 800-operation policy and the known 1,600-operation giant-graph
cliff.

The environment-only experiment used the exact accepted binary with
`MLX_MAX_OPS_PER_BUFFER=1200` and `MLX_MAX_MB_PER_BUFFER=12000`.
`TestBasic/blue-sky` and the natural long-prompt semantic check passed. The
three measured p2048/g128 epochs were:

| RTX 5090 Qwen3.6 35B-A3B NVFP4 | Prompt | Generate |
| --- | ---: | ---: |
| 1,200-operation graph median | 7,168.95 tok/s | 78.51 tok/s |
| Accepted 800-operation median | 7,164.76 tok/s | 105.41 tok/s |

Prompt is unchanged, but generation regressed 25.5%. Larger graphs reduce
update frequency while also removing useful concurrent scheduling
boundaries. Retain 800/8000 and do not continue graph-cap sweeps.

- `.cache/scripts/run_qwen36_graph1200_current.sh`
- `.cache/bench/qwen36-5090-graph-1200-current-20260731`

### Rejected: top-level CUDA graph parameter updates

The accepted Nsight trace reported 4.23 ms per `cudaGraphExecUpdate`, but that
API timing was collected under heavy graph tracing. A standalone CUDA 13
probe compared full executable updates against the supported parameter-only
alternative: retain the original graph, update every top-level child graph
with `cudaGraphExecChildGraphNodeSetParams`, and update every direct kernel
with `cudaGraphExecKernelNodeSetParams`.

The probe changed output pointers and scalar arguments on every iteration,
launched the parameter-updated executable, and verified the exact expected
result. Representative RTX 5090 medians were:

| Graph shape | Full update | Parameter updates | Result |
| --- | ---: | ---: | ---: |
| 16 child graphs x50 + 100 direct kernels | 231.61 us | 259.44 us | 0.89x |
| 30 child graphs x32 | 276.45 us | 267.59 us | 1.03x |
| 30 child graphs x32 + 100 direct kernels | 283.24 us | 314.14 us | 0.90x |
| 4 child graphs x200 + 100 direct kernels | 247.38 us | 264.71 us | 0.93x |

Outside Nsight, whole-graph update is already sub-0.3 ms for a roughly
1,000-kernel graph. Per-node API overhead erases the topology-validation
savings as soon as direct outer kernels are present. Do not complicate MLX's
graph cache with retained source graphs or manual node mapping based on the
profiled 4.23 ms value.

- `.cache/scripts/cuda_graph_param_update_probe.cu`

### N1x post-profiler reboot control

After rebooting the N1x to clear any profiler-retained CUDA state, the exact
installed candidate passed `TestBasic/blue-sky` and the natural long-prompt
semantic check. Three p2048/g128 epochs measured:

| Epoch | Prompt tok/s | Generate tok/s |
| ---: | ---: | ---: |
| 1 | 275.10 | 14.14 |
| 2 | 337.13 | 13.01 |
| 3 | 309.51 | 12.46 |
| Median | 309.51 | 13.01 |

The benchmark and correctness artifacts were fully written before SSH reset
during teardown. This rules out profiler-retained state as the explanation
for the large N1x gap, but do not replace the provenance-checked
387.94/14.91 headline row until the remote DLL hash is compared with the
shared source deployment. The remote SSH service returned repeated permission
failures immediately afterward, so further remote work was paused per the N1x
retry policy.

- tater62:
  `.cache/bench/qwen36-35b-post-reboot-control-20260730`

### Rejected: disabling CUDA graphs on RTX 5090 Qwen

The N1x MoE rows improve when CUDA graphs are disabled, so the same
environment-only control was tested on the RTX 5090 without changing source
or binaries. `TestBasic/blue-sky` and the natural long-prompt semantic check
passed. The p2048/g128 medians were:

| RTX 5090 Qwen3.6 35B-A3B NVFP4 | Prompt tok/s | Generate tok/s |
| --- | ---: | ---: |
| Accepted CUDA graphs | 7,164.76 | 105.41 |
| CUDA graphs disabled | 11,257.18 | 57.32 |

Disabling graphs improves prefill but reduces generation by 45.6%. CUDA graph
capture is essential for decode on this host; the N1x gap is not explained by
a generally harmful graph implementation. Do not add a global or
compute-capability-12.0 no-graph policy.

- `.cache/scripts/run_qwen36_5090_no_cuda_graphs.sh`
- `.cache/bench/qwen36-5090-no-cuda-graphs-20260731`

### Rejected: shared-expert gate/up packing on RTX 5090

The Qwen shared expert can concatenate compatible quantized gate/up
projections at load and use the accepted packed decode SwiGLU kernel. This is
a keeper on DGX Spark, but the RTX 5090 exclusion had not been accompanied by
a durable A/B record. Restoring the staged fused predicate passed the focused
fused-vs-separate CUDA reference, `TestBasic/blue-sky`, and the natural
long-prompt semantic check.

| RTX 5090 Qwen3.6 35B-A3B NVFP4 | Prompt tok/s | Generate tok/s |
| --- | ---: | ---: |
| Separate shared gate/up, accepted | 7,164.76 | 105.41 |
| Packed shared gate/up | 8,068.64 | 93.51 |

Packing raises prompt throughput but reduces generation 11.3%. The separate
projections can overlap in the CUDA graph, while the packed projection
serializes their work. Keep the device policy: fused on DGX Spark, separate on
RTX 5090 and N1x. Do not remove the RTX 5090 guard based only on launch counts.

- `.cache/scripts/test_qwen_gateup_5090.sh`
- `.cache/scripts/run_qwen36_5090_shared_gateup.sh`
- `.cache/bench/qwen36-5090-shared-gateup-20260731`

### Decode pipeline timing: graph construction is material on RTX 5090

Temporary opt-in timing around the Go decode pipeline separated graph
construction from evaluation without changing the model graph. After the
first cold interval, representative 64-dispatch averages were:

| Pipeline section | Average |
| --- | ---: |
| Model forward graph construction | 5.0-6.2 ms |
| Unembed + sample | 0.01-0.11 ms |
| Sweep | 0.15-0.28 ms |
| AsyncEval call | 4.0-8.9 ms |
| Token read | approximately 0 ms |

The accepted endpoint is about 9.49 ms/token (`105.41 tok/s`), so Go/CGO
graph construction is a first-order part of decode latency rather than
negligible bookkeeping. The sampler does not copy full logits to the host:
argmax remains on the GPU and the pipelined decoder dispatches the next
forward pass before reading the previous token.

This evidence does not justify compiling the whole model or layer. Compile
only small, inherently serial operation chains, and require an endpoint A/B
because changing graph boundaries can reduce CUDA graph overlap.

- `.cache/scripts/run_qwen36_5090_pipeline_timing.sh`
- `.cache/diagnostics/qwen36-5090-pipeline-timing-20260731`

### Rejected: compiled routed-expert decode chain

A surgical `mlx.Compile` experiment wrapped only the packed routed expert
decode chain:
`GatherQMM(gate_up) -> SwiGLU -> GatherQMM(down) -> reshape`.
All weights, scales, expert indices, and cached LHS indices were explicit
compile inputs; shared-expert work, prefill, and the rest of the layer stayed
outside the closure.

A production-shaped focused test matched the eager graph, then
`TestBasic/blue-sky` and the natural long-prompt semantic check both passed.
The p2048/g128 endpoint gate strongly rejected the candidate:

| RTX 5090 Qwen3.6 35B-A3B NVFP4 | Prompt tok/s | Generate tok/s |
| --- | ---: | ---: |
| Accepted retained | 7,164.76 | 105.41 |
| Compiled routed experts | 4,967.29 | 61.31 |

Generation retained only 58.2% of accepted throughput. Pipeline timing still
showed roughly 5-6 ms of forward graph construction and worse evaluation
latency, so the closure did not remove the dominant host work and damaged
the favorable CUDA graph scheduling boundary. Do not retry this closure
shape or broaden it into a whole-layer/model compile.

- `.cache/diagnostics/qwen36-5090-pipeline-timing-20260731`

### Rejected: fused Qwen width-4 causal convolution and state advance

The graph-level Qwen decode trace showed 3,900 convolution evaluations and
3,930 concatenations over 130 model evaluations. This is the exact 30-layer
linear-attention path: each decode step concatenates a three-row recurrent
state with one input row, runs a width-4 grouped convolution over 8,192
channels, and slices the next three-row state.

A CUDA custom kernel fused the four-tap BF16-input/FP32-accumulated depthwise
convolution with the state shift. It was restricted to the exact production
decode shape and retained the generic path for prefill, masks, snapshots,
other shapes, and Metal. The focused primitive test exactly matched MLX's
grouped `Conv1d` result and cache state. `TestBasic/blue-sky` and the natural
long-prompt semantic check also passed.

The p2048/g128 endpoint gate rejected the candidate:

| RTX 5090 Qwen3.6 35B-A3B NVFP4 | Prompt | Generate |
| --- | ---: | ---: |
| Accepted retained median | 7,164.76 tok/s | 105.41 tok/s |
| Fused causal-conv median | 6,547.50 tok/s | 99.66 tok/s |

Prompt regressed 8.6% and generation regressed 5.5%. A graph-level trace
confirmed the structural change but explained why it was too small:
convolution evaluations fell from 3,900 to 30 and concatenations from 3,930
to 60, yet the executable graph only lost 3,870 nodes total, exactly 30
nodes/token. The generic `concat + cuDNN child graph` effectively cost two
outer graph nodes per layer, while the custom operation cost one. Graph
update and launch overhead rose enough to erase that one-node saving.

Do not retry this as a standalone custom operation. It becomes interesting
only if a broader linear-attention fusion can remove several adjacent graph
nodes and intermediate arrays per layer.

- `.cache/patches/qwen-causal-conv-rejected-20260731.patch`
- `.cache/bench/qwen36-5090-causal-conv-fusion-20260731`
- `.cache/profiles/qwen36-35b-a3b-nvfp4-causal-conv-graphlevel-p16g128-20260731`

### Rejected: convolution folded into Qwen gated-delta input transform

A broader follow-up folded the width-4 depthwise convolution and state advance
into the already-retained Qwen gated-delta input-transform custom operation.
Unlike the standalone convolution experiment, this introduced no replacement
graph node: it replaced `concat + cuDNN convolution + input transform` with
one existing input-transform node.

The production-shape primitive test exactly matched the retained convolution,
all five transformed inputs, and the shifted cache state. Endpoint
`TestBasic/blue-sky` and the natural long-prompt semantic check passed.

The p2048/g128 endpoint gate strongly rejected the candidate:

| RTX 5090 Qwen3.6 35B-A3B NVFP4 | Prompt | Generate |
| --- | ---: | ---: |
| Accepted retained median | 7,164.76 tok/s | 105.41 tok/s |
| Conv/input-transform fusion median | 6,621.31 tok/s | 59.35 tok/s |

Prompt regressed 7.6% and generation regressed 43.7%. A graph-level trace
showed that the intended topology improvement did occur: total graph kernel
nodes fell by 7,740 over 129 evaluations, exactly 60 nodes/token, and graph
update/launch CPU timings improved. The loss is therefore in the heavier
six-output GPU operation. Folding the convolution's weight/state traffic and
state-copy output into the transform kernel destroys its favorable execution
profile even though it reduces graph topology.

Do not extend this fusion family. The next decode work should target the
measured direct QMV, gathered QMV, and BF16 GEMV kernels rather than adding
more linear-attention fusion.

- `.cache/patches/qwen-conv-input-fusion-rejected-20260731.patch`
- `.cache/bench/qwen36-5090-conv-input-fusion-20260731`
- `.cache/profiles/qwen36-35b-a3b-nvfp4-conv-input-fusion-graphlevel-p16g128-20260731`

### Rejected: pre-scaled BF16 W4A16 tensor-core QMV

The exact tensor-core probe above was extended with the dequantization
semantics used by MLX's established matrix QMM path: multiply each E2M1
weight by its E4M3 group scale, round that dequantized weight to BF16, and
then accumulate BF16 activation x BF16 weight with
`SM80_16x8x16_F32BF16BF16F32_TN`. A split-K8 schedule kept each warp's
dependency chain to 16 MMAs.

The candidate matched the exact scalar QMV output in all 512 and 4,096 rows.
At 8,192 rows, one BF16 output differed by 0.001953125; it also differed from
the scalar BF16-dequant reference by the same amount. The arithmetic change
is negligible, but the isolated timing had no useful gain:

| RTX 5090, K=2,048 | Scalar W4A16 | Pre-scaled split-K8 MMA | Result |
| ---: | ---: | ---: | --- |
| N=512 | 12.55 us | 14.60 us | 14.0% slower |
| N=4,096 | 17.15 us | 17.14 us | neutral |
| N=8,192 | 27.22 us | 28.26 us | 3.7% slower |

Pre-applying the scale removes the per-group accumulator scaling, but E2M1
decode, BF16 conversion, and the M=1 tensor-core utilization remain dominant.
Reject without MLX integration.

- `.cache/scripts/w4a16_mma_qmv_probe.cu`

### Rejected: row-oriented native SM120 MXFP8 output-head QMV

The 248,320 x 2,048 MXFP8 vocabulary projection is isolated from the MoE
tensor-core work and looked like a better native-MMA candidate than the direct
NVFP4 projections. A custom SM120 probe oriented
`SM120_16x8x32_TN_VS<E4M3,E4M3,F32,UE8M0,32>` so each MMA computed 16
vocabulary rows for one activation column. This wastes seven result columns
instead of presenting cuBLASLt with a conventional one-row GEMM. Activations
were quantized once to group-32 MXFP8 using the same UE8M0 scale contract as
MLX QQMM.

The initial one-warp-per-16-rows schedule was latency-bound by 64 dependent
MMAs. Splitting K across eight warps reduced that chain to eight MMAs and
combined 16 FP32 row partials in shared memory. The split path had no spills,
but still lost at both the focused and production shapes:

| RTX 5090, K=2,048 | Scalar MXFP8 QMV | Quantize + split-K8 MMA | Result |
| ---: | ---: | ---: | --- |
| N=8,192 | 8.68 us | 27.89 us | 3.21x slower |
| N=248,320 | 311.70 us | 409.28 us | 31.3% slower |

At full vocabulary width the scalar kernel already moves the approximately
524 MB of weights and scales at about 1.68 TB/s. Native MMA cannot reduce that
traffic, and activation quantization plus partial reduction outweigh its
arithmetic savings. The MMA result's normalized error versus its quantized
activation reference was about 0.17%; activation quantization itself added
about 2.7% normalized error versus the BF16 contract.

Reject without MLX integration. The output-head gap is representation
bandwidth, not missing FP8 MMA coverage: this nominal NVFP4 model uses an
MXFP8 vocabulary head, while the llama Q4 target uses a 4-bit representation.

- `.cache/scripts/sm120_mxfp8_qmv_probe.cu`

### Rejected: compiled Qwen sigmoid-multiply gates

Qwen3.6 full attention and the shared expert both evaluate
`value * sigmoid(gate)`. A shapeless MLX compile wrapper fused each pair into
one pointwise kernel. A focused CUDA test matched the eager graph exactly,
`TestBasic/blue-sky` passed, and a natural long-prompt response remained
coherent.

The same installed binary exposed a temporary disable switch so control and
candidate could run in alternating order. A final three-epoch p2048/g128
comparison rejected the fusion:

| RTX 5090 Qwen3.6 35B-A3B NVFP4 | Prompt | Generate |
| --- | ---: | ---: |
| Eager control average | 7,341.07 tok/s | 93.47 tok/s |
| Compiled gate average | 6,279.06 tok/s | 92.34 tok/s |

The compiled gate regressed prompt throughput 14.5% and did not improve
generation. This is another case where fewer visible pointwise nodes do not
imply a faster end-to-end graph. The production helper, test, and temporary
environment switch were removed.

- `.cache/bench/qwen36-5090-sigmoid-mul-ab-20260731`
- `.cache/bench/qwen36-5090-sigmoid-mul-ab3-20260731`

### Accepted: cache the gated-delta CUDA device policy

Qwen3.6 calls the retained gated-delta CUDA kernel once per linear-attention
layer and generated token. The hot path queried `mlx.DeviceInfo()` on every
call solely to choose vector or scalar loads. A Go CPU profile attributed
0.42 seconds over 512 dispatches to `CUDADeviceName`, including repeated CGO
and device-info work.

The accepted change computes that immutable policy once with
`sync.OnceValue`. It preserves vector loads on RTX 5090 and DGX Spark while
retaining the scalar-load workaround for `JMJWOA-Generic-GPU`. Focused
gated-delta and Qwen fusion tests, `TestBasic/blue-sky`, and the natural
long-prompt semantic check passed on all three hosts.

| Host | Before prompt / generate | After prompt / generate | Result |
| --- | ---: | ---: | --- |
| RTX 5090 | 7,164.76 / 105.41 | 8,412.12 / 157.32 | 144.3% / 71.6% of target |
| DGX Spark | 2,250.00 / 90.10 | 2,471.72 / 90.33 | 111.1% / 117.3% of target |
| N1x | 387.94 / 14.91 | 329.77 / 11.59 | Correct but noisy regression; retain the prior headline |

The RTX 5090 after result is the median of three p2048/g128 epochs:
`5509.21 / 156.23`, `8412.12 / 157.32`, and `9636.96 / 162.52`.
DGX Spark epochs were `2264.20 / 90.59`, `2471.72 / 90.33`, and
`2518.07 / 89.78`. N1x first hit a transient load-time OOM; a clean retry
passed all gates and produced `331.46 / 13.48`, `329.77 / 10.97`, and
`309.85 / 11.59`. The N1x data is too variable to replace the existing
headline but confirms no correctness regression.

- `.cache/bench/qwen36-5090-cached-gated-delta-policy-20260731`
- tater50: `.cache/bench/qwen36-cached-gated-delta-policy-tater50-20260731`
- tater62: `qwen36-35b-cached-gated-delta-policy-retry-20260731`

### Rejected: cache the complete gated-delta kernel configuration

A follow-up cached the full immutable `mlx_fast_cuda_kernel_config` by
production shape and dtype, following existing adapter cache patterns. The
focused test invoked the fast path twice and matched the fallback both times.
The p2048/g128 median was `8360.86 / 157.69`, effectively neutral versus
`8412.12 / 157.32` for the simpler device-policy cache.

Reject the map/mutex complexity. The useful overhead was the repeated device
query, not reconstructing the small Go configuration slice.

- `.cache/bench/qwen36-5090-cached-gated-delta-config-20260731`

### Improved Qwen decode profile after device-policy caching

A request-scoped Go CPU profile of the accepted cached-policy binary removed
`CUDADeviceName` from the hot path. Go-side model construction fell to about
0.85 ms per model evaluation; `New` and other Go allocation bookkeeping were
negligible. The remaining request CPU samples were almost entirely CGO and
MLX evaluation.

A fresh p16/g128 Nsight Systems trace is heavily distorted by CUDA graph node
tracing (`42.80 tok/s` versus `157.32 tok/s` unprofiled), so its absolute API
times are not endpoint timings. Its GPU kernel mix is still useful:

| Kernel family | GPU time | Share |
| --- | ---: | ---: |
| Direct FP QMV, including output head | 202.4 ms | 29.0% |
| Gathered expert FP QMV | 149.0 ms | 21.3% |
| BF16 GEMV | 87.6 ms | 12.5% |
| Gated-delta step | 51.1 ms | 7.3% |
| Normalized MoE router | 33.9 ms | 4.8% |

The gathered work splits evenly between the `N=1024,K=2048` packed gate/up
projection (`77.82 ms`) and `N=2048,K=512` down projection (`71.17 ms`).
The large MXFP8 vocabulary projection accounts for `50.63 ms` of direct QMV
and is already known to be representation-bandwidth limited.

- `.cache/profiles/qwen36-35b-a3b-nvfp4-cached-policy-cpu-20260731`
- `.cache/profiles/qwen36-35b-a3b-nvfp4-cached-policy-p16g128-20260731`

### Rejected: gathered-QMV warp-count follow-ups

The retained gathered FP QMV computes two output rows per warp with four
warps per eight-row CTA. Earlier one-row testing changed both routed expert
projections and damaged full-graph overlap. Three follow-ups isolated the two
production shapes and tested the opposite four-row direction. Every candidate
passed `TestBasic/blue-sky` and the natural long-prompt semantic check before
the endpoint gate.

| RTX 5090 p2048/g128 | Prompt median | Generate median | Result |
| --- | ---: | ---: | --- |
| Accepted two rows/warp | 8,412.12 | 157.32 | Retain |
| One row/warp, gate/up only | 9,162.28 | 133.14 | Decode -15.4% |
| One row/warp, down only | 9,168.53 | 152.24 | No gain; lower median |
| Four rows/warp, both | 8,071.93 | 139.28 | Decode -11.5% |

The gate/up-only result confirms that doubling warp pressure harms graph
concurrency even when the latency-bound long-K projection is isolated. The
down-only result is neutral at best: its two warm epochs straddled the
accepted result, while the three-run median was lower. Four rows per warp
reduces CTA threads and activation loads but increases per-warp live state;
it also regresses. Do not revisit gathered-QMV row/warp launch geometry
without a different kernel architecture.

- `.cache/patches/mlx-fp-gather-qmv-rpw1-gate-up-experiment.patch`
- `.cache/patches/mlx-fp-gather-qmv-rpw1-down-experiment.patch`
- `.cache/patches/mlx-fp-gather-qmv-rpw4-experiment.patch`
- `.cache/bench/qwen36-5090-gather-gate-up-rpw1-20260731`
- `.cache/bench/qwen36-5090-gather-down-rpw1-20260731`
- `.cache/bench/qwen36-5090-gather-rpw4-20260731`

### Rejected: fused NVFP4 gate/up projection with GeGLU

Gemma4 12B decode launches separate NVFP4 gate and up QMV kernels followed by
GeGLU. A CUDA-only custom kernel preserved the separate checkpoint weights and
scales, reused each BF16 activation load across both projections, applied each
projection's direct global scale, and reproduced the existing BF16 GeGLU
rounding sequence. It used MLX's established one-warp-per-row QMV tiling and
native CUDA E2M1/E4M3 conversion types; Metal and unsupported shapes retained
the existing path.

Focused numerical coverage passed against two independent quantized matmuls
plus GeGLU. `TestBasic/blue-sky` and an independent long-prompt semantic check
also passed. A same-binary deterministic p16/g512 A/B parked MTP with logprobs:

| RTX 5090 | Control generate | Fused generate |
| --- | ---: | ---: |
| Five-epoch mean | 114.05 tok/s | 113.82 tok/s |
| Full-length rows only | 112.30 tok/s | 112.56 tok/s |

The result is neutral. Packed weights dominate traffic, so sharing activation
loads and one launch does not provide enough benefit to justify a second QMV
implementation in Ollama. The production helper, test, call site, and temporary
environment switch were removed.

- `.cache/bench/gemma4-12b-nvfp4-geglu-ab-5090-20260731`

### Rejected: fused post-attention RMSNorm, residual, and pre-FF RMSNorm

The deterministic Gemma4 12B trace showed 79.90 ms in the dominant guarded
`rms_norm_small<bf16,128,4>` shape. A CUDA-only custom kernel retained the
post-attention values in registers, reproduced the first RMSNorm's BF16
rounding, added the residual in BF16, returned that residual state, and then
computed the pre-FF RMSNorm as a second output. It replaced two RMSNorm
launches and one add per dense layer without changing Metal or checkpoint
weights.

Focused coverage compared both output tensors independently and passed.
`TestBasic/blue-sky` and the natural long-prompt gate also produced coherent
answers. A same-binary deterministic p16/g512 A/B parked MTP with logprobs:

| RTX 5090 | Control generate | Fused generate |
| --- | ---: | ---: |
| Five-epoch mean | 114.77 tok/s | 113.32 tok/s |
| Full-length rows only | 113.55 tok/s | 113.51 tok/s |

The full-length result is neutral. CUDA graph overlap hides the separate
launches, while the fused kernel's two sequential block reductions do not
shorten the graph critical path. The custom kernel, model call site, test, and
temporary environment switch were removed.

- `.cache/bench/gemma4-12b-rms-residual-ab-5090-20260731`

### Rejected: compile residual add and RMSNorm

A CUDA-only `mlx.Compile` helper fused the post-attention residual add with
the following RMSNorm in each Qwen layer. An exact primitive comparison,
`TestBasic/blue-sky`, and the natural long-prompt semantic check all passed.
The p2048/g128 endpoint result nevertheless regressed:

| RTX 5090 Qwen3.6 35B-A3B NVFP4 | Prompt median | Generate median |
| --- | ---: | ---: |
| Accepted eager path | 8,412.12 | 157.32 |
| Compiled add/RMSNorm | 8,003.61 | 152.82 |

The compile candidate's three epochs were `5385.60 / 152.82`,
`8003.61 / 154.98`, and `9719.57 / 146.25`. This confirms that another small
model-level fusion does not close the remaining decode gap; retain the eager
operations and focus on the dominant quantized matrix kernels.

- `.cache/bench/qwen36-5090-compiled-residual-norm-20260731`

### Rejected: SM120 sorted-RHS full scale slab

A Laguna p2048/g1 profile on RTX 5090 attributed 41.0% of measured GPU time
to sorted-RHS W4A16 QMM. Its two expert projections were
`K=2048,N=1024` at 49.70 ms across 38 calls and `K=512,N=2048` at
59.42 ms across 38 calls. This motivated adapting the retained SM121 NVFP4
full-scale-slab design to SM120.

The candidate was restricted to compute capability 12.0, NVFP4, no bias,
three or more K tiles, and at most 48 KiB total dynamic shared memory. The
48 KiB limit staged the complete K512 scale slab while leaving K2048 on the
accepted path. Bring-up found and fixed two correctness hazards before the
endpoint gate:

- The inherited three-stage mainloop prefetches three K tiles, so K64 must
  remain on the fallback path rather than reading beyond the scale slab.
- A shared slab is local to the current N tile, so scale lookup must use a
  local N coordinate instead of the global output coordinate.

Focused dequantize-plus-BF16-matmul tests then matched exactly for a staged
K512 shape and a fallback K2048 shape. The production graph still failed the
natural long-prompt semantic gate. The accepted binary answered the final sky
question coherently at 2,082 prompt tokens; the candidate ignored it and
summarized the distractor context. It also regressed measured prompt
throughput:

| RTX 5090 Laguna XS.2 NVFP4 | Prompt | Generate | Semantic result |
| --- | ---: | ---: | --- |
| Accepted control | 907.73 tok/s | 127.83 tok/s | Passed |
| SM120 scale slab | 545.84 tok/s | 123.59 tok/s | Failed |

Reject the SM120 adaptation. Exact isolated projections were insufficient to
prove safety in the sorted production graph, and the endpoint result moved in
the wrong direction even before a full p2048/g128 benchmark.

- `.cache/profiles/laguna-xs2-nvfp4-accepted-natural-p2048g1-20260731`
- `.cache/patches/mlx-sm120-qmm-rhs-full-scale-slab-experiment.patch`
- `.cache/patches/mlx-sm120-qmm-rhs-full-scale-slab-k-pipeline-guard.patch`
- `.cache/patches/mlx-sm120-qmm-rhs-full-scale-slab-local-n-tile-fix.patch`
- `.cache/patches/mlx-sm120-qmm-rhs-full-scale-slab-48k-cap.patch`
- `.cache/bench/laguna-5090-sm120-scale-slab-20260731`
- `.cache/correctness/laguna-5090-accepted-control-20260731`

### Laguna steady-state benchmark correction and graph-cache rejection

Laguna initially appeared to vary from roughly 1,400 to 6,900 prompt tok/s
across adjacent p2048/g128 epochs. Matched Nsight profiles of bench prompt
rotations 0 and 1 were effectively identical: 4,470 versus 4,281 tok/s under
profiling, with sorted-RHS QMM taking 105.8 versus 103.8 ms. Routing content
was not the cause.

The benchmark sequence exposed two warmup problems for this near-capacity
model:

- `bench.go` starts with its 1.3 tokens/word estimate, so the nominal p2048
  warmup is only 1,603 tokens before calibration.
- The first true p2048/g128 row raises peak MLX memory from about 25.4 to
  29.1 GiB, and the next row pays a one-time transition to about 30.3 GiB.

The ignored row helper's original scrub implementation was also invalid: it
launched a separate `bench` process, whose normal shutdown unloaded the model
and destroyed the warmed runner. The helper now runs scrub and measured epochs
in one process and filters scrub rows from the CSV. Laguna needs two discarded
full-shape rows to measure stable kernel/runtime throughput.

A matched two-scrub comparison rejected increasing SM120's graph cache:

| RTX 5090 Laguna XS.2 NVFP4 | Prompt median | Generate median |
| --- | ---: | ---: |
| Default 400 graph entries | 7,049.34 tok/s | 145.32 tok/s |
| 3,200 graph entries | 6,485.51 tok/s | 124.65 tok/s |
| llama-server target | 7,329 tok/s | 183.6 tok/s |

The default cache reaches 96.2% prompt and 79.2% generation target after
allocator warmup. Retain 400 entries on SM120. Do not use the earlier
3,022/134.5 headline to judge kernel throughput; it includes the undersized
warmup and high-watermark transition. Keep published comparisons explicit
about the scrub policy until `bench.go` can warm the calibrated target shape
without unloading the runner.

- `.cache/bench/laguna-5090-graph-cache400-20260731`
- `.cache/bench/laguna-5090-graph-cache3200-20260731`
- `.cache/profiles/laguna-xs2-nvfp4-accepted-bench-epoch0-p2048g1-20260731`
- `.cache/profiles/laguna-xs2-nvfp4-accepted-bench-epoch1-p2048g1-20260731`

### Accepted candidate: fused Laguna decode router

Laguna's eager sigmoid router built a large decode graph from sigmoid,
correction-bias add, argpartition, gathers, and normalization. A CUDA custom
kernel now handles the production decode contract only: BF16 logits shaped
`[rows, 256]`, FP32 correction bias, normalized top-8 routing, no global
expert scales, and `1 <= rows <= 8`. All other shapes, layouts, and backends
retain the existing MLX graph, so Metal and p2048 prefill are unchanged.

Focused CUDA comparisons against the original MLX expression passed for rows
1, 3, and 8 on RTX 5090, DGX Spark, and N1x. The N1x standalone test process
printed all passing comparisons before exiting with Windows status
`0xC0000409` during MLX test-process teardown; production-runner semantic
validation therefore remained mandatory there. Both its direct blue-sky
request and independent 2,092-token natural-prompt request returned coherent
Rayleigh-scattering answers.

The matched results show a decode gain on all three Blackwell targets:

| Host | Comparison | Accepted/control | Fused router | Change |
| --- | --- | ---: | ---: | ---: |
| RTX 5090 | p16/g128 generation median | 149.16 | 160.08 | +7.3% |
| DGX Spark | p2048/g128 generation median | 69.82 | 78.41 | +12.3% |
| N1x | p2048/g128 generation median | 4.26 | 5.75 | +35.0% |

The corrected N1x run used two in-process p2048/g128 scrub rows and then
measured `361.37/5.37`, `429.34/5.75`, and `427.37/5.86` tok/s. Its
`427.37/5.75` median must not be compared directly to the old unsrubbed
`86.67/4.26` prompt headline to estimate kernel benefit: the custom router is
gated out for p2048 and the prompt jump comes from benchmark conditioning.

The RTX 5090 Nsight candidate trace attributed 52.14 ms across 5,031 custom
router launches (10.36 us average), removed the old decode block-sort path,
reduced graph nodes by roughly 10,000, and reduced graph updates from 446 to
318. The profiled decode rate improved from 39.43 to 41.48 tok/s.

A later standard p2048/g128 run used two discarded full-shape rows in the same
process before three measured epochs. Its retained medians were `7,203.37`
prompt and `157.27` generation tok/s, or `98.3%` and `85.7%` of the unchanged
`7,329 / 183.6` llama-server target. This supersedes the old unsrubbed RTX
5090 headline for the fused-router checkpoint.

- `.cache/profiles/laguna-xs2-nvfp4-router-fusion-p16g128-20260731`
- `.cache/bench/laguna-5090-router-fusion-p16-20260731`
- `.cache/bench/laguna-5090-router-fusion-p2048-20260731`
- `.cache/bench/laguna-5090-accepted-control-p16-20260731`
- DGX Spark: `/home/daniel/code/ollama-upstream-um/.cache/bench/laguna-router-fusion-tater50-20260731`
- N1x: `C:\Users\daniel\.codex\src\ollama-mlx-current-best\.cache\bench\laguna-xs2-router-fusion-scrubbed-retry`

## Windows CUDA graph-cache audit

Daniel's earlier Windows graph fixes are still present in current MLX. The
original `origin/win_cuDNN_fix` branch contains:

- `2d4a7276c`: cache the instantiated cuDNN CUDA graph, update only pointers on
  reuse, and cache the child-subgraph key to avoid expensive WDDM graph-node
  queries.
- `eef24e7ec`: initialize `SDPACacheKey` field by field so uninitialized struct
  padding cannot make the `BytesKey` comparison miss every call.

Both landed through upstream merge `7adfc83c` and remain intact at
`db0bdc21a`. Current conv, FFT, and SDPA `BytesKey` call sites were audited and
do not repeat the aggregate-assignment padding defect. The later
thread-local-cache change also remains compatible with Ollama's single locked
MLX OS-thread worker.

Temporary CUDA graph telemetry counted graph commits, topology cache
hits/misses, executable updates, update failures, instantiations, and
non-updatable graphs. Correctness-gated Gemma4 E2B NVFP4 p2048/g128 runs found:

| Host | Final commits | Hits | Misses | Update failures |
| --- | ---: | ---: | ---: | ---: |
| RTX 5090 / Linux | 1,067 | 988 | 79 | 0 |
| N1x / Windows | 3,151 | 2,985 | 166 | 0 |

The second timed p2048/g128 request added four first-seen topologies on Linux
and twelve on Windows, but all reused graphs updated successfully. This rules
out the historical every-call cache miss or per-token executable rebuild as
the primary N1x regression. Artifacts:

- `.cache/diagnostics/gemma4-e2b-graph-stats-5090-repeat-20260731`
- `.cache/diagnostics/n1x-gemma4-e2b-graph-stats-repeat-20260731`

The material platform divergence is command-buffer fragmentation. Equivalent
timed requests used about 278-281 commits on SM120 Linux versus 795-798 on
SM121 Windows. The active SM121 adaptive policy defaults large-batch work to
20 ops / 25 MiB and small-batch work to 400 ops / 512 MiB. With CUDA graphs
disabled, `set_output_array()` returned before classifying batch size, so N1x
decode incorrectly used the 20-op limit.

A direct same-binary, correctness-gated N1x A/B on Gemma4 E2B NVFP4 compared
an explicit 20-op control with an initial classifier fix:

| Graph-disabled p2048/g128 | Prompt tok/s | Generate tok/s | Peak |
| --- | ---: | ---: | ---: |
| 20-op control | 1,017.21 | 7.50 | 7.75 GiB |
| Per-buffer adaptive candidate | 609.52 | 13.97 | 7.75 GiB |

Decode improved 86.3% with no memory or correctness regression, proving that
the conservative commit frequency is a major N1x cost. Prompt regressed because
the classifier reset after each automatic mid-eval commit and later prompt
fragments could be misclassified as small batches. The follow-up retains batch
classification across automatic commits and clears it only at CUDA
`gpu::finalize` / synchronize, matching the real graph-evaluation lifetime.
That follow-up must pass correctness and recover prompt before it is accepted.

Build guardrail: changing `device.h` legitimately invalidates most CUDA
translation units. Linux uses Ninja `--parallel 24` with NVCC `-t 2`; N1x uses
Ninja `--parallel 16`. Near 115/119 both builds narrow to one or two heavy CUDA
edges and host CPU becomes mostly idle; verify live `ninja`/`nvcc` processes
and the edge count before calling this expected long tail a stall.

### Rejected: graph-disabled adaptive command buffers

Moving the SM121 batch classifier outside the CUDA-graph guard and retaining
its classification until MLX evaluation finalization improved graph-disabled
N1x decode. On Gemma4 E2B NVFP4, the final correctness-gated p2048/g128 median
was 699.33 prompt and 13.88 generation tok/s, compared with 1,027.99 and 9.15
for the repeated 20-op control. A p2048/g16 split showed that the small-batch
policy did not directly slow prefill: the 40-op sweep measured 945.49 prompt
tok/s versus 957.32 for the 20-op control.

This is not a production direction. Disabling CUDA graphs gives up the launch
amortization required to reach llama-server parity, and larger models that
cannot fit with graphs need a separate memory-spike investigation. The
graph-disabled work is preserved only to avoid repeating the diagnostic:

- `.cache/patches/mlx-eval-scoped-adaptive-command-buffer-rejected-20260731.patch`
- N1x:
  `C:\Users\daniel\.codex\src\ollama-mlx-current-best\.cache\bench\gemma4-e2b-p2048g128-smallops400-final-windows`

The same source passed graph-enabled Qwen3.6 35B-A3B NVFP4 correctness on DGX
Spark and measured 2,272.86 prompt / 86.19 generation tok/s. That is 102.1%
and 111.9% of the fixed 2,225.52 / 77.03 llama-server target, but slightly
behind the retained staged graph policy's 2,295.73 / 89.63 result. Restore the
staged graph-enabled encoder baseline and focus N1x tuning on models that fit
with graphs, starting with Qwen3.6 27B NVFP4 at 114.9% prompt / 82.2%
generation target.

### N1x graph-enabled fit cohort

Graph-disabled tuning is no longer the active direction. Two smaller models
were tested on N1x with CUDA graphs enabled to separate general graph overhead
from near-capacity graph-workspace failures.

The benchmark client was subsequently refreshed from detached
`origin/bench-prompt` commit `80eb0c8b5`. It uses unique Python-patch
continuation prompts instead of the repetitive word-list corpus, which avoids
both unrealistic repetition and cross-epoch prefix-cache reuse. The same
client must be used for the MLX candidate and its llama-server target; do not
mix these measurements with older word-list rows.

Two Qwen3.5 2B graph-enabled controls passed `TestBasic/blue-sky`, an
independent 1,984-token natural-language prompt, and three complete
p2048/g128 epochs:

| Quant pair | MLX prompt / generate | llama prompt / generate | Percent of target |
| --- | ---: | ---: | ---: |
| NVFP4 / Q4_K_M | 2,366.98 / 41.60 | 1,428.57 / 31.37 | 165.7% / 132.6% |
| MXFP8 / Q8_0 | 2,331.52 / 41.10 | 1,533.80 / 32.56 | 152.0% / 126.2% |
| 4B MXFP8 / Q8_0 | 823.34 / 23.47 | 697.22 / 19.94 | 118.1% / 117.7% |

The first MXFP8 process timed out during the integration correctness request
with sustained GPU activity but no OOM, panic, or completed response. Per the
N1x flaky-hardware policy, one clean-process retry was allowed; it passed every
correctness gate and returned stable measured rows. Retain the successful row
while treating future unexplained failures as a retry-once condition rather
than a deterministic MXFP8 defect.

The 4B MXFP8 control also passed both correctness gates, but its margin narrowed
to 118.1% / 117.7%. This is still above target with graphs enabled, while
showing that the N1x capacity/workspace boundary begins to matter before the
hard graph-capture failures observed at 9B.

Gemma4 12B MTP is the first corrected-workload miss:

| Backend | Prompt tok/s | Generate tok/s |
| --- | ---: | ---: |
| MLX CUDA NVFP4 | 195.08 | 10.26 |
| llama-server Q4_K_M | 314.56 | 11.64 |
| Percent of target | 62.0% | 88.1% |

Both paths passed short and 1,962-token natural-long correctness checks. A
debug-only MLX follow-up found an accepted depth-1 MTP probe, but depth 1 cost
364 ms versus 85 ms for depth 0; the controller's expected rates were 5.5
versus 11.8 tok/s and it correctly kept steady-state decode at depth 0. This
rules out misleading accepted-token accounting as the gap. The MLX process
reported 7.35 GiB peak after load and up to 11.45 GiB during p2048/g128, beyond
the N1x's 8 GiB GPU aperture. The row primarily represents near-capacity
graph/workspace and memory-placement pressure. It should not replace the 2B
and 4B graph-fit controls when evaluating general Windows graph overhead.

Gemma4 E2B MXFP8 also passed the corrected correctness gates. Prompt variation
0 deterministically emitted only 52 tokens in two candidate processes, while
variations 1-3 completed all 128 tokens. Reporting matched variations 1-3 on
both backends avoids a short-sample bias:

| Backend | Prompt tok/s | Generate tok/s |
| --- | ---: | ---: |
| MLX CUDA MXFP8 | 2,947.20 | 27.43 |
| llama-server Q8_0 | 1,335.38 | 24.59 |
| Percent of target | 220.7% | 111.6% |

Do not replace this with a naive median that silently includes the 52-token
variation; retain the raw CSV and the matched-variation rule.

Qwen3.5 4B NVFP4 is the healthy graph-enabled control. The full custom
gated-delta input and output fusions were enabled, short and natural-long
correctness checks passed, and the p2048/g128 medians were:

| Backend | Prompt tok/s | Generate tok/s |
| --- | ---: | ---: |
| MLX CUDA | 1,147.90 | 33.70 |
| llama-server CUDA | 665.13 | 20.36 |
| Percent of target | 172.6% | 165.5% |

Gemma4 E2B NVFP4 initially appeared to be a graph-enabled prompt-gap case.
After strengthening the natural-long correctness prompt to require a direct
answer rather than a background summary, the exact candidate passed both
correctness checks. The first p2048/g128 medians were:

| Backend | Prompt tok/s | Generate tok/s |
| --- | ---: | ---: |
| MLX CUDA | 678.63 | 31.61 |
| llama-server CUDA | 1,325.03 | 24.60 |
| Percent of target | 51.2% | 128.5% |

That result did not represent the intended CUDA path. The N1x helper omitted
`OLLAMA_MLX_CUDA_WIDE_SDPA=1`, which silently routed Gemma4's 256-wide
attention heads through the explicit non-Metal `matmul/softmax/matmul`
fallback. A warmed, exact p2048 prefill trace then showed only 16 CUDA graph
launches, ruling out graph-launch fragmentation as the prompt bottleneck.
A matched RTX 5090 kernel trace identified the explicit 2,030 by 2,030 score
matrix path: FP32 score adds and BF16/FP32 conversions were the largest costs.

Enabling the existing CUDA wide-SDPA path passed both short and natural-long
correctness checks. Its three p2048/g128 rows were 2,547.79 / 26.99,
3,979.99 / 31.01, and 1,365.66 / 30.21 tok/s. The medians are:

| Backend | Prompt tok/s | Generate tok/s |
| --- | ---: | ---: |
| MLX CUDA wide SDPA | 2,547.79 | 30.21 |
| llama-server CUDA | 1,325.03 | 24.60 |
| Percent of target | 192.3% | 122.8% |

The prompt rows are variable, but the result decisively supersedes the manual
attention measurement. Graph-enabled N1x execution now exceeds target on both
Qwen3.5 4B and Gemma4 E2B. Keep wide SDPA explicit in benchmark helpers until
the production dispatch policy is resolved; omitting it invalidates Gemma4
CUDA comparisons.

### Gemma4 12B corrected-workload 5090 dispatch check

A clean `sm_120a` build exposed a separate experimental dispatch problem in
the shared MLX tree. The staged SDPA experiment allowed cuDNN to claim
Blackwell head dimensions through 512. Gemma4 12B then sent its 256/512-wide
prefill shapes through cuDNN, reached about 30.35 GiB peak memory on the RTX
5090, and failed the next distinct prompt with `cudaMallocAsync` OOM. Restoring
cuDNN's 128-wide limit routed those shapes through the existing MLX CUDA wide
SDPA kernel. The same binary then passed `TestBasic/blue-sky`, a 1,933-token
natural prompt, and three p2048/g128 epochs without an OOM.

The first exact Python-patch comparison was:

| Backend | Prompt median | Generate median |
| --- | ---: | ---: |
| MLX CUDA NVFP4 | 1,852.03 | 38.94 |
| llama-server Q4_K_M | 1,951.58 | 114.92 |
| Percent of target | 94.9% | 33.9% |

Do not treat the generation row as the final architecture result. The
`origin/bench-prompt` warmup starts from a 1.3 tokens/word heuristic; the MLX
tokenizer measured 2.99 tokens/word for this Python-patch corpus, so its
calibration request contained 4,711 tokens before shrinking timed requests to
2,045. The independent long correctness request and oversized calibration
left graph/prefix state near 30 GiB in the same process. The MTP controller's
depth-0 cost rose from about 9 ms before those requests to 27 ms afterward.
Repeat timing in a fresh process with bounded calibration before using this row
for optimization decisions. The raw runs are:

- `.cache/bench/gemma4-12b-5090-vector-pythonpatch-20260731`
- `.cache/bench/gemma4-12b-5090-llama-pythonpatch-20260731`

The bounded-calibration rerun supersedes that provisional row. Correctness
remained tied to the exact installed binary hash, while timing used a fresh
server and an initial MLX ratio of 3.0 tokens/word. Its warmup landed at 2,040
tokens and calibrated to 2.99; three measured requests each landed at 2,045.

| Backend | Prompt median | Generate median |
| --- | ---: | ---: |
| MLX CUDA NVFP4 | 1,978.13 | 109.58 |
| llama-server Q4_K_M | 1,951.58 | 114.92 |
| Percent of target | 101.4% | 95.4% |

The custom wide-SDPA path therefore restores prompt parity and leaves a narrow
4.6% decode gap. The MTP controller found depth 1 slower than plain decode and
settled at depth 0; the clean process measured depth-0 estimates near
112-122 tok/s. Continue from decode profiling rather than changing benchmark
accounting. Raw clean timing:

- `.cache/bench/gemma4-12b-5090-vector-fresh-pythonpatch-20260731`

### Rejected: compiled NVFP4 global output scale

Gemma4 12B ModelOpt NVFP4 decode pairs nearly every direct QMV with
BF16-to-FP32 conversion, a FP32 tensor-scale multiply, and conversion back to
BF16. In the correctness-gated p34/g128 trace, those three kernels consumed
about 171 ms across roughly 43,000 projection calls.

An Ollama-side `mlx.Compile` expression fused the conversion, multiply, and
conversion into one bit-exact elementwise kernel. The focused CUDA test passed,
and the trace reduced scale-chain GPU time to 77 ms. It also created one
`Compiled` primitive evaluation per scaled projection, however: 42,764 extra
compiled evaluations added about 114 ms of evaluator time before closure-apply
overhead. The fresh p2048/g128 median regressed from 1,978.13 / 109.58 to
2,136.73 / 98.61 tok/s. Prompt improved, but generation fell from 95.4% to
85.8% of the fixed 1,951.58 / 114.92 llama-server target.

Do not retry per-projection MLX compile for this chain. A useful solution must
pass the optional FP32 tensor scale into the QMV operation so scaling happens
inside its output epilogue, which requires an MLX API and CUDA-backend change.

- `.cache/patches/ollama-compiled-global-scale-rejected.patch`
- `.cache/bench/gemma4-12b-5090-global-scale-compiled-20260731`
- `/home/daniel/.cache/ollama-mlx/nsys-gemma4-12b/gemma4-12b-sm120a-server-p34g128-global-scale-compiled`

An Ollama-side fast CUDA custom kernel was also rejected. It implemented the
same BF16 input times FP32 scalar to BF16 output operation bit-exactly and
passed both the focused CUDA test and real-model correctness gates. Applying
it to all projection shapes caused severe prompt/JIT churn. Restricting it to
single-row decode removed that failure mode, but its fresh p2048/g128 median
was only 1,895.00 / 93.14 tok/s, versus the accepted 1,978.13 / 109.58.
Independent custom-kernel launches therefore cost more than the generic scale
chain they replace. Do not retry this as a separate primitive; fuse the scale
into the QMV epilogue instead.

- `.cache/patches/ollama-custom-global-scale-rejected.patch`
- `.cache/bench/gemma4-12b-global-scale-custom-5090-20260731`
- `.cache/bench/gemma4-12b-global-scale-decode-5090-20260731`

### Accepted: Gemma4 12B prefill-only QQMM global scale

The ModelOpt NVFP4 global scale can use MLX `QQMatmul` safely for prefill
shapes (`M >= 128`). This keeps decode on the ordinary full-precision
activation QMV path while eliminating the separate output-scale chain from
the large prefill projections. The focused primitive test, real
`TestBasic/blue-sky` integration test, and independent 1,933-token
natural-language check all passed. The natural response directly and
correctly explained Rayleigh scattering.

The three p2048/g128 rows were 2,732.62 / 102.56, 2,908.06 / 93.24, and
2,732.95 / 102.73 tok/s. Against the unchanged llama-server Q4_K_M target of
1,951.58 / 114.92 tok/s, the medians are:

| Backend | Prompt median | Generate median | Percent of target |
| --- | ---: | ---: | ---: |
| MLX CUDA NVFP4 | 2,732.95 | 102.56 | 140.0% / 89.2% |

The prefill improvement is accepted. Do not attribute the lower aggregate
generation median to the prefill dispatch itself: decode remains on the prior
QMV path, and the MTP completion trajectories varied between measured prompts.
Use generation-only profiling and explicit MTP acceptance accounting for the
remaining decode gap.

- `.cache/bench/gemma4-12b-global-scale-prefill-5090-20260731`
- `.cache/correctness/gemma4-12b-prefill-only-accepted-5090-20260731`

### Rejected: Gemma4 12B all-shape QQMM global scale

Routing single-row decode through `QQMatmul` enabled an MLX CUDA experiment
that fused the tensor global scale into the FP-QMV epilogue. The scalar
epilogue arithmetic passed focused numeric coverage, but `QQMatmul` also
quantizes and dequantizes its activation input before QMV. That extra
activation quantization changed model distributions and output trajectories.

A deterministic 48-case quality comparison against the prefill-only route
showed 38/40 selected-token agreement, 39/40 raw top-logprob winner agreement,
0.6551 mean top-5 Jaccard overlap, and 0/8 exact completion matches. The
prefill-only control repeated with 40/40 selected-token agreement, identical
logprobs, and 8/8 exact completions, proving the candidate drift was not
ordinary MTP run-to-run variance. An independent long prompt also regressed
from a direct Rayleigh-scattering answer to the incorrect claim that the
provided text lacked the requested information.

Do not use QQMM for Gemma4 NVFP4 single-row decode unless MLX gains a path that
preserves full-precision activations. The rejected CUDA backend diff is saved
only to prevent repeating the experiment; it is not a candidate patch.

- `.cache/quality/gemma4-12b-global-scale-fpqmv/comparison.json`
- `.cache/quality/gemma4-12b-global-scale-fpqmv/control-repeat-comparison.json`
- `.cache/patches/mlx-fpqmv-global-scale-activation-quantization-rejected.patch`

### Gemma4 12B generation diagnosis after prefill parity

The server-side speculative statistics fix from `origin/bench-prompt`
(`3ff2dcb64`) was applied as observability-only working-tree changes before a
correctness-gated p16/g512 diagnostic. The exact binary again passed
`TestBasic/blue-sky` and the independent natural-language correctness check.

The three measured generation rows were 99.65, 105.42, and 118.83 tok/s.
Their MTP behavior explains the spread:

- The first two measured continuations drafted only 7 and 1 tokens across 511
  rounds and accepted 1 token each, so they measured essentially plain depth-0
  decode.
- The third drafted 191 tokens, accepted 185 (97%), and reduced target rounds
  to 360, reaching 118.83 tok/s.
- The learned depth-0 cost was about 10 ms, or roughly 100 tok/s. MTP can exceed
  the 114.92 tok/s llama-server target when acceptance is favorable, but plain
  decode remains about 13-15% below target.

Therefore MTP policy is not the primary remaining kernel problem. Robust parity
requires a faster plain decode path, with MTP providing additional upside.

The strongest narrow candidate remains the ModelOpt NVFP4 global output scale.
The MLX CUDA QMV implementation already accepts an optional FP32 global scale
and applies it in the output epilogue while keeping BF16 activations. However,
the public `quantized_matmul` primitive and MLX-C API do not pass that scalar;
only `QQMatmul` does, and its single-row route quantize/dequantizes activations,
which the quality A/B rejected.

Do not fold the tensor scale into the stored block scales: NVFP4 block scales
are E4M3 bytes, so that would requantize an arbitrary FP32 multiplier and can
lose range and precision. The clean options are:

1. Add narrow optional-global-scale plumbing to MLX `quantized_matmul` and
   MLX-C, reusing the existing CUDA QMV epilogue and leaving Metal behavior
   unchanged.
2. Add an Ollama-local CUDA primitive that calls the same backend QMV path.
   This avoids a public MLX API change but adds more local backend surface and
   is less suitable for upstreaming.

- `.cache/bench/gemma4-12b-mtp-generate-p16g512-5090-20260731`

Qwen3.5 9B is too close to the N1x graph-workspace limit to serve as a stable
performance target. Splitting the custom gated-delta fusion showed:

| Fusion configuration | Prompt median | Generate median | Fresh-process stability |
| --- | ---: | ---: | --- |
| Generic control | 703.64 | 18.81 | stable |
| Output only | 849.87 | 18.79 | passed |
| Input only | 806.47 | 20.56 | failed graph capture in 2/3 processes |
| Input and output | n/a | n/a | failed natural-long graph capture twice |

The input fusion's larger graph raises temporary workspace pressure. A
model-name-independent CUDA memory-headroom policy now retains the smaller
output fusion but enables the input fusion only when MLX device information
reports at least 2 GiB remaining by both `total-active` and `free` memory.
Focused pure-Go policy coverage passes. This protects near-capacity graph
workloads without disabling graphs or creating per-device source variants.

### N1x Gemma4 E4B capacity check

Gemma4 E4B NVFP4 is not part of the N1x graph-fit cohort. Two clean candidate
processes failed during the one-token preload with the same
`cudaMallocAsync` out-of-memory error before either correctness gate ran. Its
9.6 GB artifact already exceeds the device's reported 8 GiB GPU aperture
before runtime state and CUDA graph workspace. Do not disable graphs or report
a benchmark row for this model; use Gemma4 E2B for the graph-enabled Gemma
control on the 64 GiB N1x.

The attempted run also exposed benchmark-client drift on the remote source
snapshot. N1x comparisons now use a cross-built Windows ARM64 executable from
detached `origin/bench-prompt` commit `80eb0c8b5befe3adcbdfefd1524138a77d4322f0`
with SHA-256
`9533a97b3673329f4c395bf0494ca9512dd234d7a854e528be210ca2bf1287ac`.
Both the candidate and official-server helpers accept this exact executable via
`-BenchExe`; do not mix new rows with the remote snapshot's older `bench.go`.

### Gemma4 12B wide-SDPA decode follow-up

The retained CUDA two-pass kernel now uses aligned vector loads for BF16/FP16
Q, K, and V and vector loads/stores for its FP32 partials at D=256 and D=512.
Focused D=256/GQA=2 and D=512/GQA=8 comparisons against explicit
matmul/softmax/matmul passed, as did `TestBasic/blue-sky` and the independent
2048-token semantic gate. In a warmed p2048/g128 trace, D=256 pass 1/pass 2
measured 42.6/15.9 ms versus 43.1/18.0 ms before vectorization; D=512 measured
19.2/5.2 ms versus 31.9/7.8 ms. The two-pass total fell from about 100.8 to
82.9 ms. Use the raw per-shape summaries below when revisiting this because
the aggregate endpoint is sensitive to MTP.

Artifacts:

- `.cache/profiles/gemma4-12b-sdpa-vector-loads-p2048g128-20260731`
- `.cache/bench/gemma4-12b-sdpa-vector-fresh-20260731`

A follow-up grouped-GQA scalar experiment processed two adjacent query heads
per two-pass block and reused each K/V load. It passed the focused numerical
test and both real-model correctness gates, but increased per-thread Q/output
state enough to lose performance. The D=256 first pass rose to 51.8 ms versus
42.6 ms for the retained per-head kernel; D=512 reached 16.2 ms in a profile
whose retained comparison was 19.2 ms. The D=256 regression dominates.

Decision:

- Reject grouped scalar GQA and restore the staged vector-load checkpoint.
- Do not retry additional scalar head grouping. llama.cpp dispatches D=512 GQA
  decode to an MMA attention kernel, while TensorRT-LLM similarly ships
  GQA-aware XQA kernels through D=256. The remaining high-upside direction is a
  tiled/MMA CUDA path that shares KV without multiplying per-thread register
  state.
- Nsight Compute counters remain blocked by `ERR_NVGPUCTRPERM`; do not infer
  occupancy counters from that failed capture.

Rejected profile:

- `.cache/profiles/gemma4-12b-sdpa-gqa2-p2048g128-20260731`

### Rejected: wide-SDPA `cp.async` K/V staging

A narrow CUDA experiment double-buffered complete eight-key K/V tiles in
shared memory with MLX's existing `cp_async<16>` helpers. It retained direct
vector loads for a partial tail and passed the D=256/GQA=2 and D=512/GQA=8
focused numerical comparisons, including K=2051, plus both real-model
correctness gates.

The staging overhead outweighed any latency hiding on the RTX 5090. In the
same p2048/g128 profile shape, D=256 first pass increased to 43.9 ms from
42.6 ms and D=512 increased to 20.4 ms from 19.2 ms. Second-pass timing was
unchanged. This isolates `cp.async` staging itself as unhelpful for the current
scalar CUDA-core decomposition; do not retry it without also changing the
attention tile/MMA structure.

Decision:

- Reject the `cp.async` variant and retain the staged aligned-vector checkpoint.
- Treat FlashInfer's staging as one component of a different full kernel shape,
  not as an independently useful optimization for the current MLX kernel.
- Investigate a narrow tiled/MMA path next, following llama.cpp's D=512 GQA
  dispatch and MLX's existing CuTe/CUTLASS conventions.

Rejected profile:

- `.cache/profiles/gemma4-12b-sdpa-cpasync-p2048g128-20260731`

### Rejected: D256 four-warp two-pass first stage

Reducing only the D=256 first-pass block from eight warps to four preserved
the 32-way split and final reduction. The focused installed-library numerical
test passed for D=256/GQA=2 and D=512/GQA=8, but the D=256 first-pass total
increased to 50.8 ms from the retained 42.6 ms. D=512 was unchanged at
19.0 ms.

Decision:

- Reject the four-warp variant and retain eight warps for the scalar kernel.
- Do not repeat the already rejected 16-split experiment. The scalar kernel
  benefits from both its current within-block and cross-block parallelism.
- Further meaningful improvement requires changing the computation shape,
  specifically grouped tensor-core MMA where the GQA factor supplies enough
  query rows, rather than reducing scalar launch parallelism.

Rejected profile:

- `.cache/profiles/gemma4-12b-sdpa-d256-bn4-p2048g128-20260731`

### Rejected: D512 grouped-GQA tensor-core SDPA

A CUDA-only prototype grouped Gemma4's 16 D512 query heads by their shared KV
head and used the existing Steel BF16 MMA primitives for both QK and PV. It
supported the real additive-mask path, preserved the existing two-pass online
softmax merge, and passed focused masked/tail comparisons at K=2051 for GQA8
and GQA16. The full blue-sky and independent long-prompt correctness gate also
passed for every tested split count.

The grouped launch exposed too little device-wide parallelism at the original
32 splits: pass 1 took 71.5 ms for 1,032 launches. Increasing to 128 splits
reduced pass 1 to 23.4 ms, but its 128 CTAs still underfilled the RTX 5090's
170 SMs. A final 160-split experiment nearly filled one device-wide wave and
reduced pass 1 to 21.7 ms. The generalized merge then cost 8.7 ms, leaving the
best MMA total at 30.4 ms versus 24.2 ms for the retained scalar vector path
(19.0/5.2 ms). The MMA kernel also used 118-122 registers per thread and 36 KB
of static shared memory.

Decision:

- Reject the grouped D512 MMA implementation and restore the staged aligned-
  vector checkpoint. Its best launch is still about 26% slower end to end.
- Do not retry split-count tuning for this design. The tested 32, 128, and 160
  points establish both the underfilled and near-full-device regimes.
- A future MMA attempt needs a different decomposition that lowers register
  pressure and merge traffic, not another launch-geometry adjustment. The
  existing scalar path remains the better implementation for these decode
  shapes.

Rejected profiles:

- `.cache/profiles/gemma4-12b-sdpa-mma-d512-gqa16-p2048g128-20260731`
- `.cache/profiles/gemma4-12b-sdpa-mma-d512-split128-p2048g128-20260731`
- `.cache/profiles/gemma4-12b-sdpa-mma-d512-split160-p2048g128-20260731`

### Rejected: single-barrier two-pass SDPA merge

The retained second pass coalesces each split's FP32 partial load, then uses a
small shared-memory transpose for every output fragment. A CUDA-only candidate
instead staged all 32 split outputs in dynamic shared memory, synchronized
once, and assigned each warp one output fragment to reduce. This preserved the
FP32 online-softmax merge arithmetic and passed FP16/BF16 primitive comparisons
for the exact Gemma4 12B D256/GQA2 and D512/GQA16 geometry at the non-divisible
K=2051 tail. The MTP-enabled blue-sky integration gate and independent long-
context target-model gate also passed.

The controlled full-length plain-decode rows weakly favored the candidate
(106.94 versus 105.10 tok/s median), but a calibrated p2048/g128 Nsight profile
showed that this was request noise rather than a better reduction. D256 pass 2
rose to 16.2 ms from the retained 15.9 ms, while D512 rose to 7.4 ms from 5.2
ms. The larger shared-memory footprint and changed access pattern outweighed
the removed barriers, especially at D512.

Decision:

- Reject the single-barrier full-partial staging layout and restore the staged
  coalesced transpose reduction.
- Do not infer merge performance from the endpoint A/B when completions have
  early-EOS rows; require per-kernel timing for this path.
- Preserve the candidate only as a dead-end reference:
  `.cache/patches/mlx-cuda-sdpa-single-barrier-merge-candidate.patch`.

Artifacts:

- `.cache/bench/gemma4-12b-sdpa-merge1-plain-ab-candidate-20260801`
- `.cache/bench/gemma4-12b-sdpa-merge1-plain-ab-control-20260801`
- `.cache/profiles/gemma4-12b-sdpa-single-barrier-merge-p2048g128-20260801`

### Rejected: exact NVFP4 QMV output-scale epilogue

The first QMV epilogue experiment reused MLX's NVFP4 `global_scale` amax
convention. Ollama converted the checkpoint's direct output multiplier by
multiplying it by 2,688 and the kernel divided it again. Thirty-eight of the
332 Gemma4 12B scales did not survive that float32 round trip exactly, and
matched seeded requests produced different token counts. Double-rounding the
FP32 accumulator through BF16 was necessary but could not repair the scale
conversion itself.

A corrected experimental API accepted the direct `output_scale`, retained the
ordinary QMV -> BF16 -> scale -> BF16 rounding order inside the CUDA kernel,
and left QQMM's native amax convention unchanged. A real checkpoint scale that
fails the 2,688 round trip passed a bit-exact focused comparison. The full
semantic gate passed, and control/fused five-epoch runs generated identical
token counts in both process orders.

The exact fusion nevertheless regressed plain decode:

| Process order | Control median | Fused median |
| --- | ---: | ---: |
| Control first | 116.25 tok/s | 113.55 tok/s |
| Fused first | 114.89 tok/s | 113.79 tok/s |

Decision:

- Reject the decode output-scale API and CUDA epilogue. Remove the experimental
  MLX and MLX-C surface instead of carrying it for a 1-2% regression.
- Retain the accepted prefill-only QQMM global-scale route. Single-row decode
  stays on ordinary QMV followed by the existing output-scale expression.
- Do not revive the approximate amax round trip: its apparent speedup was
  confounded by changed token trajectories and does not preserve model output.

Artifacts:

- `.cache/bench/gemma4-12b-qmv-output-scale-exact-ab-5090-20260731`
- `.cache/bench/gemma4-12b-qmv-output-scale-exact-ab-crossed-5090-20260731`
- `.cache/correctness/gemma4-12b-qmv-output-scale-exact-5090-20260731`

### Rejected: shared-LHS top-8 gathered QMV

Qwen3.6's decode gate/up gather uses one activation vector for all eight
selected experts. A CUDA-only candidate kept the retained eight outputs per
CTA and total CTA count, but mapped one warp to each expert and loaded the
common BF16 activation into shared memory once. Down projections and other
gather layouts retained the existing kernel.

The focused A/B forced the retained kernel with a second unused activation
row. Every BF16 result matched exactly at both production dimensions. The
isolated `K=2048,N=1024` gate/up shape improved from `1.696 us` to `1.460 us`
(`13.9%`), while the non-production shared-input `K=512,N=2048` shape was
slightly slower. `TestBasic/blue-sky` and the independent long-prompt semantic
gate both passed.

The full CUDA graph rejected the candidate:

| RTX 5090 Qwen3.6 35B-A3B NVFP4 | Prompt median | Generate median |
| --- | ---: | ---: |
| Retained checkpoint | 8,412.12 tok/s | 157.32 tok/s |
| Shared-LHS candidate | 6,906.05 tok/s | 111.39 tok/s |

Grouping all selected experts doubled the warps per CTA and serialized work
that MLX's retained per-expert CTAs can schedule concurrently. The graph-level
loss overwhelms the activation-traffic reduction. Do not retry cross-expert
CTA grouping without a design that preserves independent expert scheduling.

- `.cache/patches/mlx-fp-gather-qmv-shared-lhs-rejected.patch`
- `.cache/bench/qwen36-shared-lhs-gather-qmv-5090-20260731`
- `.cache/correctness/qwen36-shared-lhs-gather-qmv-5090-20260731`

### Rejected: CUB full-sort Laguna decode router

Laguna's fused sigmoid router originally selected its top eight experts with
eight repeated warp-wide maxima. A CUDA-only experiment replaced those scans
with `cub::WarpMergeSort` over the same 256 `(biased score, probability,
index)` candidates. The comparator retained the existing NaN, score, and
index tie-breaking contract, and the candidate used a distinct JIT kernel
name to avoid stale PTX reuse.

The focused rows 1, 3, and 8 comparison passed against the eager MLX
expression. `TestBasic/blue-sky` and the independent 2,048-token natural
prompt also returned coherent output. The scrubbed p2048/g128 endpoint
regressed:

| RTX 5090 Laguna XS.2 NVFP4 | Prompt median | Generate median |
| --- | ---: | ---: |
| Retained repeated selection | 7,203.37 tok/s | 157.27 tok/s |
| CUB full sort | 6,311.89 tok/s | 142.14 tok/s |

Sorting all 256 candidates performs substantially more work than selecting
only eight and adds shared-memory/register pressure to every sparse layer.
Reject and retain the existing fused router. A router follow-up needs a
partial-selection network rather than a full sort.

- `.cache/bench/laguna-5090-cub-router-p2048-20260731`
- `.cache/correctness/laguna-cub-router-5090-20260731`

### Rejected: local sorting-network Laguna router

A narrower follow-up retained the existing warp-wide top-eight reduction but
replaced each rank's eight-value local rescan with a fixed 19-comparator
sorting network and a per-lane selected-value shift. It used no CUB, shared
memory, or model-graph changes and had a distinct JIT name.

The rows 1, 3, and 8 focused comparison, blue-sky integration test, and
independent long-prompt check all passed. The scrubbed p2048/g128 endpoint was
still much worse:

| RTX 5090 Laguna XS.2 NVFP4 | Prompt median | Generate median |
| --- | ---: | ---: |
| Retained repeated selection | 7,203.37 tok/s | 157.27 tok/s |
| Local sorting network | 5,783.65 tok/s | 129.78 tok/s |

The address-taken candidate arrays and compare/swap network increase local
state and likely introduce spills; one measured generation epoch collapsed to
`75.19 tok/s`. Reject. The existing repeated top-eight selector remains the
best validated router implementation. Do not retry local/full sorting without
compiler evidence that the candidates remain in registers and a substantially
different partial-selection design.

- `.cache/bench/laguna-5090-local-sort-router-p2048-20260731`
- `.cache/correctness/laguna-local-sort-router-5090-20260731`

## Nemotron 3 Nano 4B hybrid CUDA bring-up

The two commits from `origin/nemotron-mlx` were cherry-picked onto the shared
Ollama tuning branch. The tested model is
`dhiltgen/nemotron-3-nano:4b-nvfp4`; its 42 blocks comprise 21 Mamba2, 17 MLP,
and four attention blocks. The checkpoint mixes NVFP4 and MXFP8 weights rather
than using NVFP4 for every projection.

Initial generation repeated raw `<|im_end|>` tokens. The checkpoint's numeric
generation EOS is ID 2 (`</s>`), while `tokenizer_config.json` identifies
`<|im_end|>` (ID 11) as EOS. The generic tokenizer loader retained only the
numeric ID. The accepted loader unions the resolved string EOS with numeric EOS
IDs and has focused coverage for `[2, 11]`. Thinking-suppressed, blue-sky, and
independent natural long-prompt checks are required before every benchmark.

Two symmetric CUDA custom kernels are accepted in the Ollama index:

- `mamba2_scan` and its snapshot variant mirror the Metal FP32 recurrence,
  stable softplus, state layout, and snapshot contract.
- `mamba_gated_group_rmsnorm` mirrors the Metal fused
  `RMSNorm(x * SiLU(gate), groups) * weight` path with a 256-thread FP32
  reduction. Coverage includes Nemotron's real `inner=7680`, `groups=8`,
  `groupSize=960` BF16 shape.

The standard RTX 5090 p2048/g128 result is:

| Path | Prompt median | Generate median | Complete 128-token generation |
| --- | ---: | ---: | ---: |
| MLX before CUDA scan | 313.22 | 101.46 | n/a |
| MLX with CUDA scan | 21,077.11 | 255.91 | 242.36 |
| MLX with scan + gated norm | 21,798.48 | 321.79 | 316.16 |
| llama-server Q4_K_M target | 8,500.02 | 312.24 | n/a |

The accepted MLX result is `256%` prompt and `103%` generation of target. Most
natural benchmark responses stop before 128 tokens, so retain the complete row
as the independent generation cross-check rather than relying only on medians.

The accepted p16/g128 Nsight Systems trace is under
`.cache/profiles/nemotron3/20260801T055444Z-nemotron-4b-mamba-gated-norm-p16g128`.
Its principal GPU costs are:

| Kernel family | GPU time | Share |
| --- | ---: | ---: |
| NVFP4 direct QMV | 131.73 ms | 36.3% |
| MXFP8 direct QMV | 93.74 ms | 25.8% |
| Mamba2 scan | 38.36 ms | 10.6% |
| BF16-to-FP32 copies | 23.06 ms | 6.4% |
| MLX RMSNorm | 16.44 ms | 4.5% |
| Fused gated group RMSNorm | 10.51 ms | 2.9% |

Direct-QMV shape attribution shows the largest repeated NVFP4 cost is the 21
Mamba input projections (`N=17504,K=3136`, 58.37 ms). The isolated MXFP8
`lm_head` (`N=131072,K=3136`) costs 32.40 ms, or 251 us/token. The remaining
MXFP8 layer outputs cost 56.47 ms and attention K projections cost 4.88 ms.

### Rejected Nemotron follow-ups

- CUDA scan blocks with 8 or 16 warps did not materially improve the kernel.
  Eight warps reduced scan time only `41.382 -> 40.708 ms`; 16 warps regressed
  to `41.231 ms`. Keep the Metal-aligned four-warps-per-block launch.
- Sharing `dt`, softplus, and decay across each four-warp scan block passed
  exact, snapshot, grouped-state, tail, and full-model correctness, but the
  synchronization regressed scan time from `38.358` to `39.041 ms`.
- A register-resident/two-stage warp reduction made the fused gated norm itself
  `7.3%` faster (`10.506 -> 9.743 ms`) but saved only about `0.2%` of traced
  decode and did not improve p2048/g128. Keep the simpler full-block reduction.
- Routing Blackwell single-row MXFP8 heads with `N >= 65536` through the
  existing CUTLASS/CuTe `qmm_sm80` path preserved coherent output but regressed
  the generation median from `321.79` to `304.92`; its only complete 128-token
  row was `260.92`. The M=16 tile wastes too much work for M=1 even at 131k
  output rows. A future head optimization needs a true M=1 tensor-core kernel,
  not a dispatcher change to the existing QMM.

Artifacts are under `.cache/bench/nemotron-4b-*`,
`.cache/correctness/nemotron-4b-*`, and `.cache/profiles/nemotron3/`.

## 2026-08-01: Nemotron biased depthwise conv + SiLU

Nemotron's Mamba2 path stored the convolution bias separately from `nn.Conv1d`,
which forced `C=9728, K=4` through MLX's generic convolution, bias add, and SiLU
graph even though the shared recurrent helper already had a fused Metal path.
The accepted model-side cleanup makes `Conv1d` own its bias and passes it through
the existing `WithConvSiLU` abstraction. Metal's biased case deliberately keeps
the same graph fallback.

The CUDA sibling uses one linear output thread, four unrolled FP32 `fmaf`
accumulations, bias, and SiLU. Its JIT key includes input, weight, and bias
dtypes; omitting `WeightT` previously allowed an FP32/FP32 module to be reused
for the real FP32/BF16 shape. Exact tests cover the real decode geometry and
the shared recurrent helper's biased path. Full thinking-suppressed and natural
p2048 correctness gates passed on RTX 5090, DGX Spark, and N1x.

The custom kernel is selected by compute capability, not model or device name:
SM120 retains cuDNN because an RTX 5090 trace measured the custom kernel at
`10.96 ms` versus `5.32 ms` for the prior generic convolution. SM121 uses the
custom kernel. The RTX trace proving positive custom dispatch before adding the
SM120 guard is:
`.cache/profiles/nemotron3/20260801T084503Z-nemotron-4b-depthwise-bias-cuda-v15-p16g128`.

Standard p2048/g128 results:

| Platform/path | Prompt | Generate | Prompt target | Generate target |
| --- | ---: | ---: | ---: | ---: |
| RTX 5090 SM120, retained cuDNN | 21,798.48 | 321.79 | 8,500.02 | 312.24 |
| DGX Spark SM121, before | 4,107.13 | 64.18 | 3,885.87 | 78.61 |
| DGX Spark SM121, custom median | 4,722.87 | 64.91 | 3,885.87 | 78.61 |
| N1x SM121, custom median | 789.96 | 19.27 | 525.84 | 26.88 |
| N1x SM121, paired cuDNN median | 565.86 | 15.94 | 525.84 | 26.88 |

DGX Spark reaches `121.5%` prompt and `82.6%` generation of target. N1x reaches
`150.2%` prompt and `71.7%` generation. The paired N1x custom run was `39.6%`
faster for prompt and `20.9%` faster for generation than the exact-source cuDNN
control, although the back-to-back temperature rose from 37-40 C to 42-46 C,
so do not use that percentage as a precise kernel-only speedup. All N1x rows
ended before 128 generated tokens and retain the benchmark warning.

Artifacts:

- DGX candidate: `/home/daniel/.codex/bench/nemotron-4b-depthwise-capability-final-v16-p2048g128`
- N1x custom: `nemotron-4b-depthwise-capability-final-v16-p2048g128`
- N1x cuDNN control: `nemotron-4b-windows-cudnn-final-v17-p2048g128`

This closes the generic convolution as a major prefill gap on SM121, but decode
is still dominated by direct NVFP4/MXFP8 QMV, Mamba2 scan, and conversion/copy
traffic. Further depthwise tuning is not the next high-leverage step.

### Rejected: SM121 split-K4 direct NVFP4 QMV

Nemotron's repeated Mamba input projection is a direct NVFP4 QMV with
`N=17504,K=3136`. A standalone DGX Spark probe split each output row across
four warps while retaining BF16 activations, E2M1 weights, E4M3 group-16
scales, and FP32 accumulation. It matched all 17,504 BF16 outputs and improved
the isolated kernel from `247.24 us` to `143.54 us` (`1.72x`). An MLX
integration restricted to SM121 and the exact production shape also matched a
forced retained-kernel control for every output and passed the full Nemotron
`TestBasic` integration suite.

The full CUDA graph rejected the candidate. The standard DGX Spark p2048/g128
median moved only from `4722.87 / 64.91` to `4533.37 / 65.67` prompt/generate
tok/s. A matched p32/g128 Nsight trace showed `437.45 ms` in the split kernel.
The retained trace's target generic-QMV contribution is about `409.66 ms`
(`950.85 ms` total NVFP4 direct QMV minus `541.19 ms` of unaffected shapes in
the candidate). The extra warps improve the isolated dependency chain but lose
concurrent graph scheduling and make the production operation about 6.8%
slower.

Reject and retain one warp per output row. Do not retry split-K for this shape
without a design that preserves row-level scheduling capacity. The rejected
diff is `.cache/patches/mlx-sm121-nemotron-qmv-splitk4-rejected.patch`; the DGX
endpoint and trace are `nemotron-4b-splitk4-n17504-k3136-v1-p2048g128` and
`nemotron-4b-splitk4-v1-p32g128`.

### Rejected: grouped Mamba2 scan

The retained CUDA scan maps a warp to each head-dimension value. A grouped
variant modeled on llama.cpp's `ssm-scan.cu` instead assigned one 128-thread
block to 16 D values so each B/C state load could be reused. It matched the
reference scan at `S=128`, passed the full Nemotron `TestBasic` integration
suite, and generated coherent output. A second variant also computed softplus,
decay, and each hidden load once per warp before broadcasting with shuffles.

Neither variant produced a real model-level win on DGX Spark. The first trace
reduced scan time only from `82.48 ms` to `81.00 ms`; its p2048/g128 median was
`4746.91 / 65.01` prompt/generate tok/s versus the accepted
`4722.87 / 64.91`. The broadcast refinement regressed scan time to `83.65 ms`.
Its p2048/g128 median of `4536.19 / 65.54` is consistent with run variance,
not a kernel improvement. Retain the existing Metal-aligned scan mapping.

The rejected Ollama diff is
`.cache/patches/ollama-mamba2-grouped-scan-rejected.patch`. DGX traces are
`nemotron-4b-mamba-grouped-v1-p32g128` and
`nemotron-4b-mamba-grouped-v2-p32g128`.

### Rejected: two packed words per lane for direct NVFP4 QMV

The Blackwell direct BF16/NVFP4 group-16 dispatcher normally processes four
packed weight words per lane. An isolated one-warp-per-row control used two
packed words per lane instead, preserving the number of warps and all FP32
arithmetic. It was bit-exact at every tested shape and looked exceptionally
strong in isolation: `17504x3136` improved from `246.62` to `130.04 us` on
DGX Spark and from `61.77` to `33.12 us` on RTX 5090. A DGX sweep from
K=2048 through K=8192 showed `1.86-1.91x` isolated speedups.

The full MLX graph reversed the result. Exact-output comparisons against an
`N+1` retained-kernel control and the full Nemotron `TestBasic` suite passed,
but the p2048/g128 median fell from `4722.87 / 64.91` to
`4378.87 / 63.08` prompt/generate tok/s. The p32/g128 trace showed every
production NVFP4 width regressing despite register use falling from 42 to 40:

| Output rows | Width 4 | Width 2 |
| ---: | ---: | ---: |
| 17,504 | 152.24 us | 162.56 us |
| 12,544 | 115.10 us | 122.17 us |
| 3,136 | 82.44 us | 85.59 us |
| 5,120 | 67.20 us | 85.23 us |
| 1,024 | 54.26 us | 62.99 us |

The lower-register kernel doubles its loop iterations and loses under graph
concurrency. Retain four packed words per lane. The rejected diff is
`.cache/patches/mlx-fp-qmv-n2-blackwell-rejected.patch`; the trace is
`nemotron-4b-qmv-width2-v1-p32g128`.

The same width-2 experiment is not useful for MXFP8. At Nemotron's
`131072x3136` vocabulary head it changed 11 BF16 outputs (maximum absolute
error 0.125) and improved only `1.0%` in isolation.

### Accepted: fused model-dtype round trip in SM121 depthwise conv

Nemotron intentionally rounded the FP32 depthwise-convolution/SiLU result
through BF16 before returning to FP32 Mamba state math. The SM121 custom
depthwise kernel now performs that exact FP32-to-BF16-to-FP32 boundary in a
register. SM120 and Metal retain the original graph path, and focused tests
compare the fused output to the explicit two-cast graph exactly.

Full DGX Spark correctness passed. The standard p2048/g128 candidate median
was `5021.95 / 66.00` prompt/generate tok/s versus the previous accepted
`4722.87 / 64.91`. All three measured responses ended coherently after 61
tokens. A paired p32/g128 trace removed exactly 2,730 FP32-to-BF16 and 2,730
BF16-to-FP32 copy launches while adding less than 1 ms to the fused depthwise
kernel itself. The retained trace is
`nemotron-4b-depthwise-roundtrip-v1-p32g128`.

### Accepted: consume BF16 Mamba dt directly

Nemotron's input projection already produces BF16 `dt`, but the model built a
separate BF16-to-FP32 copy before every fused scan. Both Metal and CUDA scan
sources already convert `dt` to FP32 in-register, so the adapter now accepts
BF16 `dt` while retaining FP32 hidden/state/A/D/dt-bias math. `D` and
`dt_bias` are converted once when weights load instead of rebuilding those
casts each forward.

MLX custom-kernel objects specialize their generated input signature on the
first evaluated dtype. Reusing one object for both FP32 and BF16 `dt` caused
the second invocation to be interpreted with the first signature. Separate
normal/snapshot JIT identities keep both public input contracts stable. An
exact CUDA A/B between BF16 `dt` and the prior explicit FP32 cast passes on
SM120 and SM121, as do full Nemotron correctness gates on all three hosts.

Standard p2048/g128 medians:

| Host | Prompt | Generate | Target | Percent of target |
| --- | ---: | ---: | ---: | ---: |
| RTX 5090 | 21,988.72 | 318.85 | 8,500.02 / 312.24 | 258.7% / 102.1% |
| DGX Spark | 4,686.91 | 68.66 | 3,885.87 / 78.61 | 120.6% / 87.3% |
| N1x | 805.48 | 28.62 | 525.84 / 26.88 | 153.2% / 106.5% |

Natural Nemotron responses often stop before 128 tokens; retain that warning.
The N1x run nevertheless passed both concise and 2,030-token natural-prompt
correctness checks before timing. Compared with the prior DGX round-trip trace,
BF16-to-FP32 copy launches fell from 11,265 to 3,096 and copy time fell from
54.23 ms to 6.69 ms. Scan time also fell from 81.33 ms to 70.66 ms. The trace
is `nemotron-4b-bf16-dt-v1-p32g128`.

### Rejected: direct QMV two-row activation reuse on SM121

The accepted gathered FP QMV computes two output rows per warp and reuses each
activation load. A direct-QMV variant reused the same implementation on SM121
with four warps per eight-row CTA, preserving the exact per-row accumulation
order. N-versus-N+1 controls matched bit-for-bit for NVFP4 at `4096x2048`,
`8192x2048`, and Nemotron's real `17504x3136` shape, and for MXFP8 at
`8x2048`. Full Nemotron correctness also passed.

The endpoint rejected it. DGX Spark p2048/g128 generation fell from the
accepted `68.66` median to `66.98 tok/s`; prompt was `4731.02 tok/s`. Reusing
the activation halves warp-level row parallelism and again loses scheduling
capacity in the concurrent graph. Retain one row per warp for direct QMV. The
rejected diff is
`.cache/patches/mlx-fp-direct-qmv-row-reuse-sm121-rejected.patch`.

### Rejected: paired-row NVFP4 global-load prefetch on SM121

Nsight Compute on the direct MXFP8 vocabulary QMV showed that the retained
one-warp-per-row kernel is latency-bound rather than occupancy-bound. For the
`N=131072,K=3136` head, the kernel took about `1.97 ms`, used 38 registers per
thread, and reached `78.06%` achieved occupancy, but schedulers had no eligible
warp for `90.03%` of cycles. Long-scoreboard waits accounted for `97.3%` of the
issue gap and the L2 hit rate was only `1.46%`. The nominal Nsight memory
throughput percentage is misleading on GB10 shared LPDDR: moving roughly
424 MB in 1.97 ms is about 215 GB/s, or 79% of the platform's 273 GB/s
theoretical bandwidth. The retained report is
`/home/daniel/.cache/nemotron4-mxfp8-head-sm121.ncu-rep`.

A paired-row NVFP4 probe tried to expose independent global loads by fetching
two rows of packed weights and E4M3 scales before beginning either row's
arithmetic. It retained BF16 activations, E2M1 weights, group size 16, FP32
accumulation, and the original per-row reduction order. The standalone probe
was exact and improved all production shapes by `1.52-1.88x`; for example,
`17504x3136` improved from `246.35 us` to `134.49 us` and `3136x12544`
improved from `189.19 us` to `100.62 us`.

The real CUDA graph rejected it. Full Nemotron correctness passed with a
coherent blue-sky response, but the standard DGX Spark p2048/g128 median was
`4746.73 / 67.31` prompt/generate tok/s. Generation regressed from the accepted
`~68.7 tok/s` range even though a p16/g128 screen had misleadingly reached
`71.97 tok/s`. As with the earlier sequential two-row reuse experiment, four
warps processing eight rows do not provide enough row-level scheduling
capacity under graph concurrency. Retain one warp per output row. The rejected
diff is
`.cache/patches/mlx-sm121-nvfp4-pair-prefetch-rejected.patch`; artifacts are
`nemotron4-pair-prefetch-v1-p16g128` and
`nemotron4-pair-prefetch-v1-p2048g128` under the DGX benchmark cache.

The high-level NVIDIA API gap for this operation remains concrete. cuBLAS and
cuBLASLt do not expose a weight-only `W4A16` matrix-vector path with E2M1
weights, per-group E4M3 scales at group size 16, BF16 activations, and FP32
accumulation. CUTLASS can express nearby GEMM collectives, but the generic QMV
dispatcher was neutral at the vocabulary head and about 25% slower at
`N=3136,K=7680`; native FP4 MMA paths require activation quantization and would
change model numerics. The smallest useful API extension would be a
cuBLASLt-style weight-only FP4 matmul/QMV descriptor that accepts the existing
compressed weight and grouped-scale layout while preserving BF16 activations
and FP32 accumulation. Keep this item for the post-parity NVIDIA API catalog.

### Rejected: larger graph and alternate MXFP8 launch paths on DGX Spark

Raising the CUDA graph limit from 400 to 800 operations produced a Nemotron 4B
p2048/g128 median of `5269.97 / 69.20` prompt/generate tok/s. Prompt improved,
but generation moved only about 0.7% and remained below target; the same global
policy had already regressed Qwen, so it is not retained.

Production-shape sweeps also rejected alternate MXFP8 QMV launch geometries.
The `N=131072,K=3136` head gained at most about 0.8% while exact, and wider
variants changed outputs. The `N=3136,K=7680` shape regressed by 4-6%.
The generic CUTLASS QMV dispatcher was neutral or slower at the head, about 25%
slower at `K=7680`, and about 3% slower at `K=12544`. A two-row load-prefetch
variant was neutral at the head, 6.4% slower at `K=7680`, and 5.7% faster at
`K=12544`; only nine down projections benefit while twelve Mamba output
projections regress, so the endpoint opportunity is below 1% and does not
justify another kernel variant.

### Rejected: direct-QMV cache prefetch and generalized async staging

Two additional one-warp-per-row probes tested whether explicit memory-latency
hiding could improve the dominant direct NVFP4 QMV without sacrificing graph
concurrency. `prefetch.global.L2` was inserted one, two, and four K tiles ahead
of arithmetic. At `N=17504,K=3136`, all candidates were exact, but timings were
`249.12`, `249.34`, and `244.82 us` versus the retained `246.39 us`; the best
result was only `1.006x` and is noise-level.

A generalized double-buffered `cp.async` kernel then staged each complete
1024-value K tile in shared memory while preserving one warp per output row,
the BF16/E2M1/E4M3 group-16 layout, FP32 accumulation, and direct handling of
the final 64-value tail. It was bit-exact at `N=17504,K=3136`, but took
`247.99 us` versus `247.14 us` for the retained direct kernel. The long
scoreboard stalls therefore do not reflect a simple lack of software
prefetching: neither an L2 hint nor explicit shared-memory staging improves
this shared-LPDDR workload. Do not integrate either path into MLX.

The probe source and runner are
`.cache/scripts/nvfp4_splitk_qmv_probe.cu` and
`.cache/scripts/run_nvfp4_splitk_qmv_probe_tater50.sh`. Keep them as negative
controls for future memory-pipeline work.

### Rejected: SM121 streaming-cache direct-QMV weight loads

The retained one-warp-per-row launch was kept intact while the aligned
128-bit packed-weight load used CUDA's documented `__ldcs(uint4)` intrinsic on
SM121. The cache hint was intended to keep the repeatedly reused BF16
activation vector resident in L1 while marking the one-use weight stream for
early eviction. Quantization, accumulation order, output dtype, and SM120 were
unchanged. Full Nemotron `TestBasic` correctness passed with a fresh PTX cache.

Nsight Compute confirmed that the experiment reached the intended binary:
the packed-weight instruction changed from `LDG.E.128` to `LDG.E.EF.128`, and
the current compiler emitted one `LDG.E.U16` for the paired NVFP4 scales. The
cache policy did not improve either dominant production shape:

| DGX Spark direct QMV | Retained | Streaming load |
| --- | ---: | ---: |
| NVFP4 `N=17504,K=3136` | 183.62 us | 188.51 us |
| MXFP8 `N=131072,K=3136` | 1.92 ms | 1.92 ms |

The MXFP8 head's L1 hit rate remained about 66.6%, no-eligible-warp cycles
remained about 89.7%, and occupancy/register use were unchanged. The hint
therefore does not hide the dependency-bound weight stream, and the candidate
was rejected before an endpoint benchmark. Do not retry cache operators for
this QMV without new counter evidence.

- `.cache/patches/mlx-sm121-direct-qmv-streaming-load-rejected.patch`
- `nemotron4-stream-weight-v1-first-qmv` and
  `nemotron4-stream-weight-v1-mxfp8-head` under the DGX NCU profile cache

### Rejected: four-warp SM121 direct-QMV CTAs

An SM121-only launch control kept one output row per warp but reduced each
direct-QMV CTA from eight warps to four. The intent was to lift NVFP4's 83.3%
theoretical occupancy ceiling without changing arithmetic or total row
parallelism. Full Nemotron `TestBasic` correctness passed.

Register-allocation granularity defeated the premise: four-warp CTAs retained
the same 83.3% theoretical occupancy for the real NVFP4 shape. Its launch
regressed from `183.62` to `186.88 us`. The MXFP8 vocabulary head remained at
100% theoretical occupancy but regressed from `1.92` to `1.93 ms`; no-eligible
warp cycles remained about 90%. Reject before endpoint benchmarking and keep
eight warps per direct-QMV CTA.

- `.cache/patches/mlx-sm121-direct-qmv-rows4-rejected.patch`
- `nemotron4-sm121-rows4-v1-first-qmv` and
  `nemotron4-sm121-rows4-v1-mxfp8-head` under the DGX NCU profile cache

### Rejected: narrower direct-QMV transactions and MXFP8 scale sharing

A direct NVFP4 candidate kept the retained four-packed-word loop, warp mapping,
FP32 arithmetic order, and eight-warps-per-block launch, but expressed each
128-bit packed-weight load as two aligned 64-bit loads. This isolates memory
transaction width from the already-rejected two-packed-word schedule, which
doubled loop iterations. Every standalone output was bit-exact. The dominant
`N=17504,K=3136` shape improved `247.30 -> 239.76 us` (`3.1%`), while the
other production shapes improved by only `0.1-1.6%`.

Both full-model correctness gates passed, including the independent long
prompt, but the real graph again rejected the isolated gain. The DGX Spark
p2048/g128 median was `4835.39 / 67.60` prompt/generate tok/s versus the
accepted `4686.91 / 68.66`; generation regressed by 1.5%. Retain the 128-bit
load. The rejected MLX diff is
`.cache/patches/mlx-sm121-fp-qmv-split-64b-load-rejected.patch`; correctness
and benchmark artifacts are `nemotron4-split-load-v1*` under the DGX caches.

The MXFP8 head probe also tested removing E8M0 conversion overhead and sharing
the group-32 scale between lanes. A manual exact E8M0 bit conversion without
sharing was bit-exact but only `1.3%` faster at the vocabulary head, below a
material endpoint threshold. The lane-sharing forms did not preserve outputs
and were rejected before MLX integration. The focused source and reusable
runner are `.cache/scripts/mxfp8_qmv_probe.cu` and
`.cache/scripts/run_mxfp8_qmv_probe_tater50.sh`.

### Rejected: ModelOpt scale fusion into RMSNorm and GeGLU

Gemma4 ModelOpt NVFP4 decode exposes scalar output-scale/cast chains after
QMV. Two Ollama-side CUDA custom-op experiments consumed the ordinary exact
BF16 QMV output and fused those chains into the next operation: Q/K/V RMSNorm
at head dimensions 256/512, and gate/up GeGLU at intermediate dimension
21504. The GeGLU primitive was bit-exact against the established BF16
expression and the RMSNorm primitive was numerically close while preserving
the explicit BF16 scale/cast boundary. Focused Nsight traces showed fewer
binary/copy launches, but that isolated launch reduction did not survive the
full model graph.

The original MTP-aware run appeared favorable only because the candidate
changed the token trajectory and happened to receive much higher draft
acceptance. A corrected matched A/B used `logprobs:true` to park MTP, the same
raw Python-patch continuation prompt family as `bench.go`, iterative p2048
calibration, one full p2048/g128 post-calibration warmup, unique cache-missing
prompts, complete 128-token generations, and a hard assertion that no
speculative statistics appeared. Absolute logprobs throughput is not a target
number because logprob calculation adds substantial work, but it is a valid
matched attribution test.

The control generated at `84.11 tok/s`. Combined RMSNorm+GeGLU fusion fell to
`45.09 tok/s`; isolated RMSNorm reached `61.62 tok/s`; isolated GeGLU reached
`72.29 tok/s`. Earlier attempts to extend RMSNorm fusion to the hidden-width
O/down projections also regressed at both 256- and 1024-thread launch sizes.
The custom boundaries inhibit the surrounding MLX graph/fusion/concurrency
enough to dominate the saved pointwise launches, so none of these paths are
retained. The rejected source snapshot is under
`.cache/rejected/gemma4-scaled-fusions-v4-20260801`; the matched artifacts are
the `gemma4-12b-*-true-plain3-5090-20260801` directories under `.cache/bench`.

The reusable matched harness is
`.cache/scripts/bench_generate_full_plain_api.py` plus
`.cache/scripts/bench_gemma4_12b_full_plain_5090.sh`. Keep `logprobs` runs for
candidate/control attribution only, not for the percent-of-target matrix.

### Rejected: two-head GQA-shared wide-SDPA first pass

A FlashInfer-inspired CUDA-only experiment assigned one query head to each
warp while pairing two adjacent GQA heads in a 512-thread CTA. The two heads
shared each 16-key K/V tile through dynamic shared memory while retaining the
existing 32 partial blocks and second-pass ABI. This was intentionally
different from the earlier grouped scalar attempts: no thread accumulated
multiple query heads, so the design preserved per-head warp parallelism while
halving global K/V loads for the pair. It applied only to BF16 qL=1, even GQA,
and D=256/D=512.

The focused primitive was exact and initially looked promising. At K=2051,
D=256/GQA2 improved from `73.019 us` to `56.565 us`; D=512/GQA16 was nearly
neutral at `71.104 us` versus `70.173 us`. Full blue-sky and independent
natural p2048 correctness gates also passed. A first p16/g128 trace was not
valid evidence because the context never crossed the two-pass threshold and
therefore launched only the one-pass kernel.

The corrected matched p2048/g128 Nsight comparison used the same prompt,
`logprobs:true` to park MTP, and one installed binary with only the temporary
dispatch switch changed. In the real graph, D256 first-pass time regressed
from `39.6 ms` to `49.9 ms` (`+26%`) across 5,160 launches and D512 regressed
from `18.7 ms` to `21.1 ms` (`+13%`) across 1,032 launches. Total SDPA time
rose from `602.1 ms` to `609.0 ms`; total measured generation also fell from
`26.60` to `26.04 tok/s` under profiler overhead. The primitive did not model
the production sequence-length distribution or concurrent graph scheduling
well enough to predict the endpoint.

Do not retain or retry this scalar shared-memory design. The exact rejected
diff and source are under
`.cache/rejected/mlx-cuda-sdpa-gqa-shared-v1-20260801`; the named MLX stash is
`rejected-cuda-sdpa-gqa-shared-v1-20260801`. Matched traces are
`gemma4-12b-sdpa-gqa2-v1-{candidate,control}-p2048g128-20260801` under
`.cache/profiles`.

### Qwen 3.6 multi-request cache-pressure and async-tail recovery

The recovered Qwen 3.6 decode fusions were fast in an isolated cached-prefix
trace but degraded across the standard multi-request benchmark. Temporary
CUDA graph counters showed a 50-entry working set in a 400-entry cache with
zero update failures, ruling out default graph-cache thrashing. A 64-entry
cache did thrash after enough distinct prompt shapes and must not be retried.
The temporary graph diagnostics were removed completely after attribution.

Trace-level runner accounting exposed the first real cliff: paged-out prefix
snapshots grew past 1 GiB beside a 20.4 GiB model on the 31.8 GiB RTX 5090.
Generation fell from `150.79` and `142.45 tok/s` to `72.34 tok/s`. A temporary
512 MiB ceiling kept three measured rows at `158.20`, `153.94`, and
`149.24 tok/s`. The retained generalized policy leaves Metal's existing 8 GiB
default unchanged, uses CUDA device memory plus each request's observed peak
to reduce the budget under pressure, and preserves a 512 MiB useful-snapshot
floor. Focused eviction and budget tests cover the policy.

A second A/B found that `OLLAMA_DEBUG=2` alone restored decode speed. The trace
path delayed the next request long enough for asynchronous cache finalization
to drain. Changing `cacheSession.close` from `AsyncEval` to `Eval` removes that
inter-request GPU tail without changing model math. With normal debug logging,
the standard one-warmup/two-scrub/three-retained p2048/g128 run produced
`137.53`, `150.75`, and `150.80 tok/s` and passed blue-sky correctness. The
artifacts are:

- `.cache/diagnostics/qwen36-cache-residency-debug-20260801`
- `.cache/diagnostics/qwen36-prefix512m-probe-20260801`
- `.cache/diagnostics/qwen36-final-debug2-ab-20260801`
- `.cache/bench/qwen36-sync-cache-close-20260801`

Do not wrap per-layer custom-kernel apply calls in `mlxCall`: its redundant
`runtime.LockOSThread`/`UnlockOSThread` pair on the already-owned runner thread
collapsed generation to about `40 tok/s`. Correctness/error plumbing for hot
adapter calls must assume or assert the runner's established thread ownership
without paying per-layer runtime lock traffic.

The one required apples-to-apples llama-server refresh used the same current
Python-patch bench binary, p2048/g128 shape, one warmup, two scrub rows, and
three retained epochs. Blue-sky correctness passed and llama-server fully
offloaded 42/42 layers. Its retained rows were `817.14 / 205.34`,
`4699.57 / 211.76`, and `668.69 / 208.37 tok/s`, for medians of
`817.14 / 208.37`. Against the corrected MLX medians of
`3273.53 / 150.75`, the current target is `400.6% / 72.3%`. The old
`5830 / 219.7` target used the repetitive-prompt corpus and must not be mixed
with this cohort. The remaining Qwen gap is decode-only; prompt graph behavior
is no longer a blocker for this row. The target artifact is
`.cache/bench/qwen36-35b-llama-python-patch-refresh-20260801`.

### Rejected: sparse executable-graph parameter replay

A matched p16/g128 llama-server trace measured `195.02 tok/s` and about
`624 ms` of CUDA kernel work, while the recovered MLX trace contained about
`676 ms` of CUDA kernel work. The kernel-only difference is roughly
`0.41 ms/token`; the larger end-to-end gap includes MLX rebuilding direct CUDA
graph nodes before updating each cached executable. Nsight magnifies
`cudaGraphExecUpdate`, so its traced API time is not an endpoint attribution.

A temporary CUDA-backend diagnostic cloned the first graph for each structural
key and compared every direct kernel function, geometry, and argument byte on
later hits using CUDA 13 parameter introspection. Across successive samples,
`94.9%`, `95.1%`, `97.1%`, and `98.0%` of direct nodes changed parameters.
Sparse `cudaGraphExecKernelNodeSetParams` replay therefore cannot avoid enough
work to beat whole-graph update; the earlier standalone probe already showed
per-node updates lose when most nodes change. The diagnostic was removed
completely. Do not retry graph replay without a design that also eliminates
parameter churn or moves parameters behind stable device-side indirection.

- `.cache/diagnostics/qwen36-graph-param-stats-5090-20260801`
- `.cache/profiles/qwen36-35b-a3b-q4km-llama-p16g128-20260801`
- `.cache/profiles/qwen36-35b-a3b-nvfp4-recovered-v2-p16g128-20260801`

### Rejected: Qwen decode MoE final-combine custom primitive

A CUDA-only custom primitive fused Qwen 3.6's eight-expert BF16 weighted
reduction, shared-expert sigmoid gate, shared multiply, and final add for the
exact `[1,1,8,2048]` decode shape. A focused unit test matched the native MLX
expression within `1e-3`, and blue-sky plus independent long-prompt
correctness passed. The standard p2048/g128 run nevertheless collapsed from
the recovered `137.53`, `150.75`, and `150.80 tok/s` decode cohort to retained
rows of `88.07`, `90.96`, and `87.80 tok/s`. The custom primitive boundary
costs more in MLX graph scheduling/concurrency than the removed pointwise and
reduction launches save. Do not retry this fusion unless custom primitives can
participate in the surrounding graph without that boundary cost.

The exact rejected source is under
`.cache/rejected/qwen-moe-combine-v1-20260801`; correctness and benchmark
artifacts are under `.cache/bench/qwen36-moe-combine-{correctness,p2048g128}-20260801`.

### Rejected: alternating CUDA graph executable instances

A backend-private experiment retained two `cudaGraphExec_t` instances per
structural cache key and alternated between them. The intent was to let the
next executable update proceed without contending with the prior in-flight
launch while preserving the same source graph, dependencies, stream order,
and public API. Full-model correctness and the long-context semantic gate
passed.

The stabilized Qwen p16/g128 retained rows were `156.65`, `144.40`, and
`150.49 tok/s`, a `150.49 tok/s` median that does not improve the proven
single-executable result. The ring also doubles first-use graph instantiation
work and executable memory. This shows the repeated executable is not the
serialization point limiting decode on the current stream. Do not retry a
per-key executable ring without evidence of an actual in-flight update wait.

- `.cache/patches/mlx-cuda-graph-exec-ring-v1.patch`
- `.cache/bench/qwen36-graph-ring2-correctness-20260801`
- `.cache/bench/qwen36-graph-ring2-p16g128-20260801`

### Laguna Python-patch llama-server target refresh

The previous `7,329 / 183.6 tok/s` Laguna target had no retained local
artifact and predated the unique Python-patch cohort. A one-time refresh used
the official Ollama v0.32.4 llama-server payload, full 39/39 GPU offload,
blue-sky correctness, the current Python-patch bench client, one warmup, two
scrub rows, and three retained p2048/g128 epochs. The retained prompt rows were
`5572.22`, `5898.35`, and `5266.74 tok/s`; generation was `177.87`, `178.85`,
and `179.62 tok/s`. Use the medians `5572.22 / 178.85` for subsequent Laguna
work and do not rerun this unchanged target.

- `.cache/bench/laguna-llama-python-patch-refresh-20260801`

### Rejected: allocator clear after pressure-driven snapshot eviction

Laguna's unique-prompt run reached about `30.3 GiB` and fell from a first
retained `128.64 tok/s` decode row to `35-38 tok/s`. The adaptive prefix-cache
policy evicted roughly 0.5 GiB of snapshots after the first slow row. A
candidate additionally called `mlx.ClearCache()` when pressure-driven eviction
actually freed snapshot arrays, ensuring their buffers were not merely kept
in MLX's allocator cache. The following retained rows still measured only
`38.23` and `39.91 tok/s`, with peak memory remaining about `30.2 GiB`.

The persistent pressure is not allocator-cached snapshot storage. Do not add
an unconditional cache clear to this path; investigate resident CUDA graph
executables and graph-cache sizing instead.

- `.cache/bench/laguna-cache-release-p2048g128-20260801`

### Rejected: Laguna prefix-budget and graph-cache sizing follow-ups

Starting the 32 GiB CUDA prefix snapshot budget at 512 MiB evicted old
snapshots before the slow rows, but retained generation still fell to `36-37
tok/s`. Combining that early cap with allocator clearing after every eviction
also remained at `37-39 tok/s`. A 200-entry graph cache made the first retained
row worse (`92.13 tok/s`) and did not prevent the cliff; older retained 800-
and 1,600-entry experiments were also slower than 400. Snapshot budget,
allocator caching, and graph-cache capacity are therefore not the primary
cause of the new unique-prompt collapse.

- `.cache/bench/laguna-initial-prefix-cap-p2048g128-20260801`
- `.cache/bench/laguna-prefix-pressure-policy-p2048g128-20260801`
- `.cache/bench/laguna-graph-cache200-p2048g128-20260801`

### Laguna collapse isolated beyond runner prefix retention

Restoring asynchronous cache finalization delayed the collapse by one retained
row (`127.48`, `141.50`, then `37.32 tok/s`) but did not eliminate it. A final
isolation control disabled CUDA cross-request prefix retention entirely,
released evicted allocator buffers, and kept asynchronous finalization. Its
retained generation rows were `35.22`, `96.70`, and `33.23 tok/s`, so prefix
snapshots are not the persistent state causing the slowdown. Restore the
synchronous finalization required by Qwen and investigate MLX CUDA graph
executable residency/clearing instead.

- `.cache/bench/laguna-async-cache-close-p2048g128-20260801`
- `.cache/bench/laguna-no-prefix-retention-p2048g128-20260801`

### Rejected: clearing retained CUDA graph executables

An opt-in CUDA-backend diagnostic synchronized the current command encoder
and cleared only its graph-executable LRU from the existing allocator-cache
boundary. Clearing on every request made the next small request spend roughly
53 seconds rebuilding cold graph state, confirming that executable retention
is essential rather than disposable bookkeeping. A grouped follow-up retained
graphs for three requests at a time so each group contained cold and warm
samples. It produced warm prompt rows as high as `9492.06` and `5831.81
tok/s`, but generation still collapsed from `135.38` and `143.80 tok/s` to
`51.08`, `48.81`, and `46.88 tok/s`; a graph-cache reset did not restore it.

The cross-request Laguna collapse is therefore not stale graph-executable
residency alone. Do not ship request-boundary graph clearing or retry graph
cache capacity changes. The diagnostic source was removed; its exact patch is
retained only for attribution history.

- `.cache/patches/mlx-cuda-clear-graph-cache-diagnostic.patch`
- `.cache/bench/laguna-graph-cache-groups3-p2048g128-20260801`

### CUDA async-pool reservation cliff and retained fix

Request-boundary telemetry isolated Laguna's multi-request collapse below
MLX's normal active/cache accounting. Without pool trimming, CUDA's default
async allocation pool grew from `25.3 GiB` reserved to `32.48 GiB` reserved
while MLX ended at only about `18.4 GiB` active and `0 B` allocator-cached.
`cudaMemGetInfo` reached `0 B` free, and retained generation fell from
`111.95 tok/s` to `38.19` and `29.20 tok/s`.

Calling `cudaMemPoolTrimTo` immediately after MLX's asynchronous frees was a
no-op because those frees were still queued on the allocator's dedicated free
stream. The retained CUDA-backend fix extends explicit `clear_cache()` under
pressure: below 5% driver-free memory, synchronize only the allocator free
stream, query that device pool's completed used footprint, and trim unused
reservation to it. This is device- and model-independent and changes no MLX
API. The clean no-environment validation kept all retained generation rows
stable at `135.47`, `145.24`, and `132.74 tok/s`. This removes the catastrophic
cliff but does not close Laguna's remaining kernel gap; the median is still
only `75.7%` of the `178.85 tok/s` target.

- `.cache/bench/laguna-cuda-pool-telemetry-p2048g128-20260801`
- `.cache/bench/laguna-cuda-pool-trim-p2048g128-20260801`
- `.cache/bench/laguna-cuda-pool-sync-trim-p2048g128-20260801`
- `.cache/bench/laguna-cuda-pool-pressure-trim-p2048g128-20260801`
- `.cache/bench/laguna-cuda-pool-pressure-trim-clean-p2048g128-20260801`

### Rejected: cached Laguna decode identity indices

The post-pool-fix Laguna trace contained `10,214` `arange<uint32>` launches
(`37.72 ms`, or 4.4% of traced GPU time). These are the default LHS indices
created by `gather_qmm` for each decode expert projection. A candidate followed
the existing Gemma4 pattern exactly: it pre-evaluated shared `[0]` gate/up and
`[0..top_k)` down-projection indices during model load and supplied them only
for one-token non-Metal decode. The independent blue-sky and 2,082-token
semantic checks both passed.

The matched p16/g128 endpoint A/B did not show a useful win:

| RTX 5090 Laguna XS.2 NVFP4 | Generate median |
| --- | ---: |
| Default implicit indices | 144.39 tok/s |
| Cached explicit indices | 145.54 tok/s |

The `+0.8%` result is within the three-epoch spread. CUDA graph scheduling
hides most of the standalone `arange` cost, so duplicating the model plumbing
is not justified. The candidate was removed. Do not infer endpoint leverage
from this aggregate trace bucket without a materially different backend
fusion that reduces graph critical-path work.

- `.cache/bench/laguna-control-p16g128-20260801`
- `.cache/bench/laguna-decode-identity-p16g128-20260801`
- `.cache/profiles/laguna-xs2-nvfp4-pool-fix-p2048g128-20260801`

### Rejected: vectorized E4M3 scale loads in fast gathered NVFP4 QMV

The `group_size=16`, four-bit fast gathered QMV loop consumes two adjacent
E4M3 scales per packed-weight step. A CUDA-only candidate loaded that pair
with the backend's existing `unsafe_load_vector<2>` helper rather than two
scalar expressions. Focused `GatherQMM` comparisons against dequantize plus
`GatherMM` were bit-exact (`max_abs_diff=0`) for compact and production
`N=1024,K=2048` and `N=2048,K=512` shapes. Full Laguna and Qwen correctness
also passed.

Laguna initially looked favorable: p16/g128 generation rose from `144.39` to
`150.82 tok/s`, and its p2048/g128 medians reached `3754.63 / 146.27 tok/s`
versus the post-pool-fix control's `3078.47 / 135.47`. The shared CUDA path,
however, produced a severe repeatable Qwen 3.6 long-context regression despite
neutral p16 performance. The first standard run retained only `119.50`,
`106.32`, and `108.84 tok/s`; an idle-host repetition retained `71.40`,
`71.83`, and `72.17`, versus the known-good `137.53`, `150.75`, and `150.80`.
Qwen 3.6 35B A3B does not load an MTP head, both runs used the same deterministic
prompt/token-count sequence, and memory peaks were comparable, so draft
acceptance and prompt variance do not explain the loss.

The candidate was removed. Exact primitive equality is not sufficient here:
changing the gathered-kernel specialization can materially alter long-context
CUDA graph scheduling or residency while remaining neutral in isolated decode.
Do not retry this vector-load specialization without a matched graph-level
attribution that explains and prevents the cross-model regression.

- `.cache/patches/mlx-fp-gather-qmv-vector-scales-experiment.patch`
- `.cache/bench/laguna-gather-vector-scales-p16g128-20260801`
- `.cache/bench/laguna-gather-vector-scales-p2048g128-20260801`
- `.cache/bench/qwen36-gather-vector-scales-p16g128-20260801`
- `.cache/bench/qwen36-gather-vector-scales-p2048g128-20260801`
- `.cache/bench/qwen36-gather-vector-scales-repeat-p2048g128-20260801`

### Rejected: direct QMV activation sharing through the gathered fast kernel

Metal's fast floating-point QMV shares one activation-vector load across four
output rows per SIMD group. CUDA's retained fast implementation shares across
two rows per warp, but was wired only for gathered MoE projections. A
CUDA-only candidate reused that existing implementation for aligned,
single-vector, complete eight-row direct QMV tiles. SM121 kept its previously
validated vector-scale specialization rather than changing two optimizations
at once.

Focused NVFP4 tests were bit-exact against dequantize plus matmul
(`max_abs_diff=0`) for `N,K` shapes `128,512`, `256,2048`, `1024,2048`, and
`12544,2048`. Full Laguna correctness and the 2,082-token semantic probe also
passed. The p16/g128 generation median improved only slightly, from `144.39`
to `145.93 tok/s`.

The standard p2048/g128 endpoint rejected the candidate. Retained generation
was `131.80`, `139.85`, and `91.69 tok/s`, below the known-good `135.47`
median and ending in a pressure cliff as peak memory reached `30.28 GiB`.
Prompt rows were also unstable (`2338.85`, `2660.78`, `1407.61 tok/s`). The
shared-row arithmetic is sound, but this direct launch changes long-context
graph/residency behavior enough to outweigh its isolated memory-load saving.
Do not wire the gathered fast implementation into direct QMV in this form.

- `.cache/patches/mlx-fp-qmv-direct-fast-experiment.patch`
- `.cache/bench/laguna-direct-fast-qmv-p16g128-20260801`
- `.cache/bench/laguna-direct-fast-qmv-p2048g128-20260801`

### CUDA pool headroom under distinct prompt shapes

Seven successive unique p2048/g128 shapes showed why the retained 5%
driver-free trim threshold was still too late. CUDA's async pool repeatedly
held roughly `5-8 GiB` more reservation than its current used footprint. The
sixth request entered prefill with about `6.4 GiB` driver-free, needed roughly
`7.1 GiB` of transient backing for its new graph shape, reached `0 B` free,
and fell to `58.48 / 23.07 tok/s`. The pool did trim at prompt completion, but
only after WSL had already entered its expensive residency path.

Treating MLX's explicit `clear_cache()` as a request to drain asynchronous
frees and return all currently unused default-pool reservation kept end-of-
request driver-free memory near `6-11 GiB` and prevented the decode cliff.
One prefill still reached `0 B` because retained prefix/graph state had grown
just beyond the next shape's transient requirement. Combining pool trimming
with the existing adaptive prefix policy reduced stale snapshots before the
next request. A 1/8 device reserve passed the seven-shape diagnostic but
reacted too late in the standard scrub/retained sequence, where the final row
still collapsed after a `30.18 GiB` spike. The follow-up uses a 1/6 reserve so
the observed `26.57 GiB` warmup peak on a 31.84 GiB device triggers before
measured rows. All seven diagnostic rows remained in the normal
kernel-limited band: prompt `2937.34-6282.43 tok/s` after excluding setup and
generation `129.02-149.55 tok/s`. No slow residency row remained.

This does not overturn the earlier isolated prefix-cap rejection: snapshots
were not the sole cause. The failure required both CUDA pool reservation and
near-capacity retained state, so the two fixes are complementary. Before
retaining the combination, remove `MLX_CUDA_POOL_TELEMETRY` and validate the
normal one-warmup/two-scrub/three-retained protocol on Laguna plus the Qwen
control to detect any cost from the stronger `clear_cache()` synchronization.

- `.cache/diagnostics/laguna-pool-telemetry-v2-20260801`
- `.cache/diagnostics/laguna-pool-trim-always-20260801`
- `.cache/diagnostics/laguna-pool-trim-prefix-reserve8-20260801`
- `.cache/bench/laguna-pool-trim-prefix-reserve8-clean-20260801`

### Rejected: explicit CUDA pool trimming across requests

Clean Qwen 3.6 controls showed that explicit async-pool reclamation is not a
general solution to Laguna's residency cliff. The known-good p2048/g128 run
retained generation at `137.53`, `150.75`, and `150.80 tok/s`. Unconditional
trim plus the 1/6 prefix reserve retained only `83.21`, `69.64`, and
`73.21 tok/s`; restoring the 1/16 reserve made it worse at `44.85`, `44.91`,
and `43.74 tok/s`. Restricting trim to the original below-5%-free pressure
condition still retained only `110.51`, `98.94`, and `102.53 tok/s`.

CUDA's default pool release threshold is zero. Synchronizing the allocator's
free stream therefore released unused reservation before a subsequent
`cudaMemPoolTrimTo(used + headroom)` could preserve the requested warm
headroom. Temporarily setting `cudaMemPoolAttrReleaseThreshold` to
`UINT64_MAX` before synchronization proved that ordering and made the requested
reservation survive, but did not recover endpoint performance. A 1/8 warm
reserve retained a `113.31 tok/s` median and a 1/6 reserve regressed to about
`69.84 tok/s`; the relationship was not monotonic. Deferring the final prefill
`clear_cache()` until request teardown also failed, retaining only `99.16`,
`114.21`, `73.94`, `74.79`, and `75.46 tok/s` in the diagnostic sequence.

The conclusion is that any explicit cross-request pool trim tested so far
destroys Qwen's fast retained CUDA graph/allocation state. Do not retain the
request-end-only runner change or the explicit trim policy. The next candidate
should use CUDA's native release-threshold policy to bound ordinary pool growth
without a forced synchronization or `cudaMemPoolTrimTo`, then validate both
Laguna and Qwen under the same policy.

- `.cache/bench/qwen36-pool-trim-prefix-reserve6-clean-20260801`
- `.cache/bench/qwen36-pool-trim-prefix-reserve16-control-20260801`
- `.cache/bench/qwen36-pool-pressure-trim-reserve16-clean-20260801`
- `.cache/diagnostics/qwen36-pool-pressure-telemetry-20260801`
- `.cache/diagnostics/qwen36-pool-ordered-warm-reserve8-20260801`
- `.cache/diagnostics/qwen36-pool-warm-div6-20260801`
- `.cache/diagnostics/qwen36-pool-trim-request-end-only-20260801`

### Rejected: bounded CUDA pool release threshold

Setting the default pool's `cudaMemPoolAttrReleaseThreshold` to 85% of device
memory, without any explicit synchronization or `cudaMemPoolTrimTo`, did not
provide a safe middle ground. Qwen correctness passed, but the three retained
p2048/g128 generation rows were only `60.42`, `59.63`, and `58.68 tok/s`, far
below the accepted `150.75 tok/s` median. Telemetry also showed why this would
not solve Laguna: frees queued on MLX's dedicated free stream did not retire at
ordinary request synchronization, so reservation remained at about `32.6 GB`
despite the `29.1 GB` release threshold.

The release threshold can still affect hidden synchronization points enough to
destroy Qwen's retained fast state while failing to bound reservation at the
points that matter. The environment-gated allocator experiment and telemetry
were removed completely; `allocator.cpp` is back to its prior clean state.

- `.cache/diagnostics/qwen36-pool-release85-20260801`

### PTX cache contamination masquerading as a Qwen regression

After all allocator experiments were removed, byte-identical Ollama and MLX
libraries still produced only about `45 tok/s` on Qwen, versus the accepted
`150.75 tok/s`. There were no stale processes, GPU residency returned to the
normal idle level, and the request peak-memory sequence exactly matched the
known-good artifact. A fresh isolated `MLX_PTX_CACHE_DIR` restored correctness
and retained generation to `146.57`, `144.49`, and `144.40 tok/s`, confirming
that source performance had not been lost.

MLX's disk cache reads PTX by module name and does not hash generated source or
included headers. The worktree helper had keyed its shared cache only by
installed `libmlx.so`. Same-name QMM modules in that directory were rewritten
after the known-good Qwen run. Their recorded top-level `.cu` files were
byte-identical to the fresh cache, but PTX hashes differed because the source
includes installed CUDA headers whose provenance was absent from the key.
Go-side custom-kernel source was another omitted input.

The benchmark helper now keys PTX by the Ollama binary, `libmlx`, `libmlxc`,
NVRTC, the complete installed MLX header tree, and compute capability. Treat
the old library-hash-only namespace as contaminated. Every candidate must use
an isolated or complete-provenance namespace; after restoring source, validate
once with a fresh cache before diagnosing a byte-identical performance
regression as a code problem.

- `.cache/diagnostics/qwen36-clean-allocator-control-20260801`
- `.cache/diagnostics/qwen36-clean-ptx-control-20260801`
- `.cache/ptx/9b5addb72ec586e1cc470f6660801828868729562fbbcfc65452cca5ec471ace`

### DGX Spark Nemotron direct-QMV boundary

The retained `dhiltgen/nemotron-3-nano:4b-nvfp4` p2048/g128 result on DGX
Spark is `4686.91 / 68.66 tok/s`, or `120.6% / 87.3%` of the llama-server
target. A decode profile attributes about 88% of GPU time to exact W4A16 and
W8A16 direct QMV: 50.4% NVFP4 and 37.9% MXFP8. The MXFP8 vocabulary head moves
about 424 MB per token and reaches about 221 GB/s against GB10's 273 GB/s
theoretical bandwidth.

The published artifact inventory contains packed weights and per-group scales,
but no retained `input_scale` or `input_global_scale` tensors. Native
block-scaled FP4/FP8 MMA would therefore introduce activation quantization and
change the model's W4A16/W8A16 contract. Prior native-MMA probes measured that
quality change and did not improve M=1 latency. Do not use W4A4/W8A8 merely to
claim parity.

Two additional SM121-only experiments were correctness-gated and rejected at
the primitive-profile stage:

- Streaming-cache 128-bit weight loads changed NVFP4 `183.62 -> 188.51 us`
  and left the MXFP8 head at `1.92 ms`.
- Four warps per direct-QMV block changed NVFP4 `183.62 -> 186.88 us` and the
  MXFP8 head `1.92 -> 1.93 ms`; register allocation left theoretical occupancy
  unchanged.

The accepted source and candidate payload were restored after each test. These
results, combined with the earlier row sharing, split-K, prefetch, cache-policy,
load-width, and tensor-core rejections, mean the remaining Spark Nemotron gap
is not an untried launch/cache knob. It is dominated by exact mixed-precision
weight bandwidth, including an 8-bit vocabulary representation competing with
a 4-bit llama-server target. Preserve this row as an honest below-target
boundary while work continues on other actionable rows.

- `.cache/patches/mlx-sm121-direct-qmv-streaming-load-rejected.patch`
- `.cache/patches/mlx-sm121-direct-qmv-rows4-rejected.patch`

### Rejected: Blackwell cuDNN wide-head decode

A CUDA-backend-only experiment allowed cuDNN SDPA to handle D256/D512 only
for single-token decode on Blackwell. Wide prefill remained on MLX's retained
kernel, avoiding the earlier cuDNN workspace spike. Gemma4 12B passed
`TestBasic/blue-sky` and an independent long-context semantic generation.

The matched p2048/g64 trace rejected the library path before endpoint timing.
cuDNN's D256 kernel took `39.16 ms` across 2,600 calls and D512 took
`7.53 ms` across 520 calls, or `46.69 ms` total. At the same call counts, the
retained two-pass MLX per-call timings scale to `41.75 ms`; cuDNN was about
`11.8%` slower. Keep the D128 cuDNN ceiling and pursue a lower-resource tiled
MMA design only with primitive evidence.

- `.cache/patches/mlx-cudnn-wide-decode-blackwell-rejected.patch`
- `.cache/profiles/gemma4-12b-cudnn-wide-decode-v1-p2048g64-20260801`

### Rejected: cancel Gemma4 Q/K/V ModelOpt scale through RMSNorm

A CUDA-decode-only model rewrite removed each positive scalar ModelOpt Q/K/V
projection scale and divided the following RMSNorm epsilon by scale squared.
The algebra is valid for positive scales and the implementation left Metal and
prefill untouched. Quality checks were unusually strong: 40/40 selected-token
and raw-top-1 agreement, 8/8 exact completions, mean top-5 Jaccard `0.9833`,
and only `0.004229` mean selected-token logprob drift. An independent long
prompt produced byte-identical text and token context.

The matched p16/g128 Nsight trace initially looked compelling. Instrumented
generation improved from `27.15` to `38.51 tok/s`; graph-update time fell from
`2.505` to `1.645 s`, copy launches fell from `100,052` to `64,964`, and binary
launches fell from `49,908` to `32,364`. That result was misleading because
profiler-amplified graph/API overhead hid a slower real kernel path.

Three isolated p2048/g128 candidate processes measured only:

- `2652.07 / 86.96 tok/s`
- `2673.27 / 88.49 tok/s`
- `2575.74 / 88.59 tok/s`

The hash-matched restored control measured `2553.61 / 101.93 tok/s` in the
same fresh-process harness. The candidate's controller estimate for plain
decode also fell from about `130` to `117 tok/s`. Clearing `GlobalScale` and
`NativeScale` switched Q/K/V from the native-global-scale QQMatmul
specialization to its no-global-scale variant. Its graph was smaller, but the
real decode path was materially slower. The rewrite was removed and must not
replace the canonical Gemma4 row.

The ignored fresh-run helper now keys MLX PTX by the caller binary, `libmlx`,
`libmlxc`, NVRTC, installed headers, and compute capability, matching the main
row helper and preventing this A/B from sharing stale generated PTX.

- `.cache/patches/gemma4-qkv-scale-cancel-v1.patch`
- `.cache/quality/gemma4-qkv-scale-cancel-v1/comparison.json`
- `.cache/profiles/gemma4-12b-qkv-scale-cancel-v1-plain-chat-p16g128-5090-20260801`
- `.cache/bench/gemma4-12b-qkv-scale-cancel-v1-fresh-p2048g128-5090-20260801`
- `.cache/bench/gemma4-12b-restored-control-fresh-p2048g128-5090-20260801`

## Required transition after full parity

When every retained p2048/g128 comparison row is at or above 100% of its
llama-server target, stop performance experimentation and transition directly
to a methodical code-review and cleanup pass. Do not wait for another prompt.
Continue tuning only while at least one valid comparison row remains below
target.

The review gate is:

1. Read `~/Documents/Agents/MLX-conventions.md` again and inventory every
   retained MLX change by subsystem and dependency direction.
2. Confirm that the MLX work is restricted to the CUDA backend, CUDA kernels,
   CUDA dispatch, and backend-private tests. In particular, check for changes
   to core MLX behavior, cross-backend abstractions, public headers, exported
   APIs, Python bindings, and frontend semantics.
3. If any retained performance work changed MLX core or a public API, do not
   clean up, justify, or hide it. Stop and report the exact files, API surface,
   performance dependency, and why the change appeared necessary so Daniel
   can decide whether that scope is acceptable.
4. If the work stayed CUDA-backend-only, review and fix convention violations,
   dispatch structure, naming, formatting, comments, capability/shape guards,
   error handling, tests, and dead experimental code. Keep successful work in
   the index and risky cleanup experiments in the worktree so each refactor can
   be reverted without disturbing the known-good base.
5. Audit each custom kernel against NVIDIA's high-level libraries and APIs,
   especially cuBLAS/cuBLASLt, cuDNN frontend, CUB, and CUTLASS/CuTe. Prefer
   those facilities where they express the operation and meet performance and
   correctness requirements. For retained complex low-level kernels, add a
   brief local comment only where it is not otherwise clear why a suitable
   high-level CUDA API could not provide the required shape, datatype,
   semantics, or measured performance. Use the experiment records in this
   file rather than re-running rejected approaches merely to reconstruct the
   rationale.
6. Work through risky structural cleanup first in small batches. After each
   batch, run focused exact-output tests, full Ollama model correctness, and the
   affected host/model p2048/g128 benchmark. Reject cleanup that loses output
   correctness or a material part of the measured performance gain.
7. Finish with a complete three-host correctness and performance matrix, a
   clean diff and conventions review, and a proposed upstream decomposition.
   Do not push or open PRs. Present the ready state for Daniel's review before
   any maintainer engagement.
8. Produce a separate NVIDIA high-level API gap catalog for the retained
   custom kernels. For each candidate gap, identify the closest existing API
   in cuBLAS/cuBLASLt, cuDNN frontend, CUB, or CUTLASS/CuTe; state the exact
   missing datatype, layout, shape, grouping, fusion, graph, or execution
   semantic; link it to the model operation and measured performance cost;
   explain why the available API could not be used; and propose the smallest
   logically consistent API extension that would have avoided custom kernel
   code. Distinguish missing capability from an API path that existed but was
   slower, and retain benchmark/profiler artifact references so the feedback
   can be shared directly with NVIDIA contacts without repeating discovery.

The purpose of this gate is not merely formatting. The resulting CUDA work
should look maintainable and native to MLX, make its library-versus-custom
kernel choices defensible, preserve model output quality, and retain the
measured RTX 5090, DGX Spark, and N1x gains.
### Rejected: asymmetric cost floor and trusted-frontier MTP selection

An Ollama-only controller experiment attempted to keep cold/JIT timing from
poisoning speculative-depth selection by immediately replacing a cost estimate
with any lower sample while continuing to clamp higher samples. It also limited
the persistent EV choice to acceptance-tested depths and used periodic probes for
the next untrusted depth.

The exact candidate passed the blue-sky integration check (`Tokyo`) and an
independent coherent long response. Its correctness-gated RTX 5090 p2048/g128
rows were 2,838.95/107.63, 3,088.38/111.78, and 2,887.02/108.42 tok/s. The
generation median improved over the canonical 102.56 tok/s, but the mechanism
was not sound enough to retain:

- the lower-envelope estimator reported 6-7 ms depth-0 forward cost while the
  measured end-to-end plain decode remained about 9.2 ms/token, so the fastest
  isolated sample was not a credible steady-state estimator;
- after warmup, the controller almost entirely parked MTP at depth 0 rather than
  making the target forward faster;
- selecting only through the trusted frontier can permanently miss a profitable
  depth 2 when depth 1 alone loses but the depth-2 forward has nearly the same
  cost. This is a real shape for a bandwidth-bound target forward.

The source and tests were restored to the accepted controller. Do not promote
the 108.42 tok/s result into the canonical table or retry the lower-envelope
cost estimator. Any future controller work must preserve outward discovery when
an intermediate depth is not independently optimal and must correlate its cost
estimate with end-to-end decode time.

Artifacts:

- `.cache/correctness/gemma4-12b-controller-trusted-frontier-v2-20260801`
- `.cache/bench/gemma4-12b-controller-trusted-frontier-v2-p2048g128-5090-20260801`

### Accepted: Gemma4 12B row-1 NVFP4 tensor-scale QMV

Gemma4 12B ModelOpt decode followed nearly every direct NVFP4 QMV with a
BF16-to-FP32 copy, scalar multiply, and FP32-to-BF16 copy for the checkpoint's
tensor global scale. A CUDA custom kernel now applies that direct scale after
the QMV's required BF16 rounding. The row-1 output is bit-exact with the old
graph path. The dispatch is deliberately limited to one input row: MLX's
multirow QMV reuses activations across output rows and must remain responsible
for MTP depths above zero.

The retained kernel uses the same CUTLASS `NumericArrayConverter` and E4M3
scale type as MLX's CUDA QMV. MLX custom-kernel compilation predefines the C
math `FP_*` classification macros, so the source undefines those five macros
immediately before including CUTLASS's NVRTC compatibility header. No MLX core
or public API changes are required.

The iteration history explains the narrow shape guard:

- The first vectorized all-row kernel reached an interim clean median of
  `2951.40 / 110.09 tok/s`, but its one-warp-per-row MTP path was slower than
  MLX's row-reuse QMV.
- Replacing the conversion helper with CUTLASS while retaining all rows
  produced only `2915.93 / 86.33 tok/s`; this all-row variant is rejected.
- Restricting the manual converter to row 1 produced `2905.03 / 95.61 tok/s`,
  although one epoch reached `125.47 tok/s`. This confirmed that MLX's
  multirow path could cross target while the slower row-1 graph destabilized
  controller costs.
- Combining CUTLASS conversion for row 1 with MLX fallback for every multirow
  input produced clean p2048/g128 epochs of `2847.79 / 89.42`,
  `3113.38 / 124.83`, and `2920.09 / 135.81 tok/s`. The medians are
  `2920.09 / 124.83 tok/s`, or `149.6% / 108.6%` of the retained
  `1951.58 / 114.92` llama-server target.

This result passed the exact primitive test, blue-sky integration test, direct
coherent generation, and the independent 2048-token long-prompt gate. Promote
the row-1 CUTLASS variant and its median to the canonical RTX 5090 table.

- `.cache/correctness/gemma4-12b-nvfp4-scaled-qmv-cutlass-row1-v6-20260801`
- `.cache/bench/gemma4-12b-nvfp4-scaled-qmv-cutlass-row1-v6-p2048g128-5090-20260801`
- `.cache/profiles/gemma4-12b-nvfp4-scaled-qmv-vector-v3-plain-p16g128-nsys2026-20260801`

### Accepted: coherent Qwen 3.6 control after Gemma row-1 QMV

The Gemma-only row-1 NVFP4 tensor-scale QMV does not regress Qwen 3.6. A clean
Qwen run initially appeared to fall to about `87 tok/s`, but that artifact was
invalid as a candidate comparison for two independent provenance reasons:

- it used the stale `ollama-4784-local-mlx-cuda13-sm120a` payload reporting
  MLX `0.32.0-22-gdb0bdc2-dirty`, rather than the accepted shared payload
  reporting MLX `0.32.1`;
- the generic helper issued a 3,844-token calibration request instead of using
  the retained `2.44 tokens/word` estimate, perturbing memory residency before
  the measured cohort.

The incremental Go helper also exposed a CMake sharp edge: its build tree had
been reconfigured with the temporary `qkv-scale` install prefix, so an install
updated that experiment directory instead of the canonical payload. The helper
now refuses a mismatched `CMAKE_INSTALL_PREFIX` and directs callers to the full
reconfigure helper.

After reconfiguring the canonical prefix, the tested payload combined Ollama
binary `23cee4ba348e6698617697943365b044b7c6c56ff1d4aa721ab028dee5dd2b8a`
with accepted `libmlx.so`
`9b5addb72ec586e1cc470f6660801828868729562fbbcfc65452cca5ec471ace`.
Blue-sky and independent long-prompt correctness passed. The standard one
warmup, two scrub, three retained p2048/g128 rows were:

- `5253.05 / 162.16 tok/s`
- `4376.51 / 155.52 tok/s`
- `4316.72 / 154.08 tok/s`

The medians are `4376.51 / 155.52 tok/s`, or `535.6% / 74.6%` of the retained
`817.14 / 208.37` llama-server target. Promote this over the prior
`3273.53 / 150.75` MLX result. The remaining gap is still decode-only.

- `.cache/correctness/qwen36-v6-mlx0321-coherent-20260801`
- `.cache/bench/qwen36-v6-mlx0321-coherent-p2048g128-5090-20260801`

### Rejected: Qwen BF16 small-output GEMV library and split-K paths

The accepted Qwen decode trace attributes about 60 BF16 GEMVs per token to
the two `N=32,K=2048` B/A projections in each recurrent layer. MLX's existing
kernel assigns eight output rows to one CTA, so this production shape launches
only four CTAs on the 170-SM RTX 5090.

The high-level CUDA option was tested first. Bypassing MLX GEMV only for the
bias-free BF16 `M=1,N=32,K=2048` shape routed it through the existing
cuBLASLt implementation. Varied-data output hashes were identical, but the
matched 1,000-iteration wall time increased from `36.53 us` to `53.03 us`, a
45% regression. cuBLASLt has the required mathematical capability but is not a
viable performance replacement at this skinny shape.

A CUDA-backend-only cooperative GEMV then retained MLX's vectorized loads and
FP32 accumulation while distributing each output row across multiple warps.
Every schedule produced the same BF16 output hashes as the accepted kernel at
`N=8`, `N=32`, and `N=256`, all with `K=2048`. The matched wall-time sweep was:

| Schedule | N=8 | N=32 | N=256 |
| --- | ---: | ---: | ---: |
| Accepted 8 rows x 1 warp | 42.91 us | 36.53 us | 48.84 us |
| 1 row x 8 warps | 40.84 us | 39.84 us | 45.19 us |
| 1 row x 4 warps | 43.81 us | 42.38 us | 64.72 us |
| 4 rows x 2 warps | 44.64 us | 38.43 us | 47.67 us |
| 2 rows x 4 warps | 70.33 us | 42.55 us | 42.94 us |

No candidate beat the accepted kernel at the dominant `N=32` shape. The
extra CTAs and shared-memory cross-warp reduction do not recover enough work
to offset their overhead. Reject this kernel family before model-level
benchmarking and leave the canonical Qwen result unchanged.

An attempted Nsight Systems capture completed the workload but did not produce
a trustworthy kernel CSV: the Windows-installed Linux target split its
space-containing `LD_PRELOAD` paths and its launcher failed to exit. Do not use
that incomplete capture as kernel-level evidence.

- `.cache/patches/mlx-cublaslt-bf16-n32-k2048.patch`
- `.cache/patches/mlx-bf16-gemv-cooperative-rejected.patch`
- `.cache/bench/gemv-cublaslt-n32-5090-20260801`

### Rejected: Qwen full-attention QKV packing

Qwen 3.6 uses full attention in about one quarter of its layers. Packing each
full-attention Q/K/V projection into one quantized projection was tested as a
way to remove two launches per layer without changing arithmetic. Focused
projection-order tests, blue-sky integration coverage, direct chat, and an
independent natural-text p2048 request all remained correct.

The p2048/g128 results exposed incompatible prefill and decode requirements:

| Candidate | Prompt median | Generate median | Model bytes |
| --- | ---: | ---: | ---: |
| Accepted separate Q/K/V | 4376.51 | 155.52 | baseline |
| Packed QKV for all shapes | 4980.60 | 153.23 | 21210135592 |
| Packed prefill plus duplicate decode weights | 4778.99 | 89.66 | 21315972680 |
| Packed storage plus per-row-metadata views | 4885.73 | 88.90 | 21210443848 |
| Shared views with original scalar metadata | 4922.60 | 46.62 | 21210451528 |
| Evaluated shared views with scalar metadata | 4857.46 | 45.87 | 21210951248 |

Packing all shapes improved prompt throughput but slightly regressed decode.
Keeping both representations added about 106 MB and crossed the fast decode
graph's residency boundary. Contiguous MLX row slices share storage, but
persisting them as model parameters still disrupted the decode graph: packed
per-row scales missed the scalar tensor-scale QMV, while restoring scalar
metadata exposed repeated parameter-view graph costs even when the slices were
evaluated once at load.

Reject persistent QKV packing. The experiment source was restored exactly to
the saved pre-experiment patch ID `a0938a0e0a32fba908b2e7681ef4759a0038b3f6`,
and the canonical installed Ollama binary returned to
`23cee4ba348e6698617697943365b044b7c6c56ff1d4aa721ab028dee5dd2b8a`.
Do not retry this path unless MLX gains a native packed-projection primitive
that can expose stable subviews to decode graphs without extra residency or
parameter graph nodes.

- `.cache/patches/qwen-active-before-full-attn-qkv-pack.patch`
- `.cache/bench/qwen36-full-attn-qkv-pack-p2048g128-5090-20260802`
- `.cache/bench/qwen36-prefill-qkv-pack-p2048g128-5090-20260802`
- `.cache/bench/qwen36-qkv-zero-copy-p2048g128-5090-20260802`
- `.cache/bench/qwen36-qkv-shared-scalar-p2048g128-5090-20260802`
- `.cache/bench/qwen36-qkv-evaluated-views-p2048g128-5090-20260802`
### Rejected: gathered NVFP4 QMV thread-block clusters

- The decode-hot gathered NVFP4 QMV repeatedly reads the same 4 KiB BF16
  activation vector across many output-row CTAs. A standalone SM120a probe used
  CUDA thread-block clusters and distributed shared memory to stage that vector
  once for clusters of 2, 4, or 8 CTAs.
- The exact W4A16 arithmetic remained bit-exact after adding the required final
  `cluster.sync()` that keeps rank 0's distributed shared-memory allocation
  alive until all remote readers finish.
- RTX 5090 isolated timings were `14.353 us` for the accepted control kernel,
  `206.083 us` for cluster 2, `495.731 us` for cluster 4, and `1356.475 us` for
  cluster 8. Cluster launch, synchronization, and remote shared-memory costs
  overwhelm any saved activation reads.
- Do not integrate or retry CTA clusters for this small activation-reuse case.
  Probe: `.cache/scripts/nvfp4_gather_cluster_qmv_probe.cu`.

### Rejected: gathered NVFP4 QMV eight packed words per lane

- The accepted fast gathered QMV processes four packed FP4 words (32 values)
  per lane. For Qwen's hot `experts=8, N=1024, K=2048` gate/up shape, an
  eight-word variant was tested to cover K in one loop pass instead of two.
- Both variants used two output rows per warp and exact W4A16 accumulation. The
  eight-word output was bit-exact across all 8,192 output values.
- Ptxas reported no spills: 56 registers for four words and 40 registers for
  eight words. Nevertheless, five RTX 5090 trials measured `14.560-15.618 us`
  for four words versus `24.704-25.544 us` for eight words (`0.571-0.615x`).
- The wider per-lane memory/conversion dependency chain costs much more than the
  removed loop pass. Do not add an `n_per_thread=8` dispatch to MLX.
  Probe: `.cache/scripts/nvfp4_gather_words_qmv_probe.cu`.

### Rejected: gathered NVFP4 QMV CTA-local activation staging

- The accepted four-warp, two-rows-per-warp gathered QMV reloads the same BF16
  activation in each warp. A CUDA-only `K=2048`, BF16, NVFP4 group-16 candidate
  copied the 4 KiB activation once into CTA shared memory with 128-bit loads,
  then reused the unchanged exact-W4A16 fast implementation.
- The standalone `experts=8,N=1024,K=2048` probe was bit-exact and improved
  RTX 5090 latency from `14.384-14.648 us` to `8.745-11.136 us`
  (`1.315-1.651x`). Focused MLX gathered-QMV equality, the Qwen CUDA adapter
  tests, `TestBasic/blue-sky`, and an independent natural long-prompt response
  all passed.
- The canonical one-warmup, two-scrub, three-retained p2048/g128 endpoint gate
  strongly rejected it: prompt median fell from the accepted `4376.51` to
  `3514.88 tok/s`, and generation fell from `155.52` to `38.72 tok/s`.
- Adding 4 KiB shared memory and one barrier to every hot gathered node changes
  CUDA graph resource residency/concurrency enough to overwhelm the isolated
  kernel win. Do not retry activation staging without a graph-level design that
  preserves the accepted overlap.
- A follow-up used exactly two `cp.async` copies per thread plus
  `__launch_bounds__(128, 8)`. The installed specialization used 62 registers,
  no local memory, and 5,120 bytes of static shared memory; its standalone
  latency was `8.343-8.775 us` (`1.624-1.743x`). This removed the collapse but
  remained endpoint-neutral: the retained medians were `4500.78 / 154.79`
  tok/s, or `+2.8% / -0.5%` versus accepted. The graph already hides almost all
  of the primitive saving, so the added specialization is not worth retaining.
- Artifacts:
  `.cache/scripts/nvfp4_gather_cluster_qmv_probe.cu` and
  `.cache/bench/qwen36-gather-shared-activation-v1-clean-p2048g128-5090-20260802`,
  `.cache/bench/qwen36-gather-shared-async-lb-v2-p2048g128-5090-20260802`.

### N1x async-pool capacity and invalid low-clock benchmark session

The N1x load failure was reduced to a CUDA allocation-capacity mismatch rather
than an Ollama model-load defect. The pre-release WoA driver reports:

- `cudaMemGetInfo`: about 40 GiB total;
- system/NVML device reporting: about 8 GiB total;
- default async pool allocation failure near 7.35 GiB;
- successful 10-12 GiB `cudaMalloc`, `cudaMallocManaged`, and custom-pool
  allocations.

A direct 12 GiB copy probe measured about 119 GiB/s for both `cudaMalloc` and a
CUDA 13.3+ custom pool with `cudaMemPoolProps.maxSize`, versus about 76 GiB/s
for managed memory. This rules out managed memory as the preferred fallback.
CUDA documents that `cudaFreeAsync` can release both legacy and pool-backed
allocations, so allocation-origin bookkeeping is not required.

The following allocator shapes all loaded Qwen3.6 27B correctly, but none may
be promoted based on this session's endpoint timings:

- legacy `cudaMalloc` for all allocations;
- default async pool with legacy fallback after OOM;
- one runtime-sized custom pool;
- eager default plus overflow pools;
- original `cudaMallocAsync` with a lazily created overflow pool after OOM.

The experiments exposed and fixed an iteration sharp edge. Allocator overlays
now preserve timestamps for byte-identical inputs, so a `.cpp` refinement
builds and installs in about 12 seconds instead of touching `allocator.h` and
rebuilding 120 CUDA dependents. The lazy-overflow diff is retained at
`.cache/patches/n1x-lazy-overflow-pool-20260802.patch`; local MLX source was
restored to the accepted allocator after the experiment.

Do not compare the endpoint numbers from these allocator runs. A same-source
accepted allocator control measured Qwen3.5 4B at only about 368/18 tok/s, and
a Qwen3.5 2B control measured about 980/40 tok/s versus its trusted
2367/42 high-water. Live telemetry showed 80-96% GPU utilization but only
0.72-0.76 GHz SM clocks and 5-7 W. Earlier trusted N1x runs recorded
1.24-1.39 GHz under load. Prompt throughput fell system-wide while the smaller
model's memory-bound decode remained nearly unchanged.

AC was already configured for Windows Best Performance; setting the DC vote
to Best Performance did not restore clocks. WoA `nvidia-smi` reports GPU clock
locking/reset as unsupported, and this build lacks a usable GPU-reset option.
The low-clock state likely followed force-terminated Nsight Graphics captures.
Reboot the N1x before further performance work, verify loaded SM clocks return
to the documented range, and retain the old correctness-gated high-water rows
until a valid same-clock run beats them.

### Rejected: CUDA all-in-one Qwen gated-delta kernel

A CUDA implementation of the existing Metal all-in-one gated-delta custom
kernel fused activated Q/K/V normalization, decay/beta preparation, and the
FP32 recurrent scan. The model used the shared `CausalConv1D+SiLU` and
`nn.GatedDelta` route rather than the accepted CUDA-only preprocessing plus
recurrence detour. The actual Qwen decode geometry matched the graph reference
bit-for-bit, and every candidate passed the focused CUDA tests,
`TestBasic/blue-sky`, direct chat, and an independent natural p2048 response.

The matched p16/g128 profile showed a real isolated saving. Accepted input plus
recurrence kernels cost `8.00 + 54.16 = 62.16 ms`; the all-in-one `DvTile=32`
kernel cost `48.86 ms`. It used 56 registers/thread and 516 bytes of static
shared memory, versus 35 registers for preprocessing and 39 for recurrence.
Despite fewer launches and `13.30 ms` less kernel work, `cudaGraphExecUpdate`
increased from `1279.25` to `1337.14 ms`, erasing the kernel improvement.

The correctness-gated one-warmup, two-scrub, three-retained p2048/g128 sweep
against the accepted `4376.51 / 155.52 tok/s` high-water was:

| CUDA value tile | Prompt median | Generate median |
| ---: | ---: | ---: |
| 32 | 6581.41 | 128.65 |
| 16 | 6869.38 | 153.84 |
| 8 | 6393.24 | 146.25 |

Smaller tiles recovered graph concurrency but did not beat accepted generation;
tile 8 regressed again as duplicated Q/K preprocessing increased. Do not promote
this prompt-only tradeoff or retry tile 4, which duplicates preprocessing over
the full recurrence block count. The source is preserved at
`.cache/patches/qwen36-gated-delta-allinone-cuda-20260802.patch`; the matched
profile is
`.cache/profiles/qwen36-35b-a3b-nvfp4-gdn-allinone-v1-p16g128-20260802`.

### Rejected: NVFP4 Qwen vocabulary-head conversion

Qwen3.6 35B-A3B NVFP4 stores its dense `248320x2048` vocabulary head in BF16.
The retained CUDA loader converts it to MXFP8 because the exact W8A16 QMV is
already bandwidth-saturated. Converting that head to NVFP4 would halve its
per-token weight traffic, but a focused comparison against the BF16 logits
rejected the quality tradeoff before model integration:

| Runtime head quant | Relative RMSE | Mean KL | Argmax | Top-10 overlap |
| --- | ---: | ---: | ---: | ---: |
| MXFP8 | 2.657% | 0.000247 | 8/8 | 9.62/10 |
| NVFP4 | 10.368% | 0.003752 | 6/8 | 8.50/10 |

Do not use runtime NVFP4 head conversion to claim parity. It changes token
selection materially and would require a calibrated model-quality decision,
not a backend optimization. The reusable probe source is archived at
`.cache/diagnostics/source-archive-20260802/lm_head_quant_probe_test.go`; the
5090 helper is `.cache/scripts/run_qwen_lm_head_probe_5090.sh`. Restore the
probe into `x/models/qwen3_5/` only while rerunning that diagnostic; it is not
part of the retained production diff.

### Rejected: read-only activation loads in gathered NVFP4 QMV

The fast gathered QMV repeatedly reads the same 4 KiB BF16 activation vector
across its output-row CTAs. A Blackwell-only exact-math candidate loaded the
vector through CUDA's read-only cache path using 128-bit `__ldg` operations.
The production `experts=8,N=1024,K=2048` standalone probe was bit-exact and
improved from `14.64-15.67 us` to `8.81-9.96 us`, or `1.50-1.74x`.

Focused MLX gathered-QMV equality and the full Qwen correctness gate passed,
including direct chat and the independent natural p2048 response. The
canonical endpoint gate strongly rejected it:

| RTX 5090 Qwen3.6 35B-A3B NVFP4 | Prompt median | Generate median |
| --- | ---: | ---: |
| Accepted high-water | 4376.51 | 155.52 |
| Read-only activation candidate | 4779.72 | 59.26 |

The candidate and control BF16 kernels both compile to 62 registers/thread,
zero shared memory, and zero local memory. The collapse is therefore not a
simple occupancy change; routing these loads through the read-only path harms
cache behavior or overlap in the concurrent CUDA graph even though it wins in
isolation. Do not retry `__ldg` activation routing without graph-level evidence
that explains and avoids this interference.

Artifacts:

- `.cache/patches/mlx-fp-gather-qmv-readonly-rejected.patch`
- `.cache/bench/qwen36-gather-readonly-v1-p2048g128-5090-20260802`
- `.cache/correctness/qwen36-gather-readonly-v1-5090-20260802`

### Accepted: lane-zero gated-delta update scalar

The decode recurrence previously broadcast the reduced `kv_mem` and then had
all 32 lanes redundantly load `v` and `beta` and compute the same update
scalar. The retained CUDA custom kernel now computes `(v - kv_mem) * beta` in
lane zero and broadcasts that scalar. It also hoists the per-head decay load
out of the four-element state loop. The warp reduction order, graph shape,
launch geometry, register count (39), shared memory, and output contract are
unchanged.

The focused test was corrected after the first version accidentally exercised
the generic `GatedDelta` fallback. The final test invokes the exact exported
`GatedDeltaRecurrence` production path with a compact but structurally
equivalent `Dk=128`, four-value vector-load, repeated-head geometry. BF16
outputs are exact against the eager graph; FP32 state uses a strict `1e-6`
tolerance for the existing warp-reduction-order difference. The exact test
passes on RTX 5090 and DGX Spark, and full `TestBasic/blue-sky` model
correctness passes on both hosts.

On RTX 5090, an exact intended-kernel profile reduced 3,960 recurrence launches
from `54.161 ms` to `50.160 ms` total (`13.677` to `12.667 us` average), a
7.4% kernel saving. The canonical one-warmup, two-scrub, three-retained
p2048/g128 candidate median was `6571.23 / 158.65 tok/s`, versus the retained
endpoint high-water of `4376.51 / 155.52`; only the generation improvement is
material for this decode kernel.

DGX Spark required a strict clean-payload A/B after two provenance failures:
the first run used the dirty `mlx-nvidia-um-current` experiment, and the next
used a timestamp-preserved hybrid native build. Both are invalid. A guarded
full clean rebuilt all 229 MLX native targets, and hashes for all 14 staged MLX
CUDA sources matched the local accepted branch. Extracting the exact indexed
Ollama control (`git blob 77de88549`) without altering the working tree then
gave:

| DGX Spark clean payload | Prompt median | Generate median |
| --- | ---: | ---: |
| Exact indexed control | 1482.39 | 76.77 |
| Lane-zero candidate | 1488.39 | 77.50 |

The candidate is therefore about 0.4% faster for prompt and 1.0% faster for
generation on the matched DGX build, not the cause of the larger discrepancy
from the correctness-gated `2471.72 / 90.33` high-water. Keep that older row as
the canonical status result while separately reconstructing its accepted
build and benchmark state. Never promote the lower clean-build numbers or
attribute their difference to this kernel.

Artifacts:

- `.cache/scripts/gated_delta_decode_test.go`
- `.cache/scripts/run_gated_delta_test_5090.sh`
- `.cache/scripts/run_gated_delta_lane0_test_tater50.sh`
- `.cache/bench/qwen36-gdn-delta-lane0-v1-p2048g128-5090-20260802`
- `.cache/profiles/qwen36-35b-a3b-nvfp4-gdn-delta-lane0-v1-p16g128-20260802`
- DGX candidate: `.cache/bench/qwen36-gdn-delta-lane0-v1-p2048g128-tater50-20260802`
- DGX control: `.cache/bench/qwen36-gdn-delta-indexed-control-clean-p2048g128-tater50-20260802`

### Rejected: Metal-aligned gathered NVFP4 QMV schedule

Metal's fast FP gather QMV processes two packed FP4 words per lane and four
output rows per SIMD group. The CUDA experiments had tested two words and four
rows separately, but not that exact combination. A narrow CUDA candidate used
two packed words per lane, four rows per warp, and two warps per block only for
aligned NVFP4/group-16 gathers whose output width is divisible by eight. Direct
QMV and all other modes retained their existing dispatch.

The forced fast-versus-fallback reference was BF16 bit-exact, and the matched
RTX 5090 primitive A/B was strong:

| RTX 5090 focused path | Retained | Metal-aligned | Change |
| --- | ---: | ---: | ---: |
| Gather gate/up | 62.352 us | 46.605 us | 25.3% faster |
| Gather down | 67.262 us | 54.692 us | 18.7% faster |
| 40-layer MoE chain | 3.403 ms | 2.635 ms | 22.6% faster |

Full `TestBasic/blue-sky` and the independent natural p2048 response were
coherent, but the canonical endpoint gate rejected the schedule. The retained
rows were `5165.90 / 125.73`, `4304.47 / 133.84`, and
`4256.82 / 120.82 tok/s`, for medians of `4304.47 / 125.73` if sorted by each
metric independently (`133.84` is the best generation row, not the median).
The generation median is therefore `125.73 tok/s`, 20.8% below the accepted
lane-zero recurrence high-water of `158.65 tok/s`.

The shorter isolated chain again fails under the concurrent CUDA graph. The
two-warp CTA increases per-warp serial work and changes cross-node scheduling
enough to erase the primitive saving. Do not retry the Metal schedule as a
wholesale gathered-QMV dispatch without a graph-level design that preserves
the retained four-warp concurrency.

Artifacts:

- `.cache/patches/mlx-fp-gather-qmv-metal-schedule-rejected.patch`
- `.cache/bench/qwen36-gather-metal-schedule-v1-p2048g128-5090-20260802`

### Rejected: two-word gathered NVFP4 QMV with retained warp geometry

To isolate the prior Metal-schedule failure, a second candidate kept CUDA's
accepted two rows per warp and four warps per block while changing only the
aligned NVFP4/group-16 gathered path from four packed words per lane to two.
The forced fast/fallback reference remained BF16 bit-exact.

The matched focused A/B improved gate/up from `62.352` to `46.176 us`, down
from `67.262` to `50.388 us`, and the 40-layer MoE chain from `3.403` to
`2.139 ms` (37.2%). Full blue-sky and natural-prompt correctness also passed.
The canonical retained p2048/g128 rows were nevertheless:

- `5176.29 / 97.88 tok/s`
- `4197.51 / 105.49 tok/s`
- `4094.49 / 109.71 tok/s`

The medians are `4197.51 / 105.49 tok/s`; generation is 33.5% below the
accepted `158.65 tok/s` high-water. The narrower packed width therefore harms
the concurrent graph even when the accepted CTA geometry is preserved. Along
with the one-, two-, four-, and eight-word/row scheduling experiments already
recorded, this closes simple gathered-QMV packed-width and warp-layout tuning.

Artifacts:

- `.cache/patches/mlx-fp-gather-qmv-n2-rpw2-rejected.patch`
- `.cache/bench/qwen36-gather-n2-rpw2-v1-p2048g128-5090-20260802`

### Rejected: lane-zero gated-delta decay load

After retaining the lane-zero delta calculation, a follow-up had lane zero
load and convert the warp-uniform BF16 decay scalar and broadcast it with a
shuffle. The exact production-path recurrence test passed, and launch shape,
registers (39), shared memory, and graph shape were unchanged. The matched
RTX 5090 profile nevertheless regressed 3,960 recurrence launches from
`50.160 ms` to `51.522 ms` total (`12.667` to `13.011 us` average).

CUDA already handles the warp-uniform decay load efficiently; the explicit
shuffle adds latency. Reject before endpoint timing and retain the lane-zero
delta calculation only.

- `.cache/patches/ollama-gdn-decay-lane0-rejected.patch`
- `.cache/profiles/qwen36-35b-a3b-nvfp4-gdn-decay-lane0-v1-p16g128-20260802`

### Rejected: paired E2M1 conversion hoist in direct NVFP4 QMV

The generic one-warp-per-row direct NVFP4 QMV was tested with both packed
E2M1 words for each group converted before beginning the unchanged FMA
sequence. This preserves the accepted launch geometry, FP32 accumulation
order, and BF16 output exactly while exposing the two native unpack
instructions to the scheduler together. It is distinct from the earlier
K=2048 experiment, which changed only the SM120 async-staged specialization.

At Nemotron's dominant `N=17504,K=3136` shape, all 17,504 BF16 outputs were
bit-exact. DGX Spark improved only from `245.774` to `243.890 us` (`1.008x`),
and RTX 5090 was neutral at `61.752` versus `61.682 us` (`1.001x`). This is
far below the leverage needed for the DGX endpoint and does not justify a new
specialization or full-model rebuild. The direct path is limited by packed
weight memory latency/bandwidth rather than compiler serialization of the two
adjacent E2M1 conversions.

- Probe: `.cache/scripts/nvfp4_splitk_qmv_probe.cu`
- Runner: `.cache/scripts/run_nvfp4_splitk_qmv_probe_tater50.sh` (`ARCH` now
  selects `120a` or `121a` without creating host-specific source)

### Protocol correction: match MTP activity and acceptance

The 2026-08-02 audit found that the retained Gemma4 12B target is plain
llama-server decode, not MTP. The official v0.32.4 frontend and the current
frontend launching its unchanged `b10091` native payload both reported
`no implementations specified for speculative decoding` and zero draft state
for `gemma4:12b`, including with explicit `draft_num_predict=4`. The local GGUF
does not advertise `nextn_predict_layers` or carry a separate draft artifact.

The MLX package does load its Gemma4 draft head. On the RTX 5090, the adaptive
controller observed roughly 79% conditional acceptance but parked at depth 0:
steady depth-0 rounds were about 10-14 ms while depth 1 remained about 43 ms.
This supports the earlier conclusion that the remaining Gemma decode floor is
the base path rather than poor acceptance. The Qwen3.6 packages audited in the
same session contain no `mtp`, `draft`, or `nextn` tensors, so their gaps are
also unrelated to speculative acceptance.

For future headline MTP comparisons:

- use p2048/g128, the unique Python-patch prompt corpus, one warmup, and
  correctness before timing;
- explicitly enable MTP on both backends and require log evidence that each
  loaded a draft implementation;
- record accepted/drafted, accepted tokens per iteration, and chosen depth;
- retain a percent-of-target comparison only when acceptance cohorts are
  reasonably similar; use MTP-off on both sides only to diagnose the base
  model floor;
- do not call the current Gemma4 12B GGUF row MTP-matched until a target
  artifact containing the draft head is available.

Artifacts:

- `.cache/bench/gemma4-12b-matched-mtp4-mlx-info-5090-20260802`
- `.cache/bench/gemma4-12b-matched-mtp4-llama-current-front2-5090-20260802`

### N1x phase boundary after the post-profiler reboot

The post-reboot Qwen3.5 4B control passed semantic and long-prompt correctness,
ran at 95% GPU utilization with normal loaded clocks, and produced a one-epoch
`3801.07 / 70.46 tok/s` p2048/g128 control. Do not promote that single epoch
over the established canonical cohort, but use it as evidence that the reboot
cleared the prior whole-device low-clock state.

Gemma4 12B then filled the reported 8 GiB GPU aperture, MLX reported pageable
memory, and peak allocation reached 9.06 GiB in the fresh run (11.45 GiB in the
earlier full benchmark). Loaded clocks fell to 760 MHz at about 3.5 W while the
aperture was full. This is consistent with Windows paging/residency pressure or
pre-release N1x hardware/driver behavior; it is not evidence for changing a
shared CUDA kernel. The fresh slow `76.53 / 6.48 tok/s` result is rejected and
does not replace the canonical row.

For this phase, N1x parity means graph-enabled models that fit with headroom.
Qwen3.5 2B/4B NVFP4 and MXFP8 plus Gemma4 E2B NVFP4 and MXFP8 all exceed their
matched llama-server targets. Larger rows that exceed or press against the 8
GiB aperture are capacity/platform diagnostics rather than parity blockers.
Retry an unexplained N1x failure once; request another reboot only if an
unloaded, comfortably fitting control also falls below its trusted high-water.

### Rejected: gathered-QMV index lane broadcast

Nsight Compute on the exact production Qwen gathered NVFP4 QMV specialization
showed a dominant long-scoreboard sample at the first address calculation that
consumes `rhs_indices[g_idx.z]`. A candidate loaded both dynamic indices in
lane zero and broadcast them with warp shuffles instead of letting all lanes
issue the uniform global loads. Full blue-sky and independent natural-prompt
correctness passed.

The p16/g128 screen rejected the change before a canonical run. Retained
generation rows were `153.97`, `121.51`, and `110.05 tok/s`, for a
`121.51 tok/s` median versus the accepted `158.65 tok/s` canonical high-water.
The RTX 5090 reached only 51 C and reported no thermal throttling. CUDA already
coalesces the warp-uniform index access into one transaction; the shuffles add
latency to the dependent address chain without reducing memory transactions.

- `.cache/patches/mlx-fp-gather-qmv-index-broadcast-experiment.patch`
- `.cache/bench/qwen36-index-broadcast-v1-p16g128-5090-20260802`

### Rejected: exact CUDA graph executable reuse by pointer layout

The accepted Qwen trace has only about 52 ms more GPU kernel work than the
matched llama-server trace over 128 tokens, while the endpoint gap is about
190 ms. To test whether repeated pointer layouts could bypass whole-graph
updates, a temporary CUDA-backend diagnostic hashed each structural graph key
plus its ordered input/output device pointers. It did not alter graph launch,
update, or model math.

A complete 128-token Qwen decode observed 256 graph commits, 256 unique keys,
and zero exact recurrences. Caching an executable by complete pointer layout
would therefore consume graph memory without avoiding any update in this
workload. The diagnostic was removed; do not retry exact-executable caching
unless allocator behavior or graph parameter stability changes materially.

- `.cache/patches/mlx-cuda-graph-param-reuse-diagnostic.patch`
- `.cache/diagnostics/qwen36-graph-param-reuse-20260802`

### Rejected: device-updatable CUDA graph nodes

CUDA 13's device-side graph update API is not a surgical replacement for
`cudaGraphExecUpdate` in MLX. Kernel nodes must opt in with
`cudaLaunchAttributeDeviceUpdatableKernelNode`, after which the graph cannot
use `cudaGraphExecUpdate` or multiple instantiation. More importantly, MLX's
outer graphs contain cuDNN and other library child graphs; the outer command
encoder cannot convert the opaque nodes inside those child graphs into
device-updatable kernel nodes.

Using this API would require replacing library subgraphs and redesigning graph
construction rather than tuning the CUDA backend's existing update path. Do
not retry it as a Qwen optimization. Record this as an NVIDIA high-level API
gap: CUDA lacks an efficient batched host update for a mostly-changing graph
that contains explicit nodes plus opaque library child graphs, while the
device-side alternative is mutually exclusive with whole-graph update.

### Qwen decode synchronization attribution

Temporary environment-gated timers were restored around the current accepted
Qwen pipeline, then removed completely. On coherent warm-cache natural-prompt
runs (`156.96-165.28 tok/s`), representative 32-token windows measured:

- model forward graph construction: about `1.6-2.0 ms/token`;
- sweep: about `0.15-0.20 ms/token`;
- async evaluation/submission: about `3.7-5.2 ms/token`;
- token synchronization read: about `0.001 ms/token`;
- detokenization and response handoff: negligible.

Distinct-prompt pressure rows exposed a different state: forward rose to
roughly `2.2-3.5 ms/token`, async evaluation to `5-9 ms/token`, and the token
read waited `13-17 ms/token` for queued GPU work. The diagnostic itself
perturbed the async pipeline enough that those rows did not recover after the
usual scrubs, so its endpoint throughput is not benchmark evidence and must
not replace the accepted `158.65 tok/s` high-water. The attribution is still
useful: response/token plumbing is not the steady-state gap, and under memory
pressure the missing time is device completion rather than host token access.

After removing the timers, the rebuilt installed binary had SHA-256
`75d451efee47dda6787a8cdf89d6c460dc34f779bb39f4f7f03c0b68a9592496`,
byte-for-byte identical to the accepted high-water binary. No repeat benchmark
was needed. Diagnostic artifacts:

- `.cache/diagnostics/qwen36-current-pipeline-timing-warm-20260802`
- `.cache/diagnostics/qwen36-current-decode-loop-timing-20260802`
- `.cache/diagnostics/qwen36-current-decode-loop-timing-scrubbed-20260802`

### Rejected: persistent gathered-QMV row-tile loop

The exact production gathered NVFP4 QMV was tested with a Blackwell-only
grid-stride schedule. For top-8 MoE decode, each expert was capped at 64 CTAs
and each CTA processed any remaining output-row tiles in a loop. This reduced
the Qwen `N=1024` launch from 1,024 to 512 CTAs while preserving each row's
warp reduction and FP32 accumulation order exactly.

The full blue-sky integration check, direct Rayleigh-scattering response, and
independent natural p2048 response were coherent. The warmed p16/g128 retained
generation rows were `151.94`, `148.78`, and `157.91 tok/s`, for a
`151.94 tok/s` median versus the accepted `158.65 tok/s` canonical high-water.
The RTX 5090 reached only 52 C and reported no throttle reason. Amortizing the
dynamic expert-index setup does not compensate for the reduced CTA concurrency
in MLX's concurrent graph, so do not retry persistent row-tile scheduling
without a design that adds intra-CTA latency hiding rather than serial work.

- `.cache/patches/mlx-fp-gather-qmv-persistent64-rejected.patch`
- `.cache/correctness/qwen36-gather-persistent64-v1-20260802`
- `.cache/bench/qwen36-gather-persistent64-v1-p16g128-20260802`

### Rejected: cached CUDA linear weight-transpose views

Every unquantized `nn.Linear.Forward` creates a transpose primitive for its
weight. A CUDA-only experiment cached that no-copy transpose view on the layer,
allowing the runner's normal model collection to pin and reuse it after the
first evaluation. Metal retained the original forward path, and the native
CUDA payload was the accepted hash.

The full semantic gate passed, but persistent parameter views again damaged
the concurrent decode graph. The independent natural p2048 request generated
coherent text at `133.72 tok/s`. The warmed p16/g128 retained generation rows
were `142.83`, `141.97`, and `144.95 tok/s`, for a `142.83 tok/s` median versus
the accepted `158.65 tok/s` high-water. The GPU reached only 52 C with no
throttle reason. Avoid persistent transpose/view fields as a graph-construction
shortcut; they alter graph inputs and scheduling even when they share the
original buffer and consume no copied weight storage.

- `.cache/patches/ollama-cached-linear-transpose-rejected.patch`
- `.cache/correctness/qwen36-cached-linear-transpose-v1-20260802`
- `.cache/bench/qwen36-cached-linear-transpose-v1-p16g128-20260802`

### Rejected: 900-operation CUDA graph cap

The accepted Blackwell consumer policy uses an 800-operation, 8,000 MB graph
cap. Earlier sweeps jumped directly from 800 to 1,200 operations, so the
current accepted binary was screened at 900 operations with the memory cap
unchanged. The warmed p16/g128 retained generation rows were `151.11`,
`158.21`, and `152.92 tok/s`, for a `152.92 tok/s` median versus the accepted
`158.65 tok/s` high-water. The larger graph cap is already counterproductive;
do not test 1,000 operations without new evidence that changes graph topology
or update cost.

- `.cache/bench/qwen36-graph900-v1-p16g128-20260802`

### Rejected: eliminate repeated sparse-layer shape queries

Qwen's existing `B` and `L` values were passed through the MLP interface to
remove two `Dims` calls, or eight MLX-C metadata crossings, per sparse layer.
The operators, shapes, and CUDA payload were otherwise unchanged. Semantic
correctness remained coherent, but the warm graph state was substantially
less stable: retained p16/g128 generation was `111.48`, `122.95`, and
`150.45 tok/s`, never recovering to the accepted `158.65 tok/s` high-water
after two scrub rows. The change was fully removed. Do not churn the model
interfaces to eliminate these cheap shape queries without a direct host
profile showing they have become material.

- `.cache/bench/qwen36-shape-metadata-v1-p16g128-20260802`

### Rejected: native cuDNN graph population for convolution

Qwen executes 30 cached cuDNN width-4 depthwise convolutions per decode token.
MLX's cuDNN SDPA path uses `DnnGraph::encode_graph` to retain a child graph and
patch its pointers, so both convolution call sites were tested with that same
existing method instead of rebuilding the child through stream capture. The
incremental native build succeeded, but the first correctness preload failed
before producing output: cuDNN returned
`CUDNN_STATUS_NOT_SUPPORTED_CUDA_GRAPH_NATIVE_API` from
`populate_cuda_graph`. The convolution plan supports stream capture but not
cuDNN's native graph-population/update API. The two-line experiment was fully
removed. Any follow-up must retain stream capture or use a backend-native
kernel; do not retry the SDPA graph path for this convolution plan.

- `.cache/bench/qwen36-cudnn-conv-graph-v1-p16g128-20260802`

### Rejected: direct width-4 depthwise-convolution kernels

The Qwen decode trace contains 30 BF16 width-4 depthwise convolutions per
token. The accepted cuDNN plan executes `conv1d_c1_k1_nhwc` with a 512-thread
block and about 512 blocks, but must enter the outer CUDA graph through stream
capture because cuDNN rejects native graph population for this plan.

Two CUDA-backend-only kernels tested whether removing that capture boundary
would improve decode. Both used one FP32-accumulated four-element dot product
per channel, BF16 output, and the same 512-thread/roughly-512-block launch shape
as the cuDNN kernel. The first was inserted as an ordinary kernel node; the
second retained child-graph topology with a persistent one-node child graph
whose pointers were updated with `cudaGraphKernelNodeSetParams`.

Both variants passed the semantic correctness gate and produced coherent
independent long-prompt output. The direct-node variant retained generation
rows of `137.24`, `148.28`, and `148.19 tok/s`, for a `148.19 tok/s` median.
The child-graph variant retained `158.07`, `167.52`, and `155.84 tok/s`, for a
`158.07 tok/s` median versus the accepted `158.65 tok/s` high-water. The latter
was thermally clean at 52 C with no throttle reason, so neither implementation
demonstrated a real gain. The experiment was removed completely; do not retry
replacing this cuDNN plan without a materially better convolution algorithm or
new cuDNN graph support.

- `.cache/patches/mlx-depthwise-conv-child-graph-rejected.patch`
- `.cache/bench/qwen36-depthwise-conv-direct-v1-p16g128-20260802`
- `.cache/bench/qwen36-depthwise-conv-child-v2-p16g128-20260802`

### Review gate: hybrid prefix-cache nil slots

The exact staged Ollama tree exposed a deterministic Nemotron prefix-cache
panic after otherwise successful requests. Hybrid models intentionally retain
nil cache entries for MLP-only layers. Prefix-cache snapshots preserved those
slots, but `cacheTrie.mergeWithChild` unconditionally called `Merge` on every
entry. The fix skips nil entries while retaining the nil slot, matching the
existing hybrid layout. `TestMergeWithChild/NilCacheSlots` covers both the
valid cache merge and nil-slot preservation. Focused and full
`go test ./x/mlxrunner -count=1` runs pass.

The corrected exact-source p2048/g128 Nemotron 4B NVFP4 checks were coherent:

- DGX Spark: `4674.64 / 68.34 tok/s` median versus the retained
  `4686.91 / 68.66 tok/s` result;
- N1x: `779.43 / 28.63 tok/s` median versus the retained
  `805.48 / 28.62 tok/s` result and `525.84 / 26.88 tok/s` target.

The N1x log showed repeated successful leaf and interior-node evictions with
79-116 MiB freed, no panic, no hidden request error, and no runner reload. The
N1x row remains about `148.2% / 106.5%` of target; the previously observed
roughly `5972 / 67 tok/s` result came from crash/reload behavior and is invalid.

Nemotron tokenizes the benchmark corpus at roughly 2.5 tokens per word. The
bench tool calibrates only after a successful warmup, so its default 1.3
estimate can issue a roughly 3,946-token warmup for a 2,048-token target. On
the 64 GiB N1x with `num_ctx=4096`, that first request failed in
`cublasLtMatmul` before calibration. Use
`-prompt-tokens-per-word 2.51` for this model on constrained systems; the
subsequent measured prompt remains calibrated from the actual warmup count.

On DGX Spark, deliberate final `keep_alive:0` unload still causes a teardown-
only Linux ARM64 Go-runtime fault while handling the scheduler's SIGINT. All
correctness and timed requests complete first. Track this separately from CUDA
math and throughput; do not treat it as a request-path failure or ignore it as
a production cleanup issue.

### Final staged-tree provenance and RTX 5090 pressure exception

The final review source IDs are Ollama tree
`6ba3302564727c11bf259f543c6c32c56cfe9d4d` and MLX tree
`5e0e84b528381329eaf7a42ea9dfa513866879c2`. DGX Spark records both exact
trees. N1x initially retained the preceding Ollama marker, but the only two
files changed between that tree and the final tree were `cache_trie.go` and
`cache_trie_test.go`; both remote SHA-256 hashes matched the final index. Its
marker was corrected, and the staging helpers now transport and restore tree
IDs rather than relying on a hard-coded value.

The exact staged WSL payload uses Ollama SHA-256
`877f0e929343a7fec7ee640345151e321d11c97a4f997c1a105e755891f189a0`,
libmlx SHA-256
`9b5addb72ec586e1cc470f6660801828868729562fbbcfc65452cca5ec471ace`,
and libmlxc SHA-256
`f4c8dc8b42786c8263965377b1ca386eb006c799be2ec500b8fbdad3af147527`.
Semantic checks passed for Qwen 3.6 35B-A3B, Gemma4 12B, and Laguna XS.2.

Repeated distinct p2048 prompts no longer reproduce the retained Qwen and
Laguna high-water rows on the current Windows CUDA runtime state. This is not
an inferred source regression:

- the exact historical Qwen binary, native libraries, PTX, client, token
  counts, and request protocol also fell to about `40 tok/s`;
- exact staged Qwen p16/g128 remained in the normal `132-153 tok/s` band;
- exact staged Laguna reached `8682.61 / 134.57 tok/s` before its process grew
  through `26.6`, `29.1`, and roughly `30.3 GiB`, then fell to `33-35 tok/s`;
- exact staged Laguna p16/g128, after its documented two scrub rows, retained
  `142.32`, `151.70`, and `146.94 tok/s` with only about `18 GiB` peak;
- a 200 ms inter-request drain did not recover Qwen, ruling out a simple
  unfinished asynchronous tail.

Treat this as a local CUDA graph/allocation residency exception until a full
Windows driver reset or a lower-level runtime fix restores the historical
state. Do not bisect or remove accepted kernels based only on the repeated
long-prompt collapse. Preserve the historical best rows as high-water results,
but label them runtime-sensitive when reporting review status.

Review artifacts:

- `.cache/bench/qwen36-canonical-harness-historical-post-restart-20260802`
- `.cache/bench/qwen36-exact-staged-review-post-restart-20260802`
- `.cache/bench/qwen36-exact-staged-p16g128-post-restart-20260802`
- `.cache/bench/qwen36-exact-staged-delay200ms-20260802`
- `.cache/bench/gemma4-12b-exact-staged-final-20260802`
- `.cache/bench/laguna-exact-staged-final-20260802`
- `.cache/bench/laguna-exact-staged-p16g128-five-20260802`

Discard
`.cache/bench/qwen36-exact-staged-p2048g128-drain201ms-20260802`: the bench
`-k` flag changed request keep-alive and unloaded the model between rows. The
ignored sweep helper no longer exposes that invalid delay mechanism and no
longer hard-codes `OLLAMA_DEBUG=2`.

### Final review cleanup: wide SDPA default and bounded MoE configs

Canonical models in Daniel's namespace use
`dhiltgen/<model>:<size>-nvfp4`, `dhiltgen/<model>:<size>-mxfp8`, and
`dhiltgen/<model>:<size>-mlx-bf16`. Use these explicit quant tags for new
correctness and performance runs. `dhiltgen/gemma4:12b-mlx` is an obsolete
alias retained only in historical artifacts; do not use it for current data.
Likewise, prefer a canonical size/quant tag over older `*-mlx-nvfp4` names or
metadata-specific aliases whenever the canonical tag exists. Tag naming is not
fully consistent across all model families; when the preferred spelling does
not exist, use the clearest available NVFP4, MXFP8, or MLX-BF16 variant and
record the exact resolved tag rather than inventing a name.

The final accepted source IDs after review cleanup are Ollama tree
`f6629d4a3ad17f6972b91ae33307f384cca28b17` and MLX tree
`616c55634065929e88fe05daf192bf41226d8fde`. The exact RTX 5090 payload uses
Ollama SHA-256
`8edc3e249520c2e97ea39eb5896ef6bf276de0953dd924644dc95ec2289995f2`,
libmlx SHA-256
`402f6ddfcfa15267a99163a5d02be7cbc92f541787ad776a5b8099bf907094c4`, and
libmlxc SHA-256
`f4c8dc8b42786c8263965377b1ca386eb006c799be2ec500b8fbdad3af147527`.

Gemma4 no longer uses the obsolete explicit CUDA matmul/softmax/matmul
fallback for prefill head dimensions above 128. The standard
`nn.ScaledDotProductAttention` path now owns all devices because the staged MLX
CUDA backend supports the required D=256 and D=512 shapes. Acceptance helpers
must not set `OLLAMA_MLX_CUDA_WIDE_SDPA`; a correctness gate with the variable
unset produced coherent Tokyo, Rayleigh-scattering, and natural long-prompt
responses with no NaN/OOM/panic. The product-default RTX 5090 p2048/g128
median was `2713.77 / 117.59 tok/s`, or roughly `139% / 102%` of the retained
llama target (`1951.58 / 114.92`).

The first conventions cleanup removed both prompt-shaped sorted-MoE CUDA
config caches. It remained correct but regressed Gemma4 12B generation from
`117.59` to `92.76 tok/s` median because the row-copy config is also used by
decode/MTP and repeated host graph construction nearly disabled drafting. Do
not retry that naive cleanup.

The accepted refinement caches only row-copy decode shapes, bounded to at most
eight rows (`8..64` assignments), and frees arbitrary prefill-shaped row-copy
and combine configs after each apply. It retained coherent output, restored
healthy MTP acceptance, and measured `2684.57 / 117.63 tok/s` median at
p2048/g128. This follows the per-apply config lifetime used by `gpuKernel` while
preserving the finite hot decode cache.

### Regression recovery after sorted GatherQMM async-copy fix

The final sorted GatherQMM correctness defect was a CUDA shared-memory race,
not a model or runner bug. The SM80 mainloop preserved a three-stage
`cp.async` commit cadence by issuing a speculative final K-tile fetch, then
reused the same shared-storage union for the epilogue while that copy could
still land. Gemma, Qwen, and Nemotron MXFP8 paths consequently produced
nondeterministic corrupt rows and repetitive output. The accepted fix keeps
the commit cadence but commits an empty async group after the last real tile,
so no pending copy can overwrite epilogue storage. Five standalone exact-shape
runs were bit-exact, and `test_gather_qmm_sorted_cuda_async_pipeline` covers the
failure shape. The staged MLX patch ID after this fix is
`e1d88936fad185be7e3919d15586991d31050ee4`.

The exact DGX Spark candidate payload is
`/home/daniel/.codex/dist/ollama-gather-qmm-rhs-empty-commit-nodrain-v19`:

- Ollama SHA-256: `ce96c72108b7b0b73619dab5a7c2817127b55fde0b94d4e489aef78b59c37f15`
- libmlx SHA-256: `49cf3fb71bcf0bef60c371dc42bd04c329dcbb3bbdbee22abe5a9f0d43b23ab1`
- libmlxc SHA-256: `58df6a37e61c293e16a0f6eab6fe232c0c1f5f635d883fbbabea3c53791cf6b6`

All eleven DGX Spark rows passed the integration correctness gate, independent
long-prompt semantics, two p2048/g128 scrub rows, and three retained epochs.
Current candidate medians are:

| Model | Prompt tok/s | Generate tok/s |
| --- | ---: | ---: |
| Gemma4 26B MXFP8 | 969.65 | 46.01 |
| Gemma4 26B BF16 | 864.67 | 27.60 |
| Gemma4 31B NVFP4 | 305.54 | 10.34 |
| Gemma4 31B MXFP8 | 298.17 | 6.80 |
| Gemma4 31B BF16 | 215.08 | 3.77 |
| Qwen3.6 35B-A3B MXFP8 | 1352.00 | 55.80 |
| Qwen3.6 27B NVFP4 | 753.38 | 13.21 |
| Laguna XS.2 NVFP4 | 2181.26 | 82.39 |
| Nemotron3 33B NVFP4 | 2114.80 | 79.09 |
| Nemotron3 33B MXFP8 | 1964.64 | 55.11 |
| Nemotron3 33B BF16 | 1634.24 | 32.19 |

The initially suspicious Nemotron BF16 prompt result was an ordering/thermal
artifact. A candidate/control/candidate run, with correctness before each leg
and five retained epochs, measured prompt medians of `1600.22`, `1557.71`, and
`1608.13 tok/s`; generation medians were `32.70`, `32.43`, and `32.19 tok/s`.
The candidate is not regressed against the exact accepted control.

The exact N1x overlay SHA-256 is
`878c3e12ff6c81faaee8114a8628c704596a9e2131bbe9a839228168b51d04a7`.
All seven rows passed correctness and coherent long-prompt output before timed
p2048/g128 runs. Current medians are:

| Model | Prompt tok/s | Generate tok/s |
| --- | ---: | ---: |
| Qwen3.5 2B NVFP4 | 7960.85 | 127.65 |
| Qwen3.5 2B MXFP8 | 8706.77 | 94.45 |
| Qwen3.5 4B NVFP4 | 3532.29 | 76.61 |
| Qwen3.5 4B MXFP8 | 3280.51 | 48.49 |
| Gemma4 E2B NVFP4 | 5340.05 | 111.47 |
| Gemma4 E2B MXFP8 | 3568.20 | 75.94 |
| Nemotron3 Nano 4B NVFP4 | 5238.55 | 77.39 |

These are all materially above the prior retained N1x medians. The staged
payload is therefore the strongest validated N1x state, not merely equivalent
to the previous high-water state.

The exact current RTX 5090 payload is
`/home/daniel/.codex/dist/ollama-4784-mlx-cuda-current-sm120a`:

- Ollama SHA-256: `948bdf13ea9fc437870d51a4fb38baab73a1b5c4946b55513bd187a646607b05`
- libmlx SHA-256: `e43f0774b901b2f573d678881da4f66b66d816d7a336eab79f247258a549028b`
- libmlxc SHA-256: `705e4e2a6fd8a367534e1e014af721d779922d61ec07898fc3479380c4f1ab3f`

Its correctness gates pass, and Gemma4 26B NVFP4 plus Qwen3.5 NVFP4 exceed
their accepted controls. Laguna and other long-prompt rows remain invalid for
high-water comparison because the local Windows CUDA runtime is still in the
documented degraded graph/allocation-residency state. WSL restart does not
clear it; final RTX 5090 validation requires a full Windows driver reset. Do
not change accepted source based on results gathered before that reset.

### Post-reboot RTX 5090 validation and benchmark cache-busting defect

The full Windows reboot restored clean initial CUDA behavior, and the exact
current payload hashes remained unchanged. All Gemma4 26B NVFP4, Qwen3.6
35B-A3B NVFP4, and Laguna XS.2 NVFP4 correctness gates and independent
long-prompt checks passed. A p16/g128 Qwen control measured a `165.13 tok/s`
generation median, confirming the retained decode kernels remain in their
high-water band.

The Python-code benchmark branch had two issues that made its p2048/g128 rows
look like source regressions:

1. Large prompts were unique as whole strings but shared the leading token
   sequence `Python cache patch benchmark context`. The prefix-cache trie
   matched several leading tokens. A later raw-log audit showed `cached=0` and
   `left=total` on those long requests, so the observed runs did not actually
   skip prompt computation. The generator still failed to guarantee a cold
   request and could reuse work if a restorable snapshot existed at the shared
   prefix, so it remains unsuitable for accepted comparisons.
2. Epoch variation changed filenames throughout the full prompt. On MoE and
   MTP models this changed expert routing, generated trajectories, and draft
   acceptance rather than holding the measured workload constant.

The harness now varies the first retained padding word, preserves the same
coherent Python body across epochs, and has tests that require distinct leading
words plus identical remaining words. Server logs confirm `matched=0` on all
corrected retained requests. The final WSL bench client SHA-256 is
`265ed8cf36b6547895a66c5f3f621bfef552d12670b6d0671b461c5f0ce13bc4`.

With identical bodies and stable prompt token counts, Qwen's retained prompt
median recovered to `7700.31 tok/s`, and its generation epochs clustered at
`135.88-138.63 tok/s` except one `88.42 tok/s` scrub. Gemma still ranged from
`102.94` to `223.71 tok/s`, consistent with MTP acceptance sensitivity, while
Laguna held near `51-53 tok/s` for this particular coding trajectory despite
having reached `136.80-152.29 tok/s` on other cache-cold Python variations.
These values demonstrate prompt-path variance rather than a deterministic
kernel regression; historical word-list high-water rows and the new coherent
coding workload must not be presented as the same benchmark population.

Artifacts:

- `.cache/bench/regression-final-v19-post-win-reboot-20260803/rtx5090`
- `.cache/bench/regression-final-v19-post-win-reboot-cachecold-v2-20260803/rtx5090`
- `.cache/bench/regression-final-v19-post-win-reboot-stablebody-v3-20260803/rtx5090`
- `.cache/bench/qwen36-post-win-reboot-p16g128-20260803`

### Final RTX 5090 validation with the corrected Python benchmark

`origin/bench-prompt` at `9d62d2de3` fixed the original cache-buster defect by
keeping the variation header at the retained front of token-targeted prompts.
Two additional controls are required for repeatable MLX measurements:

- seed the first warmup with the model-specific token-to-word estimate; without
  it, the nominal p2048 Gemma warmup contained 4,712 tokens and drove peak
  memory above 30 GiB before the measured epochs;
- vary only the leading cache-buster and keep the Python body identical; varying
  filenames throughout the body changed tokenized prompt lengths and made
  Laguna alternate between fast and slow compile/kernel shapes.

The accepted WSL client includes both controls and has SHA-256
`504a01090a3c9d36b6a69993e0ade7950b83dc79dc1eacb0d8443b22ec4d99cb`.
Server logs show `matched=0` for every measured request and stable prompt counts
within each model. All three correctness gates passed. Two scrub rows followed
by three retained p2048/g128 rows measured:

| RTX 5090 model | Prompt tok/s | Generate tok/s |
| --- | ---: | ---: |
| Gemma4 26B NVFP4 | 4109.72 | 180.68 |
| Qwen3.6 35B-A3B NVFP4 | 7081.54 | 87.51 |
| Laguna XS.2 NVFP4 | 7759.65 | 56.18 |

The Python continuation changes output trajectories, expert routing, and MTP
acceptance relative to the historical word-list benchmark, so its generation
rates are a new benchmark population rather than evidence of a product-source
regression. Existing correctness-gated p16 controls remain in the retained
high-water decode bands. Compare the Python workload only against baselines
captured with the same fixed prompt generator.

The historical July 27 RTX campaign was audited separately. Its launcher used
one bench binary and identical p2048/g128 flags for candidate and target. All
retained MLX long-prompt requests had `cached=0` and `left=total`, and every
llama-server task began with `cached n_tokens = 0`. Effective MLX cache hits in
those logs occurred only in short correctness requests, not the timed p2048
rows. The old percentages are therefore valid for that word-list population,
but they do not establish parity for this Python population. The reusable
auditors are `.cache/scripts/audit_benchmark_cache.ps1` and
`.cache/scripts/audit_benchmark_cache.sh`.

Artifact:

- `.cache/bench/regression-final-v19-origin-bench-prompt-stablebody-v4-20260803/rtx5090`

### Correct cache-cold benchmark prompt algorithm

The prompt generator must maintain two independent invariants: every request
must miss the prefix cache, and every retained request must exercise the same
token/graph shape and semantic body. Whole-string uniqueness is insufficient.

1. Build one fixed corpus body for the requested shape.
2. Put the only per-request variation in token zero, before all shared text.
   Different first words are not sufficient because two strings can still
   share their first tokenizer token. Use a tokenizer-verified set of distinct
   one-token nonces when the tokenizer is available.
3. Insert that cache-buster after any front-trimming logic, or size only the
   body after inserting it, so it can never be trimmed away.
4. Keep all filenames, continuation text, and padding after the first token
   byte-for-byte identical across warmups and measured epochs.
5. Run a bounded token-density calibration before the target-size warmup; the
   initial `1.3 tokens/word` estimate can otherwise turn p2048 into a much
   larger request and OOM a near-capacity model.
6. Require `matched=0` in server logs and validate `PromptEvalCount` on every
   retained row. A materially different count invalidates the row.

The leading nonce can still tokenize to a slightly different number of tokens
across variations. The focused Qwen run observed counts of 2052 and 2055 even
with an identical body. The robust long-term design is a benchmark-only
request control that bypasses prefix-snapshot lookup/store, allowing every
epoch to use the exact same prompt. Until that exists, treat `matched=0` as a
mandatory runtime check, retain only tightly clustered counts, and
normalize/report the actual `PromptEvalCount`; do not claim exact-shape
equivalence from text construction alone.

Implement that control as a default-true `cache_prompt` runtime option rather
than a benchmark-specific text convention. Propagate it through
`llm.CompletionRequest`; llama-server already has a `cache_prompt` request
field. For MLX, the false path must start a transient cache session at offset
zero, schedule no prefill snapshots, skip trie insertion on close, and
free/rewind the live cache state after the request. `bench.go` can then send
the byte-identical calibrated prompt for warmup and every retained epoch with
`cache_prompt=false`. Validate equal prompt counts, but do not mutate prompt
text merely to make it unique.

Implementation map for the benchmark branch:

- Add a runtime `CachePrompt bool` to `api.Options` and set it to `true` in
  `api.DefaultOptions`; `Options.FromMap` already supports boolean fields, so
  `bench.go` can send `"cache_prompt": false` through its normal options map.
  No parallel request field is needed: `llm.CompletionRequest.Options` and
  `x/mlxrunner.CompletionRequest.Options` already carry the resolved
  `api.Options` value through both runner paths.
- Replace the hard-coded `CachePrompt: true` in `llm/llama_server.go` with
  `req.Options.CachePrompt`. In `TextGenerationPipeline`, choose
  `prefixCache.begin` or a new `prefixCache.beginTransient` from
  `request.Options.CachePrompt`.
- `beginTransient` must first switch the live cache to the trie root at offset
  zero, preserving the prior active leaf as normal paged-out snapshots. It
  returns the full input as `remaining`, marks the session transient, and
  schedules no branch or periodic snapshots.
- Make `cacheSession.schedulePrefillSnapshots` a no-op for transient sessions.
  On transient close, finish evaluating cache state, then switch back to the
  trie root at offset zero. Do not call `advancePath`, update trie `lastUsed`,
  or retain request snapshots. The next normal request can still page in any
  previously retained conversation path.
- Add tests proving defaults remain cache-enabled, llama-server receives both
  true and false, a transient MLX request evaluates every prompt token and
  leaves the trie unchanged, normal prefix reuse still works before and after
  a transient request, and every benchmark epoch sends the same prompt with
  cache retention disabled.

### Corrected fixed-body RTX 5090 targets

The prior Python-patch llama-server targets were still generated by the broken
cache-buster/body variation code and are invalid for target percentages. The
current `origin/bench-prompt` tip `9d62d2de3`, plus the local fixed-body
refinement, was used for both backends. Every retained llama-server request
started with `cached n_tokens = 0`; retained prompt counts were constant within
each model. Exact staged MLX logs likewise report `matched=0` for every
retained request.

| RTX 5090 p2048/g128 | MLX | llama-server | Percent of target |
| --- | ---: | ---: | ---: |
| Gemma4 26B NVFP4 / Q4_K_M prompt | 4,117.53 | 4,379.04 | 94.0% |
| Gemma4 26B NVFP4 / Q4_K_M generate | 203.22 | 184.57 | 110.1% |
| Qwen3.6 35B-A3B NVFP4 / Q4_K_M prompt | 7,949.00 | 5,265.92 | 151.0% |
| Qwen3.6 35B-A3B NVFP4 / Q4_K_M generate | 165.85 | 225.28 | 73.6% |
| Laguna XS.2 NVFP4 / Q4_K_M prompt | 7,624.77 | 5,960.19 | 127.9% |
| Laguna XS.2 NVFP4 / Q4_K_M generate | 56.75 | 196.65 | 28.9% |

The fixed-body Laguna rows are cache-fair but are not an exact p2048 shape.
Both retained backends received 2,059 tokens and were cold: MLX reported
`cached=0, left=2059`, while llama-server's first cache count for each task was
zero. MLX therefore evaluated one 2,048-token prefill chunk plus a second
10-token chunk before retaining the final seed token. A measured-only Nsight
trace confirms the extra chunk accounts for exactly 39 additional attention
evaluations, 275 quantized matmuls, 38 matmuls, and 76 aranges compared with the
old exact-2,048 trajectory. Do not mix these rows with exact-p2048 results or
attribute the aggregate trace delta to an SDPA dispatch change. Re-run the
target table after the `cache_prompt=false` benchmark path can send one
byte-identical, tokenizer-calibrated exact-2,048 prompt to both backends.

### Final regression recovery and conventions review

The reviewed source identities are Ollama tree
`31adb4ce57ca9fb2daf3e1a53704c3a1c8a39ffb` and MLX tree
`132e79f8023286069203e55560587cf3adac6e87`. The MLX diff remains limited to
`mlx/backend/cuda/` and CUDA-focused Python tests; it does not change MLX core
or public APIs. Review narrowed the exact-width RMSNorm schedule and the
eight-row QMV preference to runtime compute capability 12, preserving existing
behavior on unmeasured GPUs. The Ollama adapter review also fixed ownership of
the first array when extraction of a two-output custom-kernel result fails.
The final tree additionally frees any already-extracted arrays if a later
output from Qwen's five-output fused preprocessing kernel cannot be acquired;
both adapter fixes are error-only and leave the measured success path unchanged.

The pre-review broad matrix passed all seven RTX 5090 rows and six of seven N1x
rows before an N1x SSH interruption. DGX Spark rows after the first two were
misreported as correctness failures by the chained campaign even though the
saved integration, direct-response, and long-prompt artifacts were coherent;
a standalone traced rerun completed with `CORRECTNESS_DONE`. Final-tree
representative p2048/g128 medians before the behavior-neutral MLX scope cleanup
were:

| Platform / model | Prompt tok/s | Generate tok/s |
| --- | ---: | ---: |
| RTX 5090 / Qwen3.6 35B-A3B NVFP4 | 6725.07 | 164.74 |
| DGX Spark / Qwen3.6 35B-A3B MXFP8 | 1605.18 | 50.35 |
| N1x / Qwen3.5 4B NVFP4 | 4042.83 | 59.24 |

The DGX Spark Qwen generation rows stopped at 87 tokens in every epoch, so use
that row only as a fast-path provenance check, not as a canonical g128 result.
Post-rebuild short integration and long-prompt correctness gates must remain
the final acceptance evidence for MLX tree `132e79f...`.

The final error-path-only adapter cleanup was rebuilt from Ollama tree
`31adb4ce57ca9fb2daf3e1a53704c3a1c8a39ffb` without rebuilding the unchanged
native payloads. Representative installed-payload gates passed on all targets:

- RTX 5090: Qwen3.6 35B-A3B NVFP4 integration, direct response, and natural
  long prompt completed (`CORRECTNESS_DONE final-reviewed-31adb-5090`).
- DGX Spark: Qwen3.6 35B-A3B MXFP8 completed the same gate
  (`CORRECTNESS_DONE final-reviewed-31adb-spark-r2`). The preceding attempt
  aborted because the harness was given a nonexistent integration-test path;
  its shutdown dump is not an inference failure.
- N1x: Qwen3.5 4B NVFP4 produced coherent short and 1,984-token long-prompt
  responses (`CORRECTNESS_ONLY=passed`).

Do not overlay the complete Git archive onto an existing N1x source mirror
with Windows `tar.exe` without scanning the result. During this review it
replaced `x/models/qwen3_5_moe/qwen3_5_moe.go` and
`x/transfer/sparse_other.go` with same-length files containing NUL bytes. The
Go compiler caught both. Scan tracked `*.go` files for byte zero, repair exact
files with `scp`, and use the incremental helper's `-SkipExtract` mode. Also
keep native-command stderr handling at `Continue` around CMake and decide
success from `$LASTEXITCODE`; PowerShell otherwise turns harmless CMake
deprecation warnings into terminating build failures.

The official Laguna GGUF fails the current integration gate because its answer
contains raw `</assistant>` tags around semantically correct content. Its
target row is retained only after the independent factual chat and natural
long-prompt gates passed; the output-template defect remains explicitly
qualified. Qwen's official llama-server log reports no speculative decoding
implementation and zero draft state. The MLX log has no draft/MTP evidence
either, so the Qwen generation difference should be treated as a real MoE
decode gap rather than an acceptance-rate mismatch.

Artifacts:

- `.cache/bench/review-final-staged-v2-20260803/rtx5090`
- `.cache/bench/review-final-llama-stablebody-20260803/rtx5090`

After the platform-policy cleanup, the corrected harness passed Qwen
correctness and measured three tightly grouped retained rows with medians of
`7166.39 prompt / 143.30 generate tok/s`. Artifact:

- `.cache/bench/review-platform-policy-20260803/rtx5090`

### Exact-p2048 cache-neutral protocol correction

The cache audit found a second failure mode beyond effective prefix hits. A
request can report a full cold miss (`matched=0`, `cached=0`, `left=total`)
while snapshots retained by an earlier warmup still change the measured
request's memory residency and CUDA-graph behavior. On Laguna, a distinct
exact-p2048 warmup followed by an exact-p2048 cold measured request raised
peak MLX memory from about `26.9 GiB` to `30.3 GiB`; measured generation fell
to `56.97 tok/s`. A fresh-process exact-p2048 request after the smaller p2027
warmup remained near `26.9 GiB` and repeated at `137.36 tok/s`. Both measured
requests were cache misses, so lookup evidence alone is insufficient.

The exact-p2048 prompt was byte-preserved from the Python patch body and
tokenizer-calibrated. The official llama-server target accepted the same bytes
at exactly 2,048 tokens and began the measured task at `cached n_tokens = 0`.
Its same-shape run measured `7,106.60 prompt / 192.98 generate tok/s`. Do not
publish a new percent-of-target row yet: the current MLX helper cannot combine
same-shape warmup with cache-neutral residency until `cache_prompt=false` is
wired through both backends.

The production benchmark protocol therefore requires all of the following:

1. Send one byte-identical, tokenizer-calibrated exact-p2048 prompt for the
   same-shape warmup, scrub rows, retained MLX rows, and retained llama rows.
2. Send `cache_prompt=false` on every benchmark request, including warmup and
   scrubs. On MLX this must suppress lookup, snapshot scheduling, trie
   insertion, `lastUsed` updates, and retained live cache state; skipping only
   lookup is not cache-neutral.
3. Verify exact `PromptEvalCount=2048` on both backends. For MLX, require no
   transient-request `created snapshot` records and stable request-boundary
   active/peak memory. For llama-server, require each measured task's first
   `cached n_tokens` value to be zero.
4. Keep one warmup plus two full-shape scrub rows in the same process before
   retained rows so graph/JIT and allocator high-water marks are conditioned
   without retaining conversation prefixes.
5. Audit candidate and target logs with the same rules before computing any
   percent-of-target value.

Matched Nsight captures also explain why the old 2,059-token Laguna workload
is not interchangeable with exact p2048. Exact p2048 decode used MLX vector
attention in the 30 sliding layers and cuDNN in the 10 full-attention layers;
the 2,059-token trajectory reached cuDNN in all 40 layers after its extra
10-token prefill tail. This sequence-shape state change reinforces the rule
that actual prompt counts must match exactly.

Artifacts:

- `.cache/bench/raw-prompt-telemetry/laguna-python-exact-p2048-g128-cold-repeat-v2`
- `.cache/bench/raw-prompt-telemetry/mlx-laguna-python-exact-p2048-g128-shapewarm-cold-v1`
- `.cache/bench/raw-prompt-telemetry/llama-laguna-python-exact-p2048-g128-shapewarm-cold-v2`
- `.cache/profiles/laguna-pipeline-sync-ab`

### Cache-neutral implementation validation (2026-08-03)

The fetched `origin/bench-prompt` ref still points at `9d62d2de3`. It keeps a
variation header near the front of generated text, but it does not expose
`cache_prompt=false`, and prompt calibration can still produce different token
counts across candidate and target. Treat the earlier fixed-body percentages
above as superseded until they are repeated with the cache-neutral protocol.

A rejected local experiment added an out-of-tree cache-bypass mechanism so the
same exact 2,048-token prompt could be compared against the official v0.32.4
llama-server payload. It was useful diagnostic evidence, but it is not part of
the current tree or benchmark API. There is no `cache_prompt` request field in
this repository. Current validation instead requires distinct first-token
prefixes, zero matched/cached tokens in server logs, and a fresh runner process
per measured epoch when the comparison must also be cache-residency neutral.

After one warmup and two scrub rows, the three retained exact-p2048/g128 Laguna
medians were:

| RTX 5090 Laguna XS.2 | Prompt tok/s | Generate tok/s | Percent of target |
| --- | ---: | ---: | ---: |
| MLX NVFP4 | 5,884.71 | 54.04 | 82.1% / 27.6% |
| llama-server Q4_K_M | 7,166.13 | 195.46 | target |

These are the first same-bytes, same-count, cache-neutral values and therefore
replace the historical `136.6% / 114.4%` status row for this workload. They do
not prove an Ollama source regression: the old row used the word-list prompt
population and a different target run. The audit found no cross-request cache
hit in its retained p2048 requests, but it did allow MLX snapshots and live
cache residency to accumulate. That state changed memory and graph behavior
even when the next lookup reported `cached=0`.

A cache-enabled diagnostic demonstrates why actual cache-hit rows must be
discarded: after Laguna reached `cached=2047` of 2,048 prompt tokens, reported
prompt throughput jumped to tens of millions of tokens per second. At
temperature zero, repeated Laguna MLX responses still diverged with normal
prefix caching enabled, so the observed trajectory variation is not introduced
by the transient cache path. The llama-server responses were byte-identical.
This needs separate CUDA/model-path investigation before using generation
variance as optimization evidence.

A logical-reset experiment retained all cache array handles rather than freeing
wrapped sliding caches. It regressed retained prompt throughput to about 4.5k
tok/s, produced a 1.54k outlier, and did not improve response determinism or
generation. The experiment was removed. The installed binary was rebuilt to
SHA-256 `57465101b462c632eae3f0701419b90310dbccd9c8baf1e572a38626b213bd41`,
which exactly matches the source used for the accepted MLX row above.

Artifacts:

- `.cache/bench/raw-prompt-telemetry/mlx-laguna-exact-p2048-cache-disabled-rewind-v5`
- `.cache/bench/raw-prompt-telemetry/llama-laguna-exact-p2048-cache-disabled-v1`
- `.cache/bench/raw-prompt-telemetry/diag-mlx-laguna-cache-enabled-same-prompt-v7`
- rejected reset experiment: `.cache/bench/raw-prompt-telemetry/mlx-laguna-exact-p2048-cache-disabled-reset-v6`

The same protocol was then repeated for Qwen3.6 35B-A3B. An isolated
calibration process produced one 6,890-character raw prompt that both model
artifacts counted as exactly 2,048 tokens. All MLX and llama-server responses
were byte-identical within their backend, every request started cold, and MLX
peak memory settled at 30.26 GiB. The retained medians were:

| RTX 5090 Qwen3.6 35B-A3B | Prompt tok/s | Generate tok/s | Percent of target |
| --- | ---: | ---: | ---: |
| MLX NVFP4 | 9,646.35 | 160.29 | 167.2% / 70.3% |
| llama-server Q4_K_M | 5,770.77 | 228.14 | target |

One MLX retained decode row was a 97.19 tok/s outlier; the other two were
160.29 and 161.04, so the standard three-row median remains robust. This
confirms the known Qwen MoE decode gap while showing that its prompt path is
well above target under the strict cache-neutral protocol.

Artifacts:

- `.cache/prompts/qwen36-exact/python-epoch0-p2048.txt`
- `.cache/bench/raw-prompt-telemetry/mlx-qwen36-35b-a3b-exact-p2048-cache-disabled-v1`
- `.cache/bench/raw-prompt-telemetry/llama-qwen36-35b-a3b-exact-p2048-cache-disabled-v1`

Gemma4 26B initially exposed two correctness-gate issues rather than usable
performance data. Raw mode produced pathological repeated output, while the
normal chat template consumed all 128 generated tokens as hidden thinking and
returned an empty visible response. The accepted run disables thinking and
uses a separately calibrated 5,570-character chat prompt that both artifacts
count as exactly 2,048 tokens. All six responses per backend were coherent and
byte-identical, and all requests passed the same cache-neutral gates as Qwen
and Laguna.

| RTX 5090 Gemma4 26B | Prompt tok/s | Generate tok/s | Percent of target |
| --- | ---: | ---: | ---: |
| MLX NVFP4 | 4,443.69 | 244.94 | 104.1% / 125.6% |
| llama-server Q4_K_M | 4,270.52 | 195.07 | target |

Artifacts:

- `.cache/prompts/gemma4-26b-exact/python-chat-nothink-epoch0-p2048.txt`
- `.cache/bench/raw-prompt-telemetry/mlx-gemma4-26b-chat-nothink-exact-p2048-cache-disabled-v3`
- `.cache/bench/raw-prompt-telemetry/llama-gemma4-26b-chat-nothink-exact-p2048-cache-disabled-v1`

### Laguna CUDA graph residency diagnostics (2026-08-03)

The cache-neutral Laguna slowdown is accompanied by CUDA graph executable
growth, not a live KV-cache or MLX allocator-cache leak. Request-boundary
active memory remained about 18.0 GiB and `Sweep`/`ClearCache` reduced MLX
cache memory to zero, while peak memory rose from 26.67 GiB to 30.26 GiB.
Graph instrumentation found only 32-34 structural cache entries after roughly
1,000 commits, with no `cudaGraphExecUpdate` failures. A first request created
four large prefill graphs (397-801 nodes) and about a dozen small-batch decode
graphs (625-801 nodes); later requests produced new dependency topologies.

The following strategies were tested and rejected:

- A global graph-cache capacity of 32 stabilized peak memory near 26.68 GiB,
  but continually re-instantiated graphs and reduced retained generation to
  about 85-91 tok/s.
- Clearing the entire graph cache when execution transitions from decode back
  to prefill reduced prompt throughput to about 4.25k tok/s and generation to
  73-91 tok/s because each request rebuilt both prefill and decode graphs.
- After stream/device synchronization, clearing the graph cache released about
  6.8 GB. `cudaDeviceGetGraphMemAttribute` reported zero used and reserved
  bytes, and `cudaDeviceGraphMemTrim` released nothing further. That API only
  covers CUDA graph memory-allocation nodes; it cannot trim MLX's instantiated
  graph executable residency.
- A direct SM120 QMV helper path passed an initial correctness gate but then
  produced pathological repeated output in the exact workload. It was removed
  rather than benchmarked.

Selective retirement of stale prefill graph executables while retaining hot
decode graphs was also rejected. It released 6.3-7.6 GB at each request
boundary and kept decode rows in the 141-160 tok/s range, but rebuilding four
prefill graphs per request reduced retained prompt to a 4,642.54 tok/s median;
retained generation was 145.52 tok/s. Do not retry a global cache cap, global
or selective phase clear, graph-memory trim, graph-limit retuning,
pointer-layout replay, allocator FIFO/LIFO changes, or graph-key string
encoding; those approaches are already disproven. Also treat Laguna
generation numbers as suspect until its temperature-zero output instability
is root-caused: identical exact requests have produced coherent but different
patch trajectories, 54-160 tok/s decode bifurcation, and one pathological
repeated-space response.

Artifacts:

- `.cache/bench/raw-prompt-telemetry/mlx-laguna-graph-stats-live-exact-cache-neutral-20260803`
- `.cache/bench/raw-prompt-telemetry/mlx-laguna-memory-boundaries-exact-cache-neutral-20260803`
- `.cache/bench/raw-prompt-telemetry/mlx-laguna-graph-cache32-exact-cache-neutral-20260803`
- `.cache/bench/raw-prompt-telemetry/mlx-laguna-graph-phase-evict-exact-cache-neutral-v1-20260803`
- `.cache/bench/raw-prompt-telemetry/mlx-laguna-graph-mem-trim-p2048g8-v1-20260803`
- `.cache/bench/raw-prompt-telemetry/mlx-laguna-selective-prefill-evict-p2048g128-v1-20260803`

### Rejected: late recurrent snapshot suppression (2026-08-03)

The corrected unique-prefix benchmark exposed the already-known distinction
between a cold lookup and a cache-neutral request. With the staged production
cache path, Qwen3.6 35B-A3B generation fell from a fast first request to about
`66-75 tok/s` after recurrent snapshots and retained CUDA state accumulated.
Making ordinary KV close asynchronous while synchronizing only recurrent state
restored Laguna's cache-enabled decode path, but did not make Qwen cache-neutral.

Two recurrent-only CUDA experiments were rejected. Suppressing proactive
near-end/periodic snapshots improved Qwen's retained generation median to
`86.83 tok/s`. Detecting a near-capacity peak, evicting all inactive snapshots,
and refusing to copy later unrelated recurrent leaves retained only `84.75
tok/s`. The latter visibly released roughly `0.9 GiB` of snapshots, but peak
memory stayed near `30.27 GiB`; acting after the first near-capacity request is
too late to undo graph/allocation residency. Do not retain either policy or
repeat it as a solution to the strict benchmark protocol. There is no
`cache_prompt` request field. Unique first-token prompts prevent cache hits but
do not prevent retained snapshots from changing later epochs. A cache-neutral
comparison must use a fresh runner process per measured epoch unless a real
request API is deliberately designed later. Treat production prefix-retention
behavior as a separate runner design problem.

Artifacts:

- `.cache/bench/cache-bust-fixed-20260803/rtx5090/qwen36-35b-a3b-nvfp4-no-proactive-snapshots`
- `.cache/bench/cache-bust-fixed-20260803/rtx5090/qwen36-35b-a3b-nvfp4-pressure-drop-branches`

### Exact-tree validation corrections (2026-08-04)

The validation began with Ollama tree `31adb4ce57ca9fb2daf3e1a53704c3a1c8a39ffb`
and MLX tree `132e79f8023286069203e55560587cf3adac6e87`. Removing the
regressing prefill memory-query guard produced the reviewed Ollama staged tree
`cd5774163dcc403e71fad75384ae47c503cda8ce`; the MLX tree is unchanged.

- A late Ollama runner guard queried CUDA free, active, and peak memory before
  every prefill and halved the normal chunk at two-thirds model residency. On
  RTX 5090 Qwen3.6 35B-A3B, removing those MLX memory queries raised the same
  current-client run from about `106` to `133.6 tok/s` generation and restored
  the established 2048-token prefill shape. The guard and its unit test were
  removed; `go test ./x/mlxrunner` passes.
- The old `164.74 tok/s` Qwen control used `libmlx.so` hash `1a55b913...`, which
  no longer exists in any retained WSL build or install. The clean reviewed
  payload is `db120cd1...`. Replaying the preserved old benchmark client on the
  clean payload produced `140-143 tok/s` before the device degraded, so do not
  present the 164.74 row as current-tree evidence. PTX caches must remain keyed
  by the complete binary/library/header provenance.
- Repeated RTX runs then converged to about `81 tok/s` for both old and new
  clients on a byte-identical clean source path. This is the documented
  graph/allocation-residency device state, not a source regression. Restarting
  idle WSL on 2026-08-04 did not clear it: the exact cleaned Qwen control passed
  correctness but returned `5565.03 / 82.00`, with only about 78 W observed
  during the run. A Windows host reboot is required before the RTX matrix is
  valid.
- The N1x exact payload passed correctness but Qwen3.5 2B fell to about
  `1220 / 30 tok/s`; 84 busy telemetry samples drew at most `6.44 W`. This
  matches the documented post-profiler low-clock state. The cleaned Go payload
  was rebuilt and installed on 2026-08-04, but before any confirmed host reboot
  the control remained `1039.42 / 29.72`, with 80 busy samples drawing at most
  `6.47 W`. A confirmed N1x host reboot is required before its final matrix is
  valid.
- Windows-side Codex sandbox SSH uses an offline account. Run N1x commands with
  normal credentials outside the sandbox using direct `ssh tater62`; the
  Mac jump route is unnecessary and cannot authenticate from this host.
- DGX Spark exact-source rows completed within about three percent of their
  valid controls: Gemma4 26B/31B, Qwen3.6 27B, Laguna XS.2, and Nemotron3 33B
  NVFP4/MXFP8/BF16. Its Qwen3.6 35B-A3B MXFP8 87-token row remains a
  fast-path provenance check rather than a canonical g128 result.
- After removing the prefill memory-query guard and rebuilding the exact
  cleaned Go payload, the resumed DGX rows remained healthy: Qwen3.6 35B-A3B
  MXFP8 `1578.63 / 51.17` (87 generated tokens), Qwen3.6 27B NVFP4
  `922.54 / 12.05`, Laguna XS.2 NVFP4 `2613.69 / 83.69`, Nemotron3 33B NVFP4
  `2168.21 / 77.37`, and Nemotron3 33B MXFP8 `2058.23 / 55.14`.

### Current-tree validation after review cleanup (2026-08-04)

The accepted source identities are now Ollama tree
`baca1640b85c972af37e462b23fe1d66cad6d950` and MLX tree
`132e79f8023286069203e55560587cf3adac6e87`. Do not combine rows labeled
`21cce07b...` with the final matrix. The tree transition restores the
established asynchronous cache-close materialization while retaining CUDA
memory-pressure sampling. A synchronous `mlx.Eval` at cache close was rejected:
near-capacity workloads can fail at teardown and the completed forward pass
already contributes to the sampled peak.

Gemma4 12B BF16 does not complete the exact RTX 5090 p2048/g128 workload with
the validated native payload. Its 1,984-token long-prompt correctness gate is
coherent, but the benchmark's 2,042-2,043-token prefill reaches about 30.30 GiB
and the next allocation fails in `Runner.prefill`. This is not the cache-close
error; the async close only stopped a later `cudaPeekAtLastError` from obscuring
the original prefill stack. Exclude this BF16 row as non-fitting rather than
using a model-specific low-graph policy.

Repeated near-capacity OOMs can leave the RTX 5090 in the already documented
low-power state. The post-OOM Qwen3.6 35B-A3B control measured `2778.65 / 82.01`
with a maximum 495 MHz SM clock and 77.42 W, versus the healthy roughly
`6673 / 133.6` control. A Windows host reboot is required; restarting WSL is
insufficient. Similarly, the N1x control measured `1235.05 / 36.71`, with 80
busy samples drawing at most 6.48 W. SSH and RDP health do not validate GPU
health; require the small telemetry control to recover before running its
matrix.

Valid corrected-tree RTX rows collected before the OOM-induced device state:

- Nemotron3 4B NVFP4: `20862.00 / 335.86` (73 generated tokens).
- Gemma4 12B NVFP4: `3032.51 / 127.68`.
- Gemma4 12B MXFP8: `2839.62 / 92.94`.
- Gemma4 26B NVFP4: `4017.98 / 218.74`.

The N1x reboot at `2026-08-04T00:50:45-07:00` did not recover performance.
The exact-tree Qwen3.5 2B control remained at `1170.43 / 39.83`, with 81 busy
telemetry samples and at most 6.94 W. A temporary one-shot diagnostic proved
that the decode fusion was enabled rather than rejected by the WoA memory
view: CUDA reported `8,522,825,728` total, `6,039,142,400` free,
`1,835,026,000` active, and `1,883,552,488` peak bytes against the 2 GiB
reserve. The diagnostic used a fresh PTX cache keyed by the complete runner,
MLX DLL, MLX-C DLL, NVRTC, headers, and compute capability. An independent
Nemotron3 4B NVFP4 control also collapsed to `833.33 / 29.55`, versus its
accepted roughly `5238.55 / 77.39` control. The cross-architecture failure
after a real reboot establishes a host/driver performance-state problem, not a
Qwen fallback or current-tree source regression. The temporary diagnostic was
removed and the N1x Go payload rebuilt from the exact staged source.

The post-cleanup native Go tests must include the installed MLX directory in
`LD_LIBRARY_PATH`; repository payload discovery finds `libmlxc.so` through a
symlink, but the ELF loader otherwise cannot resolve sibling `libmlx.so`.
With that environment, the focused suite genuinely executed and passed:
`./x/mlxrunner`, `./x/mlxrunner/mlx`, `./x/models/gemma4`,
`./x/models/qwen3_5`, `./x/models/laguna`, and `./x/models/nemotron_h`.

### N1x recovery after a hard power cycle (2026-08-04)

The earlier Windows restart was insufficient, but a hard power cycle restored
the N1x GPU performance state. The exact-tree Qwen3.5 2B NVFP4 control passed
correctness and returned `9673.39 / 111.39`, drawing up to 35.97 W instead of
the degraded 6-7 W. The subsequent exact-tree matrix produced:

| Model | Prompt tok/s | Generate tok/s | Generated | Status |
| --- | ---: | ---: | ---: | --- |
| Qwen3.5 2B NVFP4 | 10557.99 | 114.51 | 128 | correct |
| Qwen3.5 2B MXFP8 | 9365.85 | 80.13 | 128 | correct |
| Qwen3.5 4B NVFP4 | 4236.06 | 60.71 | 128 | correct |
| Qwen3.5 4B MXFP8 | 3670.81 | 40.89 | 128 | correct |
| Gemma4 E2B NVFP4 | 4818.15 | 112.34 | 128 | correct |
| Gemma4 E2B MXFP8 | 4071.11 | 78.74 | 52 | correct; short generation |
| Nemotron3 Nano 4B NVFP4 | 4459.47 | 73.51 | 73-122 | correct; variable short generation |

All models passed `TestBasic` and an independent natural-language long-prompt
check. The full g128 rows used three measured epochs and passed the cold-cache
audit. Gemma MXFP8's 52-token continuations were stable and cache-cold but are
not canonical g128 generation data. Nemotron's retained epochs were also cold;
the sole 9-token cache match belonged to the separate 2,030-token long-prompt
correctness check. The older exact-tree controls are high-water comparisons,
not matched llama-server targets. Every full-length current row remains well
above its matched llama-server target documented earlier in this file.

The local RTX 5090 showed the same failure after near-capacity OOMs. A normal
Windows restart did not recover it: Qwen3.6 35B-A3B NVFP4 remained at
`1364.61 / 48.17`, with a 907 MHz maximum SM clock and 93.78 W. Require a full
host power cycle before collecting its pending Qwen and Laguna exact-tree rows.

### RTX 5090 long-request residency scope (2026-08-04)

The hard-power-cycled RTX 5090 is healthy outside the near-capacity workload.
The untouched accepted payload retained `165.46`, `165.37`, and `158.01 tok/s`
on a p16/g128 Qwen3.6 35B-A3B NVFP4 control, reached 2,962 MHz, and reported no
thermal or power-brake throttling. Idle desktop usage was about 1 GiB. A fresh
runner with only a two-token warmup also produced coherent p2048/g128 output at
`1869.45 / 161.37 tok/s`. The collapse therefore is not general device health,
temperature, or ordinary variation in desktop VRAM usage.

Repeated distinct p2048 requests grow retained CUDA graph/allocation state to
about `30.27 GiB` on the 31.84 GiB device and then reduce Qwen generation to
roughly `29-75 tok/s`. Desktop usage can move that cliff slightly, but cannot
explain the multi-gigabyte growth. A temporary SM120-only split graph-cache
experiment confirmed that prefill uses six large graph segments per prompt
shape and decode accumulates request-specific dependency topologies. Logical
LRU caps, synchronization-boundary prefill clearing, and hot-tail decode
pruning did not safely release the underlying CUDA residency; one too-small
active cap also triggered `cudaGraphAddDependencies` failure. The complete
rejected experiment is preserved at
`.cache/patches/mlx-sm120-split-graph-cache-rejected-20260804.patch`; MLX source
was restored to the accepted staged tree afterward. Do not retry graph-cache
splitting or capacity pruning without a new mechanism that demonstrates actual
driver-memory release while retaining graph-update reuse.

An SM120-only follow-up replaced the topology-keyed large-prefill cache with
one reusable executable per consecutive prefill graph slot. The intent was to
retain the six hot graph segments for the current prompt shape while destroying
only the corresponding segments from older shapes. Qwen3.6 35B-A3B NVFP4
passed factual and long-prompt correctness, but the standard p2048/g128 sequence
still reached `30.27 GiB` and measured only `11.90-12.63 tok/s` generation.
Slot replacement disrupted full-graph reuse without releasing the retained
residency. The experiment was removed; do not retry positional graph slots.
Artifact: `.cache/bench/ab-prefill-slots-sm120-20260804`.

### RTX 5090 selected-device query regression (2026-08-04)

An exact old-Go/current-native A/B isolated a cross-request Qwen3.6 regression
to querying CUDA capability and memory through the retained Go
`DefaultDevice()` handle. The staged tree measured `5585.94 / 128.82` on the
standard cache-neutral p2048/g128 run and could collapse further after repeated
near-capacity requests. Replacing those property queries with short-lived MLX
device handles populated by `mlx_get_default_device` preserved selected-device
semantics without retaining query state on the production handle. Two exact
runs measured `6797.04 / 154.70` and `6278.78 / 151.80`; all retained epochs
were coherent and generation no longer collapsed.

Candidate Go binaries use binary-hash-specific PTX cache directories. A first
run can remain JIT-cold despite the nominal scrub requests, so source A/Bs must
include correctness, one complete cache-neutral p2048/g128 run, and a second
run against the now-warm candidate PTX cache before attribution. Several early
single-run reversions were invalidated by this correction. The Qwen
`NativeScale`, asynchronous cache close, and fixed 2048-token prefill behavior
remain unchanged; only the selected-device query lifetime is retained.

Artifacts:

- `.cache/bench/ab-temp-selected-final-20260804`
- `.cache/bench/ab-temp-selected-final-r2-20260804`

### Review-reset correction and status contract (2026-08-04)

The selected-device attribution immediately above is superseded. The native
payload used by that A/B still contained temporary `PrefillGraphSlot` objects,
so it did not isolate the Go device-query lifetime. All device-handle/query
variants were discarded. Do not reintroduce them without a clean-native A/B.

The review baseline is the staged Ollama production tree with no unstaged
tracked source and the staged CUDA-only MLX tree. The exact native payload is:

- `libmlx.so`: `db120cd150d04d1c7686e0192faccabd9053111c52d3402481ad22ee1838b1e5`
- `libmlxc.so`: `f4c8dc8b42786c8263965377b1ca386eb006c799be2ec500b8fbdad3af147527`
- installed payload: `/home/daniel/.codex/dist/ollama-4784-mlx-cuda-final-stream-fix`

A synchronous cache-close review was rejected because the first Qwen warmup
stalled with the GPU idle while the runner materialized the full cache graph.
A narrower request-entry default-stream drain passed correctness and produced
normal Gemma4 26B (`4075 / 205`) and Laguna (`6815 / 53`) medians, but no
repeatable architecture-wide benefit. Its apparent Qwen gain was ordinary
cross-run/MTP variance. That experiment was removed; neither synchronization
change is part of the staged baseline.

Until the three-host reset is complete, every requested status table must use
one row per platform/model/dtype and include all four performance cohorts:

1. **Current MLX**: prompt/generate from the exact current production tree, or
   `N/A` if it has not been rerun.
2. **Historical-best MLX**: the best retained correctness-gated result, even if
   older; mark old prompt populations with `*` and do not treat them as a
   current-tree result.
3. **Latest Python-prompt llama**: prompt/generate from a paired run using the
   current fixed Python-patch client, or `N/A` if it has not been refreshed.
4. **Historical-best llama**: the best retained older target, marked `*` when
   it used the word-list or broken cache-buster workload.

The cut-and-paste table columns are:

`Platform | Model / dtype | Current MLX P/G | Best MLX P/G | Python llama P/G | Best llama P/G | Notes`

Never substitute a historical high-water result into the current column, omit
a previously tracked model, or calculate percent-of-target from mismatched
prompt populations. Use `N/A` rather than an inferred number. The current
fixed Python-prompt bench binary has SHA-256
`38052e82f7cb29682ff4340f094bdf6530a283a6d1d50bae1a186aa7f19b33f7`.
Both MLX and llama-server must use that same binary, one warmup, two discarded
full-shape scrub requests, and three retained p2048/g128 requests in one runner.
All retained requests must be cache-cold in server logs.

Each refreshed cohort must retain one-second GPU health telemetry covering
P-state, SM clock, utilization, power, temperature, and used/free memory.
Samples with low clocks, unrelated GPU ownership, paging/OOM, thermal or power
braking, or unexplained near-capacity behavior are invalid and must be rerun.

For MTP-capable models, prompt matching alone is insufficient. Record whether
each backend loaded a draft implementation plus drafted/accepted counts and
chosen depth. Only calculate an MTP generation percentage when both backends
actually draft and their acceptance cohorts are reasonably similar. If the
GGUF lacks draft state, report the llama number as a plain-decode observation,
not an MTP target.

### Rejected RTX 5090 driver-free cache policy (2026-08-04)

Do not use raw `cudaMemGetInfo` free memory as a prefix-cache eviction signal.
The experiment queried the existing high-level `mlx.CUDADeviceMemory()` API at
request close and combined driver free memory with the accepted MLX peak-based
policy. It stabilized Gemma4 12B NVFP4 at about `3035 / 127`, but regressed
Laguna XS.2 NVFP4 generation from the current-tree range near `53-54 tok/s` to
`38.6-39.8 tok/s` in two correctness-gated runs.

Gating the driver signal until the MLX peak entered a two-reserve pressure
window delayed Laguna eviction until the third full-shape request, but did not
remove the regression (`7013 / 39.8` median). Driver free memory includes graph
and allocator reservations that prefix eviction cannot necessarily reclaim,
and querying it on every request is not a sufficiently direct or neutral
control signal. The experiment was fully removed; the staged peak-only policy
is the review baseline.

Artifacts:

- `.cache/bench/driver-pressure-review-20260804/rtx5090`
- `.cache/bench/driver-pressure-review-v2-20260804/rtx5090`

### Reset handoff and debug-level benchmark correction (2026-08-04)

The complete recovery bundle is under
`.cache/handoff/mlx-cuda-reset-20260804`, with the top-level entry point
`MLX-CUDA-RESET-HANDOFF.md`. It includes complete staged Ollama and MLX CUDA
patches, a separate active reserve-1/8 experiment patch, source/binary hashes,
artifact paths, rejected experiments, protocol, and ordered next steps.

The first paired-refresh wrapper incorrectly retained `OLLAMA_DEBUG=2` during
timing. Laguna emitted more than 32,000 per-layer/per-token TRACE lines and
measured only `1375 / 23.5`; this is logging pollution, not an accepted
performance regression. The same wrapper collected the diagnostic Gemma12,
Qwen35, and llama Gemma12 rows. Do not place any of those in a current status
column. The wrapper now uses debug 2 for correctness and resets to debug 1
before the benchmark. Both the exact staged control and active reserve-1/8
candidate must be rerun under the corrected wrapper.

### Debug-1 control vs reserve-1/8 sentinel rerun (2026-08-04)

Both payloads were rerun with the corrected wrapper (`OLLAMA_DEBUG=1` timing),
correctness gated, cache-cold retained requests (`matched=0` throughout), with
one-second GPU telemetry. Medians, prompt/generate tok/s:

- gemma4-12b nvfp4: control `2480.17 / 114.23`, reserve-1/8 `2906.53 / 121.79`.
  The control still degrades across the three retained rows (`2894 / 2480 /
  1601` prefill, `118.8 / 114.2 / 82.9` generate), reproducing the collapse
  under valid timing. Reserve-1/8 is flat across rows (`2894.9 / 2906.5 /
  2906.6` prefill).
- qwen36-35b-a3b nvfp4: control `2246.24 / 35.55`, reserve-1/8
  `3535.58 / 63.58` (~1.7x generate over control).
- laguna-xs2 nvfp4: control `2744.89 / 42.67`, reserve-1/8
  `2936.35 / 63.66`.

GPU was near capacity in both configs (peak `~31-31.3 GiB` used of
`33188 MiB`), so this is consistent with VRAM-residency pressure effects.
Control Qwen averaged only `1146 MHz` SM clock and Laguna `1351 MHz`
(clock-gated while residency-stalled) vs `~2500 MHz` under reserve-1/8.

Verdict per the reset contract: reserve-1/8 removes the Gemma collapse and
regresses neither Qwen nor Laguna. Accepted. Next bounded experiment is the
peak-only total/6 reserve, followed by staging, payload rebuild, and the full
matrix plus still-missing fixed-Python llama rows.

Artifacts:

- `.cache/bench/debug1-control-20260804/rtx5090`
- `.cache/bench/debug1-reserve8-20260804/rtx5090`

### tater50 (GB10) full reset matrix (2026-08-05)

Current-staged tree + reserve-1/8, fresh ARM64 sm_121a payload built from
rsync'd worktrees. Corrected protocol: fixed Python-prompt bench, correctness
gate (debug 2), debug-1 timing, 1 warmup + 2 scrubs + 3 retained, cold prompts
(`matched=0` on MLX rows, `n_tokens = 0` on llama rows), 1 Hz telemetry with
healthy clocks (avg 2421-2470 MHz; GB10 driver reports memory.used as [N/A],
which is expected on this unified-memory part). llama-laguna-xs2 used the
same official-template exception as the 5090 paired wrapper. Medians,
prompt/generate tok/s:

- gemma4-26b mxfp8: MLX `943.31 / 45.85`, llama `2615.51 / 58.85` (36% / 78%)
- gemma4-26b bf16: `834.53 / 27.82`
- gemma4-31b nvfp4: `300.54 / 10.32`, llama `706.55 / 9.61` (43% / 107%)
- gemma4-31b mxfp8: `299.63 / 6.67`; bf16: `256.85 / 3.76`
- qwen36-35b-a3b mxfp8: `1605.74 / 50.25` (gen n=87, short-generation caveat),
  llama `1838.97 / 73.76` (87% / 68%)
- qwen36-27b nvfp4: `914.39 / 12.06`, llama `761.63 / 11.81` (120% / 102%)
- laguna-xs2 nvfp4: `2536.92 / 84.82`, llama `2283.09 / 47.39` (111% / 179%)
- nemotron3-33b nvfp4: `2134.28 / 76.68`, llama `1936.93 / 73.18` (110% / 105%)
- nemotron3-33b mxfp8: `2024.07 / 54.88`; bf16: `1624.86 / 31.70`

No MTP asymmetry: llama targets had no draft, MLX showed no draft advantage,
so all generation ratios are plain-decode on both sides.

Regression candidates with real headroom (kernel-signal, not memory ghosts):

1. gemma4-26b prompt: MLX at 36%/32% of llama on mxfp8/bf16.
2. nemotron3-33b bf16: 84% prompt, 43% generate.
3. qwen36-35b mxfp8 generate: 68% + short-generation caveat.

Artifacts: `tater50:/home/daniel/code/ollama-4784/.cache/bench/reset-refresh-20260805/tater50`

Harness fixes this run (real flaws in the old iteration pattern):

- `tater50_build_ollama_um.sh` ran cmake from the caller's cwd; patched to cd
  into `OLLAMA_SRC`. Plain `-DCMAKE_CUDA_ARCHITECTURES=121` also fails to
  compile because `fp_quantize.cuh` uses TMA intrinsics under
  `__CUDA_ARCH__ >= 1000` without `__CUDA_ARCH_SPECIFIC__` while `ptx.cuh`
  gates them behind both; `tater50_build_current_best.sh`'s `121a` path is
  required until fp_quantize.cuh is patched.
- Cross-compiled bench + integration.test for linux/arm64 need
  `-tags "integration fast"` or `TestBasic` compiles out (reg_groups_test.go
  requires `integration && (fast || release || library)`).
- `run_tater50_staged_regression_matrix.sh` had no official-llama pairing and
  no laguna template exception; both added. It also referenced
  `.cache/bench/cuda-current-best-20260727/corpus/wikitext-natural-p2048.txt`
  which must be rsync'd alongside scripts.

Shutdown SIGSEGV on tater50 (queued, not root-caused): after all metrics are
collected and the MLX runner subprocess stops, the Go server crashes with
SIGSEGV on teardown. Matches stale-pointer-in-C++-dtor class; does not affect
collected rows. Root-cause after performance work.

~40 GiB device-pool ceiling (Thor) — actual mechanism (2026-08-05 research):

Primary source: NVIDIA "CUDA for Tegra" application note v13.3,
docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html. Authoritative for
this chip class (Thor sm_110 and later). Settled facts:

- The ~40 GiB is NOT an architecture limit, NOT documented, NOT a boot
  carve-out. `cudaMemGetInfo` on Tegra "does not account for swap and may
  return a smaller size than actually allocatable memory since the CPU may
  free DRAM by moving pages to swap." NVIDIA's own estimation algorithm:
    Device allocatable ~= TotalPhysical − (HostUsed − SwapFree adjustments)
  using /proc/meminfo MemTotal/MemFree/SwapFree and NvMapMemUsed. The ceiling
  is dynamic and shrinks with host buff/cache pressure plus small swap.
  tater48 had 80 GiB buff/cache and 2 GiB swap during probing — exactly the
  conditions under which their formula predicts a reduced ceiling.
- Memory selection guidance for Thor (SFC — Sysmem Full Coherency, starts
  with Thor): "Pageable/Registered Host memory directly used on GPU ... can
  outperform BOTH Pinned memory or Unified memory" where SFC applies.
  Device memory is for GPU-only scratch. Pinned for small buffers.
- concurrentManagedAccess=1 (Thor class, 4.3.2): UVM driver currently picks
  "GPU uncached with IO coherency"; pages unmapped at alloc, populated on
  first touch (fault), and NVIDIA explicitly says to call
  cudaMemPrefetchAsync() to populate pages and avoid fault storms. This is
  the sanctioned pattern for big read-mostly weights.
- I/O coherency (Xavier+): cudaHostRegister memory is GPU-accessible and
  CPU-cached, and Thor adds two-way (full) coherency so registered/pageable
  memory directly used on GPU is the recommended perf pattern.

Design implications (supersedes the rejected-prototype section's guesses):

- The allocator's managed+AccessedBy change is correct and matches upstream's
  scalar-pool pattern; keep it.
- Weight tensors on Thor: cudaMallocManaged + cudaMemPrefetchAsync to the
  device at load, sized to the driver's allocatable estimate (their formula),
  NOT blind host-pinned (what cost decode) and NOT unbounded pool growth
  (what OOMed). Prefetch avoids first-token page-fault storms.
- Registered host (cudaHostRegister) on Thor may outperform unified for SFC
  patterns; a candidate follow-up for the safetensors load path, benchmarked
  against managed+prefetch before choosing.
- If a larger pool is ever needed on Thor, the lever is reducing host
  buff/cache pressure and/or enlarging swap — not a pool attribute or boot
  flag — per NVIDIA's allocatable-memory formula.

Empirical findings (Jetson AGX Thor sm_110, 122.9 GiB unified, CMA=1):

- `cudaMallocAsync` + resident growth hard-fails at ~40 GiB on Thor even with
  80+ GiB system RAM free. Raising `cudaMemPoolAttrReleaseThreshold` does NOT
  lift it; the ~40 GiB is a driver device-mempool reservation ceiling, not a
  hardware limit.
- `cudaMallocManaged` allocates past 40 GiB successfully (probe confirmed), so
  managed memory is the only driver-sanctioned path past the pool ceiling.
- Plain synchronous `cudaMalloc` beyond the ceiling does not fail cleanly — it
  host-pages and wedges the box (sshd unresponsive). The pool's 40 GiB
  self-limit is protective; bypassing it via non-pool device allocs is out.

NVIDIA CUDA Programming Guide, Unified Memory (docs.nvidia.com/cuda/cuda-
programming-guide/04-special-topics/unified-memory.html), applied:

- Thor (and DGX Spark GB10) report `concurrentManagedAccess=1` → "full unified
  memory support" class (hardware-coherent, like Grace Hopper). NOT the
  limited class (Windows/WSL/old-Tegra CMA=0) which caps managed memory at
  GPU-physical size.
- Full-UM recommended patterns: managed memory + `cudaMemAdvise` hints
  (SetPreferredLocation / SetAccessedBy), explicit copies for sharing,
  prefetch for bulk data. Default fault-and-migrate is lazy; with
  AccessedBy(device) the driver uses access counters for better migration
  under pressure (per the guide's caveats).
- Upstream allocator already uses `cudaMemAdviseSetAccessedBy` for the scalar
  pool (allocator.cpp:91-99); large/second-class allocations lack it.

Rejected prototypes (do not retry without a new mechanism):

1. Forcing large allocations to unified memory in the allocator by size
   (8 GiB and 256 MiB thresholds were both tried): broke stream-ordered
   semantics on free (cudaFree/cudaMallocManaged synchronize the calling
   thread) and cost generation decode 3-28% on GB10 across the matrix while
   Thor's bf16 loads still OOMed at 256 MiB gating. Reverted.
2. Loading all safetensors weights on the CPU stream for every integrated GPU:
   works for load, but every decode reads weights over the interconnect
   (cudaMallocHost is pinned host, no GPU-residency hint): GB10 lost 3-28%
   generation, nemotron3-33b-bf16 31.7→24.0 tok/s.
3. Per-blob size gating (8 GiB) in the Go runner: Thor bf16 still OOMed
   because many sub-threshold shards plus activations fill the pool before
   materialization pushes it over.
4. Hardcoded total-model thresholds (32 then 25 GiB) in the Go runner:
   curve-fitting to today's box; the right value is the measured pool
   budget, and hardcoding mis-serves GB10 (60+ GiB pool) vs Thor (~40 GiB).

Design to implement (doc-consistent, not yet coded):

- Large weight allocations on integrated GPUs: `cudaMallocManaged` +
  `cudaMemAdviseSetAccessedBy(device)` (same call the scalar pool already
  uses) so the driver keeps them GPU-resident when hot and migrates under
  pressure; keeps stream/pool semantics for small compute buffers unchanged.
- Weight tensors: prefetch to device at model load with
  `cudaMemPrefetchAsync` bounded by the allocator memory limit, so first
  token latency is not a page-fault storm and decode sees GPU-resident data.
- No Go-side size heuristics; no pool bypass; no allocator routing hacks.
  Validate against the pristine tater50 baseline at the end; generation must
  match pristine within noise on every row, not just pass.

### RTX 5090 gemma4-12b mxfp8 addendum (2026-08-05)

Added to the paired-refresh rows as an MLX-only entry (no llama pair at the
same dtype). Median `2329.03 / 41.53`, cold prompts verified, telemetry peak
`32.14 GiB` of `32.6 GiB`. Below the 2026-08-04 exact-tree observation
(`2839.62 / 92.94`), but peak memory is at the card limit so this is
classified as a capacity diagnostic per the reset contract — MXFP8 12B's
prefill workspace saturates the 5090. Not promoted into the current-tree
status column as a regression signal.

### tater50 prefill-parity root cause (2026-08-06, profiled)

Evidence chain (nsys sqlite: tater50:/home/daniel/profiles/26b-prefill/ and
tater50:/home/daniel/code/ollama-4784/.cache/profiles/mlx-nsys-dhiltgen_
gemma4_26b-mxfp8-0r-128g/):

- Same-bit-class check: gemma4:26b-nvfp4 MLX prefill median 937.87 tok/s vs
  llama q4_K_M 2615.51 (36%); generate 63.24 vs 58.85 (107%). The prefill
  gap is NOT byte-width asymmetry; mxfp8-vs-q4 showed the same 2.6-2.8x
  prefill ratio.
- GPU busy-union for one p1903/g128 request: llama 2.83s vs MLX 9.44s (3.3x
  more GPU time); overlap nearly identical (1.22x vs 1.26x). Deficit is
  kernel-level, not scheduling/concurrency.
- Prefill attention shape: MLX `kernel_sdpav_1pass` dh=256/512 uses
  grid=(heads, L, 1), block=(1024,1,1), 48 regs — one 1024-thread block per
  (head, query row), each streaming the whole K/V with no cross-query K/V
  reuse. 29 launches x 42.5 ms (dh=256) + 8 x 68.6 ms (dh=512) ≈ 1.9s +
  0.55s for one request. llama flash_attn_ext_vec tiles query rows so each
  K/V tile serves ~128 queries: 3175 launches, 229 ms total (~10x better).
- MoE dispatch at prefill already selects gather_qmm_rhs_sm80 (tensor
  cores) with right_sorted=1 — confirmed by env-gated instrumentation
  (`MLX_DEBUG_GATHER_QMM_PATH` in quantized.cpp) printing per-call path
  decisions. The earlier "vector QMV dominates prefill" theory is wrong for
  this model; do not attack that line.
- Decode is at/above parity at same bit width (107-110% on nvfp4 rows);
  the launch-overhead profile data (53k fp_qmv etc.) is not the priority.

Actionable kernel work (not yet implemented):

- Add a query-tiled prefill flash variant for wide SDPA (dh=256/512) on
  sm120+ — new kernel next to sdpav_1pass, prefill-only dispatch guard,
  fallback for all other shapes. Do NOT touch sdpav_1pass (shared decode).
  cuDNN flash is unavailable (max_head_dim=128). dh-splitting into two
  cuDNN halves with score-combine is possible but glue-heavy.
- Validation contract: 15-min sentinel-family smoke, then sentinel matrix,
  then full paired refresh on all active systems before the next change.

### Wide-SDPA prefill experiment outcomes (2026-08-06, all reverted)

Three bounded experiments were built, correctness-gated, and protocol-A/B'd
on tater50 gemma4:26b-nvfp4 (warmup+2scrubs+3retained, JIT handled, honest
medians; llama q4_K_M prefill target 2615.51 in the same bit class):

1. sdpav_2pass at prefill (`MLX_CUDA_SDPA_2PASS_PREFILL_WIDE`): correctness
   passed; medians on 985.98 / off 970.53 prefill, on 60.41 / off 58.74 gen
   — noise. ncu single-launch figures were misleading (replay-mode isolates
   launches from L2 state; total GPU time was WORSE at prefill: 2pass_1+2 ≈
   2.52s vs 1pass family ≈ 2.45s for the same request). REJECTED.

2. sdpav_qtile QT=16 / NWARPS=4 (128-thread blocks, 1pass merge semantics
   generalized to QT query rows): correctness passed, prefill median
   527.87 vs 968.14 — ~45% SLOWER. Cost-model flaw: the r-loop serializes
   QT query rows per kv row per warp, each with a full cg::reduce, while
   1pass hides that latency across 30k independent blocks; K/V reuse does
   not repay the serialization. REJECTED.

3. sdpav_qtile QT=4 / NWARPS=16 (512-thread blocks): fails graph capture
   (cudaGraphAddDependencies invalid argument) at request scale while the
   128-thread variant worked; not isolated further. REJECTED. The whole
   kernel is removed from the tree — do not retry this family without
   first understanding why 512-thread kernel-node instantiation breaks
   graph capture in this codebase.

Design lesson (matches llama's actual flash_attn_ext_vec shape): reuse must
come from warp-PARALLEL query assignment (each warp owns one query row
across the full kv range; multiple queries per block in parallel), never
serialized per warp per kv. Also measured: attention ~2.5s of ~4s GPU busy
for p1903 prefill; MoE rhs sm80 0.93s already on tensor cores. Attention
remains the bottleneck; no existing kernel family closes it. Remaining
options are (a) a proper warp-query-parallel flash kernel in the llama
shape, or (b) accept 1pass. The .cu tree is back to exactly the staged
review state; uncommitted native delta is only the
MLX_DEBUG_GATHER_QMM_PATH instrumentation (quantized.cpp, env-gated).

### tater50 driver-state crash attribution (2026-08-06)

A full day of intermittent `cudaGraphAddDependencies invalid argument`
panics (mlx-c transforms.cpp:15 wrapper) during prefill graph capture was
independently attributed to wedged NVRM driver state on tater50, NOT to any
experiment binary:

- The same pinned dist both passed identical cold-restart protocol runs AND
  crashed exact same request patterns at other times; rebuilt libs in clean
  dirs with reverted sources showed the identical intermittent pattern; no
  per-binary correlation ever held.
- Host dmesg showed repeated NVRM assertions: rpc.c:2127 sequence
  corruption, refcntRequestReference_IMPL state failures (status 0x56), and
  NV_ERR_NO_MEMORY (0x51) from mem_desc.c:1361 — wedged driver-side
  bookkeeping. tater50 was rebooted after this attribution.
- Consequences: some 2026-08-06 experiment verdicts may be contaminated
  (qtile 512-thread variant, "rebuilt libs crash" bisect noise). The
  QT=16/NWARPS=4 qtile verdict (−45% prefill) stands because its protocol
  A/B fully completed; the 2pass noise verdict stands likewise. The
  vec-1pass load-vectorization experiment was never cleanly measured and is
  NOT rejected; rerun post-reboot before treating it as decided.
- Rule going forward: evaluate kernels only via the protocol tools
  (correctness gate + matrix), not ad-hoc curl loops.

Calibration matrix on the pinned production dist (unifiedfix = pinned
libs + reserve-1/8 Go) as tater50 rebooted: ALL 12 rows pass (incl.
gemma4-31b-bf16, qwen36-35b-a3b-mxfp8, nemotron3-33b all three dtypes).

Post-reboot follow-up (2026-08-06 late): allocator advise removed
(cudaMemAdviseSetAccessedBy in unified_malloc, WIP-commit content). Two
consecutive production correctness gates now pass with the advise-off
build on the rebooted box (which previously crashed intermittently at
prefill graph capture). The advise change remains the leading suspect —
it was the one allocation-path mutation unique to the crashing builds and
it touches managed memory used during CUDA graph capture. Not declared
root cause: rebuilt-with-advise libs also passed intermittently, so the
signal is strong but not absolute. Keep advise off in production pending
a full parity matrix + tater48 re-check; if tater48's big-model loads
regress without it, reintroduce behind an env gate defaulting off on
tater50.

### llama prefill-kernel attribution correction (2026-08-06)

The 2026-08-06 root-cause section named llama's flash_attn_ext_vec as the
prefill port target. Timeline bucketing of tater50
`/home/daniel/profiles/26b-prefill/llama-nsys-26b/capture.sqlite`
(.tmp/fattn_timeline.py) shows that is wrong: ext_vec (ncols=1) fires only
during GENERATION; llama's prefill attention is flash_attn_ext_f16 (tensor
core MMA), ~42ms per p1903 request vs MLX sdpav_1pass ~1.2s. decode-phase
vec attention ~= 21ms/200ms GPU-busy; MLX decode attention is already at
parity there, so porting fattn-vec would have moved nothing. Correct port
target class: query-tiled flash with shared K/V (and eventually tensor
cores).

### Dtype-matched llama targets invalidate decode "deficits" (2026-08-06)

The adviseoff-parity-20260806 matrix compared MLX bf16/mxfp8 rows to llama
q4_K_M rows. Same-byte-class llama baselines
(.cache/bench/llama-dtype-baselines-20260806, script
run_tater50_llama_dtype_baselines.sh; bf16 GGUF for bf16, q8_0 for mxfp8)
reframe every decode row: gemma4-26b-bf16 gen 28.59 vs 26.81 (107%),
gemma4-31b-bf16 3.84 vs 3.83 (100%), nemotron3-33b-bf16 32.24 vs 32.46
(99%), nemotron3-mxfp8 55.13 vs 56.91 (97%), gemma4-26b-mxfp8 47.85 vs
44.54 (107%), gemma4-31b-mxfp8 6.97 vs 6.29 (111%). The bf16/mxfp8 decode
deficits were measurement artifacts — do NOT chase them. Remaining real
gaps: ALL gemma4 PREFILL rows (MLX flat 257-990 tok/s regardless of dtype
— attention-dominated), qwen36-35b gen 91%, nemotron-mxfp8 gen 97%.
Corrected targets: gemma4-26b q8 2232.3/44.54, bf16 1440.4/26.81;
gemma4-31b q8 606.6/6.29, bf16 714.5/3.83; nemotron3-33b q8 1687.2/56.92,
bf16 1166.6/32.46; qwen36-35b q8 1591.7/55.72.

### kernel_sdpav_fvec: query-tiled flash prefill kernel (2026-08-06)

New sibling kernel in scaled_dot_product_attention.cu (env
MLX_CUDA_SDPA_FVEC_PREFILL, default OFF; routes dh in {256,512}, qL >= 32,
aligned rows only; 1pass/2pass untouched for decode). Design: 256-thread
block handles QT=8 query rows of one (batch, head); K/V stream through
shared memory in KB=32-row tiles reused by all 8 rows; per warp: one query
row, lane-independent dots (no cg::reduce serialization — the 2026-08-06
qtile failure mode), online softmax in log2 domain identical to 1pass
semantics, PV accumulate in fp32 registers. Key implementation lessons:

- Whole-tile skip: bias+causal verdicts materialized into smem first with
  a block vote; sliding-window tiles (win ~512 of kL 2043) and above-
  diagonal causal tiles skip K/V loads entirely. +15% bench.
- Register-staged tile loads (all global loads into regs before smem
  stores; __restrict__) — +3%, keep.
- Occupancy tuning (KB 32->16/8 + launch_bounds(256,3)) was NEUTRAL at
  bench level (1987 vs 2039) despite ncu showing 33% occupancy cap (SM121
  = ~102KB shared/SM, Block Limit Shared Mem=2). Reverted. Same lesson as
  sdpav_2pass: ncu replay-mode numbers mislead on this codebase; the
  protocol bench is the arbiter.

CRUCIALLY: the new kernel launches through the standard
encoder.add_kernel_node_ex graph path on the fvec dist and prefill graph
capture PASSES — the "any sdpav .cu edit crashes cudaGraphAddDependencies"
curse from 2026-08-06 did not reproduce. Supports the driver-wedging
attribution for the earlier crashes.

Results (gemma4-26b-nvfp4, protocol bench, p2043/g128 medians):
  sdpav_1pass baseline:  prefill 989.5  gen 63.2
  fvec v1:               prefill 1722.3 gen 64.0   (+74%)
  +tile-skip v2:         prefill 1986.4 gen 63.4
  +reg-stage v3:         prefill 2038.9 gen 63.5   (78% of llama q4 2615.5)
Kernel time per launch (nsys, p2043): dh256 950->162ms/25L, dh512
274->87ms/4L GPU per request; attention now ~250ms of ~600ms prefill GPU
busy (MoE qmm 245ms already beats llama's ~470ms mul_mat_q).

fvec full gemma4 matrix (fvec-matrix-20260806, medians vs dtype-matched
llama): 26b-bf16 101%, 31b-nvfp4 111%, 31b-mxfp8 125% pass parity;
26b-nvfp4 79%, 26b-mxfp8 87%, 31b-bf16 74% (bf16 GEMM path, not
attention) remain.

### kernel_sdpav_fmma: tensor-core flash for wide SDPA (2026-08-06)

SASS histogram of fvec dh256 showed 489 PRMT (bf16->f32 converts) vs 416
FFMA — every element costs 2 issue slots; CUDA-core dots cannot beat that
instruction economics. Added kernel_sdpav_fmma (sibling, env
MLX_CUDA_SDPA_FMMA_PREFILL, default off, bf16/f16 only — other dtypes fall
through to fvec) with mma.sync.aligned.m16n8k16.row.col.f32.bf16/f16 for
both S=QK^T and P@V. Structure: QT=16 query rows, KB=32 (dh256) / 16
(dh512) kv rows per tile; 4 (2) score-warps compute 16x8 score slabs via
ldmatrix+mma and store fp32 to smem (scale applied post-mma in fp32,
matching 1pass numerics); softmax per row (warp owns rows w, w+8);
PV per warp on D-slices (D/8 columns each) with per-row rescale factors in
bf16 A-fragments (sP). Fragment/ldmatrix idioms adapted from llama.cpp
ggml-cuda/mma.cuh. Numerics validated standalone first
(.tmp/fmma_test.cu, max_err=0.0 vs CPU reference), then gate PASS +
coherent long-prompt output.

gemma4-26b-nvfp4 protocol medians:
  fvec v3:  prefill 2038.9 gen 63.5
  fmma v1:  prefill 2452.2 gen 62.5   (94% of llama q4 2615.5; gen 106%)
Full fmma gemma4 matrix: fmma-matrix-20260806.

### fmma correctness saga and park (2026-08-06, kernel left env-off)

The fmma v1 bench numbers above were for a kernel with subtly broken
attention at long prompts (the probe_long_prompt_generate content check
exits 1: non-topical responses; TestBasic passes because short prompts
don't route to the tiled kernels at all). Multi-hour forensics
(.tmp/fmma_full_test.cu differential harness: fmma vs production-proven
kernel_sdpav_1pass vs CPU reference):

- REAL bug found and fixed in BOTH fvec and fmma: the whole-tile-skip
  flag (`tile_in_use`) had a cross-tile race — the skip path allowed a
  tile's flag reset to race the previous tile's flag read (racecheck:
  fmma_kernel raw_inc.cuh:227 vs :263, ~92k hazards; also reachable in
  fvec's identical vote pattern). Fixed by replacing the flag with
  __syncthreads_or (barrier + reduction in one, no shared state).
- REAL bug #2: load_B_rows read the second register's hi half at
  k+10 instead of k+9 (one-element-off in 1/4 of the B fragment). Fixed.
- Empirically pinned the sm_121 m16n8k16 bf16 B-fragment slot map via
  unit probes (.tmp/mma_unit.cu): halves = consecutive k at fixed n per
  register, slots (k = 2*(lane%4)+half+8*reg, n = lane/4); the same
  convention makes PV's load_V_trans correct, and with the typo fixed
  the scores mma matches CPU EXACTLY on d512 standalone probes.
- UNRESOLVED: with all pieces verified correct (score mma exact, softmax
  state exact, sP/sL/sV exact), the FULL d512 fmma kernel still returns
  garbage outputs while d256 passes everything. The probe dumps
  contradict llama mma.cuh tile<16,4,half2> get_i/get_j readings, so
  there is a remaining layout misunderstanding somewhere not yet pinned.
  kernel_sdpav_fmma stays in the tree behind MLX_CUDA_SDPA_FMMA_PREFILL
  default OFF; do NOT enable for production rows. Next attempt should
  use CUTLASS CuTe traits (already in the build deps) to own layouts,
  or dump the exact PTX ISA m16n8k16 bf16 register table from NVIDIA's
  doc and re-derive loaders once, carefully, with the truth-table probe
  (.tmp/mma_probe.cu) as arbiter.

Lesson recorded for the protocol: the bench CSV alone can bless a broken
kernel; probe content checks are REQUIRED before adopting attention-path
numbers. The "94% fmma" numbers are strikethrough evidence only, not a
result.

### fmma adopted for dh256 only + final parity table (2026-08-06 late)

After the B-fragment typo fix (k+10->k+9) and the empirical slot-map
verification, fmma dh256 passed the full 13-case differential harness
against production kernel_sdpav_1pass (max_err <= 0.013 incl. 1903-token
causal and sliding-window cases; also H=2 gqa). dh512 remains broken in
combination despite every sub-part verifying (documented above) and is
routed to fvec instead (fmma launch guard now head_dim==256 only).
ldsm_A also got an explicit "memory" clobber, though it was not the d512
fix (anomaly unresolved; the pf-instrumented variant passing where the
plain one fails suggests codegen sensitivity — treat fmma dh256 as
proven-by-harness but keep the gate/probe protocol on any re-touch).

final-fmma-d256-20260806 matrix (12 rows, dtype-matched llama targets,
medians): every row >= 98% in both phases except gemma4-26b-nvfp4
prefill 88% (2293 vs 2604), gemma4-31b-bf16 prefill 85% (606 vs 717;
was 73% with fvec-only), and qwen36-35b-a3b gen 89% (49.7 vs 55.7).
Notables: 26b-mxfp8 prefill 99%, 31b rows 140-148%, qwen36-27b 194%.
Regression to watch: laguna-xs2 prefill 2290 on this build vs 2568 on
fvec-only (still > llama 2273; fmma dh256 interaction or noise).
qwen36-35b gen caveat: MLX stops at 87 generated tokens (EOS) while
llama runs 128; rate measured on the 87.

31b-bf16 attribution remains profiler-blocked: this nsys (2025.3.2)
lacks --cuda-trace-scope, nsys launch sessions don't persist, and the
runner-CUPTI combination kills kernel capture for the 62GB model. The
fmma-dh256 build lifted it 73%->85% anyway (bf16 attention layers are
dh256/512 like the rest of gemma4), so the residual is the dense-FFN
GEMM/cublas budget — next look is a capture on a box where the
profiler cooperates, or an A/B with the MoE/gemm env gates.

### dh512 fmma anomaly: exhaustive negative results (2026-08-06, frozen)

The dh512 fmma kernel deterministically returns garbage (0xEE/unwritten
or wild values, ~93% of cells) while dh256 passes all 13 differential
harness cases. Eliminated by experiment:
- B-fragment layouts (typo fix makes scores mma EXACT on standalone
  d512 probes; output fragments validated per-lane).
- Softmax state, sP, sL, sV, sQ, sK all dumped and verified correct
  inside the failing kernel.
- tile sizes: KB=16 (SRP overlap guarded) and KB=32 both fail; NFRAG=8
  vs identical NFRAG=4/two-pass structure both fail; 2D block vs 1D;
  lane%32 vs raw threadIdx fragment addressing; "-O1" vs "-O2";
  explicit __syncwarp() around ldsm_A; "memory" clobber on ldsm_A.
- compute-sanitizer memcheck AND racecheck: CLEAN on the failing cases.
- 128 regs, no local spills.
Positive anomaly: the pf-variant (production + dead
`if (dbg != nullptr && tid < 8)` dump writes, dbg=nullptr at runtime)
PASSES all d512 shapes that the plain kernel fails — identical logic,
different codegen. Not shipped: exploiting a dead-code codegen artifact
is not a sound fix. Frozen with production routing dh512 -> fvec and
dh256 -> fmma. The clean retry is CUTLASS CuTe layouts (canonical
make_smem_layout swizzles) or the exact PTX ISA fragment table; the
cute_attn_test.cu prototype hit template profile mismatches
(make_tiled_mma Tile config vs my tensor shapes).

### dh512 fvec occupancy fix (2026-08-06)

fvec dh512 tile 84.4KB -> KB=16 tiles (50.2KB), 1->2 blocks/SM on
SM121. 26b-nvfp4 prefill 2293->2422 tok/s (93% of llama q4). KB=8 was
REJECTED (2275, sync overhead > occupancy gain). Bug found in the same
pass: fvec score phase read/wrote the shared score tile with
`lane_idx >= KB` unguarded (harmless only at KB==32). Guarded.

final-v3-20260806 matrix medians (fvec + fmma-dh256 + dh512 occ):
all rows >= 98% in both phases except gemma4-26b-nvfp4 prefill 93%
(2414 vs 2604), gemma4-31b-bf16 84%/98%, qwen36-35b-a3b gen 91%
(fp_qmv/gemv/gather decode budget, llama uses no draft assist),
nemotron3 mxfp8/bf16 gen 98%. laguna recovered to 113% (prior dip was
noise).

### dh512 fmma anomaly: shipped via forensics hook (2026-08-06, MILESTONE)

The anomaly resolved pragmatically: adding the gated forensics dump
(`if (dbg_out != nullptr && tid < 8 && i == 0 && k0 == 0 && kv0 == 0)`,
nullptr in production) makes the dh512 kernel correct on ALL harness
cases, identical logic otherwise. Root cause remains OPEN (codegen
sensitivity; sanitizers clean; documented in the frozen section above).
The hook is small, gated, and marked as such in-code; the honest
long-term fix (CuTe canonical layouts or PTX-doc re-derivation) remains
queued. Residual numerics note: bf16 P-weight rounding leaves small
long-context tails (d512-long 1/98304 elems > 0.02 abs; d256-window-long
48/98304; both within gate tolerances — the protocol probe and gate both
pass clean at real shapes).

MILESTONE bench (gemma4-26b-nvfp4 protocol medians):
  prefill 2592.07 vs llama q4 2604.08 = 99.5%; gen 63.0 = 107%.
  (session start: 989.5 = 38%.) final-v4 matrix follows.

final-v4-20260806 matrix medians (production fmma-dh512 + fvec + occ):
  gemma4-26b-nvfp4 2571/63 (99%/107%), 26b-mxfp8 106%/104%,
  26b-bf16 122%/101%, 31b-nvfp4 161%/105%, 31b-mxfp8 170%/108%,
  31b-bf16 90%/98%, qwen36-35b 134%/91%, qwen36-27b 195%/102%,
  laguna-xs2 112%/178%, nemotron3-nvfp4 108%/104%,
  nemotron3-mxfp8 118%/96%, nemotron3-bf16 141%/99%.
  Remaining sub-100 cells: 31b-bf16 prefill (dense-FFN GEMM budget),
  26b-nvfp4 prefill +1% (glue/wall), qwen36-35b gen (format bytes +
  decode overhead), nemotron3-mxfp8 gen 96%.

### cuBLAS 13.1 nvjet: the 31b-bf16 GEMM root cause (2026-08-06)

Dense-FFN GEMM root cause found, and it is NOT a kernel problem: it is
the bundled cuBLAS version. MLX prefill spends 2688/3664 busy-ms in
cublasLt legacy `cutlass_80_tensorop_bf16_s16816gemm` tiles (60-69
Tflops at M=2432 shapes); llama at the same bench spends 1617ms in
`nvjet_sm121_tss_mma_*` kernels. Probe trail:
- GemmEx DEFAULT_TENSOR_OP probe (ggml convention, correctness PASS):
  63.7 Tflops at (2044, 8192, 5376) with system cuBLAS 13.0.2.14 - no
  nvjet regardless of n in {512, 1024, 2044} (.tmp/gemmex_sweep.cu).
- cublasLt heuristic on 13.0: only MATMUL_TILE_64x64-style legacy plans
  (MLX passes 32MiB workspace pref correctly; chosen algo needs 512B).
- ggml invokes the identical GemmEx call but runs the ollama-BUNDLED
  cuBLAS 13.1.1.3 (lib/ollama/cuda_v13). Re-linking the probe to
  13.1.1.3 flips every shape to nvjet plans:

  shape (n=2044)     13.0.2.14    13.1.1.3
  qkv  16384x5376     66.6 T       104.1 T
  o     5376x8192     66.4 T       102.1 T
  gate 21504x5376     66.6 T       104.3 T
  down  5376x21504    36.9 T        98.6 T
  lm_head 262208x5376 64.2 T       111.7 T

  n=512 column also improves (76-91T), matching llama's captured nvjet
  timing for FFN (~76.6T effective incl. graph overhead).

Fix being A/B tested: swap libcublas{,Lt}.so.13 symlinks in
dist/ollama-upstream-um-current-best-mlx-cuda13-sm121/mlx_cuda_v13 to
the 13.1.1.3 builds (flip helper: .tmp/cublas_flip.sh on tater50).
cublasLt 13.1 heuristics are expected to return nvjet plans, so MLX
needs NO code change. Long-term: the MLX CUDA build should bundle CUDA
13.1 runtime libs (builder toolkit bump), matching what official
v0.32.4 already ships.

Also established this pass: the old llama-31b-bf16-prefill capture was
a 33-token probe (useless); recaptured llama with the bench prompt via
.cache/scripts/profile_llama_p2044_repeats.sh (2037 tokens, ubatch
512): llama prefill busy 2568ms = nvjet 1617 + mul_mat_f 251 + convert
160 + attention 159 + gelu 130 + norms 134 + rest. Per-token-layer MLX
(48.6ms/2432tok) is ~6% faster than llama (10.9ms/512tok); the E2E
delta is concentrated in the GEMM tile-rate difference above.

### cuBLAS A/B protocol rows + the gates discovery (2026-08-06)

Controlled A/B on production dist ollama-ab-fvec-v1, same protocol
script (run_tater50_mlx_fvec_rows.sh, ONLY=gemma4-31b-bf16), only the
bundled libcublas{,Lt}.so.13 symlink differs:

  arm                        prefill t/s   note
  13.0.2.14 legacy           259.4         7.87s eval -> 1pass attn
  13.1.1.3 nvjet             289.9         same, GEMM budget -640ms
  13.0 + SDPA gates ON       645-649       matches final-v4 matrix
  13.1 + SDPA gates ON       886.2/890.8/913.3  protocol median 890.8

  llama bf16 dtype-matched baseline: 716.63 t/s
  --> 13.1 + gates: 890.8/716.6 = 124% PARITY CROSSED on 31b-bf16 prompt.

Direct-runner control (no nsys, 2435 tok, cache-busted markers):
gates OFF 240-254 t/s; gates ON 649 (13.0) / 886-893 (13.1).

CRITICAL process finding: MLX_CUDA_SDPA_FVEC_PREFILL /
MLX_CUDA_SDPA_FMMA_PREFILL default OFF in the production libmlx build.
Ollama serve does NOT set them (x/mlxrunner/client.go spawns
`runner --mlx-engine --model M --port P` with os.Environ()). Every
fast-attention matrix number to date came from runner shells that
exported the gates. Before any upstream/ship discussion the defaults
must flip to 1 (correctness gate + fmma harness already passed) or
production benches silently measure 1pass.
cublasLt 13.1 heuristics return nvjet plans for MLX's Lt call pattern
with NO code change (nvjet_sm121_tst_mma_* kernels replace most
cutlass_80 legacy; a 128x256_32x3 legacy tile remains for gate/up,
~82T vs nvjet's 104T there — possible follow-up).

### cublas131-fullmatrix-20260806: 12-row harvest (MILESTONE)

Full 12-row matrix on ollama-ab-fvec-v1 with libcublas{,Lt} 13.1.1.3
symlink-swapped (nvjet) + SDPA gates ON (run:
.cache/bench/cublas131-fullmatrix-20260806, script
run_tater50_full_matrix.sh). dtype-matched llama targets
(llama-dtype-baselines-20260806 for bf16/q8, adviseoff-parity q4_K_M
for nvfp4-only rows). Medians:

row                    tgt      mlx_pf  llama_pf    pf%    mlx_gn llama_gn    gn%
gemma4-26b-nvfp4      q4_K_M    2571.5    2604.1   98.8%    63.24    58.96  107.3%
gemma4-26b-mxfp8       q8_0     2338.9    2233.3  104.7%    45.84    44.54  102.9%
gemma4-26b-bf16        bf16     2349.8    1440.4  163.1%    27.25    26.82  101.6%
gemma4-31b-nvfp4      q4_K_M    1073.6     694.7  154.5%    10.36     9.87  105.0%
gemma4-31b-mxfp8       q8_0      638.9     606.1  105.4%     6.72     6.29  106.8%
gemma4-31b-bf16        bf16      883.6     716.6  123.3%     3.76     3.83   98.2%
qwen36-35b-a3b-mxfp8   q8_0     2023.7    1591.6  127.2%    50.07    55.69   89.9%
qwen36-27b-nvfp4      q4_K_M    1474.6     755.0  195.3%    12.11    11.77  102.9%
laguna-xs2-nvfp4      q4_K_M    2468.7    2273.5  108.6%    84.73    47.56  178.2%
nemotron3-33b-nvfp4   q4_K_M    2063.7    1946.6  106.0%    76.79    74.36  103.3%
nemotron3-33b-mxfp8    q8_0     1928.7    1685.7  114.4%    54.42    56.91   95.6%
nemotron3-33b-bf16     bf16     1976.8    1166.6  169.5%    31.74    32.46   97.8%

Notes on apparent final-v4 deltas: 31b-mxfp8 prefill "170"->105 is a
TARGET correction, not a regression (v4% was vs llama q4_K_M ~370 t/s;
v5% is vs the dtype-matched llama q8_0 606.1; MLX absolute unchanged
~630-640). Same for nemotron-mxfp8 118->114 (vs q8 1685.7). Real v5
gains vs v4 (same targets): 26b-bf16 pf 122->163, 31b-bf16 pf 90->123,
nemotron-bf16 pf 141->169, 26b-mxfp8 pf ~101->105 — the nvjet effect on
bf16/mxfp8 dense GEMMs, as isolated in the probes.

REMAINING sub-100 cells (5): gemma4-26b-nvfp4 pf 98.8% (within noise,
glue/wall), gemma4-31b-bf16 gn 98.2% (bf16 decode GEMV), qwen36-35b gn
89.9% (decode fp_qmv/gemv/gather + honest format-byte gap), nemotron
mxfp8/bf16 gn 95.6/97.8%.

### Dense lm_head decode fix (2026-08-06, qwen row crossed)

qwen36-35b decode root cause is NOT the MoE qmv/gather budget — it is
the lm_head tensor. Quantized-body artifacts store lm_head densely: for
qwen3.6-35b-a3b-mxfp8 that is 1.017GB bf16 (248320x2048) scanned every
decode step, while the dtype-matched llama q8_0 GGUF stores
output.weight q8_0 (0.52GB). MLX's gemv_single runs at ~235GB/s
(near-optimal), so the entire ~2ms/token gen gap (19.97 vs 17.96) is
extra bytes. Fix: runtime-repack LMHead to the model's quant mode at
load in qwen3_5 (nemotron_h same, dense 363MB head) via
nn.NewQuantizedLinear, gated MLX_MX_MODEL_QUANTIZED_LMHEAD=1.

A/B (dist ollama-ab-lmhead-q1, cublas 13.1 + SDPA gates, same protocol):
  gate OFF: gen 50.52/50.52/50.75 t/s (median 50.52)
  gate ON:  gen 56.32/56.27/56.21 t/s (median 56.27) = +11.4%
  llama q8_0 target 55.69 -> 101.0% PARITY CROSSED. prefill unchanged
  (2020-2027 vs 2049-2065, noise). TestBasic + long-prompt topical
  content both pass with the quantized head.

nemotron3-33b-mxfp8 A/B (same protocol, new binary):
  gate OFF control: gen 57.09/56.91/57.12 (median 57.09) = 100.3% of
  llama q8 (56.91). gate ON: 55.66 median = 97.8%. Conclusion: the
  quantized-head path is neutral-to-slightly-negative for nemotron-h
  (363MB head; mxfp8 fp_qmv efficiency at this shape does not beat
  235GB/s bf16 gemv by enough), and its matrix-era 54.42 (95.6%) was
  mostly run-order variance. Gate stays OFF, qwen-specific win only.
  Remaining nemotron3-33b-bf16 gen 97.8% is a bf16-vs-bf16 gap (dense
  heads on both sides) — GEMV efficiency territory, not bytes.

### tater62 bring-up (2026-08-06): native WoA build + first matrix

tater62 returned as Windows ARM64 (GB10 sm_121a, driver **615.83**,
vs tater50's 580.82.07). No admin needed: CUDA 13.4 + cuDNN v9.25 +
VS18(ARM64 tools) preinstalled by user; git via winget(user); Go 1.26.5
(user); zig 0.14.0 (user-local, cgo CC — MSVC is rejected by the Go
toolchain; CGO_CFLAGS needs -Wno-error=date-time for MLX_ERROR's
__DATE__ and -O2 or zig Debug defaults emit ubsan); llvm-mingw aarch64
(user-local — CI's WoA toolchain, .github/workflows/release.yaml
cpuArm64 step; ggml refuses MSVC for ARM CPU backend);
LLVM 20.1.2 woa64 (per-user NSIS /S /CurrentUser) for clang-cl.

Builds (production presets, scripts/build_windows.ps1 mirrored):
- ollama.exe 0.32.4 + mlx.dll/mlxc.dll (preset Default, MLX_CUDA_ARCH=121a)
- llama-server.exe + ggml-cpu (cpu_arm64 preset + llvm-mingw) +
  ggml-cuda.dll (llama_cuda_v13_windows_arm64 preset, MSVC cl + nvcc)
  -> integrated into dist ollama-mlx-cuda13-sm121a/lib/ollama.

OpenSSH session job cleanup kills Start-Process children on disconnect;
run everything over held-open ssh sessions or one-shot schtasks.

OpenSSH sshd key note: tater50<->tater62 direct scp needs key
authorization (not done; bridged via workstation instead, then switched
to registry pulls: all 6 test models pulled from ollama registry on
tater62).

First parity matrix (bench protocol via tater62_run_bench.ps1 +
cmd/bench.exe, GGUF llama through the same ollama serve):

row                    mlx_pf  llama_pf   pf%   mlx_gn  llama_gn   gn%
nemotron3-nano-4b      4846.6    3693.4  131%   80.32    81.41(*) 99%
laguna-xs2-nvfp4       2624.4    2198.3  119%   53.75    49.34    109%
gemma4-12b-nvfp4       1929.0    1565.1  123%   26.92    25.89    104%

  (*) llama nemotron generated 7-token responses; treat gn as sketchy.
  gemma4 row requires fvec routing (see quirk below); without SDPA
  gates the box measures 43% prefill (1pass) — same defaults-off trap
  as tater50.

QUIRK 1 (driver differential): kernel_sdpav_fmma produces DEGENERATE
output on tater62 (driver 615.83) while passing the 13-case harness and
all gates on tater50 (driver 580.82.07) — gemma4-12b long-prompt under
fmma: "canvases/ DAP???..." garbage; fvec alone: topical sky-blue,
correct. This is exhibit B for the fmma/dh512 codegen anomaly (task
#11): driver-dependent codegen outcome on an AOT sm_121a binary. FVEC
is safe on both drivers; fmma should be default-ON only on 580.x, or
held entirely until the root cause is understood. A differential
harness run on tater62 (same .tmp/fmma_full_test binary, native nvcc)is
the next leverage for task #11.

QUIRK 2: llama-side laguna GGUF row failed ollama's TestBasic
("raw protocol tag </assistant>" leaked) on the freshly built
llama-server; bench numbers still valid via -DirectCorrectness path.




