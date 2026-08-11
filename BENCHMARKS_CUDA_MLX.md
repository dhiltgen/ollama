# MLX CUDA Current-Best Benchmarks

Date: 2026-07-27 to 2026-08-04

Shape: p2048/g128, context 4096, one warmup, three measured epochs.
Rates are medians. Percent of target is `MLX / llama-server * 100`.
Every accepted row passed `TestBasic/blue-sky` before benchmarking.

Candidate:

- Ollama staged tree: `baca1640b85c972af37e462b23fe1d66cad6d950`.
- MLX staged tree: `132e79f8023286069203e55560587cf3adac6e87`.
- MLX backend: `mlx_cuda_v13`.
- Gemma4 CUDA wide-head SDPA is selected by the reviewed model policy.

Target:

- Official Ollama v0.32.4 Linux artifact.
- llama-server backend: `cuda_v13`.

## Exact-tree validation (2026-08-04)

This is the authoritative current-source inventory. It does not mix historical
best rows from older staged trees. Percent-of-target values remain omitted
until a matching current-protocol llama-server row exists. The historical
sections below remain experiment context only.

| Platform | Model | MLX dtype | Prompt | Generate | Generated | Status |
| --- | --- | --- | ---: | ---: | ---: | --- |
| RTX 5090 | Nemotron3 4B | NVFP4 | 20,862.00 | 335.86 | 73 | correct; short generation |
| RTX 5090 | Gemma4 12B | NVFP4 | 3,032.51 | 127.68 | 128 | correct |
| RTX 5090 | Gemma4 12B | MXFP8 | 2,839.62 | 92.94 | 128 | correct |
| RTX 5090 | Gemma4 26B | NVFP4 | 4,017.98 | 218.74 | 128 | correct |
| N1x | Qwen3.5 2B | NVFP4 | 10,557.99 | 114.51 | 128 | correct |
| N1x | Qwen3.5 2B | MXFP8 | 9,365.85 | 80.13 | 128 | correct |
| N1x | Qwen3.5 4B | NVFP4 | 4,236.06 | 60.71 | 128 | correct |
| N1x | Qwen3.5 4B | MXFP8 | 3,670.81 | 40.89 | 128 | correct |
| N1x | Gemma4 E2B | NVFP4 | 4,818.15 | 112.34 | 128 | correct |
| N1x | Gemma4 E2B | MXFP8 | 4,071.11 | 78.74 | 52 | correct; short generation |
| N1x | Nemotron3 4B | NVFP4 | 4,459.47 | 73.51 | 73-122 | correct; variable short generation |
| DGX Spark | Gemma4 26B | MXFP8 | 921.43 | 45.22 | 128 | correct |
| DGX Spark | Gemma4 26B | BF16 | 832.47 | 27.72 | 128 | correct; idle repeat |
| DGX Spark | Gemma4 31B | NVFP4 | 302.41 | 10.36 | 128 | correct |
| DGX Spark | Gemma4 31B | MXFP8 | 303.76 | 6.79 | 128 | correct |
| DGX Spark | Gemma4 31B | BF16 | 259.64 | 3.77 | 128 | correct |
| DGX Spark | Qwen3.6 35B-A3B | MXFP8 | 1,588.65 | 50.30 | 87 | correct; short generation |
| DGX Spark | Qwen3.6 27B | NVFP4 | 949.37 | 12.09 | 128 | correct |
| DGX Spark | Laguna XS.2 | NVFP4 | 2,606.61 | 84.74 | 128 | correct |
| DGX Spark | Nemotron3 33B | NVFP4 | 2,060.46 | 79.07 | 128 | correct |
| DGX Spark | Nemotron3 33B | MXFP8 | 1,957.22 | 56.22 | 128 | correct |
| DGX Spark | Nemotron3 33B | BF16 | 1,669.89 | 32.37 | 128 | correct |

RTX Gemma4 12B BF16 does not fit the exact p2048/g128 workload: prefill
reaches about 30.30 GiB before the next allocation fails. Repeated OOMs left
the RTX 5090 in a low-power state, so Qwen3.6 and Laguna current-tree rows are
pending a full host power cycle rather than represented by degraded numbers.
An ordinary Windows restart left the Qwen control capped at 907 MHz and
`1364.61 / 48.17`; do not benchmark the remaining rows in that state.

The N1x hard power cycle recovered the same low-clock failure: its Qwen3.5 2B
control returned to `9673.39 / 111.39` before the matrix above. All seven rows
passed the integration and independent long-prompt correctness gates. Full
g128 rows used three measured epochs with cold-cache log evidence. Gemma4 E2B
MXFP8 ended at 52 tokens in all retained epochs, while Nemotron3 varied from
73 to 122; their generation rates are useful observations but not canonical
g128 comparisons. Nemotron's only non-cold event was the independent
long-prompt check (`2030` total, `9` matched), not a retained benchmark epoch.

## RTX 5090

| Model | Quant pair | MLX prompt | llama prompt | Prompt target | MLX generate | llama generate | Generate target |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemma4 12B (MLX MTP) | NVFP4 / Q4_K_M | pending refresh | pending refresh | pending | pending refresh | pending refresh | pending |
| Gemma4 26B | NVFP4 / Q4_K_M | 4,443.69 | 4,270.52 | 104.1% | 244.94 | 195.07 | 125.6% |
| Qwen3.6 35B-A3B | NVFP4 / Q4_K_M | 9,646.35 | 5,770.77 | 167.2% | 160.29 | 228.14 | 70.3% |
| Laguna XS.2 | NVFP4 / Q4_K_M | 5,884.71 | 7,166.13 | 82.1% | 54.04 | 195.46 | 27.6% |

Protocol correction (2026-08-03): earlier 5090 percentages are not current
targets unless both raw logs prove cache-cold requests from the same prompt
generator. A raw-log audit found that the July 27 campaign did use one shared
bench binary and every retained p2048 request did full prompt work on both
backends. Those comparisons remain valid for their old word-list workload, but
cannot be mixed with the newer Python workload because the prompt population
changes expert routing, generation trajectories, and MTP behavior. The current
Gemma4, Qwen, and Laguna rows use a byte-identical, tokenizer-calibrated
exact-2,048-token prompt with `cache_prompt=false` on warmup, scrub, and
retained requests.
Every MLX request logged `cache bypass` and created no snapshot; every
llama-server request began with `cached n_tokens = 0`. Gemma4 12B remains
pending because its old row predates this stricter protocol.

The corrected artifacts are under `.cache/bench/raw-prompt-telemetry/` with
labels `mlx-qwen36-35b-a3b-exact-p2048-cache-disabled-v1`,
`llama-qwen36-35b-a3b-exact-p2048-cache-disabled-v1`,
`mlx-laguna-exact-p2048-cache-disabled-rewind-v5`,
`llama-laguna-exact-p2048-cache-disabled-v1`,
`mlx-gemma4-26b-chat-nothink-exact-p2048-cache-disabled-v3`, and
`llama-gemma4-26b-chat-nothink-exact-p2048-cache-disabled-v1`.

The earlier Gemma4 12B, Qwen, and Laguna narrative and percentages are retained
in `notes/MLX-CUDA-UM-QMM-EXPERIMENTS.md` as historical experiment data only.
They must not be promoted back into this current Python-workload table without
a matched rerun under the corrected cache-cold protocol. The current Qwen target has no active
draft implementation, and the MLX log has no MTP evidence, so its generation
gap is currently classified as MoE decode rather than acceptance-rate skew.
The official Laguna target passed independent semantic checks but emits raw
assistant protocol tags in the current integration test; its template defect
is separate from the measured performance target.

The other large-model rows do not sustain p2048 fully resident in 32 GiB:

- Gemma4 31B NVFP4, Qwen3.6 27B NVFP4, and Nemotron3 33B NVFP4 OOM during the long-prompt workload.
- Gemma4 26B MXFP8 also OOMs. The other MXFP8 artifacts are 29.03-36.38 GiB before runtime allocations.
- No BF16 model in this large-model set fits.

## tater50 (refresh pending)

GPU: NVIDIA GB10, compute capability 12.1, 128 GiB unified memory.

The rows below predate the fixed cache-buster. Treat every percentage as
historical and invalid until both candidate and llama-server are refreshed
with the current fixed-body client and cold-cache log evidence.

| Model | Quant pair | MLX prompt | llama prompt | Prompt target | MLX generate | llama generate | Generate target |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemma4 26B | NVFP4 / Q4_K_M | 2,941 | 2,730 | 107.7% | 64.20 | 60.6 | 105.9% |
| Gemma4 26B | MXFP8 / Q8_0 | 2,498.95 | 2,344.95 | 106.6% | 49.81 | 45.61 | 109.2% |
| Gemma4 26B | BF16 / BF16 | 2,460.39 | 1,519.10 | 162.0% | 26.57 | 26.14 | 101.6% |
| Gemma4 31B | NVFP4 / Q4_K_M | 1,160.35 | 709.66 | 163.5% | 10.30 | 9.77 | 105.4% |
| Gemma4 31B | MXFP8 / Q8_0 | 673.49 | 601.22 | 112.0% | 6.87 | 6.30 | 109.0% |
| Gemma4 31B | BF16 / BF16 | 984.26 | 720.71 | 136.6% | 3.75 | 3.82 | 98.2% |
| Qwen3.6 35B-A3B | NVFP4 / Q4_K_M | 2,636.37 | 2,225.52 | 118.5% | 76.95 | 77.03 | 99.9% |
| Qwen3.6 35B-A3B | MXFP8 / Q8_0 | 2,346.15 | 1,972.06 | 119.0% | 56.46 | 55.62 | 101.5% |
| Qwen3.6 27B | NVFP4 / Q4_K_M | 1,618.88 | 805.2 | 201.1% | 13.16 | 12.30 | 107.0% |
| Laguna XS.2 | NVFP4 / Q4_K_M | 3,021 | 2,886 | 104.7% | 69.82 | 48.16 | 145.0% |

The Qwen3.6 rows include the CUDA gated-delta vector-load fast path. For the
model's Dk=128 shape, each thread loads four contiguous BF16 Q/K values and
four contiguous FP32 recurrent-state values with aligned 64- and 128-bit
transactions. The generic path is retained for other shapes and dtypes, and
the Metal source is unchanged. The pre-release N1x is device-guarded onto the
original scalar path after a direct A/B found the vector loads slower there.
Focused numerical tests pass on both GB10's vector path and N1x's scalar path.
`TestBasic/blue-sky` passed for the dense NVFP4 model and the MoE NVFP4 and
MXFP8 models before the accepted tater50 benchmarks.

For Qwen3.6 35B-A3B MXFP8, the p2048 gated-delta kernel time fell from 199.46
to 131.63 ms. Its median improved from 2,261.55 / 56.35 to 2,346.15 / 56.46
tok/s. The NVFP4 median improved from 2,450.72 / 77.23 to 2,636.37 / 76.95;
the standard-shape generation difference is within run noise, while a
p32/g256 decode-focused run improved from 77.75 to 78.66 tok/s. Dense Qwen3.6
27B improved from 1,439 / 12.82 to 1,618.88 / 13.16 tok/s.

## tater62 (refresh pending)

GPU: pre-release NVIDIA N1x, compute capability 12.1, Windows ARM64.
Target: official Ollama v0.32.5 Windows ARM64 artifact.

The rows below used the pre-fix `origin/bench-prompt` client. Treat every
percentage as historical and invalid until both candidate and target are
refreshed with the current fixed-body client and cold-cache log evidence.

| Model | Quant pair | MLX prompt | llama prompt | Prompt target | MLX generate | llama generate | Generate target |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5 2B | NVFP4 / Q4_K_M | 2,366.98 | 1,428.57 | 165.7% | 41.60 | 31.37 | 132.6% |
| Qwen3.5 2B | MXFP8 / Q8_0 | 2,331.52 | 1,533.80 | 152.0% | 41.10 | 32.56 | 126.2% |
| Qwen3.5 4B | MXFP8 / Q8_0 | 823.34 | 697.22 | 118.1% | 23.47 | 19.94 | 117.7% |
| Qwen3.5 4B | NVFP4 / Q4_K_M | 1,147.90 | 665.13 | 172.6% | 33.70 | 20.36 | 165.5% |
| Gemma4 E2B | NVFP4 / Q4_K_M | 2,547.79 | 1,325.03 | 192.3% | 30.21 | 24.60 | 122.8% |
| Gemma4 E2B | MXFP8 / Q8_0 | 2,947.20 | 1,335.38 | 220.7% | 27.43 | 24.59 | 111.6% |
| Gemma4 12B MTP | NVFP4 / Q4_K_M | 195.08 | 314.56 | 62.0% | 10.26 | 11.64 | 88.1% |
| Qwen3.6 35B-A3B | NVFP4 / Q4_K_M | 81.22 | 794.07 | 10.2% | 6.38 | 34.65 | 18.4% |
| Qwen3.6 35B-A3B | MXFP8 / Q8_0 | 70.81 | pending | pending | 3.45 | pending | pending |
| Qwen3.6 27B | NVFP4 / Q4_K_M | 360.5 | 313.8 | 114.9% | 8.04 | 9.78 | 82.2% |
| Laguna XS.2 | NVFP4 / pending | 85.54 | pending | pending | 3.18 | pending | pending |
| Gemma4 26B | NVFP4 / Q4_K_M | pending | 361.1 | pending | pending | 12.5 | pending |

N1x phase acceptance is based on graph-enabled models that fit with headroom
inside the device's reported 8 GiB GPU aperture. Every such row with a matched
baseline is above target: Qwen3.5 2B/4B in NVFP4 and MXFP8, and Gemma4 E2B in
NVFP4 and MXFP8. Gemma4 12B, Qwen3.6 27B/35B, Laguna, and larger rows exceed or
press against that aperture and exercise Windows pageable-memory placement,
graph-workspace pressure, or load-time allocator controls. Retain those as
capacity/platform diagnostics, not blockers for the CUDA-kernel parity phase.
Do not reboot or retune shared kernels for a slow near-capacity row unless a
comfortably fitting control also reproduces the slowdown after unload.

The Qwen3.5 4B and Gemma4 E2B rows are graph-enabled, correctness-gated
measurements from fresh candidate and official-server processes. Both passed a
short direct question and an independent natural-language long-prompt check
before the p2048/g128 benchmark. Both models substantially exceed llama-server
for prompt and generation on N1x.

The Qwen3.5 2B rows use the detached `origin/bench-prompt` client at
`80eb0c8b5` for both MLX and llama-server. Its unique Python-patch continuation
prompts avoid the repetitive word-list workload and accidental prefix-cache
reuse without changing either server binary. The NVFP4 medians were
2,366.98 / 41.60 tok/s against Q4_K_M at 1,428.57 / 31.37. The MXFP8 medians
were 2,331.52 / 41.10 tok/s against Q8_0 at 1,533.80 / 32.56. Both MLX
checkpoints passed the real `TestBasic/blue-sky` integration test and an
independent 1,984-token natural-language prompt before timing. The first MXFP8
process timed out during correctness with sustained GPU activity but no OOM or
panic; the one permitted clean retry passed all gates and produced three
complete epochs.

Qwen3.5 4B MXFP8 is the upper graph-enabled dtype control in the corrected
workload. It passed both correctness gates and measured 823.34 / 23.47 tok/s
against Q8_0 at 697.22 / 19.94, or 118.1% / 117.7% of target. The smaller
margin than the 2B MXFP8 row is useful evidence that N1x capacity and graph
workspace pressure become material before graphs fail outright.

Gemma4 12B MTP passed both correctness gates with CUDA wide SDPA enabled. Its
MLX median was 195.08 / 10.26 tok/s against Q4_K_M at 314.56 / 11.64. A
debug-only follow-up showed that the speculative controller was not
artificially inflating or depressing the measured rate: a depth-1 probe was
accepted, but cost 364 ms versus 85 ms for depth 0, yielding 5.5 versus
11.8 expected tok/s, so steady-state decode correctly stayed at depth 0.
MLX peak memory was already 7.35 GiB after load and reached 11.45 GiB during
the p2048/g128 process, beyond the N1x's reported 8 GiB GPU aperture. Treat
this as a near-capacity graph/memory row rather than evidence of a general
small-model CUDA graph gap.

Gemma4 E2B MXFP8 passed both correctness gates. Python-patch variation 0
deterministically completed after 52 tokens in both candidate attempts, while
variations 1-3 reached the requested 128 tokens. To avoid using a short
generation sample, the reported candidate and target medians use the same
matched variations 1-3. They were 2,947.20 / 27.43 tok/s for MLX and
1,335.38 / 24.59 for Q8_0.

Gemma4 E4B NVFP4 is excluded from the N1x table. Two clean candidate processes
failed during one-token preload with the same `cudaMallocAsync` out-of-memory
error before correctness. Its 9.6 GB artifact is above the reported 8 GiB GPU
aperture before runtime and graph workspace, so it is not a graph-fit control.

The first Gemma4 E2B run measured only 678.63 prompt tok/s because the N1x
helper omitted `OLLAMA_MLX_CUDA_WIDE_SDPA=1`. That silently selected Gemma4's
explicit non-Metal `matmul/softmax/matmul` fallback for its 256-wide attention
heads. A matched profile found only 16 CUDA graph launches during prefill and
healthy graph-cache updates, while the explicit 2,030 by 2,030 attention score
matrix was dominated by FP32 adds and BF16/FP32 conversions. Enabling the
existing CUDA wide-SDPA path preserved short and natural-long correctness and
produced measured rows of 2,547.79 / 26.99, 3,979.99 / 31.01, and
1,365.66 / 30.21 tok/s. The standard median is reported above; its prompt
variance should be considered when comparing small changes.

Qwen3.5 9B is not an accepted benchmark row. Its graph-enabled candidate passed
correctness and could measure about 806-850 prompt tok/s and 19-21 generate
tok/s when it loaded, but two fresh processes failed during cuDNN graph capture
near the local memory-workspace limit. This is a capacity/fragmentation case,
not a stable target for kernel-performance decisions.

The Qwen3.6 row uses the dense `dhiltgen/qwen3.6:27b-coding-nvfp4`
artifact for MLX and `qwen3.6:27b` for llama-server; the coding variant only
changes metadata, not the base model. Both paths passed `TestBasic/blue-sky`.
The refreshed MLX result includes the CUDA-only MXFP8 output head. It improved
generation from 7.50 to 8.04 tok/s while reducing the reported resident model
size from 20.09 GB to 18.35 GB. The three measured prefill epochs were 145.44,
364.26, and 360.50 tok/s; the first epoch paid an additional JIT cost, so the
standard three-epoch median remains 360.50 tok/s. The generated output was a
coherent Rayleigh-scattering answer before the benchmark was accepted.
The MLX build is based on `973e27f82ffe68dbd626cda31ba34997045d1eb7`
plus `db0bdc21afa9c813556e7abe25d522a30cd2c508`, the exact cherry-pick of the
64-bit Windows large-file seek fix. The llama-server path fully offloaded
66/66 layers.

The Qwen3.6 35B-A3B row passed a semantic smoke check and was collected as a
diagnostic direct p2048/g128 run with CUDA graphs disabled through the existing
MLX environment switch. This was an A/B measurement, not a production
model-specific graph policy. The three measured epochs were
81.22, 81.15, and 81.41 prompt tok/s and 6.21, 6.38, and 6.47 generate tok/s.
The prior graph-enabled result was 80.56 / 4.42, so disabling graphs improved
generation by 44.3% without materially changing prompt throughput. Per the
current test policy, N1x is used for correctness and direct benchmarks only;
CUDA profiling is performed on `tater50`.

The Qwen3.6 35B-A3B MXFP8 checkpoint initially OOMed while cloning its separate
quantized routed-expert gate/up stacks into packed tensors. The N1x load-safe
path retains those mathematically equivalent stacks separately and dispatches
two existing `GatherQMM` calls; GB10, RTX 5090, Metal, and checkpoints that
already contain packed gate/up tensors keep the original path. The real
`TestBasic/blue-sky` integration test passed with a coherent Rayleigh-scattering
answer before benchmarking. Its three p2048/g128 epochs were 71.26 / 3.55,
70.81 / 3.42, and 66.55 / 3.45 tok/s. No matching N1x Q8_0 baseline has been
accepted, so no target percentage is reported.

The Laguna row is the median of three direct p2048/g128 epochs after a coherent
semantic smoke response. Loading requires two N1x-specific peak-memory
controls: quantized expert gate/up weights remain separate, and Windows
materializes one lazy weight graph at a time. These controls preserve the
model's math and avoid a deterministic load-time OOM. No matching N1x
llama-server baseline has been accepted yet, so no target percentage is
reported.

Diagnostic runs with CUDA graphs disabled on N1x improved Laguna's direct
median from 85.54 / 3.18
to 86.67 / 4.26 tok/s, but it reduced dense Qwen3.6 27B from 360.50 / 8.04 to
211.42 / 2.77. It also improved Qwen3.6 35B-A3B generation from 4.42 to 6.38
tok/s. The response differs by workload graph shape, so these measurements do
not justify either a global N1x switch or model-name dispatch in production.

The Q4_K_M baseline fully offloaded 31/31 layers. Its performance numbers are
retained for reference, but correctness is not established. A follow-up using
`bench.go -p` with a contiguous 1,891-token WikiText passage reproduced
malformed repetitive output on two fresh server processes. Both runs emitted
the same `family life`, `of of`, and `_dd_` token loops, and one epoch per run
hit Ollama's `prediction aborted, token repeat limit reached` guard without
returning final metrics. There was no OOM, partial offload, panic, or server
crash. This rules out the benchmark's short synthetic word-list repetition as
the trigger and leaves a reproducible N1x / CUDA WoA correctness defect to
isolate before accepting further performance baselines.

Gemma4 26B BF16 is 52 GB and cannot remain fully resident in the 37.3 GiB CUDA
budget, so it is excluded. The matching Q8_0 model is pulled and fully
available, but its benchmark is paused until the Q4 correctness defect is
understood.

## Thermal Check

The sustained Gemma4 31B BF16 rows plateaued at 60-64 C and
2,476-2,496 MHz. NVIDIA reported no hardware thermal, software thermal, or
power-brake slowdown. The maximum observed clock spread between the MLX and
llama-server BF16 rows was about 0.8%, so thermal uncertainty is material only
for results within roughly one percentage point of parity.

## Artifact Locations

- RTX 5090: `/home/daniel/code/ollama-nemotron-current-best/.cache/bench/cuda-current-best-20260727/rtx5090`
- tater50: `/home/daniel/code/ollama-upstream-um/.cache/bench/cuda-current-best-20260727/tater50`
- tater62: `.cache/bench/cuda-current-best-20260727/tater62`
- Previous unchanged tater50 targets:
  `/home/daniel/code/ollama-upstream-um/.cache/bench/tater50-um-20260726`

The `invalid-keepalive0` and `invalid-missing-wide-sdpa` directories are
rejected harness attempts and are not included in any table.
