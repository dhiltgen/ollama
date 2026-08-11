# N1x CUDA Profiling Morning Checklist

## Goal

Get one useful Nsight Graphics Pro trace for
`dhiltgen/qwen3.5:9b-nvfp4` on tater62 without relying on an SSH-owned process.
The prior attach attempt failed because the target PID disappeared or was not
visible to the profiler session; it did not establish a CUDA counter permission
failure.

## 2026-07-29 Findings

- Direct launch works: Nsight connected to the injected MLX runner and produced
  a 788,077,168-byte trace at
  `C:\Users\daniel\.codex\profiles\ngfx-direct-20260729-075130\ollama_2026_07_29_07_53_02.ngfx-gputrace`.
- The first trace's workload timing was disturbed by the interactive
  Windows PowerShell `Invoke-WebRequest` warning. The script now uses
  `-UseBasicParsing`, decodes byte-array responses as UTF-8, and preserves
  profiler and runner logs on failure.
- Attach mode is not viable on this Nsight WoA build. It reports
  `Attach failed: Cannot find process` even when the script starts and warms the
  runner in the same elevated interactive console session.
- GPU Trace reports a fixed `GPU PMA Buffer Size (MB) = 4294.9`. NVIDIA
  documents an approximately 4 GB fixed upper memory budget for PM counters and
  warp-state sampling. That reservation overlaps MLX model materialization and
  makes `dhiltgen/qwen3.6:27b-coding-nvfp4` intermittently fail with
  `cudaMallocAsync ... out of memory`, despite the model using about 19 GiB at
  steady state and loading normally without the profiler.
- Do not repeatedly retry the 27B trace after this OOM. Preserve the existing
  trace, reboot before one final clean attempt if needed, or use a smaller
  representative model. Repeated failed profiler launches are not useful data
  on this pre-release WoA stack.
- `dhiltgen/laguna-xs.2:nvfp4` loads normally at 17.84 GiB active/peak memory,
  but it also OOMs while materializing under Nsight's 4.295 GB PMA reservation.
  It is not a viable capture model with the current profiler configuration.
- `dhiltgen/gemma4:e4b-nvfp4` is only 8.98 GiB on disk, but its PLE/MTP weight
  graph expands during materialization and still failed with
  `cudaMallocAsync ... out of memory` before the runner became ready. It is not
  a useful profiling fallback yet.
- `dhiltgen/qwen3.5:9b-nvfp4` is the selected capture model. Its direct runner
  probe produced coherent Rayleigh-scattering reasoning across 128 generated
  tokens and reported 6.61 GiB active / 6.62 GiB peak memory. It exercises the
  same NVFP4 direct QMV/QMM and hybrid-attention kernel families as the larger
  Qwen models while leaving enough headroom for the profiler reservation.
- The first clean N1X capture completed at
  `C:\Users\daniel\.codex\profiles\ngfx-direct-20260729-092914\ollama_2026_07_29_09_31_09.ngfx-gputrace`.
  It is a 988,753,569-byte, p2048/g128, 2-second GPU trace with a coherent
  correctness response. The helper originally anchored the request to process
  launch, but Nsight anchors `--start-after-ms` to target readiness. The trace
  therefore began after prefill completed and is a valid NVFP4 decode capture,
  not the intended prefill capture. The instrumented request reported 2048
  prompt tokens in 1.568 seconds and 128 generated tokens in 4.507 seconds; use
  those timings only to orient the trace, not as unprofiled benchmark results.
- `dhiltgen/qwen3.5:9b-mxfp8` is pulled and correctness-gated. A direct runner
  probe produced a coherent explanation of Rayleigh scattering and used about
  9.63 GiB active / 9.64 GiB peak memory, leaving sufficient headroom for the
  profiler reservation. It is the preferred quantization control because it
  holds the Qwen model graph constant while changing the low-precision kernel
  path.
- `dhiltgen/gemma4:e2b-nvfp4` fits in memory but is not a valid profile
  candidate: the current candidate completes immediately after prompt eval with
  zero generated tokens and an empty response. Do not benchmark or profile it
  until that correctness failure is resolved.

## Recommended Capture Matrix

Collect these four traces before another optimization pass so analysis can
separate quantization effects from prefill/decode effects:

| Model | Shape | Purpose | Status |
| --- | --- | --- | --- |
| `qwen3.5:9b-nvfp4` | p2048/g128 | NVFP4 prefill-heavy | Captured (`20260729-095547`) |
| `qwen3.5:9b-nvfp4` | p2048/g128 | NVFP4 decode-only | Captured (misaligned window) |
| `qwen3.5:9b-mxfp8` | p2048/g128 | MXFP8 prefill-heavy | Captured (`20260729-095958`) |
| `qwen3.5:9b-mxfp8` | p16/g128 | MXFP8 decode-heavy | Captured (`20260729-100344`) |

No second architecture currently passes both gates. Gemma4 E2B fails
correctness, while Gemma4 E4B, Laguna XS.2, and Nemotron3 exceed profiler
headroom during materialization.

The final MXFP8 decode trace begins after the 17-token prefill completes and
contains roughly 0.7 seconds of instrumented decode activity. Its correctness
request generated all 128 tokens coherently.

`--auto-export` emits top-level metric tables but not CUDA Event List rows for
these frameless reports. The per-kernel timeline remains in the
`.ngfx-gputrace` and must be inspected through the offline Nsight Graphics GPU
Trace UI. NVIDIA documents that the reports can be analyzed on another machine
with Nsight Graphics installed; the analysis host does not need the captured
GPU or application.

## Known Paths

- Nsight Graphics Pro:
  `C:\Program Files\NVIDIA Corporation\Nsight Graphics Pro 2026.1.0 for WOA\host\windows-desktop-nomad-t23x-a64\ngfx.exe`
- Ollama candidate:
  `C:\Users\daniel\.codex\dist\ollama-mlx-cuda13-sm121a\bin\ollama.exe`
- Direct capture script:
  `C:\Users\daniel\.codex\scripts\tater62_ngfx_direct.ps1`
- Expected output root:
  `C:\Users\daniel\.codex\profiles`
- Prior failed attach:
  `C:\Users\daniel\.codex\profiles\ngfx-mlx-1620-20260727-193145`

## One-Shot Console Procedure

1. Log in at the tater62 console rather than through SSH or RDP.
2. Open an elevated ARM64 PowerShell.
3. Record `nvidia-smi` output, including the driver version.
4. Stop only Ollama, runner, or `ngfx` processes created for this test.
5. Confirm the two executable paths above exist.
6. Confirm the model is present with the candidate binary; pull it from
   `dhiltgen/` only if it is absent.
7. Run:

```powershell
powershell -ExecutionPolicy Bypass -File C:\Users\daniel\.codex\scripts\tater62_ngfx_direct.ps1 `
  -OllamaBin C:\Users\daniel\.codex\dist\ollama-mlx-cuda13-sm121a\bin\ollama.exe `
  -Model dhiltgen/qwen3.5:9b-nvfp4 `
  -UseCUDAGraphs 1 `
  -TraceDelayMs 90000 `
  -DurationMs 2000
```

8. Let the script complete its warmup, measured request, trace, and cleanup.
9. Preserve the resulting `ngfx-direct-<timestamp>` directory, including
   `ngfx.stdout.log`, `ngfx.stderr.log`, the exit-code file, and response logs.
10. Copy the complete result directory back for analysis.

## If It Fails

- `Cannot find process`: keep the target and profiler in the same interactive
  console session. Do not retry attach over SSH.
- GPU performance counter permission error: run elevated and enable NVIDIA
  developer GPU performance counters in the driver/NVIDIA Control Panel.
- Unsupported driver/tool pairing: preserve the exact error and current
  `nvidia-smi` output before installing the newer driver or a newer public
  Nsight build.
- Model or runner failure: inspect the candidate server/runner logs and verify
  a normal direct request works before retrying the profiler.
- Export does not finish: the script uses file-backed profiler logs so the
  injected runner cannot deadlock redirected pipes. It waits three minutes,
  stops only its exact runner if necessary, and gives Nsight another two
  minutes to export. The 2-second trace duration limits report size while
  covering this model's complete warmed prefill and the start of generation.
- Machine instability: retry once. If the second attempt fails, stop and record
  the failure rather than repeatedly hammering pre-release hardware.

## Success Criteria

- A coherent measured model response is present.
- The trace contains CUDA kernel activity, not only CPU/API events.
- Kernel names, launch dimensions, durations, and available memory or
  utilization counters can be exported.
- All test-owned Ollama, runner, and profiler processes are gone afterward.

Once this works, distill the validated process into
`~/Documents/Agents/mlx-wsl-profiling.md`; do not promote this unvalidated
N1x recipe to global guidance yet.
