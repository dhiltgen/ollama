# Gemma4 26B Correctness Failure on NVIDIA N1x CUDA WoA

## Status

Gemma4 26B produces deterministic malformed repetition on `tater62` with both
Q4_K_M and Q8_0. Smaller Gemma4 E4B Q4_K_M and Qwen3.6 35B-A3B Q4_K_M are
coherent under the same test. Do not use Gemma4 26B N1x throughput numbers as
valid baselines until this is understood.

The current evidence points to a Gemma4 26B-specific CUDA operation or shape on
SM121 / Windows Arm64. It does not yet distinguish an NVIDIA CUDA Toolkit or
driver defect from llama.cpp's SM121 kernel selection or implementation.

## Environment

- Host: `tater62`, pre-release NVIDIA N1x, Windows Arm64.
- GPU: `JMJWOA-Generic-GPU`, compute capability 12.1.
- Driver reported by Ollama: 13.1.
- Ollama: official Windows Arm64 v0.32.5 release.
- Backend: `cuda_v13`; llama-server reports `CUDA : ARCHS = 1210`,
  `USE_GRAPHS = 1`, and `BLACKWELL_NATIVE_FP4 = 1`.
- Ollama reports 37.3 GiB available GPU memory.
- Server used `OLLAMA_NUM_PARALLEL=1` and a 4096-token context.
- All tested models were fully GPU resident. No OOM, partial offload, panic, or
  runner crash occurred during the accepted repros.
- The host has CUDA 13.1 and 13.4 paths visible. Determine the exact loaded DLL
  versions before attributing this to a specific CTK release.

## Prompt And Command

The prompt is a contiguous WikiText-2 passage, not `bench.go`'s short repeated
word list:

- Local path:
  `.cache/bench/cuda-current-best-20260727/corpus/wikitext-natural-p2048.txt`
- SHA-256:
  `e57ca7d3550a07d42752a2c85b7406d7f26d48ffee6f7ff386cccc80576106ba`
- Size: 8,234 characters / 1,587 whitespace-delimited words.
- Gemma4 token count: 1,891 prompt tokens.
- Qwen3.6 token count: 1,909 prompt tokens.

Repro shape:

```powershell
$env:OLLAMA_HOST = 'http://tater62:25271'
$prompt = Get-Content `
  '.cache\bench\cuda-current-best-20260727\corpus\wikitext-natural-p2048.txt' `
  -Raw

go run ./cmd/bench/bench.go `
  -model '<model>' `
  -p $prompt `
  -max-tokens 128 `
  -num-ctx 4096 `
  -warmup 1 `
  -epochs 3 `
  -format csv `
  -debug `
  -timeout 600
```

Use a fresh official server process when confirming the failure. `bench.go
-debug` prints both thinking and response tokens.

## Failing Models

### Gemma4 26B Q4_K_M

- Tag: `gemma4:26b` / `gemma4:26b-a4b-it-q4_K_M`
- Manifest ID: `5571076f3d70`
- Model blob:
  `sha256-7121486771cbfe218851513210c40b35dbdee93ab1ef43fe36283c883980f0df`
- Full offload: 31/31 layers.
- Resident runner size: 16.2 GiB.
- Reproduced on two fresh server processes with the same output streams.
- Representative malformed output:
  - `family life $_$, family life $_$, ...`
  - `much of of of of of ...`
  - `ss_ss_ss_ss...` and `ss_dd_dd_dd...`
- One measured epoch in each run hit:
  `prediction aborted, token repeat limit reached`
- The other epochs returned 128 tokens but remained semantically corrupt.

### Gemma4 26B Q8_0

- Tag: `gemma4:26b-a4b-it-q8_0`
- Manifest ID: `6bfaf9a8cb37`
- Model blob:
  `sha256-bbcf7fc45500f1df01390a0010da23d032c2a4b3e9b8b829cb8038b1bc36bc0d`
- Full offload: 31/31 layers.
- Resident runner size: 25.5 GiB.
- Representative malformed output:
  `family of the own Du family, family of the own Du family, ...`
- The corruption therefore is not specific to Q4_K_M or native FP4 handling.

## Passing Controls

### Gemma4 E4B Q4_K_M

- Tag: `gemma4:e4b`
- Manifest ID: `c6eb396dbd59`
- Model blob:
  `sha256-4c27e0f5b5adf02ac956c7322bd2ee7636fe3f45a8512c9aba5385242cb6e09a`
- Full offload: 43/43 layers.
- Resident runner size: 3.0 GiB.
- Three measured epochs produced coherent continuations.
- Natural-prompt median: 3,899.85 prompt tokens/s, 58.19 generate tokens/s.

### Qwen3.6 35B-A3B Q4_K_M

- Tag: `qwen3.6:35b-a3b`
- Manifest ID: `07d35212591f`
- Model blob:
  `sha256-f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d`
- Full offload: 42/42 layers.
- Resident runner size: 21.6 GiB.
- Three measured epochs produced coherent continuations.
- Natural-prompt median: 793.75 prompt tokens/s, 34.58 generate tokens/s.

These controls rule out a generic Q4 CUDA failure and a generic large-MoE
CUDA failure.

## Relevant Model Shapes

Gemma4 26B:

- 30 blocks, embedding width 2,816.
- 128 experts, 8 selected, expert feed-forward width 704.
- Attention key/value width 512.
- Expert gate/up tensor shape: `2816 x 1408 x 128`.
- Expert down tensor shape: `704 x 2816 x 128`.
- Q4 model gate/up experts are Q4_K; down experts are Q5_0 or Q8_0.
- Q8 model uses Q8_0 for both expert tensor families.

Passing Qwen3.6 35B-A3B:

- 40 blocks, embedding width 2,048.
- 256 experts, 8 selected, expert feed-forward width 512.
- Attention key/value width 256.
- Expert down tensor shape: `512 x 2048 x 256`.

The different expert and attention dimensions are plausible kernel-selection
boundaries. Do not assume the expert kernel is responsible until the first
divergent operation or logits are identified.

## Next Isolation Steps

1. Re-run the exact prompt and exact official model manifests on an RTX 5090
   with official Ollama v0.32.5. A pass there would establish an SM121 / WoA
   boundary rather than a bad model conversion.
2. Run the same Gemma4 26B artifact on CPU on `tater62`. If the full prompt is
   too slow, first binary-search the prompt-length threshold on CUDA and use
   the shortest reliable failing prompt for CPU comparison.
3. Compare CUDA output with graphs enabled and disabled. Verify the current
   llama.cpp environment variable name from source before running the A/B.
4. Compare the available cuBLAS and custom quantized matmul dispatch controls.
   Again, verify the exact current llama.cpp variable names before use.
5. Capture deterministic next-token logits from CPU and CUDA and locate the
   first layer or operation whose output diverges. This is stronger evidence
   than comparing final text.
6. Once narrowed, run Compute Sanitizer and a minimal GGML tensor repro using
   the failing Gemma4 expert or attention shapes.
7. Reproduce with standalone llama.cpp on Windows Arm64. This separates the
   Ollama scheduler/API from the CUDA backend.
8. Record the loaded CUDA runtime, cuBLAS, and driver DLL versions, plus
   `nvidia-smi -q`, before filing with NVIDIA.

## Artifacts

- Q4 client logs:
  `.cache/bench/cuda-current-best-20260727/tater62/gemma4-26b-q4km-official-0325-wikitext-p2048g128*.client.log`
- Q4 server logs:
  `.cache/remote/tater62/server-gemma4-26b-q4-natural-prompt-25271.err.log`
  and
  `.cache/remote/tater62/server-gemma4-26b-q4-natural-prompt-retry-25271.err.log`
- Q8 client log:
  `.cache/bench/cuda-current-best-20260727/tater62/gemma4-26b-q8-official-0325-wikitext-p2048g128.client.log`
- Q8 server log:
  `.cache/remote/tater62/server-gemma4-26b-q8-natural-control-25271.err.log`
- Passing E4B CSV:
  `.cache/bench/cuda-current-best-20260727/tater62/gemma4-e4b-q4km-official-0325-wikitext-p2048g128.csv`
- Passing Qwen CSV:
  `.cache/bench/cuda-current-best-20260727/tater62/qwen36-35b-a3b-q4km-official-0325-wikitext-p2048g128-3x.csv`
