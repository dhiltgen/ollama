# MLX CUDA Reset Handoff

The complete local recovery bundle is:

`.cache/handoff/mlx-cuda-reset-20260804/STATE.md`

Start there. It records the exact Ollama and MLX source state, complete staged
patches, the separate active experiment patch, binary/native hashes, benchmark
protocol, latest exact results, rejected paths, artifact locations, and the
ordered next steps.

Important current state:

- The intended Ollama and MLX review changes are staged but not committed or
  pushed.
- The staged MLX changes remain CUDA-backend-only plus CUDA tests; no MLX core
  or public API drift was found.
- The only unstaged tracked Ollama experiment changes the prefix-cache reserve
  from `total/16` to `total/8`. It is not accepted yet.
- The latest diagnostic staged RTX 5090 Gemma4 12B run degrades across
  requests (`2789 / 93.9` median, final row `1540 / 61.6`) despite healthy
  clocks.
- The active reserve-1/8 diagnostic removes that collapse (`3053 / 105.0`),
  while Qwen measured `6483 / 132.1` and Laguna measured `1375 / 23.5`.
- Those four rows, including fixed-Python llama Gemma4 at `2192 / 114.3`, were
  timed with `OLLAMA_DEBUG=2`. They are diagnostic only, not current status.
- The paired wrapper is now corrected to use debug 2 only for correctness and
  debug 1 for timing. The next session must rerun both the staged control and
  reserve candidate before accepting or rejecting the experiment.

Do not use the historical tables or the debug-2 diagnostics as current-tree
results. The required reset
status format and `N/A` rules are in the handoff state and
`notes/MLX-CUDA-UM-QMM-EXPERIMENTS.md`.
