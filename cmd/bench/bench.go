package main

import (
	"cmp"
	"context"
	"crypto/rand"
	"flag"
	"fmt"
	"io"
	"os"
	"runtime"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
)

type flagOptions struct {
	models              *string
	epochs              *int
	maxTokens           *int
	temperature         *float64
	seed                *int
	timeout             *int
	prompt              *string
	imageFile           *string
	keepAlive           *float64
	format              *string
	outputFile          *string
	debug               *bool
	verbose             *bool
	warmup              *int
	promptTokens        *int
	promptTokensPerWord *float64
	numCtx              *int
}

type Metrics struct {
	Model    string
	Step     string
	Count    int
	Duration time.Duration
}

type ModelInfo struct {
	Name              string
	ParameterSize     string
	QuantizationLevel string
	Family            string
	SizeBytes         int64
	VRAMBytes         int64
	NumCtx            int64
}

const (
	DefaultPrompt = `Continue this Python cache fix.

diff --git a/services/cache.py b/services/cache.py
@@ -8,8 +8,14 @@ class TTLCache:
+        if key in self.items:
`

	// Tiny prompt fallback below the compact patch floor (varies by model, e.g. ~<58 for Gemma4).
	patchPromptSliceCorpus = `def put self key value bytes now if in items old current del payload expires while limit removed return diff git services cache py class method ttl store update entry size hit miss evict lookup write read`

	// Per-variation header prepended to every padded prompt. The runner's prefix
	// cache keys on leading tokens, so the cache-busting byte and varying name
	// have to come first: front padding is what gets trimmed to hit a token
	// target, and shared text ahead of them would be restored instead of prefilled.
	patchPromptVariationHeader = "%s%s patch benchmark context.\n\n"

	// Large-prompt repeated padding diff used above the full continuation floor (varies by model, e.g. ~191+ for Gemma4).
	patchPromptPadTemplate = `diff --git a/services/%s_cache.py b/services/%s_cache.py
index 2c4aa31..9b7e021 100644
--- a/services/%s_cache.py
+++ b/services/%s_cache.py
@@ -8,8 +8,14 @@ class TTLCache:
     def put(self, key: str, value: bytes, ttl: float, now: float) -> None:
-        self.items[key] = (value, now + ttl)
-        self.current_bytes += len(value)
+        payload = bytes(value)
+        expires_at = now + ttl
+        if key in self.items:
+            old = self.items[key]
+            self.current_bytes -= len(old[0])
+            del self.items[key]
+        self.items[key] = (payload, expires_at)
+        self.current_bytes += len(payload)
         while self.current_bytes > self.byte_limit:
             _, removed = self.items.popitem(last=True)
             self.current_bytes -= len(removed[0])

`

	// Full continuation kept intact for larger targets (varies by model, e.g. ~191+ for Gemma4).
	patchPromptContinuationTemplate = `Continue the same Python cache replacement fix for the final file.

diff --git a/services/%s_cache.py b/services/%s_cache.py
index 2c4aa31..9b7e021 100644
--- a/services/%s_cache.py
+++ b/services/%s_cache.py
@@ -8,8 +8,14 @@ class TTLCache:
     def put(self, key: str, value: bytes, ttl: float, now: float) -> None:
-        self.items[key] = (value, now + ttl)
-        self.current_bytes += len(value)
+        payload = bytes(value)
+        expires_at = now + ttl
+        if key in self.items:
`

	// Compact continuation keeps medium-short targets coherent (varies by model, e.g. ~58-190 for Gemma4).
	patchPromptCompactTemplate = `Continue this Python cache fix.

diff --git a/services/%s_cache.py b/services/%s_cache.py
@@ -8,8 +8,14 @@ class TTLCache:
+        if key in self.items:
`
)

// Each generated benchmark prompt starts with a different byte. This is
// stronger than varying the first word: tokenizer vocabularies can split
// memory_0 and memory_1 into the same leading token.
const patchPromptCacheBusterAlphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!#$%&()*+,-./:;<=>?@[]^_{|}~"

var patchPadNames = []string{
	"memory",
	"disk",
	"tenant",
	"project",
	"request",
	"worker",
	"session",
}

func patchPadName(index int) string {
	n := len(patchPadNames)
	name := patchPadNames[((index%n)+n)%n]
	if index < 0 {
		return fmt.Sprintf("%s_m%d", name, -index)
	}
	return fmt.Sprintf("%s_%d", name, index)
}

func patchPromptCacheBuster(variation int) string {
	n := len(patchPromptCacheBusterAlphabet)
	index := ((variation % n) + n) % n
	return patchPromptCacheBusterAlphabet[index : index+1]
}

// tokensPerWord is the calibrated ratio of tokens to words for the current model.
// Initialized with a heuristic, then updated during warmup based on actual tokenization.
var tokensPerWord = 1.3

// generatePromptForTokenCount builds a Python patch-continuation prompt near
// targetTokens. The continuation body stays at the end; prompt sizing trims or
// extends only front padding. Targets below the compact patch floor use a
// moving slice from a normalized Python/diff vocabulary instead.
func generatePromptForTokenCount(targetTokens int, variation int) string {
	targetWords := int(float64(targetTokens) / tokensPerWord)
	if targetWords < 1 {
		targetWords = 1
	}

	if targetWords < minPatchPromptWords() {
		return rawPatchPromptSlice(variation, targetWords)
	}

	continuation := rawPatchPromptContinuation(0)
	if targetWords < len(strings.Fields(continuation)) {
		continuation = rawPatchPromptCompact(0)
	}
	continuationWords := strings.Fields(continuation)

	// Lead with the variation header so no two prompts share a token prefix, then
	// spend whatever budget is left on front padding. When the continuation alone
	// already fills the target, the header still goes in and the prompt runs one
	// word long: a cache hit would corrupt the prefill measurement outright.
	budget := targetWords - len(continuationWords)
	words := patchPromptHeaderWords(variation)
	if budget < len(words) {
		words = words[:max(budget, 1)]
	}

	if padWords := budget - len(words); padWords > 0 {
		pad := strings.Fields(rawPatchPromptPad(0, padWords))
		if len(pad) > padWords {
			pad = pad[len(pad)-padWords:]
		}
		words = append(words, pad...)
	}
	return strings.Join(words, " ") + "\n\n" + continuation
}

// patchPromptHeaderWords returns the unique leading words for a variation. The
// pad name embeds the variation index, so consecutive variations differ in the
// very first character and cannot share a leading token.
func patchPromptHeaderWords(variation int) []string {
	return strings.Fields(fmt.Sprintf(
		patchPromptVariationHeader,
		patchPromptCacheBuster(variation),
		patchPadName(variation),
	))
}

func minPatchPromptWords() int {
	return len(strings.Fields(rawPatchPromptCompact(0)))
}

func rawPatchPromptSlice(variation int, targetWords int) string {
	words := strings.Fields(patchPromptSliceCorpus)
	if targetWords < len(words) {
		maxStart := len(words) - targetWords + 1
		start := ((variation % maxStart) + maxStart) % maxStart
		words = words[start : start+targetWords]
	}
	words[0] = patchPromptCacheBuster(variation) + words[0]
	return strings.Join(words, " ")
}

func buildUniqueGenerateRequest(model string, fOpt flagOptions, imgData api.ImageData, variation int, seenPrompts map[string]struct{}) (*api.GenerateRequest, error) {
	maxAttempts := len(patchPromptCacheBusterAlphabet)

	if *fOpt.promptTokens == 0 || seenPrompts == nil {
		return buildGenerateRequest(model, fOpt, imgData, variation), nil
	}

	for attempt := range maxAttempts {
		req := buildGenerateRequest(model, fOpt, imgData, variation+attempt)
		prefixKey := "\x00prefix:" + req.Prompt[:1]
		_, promptSeen := seenPrompts[req.Prompt]
		_, prefixSeen := seenPrompts[prefixKey]
		if !promptSeen && !prefixSeen {
			seenPrompts[req.Prompt] = struct{}{}
			seenPrompts[prefixKey] = struct{}{}
			return req, nil
		}
	}

	return nil, fmt.Errorf("could not generate a unique prompt for --prompt-tokens=%d after %d attempts", *fOpt.promptTokens, maxAttempts)
}

func rawPatchPromptPad(variation int, targetWords int) string {
	var prompt strings.Builder
	wordCount := 0
	for i := 0; wordCount < targetWords; i++ {
		name := patchPadName(i + variation)
		hunk := fmt.Sprintf(patchPromptPadTemplate, name, name, name, name)
		prompt.WriteString(hunk)
		wordCount += len(strings.Fields(hunk))
	}
	return prompt.String()
}

func rawPatchPromptContinuation(variation int) string {
	name := patchPadName(variation + 3)
	return fmt.Sprintf(patchPromptContinuationTemplate, name, name, name, name)
}

func rawPatchPromptCompact(variation int) string {
	name := patchPadName(variation + 3)
	return fmt.Sprintf(patchPromptCompactTemplate, name, name)
}

// calibratePromptTokens adjusts tokensPerWord based on actual tokenization from a warmup run.
func calibratePromptTokens(targetTokens, actualTokens, wordCount int) {
	if actualTokens <= 0 || wordCount <= 0 {
		return
	}
	tokensPerWord = float64(actualTokens) / float64(wordCount)
	newWords := int(float64(targetTokens) / tokensPerWord)
	fmt.Fprintf(os.Stderr, "bench: calibrated %.2f tokens/word (target=%d, got=%d, words=%d → %d)\n",
		tokensPerWord, targetTokens, actualTokens, wordCount, newWords)
}

func buildGenerateRequest(model string, fOpt flagOptions, imgData api.ImageData, epoch int) *api.GenerateRequest {
	options := make(map[string]interface{})
	if *fOpt.maxTokens > 0 {
		options["num_predict"] = *fOpt.maxTokens
	}
	options["temperature"] = *fOpt.temperature
	if fOpt.seed != nil && *fOpt.seed > 0 {
		options["seed"] = *fOpt.seed
	}
	if fOpt.numCtx != nil && *fOpt.numCtx > 0 {
		options["num_ctx"] = *fOpt.numCtx
	}

	var keepAliveDuration *api.Duration
	if *fOpt.keepAlive > 0 {
		duration := api.Duration{Duration: time.Duration(*fOpt.keepAlive * float64(time.Second))}
		keepAliveDuration = &duration
	}

	prompt := *fOpt.prompt
	if *fOpt.promptTokens > 0 {
		prompt = generatePromptForTokenCount(*fOpt.promptTokens, epoch)
	} else if prompt == "" {
		prompt = DefaultPrompt
	} else {
		prompt = strings.TrimSpace(prompt)
	}
	if *fOpt.promptTokens == 0 {
		prompt = fmt.Sprintf("%s [%d] %s", rand.Text(), epoch, prompt)
	}

	req := &api.GenerateRequest{
		Model:     model,
		Prompt:    prompt,
		Raw:       true,
		Options:   options,
		KeepAlive: keepAliveDuration,
	}

	if imgData != nil {
		req.Images = []api.ImageData{imgData}
	}

	return req
}

func fetchModelInfo(ctx context.Context, client *api.Client, model string) ModelInfo {
	info := ModelInfo{Name: model}
	resp, err := client.Show(ctx, &api.ShowRequest{Model: model})
	if err != nil {
		fmt.Fprintf(os.Stderr, "WARNING: Could not fetch model info for '%s': %v\n", model, err)
		return info
	}
	info.ParameterSize = resp.Details.ParameterSize
	info.QuantizationLevel = resp.Details.QuantizationLevel
	info.Family = resp.Details.Family
	return info
}

func fetchMemoryUsage(ctx context.Context, client *api.Client, model string) (size, vram int64) {
	resp, err := client.ListRunning(ctx)
	if err != nil {
		if debug := os.Getenv("OLLAMA_DEBUG"); debug != "" {
			fmt.Fprintf(os.Stderr, "WARNING: Could not fetch memory usage: %v\n", err)
		}
		return 0, 0
	}
	for _, m := range resp.Models {
		if m.Name == model || m.Model == model {
			return m.Size, m.SizeVRAM
		}
	}
	for _, m := range resp.Models {
		if strings.HasPrefix(m.Name, model) || strings.HasPrefix(m.Model, model) {
			return m.Size, m.SizeVRAM
		}
	}
	return 0, 0
}

func fetchContextLength(ctx context.Context, client *api.Client, model string) int64 {
	resp, err := client.ListRunning(ctx)
	if err != nil {
		return 0
	}
	for _, m := range resp.Models {
		if m.Name == model || m.Model == model || strings.HasPrefix(m.Name, model) || strings.HasPrefix(m.Model, model) {
			return int64(m.ContextLength)
		}
	}
	return 0
}

func outputFormatHeader(w io.Writer, format string, verbose bool) {
	switch format {
	case "benchstat":
		if verbose {
			fmt.Fprintf(w, "goos: %s\n", runtime.GOOS)
			fmt.Fprintf(w, "goarch: %s\n", runtime.GOARCH)
		}
	case "csv":
		headings := []string{"NAME", "STEP", "COUNT", "NS_PER_COUNT", "TOKEN_PER_SEC"}
		fmt.Fprintln(w, strings.Join(headings, ","))
	}
}

func outputModelInfo(w io.Writer, format string, info ModelInfo) {
	params := cmp.Or(info.ParameterSize, "unknown")
	quant := cmp.Or(info.QuantizationLevel, "unknown")
	family := cmp.Or(info.Family, "unknown")

	memStr := ""
	if info.SizeBytes > 0 {
		memStr = fmt.Sprintf(" | Size: %d | VRAM: %d", info.SizeBytes, info.VRAMBytes)
	}
	ctxStr := ""
	if info.NumCtx > 0 {
		ctxStr = fmt.Sprintf(" | NumCtx: %d", info.NumCtx)
	}
	fmt.Fprintf(w, "# Model: %s | Params: %s | Quant: %s | Family: %s%s%s\n",
		info.Name, params, quant, family, memStr, ctxStr)
}

func OutputMetrics(w io.Writer, format string, metrics []Metrics, verbose bool) {
	switch format {
	case "benchstat":
		for _, m := range metrics {
			if m.Step == "generate" || m.Step == "prefill" {
				if m.Count > 0 {
					nsPerToken := float64(m.Duration.Nanoseconds()) / float64(m.Count)
					tokensPerSec := float64(m.Count) / (float64(m.Duration.Nanoseconds()) + 1e-12) * 1e9
					fmt.Fprintf(w, "BenchmarkModel/name=%s/step=%s 1 %.2f ns/token %.2f token/sec\n",
						m.Model, m.Step, nsPerToken, tokensPerSec)
				} else {
					fmt.Fprintf(w, "BenchmarkModel/name=%s/step=%s 1 0 ns/token 0 token/sec\n",
						m.Model, m.Step)
				}
			} else if m.Step == "ttft" {
				fmt.Fprintf(w, "BenchmarkModel/name=%s/step=ttft 1 %d ns/op\n",
					m.Model, m.Duration.Nanoseconds())
			} else {
				fmt.Fprintf(w, "BenchmarkModel/name=%s/step=%s 1 %d ns/op\n",
					m.Model, m.Step, m.Duration.Nanoseconds())
			}
		}
	case "csv":
		for _, m := range metrics {
			if m.Step == "generate" || m.Step == "prefill" {
				var nsPerToken float64
				var tokensPerSec float64
				if m.Count > 0 {
					nsPerToken = float64(m.Duration.Nanoseconds()) / float64(m.Count)
					tokensPerSec = float64(m.Count) / (float64(m.Duration.Nanoseconds()) + 1e-12) * 1e9
				}
				fmt.Fprintf(w, "%s,%s,%d,%.2f,%.2f\n", m.Model, m.Step, m.Count, nsPerToken, tokensPerSec)
			} else {
				fmt.Fprintf(w, "%s,%s,1,%d,0\n", m.Model, m.Step, m.Duration.Nanoseconds())
			}
		}
	default:
		fmt.Fprintf(os.Stderr, "Unknown output format '%s'\n", format)
	}
}

func BenchmarkModel(fOpt flagOptions) error {
	models := strings.Split(*fOpt.models, ",")
	if fOpt.promptTokensPerWord != nil && *fOpt.promptTokensPerWord > 0 {
		tokensPerWord = *fOpt.promptTokensPerWord
	}

	var imgData api.ImageData
	var err error
	if *fOpt.imageFile != "" {
		imgData, err = readImage(*fOpt.imageFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "ERROR: Couldn't read image '%s': %v\n", *fOpt.imageFile, err)
			return err
		}
	}

	if *fOpt.debug && imgData != nil {
		fmt.Fprintf(os.Stderr, "Read file '%s'\n", *fOpt.imageFile)
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: Couldn't create ollama client: %v\n", err)
		return err
	}

	var out io.Writer = os.Stdout
	if fOpt.outputFile != nil && *fOpt.outputFile != "" {
		f, err := os.OpenFile(*fOpt.outputFile, os.O_CREATE|os.O_WRONLY, 0o644)
		if err != nil {
			fmt.Fprintf(os.Stderr, "ERROR: cannot open output file %s: %v\n", *fOpt.outputFile, err)
			return err
		}
		defer f.Close()
		out = f
	}

	outputFormatHeader(out, *fOpt.format, *fOpt.verbose)

	// Log prompt-tokens info in debug mode
	if *fOpt.debug && *fOpt.promptTokens > 0 {
		prompt := generatePromptForTokenCount(*fOpt.promptTokens, 0)
		wordCount := len(strings.Fields(prompt))
		fmt.Fprintf(os.Stderr, "Generated prompt targeting ~%d tokens (%d words, varied per epoch)\n", *fOpt.promptTokens, wordCount)
	}

	for _, model := range models {
		seenPrompts := map[string]struct{}{}

		// Fetch model info
		infoCtx, infoCancel := context.WithTimeout(context.Background(), 10*time.Second)
		info := fetchModelInfo(infoCtx, client, model)
		infoCancel()

		// Warmup phase (uses negative epoch numbers to avoid colliding with timed epochs)
		for i := range *fOpt.warmup {
			req, buildErr := buildUniqueGenerateRequest(model, fOpt, imgData, -(i + 1), seenPrompts)
			if buildErr != nil {
				fmt.Fprintf(os.Stderr, "WARNING: Warmup %d/%d for %s skipped: %v\n", i+1, *fOpt.warmup, model, buildErr)
				continue
			}
			ctx, cancel := context.WithTimeout(context.Background(), time.Duration(*fOpt.timeout)*time.Second)

			var warmupMetrics *api.Metrics
			err = client.Generate(ctx, req, func(resp api.GenerateResponse) error {
				if resp.Done {
					warmupMetrics = &resp.Metrics
				}
				return nil
			})
			cancel()

			if err != nil {
				fmt.Fprintf(os.Stderr, "WARNING: Warmup %d/%d for %s failed: %v\n", i+1, *fOpt.warmup, model, err)
			} else {
				if *fOpt.debug {
					fmt.Fprintf(os.Stderr, "Warmup %d/%d for %s complete\n", i+1, *fOpt.warmup, model)
				}
				// Calibrate prompt token count on last warmup run
				if i == *fOpt.warmup-1 && *fOpt.promptTokens > 0 && warmupMetrics != nil {
					calibratePromptTokens(*fOpt.promptTokens, warmupMetrics.PromptEvalCount, len(strings.Fields(req.Prompt)))
				}
			}
		}

		// Fetch memory/context info once after warmup (model is loaded and stable)
		memCtx, memCancel := context.WithTimeout(context.Background(), 5*time.Second)
		info.SizeBytes, info.VRAMBytes = fetchMemoryUsage(memCtx, client, model)
		if fOpt.numCtx != nil && *fOpt.numCtx > 0 {
			info.NumCtx = int64(*fOpt.numCtx)
		} else {
			info.NumCtx = fetchContextLength(memCtx, client, model)
		}
		memCancel()

		outputModelInfo(out, *fOpt.format, info)

		// Timed epoch loop
		shortCount := 0
		for epoch := range *fOpt.epochs {
			var responseMetrics *api.Metrics
			var ttft time.Duration
			short := false

			// Retry loop: if the model hits a stop token before max-tokens,
			// retry with a different prompt (up to maxRetries times).
			const maxRetries = 3
			for attempt := range maxRetries + 1 {
				responseMetrics = nil
				ttft = 0
				var ttftOnce sync.Once

				req, buildErr := buildUniqueGenerateRequest(model, fOpt, imgData, epoch+attempt*1000, seenPrompts)
				if buildErr != nil {
					err = buildErr
					fmt.Fprintf(os.Stderr, "ERROR: Couldn't generate unique prompt for model '%s': %v\n", model, buildErr)
					break
				}
				requestStart := time.Now()

				ctx, cancel := context.WithTimeout(context.Background(), time.Duration(*fOpt.timeout)*time.Second)

				err = client.Generate(ctx, req, func(resp api.GenerateResponse) error {
					if *fOpt.debug {
						fmt.Fprintf(os.Stderr, "%s", cmp.Or(resp.Thinking, resp.Response))
					}

					// Capture TTFT on first content
					ttftOnce.Do(func() {
						if resp.Response != "" || resp.Thinking != "" {
							ttft = time.Since(requestStart)
						}
					})

					if resp.Done {
						responseMetrics = &resp.Metrics
					}
					return nil
				})
				cancel()

				if *fOpt.debug {
					fmt.Fprintln(os.Stderr)
				}

				if err != nil {
					if ctx.Err() == context.DeadlineExceeded {
						fmt.Fprintf(os.Stderr, "ERROR: Request timed out with model '%s' after %vs\n", model, *fOpt.timeout)
					} else {
						fmt.Fprintf(os.Stderr, "ERROR: Couldn't generate with model '%s': %v\n", model, err)
					}
					break
				}

				if responseMetrics == nil {
					fmt.Fprintf(os.Stderr, "ERROR: No metrics received for model '%s'\n", model)
					break
				}

				// Check if the response was shorter than requested
				short = *fOpt.maxTokens > 0 && responseMetrics.EvalCount < *fOpt.maxTokens
				if !short || attempt == maxRetries {
					break
				}

				if *fOpt.debug {
					fmt.Fprintf(os.Stderr, "Short response (%d/%d tokens), retrying with different prompt (attempt %d/%d)\n",
						responseMetrics.EvalCount, *fOpt.maxTokens, attempt+1, maxRetries)
				}
			}

			if err != nil || responseMetrics == nil {
				continue
			}

			if short {
				shortCount++
				if *fOpt.debug {
					fmt.Fprintf(os.Stderr, "WARNING: Short response (%d/%d tokens) after %d retries for epoch %d\n",
						responseMetrics.EvalCount, *fOpt.maxTokens, maxRetries, epoch+1)
				}
			}

			metrics := []Metrics{
				{
					Model:    model,
					Step:     "prefill",
					Count:    responseMetrics.PromptEvalCount,
					Duration: responseMetrics.PromptEvalDuration,
				},
				{
					Model:    model,
					Step:     "generate",
					Count:    responseMetrics.EvalCount,
					Duration: responseMetrics.EvalDuration,
				},
				{
					Model:    model,
					Step:     "ttft",
					Count:    1,
					Duration: ttft,
				},
				{
					Model:    model,
					Step:     "load",
					Count:    1,
					Duration: responseMetrics.LoadDuration,
				},
				{
					Model:    model,
					Step:     "total",
					Count:    1,
					Duration: responseMetrics.TotalDuration,
				},
			}

			OutputMetrics(out, *fOpt.format, metrics, *fOpt.verbose)

			if *fOpt.debug && *fOpt.promptTokens > 0 {
				fmt.Fprintf(os.Stderr, "Generated prompt targeting ~%d tokens (actual: %d)\n",
					*fOpt.promptTokens, responseMetrics.PromptEvalCount)
			}

			if *fOpt.keepAlive > 0 {
				time.Sleep(time.Duration(*fOpt.keepAlive*float64(time.Second)) + 200*time.Millisecond)
			}
		}

		if shortCount > 0 {
			fmt.Fprintf(os.Stderr, "WARNING: %d/%d epochs for '%s' had short responses (<%d tokens). Generation metrics may be unreliable.\n",
				shortCount, *fOpt.epochs, model, *fOpt.maxTokens)
		}

		// Unload model before moving to the next one
		unloadModel(client, model, *fOpt.timeout)
	}

	return nil
}

func unloadModel(client *api.Client, model string, timeout int) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(timeout)*time.Second)
	defer cancel()

	zero := api.Duration{Duration: 0}
	req := &api.GenerateRequest{
		Model:     model,
		KeepAlive: &zero,
	}
	_ = client.Generate(ctx, req, func(resp api.GenerateResponse) error {
		return nil
	})
}

func readImage(filePath string) (api.ImageData, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		return nil, err
	}

	return api.ImageData(data), nil
}

func main() {
	fOpt := flagOptions{
		models:              flag.String("model", "", "Model to benchmark"),
		epochs:              flag.Int("epochs", 6, "Number of epochs (iterations) per model"),
		maxTokens:           flag.Int("max-tokens", 200, "Maximum tokens for model response"),
		temperature:         flag.Float64("temperature", 0, "Temperature parameter"),
		seed:                flag.Int("seed", 0, "Random seed"),
		timeout:             flag.Int("timeout", 60*5, "Timeout in seconds (default 300s)"),
		prompt:              flag.String("p", DefaultPrompt, "Prompt to use"),
		imageFile:           flag.String("image", "", "Filename for an image to include"),
		keepAlive:           flag.Float64("k", 0, "Keep alive duration in seconds"),
		format:              flag.String("format", "benchstat", "Output format [benchstat|csv]"),
		outputFile:          flag.String("output", "", "Output file for results (stdout if empty)"),
		verbose:             flag.Bool("v", false, "Show system information"),
		debug:               flag.Bool("debug", false, "Show debug information"),
		warmup:              flag.Int("warmup", 1, "Number of warmup requests before timing"),
		promptTokens:        flag.Int("prompt-tokens", 0, "Generate prompt targeting ~N tokens (0 = use -p prompt)"),
		promptTokensPerWord: flag.Float64("prompt-tokens-per-word", 0, "Initial token-to-word ratio for generated prompts (0 = heuristic)"),
		numCtx:              flag.Int("num-ctx", 0, "Context size (0 = server default)"),
	}

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [OPTIONS]\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Description:\n")
		fmt.Fprintf(os.Stderr, "  Model benchmarking tool with configurable parameters\n\n")
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "  bench -model gemma3,llama3 -epochs 6\n")
		fmt.Fprintf(os.Stderr, "  bench -model gemma3 -epochs 6 -prompt-tokens 512 -format csv\n")
	}
	flag.Parse()

	if !slices.Contains([]string{"benchstat", "csv"}, *fOpt.format) {
		fmt.Fprintf(os.Stderr, "ERROR: Unknown format '%s'\n", *fOpt.format)
		os.Exit(1)
	}

	if len(*fOpt.models) == 0 {
		fmt.Fprintf(os.Stderr, "ERROR: No model(s) specified to benchmark.\n")
		flag.Usage()
		return
	}

	BenchmarkModel(fOpt)
}
