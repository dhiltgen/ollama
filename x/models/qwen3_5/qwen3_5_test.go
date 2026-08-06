package qwen3_5

import (
	"runtime"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

func TestNormalizedMoERouter256CUDAMatchesMLX(t *testing.T) {
	useMLXTestThread(t)

	for _, rows := range []int{1, 3, 8} {
		t.Logf("rows=%d", rows)
		values := make([]float32, rows*256)
		for row := range rows {
			for i := range 256 {
				values[row*256+i] = float32((i*73+row*29)%257-128) * 0.03125
			}
		}
		logits := mlx.FromValues(values, rows, 256).AsType(mlx.DTypeBFloat16)
		gotWeights, gotIndices, ok := mlx.FastNormalizedMoERouter256(logits)
		if !ok {
			t.Skip("MLX CUDA custom kernels unavailable")
		}

		wantIndices := mlx.Argpartition(mlx.Neg(logits), 7, -1)
		wantIndices = mlx.SliceStartStop(
			wantIndices,
			[]int32{0, 0},
			[]int32{int32(rows), 8},
		)
		wantWeights := mlx.SoftmaxAxis(
			mlx.TakeAlongAxis(logits, wantIndices, -1),
			-1,
			true,
		)

		gotWeightsF32 := gotWeights.AsType(mlx.DTypeFloat32)
		wantWeightsF32 := wantWeights.AsType(mlx.DTypeFloat32)
		gotIndicesI32 := gotIndices.AsType(mlx.DTypeInt32)
		wantIndicesI32 := wantIndices.AsType(mlx.DTypeInt32)
		mlx.Eval(gotWeightsF32, wantWeightsF32, gotIndicesI32, wantIndicesI32)

		gotWeightValues := gotWeightsF32.Floats()
		wantWeightValues := wantWeightsF32.Floats()
		gotIndexValues := gotIndicesI32.Ints()
		wantIndexValues := wantIndicesI32.Ints()
		for row := range rows {
			got := make(map[int]float32, 8)
			want := make(map[int]float32, 8)
			for i := range 8 {
				offset := row*8 + i
				got[gotIndexValues[offset]] = gotWeightValues[offset]
				want[wantIndexValues[offset]] = wantWeightValues[offset]
			}
			if len(got) != len(want) {
				t.Fatalf("row %d: selected experts = %v, want %v", row, got, want)
			}
			for index, wantWeight := range want {
				gotWeight, exists := got[index]
				if !exists {
					t.Fatalf("row %d: selected experts = %v, want %v", row, got, want)
				}
				diff := gotWeight - wantWeight
				if diff < 0 {
					diff = -diff
				}
				if diff > 1e-3 {
					t.Fatalf(
						"row %d expert %d: weight = %v, want %v",
						row, index, gotWeight, wantWeight,
					)
				}
			}
		}
	}
}

func TestQwenGatedDeltaCUDAFusionsMatchMLX(t *testing.T) {
	useMLXTestThread(t)

	convValues := make([]float32, 8192)
	for i := range convValues {
		convValues[i] = float32((i*37)%257-128) * 0.0078125
	}
	alphaValues := make([]float32, 32)
	betaValues := make([]float32, 32)
	aExpValues := make([]float32, 32)
	dtBiasValues := make([]float32, 32)
	for i := range 32 {
		alphaValues[i] = float32(i-16) * 0.03125
		betaValues[i] = float32(15-i) * 0.046875
		aExpValues[i] = 0.5 + float32(i)*0.03125
		dtBiasValues[i] = float32((i*7)%17-8) * 0.015625
	}

	conv := mlx.FromValues(convValues, 1, 1, 8192).AsType(mlx.DTypeBFloat16)
	alpha := mlx.FromValues(alphaValues, 1, 1, 32).AsType(mlx.DTypeBFloat16)
	beta := mlx.FromValues(betaValues, 1, 1, 32).AsType(mlx.DTypeBFloat16)
	aExp := mlx.FromValues(aExpValues, 32)
	dtBias := mlx.FromValues(dtBiasValues, 32).AsType(mlx.DTypeBFloat16)

	gotQ, gotK, gotV, gotG, gotBeta, ok := mlx.FastQwenGatedDeltaInputs(
		conv, alpha, beta, aExp, dtBias,
	)
	if !ok {
		t.Skip("MLX CUDA custom kernels unavailable")
	}

	activated := mlx.SiLU(conv)
	wantQ := mlx.Reshape(
		mlx.SliceStartStop(activated, []int32{0, 0, 0}, []int32{1, 1, 2048}),
		1, 1, 16, 128,
	)
	wantK := mlx.Reshape(
		mlx.SliceStartStop(activated, []int32{0, 0, 2048}, []int32{1, 1, 4096}),
		1, 1, 16, 128,
	)
	wantV := mlx.Reshape(
		mlx.SliceStartStop(activated, []int32{0, 0, 4096}, []int32{1, 1, 8192}),
		1, 1, 32, 128,
	)
	wantQ = mlx.MulScalar(mlx.RMSNormFn(wantQ, nil, 1e-6), 1.0/128.0)
	wantK = mlx.MulScalar(mlx.RMSNormFn(wantK, nil, 1e-6), 0.08838835)
	wantG := mlx.Softplus(mlx.Add(alpha, dtBias))
	wantG = mlx.Exp(mlx.MulScalar(mlx.Mul(wantG, aExp), -1)).AsType(mlx.DTypeBFloat16)
	wantBeta := mlx.Sigmoid(beta)

	assertArrayClose(t, "v", gotV, wantV, 1e-3)
	assertArrayClose(t, "q", gotQ, wantQ, 1e-3)
	assertArrayClose(t, "k", gotK, wantK, 1e-3)
	assertArrayClose(t, "g", gotG, wantG, 1e-3)
	assertArrayClose(t, "beta", gotBeta, wantBeta, 1e-3)

	stateValues := make([]float32, 4096)
	zValues := make([]float32, 4096)
	normValues := make([]float32, 128)
	for i := range stateValues {
		stateValues[i] = float32((i*19)%127-63) * 0.015625
		zValues[i] = float32((i*23)%131-65) * 0.015625
	}
	for i := range normValues {
		normValues[i] = 0.75 + float32(i%17)*0.015625
	}
	state := mlx.FromValues(stateValues, 1, 1, 32, 128).AsType(mlx.DTypeBFloat16)
	z := mlx.FromValues(zValues, 1, 1, 32, 128).AsType(mlx.DTypeBFloat16)
	norm := mlx.FromValues(normValues, 128).AsType(mlx.DTypeBFloat16)

	gotOut, ok := mlx.FastQwenGatedDeltaOutput(state, z, norm)
	if !ok {
		t.Fatal("MLX CUDA output fusion unexpectedly unavailable")
	}
	wantOut := mlx.RMSNormFn(state, norm, 1e-6)
	wantOut = mlx.Mul(
		wantOut.AsType(mlx.DTypeFloat32),
		mlx.SiLU(z.AsType(mlx.DTypeFloat32)),
	).AsType(mlx.DTypeBFloat16)
	assertArrayClose(t, "output", gotOut, wantOut, 1e-3)
}

func TestSwitchMLPDecodeLHSIndicesMatchImplicit(t *testing.T) {
	useMLXTestThread(t)

	const (
		hidden  = 32
		experts = 2
		topK    = 2
	)
	pattern := func(size int, multiplier int) []float32 {
		values := make([]float32, size)
		for i := range values {
			values[i] = float32((i*multiplier)%127-63) * 0.00390625
		}
		return values
	}

	x := mlx.FromValues(pattern(hidden, 17), 1, 1, hidden).AsType(mlx.DTypeBFloat16)
	indices := mlx.FromValues([]uint32{0, 1}, 1, 1, topK)
	gateUpWeight := mlx.FromValues(
		pattern(experts*hidden*hidden*2, 19),
		experts, hidden, hidden*2,
	).AsType(mlx.DTypeBFloat16)
	downWeight := mlx.FromValues(
		pattern(experts*hidden*hidden, 23),
		experts, hidden, hidden,
	).AsType(mlx.DTypeBFloat16)

	implicit := &SwitchMLP{
		GateUpWeight: gateUpWeight,
		DownWeight:   downWeight,
	}
	explicit := &SwitchMLP{
		GateUpWeight:           gateUpWeight,
		DownWeight:             downWeight,
		DecodeGateUpLHSIndices: mlx.FromValues([]uint32{0, 0}, 1, topK),
		DecodeDownLHSIndices:   mlx.FromValues([]uint32{0, 1}, 1, topK),
	}
	cfg := &Config{HiddenSize: hidden, NumExpertsPerTok: topK}

	want := implicit.Forward(x, indices, cfg)
	got := explicit.Forward(x, indices, cfg)
	assertArrayClose(t, "switch mlp cached decode indices", got, want, 1e-5)
}

func TestDenseGateUpFusionCUDAMatchesSeparate(t *testing.T) {
	useMLXTestThread(t)
	if !mlx.CUDAIsAvailable() {
		t.Skip("CUDA is unavailable")
	}

	values := func(size, multiplier int) []float32 {
		out := make([]float32, size)
		for i := range out {
			out[i] = float32((i*multiplier)%127-63) * 0.0078125
		}
		return out
	}
	x := mlx.FromValues(values(32, 17), 1, 1, 32).AsType(mlx.DTypeBFloat16)
	gate := nn.NewQuantizedLinear(
		mlx.FromValues(values(16*32, 19), 16, 32).AsType(mlx.DTypeBFloat16),
		nil, 16, 4, "nvfp4",
	)
	up := nn.NewQuantizedLinear(
		mlx.FromValues(values(16*32, 23), 16, 32).AsType(mlx.DTypeBFloat16),
		nil, 16, 4, "nvfp4",
	)

	fused, gateScale, upScale := fuseDenseQuantizedLinears(gate, up)
	if fused == nil {
		t.Fatal("expected compatible CUDA gate/up projections to fuse")
	}
	gateUp := fused.Forward(x)
	gotGate, gotUp := splitLastAxisHalves(gateUp)
	gotGate = applyDenseGlobalScale(gotGate, gateScale)
	gotUp = applyDenseGlobalScale(gotUp, upScale)

	assertArrayClose(t, "fused gate", gotGate, gate.Forward(x), 1e-5)
	assertArrayClose(t, "fused up", gotUp, up.Forward(x), 1e-5)
}

func TestCUDAPlatformSharedGateUpFusionPolicy(t *testing.T) {
	for _, tc := range []struct {
		goos, goarch string
		major, minor int
		want         bool
	}{
		{goos: "linux", goarch: "arm64", major: 12, minor: 1, want: true},
		{goos: "linux", goarch: "amd64", major: 12, minor: 0, want: false},
		{goos: "windows", goarch: "arm64", major: 12, minor: 1, want: false},
		{goos: "linux", goarch: "amd64", major: 9, minor: 0, want: true},
	} {
		if got := cudaPlatformSupportsSharedGateUpFusion(tc.goos, tc.goarch, tc.major, tc.minor); got != tc.want {
			t.Errorf("cudaPlatformSupportsSharedGateUpFusion(%q, %q, %d, %d) = %v, want %v", tc.goos, tc.goarch, tc.major, tc.minor, got, tc.want)
		}
	}
}

func TestPlatformQuantizedExpertGateUpFusionPolicy(t *testing.T) {
	for _, tc := range []struct {
		goos, goarch string
		want         bool
	}{
		{goos: "linux", goarch: "arm64", want: true},
		{goos: "linux", goarch: "amd64", want: true},
		{goos: "windows", goarch: "arm64", want: false},
		{goos: "darwin", goarch: "arm64", want: true},
	} {
		if got := platformSupportsQuantizedExpertGateUpFusion(tc.goos, tc.goarch); got != tc.want {
			t.Errorf("platformSupportsQuantizedExpertGateUpFusion(%q, %q) = %v, want %v", tc.goos, tc.goarch, got, tc.want)
		}
	}
}

func TestSwitchMLPQuantizedFusedGateUpMatchesSeparate(t *testing.T) {
	useMLXTestThread(t)
	if !mlx.CUDAIsAvailable() {
		t.Skip("CUDA is unavailable")
	}

	const (
		hidden  = 32
		experts = 2
		topK    = 2
	)
	values := func(size, multiplier int) []float32 {
		out := make([]float32, size)
		for i := range out {
			out[i] = float32((i*multiplier)%127-63) * 0.00390625
		}
		return out
	}
	x := mlx.FromValues(values(hidden*2, 17), 1, 2, hidden).AsType(mlx.DTypeBFloat16)
	indices := mlx.FromValues([]uint32{0, 1, 1, 0}, 1, 2, topK)
	gateWeight := mlx.FromValues(values(experts*hidden*hidden, 19), experts, hidden, hidden).AsType(mlx.DTypeBFloat16)
	upWeight := mlx.FromValues(values(experts*hidden*hidden, 23), experts, hidden, hidden).AsType(mlx.DTypeBFloat16)
	downWeight := mlx.FromValues(values(experts*hidden*hidden, 29), experts, hidden, hidden).AsType(mlx.DTypeBFloat16)
	gateQ, gateScales, gateBiases := mlx.Quantize(gateWeight, 32, 8, "mxfp8")
	upQ, upScales, upBiases := mlx.Quantize(upWeight, 32, 8, "mxfp8")
	downQ, downScales, downBiases := mlx.Quantize(downWeight, 32, 8, "mxfp8")

	separate := &SwitchMLP{
		GateWeightQ:   gateQ,
		GateScales:    gateScales,
		GateBiases:    gateBiases,
		GateBits:      8,
		GateGroupSize: 32,
		GateMode:      "mxfp8",
		UpWeightQ:     upQ,
		UpScales:      upScales,
		UpBiases:      upBiases,
		UpBits:        8,
		UpGroupSize:   32,
		UpMode:        "mxfp8",
		DownWeightQ:   downQ,
		DownScales:    downScales,
		DownBiases:    downBiases,
		DownBits:      8,
		DownGroupSize: 32,
		DownMode:      "mxfp8",
	}
	fused := &SwitchMLP{
		GateUpWeightQ:   fuseExpertStacks(gateQ, upQ, 1),
		GateUpScales:    fuseExpertStacks(gateScales, upScales, 1),
		GateUpBias:      fuseExpertStacks(gateBiases, upBiases, 1),
		GateUpBits:      8,
		GateUpGroupSize: 32,
		GateUpMode:      "mxfp8",
		DownWeightQ:     downQ,
		DownScales:      downScales,
		DownBiases:      downBiases,
		DownBits:        8,
		DownGroupSize:   32,
		DownMode:        "mxfp8",
	}
	cfg := &Config{HiddenSize: hidden, NumExpertsPerTok: topK}

	got := separate.Forward(x, indices, cfg)
	want := fused.Forward(x, indices, cfg)
	assertArrayClose(t, "separate quantized switch mlp", got, want, 1e-5)
}

func TestShouldCacheDecodeExpertIndices(t *testing.T) {
	for _, tc := range []struct {
		name                string
		useQuantizedExperts bool
		topK                int32
		want                bool
	}{
		{name: "quantized moe", useQuantizedExperts: true, topK: 8, want: true},
		{name: "dense model", useQuantizedExperts: true, topK: 0, want: false},
		{name: "unquantized moe", useQuantizedExperts: false, topK: 8, want: false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := shouldCacheDecodeExpertIndices(tc.useQuantizedExperts, tc.topK); got != tc.want {
				t.Fatalf("shouldCacheDecodeExpertIndices(%v, %d) = %v, want %v",
					tc.useQuantizedExperts, tc.topK, got, tc.want)
			}
		})
	}
}

func TestDenseQuantizedLinearFusionPreservesOutputOrder(t *testing.T) {
	useMLXTestThread(t)
	if !mlx.CUDAIsAvailable() {
		t.Skip("CUDA is unavailable")
	}

	values := func(size, multiplier int) []float32 {
		out := make([]float32, size)
		for i := range out {
			out[i] = float32((i*multiplier)%127-63) * 0.0078125
		}
		return out
	}
	x := mlx.FromValues(values(32, 17), 1, 1, 32).AsType(mlx.DTypeBFloat16)
	first := nn.NewQuantizedLinear(
		mlx.FromValues(values(16*32, 19), 16, 32).AsType(mlx.DTypeBFloat16),
		nil, 16, 4, "nvfp4",
	)
	second := nn.NewQuantizedLinear(
		mlx.FromValues(values(8*32, 23), 8, 32).AsType(mlx.DTypeBFloat16),
		nil, 16, 4, "nvfp4",
	)

	fused, firstScale, secondScale := fuseDenseQuantizedLinears(first, second)
	if fused == nil || firstScale != nil || secondScale != nil {
		t.Fatal("expected compatible CUDA projections to fuse without output scales")
	}
	out := fused.Forward(x)
	gotFirst := mlx.SliceStartStop(out, []int32{0, 0, 0}, []int32{1, 1, 16})
	gotSecond := mlx.SliceStartStop(out, []int32{0, 0, 16}, []int32{1, 1, 24})
	assertArrayClose(t, "fused first projection", gotFirst, first.Forward(x), 1e-5)
	assertArrayClose(t, "fused second projection", gotSecond, second.Forward(x), 1e-5)
}

func TestQwenSharedSwiGLUCUDAMatchesMLX(t *testing.T) {
	useMLXTestThread(t)

	values := make([]float32, 1024)
	for i := range values {
		values[i] = float32((i*31)%257-128) * 0.015625
	}
	gateUp := mlx.FromValues(values, 1, 1, 1024).AsType(mlx.DTypeBFloat16)
	got, ok := mlx.FastQwenSharedSwiGLU(gateUp)
	if !ok {
		t.Skip("Qwen shared SwiGLU CUDA kernel unavailable")
	}
	gate, up := splitLastAxisHalves(gateUp)
	want := mlx.SwiGLU(gate, up)
	assertArrayClose(t, "qwen shared swiglu", got, want, 1e-3)
}

func assertArrayClose(t *testing.T, name string, got, want *mlx.Array, tolerance float32) {
	t.Helper()
	gotF32 := got.AsType(mlx.DTypeFloat32)
	wantF32 := want.AsType(mlx.DTypeFloat32)
	mlx.Eval(gotF32, wantF32)
	gotValues := gotF32.Floats()
	wantValues := wantF32.Floats()
	if len(gotValues) != len(wantValues) {
		t.Fatalf("%s: got %d values, want %d", name, len(gotValues), len(wantValues))
	}
	var maxDiff float32
	var maxIndex int
	for i := range gotValues {
		diff := gotValues[i] - wantValues[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > maxDiff {
			maxDiff = diff
			maxIndex = i
		}
	}
	if maxDiff > tolerance {
		t.Fatalf(
			"%s: max abs diff %v at %d (got %v, want %v)",
			name, maxDiff, maxIndex, gotValues[maxIndex], wantValues[maxIndex],
		)
	}
}

func useMLXTestThread(t *testing.T) {
	t.Helper()
	runtime.LockOSThread()
	initialized := false
	t.Cleanup(func() {
		if initialized {
			mlx.Sweep()
			mlx.ClearCache()
		}
		runtime.UnlockOSThread()
	})

	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX not available: %v", err)
	}
	initialized = true
	if mlx.GPUIsAvailable() {
		mlx.SetDefaultDeviceGPU()
	}
}

func TestParseConfigNestedDefaults(t *testing.T) {
	data := []byte(`{
		"model_type": "Qwen3_5MoeForConditionalGeneration",
		"text_config": {
			"hidden_size": 4096,
			"intermediate_size": 14336,
			"num_hidden_layers": 8,
			"num_attention_heads": 32,
			"num_key_value_heads": 8,
			"head_dim": 128,
			"linear_num_value_heads": 64,
			"linear_num_key_heads": 16,
			"linear_key_head_dim": 128,
			"linear_value_head_dim": 128,
			"linear_conv_kernel_dim": 4,
			"num_experts": 16,
			"num_experts_per_tok": 4,
			"moe_intermediate_size": 2048,
			"shared_expert_intermediate_size": 4096,
			"rope_parameters": {
				"rope_theta": 500000,
				"partial_rotary_factor": 0.5
			}
		}
	}`)

	cfg, err := parseConfig(data)
	if err != nil {
		t.Fatalf("parseConfig failed: %v", err)
	}

	if cfg.RopeTheta != 500000 {
		t.Fatalf("rope theta mismatch: got %v", cfg.RopeTheta)
	}
	if cfg.RopeDim != 64 {
		t.Fatalf("rope dim mismatch: got %d want 64", cfg.RopeDim)
	}
	if cfg.FullAttentionInterval != 4 {
		t.Fatalf("full_attention_interval default mismatch: got %d want 4", cfg.FullAttentionInterval)
	}
	if !cfg.NormTopKProb {
		t.Fatalf("norm_topk_prob should default to true for MoE")
	}
}

func TestLayerSelectionHelpers(t *testing.T) {
	cfg := &Config{
		NumHiddenLayers:       6,
		FullAttentionInterval: 3,
		NumExperts:            8,
		DecoderSparseStep:     2,
		MLPOnlyLayers:         []int32{1},
	}

	if !layerIsLinear(cfg, 0) {
		t.Fatalf("layer 0 should be linear")
	}
	if layerIsLinear(cfg, 2) {
		t.Fatalf("layer 2 should be full attention")
	}

	if layerUsesMoE(cfg, 1) {
		t.Fatalf("layer 1 should be forced dense by mlp_only_layers")
	}
	if !layerUsesMoE(cfg, 3) {
		t.Fatalf("layer 3 should use moe with decoder_sparse_step=2")
	}
}

func TestSupportsGatherQMM(t *testing.T) {
	tests := []struct {
		mode string
		bits int
		want bool
	}{
		{mode: "affine", bits: 4, want: true},
		{mode: "affine", bits: 8, want: true},
		{mode: "mxfp8", bits: 8, want: true},
		{mode: "nvfp4", bits: 4, want: true},
		{mode: "mxfp4", bits: 4, want: true},
		{mode: "mxfp8", bits: 4, want: false},
		{mode: "affine", bits: 3, want: false},
	}

	for _, tt := range tests {
		if got := supportsGatherQMM(tt.mode, tt.bits); got != tt.want {
			t.Fatalf("supportsGatherQMM(%q, %d) = %v, want %v", tt.mode, tt.bits, got, tt.want)
		}
	}
}

func TestResolveTensorPathLayout(t *testing.T) {
	dummy := mlx.New("dummy")

	tests := []struct {
		name          string
		key           string
		wantContainer string
		wantModel     string
	}{
		{
			name:          "standard",
			key:           "model.embed_tokens.weight",
			wantContainer: "",
			wantModel:     "model.",
		},
		{
			name:          "nested language model with inner model",
			key:           "model.language_model.model.embed_tokens.weight",
			wantContainer: "model.language_model.",
			wantModel:     "model.",
		},
		{
			name:          "nested language model without inner model",
			key:           "model.language_model.embed_tokens.weight",
			wantContainer: "model.language_model.",
			wantModel:     "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			layout := resolveTensorPathLayout(map[string]*mlx.Array{
				tt.key: dummy,
			})

			if layout.containerPrefix != tt.wantContainer || layout.modelPrefix != tt.wantModel {
				t.Fatalf(
					"resolveTensorPathLayout() = {%q %q}, want {%q %q}",
					layout.containerPrefix,
					layout.modelPrefix,
					tt.wantContainer,
					tt.wantModel,
				)
			}
		})
	}
}

func TestNewCachesLayout(t *testing.T) {
	m := &Model{
		Config: &Config{
			LinearConvKernelDim: 4,
			LinearNumKeyHeads:   2,
			LinearKeyHeadDim:    8,
			LinearNumValueHeads: 4,
			LinearValueHeadDim:  16,
		},
		Layers: []*Layer{
			{IsLinear: true},
			{IsLinear: false},
			{IsLinear: true},
		},
	}

	caches := m.NewCaches()
	if len(caches) != len(m.Layers) {
		t.Fatalf("len(caches) = %d, want %d", len(caches), len(m.Layers))
	}

	if _, ok := caches[0].(*cache.RecurrentCache); !ok {
		t.Fatalf("cache[0] = %T, want *cache.RecurrentCache", caches[0])
	}
	if _, ok := caches[1].(*cache.KVCache); !ok {
		t.Fatalf("cache[1] = %T, want *cache.KVCache", caches[1])
	}
	if _, ok := caches[2].(*cache.RecurrentCache); !ok {
		t.Fatalf("cache[2] = %T, want *cache.RecurrentCache", caches[2])
	}
}

func TestLoadWeightsPreservesLinearAttentionNormWeightDType(t *testing.T) {
	useMLXTestThread(t)

	cfg := &Config{
		HiddenSize:            4,
		IntermediateSize:      8,
		NumHiddenLayers:       2,
		NumAttentionHeads:     1,
		NumKeyValueHeads:      1,
		HeadDim:               4,
		RMSNormEps:            1e-6,
		TieWordEmbeddings:     true,
		LayerTypes:            []string{"linear", "full"},
		LinearNumValueHeads:   1,
		LinearNumKeyHeads:     1,
		LinearKeyHeadDim:      2,
		LinearValueHeadDim:    2,
		LinearConvKernelDim:   4,
		FullAttentionInterval: 2,
	}

	m := &Model{
		Config: cfg,
		Layers: make([]*Layer, cfg.NumHiddenLayers),
	}

	bf16 := mlx.DTypeBFloat16
	f32 := mlx.DTypeFloat32
	tensors := map[string]*mlx.Array{
		"model.embed_tokens.weight":                      mlx.FromValues([]float32{1, 2, 3, 4, 5, 6, 7, 8}, 2, 4).AsType(bf16),
		"model.norm.weight":                              mlx.FromValues([]float32{1, 1, 1, 1}, 4),
		"model.layers.0.input_layernorm.weight":          mlx.FromValues([]float32{1, 1, 1, 1}, 4),
		"model.layers.0.post_attention_layernorm.weight": mlx.FromValues([]float32{1, 1, 1, 1}, 4),
		"model.layers.0.linear_attn.in_proj_qkv.weight": mlx.FromValues([]float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
			1, 1, 0, 0,
			0, 1, 1, 0,
		}, 6, 4),
		"model.layers.0.linear_attn.in_proj_z.weight": mlx.FromValues([]float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
		}, 2, 4),
		"model.layers.0.linear_attn.in_proj_b.weight": mlx.FromValues([]float32{1, 0, 0, 0}, 1, 4),
		"model.layers.0.linear_attn.in_proj_a.weight": mlx.FromValues([]float32{0, 1, 0, 0}, 1, 4),
		"model.layers.0.linear_attn.out_proj.weight": mlx.FromValues([]float32{
			1, 0,
			0, 1,
			1, 1,
			0, 0,
		}, 4, 2),
		"model.layers.0.linear_attn.conv1d.weight": mlx.FromValues([]float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
			1, 1, 0, 0,
			0, 1, 1, 0,
		}, 6, 4),
		"model.layers.0.linear_attn.norm.weight": mlx.FromValues([]float32{1, 1}, 2),
		"model.layers.0.linear_attn.dt_bias":     mlx.FromValues([]float32{0}, 1),
		"model.layers.0.linear_attn.A_log":       mlx.FromValues([]float32{0}, 1),
		"model.layers.0.mlp.gate_proj.weight": mlx.FromValues([]float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
			1, 1, 0, 0,
			0, 1, 1, 0,
			0, 0, 1, 1,
			1, 0, 0, 1,
		}, 8, 4),
		"model.layers.0.mlp.up_proj.weight": mlx.FromValues([]float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
			1, 1, 0, 0,
			0, 1, 1, 0,
			0, 0, 1, 1,
			1, 0, 0, 1,
		}, 8, 4),
		"model.layers.0.mlp.down_proj.weight": mlx.FromValues([]float32{
			1, 0, 0, 0, 0, 0, 0, 0,
			0, 1, 0, 0, 0, 0, 0, 0,
			0, 0, 1, 0, 0, 0, 0, 0,
			0, 0, 0, 1, 0, 0, 0, 0,
		}, 4, 8),
		"model.layers.1.input_layernorm.weight":          mlx.FromValues([]float32{1, 1, 1, 1}, 4),
		"model.layers.1.post_attention_layernorm.weight": mlx.FromValues([]float32{1, 1, 1, 1}, 4),
		"model.layers.1.self_attn.q_proj.weight": mlx.FromValues([]float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
			1, 1, 0, 0,
			0, 1, 1, 0,
			0, 0, 1, 1,
			1, 0, 0, 1,
		}, 8, 4),
		"model.layers.1.self_attn.k_proj.weight": mlx.FromValues([]float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
		}, 4, 4),
		"model.layers.1.self_attn.v_proj.weight": mlx.FromValues([]float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
		}, 4, 4),
		"model.layers.1.self_attn.o_proj.weight": mlx.FromValues([]float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
		}, 4, 4),
		"model.layers.1.self_attn.q_norm.weight": mlx.FromValues([]float32{1, 1, 1, 1}, 4),
		"model.layers.1.self_attn.k_norm.weight": mlx.FromValues([]float32{1, 1, 1, 1}, 4),
		"model.layers.1.mlp.gate_proj.weight": mlx.FromValues([]float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
			1, 1, 0, 0,
			0, 1, 1, 0,
			0, 0, 1, 1,
			1, 0, 0, 1,
		}, 8, 4),
		"model.layers.1.mlp.up_proj.weight": mlx.FromValues([]float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
			1, 1, 0, 0,
			0, 1, 1, 0,
			0, 0, 1, 1,
			1, 0, 0, 1,
		}, 8, 4),
		"model.layers.1.mlp.down_proj.weight": mlx.FromValues([]float32{
			1, 0, 0, 0, 0, 0, 0, 0,
			0, 1, 0, 0, 0, 0, 0, 0,
			0, 0, 1, 0, 0, 0, 0, 0,
			0, 0, 0, 1, 0, 0, 0, 0,
		}, 4, 8),
	}

	if err := m.LoadWeights(tensors); err != nil {
		t.Fatalf("LoadWeights failed: %v", err)
	}

	if got := m.Layers[0].InputNorm.Weight.DType(); got != f32 {
		t.Fatalf("layer 0 input norm dtype = %v, want %v", got, f32)
	}
	if got := m.Layers[0].PostAttentionNorm.Weight.DType(); got != f32 {
		t.Fatalf("layer 0 post-attn norm dtype = %v, want %v", got, f32)
	}
	if got := m.Layers[1].InputNorm.Weight.DType(); got != f32 {
		t.Fatalf("layer 1 input norm dtype = %v, want %v", got, f32)
	}
	if got := m.Layers[1].PostAttentionNorm.Weight.DType(); got != f32 {
		t.Fatalf("layer 1 post-attn norm dtype = %v, want %v", got, f32)
	}

	if got := m.Norm.Weight.DType(); got != f32 {
		t.Fatalf("final norm dtype = %v, want %v", got, f32)
	}
	if got := m.Layers[0].Linear.NormWeight.DType(); got != f32 {
		t.Fatalf("linear-attn norm dtype = %v, want %v", got, f32)
	}
	if got := m.Layers[1].FullAttn.QNorm.Weight.DType(); got != f32 {
		t.Fatalf("q norm dtype = %v, want %v", got, f32)
	}
	if got := m.Layers[1].FullAttn.KNorm.Weight.DType(); got != f32 {
		t.Fatalf("k norm dtype = %v, want %v", got, f32)
	}
}
