package gemma4

import (
	"runtime"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func useMLXTestThread(t *testing.T) {
	t.Helper()

	runtime.LockOSThread()
	initialized := false
	t.Cleanup(func() {
		if initialized {
			mlx.Sweep()
			mlx.ClearCache()
			if mlx.GPUIsAvailable() {
				mlx.SetDefaultDeviceGPU()
			}
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

// onesLike creates a tensor of the given shape filled with a small constant.
func onesLike(shape ...int) *mlx.Array {
	return mlx.AddScalar(mlx.Zeros(mlx.DTypeBFloat16, shape...), 0.01)
}

func tinyMoEConfig() *TextConfig {
	return &TextConfig{
		HiddenSize:             16, // tiny for testing
		NumAttentionHeads:      2,
		NumKeyValueHeads:       1,
		NumGlobalKeyValueHeads: 1,
		HeadDim:                8,
		GlobalHeadDim:          8,
		NumExperts:             4,
		TopKExperts:            2,
		ExpertIntermediateSize: 8,
		EnableMoeBlock:         true,
		AttentionKEqV:          false,
		RMSNormEps:             1e-6,
		SlidingScale:           1.0,
		FullScale:              1.0,
	}
}

func newRouter(cfg *TextConfig) *Router {
	return &Router{
		Proj:  linearFromWeight(onesLike(int(cfg.NumExperts), int(cfg.HiddenSize))),
		Scale: onesLike(int(cfg.HiddenSize)),
	}
}

func newMoEBlock(cfg *TextConfig) *MoEBlock {
	return &MoEBlock{
		GateWeight:     onesLike(int(cfg.NumExperts), int(cfg.HiddenSize), int(cfg.ExpertIntermediateSize)),
		UpWeight:       onesLike(int(cfg.NumExperts), int(cfg.HiddenSize), int(cfg.ExpertIntermediateSize)),
		DownWeight:     onesLike(int(cfg.NumExperts), int(cfg.ExpertIntermediateSize), int(cfg.HiddenSize)),
		PerExpertScale: onesLike(int(cfg.NumExperts)),
	}
}

func TestMoERouterForward(t *testing.T) {
	useMLXTestThread(t)

	cfg := tinyMoEConfig()
	B, L := int32(1), int32(3)
	x := onesLike(int(B), int(L), int(cfg.HiddenSize))
	router := newRouter(cfg)

	scores, inds, _ := router.Forward(x, cfg)
	mlx.Eval(scores, inds)

	sDims := scores.Dims()
	iDims := inds.Dims()
	t.Logf("scores shape: %v, inds shape: %v", sDims, iDims)

	if len(sDims) != 2 || sDims[0] != int(B*L) || sDims[1] != int(cfg.TopKExperts) {
		t.Errorf("scores shape = %v, want [%d, %d]", sDims, B*L, cfg.TopKExperts)
	}
	if len(iDims) != 2 || iDims[0] != int(B*L) || iDims[1] != int(cfg.TopKExperts) {
		t.Errorf("inds shape = %v, want [%d, %d]", iDims, B*L, cfg.TopKExperts)
	}
}

func TestMoEBlockForward(t *testing.T) {
	useMLXTestThread(t)

	cfg := tinyMoEConfig()
	B, L := int32(1), int32(3)
	x := onesLike(int(B), int(L), int(cfg.HiddenSize))
	router := newRouter(cfg)
	moe := newMoEBlock(cfg)

	scores, inds, scoresIncludeExpertScale := router.Forward(x, cfg)
	mlx.Eval(scores, inds)

	out := moe.Forward(x, scores, inds, scoresIncludeExpertScale, cfg)
	mlx.Eval(out)

	outDims := out.Dims()
	t.Logf("MoE output shape: %v", outDims)

	if len(outDims) != 3 || outDims[0] != int(B) || outDims[1] != int(L) || outDims[2] != int(cfg.HiddenSize) {
		t.Errorf("output shape = %v, want [%d, %d, %d]", outDims, B, L, cfg.HiddenSize)
	}
}

func TestMoEBlockSortedForward(t *testing.T) {
	useMLXTestThread(t)

	cfg := tinyMoEConfig()
	B, L := int32(1), int32(128)
	x := onesLike(int(B), int(L), int(cfg.HiddenSize))
	router := newRouter(cfg)
	moe := newMoEBlock(cfg)

	scores, inds, scoresIncludeExpertScale := router.Forward(x, cfg)
	mlx.Eval(scores, inds)

	out := moe.Forward(x, scores, inds, scoresIncludeExpertScale, cfg)
	mlx.Eval(out)

	outDims := out.Dims()
	t.Logf("MoE sorted output shape: %v", outDims)

	if len(outDims) != 3 || outDims[0] != int(B) || outDims[1] != int(L) || outDims[2] != int(cfg.HiddenSize) {
		t.Errorf("output shape = %v, want [%d, %d, %d]", outDims, B, L, cfg.HiddenSize)
	}
}

func TestMoEWeightedSumCompiledMatchesEager(t *testing.T) {
	useMLXTestThread(t)

	const tokens, topK, hidden = 3, 4, 16
	downValues := make([]float32, tokens*topK*hidden)
	for i := range downValues {
		downValues[i] = float32((i%23)-11) * 0.025
	}
	scoreValues := []float32{
		0.1, 0.2, 0.3, 0.4,
		0.4, 0.3, 0.2, 0.1,
		0.15, 0.25, 0.35, 0.25,
	}
	scaleValues := []float32{
		0.75, 1.0, 1.25, 1.5,
		1.5, 1.25, 1.0, 0.75,
		0.8, 1.1, 1.4, 0.9,
	}

	down := mlx.FromValues(downValues, tokens, topK, hidden).AsType(mlx.DTypeBFloat16)
	scores := mlx.FromValues(scoreValues, tokens, topK)
	scales := mlx.FromValues(scaleValues, tokens, topK).AsType(mlx.DTypeBFloat16)

	want := gemma4MoEWeightedSumEager(down, scores, scales)
	got := gemma4MoEWeightedSumCompiled(down, scores, scales)
	mlx.Eval(want, got)

	if gotValues, wantValues := got.Floats(), want.Floats(); !floatSlicesClose(gotValues, wantValues, 1e-5) {
		t.Fatalf("compiled weighted sum mismatch:\n  got  %v\n  want %v", gotValues, wantValues)
	}
}

func TestSortedMoEDispatchCUDAKernelMatchesMLX(t *testing.T) {
	useMLXTestThread(t)

	const rows, topK, hidden = 64, 8, 2816
	assignments := rows * topK

	xValues := make([]float32, rows*hidden)
	for i := range xValues {
		xValues[i] = float32((i%97)-48) * 0.00390625
	}
	x := mlx.FromValues(xValues, rows, 1, 1, hidden).
		AsType(mlx.DTypeBFloat16)

	orderValues := make([]uint32, assignments)
	for sorted := range assignments {
		orderValues[sorted] = uint32((sorted * 37) % assignments)
	}
	order := mlx.FromValues(orderValues, assignments)

	want := mlx.ExpandDims(
		mlx.Take(
			mlx.Squeeze(x, 1),
			mlx.FloorDivideScalar(order, topK),
			0,
		),
		1,
	)
	got, ok := mlx.FastSortedMoEDispatch(x, order)
	if !ok {
		t.Skip("MLX CUDA custom kernels unavailable")
	}

	want = want.AsType(mlx.DTypeFloat32)
	got = got.AsType(mlx.DTypeFloat32)
	mlx.Eval(want, got)

	if gotValues, wantValues := got.Floats(), want.Floats(); !floatSlicesClose(gotValues, wantValues, 0) {
		t.Fatal("sorted MoE dispatch output does not match MLX gather")
	}
}

func TestSortedMoECombineCUDAKernelMatchesMLX(t *testing.T) {
	useMLXTestThread(t)

	const rows, topK, hidden = 64, 8, 2816
	assignments := rows * topK

	downValues := make([]float32, assignments*hidden)
	for i := range downValues {
		downValues[i] = float32((i%37)-18) * 0.00625
	}
	downOriginal := mlx.FromValues(downValues, assignments, hidden).
		AsType(mlx.DTypeBFloat16)

	orderValues := make([]uint32, assignments)
	invOrderValues := make([]uint32, assignments)
	for sorted := range assignments {
		original := (sorted * 37) % assignments
		orderValues[sorted] = uint32(original)
		invOrderValues[original] = uint32(sorted)
	}
	order := mlx.FromValues(orderValues, assignments)
	downSorted := mlx.Take(downOriginal, order, 0).
		Reshape(assignments, 1, 1, hidden)
	invOrder := mlx.FromValues(invOrderValues, assignments)

	scoreValues := make([]float32, assignments)
	indexValues := make([]uint32, assignments)
	for token := range rows {
		for expert := range topK {
			offset := token*topK + expert
			scoreValues[offset] = float32(expert+1) / 36
			indexValues[offset] = uint32((token*11 + expert*17) % 128)
		}
	}
	scores := mlx.FromValues(scoreValues, rows, topK).
		AsType(mlx.DTypeBFloat16)
	indices := mlx.FromValues(indexValues, rows, topK)

	scaleValues := make([]float32, 128)
	for i := range scaleValues {
		scaleValues[i] = 0.75 + float32(i%13)*0.03125
	}
	expertScales := mlx.FromValues(scaleValues, 128).
		AsType(mlx.DTypeBFloat16)

	down := mlx.Take(
		downSorted.Reshape(assignments, hidden),
		invOrder,
		0,
	).Reshape(rows, topK, hidden)
	selectedScales := mlx.Take(
		expertScales,
		mlx.Flatten(indices),
		0,
	).Reshape(rows, topK)

	tests := []struct {
		name             string
		scores           *mlx.Array
		applyExpertScale bool
		want             *mlx.Array
	}{
		{
			name:             "raw_scores",
			scores:           scores,
			applyExpertScale: true,
			want:             gemma4MoEWeightedSumEager(down, scores, selectedScales),
		},
		{
			name:             "scaled_scores",
			scores:           mlx.Mul(scores, selectedScales),
			applyExpertScale: false,
			want: gemma4MoEScaledWeightedSumCompiled(
				down,
				mlx.Mul(scores, selectedScales),
			),
		},
	}

	for _, tt := range tests {
		got, ok := mlx.FastSortedMoECombine(
			downSorted,
			invOrder,
			tt.scores,
			indices,
			expertScales,
			tt.applyExpertScale,
		)
		if !ok {
			t.Skip("MLX CUDA custom kernels unavailable")
		}

		want := tt.want.AsType(mlx.DTypeFloat32)
		got = got.AsType(mlx.DTypeFloat32)
		mlx.Eval(want, got)

		if gotValues, wantValues := got.Floats(), want.Floats(); !floatSlicesClose(gotValues, wantValues, 1e-5) {
			firstMismatch := -1
			var maxDiff float32
			gotNaNs, wantNaNs := 0, 0
			for i := range gotValues {
				if gotValues[i] != gotValues[i] {
					gotNaNs++
				}
				if wantValues[i] != wantValues[i] {
					wantNaNs++
				}
				diff := gotValues[i] - wantValues[i]
				if diff < 0 {
					diff = -diff
				}
				if diff > maxDiff {
					maxDiff = diff
				}
				if firstMismatch < 0 && diff > 1e-5 {
					firstMismatch = i
				}
			}
			t.Fatalf(
				"%s sorted MoE combine mismatch: first=%d got=%g want=%g max_diff=%g got_nans=%d want_nans=%d",
				tt.name,
				firstMismatch,
				gotValues[firstMismatch],
				wantValues[firstMismatch],
				maxDiff,
				gotNaNs,
				wantNaNs,
			)
		}
	}
}

func TestResidualScaleCompiledMatchesEager(t *testing.T) {
	useMLXTestThread(t)

	residual := mlx.FromValues([]float32{
		0.25, -0.5, 0.75, 1.0,
		-1.25, 1.5, -1.75, 2.0,
	}, 1, 2, 4).AsType(mlx.DTypeBFloat16)
	update := mlx.FromValues([]float32{
		0.125, 0.25, -0.375, 0.5,
		0.625, -0.75, 0.875, -1.0,
	}, 1, 2, 4).AsType(mlx.DTypeBFloat16)
	scale := mlx.FromValues([]float32{0.75}, 1).AsType(mlx.DTypeBFloat16)

	want := mlx.Mul(mlx.Add(residual, update), scale).AsType(mlx.DTypeFloat32)
	got := gemma4ResidualScaleCompiled(residual, update, scale).AsType(mlx.DTypeFloat32)
	mlx.Eval(want, got)

	if gotValues, wantValues := got.Floats(), want.Floats(); !floatSlicesClose(gotValues, wantValues, 1e-5) {
		t.Fatalf("compiled residual scale mismatch:\n  got  %v\n  want %v", gotValues, wantValues)
	}
}

func TestMoEFinalCUDAKernelMatchesMLX(t *testing.T) {
	useMLXTestThread(t)

	const rows, hiddenSize = 2, 2816
	values := func(period int, scale float32) []float32 {
		out := make([]float32, rows*hiddenSize)
		for i := range out {
			out[i] = float32(i%period-period/2) * scale
		}
		return out
	}

	residual := mlx.FromValues(values(41, 0.0125), 1, rows, hiddenSize).AsType(mlx.DTypeBFloat16)
	mlpOut := mlx.FromValues(values(31, 0.00625), 1, rows, hiddenSize).AsType(mlx.DTypeBFloat16)
	moeOut := mlx.FromValues(values(23, 0.003125), 1, rows, hiddenSize).AsType(mlx.DTypeBFloat16)
	normScale := mlx.FromValues(func() []float32 {
		out := make([]float32, hiddenSize)
		for i := range out {
			out[i] = 0.75 + float32(i%7)*0.03125
		}
		return out
	}(), hiddenSize).AsType(mlx.DTypeBFloat16)
	layerScale := mlx.FromValue(float32(0.875)).AsType(mlx.DTypeBFloat16)

	got, ok := mlx.FastMoEFinalResidual(
		residual,
		mlpOut,
		moeOut,
		normScale,
		layerScale,
		1e-6,
	)
	if !ok {
		t.Skip("MLX CUDA custom kernels unavailable")
	}

	combined := mlx.RMSNormFn(mlx.Add(mlpOut, moeOut), normScale, 1e-6)
	want := gemma4ResidualScaleCompiled(residual, combined, layerScale).AsType(mlx.DTypeFloat32)
	got = got.AsType(mlx.DTypeFloat32)
	mlx.Eval(want, got)

	gotValues, wantValues := got.Floats(), want.Floats()
	if !floatSlicesClose(gotValues, wantValues, 1e-5) {
		var maxDiff float32
		first := -1
		for i := range gotValues {
			diff := gotValues[i] - wantValues[i]
			if diff < 0 {
				diff = -diff
			}
			if diff > maxDiff {
				maxDiff = diff
			}
			if first < 0 && diff > 1e-5 {
				first = i
			}
		}
		t.Fatalf(
			"fused MoE tail mismatch: max diff %v, first %d: got %v, want %v",
			maxDiff,
			first,
			gotValues[first],
			wantValues[first],
		)
	}
}

func TestGatherQMMDecodeIdentityIndicesMatchDefault(t *testing.T) {
	useMLXTestThread(t)

	const experts, topK, inputSize, outputSize = 4, 2, 32, 16
	weightValues := make([]float32, experts*outputSize*inputSize)
	for i := range weightValues {
		weightValues[i] = float32((i%29)-14) * 0.0125
	}
	weight := mlx.FromValues(weightValues, experts, outputSize, inputSize)
	qweight, scales, biases := mlx.Quantize(weight, 32, 4, "affine")

	rhsIndices := mlx.FromValues([]uint32{1, 3}, 1, topK)
	gateUpLHSIndices := mlx.FromValues([]uint32{0}, 1, 1)
	downLHSIndices := mlx.FromValues([]uint32{0, 1}, 1, topK)

	for _, tt := range []struct {
		name       string
		input      *mlx.Array
		lhsIndices *mlx.Array
	}{
		{
			name: "gate-up",
			input: mlx.FromValues(func() []float32 {
				values := make([]float32, inputSize)
				for i := range values {
					values[i] = float32(i-16) * 0.025
				}
				return values
			}(), 1, 1, 1, inputSize),
			lhsIndices: gateUpLHSIndices,
		},
		{
			name: "down",
			input: mlx.FromValues(func() []float32 {
				values := make([]float32, topK*inputSize)
				for i := range values {
					values[i] = float32((i%23)-11) * 0.03125
				}
				return values
			}(), 1, topK, 1, inputSize),
			lhsIndices: downLHSIndices,
		},
	} {
		t.Log(tt.name)
		want := mlx.GatherQMM(
			tt.input, qweight, scales, biases,
			nil, rhsIndices, true, 32, 4, "affine", false,
		).AsType(mlx.DTypeFloat32)
		got := mlx.GatherQMM(
			tt.input, qweight, scales, biases,
			tt.lhsIndices, rhsIndices, true, 32, 4, "affine", false,
		).AsType(mlx.DTypeFloat32)
		mlx.Eval(want, got)

		if gotValues, wantValues := got.Floats(), want.Floats(); !floatSlicesClose(gotValues, wantValues, 1e-5) {
			t.Fatalf("%s explicit identity indices mismatch:\n  got  %v\n  want %v", tt.name, gotValues, wantValues)
		}
	}
}

func TestMoERouterCUDAMatchesMLX(t *testing.T) {
	useMLXTestThread(t)

	for _, tt := range []struct {
		name   string
		rows   int
		values []float32
	}{
		{
			name: "unique",
			rows: 3,
			values: func() []float32 {
				values := make([]float32, 3*128)
				for row := range 3 {
					for i := range 128 {
						values[row*128+i] = float32((i*37+row*13)%128 - 64)
					}
				}
				return values
			}(),
		},
		{name: "ties", rows: 2, values: make([]float32, 2*128)},
	} {
		t.Log(tt.name)
		logits := mlx.FromValues(tt.values, tt.rows, 128).AsType(mlx.DTypeBFloat16)
		expertScales := mlx.FromValues(func() []float32 {
			values := make([]float32, 128)
			for i := range values {
				values[i] = 0.75 + float32(i%11)*0.03125
			}
			return values
		}(), 128).AsType(mlx.DTypeBFloat16)
		gotWeights, gotIndices, ok := mlx.FastMoERouter(logits, expertScales)
		if !ok {
			t.Skip("MLX CUDA custom kernels unavailable")
		}

		wantIndices := mlx.Argpartition(mlx.Neg(logits), 7, -1)
		wantIndices = mlx.SliceStartStop(
			wantIndices,
			[]int32{0, 0},
			[]int32{int32(tt.rows), 8},
		)
		wantWeights := mlx.SoftmaxAxis(
			mlx.TakeAlongAxis(logits, wantIndices, -1),
			-1,
			true,
		)
		wantWeights = mlx.Mul(
			wantWeights,
			mlx.Take(expertScales, mlx.Flatten(wantIndices), 0).Reshape(tt.rows, 8),
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
		for row := range tt.rows {
			got := make(map[int]float32, 8)
			want := make(map[int]float32, 8)
			for i := range 8 {
				offset := row*8 + i
				got[gotIndexValues[offset]] = gotWeightValues[offset]
				want[wantIndexValues[offset]] = wantWeightValues[offset]
			}
			if len(got) != len(want) {
				t.Fatalf("%s row %d: selected experts = %v, want %v", tt.name, row, got, want)
			}
			for index, wantWeight := range want {
				gotWeight, exists := got[index]
				if !exists {
					t.Fatalf("%s row %d: selected experts = %v, want %v", tt.name, row, got, want)
				}
				diff := gotWeight - wantWeight
				if diff < 0 {
					diff = -diff
				}
				if diff > 1e-3 {
					t.Fatalf(
						"%s row %d expert %d: weight = %v, want %v",
						tt.name, row, index, gotWeight, wantWeight,
					)
				}
			}
		}
	}
}

// TestLoadFusedExpertsQuantized verifies that a quantized, fused gate_up
// projection is loaded onto the GatherQMM path under every name the experts
// ship as — including gemma's bare ".experts." name, which previously fell
// through to the dense branch and was loaded unquantized (the memory bloat
// bug this fix addresses).
func TestLoadFusedExpertsQuantized(t *testing.T) {
	skipIfNoMLX(t)

	const E, I, H = 4, 8, 16
	m := &Model{TextConfig: &TextConfig{QuantGroupSize: 16, QuantBits: 4, QuantMode: "nvfp4"}}

	for _, prefix := range []string{
		"model.language_model.layers.0.experts",        // gemma HF (bare .experts.)
		"model.language_model.layers.0.moe.switch_mlp", // create pipeline
	} {
		t.Run(prefix, func(t *testing.T) {
			gateUpKey := prefix + ".gate_up_proj"
			downKey := prefix + ".down_proj"
			tensors := map[string]*mlx.Array{
				gateUpKey:            onesLike(E, 2*I, H),
				gateUpKey + "_scale": onesLike(E, 2*I, H/16),
				gateUpKey + "_qbias": onesLike(E, 2*I, H/16),
				downKey:              onesLike(E, H, I),
				downKey + "_scale":   onesLike(E, H, I/16),
				downKey + "_qbias":   onesLike(E, H, I/16),
			}

			moe := &MoEBlock{}
			m.loadFusedExperts(moe, tensors, gateUpKey, tensors[gateUpKey], downKey, tensors[downKey])

			if !moe.UseQuantized {
				t.Error("UseQuantized = false, want true")
			}
			if !moe.UseFusedGateUp {
				t.Error("UseFusedGateUp = false, want true")
			}
			if moe.GateUpWeightQ == nil || moe.GateUpScales == nil || moe.GateUpBiases == nil {
				t.Error("quantized gate_up weight/scale/bias not all set")
			}
			if moe.DownWeightQ == nil || moe.DownScales == nil || moe.DownBiases == nil {
				t.Error("quantized down weight/scale/bias not all set")
			}
			// Dense fields must stay nil so Forward takes the GatherQMM path.
			if moe.GateUpWeight != nil || moe.DownWeight != nil {
				t.Error("dense weights set on a quantized block")
			}
		})
	}
}

// TestLoadFusedExpertsDense verifies that a fused gate_up projection with no
// scale companions is loaded onto the dense GatherMM path, kept fused.
func TestLoadFusedExpertsDense(t *testing.T) {
	skipIfNoMLX(t)

	const E, I, H = 4, 8, 16
	m := &Model{TextConfig: &TextConfig{}}

	gateUpKey := "model.language_model.layers.0.experts.gate_up_proj"
	downKey := "model.language_model.layers.0.experts.down_proj"
	tensors := map[string]*mlx.Array{
		gateUpKey: onesLike(E, 2*I, H),
		downKey:   onesLike(E, I, H),
	}

	moe := &MoEBlock{}
	m.loadFusedExperts(moe, tensors, gateUpKey, tensors[gateUpKey], downKey, tensors[downKey])

	if moe.UseQuantized {
		t.Error("UseQuantized = true, want false (no scales present)")
	}
	if !moe.UseFusedGateUp {
		t.Error("UseFusedGateUp = false, want true")
	}
	if moe.GateUpWeight == nil || moe.DownWeight == nil {
		t.Error("dense fused weights not set")
	}
	if moe.GateUpWeightQ != nil || moe.DownWeightQ != nil {
		t.Error("quantized weights set on a dense block")
	}
}

// TestRouterForwardMatchesLegacy verifies the optimized Router.Forward —
// which takes the top-k of the raw logits and softmaxes only the selected
// values — produces the same indices and (within tolerance) the same
// normalized scores as the legacy path that softmaxes over every expert
// first, gathers the top-k probabilities, then renormalizes.
func TestRouterForwardMatchesLegacy(t *testing.T) {
	useMLXTestThread(t)

	cfg := &TextConfig{
		HiddenSize:  8,
		NumExperts:  4,
		TopKExperts: 2,
		RMSNormEps:  1e-6,
		RouterScale: 0.5,
	}

	// Distinct per-expert weight rows so top-k has a well-defined ordering
	// (tied scores would let argpartition pick either tied expert and make
	// the index comparison below flaky).
	projWeight := mlx.FromValues([]float32{
		0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, // expert 0
		0.30, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24, 0.23, // expert 1
		-0.05, -0.06, -0.07, -0.08, -0.09, -0.10, -0.11, -0.12, // expert 2
		0.50, 0.48, 0.46, 0.44, 0.42, 0.40, 0.38, 0.36, // expert 3
	}, int(cfg.NumExperts), int(cfg.HiddenSize))

	scale := mlx.FromValues([]float32{
		1.0, 0.9, 1.1, 1.0, 1.2, 0.8, 1.0, 1.05,
	}, int(cfg.HiddenSize))

	r := &Router{
		Proj:  linearFromWeight(projWeight),
		Scale: scale,
	}
	r.NormScale = mlx.MulScalar(scale, cfg.RouterScale)
	mlx.Eval(r.NormScale)

	// Varied x so different positions potentially hit different top-k.
	x := mlx.FromValues([]float32{
		0.2, -0.1, 0.3, 0.0, 0.4, -0.2, 0.1, 0.05,
		-0.3, 0.2, -0.1, 0.4, -0.05, 0.3, 0.0, 0.2,
		0.5, 0.4, -0.2, 0.1, -0.3, 0.0, 0.3, -0.1,
	}, 1, 3, int(cfg.HiddenSize))

	gotScores, gotInds, _ := r.Forward(x, cfg)
	wantScores, wantInds := legacyRouterForward(r, x, cfg)
	gotInds = gotInds.AsType(mlx.DTypeInt32)
	wantInds = wantInds.AsType(mlx.DTypeInt32)
	mlx.Eval(gotScores, gotInds, wantScores, wantInds)

	if got, want := gotInds.Ints(), wantInds.Ints(); !intSlicesEqual(got, want) {
		t.Fatalf("indices mismatch:\n  got  %v\n  want %v", got, want)
	}
	if got, want := gotScores.Floats(), wantScores.Floats(); !floatSlicesClose(got, want, 1e-5) {
		t.Fatalf("scores mismatch:\n  got  %v\n  want %v", got, want)
	}
}

// legacyRouterForward implements the pre-optimization router: full softmax
// over every expert, gather the top-k probabilities, then renormalize them
// to sum to 1. Algebraically identical to the fused form in Router.Forward.
func legacyRouterForward(r *Router, x *mlx.Array, cfg *TextConfig) (*mlx.Array, *mlx.Array) {
	dims := x.Dims()
	BL := int32(dims[0]) * int32(dims[1])

	xFlat := mlx.Reshape(x, BL, cfg.HiddenSize)
	normed := mlx.RMSNormFn(xFlat, nil, cfg.RMSNormEps)
	normed = mlx.MulScalar(normed, cfg.RouterScale)
	normed = mlx.Mul(normed, r.Scale)

	expertScores := r.Proj.Forward(normed)
	probs := mlx.SoftmaxAxis(expertScores, -1, true)

	neg := mlx.Neg(expertScores)
	inds := mlx.Argpartition(neg, int(cfg.TopKExperts)-1, -1)
	inds = mlx.SliceStartStop(inds,
		[]int32{0, 0},
		[]int32{BL, cfg.TopKExperts},
	)

	scores := mlx.TakeAlongAxis(probs, inds, -1)
	sumScores := mlx.Sum(scores, -1, true)
	scores = mlx.Div(scores, sumScores)
	return scores, inds
}

func intSlicesEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func floatSlicesClose(a, b []float32, tol float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		d := a[i] - b[i]
		if d < 0 {
			d = -d
		}
		if d > tol {
			return false
		}
	}
	return true
}

// linearFromWeight creates a simple nn.LinearLayer from a weight tensor (no bias).
func linearFromWeight(w *mlx.Array) *simpleLinear {
	return &simpleLinear{weight: w}
}

type simpleLinear struct {
	weight *mlx.Array
}

func (l *simpleLinear) Forward(x *mlx.Array) *mlx.Array {
	return x.Matmul(mlx.Transpose(l.weight, 1, 0))
}

func (l *simpleLinear) OutputDim() int32 {
	return int32(l.weight.Dims()[0])
}
