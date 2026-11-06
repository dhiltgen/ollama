package qwen4_exp

import (
	"math"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/nn"
)

func TestHyperConnectionMatchesReferenceFormula(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.EnableCompile()
		defer mlx.DisableCompile()

		cfg := &Config{HCCount: 2, HiddenSize: 2, HCLowRank: 2, RMSNormEps: 1e-6}
		normWeight := []float32{1.1, 0.9, 1.2, 0.8}
		downWeight := [][]float32{{0.2, -0.3, 0.4, 0.1}, {-0.1, 0.5, 0.2, -0.4}}
		upWeight := [][]float32{{0.3, -0.2}, {0.1, 0.4}, {-0.5, 0.2}, {0.25, 0.15}}
		injectWeight := [][]float32{{0.2, -0.1, 0.3, 0.4}, {-0.3, 0.2, 0.1, 0.5}}
		input := []float32{1, 2, 3, 4}

		downProj := nn.NewLinear(matrix(downWeight), nil)
		injectProj := nn.NewLinear(matrix(injectWeight), nil)
		h := &hyperConnection{
			Norm:        &streamRMSNorm{Weight: mlx.FromValues(normWeight, 2, 2)},
			InputMixUp:  nn.NewLinear(matrix(upWeight), nil),
			PackedInput: packHyperConnectionInput(downProj, injectProj),
			MixDownDim:  downProj.OutputDim(),
		}
		residual := mlx.FromValues(input, 1, 1, 4)
		branch, state := h.Prepare(residual, cfg)
		got := h.Inject(state, branch, cfg).AsType(mlx.DTypeFloat32)
		reduced := h.Reduce(residual, cfg).AsType(mlx.DTypeFloat32)
		mlx.Eval(got, reduced)

		normed := append([]float32(nil), input...)
		for stream := range int(cfg.HCCount) {
			start := stream * int(cfg.HiddenSize)
			var square float64
			for i := range int(cfg.HiddenSize) {
				x := float64(input[start+i])
				square += x * x
			}
			invRMS := 1 / math.Sqrt(square/float64(cfg.HiddenSize)+float64(cfg.RMSNormEps))
			for i := range int(cfg.HiddenSize) {
				normed[start+i] = float32(float64(input[start+i]) * invRMS * float64(normWeight[start+i]))
			}
		}

		down := matvec(downWeight, normed)
		for i := range down {
			down[i] /= float32(cfg.HCCount)
			down[i] *= 1 / (1 + float32(math.Exp(float64(-down[i]))))
		}
		mix := matvec(upWeight, down)
		wantBranch := make([]float32, cfg.HiddenSize)
		for stream := range int(cfg.HCCount) {
			for i := range int(cfg.HiddenSize) {
				j := stream*int(cfg.HiddenSize) + i
				gate := 1 / (1 + float32(math.Exp(float64(-mix[j]))))
				wantBranch[i] += gate * normed[j] / float32(cfg.HCCount)
			}
		}

		injection := matvec(injectWeight, normed)
		want := append([]float32(nil), input...)
		for stream := range int(cfg.HCCount) {
			weight := 2 / (1 + float32(math.Exp(float64(-injection[stream]/float32(cfg.HCCount)))))
			for i := range int(cfg.HiddenSize) {
				want[stream*int(cfg.HiddenSize)+i] += weight * wantBranch[i]
			}
		}

		assertClose(t, "mixed branch", reduced.Floats(), wantBranch)
		assertClose(t, "injected streams", got.Floats(), want)
	})
}

func TestPackHyperConnectionInputRejectsGlobalScales(t *testing.T) {
	for _, tc := range []struct {
		name         string
		down, inject bool
		wantPacked   bool
	}{
		{"unscaled", false, false, true},
		{"input mix", true, false, false},
		{"block inject", false, true, false},
		{"both", true, true, false},
	} {
		mlxtest.RunSubtest(t, tc.name, func(t *mlxtest.T) {
			projection := func(withGlobalScale bool) *nn.QuantizedLinear {
				values := make([]float32, 2*16)
				for i := range values {
					values[i] = float32(i%7-3) * 0.1
				}
				weight, scales, _ := mlx.Quantize(mlx.FromValues(values, 2, 16), 16, 4, "nvfp4")
				p := &nn.QuantizedLinear{
					Weight:    weight,
					Scales:    scales,
					GroupSize: 16,
					Bits:      4,
					Mode:      "nvfp4",
				}
				if withGlobalScale {
					p.GlobalScale = mlx.FromValue(float32(1))
				}
				return p
			}

			got := packHyperConnectionInput(projection(tc.down), projection(tc.inject))
			if !tc.wantPacked {
				if got != nil {
					t.Fatal("packed projections with a global scale, want separate fallback")
				}
				return
			}
			if got == nil {
				t.Fatal("unscaled projections did not pack")
			}
			if got.OutputDim() != 4 {
				t.Fatalf("packed output dimension = %d, want 4", got.OutputDim())
			}
		})
	}
}

func matrix(rows [][]float32) *mlx.Array {
	values := make([]float32, 0, len(rows)*len(rows[0]))
	for _, row := range rows {
		values = append(values, row...)
	}
	return mlx.FromValues(values, len(rows), len(rows[0]))
}

func matvec(weight [][]float32, input []float32) []float32 {
	output := make([]float32, len(weight))
	for i, row := range weight {
		for j, value := range row {
			output[i] += value * input[j]
		}
	}
	return output
}

func assertClose(t *mlxtest.T, name string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length = %d, want %d", name, len(got), len(want))
	}
	if len(want) == 0 {
		t.Fatalf("%s has no values to compare", name)
	}
	for i := range want {
		gotValue, wantValue := float64(got[i]), float64(want[i])
		if math.IsNaN(gotValue) || math.IsInf(gotValue, 0) || math.IsNaN(wantValue) || math.IsInf(wantValue, 0) {
			t.Fatalf("%s[%d] = %v, want finite %v", name, i, got[i], want[i])
		}
		if math.Abs(gotValue-wantValue) > 1e-5 {
			t.Fatalf("%s[%d] = %v, want %v", name, i, got[i], want[i])
		}
	}
}
