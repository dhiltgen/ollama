package qwen4_exp

import (
	"math"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

func TestLinearAttentionOutputGateCompiledMatchesEager(t *testing.T) {
	for _, tc := range []struct {
		name   string
		dtype  mlx.DType
		tokens int
	}{
		{"f32 one token", mlx.DTypeFloat32, 1},
		{"bf16 one token", mlx.DTypeBFloat16, 1},
		{"bf16 128 tokens", mlx.DTypeBFloat16, 128},
	} {
		mlxtest.RunSubtest(t, tc.name, func(t *mlxtest.T) {
			mlx.EnableCompile()
			defer mlx.DisableCompile()
			mlx.Scoped(func() {
				values := make([]float32, tc.tokens*8)
				for i := range values {
					values[i] = float32(math.Sin(float64(i) * 0.17))
				}
				out := mlx.FromValues(values, 1, tc.tokens, 8).AsType(tc.dtype)
				gate := mlx.FromValues(values, 1, tc.tokens, 8).AsType(mlx.DTypeBFloat16)

				got := linearAttentionOutputGate(out, gate).AsType(mlx.DTypeFloat32)
				want := mlx.Mul(out.AsType(mlx.DTypeFloat32), mlx.Sigmoid(gate.AsType(mlx.DTypeFloat32)))
				want = want.AsType(tc.dtype).AsType(mlx.DTypeFloat32)
				mlx.Eval(got, want)
				assertClose(t, "output gate", got.Floats(), want.Floats())
			})
		})
	}
}
