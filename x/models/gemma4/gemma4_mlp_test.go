package gemma4

import (
	"math"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

func TestFusedGateUpLinearDenseMatchesSeparate(t *testing.T) {
	skipIfNoMLX(t)

	x := mlx.FromValues([]float32{
		0.1, -0.2, 0.3, -0.4,
		0.5, -0.6, 0.7, -0.8,
		0.9, -1.0, 1.1, -1.2,
		1.3, -1.4, 1.5, -1.6,
		1.7, -1.8, 1.9, -2.0,
		2.1, -2.2, 2.3, -2.4,
	}, 2, 3, 4).AsType(mlx.DTypeBFloat16)
	gate := nn.NewLinear(
		mlx.FromValues([]float32{
			0.1, 0.2, 0.3, 0.4,
			-0.5, 0.6, -0.7, 0.8,
			0.9, -1.0, 1.1, -1.2,
			-1.3, 1.4, -1.5, 1.6,
			1.7, 1.8, -1.9, -2.0,
		}, 5, 4).AsType(mlx.DTypeBFloat16),
		mlx.FromValues([]float32{0.01, -0.02, 0.03, -0.04, 0.05}, 5).AsType(mlx.DTypeBFloat16),
	)
	up := nn.NewLinear(
		mlx.FromValues([]float32{
			-0.2, 0.1, -0.4, 0.3,
			0.6, -0.5, 0.8, -0.7,
			-1.0, 0.9, -1.2, 1.1,
			1.4, -1.3, 1.6, -1.5,
			-1.8, -1.7, 2.0, 1.9,
		}, 5, 4).AsType(mlx.DTypeBFloat16),
		mlx.FromValues([]float32{-0.06, 0.07, -0.08, 0.09, -0.1}, 5).AsType(mlx.DTypeBFloat16),
	)
	mlx.Eval(x, gate.Weight, gate.Bias, up.Weight, up.Bias)

	fused := tryFuseGateUpLinear(gate, up)
	if fused == nil {
		t.Fatal("tryFuseGateUpLinear returned nil for compatible dense linears")
	}

	got := fused.Forward(x)
	wantGate := gate.Forward(x)
	wantUp := up.Forward(x)
	mlx.Eval(got[0], got[1], wantGate, wantUp)

	assertArraysClose(t, got[0].AsType(mlx.DTypeFloat32), wantGate.AsType(mlx.DTypeFloat32), 1e-4)
	assertArraysClose(t, got[1].AsType(mlx.DTypeFloat32), wantUp.AsType(mlx.DTypeFloat32), 1e-4)
}

func TestFusedGateUpLinearQuantizedMatchesSeparate(t *testing.T) {
	skipIfNoMLX(t)

	inputVals := make([]float32, 2*32)
	for i := range inputVals {
		inputVals[i] = float32((i%9)-4) / 8
	}
	gateVals := make([]float32, 3*32)
	upVals := make([]float32, 3*32)
	for i := range gateVals {
		gateVals[i] = float32((i%13)-6) / 11
		upVals[i] = float32((i%17)-8) / 13
	}

	x := mlx.FromValues(inputVals, 2, 32).AsType(mlx.DTypeBFloat16)
	gateWeight := mlx.FromValues(gateVals, 3, 32).AsType(mlx.DTypeBFloat16)
	upWeight := mlx.FromValues(upVals, 3, 32).AsType(mlx.DTypeBFloat16)
	gateBias := mlx.FromValues([]float32{0.1, -0.2, 0.3}, 3).AsType(mlx.DTypeBFloat16)
	upBias := mlx.FromValues([]float32{-0.4, 0.5, -0.6}, 3).AsType(mlx.DTypeBFloat16)
	mlx.Eval(x, gateWeight, upWeight, gateBias, upBias)

	gate := nn.NewQuantizedLinear(gateWeight, gateBias, 32, 4, "mxfp4")
	up := nn.NewQuantizedLinear(upWeight, upBias, 32, 4, "mxfp4")
	gate.GlobalScale = mlx.FromValues([]float32{0.75}, 1)
	up.GlobalScale = mlx.FromValues([]float32{1.25}, 1)
	mlx.Eval(gate.GlobalScale, up.GlobalScale)

	fused := tryFuseGateUpLinear(gate, up)
	if fused == nil {
		t.Fatal("tryFuseGateUpLinear returned nil for compatible quantized linears")
	}

	got := fused.Forward(x)
	wantGate := gate.Forward(x)
	wantUp := up.Forward(x)
	mlx.Eval(got[0], got[1], wantGate, wantUp)

	assertArraysClose(t, got[0].AsType(mlx.DTypeFloat32), wantGate.AsType(mlx.DTypeFloat32), 1e-3)
	assertArraysClose(t, got[1].AsType(mlx.DTypeFloat32), wantUp.AsType(mlx.DTypeFloat32), 1e-3)
}

func assertArraysClose(t *testing.T, got, want *mlx.Array, tol float32) {
	t.Helper()
	mlx.Eval(got, want)
	gotVals := got.Floats()
	wantVals := want.Floats()
	if len(gotVals) != len(wantVals) {
		t.Fatalf("array length = %d, want %d", len(gotVals), len(wantVals))
	}
	for i := range gotVals {
		if float32(math.Abs(float64(gotVals[i]-wantVals[i]))) > tol {
			t.Fatalf("value[%d] = %.6f, want %.6f", i, gotVals[i], wantVals[i])
		}
	}
}
