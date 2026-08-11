package mlx

import (
	"math"
	"testing"
)

func TestFastBF16GeGLUProjection(t *testing.T) {
	withMLXThread(t, func() {
		if !CUDAIsAvailable() {
			t.Skip("CUDA is not available")
		}

		const inputDim = 256
		const outputDim = 64

		xData := make([]float32, inputDim)
		gateData := make([]float32, inputDim*outputDim)
		upData := make([]float32, inputDim*outputDim)
		for i := range xData {
			xData[i] = float32(math.Sin(float64(i)*0.11) * 0.25)
		}
		for i := range gateData {
			gateData[i] = float32(math.Sin(float64(i)*0.017) * 0.125)
			upData[i] = float32(math.Cos(float64(i)*0.013) * 0.125)
		}

		x := FromValues(xData, 1, 1, inputDim).AsType(DTypeBFloat16)
		gateWeight := FromValues(gateData, outputDim, inputDim).AsType(DTypeBFloat16)
		upWeight := FromValues(upData, outputDim, inputDim).AsType(DTypeBFloat16)

		got, ok := FastBF16GeGLUProjection(x, gateWeight, upWeight)
		if !ok {
			t.Fatal("FastBF16GeGLUProjection did not accept a supported CUDA shape")
		}
		want := GeGLU(
			x.Matmul(gateWeight.Transpose(1, 0)),
			x.Matmul(upWeight.Transpose(1, 0)),
		)
		gotFloat := got.AsType(DTypeFloat32)
		wantFloat := want.AsType(DTypeFloat32)
		Eval(gotFloat, wantFloat)

		gotValues := gotFloat.Floats()
		wantValues := wantFloat.Floats()
		for i := range gotValues {
			diff := float32(math.Abs(float64(gotValues[i] - wantValues[i])))
			if diff > 0.0078125 {
				t.Fatalf("output[%d] = %v, want %v (diff %v)", i, gotValues[i], wantValues[i], diff)
			}
		}
	})
}
