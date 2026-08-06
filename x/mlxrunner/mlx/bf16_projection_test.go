package mlx

import (
	"math"
	"testing"
)

func TestFastBF16Projection(t *testing.T) {
	withMLXThread(t, func() {
		if !CUDAIsAvailable() {
			t.Skip("CUDA is not available")
		}

		const inputDim = 256
		const outputDim = 64

		xData := make([]float32, inputDim)
		weightData := make([]float32, outputDim*inputDim)
		for i := range xData {
			xData[i] = float32(math.Sin(float64(i)*0.11) * 0.25)
		}
		for i := range weightData {
			weightData[i] = float32(math.Cos(float64(i)*0.013) * 0.125)
		}

		x := FromValues(xData, 1, 1, inputDim).AsType(DTypeBFloat16)
		weight := FromValues(weightData, outputDim, inputDim).AsType(DTypeBFloat16)

		got, ok := FastBF16Projection(x, weight)
		if !ok {
			t.Fatal("FastBF16Projection did not accept a supported CUDA shape")
		}
		want := x.Matmul(weight.Transpose(1, 0))

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
