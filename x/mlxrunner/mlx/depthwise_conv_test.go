package mlx

import (
	"fmt"
	"testing"
)

func TestDepthwiseConvSiLUMatchesGraph(t *testing.T) {
	skipIfNoMLX(t)
	var testErr error
	withMLXThread(t, func() {
		testErr = testDepthwiseConvSiLUMatchesGraph()
	})
	if testErr != nil {
		t.Fatal(testErr)
	}
}

func testDepthwiseConvSiLUMatchesGraph() error {
	for _, dtype := range []DType{DTypeBFloat16, DTypeFloat32} {
		for _, shape := range []struct{ B, T, C, K int }{
			{1, 1, 64, 4},
			{1, 4, 64, 4},
			{1, 11, 96, 4},
			{1, 64, 64, 4},
			{1, 333, 64, 4},
			{3, 7, 64, 4},
			{2, 5, 32, 2},
			{1, 1, 9728, 4},
		} {
			name := fmt.Sprintf("%v_b%d_t%d_c%d_k%d", dtype, shape.B, shape.T, shape.C, shape.K)
			x := patternArray(dtype, []int{shape.B, shape.T + shape.K - 1, shape.C}, 0.02, 0.004, 41, 263)
			w := patternArray(dtype, []int{shape.C, shape.K}, 0.1, 0.01, 7, 53)

			ref := SiLU(Conv1d(x, Reshape(w, int32(shape.C), int32(shape.K), 1), nil, 1, 0, 1, int32(shape.C)))
			y := DepthwiseConvSiLU(x, w, nil, shape.T)
			if err := requireExact("y", y, ref); err != nil {
				return fmt.Errorf("%s: %w", name, err)
			}
		}
	}

	x := patternArray(DTypeFloat32, []int{1, 4, 9728}, 0.02, 0.004, 41, 263)
	w := patternArray(DTypeBFloat16, []int{9728, 4}, 0.1, 0.01, 7, 53)
	ref := SiLU(Conv1d(x, Reshape(w, 9728, 4, 1), nil, 1, 0, 1, 9728))
	y := DepthwiseConvSiLU(x, w, nil, 1)
	if err := requireExact("F32xBF16_b1_t1_c9728_k4", y, ref); err != nil {
		return err
	}

	bias := patternArray(DTypeFloat32, []int{9728}, 0.03, 0.002, 11, 97)
	ref = SiLU(Conv1d(x, Reshape(w, 9728, 4, 1), bias, 1, 0, 1, 9728))
	y = DepthwiseConvSiLU(x, w, bias, 1)
	if err := requireExact("F32xBF16_biasF32_b1_t1_c9728_k4", y, ref); err != nil {
		return err
	}

	ref = ref.AsType(DTypeBFloat16).AsType(DTypeFloat32)
	y = DepthwiseConvSiLURoundTrip(x, w, bias, 1, DTypeBFloat16)
	if err := requireExact("F32xBF16_biasF32_roundBF16_b1_t1_c9728_k4", y, ref); err != nil {
		return err
	}
	return nil
}
