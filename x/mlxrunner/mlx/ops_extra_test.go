package mlx

import (
	"math"
	"testing"
)

// fp4Values decodes an fp4 (E2M1) code to its value.
var fp4Values = [16]float32{0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6}

func TestDequantizeGlobalScale(t *testing.T) {
	skipIfNoMLX(t)
	withMLXThread(t, func() {
		testDequantizeGlobalScale(t)
	})
}

// The quantized payload is built directly, the way an nvfp4 checkpoint ships
// it: packed fp4 codes, e4m3 group-scale bytes, and a separate global scale.
// Only the dequantize consumer path runs, so expectations are exact.
func testDequantizeGlobalScale(t *testing.T) {
	const rows, cols, group = 4, 64, 16
	// Every group cycles through all 16 codes; group g of row r has scale
	// 2^((r+g)%4-1), a power of two so every expected product is exact.
	scaleOf := func(r, g int) float32 {
		return float32(math.Ldexp(1, (r+g)%4-1))
	}

	packed := make([]uint32, rows*cols/8)
	for i := range packed {
		for j := range 8 {
			packed[i] |= uint32((i*8+j)%16) << (4 * j)
		}
	}
	scaleBits := make([]uint8, rows*(cols/group))
	for i := range scaleBits {
		exp := (i/(cols/group)+i%(cols/group))%4 - 1
		scaleBits[i] = uint8((exp + 7) << 3)
	}
	wq := FromValues(packed, rows, cols/8)
	scales := FromValues(scaleBits, rows, cols/group)

	check := func(name string, got *Array, gs func(r int) float32) {
		t.Helper()
		g32 := got.AsType(DTypeFloat32)
		Eval(g32)
		values := g32.Floats()
		if len(values) != rows*cols {
			t.Errorf("%s: length = %d, want %d", name, len(values), rows*cols)
			return
		}
		for i, v := range values {
			r, c := i/cols, i%cols
			if want := fp4Values[c%16] * scaleOf(r, c/group) * gs(r); v != want {
				t.Errorf("%s[%d] = %v, want %v", name, i, v, want)
				return
			}
		}
	}

	base := Dequantize(wq, scales, nil, group, 4, "nvfp4", nil)
	check("no scale", base, func(int) float32 { return 1 })

	perRow := []float32{0.5, 1, 2, 4}
	cases := []struct {
		name  string
		scale *Array
		gs    func(r int) float32
	}{
		{"scalar", FromValues([]float32{2}, 1), func(int) float32 { return 2 }},
		{"perRow", FromValues(perRow, rows), func(r int) float32 { return perRow[r] }},
	}
	for _, tc := range cases {
		got := Dequantize(wq, scales, nil, group, 4, "nvfp4", tc.scale)
		if got.DType() != base.DType() {
			t.Errorf("%s: dtype = %v, want %v", tc.name, got.DType(), base.DType())
		}
		check(tc.name, got, tc.gs)
	}
}

func TestFastQuantizedMatmulDisabledOnMetal(t *testing.T) {
	withMLXThread(t, func() {
		if !MetalIsAvailable() {
			t.Skip("Metal is not available")
		}

		for _, mode := range []string{"nvfp4", "mxfp8"} {
			if supportsFastQuantizedMatmulMode(mode) {
				t.Fatalf("supportsFastQuantizedMatmulMode(%q) = true on Metal", mode)
			}
		}
	})
}

func TestFastQuantizedMatmulNVFP4GlobalScale(t *testing.T) {
	withMLXThread(t, func() {
		if !CUDAIsAvailable() || !supportsFastQuantizedMatmulMode("nvfp4") {
			t.Skip("native NVFP4 CUDA matmul is not available")
		}

		weightValues := make([]float32, 16*32)
		for i := range weightValues {
			weightValues[i] = float32((i%9)-4) / 4
		}
		weight := FromValues(weightValues, 16, 32).AsType(DTypeBFloat16)
		quantizedWeight, scales, biases := Quantize(weight, 16, 4, "nvfp4")
		if biases != nil {
			t.Fatalf("nvfp4 biases = %v, want nil", biases)
		}
		Eval(quantizedWeight, scales)

		// This real checkpoint scale does not survive an amax * 2688 / 2688
		// float32 round trip, so it exercises the direct output-scale contract.
		directScale := NewScalarArray(0.00010390509123681113)
		nativeScale := NativeQuantizedGlobalScale(directScale, "nvfp4")
		if nativeScale == nil {
			t.Fatal("scalar NVFP4 scale did not produce a native scale")
		}
		if got := NativeQuantizedGlobalScale(FromValues([]float32{1, 2}, 2), "nvfp4"); got != nil {
			t.Fatal("per-row NVFP4 scale must retain the generic broadcast path")
		}
		for _, rows := range []int{1, 2, 8, 128} {
			inputValues := make([]float32, rows*32)
			for i := range inputValues {
				inputValues[i] = float32((i%7)-3) / 3
			}
			input := FromValues(inputValues, 1, rows, 32).AsType(DTypeBFloat16)
			Eval(input)

			got := FastQuantizedMatmul(
				input,
				quantizedWeight,
				scales,
				nil,
				directScale,
				nativeScale,
				true,
				16,
				4,
				"nvfp4",
			).AsType(DTypeFloat32)
			want := Mul(
				QuantizedMatmul(
					input,
					quantizedWeight,
					scales,
					nil,
					true,
					16,
					4,
					"nvfp4",
				),
				directScale,
			).AsType(DTypeBFloat16).AsType(DTypeFloat32)
			Eval(got, want)

			gotValues := got.Floats()
			wantValues := want.Floats()
			for i := range gotValues {
				if rows == 1 && gotValues[i] != wantValues[i] {
					t.Fatalf("rows=%d output[%d] = %.6f, want bit-exact %.6f", rows, i, gotValues[i], wantValues[i])
				}
				if delta := gotValues[i] - wantValues[i]; delta < -0.02 || delta > 0.02 {
					t.Fatalf("rows=%d output[%d] = %.6f, want %.6f (delta %.6f)", rows, i, gotValues[i], wantValues[i], delta)
				}
			}
		}
	})
}

func TestFastSDPAWideTwoPass(t *testing.T) {
	withMLXThread(t, func() {
		if !CUDAIsAvailable() {
			t.Skip("CUDA is not available")
		}

		for _, tc := range []struct {
			headDim   int
			qHeads    int
			keyLength int
		}{
			{headDim: 256, qHeads: 2, keyLength: 1024},
			{headDim: 512, qHeads: 8, keyLength: 2051},
			{headDim: 512, qHeads: 16, keyLength: 2051},
		} {
			qValues := make([]float32, tc.qHeads*tc.headDim)
			kValues := make([]float32, tc.keyLength*tc.headDim)
			vValues := make([]float32, tc.keyLength*tc.headDim)
			for i := range qValues {
				qValues[i] = float32((i%17)-8) / 32
			}
			for i := range kValues {
				kValues[i] = float32((i%23)-11) / 32
				vValues[i] = float32((i%29)-14) / 32
			}

			q := FromValues(qValues, 1, tc.qHeads, 1, tc.headDim).AsType(DTypeBFloat16)
			k := FromValues(kValues, 1, 1, tc.keyLength, tc.headDim).AsType(DTypeBFloat16)
			v := FromValues(vValues, 1, 1, tc.keyLength, tc.headDim).AsType(DTypeBFloat16)
			scale := float32(1 / math.Sqrt(float64(tc.headDim)))
			var mask *Array
			if tc.headDim == 512 {
				maskValues := make([]float32, tc.keyLength)
				for i := range maskValues {
					maskValues[i] = float32((i%7)-3) / 64
				}
				mask = FromValues(maskValues, 1, 1, 1, tc.keyLength).
					AsType(DTypeBFloat16)
			}

			got := FastScaledDotProductAttention(q, k, v, scale, "", mask).
				AsType(DTypeFloat32)
			scores := MulScalar(q.Matmul(k.Transpose(0, 1, 3, 2)), scale)
			if mask != nil {
				scores = Add(scores, mask)
			}
			want := SoftmaxAxis(scores, -1, true).Matmul(v).AsType(DTypeFloat32)
			Eval(got, want)

			gotValues := got.Floats()
			wantValues := want.Floats()
			for i := range gotValues {
				if delta := float32(math.Abs(float64(gotValues[i] - wantValues[i]))); delta > 0.01 {
					t.Fatalf("headDim=%d qHeads=%d keyLength=%d output[%d] = %.6f, want %.6f (delta %.6f)", tc.headDim, tc.qHeads, tc.keyLength, i, gotValues[i], wantValues[i], delta)
				}
			}
		}
	})
}
