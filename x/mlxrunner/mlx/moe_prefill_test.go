package mlx

import "testing"

func TestFastSortedMoEDispatchDynamicHiddenSize(t *testing.T) {
	withMLXThread(t, func() {
		if !CUDAIsAvailable() {
			t.Skip("CUDA is not available")
		}

		const rows, topK, hiddenSize = 64, 8, 4096
		assignments := rows * topK

		xValues := make([]float32, rows*hiddenSize)
		for i := range xValues {
			xValues[i] = float32((i%97)-48) * 0.00390625
		}
		x := FromValues(xValues, rows, 1, 1, hiddenSize).
			AsType(DTypeBFloat16)

		orderValues := make([]uint32, assignments)
		for i := range orderValues {
			orderValues[i] = uint32(assignments - 1 - i)
		}
		order := FromValues(orderValues, assignments)

		want := ExpandDims(
			Take(
				Squeeze(x, 1),
				FloorDivideScalar(order, topK),
				0,
			),
			1,
		)
		got, ok := FastSortedMoEDispatch(x, order)
		if !ok {
			t.Fatal("FastSortedMoEDispatch rejected a supported hidden size")
		}

		want = want.AsType(DTypeFloat32)
		got = got.AsType(DTypeFloat32)
		Eval(want, got)

		gotValues, wantValues := got.Floats(), want.Floats()
		for i := range gotValues {
			if gotValues[i] != wantValues[i] {
				t.Fatalf("output[%d] = %v, want %v", i, gotValues[i], wantValues[i])
			}
		}
	})
}

func TestFastSortedMoEUnsortDynamicHiddenSize(t *testing.T) {
	withMLXThread(t, func() {
		if !CUDAIsAvailable() {
			t.Skip("CUDA is not available")
		}

		const rows, topK, hiddenSize = 64, 8, 4096
		assignments := rows * topK

		xValues := make([]float32, assignments*hiddenSize)
		for i := range xValues {
			xValues[i] = float32((i%101)-50) * 0.00390625
		}
		x := FromValues(xValues, assignments, 1, 1, hiddenSize).
			AsType(DTypeBFloat16)

		invOrderValues := make([]uint32, assignments)
		for i := range invOrderValues {
			invOrderValues[i] = uint32(assignments - 1 - i)
		}
		invOrder := FromValues(invOrderValues, assignments)

		want := Take(
			Squeeze(Squeeze(x, 2), 1),
			invOrder,
			0,
		)
		got, ok := FastSortedMoEUnsort(x, invOrder)
		if !ok {
			t.Fatal("FastSortedMoEUnsort rejected a supported hidden size")
		}

		want = want.AsType(DTypeFloat32)
		got = Reshape(got, int32(assignments), hiddenSize).AsType(DTypeFloat32)
		Eval(want, got)

		gotValues, wantValues := got.Floats(), want.Floats()
		for i := range gotValues {
			if gotValues[i] != wantValues[i] {
				t.Fatalf("output[%d] = %v, want %v", i, gotValues[i], wantValues[i])
			}
		}
	})
}
