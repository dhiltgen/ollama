package ggml

import (
	"fmt"
	"log/slog"
	"testing"
)

func TestPermute3d(t *testing.T) {
	be := newTestBackend(100)
	ctx := newTestContext(be, 100)
	shape := []int{2, 3, 5}
	data := make([]float32, shape[0]*shape[1]*shape[2])
	for i := range data {
		data[i] = float32(i + 1)
	}
	x, err := ctx.FromFloatSlice(data, shape...)
	if err != nil {
		t.Fatal(err)
	}
	slog.Info("Initial data", "tensor", x)

	type testCase struct {
		shape []int
	}
	testCases := []testCase{
		{shape: []int{0, 1, 2, 3}},
		{shape: []int{0, 2, 1, 3}},
		{shape: []int{1, 0, 2, 3}},
		{shape: []int{1, 2, 0, 3}},
		{shape: []int{2, 0, 1, 3}},
		{shape: []int{2, 1, 0, 3}},
	}
	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%v", tc), func(t *testing.T) {
			t2 := x.Permute(ctx, tc.shape...)
			slog.Info("After permute", "request", tc.shape, "result", t2)
			res := t2.Shape()
			expected := [3]int{
				shape[tc.shape[0]],
				shape[tc.shape[1]],
				shape[tc.shape[2]],
			}
			if ([3]int)(res) != expected {
				t.Fatalf("reshape %v expected %v but got %v", tc.shape, expected, res)
			}
		})
	}
}

func TestPermute4d(t *testing.T) {
	be := newTestBackend(1000)
	ctx := newTestContext(be, 1000)
	shape := []int{2, 3, 5, 7}
	data := make([]float32, shape[0]*shape[1]*shape[2]*shape[3])
	for i := range data {
		data[i] = float32(i + 1)
	}
	x, err := ctx.FromFloatSlice(data, shape...)
	if err != nil {
		t.Fatal(err)
	}
	slog.Info("Initial data", "tensor", x)

	type testCase struct {
		shape []int
	}
	testCases := []testCase{
		{shape: []int{0, 1, 2, 3}},
		{shape: []int{0, 1, 3, 2}},
		{shape: []int{0, 2, 1, 3}},
		{shape: []int{0, 2, 3, 1}},
		{shape: []int{0, 3, 1, 2}},
		{shape: []int{0, 3, 2, 1}},
		{shape: []int{1, 0, 2, 3}},
		{shape: []int{1, 0, 3, 2}},
		{shape: []int{1, 2, 0, 3}},
		{shape: []int{1, 2, 3, 0}},
		{shape: []int{1, 3, 0, 2}},
		{shape: []int{1, 3, 2, 0}},
		{shape: []int{2, 0, 1, 3}},
		{shape: []int{2, 0, 3, 1}},
		{shape: []int{2, 1, 0, 3}},
		{shape: []int{2, 1, 3, 0}},
		{shape: []int{2, 3, 0, 1}},
		{shape: []int{2, 3, 1, 0}},
		{shape: []int{3, 0, 1, 2}},
		{shape: []int{3, 0, 2, 1}},
		{shape: []int{3, 1, 0, 2}},
		{shape: []int{3, 1, 2, 0}},
		{shape: []int{3, 2, 0, 1}},
		{shape: []int{3, 2, 1, 0}},
	}
	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%v", tc), func(t *testing.T) {
			t2 := x.Permute(ctx, tc.shape...)
			slog.Info("After permute", "request", tc.shape, "result", t2)
			res := t2.Shape()
			expected := [4]int{
				shape[tc.shape[0]],
				shape[tc.shape[1]],
				shape[tc.shape[2]],
				shape[tc.shape[3]],
			}
			if ([4]int)(res) != expected {
				t.Fatalf("reshape %v expected %v but got %v", tc.shape, expected, res)
			}
		})
	}
}
