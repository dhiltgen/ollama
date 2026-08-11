package mlxrunner

import "testing"

func TestMaterializationBatchArrayLimit(t *testing.T) {
	for _, tc := range []struct {
		goos, goarch string
		want         int
	}{
		{goos: "linux", goarch: "arm64", want: 0},
		{goos: "linux", goarch: "amd64", want: 0},
		{goos: "windows", goarch: "arm64", want: 1},
		{goos: "windows", goarch: "amd64", want: 0},
	} {
		if got := materializationBatchArrayLimit(tc.goos, tc.goarch); got != tc.want {
			t.Errorf("materializationBatchArrayLimit(%q, %q) = %d, want %d", tc.goos, tc.goarch, got, tc.want)
		}
	}
}
