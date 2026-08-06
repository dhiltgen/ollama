package mlx

import "testing"

func TestHasCUDAMemoryHeadroom(t *testing.T) {
	const gib = 1 << 30

	tests := []struct {
		name                      string
		total, free, active, peak int
		reserve                   int
		want                      bool
	}{
		{
			name:    "ample local memory",
			total:   32 * gib,
			free:    24 * gib,
			active:  8 * gib,
			peak:    10 * gib,
			reserve: 2 * gib,
			want:    true,
		},
		{
			name:    "async pool reservation is reusable",
			total:   32 * gib,
			free:    64 << 20,
			active:  19 * gib,
			peak:    27 * gib,
			reserve: 2 * gib,
			want:    true,
		},
		{
			name:    "driver free memory provides direct headroom",
			total:   8 * gib,
			free:    4 * gib,
			active:  7 * gib,
			peak:    7 * gib,
			reserve: 2 * gib,
			want:    true,
		},
		{
			name:    "active allocations constrain pooled headroom",
			total:   8 * gib,
			free:    1 * gib,
			active:  7 * gib,
			peak:    5 * gib,
			reserve: 2 * gib,
			want:    false,
		},
		{
			name:    "peak working set constrains pooled headroom",
			total:   8 * gib,
			free:    1 * gib,
			active:  5 * gib,
			peak:    7 * gib,
			reserve: 2 * gib,
			want:    false,
		},
		{
			name:    "exact reserve",
			total:   8 * gib,
			free:    1 * gib,
			active:  6 * gib,
			peak:    6 * gib,
			reserve: 2 * gib,
			want:    true,
		},
		{
			name:    "oversubscribed active allocations",
			total:   8 * gib,
			free:    1 * gib,
			active:  9 * gib,
			peak:    9 * gib,
			reserve: 2 * gib,
			want:    false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := hasCUDAMemoryHeadroom(
				tc.total,
				tc.free,
				tc.active,
				tc.peak,
				tc.reserve,
			)
			if got != tc.want {
				t.Fatalf("hasCUDAMemoryHeadroom() = %v, want %v", got, tc.want)
			}
		})
	}
}
