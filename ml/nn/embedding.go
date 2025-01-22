package nn

import (
	"github.com/ollama/ollama/ml"
)

type Embedding struct {
	Weight ml.Tensor `ggml:"weight"`
}

func (m *Embedding) Forward(ctx ml.Context, hiddenState ml.Tensor) ml.Tensor {
	r := m.Weight.Rows(ctx, hiddenState)
	// slog.Info("XXX Embedding.Forward", "rows", r, "hiddenState", hiddenState)
	return r
}
