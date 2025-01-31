package nn

import (
	"fmt"

	"github.com/ollama/ollama/ml"
)

// Attention implements scaled dot-product attention for transformer models:
// Attention(Q, K, V) = softmax(QK^T/√d_k)V
//
// Parameters:
//   - ctx: Context for tensor operations
//   - query: Query tensor (Q) with shape [heads, seq_len_q, d_k]
//   - key: Key tensor (K) with shape [kv_heads, seq_len_k, d_k]
//   - value: Value tensor (V) with shape [kv_heads, d_v, seq_len_k]
//   - mask: Optional attention mask that is added to the attention score. If
//     provided, should broadcast to [heads, seq_len_q, seq_len_k]
//   - scale: Scaling factor, typically 1/√d_k where d_k is the key dimension
//
// Returns:
//
//	Attention output with shape [d_v, heads, seq_len_q]
func Attention(ctx ml.Context, query, key, value, mask ml.Tensor, scale float64) ml.Tensor {
	// slog.Info("XXX", "q", query)
	// slog.Info("XXX", "k", key)
	// slog.Info("XXX", "v", value)
	// slog.Info("XXX", "m", mask)
	// panic("XXX")
	if query.Dim(2) != key.Dim(2) {
		panic(fmt.Errorf("d_k in attention operation does not match between query(%v) and key(%v)", query.Dim(2), key.Dim(2)))
	}

	if mask != nil && query.Dim(1) != mask.Dim(0) {
		panic(fmt.Errorf("seq_len_q in attention operation does not match between query(%v) and mask(%v)", query.Dim(1), mask.Dim(0)))
	}

	if key.Dim(1) != value.Dim(2) {
		panic(fmt.Errorf("seq_len_k in attention operation does not match between key(%v) and value(%v)", key.Dim(1), value.Dim(2)))
	}

	if mask != nil && key.Dim(1) != mask.Dim(1) {
		panic(fmt.Errorf("seq_len_k in attention operation does not match between key(%v) and mask(%v)", key.Dim(1), mask.Dim(1)))
	}

	if key.Dim(0) != value.Dim(0) {
		panic(fmt.Errorf("kv_heads in attention operation does not match between key(%v) and value(%v)", key.Dim(0), value.Dim(0)))
	}

	if sdpa, ok := query.(ml.ScaledDotProductAttention); ok {
		return sdpa.ScaledDotProductAttention(ctx, key, value, mask, scale)
	} else {
		kq := key.MulmatFullPrec(ctx, query)

		kq = kq.Scale(ctx, scale)
		if mask != nil {
			kq = kq.Add(ctx, mask)
		}
		kq = kq.Softmax(ctx)

		kqv := value.Mulmat(ctx, kq)
		return kqv.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	}
}
