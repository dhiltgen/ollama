package nn

import (
	"fmt"
	"log/slog"
	"os"

	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
)

// Attention implements scaled dot-product attention for transformer models:
// Attention(Q, K, V) = softmax(QK^T/√d_k)V
//
// Parameters:
//   - ctx: Context for tensor operations
//   - query: Query tensor (Q) with shape [d_k, heads, seq_len_q]
//   - key: Key tensor (K) with shape [d_k, kv_heads, seq_len_k], can be nil to read from cache only
//   - value: Value tensor (V) with shape [d_v, kv_heads, seq_len_k], can be nil to read from cache only
//   - scale: Scaling factor, typically 1/√d_k where d_k is the key dimension
//   - cache: KV cache to store key/value and get past history, can be nil to only use provided key/value
//
// Returns:
//
//	Attention output with shape [d_v, heads, seq_len_q]
func Attention(ctx ml.Context, query, key, value ml.Tensor, scale float64, cache kvcache.Cache) ml.Tensor {
	return AttentionWithVMLA(ctx, query, key, value, nil, nil, scale, cache)
}

func AttentionWithSinks(ctx ml.Context, query, key, value, sinks ml.Tensor, scale float64, cache kvcache.Cache) ml.Tensor {
	return AttentionWithVMLA(ctx, query, key, value, sinks, nil, scale, cache)
}

func AttentionWithVMLA(ctx ml.Context, query, key, value, sinks ml.Tensor, vmla ml.Tensor, scale float64, cache kvcache.Cache) ml.Tensor {
	ctx.Forward(query)
	if key != nil && value != nil {
		if query.Dim(0) != key.Dim(0) {
			panic(fmt.Errorf("d_k in attention operation does not match between query(%v) and key(%v)", query.Dim(0), key.Dim(0)))
		}

		if key.Dim(1) != value.Dim(1) {
			panic(fmt.Errorf("kv_heads in attention operation does not match between key(%v) and value(%v)", key.Dim(1), value.Dim(1)))
		}

		if key.Dim(2) != value.Dim(2) {
			panic(fmt.Errorf("seq_len_k in attention operation does not match between key(%v) and value(%v)", key.Dim(2), value.Dim(2)))
		}

		ctx.Forward(key, value)
		if cache != nil {
			cache.Put(ctx, key, value)
		}
	} else if cache == nil {
		panic("key & value tensors must be provided if cache is nil")
	}

	var mask ml.Tensor
	if cache != nil {
		key, value, mask = cache.Get(ctx)
	}

	// Only use the fast SDPA implementation if we have a cache, since that's what
	// will do any expected backend-specific transformations for us
	slog.Info("XXX before mlx_fast_scaled_dot_product_attention", "q", query)
	slog.Info("XXX before mlx_fast_scaled_dot_product_attention", "k", key)
	slog.Info("XXX before mlx_fast_scaled_dot_product_attention", "v", value)
	slog.Info("XXX before mlx_fast_scaled_dot_product_attention", "mask", mask) // WRONG - shape is good, but all -inf values

	fmt.Fprintln(os.Stderr, key.ToString())
	// panic("Just before ScaledDotProductAttention")

	// TODO - something's wrong here...  probably mask, but rule out the others first...

	if cache != nil {
		// TODO what to do with vmla?
		return query.ScaledDotProductAttention(ctx, key, value, scale, "array", []ml.Tensor{mask}, sinks)
	} else {
		// TODO transpose shapes are wrong
		query = query.Transpose(ctx, 0, 2, 1, 3)
		key = key.Transpose(ctx, 0, 2, 1, 3)
		value = value.Transpose(ctx, 1, 2, 0, 3).Contiguous(ctx, false)

		kq := query.Matmul(ctx, key)

		kq = kq.Scale(ctx, scale)
		if mask != nil {
			kq = kq.Add(ctx, mask)
		}
		kq = kq.Softmax(ctx)

		kqv := kq.Matmul(ctx, value)

		if vmla != nil {
			kqv = kqv.Matmul(ctx, vmla)
		}

		return kqv.Transpose(ctx, 0, 2, 1, 3).Contiguous(ctx, false)
	}
}
