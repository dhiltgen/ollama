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
	// ctx.CompareWith("/tmp/test", query, true) //
	// fmt.Fprintln(os.Stderr, query.ToString())
	// panic("input to self attention") //
	// ctx.CompareWith("/tmp/test", value.Contiguous(ctx, false), true) // requires contiguous, then correct
	// fmt.Fprintln(os.Stderr, value.ToString())
	// panic("before forward") //

	slog.Info("XXX Before ScaledDotProductAttention and cache put", "query", query)
	slog.Info("XXX Before ScaledDotProductAttention and cache put", "key", key)
	slog.Info("XXX Before ScaledDotProductAttention and cache put", "value", value)
	ctx.Forward(query)
	// ctx.CompareWith("/tmp/test", query, true) //
	// fmt.Fprintln(os.Stderr, query.ToString())
	// panic("input to self attention") //

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

	// ctx.CompareWith("/tmp/test", key, true) //
	// fmt.Fprintln(os.Stderr, key.ToString())
	// panic("before cache") // CORRECT
	// ctx.CompareWith("/tmp/test", value, true) //
	// fmt.Fprintln(os.Stderr, value.ToString())
	// panic("before cache") //

	var mask ml.Tensor
	if cache != nil {
		key, value, mask = cache.Get(ctx)
	}
	// slog.Info("XXX after cache get", "key", key)
	// slog.Info("XXX after cache get", "value", value)
	// slog.Info("XXX after cache get", "mask", mask)
	// panic("before sdpa")

	// Only use the fast SDPA implementation if we have a cache, since that's what
	// will do any expected backend-specific transformations for us
	slog.Info("XXX before mlx_fast_scaled_dot_product_attention", "q", query)
	fmt.Fprintln(os.Stderr, query.ToString())
	slog.Info("XXX before mlx_fast_scaled_dot_product_attention", "k", key)
	fmt.Fprintln(os.Stderr, key.ToString())
	slog.Info("XXX before mlx_fast_scaled_dot_product_attention", "v", value)
	fmt.Fprintln(os.Stderr, value.ToString())
	slog.Info("XXX before mlx_fast_scaled_dot_product_attention", "mask", mask)
	fmt.Fprintln(os.Stderr, mask.ToString())
	slog.Info("XXX before mlx_fast_scaled_dot_product_attention", "scale", scale)
	slog.Info("XXX before mlx_fast_scaled_dot_product_attention", "sinks", sinks)

	// fmt.Fprintln(os.Stderr, key.ToString())
	// panic("Just before ScaledDotProductAttention")
	// ctx.CompareWith("/tmp/test", map[string]ml.Tensor{"q": query.Contiguous(ctx, false), "k": key.Contiguous(ctx, false), "v": value.Contiguous(ctx, false)}, true)
	// panic("input to ScaledDotProductAttention") // CORRECT 5-9's similarity
	if cache != nil {
		// TODO what to do with vmla?
		return query.ScaledDotProductAttention(ctx, key, value, scale, "array", []ml.Tensor{mask}, sinks)
		// return query.ScaledDotProductAttention(ctx, key, value, scale, "causal", []ml.Tensor{}, sinks)

		// TODO these two produce identical output, but not similar enough - 92.9% - should be 99.999%
	} else {
		panic("else case not supported")
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
