package llama

import (
	"math"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/model"
)

type Options struct {
	RopeFactors                      ml.Tensor `ggml:"rope_freqs.weight"`
	hiddenSize, numHeads, numKVHeads int64
	eps, ropeBase, ropeScale         float32
	ropeDim                          uint32
}

type Model struct {
	model.Base

	TextProcessor

	TokenEmbedding *nn.Embedding `ggml:"token_embd"`
	Layers         []Layer       `ggml:"blk"`
	OutputNorm     *nn.RMSNorm   `ggml:"output_norm"`
	Output         *nn.Linear    `ggml:"output"`

	*Options
}

func New(c ml.Config) (model.Model, error) {
	r := &Model{
		TextProcessor: newTextProcessor(c),
		Layers:        make([]Layer, c.Uint("block_count")),
		Options: &Options{
			hiddenSize: int64(c.Uint("embedding_length")),
			numHeads:   int64(c.Uint("attention.head_count")),
			numKVHeads: int64(c.Uint("attention.head_count_kv")),
			eps:        c.Float("attention.layer_norm_rms_epsilon"),
			ropeBase:   c.Float("rope.freq_base"),
			ropeScale:  c.Float("rope.freq_scale", 1),
			ropeDim:    c.Uint("rope.dimension_count"),
		},
	}
	return r, nil
}

type SelfAttention struct {
	Query  *nn.Linear `ggml:"attn_q"`
	Key    *nn.Linear `ggml:"attn_k"`
	Value  *nn.Linear `ggml:"attn_v"`
	Output *nn.Linear `ggml:"attn_output"`
}

func (sa *SelfAttention) Forward(ctx ml.Context, hiddenState ml.Tensor, offset int32, cache model.Cache, opts *Options) ml.Tensor {
	// Note: this impl was derived from multiple references and is not necessarily the most optimal.
	// https://github.com/meta-llama/llama-models/blob/main/models/llama3/reference_impl/model.py
	// https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/llama.py
	shape := hiddenState.Shape()
	bsz := shape[0]
	seqlen := shape[1]
	n_local_heads := opts.numHeads
	n_local_kv_heads := opts.numKVHeads
	// n_rep := int(n_local_heads / n_local_kv_heads)

	head_dim := opts.hiddenSize / opts.numHeads
	scale := math.Pow(float64(head_dim), -0.5)

	xq := sa.Query.Forward(ctx, hiddenState)
	xk := sa.Key.Forward(ctx, hiddenState)
	xv := sa.Value.Forward(ctx, hiddenState)

	xq = xq.Reshape(ctx, bsz, seqlen, n_local_heads, head_dim).Permute(ctx, 0, 2, 1, 3)
	xk = xk.Reshape(ctx, bsz, seqlen, n_local_kv_heads, head_dim).Permute(ctx, 0, 2, 1, 3)
	xv = xv.Reshape(ctx, bsz, seqlen, n_local_kv_heads, head_dim).Permute(ctx, 0, 2, 1, 3)

	// TODO - begin ROPE impl, this should be abstracted out someplace...
	// Reference: https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/rope_utils.py#L9
	dims := opts.ropeDim
	base := opts.ropeBase // aka rope_scale
	if base == 0 {
		base = 10000.0
	}
	low_freq_factor := opts.ropeScale // ???
	low_freq_factorA, _ := ctx.FromFloatSlice([]float32{low_freq_factor}, 1)
	high_freq_factor := float32(4.0) // TODO should attempt to get from metadata
	factor := float32(8.0)           // metadata?
	factorA, _ := ctx.FromFloatSlice([]float32{factor}, 1)
	old_context_len := float32(8192) // metadata?  (aka original_max_position_embeddings)
	old_context_lenA, _ := ctx.FromFloatSlice([]float32{old_context_len}, 1)

	// Calcs...
	low_freq_wavelen := float32(old_context_len) / low_freq_factor
	low_freq_wavelenA, _ := ctx.FromFloatSlice([]float32{low_freq_wavelen}, 1)
	high_freq_wavelen := float32(old_context_len) / high_freq_factor
	high_freq_wavelenA, _ := ctx.FromFloatSlice([]float32{high_freq_wavelen}, 1)

	// freqs = base ** (mx.arange(0, dims, 2) / dims)
	tmp := ctx.Arange(0, float64(dims), 2, ml.DTypeI32)
	dimsA, _ := ctx.FromFloatSlice([]float32{float32(dims)}, 1)
	tmp = tmp.Divide(ctx, dimsA)
	baseA, _ := ctx.FromFloatSlice([]float32{float32(base)}, 1)
	freqs := baseA.Power(ctx, ctx.Arange(0, float64(dims), 2, ml.DTypeF32).Divide(ctx, dimsA))
	two_pi, _ := ctx.FromFloatSlice([]float32{2 * math.Pi}, 1)
	wavelens := freqs.Mul(ctx, two_pi)
	freqs = ctx.Where(wavelens.Greater(ctx, low_freq_wavelenA), freqs.Mul(ctx, factorA), freqs)
	is_medium_freq := wavelens.Greater(ctx, high_freq_wavelenA).BitwiseAnd(ctx, wavelens.Less(ctx, low_freq_wavelenA))
	// smooth_factors = (old_context_len / wavelens - low_freq_factor) / (high_freq_factor - low_freq_factor)
	high_minus_low, _ := ctx.FromFloatSlice([]float32{high_freq_factor - low_freq_factor}, 1)
	smooth_factors := old_context_lenA.Divide(ctx, wavelens).Subtract(ctx, low_freq_factorA).Divide(ctx, high_minus_low)
	// smooth_freqs = freqs / ((1 - smooth_factors) / factor + smooth_factors)
	oneA, _ := ctx.FromFloatSlice([]float32{1}, 1)
	smooth_freqs := freqs.Divide(ctx, oneA.Subtract(ctx, smooth_factors).Divide(ctx, factorA).Add(ctx, smooth_factors))
	_freqs := ctx.Where(is_medium_freq, smooth_freqs, freqs)

	xq = xq.Rope(
		ctx,
		offset,
		_freqs,
		dims,
		0,   // base unused
		1.0, // scale
	)
	xk = xk.Rope(
		ctx,
		offset,
		_freqs,
		dims,
		0,   // base unused
		1.0, // scale
	)

	// TODO - when this comes back, the input should be truncated to just the latest token
	// keys, values = cache.Put(ctx, xk, xv, cache.Options)
	keys := xk
	values := xv

	// // Begin scaled dot product attention
	// // Note: this impl is close, but not quite working as an alternative to FastScaledDotProductAttention (incorrect shapes someplace)
	// keys = keys.Repeat(ctx, n_rep, 2).Contiguous(ctx)
	// values = values.Repeat(ctx, n_rep, 2).Contiguous(ctx)

	// xq = xq.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)         // (bs, n_local_heads, seqlen, head_dim)
	// keys = keys.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)     // (bs, n_local_heads, cache_len + seqlen, head_dim)
	// values = values.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx) // (bs, n_local_heads, cache_len + seqlen, head_dim)

	// kp := keys.Permute(ctx, 0, 1, 3, 2)
	// scores := kp.Mulmat(ctx, xq).Scale(ctx, 1.0/math.Sqrt(float64(head_dim)))
	// // TODO mask here
	// scores = scores.Softmax(ctx) // Without axis=-1 this starts to drift
	// output := values.Mulmat(ctx, scores)
	// // End scaled dot product attention

	output := ctx.FastScaledDotProductAttention(xq, keys, values, float32(scale), nil)
	// slog.Info("XXX output from scaled dot product attention", "output", output)

	output = output.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	output = output.Reshape(ctx, bsz, seqlen, -1)
	output = sa.Output.Forward(ctx, output)
	return output
}

type MLP struct {
	Up   *nn.Linear `ggml:"ffn_up"`
	Down *nn.Linear `ggml:"ffn_down"`
	Gate *nn.Linear `ggml:"ffn_gate"`
}

func (mlp *MLP) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *Options) ml.Tensor {
	g := mlp.Gate.Forward(ctx, hiddenState)
	x := mlp.Up.Forward(ctx, hiddenState)
	hiddenState = g.SILU(ctx).Mul(ctx, x)
	return mlp.Down.Forward(ctx, hiddenState)
}

type Layer struct {
	AttentionNorm *nn.RMSNorm `ggml:"attn_norm"`
	SelfAttention *SelfAttention
	MLPNorm       *nn.RMSNorm `ggml:"ffn_norm"`
	MLP           *MLP
}

func (l *Layer) Forward(ctx ml.Context, hiddenState ml.Tensor, offset int32, cache model.Cache, opts *Options) ml.Tensor {
	residual := hiddenState
	hiddenState = l.AttentionNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = l.SelfAttention.Forward(ctx, hiddenState, offset, cache, opts)
	hiddenState = hiddenState.Add(ctx, residual)
	residual = hiddenState

	hiddenState = l.MLPNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = l.MLP.Forward(ctx, hiddenState, opts)
	out := hiddenState.Add(ctx, residual)
	return out
}

func (m *Model) Forward(ctx ml.Context, opts model.Options) (ml.Tensor, error) {
	inputs, err := ctx.FromIntSlice(opts.Inputs(), len(opts.Inputs()))
	if err != nil {
		return nil, err
	}
	offset := int32(0)
	hiddenState := m.TokenEmbedding.Forward(ctx, inputs).Reshape(ctx, 1, -1, 4096)

	for i, layer := range m.Layers {
		hiddenState = layer.Forward(ctx, hiddenState, offset, opts.Cache.Sub(i), m.Options)
	}

	hiddenState = m.OutputNorm.Forward(ctx, hiddenState, m.eps)

	// TODO this isn't the right solution, but we need to do this only once (there's a bug here someplace...)
	s := m.Output.Weight.Shape()
	if s[0] != hiddenState.Shape()[2] {
		m.Output.Weight = m.Output.Weight.Permute(ctx, 1, 0)
	}

	outputs, err := ctx.FromIntSlice([]int32{-1}, 1, 1)
	if err != nil {
		return nil, err
	}
	t := hiddenState.Rows(ctx, outputs).Reshape(ctx, 1, -1)

	hiddenState = m.Output.Forward(ctx, t)
	return hiddenState, nil
}

func init() {
	model.Register("llama", New)
}
