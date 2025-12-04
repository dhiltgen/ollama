package gemma3

import (
	"fmt"
	"log/slog"
	"math"
	"os"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/ml/nn/fast"
	"github.com/ollama/ollama/ml/nn/rope"
	"github.com/ollama/ollama/model/input"
)

type TextConfig struct {
	hiddenSize, numHeads, numKVHeads  int
	attnKeyLen, attnValLen, vocabSize int
	eps, ropeScale                    float32
	ropeLocalBase, ropeGlobalBase     float32
	largeModelScaling                 bool
}

type TextModel struct {
	TokenEmbedding *nn.Embedding `gguf:"token_embd"`
	Layers         []TextLayer   `gguf:"blk"`
	OutputNorm     *nn.RMSNorm   `gguf:"output_norm"`
	Output         *nn.Linear    `gguf:"output,alt:token_embd"`

	*TextConfig
}

const (
	gemmaGlobalCacheCount = 6
	gemma27BLayerCount    = 62

	// Derived from mlx-lm
	hiddenSize           = int(1152)
	numHiddenLayers      = int(26)
	intermediateSize     = int(6912)
	numAttentionHeads    = int(4)
	headDim              = int(256)
	rmsNormEps           = float32(1.0e-6)
	vocabSize            = int(262144)
	numKeyValueHeads     = int(1)
	ropeGlobalBaseFreq   = float32(1000000.0)
	ropeLocalBaseFreq    = float32(10000.0)
	ropeTraditional      = false
	queryPreAttnScalar   = float32(256)
	slidingWindow        = int(512)
	slidingWindowPattern = int(6)

	// dim = args.hidden_size
	// self.n_heads = n_heads = args.num_attention_heads
	// self.n_kv_heads = n_kv_heads = args.num_key_value_heads
	// self.repeats = n_heads // n_kv_heads
	// self.head_dim = head_dim = args.head_dim
	// self.layer_idx = layer_idx

	// self.scale = args.query_pre_attn_scalar**-0.5

)

const (
	cacheTypeSWA = iota
	cacheTypeCausal
)

func newTextModel(c fs.Config) *TextModel {
	numBlocks := int(c.Uint("block_count"))

	m := TextModel{
		Layers: make([]TextLayer, numBlocks),
		TextConfig: &TextConfig{
			// hiddenSize:     hiddenSize, //int(c.Uint("embedding_length")),
			hiddenSize:     int(c.Uint("embedding_length")),
			vocabSize:      vocabSize,
			numHeads:       int(c.Uint("attention.head_count")),
			numKVHeads:     int(c.Uint("attention.head_count_kv")),
			attnKeyLen:     int(c.Uint("attention.key_length", 256)),
			attnValLen:     int(c.Uint("attention.value_length", 256)),
			eps:            c.Float("attention.layer_norm_rms_epsilon", 1e-06),
			ropeLocalBase:  c.Float("rope.local.freq_base", 10000.0),
			ropeGlobalBase: c.Float("rope.global.freq_base", 1000000.0),
			ropeScale:      1,
			// NOTE: the rope.scaling.factor is set incorrectly in the official QAT weights
			//       (8 instead of 1)
			// ropeScale:      c.Float("rope.scaling.factor", 1.0),
		},
	}

	if numBlocks == gemma27BLayerCount {
		m.largeModelScaling = true
	}

	return &m
}

type TextSelfAttention struct {
	Query     *nn.Linear  `gguf:"attn_q"`
	QueryNorm *nn.RMSNorm `gguf:"attn_q_norm"`
	Key       *nn.Linear  `gguf:"attn_k"`
	KeyNorm   *nn.RMSNorm `gguf:"attn_k_norm"`
	Value     *nn.Linear  `gguf:"attn_v"`
	Output    *nn.Linear  `gguf:"attn_output"`
}

func (sa *TextSelfAttention) Forward(ctx ml.Context, layer int, hiddenState, positionIDs ml.Tensor, cache kvcache.Cache, opts *TextConfig) ml.Tensor {
	B := hiddenState.Dim(0)
	L := hiddenState.Dim(1)
	slog.Info("XXX start of Forward", "B", B, "L", L, "hiddenState", hiddenState)

	ropeBase := opts.ropeLocalBase
	if (layer+1)%gemmaGlobalCacheCount == 0 {
		ropeBase = opts.ropeGlobalBase
	}
	// fmt.Fprintf(os.Stderr, hiddenState.ToString())
	// panic("before q forward") // CORRECT

	q := sa.Query.Forward(ctx, hiddenState)
	// fmt.Fprintf(os.Stderr, q.ToString())
	// panic("after q forward") // CORRECT
	// slog.Info("XXX before reshape+transpose", "q", q)
	// slog.Info("XXX", "reshape", []int{B, L, opts.numHeads, -1})
	q = q.Reshape(ctx, B, L, opts.numHeads, -1).Transpose(ctx, 0, 2, 1, 3)
	// slog.Info("XXX after reshape+transpose", "q", q)
	q = sa.QueryNorm.Forward(ctx, q, opts.eps)
	// slog.Info("XXX after querynorm", "q", q)
	traditional := false
	offset := int(0) // TODO is this right?

	// fmt.Fprintln(os.Stderr, q.ToString())
	// panic("before q rope") // CORRECT

	q = q.RoPE(ctx, opts.attnKeyLen, traditional, opts.ropeScale, offset, ml.WithRoPEBase(ropeBase))
	// fmt.Fprintln(os.Stderr, q.ToString())
	// panic("after q rope") // CORRECT

	// TODO - this is wrong somehow so commenting out
	// if opts.largeModelScaling {
	// 	q = q.Scale(ctx, 1.0/math.Sqrt(float64(opts.hiddenSize/opts.numHeads)))
	// } else {
	// 	q = q.Scale(ctx, 1.0/math.Sqrt(float64(opts.attnKeyLen)))
	// }

	// slog.Info("XXX before Key.Forward", "key", sa.Key.Weight, "hiddenState", hiddenState)
	k := sa.Key.Forward(ctx, hiddenState)
	// slog.Info("XXX after Key.Forward", "key", k)
	k = k.Reshape(ctx, B, L, opts.numKVHeads, -1).Transpose(ctx, 0, 2, 1, 3)
	// slog.Info("XXX after reshape", "key", k, "KeyNorm", sa.KeyNorm.Weight)
	k = sa.KeyNorm.Forward(ctx, k, opts.eps)
	// slog.Info("XXX after KeyNorm.Forward", "key", k)
	k = k.RoPE(ctx, opts.attnKeyLen, traditional, opts.ropeScale, offset, ml.WithRoPEBase(ropeBase))
	// slog.Info("XXX after RoPE", "key", k)
	// fmt.Fprintln(os.Stderr, k.ToString())
	// panic("after k rope") // CORRECT

	v := sa.Value.Forward(ctx, hiddenState)
	v = v.Reshape(ctx, B, L, opts.numKVHeads, -1).Transpose(ctx, 0, 2, 1, 3)

	scaleFactor := 1.0

	// fmt.Fprintln(os.Stderr, q.ToString()) // CORRECT now
	// fmt.Fprintln(os.Stderr, k.ToString()) // CORRECT
	// fmt.Fprintln(os.Stderr, v.ToString()) // CORRECT
	// panic("before QKV Attention")

	kqv := nn.Attention(ctx, q, k, v, scaleFactor, cache)
	// fmt.Fprintln(os.Stderr, kqv.ToString())
	// panic("after scaled dot product") // WRONG - all nans

	kqv = kqv.Transpose(ctx, 0, 2, 1, 3).Reshape(ctx, B, L, -1)

	t := sa.Output.Forward(ctx, kqv)
	// fmt.Fprintln(os.Stderr, t.ToString())
	// panic("final output") // WRONG! nan's
	return t
}

func (m *TextModel) Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	ropeBase := m.TextConfig.ropeLocalBase
	if (layer+1)%gemmaGlobalCacheCount == 0 {
		ropeBase = m.TextConfig.ropeGlobalBase
	}

	return fast.RoPE(ctx, key, shift, m.TextConfig.attnKeyLen, ropeBase, 1/m.TextConfig.ropeScale, rope.WithTypeNeoX()), nil
}

type TextMLP struct {
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
	Gate *nn.Linear `gguf:"ffn_gate"`
}

func (mlp *TextMLP) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *TextConfig) ml.Tensor {
	hiddenState = mlp.Gate.Forward(ctx, hiddenState).GELU(ctx, mlp.Up.Forward(ctx, hiddenState))
	return mlp.Down.Forward(ctx, hiddenState)
}

type TextLayer struct {
	AttentionNorm     *nn.RMSNorm `gguf:"attn_norm"`
	SelfAttention     *TextSelfAttention
	PostAttentionNorm *nn.RMSNorm `gguf:"post_attention_norm"`
	MLPNorm           *nn.RMSNorm `gguf:"ffn_norm"`
	MLP               *TextMLP
	PostMLPNorm       *nn.RMSNorm `gguf:"post_ffw_norm"`
}

func (l *TextLayer) Forward(ctx ml.Context, layer int, hiddenState, positionIDs, outputs ml.Tensor, cache kvcache.Cache, opts *TextConfig) ml.Tensor {
	residual := hiddenState
	// fmt.Fprintf(os.Stderr, hiddenState.ToString())
	// panic("before attention norm") // CORRECT

	// fmt.Fprintf(os.Stderr, l.AttentionNorm.Weight.ToString())
	// panic("l.AttentionNorm.Weight") // CORRECT
	hiddenState = l.AttentionNorm.Forward(ctx, hiddenState, opts.eps)
	// fmt.Fprintln(os.Stderr, hiddenState.ToString())
	// panic("after attention norm") // CORRECT
	hiddenState = l.SelfAttention.Forward(ctx, layer, hiddenState, positionIDs, cache, opts)
	// fmt.Fprintln(os.Stderr, hiddenState.ToString())
	// panic("after self attention")

	hiddenState = l.PostAttentionNorm.Forward(ctx, hiddenState, opts.eps)

	// In the final layer (outputs != nil), optimize by pruning to just the token positions
	// we need logits for.
	if outputs != nil {
		slog.Info("Before TakeAxes", "hiddenState", hiddenState)
		slog.Info("Before TakeAxes", "residual", residual)
		slog.Info("Before TakeAxes", "outputs", outputs)
		hiddenState = hiddenState.TakeAxes(ctx, outputs, 1)
		residual = residual.TakeAxes(ctx, outputs, 1)
		slog.Info("after TakeAxes", "hiddenState", hiddenState)
		slog.Info("after TakeAxes", "residual", residual)
		// panic("XXX")
	}

	hiddenState = hiddenState.Add(ctx, residual)
	residual = hiddenState

	hiddenState = l.MLPNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = l.MLP.Forward(ctx, hiddenState, opts)
	hiddenState = l.PostMLPNorm.Forward(ctx, hiddenState, opts.eps)
	return hiddenState.Add(ctx, residual)
}

func (m *TextModel) Forward(ctx ml.Context, batch input.Batch, cache kvcache.Cache) ml.Tensor {

	positions := ctx.Input().FromInts(batch.Positions, len(batch.Positions))

	// TODO is this the right place to create this?
	// if m.TokenEmbedding == nil {
	// 	m.TokenEmbedding = &nn.Embedding{
	// 		Weight: ctx.RandomNormal([]int{m.vocabSize, m.hiddenSize}, ml.DTypeFloat32, 0, float32(math.Sqrt(1/float64(m.hiddenSize))), nil),
	// 	}
	// }

	slog.Info("XXX TextModel.Forward", "batch", batch.Inputs)
	// fmt.Fprintln(os.Stderr, m.TokenEmbedding.Weight.ToString())
	// panic("TokenEmbedding") // CORRECT

	// fmt.Fprintln(os.Stderr, batch.Inputs.ToString())
	// panic("batch.Inputs") // CORRECT
	hiddenState := m.TokenEmbedding.Forward(ctx, batch.Inputs)
	// fmt.Fprintln(os.Stderr, hiddenState.ToString())
	// panic("TokenEmbedding.Forward") // CORRECt, but has more token zero rows at the end - probably OK
	slog.Info("XXX scale", "m.TextConfig.hiddenSize", m.TextConfig.hiddenSize, "scale", math.Sqrt(float64(m.TextConfig.hiddenSize)))
	hiddenState = hiddenState.Scale(ctx, math.Sqrt(float64(m.TextConfig.hiddenSize)))
	// fmt.Fprintln(os.Stderr, hiddenState.ToString())
	// panic("scale") // CORRECT
	slog.Info("XXX after Scale", "hiddenState", hiddenState)

	// fmt.Fprintf(os.Stderr, hiddenState.ToString())
	// panic("Does it look OK?")

	// set image embeddings
	var except []int
	for _, image := range batch.Multimodal {
		visionOutputs := image.Multimodal[0].Tensor
		ctx.Forward(visionOutputs.Copy(ctx, hiddenState.AsStrided(ctx,
			[]int{visionOutputs.Dim(0) * visionOutputs.Dim(1)},
			[]int{image.Index * hiddenState.Stride(1)}, 0)))

		for i := range visionOutputs.Dim(1) {
			except = append(except, image.Index+i)
		}
	}

	for i, layer := range m.Layers {
		// gemma alternates between the sliding window (local) and causal (global)
		// kv cache every 6 layers
		if cache != nil {
			cacheType := cacheTypeSWA
			if (i+1)%gemmaGlobalCacheCount == 0 {
				cacheType = cacheTypeCausal
			}
			cache.SetLayer(i)
			wc := cache.(*kvcache.WrapperCache)
			wc.SetLayerType(cacheType)

			if causal, ok := wc.UnderlyingCache().(*kvcache.Causal); ok {
				causal.SetCausal(ctx, kvcache.CausalOptions{Except: except})
			}
		}

		var lastLayerOutputs ml.Tensor
		if i == len(m.Layers)-1 {
			lastLayerOutputs = batch.Outputs
		}

		hiddenState = layer.Forward(ctx, i, hiddenState, positions, lastLayerOutputs, cache, m.TextConfig)
		// fmt.Fprintln(os.Stderr, hiddenState.ToString())
		// panic("after first layer")
	}

	slog.Info("XXX before OutputNorm.Forward", "hiddenState", hiddenState)
	fmt.Fprintln(os.Stderr, hiddenState.ToString())
	hiddenState = m.OutputNorm.Forward(ctx, hiddenState, m.eps)
	slog.Info("XXX after OutputNorm.Forward", "hiddenState", hiddenState)
	fmt.Fprintln(os.Stderr, hiddenState.ToString())

	out := hiddenState.Matmul(ctx, m.TokenEmbedding.Weight.Transpose(ctx))
	slog.Info("XXX after as_linear equivalent", "hiddenState", out)
	fmt.Fprintln(os.Stderr, out.ToString())

	// panic("after forward pass")

	return hiddenState
}
