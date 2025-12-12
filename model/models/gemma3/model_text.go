package gemma3

import (
	"math"

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

	// ctx.CompareWith("/tmp/test", map[string]ml.Tensor{"h": hiddenState}, true)
	// panic("starting layer forward") // h=0.9999882578849792 shape="[1 13 2560]" min_difference=[-0.7509308] max_difference=[1.9148865]

	B := hiddenState.Dim(0)
	L := hiddenState.Dim(1)
	// slog.Info("XXX start of Forward", "B", B, "L", L, "hiddenState", hiddenState)

	ropeBase := opts.ropeLocalBase
	if (layer+1)%gemmaGlobalCacheCount == 0 {
		ropeBase = opts.ropeGlobalBase
	}

	q := sa.Query.Forward(ctx, hiddenState)
	k := sa.Key.Forward(ctx, hiddenState)
	v := sa.Value.Forward(ctx, hiddenState)
	// ctx.CompareWith("/tmp/test", map[string]ml.Tensor{"q": q, "k": k, "v": v}, true)
	// 	panic("after qkv forward") //
	// 2025/12/10 15:24:26 INFO XXX tensors are similar k=0.9999958872795105 shape="[1 13 1024]" min_difference=[-0.5005493] max_difference=[0.5308075]
	// 2025/12/10 15:24:26 INFO XXX tensors are similar v=0.9999958872795105 shape="[1 13 1024]" min_difference=[-0.32923126] max_difference=[0.32646942]
	// 2025/12/10 15:24:26 INFO XXX tensors are similar q=0.9999900460243225 shape="[1 13 2048]" min_difference=[-0.8323364] max_difference=[0.83200073]

	q = q.Reshape(ctx, B, L, opts.numHeads, -1).Transpose(ctx, 0, 2, 1, 3)
	k = k.Reshape(ctx, B, L, opts.numKVHeads, -1).Transpose(ctx, 0, 2, 1, 3)
	v = v.Reshape(ctx, B, L, opts.numKVHeads, -1).Transpose(ctx, 0, 2, 1, 3).Contiguous(ctx, false)

	q = sa.QueryNorm.Forward(ctx, q, opts.eps)
	k = sa.KeyNorm.Forward(ctx, k, opts.eps)
	// ctx.CompareWith("/tmp/test", map[string]ml.Tensor{"q": q, "k": k}, true)
	// panic("after qk norm forward") //
	// 2025/12/10 15:27:45 INFO XXX tensors are similar q=0.999985933303833 shape="[1 8 13 256]" min_difference=[-0.08123684] max_difference=[0.083618164]
	// 2025/12/10 15:27:45 INFO XXX tensors are similar k=0.9999889731407166 shape="[1 4 13 256]" min_difference=[-0.20866394] max_difference=[0.19916534]

	traditional := false
	offset := int(0) // TODO is this right?

	q = q.RoPE(ctx, opts.attnKeyLen, traditional, opts.ropeScale, offset, ml.WithRoPEBase(ropeBase))
	k = k.RoPE(ctx, opts.attnKeyLen, traditional, opts.ropeScale, offset, ml.WithRoPEBase(ropeBase))
	// ctx.CompareWith("/tmp/test", map[string]ml.Tensor{"q": q, "k": k}, true)
	// panic("after qk rope") //
	// 2025/12/10 15:30:34 INFO XXX tensors are similar q=0.9999869465827942 shape="[1 8 13 256]" min_difference=[-0.07926178] max_difference=[0.07012844]
	// 2025/12/10 15:30:34 INFO XXX tensors are similar k=0.9999891519546509 shape="[1 4 13 256]" min_difference=[-0.21365738] max_difference=[0.19916534]

	// TODO - this is wrong somehow so commenting out
	// if opts.largeModelScaling {
	// 	q = q.Scale(ctx, 1.0/math.Sqrt(float64(opts.hiddenSize/opts.numHeads)))
	// } else {
	// 	q = q.Scale(ctx, 1.0/math.Sqrt(float64(opts.attnKeyLen)))
	// }

	scaleFactor := math.Pow(256, -0.5)

	kqv := nn.Attention(ctx, q, k, v, scaleFactor, cache)

	// ctx.CompareWith("/tmp/test", map[string]ml.Tensor{"kqv": kqv}, true)
	// panic("output from self attention") // kqv=0.9999763369560242 shape="[1 8 13 256]" min_difference=[-0.85839653] max_difference=[1.2572632]

	kqv = kqv.Transpose(ctx, 0, 2, 1, 3).Reshape(ctx, B, L, -1)

	t := sa.Output.Forward(ctx, kqv)
	// ctx.CompareWith("/tmp/test", map[string]ml.Tensor{"t": t}, true)
	// panic("output from self attention") // t=0.9999837279319763 shape="[1 13 2560]" min_difference=[-0.42661285] max_difference=[0.6028366]

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
	// ctx.CompareWith("/tmp/test", map[string]ml.Tensor{"h": hiddenState, "up": mlp.Up.Weight, "down": mlp.Down.Weight, "gate": mlp.Gate.Weight}, true) //
	// panic("after post attention norm")
	// 2025/12/11 08:57:21 INFO XXX tensors are similar h=0.9722424149513245 shape="[1 13 2560]" min_difference=[-4.394745] max_difference=[4.346017]
	// 2025/12/11 08:57:21 INFO XXX tensors are similar up=1.0000001192092896 shape="[10240 2560]" min_difference=[0] max_difference=[0]
	// 2025/12/11 08:57:21 INFO XXX tensors are similar down=1 shape="[2560 10240]" min_difference=[0] max_difference=[0]
	// 2025/12/11 08:57:22 INFO XXX tensors are similar gate=1 shape="[10240 2560]" min_difference=[0] max_difference=[0]                                                                                                           //

	t1 := mlp.Up.Forward(ctx, hiddenState)
	t2 := mlp.Gate.Forward(ctx, hiddenState)
	// ctx.CompareWith("/tmp/test", map[string]ml.Tensor{"upf": t1, "gatef": t2}, true)
	// panic("starting layer forward") //
	// 2025/12/11 09:06:20 INFO XXX tensors are similar upf=0.978447437286377 shape="[1 13 10240]" min_difference=[-5.1779633] max_difference=[3.4795022]
	// 2025/12/11 09:06:20 INFO XXX tensors are similar gatef=0.9751632213592529 shape="[1 13 10240]" min_difference=[-5.7206907] max_difference=[5.279685]
	hiddenState = t2.GELU(ctx, t1)
	// hiddenState = mlp.Gate.Forward(ctx, hiddenState).GELU(ctx, mlp.Up.Forward(ctx, hiddenState))
	r := mlp.Down.Forward(ctx, hiddenState)
	// ctx.CompareWith("/tmp/test", map[string]ml.Tensor{"downf": r}, true)
	// panic("starting layer forward") //

	return r
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
	// ctx.CompareWith("/tmp/test", map[string]ml.Tensor{"h": hiddenState}, true)
	// panic("starting layer forward") // h=0.9999922513961792 shape="[1 13 2560]" min_difference=[-0.03314209] max_difference=[0.10057831]

	residual := hiddenState
	hiddenState = l.AttentionNorm.Forward(ctx, hiddenState, opts.eps)
	// ctx.CompareWith("/tmp/test", map[string]ml.Tensor{"h": hiddenState}, true)
	// panic("after AttentionNorm") // h=0.9999882578849792 shape="[1 13 2560]" min_difference=[-0.7509308] max_difference=[1.9148865]
	hiddenState = l.SelfAttention.Forward(ctx, layer, hiddenState, positionIDs, cache, opts)
	// ctx.CompareWith("/tmp/test", map[string]ml.Tensor{"t": hiddenState}, true)
	// panic("after SelfAttention") // t=0.9999837279319763 shape="[1 13 2560]" min_difference=[-0.42661285] max_difference=[0.6028366]

	hiddenState = l.PostAttentionNorm.Forward(ctx, hiddenState, opts.eps)
	// ctx.CompareWith("/tmp/test", map[string]ml.Tensor{"t": hiddenState}, true)
	// panic("after post attention norm") // t=0.9999939203262329 shape="[1 13 2560]" min_difference=[-4.0560303] max_difference=[3.5961914]

	// In the final layer (outputs != nil), optimize by pruning to just the token positions
	// we need logits for.
	if outputs != nil {
		// slog.Info("Before TakeAxes", "hiddenState", hiddenState)
		// slog.Info("Before TakeAxes", "residual", residual)
		// slog.Info("Before TakeAxes", "outputs", outputs)
		hiddenState = hiddenState.TakeAxes(ctx, outputs, 1)
		residual = residual.TakeAxes(ctx, outputs, 1)
		// slog.Info("after TakeAxes", "hiddenState", hiddenState)
		// slog.Info("after TakeAxes", "residual", residual)
		// panic("XXX")
	}

	hiddenState = hiddenState.Add(ctx, residual)
	residual = hiddenState
	// ctx.CompareWith("/tmp/test", map[string]ml.Tensor{"h": hiddenState}, true)
	// panic("before MLPNorm") // h=0.9999914765357971 shape="[1 13 2560]" min_difference=[-4.541565] max_difference=[3.4016113]

	hiddenState = l.MLPNorm.Forward(ctx, hiddenState, opts.eps)
	// ctx.CompareWith("/tmp/test", map[string]ml.Tensor{"h": hiddenState}, true)
	// panic("after post attention norm") // h=0.9999774694442749 shape="[1 13 2560]" min_difference=[-0.30542183] max_difference=[0.124988556]

	hiddenState = l.MLP.Forward(ctx, hiddenState, opts) // TODO this is where it goes bad most likely...
	// ctx.CompareWith("/tmp/test", map[string]ml.Tensor{"t": hiddenState}, true)
	// panic("after post attention norm") // t=0.9998757243156433 shape="[1 13 2560]" min_difference=[-0.41888905] max_difference=[0.43873215]

	hiddenState = l.PostMLPNorm.Forward(ctx, hiddenState, opts.eps)
	// ctx.CompareWith("/tmp/test", map[string]ml.Tensor{"t": hiddenState}, true)
	// panic("after post attention norm") // t=0.9999986886978149 shape="[1 13 2560]" min_difference=[-39] max_difference=[6.501816]

	x := hiddenState.Add(ctx, residual)
	// slog.Info("5", "hiddenState", x)
	// if outputs != nil {
	// ctx.CompareWith("/tmp/test", map[string]ml.Tensor{"t": x}, true)
	// panic("after post attention norm") // t=0.9999957084655762 shape="[1 13 2560]" min_difference=[-62.378906] max_difference=[7.8084717]
	// }
	return x
}

func (m *TextModel) Forward(ctx ml.Context, batch input.Batch, cache kvcache.Cache) ml.Tensor {

	positions := ctx.Input().FromInts(batch.Positions, len(batch.Positions))

	// TODO is this the right place to create this?
	// if m.TokenEmbedding == nil {
	// 	m.TokenEmbedding = &nn.Embedding{
	// 		Weight: ctx.RandomNormal([]int{m.vocabSize, m.hiddenSize}, ml.DTypeFloat32, 0, float32(math.Sqrt(1/float64(m.hiddenSize))), nil),
	// 	}
	// }

	// slog.Info("XXX TextModel.Forward", "batch", batch.Inputs)
	// fmt.Fprintln(os.Stderr, m.TokenEmbedding.Weight.ToString())
	// panic("TokenEmbedding") // CORRECT

	// fmt.Fprintln(os.Stderr, batch.Inputs.ToString())
	// panic("batch.Inputs") // CORRECT
	hiddenState := m.TokenEmbedding.Forward(ctx, batch.Inputs)
	// fmt.Fprintln(os.Stderr, hiddenState.ToString())
	// panic("TokenEmbedding.Forward") // CORRECt, but has more token zero rows at the end - probably OK
	// slog.Info("XXX scale", "m.TextConfig.hiddenSize", m.TextConfig.hiddenSize, "scale", math.Sqrt(float64(m.TextConfig.hiddenSize)))
	hiddenState = hiddenState.Scale(ctx, math.Sqrt(float64(m.TextConfig.hiddenSize)))
	// fmt.Fprintln(os.Stderr, hiddenState.ToString())
	// panic("scale") // CORRECT
	// slog.Info("XXX after Scale", "hiddenState", hiddenState)

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
		// fmt.Fprintln(os.Stderr, hiddenState.ToString())
		// panic("before first layer") // CORRECT
		if cache != nil {
			// cacheType := cacheTypeSWA
			// if (i+1)%gemmaGlobalCacheCount == 0 {
			// 	cacheType = cacheTypeCausal
			// }
			cache.SetLayer(i)

			// TODO this needs to come back
			// wc := cache.(*kvcache.WrapperCache)
			// wc.SetLayerType(cacheType)

			// if causal, ok := wc.UnderlyingCache().(*kvcache.Causal); ok {
			// 	causal.SetCausal(ctx, kvcache.CausalOptions{Except: except})
			// }
		}

		var lastLayerOutputs ml.Tensor
		if i == len(m.Layers)-1 {
			// slog.Info("XXX last layer", "i", i)
			// panic("last layer")
			lastLayerOutputs = batch.Outputs
		}

		hiddenState = layer.Forward(ctx, i, hiddenState, positions, lastLayerOutputs, cache, m.TextConfig)
		// if i == 33 {
		// 	ctx.CompareWith("/tmp/test", map[string]ml.Tensor{"t": hiddenState}, true)
		// 	panic("after post attention norm") //
		// }
	}
	// ctx.CompareWith("/tmp/test", map[string]ml.Tensor{"t": hiddenState}, true)
	// panic("after post attention norm") // mismatched shapes:  file: [1 1 2560] vs. input [1 13 2560]

	// slog.Info("XXX before OutputNorm.Forward", "hiddenState", hiddenState)
	// fmt.Fprintln(os.Stderr, hiddenState.ToString())
	hiddenState = m.OutputNorm.Forward(ctx, hiddenState, m.eps)
	// ctx.CompareWith("/tmp/test", map[string]ml.Tensor{"t": hiddenState}, true)
	// panic("after post attention norm") //

	// out := hiddenState.Matmul(ctx, m.TokenEmbedding.Weight.Transpose(ctx))
	// slog.Info("XXX after as_linear equivalent", "hiddenState", out)
	// fmt.Fprintln(os.Stderr, out.ToString())

	// panic("after forward pass")

	return hiddenState
}
