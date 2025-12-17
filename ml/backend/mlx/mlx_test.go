package mlx

import (
	"fmt"
	"log/slog"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	fsggml "github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
	"github.com/ollama/ollama/model/models/gemma3"
	"github.com/ollama/ollama/runner/common"
	"github.com/ollama/ollama/sample"
)

func TestLoadModel(t *testing.T) {
	dir := "/Users/daniel/Models/gemma-3-4b-it/"
	b := &Backend{}
	err := b.LoadSafeTensors(dir)
	if err != nil {
		t.Fatalf("load failed: %s", err)
	}
}

func TestFromInts(t *testing.T) {
	b := &Backend{}
	c := b.NewContext()
	defer c.Close()
	data := []int32{1, 2, 3, 4, 5, 6}
	a := c.FromInts(data, 2, 3)
	slog.Info("", "array", a)
	t.Log(a.ToString())
	if !reflect.DeepEqual(a.Shape(), []int{2, 3}) {
		t.Fatalf("incorrect shape: %v", a.Shape())
	}
}

func TestFromFloats(t *testing.T) {
	b := &Backend{}
	c := b.NewContext()
	defer c.Close()
	data := []float32{1, 2, 3, 4, 5, 6}
	a := c.FromFloats(data, 2, 3)
	slog.Info("", "array", a)
	t.Log(a.ToString())
	if !reflect.DeepEqual(a.Shape(), []int{2, 3}) {
		t.Fatalf("incorrect shape: %v", a.Shape())
	}
	res := a.Floats()
	if !reflect.DeepEqual(res, data) {
		t.Fatalf("incorrect results: %v", res)
	}
}

func TestAdd(t *testing.T) {
	b := &Backend{}
	c := b.NewContext()
	defer c.Close()
	t1 := c.Arange(0, 24, 1, ml.DTypeFloat16)
	t2 := c.Arange(0, 24, 1, ml.DTypeFloat16)
	exp := c.Arange(0, 48, 2, ml.DTypeFloat16)
	t3 := t1.Add(c, t2)
	c.Compute(t3, exp)
	t3f := t3.Floats()
	if !reflect.DeepEqual(t3f, exp.Floats()) {
		t.Fatalf("incorrect result: %v", t3f)
	}
}

func TestReshapeTranspose(t *testing.T) {
	b := &Backend{}
	c := b.NewContext()
	defer c.Close()
	t1 := c.Arange(0, 24, 1, ml.DTypeFloat16).Reshape(c, 2, 3, 4).Transpose(c, 0, 2, 1).Contiguous(c, false)
	c.Compute(t1)
	t1f := t1.Floats()
	exp := []float32{
		0, 4, 8,
		1, 5, 9,
		2, 6, 10,
		3, 7, 11,
		12, 16, 20,
		13, 17, 21,
		14, 18, 22,
		15, 19, 23,
	}
	if !reflect.DeepEqual(t1f, exp) {
		t.Fatalf("incorrect results: %v", t1f)
	}
}

func prod(vals ...int) int {
	r := 1
	for _, v := range vals {
		r *= v
	}
	return r
}
func TestMatmul(t *testing.T) {
	// TODO create scenarios...
	b := &Backend{}
	c := b.NewContext()
	defer c.Close()
	s1 := []int{1, 3, 2, 4}
	t1 := c.Arange(0, float32(prod(s1...)), 1, ml.DTypeFloat16).Reshape(c, s1...)
	s2 := []int{4, 2}
	t2 := c.Arange(0, float32(prod(s2...)), 1, ml.DTypeFloat16).Reshape(c, s2...)
	t3 := t1.Matmul(c, t2)
	exp := []float32{
		28, 34,
		76, 98,

		124, 162,
		172, 226,

		220, 290,
		268, 354,
	}
	c.Compute(t3)
	t3f := t3.Floats()
	if !reflect.DeepEqual(t3f, exp) {
		t.Fatalf("incorrect result: %v", t3f)
	}
}

func TestRows(t *testing.T) {
	b := &Backend{}
	c := b.NewContext()
	defer c.Close()
	t1 := c.Arange(0, 12, 1, ml.DTypeFloat32).Reshape(c, 1, 4, 3)
	outputs := c.Zeros(ml.DTypeInt32, 1)
	t2 := t1.TakeAxes(c, outputs, 1)
	c.Forward(t1, t2).Compute(t1, t2)
	t.Log(t1.ToString())
	t.Log(t2.ToString())
	f := t2.Floats()
	t.Logf("Result: %v", f)
}

func TestCaching(t *testing.T) {
	// Validate the caching algorithm
	b := &Backend{}
	c := b.NewContext()
	defer c.Close()
	batchSize := 3
	headDim := 4
	numKVHeads := 2
	// Make cache twice the size of one test batch
	cells := batchSize * 2
	cellSize := numKVHeads * headDim
	shape := []int{1, numKVHeads, batchSize, headDim}
	stop := float32(1)
	for _, x := range shape {
		stop *= float32(x)
	}
	// Create the cache
	cache := c.Zeros(ml.DTypeBfloat16, cells, cellSize)
	t.Logf("Empty Cache shape%v\n"+cache.ToString(), []int{cells, cellSize})

	// Input tensor
	t1 := c.Arange(0, stop, 1, ml.DTypeBfloat16).Reshape(c, shape...)
	t.Logf("Initial Data shape%v\n"+t1.ToString(), shape)

	// Reshape to copy into the cache
	/*
		From MLX python/src/indexing.cpp mlx_scatter_args_array
		// The update shape must broadcast with indices.shape + [1] + src.shape[1:]
		auto up_shape = indices.shape();
		up_shape.insert(up_shape.end(), src.shape().begin() + 1, src.shape().end());
		up = broadcast_to(up, up_shape);
		up_shape.insert(up_shape.begin() + indices.ndim(), 1);
		up = reshape(up, up_shape);
	*/
	numRows := 3
	up := t1.Reshape(c, numRows, 1, cellSize) // The shape has to look like this for scatter to work properly
	t.Logf("Data reshaped for cache input shape%v\n"+up.ToString(), []int{batchSize, numKVHeads * headDim})

	// Simulate cells 1,3,5 are available
	indicies := []ml.Tensor{c.FromInts([]int32{1, 3, 5}, numRows)}
	t.Logf("Indicies shape%v\n"+indicies[0].ToString(), []int{numRows})
	axis := []int{0} // The 1,3,5 of the indicies are in reference to axis 0 in the cache shape
	cache.Scatter(c, indicies, up, axis)

	c.Forward(cache)
	// Cache should contain the data now
	t.Log("Cache after put\n" + cache.ToString())

	// Retrieve cache content and verify it matches
	out := cache.TakeAxes(c, indicies[0], 0).Reshape(c, shape...)
	t.Logf("Output shape%v\n"+out.ToString(), out.Shape())

	t1f := t1.Floats()
	outf := out.Floats()
	if !reflect.DeepEqual(t1f, outf) {
		t.Fatalf("mismatched in->out\n%v\n ->\n%v", t1f, outf)
	}
}

// TODO test case on RMSNorm and LayerNorm, RoPE, ScaledDotProductAttention, Take

func TestGemma3(t *testing.T) {
	// Hacky frankenstein partially GGUF partially safetensors...

	// Why is the sky blue
	inputs := []int32{2, 105, 2364, 107, 36425, 563, 506, 7217, 3730, 106, 107, 105, 4368}

	modelPath := "/Users/daniel/.ollama/models/blobs/sha256-2e1715faf889527461e76d271e827bbe03f3d22b4b86acf6146671d72eb6d11d"
	r, err := os.Open(modelPath)
	if err != nil {
		t.Fatalf("unable to open gemma3:4b-it-fp16: %s", err)
	}
	defer r.Close()

	meta, err := fsggml.Decode(r, -1)
	if err != nil {
		t.Fatalf("unable to decode: %s", err)
	}

	m, err := gemma3.New(meta.KV())
	if err != nil {
		t.Fatalf("unable to load model: %s", err)
	}
	dir := "/Users/daniel/Models/gemma-3-4b-it/"
	b := &Backend{}
	ctx := b.NewContext()
	err = b.LoadSafeTensors(dir)
	if err != nil {
		t.Fatalf("load failed: %s", err)
	}

	// More hacks...
	g3 := m.(*gemma3.Model)
	if g3.TextModel == nil {
		t.Fatal("nil text model")
	}

	// TODO Load the layers manually for now

	g3.TextModel.TokenEmbedding = &nn.Embedding{
		Weight: b.Get("language_model.model.embed_tokens.weight"),
	}
	g3.TextModel.OutputNorm = &nn.RMSNorm{
		Weight: b.Get("language_model.model.norm.weight"),
	}
	g3.TextModel.Output = &nn.Linear{
		Weight: b.Get("language_model.model.embed_tokens.weight"),
	}
	for i := range g3.TextModel.Layers {
		g3.TextModel.Layers[i].AttentionNorm = &nn.RMSNorm{
			Weight: b.Get(fmt.Sprintf("language_model.model.layers.%d.input_layernorm.weight", i)),
		}
		g3.TextModel.Layers[i].SelfAttention = &gemma3.TextSelfAttention{
			Query: &nn.Linear{
				Weight: b.Get(fmt.Sprintf("language_model.model.layers.%d.self_attn.q_proj.weight", i)),
			},
			QueryNorm: &nn.RMSNorm{
				Weight: b.Get(fmt.Sprintf("language_model.model.layers.%d.self_attn.q_norm.weight", i)),
			},
			Key: &nn.Linear{
				Weight: b.Get(fmt.Sprintf("language_model.model.layers.%d.self_attn.k_proj.weight", i)),
			},
			KeyNorm: &nn.RMSNorm{
				Weight: b.Get(fmt.Sprintf("language_model.model.layers.%d.self_attn.k_norm.weight", i)),
			},
			Value: &nn.Linear{
				Weight: b.Get(fmt.Sprintf("language_model.model.layers.%d.self_attn.v_proj.weight", i)),
			},
			Output: &nn.Linear{
				Weight: b.Get(fmt.Sprintf("language_model.model.layers.%d.self_attn.o_proj.weight", i)),
			},
		}
		g3.TextModel.Layers[i].PostAttentionNorm = &nn.RMSNorm{
			Weight: b.Get(fmt.Sprintf("language_model.model.layers.%d.post_attention_layernorm.weight", i)),
		}
		g3.TextModel.Layers[i].MLPNorm = &nn.RMSNorm{
			Weight: b.Get(fmt.Sprintf("language_model.model.layers.%d.pre_feedforward_layernorm.weight", i)),
		}
		g3.TextModel.Layers[i].MLP = &gemma3.TextMLP{
			Up: &nn.Linear{
				Weight: b.Get(fmt.Sprintf("language_model.model.layers.%d.mlp.up_proj.weight", i)),
			},
			Down: &nn.Linear{
				Weight: b.Get(fmt.Sprintf("language_model.model.layers.%d.mlp.down_proj.weight", i)),
			},
			Gate: &nn.Linear{
				Weight: b.Get(fmt.Sprintf("language_model.model.layers.%d.mlp.gate_proj.weight", i)),
			},
		}
		g3.TextModel.Layers[i].PostMLPNorm = &nn.RMSNorm{
			Weight: b.Get(fmt.Sprintf("language_model.model.layers.%d.post_feedforward_layernorm.weight", i)),
		}
	}

	// TODO multimodal not wired up yet

	// TODO try to run a forward pass with a hard-coded input
	batch := input.Batch{
		Inputs:    ctx.FromInts(inputs[:], 1, len(inputs)),
		Positions: make([]int32, len(inputs)),
		Sequences: make([]int, len(inputs)),
		Outputs:   ctx.FromInts([]int32{int32(len(inputs) - 1)}, 1),
		Offset:    0,
	}
	for i := range len(inputs) {
		batch.Positions[i] = int32(i)
	}
	offset := len(inputs)

	cache := g3.Config().Cache
	if cache != nil {
		numSlots := 1
		batchSize := 512
		numCtx := 4096

		// Note: this is inconsistent with mlx-py, but trying to be consistent with the GGML cache impl to get things working
		// cache.SetConfig(ml.CacheConfig{CachePadding: 256, MaskDType: ml.DTypeBfloat16, MaskBatchPadding: 64})
		cache.SetConfig(ml.CacheConfig{CachePadding: 0, MaskDType: ml.DTypeBfloat16, MaskBatchPadding: 0})

		cache.Init(b, ml.DTypeBfloat16, numSlots, int(numCtx), batchSize)
		err := cache.StartForward(ctx, batch, false)
		if err != nil {
			t.Fatalf("failed cache.StartForward: %s", err)
		}
	}
	opts := api.DefaultOptions()
	var grammar *sample.GrammarSampler
	sampler := sample.NewSampler(
		opts.Temperature,
		opts.TopK,
		opts.TopP,
		opts.MinP,
		opts.Seed,
		grammar,
	)

	t.Log("Starting Forward pass loop")
	pendingResponses := []string{}
	for {
		out, err := g3.Forward(ctx, batch)
		if err != nil {
			t.Fatalf("failed forward pass: %s", err)
		}
		ctx.Forward(out)
		outputs := out.Floats()
		t.Logf("finished forward pass!  length:%d", len(outputs))
		// sample a token
		// vocabSize := len(outputs) / batch.Outputs.Dim(0)

		logits := outputs // TODO subset?
		// ctx.CompareWith("/tmp/test", map[string]ml.Tensor{"l": out}, true)
		// panic("after post attention norm") //

		token, err := sampler.Sample(logits)
		if err != nil {
			t.Fatalf("unable to sample token: %s", err)
		}
		t.Logf("Sampled token: %v", token)
		if m.(model.TextProcessor).Is(token, model.SpecialEOS) {
			t.Log("hit EOS")
			break
		}
		piece, err := m.(model.TextProcessor).Decode([]int32{token})
		if err != nil {
			t.Fatalf("unable to decode token: %s", err)
		}

		pendingResponses = append(pendingResponses, piece)
		sequence := strings.Join(pendingResponses, "")
		if ok, stop := common.FindStop(sequence, opts.Stop); ok {
			t.Logf("hit stop token: %v", stop)
			break
		}
		t.Logf("RESULTS: %s", sequence)
		batch = input.Batch{
			Inputs:    ctx.FromInts([]int32{token}, 1, 1),
			Positions: make([]int32, 1),
			Sequences: make([]int, 1),
			Outputs:   ctx.FromInts([]int32{0}, 1),
			Offset:    offset,
		}
		offset++
		batch.Positions[0] = 0
		err = cache.StartForward(ctx, batch, false)
		if err != nil {
			t.Fatalf("failed cache.StartForward: %s", err)
		}

	}

}
