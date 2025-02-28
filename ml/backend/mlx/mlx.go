package mlx

/*
#cgo CPPFLAGS: -I${SRCDIR}/../../../build/_deps/mlx-c-src
#cgo LDFLAGS: -L${SRCDIR}/../../../build/lib/ollama/ -lmlxc -lmlx
#cgo LDFLAGS: -framework Accelerate
#cgo LDFLAGS: -Wl,-rpath,${SRCDIR}/../../../build/lib/ollama/
#include <stdlib.h>
#include "mlx/c/array.h"
#include "mlx/c/fast.h"
#include "mlx/c/ops.h"
#include "mlx/c/stream.h"
#include "mlx/c/transforms.h"
#include "mlx/c/error.h"
static inline size_t stride(const mlx_array a, int i) {return mlx_array_strides(a)[i];}

extern void goStackTrace();
static void error_handler(const char *msg, void* data) {
	fprintf(stderr, "MLX error: %s\n", msg);
	goStackTrace();
	exit(-1);
}
static void set_error_handler() {mlx_set_error_handler(&error_handler, NULL, NULL);}
*/
import "C"

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"mime"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"runtime/debug"
	"strings"
	"sync"
	"unsafe"

	"github.com/ollama/ollama/envconfig"
	fs "github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/ml"
	"golang.org/x/sync/errgroup"
)

func init() {
	ml.RegisterBackend("mlx", New)
	C.set_error_handler()
}

//export goStackTrace
func goStackTrace() {
	debug.PrintStack()
}

// HACK - defined in the server package but can't import for cycles...
type Manifest struct {
	SchemaVersion int     `json:"schemaVersion"`
	MediaType     string  `json:"mediaType"`
	Config        Layer   `json:"config"`
	Layers        []Layer `json:"layers"`

	filepath string
	fi       os.FileInfo
	digest   string
}
type Layer struct {
	MediaType string `json:"mediaType"`
	Digest    string `json:"digest"`
	Size      int64  `json:"size"`
	From      string `json:"from,omitempty"`
	status    string

	// Extracted from the media type for easy access
	Dtype string
	Shape []int
	Name  string
}

func (l Layer) ParseMediaType() (string, map[string]string, error) {
	return mime.ParseMediaType(l.MediaType)
}

func GetBlobsPath(digest string) (string, error) {
	// only accept actual sha256 digests
	pattern := "^sha256[:-][0-9a-fA-F]{64}$"
	re := regexp.MustCompile(pattern)

	if digest != "" && !re.MatchString(digest) {
		return "", errors.New("invalid digest format")
	}

	digest = strings.ReplaceAll(digest, ":", "-")
	path := filepath.Join(envconfig.Models(), "blobs", digest)
	dirPath := filepath.Dir(path)
	if digest == "" {
		dirPath = path
	}

	if err := os.MkdirAll(dirPath, 0o755); err != nil {
		return "", err
	}

	return path, nil
}

func adjustTensorName(ggufName string) string {
	r := strings.ReplaceAll(ggufName, "blk.", "model.layers.")
	r = strings.ReplaceAll(r, ".attn_k.", ".self_attn.k_proj.")
	r = strings.ReplaceAll(r, ".attn_q.", ".self_attn.q_proj.")
	r = strings.ReplaceAll(r, ".attn_v.", ".self_attn.v_proj.")
	r = strings.ReplaceAll(r, ".attn_output.", ".self_attn.o_proj.")
	r = strings.ReplaceAll(r, ".ffn_gate.", ".mlp.gate_proj.")
	r = strings.ReplaceAll(r, ".ffn_up.", ".mlp.up_proj.")
	r = strings.ReplaceAll(r, ".ffn_down.", ".mlp.down_proj.")
	r = strings.ReplaceAll(r, ".ffn_norm.", ".post_attention_layernorm.")
	r = strings.ReplaceAll(r, ".attn_norm.", ".input_layernorm.")
	if strings.HasPrefix(r, "output_norm.") {
		r = strings.ReplaceAll(r, "output_norm.", "model.norm.")
	}
	if strings.HasPrefix(r, "token_embd.") {
		r = strings.ReplaceAll(r, "token_embd.", "model.embed_tokens.")
	}
	if strings.HasPrefix(r, "output.") {
		r = strings.ReplaceAll(r, "output.", "lm_head.")
	}
	return r
}

func New(r *os.File, params ml.BackendParams) (ml.Backend, error) {
	// TODO - temporary hacks to try to load unmodified row-order tensor data
	sha256sum := sha256.New()
	var manifest Manifest
	if err := json.NewDecoder(io.TeeReader(r, sha256sum)).Decode(&manifest); err != nil {
		return nil, err
	}
	var ggufPath string

	for _, layer := range manifest.Layers {
		filename, err := GetBlobsPath(layer.Digest)
		if err != nil {
			return nil, err
		}
		switch layer.MediaType {
		case "application/vnd.ollama.image.model":
			ggufPath = filename
		}
	}

	gr, err := os.Open(ggufPath)
	if err != nil {
		return nil, err
	}

	meta, _, err := fs.Decode(gr, -1)
	if err != nil {
		return nil, err
	}
	// slog.Info("XXX Parsed meta", "meta", meta)
	// slog.Info("XXX Parsed meta", "kv", meta.KV())
	// for i, t := range meta.Tensors().Items() {
	// 	slog.Info("XXX tensor", "i", i, "t", t)

	// }
	// slog.Info("XXX Parsed meta", "tensors", meta.Tensors())

	// TODO all this loading logic will be replaced by the new model loading abstraction, including any necessary transformations
	// As currently structured, this likely causes a significant performance impact

	tensors := make(map[string]*Array, len(meta.Tensors().Items()))
	// sr := io.NewSectionReader(r, int64(meta.Tensors().Offset), n-int64(meta.Tensors().Offset))

	slog.Info("initializing MLX GPU backend")
	// stream := C.mlx_default_gpu_stream_new()

	var g errgroup.Group
	var mu sync.Mutex
	vec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(vec)
	for _, t := range meta.Tensors().Items() {
		g.Go(func() error {
			if t.Name == "rope_freqs.weight" {
				slog.Info("Skipping rope_freqs.weight")
				return nil
			}
			// Optimize this...
			var layer *Layer
			for _, l := range manifest.Layers {
				_, params, err := l.ParseMediaType()
				if err != nil {
					slog.Error("failed to parse layer", "error", err)
				}
				if params["name"] == adjustTensorName(t.Name) {
					l.Name = params["name"]
					err = json.Unmarshal([]byte(params["shape"]), &l.Shape)
					if err != nil {
						return err
					}
					l.Dtype = params["dtype"]
					layer = &l
					break
				}
			}
			if layer == nil {
				slog.Info("Unable to find tensor", "name", t.Name, "adjusted", adjustTensorName(t.Name))
				panic("missing tensor name mapping!")
			}
			layerFilename, err := GetBlobsPath(layer.Digest)
			if err != nil {
				return err
			}

			lfp, err := os.Open(layerFilename)
			if err != nil {
				return err
			}

			var b bytes.Buffer
			n, err := io.Copy(&b, lfp)
			if err != nil {
				return err
			}
			if n != int64(layer.Size) {
				return fmt.Errorf("expected %d bytes, got %d", t.Size(), n)
			}

			data := b.Bytes()
			shape := make([]C.int, len(layer.Shape))
			for i := range layer.Shape {
				shape[i] = C.int(layer.Shape[i])
			}

			// // TODO Quantization types
			// // ref: https://github.com/ml-explore/mlx/blob/main/mlx/io/gguf_quants.cpp
			var dtype C.mlx_dtype
			var dsize int
			switch layer.Dtype {
			case "BF16":
				dtype = C.MLX_BFLOAT16
				dsize = 2
			// case 1:
			// 	dtype = C.MLX_FLOAT16
			default:
				return fmt.Errorf("unsupported dtype %s", layer.Dtype)
			}
			if len(layer.Shape) == 2 && !strings.HasPrefix(layer.Name, "model.embed_tokens.") {
				tdata := make([]byte, len(data))
				t := 0
				for j := range layer.Shape[1] {
					for i := range layer.Shape[0] {
						offset := i*dsize*layer.Shape[1] + j*dsize
						tdata[t] = data[offset]
						t++
						tdata[t] = data[offset+1]
						t++
					}
				}
				data = tdata
				rshape := make([]C.int, len(shape))
				i := 0
				for r := len(shape) - 1; r >= 0; r-- {
					rshape[r] = shape[i]
					i++
				}
				shape = rshape
			}
			cbytes := C.CBytes(data)
			defer C.free(cbytes)

			a := C.mlx_array_new_data(
				cbytes,
				(*C.int)(&shape[0]),
				C.int(len(shape)),
				dtype,
			)
			tmp := &Array{a: a, name: t.Name}
			// tmp.name = layer.Name // safetensor naming
			tmp.name = t.Name // GGUF naming
			slog.Info("MLX Loaded", "tensor", tmp)
			mu.Lock()
			defer mu.Unlock()
			tensors[t.Name] = tmp
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		return nil, err
	}
	// panic("Got to end of load")
	// C.mlx_async_eval(vec)

	return &Backend{
		meta:    meta,
		tensors: tensors,
	}, nil
}

type Backend struct {
	meta    *fs.GGML
	tensors map[string]*Array
}

// Config implements ml.Backend.
func (b *Backend) Config() ml.Config {
	return b.meta.KV()
}

// Get implements ml.Backend.
func (b *Backend) Get(name string) ml.Tensor {
	if a, ok := b.tensors[name]; ok {
		return a
	}

	return nil
}

func (b *Backend) NewContext() ml.Context {
	return &Context{
		stream: C.mlx_default_gpu_stream_new(),
	}
}

func (b *Backend) SystemInfo() string {
	// TODO implement this, maybe from metal.h calls...
	return ""
}

type Context struct {
	stream C.mlx_stream

	mu     sync.Mutex
	arrays []C.mlx_array // TODO should we do some bookkeeping to ensure none of these Arrays are still lingering?
}

// Close implements ml.Context.
func (c *Context) Close() {
	// C.mlx_synchronize(c.stream) // ???
	C.mlx_stream_free(c.stream)

	c.mu.Lock()
	defer c.mu.Unlock()
	for _, a := range c.arrays {
		C.mlx_array_free(a)
	}
}

// Compute implements ml.Context.
func (c *Context) Compute(tensors ...ml.Tensor) {
	// TODO - for the zero tensor case this feels like it might not be correct...
	needSync := true
	sync := func() {
		if needSync {
			C.mlx_synchronize(c.stream)
			needSync = false
		}
	}

	vec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(vec)
	for _, t := range tensors {
		C.mlx_vector_array_append_value(vec, t.(*Array).a)
		t.(*Array).sync = sync
	}
	C.mlx_async_eval(vec)
}

// Forward implements ml.Context.
func (c *Context) Forward(t ml.Tensor) {
	t.(*Array).sync = func() {
		C.mlx_synchronize(c.stream)
	}
	vec := C.mlx_vector_array_new_value(t.(*Array).a)
	defer C.mlx_vector_array_free(vec)
	C.mlx_async_eval(vec)
}

// FromFloatSlice implements ml.Context.
func (c *Context) FromFloatSlice(s []float32, shape ...int) (ml.Tensor, error) {
	cshape := make([]C.int, len(shape))
	for i, dim := range shape {
		cshape[i] = C.int(dim)
	}
	return newArray(c,
		C.mlx_array_new_data(
			unsafe.Pointer(&s[0]),
			(*C.int)(&cshape[0]),
			C.int(len(cshape)),
			C.MLX_FLOAT32,
		),
	), nil
}

// FromIntSlice implements ml.Context.
func (c *Context) FromIntSlice(s []int32, shape ...int) (ml.Tensor, error) {
	cshape := make([]C.int, len(shape))
	for i, dim := range shape {
		cshape[i] = C.int(dim)
	}
	return newArray(c,
		C.mlx_array_new_data(
			unsafe.Pointer(&s[0]),
			(*C.int)(&cshape[0]),
			C.int(len(cshape)),
			C.MLX_INT32,
		),
	), nil
}

// Zeros implements ml.Context.
func (c *Context) Zeros(dtype ml.DType, shape ...int) ml.Tensor {
	if len(shape) < 1 || len(shape) > 4 {
		panic("unsupported number of dimensions")
	}
	for _, dim := range shape {
		if dim < 1 {
			panic("invalid shape")
		}
	}
	var dt C.mlx_dtype
	switch dtype {
	case ml.DTypeF32:
		dt = C.MLX_FLOAT32
	case ml.DTypeF16:
		dt = C.MLX_FLOAT16
	case ml.DTypeI32:
		dt = C.MLX_INT32
	default:
		panic(fmt.Sprintf("unsupported dtype %d", dtype))
	}
	sh := make([]C.int, len(shape))
	for i, s := range shape {
		sh[i] = (C.int)(s)
	}

	var r C.mlx_array
	C.mlx_zeros(
		&r,
		&sh[0],
		(C.size_t)(len(sh)),
		dt,
		c.stream,
	)
	return newArray(c, r)
}

func (c *Context) MaxTensors() int {
	// TODO actually wire up correctly
	return 9999
}

type Array struct {
	name string
	a    C.mlx_array

	sync func()
}

func newArray(ctx *Context, a C.mlx_array) *Array {
	// TODO measure impact and if this slows things down, make it conditional on some debugging flag at load time
	var name string
	_, f, l, ok := runtime.Caller(2)
	if ok {
		name = fmt.Sprintf("%s:%d", f, l)
	}

	t := &Array{
		name: name,
		a:    a,
	}
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	ctx.arrays = append(ctx.arrays, a)
	return t
}

func (a *Array) LogValue() slog.Value {
	// TODO this forces eval on every log message - find a pattern to make this configurable to aid in debugging
	// str := C.mlx_string_new()
	// C.mlx_array_tostring(&str, a.a)
	// s := C.mlx_string_data(str)
	// defer C.mlx_string_free(str)
	// fmt.Println(C.GoString(s))
	dims := int(C.mlx_array_ndim(a.a))
	strides := make([]int, dims)
	for i := range strides {
		strides[i] = int(C.stride(a.a, (C.int)(i)))
	}

	return slog.GroupValue(
		slog.String("name", a.name),
		slog.Any("shape", a.Shape()),
		slog.Any("strides", strides),
		// slog.String("values", C.GoString(s)),
	)
}

// Add implements ml.Tensor.
func (a *Array) Add(ctx ml.Context, a2 ml.Tensor) ml.Tensor {
	var r C.mlx_array
	C.mlx_add(
		&r,
		a.a,
		a2.(*Array).a,
		ctx.(*Context).stream,
	)
	return newArray(ctx.(*Context), r)
}

// Bytes implements ml.Tensor.
func (a *Array) Bytes() []byte {
	if a.sync != nil {
		a.sync()
	}

	l := (int)(C.mlx_array_nbytes(a.a))
	data := C.mlx_array_data_uint8(a.a)
	if data == nil {
		return nil
	}
	return unsafe.Slice((*byte)(data), l)
}

// Concat implements ml.Tensor.
func (a *Array) Concat(ctx ml.Context, a2 ml.Tensor, dim int) ml.Tensor {
	panic("unimplemented")
}

// Contiguous implements ml.Tensor.
func (a *Array) Contiguous(ctx ml.Context) ml.Tensor {
	var r C.mlx_array
	C.mlx_contiguous(
		&r,
		a.a,
		true, // TODO ???
		ctx.(*Context).stream,
	)
	return newArray(ctx.(*Context), r)
}

// Conv2D implements ml.Tensor.
func (a *Array) Conv2D(ctx ml.Context, weight ml.Tensor, s0 int, s1 int, p0 int, p1 int, d0 int, d1 int) ml.Tensor {
	panic("unimplemented")
}

// Copy implements ml.Tensor.
func (a *Array) Copy(ctx ml.Context, a2 ml.Tensor) ml.Tensor {
	C.mlx_copy(
		&a2.(*Array).a,
		a.a,
		ctx.(*Context).stream,
	)
	// TODO - view?
	return newArray(ctx.(*Context), a2.(*Array).a)
}

// DType implements ml.Tensor.
func (a *Array) DType() ml.DType {
	switch C.mlx_array_dtype(a.a) {
	// case	C.MLX_BOOL:
	// case	C.MLX_UINT8:
	// case	C.MLX_UINT16:
	// case	C.MLX_UINT32:
	// case	C.MLX_UINT64:
	// case	C.MLX_INT8:
	// case	C.MLX_INT16:
	case C.MLX_INT32:
		return ml.DTypeI32
	// case	C.MLX_INT64:
	case C.MLX_FLOAT16:
		return ml.DTypeF16
	case C.MLX_FLOAT32:
		return ml.DTypeF32
	default:
		panic("unsupported dtype")
	}
}

// Dim implements ml.Tensor.
func (a *Array) Dim(n int) int {
	return int(C.mlx_array_dim(a.a, C.int(n)))
}

// Floats implements ml.Tensor.
func (a *Array) Floats() []float32 {
	if a.sync != nil {
		a.sync()
	}

	f32sLen := (int)(C.mlx_array_size(a.a))
	data := C.mlx_array_data_float32(a.a)
	if data == nil {
		panic("nil data, wasn't eval'd")
	}
	f32s := unsafe.Slice((*float32)(data), f32sLen)
	return f32s
}

// GELU implements ml.Tensor.
func (a *Array) GELU(ctx ml.Context) ml.Tensor {
	panic("unimplemented")
}

// Mul implements ml.Tensor.
func (a *Array) Mul(ctx ml.Context, a2 ml.Tensor) ml.Tensor {
	var r C.mlx_array
	C.mlx_multiply(
		&r,
		a.a,
		a2.(*Array).a,
		ctx.(*Context).stream,
	)
	return newArray(ctx.(*Context), r)
}

// Mulmat implements ml.Tensor.
func (a *Array) Mulmat(ctx ml.Context, a2 ml.Tensor) ml.Tensor {
	var r C.mlx_array
	s := a.Shape()
	strides := make([]int, len(s))
	for i := range s {
		strides[i] = a.Stride(i)
	}
	sb := a2.Shape()
	stridesb := make([]int, len(sb))
	for i := range sb {
		stridesb[i] = a2.Stride(i)
	}
	C.mlx_matmul(&r,
		a2.(*Array).a,
		a.a,
		ctx.(*Context).stream)
	return newArray(ctx.(*Context), r)
}

func (a *Array) MulmatFullPrec(ctx ml.Context, a2 ml.Tensor) ml.Tensor {
	return a.Mulmat(ctx, a2)
}

// LayerNorm implements ml.Tensor.
func (a *Array) LayerNorm(ctx ml.Context, w, b ml.Tensor, eps float32) ml.Tensor {
	var r C.mlx_array
	C.mlx_fast_layer_norm(
		&r,
		a.a,
		w.(*Array).a,
		b.(*Array).a,
		C.float(eps),
		ctx.(*Context).stream,
	)
	return newArray(ctx.(*Context), r)
}

// Pad implements ml.Tensor.
func (a *Array) Pad(ctx ml.Context, shape ...int) ml.Tensor {
	panic("unimplemented")
}

// Permute implements ml.Tensor.
func (a *Array) Permute(ctx ml.Context, shape ...int) ml.Tensor {
	ndim := min(C.mlx_array_ndim(a.a), C.size_t(len(shape)))
	var r C.mlx_array
	sh := make([]C.int, ndim)
	for i := range ndim {
		sh[i] = (C.int)(shape[i])
		if int(sh[i]) >= int(ndim) {
			slog.Error("Permute error", "tensor", a, "shape", shape)
			panic("invalid pemute call")
		}
	}
	C.mlx_transpose(
		&r,
		a.a,
		&sh[0],
		ndim,
		ctx.(*Context).stream,
	)
	return newArray(ctx.(*Context), r)
}

// RMSNorm implements ml.Tensor.
func (a *Array) RMSNorm(ctx ml.Context, w ml.Tensor, eps float32) ml.Tensor {
	var r C.mlx_array
	C.mlx_fast_rms_norm(
		&r,
		a.a,
		w.(*Array).a,
		C.float(eps),
		ctx.(*Context).stream,
	)
	return newArray(ctx.(*Context), r)
}

// Reshape implements ml.Tensor.
func (a *Array) Reshape(ctx ml.Context, shape ...int) ml.Tensor {
	cshape := make([]C.int, len(shape))
	for i, dim := range shape {
		cshape[i] = C.int(dim)
	}
	var r C.mlx_array
	C.mlx_reshape(&r, a.a, (*C.int)(&cshape[0]), C.size_t(len(cshape)), ctx.(*Context).stream)
	return newArray(ctx.(*Context), r)
}

/* MLX breadcrumb for Fast RoPE
a (array) – Input array.
dims (int) – The feature dimensions to be rotated. If the input feature is larger than dims then the rest is left unchanged.
traditional (bool) – If set to True choose the traditional implementation which rotates consecutive dimensions.
base (float, optional) – The base used to compute angular frequency for each dimension in the positional encodings. Exactly one of base and freqs must be None.
scale (float) – The scale used to scale the positions.
offset (int or array) – The position offset to start at.
freqs (array, optional) – Optional frequencies to use with RoPE. If set, the base parameter must be None. Default: None.
*/

// Rope implements ml.Tensor.
func (a *Array) RoPE(
	ctx ml.Context,
	positionIDs ml.Tensor, // Unused in MLX
	ropeFactors ml.Tensor, // Unused in MLX
	freqs ml.Tensor,
	dim uint32,
	base float32,
	scale float32,
) ml.Tensor {
	a = a.Reshape(ctx, append([]int{1}, a.Shape()...)...).Permute(ctx, 0, 2, 1, 3).(*Array)
	// TODO figure out how to get offset wired up
	offset := 0
	var r C.mlx_array
	var b C.mlx_optional_float
	var _freqs C.mlx_array
	if base == 0 {
		base = 10000
	}
	if freqs == nil || len(freqs.Shape()) == 0 {
		b.value = C.float(base)
		b.has_value = true
	} else {
		_freqs = freqs.(*Array).a
	}

	C.mlx_fast_rope(
		&r,
		a.a,
		C.int(dim),
		false, // traditional=false
		b,
		C.float(scale),
		C.int(offset),
		_freqs,
		ctx.(*Context).stream,
	)

	res := newArray(ctx.(*Context), r).Permute(ctx, 0, 2, 1, 3)
	return res.Reshape(ctx, res.Shape()[1:]...)
}

// Rows implements ml.Tensor.
func (a *Array) Rows(ctx ml.Context, a2 ml.Tensor) ml.Tensor {
	var r C.mlx_array

	// HACK!
	// If the indicies is greater than 2 dimensions, assume axis 1
	var axis C.int
	if C.mlx_array_ndim(a2.(*Array).a) > 1 {
		axis = 1
	} else {
		axis = 0
	}
	C.mlx_take(&r, a.a, a2.(*Array).a, axis, ctx.(*Context).stream)
	return newArray(ctx.(*Context), r)
}

// SILU implements ml.Tensor.
func (a *Array) SILU(ctx ml.Context) ml.Tensor {
	var sig C.mlx_array
	C.mlx_sigmoid(
		&sig,
		a.a,
		ctx.(*Context).stream,
	)
	var r C.mlx_array
	C.mlx_multiply(
		&r,
		a.a,
		sig,
		ctx.(*Context).stream,
	)
	return newArray(ctx.(*Context), r)
}

// Scale implements ml.Tensor.
func (a *Array) Scale(ctx ml.Context, s float64) ml.Tensor {
	scale := C.mlx_array_new_float(C.float(s))
	var r C.mlx_array
	C.mlx_multiply(
		&r,
		a.a,
		scale,
		ctx.(*Context).stream,
	)
	return newArray(ctx.(*Context), r)
}

// Shape implements ml.Tensor.
func (a *Array) Shape() []int {
	shape := make([]int, C.mlx_array_ndim(a.a))
	for i := range shape {
		shape[i] = int(C.mlx_array_dim(a.a, C.int(i)))
	}

	return shape
}

// Softmax implements ml.Tensor.
func (a *Array) Softmax(ctx ml.Context) ml.Tensor {
	var r C.mlx_array
	axes := []C.int{-1}
	C.mlx_softmax(
		&r,
		a.a,
		&axes[0],
		C.size_t(len(axes)),
		false, //TODO - precise?
		ctx.(*Context).stream,
	)
	return newArray(ctx.(*Context), r)
}

// Stack implements ml.Tensor.
func (a *Array) Stack(ctx ml.Context, dim int, s ...ml.Tensor) ml.Tensor {
	panic("unimplemented")
}

// Stride implements ml.Tensor.
func (a *Array) Stride(n int) int {
	return (int)(C.stride(a.a, (C.int)(n)))
}

// Tanh implements ml.Tensor.
func (a *Array) Tanh(ctx ml.Context) ml.Tensor {
	panic("unimplemented")
}

// Unpad implements ml.Tensor.
func (a *Array) Unpad(ctx ml.Context, shape ...int) ml.Tensor {
	panic("unimplemented")
}

// View implements ml.Tensor.
func (a *Array) View(ctx ml.Context, offset int, shape []int, stride []int) ml.Tensor {
	if len(stride)+1 != len(shape) {
		panic(fmt.Sprintf("malformed view request: shape=%v stride=%v", shape, stride))
	}

	var r C.mlx_array
	var sh []C.int
	var st []C.size_t
	var stp *C.size_t
	switch len(shape) {
	case 1:
		sh = []C.int{
			C.int(shape[0]),
		}
	case 2:
		sh = []C.int{
			C.int(shape[0]),
			C.int(shape[1]),
		}
		// st = []C.size_t{
		// 	C.size_t(stride[0]),
		// }
	case 3:
		sh = []C.int{
			C.int(shape[0]),
			C.int(shape[1]),
			C.int(shape[2]),
		}
		// st = []C.size_t{
		// 	C.size_t(stride[0]),
		// 	C.size_t(stride[1]),
		// }
	case 4:
		sh = []C.int{
			C.int(shape[0]),
			C.int(shape[1]),
			C.int(shape[2]),
			C.int(shape[3]),
		}
		// st = []C.size_t{
		// 	C.size_t(stride[0]),
		// 	C.size_t(stride[1]),
		// 	C.size_t(stride[2]),
		// }
	default:
		panic("unsupported number of dimensions")
	}
	if len(st) > 0 {
		stp = (*C.size_t)(unsafe.Pointer(&st[0]))
	}
	C.mlx_as_strided(
		&r,
		a.a,
		(*C.int)(unsafe.Pointer(&sh[0])),
		C.size_t(len(sh)),
		stp,
		C.size_t(len(st)),
		C.size_t(offset),
		ctx.(*Context).stream,
	)

	return newArray(ctx.(*Context), r)
}

func (t *Array) ScaledDotProductAttention(ctx ml.Context, keys, values, mask ml.Tensor, scale float64) ml.Tensor {
	var r C.mlx_array
	var m C.mlx_array
	if mask != nil {
		m = mask.(*Array).a
	}

	queries := t.Reshape(ctx, append([]int{1}, t.Shape()...)...)
	keys = keys.Reshape(ctx, append([]int{1}, keys.Shape()...)...)
	values = values.Reshape(ctx, append([]int{1}, values.Shape()...)...).Permute(ctx, 0, 1, 3, 2)

	C.mlx_fast_scaled_dot_product_attention(
		&r,
		queries.(*Array).a,
		keys.(*Array).a,
		values.(*Array).a,
		C.float(scale),
		m,
		C.mlx_optional_int{},
		ctx.(*Context).stream,
	)
	res := newArray(ctx.(*Context), r)
	return res.Reshape(ctx, append([]int{}, res.Shape()[1:]...)...).Permute(ctx, 1, 0, 2, 3)
}

func (ctx *Context) SliceUpdate(target, source ml.Tensor, start, stop, strides []int) {
	cStart := make([]C.int, len(start))
	for i := range start {
		cStart[i] = C.int(start[i])
	}
	cStop := make([]C.int, len(stop))
	for i := range stop {
		cStop[i] = C.int(stop[i])
	}
	cStrides := make([]C.int, len(strides))
	for i := range strides {
		cStrides[i] = C.int(strides[i])
	}
	C.mlx_slice_update(
		&target.(*Array).a,
		target.(*Array).a,
		source.(*Array).a,
		(*C.int)(unsafe.Pointer(&cStart[0])),
		C.size_t(len(cStart)),
		(*C.int)(unsafe.Pointer(&cStop[0])),
		C.size_t(len(cStop)),
		(*C.int)(unsafe.Pointer(&cStrides[0])),
		C.size_t(len(cStrides)),
		ctx.stream,
	)
}

// TODO remove this before merging - temporary debugging aid
func (c *Context) Abort(t ml.Tensor) {
	// str := C.mlx_string_new()
	// C.mlx_array_tostring(&str, t.(*Array).a)
	// s := C.mlx_string_data(str)
	// defer C.mlx_string_free(str)
	debug.PrintStack()
	// fmt.Printf("shape%v\n", t.Shape())
	// fmt.Println(C.GoString(s))

	c.Compute(t)
	f32 := t.Floats()

	filename := os.Getenv("OLLAMA_BACKEND") + ".json"
	slog.Info("Writing tensors to", "filename", filename)
	f, err := os.Create(filename)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	encoder := json.NewEncoder(f)
	err = encoder.Encode(f32)
	if err != nil {
		panic(err)
	}

	os.Exit(1)
}
