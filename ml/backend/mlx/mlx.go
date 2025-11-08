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
static void* mlx_array_data_float16_asvoid(const mlx_array a) {return (void*)mlx_array_data_float16(a);}
*/
import "C"

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"os"
	"runtime"
	"runtime/debug"
	"sync"
	"unsafe"

	"github.com/ollama/ollama/fs"
	fsggml "github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/ml/nn/rope"

	"github.com/ollama/ollama/ml"
	"github.com/x448/float16"
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

func New(modelPath string, params ml.BackendParams) (ml.Backend, error) {
	r, err := os.Open(modelPath)
	if err != nil {
		return nil, err
	}
	defer r.Close()

	meta, err := fsggml.Decode(r, -1)
	if err != nil {
		return nil, err
	}

	// TODO all this loading logic will be replaced by the new model loading abstraction, including any necessary transformations
	// As currently structured, this likely causes a significant performance impact

	tensors := make(map[string]*Array, len(meta.Tensors().Items()))
	// sr := io.NewSectionReader(r, int64(meta.Tensors().Offset), n-int64(meta.Tensors().Offset))

	slog.Info("initializing MLX GPU backend")
	stream := C.mlx_default_gpu_stream_new()

	var g errgroup.Group
	var mu sync.Mutex
	vec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(vec)
	// unmutate := func(name string, shape []C.int, r C.mlx_array) error {
	// 	// TODO - is this code memory access safe, or does the delayed processing cause potential memory access after Go frees the stack?

	// 	// TODO performance: Since these operations are ~static yet cause a lot of additional nodes in the graph
	// 	// Ideally these should be applied "on the fly" at load time, so the tensor has the data ready to go.
	// 	defer C.mlx_array_free(r)

	// 	var n_head []uint64
	// 	if strings.Contains(name, "attn_q") {
	// 		n_head = meta.KV().HeadCount() // Q
	// 	} else {
	// 		n_head = meta.KV().HeadCountKV() // K
	// 	}
	// 	tmpShape := []C.int{C.int(n_head[0] /* WRONG? */), C.int(math.Floor(math.Floor(float64(shape[0]) / float64(n_head[0] /* WRONG?*/) / float64(2)))), 2, shape[1]}
	// 	var shaped C.mlx_array
	// 	C.mlx_reshape(&shaped, r, &tmpShape[0], C.size_t(len(tmpShape)), stream)
	// 	defer C.mlx_array_free(shaped)
	// 	var swapped C.mlx_array
	// 	C.mlx_swapaxes(
	// 		&swapped,
	// 		shaped,
	// 		1,
	// 		2,
	// 		stream,
	// 	)
	// 	defer C.mlx_array_free(swapped)

	// 	var reshaped C.mlx_array
	// 	C.mlx_reshape(
	// 		&reshaped,
	// 		swapped,
	// 		&shape[0],
	// 		C.size_t(len(shape)),
	// 		stream,
	// 	)
	// 	mu.Lock()
	// 	defer mu.Unlock()
	// 	C.mlx_vector_array_append_value(vec, reshaped)
	// 	tmp := &Array{a: reshaped, name: name}
	// 	tensors[name] = tmp
	// 	return nil
	// }
	for _, t := range meta.Tensors().Items() {
		g.Go(func() error {
			var b bytes.Buffer
			n, err := io.Copy(&b, io.NewSectionReader(r, int64(meta.Tensors().Offset+t.Offset), int64(t.Size())))
			if err != nil {
				return err
			}

			if n != int64(t.Size()) {
				return fmt.Errorf("expected %d bytes, got %d", t.Size(), n)
			}

			cbytes := C.CBytes(b.Bytes())
			defer C.free(cbytes)

			// Inverted
			shape := make([]C.int, len(t.Shape))
			i := len(t.Shape) - 1
			for _, dim := range t.Shape {
				shape[i] = C.int(dim)
				i--
			}
			var r C.mlx_array

			switch t.Kind {
			case 0: // GGML_TYPE_F32
				a := C.mlx_array_new_data(
					cbytes,
					&shape[0],
					C.int(len(shape)),
					C.MLX_FLOAT32,
				)
				// MLX fp32 ops are significantly slower than fp16
				C.mlx_astype(
					&r,
					a,
					C.MLX_FLOAT16,
					stream,
				)
				defer C.mlx_array_free(a)
			case 1: // GGML_TYPE_F16
				r = C.mlx_array_new_data(
					cbytes,
					&shape[0],
					C.int(len(shape)),
					C.MLX_FLOAT16,
				)
			case 30: // GGML_TYPE_BF16
				r = C.mlx_array_new_data(
					cbytes,
					&shape[0],
					C.int(len(shape)),
					C.MLX_BFLOAT16,
				)
			case 2, 8: // GGML_TYPE_Q4_0, GGML_TYPE_Q8_0
				// Note: theoretically GGML_TYPE_Q4_1 (3) should work, but spits out garbage so omitting for now
				r, err = gguf_load_quantized(cbytes, t.Name, shape, t.Kind, stream)
				if err != nil {
					panic(err.Error())
				}
			case 12, 14: // GGML_TYPE_Q4_K, GGML_TYPE_Q6_K
				// TODO any special cases?
				r, err = load_k_quantized(cbytes, t.Name, shape, t.Kind, stream)
				if err != nil {
					panic(err)
				}
			default:
				return fmt.Errorf("unsupported dtype %v", t)
			}

			// Q/K are are mutated and we need to reverse that mutation
			// TODO - this is only for llama based models and shouldn't be applied universally
			// but only applies to some backends at the moment...  maybe?
			// if meta.KV().Architecture() == "llama" && (strings.HasSuffix(t.Name, "attn_q.weight") || strings.HasSuffix(t.Name, "attn_q.bias") || strings.HasSuffix(t.Name, "attn_k.weight") || strings.HasSuffix(t.Name, "attn_k.bias")) {
			// 	return unmutate(t.Name, shape, r)
			// }
			mu.Lock()
			defer mu.Unlock()
			C.mlx_vector_array_append_value(vec, r)
			tmp := &Array{a: r, name: t.Name}
			tmp.name = t.Name
			tensors[t.Name] = tmp
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		return nil, err
	}
	C.mlx_async_eval(vec)

	return &Backend{
		meta:    meta,
		tensors: tensors,
	}, nil
}

type Backend struct {
	meta    *fsggml.GGML
	tensors map[string]*Array
}

// Config implements ml.Backend.
func (b *Backend) Config() fs.Config {
	return b.meta.KV()
}

// Get implements ml.Backend.
func (b *Backend) Get(name string) ml.Tensor {
	if a, ok := b.tensors[name]; ok {
		return a
	}

	return nil
}

func (b *Backend) BackendMemory() ml.BackendMemory {
	panic("not yet implemented")
}

func (b *Backend) BackendDevices() []ml.DeviceInfo {
	// TODO implement
	return []ml.DeviceInfo{
		ml.DeviceInfo{
			Name: "Metal0",
			DeviceID: ml.DeviceID{
				ID:      "0",
				Library: "Metal",
			},
			TotalMemory: 20 * 1024 * 1024 * 1024,
			FreeMemory:  20 * 1024 * 1024 * 1024,
			LibraryPath: []string{"foo"},
		},
	}
}
func (b *Backend) Close() {
	panic("not yet implemented")
}
func (b *Backend) Load(ctx context.Context, progress func(float32)) error {
	panic("not yet implemented")
}

func (b *Backend) NewContext() ml.Context {
	return &Context{
		stream: C.mlx_default_gpu_stream_new(),
	}
}

func (b *Backend) NewContextSize(_ int) ml.Context {
	return b.NewContext()
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

// FromFloats implements ml.Context.
func (c *Context) FromFloats(s []float32, shape ...int) ml.Tensor {
	u16s := make([]float16.Float16, len(s))
	for i := range u16s {
		u16s[i] = float16.Fromfloat32(s[i])
	}
	cshape := make([]C.int, len(shape))
	for i, dim := range shape {
		cshape[i] = C.int(dim)
	}
	return newArray(c,
		C.mlx_array_new_data(
			unsafe.Pointer(&u16s[0]),
			&cshape[0],
			C.int(len(cshape)),
			C.MLX_FLOAT16,
		),
	)
}

// FromInts implements ml.Context.
func (c *Context) FromInts(s []int32, shape ...int) ml.Tensor {
	cshape := make([]C.int, len(shape))
	for i, dim := range shape {
		cshape[i] = C.int(dim)
	}
	return newArray(c,
		C.mlx_array_new_data(
			unsafe.Pointer(&s[0]),
			&cshape[0],
			C.int(len(cshape)),
			C.MLX_INT32,
		),
	)
}

// Reserve implements ml.Context.
func (c *Context) Reserve() {
	panic("unimplemented")
}

// SetBatchSize implements ml.Context.
func (c *Context) SetBatchSize(int) {
	panic("unimplemented")
}

// Close implements ml.Context.
func (c *Context) Close() {
	// C.mlx_synchronize(c.stream) // ???
	C.mlx_stream_free(c.stream)

	c.mu.Lock()
	defer c.mu.Unlock()
	for _, a := range c.arrays {
		slog.Info("XXX freeing", "array", a)
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
func (c *Context) Forward(tensors ...ml.Tensor) ml.Context {
	vec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(vec)
	needSync := true
	sync := func() {
		if needSync {
			C.mlx_synchronize(c.stream)
			needSync = false
		}
	}

	for _, t := range tensors {
		t.(*Array).sync = sync
		C.mlx_vector_array_append_value(vec, t.(*Array).a)
	}
	C.mlx_async_eval(vec)
	return c
}

// FromFloatSlice implements ml.Context.
func (c *Context) FromFloatSlice(s []float32, shape ...int) (ml.Tensor, error) {
	u16s := make([]float16.Float16, len(s))
	for i := range u16s {
		u16s[i] = float16.Fromfloat32(s[i])
	}
	cshape := make([]C.int, len(shape))
	for i, dim := range shape {
		cshape[i] = C.int(dim)
	}
	return newArray(c,
		C.mlx_array_new_data(
			unsafe.Pointer(&u16s[0]),
			&cshape[0],
			C.int(len(cshape)),
			C.MLX_FLOAT16,
		),
	), nil
}

// FromIntSlice implements ml.Context.
// func (c *Context) FromIntSlice(s []int32, shape ...int) (ml.Tensor, error) {
// 	cshape := make([]C.int, len(shape))
// 	for i, dim := range shape {
// 		cshape[i] = C.int(dim)
// 	}
// 	return newArray(c,
// 		C.mlx_array_new_data(
// 			unsafe.Pointer(&s[0]),
// 			&cshape[0],
// 			C.int(len(cshape)),
// 			C.MLX_INT32,
// 		),
// 	), nil
// }

func (c *Context) Empty(dtype ml.DType, shape ...int) ml.Tensor {
	// TODO more efficient impl?
	return c.Zeros(dtype, shape...)
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
		// TODO should we just force this to fp16?
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

func (c *Context) MaxGraphNodes() int {
	// TODO actually wire up correctly
	return 9999
}

func (c *Context) Input() ml.Context {
	return c
}

func (c *Context) Output() ml.Context {
	return c
}

func (c *Context) Layer(_ int) ml.Context {
	return c
}

func (c *Context) Arange(start, stop, step float32, dtype ml.DType) ml.Tensor {
	panic("NOT YET IMPLEMENTED")
}
func (c *Context) ComputeWithNotify(func(), ...ml.Tensor) {
	panic("NOT YET IMPLEMENTED")
}
func (c *Context) FromBytes(dtype ml.DType, s []byte, shape ...int) ml.Tensor {
	panic("NOT YET IMPLEMENTED")
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
	// fmt.Println(a.name)
	// fmt.Println(C.GoString(s))

	dims := int(C.mlx_array_ndim(a.a))
	strides := make([]int, dims)
	for i := range strides {
		strides[i] = int(C.stride(a.a, (C.int)(i)))
	}

	return slog.GroupValue(
		slog.String("name", a.name),
		slog.String("type", a.TypeString()),
		slog.Any("shape", a.Shape()),
		slog.Any("strides", strides),
		// slog.String("values", C.GoString(s)),
	)
}

func (a *Array) Neg(ctx ml.Context) ml.Tensor {
	panic("NOT YET IMPLEMENTED")
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
func (a *Array) Sub(ctx ml.Context, a2 ml.Tensor) ml.Tensor {
	panic("NOT YET IMPLEMENTED")
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
func (a *Array) Contiguous(ctx ml.Context, shape ...int) ml.Tensor {
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
// GGML API
// 	input: [N, IC, IH, IW]
// 	weight: [OC，IC, KH, KW]
// 	result: [N, OC, OH, OW]
//
// MLX:
//  input: (N, KH, KW, C_in)
//  weight: (C_out, IH, IW, C_in)
//  result: XXX

func (weight *Array) Conv2D(ctx ml.Context, input ml.Tensor, s0 int, s1 int, p0 int, p1 int, d0 int, d1 int) ml.Tensor {
	var inputM C.mlx_array
	var weightM C.mlx_array
	C.mlx_transpose(
		&inputM,
		input.(*Array).a,
		&[]C.int{0, 2, 3, 1}[0],
		4,
		ctx.(*Context).stream,
	)
	defer C.mlx_array_free(inputM)
	C.mlx_transpose(
		&weightM,
		weight.a,
		&[]C.int{0, 2, 3, 1}[0],
		4,
		ctx.(*Context).stream,
	)
	defer C.mlx_array_free(weightM)
	var r C.mlx_array
	C.mlx_conv2d(
		&r,
		inputM,
		weightM,
		C.int(s0),
		C.int(s1),
		C.int(p0),
		C.int(p1),
		C.int(d0),
		C.int(d1),
		1,
		ctx.(*Context).stream,
	)
	defer C.mlx_array_free(r)
	var a C.mlx_array
	C.mlx_transpose(
		&a,
		r,
		&[]C.int{0, 3, 1, 2}[0],
		4,
		ctx.(*Context).stream,
	)
	return newArray(ctx.(*Context), a)
}

func (a *Array) Conv3D(ctx ml.Context, weight ml.Tensor, c, s0, s1, s2, p0, p1, p2, d0, d1, d2 int) ml.Tensor {
	panic("NOT YET IMPLEMENTED")
}

func (a *Array) IM2Col(ctx ml.Context, weight ml.Tensor, s0, s1, p0, p1, d0, d1 int) ml.Tensor {
	panic("NOT YET IMPLEMENTED")
}
func (a *Array) TopK(ctx ml.Context, k int) ml.Tensor {
	panic("NOT YET IMPLEMENTED")
}
func (a *Array) Argsort(ctx ml.Context) ml.Tensor {
	panic("NOT YET IMPLEMENTED")
}
func (a *Array) Mean(ctx ml.Context) ml.Tensor {
	panic("NOT YET IMPLEMENTED")
}
func (a *Array) Variance(ctx ml.Context) ml.Tensor {
	panic("NOT YET IMPLEMENTED")
}
func (a *Array) Stddev(ctx ml.Context) ml.Tensor {
	panic("NOT YET IMPLEMENTED")
}
func (a *Array) Sqr(ctx ml.Context) ml.Tensor {
	panic("NOT YET IMPLEMENTED")
}
func (a *Array) Sqrt(ctx ml.Context) ml.Tensor {
	panic("NOT YET IMPLEMENTED")
}
func (a *Array) Clamp(ctx ml.Context, min, max float32) ml.Tensor {
	panic("NOT YET IMPLEMENTED")
}
func (a *Array) Cast(ctx ml.Context, dtype ml.DType) ml.Tensor {
	panic("NOT YET IMPLEMENTED")
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
func (a *Array) Duplicate(ctx ml.Context) ml.Tensor {
	panic("NOT YET IMPLEMENTED")
}
func (c *Array) FromBytes(s []byte) {
	panic("NOT YET IMPLEMENTED")
}
func (c *Array) FromFloats([]float32) {
	panic("NOT YET IMPLEMENTED")
}
func (c *Array) FromInts([]int32) {
	panic("NOT YET IMPLEMENTED")
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
	l := (int)(C.mlx_array_size(a.a))

	switch C.mlx_array_dtype(a.a) {
	case C.MLX_BFLOAT16:
		panic("bfloat16 not yet implemented")
	case C.MLX_FLOAT16:
		data := C.mlx_array_data_float16_asvoid(a.a)
		if data == nil {
			panic("nil data, wasn't eval'd")
		}
		u16s := unsafe.Slice((*uint16)(data), l)
		f32s := make([]float32, len(u16s))
		for i := range u16s {
			f32s[i] = float16.Frombits(u16s[i]).Float32()
		}
		return f32s
	case C.MLX_FLOAT32:
		data := C.mlx_array_data_float32(a.a)
		if data == nil {
			panic("nil data, wasn't eval'd")
		}
		f32s := unsafe.Slice((*float32)(data), l)
		return f32s
	default:
		panic(fmt.Sprintf("unsupported dtype for Floats: %d", C.mlx_array_dtype(a.a)))
	}
}

// GELU implements ml.Tensor.
func (a *Array) GELU(ctx ml.Context, up ...ml.Tensor) ml.Tensor {
	// TODO precise vs fast, and compile
	// x * mx.sigmoid(1.702 * x)
	u16s := []float16.Float16{float16.Fromfloat32(1.702)}
	cshape := []C.int{1}
	f := C.mlx_array_new_data(unsafe.Pointer(&u16s[0]), &cshape[0], 1, C.MLX_FLOAT16)
	defer C.mlx_array_free(f)
	var r1, r2, r3 C.mlx_array
	C.mlx_multiply(&r1, a.a, f, ctx.(*Context).stream)
	defer C.mlx_array_free(r1)
	C.mlx_sigmoid(&r2, r1, ctx.(*Context).stream)
	defer C.mlx_array_free(r2)
	C.mlx_multiply(&r3, a.a, r2, ctx.(*Context).stream)
	return newArray(ctx.(*Context), r3)
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

func (a *Array) Div(ctx ml.Context, a2 ml.Tensor) ml.Tensor {
	panic("NOT YET IMPLEMENTED")
}

// Mulmat implements ml.Tensor.
func (a *Array) Mulmat(ctx ml.Context, a2 ml.Tensor) ml.Tensor {
	var r C.mlx_array
	slog.Info("MLX Mulmat Input", "a", a, "a2", a2)

	var ar C.mlx_array
	s := make([]C.int, C.mlx_array_ndim(a.a))
	for i := range s {
		s[i] = C.int(i)
	}
	if len(s) < 2 {
		panic("unexpected shape for Mulmat")
	}
	// TODO - does this actually need conditional logic?
	if C.mlx_array_dim(a2.(*Array).a, C.int(C.mlx_array_ndim(a2.(*Array).a))-C.int(1)) != C.mlx_array_dim(a.a, C.int(len(s)-2)) {
		slog.Info("XXX Doing transpose")
		s[len(s)-2], s[len(s)-1] = s[len(s)-1], s[len(s)-2]
		C.mlx_transpose(&ar, a.a, &s[0], C.size_t(len(s)), ctx.(*Context).stream)
		defer C.mlx_array_free(ar)
	} else {
		slog.Info("XXX leaving as is")
		// TODO panic here to see if this is ever hit
		ar = a.a
	}
	slog.Info("MLX A @ B", "A", &Array{a: a2.(*Array).a}, "B", &Array{a: ar})

	C.mlx_matmul(&r,
		a2.(*Array).a,
		ar,
		ctx.(*Context).stream)
	return newArray(ctx.(*Context), r)
}

func (a *Array) MulmatFullPrec(ctx ml.Context, a2 ml.Tensor) ml.Tensor {
	return a.Mulmat(ctx, a2)
}

func (a *Array) MulmatID(ctx ml.Context, t2, ids ml.Tensor) ml.Tensor {
	// TODO implement
	panic("NOT YET IMPLEMENTED")
}
func (a *Array) AddID(ctx ml.Context, t2, ids ml.Tensor) ml.Tensor {
	// TODO implement
	panic("NOT YET IMPLEMENTED")
}
func (a *Array) L2Norm(ctx ml.Context, eps float32) ml.Tensor {
	// TODO implement
	panic("NOT YET IMPLEMENTED")
}
func (a *Array) Sin(ctx ml.Context) ml.Tensor {
	// TODO implement
	panic("NOT YET IMPLEMENTED")
}
func (a *Array) Cos(ctx ml.Context) ml.Tensor {
	// TODO implement
	panic("NOT YET IMPLEMENTED")
}
func (a *Array) Repeat(ctx ml.Context, dim, n int) ml.Tensor {
	// TODO implement
	panic("NOT YET IMPLEMENTED")
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
	C.mlx_reshape(&r, a.a, &cshape[0], C.size_t(len(cshape)), ctx.(*Context).stream)
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
// 	RoPE(ctx ml.Context, positionIDs ml.Tensor, dim int, base, scale float32, options ...func(*rope.Options)) ml.Tensor

func (a *Array) RoPE(ctx ml.Context, positionIDs ml.Tensor, dim int, base, scale float32, options ...func(*rope.Options)) ml.Tensor {
	// 	ctx ml.Context,
	// 	positionIDs ml.Tensor, // Unused in MLX
	// 	ropeFactors ml.Tensor, // Unused in MLX
	// 	freqs ml.Tensor,
	// 	dim uint32,
	// 	ropeType uint32,
	// 	base float32,
	// 	scale float32,
	// ) ml.Tensor {
	a = a.Reshape(ctx, append([]int{1}, a.Shape()...)...).Permute(ctx, 0, 2, 1, 3).(*Array)

	// TODO this probably isn't right...

	offset := 0
	var r C.mlx_array
	var b C.mlx_optional_float
	var _freqs C.mlx_array
	if base == 0 {
		base = 10000
	}
	// if freqs == nil || len(freqs.Shape()) == 0 {
	b.value = C.float(base)
	b.has_value = true
	// } else {
	// 	_freqs = freqs.(*Array).a
	// }

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
func (a *Array) SILU(ctx ml.Context, up ...ml.Tensor) ml.Tensor {
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
func (a *Array) RELU(ctx ml.Context, up ...ml.Tensor) ml.Tensor {
	panic("NOT YET IMPLEMENTED")
}
func (a *Array) Sigmoid(ctx ml.Context) ml.Tensor {
	panic("NOT YET IMPLEMENTED")
}
func (a *Array) SILUAlphaLimit(ctx ml.Context, up ml.Tensor, alpha, limit float32) ml.Tensor {
	panic("NOT YET IMPLEMENTED")
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
func (a *Array) SumRows(ctx ml.Context) ml.Tensor {
	// TODO implement
	panic("NOT YET IMPLEMENTED")
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
		false, // TODO - precise?
		ctx.(*Context).stream,
	)
	return newArray(ctx.(*Context), r)
}

// Stack implements ml.Tensor.
func (a *Array) Stack(ctx ml.Context, dim int, s ...ml.Tensor) ml.Tensor {
	vec := C.mlx_vector_array_new_value(a.a)
	defer C.mlx_vector_array_free(vec)
	for _, t := range s {
		C.mlx_vector_array_append_value(vec, t.(*Array).a)
	}
	var r C.mlx_array
	C.mlx_concatenate(
		&r,
		vec,
		C.int(dim), // TODO - this isn't right -
		// MLX error: [concatenate] Invalid axis (2) passed to concatenate for array with shape (1280). at /Users/daniel/code/ollama/build/_deps/mlx-c-src/mlx/c/ops.cpp:635
		ctx.(*Context).stream,
	)
	return newArray(ctx.(*Context), r)
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
func (a *Array) View(ctx ml.Context, offset int, shape ...int) ml.Tensor {
	// if len(stride)+1 != len(shape) {
	// 	panic(fmt.Sprintf("malformed view request: shape=%v stride=%v", shape, stride))
	// }

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

	queries := t.Reshape(ctx, append([]int{1}, t.Shape()...)...).Permute(ctx, 0, 2, 1, 3)
	keys = keys.Reshape(ctx, append([]int{1}, keys.Shape()...)...).Permute(ctx, 0, 2, 1, 3)
	values = values.Reshape(ctx, append([]int{1}, values.Shape()...)...).Permute(ctx, 0, 2, 1, 3)

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

func (t Array) AvgPool2D(ctx ml.Context, k, s int, p float32) ml.Tensor {
	panic("NOT YET IMPLEMENTED")
}

func (t Array) Set(ctx ml.Context, t2 ml.Tensor, offset int, strides ...int) ml.Tensor {
	panic("NOT YET IMPLEMENTED")
}

func (ctx *Context) SliceUpdate(target, source ml.Tensor, start, stop, strides []int) {
	t := target.(*Array)
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
	var r C.mlx_array
	C.mlx_slice_update(
		&r,
		t.a,
		source.(*Array).a,
		(*C.int)(unsafe.Pointer(&cStart[0])),
		C.size_t(len(cStart)),
		(*C.int)(unsafe.Pointer(&cStop[0])),
		C.size_t(len(cStop)),
		(*C.int)(unsafe.Pointer(&cStrides[0])),
		C.size_t(len(cStrides)),
		ctx.stream,
	)
	// Release the old array and replace with the new one to ensure the same underlying buffer is used
	C.mlx_array_free(t.a)
	t.a = r
}

func (ctx *Context) Slice(source ml.Tensor, start, stop, strides []int) ml.Tensor {
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
	var r C.mlx_array
	C.mlx_slice(
		&r,
		source.(*Array).a,
		(*C.int)(unsafe.Pointer(&cStart[0])),
		C.size_t(len(cStart)),
		(*C.int)(unsafe.Pointer(&cStop[0])),
		C.size_t(len(cStop)),
		(*C.int)(unsafe.Pointer(&cStrides[0])),
		C.size_t(len(cStrides)),
		ctx.stream,
	)
	return newArray(ctx, r)
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

func (a *Array) TypeString() string {
	switch C.mlx_array_dtype(a.a) {
	case C.MLX_BOOL:
		return "bool"
	case C.MLX_UINT8:
		return "uint8"
	case C.MLX_UINT16:
		return "uint16"
	case C.MLX_UINT32:
		return "uint32"
	case C.MLX_UINT64:
		return "uint64"
	case C.MLX_INT8:
		return "int8"
	case C.MLX_INT16:
		return "int16"
	case C.MLX_INT32:
		return "int32"
	case C.MLX_INT64:
		return "int64"
	case C.MLX_FLOAT16:
		return "float16"
	case C.MLX_FLOAT32:
		return "float32"
	case C.MLX_BFLOAT16:
		return "bfloat16"
	case C.MLX_COMPLEX64:
		return "complex64"
	default:
		return "unknown"
	}
}
