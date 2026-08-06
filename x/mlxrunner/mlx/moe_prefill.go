package mlx

/*
#include <stdlib.h>
#include "generated.h"
*/
import "C"

import (
	"sync"
	"unsafe"
)

const (
	sortedMoETopK              = 8
	sortedMoECombineHiddenSize = 2816
	sortedMoECopyThreads       = 128
	sortedMoEThreads           = 256
)

var (
	sortedMoERowCopyCUDAKernelOnce sync.Once
	sortedMoERowCopyCUDAKernel     C.mlx_fast_cuda_kernel
	sortedMoERowCopyCUDADisabled   bool

	sortedMoERowCopyCUDAConfigMu sync.Mutex
	sortedMoERowCopyCUDAConfigs  = make(map[sortedMoERowCopyCUDAConfigKey]C.mlx_fast_cuda_kernel_config)

	sortedMoECombineCUDAKernelOnce sync.Once
	sortedMoECombineCUDAKernel     C.mlx_fast_cuda_kernel
	sortedMoECombineCUDADisabled   bool
)

type sortedMoERowCopyCUDAConfigKey struct {
	assignments       int
	hiddenSize        int
	divideIndexByTopK bool
}

const sortedMoERowCopyCUDAKernelSource = `
constexpr int TOP_K = 8;
constexpr int VECTOR_BYTES = sizeof(uint4);
constexpr int ROW_VECTORS =
    HIDDEN_SIZE * sizeof(InT) / VECTOR_BYTES;

int assignment = blockIdx.z;
int source_row = static_cast<int>(order[assignment]);
if constexpr (DivideIndexByTopK) {
  source_row /= TOP_K;
}
const uint4* source = reinterpret_cast<const uint4*>(
    x + source_row * HIDDEN_SIZE);
uint4* destination = reinterpret_cast<uint4*>(
    out + assignment * HIDDEN_SIZE);

for (int column = threadIdx.x;
     column < ROW_VECTORS;
     column += blockDim.x) {
  destination[column] = source[column];
}
`

const sortedMoECombineCUDAKernelSource = `
constexpr int TOP_K = 8;
constexpr int HIDDEN_SIZE = 2816;

int token = blockIdx.z;
int column = threadIdx.x;

for (int hidden = column; hidden < HIDDEN_SIZE; hidden += blockDim.x) {
  InT sum = static_cast<InT>(0.0f);
#pragma unroll
  for (int expert = 0; expert < TOP_K; ++expert) {
    int assignment = token * TOP_K + expert;
    long long sorted_row =
        static_cast<long long>(inv_order[assignment]);
    long long expert_index =
        static_cast<long long>(expert_indices[assignment]);

    InT scaled = down[sorted_row * HIDDEN_SIZE + hidden];
    if constexpr (ApplyExpertScale) {
      scaled = static_cast<InT>(
          static_cast<float>(scaled) *
          static_cast<float>(expert_scales[expert_index]));
    }
    InT weighted = static_cast<InT>(
        static_cast<float>(scaled) *
        static_cast<float>(scores[assignment]));
    sum = static_cast<InT>(
        static_cast<float>(sum) + static_cast<float>(weighted));
  }
  out[token * HIDDEN_SIZE + hidden] = sum;
}
`

func initSortedMoERowCopyCUDAKernel() {
	var cudaAvailable C.bool
	if C.mlx_cuda_is_available(&cudaAvailable) != 0 || !bool(cudaAvailable) {
		sortedMoERowCopyCUDADisabled = true
		return
	}

	inputs, freeInputs, ok := cStringVector([]string{"x", "order"})
	if !ok {
		sortedMoERowCopyCUDADisabled = true
		freeInputs()
		return
	}
	defer freeInputs()

	outputs, freeOutputs, ok := cStringVector([]string{"out"})
	if !ok {
		sortedMoERowCopyCUDADisabled = true
		freeOutputs()
		return
	}
	defer freeOutputs()

	cName := C.CString("sorted_moe_row_copy_top8_dynamic_hidden")
	defer C.free(unsafe.Pointer(cName))
	cSource := C.CString(sortedMoERowCopyCUDAKernelSource)
	defer C.free(unsafe.Pointer(cSource))
	cHeader := C.CString("")
	defer C.free(unsafe.Pointer(cHeader))

	sortedMoERowCopyCUDAKernel = C.mlx_fast_cuda_kernel_new(
		cName,
		inputs,
		outputs,
		cSource,
		cHeader,
		C.bool(true),
		C.int(0),
	)
}

func sortedMoERowCopyCUDAConfig(
	assignments, hiddenSize int,
	divideIndexByTopK bool,
) (C.mlx_fast_cuda_kernel_config, bool, bool) {
	cacheable := assignments <= sortedMoETopK*sortedMoETopK
	key := sortedMoERowCopyCUDAConfigKey{
		assignments:       assignments,
		hiddenSize:        hiddenSize,
		divideIndexByTopK: divideIndexByTopK,
	}
	if cacheable {
		sortedMoERowCopyCUDAConfigMu.Lock()
		defer sortedMoERowCopyCUDAConfigMu.Unlock()
		if cfg, ok := sortedMoERowCopyCUDAConfigs[key]; ok {
			return cfg, true, true
		}
	}

	cfg := C.mlx_fast_cuda_kernel_config_new()
	fail := func() (C.mlx_fast_cuda_kernel_config, bool, bool) {
		C.mlx_fast_cuda_kernel_config_free(cfg)
		return C.mlx_fast_cuda_kernel_config{}, false, false
	}

	for _, arg := range []struct {
		name  string
		dtype DType
	}{
		{name: "InT", dtype: DTypeBFloat16},
		{name: "OrderT", dtype: DTypeUint32},
	} {
		cName := C.CString(arg.name)
		rc := C.mlx_fast_cuda_kernel_config_add_template_arg_dtype(
			cfg,
			cName,
			C.mlx_dtype(arg.dtype),
		)
		C.free(unsafe.Pointer(cName))
		if rc != 0 {
			return fail()
		}
	}

	cHiddenSize := C.CString("HIDDEN_SIZE")
	defer C.free(unsafe.Pointer(cHiddenSize))
	if C.mlx_fast_cuda_kernel_config_add_template_arg_int(
		cfg,
		cHiddenSize,
		C.int(hiddenSize),
	) != 0 {
		return fail()
	}

	cDivideIndexByTopK := C.CString("DivideIndexByTopK")
	defer C.free(unsafe.Pointer(cDivideIndexByTopK))
	if C.mlx_fast_cuda_kernel_config_add_template_arg_bool(
		cfg,
		cDivideIndexByTopK,
		C.bool(divideIndexByTopK),
	) != 0 {
		return fail()
	}

	shape := []C.int{
		C.int(assignments),
		1,
		1,
		C.int(hiddenSize),
	}
	if C.mlx_fast_cuda_kernel_config_add_output_arg(
		cfg,
		unsafe.SliceData(shape),
		C.size_t(len(shape)),
		C.mlx_dtype(DTypeBFloat16),
	) != 0 {
		return fail()
	}
	if C.mlx_fast_cuda_kernel_config_set_grid(
		cfg,
		sortedMoECopyThreads,
		1,
		C.int(assignments),
	) != 0 {
		return fail()
	}
	if C.mlx_fast_cuda_kernel_config_set_thread_group(
		cfg,
		sortedMoECopyThreads,
		1,
		1,
	) != 0 {
		return fail()
	}
	if C.mlx_fast_cuda_kernel_config_set_verbose(cfg, false) != 0 {
		return fail()
	}

	if cacheable {
		// Decode admits at most eight rows, so retaining these configs is bounded.
		sortedMoERowCopyCUDAConfigs[key] = cfg
	}
	return cfg, cacheable, true
}

// fastSortedMoERowCopy applies the shared CUDA row-permutation kernel used by
// sorted MoE dispatch and unsort operations.
func fastSortedMoERowCopy(
	x, order *Array,
	divideIndexByTopK bool,
) (out *Array, ok bool) {
	if sortedMoERowCopyCUDADisabled ||
		x == nil || order == nil ||
		x.DType() != DTypeBFloat16 ||
		order.DType() != DTypeUint32 ||
		x.NumDims() != 4 ||
		x.Dim(1) != 1 ||
		x.Dim(2) != 1 ||
		x.Dim(3)%8 != 0 ||
		order.NumDims() != 1 ||
		order.Size() < sortedMoETopK ||
		order.Size()%sortedMoETopK != 0 {
		return nil, false
	}

	sortedMoERowCopyCUDAKernelOnce.Do(initSortedMoERowCopyCUDAKernel)
	if sortedMoERowCopyCUDADisabled {
		return nil, false
	}

	cfg, cached, ok := sortedMoERowCopyCUDAConfig(
		order.Size(),
		x.Dim(3),
		divideIndexByTopK,
	)
	if !ok {
		sortedMoERowCopyCUDADisabled = true
		return nil, false
	}
	if !cached {
		defer C.mlx_fast_cuda_kernel_config_free(cfg)
	}

	inputs := []C.mlx_array{x.ctx, order.ctx}
	inVec := C.mlx_vector_array_new_data(
		unsafe.SliceData(inputs),
		C.size_t(len(inputs)),
	)
	defer C.mlx_vector_array_free(inVec)

	outVec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(outVec)
	if C.mlx_fast_cuda_kernel_apply(
		&outVec,
		sortedMoERowCopyCUDAKernel,
		inVec,
		cfg,
		DefaultStream().ctx,
	) != 0 || C.mlx_vector_array_size(outVec) < 1 {
		sortedMoERowCopyCUDADisabled = true
		return nil, false
	}

	out = New("SORTED_MOE_DISPATCH_CUDA")
	if C.mlx_vector_array_get(&out.ctx, outVec, 0) != 0 {
		sortedMoERowCopyCUDADisabled = true
		return nil, false
	}
	return out, true
}

// FastSortedMoEDispatch duplicates activation rows into expert-sorted order
// using vectorized row copies. It replaces the generic indexed gather used by
// MoE prefill on CUDA; no CUDA library primitive combines the index transform
// with these full-row copies.
func FastSortedMoEDispatch(x, order *Array) (out *Array, ok bool) {
	if x == nil || order == nil ||
		x.NumDims() != 4 ||
		x.Dim(0)*sortedMoETopK != order.Size() {
		return nil, false
	}
	return fastSortedMoERowCopy(x, order, true)
}

// FastSortedMoEUnsort restores expert outputs to token-major order with
// vectorized row copies, replacing the generic indexed gather on CUDA.
func FastSortedMoEUnsort(x, invOrder *Array) (out *Array, ok bool) {
	if x == nil || invOrder == nil ||
		x.NumDims() != 4 ||
		x.Size() != invOrder.Size()*x.Dim(3) {
		return nil, false
	}
	return fastSortedMoERowCopy(x, invOrder, false)
}

func initSortedMoECombineCUDAKernel() {
	var cudaAvailable C.bool
	if C.mlx_cuda_is_available(&cudaAvailable) != 0 || !bool(cudaAvailable) {
		sortedMoECombineCUDADisabled = true
		return
	}

	inputs, freeInputs, ok := cStringVector([]string{
		"down",
		"inv_order",
		"scores",
		"expert_indices",
		"expert_scales",
	})
	if !ok {
		sortedMoECombineCUDADisabled = true
		freeInputs()
		return
	}
	defer freeInputs()

	outputs, freeOutputs, ok := cStringVector([]string{"out"})
	if !ok {
		sortedMoECombineCUDADisabled = true
		freeOutputs()
		return
	}
	defer freeOutputs()

	cName := C.CString("fused_sorted_moe_combine_top8_2816")
	defer C.free(unsafe.Pointer(cName))
	cSource := C.CString(sortedMoECombineCUDAKernelSource)
	defer C.free(unsafe.Pointer(cSource))
	cHeader := C.CString("")
	defer C.free(unsafe.Pointer(cHeader))

	sortedMoECombineCUDAKernel = C.mlx_fast_cuda_kernel_new(
		cName,
		inputs,
		outputs,
		cSource,
		cHeader,
		C.bool(true),
		C.int(0),
	)
}

func sortedMoECombineCUDAConfig(rows int, applyExpertScale bool) (C.mlx_fast_cuda_kernel_config, bool) {
	// The output shape follows the prompt, so keep this config scoped to one apply.
	cfg := C.mlx_fast_cuda_kernel_config_new()
	fail := func() (C.mlx_fast_cuda_kernel_config, bool) {
		C.mlx_fast_cuda_kernel_config_free(cfg)
		return C.mlx_fast_cuda_kernel_config{}, false
	}

	for _, arg := range []struct {
		name  string
		dtype DType
	}{
		{name: "InT", dtype: DTypeBFloat16},
		{name: "OrderT", dtype: DTypeUint32},
		{name: "ExpertIndexT", dtype: DTypeUint32},
	} {
		cName := C.CString(arg.name)
		rc := C.mlx_fast_cuda_kernel_config_add_template_arg_dtype(
			cfg,
			cName,
			C.mlx_dtype(arg.dtype),
		)
		C.free(unsafe.Pointer(cName))
		if rc != 0 {
			return fail()
		}
	}

	cApplyExpertScale := C.CString("ApplyExpertScale")
	defer C.free(unsafe.Pointer(cApplyExpertScale))
	if C.mlx_fast_cuda_kernel_config_add_template_arg_bool(
		cfg,
		cApplyExpertScale,
		C.bool(applyExpertScale),
	) != 0 {
		return fail()
	}

	shape := []C.int{C.int(rows), sortedMoECombineHiddenSize}
	if C.mlx_fast_cuda_kernel_config_add_output_arg(
		cfg,
		unsafe.SliceData(shape),
		C.size_t(len(shape)),
		C.mlx_dtype(DTypeBFloat16),
	) != 0 {
		return fail()
	}
	if C.mlx_fast_cuda_kernel_config_set_grid(
		cfg,
		sortedMoEThreads,
		1,
		C.int(rows),
	) != 0 {
		return fail()
	}
	if C.mlx_fast_cuda_kernel_config_set_thread_group(
		cfg,
		sortedMoEThreads,
		1,
		1,
	) != 0 {
		return fail()
	}
	if C.mlx_fast_cuda_kernel_config_set_verbose(cfg, false) != 0 {
		return fail()
	}

	return cfg, true
}

// FastSortedMoECombine fuses the sorted-output permutation, expert scaling,
// dispatch weighting, and top-k reduction used by Gemma4 prefill on CUDA. The
// library-level alternative materializes a gather and multiple pointwise and
// reduction nodes, so there is no equivalent single high-level CUDA call.
func FastSortedMoECombine(
	down, invOrder, scores, expertIndices, expertScales *Array,
	applyExpertScale bool,
) (out *Array, ok bool) {
	if sortedMoECombineCUDADisabled ||
		down == nil || invOrder == nil || scores == nil ||
		expertIndices == nil || expertScales == nil ||
		down.DType() != DTypeBFloat16 ||
		invOrder.DType() != DTypeUint32 ||
		scores.DType() != DTypeBFloat16 ||
		expertIndices.DType() != DTypeUint32 ||
		expertScales.DType() != DTypeBFloat16 ||
		scores.NumDims() != 2 ||
		scores.Dim(1) != sortedMoETopK ||
		expertScales.NumDims() != 1 ||
		expertScales.Size() != 128 {
		return nil, false
	}

	rows := scores.Dim(0)
	assignments := rows * sortedMoETopK
	if rows < 64 ||
		down.Size() != assignments*sortedMoECombineHiddenSize ||
		invOrder.Size() != assignments ||
		expertIndices.Size() != assignments {
		return nil, false
	}

	sortedMoECombineCUDAKernelOnce.Do(initSortedMoECombineCUDAKernel)
	if sortedMoECombineCUDADisabled {
		return nil, false
	}

	cfg, ok := sortedMoECombineCUDAConfig(rows, applyExpertScale)
	if !ok {
		sortedMoECombineCUDADisabled = true
		return nil, false
	}
	defer C.mlx_fast_cuda_kernel_config_free(cfg)

	inputs := []C.mlx_array{
		down.ctx,
		invOrder.ctx,
		scores.ctx,
		expertIndices.ctx,
		expertScales.ctx,
	}
	inVec := C.mlx_vector_array_new_data(
		unsafe.SliceData(inputs),
		C.size_t(len(inputs)),
	)
	defer C.mlx_vector_array_free(inVec)

	outVec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(outVec)
	if C.mlx_fast_cuda_kernel_apply(
		&outVec,
		sortedMoECombineCUDAKernel,
		inVec,
		cfg,
		DefaultStream().ctx,
	) != 0 || C.mlx_vector_array_size(outVec) < 1 {
		sortedMoECombineCUDADisabled = true
		return nil, false
	}

	out = New("SORTED_MOE_COMBINE_CUDA")
	if C.mlx_vector_array_get(&out.ctx, outVec, 0) != 0 {
		sortedMoECombineCUDADisabled = true
		return nil, false
	}
	return out, true
}
