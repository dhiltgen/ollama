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
	moeFinalHiddenSize = 2816
	moeFinalReads      = 8
	moeFinalThreads    = moeFinalHiddenSize / moeFinalReads
)

var (
	moeFinalCUDAKernelOnce sync.Once
	moeFinalCUDAKernel     C.mlx_fast_cuda_kernel
	moeFinalCUDADisabled   bool

	moeFinalCUDAConfigMu    sync.Mutex
	moeFinalCUDAConfigs     [9][9]C.mlx_fast_cuda_kernel_config
	moeFinalCUDAConfigValid [9][9]bool
)

const moeFinalCUDAKernelHeader = `
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
`

const moeFinalCUDAKernelSource = `
namespace cg = cooperative_groups;

constexpr int HIDDEN_SIZE = 2816;
constexpr int N_READS = 8;

auto block = cg::this_thread_block();
auto warp = cg::tiled_partition<32>(block);
int row = blockIdx.z;
int thread = threadIdx.x;
int offset = row * HIDDEN_SIZE + thread * N_READS;

__shared__ float warp_sums[32];
InT combined[N_READS];
float normalizer = 0.0f;

#pragma unroll
for (int i = 0; i < N_READS; ++i) {
  combined[i] = static_cast<InT>(
      static_cast<float>(mlp[offset + i]) +
      static_cast<float>(moe[offset + i]));
  float value = static_cast<float>(combined[i]);
  normalizer += value * value;
}

normalizer = cg::reduce(warp, normalizer, cg::plus<float>{});
if (warp.thread_rank() == 0) {
  warp_sums[warp.meta_group_rank()] = normalizer;
}
block.sync();

normalizer = warp.thread_rank() < warp.meta_group_size()
    ? warp_sums[warp.thread_rank()]
    : 0.0f;
normalizer = cg::reduce(warp, normalizer, cg::plus<float>{});
normalizer = rsqrtf(normalizer / HIDDEN_SIZE + 1e-6f);

InT layer_scale_value = layer_scale[0];
#pragma unroll
for (int i = 0; i < N_READS; ++i) {
  int index = offset + i;
  float normalized = static_cast<float>(combined[i]) * normalizer;
  InT rounded_normalized =
      norm_scale[index - row * HIDDEN_SIZE] *
      static_cast<InT>(normalized);
  InT rounded_residual = static_cast<InT>(
      static_cast<float>(residual[index]) +
      static_cast<float>(rounded_normalized));
  out[index] = layer_scale_value * rounded_residual;
}
`

func initMoEFinalCUDAKernel() {
	var cudaAvailable C.bool
	if C.mlx_cuda_is_available(&cudaAvailable) != 0 || !bool(cudaAvailable) {
		moeFinalCUDADisabled = true
		return
	}

	inputs, freeInputs, ok := cStringVector([]string{
		"residual",
		"mlp",
		"moe",
		"norm_scale",
		"layer_scale",
	})
	if !ok {
		moeFinalCUDADisabled = true
		freeInputs()
		return
	}
	defer freeInputs()

	outputs, freeOutputs, ok := cStringVector([]string{"out"})
	if !ok {
		moeFinalCUDADisabled = true
		freeOutputs()
		return
	}
	defer freeOutputs()

	cName := C.CString("fused_moe_final_residual_2816")
	defer C.free(unsafe.Pointer(cName))
	cSource := C.CString(moeFinalCUDAKernelSource)
	defer C.free(unsafe.Pointer(cSource))
	cHeader := C.CString(moeFinalCUDAKernelHeader)
	defer C.free(unsafe.Pointer(cHeader))

	moeFinalCUDAKernel = C.mlx_fast_cuda_kernel_new(
		cName,
		inputs,
		outputs,
		cSource,
		cHeader,
		C.bool(true),
		C.int(0),
	)
}

func moeFinalCUDAConfig(B, L int) (C.mlx_fast_cuda_kernel_config, bool) {
	moeFinalCUDAConfigMu.Lock()
	defer moeFinalCUDAConfigMu.Unlock()

	if moeFinalCUDAConfigValid[B][L] {
		return moeFinalCUDAConfigs[B][L], true
	}

	cfg := C.mlx_fast_cuda_kernel_config_new()
	fail := func() (C.mlx_fast_cuda_kernel_config, bool) {
		C.mlx_fast_cuda_kernel_config_free(cfg)
		return C.mlx_fast_cuda_kernel_config{}, false
	}

	cInT := C.CString("InT")
	defer C.free(unsafe.Pointer(cInT))
	if C.mlx_fast_cuda_kernel_config_add_template_arg_dtype(
		cfg, cInT, C.mlx_dtype(DTypeBFloat16),
	) != 0 {
		return fail()
	}

	shape := []C.int{C.int(B), C.int(L), moeFinalHiddenSize}
	if C.mlx_fast_cuda_kernel_config_add_output_arg(
		cfg,
		unsafe.SliceData(shape),
		C.size_t(len(shape)),
		C.mlx_dtype(DTypeBFloat16),
	) != 0 {
		return fail()
	}
	if C.mlx_fast_cuda_kernel_config_set_grid(
		cfg, moeFinalThreads, 1, C.int(B*L),
	) != 0 {
		return fail()
	}
	if C.mlx_fast_cuda_kernel_config_set_thread_group(
		cfg, moeFinalThreads, 1, 1,
	) != 0 {
		return fail()
	}
	if C.mlx_fast_cuda_kernel_config_set_verbose(cfg, false) != 0 {
		return fail()
	}

	moeFinalCUDAConfigs[B][L] = cfg
	moeFinalCUDAConfigValid[B][L] = true
	return cfg, true
}

// FastMoEFinalResidual fuses the final MoE branch combination, RMSNorm, and
// scaled residual update for the small CUDA decode shape used by Gemma4 26B.
// NVIDIA libraries expose the pieces but no operation preserving this fused
// branch and BF16-rounding contract.
func FastMoEFinalResidual(
	residual, mlp, moe, normScale, layerScale *Array,
	eps float32,
) (out *Array, ok bool) {
	if moeFinalCUDADisabled ||
		residual == nil || mlp == nil || moe == nil ||
		normScale == nil || layerScale == nil ||
		eps != 1e-6 ||
		residual.DType() != DTypeBFloat16 ||
		mlp.DType() != DTypeBFloat16 ||
		moe.DType() != DTypeBFloat16 ||
		normScale.DType() != DTypeBFloat16 ||
		layerScale.DType() != DTypeBFloat16 ||
		residual.NumDims() != 3 ||
		mlp.NumDims() != 3 ||
		moe.NumDims() != 3 ||
		normScale.NumDims() != 1 ||
		normScale.Dim(0) != moeFinalHiddenSize ||
		layerScale.Size() != 1 {
		return nil, false
	}

	B, L := residual.Dim(0), residual.Dim(1)
	if B < 1 || B > 8 || L < 1 || L > 8 || B*L > 8 ||
		residual.Dim(2) != moeFinalHiddenSize ||
		mlp.Dim(0) != B || mlp.Dim(1) != L ||
		mlp.Dim(2) != moeFinalHiddenSize ||
		moe.Dim(0) != B || moe.Dim(1) != L ||
		moe.Dim(2) != moeFinalHiddenSize {
		return nil, false
	}

	moeFinalCUDAKernelOnce.Do(initMoEFinalCUDAKernel)
	if moeFinalCUDADisabled {
		return nil, false
	}

	cfg, ok := moeFinalCUDAConfig(B, L)
	if !ok {
		moeFinalCUDADisabled = true
		return nil, false
	}

	inputs := []C.mlx_array{
		residual.ctx,
		mlp.ctx,
		moe.ctx,
		normScale.ctx,
		layerScale.ctx,
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
		moeFinalCUDAKernel,
		inVec,
		cfg,
		DefaultStream().ctx,
	) != 0 || C.mlx_vector_array_size(outVec) < 1 {
		moeFinalCUDADisabled = true
		return nil, false
	}

	out = New("MOE_FINAL_CUDA")
	if C.mlx_vector_array_get(&out.ctx, outVec, 0) != 0 {
		moeFinalCUDADisabled = true
		return nil, false
	}
	return out, true
}
