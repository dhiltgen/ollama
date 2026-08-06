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

const bf16ProjectionThreads = 256

type bf16ProjectionConfigKey struct {
	inputDim  int
	outputDim int
}

var (
	bf16ProjectionCUDAKernelOnce sync.Once
	bf16ProjectionCUDAKernel     C.mlx_fast_cuda_kernel
	bf16ProjectionCUDADisabled   bool

	bf16ProjectionCUDAConfigMu sync.Mutex
	bf16ProjectionCUDAConfigs  = make(map[bf16ProjectionConfigKey]C.mlx_fast_cuda_kernel_config)
)

const bf16ProjectionCUDAKernelHeader = `
#include <cuda_bf16.h>
`

const bf16ProjectionCUDAKernelSource = `
constexpr unsigned int FULL_MASK = 0xffffffff;
constexpr int WARP_WIDTH = 32;
constexpr int WARPS = 8;

int row = blockIdx.z;
int lane = threadIdx.x % WARP_WIDTH;
int warp = threadIdx.x / WARP_WIDTH;
int row_offset = row * INPUT_DIM;

const __nv_bfloat162* x2 =
    reinterpret_cast<const __nv_bfloat162*>(x);
const __nv_bfloat162* weight2 =
    reinterpret_cast<const __nv_bfloat162*>(weight + row_offset);

float sum = 0.0f;
if constexpr (INPUT_DIM % 8 == 0) {
  union alignas(16) BF16x8 {
    uint4 packed;
    __nv_bfloat162 pairs[4];
  };

  const uint4* x8 = reinterpret_cast<const uint4*>(x);
  const uint4* weight8 =
      reinterpret_cast<const uint4*>(weight + row_offset);
  for (int col8 = threadIdx.x; col8 < INPUT_DIM / 8;
       col8 += blockDim.x) {
    BF16x8 xv;
    BF16x8 wv;
    xv.packed = x8[col8];
    wv.packed = weight8[col8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float2 xf = __bfloat1622float2(xv.pairs[i]);
      float2 wf = __bfloat1622float2(wv.pairs[i]);
      sum = fmaf(wf.x, xf.x, sum);
      sum = fmaf(wf.y, xf.y, sum);
    }
  }
} else {
  for (int col2 = threadIdx.x; col2 < INPUT_DIM / 2;
       col2 += blockDim.x) {
    float2 xv = __bfloat1622float2(x2[col2]);
    float2 wv = __bfloat1622float2(weight2[col2]);
    sum = fmaf(wv.x, xv.x, sum);
    sum = fmaf(wv.y, xv.y, sum);
  }
}

#pragma unroll
for (int offset = WARP_WIDTH / 2; offset > 0; offset /= 2) {
  sum += __shfl_down_sync(FULL_MASK, sum, offset);
}

__shared__ float warp_sums[WARPS];
if (lane == 0) {
  warp_sums[warp] = sum;
}
__syncthreads();

if (warp == 0) {
  sum = lane < WARPS ? warp_sums[lane] : 0.0f;
#pragma unroll
  for (int offset = WARP_WIDTH / 2; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(FULL_MASK, sum, offset);
  }

  if (lane == 0) {
    output[row] = static_cast<InT>(sum);
  }
}
`

func initBF16ProjectionCUDAKernel() {
	var cudaAvailable C.bool
	if C.mlx_cuda_is_available(&cudaAvailable) != 0 || !bool(cudaAvailable) {
		bf16ProjectionCUDADisabled = true
		return
	}

	inputs, freeInputs, ok := cStringVector([]string{"x", "weight"})
	if !ok {
		bf16ProjectionCUDADisabled = true
		freeInputs()
		return
	}
	defer freeInputs()

	outputs, freeOutputs, ok := cStringVector([]string{"output"})
	if !ok {
		bf16ProjectionCUDADisabled = true
		freeOutputs()
		return
	}
	defer freeOutputs()

	cName := C.CString("bf16_projection")
	defer C.free(unsafe.Pointer(cName))
	cSource := C.CString(bf16ProjectionCUDAKernelSource)
	defer C.free(unsafe.Pointer(cSource))
	cHeader := C.CString(bf16ProjectionCUDAKernelHeader)
	defer C.free(unsafe.Pointer(cHeader))

	bf16ProjectionCUDAKernel = C.mlx_fast_cuda_kernel_new(
		cName,
		inputs,
		outputs,
		cSource,
		cHeader,
		C.bool(true),
		C.int(0),
	)
}

func bf16ProjectionCUDAConfig(inputDim, outputDim int) (C.mlx_fast_cuda_kernel_config, bool) {
	key := bf16ProjectionConfigKey{inputDim: inputDim, outputDim: outputDim}

	bf16ProjectionCUDAConfigMu.Lock()
	defer bf16ProjectionCUDAConfigMu.Unlock()

	if cfg, ok := bf16ProjectionCUDAConfigs[key]; ok {
		return cfg, true
	}

	cfg := C.mlx_fast_cuda_kernel_config_new()
	fail := func() (C.mlx_fast_cuda_kernel_config, bool) {
		C.mlx_fast_cuda_kernel_config_free(cfg)
		return C.mlx_fast_cuda_kernel_config{}, false
	}

	cInT := C.CString("InT")
	defer C.free(unsafe.Pointer(cInT))
	cInputDim := C.CString("INPUT_DIM")
	defer C.free(unsafe.Pointer(cInputDim))
	if C.mlx_fast_cuda_kernel_config_add_template_arg_dtype(
		cfg, cInT, C.mlx_dtype(DTypeBFloat16),
	) != 0 ||
		C.mlx_fast_cuda_kernel_config_add_template_arg_int(
			cfg, cInputDim, C.int(inputDim),
		) != 0 {
		return fail()
	}

	shape := []C.int{1, 1, C.int(outputDim)}
	if C.mlx_fast_cuda_kernel_config_add_output_arg(
		cfg,
		unsafe.SliceData(shape),
		C.size_t(len(shape)),
		C.mlx_dtype(DTypeBFloat16),
	) != 0 ||
		C.mlx_fast_cuda_kernel_config_set_grid(
			cfg, bf16ProjectionThreads, 1, C.int(outputDim),
		) != 0 ||
		C.mlx_fast_cuda_kernel_config_set_thread_group(
			cfg, bf16ProjectionThreads, 1, 1,
		) != 0 ||
		C.mlx_fast_cuda_kernel_config_set_verbose(cfg, false) != 0 {
		return fail()
	}

	bf16ProjectionCUDAConfigs[key] = cfg
	return cfg, true
}

// FastBF16Projection performs a bias-free BF16 decode projection on CUDA.
// Its production caller reserves it for Gemma's memory-bound fused MLP chain,
// where endpoint benchmarks beat the decomposed MLX/cuBLAS path. Attention
// projections retain the library path because isolated kernel gains reduced
// full-graph scheduling overlap. Unsupported inputs return ok=false.
func FastBF16Projection(x, weight *Array) (out *Array, ok bool) {
	if bf16ProjectionCUDADisabled ||
		x == nil || weight == nil ||
		x.DType() != DTypeBFloat16 ||
		weight.DType() != DTypeBFloat16 ||
		x.NumDims() != 3 ||
		x.Dim(0) != 1 || x.Dim(1) != 1 ||
		weight.NumDims() != 2 ||
		x.Dim(2) != weight.Dim(1) ||
		x.Dim(2)%2 != 0 {
		return nil, false
	}

	bf16ProjectionCUDAKernelOnce.Do(initBF16ProjectionCUDAKernel)
	if bf16ProjectionCUDADisabled {
		return nil, false
	}

	inputDim := x.Dim(2)
	outputDim := weight.Dim(0)
	cfg, ok := bf16ProjectionCUDAConfig(inputDim, outputDim)
	if !ok {
		bf16ProjectionCUDADisabled = true
		return nil, false
	}

	inputs := []C.mlx_array{x.ctx, weight.ctx}
	inVec := C.mlx_vector_array_new_data(
		unsafe.SliceData(inputs),
		C.size_t(len(inputs)),
	)
	defer C.mlx_vector_array_free(inVec)

	outVec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(outVec)
	if C.mlx_fast_cuda_kernel_apply(
		&outVec,
		bf16ProjectionCUDAKernel,
		inVec,
		cfg,
		DefaultStream().ctx,
	) != 0 || C.mlx_vector_array_size(outVec) < 1 {
		bf16ProjectionCUDADisabled = true
		return nil, false
	}

	out = New("BF16_PROJECTION_CUDA")
	if C.mlx_vector_array_get(&out.ctx, outVec, 0) != 0 {
		bf16ProjectionCUDADisabled = true
		return nil, false
	}
	return out, true
}
