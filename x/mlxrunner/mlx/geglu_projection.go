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

const bf16GeGLUProjectionThreads = 256

type bf16GeGLUProjectionConfigKey struct {
	inputDim  int
	outputDim int
}

var (
	bf16GeGLUProjectionCUDAKernelOnce sync.Once
	bf16GeGLUProjectionCUDAKernel     C.mlx_fast_cuda_kernel
	bf16GeGLUProjectionCUDADisabled   bool

	bf16GeGLUProjectionCUDAConfigMu sync.Mutex
	bf16GeGLUProjectionCUDAConfigs  = make(map[bf16GeGLUProjectionConfigKey]C.mlx_fast_cuda_kernel_config)
)

const bf16GeGLUProjectionCUDAKernelHeader = `
#include <cuda_bf16.h>
`

const bf16GeGLUProjectionCUDAKernelSource = `
constexpr unsigned int FULL_MASK = 0xffffffff;
constexpr int WARP_WIDTH = 32;

int lane = threadIdx.x % WARP_WIDTH;
int warp = threadIdx.x / WARP_WIDTH;
int row = blockIdx.z;
int row_offset = row * INPUT_DIM;

const __nv_bfloat162* x2 =
    reinterpret_cast<const __nv_bfloat162*>(x);
const __nv_bfloat162* gate_weight2 =
    reinterpret_cast<const __nv_bfloat162*>(gate_weight + row_offset);
const __nv_bfloat162* up_weight2 =
    reinterpret_cast<const __nv_bfloat162*>(up_weight + row_offset);

float gate_sum = 0.0f;
float up_sum = 0.0f;
if constexpr (INPUT_DIM % 8 == 0) {
  union alignas(16) BF16x8 {
    uint4 packed;
    __nv_bfloat162 pairs[4];
  };

  const uint4* x8 = reinterpret_cast<const uint4*>(x);
  const uint4* gate_weight8 =
      reinterpret_cast<const uint4*>(gate_weight + row_offset);
  const uint4* up_weight8 =
      reinterpret_cast<const uint4*>(up_weight + row_offset);
  for (int col8 = threadIdx.x; col8 < INPUT_DIM / 8;
       col8 += blockDim.x) {
    BF16x8 xv;
    BF16x8 gate;
    BF16x8 up;
    xv.packed = x8[col8];
    gate.packed = gate_weight8[col8];
    up.packed = up_weight8[col8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float2 xf = __bfloat1622float2(xv.pairs[i]);
      float2 gatef = __bfloat1622float2(gate.pairs[i]);
      float2 upf = __bfloat1622float2(up.pairs[i]);
      gate_sum = fmaf(gatef.x, xf.x, gate_sum);
      gate_sum = fmaf(gatef.y, xf.y, gate_sum);
      up_sum = fmaf(upf.x, xf.x, up_sum);
      up_sum = fmaf(upf.y, xf.y, up_sum);
    }
  }
} else {
  for (int col2 = threadIdx.x; col2 < INPUT_DIM / 2;
       col2 += blockDim.x) {
    float2 xv = __bfloat1622float2(x2[col2]);
    float2 gate = __bfloat1622float2(gate_weight2[col2]);
    float2 up = __bfloat1622float2(up_weight2[col2]);
    gate_sum = fmaf(gate.x, xv.x, gate_sum);
    gate_sum = fmaf(gate.y, xv.y, gate_sum);
    up_sum = fmaf(up.x, xv.x, up_sum);
    up_sum = fmaf(up.y, xv.y, up_sum);
  }
}

#pragma unroll
for (int offset = WARP_WIDTH / 2; offset > 0; offset /= 2) {
  gate_sum += __shfl_down_sync(FULL_MASK, gate_sum, offset);
  up_sum += __shfl_down_sync(FULL_MASK, up_sum, offset);
}

__shared__ float gate_warp_sums[8];
__shared__ float up_warp_sums[8];
if (lane == 0) {
  gate_warp_sums[warp] = gate_sum;
  up_warp_sums[warp] = up_sum;
}
__syncthreads();

if (warp == 0) {
  gate_sum = lane < 8 ? gate_warp_sums[lane] : 0.0f;
  up_sum = lane < 8 ? up_warp_sums[lane] : 0.0f;
#pragma unroll
  for (int offset = WARP_WIDTH / 2; offset > 0; offset /= 2) {
    gate_sum += __shfl_down_sync(FULL_MASK, gate_sum, offset);
    up_sum += __shfl_down_sync(FULL_MASK, up_sum, offset);
  }

  if (lane == 0) {
    InT gate = static_cast<InT>(gate_sum);
    InT up = static_cast<InT>(up_sum);
    InT gate2 = static_cast<InT>(
        static_cast<float>(gate) * static_cast<float>(gate));
    InT gate3 = static_cast<InT>(
        static_cast<float>(gate2) * static_cast<float>(gate));
    InT cubic = static_cast<InT>(
        static_cast<float>(static_cast<InT>(0.044715f)) *
        static_cast<float>(gate3));
    InT inner = static_cast<InT>(
        static_cast<float>(gate) + static_cast<float>(cubic));
    InT scaled = static_cast<InT>(
        static_cast<float>(static_cast<InT>(0.7978845608028654f)) *
        static_cast<float>(inner));
    InT activated = static_cast<InT>(tanhf(static_cast<float>(scaled)));
    InT shifted = static_cast<InT>(
        static_cast<float>(static_cast<InT>(1.0f)) +
        static_cast<float>(activated));
    InT half_gate = static_cast<InT>(
        static_cast<float>(static_cast<InT>(0.5f)) *
        static_cast<float>(gate));
    InT gelu = static_cast<InT>(
        static_cast<float>(half_gate) * static_cast<float>(shifted));
    InT hidden = static_cast<InT>(
        static_cast<float>(gelu) * static_cast<float>(up));
    output[row] = hidden;
  }
}
`

func initBF16GeGLUProjectionCUDAKernel() {
	var cudaAvailable C.bool
	if C.mlx_cuda_is_available(&cudaAvailable) != 0 || !bool(cudaAvailable) {
		bf16GeGLUProjectionCUDADisabled = true
		return
	}

	inputs, freeInputs, ok := cStringVector([]string{"x", "gate_weight", "up_weight"})
	if !ok {
		bf16GeGLUProjectionCUDADisabled = true
		freeInputs()
		return
	}
	defer freeInputs()

	outputs, freeOutputs, ok := cStringVector([]string{"output"})
	if !ok {
		bf16GeGLUProjectionCUDADisabled = true
		freeOutputs()
		return
	}
	defer freeOutputs()

	cName := C.CString("bf16_geglu_projection")
	defer C.free(unsafe.Pointer(cName))
	cSource := C.CString(bf16GeGLUProjectionCUDAKernelSource)
	defer C.free(unsafe.Pointer(cSource))
	cHeader := C.CString(bf16GeGLUProjectionCUDAKernelHeader)
	defer C.free(unsafe.Pointer(cHeader))

	bf16GeGLUProjectionCUDAKernel = C.mlx_fast_cuda_kernel_new(
		cName,
		inputs,
		outputs,
		cSource,
		cHeader,
		C.bool(true),
		C.int(0),
	)
}

func bf16GeGLUProjectionCUDAConfig(inputDim, outputDim int) (C.mlx_fast_cuda_kernel_config, bool) {
	key := bf16GeGLUProjectionConfigKey{inputDim: inputDim, outputDim: outputDim}

	bf16GeGLUProjectionCUDAConfigMu.Lock()
	defer bf16GeGLUProjectionCUDAConfigMu.Unlock()

	if cfg, ok := bf16GeGLUProjectionCUDAConfigs[key]; ok {
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
			cfg, bf16GeGLUProjectionThreads, 1, C.int(outputDim),
		) != 0 ||
		C.mlx_fast_cuda_kernel_config_set_thread_group(
			cfg, bf16GeGLUProjectionThreads, 1, 1,
		) != 0 ||
		C.mlx_fast_cuda_kernel_config_set_verbose(cfg, false) != 0 {
		return fail()
	}

	bf16GeGLUProjectionCUDAConfigs[key] = cfg
	return cfg, true
}

// FastBF16GeGLUProjection fuses two BF16 decode projections with GeGLU on
// CUDA. cuBLASLt can run each projection, but has no epilogue for combining two
// independent matrices with GeGLU; its production caller limits this path to
// the memory-bound Gemma MLP shapes where endpoint benchmarks improved.
// Unsupported inputs return ok=false.
func FastBF16GeGLUProjection(x, gateWeight, upWeight *Array) (out *Array, ok bool) {
	if bf16GeGLUProjectionCUDADisabled ||
		x == nil || gateWeight == nil || upWeight == nil ||
		x.DType() != DTypeBFloat16 ||
		gateWeight.DType() != DTypeBFloat16 ||
		upWeight.DType() != DTypeBFloat16 ||
		x.NumDims() != 3 ||
		x.Dim(0) != 1 || x.Dim(1) != 1 ||
		gateWeight.NumDims() != 2 ||
		upWeight.NumDims() != 2 ||
		gateWeight.Dim(0) != upWeight.Dim(0) ||
		gateWeight.Dim(1) != upWeight.Dim(1) ||
		x.Dim(2) != gateWeight.Dim(1) ||
		x.Dim(2)%2 != 0 {
		return nil, false
	}

	bf16GeGLUProjectionCUDAKernelOnce.Do(initBF16GeGLUProjectionCUDAKernel)
	if bf16GeGLUProjectionCUDADisabled {
		return nil, false
	}

	inputDim := x.Dim(2)
	outputDim := gateWeight.Dim(0)
	cfg, ok := bf16GeGLUProjectionCUDAConfig(inputDim, outputDim)
	if !ok {
		bf16GeGLUProjectionCUDADisabled = true
		return nil, false
	}

	inputs := []C.mlx_array{x.ctx, gateWeight.ctx, upWeight.ctx}
	inVec := C.mlx_vector_array_new_data(
		unsafe.SliceData(inputs),
		C.size_t(len(inputs)),
	)
	defer C.mlx_vector_array_free(inVec)

	outVec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(outVec)
	if C.mlx_fast_cuda_kernel_apply(
		&outVec,
		bf16GeGLUProjectionCUDAKernel,
		inVec,
		cfg,
		DefaultStream().ctx,
	) != 0 || C.mlx_vector_array_size(outVec) < 1 {
		bf16GeGLUProjectionCUDADisabled = true
		return nil, false
	}

	out = New("BF16_GEGLU_PROJECTION_CUDA")
	if C.mlx_vector_array_get(&out.ctx, outVec, 0) != 0 {
		bf16GeGLUProjectionCUDADisabled = true
		return nil, false
	}
	return out, true
}
