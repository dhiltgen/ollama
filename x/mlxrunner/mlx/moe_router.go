package mlx

/*
#include <stdlib.h>
#include "generated.h"

static int mlx_fast_cuda_kernel_get2_outputs(
    mlx_array* out0,
    mlx_array* out1,
    mlx_vector_array outputs) {
  if (mlx_vector_array_size(outputs) < 2) {
    return 1;
  }
  int rc = mlx_vector_array_get(out0, outputs, 0);
  if (rc != 0) {
    return rc;
  }
  rc = mlx_vector_array_get(out1, outputs, 1);
  if (rc != 0) {
    mlx_array_free(*out0);
  }
  return rc;
}

static int mlx_fast_cuda_kernel_apply2_out2(
    mlx_array* out0,
    mlx_array* out1,
    mlx_fast_cuda_kernel kernel,
    mlx_array input0,
    mlx_array input1,
    mlx_fast_cuda_kernel_config config,
    mlx_stream stream) {
  mlx_array input_data[] = {input0, input1};
  mlx_vector_array inputs = mlx_vector_array_new_data(input_data, 2);
  mlx_vector_array outputs = mlx_vector_array_new();
  int rc = mlx_fast_cuda_kernel_apply(
      &outputs, kernel, inputs, config, stream);
  if (rc == 0) {
    rc = mlx_fast_cuda_kernel_get2_outputs(out0, out1, outputs);
  }
  mlx_vector_array_free(outputs);
  mlx_vector_array_free(inputs);
  return rc;
}

static int mlx_fast_cuda_kernel_apply1_out2(
    mlx_array* out0,
    mlx_array* out1,
    mlx_fast_cuda_kernel kernel,
    mlx_array input0,
    mlx_fast_cuda_kernel_config config,
    mlx_stream stream) {
  mlx_array input_data[] = {input0};
  mlx_vector_array inputs = mlx_vector_array_new_data(input_data, 1);
  mlx_vector_array outputs = mlx_vector_array_new();
  int rc = mlx_fast_cuda_kernel_apply(
      &outputs, kernel, inputs, config, stream);
  if (rc == 0) {
    rc = mlx_fast_cuda_kernel_get2_outputs(out0, out1, outputs);
  }
  mlx_vector_array_free(outputs);
  mlx_vector_array_free(inputs);
  return rc;
}
*/
import "C"

import (
	"sync"
	"unsafe"
)

var (
	moeRouterCUDAKernelOnce sync.Once
	moeRouterCUDAKernel     C.mlx_fast_cuda_kernel
	moeRouterCUDADisabled   bool

	moeRouterCUDAConfigMu    sync.Mutex
	moeRouterCUDAConfigs     [9]C.mlx_fast_cuda_kernel_config
	moeRouterCUDAConfigValid [9]bool

	normalizedMoERouter256CUDAKernelOnce sync.Once
	normalizedMoERouter256CUDAKernel     C.mlx_fast_cuda_kernel
	normalizedMoERouter256CUDADisabled   bool

	normalizedMoERouter256CUDAConfigMu    sync.Mutex
	normalizedMoERouter256CUDAConfigs     [9]C.mlx_fast_cuda_kernel_config
	normalizedMoERouter256CUDAConfigValid [9]bool

	sigmoidMoERouter256CUDAKernelOnce sync.Once
	sigmoidMoERouter256CUDAKernel     C.mlx_fast_cuda_kernel
	sigmoidMoERouter256CUDADisabled   bool

	sigmoidMoERouter256CUDAConfigMu    sync.Mutex
	sigmoidMoERouter256CUDAConfigs     [9]C.mlx_fast_cuda_kernel_config
	sigmoidMoERouter256CUDAConfigValid [9]bool
)

const moeRouterCUDAKernelHeader = `
__device__ bool router_better(
    float candidate_value,
    int candidate_index,
    float current_value,
    int current_index) {
  if (candidate_index < 0) {
    return false;
  }
  if (current_index < 0) {
    return true;
  }
  bool candidate_nan = isnan(candidate_value);
  bool current_nan = isnan(current_value);
  if (candidate_nan != current_nan) {
    return candidate_nan;
  }
  return candidate_value > current_value ||
      (candidate_value == current_value && candidate_index < current_index);
}

`

const moeRouterCUDAKernelSource = `
constexpr unsigned int FULL_MASK = 0xffffffff;
constexpr int EXPERTS = 128;
constexpr int TOP_K = 8;
constexpr int VALUES_PER_THREAD = 4;

int lane = threadIdx.x;
int row = blockIdx.z;
const InT* row_logits = logits + row * EXPERTS;

float values[VALUES_PER_THREAD];
int local_indices[VALUES_PER_THREAD];
#pragma unroll
for (int i = 0; i < VALUES_PER_THREAD; ++i) {
  int index = lane + i * 32;
  values[i] = static_cast<float>(row_logits[index]);
  local_indices[i] = index;
}

float selected_value = 0.0f;
int selected_index = -1;
#pragma unroll
for (int rank = 0; rank < TOP_K; ++rank) {
  float best_value = 0.0f;
  int best_index = -1;
#pragma unroll
  for (int i = 0; i < VALUES_PER_THREAD; ++i) {
    if (router_better(
            values[i], local_indices[i], best_value, best_index)) {
      best_value = values[i];
      best_index = local_indices[i];
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    float other_value = __shfl_down_sync(FULL_MASK, best_value, offset);
    int other_index = __shfl_down_sync(FULL_MASK, best_index, offset);
    if (lane + offset < 32 &&
        router_better(other_value, other_index, best_value, best_index)) {
      best_value = other_value;
      best_index = other_index;
    }
  }
  best_value = __shfl_sync(FULL_MASK, best_value, 0);
  best_index = __shfl_sync(FULL_MASK, best_index, 0);

  if (lane == rank) {
    selected_value = best_value;
    selected_index = best_index;
  }
#pragma unroll
  for (int i = 0; i < VALUES_PER_THREAD; ++i) {
    if (local_indices[i] == best_index) {
      local_indices[i] = -1;
    }
  }
}

float max_value = lane < TOP_K ? selected_value : -INFINITY;
#pragma unroll
for (int offset = 16; offset > 0; offset >>= 1) {
  float other = __shfl_down_sync(FULL_MASK, max_value, offset);
  if (lane + offset < 32 && (isnan(other) || other > max_value)) {
    max_value = other;
  }
}
max_value = __shfl_sync(FULL_MASK, max_value, 0);

float exponent = lane < TOP_K ? expf(selected_value - max_value) : 0.0f;
float sum = exponent;
#pragma unroll
for (int offset = 16; offset > 0; offset >>= 1) {
  sum += __shfl_down_sync(FULL_MASK, sum, offset);
}
sum = __shfl_sync(FULL_MASK, sum, 0);

if (lane < TOP_K) {
  int output_index = row * TOP_K + lane;
  InT normalized_weight = static_cast<InT>(exponent / sum);
  weights[output_index] = static_cast<InT>(
      static_cast<float>(normalized_weight) *
      static_cast<float>(expert_scales[selected_index]));
  indices[output_index] = static_cast<uint32_t>(selected_index);
}
`

const normalizedMoERouter256CUDAKernelSource = `
constexpr unsigned int FULL_MASK = 0xffffffff;
constexpr int EXPERTS = 256;
constexpr int TOP_K = 8;
constexpr int VALUES_PER_THREAD = 8;

int lane = threadIdx.x;
int row = blockIdx.z;
const InT* row_logits = logits + row * EXPERTS;

float values[VALUES_PER_THREAD];
int local_indices[VALUES_PER_THREAD];
#pragma unroll
for (int i = 0; i < VALUES_PER_THREAD; ++i) {
  int index = lane + i * 32;
  values[i] = static_cast<float>(row_logits[index]);
  local_indices[i] = index;
}

float selected_value = 0.0f;
int selected_index = -1;
#pragma unroll
for (int rank = 0; rank < TOP_K; ++rank) {
  float best_value = 0.0f;
  int best_index = -1;
#pragma unroll
  for (int i = 0; i < VALUES_PER_THREAD; ++i) {
    if (router_better(
            values[i], local_indices[i], best_value, best_index)) {
      best_value = values[i];
      best_index = local_indices[i];
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    float other_value = __shfl_down_sync(FULL_MASK, best_value, offset);
    int other_index = __shfl_down_sync(FULL_MASK, best_index, offset);
    if (lane + offset < 32 &&
        router_better(other_value, other_index, best_value, best_index)) {
      best_value = other_value;
      best_index = other_index;
    }
  }
  best_value = __shfl_sync(FULL_MASK, best_value, 0);
  best_index = __shfl_sync(FULL_MASK, best_index, 0);

  if (lane == rank) {
    selected_value = best_value;
    selected_index = best_index;
  }
#pragma unroll
  for (int i = 0; i < VALUES_PER_THREAD; ++i) {
    if (local_indices[i] == best_index) {
      local_indices[i] = -1;
    }
  }
}

float max_value = lane < TOP_K ? selected_value : -INFINITY;
#pragma unroll
for (int offset = 16; offset > 0; offset >>= 1) {
  float other = __shfl_down_sync(FULL_MASK, max_value, offset);
  if (lane + offset < 32 && (isnan(other) || other > max_value)) {
    max_value = other;
  }
}
max_value = __shfl_sync(FULL_MASK, max_value, 0);

float exponent = lane < TOP_K ? expf(selected_value - max_value) : 0.0f;
float sum = exponent;
#pragma unroll
for (int offset = 16; offset > 0; offset >>= 1) {
  sum += __shfl_down_sync(FULL_MASK, sum, offset);
}
sum = __shfl_sync(FULL_MASK, sum, 0);

if (lane < TOP_K) {
  int output_index = row * TOP_K + lane;
  weights[output_index] = static_cast<InT>(exponent / sum);
  indices[output_index] = static_cast<uint32_t>(selected_index);
}
`

const sigmoidMoERouter256CUDAKernelSource = `
constexpr unsigned int FULL_MASK = 0xffffffff;
constexpr int EXPERTS = 256;
constexpr int TOP_K = 8;
constexpr int VALUES_PER_THREAD = 8;

int lane = threadIdx.x;
int row = blockIdx.z;
const InT* row_logits = logits + row * EXPERTS;

float probabilities[VALUES_PER_THREAD];
float values[VALUES_PER_THREAD];
int local_indices[VALUES_PER_THREAD];
#pragma unroll
for (int i = 0; i < VALUES_PER_THREAD; ++i) {
  int index = lane + i * 32;
  float logit = static_cast<float>(row_logits[index]);
  float probability = 1.0f / (1.0f + expf(-logit));
  probabilities[i] = probability;
  values[i] = probability + static_cast<float>(bias[index]);
  local_indices[i] = index;
}

float selected_probability = 0.0f;
int selected_index = -1;
#pragma unroll
for (int rank = 0; rank < TOP_K; ++rank) {
  float best_value = 0.0f;
  float best_probability = 0.0f;
  int best_index = -1;
#pragma unroll
  for (int i = 0; i < VALUES_PER_THREAD; ++i) {
    if (router_better(
            values[i], local_indices[i], best_value, best_index)) {
      best_value = values[i];
      best_probability = probabilities[i];
      best_index = local_indices[i];
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    float other_value = __shfl_down_sync(FULL_MASK, best_value, offset);
    float other_probability =
        __shfl_down_sync(FULL_MASK, best_probability, offset);
    int other_index = __shfl_down_sync(FULL_MASK, best_index, offset);
    if (lane + offset < 32 &&
        router_better(other_value, other_index, best_value, best_index)) {
      best_value = other_value;
      best_probability = other_probability;
      best_index = other_index;
    }
  }
  best_probability = __shfl_sync(FULL_MASK, best_probability, 0);
  best_index = __shfl_sync(FULL_MASK, best_index, 0);

  if (lane == rank) {
    selected_probability = best_probability;
    selected_index = best_index;
  }
#pragma unroll
  for (int i = 0; i < VALUES_PER_THREAD; ++i) {
    if (local_indices[i] == best_index) {
      local_indices[i] = -1;
    }
  }
}

float sum = lane < TOP_K ? selected_probability : 0.0f;
#pragma unroll
for (int offset = 16; offset > 0; offset >>= 1) {
  sum += __shfl_down_sync(FULL_MASK, sum, offset);
}
sum = __shfl_sync(FULL_MASK, sum, 0);

if (lane < TOP_K) {
  int output_index = row * TOP_K + lane;
  weights[output_index] = selected_probability / sum;
  indices[output_index] = static_cast<uint32_t>(selected_index);
}
`

func initMoERouterCUDAKernel() {
	var cudaAvailable C.bool
	if C.mlx_cuda_is_available(&cudaAvailable) != 0 || !bool(cudaAvailable) {
		moeRouterCUDADisabled = true
		return
	}

	inputs, freeInputs, ok := cStringVector([]string{"logits", "expert_scales"})
	if !ok {
		moeRouterCUDADisabled = true
		freeInputs()
		return
	}
	defer freeInputs()

	outputs, freeOutputs, ok := cStringVector([]string{"weights", "indices"})
	if !ok {
		moeRouterCUDADisabled = true
		freeOutputs()
		return
	}
	defer freeOutputs()

	cName := C.CString("fused_moe_router_top8_128")
	defer C.free(unsafe.Pointer(cName))
	cSource := C.CString(moeRouterCUDAKernelSource)
	defer C.free(unsafe.Pointer(cSource))
	cHeader := C.CString(moeRouterCUDAKernelHeader)
	defer C.free(unsafe.Pointer(cHeader))

	moeRouterCUDAKernel = C.mlx_fast_cuda_kernel_new(
		cName,
		inputs,
		outputs,
		cSource,
		cHeader,
		C.bool(true),
		C.int(0),
	)
}

func moeRouterCUDAConfig(rows int) (C.mlx_fast_cuda_kernel_config, bool) {
	moeRouterCUDAConfigMu.Lock()
	defer moeRouterCUDAConfigMu.Unlock()

	if moeRouterCUDAConfigValid[rows] {
		return moeRouterCUDAConfigs[rows], true
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

	shape := []C.int{C.int(rows), 8}
	if C.mlx_fast_cuda_kernel_config_add_output_arg(
		cfg,
		unsafe.SliceData(shape),
		C.size_t(len(shape)),
		C.mlx_dtype(DTypeBFloat16),
	) != 0 {
		return fail()
	}
	if C.mlx_fast_cuda_kernel_config_add_output_arg(
		cfg,
		unsafe.SliceData(shape),
		C.size_t(len(shape)),
		C.mlx_dtype(DTypeUint32),
	) != 0 {
		return fail()
	}
	if C.mlx_fast_cuda_kernel_config_set_grid(cfg, 32, 1, C.int(rows)) != 0 {
		return fail()
	}
	if C.mlx_fast_cuda_kernel_config_set_thread_group(cfg, 32, 1, 1) != 0 {
		return fail()
	}
	if C.mlx_fast_cuda_kernel_config_set_verbose(cfg, false) != 0 {
		return fail()
	}

	// Decode only admits rows 1-8, so retaining these immutable configs is bounded.
	moeRouterCUDAConfigs[rows] = cfg
	moeRouterCUDAConfigValid[rows] = true
	return cfg, true
}

// FastMoERouter selects and normalizes the top eight of 128 BF16 expert logits,
// then applies the corresponding static per-expert scale. It is intentionally
// limited to small CUDA decode batches. A CUB full sort does substantially more
// work than fixed top-eight selection and measured slower; other shapes and
// backends retain the regular MLX graph.
func FastMoERouter(logits, expertScales *Array) (weights, indices *Array, ok bool) {
	if moeRouterCUDADisabled || logits == nil || expertScales == nil ||
		logits.DType() != DTypeBFloat16 || logits.NumDims() != 2 ||
		logits.Dim(1) != 128 ||
		expertScales.DType() != DTypeBFloat16 ||
		expertScales.NumDims() != 1 || expertScales.Dim(0) != 128 {
		return nil, nil, false
	}
	rows := logits.Dim(0)
	if rows < 1 || rows > 8 {
		return nil, nil, false
	}

	moeRouterCUDAKernelOnce.Do(initMoERouterCUDAKernel)
	if moeRouterCUDADisabled {
		return nil, nil, false
	}

	cfg, ok := moeRouterCUDAConfig(rows)
	if !ok {
		moeRouterCUDADisabled = true
		return nil, nil, false
	}

	var weightsCtx, indicesCtx C.mlx_array
	if C.mlx_fast_cuda_kernel_apply2_out2(
		&weightsCtx,
		&indicesCtx,
		moeRouterCUDAKernel,
		logits.ctx,
		expertScales.ctx,
		cfg,
		DefaultStream().ctx,
	) != 0 {
		moeRouterCUDADisabled = true
		return nil, nil, false
	}

	weights = New("MOE_ROUTER_CUDA_WEIGHTS")
	weights.ctx = weightsCtx
	indices = New("MOE_ROUTER_CUDA_INDICES")
	indices.ctx = indicesCtx
	return weights, indices, true
}

func initNormalizedMoERouter256CUDAKernel() {
	var cudaAvailable C.bool
	if C.mlx_cuda_is_available(&cudaAvailable) != 0 || !bool(cudaAvailable) {
		normalizedMoERouter256CUDADisabled = true
		return
	}

	inputs, freeInputs, ok := cStringVector([]string{"logits"})
	if !ok {
		normalizedMoERouter256CUDADisabled = true
		freeInputs()
		return
	}
	defer freeInputs()

	outputs, freeOutputs, ok := cStringVector([]string{"weights", "indices"})
	if !ok {
		normalizedMoERouter256CUDADisabled = true
		freeOutputs()
		return
	}
	defer freeOutputs()

	cName := C.CString("normalized_moe_router_top8_256")
	defer C.free(unsafe.Pointer(cName))
	cSource := C.CString(normalizedMoERouter256CUDAKernelSource)
	defer C.free(unsafe.Pointer(cSource))
	cHeader := C.CString(moeRouterCUDAKernelHeader)
	defer C.free(unsafe.Pointer(cHeader))

	normalizedMoERouter256CUDAKernel = C.mlx_fast_cuda_kernel_new(
		cName,
		inputs,
		outputs,
		cSource,
		cHeader,
		C.bool(true),
		C.int(0),
	)
}

func normalizedMoERouter256CUDAConfig(rows int) (C.mlx_fast_cuda_kernel_config, bool) {
	normalizedMoERouter256CUDAConfigMu.Lock()
	defer normalizedMoERouter256CUDAConfigMu.Unlock()

	if normalizedMoERouter256CUDAConfigValid[rows] {
		return normalizedMoERouter256CUDAConfigs[rows], true
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

	shape := []C.int{C.int(rows), 8}
	if C.mlx_fast_cuda_kernel_config_add_output_arg(
		cfg,
		unsafe.SliceData(shape),
		C.size_t(len(shape)),
		C.mlx_dtype(DTypeBFloat16),
	) != 0 {
		return fail()
	}
	if C.mlx_fast_cuda_kernel_config_add_output_arg(
		cfg,
		unsafe.SliceData(shape),
		C.size_t(len(shape)),
		C.mlx_dtype(DTypeUint32),
	) != 0 {
		return fail()
	}
	if C.mlx_fast_cuda_kernel_config_set_grid(cfg, 32, 1, C.int(rows)) != 0 {
		return fail()
	}
	if C.mlx_fast_cuda_kernel_config_set_thread_group(cfg, 32, 1, 1) != 0 {
		return fail()
	}
	if C.mlx_fast_cuda_kernel_config_set_verbose(cfg, false) != 0 {
		return fail()
	}

	normalizedMoERouter256CUDAConfigs[rows] = cfg
	normalizedMoERouter256CUDAConfigValid[rows] = true
	return cfg, true
}

// FastNormalizedMoERouter256 selects and normalizes the top eight of 256 BF16
// expert logits. A CUB full sort measured slower than fixed top-eight selection.
// Unsupported shapes and backends retain the regular MLX graph.
func FastNormalizedMoERouter256(logits *Array) (weights, indices *Array, ok bool) {
	if normalizedMoERouter256CUDADisabled || logits == nil ||
		logits.DType() != DTypeBFloat16 || logits.NumDims() != 2 ||
		logits.Dim(1) != 256 {
		return nil, nil, false
	}
	rows := logits.Dim(0)
	if rows < 1 || rows > 8 {
		return nil, nil, false
	}

	normalizedMoERouter256CUDAKernelOnce.Do(initNormalizedMoERouter256CUDAKernel)
	if normalizedMoERouter256CUDADisabled {
		return nil, nil, false
	}

	cfg, ok := normalizedMoERouter256CUDAConfig(rows)
	if !ok {
		normalizedMoERouter256CUDADisabled = true
		return nil, nil, false
	}

	var weightsCtx, indicesCtx C.mlx_array
	if C.mlx_fast_cuda_kernel_apply1_out2(
		&weightsCtx,
		&indicesCtx,
		normalizedMoERouter256CUDAKernel,
		logits.ctx,
		cfg,
		DefaultStream().ctx,
	) != 0 {
		normalizedMoERouter256CUDADisabled = true
		return nil, nil, false
	}

	weights = New("NORMALIZED_MOE_ROUTER_256_CUDA_WEIGHTS")
	weights.ctx = weightsCtx
	indices = New("NORMALIZED_MOE_ROUTER_256_CUDA_INDICES")
	indices.ctx = indicesCtx
	return weights, indices, true
}

func initSigmoidMoERouter256CUDAKernel() {
	var cudaAvailable C.bool
	if C.mlx_cuda_is_available(&cudaAvailable) != 0 || !bool(cudaAvailable) {
		sigmoidMoERouter256CUDADisabled = true
		return
	}

	inputs, freeInputs, ok := cStringVector([]string{"logits", "bias"})
	if !ok {
		sigmoidMoERouter256CUDADisabled = true
		freeInputs()
		return
	}
	defer freeInputs()

	outputs, freeOutputs, ok := cStringVector([]string{"weights", "indices"})
	if !ok {
		sigmoidMoERouter256CUDADisabled = true
		freeOutputs()
		return
	}
	defer freeOutputs()

	cName := C.CString("sigmoid_moe_router_top8_256")
	defer C.free(unsafe.Pointer(cName))
	cSource := C.CString(sigmoidMoERouter256CUDAKernelSource)
	defer C.free(unsafe.Pointer(cSource))
	cHeader := C.CString(moeRouterCUDAKernelHeader)
	defer C.free(unsafe.Pointer(cHeader))

	sigmoidMoERouter256CUDAKernel = C.mlx_fast_cuda_kernel_new(
		cName,
		inputs,
		outputs,
		cSource,
		cHeader,
		C.bool(true),
		C.int(0),
	)
}

func sigmoidMoERouter256CUDAConfig(rows int) (C.mlx_fast_cuda_kernel_config, bool) {
	sigmoidMoERouter256CUDAConfigMu.Lock()
	defer sigmoidMoERouter256CUDAConfigMu.Unlock()

	if sigmoidMoERouter256CUDAConfigValid[rows] {
		return sigmoidMoERouter256CUDAConfigs[rows], true
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

	shape := []C.int{C.int(rows), 8}
	if C.mlx_fast_cuda_kernel_config_add_output_arg(
		cfg,
		unsafe.SliceData(shape),
		C.size_t(len(shape)),
		C.mlx_dtype(DTypeFloat32),
	) != 0 {
		return fail()
	}
	if C.mlx_fast_cuda_kernel_config_add_output_arg(
		cfg,
		unsafe.SliceData(shape),
		C.size_t(len(shape)),
		C.mlx_dtype(DTypeUint32),
	) != 0 {
		return fail()
	}
	if C.mlx_fast_cuda_kernel_config_set_grid(cfg, 32, 1, C.int(rows)) != 0 {
		return fail()
	}
	if C.mlx_fast_cuda_kernel_config_set_thread_group(cfg, 32, 1, 1) != 0 {
		return fail()
	}
	if C.mlx_fast_cuda_kernel_config_set_verbose(cfg, false) != 0 {
		return fail()
	}

	sigmoidMoERouter256CUDAConfigs[rows] = cfg
	sigmoidMoERouter256CUDAConfigValid[rows] = true
	return cfg, true
}

// FastSigmoidMoERouter256 applies Laguna's sigmoid-and-correction-bias router,
// selects the top eight experts, and normalizes their original sigmoid scores.
// A CUB full sort regressed the endpoint, so this retains fixed top-eight
// selection. Unsupported inputs retain the regular MLX graph.
func FastSigmoidMoERouter256(logits, bias *Array) (weights, indices *Array, ok bool) {
	if sigmoidMoERouter256CUDADisabled ||
		logits == nil || bias == nil ||
		logits.DType() != DTypeBFloat16 || logits.NumDims() != 2 ||
		logits.Dim(1) != 256 ||
		bias.DType() != DTypeFloat32 ||
		bias.NumDims() != 1 || bias.Dim(0) != 256 {
		return nil, nil, false
	}
	rows := logits.Dim(0)
	if rows < 1 || rows > 8 {
		return nil, nil, false
	}

	sigmoidMoERouter256CUDAKernelOnce.Do(initSigmoidMoERouter256CUDAKernel)
	if sigmoidMoERouter256CUDADisabled {
		return nil, nil, false
	}

	cfg, ok := sigmoidMoERouter256CUDAConfig(rows)
	if !ok {
		sigmoidMoERouter256CUDADisabled = true
		return nil, nil, false
	}

	var weightsCtx, indicesCtx C.mlx_array
	if C.mlx_fast_cuda_kernel_apply2_out2(
		&weightsCtx,
		&indicesCtx,
		sigmoidMoERouter256CUDAKernel,
		logits.ctx,
		bias.ctx,
		cfg,
		DefaultStream().ctx,
	) != 0 {
		sigmoidMoERouter256CUDADisabled = true
		return nil, nil, false
	}

	weights = New("SIGMOID_MOE_ROUTER_256_CUDA_WEIGHTS")
	weights.ctx = weightsCtx
	indices = New("SIGMOID_MOE_ROUTER_256_CUDA_INDICES")
	indices.ctx = indicesCtx
	return weights, indices, true
}
