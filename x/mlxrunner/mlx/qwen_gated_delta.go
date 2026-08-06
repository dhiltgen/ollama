package mlx

// #include <stdlib.h>
// #include "generated.h"
import "C"

import (
	"sync"
	"unsafe"
)

const qwenGatedDeltaInputsCUDAReserve = 2 << 30

var (
	qwenGatedDeltaInputsCUDAKernelOnce sync.Once
	qwenGatedDeltaInputsCUDAKernel     C.mlx_fast_cuda_kernel
	qwenGatedDeltaInputsCUDAConfig     C.mlx_fast_cuda_kernel_config
	qwenGatedDeltaInputsCUDADisabled   bool
	qwenGatedDeltaInputsCUDAEnabled    = sync.OnceValue(func() bool {
		total, free, ok := cudaDeviceMemory()
		if !ok {
			return false
		}
		return hasCUDAMemoryHeadroom(
			total,
			free,
			ActiveMemory(),
			PeakMemory(),
			qwenGatedDeltaInputsCUDAReserve,
		)
	})

	qwenGatedDeltaOutputCUDAKernelOnce sync.Once
	qwenGatedDeltaOutputCUDAKernel     C.mlx_fast_cuda_kernel
	qwenGatedDeltaOutputCUDAConfig     C.mlx_fast_cuda_kernel_config
	qwenGatedDeltaOutputCUDADisabled   bool
)

const qwenGatedDeltaInputsCUDAKernelSource = `
constexpr unsigned int FULL_MASK = 0xffffffff;
constexpr int DK = 128;
constexpr int HK = 16;
constexpr int DV = 128;
constexpr int HV = 32;

int lane = threadIdx.x;
int warp = threadIdx.y;
int tile = blockIdx.y;

if (blockIdx.z == 0) {
  int head = tile * blockDim.y + warp;
  if (head >= HK) {
    return;
  }

  float q_values[8] = {0.0f};
  float k_values[8] = {0.0f};
  float q_sum = 0.0f;
  float k_sum = 0.0f;
  if (lane < 16) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int d = lane * 8 + i;
      InT q_input = conv[head * DK + d];
      InT k_input = conv[HK * DK + head * DK + d];
      float q_float = static_cast<float>(q_input);
      float k_float = static_cast<float>(k_input);
      InT q_exp = static_cast<InT>(expf(fabsf(q_float)));
      InT k_exp = static_cast<InT>(expf(fabsf(k_float)));
      InT q_denom = static_cast<InT>(1.0f + static_cast<float>(q_exp));
      InT k_denom = static_cast<InT>(1.0f + static_cast<float>(k_exp));
      InT q_y = static_cast<InT>(1.0f / static_cast<float>(q_denom));
      InT k_y = static_cast<InT>(1.0f / static_cast<float>(k_denom));
      InT q_sigmoid = q_float < 0.0f
          ? q_y
          : static_cast<InT>(1.0f - static_cast<float>(q_y));
      InT k_sigmoid = k_float < 0.0f
          ? k_y
          : static_cast<InT>(1.0f - static_cast<float>(k_y));
      InT q_value_bf16 = static_cast<InT>(
          q_float * static_cast<float>(q_sigmoid));
      InT k_value_bf16 = static_cast<InT>(
          k_float * static_cast<float>(k_sigmoid));
      float q_value = static_cast<float>(q_value_bf16);
      float k_value = static_cast<float>(k_value_bf16);
      q_values[i] = q_value;
      k_values[i] = k_value;
      q_sum += q_value * q_value;
      k_sum += k_value * k_value;
    }
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    q_sum += __shfl_down_sync(FULL_MASK, q_sum, offset);
    k_sum += __shfl_down_sync(FULL_MASK, k_sum, offset);
  }
  q_sum = __shfl_sync(FULL_MASK, q_sum, 0);
  k_sum = __shfl_sync(FULL_MASK, k_sum, 0);
  float q_scale = rsqrtf(q_sum / DK + 1.0e-6f);
  float k_scale = rsqrtf(k_sum / DK + 1.0e-6f);

  if (lane < 16) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int d = lane * 8 + i;
      int offset = head * DK + d;
      InT q_normalized = static_cast<InT>(q_values[i] * q_scale);
      InT k_normalized = static_cast<InT>(k_values[i] * k_scale);
      q[offset] = static_cast<InT>(
          static_cast<float>(q_normalized) *
          static_cast<float>(static_cast<InT>(1.0f / DK)));
      k[offset] = static_cast<InT>(
          static_cast<float>(k_normalized) *
          static_cast<float>(static_cast<InT>(0.08838835f)));
    }
  }
} else if (blockIdx.z == 1) {
  int head = tile * blockDim.y + warp;
  if (head >= HV) {
    return;
  }
  int v_base = 2 * HK * DK + head * DV;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int d = lane + i * 32;
    float value = static_cast<float>(conv[v_base + d]);
    InT exponent = static_cast<InT>(expf(fabsf(value)));
    InT denominator = static_cast<InT>(
        1.0f + static_cast<float>(exponent));
    InT y_value = static_cast<InT>(
        1.0f / static_cast<float>(denominator));
    InT sigmoid = value < 0.0f
        ? y_value
        : static_cast<InT>(1.0f - static_cast<float>(y_value));
    v[head * DV + d] = static_cast<InT>(
        value * static_cast<float>(sigmoid));
  }
} else if (tile == 0 && warp == 0 && lane < HV) {
  InT sum = static_cast<InT>(
      static_cast<float>(alpha[lane]) +
      static_cast<float>(dt_bias[lane]));
  InT max_value = static_cast<float>(sum) > 0.0f
      ? sum
      : static_cast<InT>(0.0f);
  InT min_value = static_cast<float>(sum) < 0.0f
      ? sum
      : static_cast<InT>(0.0f);
  InT difference = static_cast<InT>(
      static_cast<float>(min_value) - static_cast<float>(max_value));
  InT exponent = static_cast<InT>(expf(static_cast<float>(difference)));
  InT logarithm = static_cast<InT>(log1pf(static_cast<float>(exponent)));
  InT softplus = static_cast<InT>(
      static_cast<float>(max_value) + static_cast<float>(logarithm));
  g[lane] = static_cast<InT>(
      expf(-static_cast<float>(softplus) * a_exp[lane]));

  float beta_value = static_cast<float>(beta[lane]);
  InT beta_exp = static_cast<InT>(expf(fabsf(beta_value)));
  InT beta_denom = static_cast<InT>(
      1.0f + static_cast<float>(beta_exp));
  InT beta_y = static_cast<InT>(
      1.0f / static_cast<float>(beta_denom));
  beta_gate[lane] = beta_value < 0.0f
      ? beta_y
      : static_cast<InT>(1.0f - static_cast<float>(beta_y));
}
`

const qwenGatedDeltaOutputCUDAKernelSource = `
constexpr unsigned int FULL_MASK = 0xffffffff;
constexpr int D = 128;
constexpr int HEADS = 32;

int lane = threadIdx.x;
int head = blockIdx.y * blockDim.y + threadIdx.y;
if (head >= HEADS) {
  return;
}

float values[8] = {0.0f};
float z_values[8] = {0.0f};
float sum = 0.0f;
if (lane < 16) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    int d = lane * 8 + i;
    int offset = head * D + d;
    float value = static_cast<float>(state_out[offset]);
    values[i] = value;
    z_values[i] = static_cast<float>(z[offset]);
    sum += value * value;
  }
}
#pragma unroll
for (int offset = 16; offset > 0; offset >>= 1) {
  sum += __shfl_down_sync(FULL_MASK, sum, offset);
}
sum = __shfl_sync(FULL_MASK, sum, 0);
float inv_rms = rsqrtf(sum / D + 1.0e-6f);

if (lane < 16) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    int d = lane * 8 + i;
    int offset = head * D + d;
    InT scaled = static_cast<InT>(values[i] * inv_rms);
    InT normalized = static_cast<InT>(
        static_cast<float>(norm_weight[d]) *
        static_cast<float>(scaled));
    float z_value = z_values[i];
    float z_sigmoid = 1.0f / (1.0f + expf(fabsf(z_value)));
    z_sigmoid = z_value < 0.0f ? z_sigmoid : 1.0f - z_sigmoid;
    float z_silu = z_value * z_sigmoid;
    output[offset] = static_cast<InT>(
        static_cast<float>(normalized) * z_silu);
  }
}
`

func initQwenGatedDeltaInputsCUDAKernel() {
	var cudaAvailable C.bool
	if C.mlx_cuda_is_available(&cudaAvailable) != 0 || !bool(cudaAvailable) {
		qwenGatedDeltaInputsCUDADisabled = true
		return
	}

	inputs, freeInputs, ok := cStringVector(
		[]string{"conv", "alpha", "beta", "a_exp", "dt_bias"},
	)
	if !ok {
		qwenGatedDeltaInputsCUDADisabled = true
		freeInputs()
		return
	}
	defer freeInputs()

	outputs, freeOutputs, ok := cStringVector(
		[]string{"q", "k", "v", "g", "beta_gate"},
	)
	if !ok {
		qwenGatedDeltaInputsCUDADisabled = true
		freeOutputs()
		return
	}
	defer freeOutputs()

	cName := C.CString("qwen_gated_delta_inputs")
	defer C.free(unsafe.Pointer(cName))
	cSource := C.CString(qwenGatedDeltaInputsCUDAKernelSource)
	defer C.free(unsafe.Pointer(cSource))
	cHeader := C.CString("")
	defer C.free(unsafe.Pointer(cHeader))
	qwenGatedDeltaInputsCUDAKernel = C.mlx_fast_cuda_kernel_new(
		cName,
		inputs,
		outputs,
		cSource,
		cHeader,
		C.bool(true),
		C.int(0),
	)

	cfg := C.mlx_fast_cuda_kernel_config_new()
	fail := func() {
		C.mlx_fast_cuda_kernel_config_free(cfg)
		qwenGatedDeltaInputsCUDADisabled = true
	}
	cInT := C.CString("InT")
	defer C.free(unsafe.Pointer(cInT))
	if C.mlx_fast_cuda_kernel_config_add_template_arg_dtype(
		cfg, cInT, C.mlx_dtype(DTypeBFloat16),
	) != 0 {
		fail()
		return
	}

	for _, output := range []struct {
		shape []C.int
		dtype DType
	}{
		{shape: []C.int{1, 1, 16, 128}, dtype: DTypeBFloat16},
		{shape: []C.int{1, 1, 16, 128}, dtype: DTypeBFloat16},
		{shape: []C.int{1, 1, 32, 128}, dtype: DTypeBFloat16},
		{shape: []C.int{1, 1, 32}, dtype: DTypeBFloat16},
		{shape: []C.int{1, 1, 32}, dtype: DTypeBFloat16},
	} {
		if C.mlx_fast_cuda_kernel_config_add_output_arg(
			cfg,
			unsafe.SliceData(output.shape),
			C.size_t(len(output.shape)),
			C.mlx_dtype(output.dtype),
		) != 0 {
			fail()
			return
		}
	}
	if C.mlx_fast_cuda_kernel_config_set_grid(cfg, 32, 32, 3) != 0 ||
		C.mlx_fast_cuda_kernel_config_set_thread_group(cfg, 32, 4, 1) != 0 ||
		C.mlx_fast_cuda_kernel_config_set_verbose(cfg, false) != 0 {
		fail()
		return
	}
	qwenGatedDeltaInputsCUDAConfig = cfg
}

func initQwenGatedDeltaOutputCUDAKernel() {
	var cudaAvailable C.bool
	if C.mlx_cuda_is_available(&cudaAvailable) != 0 || !bool(cudaAvailable) {
		qwenGatedDeltaOutputCUDADisabled = true
		return
	}

	inputs, freeInputs, ok := cStringVector(
		[]string{"state_out", "z", "norm_weight"},
	)
	if !ok {
		qwenGatedDeltaOutputCUDADisabled = true
		freeInputs()
		return
	}
	defer freeInputs()

	outputs, freeOutputs, ok := cStringVector([]string{"output"})
	if !ok {
		qwenGatedDeltaOutputCUDADisabled = true
		freeOutputs()
		return
	}
	defer freeOutputs()

	cName := C.CString("qwen_gated_delta_output")
	defer C.free(unsafe.Pointer(cName))
	cSource := C.CString(qwenGatedDeltaOutputCUDAKernelSource)
	defer C.free(unsafe.Pointer(cSource))
	cHeader := C.CString("")
	defer C.free(unsafe.Pointer(cHeader))
	qwenGatedDeltaOutputCUDAKernel = C.mlx_fast_cuda_kernel_new(
		cName,
		inputs,
		outputs,
		cSource,
		cHeader,
		C.bool(true),
		C.int(0),
	)

	cfg := C.mlx_fast_cuda_kernel_config_new()
	fail := func() {
		C.mlx_fast_cuda_kernel_config_free(cfg)
		qwenGatedDeltaOutputCUDADisabled = true
	}
	cInT := C.CString("InT")
	defer C.free(unsafe.Pointer(cInT))
	if C.mlx_fast_cuda_kernel_config_add_template_arg_dtype(
		cfg, cInT, C.mlx_dtype(DTypeBFloat16),
	) != 0 {
		fail()
		return
	}
	shape := []C.int{1, 1, 32, 128}
	if C.mlx_fast_cuda_kernel_config_add_output_arg(
		cfg,
		unsafe.SliceData(shape),
		C.size_t(len(shape)),
		C.mlx_dtype(DTypeBFloat16),
	) != 0 ||
		C.mlx_fast_cuda_kernel_config_set_grid(cfg, 32, 32, 1) != 0 ||
		C.mlx_fast_cuda_kernel_config_set_thread_group(cfg, 32, 4, 1) != 0 ||
		C.mlx_fast_cuda_kernel_config_set_verbose(cfg, false) != 0 {
		fail()
		return
	}
	qwenGatedDeltaOutputCUDAConfig = cfg
}

// FastQwenGatedDeltaInputs fuses the decode-only Qwen gated-delta input
// transforms. No cuDNN or cuBLASLt operation combines the head reshapes,
// normalization, decay, and gate preparation. Unsupported inputs return
// ok=false.
func FastQwenGatedDeltaInputs(
	conv, alpha, beta, aExp, dtBias *Array,
) (q, k, v, g, betaGate *Array, ok bool) {
	if !qwenGatedDeltaInputsCUDAEnabled() ||
		qwenGatedDeltaInputsCUDADisabled ||
		conv == nil || alpha == nil || beta == nil || aExp == nil || dtBias == nil ||
		conv.DType() != DTypeBFloat16 ||
		alpha.DType() != DTypeBFloat16 ||
		beta.DType() != DTypeBFloat16 ||
		aExp.DType() != DTypeFloat32 ||
		dtBias.DType() != DTypeBFloat16 ||
		!sameDims(conv, 1, 1, 8192) ||
		!sameDims(alpha, 1, 1, 32) ||
		!sameDims(beta, 1, 1, 32) ||
		!sameDims(aExp, 32) ||
		!sameDims(dtBias, 32) {
		return nil, nil, nil, nil, nil, false
	}

	qwenGatedDeltaInputsCUDAKernelOnce.Do(initQwenGatedDeltaInputsCUDAKernel)
	if qwenGatedDeltaInputsCUDADisabled {
		return nil, nil, nil, nil, nil, false
	}

	inputData := []C.mlx_array{
		conv.ctx,
		alpha.ctx,
		beta.ctx,
		aExp.ctx,
		dtBias.ctx,
	}
	inputs := C.mlx_vector_array_new_data(
		unsafe.SliceData(inputData),
		C.size_t(len(inputData)),
	)
	defer C.mlx_vector_array_free(inputs)
	outputs := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(outputs)
	if C.mlx_fast_cuda_kernel_apply(
		&outputs,
		qwenGatedDeltaInputsCUDAKernel,
		inputs,
		qwenGatedDeltaInputsCUDAConfig,
		DefaultStream().ctx,
	) != 0 || C.mlx_vector_array_size(outputs) < 5 {
		qwenGatedDeltaInputsCUDADisabled = true
		return nil, nil, nil, nil, nil, false
	}

	result := []*Array{
		New("QWEN_GATED_DELTA_INPUT_Q"),
		New("QWEN_GATED_DELTA_INPUT_K"),
		New("QWEN_GATED_DELTA_INPUT_V"),
		New("QWEN_GATED_DELTA_INPUT_G"),
		New("QWEN_GATED_DELTA_INPUT_BETA"),
	}
	for i := range result {
		if C.mlx_vector_array_get(&result[i].ctx, outputs, C.size_t(i)) != 0 {
			for j := range i {
				C.mlx_array_free(result[j].ctx)
				result[j].ctx = C.mlx_array{}
			}
			qwenGatedDeltaInputsCUDADisabled = true
			return nil, nil, nil, nil, nil, false
		}
	}
	return result[0], result[1], result[2], result[3], result[4], true
}

// FastQwenGatedDeltaOutput fuses the decode-only recurrent output RMSNorm and
// SiLU gate. cuDNN has no grouped RMSNorm epilogue with this independent gate.
// Unsupported inputs return ok=false.
func FastQwenGatedDeltaOutput(
	stateOut, z, normWeight *Array,
) (output *Array, ok bool) {
	if qwenGatedDeltaOutputCUDADisabled ||
		stateOut == nil || z == nil || normWeight == nil ||
		stateOut.DType() != DTypeBFloat16 ||
		z.DType() != DTypeBFloat16 ||
		normWeight.DType() != DTypeBFloat16 ||
		!sameDims(stateOut, 1, 1, 32, 128) ||
		!sameDims(z, 1, 1, 32, 128) ||
		!sameDims(normWeight, 128) {
		return nil, false
	}

	qwenGatedDeltaOutputCUDAKernelOnce.Do(initQwenGatedDeltaOutputCUDAKernel)
	if qwenGatedDeltaOutputCUDADisabled {
		return nil, false
	}

	inputData := []C.mlx_array{stateOut.ctx, z.ctx, normWeight.ctx}
	inputs := C.mlx_vector_array_new_data(
		unsafe.SliceData(inputData),
		C.size_t(len(inputData)),
	)
	defer C.mlx_vector_array_free(inputs)
	outputs := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(outputs)
	if C.mlx_fast_cuda_kernel_apply(
		&outputs,
		qwenGatedDeltaOutputCUDAKernel,
		inputs,
		qwenGatedDeltaOutputCUDAConfig,
		DefaultStream().ctx,
	) != 0 || C.mlx_vector_array_size(outputs) < 1 {
		qwenGatedDeltaOutputCUDADisabled = true
		return nil, false
	}

	output = New("QWEN_GATED_DELTA_OUTPUT")
	if C.mlx_vector_array_get(&output.ctx, outputs, 0) != 0 {
		qwenGatedDeltaOutputCUDADisabled = true
		return nil, false
	}
	return output, true
}

func sameDims(array *Array, dims ...int) bool {
	if array.NumDims() != len(dims) {
		return false
	}
	for i, dim := range dims {
		if array.Dim(i) != dim {
			return false
		}
	}
	return true
}
