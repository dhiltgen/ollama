package mlx

// #include <stdlib.h>
// #include "generated.h"
import "C"

import (
	"sync"
	"unsafe"
)

var (
	qwenSharedSwiGLUCUDAKernelOnce sync.Once
	qwenSharedSwiGLUCUDAKernel     C.mlx_fast_cuda_kernel
	qwenSharedSwiGLUCUDAConfig     C.mlx_fast_cuda_kernel_config
	qwenSharedSwiGLUCUDADisabled   bool
)

const qwenSharedSwiGLUCUDAKernelSource = `
constexpr int D = 512;

int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index >= D) {
  return;
}

float gate_value = static_cast<float>(gate_up[index]);
InT exponent = static_cast<InT>(expf(fabsf(gate_value)));
InT denominator = static_cast<InT>(
    1.0f + static_cast<float>(exponent));
InT y_value = static_cast<InT>(
    1.0f / static_cast<float>(denominator));
InT sigmoid = gate_value < 0.0f
    ? y_value
    : static_cast<InT>(1.0f - static_cast<float>(y_value));
InT silu = static_cast<InT>(
    gate_value * static_cast<float>(sigmoid));
output[index] = static_cast<InT>(
    static_cast<float>(silu) *
    static_cast<float>(gate_up[D + index]));
`

func initQwenSharedSwiGLUCUDAKernel() {
	var cudaAvailable C.bool
	if C.mlx_cuda_is_available(&cudaAvailable) != 0 || !bool(cudaAvailable) {
		qwenSharedSwiGLUCUDADisabled = true
		return
	}

	inputs, freeInputs, ok := cStringVector([]string{"gate_up"})
	if !ok {
		qwenSharedSwiGLUCUDADisabled = true
		freeInputs()
		return
	}
	defer freeInputs()

	outputs, freeOutputs, ok := cStringVector([]string{"output"})
	if !ok {
		qwenSharedSwiGLUCUDADisabled = true
		freeOutputs()
		return
	}
	defer freeOutputs()

	cName := C.CString("qwen_shared_swiglu")
	defer C.free(unsafe.Pointer(cName))
	cSource := C.CString(qwenSharedSwiGLUCUDAKernelSource)
	defer C.free(unsafe.Pointer(cSource))
	cHeader := C.CString("")
	defer C.free(unsafe.Pointer(cHeader))
	qwenSharedSwiGLUCUDAKernel = C.mlx_fast_cuda_kernel_new(
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
		qwenSharedSwiGLUCUDADisabled = true
	}
	cInT := C.CString("InT")
	defer C.free(unsafe.Pointer(cInT))
	shape := []C.int{1, 1, 512}
	if C.mlx_fast_cuda_kernel_config_add_template_arg_dtype(
		cfg, cInT, C.mlx_dtype(DTypeBFloat16),
	) != 0 ||
		C.mlx_fast_cuda_kernel_config_add_output_arg(
			cfg,
			unsafe.SliceData(shape),
			C.size_t(len(shape)),
			C.mlx_dtype(DTypeBFloat16),
		) != 0 ||
		C.mlx_fast_cuda_kernel_config_set_grid(cfg, 512, 1, 1) != 0 ||
		C.mlx_fast_cuda_kernel_config_set_thread_group(cfg, 256, 1, 1) != 0 ||
		C.mlx_fast_cuda_kernel_config_set_verbose(cfg, false) != 0 {
		fail()
		return
	}
	qwenSharedSwiGLUCUDAConfig = cfg
}

// FastQwenSharedSwiGLU consumes the packed decode output of Qwen's shared
// gate/up projection. cuBLASLt cannot apply SwiGLU across the two independently
// scaled halves as a matmul epilogue. Unsupported inputs return ok=false.
func FastQwenSharedSwiGLU(gateUp *Array) (*Array, bool) {
	if qwenSharedSwiGLUCUDADisabled ||
		gateUp == nil ||
		gateUp.DType() != DTypeBFloat16 ||
		!exactShape(gateUp, 1, 1, 1024) {
		return nil, false
	}

	qwenSharedSwiGLUCUDAKernelOnce.Do(initQwenSharedSwiGLUCUDAKernel)
	if qwenSharedSwiGLUCUDADisabled {
		return nil, false
	}

	inputData := []C.mlx_array{gateUp.ctx}
	inputs := C.mlx_vector_array_new_data(
		unsafe.SliceData(inputData),
		C.size_t(len(inputData)),
	)
	defer C.mlx_vector_array_free(inputs)
	outputs := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(outputs)
	if C.mlx_fast_cuda_kernel_apply(
		&outputs,
		qwenSharedSwiGLUCUDAKernel,
		inputs,
		qwenSharedSwiGLUCUDAConfig,
		DefaultStream().ctx,
	) != 0 || C.mlx_vector_array_size(outputs) < 1 {
		qwenSharedSwiGLUCUDADisabled = true
		return nil, false
	}

	output := New("QWEN_SHARED_SWIGLU")
	if C.mlx_vector_array_get(&output.ctx, outputs, 0) != 0 {
		qwenSharedSwiGLUCUDADisabled = true
		return nil, false
	}
	return output, true
}
