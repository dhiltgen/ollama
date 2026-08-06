package mlx

const nvfp4ScaledQMVRowsPerBlock = 8

var nvfp4ScaledQMVKernel = gpuKernel{
	name:    "nvfp4_scaled_qmv",
	inputs:  []string{"x", "weight", "scales", "global_scale"},
	outputs: []string{"output"},
	cuda: gpuSource{
		header: `
#include <cuda_bf16.h>
#undef FP_NAN
#undef FP_INFINITE
#undef FP_ZERO
#undef FP_SUBNORMAL
#undef FP_NORMAL
#include <cutlass/float8.h>
#include <cutlass/numeric_conversion.h>

template <typename T, int N>
struct alignas(sizeof(T) * N) OllamaAlignedVector {
  T values[N];
};

template <int N, typename T>
__device__ __forceinline__ OllamaAlignedVector<T, N>
ollama_load_vector(const T* ptr) {
  return *reinterpret_cast<const OllamaAlignedVector<T, N>*>(ptr);
}
`,
		source: `
constexpr unsigned int FULL_MASK = 0xffffffff;
constexpr int WARP_WIDTH = 32;
constexpr int VALUES_PER_WORD = 8;
constexpr int VALUES_PER_THREAD = 32;
constexpr int WORDS_PER_THREAD = VALUES_PER_THREAD / VALUES_PER_WORD;
constexpr int GROUPS_PER_THREAD = VALUES_PER_THREAD / GROUP_SIZE;

int lane = threadIdx.x;
int row = blockIdx.y * ROWS_PER_BLOCK + threadIdx.y;
int vector_index = blockIdx.x;
if (row < OUTPUT_DIM) {
  const InT* vector = x + vector_index * INPUT_DIM;
  const unsigned int* row_weight =
      weight + row * (INPUT_DIM / VALUES_PER_WORD);
  using ScaleType = cutlass::float_e4m3_t;
  const ScaleType* row_scales =
      reinterpret_cast<const ScaleType*>(scales) +
      row * (INPUT_DIM / GROUP_SIZE) +
      lane * GROUPS_PER_THREAD;

  float sum = 0.0f;
  for (int column_base = VALUES_PER_THREAD * lane;
       column_base < INPUT_DIM;
       column_base += WARP_WIDTH * VALUES_PER_THREAD) {
    auto vector_values = ollama_load_vector<VALUES_PER_THREAD>(
        vector + column_base);
    auto weight_words = ollama_load_vector<WORDS_PER_THREAD>(
        row_weight + column_base / VALUES_PER_WORD);

#pragma unroll
    for (int group = 0; group < GROUPS_PER_THREAD; ++group) {
      float2 partial = {0.0f, 0.0f};
#pragma unroll
      for (int local_word = 0; local_word < GROUP_SIZE / VALUES_PER_WORD;
           ++local_word) {
        int word = group * (GROUP_SIZE / VALUES_PER_WORD) + local_word;
        cutlass::NumericArrayConverter<
            float, cutlass::float_e2m1_t, VALUES_PER_WORD> converter;
        auto weight_values = converter(
            *reinterpret_cast<
                cutlass::Array<cutlass::float_e2m1_t, VALUES_PER_WORD>*>(
                &weight_words.values[word]));
        int value = word * VALUES_PER_WORD;
        partial.x += weight_values[0] *
            static_cast<float>(vector_values.values[value]);
        partial.y += weight_values[1] *
            static_cast<float>(vector_values.values[value + 1]);
        partial.x += weight_values[2] *
            static_cast<float>(vector_values.values[value + 2]);
        partial.y += weight_values[3] *
            static_cast<float>(vector_values.values[value + 3]);
        partial.x += weight_values[4] *
            static_cast<float>(vector_values.values[value + 4]);
        partial.y += weight_values[5] *
            static_cast<float>(vector_values.values[value + 5]);
        partial.x += weight_values[6] *
            static_cast<float>(vector_values.values[value + 6]);
        partial.y += weight_values[7] *
            static_cast<float>(vector_values.values[value + 7]);
      }
      sum += (partial.x + partial.y) * static_cast<float>(row_scales[group]);
    }
    row_scales += WARP_WIDTH * GROUPS_PER_THREAD;
  }

#pragma unroll
  for (int offset = WARP_WIDTH / 2; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(FULL_MASK, sum, offset);
  }
  if (lane == 0) {
    // Match quantized_matmul followed by the FP32 tensor-scale multiply:
    // the QMV result rounds to BF16 before the scale is applied.
    InT rounded = static_cast<InT>(sum);
    output[vector_index * OUTPUT_DIM + row] = static_cast<InT>(
        static_cast<float>(rounded) * global_scale[0]);
  }
}
`,
	},
}

// fastNVFP4ScaledQMV fuses the direct tensor scale into the CUDA decode QMV.
// cuBLASLt's native FP4 MMA path quantizes activations (W4A4), while this model
// requires weight-only W4A16 with its existing group scales and BF16 rounding.
// It was also slower at M=1. Unsupported inputs retain the MLX graph path.
func fastNVFP4ScaledQMV(x, weight, scales, globalScale *Array) (out *Array, ok bool) {
	if x == nil || weight == nil || scales == nil || globalScale == nil ||
		x.DType() != DTypeBFloat16 ||
		weight.DType() != DTypeUint32 ||
		scales.DType() != DTypeUint8 ||
		globalScale.DType() != DTypeFloat32 || globalScale.Size() != 1 ||
		x.NumDims() != 3 || x.Dim(0) != 1 ||
		weight.NumDims() != 2 || scales.NumDims() != 2 {
		return nil, false
	}

	rows := x.Dim(1)
	inputDim := x.Dim(2)
	outputDim := weight.Dim(0)
	if rows != 1 || inputDim%32 != 0 ||
		weight.Dim(1)*8 != inputDim ||
		scales.Dim(0) != outputDim || scales.Dim(1)*16 != inputDim {
		return nil, false
	}

	outs, ok := nvfp4ScaledQMVKernel.applyCUDA(gpuLaunch{
		dtypes: []gpuDTypeArg{{name: "InT", dtype: DTypeBFloat16}},
		ints: []gpuIntArg{
			{name: "INPUT_DIM", value: inputDim},
			{name: "OUTPUT_DIM", value: outputDim},
			{name: "GROUP_SIZE", value: 16},
			{name: "ROWS_PER_BLOCK", value: nvfp4ScaledQMVRowsPerBlock},
		},
		outputs: []gpuOutputSpec{{
			name:  "NVFP4_SCALED_QMV_CUDA",
			shape: []int32{1, int32(rows), int32(outputDim)},
			dtype: DTypeBFloat16,
		}},
		grid: [3]int{
			rows * 32,
			((outputDim + nvfp4ScaledQMVRowsPerBlock - 1) / nvfp4ScaledQMVRowsPerBlock) * nvfp4ScaledQMVRowsPerBlock,
			1,
		},
		threadGroup: [3]int{32, nvfp4ScaledQMVRowsPerBlock, 1},
		inputs:      []*Array{x, weight, scales, globalScale},
	})
	if !ok || len(outs) != 1 {
		return nil, false
	}
	return outs[0], true
}
