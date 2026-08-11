package mlx

import "fmt"

// B and T arrive as runtime scalars rather than template arguments so
// windows of any length share one compiled pipeline; only the channel
// geometry specializes the kernel.
const depthwiseConvSiLUMetalSource = `
auto elem = thread_position_in_grid.x;
int B = dims[0];
int T = dims[1];
uint total = uint(B) * uint(T) * uint(C);
if (elem >= total) {
  return;
}

int c = int(elem % uint(C));
int t = int(elem / uint(C)) % T;
int b = int(elem) / (C * T);
auto in_base = (b * (T + K - 1) + t) * C + c;

float acc = 0.0f;
for (int i = 0; i < K; ++i) {
  WeightT weight_value = w[c * K + i];
  acc += static_cast<float>(x[in_base + i * C]) * static_cast<float>(weight_value);
}

InT conv_out = static_cast<InT>(acc);
InT sigmoid = stable_sigmoid(conv_out);
out[elem] = static_cast<InT>(conv_out * sigmoid);
`

const depthwiseConvSiLUMetalHeader = `
template <typename T>
T stable_sigmoid(T x) {
  auto y = 1 / (1 + metal::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}
`

const depthwiseConvSiLUCUDAKernelPrefix = `
auto elem = blockIdx.x * blockDim.x + threadIdx.x;
int B = dims[0];
int T = dims[1];
unsigned int total =
    static_cast<unsigned int>(B) *
    static_cast<unsigned int>(T) *
    static_cast<unsigned int>(C);
if (elem >= total) {
  return;
}

int c = int(elem % static_cast<unsigned int>(C));
int t = int(elem / static_cast<unsigned int>(C)) % T;
int b = int(elem) / (C * T);
auto in_base = (b * (T + K - 1) + t) * C + c;

float acc = 0.0f;
#pragma unroll
for (int i = 0; i < K; ++i) {
  float xv = static_cast<float>(x[in_base + i * C]);
  WeightT weight_value = w[c * K + i];
  float wv = static_cast<float>(weight_value);
  acc = fmaf(xv, wv, acc);
}
`

const depthwiseConvSiLUCUDAKernelSuffix = `
InT conv_out = static_cast<InT>(acc);
float conv_value = static_cast<float>(conv_out);
InT y = 1 / (1 + expf(fabsf(conv_value)));
InT sigmoid = (conv_out < static_cast<InT>(0.0f))
    ? y
    : static_cast<InT>(1.0f) - y;
out[elem] = conv_out * sigmoid;
`

const depthwiseConvSiLUCUDAKernelSource = depthwiseConvSiLUCUDAKernelPrefix + depthwiseConvSiLUCUDAKernelSuffix

const depthwiseConvBiasSiLUCUDAKernelSource = depthwiseConvSiLUCUDAKernelPrefix + `
acc += static_cast<float>(bias[c]);
` + depthwiseConvSiLUCUDAKernelSuffix

const depthwiseConvSiLURoundTripCUDAKernelSuffix = `
InT conv_out = static_cast<InT>(acc);
float conv_value = static_cast<float>(conv_out);
InT y = 1 / (1 + expf(fabsf(conv_value)));
InT sigmoid = (conv_out < static_cast<InT>(0.0f))
    ? y
    : static_cast<InT>(1.0f) - y;
RoundT rounded = static_cast<RoundT>(conv_out * sigmoid);
out[elem] = static_cast<InT>(rounded);
`

const depthwiseConvSiLURoundTripCUDAKernelSource = depthwiseConvSiLUCUDAKernelPrefix + depthwiseConvSiLURoundTripCUDAKernelSuffix

const depthwiseConvBiasSiLURoundTripCUDAKernelSource = depthwiseConvSiLUCUDAKernelPrefix + `
acc += static_cast<float>(bias[c]);
` + depthwiseConvSiLURoundTripCUDAKernelSuffix

var depthwiseConvSiLU = &gpuKernel{
	name:    "depthwise_conv_silu",
	inputs:  []string{"x", "w", "dims"},
	outputs: []string{"out"},
	metal: gpuSource{
		source: depthwiseConvSiLUMetalSource,
		header: depthwiseConvSiLUMetalHeader,
	},
	cuda: gpuSource{source: depthwiseConvSiLUCUDAKernelSource},
}

var depthwiseConvBiasSiLUCUDA = &gpuKernel{
	name:    "depthwise_conv_bias_silu",
	inputs:  []string{"x", "w", "bias", "dims"},
	outputs: []string{"out"},
	cuda:    gpuSource{source: depthwiseConvBiasSiLUCUDAKernelSource},
}

var depthwiseConvSiLURoundTripCUDA = &gpuKernel{
	name:    "depthwise_conv_silu_roundtrip",
	inputs:  []string{"x", "w", "dims"},
	outputs: []string{"out"},
	cuda:    gpuSource{source: depthwiseConvSiLURoundTripCUDAKernelSource},
}

var depthwiseConvBiasSiLURoundTripCUDA = &gpuKernel{
	name:    "depthwise_conv_bias_silu_roundtrip",
	inputs:  []string{"x", "w", "bias", "dims"},
	outputs: []string{"out"},
	cuda:    gpuSource{source: depthwiseConvBiasSiLURoundTripCUDAKernelSource},
}

func depthwiseConvSiLUGraph(x, w, bias *Array) *Array {
	Cdim, K := int32(w.Dim(0)), int32(w.Dim(1))
	return SiLU(Conv1d(x, Reshape(w, Cdim, K, 1), bias, 1, 0, 1, Cdim))
}

// DepthwiseConvSiLU computes SiLU of a valid depthwise conv: x
// [B, T+K-1, C] and w [C, K] give [B, T, C], each output reading the K
// trailing input rows starting at its own index. An optional bias has shape
// [C]. Inputs that fit a fused kernel's contract run there; anything else runs
// the same computation as graph ops. The accumulator is rounded to the input
// dtype before the activation so the result matches SiLU applied to the graph
// conv's output.
func DepthwiseConvSiLU(x, w, bias *Array, outLen int) *Array {
	return depthwiseConvSiLUApply(x, w, bias, outLen, nil)
}

// DepthwiseConvSiLURoundTrip preserves a model-dtype rounding boundary while
// returning the result in the input dtype. CUDA can perform both casts in the
// fused convolution kernel; unsupported backends retain the equivalent graph.
func DepthwiseConvSiLURoundTrip(x, w, bias *Array, outLen int, roundDType DType) *Array {
	return depthwiseConvSiLUApply(x, w, bias, outLen, &roundDType)
}

func depthwiseConvSiLUApply(x, w, bias *Array, outLen int, roundDType *DType) *Array {
	if x == nil || w == nil || x.NumDims() != 3 || w.NumDims() != 2 {
		panic("mlx.DepthwiseConvSiLU: need x [B, T+K-1, C] and w [C, K]")
	}
	B, Cdim, K := x.Dim(0), x.Dim(2), w.Dim(1)
	if w.Dim(0) != Cdim || K <= 0 || x.Dim(1) != outLen+K-1 {
		panic(fmt.Sprintf("mlx.DepthwiseConvSiLU: shapes x %v, w %v do not fit outLen %d", x.Dims(), w.Dims(), outLen))
	}
	if bias != nil && (bias.NumDims() != 1 || bias.Dim(0) != Cdim) {
		panic(fmt.Sprintf("mlx.DepthwiseConvSiLU: bias shape %v does not fit %d channels", bias.Dims(), Cdim))
	}
	if (x.DType() != DTypeBFloat16 && x.DType() != DTypeFloat32) ||
		(w.DType() != DTypeBFloat16 && w.DType() != DTypeFloat32) {
		return depthwiseConvSiLUGraph(x, w, bias)
	}
	if bias != nil && bias.DType() != DTypeBFloat16 && bias.DType() != DTypeFloat32 {
		return depthwiseConvSiLUGraph(x, w, bias)
	}

	total := B * outLen * Cdim
	launch := gpuLaunch{
		dtypes: []gpuDTypeArg{{"InT", x.DType()}, {"WeightT", w.DType()}},
		ints:   []gpuIntArg{{"C", Cdim}, {"K", K}},
		outputs: []gpuOutputSpec{
			{"DEPTHWISE_CONV_SILU", []int32{int32(B), int32(outLen), int32(Cdim)}, x.DType()},
		},
		grid:        [3]int{(total + 255) / 256 * 256, 1, 1},
		threadGroup: [3]int{256, 1, 1},
		inputs:      []*Array{x, w, NewArrayInt32([]int32{int32(B), int32(outLen)}, []int32{2})},
	}
	// SM120's cuDNN path is faster for these small depthwise launches. SM121's
	// generic path has substantially higher overhead, so use the CUDA kernel
	// only where it has a measured advantage.
	if x.DType() == DTypeFloat32 && cudaComputeCapabilityAtLeast(12, 1) {
		kernel := depthwiseConvSiLU
		if roundDType != nil && *roundDType != x.DType() {
			kernel = depthwiseConvSiLURoundTripCUDA
			launch.dtypes = append(launch.dtypes, gpuDTypeArg{"RoundT", *roundDType})
		}
		if bias != nil {
			if roundDType != nil && *roundDType != x.DType() {
				kernel = depthwiseConvBiasSiLURoundTripCUDA
			} else {
				kernel = depthwiseConvBiasSiLUCUDA
			}
			launch.dtypes = append(launch.dtypes, gpuDTypeArg{"BiasT", bias.DType()})
			launch.inputs = []*Array{x, w, bias, launch.inputs[2]}
		}
		if outs, ok := kernel.applyCUDA(launch); ok {
			return outs[0]
		}
	}
	if roundDType == nil && bias == nil && x.DType() == w.DType() {
		if outs, ok := depthwiseConvSiLU.applyMetal(launch); ok {
			return outs[0]
		}
	}
	out := depthwiseConvSiLUGraph(x, w, bias)
	if roundDType != nil && *roundDType != out.DType() {
		outDType := out.DType()
		out = out.AsType(*roundDType).AsType(outDType)
	}
	return out
}
