package mlx

// #include "generated.h"
// #include <stdlib.h>
import "C"

import (
	"log/slog"
	"sync"
	"unsafe"
)

type Device struct {
	ctx C.mlx_device
}

func (d Device) LogValue() slog.Value {
	str := C.mlx_string_new()
	defer C.mlx_string_free(str)
	C.mlx_device_tostring(&str, d.ctx)
	return slog.StringValue(C.GoString(C.mlx_string_data(str)))
}

var (
	defaultDevice    Device
	defaultDeviceSet bool
	defaultStream    Stream
	defaultStreamSet bool

	cudaCapabilityMu    sync.Mutex
	cudaCapabilityValid bool
	cudaCapabilityMajor int
	cudaCapabilityMinor int
)

func resetDefaultStreamCache() {
	defaultDeviceSet = false
	defaultStreamSet = false

	cudaCapabilityMu.Lock()
	cudaCapabilityValid = false
	cudaCapabilityMajor = 0
	cudaCapabilityMinor = 0
	cudaCapabilityMu.Unlock()

	resetFastQuantizedMatmulBackendCache()
}

func DefaultDevice() Device {
	if !defaultDeviceSet {
		d := C.mlx_device_new()
		C.mlx_get_default_device(&d)
		defaultDevice = Device{d}
		defaultDeviceSet = true
	}

	return defaultDevice
}

// GPUIsAvailable returns true if a GPU device is available.
func GPUIsAvailable() bool {
	dev := C.mlx_device_new_type(C.MLX_GPU, 0)
	defer C.mlx_device_free(dev)
	var avail C.bool
	C.mlx_device_is_available(&avail, dev)
	return bool(avail)
}

// SetDefaultDeviceGPU sets the default MLX device to GPU.
func SetDefaultDeviceGPU() {
	dev := C.mlx_device_new_type(C.MLX_GPU, 0)
	C.mlx_set_default_device(dev)
	C.mlx_device_free(dev)
	resetDefaultStreamCache()
}

func cudaComputeCapability() (major, minor int, ok bool) {
	cudaCapabilityMu.Lock()
	defer cudaCapabilityMu.Unlock()

	if cudaCapabilityValid {
		return cudaCapabilityMajor, cudaCapabilityMinor, true
	}

	dev := DefaultDevice().ctx

	var available C.bool
	if C.mlx_device_is_available(&available, dev) != 0 || !bool(available) {
		return 0, 0, false
	}

	info := C.mlx_device_info_new()
	defer C.mlx_device_info_free(info)
	if C.mlx_device_info_get(&info, dev) != 0 {
		return 0, 0, false
	}

	major, ok = deviceInfoSize(info, "compute_capability_major")
	if !ok {
		return 0, 0, false
	}
	minor, ok = deviceInfoSize(info, "compute_capability_minor")
	if !ok {
		return 0, 0, false
	}

	cudaCapabilityMajor = major
	cudaCapabilityMinor = minor
	cudaCapabilityValid = true
	return major, minor, true
}

func cudaComputeCapabilityAtLeast(major, minor int) bool {
	actualMajor, actualMinor, ok := cudaComputeCapability()
	if !ok {
		return false
	}
	if actualMajor != major {
		return actualMajor > major
	}
	return actualMinor >= minor
}

// CUDAComputeCapability returns the selected CUDA device's major and minor
// compute capability. It reports ok=false on non-CUDA backends.
func CUDAComputeCapability() (major, minor int, ok bool) {
	return cudaComputeCapability()
}

func cudaDeviceMemory() (total, free int, ok bool) {
	dev := DefaultDevice().ctx

	var available C.bool
	if C.mlx_device_is_available(&available, dev) != 0 || !bool(available) {
		return 0, 0, false
	}

	info := C.mlx_device_info_new()
	defer C.mlx_device_info_free(info)
	if C.mlx_device_info_get(&info, dev) != 0 {
		return 0, 0, false
	}

	total, totalOK := deviceInfoSize(info, "total_memory")
	free, freeOK := deviceInfoSize(info, "free_memory")
	return total, free, totalOK && freeOK
}

// CUDADeviceMemory returns the selected CUDA device's total and currently
// driver-free memory. It reports ok=false on non-CUDA backends.
func CUDADeviceMemory() (total, free int, ok bool) {
	return cudaDeviceMemory()
}

func hasCUDAMemoryHeadroom(total, free, active, peak, reserve int) bool {
	if total <= 0 || free < 0 || active < 0 || peak < 0 || reserve < 0 ||
		active > total || peak > total {
		return false
	}
	if free >= reserve {
		return true
	}

	// cudaMemGetInfo excludes memory retained by CUDA's async allocation pool.
	// That memory remains reusable by MLX, so a completed high-water mark is a
	// better fallback pressure signal than raw driver-free memory alone.
	return total-active >= reserve && total-peak >= reserve
}

func deviceInfoSize(info C.mlx_device_info, key string) (int, bool) {
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))

	var value C.size_t
	if C.mlx_device_info_get_size(&value, info, cKey) != 0 {
		return 0, false
	}
	return int(value), true
}

type Stream struct {
	ctx C.mlx_stream
}

func (s Stream) LogValue() slog.Value {
	str := C.mlx_string_new()
	defer C.mlx_string_free(str)
	C.mlx_stream_tostring(&str, s.ctx)
	return slog.StringValue(C.GoString(C.mlx_string_data(str)))
}

func DefaultStream() Stream {
	if !defaultStreamSet {
		s := C.mlx_stream_new()
		C.mlx_get_default_stream(&s, DefaultDevice().ctx)
		defaultStream = Stream{s}
		defaultStreamSet = true
	}

	return defaultStream
}
