package ADLX

/*
#include "-I${SRCDIR}/../../../build/_deps/adlx-src/SDK/ADLXHelper/Windows/C/ADLXHelper.h"
#include "-I${SRCDIR}/../../../build/_deps/adlx-src/SDK/Include/IPerformanceMonitoring3.h"

typedef const char cchar;

// Helper struct to simplify the Go code for device identification
typedef struct gpuInfo {
	const char *vendorId;
	ADLX_GPU_TYPE gpuType;
	const char *gpuName;
	adlx_uint vramMB;
	const char *deviceId;
	const char *revisionId;
	const char *subSystemId;
	const char *subSystemVendorId;
	adlx_int uniqueId;
} gpuInfo;

// Wrappers since Go can't call function pointers
ADLX_RESULT GetPerformanceMonitoringServices(IADLXSystem* pThis, IADLXPerformanceMonitoringServices** ppPerformanceMonitoringServices) {
	return pThis->pVtbl->GetPerformanceMonitoringServices(pThis, ppPerformanceMonitoringServices);
}
ADLX_RESULT GetGPUs(IADLXSystem* pThis, IADLXGPUList** ppGPUs) {
	return pThis->pVtbl->GetGPUs(pThis, ppGPUs);
}
adlx_uint GPUListBegin(IADLXGPUList* pThis) {
	return pThis->pVtbl->Begin(pThis);
}
adlx_uint GPUListEnd(IADLXGPUList* pThis) {
	return pThis->pVtbl->End(pThis);
}
ADLX_RESULT At_GPUList(IADLXGPUList* pThis, const adlx_uint location, IADLXGPU** ppItem) {
	return pThis->pVtbl->At_GPUList(pThis, location, ppItem);
}
ADLX_RESULT GetGPUInfo(IADLXGPU* pThis, gpuInfo *info) {
	ADLX_RESULT r = ADLX_OK;
	r = pThis->pVtbl->VendorId(pThis, &info->vendorId);
	if (r != ADLX_OK) return r;
	r = pThis->pVtbl->Type(pThis, &info->gpuType);
	if (r != ADLX_OK) return r;
	r = pThis->pVtbl->TotalVRAM(pThis, &info->vramMB);
	if (r != ADLX_OK) return r;
	r = pThis->pVtbl->Name(pThis, &info->gpuName);
	if (r != ADLX_OK) return r;
	r = pThis->pVtbl->DeviceId(pThis, &info->deviceId);
	if (r != ADLX_OK) return r;
	r = pThis->pVtbl->RevisionId(pThis, &info->revisionId);
	if (r != ADLX_OK) return r;
	r = pThis->pVtbl->SubSystemId(pThis, &info->subSystemId);
	if (r != ADLX_OK) return r;
	r = pThis->pVtbl->SubSystemVendorId(pThis, &info->subSystemVendorId);
	if (r != ADLX_OK) return r;
	return pThis->pVtbl->UniqueId(pThis, &info->uniqueId);
}

ADLX_RESULT GetCurrentGPUMetrics(IADLXPerformanceMonitoringServices* pThis, IADLXGPU* pGPU, IADLXGPUMetrics** ppMetrics) {
	return pThis->pVtbl->GetCurrentGPUMetrics(pThis, pGPU, ppMetrics);
}
ADLX_RESULT GetSupportedGPUMetrics(IADLXPerformanceMonitoringServices* pThis, IADLXGPU* pGPU, IADLXGPUMetricsSupport** ppMetricsSupported) {
	return pThis->pVtbl->GetSupportedGPUMetrics(pThis, pGPU, ppMetricsSupported);
}
ADLX_RESULT IsSupportedGPUVRAM(IADLXGPUMetricsSupport* pThis, adlx_bool* supported) {
	return pThis->pVtbl->IsSupportedGPUVRAM(pThis, supported);
}
ADLX_RESULT GPUVRAM(IADLXGPUMetrics* pThis, adlx_int* data) {
	return pThis->pVtbl->GPUVRAM(pThis, data);
}
*/
import "C"
import (
	"fmt"
	"log/slog"
	"unsafe"

	"golang.org/x/sys/windows"
)

var (
	adlxLib                              = windows.NewLazyDLL("amdadlx64.dll")
	ADLXQueryFullVersion                 = adlxLib.NewProc("ADLXQueryFullVersion")
	ADLXQueryVersion                     = adlxLib.NewProc("ADLXQueryVersion")
	ADLXInitializeWithCallerAdl          = adlxLib.NewProc("ADLXInitializeWithCallerAdl")
	ADLXInitializeWithIncompatibleDriver = adlxLib.NewProc("ADLXInitializeWithIncompatibleDriver")
	ADLXInitialize                       = adlxLib.NewProc("ADLXInitialize")
	ADLXTerminate                        = adlxLib.NewProc("ADLXTerminate")
)

// Equivalent to the macros
func adlxSucceeded(x C.ADLX_RESULT) bool {
	return (C.ADLX_OK == (x) || C.ADLX_ALREADY_ENABLED == (x) || C.ADLX_ALREADY_INITIALIZED == (x))
}
func adlxFailed(x C.ADLX_RESULT) bool {
	return (C.ADLX_OK != (x) && C.ADLX_ALREADY_ENABLED != (x) && C.ADLX_ALREADY_INITIALIZED != (x))
}

type Gpu struct {
	ID                int
	Name              string
	VendorID          string
	DeviceID          string
	RevisionID        string
	SubSystemID       string
	SubSystemVendorID string
	Discrete          bool // False for iGPUs
	TotalVRAM         int  // In MB
	UsedVRAM          int
	UniqueID          int // TBD if this is useful for correlation
}

// TODO - unclear what the optimal lifecycle is - when to unload/free, how to efficiently refresh VRAM?

func GetGPUs() ([]Gpu, error) {
	if err := adlxLib.Load(); err != nil {
		return nil, fmt.Errorf("unable to load library %s %w", adlxLib.Name, err)
	}
	if err := ADLXQueryFullVersion.Find(); err != nil {
		return nil, fmt.Errorf("missing ADLXQueryFullVersion: %w", err)
	}
	if err := ADLXInitialize.Find(); err != nil {
		return nil, fmt.Errorf("missing ADLXInitialize: %w", err)
	}
	if err := ADLXInitializeWithIncompatibleDriver.Find(); err != nil {
		return nil, fmt.Errorf("missing ADLXInitializeWithIncompatibleDriver: %w", err)
	}

	// dllHandle := adlxLib.Handle()
	var fullVersion uint64
	res, _, err := ADLXQueryFullVersion.Call(
		(uintptr)(unsafe.Pointer(&fullVersion)),
	)
	if adlxFailed((C.ADLX_RESULT)(res)) {
		return nil, fmt.Errorf("failed to query version: %w", err)
	}

	fmt.Printf("XXX ADLX Version %d.%d.%d (build %d)\n",
		fullVersion>>48&0xf,
		fullVersion>>32&0xf,
		fullVersion>>16&0xf,
		fullVersion&0xf,
	)

	var sys *C.IADLXSystem
	res, _, _ = ADLXInitialize.Call(
		(uintptr)(fullVersion), // C.ADLX_FULL_VERSION
		(uintptr)(unsafe.Pointer(&sys)),
	)
	if adlxFailed((C.ADLX_RESULT)(res)) {
		// Try the other one...
		slog.Info("ADLXInitialize failure", "result", res)
		res, _, _ = ADLXInitializeWithIncompatibleDriver.Call(
			(uintptr)(fullVersion), // C.ADLX_FULL_VERSION
			(uintptr)(unsafe.Pointer(&sys)),
		)
		if adlxFailed((C.ADLX_RESULT)(res)) {
			return nil, fmt.Errorf("ADLXInitializeWithIncompatibleDriver failure: %d", res)
		}
	}
	// fmt.Printf("XXX Initialize res=%d m_pSystemServices=%p err=%s\n", res, sys, err)

	var perfMonitoringService *C.IADLXPerformanceMonitoringServices
	ret := C.GetPerformanceMonitoringServices(sys, &perfMonitoringService)
	if adlxFailed(ret) {
		return nil, fmt.Errorf("failed to fetch PerformanceMonitoringServices: %d %w", res, err)
	}
	// fmt.Printf("XXX PerformanceMonitoringServices %p\n", perfMonitoringService)

	var gpus *C.IADLXGPUList
	ret = C.GetGPUs(sys, &gpus)
	if adlxFailed(ret) {
		return nil, fmt.Errorf("failed to call GetGPUs: %d %w", res, err)
	}
	// fmt.Printf("XXX GetGPUs %p\n", gpus)
	begin := C.GPUListBegin(gpus)
	end := C.GPUListEnd(gpus)
	for i := begin; i < end; i++ {
		var gpu *C.IADLXGPU
		ret = C.At_GPUList(gpus, i, &gpu)
		// fmt.Printf("XXX GPU %d %p ret=%d\n", i, gpus, ret)

		var gpuMetricsSupport *C.IADLXGPUMetricsSupport
		ret = C.GetSupportedGPUMetrics(perfMonitoringService, gpu, &gpuMetricsSupport)
		if adlxFailed(ret) {
			slog.Error("Failed to call GetSupportedGPUMetrics", "status", ret)
			// TODO release things...
			continue
		}

		var gpuMetrics *C.IADLXGPUMetrics
		ret = C.GetCurrentGPUMetrics(perfMonitoringService, gpu, &gpuMetrics)
		// fmt.Printf("XXX GPU Metrics %d %p ret=%d\n", i, gpuMetrics, ret)

		var supported C.adlx_bool
		var usedVRAM C.adlx_int
		ret = C.IsSupportedGPUVRAM(gpuMetricsSupport, &supported)
		if adlxFailed(ret) || supported == 0 {
			slog.Info("GPU VRAM reporting not supported", "device", i)
		} else {
			ret = C.GPUVRAM(gpuMetrics, &usedVRAM)
			if adlxFailed(ret) {
				slog.Error("Failed to call GPUVRAM", "status", ret)
				// TODO release things...
				continue
			}
		}

		var info C.gpuInfo
		ret = C.GetGPUInfo(gpu, &info)
		if adlxFailed(ret) {
			slog.Error("Failed to get GPU Info", "status", ret)
		} else {
			var gpuType string
			switch info.gpuType {
			case C.GPUTYPE_INTEGRATED:
				gpuType = "iGPU"
			case C.GPUTYPE_DISCRETE:
				gpuType = "discrete GPU"
			default:
				gpuType = "unknown"
			}

			fmt.Printf(`XXX GPU [%d] info
    Type              = %s
    VendorID          = %s
    Name              = %s
    DeviceID          = %s
    RevisionID        = %s
    SubSystemID       = %s
    SubSystemVendorID = %s
    UniqueID          = %x
    Total VRAM        = %d MB
    Used VRAM         = %d MB
`, i,
				gpuType,
				C.GoString(info.vendorId),
				C.GoString(info.gpuName),
				C.GoString(info.deviceId),
				C.GoString(info.revisionId),
				C.GoString(info.subSystemId),
				C.GoString(info.subSystemVendorId),
				info.uniqueId,
				info.vramMB,
				usedVRAM,
			)
		}

		// TODO release gpu

	}

	// TODO release gpus
	return nil, nil

}
