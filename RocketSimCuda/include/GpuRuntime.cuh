#pragma once
// ============================================================================
// GpuRuntime.cuh — portable CUDA / HIP runtime for RocketSimCuda kernels
// ----------------------------------------------------------------------------
// NVIDIA (default): real CUDA runtime.
// AMD: define GIGA_GPU_HIP before including this header (CMake does this for
//      windows-hip / linux-rocm presets). Maps the cuda* API surface used by
//      RocketSimCuda onto hip* so the same .cu sources build with hipcc.
// ============================================================================

#include <cstdio>

#if defined(GIGA_GPU_HIP)

#include <hip/hip_runtime.h>

#ifndef GIGA_GPU_RUNTIME_CUDA_ALIASES
#define GIGA_GPU_RUNTIME_CUDA_ALIASES

using cudaError_t = hipError_t;
using cudaMemcpyKind = hipMemcpyKind;
using cudaStream_t = hipStream_t;

#define cudaSuccess                hipSuccess
#define cudaMemcpyHostToDevice     hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost     hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice   hipMemcpyDeviceToDevice
#define cudaMemcpyHostToHost       hipMemcpyHostToHost

#define cudaMalloc                 hipMalloc
#define cudaFree                   hipFree
#define cudaMemcpy                 hipMemcpy
#define cudaMemcpyAsync            hipMemcpyAsync
#define cudaMemcpyToSymbol         hipMemcpyToSymbol
#define cudaMemcpyFromSymbol       hipMemcpyFromSymbol
#define cudaMemset                 hipMemset
#define cudaMemsetAsync            hipMemsetAsync
#define cudaDeviceSynchronize      hipDeviceSynchronize
#define cudaGetLastError           hipGetLastError
#define cudaPeekAtLastError        hipPeekAtLastError
#define cudaGetErrorString         hipGetErrorString
#define cudaSetDevice              hipSetDevice
#define cudaGetDevice              hipGetDevice
#define cudaGetDeviceCount         hipGetDeviceCount
#define cudaDeviceGetAttribute     hipDeviceGetAttribute
#define cudaStreamCreate           hipStreamCreate
#define cudaStreamDestroy          hipStreamDestroy
#define cudaStreamSynchronize      hipStreamSynchronize

#endif // GIGA_GPU_RUNTIME_CUDA_ALIASES

#ifndef GPU_BACKEND_NAME
#define GPU_BACKEND_NAME "HIP"
#endif

#else // !GIGA_GPU_HIP → NVIDIA CUDA

#include <cuda_runtime.h>

#ifndef GPU_BACKEND_NAME
#define GPU_BACKEND_NAME "CUDA"
#endif

#endif // GIGA_GPU_HIP

#ifndef GPU_CHECK
#define GPU_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "RocketSimCuda %s error at %s:%d: %s\n", \
                GPU_BACKEND_NAME, __FILE__, __LINE__, cudaGetErrorString(err__)); \
    } \
} while (0)
#endif

#ifndef CUDA_CHECK
#define CUDA_CHECK GPU_CHECK
#endif
