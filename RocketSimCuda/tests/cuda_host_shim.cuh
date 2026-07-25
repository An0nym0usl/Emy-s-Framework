// ============================================================================
// cuda_host_shim.cuh  —  GPU-less host-compilation shim for RocketSimCuda
// ----------------------------------------------------------------------------
// Purpose (OPT-105): let the *real* RocketSimCuda kernel sources
// (CudaMath.cuh / GpuTypes.cuh / Collision.cuh / CarPhysics.cuh) be compiled
// and executed by a plain C++ host compiler, with NO CUDA toolkit and NO GPU.
//
// It neutralises the CUDA execution-space qualifiers to no-ops and provides
// host implementations of the handful of CUDA device intrinsics the device
// functions call. The device functions are then ordinary C++ functions that
// run on the CPU bit-for-bit as written — this is what makes the parity proof
// exercise the actual kernel algebra rather than a reimplementation.
//
// MUST be included BEFORE any RocketSimCuda header. Do NOT include this when
// compiling with nvcc (it would clobber the real qualifiers); it is guarded so
// it is inert under __CUDACC__.
// ============================================================================
#pragma once

#ifndef __CUDACC__
#ifndef __HIPCC__

// ---- Execution-space / qualifier no-ops -----------------------------------
// On device these control where code runs and is inlined; on the host they
// carry no meaning, so map them away. The device functions keep their exact
// arithmetic — only these annotations disappear.
#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __global__
#define __global__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif
#ifndef __noinline__
#define __noinline__
#endif
#ifndef __constant__
#define __constant__
#endif
#ifndef __shared__
#define __shared__ static
#endif
#ifndef __restrict__
#define __restrict__
#endif
#ifndef __launch_bounds__
#define __launch_bounds__(...)
#endif

#include <cmath>

// ---- Device math intrinsics not present in the host C library --------------
// glibc already provides fminf/fmaxf/fabsf/sqrtf/powf/atan2f/asinf/cosf/sinf,
// so only the reciprocal-sqrt intrinsic needs a host definition.
//
// NOTE (tolerance rationale): CUDA's rsqrtf is the IEEE-rounded reciprocal
// square root; `1.0f/sqrtf(x)` is the faithfully-rounded host equivalent and
// is at worst ~1 ulp from the device result. This is one of the documented,
// bounded float32 divergence sources between this host build and a real GPU
// build — it does not affect the host-vs-reference parity numbers below, which
// compare this host build against a double-precision reference of the same
// algorithm.
#ifndef RSCUDA_HOST_HAS_RSQRTF
#define RSCUDA_HOST_HAS_RSQRTF
static inline float rsqrtf(float x) { return 1.0f / sqrtf(x); }
#endif

#endif // !__HIPCC__
#endif // !__CUDACC__
