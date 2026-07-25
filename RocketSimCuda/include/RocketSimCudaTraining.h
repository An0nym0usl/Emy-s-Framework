#pragma once

#include <cstdint>

namespace rsc {

struct DeviceGAEParams {
    float gamma = 0.99f;
    float lambda = 0.95f;
    float returnStd = 0.f;
    float clipRange = 0.f;
};

// Compute GAE entirely on GPU.
// All pointers must be device pointers. `streamHandle` should be a `cudaStream_t`
// cast to `void*`, or null to use the default stream.
void ComputeGAEOnDevice(
    const float* d_rews,
    const int8_t* d_terminals,
    const float* d_valPreds,
    const float* d_truncValPreds,
    int numTruncs,
    float* d_advantages,
    float* d_targetValues,
    float* d_returns,
    float* d_rewClipPortion,
    int T,
    DeviceGAEParams params,
    void* streamHandle = nullptr
);

} // namespace rsc
