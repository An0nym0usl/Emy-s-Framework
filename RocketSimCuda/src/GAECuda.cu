#include <RocketSimCudaTraining.h>
#include <GpuRuntime.cuh>

#include <math.h>

#if !defined(GIGA_GPU_HIP)
#include <cub/cub.cuh>
#endif

namespace rsc {

static constexpr int8_t TERM_NORMAL = 1;
static constexpr int8_t TERM_TRUNCATED = 2;

// The GAE recurrence resets at every terminal (done or truncated), so the
// buffer splits into independent segments [prevTerminal+1 .. terminal]. Each
// segment runs the EXACT same backward arithmetic as the old single-thread
// kernel, just on its own thread - results are bit-identical per element.
// (Only the rewClipPortion metric is reduced with atomics, so its summation
// order can differ at float rounding level; it is a logging-only value.)

__global__ void markTerminalsKernel(
    const int8_t* __restrict__ terminals,
    uint8_t* __restrict__ termFlags,
    int* __restrict__ truncFlags,
    int T
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T)
        return;
    termFlags[i] = terminals[i] != 0 ? 1 : 0;
    truncFlags[i] = terminals[i] == TERM_TRUNCATED ? 1 : 0;
}

__global__ void zeroTotalsKernel(float* totals) {
    totals[0] = 0.0f;  // totalRew
    totals[1] = 0.0f;  // totalClipped
}

__global__ void computeGaeSegmentsKernel(
    const float* __restrict__ rews,
    const int8_t* __restrict__ terminals,
    const float* __restrict__ valPreds,
    const float* __restrict__ truncValPreds,
    const int* __restrict__ segEnds,       // ascending terminal positions
    const int* __restrict__ numSegsPtr,    // number of terminal positions
    const int* __restrict__ truncBefore,   // exclusive prefix count of truncated terminals
    int totalTruncs,
    float* __restrict__ advantages,
    float* __restrict__ returns,
    float* __restrict__ totals,            // [0]=totalRew, [1]=totalClipped
    int T,
    DeviceGAEParams params
) {
    int numSegs = *numSegsPtr;
    // +1 for the (usually empty) trailing run after the last terminal
    int numWork = numSegs + 1;

    for (int s = blockIdx.x * blockDim.x + threadIdx.x; s < numWork; s += gridDim.x * blockDim.x) {
        int start, end;
        if (s < numSegs) {
            end = segEnds[s];
            start = (s == 0) ? 0 : segEnds[s - 1] + 1;
        } else {
            start = (numSegs == 0) ? 0 : segEnds[numSegs - 1] + 1;
            end = T - 1;
        }
        if (start > end)
            continue;

        float prevAdvantage = 0.0f;
        float prevReturn = 0.0f;
        float totalRew = 0.0f;
        float totalClipped = 0.0f;

        for (int step = end; step >= start; --step) {
            int8_t terminal = terminals[step];
            float done = (terminal == TERM_NORMAL) ? 1.0f : 0.0f;
            float trunc = (terminal == TERM_TRUNCATED) ? 1.0f : 0.0f;

            float curReward;
            if (params.returnStd > 0.0f) {
                curReward = rews[step] / params.returnStd;
                totalRew += fabsf(curReward);
                if (params.clipRange > 0.0f)
                    curReward = fminf(fmaxf(curReward, -params.clipRange), params.clipRange);
                totalClipped += fabsf(curReward);
            } else {
                curReward = rews[step];
                totalRew += fabsf(curReward);
            }

            float nextValPred;
            if (terminal == TERM_TRUNCATED) {
                int truncIdx = totalTruncs - truncBefore[step] - 1;
                nextValPred = truncValPreds[truncIdx];
            } else if (terminal == TERM_NORMAL) {
                nextValPred = 0.0f;
            } else {
                nextValPred = valPreds[step + 1];
            }

            float predReturn = curReward + params.gamma * nextValPred * (1.0f - done);
            float delta = predReturn - valPreds[step];

            float notDoneNotTrunc = (1.0f - done) * (1.0f - trunc);
            float curReturn = rews[step] + prevReturn * params.gamma * notDoneNotTrunc;
            returns[step] = curReturn;

            prevAdvantage = delta + params.gamma * params.lambda * notDoneNotTrunc * prevAdvantage;
            advantages[step] = prevAdvantage;

            prevReturn = curReturn;
        }

        atomicAdd(&totals[0], totalRew);
        atomicAdd(&totals[1], totalClipped);
    }
}

__global__ void finalizeGaeKernel(
    const float* __restrict__ valPreds,
    const float* __restrict__ advantages,
    const float* __restrict__ totals,
    float* __restrict__ targetValues,
    float* __restrict__ rewClipPortion,
    int T
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < T)
        targetValues[i] = valPreds[i] + advantages[i];
    if (i == 0) {
        float totalRew = totals[0];
        float totalClipped = totals[1];
        *rewClipPortion = (totalRew - totalClipped) / fmaxf(totalRew, 1e-7f);
    }
}

#if defined(GIGA_GPU_HIP)
// CUB is NVIDIA-only. HIP GAE uses a serial exclusive-scan + compact on the
// device. GAE is not the SPS bottleneck vs physics/obs; correctness first.
__global__ void gaeExclusiveScanSerialKernel(const int* __restrict__ in, int* __restrict__ out, int n) {
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;
    int sum = 0;
    for (int i = 0; i < n; ++i) {
        out[i] = sum;
        sum += in[i];
    }
}

__global__ void gaeCompactFlaggedKernel(
    const uint8_t* __restrict__ flags,
    const int* __restrict__ exclusiveScan,
    int* __restrict__ outIdx,
    int* __restrict__ outCount,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    if (flags[i])
        outIdx[exclusiveScan[i]] = i;
    if (i == n - 1)
        *outCount = exclusiveScan[n - 1] + (flags[n - 1] ? 1 : 0);
}

__global__ void gaeFlagsToIntKernel(const uint8_t* __restrict__ flags, int* __restrict__ asInt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        asInt[i] = flags[i] ? 1 : 0;
}
#endif // GIGA_GPU_HIP

// Grow-only scratch cache (training is a single-threaded loop; buffers are
// reused across iterations and only reallocated if T grows).
struct GaeScratch {
    int capacity = 0;
    uint8_t* termFlags = nullptr;
    int* truncFlags = nullptr;
    int* truncBefore = nullptr;
    int* segEnds = nullptr;
    int* numSegs = nullptr;
    int* numTruncsOut = nullptr;
    float* totals = nullptr;
#if !defined(GIGA_GPU_HIP)
    void* cubTemp = nullptr;
    size_t cubTempBytes = 0;
#else
    int* flagAsInt = nullptr;
    int* termScan = nullptr;
#endif

    void EnsureCapacity(int T) {
        if (T <= capacity)
            return;
        Release();
        capacity = T;
        cudaMalloc(&termFlags, T * sizeof(uint8_t));
        cudaMalloc(&truncFlags, T * sizeof(int));
        cudaMalloc(&truncBefore, T * sizeof(int));
        cudaMalloc(&segEnds, T * sizeof(int));
        cudaMalloc(&numSegs, sizeof(int));
        cudaMalloc(&numTruncsOut, sizeof(int));
        cudaMalloc(&totals, 2 * sizeof(float));

#if !defined(GIGA_GPU_HIP)
        size_t flaggedBytes = 0, scanBytes = 0;
        cub::CountingInputIterator<int> counting(0);
        cub::DeviceSelect::Flagged(
            nullptr, flaggedBytes, counting, termFlags, segEnds, numSegs, T);
        cub::DeviceScan::ExclusiveSum(
            nullptr, scanBytes, truncFlags, truncBefore, T);
        cubTempBytes = flaggedBytes > scanBytes ? flaggedBytes : scanBytes;
        cudaMalloc(&cubTemp, cubTempBytes);
#else
        cudaMalloc(&flagAsInt, T * sizeof(int));
        cudaMalloc(&termScan, T * sizeof(int));
#endif
    }

    void Release() {
        if (termFlags) cudaFree(termFlags);
        if (truncFlags) cudaFree(truncFlags);
        if (truncBefore) cudaFree(truncBefore);
        if (segEnds) cudaFree(segEnds);
        if (numSegs) cudaFree(numSegs);
        if (numTruncsOut) cudaFree(numTruncsOut);
        if (totals) cudaFree(totals);
#if !defined(GIGA_GPU_HIP)
        if (cubTemp) cudaFree(cubTemp);
        cubTemp = nullptr;
        cubTempBytes = 0;
#else
        if (flagAsInt) cudaFree(flagAsInt);
        if (termScan) cudaFree(termScan);
        flagAsInt = nullptr;
        termScan = nullptr;
#endif
        termFlags = nullptr; truncFlags = nullptr; truncBefore = nullptr;
        segEnds = nullptr; numSegs = nullptr; numTruncsOut = nullptr;
        totals = nullptr;
        capacity = 0;
    }
};

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
    void* streamHandle
) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(streamHandle);

    static GaeScratch scratch;
    scratch.EnsureCapacity(T);

    constexpr int BLOCK = 256;
    int gridT = (T + BLOCK - 1) / BLOCK;

    markTerminalsKernel<<<gridT, BLOCK, 0, stream>>>(
        d_terminals, scratch.termFlags, scratch.truncFlags, T);

#if !defined(GIGA_GPU_HIP)
    cub::CountingInputIterator<int> counting(0);
    size_t tempBytes = scratch.cubTempBytes;
    cub::DeviceSelect::Flagged(
        scratch.cubTemp, tempBytes, counting, scratch.termFlags,
        scratch.segEnds, scratch.numSegs, T, stream);

    tempBytes = scratch.cubTempBytes;
    cub::DeviceScan::ExclusiveSum(
        scratch.cubTemp, tempBytes, scratch.truncFlags, scratch.truncBefore, T, stream);
#else
    gaeFlagsToIntKernel<<<gridT, BLOCK, 0, stream>>>(scratch.termFlags, scratch.flagAsInt, T);
    gaeExclusiveScanSerialKernel<<<1, 1, 0, stream>>>(scratch.flagAsInt, scratch.termScan, T);
    gaeCompactFlaggedKernel<<<gridT, BLOCK, 0, stream>>>(
        scratch.termFlags, scratch.termScan, scratch.segEnds, scratch.numSegs, T);
    gaeExclusiveScanSerialKernel<<<1, 1, 0, stream>>>(scratch.truncFlags, scratch.truncBefore, T);
#endif

    zeroTotalsKernel<<<1, 1, 0, stream>>>(scratch.totals);

    int segGrid = 64;
    computeGaeSegmentsKernel<<<segGrid, BLOCK, 0, stream>>>(
        d_rews, d_terminals, d_valPreds, d_truncValPreds,
        scratch.segEnds, scratch.numSegs, scratch.truncBefore, numTruncs,
        d_advantages, d_returns, scratch.totals, T, params);

    finalizeGaeKernel<<<gridT, BLOCK, 0, stream>>>(
        d_valPreds, d_advantages, scratch.totals,
        d_targetValues, d_rewClipPortion, T);
}

} // namespace rsc
