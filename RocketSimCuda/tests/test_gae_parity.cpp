// GAE GPU kernel parity test: random reward/terminal/valPred streams, CPU
// reference recurrence (transcribed from GigaLearnCPP GAE.cpp) vs the
// segmented CUDA kernel. Advantages/returns/targets must match EXACTLY
// (same FP ops per element); clip portion within float-sum tolerance.

#include <RocketSimCudaTraining.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <random>

static constexpr int8_t TERM_NONE = 0;
static constexpr int8_t TERM_NORMAL = 1;
static constexpr int8_t TERM_TRUNCATED = 2;

struct CpuGaeOut {
    std::vector<float> advantages, returns, targetValues;
    float clipPortion;
};

static CpuGaeOut CpuReference(
    const std::vector<float>& rews, const std::vector<int8_t>& terminals,
    const std::vector<float>& valPreds, const std::vector<float>& truncValPreds,
    float gamma, float lambda, float returnStd, float clipRange
) {
    int T = (int)rews.size();
    CpuGaeOut out;
    out.advantages.assign(T, 0);
    out.returns.assign(T, 0);
    out.targetValues.assign(T, 0);

    float prevLambda = 0, prevRet = 0;
    int truncCount = 0;
    float totalRew = 0, totalClipped = 0;

    for (int step = T - 1; step >= 0; step--) {
        int8_t terminal = terminals[step];
        float done = terminal == TERM_NORMAL;
        float trunc = terminal == TERM_TRUNCATED;

        float curReward;
        if (returnStd > 0.0f) {
            curReward = rews[step] / returnStd;
            totalRew += std::fabs(curReward);
            if (clipRange > 0)
                curReward = std::fmin(std::fmax(curReward, -clipRange), clipRange);
            totalClipped += std::fabs(curReward);
        } else {
            curReward = rews[step];
            totalRew += std::fabs(curReward);
        }

        float nextValPred;
        if (terminal == TERM_TRUNCATED) {
            nextValPred = truncValPreds[truncCount];
            truncCount++;
        } else if (terminal == TERM_NORMAL) {
            nextValPred = 0;
        } else {
            nextValPred = valPreds[step + 1];
        }

        float predReturn = curReward + gamma * nextValPred * (1 - done);
        float delta = predReturn - valPreds[step];
        float curReturn = rews[step] + prevRet * gamma * (1 - done) * (1 - trunc);
        out.returns[step] = curReturn;

        prevLambda = delta + gamma * lambda * (1 - done) * (1 - trunc) * prevLambda;
        out.advantages[step] = prevLambda;

        prevRet = curReturn;
    }

    for (int step = 0; step < T; step++)
        out.targetValues[step] = valPreds[step] + out.advantages[step];
    out.clipPortion = (totalRew - totalClipped) / std::fmax(totalRew, 1e-7f);
    return out;
}

int main() {
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> rdist(-2.f, 2.f);
    std::uniform_real_distribution<float> u01(0.f, 1.f);

    int failures = 0;

    struct Case { int T; float terminalProb; float returnStd; float clipRange; bool endTerminal; };
    Case cases[] = {
        {200000, 1.f / 300.f, 1.7f, 4.f, true},   // training-like
        {50000, 1.f / 50.f, 0.f, 0.f, true},      // no standardization
        {1000, 0.f, 1.f, 2.f, false},             // single segment, no terminals
        {777, 1.f / 20.f, 1.f, 2.f, false},       // trailing partial segment
        {1, 0.f, 1.f, 2.f, true},                 // tiny
    };

    for (auto& c : cases) {
        int T = c.T;
        std::vector<float> rews(T), valPreds(T);
        std::vector<int8_t> terminals(T, TERM_NONE);
        std::vector<float> truncValPreds;

        for (int i = 0; i < T; i++) {
            rews[i] = rdist(rng);
            valPreds[i] = rdist(rng);
            if (u01(rng) < c.terminalProb)
                terminals[i] = (u01(rng) < 0.4f) ? TERM_TRUNCATED : TERM_NORMAL;
        }
        if (c.endTerminal)
            terminals[T - 1] = TERM_TRUNCATED;
        else if (T > 1)
            terminals[T - 1] = TERM_NONE;  // exercise the trailing segment;
                                           // valPreds[T] is never read because
                                           // step T-1 only reads valPreds[step+1]
                                           // when non-terminal... so pad below.

        // The recurrence reads valPreds[step+1] at non-terminal steps; the CPU
        // reference and kernel both index step+1 <= T-1 only when step < T-1,
        // EXCEPT a non-terminal final step. Keep the final step terminal-free
        // only with a sentinel pad to keep both sides defined identically:
        if (terminals[T - 1] == TERM_NONE)
            valPreds.push_back(0.f);  // CPU side pad

        int numTruncs = 0;
        for (int i = 0; i < T; i++)
            if (terminals[i] == TERM_TRUNCATED) numTruncs++;
        // Backward scan consumes truncValPreds in reverse order of position
        truncValPreds.resize(numTruncs);
        for (auto& v : truncValPreds) v = rdist(rng);

        CpuGaeOut ref = CpuReference(rews, terminals, valPreds, truncValPreds,
                                     0.99f, 0.95f, c.returnStd, c.clipRange);

        // ---- GPU ----
        float *d_rews, *d_valPreds, *d_truncVals, *d_adv, *d_tgt, *d_ret, *d_clip;
        int8_t* d_terms;
        cudaMalloc(&d_rews, T * sizeof(float));
        cudaMalloc(&d_valPreds, (T + 1) * sizeof(float));
        cudaMalloc(&d_truncVals, std::max(numTruncs, 1) * sizeof(float));
        cudaMalloc(&d_adv, T * sizeof(float));
        cudaMalloc(&d_tgt, T * sizeof(float));
        cudaMalloc(&d_ret, T * sizeof(float));
        cudaMalloc(&d_clip, sizeof(float));
        cudaMalloc(&d_terms, T * sizeof(int8_t));
        cudaMemcpy(d_rews, rews.data(), T * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_valPreds, valPreds.data(), valPreds.size() * sizeof(float), cudaMemcpyHostToDevice);
        if (numTruncs > 0)
            cudaMemcpy(d_truncVals, truncValPreds.data(), numTruncs * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_terms, terminals.data(), T * sizeof(int8_t), cudaMemcpyHostToDevice);

        rsc::DeviceGAEParams params = {};
        params.gamma = 0.99f;
        params.lambda = 0.95f;
        params.returnStd = c.returnStd;
        params.clipRange = c.clipRange;

        rsc::ComputeGAEOnDevice(
            d_rews, d_terms, d_valPreds, numTruncs > 0 ? d_truncVals : nullptr,
            numTruncs, d_adv, d_tgt, d_ret, d_clip, T, params, (void*)0 /*default stream*/);

        std::vector<float> gAdv(T), gTgt(T), gRet(T);
        float gClip = 0;
        cudaMemcpy(gAdv.data(), d_adv, T * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(gTgt.data(), d_tgt, T * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(gRet.data(), d_ret, T * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&gClip, d_clip, sizeof(float), cudaMemcpyDeviceToHost);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::printf("CUDA error: %s\n", cudaGetErrorString(err));
            failures++;
        }

        int exactBad = 0;
        float worst = 0;
        for (int i = 0; i < T; i++) {
            if (gAdv[i] != ref.advantages[i] || gRet[i] != ref.returns[i] || gTgt[i] != ref.targetValues[i]) {
                exactBad++;
                worst = std::fmax(worst, std::fabs(gAdv[i] - ref.advantages[i]));
                if (exactBad <= 3)
                    std::printf("  T=%d i=%d adv cpu=%g gpu=%g | ret cpu=%g gpu=%g\n",
                                T, i, ref.advantages[i], gAdv[i], ref.returns[i], gRet[i]);
            }
        }
        float clipDiff = std::fabs(gClip - ref.clipPortion);
        bool ok = exactBad == 0 && clipDiff < 1e-4f;
        std::printf("T=%6d termP=%.4f std=%.1f clip=%.1f -> %s (badVals=%d worst=%g clipDiff=%g)\n",
                    T, c.terminalProb, c.returnStd, c.clipRange,
                    ok ? "PASS" : "FAIL", exactBad, worst, clipDiff);
        if (!ok)
            failures++;

        cudaFree(d_rews); cudaFree(d_valPreds); cudaFree(d_truncVals);
        cudaFree(d_adv); cudaFree(d_tgt); cudaFree(d_ret); cudaFree(d_clip); cudaFree(d_terms);
    }

    std::printf("Overall: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
