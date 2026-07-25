#pragma once
// ============================================================================
// TrainingObs.cuh — GGL v2 ContinuousV2 + Attention observation builder
// ----------------------------------------------------------------------------
// Single source of truth for the AdvancedObs device path consumed by the
// ContinuousV2 + Attention policy. These device functions are included by BOTH
// the production kernel (RocketSimCuda.cu, buildAdvancedObsAndDefaultMasksKernel)
// and the GPU-less CPU<->CUDA parity harness (tests/cpu_parity_harness.cpp), so
// the parity proof exercises the *actual* obs-assembly code rather than a copy.
//
// Layout / semantics mirror RLGymCPP AdvancedObs::BuildObs (the CPU reference):
//   ball{pos,vel,angVel} (team-inverted, normalized), self.lastControls (the 8
//   continuous ContinuousV2 action values), boost-pad timers (team-mapped),
//   then self / teammates / opponents player blocks.
// ============================================================================

#include "CarPhysics.cuh"   // GpuCarState/GpuBallState/RotMat/Vec3 + rotmat_dot_vec

namespace rsc {

static constexpr float ADV_POS_COEF = 1.f / 5000.f;
static constexpr float ADV_VEL_COEF = 1.f / 2300.f;
static constexpr float ADV_ANG_VEL_COEF = 1.f / 3.f;
static constexpr float OBS_DOUBLEJUMP_MAX_DELAY = 1.25f;

__device__ __forceinline__ Vec3 maybe_invert_obs_vec(Vec3 v, bool inv) {
    if (!inv) return v;
    return {-v.x, -v.y, v.z};
}

__device__ __forceinline__ RotMat maybe_invert_obs_rot(RotMat m, bool inv) {
    if (!inv) return m;
    m.forward = maybe_invert_obs_vec(m.forward, true);
    m.right = maybe_invert_obs_vec(m.right, true);
    m.up = maybe_invert_obs_vec(m.up, true);
    return m;
}

__device__ __forceinline__ bool car_has_flip_or_jump(const GpuCarState& car) {
    return car.isOnGround || (!car.hasFlipped && !car.hasDoubleJumped && car.airTimeSinceJump < OBS_DOUBLEJUMP_MAX_DELAY);
}

__device__ __forceinline__ void obs_write_vec3(float* out, int& idx, Vec3 v) {
    out[idx++] = v.x;
    out[idx++] = v.y;
    out[idx++] = v.z;
}

__device__ __forceinline__ void add_advanced_obs_player(
    float* out, int& idx,
    const GpuCarState& player,
    bool inv,
    Vec3 invBallPos,
    Vec3 invBallVel
) {
    Vec3 pos = maybe_invert_obs_vec(player.pos, inv);
    RotMat rot = maybe_invert_obs_rot(player.rotMat, inv);
    Vec3 vel = maybe_invert_obs_vec(player.vel, inv);
    Vec3 angVel = maybe_invert_obs_vec(player.angVel, inv);

    obs_write_vec3(out, idx, pos * ADV_POS_COEF);
    obs_write_vec3(out, idx, rot.forward);
    obs_write_vec3(out, idx, rot.up);
    obs_write_vec3(out, idx, vel * ADV_VEL_COEF);
    obs_write_vec3(out, idx, angVel * ADV_ANG_VEL_COEF);
    obs_write_vec3(out, idx, rotmat_dot_vec(rot, angVel) * ADV_ANG_VEL_COEF);
    obs_write_vec3(out, idx, rotmat_dot_vec(rot, invBallPos - pos) * ADV_POS_COEF);
    obs_write_vec3(out, idx, rotmat_dot_vec(rot, invBallVel - vel) * ADV_VEL_COEF);

    out[idx++] = player.boost / 100.f;
    out[idx++] = player.isOnGround ? 1.f : 0.f;
    out[idx++] = car_has_flip_or_jump(player) ? 1.f : 0.f;
    out[idx++] = player.isDemoed ? 1.f : 0.f;
    out[idx++] = player.hasJumped ? 1.f : 0.f;
}

// Assemble the full ContinuousV2 obs row for `cars[carIdx]`. `padMap` is the
// team-mapped boost-pad ordering (length NUM_BOOST_PADS) and `inv` is the
// team-inversion flag (ORANGE perspective). Returns the number of floats
// written, so callers can keep the row-size invariant.
__device__ __forceinline__ int device_build_advanced_obs(
    float* obsRow,
    const GpuCarState* cars, int numCars, int carIdx,
    const GpuBallState& ball,
    const GpuBoostPadState* pads, const int* padMap, bool inv
) {
    const GpuCarState& self = cars[carIdx];

    Vec3 invBallPos = maybe_invert_obs_vec(ball.pos, inv);
    Vec3 invBallVel = maybe_invert_obs_vec(ball.vel, inv);
    Vec3 invBallAngVel = maybe_invert_obs_vec(ball.angVel, inv);

    int obsIdx = 0;
    obs_write_vec3(obsRow, obsIdx, invBallPos * ADV_POS_COEF);
    obs_write_vec3(obsRow, obsIdx, invBallVel * ADV_VEL_COEF);
    obs_write_vec3(obsRow, obsIdx, invBallAngVel * ADV_ANG_VEL_COEF);

    obsRow[obsIdx++] = self.lastControls.throttle;
    obsRow[obsIdx++] = self.lastControls.steer;
    obsRow[obsIdx++] = self.lastControls.pitch;
    obsRow[obsIdx++] = self.lastControls.yaw;
    obsRow[obsIdx++] = self.lastControls.roll;
    obsRow[obsIdx++] = self.lastControls.jump ? 1.f : 0.f;
    obsRow[obsIdx++] = self.lastControls.boost ? 1.f : 0.f;
    obsRow[obsIdx++] = self.lastControls.handbrake ? 1.f : 0.f;

    for (int i = 0; i < NUM_BOOST_PADS; i++) {
        const GpuBoostPadState& pad = pads[padMap[i]];
        obsRow[obsIdx++] = pad.isActive ? 1.f : (1.f / (1.f + pad.cooldown));
    }

    add_advanced_obs_player(obsRow, obsIdx, self, inv, invBallPos, invBallVel);

    for (int otherIdx = 0; otherIdx < numCars; otherIdx++) {
        if (otherIdx == carIdx || cars[otherIdx].team != self.team)
            continue;
        add_advanced_obs_player(obsRow, obsIdx, cars[otherIdx], inv, invBallPos, invBallVel);
    }

    for (int otherIdx = 0; otherIdx < numCars; otherIdx++) {
        if (otherIdx == carIdx || cars[otherIdx].team == self.team)
            continue;
        add_advanced_obs_player(obsRow, obsIdx, cars[otherIdx], inv, invBallPos, invBallVel);
    }

    return obsIdx;
}

} // namespace rsc
