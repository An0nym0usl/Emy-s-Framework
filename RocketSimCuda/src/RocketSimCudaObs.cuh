#pragma once
// AdvancedObs + VantaObs kernels
// Extracted from RocketSimCuda.cu (modular split; same TU via include).

// ============================================================================
// Training interop kernel (AdvancedObs + DefaultAction masks)
// ----------------------------------------------------------------------------
// The AdvancedObs device helpers (maybe_invert_obs_*, add_advanced_obs_player,
// device_build_advanced_obs) live in TrainingObs.cuh so the GPU-less parity
// harness can exercise the exact same obs-assembly code (OPT-124).
// ============================================================================

__global__ void buildAdvancedObsAndDefaultMasksKernel(
    const GpuCarState* allCars,
    const GpuBallState* allBalls,
    const GpuBoostPadState* allPads,
    const GpuArenaState* allArenas,
    int numArenas,
    int maxCarsPerArena,
    int obsRowSize,
    float* outObs,
    uint8_t* outMasks
) {
    int flatPlayerIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPlayers = numArenas * maxCarsPerArena;
    if (flatPlayerIdx >= totalPlayers) return;

    int arenaIdx = flatPlayerIdx / maxCarsPerArena;
    int carIdx = flatPlayerIdx % maxCarsPerArena;

    float* obsRow = outObs + flatPlayerIdx * obsRowSize;
    uint8_t* maskRow = outMasks + flatPlayerIdx * DEFAULT_ACTION_COUNT;

    const GpuArenaState& arena = allArenas[arenaIdx];
    if (carIdx >= arena.numCars) {
        for (int i = 0; i < obsRowSize; i++)
            obsRow[i] = 0.f;
        for (int i = 0; i < DEFAULT_ACTION_COUNT; i++)
            maskRow[i] = 0;
        return;
    }

    const GpuCarState* cars = &allCars[arenaIdx * maxCarsPerArena];
    const GpuBallState& ball = allBalls[arenaIdx];
    const GpuBoostPadState* pads = &allPads[arenaIdx * BoostPadData::NUM_TOTAL];
    const GpuCarState& self = cars[carIdx];

    bool inv = (self.team == static_cast<uint8_t>(Team::ORANGE));
    const int* padMap = inv ? c_obsPadMapInv : c_obsPadMap;
    device_build_advanced_obs(obsRow, cars, arena.numCars, carIdx, ball, pads, padMap, inv);

    bool jumpAllowed = car_has_flip_or_jump(self) || (self.worldContactHasContact && self.worldContactNormal.z > 0.9f);
    bool useGroundMask = self.isOnGround;
    bool hasBoost = self.boost > 0.f;

    for (int i = 0; i < DEFAULT_ACTION_COUNT; i++) {
        uint8_t available = useGroundMask ? c_defaultGroundMask[i] : c_defaultAirMask[i];
        if (!hasBoost && c_defaultBoostMask[i])
            available = 0;
        if (jumpAllowed && c_defaultJumpMask[i])
            available = 1;
        maskRow[i] = available;
    }
}

// ============================================================================
// VantaObs kernels
// ============================================================================

__global__ void vantaArenaUpdateKernel(
    const GpuCarState* allCars,
    const GpuBallState* allBalls,
    const GpuBoostPadState* allPads,
    const GpuArenaState* allArenas,
    GpuVantaArenaState* vantaArenas,
    GpuVantaTimers* vantaTimers,
    int numArenas,
    int maxCarsPerArena,
    float timeInterval,
    int predictBall
) {
    int arenaIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (arenaIdx >= numArenas)
        return;

    GpuVantaArenaState& va = vantaArenas[arenaIdx];
    const GpuArenaState& arena = allArenas[arenaIdx];
    const GpuCarState* cars = &allCars[arenaIdx * maxCarsPerArena];

    // Ball prediction: roll a copy of the ball forward with the exact same
    // bullet-transcribed ball physics (BallPredTracker's ball-only arena),
    // capturing the 1.0 / 2.0 / 3.0 second slices (ticks 120/240/360).
    if (predictBall) {
        GpuBallState predBall = allBalls[arenaIdx];
        int sliceIdx = 0;
        for (int t = 1; t <= 360; t++) {
            device_bullet_world_step(
                nullptr, 0, predBall,
                c_surfaces, c_numSurfaces, c_meshGrid,
                nullptr, nullptr,
                c_gravity, c_ballDrag,
                c_ballRadius, c_ballInvInertia,
                c_ballWorldFrictionCombined, c_ballWorldRestitutionCombined,
                PhysConst::CARWORLD_COLLISION_FRICTION, PhysConst::CARWORLD_COLLISION_RESTITUTION,
                c_ballHitExtraForceScale, c_bumpForceScale, c_bumpCooldownTime,
                arena.tickCount + (uint64_t)t, c_tickTime);
            device_ball_finish_tick(predBall, c_ballMaxSpeed);

            if (t == 120 || t == 240 || t == 360) {
                va.predPos[sliceIdx] = predBall.pos;
                va.predVel[sliceIdx] = predBall.vel;
                sliceIdx++;
            }
        }
        va.hasPred = 1;
    } else {
        va.hasPred = 0;
    }

    vanta_update_arena(
        va, &vantaTimers[arenaIdx * maxCarsPerArena],
        cars, arena.numCars,
        &allPads[arenaIdx * BoostPadData::NUM_TOTAL],
        c_vantaPadMap, c_vantaPadLocations,
        arena.tickCount, timeInterval);
}

__global__ void vantaObsKernel(
    const GpuCarState* allCars,
    const GpuBallState* allBalls,
    const GpuArenaState* allArenas,
    const GpuVantaArenaState* vantaArenas,
    GpuVantaTimers* vantaTimers,
    int numArenas,
    int maxCarsPerArena,
    int obsRowSize,
    float timeInterval,
    float dodgeDeadzone,
    int onlyClosestOpp,
    float* outObs
) {
    int flatPlayerIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPlayers = numArenas * maxCarsPerArena;
    if (flatPlayerIdx >= totalPlayers)
        return;

    int arenaIdx = flatPlayerIdx / maxCarsPerArena;
    int carIdx = flatPlayerIdx % maxCarsPerArena;

    float* obsRow = outObs + flatPlayerIdx * obsRowSize;
    const GpuArenaState& arena = allArenas[arenaIdx];

    if (carIdx >= arena.numCars) {
        for (int i = 0; i < obsRowSize; i++)
            obsRow[i] = 0.f;
        return;
    }

    const GpuCarState* cars = &allCars[arenaIdx * maxCarsPerArena];
    const GpuBallState& ball = allBalls[arenaIdx];

    // GameState.lastTouchCarID
    uint32_t lastTouchCarID = 0;
    {
        uint64_t bestTick = 0;
        for (int i = 0; i < arena.numCars; i++) {
            if (cars[i].ballHitValid && cars[i].ballHitTickCount != ~0ULL &&
                cars[i].ballHitTickCount >= bestTick) {
                bestTick = cars[i].ballHitTickCount;
                lastTouchCarID = cars[i].id;
            }
        }
    }

    vanta_build_obs(
        obsRow,
        cars, arena.numCars, carIdx,
        ball,
        vantaArenas[arenaIdx],
        &vantaTimers[arenaIdx * maxCarsPerArena],
        c_vantaPadMap, c_vantaPadLocations, c_vantaMirrorPadMap,
        lastTouchCarID,
        arena.tickCount,
        timeInterval, dodgeDeadzone,
        onlyClosestOpp != 0);
}

