#pragma once
// Constant upload + host converters
// Extracted from RocketSimCuda.cu (modular split; same TU via include).

// ============================================================================
// Upload constant memory data
// ============================================================================

static void initConstantMemory(const MutatorConfig& mc, float tickTime) {
    // Build arena surfaces
    ArenaSurface surfaces[MAX_ARENA_SURFACES];
    int numSurf = buildArenaSurfaces(surfaces);

    CUDA_CHECK(cudaMemcpyToSymbol(c_surfaces, surfaces, sizeof(ArenaSurface) * numSurf));
    CUDA_CHECK(cudaMemcpyToSymbol(c_numSurfaces, &numSurf, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_tickTime, &tickTime, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_ballRadius, &mc.ballRadius, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_ballDrag, &mc.ballDrag, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_ballFriction, &mc.ballWorldFriction, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_ballRestitution, &mc.ballWorldRestitution, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_ballMaxSpeed, &mc.ballMaxSpeed, sizeof(float)));

    // Ball inverse inertia (btSphereShape::calculateLocalInertia: 0.4*m*r^2, BT units)
    {
        float radiusBT = mc.ballRadius * (1.f / 50.f);
        float inertia = 0.4f * mc.ballMass * radiusBT * radiusBT;
        float invInertia = 1.f / inertia;
        CUDA_CHECK(cudaMemcpyToSymbol(c_ballInvInertia, &invInertia, sizeof(float)));
    }

    // Combined ball-world contact properties. The arena collision bodies use
    // friction 0.6 / restitution 0.3 (Arena.cpp:505-507); RocketSim's patched
    // static-vs-dynamic combiners are min(friction), max(restitution).
    {
        float combinedFriction = fminf(mc.ballWorldFriction, 0.6f);
        float combinedRestitution = fmaxf(mc.ballWorldRestitution, 0.3f);
        CUDA_CHECK(cudaMemcpyToSymbol(c_ballWorldFrictionCombined, &combinedFriction, sizeof(float)));
        CUDA_CHECK(cudaMemcpyToSymbol(c_ballWorldRestitutionCombined, &combinedRestitution, sizeof(float)));
    }

    CUDA_CHECK(cudaMemcpyToSymbol(c_ballHitExtraForceScale, &mc.ballHitExtraForceScale, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_bumpForceScale, &mc.bumpForceScale, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_bumpCooldownTime, &mc.bumpCooldownTime, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_boostUsedPerSecond, &mc.boostUsedPerSecond, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_boostAccelGround, &mc.boostAccelGround, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_boostAccelAir, &mc.boostAccelAir, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_boostPadCooldownBig, &mc.boostPadCooldown_Big, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_boostPadCooldownSmall, &mc.boostPadCooldown_Small, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_goalThresholdY, &mc.goalBaseThresholdY, sizeof(float)));

    Vec3 grav = {mc.gravity.x, mc.gravity.y, mc.gravity.z};
    CUDA_CHECK(cudaMemcpyToSymbol(c_gravity, &grav, sizeof(Vec3)));
}

static void initTrainingConstantMemory() {
    const int padMap[NUM_BOOST_PADS] = {
        0, 1, 2, 32, 33, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 28, 13,
        14, 29, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 30, 31, 25, 26, 27
    };
    int padMapInv[NUM_BOOST_PADS];
    for (int i = 0; i < NUM_BOOST_PADS; i++)
        padMapInv[i] = padMap[NUM_BOOST_PADS - i - 1];

    CUDA_CHECK(cudaMemcpyToSymbol(c_obsPadMap, padMap, sizeof(padMap)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_obsPadMapInv, padMapInv, sizeof(padMapInv)));

    {
        // VantaObs pad tables: locations in RLGymCPP order + the X-mirror
        // permutation (matches the VantaObs constructor).
        Vec3 vantaLocs[NUM_BOOST_PADS];
        for (int i = 0; i < NUM_BOOST_PADS; i++)
            vantaLocs[i] = BoostPadData::POSITIONS[padMap[i]];

        int vantaMirror[NUM_BOOST_PADS];
        for (int i = 0; i < NUM_BOOST_PADS; i++) {
            int match = i;
            for (int j = 0; j < NUM_BOOST_PADS; j++) {
                if (fabsf(vantaLocs[j].x + vantaLocs[i].x) < 1.f &&
                    fabsf(vantaLocs[j].y - vantaLocs[i].y) < 1.f &&
                    fabsf(vantaLocs[j].z - vantaLocs[i].z) < 1.f) {
                    match = j;
                    break;
                }
            }
            vantaMirror[i] = match;
        }

        CUDA_CHECK(cudaMemcpyToSymbol(c_vantaPadLocations, vantaLocs, sizeof(vantaLocs)));
        CUDA_CHECK(cudaMemcpyToSymbol(c_vantaMirrorPadMap, vantaMirror, sizeof(vantaMirror)));
        CUDA_CHECK(cudaMemcpyToSymbol(c_vantaPadMap, padMap, sizeof(padMap)));
    }

    constexpr float R_B[] = {0.f, 1.f};
    constexpr float R_F[] = {-1.f, 0.f, 1.f};

    std::vector<CarControls> actions;
    for (float throttle : R_F) {
        for (float steer : R_F) {
            for (float boost : R_B) {
                for (float handbrake : R_B) {
                    if (boost == 1.f && throttle != 1.f)
                        continue;
                    actions.push_back({throttle, steer, 0.f, steer, 0.f, false, boost == 1.f, handbrake == 1.f});
                }
            }
        }
    }

    int numGroundActions = (int)actions.size();

    for (float pitch : R_F) {
        for (float yaw : R_F) {
            for (float roll : R_F) {
                for (float jump : R_B) {
                    for (float boost : R_B) {
                        if (jump == 1.f && yaw != 0.f)
                            continue;
                        if (pitch == roll && roll == jump && jump == 0.f)
                            continue;

                        float handbrake = (jump == 1.f) && (pitch != 0.f || yaw != 0.f || roll != 0.f);
                        actions.push_back({boost, yaw, pitch, yaw, roll, jump == 1.f, boost == 1.f, handbrake == 1.f});
                    }
                }
            }
        }
    }

    uint8_t groundMask[DEFAULT_ACTION_COUNT] = {};
    uint8_t airMask[DEFAULT_ACTION_COUNT] = {};
    uint8_t jumpMask[DEFAULT_ACTION_COUNT] = {};
    uint8_t boostMask[DEFAULT_ACTION_COUNT] = {};

    for (int i = 0; i < (int)actions.size(); i++) {
        const CarControls& action = actions[i];
        if (action.jump)
            jumpMask[i] = 1;
        if (action.boost)
            boostMask[i] = 1;
        if (i < numGroundActions)
            groundMask[i] = 1;
        if (i > numGroundActions && !action.jump)
            airMask[i] = 1;
        if (i < numGroundActions) {
            if (action.throttle == (action.boost ? 1.f : 0.f) && ((action.yaw != 0.f) == action.handbrake))
                airMask[i] = 1;
        }
    }

    CUDA_CHECK(cudaMemcpyToSymbol(c_defaultGroundMask, groundMask, sizeof(groundMask)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_defaultAirMask, airMask, sizeof(airMask)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_defaultJumpMask, jumpMask, sizeof(jumpMask)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_defaultBoostMask, boostMask, sizeof(boostMask)));
}

// ============================================================================
// Helper: convert between public and GPU types
// ============================================================================

static GpuCarControls toGpuControls(const CarControls& c) {
    return {c.throttle, c.steer, c.pitch, c.yaw, c.roll, c.jump, c.boost, c.handbrake};
}

static void computeCarInertia(GpuCarState& g) {
    // Must match car_compute_inertia in CarPhysics.cuh (btBoxShape margins).
    float hx = g.config.hitboxSize.x * 0.5f / 50.f;
    float hy = g.config.hitboxSize.y * 0.5f / 50.f;
    float hz = g.config.hitboxSize.z * 0.5f / 50.f;

    constexpr float CONVEX_DISTANCE_MARGIN = 0.04f;
    float minHalf = fminf(hx, fminf(hy, hz));
    float safeMargin = fminf(CONVEX_DISTANCE_MARGIN, 0.1f * minHalf);
    float adjust = -CONVEX_DISTANCE_MARGIN + safeMargin;
    hx += adjust;
    hy += adjust;
    hz += adjust;

    float m = PhysConst::CAR_MASS;
    float lx = 2.f * hx, ly = 2.f * hy, lz = 2.f * hz;
    g.localInertia = {m / 12.f * (ly * ly + lz * lz), m / 12.f * (lx * lx + lz * lz), m / 12.f * (lx * lx + ly * ly)};
    g.invLocalInertia = {1.f / g.localInertia.x, 1.f / g.localInertia.y, 1.f / g.localInertia.z};
}

static void initGpuCarState(GpuCarState& g, uint32_t id, uint8_t team, uint8_t preset, float spawnBoost) {
    memset(&g, 0, sizeof(GpuCarState));
    g.id = id;
    g.team = team;
    g.preset = preset;
    g.config = CarPresets::Get(preset);
    g.boost = spawnBoost;
    g.isOnGround = true;
    g.rotMat = {{1,0,0},{0,1,0},{0,0,1}};
    g.pos = {0, 0, PhysConst::CAR_SPAWN_REST_Z};
    computeCarInertia(g);
}

static CarState gpuCarToPublic(const GpuCarState& g) {
    CarState s;
    memcpy(&s.pos, &g.pos, sizeof(Vec3));
    memcpy(&s.rotMat, &g.rotMat, sizeof(RotMat));
    memcpy(&s.vel, &g.vel, sizeof(Vec3));
    memcpy(&s.angVel, &g.angVel, sizeof(Vec3));
    s.boost = g.boost;
    s.timeSpentBoosting = g.timeSpentBoosting;
    s.isOnGround = g.isOnGround;
    s.hasJumped = g.hasJumped;
    s.hasDoubleJumped = g.hasDoubleJumped;
    s.hasFlipped = g.hasFlipped;
    s.isFlipping = g.isFlipping;
    s.isJumping = g.isJumping;
    memcpy(&s.flipRelTorque, &g.flipRelTorque, sizeof(Vec3));
    s.jumpTime = g.jumpTime;
    s.flipTime = g.flipTime;
    s.airTime = g.airTime;
    s.airTimeSinceJump = g.airTimeSinceJump;
    s.isSupersonic = g.isSupersonic;
    s.supersonicTime = g.supersonicTime;
    s.handbrakeVal = g.handbrakeVal;
    s.isAutoFlipping = g.isAutoFlipping;
    s.autoFlipTimer = g.autoFlipTimer;
    s.autoFlipTorqueScale = g.autoFlipTorqueScale;
    s.isDemoed = g.isDemoed;
    s.demoRespawnTimer = g.demoRespawnTimer;
    s.worldContactHasContact = g.worldContactHasContact;
    memcpy(&s.worldContactNormal, &g.worldContactNormal, sizeof(Vec3));
    s.carContactOtherID = g.carContactOtherID;
    s.carContactCooldownTimer = g.carContactCooldownTimer;
    s.ballHitValid = g.ballHitValid;
    s.ballHitTickCount = g.ballHitTickCount;
    s.team = static_cast<Team>(g.team);
    s.preset = static_cast<CarPreset>(g.preset);
    s.id = g.id;
    s.lastControls = g.lastControls;
    return s;
}

static ArenaInfo toPublicArenaInfo(const GpuArenaState& g) {
    ArenaInfo a;
    a.tickCount = g.tickCount;
    a.numCars = g.numCars;
    a.gameMode = static_cast<GameMode>(g.gameMode);
    a.goalScored = g.goalScored;
    a.goalTeam = g.goalTeam;
    return a;
}
