#pragma once
// RocketSimCudaBatch host API
// Extracted from RocketSimCuda.cu (modular split; same TU via include).

// ============================================================================
// HOST API IMPLEMENTATION
// ============================================================================

// Use the public namespace types by casting (binary compatible layout)
using PubVec3 = ::rsc::Vec3;

RocketSimCudaBatch::RocketSimCudaBatch() {}

RocketSimCudaBatch::~RocketSimCudaBatch() {
    if (m_initialized) Destroy();
}

void RocketSimCudaBatch::Init(const BatchConfig& config) {
    if (m_initialized) Destroy();

    m_config = config;
    int N = config.numArenas;
    int M = config.maxCarsPerArena;

    // Allocate GPU memory
    CUDA_CHECK(cudaMalloc(&d_carStates,  N * M * sizeof(GpuCarState)));
    CUDA_CHECK(cudaMalloc(&d_ballStates, N * sizeof(GpuBallState)));
    CUDA_CHECK(cudaMalloc(&d_padStates,  N * BoostPadData::NUM_TOTAL * sizeof(GpuBoostPadState)));
    CUDA_CHECK(cudaMalloc(&d_arenaInfos, N * sizeof(GpuArenaState)));
    CUDA_CHECK(cudaMalloc(&d_controls,   N * M * sizeof(GpuCarControls)));
    CUDA_CHECK(cudaMalloc(&d_trainingObs, N * M * GetObsRowSize() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_trainingActionMasks, N * M * DEFAULT_ACTION_COUNT * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_prevCarStates, N * M * sizeof(GpuCarState)));
    CUDA_CHECK(cudaMalloc(&d_prevBallStates, N * sizeof(GpuBallState)));
    CUDA_CHECK(cudaMalloc(&d_trainingRewards, N * M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_trainingTerminals, N * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_noTouchTimers, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rewardAuxStates, N * M * sizeof(GpuRewardAuxState)));
    CUDA_CHECK(cudaMalloc(&d_vantaTimers, N * M * sizeof(GpuVantaTimers)));
    CUDA_CHECK(cudaMalloc(&d_vantaArena, N * sizeof(GpuVantaArenaState)));

    // Zero-initialize
    CUDA_CHECK(cudaMemset(d_carStates,  0, N * M * sizeof(GpuCarState)));
    CUDA_CHECK(cudaMemset(d_ballStates, 0, N * sizeof(GpuBallState)));
    CUDA_CHECK(cudaMemset(d_padStates,  0, N * BoostPadData::NUM_TOTAL * sizeof(GpuBoostPadState)));
    CUDA_CHECK(cudaMemset(d_arenaInfos, 0, N * sizeof(GpuArenaState)));
    CUDA_CHECK(cudaMemset(d_controls,   0, N * M * sizeof(GpuCarControls)));
    CUDA_CHECK(cudaMemset(d_trainingObs, 0, N * M * GetObsRowSize() * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_trainingActionMasks, 0, N * M * DEFAULT_ACTION_COUNT * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemset(d_prevCarStates, 0, N * M * sizeof(GpuCarState)));
    CUDA_CHECK(cudaMemset(d_prevBallStates, 0, N * sizeof(GpuBallState)));
    CUDA_CHECK(cudaMemset(d_trainingRewards, 0, N * M * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_trainingTerminals, 0, N * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemset(d_noTouchTimers, 0, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_rewardAuxStates, 0, N * M * sizeof(GpuRewardAuxState)));
    CUDA_CHECK(cudaMemset(d_vantaTimers, 0, N * M * sizeof(GpuVantaTimers)));
    CUDA_CHECK(cudaMemset(d_vantaArena, 0, N * sizeof(GpuVantaArenaState)));

    // Allocate host staging
    h_carStates.resize(N * M);
    h_ballStates.resize(N);
    h_arenaInfos.resize(N);

    // Upload constant memory
    float tickTime = 1.f / config.tickRate;
    initConstantMemory(config.mutatorConfig, tickTime);
    initTrainingConstantMemory();
    ConfigureTrainingRewards({});
    ConfigureTrainingTerminals({});

    // Load the real arena collision meshes (exact reference geometry for
    // ball-world contacts and suspension raycasts).
    {
        MeshGridView view = {};
        if (config.collisionMeshesPath && config.collisionMeshesPath[0] &&
            config.gameMode == GameMode::SOCCAR) {
            std::vector<MeshTriangle> tris;
            if (loadCollisionMeshes(config.collisionMeshesPath, tris)) {
                MeshGridConfig gcfg;
                std::vector<int> cellStart, cellTris;
                buildMeshGrid(tris, gcfg, cellStart, cellTris);

                CUDA_CHECK(cudaMalloc(&d_meshTris, tris.size() * sizeof(MeshTriangle)));
                CUDA_CHECK(cudaMalloc(&d_meshCellStart, cellStart.size() * sizeof(int)));
                CUDA_CHECK(cudaMalloc(&d_meshCellTris, (cellTris.size() > 0 ? cellTris.size() : 1) * sizeof(int)));
                CUDA_CHECK(cudaMemcpy(d_meshTris, tris.data(), tris.size() * sizeof(MeshTriangle), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_meshCellStart, cellStart.data(), cellStart.size() * sizeof(int), cudaMemcpyHostToDevice));
                if (!cellTris.empty())
                    CUDA_CHECK(cudaMemcpy(d_meshCellTris, cellTris.data(), cellTris.size() * sizeof(int), cudaMemcpyHostToDevice));

                view.cfg = gcfg;
                view.tris = static_cast<const MeshTriangle*>(d_meshTris);
                view.cellStart = static_cast<const int*>(d_meshCellStart);
                view.cellTris = static_cast<const int*>(d_meshCellTris);
                view.numTris = (int)tris.size();

                printf("RocketSimCuda: loaded %d arena triangles (grid %dx%dx%d)\n",
                       view.numTris, gcfg.nx, gcfg.ny, gcfg.nz);
            } else {
                fprintf(stderr, "RocketSimCuda: failed to load collision meshes from %s; "
                                "falling back to analytic arena surfaces\n",
                        config.collisionMeshesPath);
            }
        }
        CUDA_CHECK(cudaMemcpyToSymbol(c_meshGrid, &view, sizeof(view)));
    }

    m_initialized = true;

    // Initialize all arenas
    ResetAllArenas();
}

void RocketSimCudaBatch::Destroy() {
    if (!m_initialized) return;

    CUDA_CHECK(cudaFree(d_carStates));
    CUDA_CHECK(cudaFree(d_ballStates));
    CUDA_CHECK(cudaFree(d_padStates));
    CUDA_CHECK(cudaFree(d_arenaInfos));
    CUDA_CHECK(cudaFree(d_controls));
    if (d_actionLut) CUDA_CHECK(cudaFree(d_actionLut));
    if (d_actionIndices) CUDA_CHECK(cudaFree(d_actionIndices));
    CUDA_CHECK(cudaFree(d_trainingObs));
    CUDA_CHECK(cudaFree(d_trainingActionMasks));
    CUDA_CHECK(cudaFree(d_prevCarStates));
    CUDA_CHECK(cudaFree(d_prevBallStates));
    CUDA_CHECK(cudaFree(d_trainingRewards));
    CUDA_CHECK(cudaFree(d_trainingTerminals));
    CUDA_CHECK(cudaFree(d_noTouchTimers));
    CUDA_CHECK(cudaFree(d_rewardAuxStates));
    CUDA_CHECK(cudaFree(d_vantaTimers));
    CUDA_CHECK(cudaFree(d_vantaArena));
    if (d_meshTris) CUDA_CHECK(cudaFree(d_meshTris));
    if (d_meshCellStart) CUDA_CHECK(cudaFree(d_meshCellStart));
    if (d_meshCellTris) CUDA_CHECK(cudaFree(d_meshCellTris));

    d_carStates = d_ballStates = d_padStates = d_arenaInfos = d_controls = nullptr;
    d_actionLut = d_actionIndices = nullptr;
    m_actionLutCount = 0;
    m_actionIndicesCapacity = 0;
    d_trainingObs = d_trainingActionMasks = nullptr;
    d_prevCarStates = d_prevBallStates = nullptr;
    d_trainingRewards = d_trainingTerminals = d_noTouchTimers = nullptr;
    d_rewardAuxStates = d_vantaTimers = d_vantaArena = nullptr;
    d_meshTris = d_meshCellStart = d_meshCellTris = nullptr;
    m_initialized = false;
}

int RocketSimCudaBatch::AddCar(int arenaIdx, Team team, CarPreset preset) {
    int M = m_config.maxCarsPerArena;

    // Read arena info
    GpuArenaState arenaState;
    CUDA_CHECK(cudaMemcpy(&arenaState,
        static_cast<GpuArenaState*>(d_arenaInfos) + arenaIdx,
        sizeof(GpuArenaState), cudaMemcpyDeviceToHost));

    int carIdx = arenaState.numCars;
    if (carIdx >= M) return -1;

    // Create car state
    GpuCarState carState;
    static uint32_t nextID = 1;
    initGpuCarState(carState, nextID++, static_cast<uint8_t>(team),
                    static_cast<uint8_t>(preset), m_config.mutatorConfig.carSpawnBoostAmount);

    // Kickoff pose: team-local spawn index (RocketSim assigns the same slot to
    // both teams, then mirrors orange with pos *= {-1,-1,1} and yaw += Ï€).
    int teamLocal = 0;
    for (int i = 0; i < carIdx; i++) {
        GpuCarState existing;
        CUDA_CHECK(cudaMemcpy(&existing,
            static_cast<GpuCarState*>(d_carStates) + arenaIdx * M + i,
            sizeof(GpuCarState), cudaMemcpyDeviceToHost));
        if (existing.team == static_cast<uint8_t>(team))
            teamLocal++;
    }
    int spawnIdx = teamLocal % SpawnData::SPAWN_COUNT;
    SpawnData::SpawnPos sp = SpawnData::SpawnSoccar(spawnIdx);
    float px, py, yaw;
    SpawnData::KickoffPoseForTeam(sp, static_cast<int>(team), px, py, yaw);
    carState.pos = {px, py, PhysConst::CAR_SPAWN_REST_Z};
    carState.rotMat = SpawnData::YawToRotMat(yaw);

    // Upload car
    CUDA_CHECK(cudaMemcpy(
        static_cast<GpuCarState*>(d_carStates) + arenaIdx * M + carIdx,
        &carState, sizeof(GpuCarState), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        static_cast<GpuCarState*>(d_prevCarStates) + arenaIdx * M + carIdx,
        &carState, sizeof(GpuCarState), cudaMemcpyHostToDevice));

    // Update arena car count
    arenaState.numCars = carIdx + 1;
    CUDA_CHECK(cudaMemcpy(
        static_cast<GpuArenaState*>(d_arenaInfos) + arenaIdx,
        &arenaState, sizeof(GpuArenaState), cudaMemcpyHostToDevice));

    return carIdx;
}

void RocketSimCudaBatch::SetResetCurriculum(float kickoffW, float fuzzedW, float aerialW, float chaseW) {
    auto clampW = [](float w) -> float {
        if (!(w >= 0.f) || w > 1e6f) return 0.f;
        return w;
    };
    m_resetKickoffW = clampW(kickoffW);
    m_resetFuzzedW = clampW(fuzzedW);
    m_resetAerialW = clampW(aerialW);
    m_resetChaseW = clampW(chaseW);
    if (m_resetKickoffW + m_resetFuzzedW + m_resetAerialW + m_resetChaseW <= 0.f)
        m_resetKickoffW = 1.f;
}

void RocketSimCudaBatch::ResetArena(int arenaIdx) {
    int M = m_config.maxCarsPerArena;

    // Cheap host LCG for curriculum reset variety (not cryptographic).
    auto nextU01 = [this]() -> float {
        m_resetRng = m_resetRng * 1664525u + 1013904223u;
        return (m_resetRng >> 8) * (1.f / 16777216.f);
    };
    auto nextSigned = [&](float amp) -> float {
        return (nextU01() * 2.f - 1.f) * amp;
    };
    auto clampField = [](float v, float lim) -> float {
        if (v > lim) return lim;
        if (v < -lim) return -lim;
        return v;
    };

    // 0=kickoff, 1=fuzzed kickoff, 2=aerial-ish, 3=ball-chase (cars near ball, ground)
    int resetMode = 0;
    {
        const float total = m_resetKickoffW + m_resetFuzzedW + m_resetAerialW + m_resetChaseW;
        float r = nextU01() * total;
        if (r < m_resetKickoffW)
            resetMode = 0;
        else if (r < m_resetKickoffW + m_resetFuzzedW)
            resetMode = 1;
        else if (r < m_resetKickoffW + m_resetFuzzedW + m_resetAerialW)
            resetMode = 2;
        else
            resetMode = 3;
    }

    // Reset arena state
    GpuArenaState arenaState;
    CUDA_CHECK(cudaMemcpy(&arenaState,
        static_cast<GpuArenaState*>(d_arenaInfos) + arenaIdx,
        sizeof(GpuArenaState), cudaMemcpyDeviceToHost));

    arenaState.tickCount = 0;
    arenaState.goalScored = false;
    arenaState.goalTeam = -1;

    // Reward::Reset â€” clear stateful reward memories for this arena
    if (d_rewardAuxStates) {
        CUDA_CHECK(cudaMemset(
            static_cast<GpuRewardAuxState*>(d_rewardAuxStates) + arenaIdx * M,
            0, M * sizeof(GpuRewardAuxState)));
    }

    // ObsBuilder::Reset â€” clear the per-arena VantaObs state
    if (d_vantaTimers) {
        CUDA_CHECK(cudaMemset(
            static_cast<GpuVantaTimers*>(d_vantaTimers) + arenaIdx * M,
            0, M * sizeof(GpuVantaTimers)));
    }
    if (d_vantaArena) {
        CUDA_CHECK(cudaMemset(
            static_cast<GpuVantaArenaState*>(d_vantaArena) + arenaIdx,
            0, sizeof(GpuVantaArenaState)));
    }

    // Reset ball
    GpuBallState ballState;
    memset(&ballState, 0, sizeof(ballState));
    ballState.pos = {0, 0, PhysConst::BALL_REST_Z};
    ballState.rotMat = {{1,0,0},{0,1,0},{0,0,1}};

    if (resetMode == 1) {
        // Fuzzed kickoff: small ball offset + mild velocity (ContinuousV2-style).
        ballState.pos.x = nextSigned(180.f);
        ballState.pos.y = nextSigned(220.f);
        ballState.vel = { nextSigned(90.f), nextSigned(90.f), nextSigned(35.f) };
    } else if (resetMode == 2) {
        // Aerial-ish: elevated ball near midfield for touch/air skill signal.
        ballState.pos = { nextSigned(1400.f), nextSigned(1600.f), 420.f + nextU01() * 360.f };
        ballState.vel = { nextSigned(100.f), nextSigned(100.f), nextSigned(50.f) };
    } else if (resetMode == 3) {
        // Ball-chase: ball in play; cars spawn nearby on ground facing it.
        ballState.pos = { nextSigned(1900.f), nextSigned(2200.f), PhysConst::BALL_REST_Z };
        ballState.vel = { nextSigned(500.f), nextSigned(500.f), nextSigned(80.f) };
    }

    CUDA_CHECK(cudaMemcpy(
        static_cast<GpuBallState*>(d_ballStates) + arenaIdx,
        &ballState, sizeof(GpuBallState), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        static_cast<GpuBallState*>(d_prevBallStates) + arenaIdx,
        &ballState, sizeof(GpuBallState), cudaMemcpyHostToDevice));

    // Shared kickoff slot order for both teams (RocketSim ResetToRandomKickoff).
    int kickoffOrder[SpawnData::SPAWN_COUNT];
    for (int i = 0; i < SpawnData::SPAWN_COUNT; i++)
        kickoffOrder[i] = i;
    for (int i = SpawnData::SPAWN_COUNT - 1; i > 0; i--) {
        int j = (int)(nextU01() * (float)(i + 1));
        if (j < 0) j = 0;
        if (j > i) j = i;
        int tmp = kickoffOrder[i];
        kickoffOrder[i] = kickoffOrder[j];
        kickoffOrder[j] = tmp;
    }

    // Reset cars to spawn positions (aerial mode places one car under the ball).
    const int setupCar = (resetMode == 2 && arenaState.numCars > 0)
        ? (int)(nextU01() * (float)arenaState.numCars) % arenaState.numCars
        : -1;
    int blueSeen = 0, orangeSeen = 0;
    for (int c = 0; c < arenaState.numCars; c++) {
        GpuCarState car;
        CUDA_CHECK(cudaMemcpy(&car,
            static_cast<GpuCarState*>(d_carStates) + arenaIdx * M + c,
            sizeof(GpuCarState), cudaMemcpyDeviceToHost));

        uint32_t id = car.id;
        uint8_t team = car.team;
        uint8_t preset = car.preset;

        initGpuCarState(car, id, team, preset, m_config.mutatorConfig.carSpawnBoostAmount);

        if (resetMode == 3) {
            // Ground chase: every car near ball, facing it â€” VelPlayerToBall has gradient.
            float dx = nextSigned(1.f);
            float dy = nextSigned(1.f);
            float len = sqrtf(dx * dx + dy * dy);
            if (len < 1e-3f) { dx = 1.f; dy = 0.f; len = 1.f; }
            dx /= len; dy /= len;
            const float dist = 450.f + nextU01() * 1400.f;
            car.pos = {
                clampField(ballState.pos.x + dx * dist, 3500.f),
                clampField(ballState.pos.y + dy * dist, 4500.f),
                PhysConst::CAR_SPAWN_REST_Z
            };
            float yaw = atan2f(ballState.pos.y - car.pos.y, ballState.pos.x - car.pos.x);
            car.rotMat = SpawnData::YawToRotMat(yaw);
            const float spd = nextU01() * 600.f;
            const float toBx = ballState.pos.x - car.pos.x;
            const float toBy = ballState.pos.y - car.pos.y;
            const float toLen = sqrtf(toBx * toBx + toBy * toBy);
            if (toLen > 1.f)
                car.vel = { (toBx / toLen) * spd, (toBy / toLen) * spd, 0.f };
            car.isOnGround = true;
            car.boost = 30.f + nextU01() * 70.f;
            if (team == 0) blueSeen++; else orangeSeen++;
        } else if (resetMode == 2 && c == setupCar) {
            float dx = nextSigned(1.f);
            float dy = nextSigned(1.f);
            float len = sqrtf(dx * dx + dy * dy);
            if (len < 1e-3f) { dx = 1.f; dy = 0.f; len = 1.f; }
            dx /= len; dy /= len;
            const float dist = 180.f + nextU01() * 160.f;
            car.pos = {
                ballState.pos.x - dx * dist,
                ballState.pos.y - dy * dist,
                ballState.pos.z - (40.f + nextU01() * 120.f)
            };
            if (car.pos.z < PhysConst::CAR_SPAWN_REST_Z)
                car.pos.z = PhysConst::CAR_SPAWN_REST_Z + 80.f;
            float yaw = atan2f(ballState.pos.y - car.pos.y, ballState.pos.x - car.pos.x);
            car.rotMat = SpawnData::YawToRotMat(yaw);
            const float spd = 500.f + nextU01() * 400.f;
            car.vel = { dx * spd, dy * spd, 80.f + nextU01() * 120.f };
            car.isOnGround = false;
            car.hasJumped = nextU01() < 0.55f;
            car.boost = 55.f + nextU01() * 45.f;
            if (team == 0) blueSeen++; else orangeSeen++;
        } else {
            int teamLocal = (team == 0) ? blueSeen++ : orangeSeen++;
            int spawnIdx = kickoffOrder[teamLocal % SpawnData::SPAWN_COUNT];
            SpawnData::SpawnPos sp = SpawnData::SpawnSoccar(spawnIdx);
            float px, py, yaw;
            SpawnData::KickoffPoseForTeam(sp, (int)team, px, py, yaw);
            car.pos = {px, py, PhysConst::CAR_SPAWN_REST_Z};
            if (resetMode == 1) {
                // Mild positional fuzz only â€” keep mirrored slot + yaw (RocketSim-like).
                car.pos.x += nextSigned(40.f);
                car.pos.y += nextSigned(40.f);
            }
            car.rotMat = SpawnData::YawToRotMat(yaw);
        }

        CUDA_CHECK(cudaMemcpy(
            static_cast<GpuCarState*>(d_carStates) + arenaIdx * M + c,
            &car, sizeof(GpuCarState), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            static_cast<GpuCarState*>(d_prevCarStates) + arenaIdx * M + c,
            &car, sizeof(GpuCarState), cudaMemcpyHostToDevice));
    }

    // Reset boost pads
    GpuBoostPadState defaultPad;
    defaultPad.isActive = true;
    defaultPad.cooldown = 0.f;
    defaultPad.prevLockedCarID = 0;
    defaultPad.curLockedCarID = 0;
    for (int p = 0; p < BoostPadData::NUM_TOTAL; p++) {
        CUDA_CHECK(cudaMemcpy(
            static_cast<GpuBoostPadState*>(d_padStates) + arenaIdx * BoostPadData::NUM_TOTAL + p,
            &defaultPad, sizeof(GpuBoostPadState), cudaMemcpyHostToDevice));
    }

    // Reset controls for this arena so a reset does not immediately replay stale inputs.
    CUDA_CHECK(cudaMemset(
        static_cast<GpuCarControls*>(d_controls) + arenaIdx * M,
        0, M * sizeof(GpuCarControls)));
    CUDA_CHECK(cudaMemset(
        static_cast<float*>(d_trainingRewards) + arenaIdx * M,
        0, M * sizeof(float)));
    CUDA_CHECK(cudaMemset(
        static_cast<uint8_t*>(d_trainingTerminals) + arenaIdx,
        0, sizeof(uint8_t)));
    // Stagger no-touch age so 8k arenas do not all expire on the same env step
    // (was a ~4s Collection hitch every ~12.5 iters at noTouch=10s / steps=16).
    {
        float initialNoTouch = 0.f;
        const float timeout = m_trainingTerminalConfig.noTouchTimeoutSeconds;
        if (timeout > 0.f)
            initialNoTouch = nextU01() * timeout * 0.9f;
        CUDA_CHECK(cudaMemcpy(
            static_cast<float*>(d_noTouchTimers) + arenaIdx,
            &initialNoTouch, sizeof(float), cudaMemcpyHostToDevice));
    }

    // Upload arena state
    CUDA_CHECK(cudaMemcpy(
        static_cast<GpuArenaState*>(d_arenaInfos) + arenaIdx,
        &arenaState, sizeof(GpuArenaState), cudaMemcpyHostToDevice));
}

void RocketSimCudaBatch::ResetAllArenas() {
    for (int i = 0; i < m_config.numArenas; i++)
        ResetArena(i);

    CUDA_CHECK(cudaMemset(
        d_controls, 0,
        m_config.numArenas * m_config.maxCarsPerArena * sizeof(GpuCarControls)));
    CUDA_CHECK(cudaMemset(
        d_trainingRewards, 0,
        m_config.numArenas * m_config.maxCarsPerArena * sizeof(float)));
    CUDA_CHECK(cudaMemset(
        d_trainingTerminals, 0,
        m_config.numArenas * sizeof(uint8_t)));
    // Keep per-arena staggered no-touch ages from ResetArena (do NOT zero â€” that
    // re-synchronized mass timeouts and tanked sustained Overall median).
}


__global__ void expandDiscreteActionsKernel(
    const int* __restrict__ indices,
    const GpuCarControls* __restrict__ lut,
    GpuCarControls* __restrict__ controls,
    int count,
    int actionCount)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int idx = indices[i];
    if (idx < 0) idx = 0;
    if (actionCount > 0 && idx >= actionCount) idx = actionCount - 1;
    controls[i] = lut[idx];
}

void RocketSimCudaBatch::SetCarControls(int arenaIdx, int carIdx, const CarControls& controls) {
    int M = m_config.maxCarsPerArena;
    int idx = arenaIdx * M + carIdx;
    GpuCarControls gc = toGpuControls(controls);
    CUDA_CHECK(cudaMemcpy(
        static_cast<GpuCarControls*>(d_controls) + idx,
        &gc, sizeof(GpuCarControls), cudaMemcpyHostToDevice));
}

void RocketSimCudaBatch::SetAllCarControls(const CarControls* controls) {
    int total = m_config.numArenas * m_config.maxCarsPerArena;
    CUDA_CHECK(cudaMemcpy(
        d_controls, controls,
        total * sizeof(GpuCarControls), cudaMemcpyHostToDevice));
}

void RocketSimCudaBatch::UploadDiscreteActionLut(const CarControls* lut, int actionCount) {
    if (!m_initialized || !lut || actionCount <= 0) return;
    if (d_actionLut && m_actionLutCount != actionCount) {
        CUDA_CHECK(cudaFree(d_actionLut));
        d_actionLut = nullptr;
        m_actionLutCount = 0;
    }
    if (!d_actionLut) {
        CUDA_CHECK(cudaMalloc(&d_actionLut, (size_t)actionCount * sizeof(GpuCarControls)));
        m_actionLutCount = actionCount;
    }
    CUDA_CHECK(cudaMemcpy(
        d_actionLut, lut,
        (size_t)actionCount * sizeof(GpuCarControls), cudaMemcpyHostToDevice));
}

void RocketSimCudaBatch::SetDiscreteActionsFromIndices(const int* indices, int count) {
    if (!m_initialized || !indices || count <= 0) return;
    if (!d_actionLut || m_actionLutCount <= 0) {
        // No LUT uploaded â€” leave controls unchanged.
        return;
    }
    if (count > m_actionIndicesCapacity) {
        if (d_actionIndices) CUDA_CHECK(cudaFree(d_actionIndices));
        CUDA_CHECK(cudaMalloc(&d_actionIndices, (size_t)count * sizeof(int)));
        m_actionIndicesCapacity = count;
    }
    CUDA_CHECK(cudaMemcpy(
        d_actionIndices, indices,
        (size_t)count * sizeof(int), cudaMemcpyHostToDevice));
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    expandDiscreteActionsKernel<<<blocks, threads>>>(
        static_cast<const int*>(d_actionIndices),
        static_cast<const GpuCarControls*>(d_actionLut),
        static_cast<GpuCarControls*>(d_controls),
        count,
        m_actionLutCount);
    CUDA_CHECK(cudaGetLastError());
}

void RocketSimCudaBatch::SetDiscreteActionsFromDevice(const int* deviceIndices, int count) {
    if (!m_initialized || !deviceIndices || count <= 0) return;
    if (!d_actionLut || m_actionLutCount <= 0)
        return;
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    expandDiscreteActionsKernel<<<blocks, threads>>>(
        deviceIndices,
        static_cast<const GpuCarControls*>(d_actionLut),
        static_cast<GpuCarControls*>(d_controls),
        count,
        m_actionLutCount);
    CUDA_CHECK(cudaGetLastError());
}



void RocketSimCudaBatch::Step(int ticksToSimulate) {
    if (!m_initialized || ticksToSimulate <= 0) return;

    int N = m_config.numArenas;
    int M = m_config.maxCarsPerArena;
    int threadsPerBlock = 128;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    stepArenaKernel<<<numBlocks, threadsPerBlock>>>(
        static_cast<GpuCarState*>(d_carStates),
        static_cast<GpuBallState*>(d_ballStates),
        static_cast<GpuBoostPadState*>(d_padStates),
        static_cast<GpuArenaState*>(d_arenaInfos),
        static_cast<GpuCarControls*>(d_controls),
        N, M, ticksToSimulate
    );

    CUDA_CHECK(cudaGetLastError());
    if (m_kernelSync)
        CUDA_CHECK(cudaDeviceSynchronize());
}

void RocketSimCudaBatch::BuildAdvancedObsAndDefaultMasks() {
    if (!m_initialized) return;

    int totalPlayers = m_config.numArenas * m_config.maxCarsPerArena;
    int threadsPerBlock = 128;
    int numBlocks = (totalPlayers + threadsPerBlock - 1) / threadsPerBlock;

    buildAdvancedObsAndDefaultMasksKernel<<<numBlocks, threadsPerBlock>>>(
        static_cast<GpuCarState*>(d_carStates),
        static_cast<GpuBallState*>(d_ballStates),
        static_cast<GpuBoostPadState*>(d_padStates),
        static_cast<GpuArenaState*>(d_arenaInfos),
        m_config.numArenas,
        m_config.maxCarsPerArena,
        GetAdvancedObsSize(),
        static_cast<float*>(d_trainingObs),
        static_cast<uint8_t*>(d_trainingActionMasks)
    );

    CUDA_CHECK(cudaGetLastError());
    if (m_kernelSync)
        CUDA_CHECK(cudaDeviceSynchronize());
}

void RocketSimCudaBatch::BuildVantaObs() {
    if (!m_initialized) return;

    float timeInterval = (float)m_config.vantaTickSkip / 120.f;

    {
        int threadsPerBlock = 64;
        int numBlocks = (m_config.numArenas + threadsPerBlock - 1) / threadsPerBlock;
        vantaArenaUpdateKernel<<<numBlocks, threadsPerBlock>>>(
            static_cast<GpuCarState*>(d_carStates),
            static_cast<GpuBallState*>(d_ballStates),
            static_cast<GpuBoostPadState*>(d_padStates),
            static_cast<GpuArenaState*>(d_arenaInfos),
            static_cast<GpuVantaArenaState*>(d_vantaArena),
            static_cast<GpuVantaTimers*>(d_vantaTimers),
            m_config.numArenas,
            m_config.maxCarsPerArena,
            timeInterval,
            m_config.vantaPredictBall ? 1 : 0
        );
        CUDA_CHECK(cudaGetLastError());
    }

    {
        int totalPlayers = m_config.numArenas * m_config.maxCarsPerArena;
        int threadsPerBlock = 128;
        int numBlocks = (totalPlayers + threadsPerBlock - 1) / threadsPerBlock;
        vantaObsKernel<<<numBlocks, threadsPerBlock>>>(
            static_cast<GpuCarState*>(d_carStates),
            static_cast<GpuBallState*>(d_ballStates),
            static_cast<GpuArenaState*>(d_arenaInfos),
            static_cast<GpuVantaArenaState*>(d_vantaArena),
            static_cast<GpuVantaTimers*>(d_vantaTimers),
            m_config.numArenas,
            m_config.maxCarsPerArena,
            GetObsRowSize(),
            timeInterval,
            m_config.vantaDodgeDeadzone,
            m_config.vantaOnlyClosestOpp ? 1 : 0,
            static_cast<float*>(d_trainingObs)
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

void RocketSimCudaBatch::ConfigureTrainingRewards(const TrainingRewardConfig& config) {
    m_trainingRewardConfig = config;
    int numEntries = config.numEntries;
    if (numEntries < 0)
        numEntries = 0;
    if (numEntries > MAX_TRAINING_REWARD_ENTRIES)
        numEntries = MAX_TRAINING_REWARD_ENTRIES;
    m_trainingRewardConfig.numEntries = numEntries;
    CUDA_CHECK(cudaMemcpyToSymbol(c_trainingRewardEntries, config.entries, numEntries * sizeof(TrainingRewardEntry)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_numTrainingRewardEntries, &numEntries, sizeof(int)));
}

void RocketSimCudaBatch::ConfigureTrainingTerminals(const TrainingTerminalConfig& config) {
    m_trainingTerminalConfig = config;
    CUDA_CHECK(cudaMemcpyToSymbol(c_trainingTerminalConfig, &config, sizeof(TrainingTerminalConfig)));
    // After timeout is known, spread ages so the first no-touch wave is not a synchronized
    // 8k-arena reset (ResetAllArenas runs before terminals are configured).
    if (m_initialized && config.noTouchTimeoutSeconds > 0.f && d_noTouchTimers) {
        const int N = m_config.numArenas;
        std::vector<float> ages((size_t)N);
        for (int i = 0; i < N; i++) {
            m_resetRng = m_resetRng * 1664525u + 1013904223u;
            const float u = (m_resetRng >> 8) * (1.f / 16777216.f);
            ages[(size_t)i] = u * config.noTouchTimeoutSeconds * 0.9f;
        }
        CUDA_CHECK(cudaMemcpy(
            d_noTouchTimers, ages.data(), (size_t)N * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void RocketSimCudaBatch::SnapshotTrainingState() {
    if (!m_initialized) return;

    int totalCars = m_config.numArenas * m_config.maxCarsPerArena;
    CUDA_CHECK(cudaMemcpy(d_prevCarStates, d_carStates, totalCars * sizeof(GpuCarState), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_prevBallStates, d_ballStates, m_config.numArenas * sizeof(GpuBallState), cudaMemcpyDeviceToDevice));
}

void RocketSimCudaBatch::BuildRewardsAndTerminals(int stepTicks) {
    if (!m_initialized || stepTicks <= 0) return;

    int threadsPerBlock = 128;
    int numBlocks = (m_config.numArenas + threadsPerBlock - 1) / threadsPerBlock;
    buildRewardsAndTerminalsKernel<<<numBlocks, threadsPerBlock>>>(
        static_cast<GpuCarState*>(d_carStates),
        static_cast<GpuCarState*>(d_prevCarStates),
        static_cast<GpuBallState*>(d_ballStates),
        static_cast<GpuBallState*>(d_prevBallStates),
        static_cast<GpuArenaState*>(d_arenaInfos),
        static_cast<GpuRewardAuxState*>(d_rewardAuxStates),
        m_config.numArenas,
        m_config.maxCarsPerArena,
        stepTicks,
        static_cast<float*>(d_trainingRewards),
        static_cast<uint8_t*>(d_trainingTerminals),
        static_cast<float*>(d_noTouchTimers)
    );

    CUDA_CHECK(cudaGetLastError());
    if (m_kernelSync)
        CUDA_CHECK(cudaDeviceSynchronize());
}

void RocketSimCudaBatch::CopyBuiltAdvancedObs(float* outObs) const {
    if (!m_initialized || outObs == nullptr) return;
    int totalFloats = m_config.numArenas * m_config.maxCarsPerArena * GetObsRowSize();
    CUDA_CHECK(cudaMemcpy(outObs, d_trainingObs, totalFloats * sizeof(float), cudaMemcpyDeviceToHost));
}

void RocketSimCudaBatch::CopyBuiltDefaultActionMasks(uint8_t* outMasks) const {
    if (!m_initialized || outMasks == nullptr) return;
    int totalMasks = m_config.numArenas * m_config.maxCarsPerArena * DEFAULT_ACTION_COUNT;
    CUDA_CHECK(cudaMemcpy(outMasks, d_trainingActionMasks, totalMasks * sizeof(uint8_t), cudaMemcpyDeviceToHost));
}

void RocketSimCudaBatch::CopyBuiltRewards(float* outRewards) const {
    if (!m_initialized || outRewards == nullptr) return;
    int totalRewards = m_config.numArenas * m_config.maxCarsPerArena;
    CUDA_CHECK(cudaMemcpy(outRewards, d_trainingRewards, totalRewards * sizeof(float), cudaMemcpyDeviceToHost));
}

void RocketSimCudaBatch::CopyBuiltTerminals(uint8_t* outTerminals) const {
    if (!m_initialized || outTerminals == nullptr) return;
    CUDA_CHECK(cudaMemcpy(outTerminals, d_trainingTerminals, m_config.numArenas * sizeof(uint8_t), cudaMemcpyDeviceToHost));
}

void RocketSimCudaBatch::Synchronize() const {
    if (!m_initialized) return;
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ---- State getters ----

CarState RocketSimCudaBatch::GetCarState(int arenaIdx, int carIdx) const {
    int idx = arenaIdx * m_config.maxCarsPerArena + carIdx;
    GpuCarState g;
    CUDA_CHECK(cudaMemcpy(&g, static_cast<GpuCarState*>(d_carStates) + idx,
                          sizeof(GpuCarState), cudaMemcpyDeviceToHost));
    return gpuCarToPublic(g);
}

int RocketSimCudaBatch::DebugCopyCarInternal(int arenaIdx, int carIdx, void* dst, int maxBytes) const {
    int idx = arenaIdx * m_config.maxCarsPerArena + carIdx;
    int bytes = (maxBytes < (int)sizeof(GpuCarState)) ? maxBytes : (int)sizeof(GpuCarState);
    CUDA_CHECK(cudaMemcpy(dst, static_cast<GpuCarState*>(d_carStates) + idx,
                          bytes, cudaMemcpyDeviceToHost));
    return bytes;
}

void RocketSimCudaBatch::GetAllCarStates(CarState* outStates) const {
    int total = m_config.numArenas * m_config.maxCarsPerArena;
    std::vector<GpuCarState> gpuStates(total);
    CUDA_CHECK(cudaMemcpy(gpuStates.data(), d_carStates,
                          total * sizeof(GpuCarState), cudaMemcpyDeviceToHost));
    for (int i = 0; i < total; i++)
        outStates[i] = gpuCarToPublic(gpuStates[i]);
}

BallState RocketSimCudaBatch::GetBallState(int arenaIdx) const {
    GpuBallState g;
    CUDA_CHECK(cudaMemcpy(&g, static_cast<GpuBallState*>(d_ballStates) + arenaIdx,
                          sizeof(GpuBallState), cudaMemcpyDeviceToHost));
    BallState s;
    memcpy(&s.pos, &g.pos, sizeof(Vec3));
    memcpy(&s.rotMat, &g.rotMat, sizeof(RotMat));
    memcpy(&s.vel, &g.vel, sizeof(Vec3));
    memcpy(&s.angVel, &g.angVel, sizeof(Vec3));
    return s;
}

void RocketSimCudaBatch::GetAllBallStates(BallState* outStates) const {
    int N = m_config.numArenas;
    std::vector<GpuBallState> gpuStates(N);
    CUDA_CHECK(cudaMemcpy(gpuStates.data(), d_ballStates,
                          N * sizeof(GpuBallState), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; i++) {
        memcpy(&outStates[i].pos, &gpuStates[i].pos, sizeof(Vec3));
        memcpy(&outStates[i].rotMat, &gpuStates[i].rotMat, sizeof(RotMat));
        memcpy(&outStates[i].vel, &gpuStates[i].vel, sizeof(Vec3));
        memcpy(&outStates[i].angVel, &gpuStates[i].angVel, sizeof(Vec3));
    }
}

void RocketSimCudaBatch::GetBoostPadStates(int arenaIdx, BoostPadState* outStates) const {
    std::vector<GpuBoostPadState> gpuPads(BoostPadData::NUM_TOTAL);
    CUDA_CHECK(cudaMemcpy(gpuPads.data(),
        static_cast<GpuBoostPadState*>(d_padStates) + arenaIdx * BoostPadData::NUM_TOTAL,
        BoostPadData::NUM_TOTAL * sizeof(GpuBoostPadState), cudaMemcpyDeviceToHost));
    for (int i = 0; i < BoostPadData::NUM_TOTAL; i++) {
        outStates[i].isActive = gpuPads[i].isActive;
        outStates[i].cooldown = gpuPads[i].cooldown;
    }
}

void RocketSimCudaBatch::GetAllBoostPadStates(BoostPadState* outStates) const {
    int total = m_config.numArenas * BoostPadData::NUM_TOTAL;
    std::vector<GpuBoostPadState> gpuPads(total);
    CUDA_CHECK(cudaMemcpy(gpuPads.data(), d_padStates, total * sizeof(GpuBoostPadState), cudaMemcpyDeviceToHost));
    for (int i = 0; i < total; i++) {
        outStates[i].isActive = gpuPads[i].isActive;
        outStates[i].cooldown = gpuPads[i].cooldown;
    }
}

ArenaInfo RocketSimCudaBatch::GetArenaInfo(int arenaIdx) const {
    GpuArenaState g;
    CUDA_CHECK(cudaMemcpy(&g, static_cast<GpuArenaState*>(d_arenaInfos) + arenaIdx,
                          sizeof(GpuArenaState), cudaMemcpyDeviceToHost));
    return toPublicArenaInfo(g);
}

void RocketSimCudaBatch::GetAllArenaInfos(ArenaInfo* outInfos) const {
    int N = m_config.numArenas;
    std::vector<GpuArenaState> gpuInfos(N);
    CUDA_CHECK(cudaMemcpy(gpuInfos.data(), d_arenaInfos,
                          N * sizeof(GpuArenaState), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; i++)
        outInfos[i] = toPublicArenaInfo(gpuInfos[i]);
}

// ---- State setters ----

void RocketSimCudaBatch::SetCarState(int arenaIdx, int carIdx, const CarState& state) {
    int idx = arenaIdx * m_config.maxCarsPerArena + carIdx;

    GpuCarState g;
    CUDA_CHECK(cudaMemcpy(&g, static_cast<GpuCarState*>(d_carStates) + idx,
                          sizeof(GpuCarState), cudaMemcpyDeviceToHost));

    // Copy public fields into GPU state (preserving internal fields like config, inertia)
    memcpy(&g.pos, &state.pos, sizeof(Vec3));
    memcpy(&g.rotMat, &state.rotMat, sizeof(RotMat));
    memcpy(&g.vel, &state.vel, sizeof(Vec3));
    memcpy(&g.angVel, &state.angVel, sizeof(Vec3));
    g.boost = state.boost;
    g.timeSpentBoosting = state.timeSpentBoosting;
    g.isOnGround = state.isOnGround;
    g.hasJumped = state.hasJumped;
    g.hasDoubleJumped = state.hasDoubleJumped;
    g.hasFlipped = state.hasFlipped;
    g.isFlipping = state.isFlipping;
    g.isJumping = state.isJumping;
    memcpy(&g.flipRelTorque, &state.flipRelTorque, sizeof(Vec3));
    g.jumpTime = state.jumpTime;
    g.flipTime = state.flipTime;
    g.airTime = state.airTime;
    g.airTimeSinceJump = state.airTimeSinceJump;
    g.isSupersonic = state.isSupersonic;
    g.supersonicTime = state.supersonicTime;
    g.handbrakeVal = state.handbrakeVal;
    g.isAutoFlipping = state.isAutoFlipping;
    g.autoFlipTimer = state.autoFlipTimer;
    g.autoFlipTorqueScale = state.autoFlipTorqueScale;
    g.isDemoed = state.isDemoed;
    g.demoRespawnTimer = state.demoRespawnTimer;
    g.worldContactHasContact = state.worldContactHasContact;
    memcpy(&g.worldContactNormal, &state.worldContactNormal, sizeof(Vec3));
    g.carContactOtherID = state.carContactOtherID;
    g.carContactCooldownTimer = state.carContactCooldownTimer;
    g.ballHitValid = state.ballHitValid;
    g.ballHitTickCount = state.ballHitTickCount;
    g.lastControls = toGpuControls(state.lastControls);
    g.team = static_cast<uint8_t>(state.team);
    g.id = state.id;

    if (g.preset != static_cast<uint8_t>(state.preset)) {
        g.preset = static_cast<uint8_t>(state.preset);
        g.config = CarPresets::Get(g.preset);
        computeCarInertia(g);
    }

    g.velocityImpulseCache = {0, 0, 0};

    CUDA_CHECK(cudaMemcpy(
        static_cast<GpuCarState*>(d_carStates) + idx,
        &g, sizeof(GpuCarState), cudaMemcpyHostToDevice));
}

void RocketSimCudaBatch::SetBallState(int arenaIdx, const BallState& state) {
    GpuBallState g;
    memset(&g, 0, sizeof(g));
    memcpy(&g.pos, &state.pos, sizeof(Vec3));
    memcpy(&g.rotMat, &state.rotMat, sizeof(RotMat));
    memcpy(&g.vel, &state.vel, sizeof(Vec3));
    memcpy(&g.angVel, &state.angVel, sizeof(Vec3));

    CUDA_CHECK(cudaMemcpy(
        static_cast<GpuBallState*>(d_ballStates) + arenaIdx,
        &g, sizeof(GpuBallState), cudaMemcpyHostToDevice));
}

void RocketSimCudaBatch::SetBoostPadStates(int arenaIdx, const BoostPadState* states) {
    if (!states)
        return;
    std::vector<GpuBoostPadState> gpuPads(BoostPadData::NUM_TOTAL);
    for (int i = 0; i < BoostPadData::NUM_TOTAL; i++) {
        gpuPads[i].isActive = states[i].isActive;
        gpuPads[i].cooldown = states[i].cooldown;
    }
    CUDA_CHECK(cudaMemcpy(
        static_cast<GpuBoostPadState*>(d_padStates) + arenaIdx * BoostPadData::NUM_TOTAL,
        gpuPads.data(),
        BoostPadData::NUM_TOTAL * sizeof(GpuBoostPadState),
        cudaMemcpyHostToDevice));
}

void RocketSimCudaBatch::SetAllBoostPadStates(const BoostPadState* states) {
    if (!states)
        return;
    int total = m_config.numArenas * BoostPadData::NUM_TOTAL;
    std::vector<GpuBoostPadState> gpuPads(total);
    for (int i = 0; i < total; i++) {
        gpuPads[i].isActive = states[i].isActive;
        gpuPads[i].cooldown = states[i].cooldown;
    }
    CUDA_CHECK(cudaMemcpy(d_padStates, gpuPads.data(), total * sizeof(GpuBoostPadState), cudaMemcpyHostToDevice));
}

