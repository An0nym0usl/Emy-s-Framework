#include <GpuRuntime.cuh>
#include "CarPhysics.cuh"
#include "Collision.cuh"
#include "TrainingObs.cuh"
#include "VantaObsGpu.cuh"
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

namespace rsc {

// Reserved per-car aux slot for optional future stateful rewards (public stubs use pad only).
struct GpuRewardAuxState {
	uint8_t _pad;
};


// ============================================================================
// Constant memory
// ============================================================================

__constant__ ArenaSurface c_surfaces[MAX_ARENA_SURFACES];
__constant__ int c_numSurfaces;
__constant__ float c_tickTime;
__constant__ float c_ballRadius;
__constant__ float c_ballDrag;
__constant__ float c_ballFriction;
__constant__ float c_ballRestitution;
__constant__ float c_ballMaxSpeed;
__constant__ float c_ballInvInertia;
__constant__ float c_ballWorldFrictionCombined;
__constant__ float c_ballWorldRestitutionCombined;
__constant__ float c_ballHitExtraForceScale;
__constant__ float c_bumpForceScale;
__constant__ float c_bumpCooldownTime;
__constant__ float c_boostUsedPerSecond;
__constant__ float c_boostAccelGround;
__constant__ float c_boostAccelAir;
__constant__ float c_boostPadCooldownBig;
__constant__ float c_boostPadCooldownSmall;
__constant__ float c_goalThresholdY;
__constant__ Vec3 c_gravity;
__constant__ int c_obsPadMap[NUM_BOOST_PADS];
__constant__ int c_obsPadMapInv[NUM_BOOST_PADS];
__constant__ uint8_t c_defaultGroundMask[DEFAULT_ACTION_COUNT];
__constant__ uint8_t c_defaultAirMask[DEFAULT_ACTION_COUNT];
__constant__ uint8_t c_defaultJumpMask[DEFAULT_ACTION_COUNT];
__constant__ uint8_t c_defaultBoostMask[DEFAULT_ACTION_COUNT];
__constant__ TrainingRewardEntry c_trainingRewardEntries[MAX_TRAINING_REWARD_ENTRIES];
__constant__ int c_numTrainingRewardEntries;
__constant__ TrainingTerminalConfig c_trainingTerminalConfig;
__constant__ MeshGridView c_meshGrid;  // real arena triangles (numTris==0 -> analytic fallback)
__constant__ Vec3 c_vantaPadLocations[VantaConst::BOOST_SIZE];  // RLGymCPP pad order
__constant__ int c_vantaMirrorPadMap[VantaConst::BOOST_SIZE];   // X-mirror permutation (RLGym order)
__constant__ int c_vantaPadMap[VantaConst::BOOST_SIZE];         // RLGym order -> GPU pad index

// ============================================================================
// Collision mesh loading (CMF format, vertices already in BT units)
// ============================================================================

static bool loadCollisionMeshes(
    const char* collisionMeshesPath,
    std::vector<MeshTriangle>& outTris
) {
    namespace fs = std::filesystem;

    fs::path base = fs::path(collisionMeshesPath) / "soccar";
    if (!fs::exists(base)) {
        fprintf(stderr, "RocketSimCuda: collision mesh folder not found: %s\n", base.string().c_str());
        return false;
    }

    int meshId = 0;
    for (const auto& entry : fs::directory_iterator(base)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".cmf")
            continue;

        std::ifstream in(entry.path(), std::ios::binary);
        if (!in)
            continue;

        int32_t numTris = 0, numVerts = 0;
        in.read(reinterpret_cast<char*>(&numTris), 4);
        in.read(reinterpret_cast<char*>(&numVerts), 4);
        if (numTris <= 0 || numVerts <= 0 || numTris > 1000000 || numVerts > 1000000) {
            fprintf(stderr, "RocketSimCuda: bad CMF header in %s\n", entry.path().string().c_str());
            continue;
        }

        std::vector<int32_t> indices(numTris * 3);
        std::vector<float> verts(numVerts * 3);
        in.read(reinterpret_cast<char*>(indices.data()), numTris * 3 * 4);
        in.read(reinterpret_cast<char*>(verts.data()), numVerts * 3 * 4);
        if (!in) {
            fprintf(stderr, "RocketSimCuda: truncated CMF: %s\n", entry.path().string().c_str());
            continue;
        }

        for (int t = 0; t < numTris; t++) {
            int i0 = indices[t * 3 + 0], i1 = indices[t * 3 + 1], i2 = indices[t * 3 + 2];
            if (i0 < 0 || i0 >= numVerts || i1 < 0 || i1 >= numVerts || i2 < 0 || i2 >= numVerts)
                continue;
            MeshTriangle tri;
            tri.v0 = {verts[i0 * 3], verts[i0 * 3 + 1], verts[i0 * 3 + 2]};
            tri.v1 = {verts[i1 * 3], verts[i1 * 3 + 1], verts[i1 * 3 + 2]};
            tri.v2 = {verts[i2 * 3], verts[i2 * 3 + 1], verts[i2 * 3 + 2]};
            tri.meshId = meshId;
            outTris.push_back(tri);
        }
        meshId++;
    }

    return !outTris.empty();
}

// Uniform grid over the arena (BT units).
static void buildMeshGrid(
    const std::vector<MeshTriangle>& tris,
    MeshGridConfig& cfg,
    std::vector<int>& cellStart,
    std::vector<int>& cellTris
) {
    constexpr float CELL = 4.f;  // BT (= 200 UU)

    Vec3 mn = {1e30f, 1e30f, 1e30f}, mx = {-1e30f, -1e30f, -1e30f};
    for (const auto& t : tris) {
        const Vec3* vs[3] = {&t.v0, &t.v1, &t.v2};
        for (auto* v : vs) {
            mn.x = fminf(mn.x, v->x); mn.y = fminf(mn.y, v->y); mn.z = fminf(mn.z, v->z);
            mx.x = fmaxf(mx.x, v->x); mx.y = fmaxf(mx.y, v->y); mx.z = fmaxf(mx.z, v->z);
        }
    }
    // pad so queries at the boundary clamp inside
    mn.x -= 1.f; mn.y -= 1.f; mn.z -= 1.f;
    mx.x += 1.f; mx.y += 1.f; mx.z += 1.f;

    cfg.minPos = mn;
    cfg.nx = (int)ceilf((mx.x - mn.x) / CELL);
    cfg.ny = (int)ceilf((mx.y - mn.y) / CELL);
    cfg.nz = (int)ceilf((mx.z - mn.z) / CELL);
    cfg.invCellSize = {1.f / CELL, 1.f / CELL, 1.f / CELL};

    int numCells = cfg.nx * cfg.ny * cfg.nz;
    std::vector<std::vector<int>> cells(numCells);

    for (int i = 0; i < (int)tris.size(); i++) {
        const auto& t = tris[i];
        Vec3 tmn = {fminf(t.v0.x, fminf(t.v1.x, t.v2.x)),
                    fminf(t.v0.y, fminf(t.v1.y, t.v2.y)),
                    fminf(t.v0.z, fminf(t.v1.z, t.v2.z))};
        Vec3 tmx = {fmaxf(t.v0.x, fmaxf(t.v1.x, t.v2.x)),
                    fmaxf(t.v0.y, fmaxf(t.v1.y, t.v2.y)),
                    fmaxf(t.v0.z, fmaxf(t.v1.z, t.v2.z))};

        int cx0 = (int)floorf((tmn.x - mn.x) / CELL), cx1 = (int)floorf((tmx.x - mn.x) / CELL);
        int cy0 = (int)floorf((tmn.y - mn.y) / CELL), cy1 = (int)floorf((tmx.y - mn.y) / CELL);
        int cz0 = (int)floorf((tmn.z - mn.z) / CELL), cz1 = (int)floorf((tmx.z - mn.z) / CELL);
        cx0 = cx0 < 0 ? 0 : cx0; cy0 = cy0 < 0 ? 0 : cy0; cz0 = cz0 < 0 ? 0 : cz0;
        cx1 = cx1 >= cfg.nx ? cfg.nx - 1 : cx1;
        cy1 = cy1 >= cfg.ny ? cfg.ny - 1 : cy1;
        cz1 = cz1 >= cfg.nz ? cfg.nz - 1 : cz1;

        for (int cz = cz0; cz <= cz1; cz++)
            for (int cy = cy0; cy <= cy1; cy++)
                for (int cx = cx0; cx <= cx1; cx++)
                    cells[(cz * cfg.ny + cy) * cfg.nx + cx].push_back(i);
    }

    cellStart.resize(numCells + 1);
    cellTris.clear();
    for (int c = 0; c < numCells; c++) {
        cellStart[c] = (int)cellTris.size();
        for (int idx : cells[c])
            cellTris.push_back(idx);
    }
    cellStart[numCells] = (int)cellTris.size();
}

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

// ============================================================================
// Training rewards + terminals
// ============================================================================

static constexpr int MAX_TRAINING_CARS_PER_ARENA = 8;

__device__ __forceinline__ float training_vec_length(Vec3 v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ __forceinline__ bool training_ball_touched_step(const GpuCarState& car, uint64_t tickCount, int stepTicks) {
    return car.ballHitValid && car.ballHitTickCount >= (tickCount - (uint64_t)stepTicks);
}

__device__ __forceinline__ int training_find_car_idx_by_id(const GpuCarState* cars, int numCars, uint32_t id) {
    for (int i = 0; i < numCars; i++) {
        if (cars[i].id == id)
            return i;
    }
    return -1;
}

__device__ __forceinline__ bool training_is_new_contact(
    const GpuCarState& cur,
    const GpuCarState& prev
) {
    if (!cur.carContactOtherID || cur.carContactCooldownTimer <= 0.f)
        return false;

    return
        prev.carContactOtherID != cur.carContactOtherID ||
        prev.carContactCooldownTimer <= 0.f ||
        prev.carContactCooldownTimer < cur.carContactCooldownTimer;
}

__device__ float training_eval_reward(
    const TrainingRewardEntry& entry,
    const GpuCarState* cars,
    const GpuCarState* prevCars,
    const GpuBallState& ball,
    const GpuBallState& prevBall,
    const GpuArenaState& arena,
    int numCars,
    int carIdx,
    int stepTicks,
    GpuRewardAuxState* rewardAuxRow,
    uint32_t lastTouchCarID
) {
    const GpuCarState& car = cars[carIdx];
    const GpuCarState& prevCar = prevCars[carIdx];
    bool touchedStep = training_ball_touched_step(car, arena.tickCount, stepTicks);

    switch (entry.id) {
    case TrainingRewardID::GOAL_REWARD:
        if (!arena.goalScored || arena.goalTeam < 0)
            return 0.f;
        return ((int)car.team == arena.goalTeam) ? 1.f : entry.params[0];

    case TrainingRewardID::VELOCITY_BALL_TO_GOAL: {
        bool targetOrangeGoal = (car.team == (uint8_t)Team::BLUE);
        if (entry.params[0] > 0.5f)
            targetOrangeGoal = !targetOrangeGoal;

        float targetY = targetOrangeGoal ? 6000.f : -6000.f;
        Vec3 dir = { -ball.pos.x, targetY - ball.pos.y, 321.3875f - ball.pos.z };
        dir = v3_safe_normalize(dir);
        Vec3 normBallVel = ball.vel / PhysConst::BALL_MAX_SPEED;
        return v3_dot(dir, normBallVel);
    }

    case TrainingRewardID::VELOCITY_PLAYER_TO_BALL: {
        Vec3 dirToBall = v3_safe_normalize(ball.pos - car.pos);
        Vec3 normVel = car.vel / PhysConst::CAR_MAX_SPEED;
        return v3_dot(dirToBall, normVel);
    }

    case TrainingRewardID::FACE_BALL: {
        Vec3 dirToBall = v3_safe_normalize(ball.pos - car.pos);
        return v3_dot(car.rotMat.forward, dirToBall);
    }

    case TrainingRewardID::TOUCH_BALL:
        return touchedStep ? 1.f : 0.f;

    case TrainingRewardID::TOUCH_ACCEL: {
        if (!touchedStep)
            return 0.f;

        constexpr float MAX_REWARDED_BALL_SPEED = 110.f * (100000.f / 36000.f);
        float prevSpeedFrac = rs_min(1.f, training_vec_length(prevBall.vel) / MAX_REWARDED_BALL_SPEED);
        float curSpeedFrac = rs_min(1.f, training_vec_length(ball.vel) / MAX_REWARDED_BALL_SPEED);
        return (curSpeedFrac > prevSpeedFrac) ? (curSpeedFrac - prevSpeedFrac) : 0.f;
    }

    case TrainingRewardID::STRONG_TOUCH: {
        if (!touchedStep)
            return 0.f;

        float hitForce = training_vec_length(ball.vel - prevBall.vel);
        if (hitForce < entry.params[0])
            return 0.f;
        return rs_min(1.f, hitForce / rs_max(entry.params[1], 1e-6f));
    }

    case TrainingRewardID::SPEED:
        return training_vec_length(car.vel) / PhysConst::CAR_MAX_SPEED;

    case TrainingRewardID::VELOCITY:
        return (training_vec_length(car.vel) / PhysConst::CAR_MAX_SPEED) * entry.params[0];

    case TrainingRewardID::AIR:
        return car.isOnGround ? 0.f : 1.f;

    case TrainingRewardID::WAVEDASH:
        return (car.isOnGround && prevCar.isFlipping && !prevCar.isOnGround) ? 1.f : 0.f;

    case TrainingRewardID::PICKUP_BOOST:
        if (car.boost > prevCar.boost)
            return sqrtf(car.boost / 100.f) - sqrtf(prevCar.boost / 100.f);
        return 0.f;

    case TrainingRewardID::SAVE_BOOST:
        return rs_clamp(powf(car.boost / 100.f, entry.params[0]), 0.f, 1.f);

    case TrainingRewardID::BUMP: {
        if (!training_is_new_contact(car, prevCar))
            return 0.f;
        int otherIdx = training_find_car_idx_by_id(cars, numCars, car.carContactOtherID);
        if (otherIdx < 0 || cars[otherIdx].team == car.team)
            return 0.f;
        return 1.f;
    }

    case TrainingRewardID::BUMPED_PENALTY: {
        for (int otherIdx = 0; otherIdx < numCars; otherIdx++) {
            if (otherIdx == carIdx || cars[otherIdx].team == car.team)
                continue;
            if (!training_is_new_contact(cars[otherIdx], prevCars[otherIdx]))
                continue;
            if (cars[otherIdx].carContactOtherID == car.id)
                return -1.f;
        }
        return 0.f;
    }

    case TrainingRewardID::DEMO: {
        if (!training_is_new_contact(car, prevCar))
            return 0.f;
        int otherIdx = training_find_car_idx_by_id(cars, numCars, car.carContactOtherID);
        if (otherIdx < 0 || cars[otherIdx].team == car.team)
            return 0.f;
        return (cars[otherIdx].isDemoed && !prevCars[otherIdx].isDemoed) ? 1.f : 0.f;
    }

    case TrainingRewardID::DEMOED_PENALTY:
        return (car.isDemoed && !prevCar.isDemoed) ? -1.f : 0.f;

    case TrainingRewardID::KICKOFF_PROXIMITY: {
        constexpr float KICKOFF_POS_XY_TOLERANCE = 1.0f;
        if (rs_abs(ball.pos.x) >= KICKOFF_POS_XY_TOLERANCE || rs_abs(ball.pos.y) >= KICKOFF_POS_XY_TOLERANCE)
            return 0.f;

        float playerDistSq = v3_dist_sq(car.pos, ball.pos);

        float minOpponentDistSq = 3.402823466e+38f;
        bool opponentFound = false;
        for (int i = 0; i < numCars; i++) {
            if (cars[i].id == car.id)
                continue;
            if (cars[i].team != car.team) {
                opponentFound = true;
                float d = v3_dist_sq(cars[i].pos, ball.pos);
                minOpponentDistSq = rs_min(minOpponentDistSq, d);
            }
        }

        if (!opponentFound || playerDistSq < minOpponentDistSq) {
            if (opponentFound && playerDistSq == minOpponentDistSq)
                return 0.f;
            return 1.f;
        }
        return -1.f;
    }

    case TrainingRewardID::TEAMMATE_BUMP_PENALTY: {
        uint32_t curId = car.carContactOtherID;
        float curTimer = car.carContactCooldownTimer;
        if (curTimer <= 0.f || curId == 0)
            return 0.f;

        uint32_t prevId = prevCar.carContactOtherID;
        float prevTimer = prevCar.carContactCooldownTimer;
        bool isNewBump = (curTimer > prevTimer) || (curId != prevId && prevTimer <= 0.f);
        if (!isNewBump)
            return 0.f;

        for (int i = 0; i < numCars; i++) {
            if (cars[i].id == curId) {
                if (cars[i].team == car.team)
                    return -1.f;
                break;
            }
        }
        return 0.f;
    }

    // Reserved team/aerial IDs — public stubs (no proprietary pack).
    case TrainingRewardID::PASS_INCENTIVE:
    case TrainingRewardID::TEAM_PASSING:
    case TrainingRewardID::MIXED_INITIATE:
    case TrainingRewardID::MIXED_DOUBLETAP:
    case TrainingRewardID::SPEED_AERIAL:
        (void)rewardAuxRow;
        (void)lastTouchCarID;
        return 0.f;

    case TrainingRewardID::PRESSURE_FLICK: {
        // Approx Leak PressureFlickReward: flip-touch near ball under pressure, ball→goal speed.
        if (!touchedStep || !car.isFlipping)
            return 0.f;
        float dist = sqrtf(v3_dist_sq(prevCar.pos, prevBall.pos));
        if (dist > 250.f)
            return 0.f;
        float speedDiff = rs_abs(training_vec_length(prevCar.vel) - training_vec_length(prevBall.vel));
        if (speedDiff > 500.f)
            return 0.f;
        float closestOpp = 1e9f;
        for (int i = 0; i < numCars; i++) {
            if (cars[i].team == car.team || cars[i].id == car.id)
                continue;
            closestOpp = rs_min(closestOpp, sqrtf(v3_dist_sq(car.pos, cars[i].pos)));
        }
        if (closestOpp > 700.f)
            return 0.f;
        bool targetOrange = (car.team == (uint8_t)Team::BLUE);
        float targetY = targetOrange ? 5120.f : -5120.f;
        Vec3 dir = v3_safe_normalize(Vec3{ 0.f, targetY - ball.pos.y, 0.f });
        float velGoal = v3_dot(ball.vel, dir);
        if (velGoal <= 1000.f)
            return 0.f;
        float ratio = velGoal / 2920.f;
        float r = ratio * ratio * sqrtf(rs_max(ratio, 0.f)); // ~pow 2.5
        return rs_min(r, 2.f);
    }

    case TrainingRewardID::MAWKZY_FLICK: {
        // Lighter flick proxy: flip-touch that accelerates ball toward goal.
        if (!touchedStep || !car.isFlipping)
            return 0.f;
        bool targetOrange = (car.team == (uint8_t)Team::BLUE);
        float targetY = targetOrange ? 5120.f : -5120.f;
        Vec3 dir = v3_safe_normalize(Vec3{ 0.f, targetY - ball.pos.y, 0.f });
        float accel = v3_dot(ball.vel - prevBall.vel, dir);
        return rs_clamp(accel / 1500.f, 0.f, 1.f);
    }

    case TrainingRewardID::AIR_CARRY: {
        // Air + ball touch / soft carry shaping.
        if (car.isOnGround)
            return 0.f;
        if (touchedStep)
            return 1.f;
        float dist = sqrtf(v3_dist_sq(car.pos, ball.pos));
        if (dist > 300.f)
            return 0.f;
        float rel = training_vec_length(car.vel - ball.vel);
        return rs_clamp(1.f - rel / 1200.f, 0.f, 0.35f);
    }

    case TrainingRewardID::ANTI_BALL_STACK: {
        // Discourage sitting on ball / stacking (lightweight proxy).
        float dz = rs_abs(car.pos.z - ball.pos.z);
        float dxy = sqrtf((car.pos.x - ball.pos.x) * (car.pos.x - ball.pos.x)
            + (car.pos.y - ball.pos.y) * (car.pos.y - ball.pos.y));
        if (dxy < 120.f && car.pos.z > ball.pos.z && dz < 180.f && car.isOnGround)
            return -1.f;
        return 0.f;
    }

    default:
        return 0.f;
    }
}

__global__ void buildRewardsAndTerminalsKernel(
    const GpuCarState* allCars,
    const GpuCarState* prevCars,
    const GpuBallState* allBalls,
    const GpuBallState* prevBalls,
    const GpuArenaState* allArenas,
    GpuRewardAuxState* allRewardAuxStates,
    int numArenas,
    int maxCarsPerArena,
    int stepTicks,
    float* outRewards,
    uint8_t* outTerminals,
    float* noTouchTimers
) {
    int arenaIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (arenaIdx >= numArenas)
        return;

    const GpuArenaState& arena = allArenas[arenaIdx];
    const GpuCarState* cars = &allCars[arenaIdx * maxCarsPerArena];
    const GpuCarState* prevArenaCars = &prevCars[arenaIdx * maxCarsPerArena];
    const GpuBallState& ball = allBalls[arenaIdx];
    const GpuBallState& prevBall = prevBalls[arenaIdx];
    GpuRewardAuxState* rewardAuxRow = &allRewardAuxStates[arenaIdx * maxCarsPerArena];
    float* rewardRow = outRewards + arenaIdx * maxCarsPerArena;

    for (int i = 0; i < maxCarsPerArena; i++)
        rewardRow[i] = 0.f;

    // GameState.lastTouchCarID: the most recent ball toucher.
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

    (void)rewardAuxRow;

    bool supportedArena = arena.numCars <= MAX_TRAINING_CARS_PER_ARENA;
    if (supportedArena) {
        float totalRewards[MAX_TRAINING_CARS_PER_ARENA] = {};
        float rawRewards[MAX_TRAINING_CARS_PER_ARENA] = {};

        for (int rewardIdx = 0; rewardIdx < c_numTrainingRewardEntries; rewardIdx++) {
            const TrainingRewardEntry& entry = c_trainingRewardEntries[rewardIdx];

            for (int carIdx = 0; carIdx < arena.numCars; carIdx++) {
                rawRewards[carIdx] = training_eval_reward(
                    entry, cars, prevArenaCars, ball, prevBall, arena, arena.numCars, carIdx, stepTicks,
                    rewardAuxRow, lastTouchCarID
                );
            }

            if (entry.isZeroSum) {
                for (int carIdx = 0; carIdx < arena.numCars; carIdx++) {
                    float teamSum = 0.f;
                    int teamCount = 0;
                    float oppSum = 0.f;
                    int oppCount = 0;
                    uint8_t myTeam = cars[carIdx].team;

                    for (int otherIdx = 0; otherIdx < arena.numCars; otherIdx++) {
                        if (cars[otherIdx].team == myTeam) {
                            teamSum += rawRewards[otherIdx];
                            teamCount++;
                        } else {
                            oppSum += rawRewards[otherIdx];
                            oppCount++;
                        }
                    }

                    float avgTeam = teamCount > 0 ? (teamSum / (float)teamCount) : 0.f;
                    float avgOpp = oppCount > 0 ? (oppSum / (float)oppCount) : 0.f;
                    float zeroSumReward =
                        rawRewards[carIdx] * (1.f - entry.teamSpirit) +
                        avgTeam * entry.teamSpirit -
                        avgOpp * entry.opponentScale;
                    totalRewards[carIdx] += zeroSumReward * entry.weight;
                }
            } else {
                for (int carIdx = 0; carIdx < arena.numCars; carIdx++)
                    totalRewards[carIdx] += rawRewards[carIdx] * entry.weight;
            }
        }

        for (int carIdx = 0; carIdx < arena.numCars; carIdx++)
            rewardRow[carIdx] = totalRewards[carIdx];
    }

    bool touchedThisStep = false;
    for (int carIdx = 0; carIdx < arena.numCars; carIdx++) {
        if (training_ball_touched_step(cars[carIdx], arena.tickCount, stepTicks)) {
            touchedThisStep = true;
            break;
        }
    }

    float noTouchTime = noTouchTimers[arenaIdx];
    if (touchedThisStep) {
        noTouchTime = 0.f;
    } else {
        noTouchTime += (float)stepTicks * c_tickTime;
    }
    noTouchTimers[arenaIdx] = noTouchTime;

    uint8_t terminal = 0;
    if (c_trainingTerminalConfig.useGoalScore && arena.goalScored)
        terminal = 1;
    if (!terminal &&
        c_trainingTerminalConfig.noTouchTimeoutSeconds > 0.f &&
        noTouchTime >= c_trainingTerminalConfig.noTouchTimeoutSeconds) {
        terminal = 2;
    }

    outTerminals[arenaIdx] = terminal;
}

// ============================================================================
// MAIN SIMULATION KERNEL
// 1 thread = 1 arena
// ============================================================================

__global__ void stepArenaKernel(
    GpuCarState* allCars,       // [numArenas * maxCarsPerArena]
    GpuBallState* allBalls,     // [numArenas]
    GpuBoostPadState* allPads,  // [numArenas * NUM_BOOST_PADS]
    GpuArenaState* allArenas,   // [numArenas]
    const GpuCarControls* allControls, // [numArenas * maxCarsPerArena]
    int numArenas,
    int maxCarsPerArena,
    int ticksToSimulate
) {
    int arenaIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (arenaIdx >= numArenas) return;

    // Pointers to this arena's data
    GpuCarState* cars = &allCars[arenaIdx * maxCarsPerArena];
    GpuBallState& ball = allBalls[arenaIdx];
    GpuBoostPadState* pads = &allPads[arenaIdx * BoostPadData::NUM_TOTAL];
    GpuArenaState& arena = allArenas[arenaIdx];

    int numCars = arena.numCars;
    bool isVoid = (arena.gameMode == 1);
    float dt = c_tickTime;

    for (int tick = 0; tick < ticksToSimulate; tick++) {

        // Per-tick force accumulators (Bullet applyCentralForce/applyTorque:
        // integrated by the solver, applied to velocities at writeback).
        Vec3 carAccel[MAX_CARS_PER_ARENA];
        Vec3 carAngAccel[MAX_CARS_PER_ARENA];
        for (int c = 0; c < numCars; c++) {
            carAccel[c] = v3_zero();
            carAngAccel[c] = v3_zero();
        }

        // ============================================================
        // 1. CAR PRE-TICK UPDATE (Car::_PreTickUpdate)
        // ============================================================
        for (int c = 0; c < numCars; c++) {
            GpuCarState& car = cars[c];
            car.controls = allControls[arenaIdx * maxCarsPerArena + c];

            // Clamp controls
            car.controls.throttle = rs_clamp(car.controls.throttle, -1.f, 1.f);
            car.controls.steer    = rs_clamp(car.controls.steer, -1.f, 1.f);
            car.controls.pitch    = rs_clamp(car.controls.pitch, -1.f, 1.f);
            car.controls.yaw      = rs_clamp(car.controls.yaw, -1.f, 1.f);
            car.controls.roll     = rs_clamp(car.controls.roll, -1.f, 1.f);

            // Handle demo state
            if (car.isDemoed) {
                device_handle_demo(car, dt, arenaIdx);
                continue;
            }

            bool jumpPressed = car.controls.jump && !car.lastControls.jump;

            // Wheel raycasts
            if (!isVoid) {
                device_wheel_raycast(car, c_surfaces, c_numSurfaces, c_meshGrid, dt);
                // Match RocketSim's updateVehicleFirst(): compute friction impulses
                // using wheel state from the previous tick before controls update.
                device_calc_friction_impulses(car, dt);
            } else {
                car.isOnGround = false;
                for (int w = 0; w < 4; w++) car.wheels[w].isInContact = false;
            }

            // Count wheels in contact
            int numWheelsInContact = 0;
            for (int w = 0; w < 4; w++)
                numWheelsInContact += car.wheels[w].isInContact ? 1 : 0;

            float forwardSpeed = car_forward_speed(car);

            // Update wheels (throttle, brake, steer, friction, sticky force)
            if (!isVoid)
                device_update_wheels(car, dt, carAccel[c]);

            // Air torque (if not fully grounded)
            if (numWheelsInContact < 3) {
                device_update_air_torque(car, dt, numWheelsInContact == 0,
                                         carAccel[c], carAngAccel[c]);
            } else {
                car.isFlipping = false;
            }

            // Jump
            device_update_jump(car, dt, jumpPressed, carAccel[c]);

            // Auto-flip
            device_update_autoflip(car, dt, jumpPressed);

            // Double jump / Flip
            device_update_double_jump_or_flip(car, dt, jumpPressed, forwardSpeed);

            // Auto-roll
            if (car.controls.throttle != 0.f &&
                ((numWheelsInContact > 0 && numWheelsInContact < 4) || car.worldContactHasContact)) {
                device_update_autoroll(car, dt, numWheelsInContact,
                                       carAccel[c], carAngAccel[c]);
            }

            car.worldContactHasContact = false;

            // Suspension + friction (direct impulses, like btVehicleRL)
            if (!isVoid) {
                device_update_suspension(car, dt);
                device_apply_friction_impulses(car, dt);
            }

            // Boost
            device_update_boost(car, dt, c_boostUsedPerSecond, c_boostAccelGround, c_boostAccelAir,
                                carAccel[c]);
        }

        // ============================================================
        // 2. BOOST PADS PRE-TICK
        // ============================================================
        if (!isVoid) {
            for (int p = 0; p < BoostPadData::NUM_TOTAL; p++)
                device_boostpad_pre_tick(pads[p], dt);
        }

        // ============================================================
        // 3+4. BULLET WORLD STEP
        // (damping -> collision detection at pre-move positions ->
        //  sequential impulse solve -> writeback -> integrate transforms)
        // ============================================================
        device_bullet_world_step(
            cars, numCars, ball,
            c_surfaces, isVoid ? 0 : c_numSurfaces,
            c_meshGrid,
            carAccel, carAngAccel,
            c_gravity,
            c_ballDrag,
            c_ballRadius, c_ballInvInertia,
            c_ballWorldFrictionCombined, c_ballWorldRestitutionCombined,
            PhysConst::CARWORLD_COLLISION_FRICTION, PhysConst::CARWORLD_COLLISION_RESTITUTION,
            c_ballHitExtraForceScale, c_bumpForceScale, c_bumpCooldownTime,
            arena.tickCount, dt);

        // ============================================================
        // 5. POST-TICK
        // ============================================================
        for (int c = 0; c < numCars; c++) {
            device_car_post_tick(cars[c], dt);
            device_car_finish_tick(cars[c]);
        }

        // Boost pad collisions + pickup
        if (!isVoid) {
            for (int c = 0; c < numCars; c++) {
                if (cars[c].isDemoed) continue;
                for (int p = 0; p < BoostPadData::NUM_TOTAL; p++)
                    device_boostpad_check_collide(pads[p], p, cars[c]);
            }
            for (int p = 0; p < BoostPadData::NUM_TOTAL; p++)
                device_boostpad_post_tick(pads[p], p, cars, numCars, dt,
                                         c_boostPadCooldownBig, c_boostPadCooldownSmall);
        }

        // Ball finish tick
        device_ball_finish_tick(ball, c_ballMaxSpeed);

        // ============================================================
        // 6. GOAL CHECK
        // ============================================================
        if (!isVoid) {
            if (rs_abs(ball.pos.y) > c_goalThresholdY &&
                rs_abs(ball.pos.x) < PhysConst::GOAL_HALF_WIDTH &&
                ball.pos.z < PhysConst::GOAL_HEIGHT) {
                arena.goalScored = true;
                arena.goalTeam = (ball.pos.y > 0.f) ? 0 : 1; // 0=blue scored (ball in orange end)
            }
        }

        arena.tickCount++;
    }
}

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
    // both teams, then mirrors orange with pos *= {-1,-1,1} and yaw += π).
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

void RocketSimCudaBatch::SetResetCurriculum(float kickoffW, float fuzzedW, float aerialW) {
    auto clampW = [](float w) -> float {
        if (!(w >= 0.f) || w > 1e6f) return 0.f;
        return w;
    };
    m_resetKickoffW = clampW(kickoffW);
    m_resetFuzzedW = clampW(fuzzedW);
    m_resetAerialW = clampW(aerialW);
    if (m_resetKickoffW + m_resetFuzzedW + m_resetAerialW <= 0.f)
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

    // 0=kickoff, 1=fuzzed kickoff, 2=aerial-ish setup
    int resetMode = 0;
    {
        const float total = m_resetKickoffW + m_resetFuzzedW + m_resetAerialW;
        float r = nextU01() * total;
        if (r < m_resetKickoffW)
            resetMode = 0;
        else if (r < m_resetKickoffW + m_resetFuzzedW)
            resetMode = 1;
        else
            resetMode = 2;
    }

    // Reset arena state
    GpuArenaState arenaState;
    CUDA_CHECK(cudaMemcpy(&arenaState,
        static_cast<GpuArenaState*>(d_arenaInfos) + arenaIdx,
        sizeof(GpuArenaState), cudaMemcpyDeviceToHost));

    arenaState.tickCount = 0;
    arenaState.goalScored = false;
    arenaState.goalTeam = -1;

    // Reward::Reset — clear stateful reward memories for this arena
    if (d_rewardAuxStates) {
        CUDA_CHECK(cudaMemset(
            static_cast<GpuRewardAuxState*>(d_rewardAuxStates) + arenaIdx * M,
            0, M * sizeof(GpuRewardAuxState)));
    }

    // ObsBuilder::Reset — clear the per-arena VantaObs state
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

        if (resetMode == 2 && c == setupCar) {
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
                // Mild positional fuzz only — keep mirrored slot + yaw (RocketSim-like).
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
    // Keep per-arena staggered no-touch ages from ResetArena (do NOT zero — that
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
        // No LUT uploaded — leave controls unchanged.
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

} // namespace rsc
