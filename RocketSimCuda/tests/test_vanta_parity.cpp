// VantaObs CPU-vs-GPU parity test.
// Runs both sims in lockstep from kickoff with scripted controls, builds the
// observation on both sides every tickSkip ticks and compares all 366 floats
// per player. Exit 0 = parity within tolerance.

#include "cuda_host_shim.cuh"
#include "../src/GpuTypes.cuh"

#include <RocketSimCuda.h>
#include <RocketSim.h>

#include <RLGymCPP/OBSBuilders/VantaObs.h>
#include <RLGymCPP/Gamestates/GameState.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace cpu = RocketSim;
namespace gpu = rsc;
using RLGC::VantaObs;
using RLGC::VantaObsConfig;
using RLGC::GameState;

static constexpr float PI_F = 3.14159265358979323846f;
static constexpr int TICK_SKIP = 8;
static constexpr float TOLERANCE = 2e-3f;  // obs are normalized; ~5 UU absolute

static cpu::Vec ToCpuVec(const gpu::Vec3& v) { return {v.x, v.y, v.z}; }
static cpu::RotMat ToCpuRot(const gpu::RotMat& m) {
    return {ToCpuVec(m.forward), ToCpuVec(m.right), ToCpuVec(m.up)};
}

static cpu::CarControls ToCpuControls(const gpu::CarControls& c) {
    cpu::CarControls out;
    out.throttle = c.throttle; out.steer = c.steer;
    out.pitch = c.pitch; out.yaw = c.yaw; out.roll = c.roll;
    out.jump = c.jump; out.boost = c.boost; out.handbrake = c.handbrake;
    return out;
}

static void CopyGpuToCpuCarState(cpu::CarState& dst, const gpu::CarState& src) {
    dst.pos = ToCpuVec(src.pos);
    dst.rotMat = ToCpuRot(src.rotMat);
    dst.vel = ToCpuVec(src.vel);
    dst.angVel = ToCpuVec(src.angVel);
    dst.boost = src.boost;
    dst.isOnGround = src.isOnGround;
    dst.hasJumped = src.hasJumped;
    dst.hasDoubleJumped = src.hasDoubleJumped;
    dst.hasFlipped = src.hasFlipped;
    dst.isFlipping = src.isFlipping;
    dst.isJumping = src.isJumping;
    dst.jumpTime = src.jumpTime;
    dst.flipTime = src.flipTime;
    dst.airTime = src.airTime;
    dst.airTimeSinceJump = src.airTimeSinceJump;
    dst.isSupersonic = src.isSupersonic;
    dst.supersonicTime = src.supersonicTime;
    dst.handbrakeVal = src.handbrakeVal;
    dst.isDemoed = src.isDemoed;
    dst.demoRespawnTimer = src.demoRespawnTimer;
    dst.lastControls = ToCpuControls(src.lastControls);
}

int main() {
    std::filesystem::path meshes =
        std::filesystem::path(RSCUDA_WORKSPACE_ROOT) / "collision_meshes";
    cpu::Init(meshes, true);
    std::string meshPath = meshes.string();

    constexpr int NUM_CARS = 4;  // 2v2 like the training config

    // ---- GPU batch ----
    gpu::RocketSimCudaBatch batch;
    gpu::BatchConfig cfg;
    cfg.numArenas = 1;
    cfg.maxCarsPerArena = NUM_CARS;
    cfg.tickRate = 120.f;
    cfg.collisionMeshesPath = meshPath.c_str();
    cfg.obsMode = gpu::ObsMode::VANTA;
    cfg.vantaPredictBall = 1;
    cfg.vantaTickSkip = TICK_SKIP;
    batch.Init(cfg);

    batch.AddCar(0, gpu::Team::BLUE, gpu::CarPreset::OCTANE);
    batch.AddCar(0, gpu::Team::BLUE, gpu::CarPreset::OCTANE);
    batch.AddCar(0, gpu::Team::ORANGE, gpu::CarPreset::OCTANE);
    batch.AddCar(0, gpu::Team::ORANGE, gpu::CarPreset::OCTANE);
    batch.ResetArena(0);

    // ---- CPU arena, mirrored from the GPU spawn state ----
    cpu::Arena* arena = cpu::Arena::Create(cpu::GameMode::SOCCAR);
    std::vector<cpu::Car*> cars;
    cars.push_back(arena->AddCar(cpu::Team::BLUE, cpu::CAR_CONFIG_OCTANE));
    cars.push_back(arena->AddCar(cpu::Team::BLUE, cpu::CAR_CONFIG_OCTANE));
    cars.push_back(arena->AddCar(cpu::Team::ORANGE, cpu::CAR_CONFIG_OCTANE));
    cars.push_back(arena->AddCar(cpu::Team::ORANGE, cpu::CAR_CONFIG_OCTANE));

    for (int i = 0; i < NUM_CARS; i++) {
        gpu::CarState g = batch.GetCarState(0, i);
        cpu::CarState c = cars[i]->GetState();
        CopyGpuToCpuCarState(c, g);
        cars[i]->SetState(c);
    }
    {
        gpu::BallState gb = batch.GetBallState(0);
        cpu::BallState b;
        b.pos = ToCpuVec(gb.pos);
        b.rotMat = ToCpuRot(gb.rotMat);
        b.vel = ToCpuVec(gb.vel);
        b.angVel = ToCpuVec(gb.angVel);
        arena->ball->SetState(b);
    }

    // ---- CPU obs builder (one per arena, like EnvSet) ----
    VantaObsConfig obsCfg = {};
    obsCfg.tickSkip = TICK_SKIP;
    obsCfg.predictBall = true;
    VantaObs cpuObs(obsCfg);

    {
        GameState resetState = GameState(arena);
        cpuObs.Reset(resetState);
    }

    std::vector<float> gpuObs(NUM_CARS * gpu::VANTA_OBS_SIZE);

    int worstIdx = -1, worstPlayer = -1;
    float worstDiff = 0.f;
    int failCount = 0;
    int steps = 30;  // 240 ticks of play

    for (int step = 0; step < steps; step++) {
        // Scripted controls (deterministic variety)
        std::vector<gpu::CarControls> controls(NUM_CARS);
        for (int i = 0; i < NUM_CARS; i++) {
            gpu::CarControls c;
            c.throttle = 1.f;
            c.steer = 0.4f * sinf(0.21f * step + i);
            c.boost = ((step + i) % 5) < 2;
            c.jump = (step > 6) && ((step + 2 * i) % 11) == 0;
            c.pitch = (step > 8 && ((step + i) % 13) == 0) ? -0.8f : 0.f;
            controls[i] = c;
            batch.SetCarControls(0, i, c);
            cars[i]->controls = ToCpuControls(c);
        }

        arena->Step(TICK_SKIP);
        batch.Step(TICK_SKIP);

        // Resync the CPU world from the GPU so both obs builders see the
        // exact same physics state (this isolates obs parity from the tiny
        // legitimate physics divergences at chaotic events).
        for (int i = 0; i < NUM_CARS; i++) {
            gpu::CarState g = batch.GetCarState(0, i);
            cpu::CarState c = cars[i]->GetState();
            CopyGpuToCpuCarState(c, g);

            // Wheel contacts are computed during the step (not part of the
            // public state) — read them from the GPU internals so both obs
            // builders agree on wheelsWithContact.
            rsc::GpuCarState internal;
            batch.DebugCopyCarInternal(0, i, &internal, sizeof(internal));
            for (int w = 0; w < 4; w++)
                c.wheelsWithContact[w] = internal.wheels[w].isInContact;
            c.worldContact.hasContact = internal.worldContactHasContact;
            c.worldContact.contactNormal = ToCpuVec(internal.worldContactNormal);

            cars[i]->SetState(c);
        }
        {
            gpu::BallState gb = batch.GetBallState(0);
            cpu::BallState b = arena->ball->GetState();
            b.pos = ToCpuVec(gb.pos);
            b.rotMat = ToCpuRot(gb.rotMat);
            b.vel = ToCpuVec(gb.vel);
            b.angVel = ToCpuVec(gb.angVel);
            arena->ball->SetState(b);
        }

        // ---- Build obs on both sides ----
        GameState state = GameState(arena);
        state.lastArena = arena;
        // Match players by carId (GameState ordering is not index order);
        // set prevAction = applied controls.
        int playerIdxOfCar[NUM_CARS];
        for (int i = 0; i < NUM_CARS; i++) {
            playerIdxOfCar[i] = -1;
            for (int j = 0; j < (int)state.players.size(); j++) {
                if (state.players[j].carId == cars[i]->id) {
                    playerIdxOfCar[i] = j;
                    auto& pa = state.players[j].prevAction;
                    const auto& c = controls[i];
                    pa.throttle = c.throttle; pa.steer = c.steer;
                    pa.pitch = c.pitch; pa.yaw = c.yaw; pa.roll = c.roll;
                    pa.jump = c.jump; pa.boost = c.boost; pa.handbrake = c.handbrake;
                    break;
                }
            }
            if (playerIdxOfCar[i] < 0) {
                std::printf("FATAL: car id %u not found in GameState\n", cars[i]->id);
                return 1;
            }
        }

        batch.BuildVantaObs();
        batch.CopyBuiltAdvancedObs(gpuObs.data());

        for (int p = 0; p < NUM_CARS; p++) {
            RLGC::FList cpuRow = cpuObs.BuildObs(state.players[playerIdxOfCar[p]], state);
            if ((int)cpuRow.size() != gpu::VANTA_OBS_SIZE) {
                std::printf("FATAL: CPU obs size %d != %d\n",
                            (int)cpuRow.size(), gpu::VANTA_OBS_SIZE);
                return 1;
            }
            const float* gpuRow = gpuObs.data() + p * gpu::VANTA_OBS_SIZE;
            for (int k = 0; k < gpu::VANTA_OBS_SIZE; k++) {
                float d = std::fabs(cpuRow[k] - gpuRow[k]);
                if (d > worstDiff) {
                    worstDiff = d;
                    worstIdx = k;
                    worstPlayer = p;
                }
                if (d > TOLERANCE) {
                    if (failCount < 12) {
                        std::printf(
                            "step %2d player %d obs[%3d]: cpu=%9.5f gpu=%9.5f (d=%.6f)\n",
                            step, p, k, cpuRow[k], gpuRow[k], d);
                    }
                    failCount++;
                }
            }
        }
    }

    std::printf("\nWorst diff: %.6f at obs[%d] (player %d); mismatches over tol: %d\n",
                worstDiff, worstIdx, worstPlayer, failCount);
    std::printf("Overall: %s\n", failCount == 0 ? "PASS" : "FAIL");

    delete arena;
    batch.Destroy();
    return failCount == 0 ? 0 : 1;
}
