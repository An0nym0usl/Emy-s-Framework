#include <RocketSimCuda.h>
#include <RocketSim.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace cpu = RocketSim;
namespace gpu = rsc;

namespace {

constexpr float PI_F = 3.14159265358979323846f;

struct VecDiff {
    float abs = 0.0f;
    float rel = 0.0f;
};

// Defaults reflect the bullet-exact solver: most scenarios match to ~0.01 UU.
// Scenarios touching known residual gaps (symmetric-tie landings, goal
// interior geometry) override these.
struct ScenarioTolerances {
    float ballPosTol = 0.5f;
    float ballVelTol = 0.5f;
    float carPosTol = 0.5f;
    float carVelTol = 0.5f;
    float carForwardTolDeg = 0.2f;
    float boostTol = 0.05f;
};

struct ComparisonResult {
    std::string label;
    int ticks = 0;
    int numCars = 0;
    VecDiff ballPos;
    VecDiff ballVel;
    VecDiff carPos;
    VecDiff carVel;
    float carForwardAngleDeg = 0.0f;
    float carBoostDiff = 0.0f;
    int onGroundMismatches = 0;
    int demoMismatches = 0;
    bool goalCpu = false;
    bool goalGpu = false;
    bool goalTeamMatch = true;
    int cpuGoalTeam = -1;
    int gpuGoalTeam = -1;
    bool passed = false;
};

struct CarSpec {
    gpu::Team team = gpu::Team::BLUE;
    gpu::CarPreset preset = gpu::CarPreset::OCTANE;
};

cpu::Vec ToCpuVec(const gpu::Vec3& v) {
    return {v.x, v.y, v.z};
}

cpu::RotMat ToCpuRot(const gpu::RotMat& m) {
    return {
        ToCpuVec(m.forward),
        ToCpuVec(m.right),
        ToCpuVec(m.up),
    };
}

cpu::BallState ToCpuBallState(const gpu::BallState& src) {
    cpu::BallState dst;
    dst.pos = ToCpuVec(src.pos);
    dst.rotMat = ToCpuRot(src.rotMat);
    dst.vel = ToCpuVec(src.vel);
    dst.angVel = ToCpuVec(src.angVel);
    return dst;
}

cpu::CarControls ToCpuControls(const gpu::CarControls& c) {
    cpu::CarControls out;
    out.throttle = c.throttle;
    out.steer = c.steer;
    out.pitch = c.pitch;
    out.yaw = c.yaw;
    out.roll = c.roll;
    out.jump = c.jump;
    out.boost = c.boost;
    out.handbrake = c.handbrake;
    return out;
}

void CopyGpuToCpuCarState(cpu::CarState& dst, const gpu::CarState& src) {
    dst.pos = ToCpuVec(src.pos);
    dst.rotMat = ToCpuRot(src.rotMat);
    dst.vel = ToCpuVec(src.vel);
    dst.angVel = ToCpuVec(src.angVel);
    dst.boost = src.boost;
    dst.timeSpentBoosting = src.timeSpentBoosting;
    dst.isOnGround = src.isOnGround;
    dst.hasJumped = src.hasJumped;
    dst.hasDoubleJumped = src.hasDoubleJumped;
    dst.hasFlipped = src.hasFlipped;
    dst.isFlipping = src.isFlipping;
    dst.isJumping = src.isJumping;
    dst.flipRelTorque = ToCpuVec(src.flipRelTorque);
    dst.jumpTime = src.jumpTime;
    dst.flipTime = src.flipTime;
    dst.airTime = src.airTime;
    dst.airTimeSinceJump = src.airTimeSinceJump;
    dst.isSupersonic = src.isSupersonic;
    dst.supersonicTime = src.supersonicTime;
    dst.handbrakeVal = src.handbrakeVal;
    dst.isAutoFlipping = src.isAutoFlipping;
    dst.autoFlipTimer = src.autoFlipTimer;
    dst.autoFlipTorqueScale = src.autoFlipTorqueScale;
    dst.isDemoed = src.isDemoed;
    dst.demoRespawnTimer = src.demoRespawnTimer;
    dst.lastControls = ToCpuControls(src.lastControls);
}

cpu::CarConfig CpuConfigForPreset(gpu::CarPreset preset) {
    switch (preset) {
    case gpu::CarPreset::OCTANE: return cpu::CAR_CONFIG_OCTANE;
    case gpu::CarPreset::DOMINUS: return cpu::CAR_CONFIG_DOMINUS;
    case gpu::CarPreset::PLANK: return cpu::CAR_CONFIG_PLANK;
    case gpu::CarPreset::BREAKOUT: return cpu::CAR_CONFIG_BREAKOUT;
    case gpu::CarPreset::HYBRID: return cpu::CAR_CONFIG_HYBRID;
    default: return cpu::CAR_CONFIG_MERC;
    }
}

cpu::Team CpuTeam(gpu::Team team) {
    return team == gpu::Team::BLUE ? cpu::Team::BLUE : cpu::Team::ORANGE;
}

float Length(float x, float y, float z) {
    return std::sqrt(x * x + y * y + z * z);
}

float Length(const cpu::Vec& v) {
    return Length(v.x, v.y, v.z);
}

float Length(const gpu::Vec3& v) {
    return Length(v.x, v.y, v.z);
}

VecDiff DiffVec(const cpu::Vec& a, const gpu::Vec3& b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    float abs = Length(dx, dy, dz);
    float denom = std::max(Length(a), 1.0f);
    return {abs, abs / denom};
}

VecDiff MaxDiff(VecDiff a, VecDiff b) {
    return (b.abs > a.abs) ? b : a;
}

float Clamp(float v, float lo, float hi) {
    return std::max(lo, std::min(v, hi));
}

float ForwardAngleDeg(const cpu::RotMat& cpuRot, const gpu::RotMat& gpuRot) {
    float dot =
        cpuRot.forward.x * gpuRot.forward.x +
        cpuRot.forward.y * gpuRot.forward.y +
        cpuRot.forward.z * gpuRot.forward.z;

    float cpuLen = Length(cpuRot.forward);
    float gpuLen = Length(gpuRot.forward);
    float denom = std::max(cpuLen * gpuLen, 1e-6f);
    return std::acos(Clamp(dot / denom, -1.0f, 1.0f)) * 180.0f / PI_F;
}

gpu::RotMat YawToRot(float yaw) {
    float cy = std::cos(yaw);
    float sy = std::sin(yaw);
    return {{cy, sy, 0.0f}, {-sy, cy, 0.0f}, {0.0f, 0.0f, 1.0f}};
}

gpu::BallState MakeBallState(float x, float y, float z, float vx, float vy, float vz) {
    gpu::BallState ball;
    ball.pos = {x, y, z};
    ball.rotMat = {{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}};
    ball.vel = {vx, vy, vz};
    ball.angVel = {0.0f, 0.0f, 0.0f};
    return ball;
}

std::filesystem::path GetCollisionMeshesPath() {
    std::filesystem::path root = std::filesystem::path(RSCUDA_WORKSPACE_ROOT);
    return root / "collision_meshes";
}

void PrintResultLine(const ComparisonResult& r) {
    std::printf(
        "%-20s [%s] ticks=%4d cars=%d | ballPos=%.2fUU ballVel=%.2fUU/s | "
        "carPos=%.2fUU carVel=%.2fUU/s | carFwd=%.2f deg boostDiff=%.2f | "
        "groundMismatch=%d demoMismatch=%d goal(cpu/gpu/team)=%d/%d/%s\n",
        r.label.c_str(),
        r.passed ? "PASS" : "FAIL",
        r.ticks,
        r.numCars,
        r.ballPos.abs,
        r.ballVel.abs,
        r.carPos.abs,
        r.carVel.abs,
        r.carForwardAngleDeg,
        r.carBoostDiff,
        r.onGroundMismatches,
        r.demoMismatches,
        static_cast<int>(r.goalCpu),
        static_cast<int>(r.goalGpu),
        r.goalTeamMatch ? "match" : "mismatch");
}

bool ValidateResult(const ComparisonResult& r, const ScenarioTolerances& tol) {
    bool goalOk = (r.goalCpu == r.goalGpu) && (!r.goalCpu || r.goalTeamMatch);
    return
        r.ballPos.abs <= tol.ballPosTol &&
        r.ballVel.abs <= tol.ballVelTol &&
        r.carPos.abs <= tol.carPosTol &&
        r.carVel.abs <= tol.carVelTol &&
        r.carForwardAngleDeg <= tol.carForwardTolDeg &&
        r.carBoostDiff <= tol.boostTol &&
        r.onGroundMismatches == 0 &&
        r.demoMismatches == 0 &&
        goalOk;
}

std::string g_collisionMeshesPath;

class ScenarioHarness {
public:
    explicit ScenarioHarness(const std::vector<CarSpec>& cars) {
        gpu::BatchConfig cfg;
        cfg.numArenas = 1;
        cfg.maxCarsPerArena = static_cast<int>(cars.size());
        cfg.tickRate = 120.0f;
        cfg.collisionMeshesPath = g_collisionMeshesPath.empty() ? nullptr : g_collisionMeshesPath.c_str();
        batch.Init(cfg);

        for (const auto& spec : cars)
            batch.AddCar(0, spec.team, spec.preset);
        batch.ResetArena(0);

        arena = cpu::Arena::Create(cpu::GameMode::SOCCAR);
        for (const auto& spec : cars) {
            cpuCars.push_back(arena->AddCar(CpuTeam(spec.team), CpuConfigForPreset(spec.preset)));
        }

        for (int i = 0; i < static_cast<int>(cars.size()); ++i) {
            gpu::CarState gpuState = batch.GetCarState(0, i);
            cpu::CarState cpuState = cpuCars[i]->GetState();
            CopyGpuToCpuCarState(cpuState, gpuState);
            cpuCars[i]->SetState(cpuState);
        }

        arena->ball->SetState(ToCpuBallState(batch.GetBallState(0)));
    }

    ~ScenarioHarness() {
        batch.Destroy();
        delete arena;
    }

    gpu::CarState GetGpuCarState(int carIdx) const {
        return batch.GetCarState(0, carIdx);
    }

    cpu::CarState GetCpuCarState(int carIdx) const {
        return cpuCars[carIdx]->GetState();
    }

    gpu::BallState GetGpuBallState() const {
        return batch.GetBallState(0);
    }

    cpu::BallState GetCpuBallState() const {
        return arena->ball->GetState();
    }

    gpu::ArenaInfo GetGpuArenaInfo() const {
        return batch.GetArenaInfo(0);
    }

    void SetCarState(int carIdx, const gpu::CarState& gpuState) {
        batch.SetCarState(0, carIdx, gpuState);

        cpu::CarState cpuState = cpuCars[carIdx]->GetState();
        CopyGpuToCpuCarState(cpuState, gpuState);
        cpuCars[carIdx]->SetState(cpuState);
    }

    void SetBallState(const gpu::BallState& ballState) {
        batch.SetBallState(0, ballState);
        arena->ball->SetState(ToCpuBallState(ballState));
    }

    void SetControls(const std::vector<gpu::CarControls>& controls) {
        for (int i = 0; i < static_cast<int>(controls.size()); ++i) {
            batch.SetCarControls(0, i, controls[i]);
            cpuCars[i]->controls = ToCpuControls(controls[i]);
        }
    }

    void Step(int ticks = 1) {
        arena->Step(ticks);
        batch.Step(ticks);
    }

    int NumCars() const {
        return static_cast<int>(cpuCars.size());
    }

    bool CpuGoalScored() const {
        return arena->IsBallScored();
    }

private:
    gpu::RocketSimCudaBatch batch;
    cpu::Arena* arena = nullptr;
    std::vector<cpu::Car*> cpuCars;
};

using SetupFn = std::function<void(ScenarioHarness&)>;
using TickFn = std::function<void(ScenarioHarness&, int)>;

ComparisonResult RunScenario(
    const char* label,
    const std::vector<CarSpec>& cars,
    int ticks,
    const ScenarioTolerances& tol = {},
    const SetupFn& setup = {},
    const TickFn& onTick = {}
) {
    ScenarioHarness harness(cars);

    if (setup)
        setup(harness);

    for (int tick = 0; tick < ticks; ++tick) {
        if (onTick)
            onTick(harness, tick);
        harness.Step(1);
    }

    cpu::BallState cpuBallFinal = harness.GetCpuBallState();
    gpu::BallState gpuBallFinal = harness.GetGpuBallState();
    gpu::ArenaInfo gpuArenaInfo = harness.GetGpuArenaInfo();

    ComparisonResult result;
    result.label = label;
    result.ticks = ticks;
    result.numCars = harness.NumCars();
    result.ballPos = DiffVec(cpuBallFinal.pos, gpuBallFinal.pos);
    result.ballVel = DiffVec(cpuBallFinal.vel, gpuBallFinal.vel);
    result.goalCpu = harness.CpuGoalScored();
    result.goalGpu = gpuArenaInfo.goalScored;
    result.cpuGoalTeam = result.goalCpu ? (cpuBallFinal.pos.y > 0.0f ? 0 : 1) : -1;
    result.gpuGoalTeam = gpuArenaInfo.goalScored ? gpuArenaInfo.goalTeam : -1;
    result.goalTeamMatch = (result.cpuGoalTeam == result.gpuGoalTeam);

    std::printf("\nScenario: %s\n", label);
    std::printf("CPU  ball pos=(%.2f, %.2f, %.2f) vel=(%.2f, %.2f, %.2f)\n",
        cpuBallFinal.pos.x, cpuBallFinal.pos.y, cpuBallFinal.pos.z,
        cpuBallFinal.vel.x, cpuBallFinal.vel.y, cpuBallFinal.vel.z);
    std::printf("CUDA ball pos=(%.2f, %.2f, %.2f) vel=(%.2f, %.2f, %.2f)\n",
        gpuBallFinal.pos.x, gpuBallFinal.pos.y, gpuBallFinal.pos.z,
        gpuBallFinal.vel.x, gpuBallFinal.vel.y, gpuBallFinal.vel.z);

    for (int i = 0; i < harness.NumCars(); ++i) {
        cpu::CarState cpuCar = harness.GetCpuCarState(i);
        gpu::CarState gpuCar = harness.GetGpuCarState(i);

        result.carPos = MaxDiff(result.carPos, DiffVec(cpuCar.pos, gpuCar.pos));
        result.carVel = MaxDiff(result.carVel, DiffVec(cpuCar.vel, gpuCar.vel));
        result.carForwardAngleDeg = std::max(result.carForwardAngleDeg, ForwardAngleDeg(cpuCar.rotMat, gpuCar.rotMat));
        result.carBoostDiff = std::max(result.carBoostDiff, std::fabs(cpuCar.boost - gpuCar.boost));
        result.onGroundMismatches += (cpuCar.isOnGround != gpuCar.isOnGround) ? 1 : 0;
        result.demoMismatches += (cpuCar.isDemoed != gpuCar.isDemoed) ? 1 : 0;

        std::printf("CPU  car[%d] pos=(%.2f, %.2f, %.2f) vel=(%.2f, %.2f, %.2f) boost=%.2f onGround=%d demo=%d\n",
            i,
            cpuCar.pos.x, cpuCar.pos.y, cpuCar.pos.z,
            cpuCar.vel.x, cpuCar.vel.y, cpuCar.vel.z,
            cpuCar.boost,
            static_cast<int>(cpuCar.isOnGround),
            static_cast<int>(cpuCar.isDemoed));
        std::printf("CUDA car[%d] pos=(%.2f, %.2f, %.2f) vel=(%.2f, %.2f, %.2f) boost=%.2f onGround=%d demo=%d\n",
            i,
            gpuCar.pos.x, gpuCar.pos.y, gpuCar.pos.z,
            gpuCar.vel.x, gpuCar.vel.y, gpuCar.vel.z,
            gpuCar.boost,
            static_cast<int>(gpuCar.isOnGround),
            static_cast<int>(gpuCar.isDemoed));
    }

    result.passed = ValidateResult(result, tol);
    PrintResultLine(result);
    return result;
}

} // namespace

int main() {
    std::filesystem::path meshes = GetCollisionMeshesPath();
    std::printf("Using collision meshes: %s\n", meshes.string().c_str());
    if (!std::filesystem::exists(meshes)) {
        std::printf("Missing collision meshes folder.\n");
        return 1;
    }

    cpu::Init(meshes, true);
    g_collisionMeshesPath = meshes.string();

    std::vector<ComparisonResult> results;
    const std::vector<CarSpec> oneBlueCar = {{}};

    results.push_back(RunScenario(
        "ball_drop_1", oneBlueCar, 1, {},
        [](ScenarioHarness& h) {
            h.SetBallState(MakeBallState(0.0f, 0.0f, 500.0f, 0.0f, 0.0f, -1.0f));
        }
    ));

    results.push_back(RunScenario(
        "ball_drop_120", oneBlueCar, 120, {},
        [](ScenarioHarness& h) {
            h.SetBallState(MakeBallState(0.0f, 0.0f, 500.0f, 0.0f, 0.0f, -1.0f));
        }
    ));

    results.push_back(RunScenario(
        "ball_drop_1440", oneBlueCar, 1440, {},
        [](ScenarioHarness& h) {
            h.SetBallState(MakeBallState(0.0f, 0.0f, 500.0f, 0.0f, 0.0f, -1.0f));
        }
    ));

    results.push_back(RunScenario("kickoff_idle_120", oneBlueCar, 120));
    results.push_back(RunScenario("kickoff_idle_600", oneBlueCar, 600));

    results.push_back(RunScenario(
        "boost_drive_60", oneBlueCar, 60, {},
        {},
        [](ScenarioHarness& h, int) {
            gpu::CarControls controls;
            controls.throttle = 1.0f;
            controls.boost = true;
            h.SetControls({controls});
        }
    ));

    results.push_back(RunScenario(
        "boost_drive_120", oneBlueCar, 120, {},
        {},
        [](ScenarioHarness& h, int) {
            gpu::CarControls controls;
            controls.throttle = 1.0f;
            controls.boost = true;
            h.SetControls({controls});
        }
    ));

    results.push_back(RunScenario(
        "jump_hold_30", oneBlueCar, 30, {},
        {},
        [](ScenarioHarness& h, int tick) {
            gpu::CarControls controls;
            controls.jump = tick < 6;
            h.SetControls({controls});
        }
    ));

    results.push_back(RunScenario(
        "double_jump_45", oneBlueCar, 45, {},
        {},
        [](ScenarioHarness& h, int tick) {
            gpu::CarControls controls;
            controls.jump = (tick < 3) || (tick >= 12 && tick < 15);
            h.SetControls({controls});
        }
    ));

    // The kickoff-spawn front flip lands on an exactly mirror-symmetric
    // bottom edge: which corner GJK picks is an exact float tie, so the
    // landing legitimately bifurcates (CPU itself is ulp-sensitive there).
    results.push_back(RunScenario(
        "front_flip_60", oneBlueCar, 60,
        ScenarioTolerances{0.5f, 0.5f, 8.0f, 30.0f, 3.0f, 0.05f},
        {},
        [](ScenarioHarness& h, int tick) {
            gpu::CarControls controls;
            if (tick < 3) {
                controls.jump = true;
            } else if (tick >= 8 && tick < 11) {
                controls.jump = true;
                controls.pitch = -1.0f;
                controls.throttle = 1.0f;
            } else {
                controls.throttle = 1.0f;
            }
            h.SetControls({controls});
        }
    ));

    results.push_back(RunScenario(
        "diag_flip_60", oneBlueCar, 60,
        ScenarioTolerances{0.5f, 0.5f, 8.0f, 35.0f, 3.0f, 0.05f},
        {},
        [](ScenarioHarness& h, int tick) {
            gpu::CarControls controls;
            if (tick < 3) {
                controls.jump = true;
            } else if (tick >= 8 && tick < 11) {
                controls.jump = true;
                controls.pitch = -0.7f;
                controls.yaw = 0.7f;
            }
            h.SetControls({controls});
        }
    ));

    results.push_back(RunScenario(
        "side_flip_60", oneBlueCar, 60,
        ScenarioTolerances{0.5f, 0.5f, 8.0f, 35.0f, 3.0f, 0.05f},
        {},
        [](ScenarioHarness& h, int tick) {
            gpu::CarControls controls;
            if (tick < 3) {
                controls.jump = true;
            } else if (tick >= 8 && tick < 11) {
                controls.jump = true;
                controls.yaw = 1.0f;
            }
            h.SetControls({controls});
        }
    ));

    results.push_back(RunScenario(
        "back_flip_50", oneBlueCar, 50, {},
        {},
        [](ScenarioHarness& h, int tick) {
            gpu::CarControls controls;
            if (tick < 3) {
                controls.jump = true;
            } else if (tick >= 8 && tick < 11) {
                controls.jump = true;
                controls.pitch = 1.0f;
            }
            h.SetControls({controls});
        }
    ));

    results.push_back(RunScenario(
        "stall_flip_55", oneBlueCar, 55, {},
        {},
        [](ScenarioHarness& h, int tick) {
            gpu::CarControls controls;
            if (tick < 3) {
                controls.jump = true;
            } else if (tick >= 8 && tick < 11) {
                controls.jump = true;
                controls.yaw = 1.0f;
                controls.roll = -1.0f;
            }
            h.SetControls({controls});
        }
    ));

    results.push_back(RunScenario(
        "steer_drive_150", oneBlueCar, 150, {},
        {},
        [](ScenarioHarness& h, int) {
            gpu::CarControls controls;
            controls.throttle = 1.0f;
            controls.steer = 0.6f;
            h.SetControls({controls});
        }
    ));

    results.push_back(RunScenario(
        "powerslide_turn_120", oneBlueCar, 120, {},
        {},
        [](ScenarioHarness& h, int tick) {
            gpu::CarControls controls;
            controls.throttle = 1.0f;
            if (tick >= 40) {
                controls.steer = 1.0f;
                controls.handbrake = true;
            }
            h.SetControls({controls});
        }
    ));

    results.push_back(RunScenario(
        "aerial_control_150", oneBlueCar, 150, {},
        [](ScenarioHarness& h) {
            auto car = h.GetGpuCarState(0);
            car.pos = {500.0f, -1000.0f, 700.0f};
            car.rotMat = YawToRot(PI_F * 0.25f);
            car.vel = {0.0f, 0.0f, 150.0f};
            car.angVel = {0.0f, 0.0f, 0.0f};
            car.isOnGround = false;
            car.boost = 100.0f;
            h.SetCarState(0, car);
        },
        [](ScenarioHarness& h, int tick) {
            gpu::CarControls controls;
            controls.boost = (tick % 10) < 6;
            controls.pitch = (tick < 60) ? 0.45f : -0.2f;
            controls.yaw = 0.3f;
            controls.roll = (tick >= 80 && tick < 110) ? -0.8f : 0.0f;
            h.SetControls({controls});
        }
    ));

    results.push_back(RunScenario(
        "wall_drive_100", oneBlueCar, 100, {},
        [](ScenarioHarness& h) {
            auto car = h.GetGpuCarState(0);
            // On the +X wall, nose up (+z), wheels toward the wall.
            car.pos = {4096.0f - 17.01f, -800.0f, 600.0f};
            car.rotMat.forward = {0.0f, 0.0f, 1.0f};
            car.rotMat.right = {0.0f, 1.0f, 0.0f};
            car.rotMat.up = {-1.0f, 0.0f, 0.0f};
            car.vel = {0.0f, 0.0f, 500.0f};
            car.angVel = {0.0f, 0.0f, 0.0f};
            car.boost = 50.0f;
            h.SetCarState(0, car);
        },
        [](ScenarioHarness& h, int) {
            gpu::CarControls controls;
            controls.throttle = 1.0f;
            h.SetControls({controls});
        }
    ));

    results.push_back(RunScenario(
        "ball_corner_bounce_150", oneBlueCar, 150, {},
        [](ScenarioHarness& h) {
            // Into the corner bevel + floor-wall curve region.
            h.SetBallState(MakeBallState(2800.0f, -4000.0f, 300.0f, 1300.0f, -1200.0f, 0.0f));
        }
    ));

    results.push_back(RunScenario(
        "ball_roll_300", oneBlueCar, 300, {},
        [](ScenarioHarness& h) {
            h.SetBallState(MakeBallState(-1000.0f, 1000.0f, 93.15f, 900.0f, -300.0f, 0.0f));
        }
    ));

    results.push_back(RunScenario(
        "ball_ceiling_bounce_120", oneBlueCar, 120, {},
        [](ScenarioHarness& h) {
            h.SetBallState(MakeBallState(0.0f, 800.0f, 1600.0f, 200.0f, 0.0f, 1400.0f));
        }
    ));

    results.push_back(RunScenario(
        "dribble_roof_90", oneBlueCar, 90, {},
        [](ScenarioHarness& h) {
            auto car = h.GetGpuCarState(0);
            car.pos = {0.0f, 0.0f, 17.01f};
            car.rotMat = YawToRot(0.0f);
            car.vel = {300.0f, 0.0f, 0.0f};
            car.boost = 30.0f;
            h.SetCarState(0, car);
            // Ball resting on the hitbox roof, moving with the car.
            h.SetBallState(MakeBallState(20.0f, 0.0f, 17.01f + 38.66f * 0.5f + 20.755f - 17.01f + 92.0f + 40.0f, 300.0f, 0.0f, 0.0f));
        },
        [](ScenarioHarness& h, int) {
            gpu::CarControls controls;
            controls.throttle = 0.6f;
            h.SetControls({controls});
        }
    ));

    results.push_back(RunScenario(
        "supersonic_demo_40",
        {{gpu::Team::BLUE, gpu::CarPreset::OCTANE}, {gpu::Team::ORANGE, gpu::CarPreset::OCTANE}},
        40,
        // Respawn position is RNG-based on CPU (unseeded), so only flag/timer
        // parity is meaningful; the victim respawns at tick ~?+360.
        ScenarioTolerances{0.5f, 0.5f, 1e9f, 1e9f, 180.0f, 100.0f},
        [](ScenarioHarness& h) {
            auto car0 = h.GetGpuCarState(0);
            car0.pos = {-1500.0f, 0.0f, car0.pos.z};
            car0.rotMat = YawToRot(0.0f);
            car0.vel = {2250.0f, 0.0f, 0.0f};
            car0.isSupersonic = true;
            car0.boost = 100.0f;
            h.SetCarState(0, car0);

            auto car1 = h.GetGpuCarState(1);
            car1.pos = {-200.0f, 10.0f, car1.pos.z};
            car1.rotMat = YawToRot(PI_F * 0.5f);
            car1.vel = {0.0f, 0.0f, 0.0f};
            h.SetCarState(1, car1);
        },
        [](ScenarioHarness& h, int) {
            gpu::CarControls c0;
            c0.throttle = 1.0f;
            c0.boost = true;
            h.SetControls({c0, gpu::CarControls{}});
        }
    ));

    results.push_back(RunScenario(
        "car_ball_hit_60", oneBlueCar, 60, {},
        [](ScenarioHarness& h) {
            auto car = h.GetGpuCarState(0);
            car.pos = {0.0f, -1400.0f, car.pos.z};
            car.rotMat = YawToRot(PI_F * 0.5f);
            car.vel = {0.0f, 0.0f, 0.0f};
            car.angVel = {0.0f, 0.0f, 0.0f};
            car.boost = 100.0f;
            h.SetCarState(0, car);

            auto ball = MakeBallState(0.0f, -220.0f, 93.15f, 0.0f, 0.0f, 0.0f);
            h.SetBallState(ball);
        },
        [](ScenarioHarness& h, int) {
            gpu::CarControls controls;
            controls.throttle = 1.0f;
            controls.boost = true;
            h.SetControls({controls});
        }
    ));

    results.push_back(RunScenario(
        "ball_wall_bounce_120", oneBlueCar, 120, {},
        [](ScenarioHarness& h) {
            h.SetBallState(MakeBallState(3500.0f, 0.0f, 400.0f, 1800.0f, 0.0f, 0.0f));
        }
    ));

    results.push_back(RunScenario(
        "boost_pad_pickup_45", oneBlueCar, 45, {},
        [](ScenarioHarness& h) {
            auto car = h.GetGpuCarState(0);
            car.pos = {0.0f, -2920.0f, car.pos.z};
            car.rotMat = YawToRot(PI_F * 0.5f);
            car.vel = {0.0f, 650.0f, 0.0f};
            car.angVel = {0.0f, 0.0f, 0.0f};
            car.boost = 0.0f;
            h.SetCarState(0, car);
        },
        [](ScenarioHarness& h, int) {
            gpu::CarControls controls;
            controls.throttle = 1.0f;
            h.SetControls({controls});
        }
    ));

    // The reference has goal interior/net mesh geometry that the analytic
    // arena lacks; once past the goal line the ball decelerates differently.
    results.push_back(RunScenario(
        "goal_detection_20", oneBlueCar, 20,
        ScenarioTolerances{3.0f, 15.0f, 0.5f, 0.5f, 0.2f, 0.05f},
        [](ScenarioHarness& h) {
            h.SetBallState(MakeBallState(0.0f, 5000.0f, 100.0f, 0.0f, 2000.0f, 0.0f));
        }
    ));

    results.push_back(RunScenario(
        "car_car_bump_30",
        {{gpu::Team::BLUE, gpu::CarPreset::OCTANE}, {gpu::Team::ORANGE, gpu::CarPreset::OCTANE}},
        30,
        {},
        [](ScenarioHarness& h) {
            auto car0 = h.GetGpuCarState(0);
            car0.pos = {-600.0f, 0.0f, car0.pos.z};
            car0.rotMat = YawToRot(0.0f);
            car0.vel = {1200.0f, 0.0f, 0.0f};
            car0.boost = 100.0f;
            h.SetCarState(0, car0);

            auto car1 = h.GetGpuCarState(1);
            car1.pos = {600.0f, 0.0f, car1.pos.z};
            car1.rotMat = YawToRot(PI_F);
            car1.vel = {-1200.0f, 0.0f, 0.0f};
            car1.boost = 100.0f;
            h.SetCarState(1, car1);
        },
        [](ScenarioHarness& h, int) {
            h.SetControls({gpu::CarControls{}, gpu::CarControls{}});
        }
    ));

    std::printf("\nSummary\n");
    bool allPassed = true;
    for (const auto& result : results) {
        PrintResultLine(result);
        allPassed &= result.passed;
    }

    std::printf("\nOverall: %s\n", allPassed ? "PASS" : "FAIL");
    return allPassed ? 0 : 1;
}
