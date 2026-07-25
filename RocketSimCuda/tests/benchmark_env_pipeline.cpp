#include <RocketSimCuda.h>
#include <RocketSim.h>

#include <RLGymCPP/EnvSet/EnvSet.h>
#include <RLGymCPP/RewardCore/CommonRewards.h>
#include <RLGymCPP/RewardCore/ZeroSumReward.h>
#include <RLGymCPP/TerminalConditions/NoTouchCondition.h>
#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>
#include <RLGymCPP/OBSBuilders/AdvancedObs.h>
#include <RLGymCPP/StateSetters/KickoffState.h>
#include <RLGymCPP/ActionParsers/DefaultAction.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <vector>

namespace cpu = RocketSim;
namespace gpu = rsc;
using namespace RLGC;

namespace {

constexpr int NUM_ARENAS = 256;
constexpr int CARS_PER_ARENA = 2;
constexpr int TICK_SKIP = 8;
constexpr int ACTION_DELAY = 7;
constexpr int RAW_ITERS = 160;
constexpr int ENVSET_ITERS = 120;

struct RawBenchResult {
    double controlMs = 0.0;
    double stepMs = 0.0;
    double readMs = 0.0;
    double totalMs = 0.0;
    double arenaTicksPerSecond = 0.0;
    double checksum = 0.0;
};

struct EnvBenchResult {
    double resetMs = 0.0;
    double firstHalfMs = 0.0;
    double secondHalfMs = 0.0;
    double obsReadMs = 0.0;
    double totalMs = 0.0;
    double envStepsPerSecond = 0.0;
    double playerStepsPerSecond = 0.0;
    double checksum = 0.0;
};

double ElapsedMs(const std::chrono::steady_clock::time_point& start) {
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::filesystem::path GetCollisionMeshesPath() {
    std::filesystem::path root = std::filesystem::path(RSCUDA_WORKSPACE_ROOT);
    return root / "collision_meshes";
}

gpu::CarControls MakeBenchControls(int arenaIdx, int carIdx) {
    gpu::CarControls controls;
    controls.throttle = (carIdx % 2 == 0) ? 1.0f : 0.5f;
    controls.steer = (arenaIdx % 3 == 0) ? 0.25f : 0.0f;
    controls.boost = (arenaIdx + carIdx) % 5 == 0;
    controls.jump = false;
    controls.handbrake = false;
    return controls;
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

EnvCreateResult BenchEnvCreate(int) {
    std::vector<WeightedReward> rewards = {
        { new AirReward(), 0.25f },
        { new FaceBallReward(), 0.25f },
        { new VelocityPlayerToBallReward(), 4.0f },
        { new StrongTouchReward(20, 100), 60.0f },
        { new ZeroSumReward(new VelocityBallToGoalReward(), 1), 2.0f },
        { new PickupBoostReward(), 10.0f },
        { new SaveBoostReward(), 0.2f },
        { new ZeroSumReward(new BumpReward(), 0.5f), 20.0f },
        { new ZeroSumReward(new DemoReward(), 0.5f), 80.0f },
        { new GoalReward(), 150.0f }
    };

    std::vector<TerminalCondition*> terminalConditions = {
        new NoTouchCondition(10),
        new GoalScoreCondition()
    };

    auto arena = cpu::Arena::Create(cpu::GameMode::SOCCAR);
    arena->AddCar(cpu::Team::BLUE);
    arena->AddCar(cpu::Team::ORANGE);

    EnvCreateResult result = {};
    result.arena = arena;
    result.rewards = rewards;
    result.terminalConditions = terminalConditions;
    result.obsBuilder = new AdvancedObs();
    result.actionParser = new DefaultAction();
    result.stateSetter = new KickoffState();
    return result;
}

RawBenchResult BenchmarkCpuRaw() {
    std::vector<cpu::Arena*> arenas;
    arenas.reserve(NUM_ARENAS);

    for (int arenaIdx = 0; arenaIdx < NUM_ARENAS; ++arenaIdx) {
        cpu::Arena* arena = cpu::Arena::Create(cpu::GameMode::SOCCAR);
        for (int carIdx = 0; carIdx < CARS_PER_ARENA; ++carIdx) {
            cpu::Team team = (carIdx % 2 == 0) ? cpu::Team::BLUE : cpu::Team::ORANGE;
            arena->AddCar(team, cpu::CAR_CONFIG_OCTANE);
        }
        arenas.push_back(arena);
    }

    RawBenchResult result;
    auto totalStart = std::chrono::steady_clock::now();

    for (int iter = 0; iter < RAW_ITERS; ++iter) {
        auto controlStart = std::chrono::steady_clock::now();
        for (int arenaIdx = 0; arenaIdx < NUM_ARENAS; ++arenaIdx) {
            int carIdx = 0;
            for (auto* car : arenas[arenaIdx]->_cars) {
                car->controls = ToCpuControls(MakeBenchControls(arenaIdx + iter, carIdx));
                carIdx++;
            }
        }
        result.controlMs += ElapsedMs(controlStart);

        auto stepStart = std::chrono::steady_clock::now();
        for (auto* arena : arenas)
            arena->Step(TICK_SKIP);
        result.stepMs += ElapsedMs(stepStart);

        auto readStart = std::chrono::steady_clock::now();
        for (auto* arena : arenas) {
            auto ball = arena->ball->GetState();
            result.checksum += ball.pos.x + ball.pos.y + ball.pos.z + ball.vel.Length();
            for (auto* car : arena->_cars) {
                auto state = car->GetState();
                result.checksum += state.pos.x + state.pos.y + state.pos.z + state.vel.Length() + state.boost;
            }
        }
        result.readMs += ElapsedMs(readStart);
    }

    result.totalMs = ElapsedMs(totalStart);
    result.arenaTicksPerSecond = (NUM_ARENAS * RAW_ITERS * TICK_SKIP) / (result.totalMs / 1000.0);

    for (auto* arena : arenas)
        delete arena;

    return result;
}

RawBenchResult BenchmarkCudaRaw() {
    gpu::RocketSimCudaBatch batch;
    gpu::BatchConfig cfg;
    cfg.numArenas = NUM_ARENAS;
    cfg.maxCarsPerArena = CARS_PER_ARENA;
    cfg.tickRate = 120.0f;
    batch.Init(cfg);

    for (int arenaIdx = 0; arenaIdx < NUM_ARENAS; ++arenaIdx) {
    batch.AddCar(arenaIdx, gpu::Team::BLUE, gpu::CarPreset::OCTANE);
    batch.AddCar(arenaIdx, gpu::Team::ORANGE, gpu::CarPreset::OCTANE);
    }
    batch.ResetAllArenas();

    std::vector<gpu::CarControls> controls(NUM_ARENAS * CARS_PER_ARENA);
    std::vector<gpu::CarState> carStates(NUM_ARENAS * CARS_PER_ARENA);
    std::vector<gpu::BallState> ballStates(NUM_ARENAS);

    RawBenchResult result;
    auto totalStart = std::chrono::steady_clock::now();

    for (int iter = 0; iter < RAW_ITERS; ++iter) {
        auto controlStart = std::chrono::steady_clock::now();
        for (int arenaIdx = 0; arenaIdx < NUM_ARENAS; ++arenaIdx) {
            for (int carIdx = 0; carIdx < CARS_PER_ARENA; ++carIdx) {
                controls[arenaIdx * CARS_PER_ARENA + carIdx] = MakeBenchControls(arenaIdx + iter, carIdx);
            }
        }
        batch.SetAllCarControls(controls.data());
        result.controlMs += ElapsedMs(controlStart);

        auto stepStart = std::chrono::steady_clock::now();
        batch.Step(TICK_SKIP);
        result.stepMs += ElapsedMs(stepStart);

        auto readStart = std::chrono::steady_clock::now();
        batch.GetAllCarStates(carStates.data());
        batch.GetAllBallStates(ballStates.data());
        for (const auto& ball : ballStates)
            result.checksum += ball.pos.x + ball.pos.y + ball.pos.z + std::sqrt(ball.vel.x * ball.vel.x + ball.vel.y * ball.vel.y + ball.vel.z * ball.vel.z);
        for (const auto& car : carStates)
            result.checksum += car.pos.x + car.pos.y + car.pos.z + std::sqrt(car.vel.x * car.vel.x + car.vel.y * car.vel.y + car.vel.z * car.vel.z) + car.boost;
        result.readMs += ElapsedMs(readStart);
    }

    result.totalMs = ElapsedMs(totalStart);
    result.arenaTicksPerSecond = (NUM_ARENAS * RAW_ITERS * TICK_SKIP) / (result.totalMs / 1000.0);

    batch.Destroy();
    return result;
}

EnvBenchResult BenchmarkCpuEnvSet() {
    EnvSetConfig cfg = {};
    cfg.envCreateFn = BenchEnvCreate;
    cfg.numArenas = NUM_ARENAS;
    cfg.tickSkip = TICK_SKIP;
    cfg.actionDelay = ACTION_DELAY;
    cfg.saveRewards = true;

    EnvSet env(cfg);
    IList actions(env.state.numPlayers, 0);

    EnvBenchResult result;
    auto totalStart = std::chrono::steady_clock::now();

    for (int iter = 0; iter < ENVSET_ITERS; ++iter) {
        for (int i = 0; i < env.state.numPlayers; ++i)
            actions[i] = (i + iter) % env.actionParsers[0]->GetActionAmount();

        auto resetStart = std::chrono::steady_clock::now();
        env.Reset();
        result.resetMs += ElapsedMs(resetStart);

        auto firstHalfStart = std::chrono::steady_clock::now();
        env.StepFirstHalf(true);
        env.Sync();
        result.firstHalfMs += ElapsedMs(firstHalfStart);

        auto secondHalfStart = std::chrono::steady_clock::now();
        env.StepSecondHalf(actions, false);
        result.secondHalfMs += ElapsedMs(secondHalfStart);

        auto readStart = std::chrono::steady_clock::now();
        for (float obs : env.state.obs.data)
            result.checksum += obs;
        for (uint8_t mask : env.state.actionMasks.data)
            result.checksum += mask;
        for (float reward : env.state.rewards)
            result.checksum += reward;
        result.obsReadMs += ElapsedMs(readStart);
    }

    result.totalMs = ElapsedMs(totalStart);
    result.envStepsPerSecond = (NUM_ARENAS * ENVSET_ITERS) / (result.totalMs / 1000.0);
    result.playerStepsPerSecond = (env.state.numPlayers * ENVSET_ITERS) / (result.totalMs / 1000.0);
    return result;
}

EnvBenchResult BenchmarkCudaEnvSet() {
    EnvSetConfig cfg = {};
    cfg.envCreateFn = BenchEnvCreate;
    cfg.numArenas = NUM_ARENAS;
    cfg.tickSkip = TICK_SKIP;
    cfg.actionDelay = ACTION_DELAY;
    cfg.saveRewards = false;
    cfg.physicsBackend = EnvPhysicsBackend::ROCKETSIM_CUDA;
    cfg.syncCudaWorldStateToCpu = false;

    EnvSet env(cfg);
    IList actions(env.state.numPlayers, 0);

    EnvBenchResult result;
    auto totalStart = std::chrono::steady_clock::now();

    for (int iter = 0; iter < ENVSET_ITERS; ++iter) {
        for (int i = 0; i < env.state.numPlayers; ++i)
            actions[i] = (i + iter) % env.actionParsers[0]->GetActionAmount();

        auto resetStart = std::chrono::steady_clock::now();
        env.Reset();
        result.resetMs += ElapsedMs(resetStart);

        auto firstHalfStart = std::chrono::steady_clock::now();
        env.StepFirstHalf(true);
        env.Sync();
        result.firstHalfMs += ElapsedMs(firstHalfStart);

        auto secondHalfStart = std::chrono::steady_clock::now();
        env.StepSecondHalf(actions, false);
        result.secondHalfMs += ElapsedMs(secondHalfStart);

        auto readStart = std::chrono::steady_clock::now();
        for (float obs : env.state.obs.data)
            result.checksum += obs;
        for (uint8_t mask : env.state.actionMasks.data)
            result.checksum += mask;
        for (float reward : env.state.rewards)
            result.checksum += reward;
        result.obsReadMs += ElapsedMs(readStart);
    }

    result.totalMs = ElapsedMs(totalStart);
    result.envStepsPerSecond = (NUM_ARENAS * ENVSET_ITERS) / (result.totalMs / 1000.0);
    result.playerStepsPerSecond = (env.state.numPlayers * ENVSET_ITERS) / (result.totalMs / 1000.0);
    return result;
}

void PrintRawResult(const char* label, const RawBenchResult& r) {
    std::printf(
        "%s\n"
        "  controls: %.2f ms\n"
        "  step:     %.2f ms\n"
        "  read:     %.2f ms\n"
        "  total:    %.2f ms\n"
        "  throughput: %.0f arena-ticks/s\n"
        "  checksum: %.3f\n",
        label,
        r.controlMs,
        r.stepMs,
        r.readMs,
        r.totalMs,
        r.arenaTicksPerSecond,
        r.checksum);
}

void PrintEnvResult(const char* label, const EnvBenchResult& r) {
    std::printf(
        "%s\n"
        "  reset:      %.2f ms\n"
        "  first half: %.2f ms\n"
        "  second half:%.2f ms\n"
        "  obs read:   %.2f ms\n"
        "  total:      %.2f ms\n"
        "  throughput: %.0f env-steps/s | %.0f player-steps/s\n"
        "  checksum: %.3f\n",
        label,
        r.resetMs,
        r.firstHalfMs,
        r.secondHalfMs,
        r.obsReadMs,
        r.totalMs,
        r.envStepsPerSecond,
        r.playerStepsPerSecond,
        r.checksum);
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

    std::printf("Benchmark config: %d arenas, %d cars/arena, tickSkip=%d, actionDelay=%d\n\n",
        NUM_ARENAS, CARS_PER_ARENA, TICK_SKIP, ACTION_DELAY);

    RawBenchResult cpuRaw = BenchmarkCpuRaw();
    RawBenchResult cudaRaw = BenchmarkCudaRaw();
    EnvBenchResult cpuEnv = BenchmarkCpuEnvSet();
    EnvBenchResult cudaEnv = BenchmarkCudaEnvSet();

    PrintRawResult("CPU raw RocketSim", cpuRaw);
    std::printf("\n");
    PrintRawResult("CUDA raw RocketSimCudaBatch", cudaRaw);
    std::printf("\n");
    PrintEnvResult("CPU EnvSet (obs/rewards/masks included)", cpuEnv);
    std::printf("\n");
    PrintEnvResult("CUDA EnvSet (obs/rewards/masks included)", cudaEnv);

    return 0;
}
