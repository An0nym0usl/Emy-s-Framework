#include <RocketSimCuda.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

using namespace rsc;

int main(int argc, char** argv) {
    constexpr int kNumArenas = 1024;
    constexpr int kCarsPerArena = 2;
    constexpr int kWarmupTicks = 120;
    constexpr int kMeasuredSteps = 10;
    constexpr int kTicksPerStep = 120;

    printf("========================================\n");
    printf("  RocketSimCuda 1024-Arena Benchmark\n");
    printf("========================================\n");

    RocketSimCudaBatch batch;
    BatchConfig cfg;
    cfg.numArenas = kNumArenas;
    cfg.maxCarsPerArena = kCarsPerArena;
    cfg.tickRate = 120.f;

    // Use the real collision meshes when available (training configuration).
    std::string meshPath;
#ifdef RSCUDA_WORKSPACE_ROOT
    meshPath = (std::filesystem::path(RSCUDA_WORKSPACE_ROOT) / "collision_meshes").string();
#endif
    if (argc > 1)
        meshPath = argv[1];
    if (!meshPath.empty() && std::filesystem::exists(meshPath)) {
        printf("Collision meshes: %s\n", meshPath.c_str());
        cfg.collisionMeshesPath = meshPath.c_str();
    } else {
        printf("Collision meshes: (analytic fallback)\n");
    }

    batch.Init(cfg);

    for (int arena = 0; arena < kNumArenas; arena++) {
        batch.AddCar(arena, Team::BLUE, CarPreset::OCTANE);
        batch.AddCar(arena, Team::ORANGE, CarPreset::OCTANE);
    }

    std::vector<CarControls> controls(kNumArenas * kCarsPerArena);
    for (int arena = 0; arena < kNumArenas; arena++) {
        controls[arena * kCarsPerArena + 0].throttle = 1.0f;
        controls[arena * kCarsPerArena + 0].boost = (arena % 2) == 0;
        controls[arena * kCarsPerArena + 1].throttle = 1.0f;
        controls[arena * kCarsPerArena + 1].steer = -0.25f;
    }
    batch.SetAllCarControls(controls.data());

    printf("Warmup: %d arenas, %d ticks...\n", kNumArenas, kWarmupTicks);
    batch.Step(kWarmupTicks);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kMeasuredSteps; i++) {
        batch.Step(kTicksPerStep);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsedMs =
        std::chrono::duration<double, std::milli>(end - start).count();
    int measuredTicks = kMeasuredSteps * kTicksPerStep;
    double arenaStepsPerSecond =
        (static_cast<double>(kNumArenas) * measuredTicks) / (elapsedMs / 1000.0);

    ArenaInfo info0 = batch.GetArenaInfo(0);
    ArenaInfo infoLast = batch.GetArenaInfo(kNumArenas - 1);
    BallState b0 = batch.GetBallState(0);
    BallState bLast = batch.GetBallState(kNumArenas - 1);
    CarState c0 = batch.GetCarState(0, 0);
    CarState cLast = batch.GetCarState(kNumArenas - 1, 1);

    printf("Measured: %d arenas x %d ticks in %.2f ms\n",
           kNumArenas, measuredTicks, elapsedMs);
    printf("Throughput: %.0f arena-ticks/s\n", arenaStepsPerSecond);
    printf("Arena 0 tickCount: %llu\n", static_cast<unsigned long long>(info0.tickCount));
    printf("Arena %d tickCount: %llu\n", kNumArenas - 1,
           static_cast<unsigned long long>(infoLast.tickCount));
    printf("Arena 0 ball: pos=(%.1f, %.1f, %.1f) vel=(%.1f, %.1f, %.1f)\n",
           b0.pos.x, b0.pos.y, b0.pos.z, b0.vel.x, b0.vel.y, b0.vel.z);
    printf("Arena %d ball: pos=(%.1f, %.1f, %.1f) vel=(%.1f, %.1f, %.1f)\n",
           kNumArenas - 1,
           bLast.pos.x, bLast.pos.y, bLast.pos.z, bLast.vel.x, bLast.vel.y, bLast.vel.z);
    printf("Arena 0 car 0: pos=(%.1f, %.1f, %.1f) vel=(%.1f, %.1f, %.1f) boost=%.2f\n",
           c0.pos.x, c0.pos.y, c0.pos.z, c0.vel.x, c0.vel.y, c0.vel.z, c0.boost);
    printf("Arena %d car 1: pos=(%.1f, %.1f, %.1f) vel=(%.1f, %.1f, %.1f) boost=%.2f\n",
           kNumArenas - 1,
           cLast.pos.x, cLast.pos.y, cLast.pos.z,
           cLast.vel.x, cLast.vel.y, cLast.vel.z, cLast.boost);

    bool ok = true;
    if (info0.tickCount != static_cast<uint64_t>(kWarmupTicks + measuredTicks)) {
        printf("[FAIL] Arena 0 tick count mismatch\n");
        ok = false;
    }
    if (infoLast.tickCount != static_cast<uint64_t>(kWarmupTicks + measuredTicks)) {
        printf("[FAIL] Arena %d tick count mismatch\n", kNumArenas - 1);
        ok = false;
    }
    if (!(b0.pos.z > 0.f && bLast.pos.z > 0.f)) {
        printf("[FAIL] Ball went below ground\n");
        ok = false;
    }
    if (!(c0.pos.z > 0.f && cLast.pos.z > 0.f)) {
        printf("[FAIL] Car went below ground\n");
        ok = false;
    }

    batch.Destroy();

    printf("Status: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
