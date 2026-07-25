#include <RocketSimCuda.h>
#include <RocketSim.h>

#include <RLGymCPP/Gamestates/GameState.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <vector>

namespace cpu = RocketSim;
namespace gpu = rsc;
using namespace RLGC;

namespace {

static int g_passed = 0;
static int g_failed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_passed++; std::printf("  [PASS] %s\n", msg); } \
    else      { g_failed++; std::printf("  [FAIL] %s\n", msg); } \
} while(0)

bool Near(float a, float b, float eps = 1e-3f) {
    return std::fabs(a - b) <= eps;
}

float VecDiff(const Vec& a, const Vec& b) {
    return (a - b).Length();
}

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
    dst.ballHitInfo.isValid = src.ballHitValid;
    dst.ballHitInfo.tickCountWhenHit = src.ballHitTickCount;
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

std::filesystem::path GetCollisionMeshesPath() {
    std::filesystem::path root = std::filesystem::path(RSCUDA_WORKSPACE_ROOT);
    return root / "collision_meshes";
}

const Player* FindPlayerByCarId(const GameState& state, uint32_t carId) {
    for (const auto& player : state.players) {
        if (player.carId == carId)
            return &player;
    }
    return nullptr;
}

struct BridgeHarness {
    gpu::RocketSimCudaBatch batch;
    cpu::Arena* arena = nullptr;
    std::vector<cpu::Car*> cars;

    BridgeHarness(int numCars) {
        gpu::BatchConfig cfg;
        cfg.numArenas = 1;
        cfg.maxCarsPerArena = numCars;
        cfg.tickRate = 120.0f;
        batch.Init(cfg);

        arena = cpu::Arena::Create(cpu::GameMode::SOCCAR);

        for (int i = 0; i < numCars; ++i) {
            gpu::Team team = (i % 2 == 0) ? gpu::Team::BLUE : gpu::Team::ORANGE;
            batch.AddCar(0, team, gpu::CarPreset::OCTANE);
            cars.push_back(arena->AddCar(team == gpu::Team::BLUE ? cpu::Team::BLUE : cpu::Team::ORANGE, cpu::CAR_CONFIG_OCTANE));
        }

        batch.ResetArena(0);

        for (int i = 0; i < numCars; ++i) {
            gpu::CarState gpuState = batch.GetCarState(0, i);
            cars[i]->id = gpuState.id;
            cpu::CarState cpuState = cars[i]->GetState();
            CopyGpuToCpuCarState(cpuState, gpuState);
            cars[i]->SetState(cpuState);
        }
        arena->ball->SetState(ToCpuBallState(batch.GetBallState(0)));
    }

    ~BridgeHarness() {
        batch.Destroy();
        delete arena;
    }
};

void CompareGameStates(
    const GameState& cpuState,
    const GameState& gpuState,
    bool expectTouch = false,
    bool expectGoal = false,
    float ballPosTol = 1.0f,
    float ballVelTol = 25.0f,
    float playerPosTol = 1.0f,
    float playerVelTol = 2.0f,
    bool compareBallTouchedTick = true
) {
    CHECK(cpuState.players.size() == gpuState.players.size(), "Player count matches");
    CHECK(cpuState.goalScored == gpuState.goalScored, "goalScored matches");
    CHECK(cpuState.lastTickCount == gpuState.lastTickCount, "tickCount matches");
    CHECK(Near(cpuState.deltaTime, gpuState.deltaTime, 1e-5f), "deltaTime matches");
    CHECK(VecDiff(cpuState.ball.pos, gpuState.ball.pos) < ballPosTol, "Ball position matches");
    CHECK(VecDiff(cpuState.ball.vel, gpuState.ball.vel) < ballVelTol, "Ball velocity matches");

    CHECK(cpuState.boostPads.size() == gpuState.boostPads.size(), "Boost pad count matches");
    int padMismatch = 0;
    for (int i = 0; i < cpuState.boostPads.size(); ++i) {
        padMismatch += (cpuState.boostPads[i] != gpuState.boostPads[i]) ? 1 : 0;
        padMismatch += !Near(cpuState.boostPadTimers[i], gpuState.boostPadTimers[i], 0.05f) ? 1 : 0;
    }
    CHECK(padMismatch == 0, "Boost pad states/timers match");

    for (int i = 0; i < cpuState.players.size(); ++i) {
        const auto& c = cpuState.players[i];
        const Player* g = FindPlayerByCarId(gpuState, c.carId);
        CHECK(g != nullptr, "Player car id matches");
        if (!g)
            continue;

        CHECK(c.team == g->team, "Player team matches");
        CHECK(c.isOnGround == g->isOnGround, "Player on-ground matches");
        CHECK(c.isDemoed == g->isDemoed, "Player demo matches");
        CHECK(VecDiff(c.pos, g->pos) < playerPosTol, "Player position matches");
        CHECK(VecDiff(c.vel, g->vel) < playerVelTol, "Player velocity matches");
        CHECK(Near(c.boost, g->boost, 0.1f), "Player boost matches");
        CHECK(c.ballTouchedStep == g->ballTouchedStep, "Player ballTouchedStep matches");
        if (compareBallTouchedTick)
            CHECK(c.ballTouchedTick == g->ballTouchedTick, "Player ballTouchedTick matches");
    }

    if (expectTouch) {
        bool anyTouch = false;
        for (const auto& player : gpuState.players)
            anyTouch |= player.ballTouchedStep;
        CHECK(anyTouch, "CUDA bridge preserves a ball touch event");
    }

    if (expectGoal)
        CHECK(gpuState.goalScored, "CUDA bridge preserves a goal event");
}

void test_kickoff_idle_bridge() {
    std::printf("\n=== Test: GameState bridge kickoff idle ===\n");
    BridgeHarness h(2);

    std::vector<Action> actions(2);
    GameState cpuPrev(h.arena);
    GameState gpuPrev(h.batch, 0);

    h.arena->Step(120);
    h.batch.Step(120);

    GameState cpuCur;
    cpuCur.UpdateFromArena(h.arena, actions, &cpuPrev);
    GameState gpuCur;
    gpuCur.UpdateFromCudaBatch(h.batch, 0, actions, &gpuPrev);

    CompareGameStates(cpuCur, gpuCur);
}

void test_boost_pad_bridge() {
    std::printf("\n=== Test: GameState bridge boost pad ===\n");
    BridgeHarness h(1);

    gpu::CarState car = h.batch.GetCarState(0, 0);
    car.pos = {0.0f, -2920.0f, car.pos.z};
    car.rotMat = {{0.0f, 1.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}};
    car.vel = {0.0f, 650.0f, 0.0f};
    car.boost = 0.0f;
    h.batch.SetCarState(0, 0, car);

    cpu::CarState cpuCar = h.cars[0]->GetState();
    CopyGpuToCpuCarState(cpuCar, car);
    h.cars[0]->SetState(cpuCar);

    gpu::CarControls controls;
    controls.throttle = 1.0f;
    h.batch.SetCarControls(0, 0, controls);
    h.cars[0]->controls = ToCpuControls(controls);

    std::vector<Action> actions = { Action(ToCpuControls(controls)) };
    GameState cpuPrev;
    cpuPrev.UpdateFromArena(h.arena, actions, nullptr);
    GameState gpuPrev;
    gpuPrev.UpdateFromCudaBatch(h.batch, 0, actions, nullptr);

    h.arena->Step(45);
    h.batch.Step(45);

    GameState cpuCur;
    cpuCur.UpdateFromArena(h.arena, actions, &cpuPrev);
    GameState gpuCur;
    gpuCur.UpdateFromCudaBatch(h.batch, 0, actions, &gpuPrev);

    CompareGameStates(cpuCur, gpuCur);
}

void test_ball_touch_bridge() {
    std::printf("\n=== Test: GameState bridge ball touch ===\n");
    BridgeHarness h(1);

    gpu::CarState car = h.batch.GetCarState(0, 0);
    car.pos = {0.0f, -1400.0f, car.pos.z};
    car.rotMat = {{0.0f, 1.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}};
    car.vel = {0.0f, 0.0f, 0.0f};
    car.boost = 100.0f;
    h.batch.SetCarState(0, 0, car);

    cpu::CarState cpuCar = h.cars[0]->GetState();
    CopyGpuToCpuCarState(cpuCar, car);
    h.cars[0]->SetState(cpuCar);

    gpu::BallState ball;
    ball.pos = {0.0f, -850.0f, 93.15f};
    ball.rotMat = {{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}};
    ball.vel = {0.0f, 0.0f, 0.0f};
    ball.angVel = {0.0f, 0.0f, 0.0f};
    h.batch.SetBallState(0, ball);
    h.arena->ball->SetState(ToCpuBallState(ball));

    gpu::CarControls controls;
    controls.throttle = 1.0f;
    controls.boost = true;
    h.batch.SetCarControls(0, 0, controls);
    h.cars[0]->controls = ToCpuControls(controls);

    std::vector<Action> actions = { Action(ToCpuControls(controls)) };
    GameState cpuPrev;
    cpuPrev.UpdateFromArena(h.arena, actions, nullptr);
    GameState gpuPrev;
    gpuPrev.UpdateFromCudaBatch(h.batch, 0, actions, nullptr);

    h.arena->Step(30);
    h.batch.Step(30);

    gpu::CarState gpuTouchedCar = h.batch.GetCarState(0, 0);
    gpuTouchedCar.ballHitValid = true;
    gpuTouchedCar.ballHitTickCount = h.batch.GetArenaInfo(0).tickCount - 1;
    h.batch.SetCarState(0, 0, gpuTouchedCar);

    cpu::CarState cpuTouchedCar = h.cars[0]->GetState();
    cpuTouchedCar.ballHitInfo.isValid = true;
    cpuTouchedCar.ballHitInfo.tickCountWhenHit = h.arena->tickCount - 1;
    h.cars[0]->SetState(cpuTouchedCar);

    GameState cpuCur;
    cpuCur.UpdateFromArena(h.arena, actions, &cpuPrev);
    GameState gpuCur;
    gpuCur.UpdateFromCudaBatch(h.batch, 0, actions, &gpuPrev);

    CompareGameStates(cpuCur, gpuCur, true, false, 150.0f, 500.0f, 50.0f, 200.0f);
}

void test_goal_bridge() {
    std::printf("\n=== Test: GameState bridge goal ===\n");
    BridgeHarness h(1);

    gpu::BallState ball;
    ball.pos = {0.0f, 5000.0f, 100.0f};
    ball.rotMat = {{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}};
    ball.vel = {0.0f, 2000.0f, 0.0f};
    ball.angVel = {0.0f, 0.0f, 0.0f};
    h.batch.SetBallState(0, ball);
    h.arena->ball->SetState(ToCpuBallState(ball));

    std::vector<Action> actions(1);
    GameState cpuPrev;
    cpuPrev.UpdateFromArena(h.arena, actions, nullptr);
    GameState gpuPrev;
    gpuPrev.UpdateFromCudaBatch(h.batch, 0, actions, nullptr);

    h.arena->Step(20);
    h.batch.Step(20);

    GameState cpuCur;
    cpuCur.UpdateFromArena(h.arena, actions, &cpuPrev);
    GameState gpuCur;
    gpuCur.UpdateFromCudaBatch(h.batch, 0, actions, &gpuPrev);

    CompareGameStates(cpuCur, gpuCur, false, true);
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

    std::printf("========================================\n");
    std::printf("  RocketSimCuda GameState Bridge Tests\n");
    std::printf("========================================\n");

    test_kickoff_idle_bridge();
    test_boost_pad_bridge();
    test_ball_touch_bridge();
    test_goal_bridge();

    std::printf("\n========================================\n");
    std::printf("  Results: %d passed, %d failed\n", g_passed, g_failed);
    std::printf("========================================\n");

    return g_failed > 0 ? 1 : 0;
}
