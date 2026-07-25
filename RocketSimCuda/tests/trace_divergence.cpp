// Tick-by-tick CPU vs CUDA divergence tracer.
// Steps both sims through a scenario and prints full car state each tick so
// the first diverging tick (and field) can be identified precisely.
//
// Usage: RocketSimCudaTraceDivergence [scenario] [ticks]
//   scenarios: front_flip (default), ball_wall, side_flip, back_flip,
//              aerial_yaw, aerial_pitch_roll, diag_flip, stall

#include "cuda_host_shim.cuh"
#include "../src/GpuTypes.cuh"
#include "../src/CudaMath.cuh"
#include "../src/MeshCollision.cuh"

#include <RocketSimCuda.h>
#include <RocketSim.h>

#include <cstdint>
#include <fstream>

#include "../../GigaLearnCPP/RLGymCPP/RocketSim/libsrc/bullet3-3.24/BulletCollision/CollisionDispatch/btCollisionDispatcher.h"
#include "../../GigaLearnCPP/RLGymCPP/RocketSim/libsrc/bullet3-3.24/BulletCollision/NarrowPhaseCollision/btPersistentManifold.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace cpu = RocketSim;
namespace gpu = rsc;

static constexpr float PI_F = 3.14159265358979323846f;

static cpu::Vec ToCpuVec(const gpu::Vec3& v) { return {v.x, v.y, v.z}; }

static cpu::RotMat ToCpuRot(const gpu::RotMat& m) {
    return {ToCpuVec(m.forward), ToCpuVec(m.right), ToCpuVec(m.up)};
}

static cpu::CarControls ToCpuControls(const gpu::CarControls& c) {
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

static void CopyGpuToCpuCarState(cpu::CarState& dst, const gpu::CarState& src) {
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

static gpu::RotMat YawToRot(float yaw) {
    float cy = std::cos(yaw), sy = std::sin(yaw);
    return {{cy, sy, 0.f}, {-sy, cy, 0.f}, {0.f, 0.f, 1.f}};
}

static std::string g_collisionMeshesPath;
static std::vector<rsc::MeshTriangle> g_hostTris;

static void LoadHostTris() {
    namespace fs = std::filesystem;
    fs::path base = fs::path(g_collisionMeshesPath) / "soccar";
    int meshId = 0;
    for (const auto& entry : fs::directory_iterator(base)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".cmf") continue;
        std::ifstream in(entry.path(), std::ios::binary);
        int32_t numTris = 0, numVerts = 0;
        in.read(reinterpret_cast<char*>(&numTris), 4);
        in.read(reinterpret_cast<char*>(&numVerts), 4);
        std::vector<int32_t> idx(numTris * 3);
        std::vector<float> verts(numVerts * 3);
        in.read(reinterpret_cast<char*>(idx.data()), numTris * 3 * 4);
        in.read(reinterpret_cast<char*>(verts.data()), numVerts * 3 * 4);
        for (int t = 0; t < numTris; t++) {
            rsc::MeshTriangle tri;
            int i0 = idx[t*3], i1 = idx[t*3+1], i2 = idx[t*3+2];
            tri.v0 = {verts[i0*3], verts[i0*3+1], verts[i0*3+2]};
            tri.v1 = {verts[i1*3], verts[i1*3+1], verts[i1*3+2]};
            tri.v2 = {verts[i2*3], verts[i2*3+1], verts[i2*3+2]};
            tri.meshId = meshId;
            g_hostTris.push_back(tri);
        }
        meshId++;
    }
}

// Host replication of the GPU per-mesh manifold contact generation, for
// side-by-side comparison with the CPU bullet manifolds.
static void DumpGpuBallContacts(rsc::Vec3 ballPosUU, float ballRadiusUU) {
    using namespace rsc;
    float radiusBT = ballRadiusUU / 50.f;
    float breakingBT = 1.825f / 50.f;
    Vec3 centerBT = {ballPosUU.x / 50.f, ballPosUU.y / 50.f, ballPosUU.z / 50.f};

    struct Pt { Vec3 point, normal; float depth; bool valid; };
    Pt mani[8][4] = {};
    int maniMesh[8];
    int numMani = 0;

    for (const auto& tri : g_hostTris) {
        Vec3 point, normal;
        float depth;
        if (!mesh_sphere_triangle_collide(centerBT, radiusBT, breakingBT, tri, point, normal, depth))
            continue;

        int m = -1;
        for (int i = 0; i < numMani; i++) if (maniMesh[i] == tri.meshId) { m = i; break; }
        if (m < 0) {
            if (numMani >= 8) continue;
            m = numMani++;
            maniMesh[m] = tri.meshId;
            for (int i = 0; i < 4; i++) mani[m][i].valid = false;
        }
        Pt* cache = mani[m];

        float shortest = breakingBT * breakingBT;
        int nearest = -1, firstFree = -1;
        for (int i = 0; i < 4; i++) {
            if (!cache[i].valid) { if (firstFree < 0) firstFree = i; continue; }
            Vec3 d = cache[i].point - point;
            float dd = v3_dot(d, d);
            if (dd < shortest) { shortest = dd; nearest = i; }
        }
        if (nearest >= 0) { cache[nearest] = {point, normal, depth, true}; continue; }
        if (firstFree >= 0) { cache[firstFree] = {point, normal, depth, true}; continue; }

        int maxPenIdx = -1;
        float maxPen = depth;
        for (int i = 0; i < 4; i++) if (cache[i].depth < maxPen) { maxPenIdx = i; maxPen = cache[i].depth; }
        auto area4 = [](Vec3 p0, Vec3 p1, Vec3 p2, Vec3 p3) {
            Vec3 a0 = p0-p1, b0 = p2-p3, a1 = p0-p2, b1 = p1-p3, a2 = p0-p3, b2 = p1-p2;
            float l0 = v3_length_sq(v3_cross(a0,b0)), l1 = v3_length_sq(v3_cross(a1,b1)), l2 = v3_length_sq(v3_cross(a2,b2));
            return fmaxf(fmaxf(l0,l1),l2);
        };
        float res[4] = {};
        if (maxPenIdx != 0) res[0] = area4(point, cache[1].point, cache[2].point, cache[3].point);
        if (maxPenIdx != 1) res[1] = area4(point, cache[0].point, cache[2].point, cache[3].point);
        if (maxPenIdx != 2) res[2] = area4(point, cache[0].point, cache[1].point, cache[3].point);
        if (maxPenIdx != 3) res[3] = area4(point, cache[0].point, cache[1].point, cache[2].point);
        int big = 0; float bigArea = res[0];
        for (int i = 1; i < 4; i++) if (res[i] > bigArea) { bigArea = res[i]; big = i; }
        cache[big] = {point, normal, depth, true};
    }

    for (int m = 0; m < numMani; m++) {
        std::printf("  GPU manifold mesh%d:\n", maniMesh[m]);
        for (int i = 0; i < 4; i++) {
            if (!mani[m][i].valid) continue;
            std::printf(
                "    pt pos=(%8.4f,%8.4f,%8.4f) n=(%7.4f,%7.4f,%7.4f) dist=%9.5f\n",
                mani[m][i].point.x * 50.f, mani[m][i].point.y * 50.f, mani[m][i].point.z * 50.f,
                mani[m][i].normal.x, mani[m][i].normal.y, mani[m][i].normal.z,
                mani[m][i].depth * 50.f);
        }
    }
}

struct Tracer {
    gpu::RocketSimCudaBatch batch;
    cpu::Arena* arena = nullptr;
    cpu::Car* cpuCar = nullptr;
    cpu::Car* cpuCar2 = nullptr;

    explicit Tracer(int numCars = 1) {
        gpu::BatchConfig cfg;
        cfg.numArenas = 1;
        cfg.maxCarsPerArena = numCars;
        cfg.tickRate = 120.f;
        cfg.collisionMeshesPath = g_collisionMeshesPath.empty() ? nullptr : g_collisionMeshesPath.c_str();
        batch.Init(cfg);
        batch.AddCar(0, gpu::Team::BLUE, gpu::CarPreset::OCTANE);
        if (numCars > 1)
            batch.AddCar(0, gpu::Team::ORANGE, gpu::CarPreset::OCTANE);
        batch.ResetArena(0);

        arena = cpu::Arena::Create(cpu::GameMode::SOCCAR);
        cpuCar = arena->AddCar(cpu::Team::BLUE, cpu::CAR_CONFIG_OCTANE);
        if (numCars > 1)
            cpuCar2 = arena->AddCar(cpu::Team::ORANGE, cpu::CAR_CONFIG_OCTANE);

        SyncCpuFromGpu();
    }

    ~Tracer() {
        batch.Destroy();
        delete arena;
    }

    void SyncCpuFromGpu() {
        {
            gpu::CarState g = batch.GetCarState(0, 0);
            cpu::CarState c = cpuCar->GetState();
            CopyGpuToCpuCarState(c, g);
            cpuCar->SetState(c);
        }
        if (cpuCar2) {
            gpu::CarState g = batch.GetCarState(0, 1);
            cpu::CarState c = cpuCar2->GetState();
            CopyGpuToCpuCarState(c, g);
            cpuCar2->SetState(c);
        }
        cpu::BallState b;
        gpu::BallState gb = batch.GetBallState(0);
        b.pos = ToCpuVec(gb.pos);
        b.rotMat = ToCpuRot(gb.rotMat);
        b.vel = ToCpuVec(gb.vel);
        b.angVel = ToCpuVec(gb.angVel);
        arena->ball->SetState(b);
    }

    void SetCar(const gpu::CarState& g) {
        batch.SetCarState(0, 0, g);
        cpu::CarState c = cpuCar->GetState();
        CopyGpuToCpuCarState(c, g);
        cpuCar->SetState(c);
    }

    void SetBall(const gpu::BallState& gb) {
        batch.SetBallState(0, gb);
        cpu::BallState b;
        b.pos = ToCpuVec(gb.pos);
        b.rotMat = ToCpuRot(gb.rotMat);
        b.vel = ToCpuVec(gb.vel);
        b.angVel = ToCpuVec(gb.angVel);
        arena->ball->SetState(b);
    }

    void Step(const gpu::CarControls& controls) {
        batch.SetCarControls(0, 0, controls);
        cpuCar->controls = ToCpuControls(controls);
        if (cpuCar2) {
            gpu::CarControls idle;
            batch.SetCarControls(0, 1, idle);
            cpuCar2->controls = ToCpuControls(idle);
        }
        arena->Step(1);
        batch.Step(1);
    }

    void PrintTick(int tick) {
        cpu::CarState c = cpuCar->GetState();
        gpu::CarState g = batch.GetCarState(0, 0);
        cpu::BallState cb = arena->ball->GetState();
        gpu::BallState gb = batch.GetBallState(0);

        float posD = std::sqrt(
            (c.pos.x - g.pos.x) * (c.pos.x - g.pos.x) +
            (c.pos.y - g.pos.y) * (c.pos.y - g.pos.y) +
            (c.pos.z - g.pos.z) * (c.pos.z - g.pos.z));
        float velD = std::sqrt(
            (c.vel.x - g.vel.x) * (c.vel.x - g.vel.x) +
            (c.vel.y - g.vel.y) * (c.vel.y - g.vel.y) +
            (c.vel.z - g.vel.z) * (c.vel.z - g.vel.z));
        float avD = std::sqrt(
            (c.angVel.x - g.angVel.x) * (c.angVel.x - g.angVel.x) +
            (c.angVel.y - g.angVel.y) * (c.angVel.y - g.angVel.y) +
            (c.angVel.z - g.angVel.z) * (c.angVel.z - g.angVel.z));
        float ballD = std::sqrt(
            (cb.pos.x - gb.pos.x) * (cb.pos.x - gb.pos.x) +
            (cb.pos.y - gb.pos.y) * (cb.pos.y - gb.pos.y) +
            (cb.pos.z - gb.pos.z) * (cb.pos.z - gb.pos.z));
        float ballVD = std::sqrt(
            (cb.vel.x - gb.vel.x) * (cb.vel.x - gb.vel.x) +
            (cb.vel.y - gb.vel.y) * (cb.vel.y - gb.vel.y) +
            (cb.vel.z - gb.vel.z) * (cb.vel.z - gb.vel.z));

        std::printf(
            "t=%3d dPos=%8.4f dVel=%8.4f dAngV=%8.5f dBall=%8.4f dBallV=%8.4f\n",
            tick, posD, velD, avD, ballD, ballVD);
        std::printf(
            "  CPU pos=(%9.3f,%9.3f,%8.3f) vel=(%9.3f,%9.3f,%9.3f) av=(%7.4f,%7.4f,%7.4f) f=(%6.3f,%6.3f,%6.3f) g=%d j=%d(%0.4f) fl=%d(%0.4f) dj=%d hf=%d\n",
            c.pos.x, c.pos.y, c.pos.z, c.vel.x, c.vel.y, c.vel.z,
            c.angVel.x, c.angVel.y, c.angVel.z,
            c.rotMat.forward.x, c.rotMat.forward.y, c.rotMat.forward.z,
            (int)c.isOnGround, (int)c.isJumping, c.jumpTime,
            (int)c.isFlipping, c.flipTime, (int)c.hasDoubleJumped, (int)c.hasFlipped);
        std::printf(
            "  GPU pos=(%9.3f,%9.3f,%8.3f) vel=(%9.3f,%9.3f,%9.3f) av=(%7.4f,%7.4f,%7.4f) f=(%6.3f,%6.3f,%6.3f) g=%d j=%d(%0.4f) fl=%d(%0.4f) dj=%d hf=%d\n",
            g.pos.x, g.pos.y, g.pos.z, g.vel.x, g.vel.y, g.vel.z,
            g.angVel.x, g.angVel.y, g.angVel.z,
            g.rotMat.forward.x, g.rotMat.forward.y, g.rotMat.forward.z,
            (int)g.isOnGround, (int)g.isJumping, g.jumpTime,
            (int)g.isFlipping, g.flipTime, (int)g.hasDoubleJumped, (int)g.hasFlipped);

        if (ballD > 0.005f || ballVD > 0.005f) {
            std::printf(
                "  CPU ball pos=(%9.3f,%9.3f,%8.3f) vel=(%9.3f,%9.3f,%9.3f) av=(%7.4f,%7.4f,%7.4f)\n",
                cb.pos.x, cb.pos.y, cb.pos.z, cb.vel.x, cb.vel.y, cb.vel.z,
                cb.angVel.x, cb.angVel.y, cb.angVel.z);
            std::printf(
                "  GPU ball pos=(%9.3f,%9.3f,%8.3f) vel=(%9.3f,%9.3f,%9.3f) av=(%7.4f,%7.4f,%7.4f)\n",
                gb.pos.x, gb.pos.y, gb.pos.z, gb.vel.x, gb.vel.y, gb.vel.z,
                gb.angVel.x, gb.angVel.y, gb.angVel.z);
        }
    }
};

int main(int argc, char** argv) {
    std::string scenario = (argc > 1) ? argv[1] : "front_flip";
    int ticks = (argc > 2) ? std::atoi(argv[2]) : 60;

    std::filesystem::path meshes =
        std::filesystem::path(RSCUDA_WORKSPACE_ROOT) / "collision_meshes";
    cpu::Init(meshes, true);
    g_collisionMeshesPath = meshes.string();
    LoadHostTris();

    std::printf("scenario=%s ticks=%d\n", scenario.c_str(), ticks);

    Tracer tracer;

    if (scenario == "ball_wall") {
        gpu::BallState ball;
        ball.pos = {3500.f, 0.f, 400.f};
        ball.rotMat = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        ball.vel = {1800.f, 0.f, 0.f};
        ball.angVel = {0.f, 0.f, 0.f};
        tracer.SetBall(ball);
    } else if (scenario == "ball_corner") {
        gpu::BallState ball;
        ball.pos = {2800.f, -4000.f, 300.f};
        ball.rotMat = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        ball.vel = {1300.f, -1200.f, 0.f};
        ball.angVel = {0.f, 0.f, 0.f};
        tracer.SetBall(ball);
    } else if (scenario == "dribble") {
        gpu::CarState car = tracer.batch.GetCarState(0, 0);
        car.pos = {0.f, 0.f, 17.01f};
        car.rotMat = YawToRot(0.f);
        car.vel = {300.f, 0.f, 0.f};
        car.boost = 30.f;
        tracer.SetCar(car);
        gpu::BallState ball;
        ball.pos = {20.f, 0.f, 172.1f};
        ball.rotMat = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        ball.vel = {300.f, 0.f, 0.f};
        ball.angVel = {0.f, 0.f, 0.f};
        tracer.SetBall(ball);
    } else if (scenario == "aerial_yaw" || scenario == "aerial_pitch_roll") {
        gpu::CarState car = tracer.batch.GetCarState(0, 0);
        car.pos = {0.f, 0.f, 800.f};
        car.rotMat = YawToRot(PI_F * 0.25f);
        car.vel = {0.f, 0.f, 100.f};
        car.angVel = {0.f, 0.f, 0.f};
        car.isOnGround = false;
        car.boost = 100.f;
        tracer.SetCar(car);
    }

    for (int t = 0; t < ticks; t++) {
        gpu::CarControls controls;

        if (scenario == "front_flip") {
            if (t < 3) {
                controls.jump = true;
            } else if (t >= 8 && t < 11) {
                controls.jump = true;
                controls.pitch = -1.f;
                controls.throttle = 1.f;
            } else {
                controls.throttle = 1.f;
            }
        } else if (scenario == "side_flip") {
            if (t < 3) {
                controls.jump = true;
            } else if (t >= 8 && t < 11) {
                controls.jump = true;
                controls.yaw = 1.f;
            }
        } else if (scenario == "back_flip") {
            if (t < 3) {
                controls.jump = true;
            } else if (t >= 8 && t < 11) {
                controls.jump = true;
                controls.pitch = 1.f;
            }
        } else if (scenario == "diag_flip") {
            if (t < 3) {
                controls.jump = true;
            } else if (t >= 8 && t < 11) {
                controls.jump = true;
                controls.pitch = -0.7f;
                controls.yaw = 0.7f;
            }
        } else if (scenario == "stall") {
            if (t < 3) {
                controls.jump = true;
            } else if (t >= 8 && t < 11) {
                controls.jump = true;
                controls.yaw = 1.f;
                controls.roll = -1.f;
            }
        } else if (scenario == "aerial_yaw") {
            controls.yaw = 1.f;
            controls.boost = (t % 8) < 4;
        } else if (scenario == "aerial_pitch_roll") {
            controls.pitch = 0.6f;
            controls.roll = -0.8f;
        } else if (scenario == "powerslide") {
            controls.throttle = 1.f;
            if (t >= 40) {
                controls.steer = 1.f;
                controls.handbrake = true;
            }
        } else if (scenario == "steer_late") {
            controls.throttle = 1.f;
            if (t >= 40)
                controls.steer = 1.f;
        } else if (scenario == "handbrake_straight") {
            controls.throttle = 1.f;
            if (t >= 40)
                controls.handbrake = true;
        } else if (scenario == "dribble") {
            controls.throttle = 0.6f;
        }
        // ball_wall / ball_corner: idle car

        tracer.Step(controls);
        tracer.PrintTick(t);

        // Ball-world manifold dump for corner-bounce debugging.
        if (scenario == "ball_corner" && t >= 53 && t <= 56) {
            gpu::BallState gb = tracer.batch.GetBallState(0);
            DumpGpuBallContacts({gb.pos.x, gb.pos.y, gb.pos.z}, 91.25f);
            auto* dispatcher = tracer.arena->_bulletWorld.getDispatcher();
            int numManifolds = dispatcher->getNumManifolds();
            for (int m = 0; m < numManifolds; m++) {
                btPersistentManifold* manifold = dispatcher->getManifoldByIndexInternal(m);
                const btCollisionObject* b0 = manifold->getBody0();
                const btCollisionObject* b1 = manifold->getBody1();
                bool ballInvolved =
                    b0 == &tracer.arena->ball->_rigidBody || b1 == &tracer.arena->ball->_rigidBody;
                if (!ballInvolved || manifold->getNumContacts() == 0)
                    continue;
                std::printf("  CPU manifold %d (%d pts):\n", m, manifold->getNumContacts());
                for (int p = 0; p < manifold->getNumContacts(); p++) {
                    const btManifoldPoint& pt = manifold->getContactPoint(p);
                    std::printf(
                        "    pt pos=(%8.4f,%8.4f,%8.4f) n=(%7.4f,%7.4f,%7.4f) dist=%9.5f\n",
                        pt.getPositionWorldOnB().x() * 50.f, pt.getPositionWorldOnB().y() * 50.f,
                        pt.getPositionWorldOnB().z() * 50.f,
                        pt.m_normalWorldOnB.x(), pt.m_normalWorldOnB.y(), pt.m_normalWorldOnB.z(),
                        pt.getDistance() * 50.f);
                }
            }
        }

        // Per-wheel internals around the steering onset for steer scenarios.
        if ((scenario == "steer_late" || scenario == "powerslide") && t >= 39 && t <= 43) {
            rsc::GpuCarState g;
            tracer.batch.DebugCopyCarInternal(0, 0, &g, sizeof(g));
            for (int w = 0; w < 4; w++) {
                auto& cw = tracer.cpuCar->_bulletVehicle.m_wheelInfo[w];
                auto& gw = g.wheels[w];
                std::printf(
                    "  w%d CPU steer=%8.5f latF=%7.4f longF=%7.4f eng=%9.4f brk=%8.4f imp=(%9.4f,%9.4f,%9.4f) susF=%9.4f\n",
                    w, cw.m_steerAngle, cw.m_latFriction, cw.m_longFriction,
                    cw.m_engineForce, cw.m_brake,
                    cw.m_impulse.x(), cw.m_impulse.y(), cw.m_impulse.z(),
                    cw.m_wheelsSuspensionForce);
                std::printf(
                    "  w%d GPU steer=%8.5f latF=%7.4f longF=%7.4f eng=%9.4f brk=%8.4f imp=(%9.4f,%9.4f,%9.4f) susF=%9.4f\n",
                    w, gw.steerAngle, gw.latFriction, gw.longFriction,
                    gw.engineForce, gw.brake,
                    gw.frictionImpulse.x, gw.frictionImpulse.y, gw.frictionImpulse.z,
                    gw.suspensionForce);
            }
        }
    }

    return 0;
}
