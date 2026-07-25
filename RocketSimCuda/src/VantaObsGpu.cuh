#pragma once

// ============================================================================
// VantaObsGpu.cuh — GPU transcription of VantaObs v3.5
// (GigaLearnCPP/RLGymCPP/src/RLGymCPP/OBSBuilders/VantaObs.h)
//
// Layout (366 floats): ego(81) + ball(29, incl. 3 ball-prediction slices at
// 1/2/3 s) + goal(16) + teamCtx(6) + 39*2 teammates + 39*3 opponents +
// mask(5) + boostPads(34).
//
// Stateful pieces (per the CPU builder, one instance per arena):
//  - per-arena boost pad timers/availability (RLGymCPP pad order)
//  - per-player "coyote" timers driven by the previous action
//  - ball prediction slices, recomputed from a ball-only rollout using the
//    exact same bullet-transcribed ball physics as the main step
// ============================================================================

#include "CudaMath.cuh"

namespace rsc {

namespace VantaConst {
    constexpr float POS_STD = 2300.f;
    constexpr float VEL_STD = 2300.f;
    constexpr float BALL_VEL_STD = 6000.f;
    constexpr float ANG_STD = 5.5f;
    constexpr float DEMO_TIMER_STD = 3.f;
    constexpr float GOAL_DIST_COEF = 1.f / 10240.f;

    constexpr int EGO_SIZE = 81;
    constexpr int BALL_SIZE = 29;
    constexpr int GOAL_SIZE = 16;
    constexpr int TEAM_CTX_SIZE = 6;
    constexpr int CAR_SIZE = 39;
    constexpr int MAX_TEAMMATES = 2;
    constexpr int MAX_OPPONENTS = 3;
    constexpr int MASK_SIZE = MAX_TEAMMATES + MAX_OPPONENTS;
    constexpr int BOOST_SIZE = 34;

    constexpr int TOTAL_OBS_SIZE =
        EGO_SIZE + BALL_SIZE + GOAL_SIZE + TEAM_CTX_SIZE
        + (CAR_SIZE * MAX_TEAMMATES)
        + (CAR_SIZE * MAX_OPPONENTS)
        + MASK_SIZE + BOOST_SIZE;  // = 366

    constexpr float DOUBLEJUMP_MAX_DELAY = 1.25f;
}

// ---- persistent state --------------------------------------------------------

struct GpuVantaTimers {
    float boostTime;
    float boostHoldTime;
    float jumpTime;
    float flipTime;
    float airTime;
    float handbrakeTime;
    float demoTimer;

    uint8_t isJumping;
    uint8_t hasJumped;
    uint8_t hasDoubleJumped;
    uint8_t hasFlipped;
    uint8_t onGround;
    uint8_t wasDemoed;
    uint8_t timersInit;

    Vec3 flipDir;
    GpuCarControls prevPrevAction;
    uint64_t lastUpdatedTick;
};

struct GpuVantaArenaState {
    float boostTimers[VantaConst::BOOST_SIZE];     // RLGymCPP pad order
    uint8_t boostAvail[VantaConst::BOOST_SIZE];
    uint8_t initialized;
    uint64_t lastUpdatedTick;

    // Ball prediction slices (world frame) at 1.0 / 2.0 / 3.0 seconds
    Vec3 predPos[3];
    Vec3 predVel[3];
    uint8_t hasPred;
};

// ---- helpers -------------------------------------------------------------------

struct VantaPhys {
    Vec3 pos, vel, angVel;
    RotMat rotMat;
};

__device__ __forceinline__ VantaPhys vanta_phys_of_car(const GpuCarState& c) {
    return {c.pos, c.vel, c.angVel, c.rotMat};
}

__device__ __forceinline__ VantaPhys vanta_phys_of_ball_state(Vec3 pos, Vec3 vel, Vec3 angVel) {
    VantaPhys p;
    p.pos = pos; p.vel = vel; p.angVel = angVel;
    p.rotMat = rotmat_identity();
    return p;
}

// InvertPhys: pos/vel/angVel *= (-1,-1,1); each rotMat basis vector *= (-1,-1,1)
__device__ inline VantaPhys vanta_invert(const VantaPhys& p, bool inv) {
    if (!inv) return p;
    VantaPhys out = p;
    Vec3 INV = {-1.f, -1.f, 1.f};
    out.pos = out.pos * INV;
    out.vel = out.vel * INV;
    out.angVel = out.angVel * INV;
    out.rotMat.forward = out.rotMat.forward * INV;
    out.rotMat.right = out.rotMat.right * INV;
    out.rotMat.up = out.rotMat.up * INV;
    return out;
}

// MirrorPhysX: pos.x/vel.x negated; angVel.y/z negated; basis x components negated
__device__ inline VantaPhys vanta_mirror_x(const VantaPhys& p) {
    VantaPhys out = p;
    out.pos.x = -out.pos.x;
    out.vel.x = -out.vel.x;
    out.angVel.y = -out.angVel.y;
    out.angVel.z = -out.angVel.z;
    out.rotMat.forward.x = -out.rotMat.forward.x;
    out.rotMat.right.x = -out.rotMat.right.x;
    out.rotMat.up.x = -out.rotMat.up.x;
    return out;
}

__device__ __forceinline__ VantaPhys vanta_canonical(const VantaPhys& world, bool teamInv, bool mirrorX) {
    VantaPhys p = vanta_invert(world, teamInv);
    if (mirrorX) p = vanta_mirror_x(p);
    return p;
}

__device__ __forceinline__ Vec3 vanta_canonical_normal(Vec3 n, bool teamInv, bool mirrorX) {
    if (teamInv) { n.x = -n.x; n.y = -n.y; }
    if (mirrorX) { n.x = -n.x; }
    return n;
}

// rotMat.Dot(v): world -> body frame (transpose multiply)
__device__ __forceinline__ Vec3 vanta_rot_dot(const RotMat& R, Vec3 v) {
    return rotmat_dot_vec(R, v);
}

__device__ __forceinline__ Vec3 vanta_safe_norm(Vec3 v) {
    float l = v3_length(v);
    return (l > 1e-6f) ? v / l : v3_zero();
}

__device__ __forceinline__ bool vanta_has_flip_or_jump(const GpuCarState& c) {
    return c.isOnGround ||
           (!c.hasFlipped && !c.hasDoubleJumped &&
            c.airTimeSinceJump < VantaConst::DOUBLEJUMP_MAX_DELAY);
}

__device__ __forceinline__ bool vanta_has_flip_reset(const GpuCarState& c) {
    return !c.isOnGround && vanta_has_flip_or_jump(c) && !c.hasJumped;
}

__device__ __forceinline__ bool vanta_got_flip_reset(const GpuCarState& c) {
    return !c.isOnGround && !c.hasJumped;
}

// Small pad influence/density pooling. Pad locations in RLGymCPP order.
__device__ inline void vanta_small_pad_stats(
    const Vec3* padLocations, const GpuVantaArenaState& va,
    Vec3 carPos, float& outMaxInf, float& outDensity
) {
    float maxInf = 0.f;
    float density = 0.f;
    for (int i = 0; i < VantaConst::BOOST_SIZE; i++) {
        if (padLocations[i].z < 72.f) {  // small pad
            if (va.boostAvail[i]) {
                float dist = v3_length(padLocations[i] - carPos);
                float inf = 1.f / (1.f + dist / 500.f);
                maxInf = fmaxf(maxInf, inf);
                if (dist < 2000.f)
                    density += (1.f - dist / 2000.f);
            }
        }
    }
    outMaxInf = maxInf;
    outDensity = density / 4.f;
}

// ---- per-arena update (UpdateBoostPads + UpdateDemoTimers) -------------------
// One thread per arena, before the per-player obs kernel.
__device__ inline void vanta_update_arena(
    GpuVantaArenaState& va,
    GpuVantaTimers* timersRow,            // [maxCarsPerArena]
    const GpuCarState* cars, int numCars,
    const GpuBoostPadState* pads,         // GPU pad order
    const int* padMap,                    // RLGym order -> GPU pad index
    const Vec3* padLocations,             // RLGym order
    uint64_t tickCount, float timeInterval
) {
    if (!va.initialized) {
        for (int i = 0; i < VantaConst::BOOST_SIZE; i++) {
            va.boostAvail[i] = 1;
            va.boostTimers[i] = 0.f;
        }
        va.initialized = 1;
        va.lastUpdatedTick = ~0ULL;
    }

    if (va.lastUpdatedTick == tickCount)
        return;
    va.lastUpdatedTick = tickCount;

    // UpdateBoostPads
    for (int i = 0; i < VantaConst::BOOST_SIZE; i++) {
        uint8_t isActive = pads[padMap[i]].isActive ? 1 : 0;
        if (isActive != va.boostAvail[i]) {
            va.boostAvail[i] = isActive;
            va.boostTimers[i] = isActive ? 0.f
                : ((padLocations[i].z > 72.f) ? 10.f : 4.f);
        } else if (!va.boostAvail[i]) {
            va.boostTimers[i] = fmaxf(0.f, va.boostTimers[i] - timeInterval);
        }
    }

    // UpdateDemoTimers
    for (int c = 0; c < numCars; c++) {
        GpuVantaTimers& pt = timersRow[c];
        if (!pt.timersInit) {
            // GetOrCreateTimers default state
            pt = {};
            pt.onGround = cars[c].isOnGround ? 1 : 0;
            pt.lastUpdatedTick = ~0ULL;
            pt.timersInit = 1;
        }
        if (cars[c].isDemoed) {
            if (!pt.wasDemoed) {
                pt.demoTimer = 3.f;
                pt.wasDemoed = 1;
            }
            pt.demoTimer = fmaxf(0.f, pt.demoTimer - timeInterval);
        } else {
            pt.demoTimer = 0.f;
            pt.wasDemoed = 0;
        }
    }
}

// ---- per-player coyote timers (UpdateAdditionalTimers) ------------------------
__device__ inline void vanta_update_player_timers(
    GpuVantaTimers& pt, const GpuCarState& player,
    uint64_t currentTick, float timeInterval, float dodgeDeadzone
) {
    if (!pt.timersInit) {
        pt = {};
        pt.onGround = player.isOnGround ? 1 : 0;
        pt.lastUpdatedTick = ~0ULL;
        pt.timersInit = 1;
    }

    if (pt.lastUpdatedTick == currentTick)
        return;
    pt.lastUpdatedTick = currentTick;

    const GpuCarControls& prev = player.lastControls;

    bool prevBoost = prev.boost;
    if (prevBoost)
        pt.boostTime = fminf(pt.boostTime + timeInterval * 120.f, 12.f);
    else if (pt.boostTime >= 12.f)
        pt.boostTime = 0.f;

    if (prevBoost)
        pt.boostHoldTime = fminf(pt.boostHoldTime + timeInterval * 120.f, 12.f);
    else
        pt.boostHoldTime = fmaxf(pt.boostHoldTime - 2.f * timeInterval * 120.f, 0.f);

    if (pt.onGround && !pt.isJumping)
        pt.hasJumped = 0;

    const bool jumpHeld = prev.jump;
    const bool jumpPrevRel = !pt.prevPrevAction.jump;

    if (pt.isJumping)
        pt.isJumping = (pt.jumpTime < 3.f || (jumpHeld && pt.jumpTime < 24.f)) ? 1 : 0;
    else if (jumpHeld && jumpPrevRel && pt.onGround) {
        pt.isJumping = 1;
        pt.jumpTime = 0.f;
    }

    if (pt.isJumping) {
        pt.hasJumped = 1;
        pt.jumpTime = fminf(pt.jumpTime + timeInterval * 120.f, 24.f);
    } else {
        pt.jumpTime = 0.f;
    }

    if (player.isOnGround) {
        pt.isJumping = 0;
        pt.hasJumped = 0;
        pt.hasDoubleJumped = 0;
        pt.hasFlipped = 0;
        pt.airTime = 0.f;
        pt.flipTime = 0.f;
        pt.flipDir = v3_zero();
        pt.onGround = 1;
    } else {
        pt.airTime = (pt.hasJumped && !pt.isJumping)
            ? fminf(pt.airTime + timeInterval * 120.f, 150.f) : 0.f;

        if (jumpHeld && jumpPrevRel && (!pt.hasJumped || pt.airTime < 150.f)) {
            if (!pt.hasDoubleJumped && !pt.hasFlipped) {
                float absMax = fmaxf(fmaxf(fabsf(prev.pitch), fabsf(prev.yaw)), fabsf(prev.roll));
                if (absMax >= dodgeDeadzone) {
                    pt.flipTime = 0.f;
                    pt.hasFlipped = 1;
                    float fx = prev.yaw;
                    if (fabsf(fx) < 1e-6f && fabsf(prev.roll) >= dodgeDeadzone)
                        fx = prev.roll;
                    float fy = -prev.pitch;
                    float n = sqrtf(fx * fx + fy * fy);
                    pt.flipDir = (n > 1e-6f) ? v3(fx / n, fy / n, 0.f) : v3_zero();
                } else {
                    pt.hasDoubleJumped = 1;
                }
            }
        }
        pt.onGround = 0;
    }

    if (pt.hasFlipped)
        pt.flipTime = fminf(pt.flipTime + timeInterval * 120.f, 78.f);

    if (prev.handbrake)
        pt.handbrakeTime = fminf(pt.handbrakeTime + 5.f * timeInterval, 1.f);
    else
        pt.handbrakeTime = fmaxf(pt.handbrakeTime - 2.f * timeInterval, 0.f);

    pt.prevPrevAction = prev;
}

// ---- obs row builder ------------------------------------------------------------
// Writes VantaConst::TOTAL_OBS_SIZE floats into `obs`.
__device__ inline void vanta_build_obs(
    float* obs,
    const GpuCarState* cars, int numCars, int carIdx,
    const GpuBallState& ball,
    const GpuVantaArenaState& va,
    GpuVantaTimers* timersRow,
    const int* padMap,            // unused here (timers already in RLGym order)
    const Vec3* padLocations,     // RLGym order
    const int* mirrorPadMap,      // RLGym order X-mirror permutation
    uint32_t lastTouchCarID,
    uint64_t tickCount,
    float timeInterval, float dodgeDeadzone,
    bool onlyClosestOpp
) {
    using namespace VantaConst;

    const GpuCarState& player = cars[carIdx];

    vanta_update_player_timers(timersRow[carIdx], player, tickCount, timeInterval, dodgeDeadzone);
    const GpuVantaTimers& pt = timersRow[carIdx];

    const bool inv = (player.team == 1);  // ORANGE
    const bool mirrorX = (player.pos.x < 0.f);

    VantaPhys self = vanta_canonical(vanta_phys_of_car(player), inv, mirrorX);
    VantaPhys ballC = vanta_canonical(
        vanta_phys_of_ball_state(ball.pos, ball.vel, ball.angVel), inv, mirrorX);

    int o = 0;

    // ================= A. Ego (81) =================
    {
        Vec3 posDiff = ballC.pos - self.pos;
        Vec3 velDiff = ballC.vel - self.vel;
        Vec3 ballRelPos = vanta_rot_dot(self.rotMat, posDiff);
        Vec3 ballRelVel = vanta_rot_dot(self.rotMat, velDiff);
        Vec3 angVelBody = vanta_rot_dot(self.rotMat, self.angVel);

        obs[o++] = self.pos.x / POS_STD;  obs[o++] = self.pos.y / POS_STD;  obs[o++] = self.pos.z / POS_STD;
        obs[o++] = self.vel.x / VEL_STD;  obs[o++] = self.vel.y / VEL_STD;  obs[o++] = self.vel.z / VEL_STD;
        obs[o++] = self.angVel.x / ANG_STD; obs[o++] = self.angVel.y / ANG_STD; obs[o++] = self.angVel.z / ANG_STD;

        obs[o++] = ballRelPos.x / POS_STD; obs[o++] = ballRelPos.y / POS_STD; obs[o++] = ballRelPos.z / POS_STD;
        obs[o++] = ballRelVel.x / BALL_VEL_STD; obs[o++] = ballRelVel.y / BALL_VEL_STD; obs[o++] = ballRelVel.z / BALL_VEL_STD;

        obs[o++] = self.rotMat.forward.x; obs[o++] = self.rotMat.forward.y; obs[o++] = self.rotMat.forward.z;
        obs[o++] = self.rotMat.up.x;      obs[o++] = self.rotMat.up.y;      obs[o++] = self.rotMat.up.z;

        obs[o++] = angVelBody.x / ANG_STD;
        obs[o++] = angVelBody.y / ANG_STD;
        obs[o++] = angVelBody.z / ANG_STD;

        obs[o++] = v3_length(self.vel) / VEL_STD;

        obs[o++] = player.boost / 100.f;
        obs[o++] = player.isOnGround ? 1.f : 0.f;
        obs[o++] = vanta_has_flip_or_jump(player) ? 1.f : 0.f;
        obs[o++] = player.isDemoed ? 1.f : 0.f;
        obs[o++] = player.hasJumped ? 1.f : 0.f;
        obs[o++] = player.isSupersonic ? 1.f : 0.f;
        {
            float w = 0.f;
            for (int i = 0; i < 4; i++)
                w += player.wheels[i].isInContact ? 1.f : 0.f;
            obs[o++] = w / 4.f;
        }
        obs[o++] = vanta_has_flip_reset(player) ? 1.f : 0.f;
        obs[o++] = vanta_got_flip_reset(player) ? 1.f : 0.f;
        obs[o++] = player.handbrakeVal;

        obs[o++] = mirrorX ? 1.f : 0.f;
        obs[o++] = inv ? 1.f : 0.f;

        // Threat frame
        {
            Vec3 ownGoal = v3(0.f, -5120.f, 0.f);
            Vec3 toGoal = ownGoal - ballC.pos;
            float len = v3_length(toGoal);
            Vec3 dangerAxis = (len > 1.f) ? toGoal / len : v3(0.f, -1.f, 0.f);
            Vec3 lateralAxis = v3(-dangerAxis.y, dangerAxis.x, 0.f);
            float latLen = v3_length(lateralAxis);
            if (latLen > 1e-6f) lateralAxis = lateralAxis / latLen;

            obs[o++] = v3_dot(ballC.vel, dangerAxis) / BALL_VEL_STD;
            obs[o++] = v3_dot(self.vel, dangerAxis) / VEL_STD;
            obs[o++] = v3_dot(self.vel, lateralAxis) / VEL_STD;
            Vec3 carToBall = ballC.pos - self.pos;
            obs[o++] = v3_dot(carToBall, dangerAxis) / POS_STD;
            obs[o++] = v3_dot(carToBall, lateralAxis) / POS_STD;
            obs[o++] = v3_length(ownGoal - ballC.pos) / (2.f * POS_STD);
        }

        // Alignment scalars
        {
            Vec3 attackGoal = v3(0.f, 5120.f, 0.f);
            Vec3 fwd = self.rotMat.forward;
            Vec3 velNorm = vanta_safe_norm(self.vel);
            Vec3 toBallNorm = vanta_safe_norm(ballC.pos - self.pos);
            Vec3 toGoalNorm = vanta_safe_norm(attackGoal - self.pos);

            obs[o++] = v3_dot(fwd, toBallNorm);
            obs[o++] = v3_dot(fwd, toGoalNorm);
            obs[o++] = v3_dot(velNorm, toBallNorm);
            obs[o++] = v3_dot(self.rotMat.up, v3(0.f, 0.f, 1.f));
        }

        // Previous action (mirrored steer/yaw/roll on mirrored X)
        {
            const GpuCarControls& pa = player.lastControls;
            obs[o++] = pa.throttle;
            obs[o++] = mirrorX ? -pa.steer : pa.steer;
            obs[o++] = pa.pitch;
            obs[o++] = mirrorX ? -pa.yaw : pa.yaw;
            obs[o++] = mirrorX ? -pa.roll : pa.roll;
            obs[o++] = pa.jump ? 1.f : 0.f;
            obs[o++] = pa.boost ? 1.f : 0.f;
            obs[o++] = pa.handbrake ? 1.f : 0.f;
        }

        // Wall/ceiling contact flag
        obs[o++] = (player.worldContactHasContact && player.worldContactNormal.z < 0.7f) ? 1.f : 0.f;

        // Coyote demo timer
        obs[o++] = pt.demoTimer / DEMO_TIMER_STD;

        // Coyote timers
        obs[o++] = pt.boostTime / 12.f;
        obs[o++] = pt.jumpTime / 24.f;
        obs[o++] = pt.airTime / 150.f;
        obs[o++] = pt.flipTime / 78.f;
        obs[o++] = pt.handbrakeTime;
        obs[o++] = mirrorX ? -pt.flipDir.x : pt.flipDir.x;
        obs[o++] = pt.flipDir.y;
        obs[o++] = pt.boostHoldTime / 12.f;

        obs[o++] = pt.isJumping ? 1.f : 0.f;
        obs[o++] = pt.hasFlipped ? 1.f : 0.f;
        obs[o++] = pt.hasDoubleJumped ? 1.f : 0.f;

        // Ego enrichments
        float boostFrac = player.boost / 100.f;
        float distToBall = v3_length(posDiff);
        float closing = (distToBall > 1e-3f)
            ? (-v3_dot(velDiff, posDiff) / distToBall) / VEL_STD
            : 0.f;
        obs[o++] = sqrtf(fmaxf(0.f, boostFrac));
        obs[o++] = (ballC.pos.z > self.pos.z + 200.f) ? 1.f : 0.f;
        obs[o++] = closing;
        obs[o++] = distToBall / POS_STD;

        // Full surface contact
        bool hasContact = player.worldContactHasContact;
        Vec3 contactNormal = hasContact
            ? vanta_canonical_normal(player.worldContactNormal, inv, mirrorX)
            : v3_zero();
        obs[o++] = hasContact ? 1.f : 0.f;
        obs[o++] = contactNormal.x;
        obs[o++] = contactNormal.y;
        obs[o++] = contactNormal.z;

        obs[o++] = player.isFlipping ? 1.f : 0.f;

        obs[o++] = 0.f;  // headroom slot

        // Kickoff Left-Goes disambiguation (team-inverted, NOT mirrored)
        {
            VantaPhys teamFrame = vanta_invert(vanta_phys_of_car(player), inv);
            obs[o++] = (teamFrame.pos.x < 0.f) ? 1.f : 0.f;
        }

        // Small pad density pooling (world pos)
        {
            float maxInf, density;
            vanta_small_pad_stats(padLocations, va, player.pos, maxInf, density);
            obs[o++] = maxInf;
            obs[o++] = density;
        }
    }

    // ================= B. Ball (29) =================
    {
        obs[o++] = ballC.pos.x / POS_STD;
        obs[o++] = ballC.pos.y / POS_STD;
        obs[o++] = ballC.pos.z / POS_STD;
        obs[o++] = ballC.vel.x / BALL_VEL_STD;
        obs[o++] = ballC.vel.y / BALL_VEL_STD;
        obs[o++] = ballC.vel.z / BALL_VEL_STD;
        obs[o++] = ballC.angVel.x / ANG_STD;
        obs[o++] = ballC.angVel.y / ANG_STD;
        obs[o++] = ballC.angVel.z / ANG_STD;
        obs[o++] = v3_length(ballC.vel) / BALL_VEL_STD;
        obs[o++] = (ballC.pos.z <= 100.f) ? 1.f : 0.f;

        if (va.hasPred) {
            for (int i = 0; i < 3; i++) {
                VantaPhys canSlice = vanta_canonical(
                    vanta_phys_of_ball_state(va.predPos[i], va.predVel[i], v3_zero()),
                    inv, mirrorX);
                obs[o++] = canSlice.pos.x / POS_STD;
                obs[o++] = canSlice.pos.y / POS_STD;
                obs[o++] = canSlice.pos.z / POS_STD;
                obs[o++] = canSlice.vel.x / BALL_VEL_STD;
                obs[o++] = canSlice.vel.y / BALL_VEL_STD;
                obs[o++] = canSlice.vel.z / BALL_VEL_STD;
            }
        } else {
            for (int i = 0; i < 18; i++)
                obs[o++] = 0.f;
        }
    }

    // ================= C. Goal Context (16) =================
    {
        Vec3 OPP_GOAL = v3(0.f, 5120.f, 0.f);
        Vec3 OWN_GOAL = v3(0.f, -5120.f, 0.f);

        Vec3 selfToOpp = OPP_GOAL - self.pos;
        Vec3 localSelfToOpp = vanta_rot_dot(self.rotMat, selfToOpp);
        obs[o++] = localSelfToOpp.x * GOAL_DIST_COEF;
        obs[o++] = localSelfToOpp.y * GOAL_DIST_COEF;
        obs[o++] = localSelfToOpp.z * GOAL_DIST_COEF;
        obs[o++] = v3_length(selfToOpp) * GOAL_DIST_COEF;

        Vec3 selfToOwn = OWN_GOAL - self.pos;
        Vec3 localSelfToOwn = vanta_rot_dot(self.rotMat, selfToOwn);
        obs[o++] = localSelfToOwn.x * GOAL_DIST_COEF;
        obs[o++] = localSelfToOwn.y * GOAL_DIST_COEF;
        obs[o++] = localSelfToOwn.z * GOAL_DIST_COEF;
        obs[o++] = v3_length(selfToOwn) * GOAL_DIST_COEF;

        Vec3 ballToOpp = OPP_GOAL - ballC.pos;
        obs[o++] = ballToOpp.x * GOAL_DIST_COEF;
        obs[o++] = ballToOpp.y * GOAL_DIST_COEF;
        obs[o++] = ballToOpp.z * GOAL_DIST_COEF;
        obs[o++] = v3_length(ballToOpp) * GOAL_DIST_COEF;

        Vec3 ballToOwn = OWN_GOAL - ballC.pos;
        obs[o++] = ballToOwn.x * GOAL_DIST_COEF;
        obs[o++] = ballToOwn.y * GOAL_DIST_COEF;
        obs[o++] = ballToOwn.z * GOAL_DIST_COEF;
        obs[o++] = v3_length(ballToOwn) * GOAL_DIST_COEF;
    }

    // ================= D. Team Context (6) =================
    {
        int lastTouchTeam = -1;
        if (lastTouchCarID != 0) {
            for (int i = 0; i < numCars; i++) {
                if (cars[i].id == lastTouchCarID) {
                    if (cars[i].id == player.id)          lastTouchTeam = 0;
                    else if (cars[i].team == player.team) lastTouchTeam = 1;
                    else                                  lastTouchTeam = 2;
                    break;
                }
            }
        }
        obs[o++] = (lastTouchTeam == 0) ? 1.f : 0.f;
        obs[o++] = (lastTouchTeam == 1) ? 1.f : 0.f;
        obs[o++] = (lastTouchTeam == 2) ? 1.f : 0.f;

        int blueCount = 0, orangeCount = 0;
        for (int i = 0; i < numCars; i++)
            cars[i].team == 0 ? blueCount++ : orangeCount++;
        int myCount = inv ? orangeCount : blueCount;
        int theirCount = inv ? blueCount : orangeCount;
        obs[o++] = myCount / 3.f;
        obs[o++] = theirCount / 3.f;

        obs[o++] = (ballC.pos.y < 0.f) ? 1.f : 0.f;
    }

    // ================= E/F. Cars sorted by canonical dist-to-ball ============
    {
        int mateIdx[VantaConst::MAX_TEAMMATES + 4];
        float mateDist[VantaConst::MAX_TEAMMATES + 4];
        int numMates = 0;
        int oppIdx[VantaConst::MAX_OPPONENTS + 4];
        float oppDist[VantaConst::MAX_OPPONENTS + 4];
        int numOpps = 0;

        for (int i = 0; i < numCars; i++) {
            if (cars[i].id == player.id) continue;
            VantaPhys pCan = vanta_canonical(vanta_phys_of_car(cars[i]), inv, mirrorX);
            float dist = cars[i].isDemoed
                ? 3.402823466e+38f
                : v3_length(pCan.pos - ballC.pos);
            if (cars[i].team == player.team) {
                mateIdx[numMates] = i; mateDist[numMates] = dist; numMates++;
            } else {
                oppIdx[numOpps] = i; oppDist[numOpps] = dist; numOpps++;
            }
        }

        // insertion sort by distance (stable like std::sort for small N)
        for (int i = 1; i < numMates; i++) {
            int idx = mateIdx[i]; float d = mateDist[i]; int j = i - 1;
            while (j >= 0 && mateDist[j] > d) {
                mateIdx[j + 1] = mateIdx[j]; mateDist[j + 1] = mateDist[j]; j--;
            }
            mateIdx[j + 1] = idx; mateDist[j + 1] = d;
        }
        for (int i = 1; i < numOpps; i++) {
            int idx = oppIdx[i]; float d = oppDist[i]; int j = i - 1;
            while (j >= 0 && oppDist[j] > d) {
                oppIdx[j + 1] = oppIdx[j]; oppDist[j + 1] = oppDist[j]; j--;
            }
            oppIdx[j + 1] = idx; oppDist[j + 1] = d;
        }

        auto writeCarPacket = [&](const GpuCarState& other) {
            VantaPhys oC = vanta_canonical(vanta_phys_of_car(other), inv, mirrorX);

            Vec3 worldRelPos = oC.pos - self.pos;
            Vec3 worldRelVel = oC.vel - self.vel;
            Vec3 bodyRelPos = vanta_rot_dot(self.rotMat, worldRelPos);
            Vec3 bodyRelVel = vanta_rot_dot(self.rotMat, worldRelVel);
            Vec3 bodyOtherAngVel = vanta_rot_dot(self.rotMat, oC.angVel);
            float distToEgo = v3_length(worldRelPos);
            float distToBall2 = v3_length(oC.pos - ballC.pos);

            obs[o++] = oC.pos.x / POS_STD;  obs[o++] = oC.pos.y / POS_STD;  obs[o++] = oC.pos.z / POS_STD;
            obs[o++] = oC.vel.x / VEL_STD;  obs[o++] = oC.vel.y / VEL_STD;  obs[o++] = oC.vel.z / VEL_STD;
            obs[o++] = oC.angVel.x / ANG_STD; obs[o++] = oC.angVel.y / ANG_STD; obs[o++] = oC.angVel.z / ANG_STD;

            obs[o++] = bodyRelPos.x / POS_STD; obs[o++] = bodyRelPos.y / POS_STD; obs[o++] = bodyRelPos.z / POS_STD;
            obs[o++] = bodyRelVel.x / VEL_STD; obs[o++] = bodyRelVel.y / VEL_STD; obs[o++] = bodyRelVel.z / VEL_STD;

            obs[o++] = bodyOtherAngVel.x / ANG_STD;
            obs[o++] = bodyOtherAngVel.y / ANG_STD;
            obs[o++] = bodyOtherAngVel.z / ANG_STD;

            obs[o++] = oC.rotMat.forward.x; obs[o++] = oC.rotMat.forward.y; obs[o++] = oC.rotMat.forward.z;
            obs[o++] = oC.rotMat.up.x;      obs[o++] = oC.rotMat.up.y;      obs[o++] = oC.rotMat.up.z;

            obs[o++] = other.boost / 100.f;
            obs[o++] = other.isOnGround ? 1.f : 0.f;
            obs[o++] = vanta_has_flip_or_jump(other) ? 1.f : 0.f;
            obs[o++] = other.isDemoed ? 1.f : 0.f;
            obs[o++] = other.isSupersonic ? 1.f : 0.f;

            obs[o++] = distToEgo / POS_STD;
            obs[o++] = distToBall2 / POS_STD;

            // LookupDemoTimer by car id
            float demoT = 0.f;
            for (int i = 0; i < numCars; i++) {
                if (cars[i].id == other.id) {
                    if (timersRow[i].timersInit)
                        demoT = timersRow[i].demoTimer;
                    break;
                }
            }
            obs[o++] = demoT / DEMO_TIMER_STD;

            obs[o++] = other.isJumping ? 1.f : 0.f;
            obs[o++] = other.isFlipping ? 1.f : 0.f;
            obs[o++] = other.hasDoubleJumped ? 1.f : 0.f;

            obs[o++] = v3_length(oC.vel) / VEL_STD;
            obs[o++] = other.hasFlipped ? 1.f : 0.f;

            float maxInf, density;
            vanta_small_pad_stats(padLocations, va, other.pos, maxInf, density);
            obs[o++] = maxInf;
            obs[o++] = density;
        };

        for (int i = 0; i < VantaConst::MAX_TEAMMATES; i++) {
            if (i < numMates) {
                writeCarPacket(cars[mateIdx[i]]);
            } else {
                for (int k = 0; k < VantaConst::CAR_SIZE; k++) obs[o++] = 0.f;
            }
        }

        const int oppLimit = onlyClosestOpp ? 1 : VantaConst::MAX_OPPONENTS;
        for (int i = 0; i < VantaConst::MAX_OPPONENTS; i++) {
            if (i < oppLimit && i < numOpps) {
                writeCarPacket(cars[oppIdx[i]]);
            } else {
                for (int k = 0; k < VantaConst::CAR_SIZE; k++) obs[o++] = 0.f;
            }
        }

        // G. Validity mask (5)
        for (int i = 0; i < VantaConst::MAX_TEAMMATES; i++)
            obs[o++] = (i < numMates) ? 1.f : 0.f;
        for (int i = 0; i < VantaConst::MAX_OPPONENTS; i++)
            obs[o++] = (i < oppLimit && i < numOpps) ? 1.f : 0.f;
    }

    // ================= H. Boost Pads (34) =================
    {
        const int n = VantaConst::BOOST_SIZE;
        for (int i = 0; i < n; i++) {
            int idx = inv ? (n - 1 - i) : i;
            if (mirrorX) idx = mirrorPadMap[idx];

            if (va.boostAvail[idx]) {
                obs[o++] = 1.f;
            } else {
                float tMax = (padLocations[idx].z > 72.f) ? 10.f : 4.f;
                obs[o++] = 1.f / (1.f + va.boostTimers[idx] / tMax);
            }
        }
    }
}

}  // namespace rsc
