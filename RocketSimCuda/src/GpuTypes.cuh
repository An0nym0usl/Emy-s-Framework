#pragma once

#include <cstdint>
#include <cfloat>
#include <cmath>
#include <type_traits>
#include <RocketSimCuda.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

namespace rsc {

// Vec3 and RotMat are defined in RocketSimCuda.h (public header)
static_assert(std::is_trivially_copyable_v<Vec3>);
static_assert(std::is_trivially_copyable_v<RotMat>);



struct GpuWheelPairConfig {
    float wheelRadius;
    float suspensionRestLength; 
    Vec3 connectionPointOffset;
};

struct GpuCarConfig {
    Vec3 hitboxSize;
    Vec3 hitboxPosOffset;
    GpuWheelPairConfig frontWheels;
    GpuWheelPairConfig backWheels;
    float dodgeDeadzone;
};


struct GpuWheelState {
    bool isInContact;
    bool isInContactWithWorld;
    Vec3 contactNormal;
    Vec3 contactPoint;
    Vec3 hardPointWS;
    float suspensionLength;
    float suspensionRelVel;
    float clippedInvContactDotSuspension;
    float suspensionForce;
    float extraPushback;  // btVehicleRL::m_extraPushback (BT impulse units)
    Vec3 frictionImpulse;
    float engineForce;
    float brake;
    float steerAngle;
    float latFriction;
    float longFriction;
};


using GpuCarControls = CarControls;
static_assert(std::is_trivially_copyable_v<GpuCarControls>);



struct GpuCarState {
    // PhysState
    Vec3 pos;
    RotMat rotMat;
    Vec3 vel;
    Vec3 angVel;

    // Game state
    float boost;
    float timeSpentBoosting;
    bool isOnGround;
    bool hasJumped;
    bool hasDoubleJumped;
    bool hasFlipped;
    bool isFlipping;
    bool isJumping;
    Vec3 flipRelTorque;
    float jumpTime;
    float flipTime;
    float airTime;
    float airTimeSinceJump;
    bool isSupersonic;
    float supersonicTime;
    float handbrakeVal;

    // Auto-flip
    bool isAutoFlipping;
    float autoFlipTimer;
    float autoFlipTorqueScale;

    // Demo
    bool isDemoed;
    float demoRespawnTimer;

    // Contact tracking
    bool worldContactHasContact;
    Vec3 worldContactNormal;
    uint32_t carContactOtherID;
    float carContactCooldownTimer;

    // Ball hit tracking
    bool ballHitValid;
    uint64_t ballHitTickCount;
    uint64_t lastExtraImpulseTick;

    // Vehicle state
    GpuWheelState wheels[4];
    Vec3 velocityImpulseCache;

    // Persistent car-world contact manifold (btPersistentManifold semantics:
    // points accumulate across ticks, refreshed/expired, impulses warm-started).
    struct WorldManifoldPoint {
        Vec3 localPointA;    // contact point in car-local space
        Vec3 worldPointB;    // contact point on the static world (world space)
        Vec3 normal;         // m_normalWorldOnB
        float appliedImpulse;
        uint8_t valid;
        uint8_t algo;        // 0 = mesh narrowphase, 1 = static plane
    };
    WorldManifoldPoint worldManifold[4];

    float wsBallImpulse;       // car-ball contact warm impulse
    uint8_t wsBallContact;     // had car-ball contact last tick
    float wsCarImpulse;        // car-car contact (single slot)
    uint8_t wsCarContact;

    // Controls
    GpuCarControls controls;
    GpuCarControls lastControls;

    // Identity
    uint8_t team;   
    uint8_t preset; 
    uint32_t id;

    // Config 
    GpuCarConfig config;

    // Precomputed inertia (diagonal, in UU)
    Vec3 localInertia;
    Vec3 invLocalInertia;
};



struct GpuBallState {
    Vec3 pos;
    RotMat rotMat;
    Vec3 vel;
    Vec3 angVel;
    Vec3 velocityImpulseCache;
};



struct GpuBoostPadState {
    bool isActive;
    float cooldown;
    uint32_t prevLockedCarID;
    uint32_t curLockedCarID;
};



struct GpuArenaState {
    uint64_t tickCount;
    int numCars;
    uint8_t gameMode;
    bool goalScored;
    int goalTeam;
};


// ArenaSurface — analytical arena geometry primitive.
//
// kind == kPlane:
//   Half-space defined by `normal` (pointing into playable space) and `offset`.
//   A point p satisfies the plane equation as `dot(p, normal) - offset >= 0`,
//   i.e. p is on the playable side when the signed distance is non-negative.
//
// kind == kQuarterCylinder:
//   A quarter-arc cylindrical surface used for the floor-wall and wall-ceiling
//   blends in the soccar arena. The cylinder is defined by:
//     - `center`     : a point on the axis (the centre of the circular slice)
//     - `axis`       : unit vector along the cylinder axis
//     - `radius`     : circle radius (256 UU for floor-wall, 512 UU for wall-ceiling)
//     - `quadN1/N2`  : the two unit radial directions that span the valid quadrant.
//                      A test point p is in the quadrant iff
//                          dot(p - center - axis*axialT, quadN1) >= 0
//                       && dot(p - center - axis*axialT, quadN2) >= 0
//   Playable space is INSIDE the circle (radial distance < radius); a test point
//   that ends up OUTSIDE the circle while in the quadrant has crossed the curve
//   and must be pushed back toward `center` along the radial direction.
//
// `boundsMin/Max` is a coarse AABB used as a broadphase culling test for both
// kinds; the test point is skipped against this surface entirely if it lies
// outside the AABB.
struct ArenaSurface {
    enum Kind : int { kPlane = 0, kQuarterCylinder = 1 };
    int kind;

    // True for the four btStaticPlaneShape colliders of the reference arena
    // (floor, ceiling, +X/-X walls). These use btConvexPlaneCollisionAlgorithm
    // (single support point, no triangle margin). All other surfaces come from
    // the BVH triangle meshes (GJK with box+mesh margins for convex bodies).
    int isStaticPlane;

    // Plane data
    Vec3 normal;
    float offset;

    // Quarter-cylinder data
    Vec3 center;
    Vec3 axis;
    float radius;
    Vec3 quadN1;
    Vec3 quadN2;

    // Shared broadphase AABB
    Vec3 boundsMin;
    Vec3 boundsMax;
};


struct CurvePoint {
    float input;
    float output;
};



namespace PhysConst {
    // General
    constexpr float GRAVITY_Z           = -650.f;
    constexpr float TICK_TIME           = 1.f / 120.f;

    // Arena extents (SOCCAR)
    constexpr float ARENA_EXTENT_X      = 4096.f;
    constexpr float ARENA_EXTENT_Y      = 5120.f;
    constexpr float ARENA_HEIGHT        = 2048.f;
    constexpr float GOAL_HALF_WIDTH     = 892.755f;
    constexpr float GOAL_HEIGHT         = 642.775f;

    // Mass (RocketSim RLConst: BALL_MASS_BT = CAR_MASS_BT / 6)
    constexpr float CAR_MASS            = 180.f;
    constexpr float BALL_MASS           = CAR_MASS / 6.f; // 30.f — keep formula 1:1 with RocketSim

    // Collision parameters
    constexpr float CAR_COLLISION_FRICTION     = 0.3f;
    constexpr float CAR_COLLISION_RESTITUTION  = 0.1f;
    constexpr float CARBALL_COLLISION_FRICTION = 2.0f;
    constexpr float CARBALL_COLLISION_RESTITUTION = 0.0f;
    constexpr float CARWORLD_COLLISION_FRICTION   = 0.3f;
    constexpr float CARWORLD_COLLISION_RESTITUTION = 0.3f;
    constexpr float CARCAR_COLLISION_FRICTION  = 0.09f;
    constexpr float CARCAR_COLLISION_RESTITUTION = 0.1f;
    constexpr float BALL_FRICTION       = 0.35f;
    constexpr float BALL_RESTITUTION    = 0.6f;
    constexpr float BALL_DRAG           = 0.03f;

    // Speed limits
    constexpr float CAR_MAX_SPEED       = 2300.f;
    constexpr float BALL_MAX_SPEED      = 6000.f;
    constexpr float CAR_MAX_ANG_SPEED   = 5.5f;
    constexpr float BALL_MAX_ANG_SPEED  = 6.f;

    // Boost
    constexpr float BOOST_MAX           = 100.f;
    constexpr float BOOST_USED_PER_SECOND = 100.f / 3.f;
    constexpr float BOOST_MIN_TIME      = 0.1f;
    constexpr float BOOST_ACCEL_GROUND  = 2975.f / 3.f;
    constexpr float BOOST_ACCEL_AIR     = 3175.f / 3.f;
    constexpr float BOOST_SPAWN_AMOUNT  = 100.f / 3.f;

    // Supersonic
    constexpr float SUPERSONIC_START_SPEED        = 2200.f;
    constexpr float SUPERSONIC_MAINTAIN_MIN_SPEED = 2100.f;
    constexpr float SUPERSONIC_MAINTAIN_MAX_TIME  = 1.f;

    // Powerslide
    constexpr float POWERSLIDE_RISE_RATE = 5.f;
    constexpr float POWERSLIDE_FALL_RATE = 2.f;

    // Throttle / Brake
    constexpr float THROTTLE_TORQUE_AMOUNT = CAR_MASS * 400.f; // Force in mass*UU/s^2
    constexpr float BRAKE_TORQUE_AMOUNT    = CAR_MASS * (14.25f + 1.f/3.f);
    constexpr float STOPPING_FORWARD_VEL   = 25.f;
    constexpr float COASTING_BRAKE_FACTOR  = 0.15f;
    constexpr float BRAKING_NO_THROTTLE_SPEED_THRESH = 0.01f;
    constexpr float THROTTLE_DEADZONE      = 0.001f;
    constexpr float THROTTLE_AIR_ACCEL     = 200.f / 3.f;

    // Jump
    constexpr float JUMP_ACCEL             = 4375.f / 3.f;
    constexpr float JUMP_IMMEDIATE_FORCE   = 875.f / 3.f;
    constexpr float JUMP_MIN_TIME          = 0.025f;
    constexpr float JUMP_RESET_TIME_PAD    = 1.f / 40.f;
    constexpr float JUMP_MAX_TIME          = 0.2f;
    constexpr float JUMP_PRE_MIN_ACCEL_SCALE = 0.62f;
    constexpr float DOUBLEJUMP_MAX_DELAY   = 1.25f;

    // Flip / Dodge
    constexpr float FLIP_Z_DAMP_120        = 0.35f;
    constexpr float FLIP_Z_DAMP_START      = 0.15f;
    constexpr float FLIP_Z_DAMP_END        = 0.21f;
    constexpr float FLIP_TORQUE_TIME       = 0.65f;
    constexpr float FLIP_TORQUE_MIN_TIME   = 0.41f;
    constexpr float FLIP_PITCHLOCK_TIME    = 1.f;
    constexpr float FLIP_PITCHLOCK_EXTRA_TIME = 0.3f;
    constexpr float FLIP_INITIAL_VEL_SCALE = 500.f;
    constexpr float FLIP_TORQUE_X          = 260.f;
    constexpr float FLIP_TORQUE_Y          = 224.f;
    constexpr float FLIP_FORWARD_IMPULSE_MAX_SPEED_SCALE  = 1.f;
    constexpr float FLIP_SIDE_IMPULSE_MAX_SPEED_SCALE     = 1.9f;
    constexpr float FLIP_BACKWARD_IMPULSE_MAX_SPEED_SCALE = 2.5f;
    constexpr float FLIP_BACKWARD_IMPULSE_SCALE_X         = 16.f / 15.f;

    // Air control torque (PYR order in original)
    constexpr float AIR_CONTROL_TORQUE_PITCH = 130.f;
    constexpr float AIR_CONTROL_TORQUE_YAW   = 95.f;
    constexpr float AIR_CONTROL_TORQUE_ROLL  = 400.f;
    constexpr float AIR_CONTROL_DAMPING_PITCH = 30.f;
    constexpr float AIR_CONTROL_DAMPING_YAW   = 20.f;
    constexpr float AIR_CONTROL_DAMPING_ROLL  = 50.f;
    constexpr float CAR_TORQUE_SCALE = 2.f * M_PI / (1 << 16) * 1000.f;

    // Auto-flip
    constexpr float CAR_AUTOFLIP_IMPULSE    = 200.f;
    constexpr float CAR_AUTOFLIP_TORQUE     = 50.f;
    constexpr float CAR_AUTOFLIP_TIME       = 0.4f;
    constexpr float CAR_AUTOFLIP_NORMZ_THRESH = 0.70710678f; // sqrt(0.5)
    constexpr float CAR_AUTOFLIP_ROLL_THRESH  = 2.8f;

    // Auto-roll
    constexpr float CAR_AUTOROLL_FORCE  = 100.f;
    constexpr float CAR_AUTOROLL_TORQUE = 80.f;

    // Ball-car extra impulse
    constexpr float BALL_CAR_EXTRA_IMPULSE_Z_SCALE       = 0.35f;
    constexpr float BALL_CAR_EXTRA_IMPULSE_FORWARD_SCALE = 0.65f;
    constexpr float BALL_CAR_EXTRA_IMPULSE_MAXDELTAVEL   = 4600.f;

    // Bumps / Demos
    constexpr float BUMP_COOLDOWN_TIME   = 0.25f;
    constexpr float BUMP_MIN_FORWARD_DIST = 64.5f;
    constexpr float DEMO_RESPAWN_TIME    = 3.f;

    // Ball
    constexpr float BALL_REST_Z          = 93.15f;
    constexpr float BALL_COLLISION_RADIUS_SOCCAR = 91.25f;
    constexpr float BALL_WORLD_COLLISION_MARGIN = BALL_REST_Z - BALL_COLLISION_RADIUS_SOCCAR;
    constexpr float BALL_FLOOR_REST_BOUNCE_SPEED = 6.0f;
    constexpr float BALL_REST_POS_EPS = 0.5f;
    constexpr float BALL_REST_VERTICAL_SPEED_EPS = 3.0f;
    constexpr float BALL_REST_HORIZONTAL_SPEED_EPS = 0.5f;
    constexpr float BALL_REST_ANG_SPEED_EPS = 0.5f;

    // Goal
    constexpr float SOCCAR_GOAL_SCORE_THRESHOLD_Y = 5124.25f;

    // Spawn
    constexpr float CAR_SPAWN_REST_Z  = 17.f;
    constexpr float CAR_RESPAWN_Z     = 36.f;

    // Suspension (btVehicleRL)
    constexpr float SUSPENSION_STIFFNESS        = 500.f;
    constexpr float WHEELS_DAMPING_COMPRESSION  = 25.f;
    constexpr float WHEELS_DAMPING_RELAXATION   = 40.f;
    constexpr float MAX_SUSPENSION_TRAVEL       = 12.f;
    constexpr float SUSPENSION_SUBTRACTION      = 0.05f * 50.f; 
    constexpr float SUSPENSION_FORCE_SCALE_FRONT = 36.f - 0.25f;
    constexpr float SUSPENSION_FORCE_SCALE_BACK  = 54.f + 0.25f + 0.015f;

    // Boost pads
    constexpr float PAD_CYL_HEIGHT   = 95.f;
    constexpr float PAD_CYL_RAD_BIG  = 208.f;
    constexpr float PAD_CYL_RAD_SMALL = 144.f;
    constexpr float PAD_BOX_HEIGHT   = 64.f;
    constexpr float PAD_BOX_RAD_BIG  = 160.f;
    constexpr float PAD_BOX_RAD_SMALL = 120.f;
    constexpr float PAD_COOLDOWN_BIG   = 10.f;
    constexpr float PAD_COOLDOWN_SMALL = 4.f;
    constexpr float PAD_BOOST_BIG    = 100.f;
    constexpr float PAD_BOOST_SMALL  = 12.f;
}



namespace Curves {
    // Steer angle from speed
    constexpr int STEER_ANGLE_N = 6;
    constexpr __device__ CurvePoint STEER_ANGLE[STEER_ANGLE_N] = {
        {0.f,    0.53356f},
        {500.f,  0.31930f},
        {1000.f, 0.18203f},
        {1500.f, 0.10570f},
        {1750.f, 0.08507f},
        {3000.f, 0.03454f},
    };

    // Powerslide steer angle from speed
    constexpr int POWERSLIDE_STEER_N = 2;
    constexpr __device__ CurvePoint POWERSLIDE_STEER[POWERSLIDE_STEER_N] = {
        {0.f,    0.39235f},
        {2500.f, 0.12610f},
    };

    // Drive speed torque factor
    constexpr int DRIVE_TORQUE_N = 3;
    constexpr __device__ CurvePoint DRIVE_TORQUE[DRIVE_TORQUE_N] = {
        {0.f,    1.0f},
        {1400.f, 0.1f},
        {1410.f, 0.0f},
    };

    // Non-sticky friction factor
    constexpr int NON_STICKY_FRICTION_N = 3;
    constexpr __device__ CurvePoint NON_STICKY_FRICTION[NON_STICKY_FRICTION_N] = {
        {0.f,      0.1f},
        {0.7075f,  0.5f},
        {1.f,      1.0f},
    };

    // Lateral friction
    constexpr int LAT_FRICTION_N = 2;
    constexpr __device__ CurvePoint LAT_FRICTION[LAT_FRICTION_N] = {
        {0.f, 1.0f},
        {1.f, 0.2f},
    };

    // Longitudinal friction (empty -> always returns 1.0)
    constexpr int LONG_FRICTION_N = 0;

    // Handbrake lateral friction factor
    constexpr int HANDBRAKE_LAT_N = 1;
    constexpr __device__ CurvePoint HANDBRAKE_LAT[HANDBRAKE_LAT_N] = {
        {0.f, 0.1f},
    };

    // Handbrake longitudinal friction factor
    constexpr int HANDBRAKE_LONG_N = 2;
    constexpr __device__ CurvePoint HANDBRAKE_LONG[HANDBRAKE_LONG_N] = {
        {0.f, 0.5f},
        {1.f, 0.9f},
    };

    // Ball-car extra impulse factor
    constexpr int BALL_CAR_IMPULSE_N = 4;
    constexpr __device__ CurvePoint BALL_CAR_IMPULSE[BALL_CAR_IMPULSE_N] = {
        {0.f,    0.65f},
        {500.f,  0.65f},
        {2300.f, 0.55f},
        {4600.f, 0.30f},
    };

    // Bump velocity (ground)
    constexpr int BUMP_VEL_GROUND_N = 3;
    constexpr __device__ CurvePoint BUMP_VEL_GROUND[BUMP_VEL_GROUND_N] = {
        {0.f,    5.f / 6.f},
        {1400.f, 1100.f},
        {2200.f, 1530.f},
    };

    // Bump velocity (air)
    constexpr int BUMP_VEL_AIR_N = 3;
    constexpr __device__ CurvePoint BUMP_VEL_AIR[BUMP_VEL_AIR_N] = {
        {0.f,    5.f / 6.f},
        {1400.f, 1390.f},
        {2200.f, 1945.f},
    };

    // Bump upward velocity
    constexpr int BUMP_UPWARD_N = 3;
    constexpr __device__ CurvePoint BUMP_UPWARD[BUMP_UPWARD_N] = {
        {0.f,    2.f / 6.f},
        {1400.f, 278.f},
        {2200.f, 417.f},
    };
}


namespace CarPresets {
    // hitboxSize, hitboxPosOffset
    // frontWheels: {radius, suspRestLen - MAX_SUSP_TRAVEL, connectionOffset}
    // backWheels:  {radius, suspRestLen - MAX_SUSP_TRAVEL, connectionOffset}
    // dodgeDeadzone

    inline const GpuCarConfig OCTANE = {
        {120.507f, 86.6994f, 38.6591f},
        {13.87566f, 0.f, 20.755f},
        {12.50f, 38.755f - 12.f, {51.25f, 25.90f, 20.755f}},
        {15.00f, 37.055f - 12.f, {-33.75f, 29.50f, 20.755f}},
        0.5f
    };
    inline const GpuCarConfig DOMINUS = {
        {130.427f, 85.7799f, 33.8f},
        {9.f, 0.f, 15.75f},
        {12.00f, 33.95f - 12.f, {50.30f, 31.10f, 15.75f}},
        {13.50f, 33.85f - 12.f, {-34.75f, 33.00f, 15.75f}},
        0.5f
    };
    inline const GpuCarConfig PLANK = {
        {131.32f, 87.1704f, 31.8944f},
        {9.00857f, 0.f, 12.0942f},
        {12.50f, 31.9242f - 12.f, {49.97f, 27.80f, 12.0942f}},
        {17.00f, 27.9242f - 12.f, {-35.43f, 20.28f, 12.0942f}},
        0.5f
    };
    inline const GpuCarConfig BREAKOUT = {
        {133.992f, 83.021f, 32.8f},
        {12.5f, 0.f, 11.75f},
        {13.50f, 29.7f - 12.f, {51.50f, 26.67f, 11.75f}},
        {15.00f, 29.666f - 12.f, {-35.75f, 35.00f, 11.75f}},
        0.5f
    };
    inline const GpuCarConfig HYBRID = {
        {129.519f, 84.6879f, 36.6591f},
        {13.8757f, 0.f, 20.755f},
        {12.50f, 38.755f - 12.f, {51.25f, 25.90f, 20.755f}},
        {15.00f, 37.055f - 12.f, {-34.00f, 29.50f, 20.755f}},
        0.5f
    };
    inline const GpuCarConfig MERC = {
        {123.22f, 79.2103f, 44.1591f},
        {11.3757f, 0.f, 21.505f},
        {15.00f, 39.505f - 12.f, {51.25f, 25.90f, 21.505f}},
        {15.00f, 39.105f - 12.f, {-33.75f, 29.50f, 21.505f}},
        0.5f
    };

    inline const GpuCarConfig& Get(int preset) {
        switch (preset) {
        case 0: return OCTANE;
        case 1: return DOMINUS;
        case 2: return PLANK;
        case 3: return BREAKOUT;
        case 4: return HYBRID;
        default: return MERC;
        }
    }
}

namespace BoostPadData {
    constexpr int NUM_SMALL = 28;
    constexpr int NUM_BIG   = 6;
    constexpr int NUM_TOTAL = NUM_SMALL + NUM_BIG;

    // Small pads first, then big pads
    constexpr Vec3 POSITIONS[NUM_TOTAL] = {
        // 28 small pads
        {0.f,      -4240.f, 70.f},
        {-1792.f,  -4184.f, 70.f},
        {1792.f,   -4184.f, 70.f},
        {-940.f,   -3308.f, 70.f},
        {940.f,    -3308.f, 70.f},
        {0.f,      -2816.f, 70.f},
        {-3584.f,  -2484.f, 70.f},
        {3584.f,   -2484.f, 70.f},
        {-1788.f,  -2300.f, 70.f},
        {1788.f,   -2300.f, 70.f},
        {-2048.f,  -1036.f, 70.f},
        {0.f,      -1024.f, 70.f},
        {2048.f,   -1036.f, 70.f},
        {-1024.f,  0.f,     70.f},
        {1024.f,   0.f,     70.f},
        {-2048.f,  1036.f,  70.f},
        {0.f,      1024.f,  70.f},
        {2048.f,   1036.f,  70.f},
        {-1788.f,  2300.f,  70.f},
        {1788.f,   2300.f,  70.f},
        {-3584.f,  2484.f,  70.f},
        {3584.f,   2484.f,  70.f},
        {0.f,      2816.f,  70.f},
        {-940.f,   3308.f,  70.f},
        {940.f,    3308.f,  70.f},
        {-1792.f,  4184.f,  70.f},
        {1792.f,   4184.f,  70.f},
        {0.f,      4240.f,  70.f},
        // 6 big pads
        {-3584.f,  0.f,     73.f},
        {3584.f,   0.f,     73.f},
        {-3072.f,  4096.f,  73.f},
        {3072.f,   4096.f,  73.f},
        {-3072.f,  -4096.f, 73.f},
        {3072.f,   -4096.f, 73.f},
    };

    constexpr bool IS_BIG[NUM_TOTAL] = {
        // 28 small
        false, false, false, false, false, false, false,
        false, false, false, false, false, false, false,
        false, false, false, false, false, false, false,
        false, false, false, false, false, false, false,
        // 6 big
        true, true, true, true, true, true,
    };

    __host__ __device__ inline bool IsBig(int idx) {
        return idx >= NUM_SMALL;
    }

    __host__ __device__ inline Vec3 Position(int idx) {
        switch (idx) {
        case 0: return {0.f, -4240.f, 70.f};
        case 1: return {-1792.f, -4184.f, 70.f};
        case 2: return {1792.f, -4184.f, 70.f};
        case 3: return {-940.f, -3308.f, 70.f};
        case 4: return {940.f, -3308.f, 70.f};
        case 5: return {0.f, -2816.f, 70.f};
        case 6: return {-3584.f, -2484.f, 70.f};
        case 7: return {3584.f, -2484.f, 70.f};
        case 8: return {-1788.f, -2300.f, 70.f};
        case 9: return {1788.f, -2300.f, 70.f};
        case 10: return {-2048.f, -1036.f, 70.f};
        case 11: return {0.f, -1024.f, 70.f};
        case 12: return {2048.f, -1036.f, 70.f};
        case 13: return {-1024.f, 0.f, 70.f};
        case 14: return {1024.f, 0.f, 70.f};
        case 15: return {-2048.f, 1036.f, 70.f};
        case 16: return {0.f, 1024.f, 70.f};
        case 17: return {2048.f, 1036.f, 70.f};
        case 18: return {-1788.f, 2300.f, 70.f};
        case 19: return {1788.f, 2300.f, 70.f};
        case 20: return {-3584.f, 2484.f, 70.f};
        case 21: return {3584.f, 2484.f, 70.f};
        case 22: return {0.f, 2816.f, 70.f};
        case 23: return {-940.f, 3308.f, 70.f};
        case 24: return {940.f, 3308.f, 70.f};
        case 25: return {-1792.f, 4184.f, 70.f};
        case 26: return {1792.f, 4184.f, 70.f};
        case 27: return {0.f, 4240.f, 70.f};
        case 28: return {-3584.f, 0.f, 73.f};
        case 29: return {3584.f, 0.f, 73.f};
        case 30: return {-3072.f, 4096.f, 73.f};
        case 31: return {3072.f, 4096.f, 73.f};
        case 32: return {-3072.f, -4096.f, 73.f};
        default: return {3072.f, -4096.f, 73.f};
        }
    }
}

// KickOff
namespace SpawnData {
    struct SpawnPos { float x, y, yawAng; };

    constexpr int SPAWN_COUNT = 5;
    constexpr SpawnPos SPAWNS_SOCCAR[SPAWN_COUNT] = {
        {-2048.f, -2560.f, M_PI / 4.f * 1.f},
        { 2048.f, -2560.f, M_PI / 4.f * 3.f},
        { -256.f, -3840.f, M_PI / 4.f * 2.f},
        {  256.f, -3840.f, M_PI / 4.f * 2.f},
        {    0.f, -4608.f, M_PI / 4.f * 2.f},
    };

    constexpr int RESPAWN_COUNT = 4;
    constexpr SpawnPos RESPAWNS_SOCCAR[RESPAWN_COUNT] = {
        {-2304.f, -4608.f, M_PI / 2.f},
        {-2688.f, -4608.f, M_PI / 2.f},
        { 2304.f, -4608.f, M_PI / 2.f},
        { 2688.f, -4608.f, M_PI / 2.f},
    };

    __host__ __device__ inline SpawnPos SpawnSoccar(int idx) {
        switch (idx) {
        case 0: return {-2048.f, -2560.f, M_PI / 4.f * 1.f};
        case 1: return {2048.f, -2560.f, M_PI / 4.f * 3.f};
        case 2: return {-256.f, -3840.f, M_PI / 4.f * 2.f};
        case 3: return {256.f, -3840.f, M_PI / 4.f * 2.f};
        default: return {0.f, -4608.f, M_PI / 4.f * 2.f};
        }
    }

    __host__ __device__ inline SpawnPos RespawnSoccar(int idx) {
        switch (idx) {
        case 0: return {-2304.f, -4608.f, M_PI / 2.f};
        case 1: return {-2688.f, -4608.f, M_PI / 2.f};
        case 2: return {2304.f, -4608.f, M_PI / 2.f};
        default: return {2688.f, -4608.f, M_PI / 2.f};
        }
    }

    // RocketSim Arena::ResetToRandomKickoff orange mirror: pos *= {-1,-1,1}; yaw += π
    // (Demo respawn is different — Y-only flip — see Car::Respawn / device_handle_demo.)
    __host__ __device__ inline void KickoffPoseForTeam(
        const SpawnPos& sp, int team, float& outX, float& outY, float& outYaw)
    {
        if (team == 0) {
            outX = sp.x;
            outY = sp.y;
            outYaw = sp.yawAng;
        } else {
            outX = -sp.x;
            outY = -sp.y;
            outYaw = sp.yawAng + (float)M_PI;
        }
    }

    __host__ __device__ inline RotMat YawToRotMat(float yaw) {
        float cy = cosf(yaw), sy = sinf(yaw);
        return {{cy, sy, 0}, {-sy, cy, 0}, {0, 0, 1}};
    }
}



// Max cars per arena and max arena surfaces
constexpr int MAX_CARS_PER_ARENA = 6;
// 14 planes (floor, ceiling, 6 wall segments, 4 corner bevels) +
// 10 quarter-cylinders (4 X-wall floor/ceil curves + 6 Y-wall floor/ceil curves) +
// headroom for the future triangle-mesh goal-back surfaces.
constexpr int MAX_ARENA_SURFACES = 32;

} // namespace rsc
