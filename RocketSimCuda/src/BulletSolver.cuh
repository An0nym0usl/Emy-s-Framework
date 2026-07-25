#pragma once

// ============================================================================
// BulletSolver.cuh — exact transcription of the contact-solving pieces of
// RocketSim's patched bullet3-3.24 (btSequentialImpulseConstraintSolver and
// friends), in BT units (1 BT = 50 UU), float32, matching operation order.
//
// Sources transcribed (GigaLearnCPP/RLGymCPP/RocketSim/libsrc/bullet3-3.24):
//  - btSequentialImpulseConstraintSolver.cpp:
//      gResolveSingleConstraintRowGeneric_scalar_reference
//      gResolveSingleConstraintRowLowerLimit_scalar_reference
//      gResolveSplitPenetrationImpulse_scalar_reference
//      setupContactConstraint / setupFrictionConstraint / convertContactInner
//      convertContactSpecial (RocketSim aggregated ball-world contact)
//      restitutionCurve, solve loop structure (10 iterations, no early-out)
//  - btContactConstraint.cpp: resolveSingleBilateral, resolveSingleCollision
//  - btManifoldResult.cpp (ROCKETSIM CHANGE): static-vs-dynamic combiners are
//      friction = min(a,b), restitution = max(a,b)
//  - LinearMath/btTransformUtil.h: integrateTransform (quaternion exp map)
//  - LinearMath/btMatrix3x3.h: getRotation / setRotation
//  - Solver config from Arena.cpp: erp2 = 0.8, splitImpulsePenetrationThreshold
//    = 1e30 (=> always split impulse), default solver mode = warmstarting only
//    (single velocity-dependent friction direction, friction NOT warm-started).
// ============================================================================

#include "CudaMath.cuh"

namespace rsc {
namespace bts {

constexpr float UU_TO_BT = 1.f / 50.f;
constexpr float BT_TO_UU = 50.f;

constexpr float ERP2 = 0.8f;                            // Arena.cpp:488
constexpr float RESTITUTION_VELOCITY_THRESHOLD = 0.2f;  // btContactSolverInfo.h:108
constexpr float WARMSTARTING_FACTOR = 0.85f;            // btContactSolverInfo.h:99
constexpr int   NUM_ITERATIONS = 10;                    // btContactSolverInfo.h:85
constexpr float LINEAR_SLOP = 0.f;                      // btContactSolverInfo.h:98
constexpr float SPLIT_IMPULSE_TURN_ERP = 0.1f;          // btContactSolverInfo.h
constexpr float SIMD_EPSILON_F = 1.19209290e-07f;       // FLT_EPSILON
constexpr float ANGULAR_MOTION_THRESHOLD = 0.5f * 1.57079632679489661923f; // 0.5*SIMD_HALF_PI

// Combined contact properties for STATIC vs DYNAMIC pairs.
// ROCKETSIM CHANGE in btManifoldResult.cpp: min for friction, max for restitution.
__device__ __forceinline__ float combined_friction_static(float a, float b) {
    return (a < b) ? a : b;
}
__device__ __forceinline__ float combined_restitution_static(float a, float b) {
    return (a > b) ? a : b;
}

// btSequentialImpulseConstraintSolver::restitutionCurve
__device__ __forceinline__ float restitution_curve(float rel_vel, float restitution, float velocityThreshold) {
    if (fabsf(rel_vel) < velocityThreshold)
        return 0.f;
    return restitution * -rel_vel;
}

// btPlaneSpace1 (btVector3.h) — fallback friction basis when there is no
// lateral relative velocity.
__device__ inline void bt_plane_space1(Vec3 n, Vec3& p, Vec3& q) {
    constexpr float SIMDSQRT12 = 0.7071067811865475244008443621048490f;
    if (fabsf(n.z) > SIMDSQRT12) {
        // choose p in y-z plane
        float a = n.y * n.y + n.z * n.z;
        float k = 1.0f / sqrtf(a);
        p.x = 0.f;
        p.y = -n.z * k;
        p.z = n.y * k;
        // set q = n x p
        q.x = a * k;
        q.y = -n.x * p.z;
        q.z = n.x * p.y;
    } else {
        // choose p in x-y plane
        float a = n.x * n.x + n.y * n.y;
        float k = 1.0f / sqrtf(a);
        p.x = -n.y * k;
        p.y = n.x * k;
        p.z = 0.f;
        // set q = n x p
        q.x = -n.z * p.y;
        q.y = n.z * p.x;
        q.z = a * k;
    }
}

// ----------------------------------------------------------------------------
// Symmetric 3x3 (world-space inverse inertia tensor), stored as rows.
// invInertiaWorld = basis * diag(invInertiaLocal) * basis^T
// (btRigidBody::updateInertiaTensor uses m_worldTransform.getBasis().scaled())
// ----------------------------------------------------------------------------
struct Mat3 {
    Vec3 r0, r1, r2;
};

__device__ __forceinline__ Vec3 mat3_mul(const Mat3& m, Vec3 v) {
    return {v3_dot(m.r0, v), v3_dot(m.r1, v), v3_dot(m.r2, v)};
}

// basis columns are (forward, right, up); element m[i][j] is row i, col j of
// the bullet matrix. scaled(invDiag) multiplies columns; result times
// transpose(basis).
__device__ inline Mat3 make_inv_inertia_world(const RotMat& b, Vec3 invDiag) {
    // bullet: m_invInertiaTensorWorld = basis.scaled(invDiag) * basis.transpose()
    // basis.scaled(d) row i = (m[i][0]*d.x, m[i][1]*d.y, m[i][2]*d.z)
    // basis rows: row0 = (f.x, r.x, u.x), row1 = (f.y, r.y, u.y), row2 = (f.z, r.z, u.z)
    Vec3 row0 = {b.forward.x, b.right.x, b.up.x};
    Vec3 row1 = {b.forward.y, b.right.y, b.up.y};
    Vec3 row2 = {b.forward.z, b.right.z, b.up.z};

    Vec3 s0 = {row0.x * invDiag.x, row0.y * invDiag.y, row0.z * invDiag.z};
    Vec3 s1 = {row1.x * invDiag.x, row1.y * invDiag.y, row1.z * invDiag.z};
    Vec3 s2 = {row2.x * invDiag.x, row2.y * invDiag.y, row2.z * invDiag.z};

    // (scaled * basis^T)[i][j] = s_i . row_j
    Mat3 out;
    out.r0 = {v3_dot(s0, row0), v3_dot(s0, row1), v3_dot(s0, row2)};
    out.r1 = {v3_dot(s1, row0), v3_dot(s1, row1), v3_dot(s1, row2)};
    out.r2 = {v3_dot(s2, row0), v3_dot(s2, row1), v3_dot(s2, row2)};
    return out;
}

// ----------------------------------------------------------------------------
// Quaternion helpers (btQuaternion / btMatrix3x3 transcriptions)
// ----------------------------------------------------------------------------
struct Quat {
    float x, y, z, w;
};

// btMatrix3x3::getRotation
__device__ inline Quat rotmat_to_quat(const RotMat& b) {
    // m[i][j]: row i, col j; columns are forward/right/up
    float m00 = b.forward.x, m01 = b.right.x, m02 = b.up.x;
    float m10 = b.forward.y, m11 = b.right.y, m12 = b.up.y;
    float m20 = b.forward.z, m21 = b.right.z, m22 = b.up.z;

    float trace = m00 + m11 + m22;
    Quat q;
    if (trace > 0.f) {
        float s = sqrtf(trace + 1.0f);
        q.w = s * 0.5f;
        s = 0.5f / s;
        q.x = (m21 - m12) * s;
        q.y = (m02 - m20) * s;
        q.z = (m10 - m01) * s;
    } else {
        int i = m00 < m11 ? (m11 < m22 ? 2 : 1) : (m00 < m22 ? 2 : 0);
        int j = (i + 1) % 3;
        int k = (i + 2) % 3;

        // matrix accessor m[r][c]
        float m[3][3] = {{m00, m01, m02}, {m10, m11, m12}, {m20, m21, m22}};

        float s = sqrtf(m[i][i] - m[j][j] - m[k][k] + 1.0f);
        float temp[4];
        temp[i] = s * 0.5f;
        s = 0.5f / s;
        temp[3] = (m[k][j] - m[j][k]) * s;
        temp[j] = (m[j][i] + m[i][j]) * s;
        temp[k] = (m[k][i] + m[i][k]) * s;
        q.x = temp[0];
        q.y = temp[1];
        q.z = temp[2];
        q.w = temp[3];
    }
    return q;
}

__device__ __forceinline__ Quat quat_mul(const Quat& a, const Quat& b) {
    // btQuaternion operator*
    return {
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y + a.y * b.w + a.z * b.x - a.x * b.z,
        a.w * b.z + a.z * b.w + a.x * b.y - a.y * b.x,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z};
}

__device__ __forceinline__ Quat quat_normalize(const Quat& q) {
    float len = sqrtf(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    float inv = 1.0f / len;
    return {q.x * inv, q.y * inv, q.z * inv, q.w * inv};
}

// btMatrix3x3::setRotation
__device__ inline RotMat quat_to_rotmat(const Quat& q) {
    float d = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
    float s = 2.0f / d;
    float xs = q.x * s, ys = q.y * s, zs = q.z * s;
    float wx = q.w * xs, wy = q.w * ys, wz = q.w * zs;
    float xx = q.x * xs, xy = q.x * ys, xz = q.x * zs;
    float yy = q.y * ys, yz = q.y * zs, zz = q.z * zs;

    // setValue(row-major): m00..m22
    float m00 = 1.0f - (yy + zz), m01 = xy - wz, m02 = xz + wy;
    float m10 = xy + wz, m11 = 1.0f - (xx + zz), m12 = yz - wx;
    float m20 = xz - wy, m21 = yz + wx, m22 = 1.0f - (xx + yy);

    RotMat out;
    out.forward = {m00, m10, m20};
    out.right   = {m01, m11, m21};
    out.up      = {m02, m12, m22};
    return out;
}

// btTransformUtil::integrateTransform — rotation part (exponential map).
// Returns the new basis after rotating `basis` by `angVel` for `dt`.
__device__ inline RotMat bt_integrate_rotation(const RotMat& basis, Vec3 angvel, float dt) {
    Vec3 axis;
    float fAngle2 = v3_length_sq(angvel);
    float fAngle = 0.f;
    if (fAngle2 > SIMD_EPSILON_F)
        fAngle = sqrtf(fAngle2);

    // limit the angular motion
    if (fAngle * dt > ANGULAR_MOTION_THRESHOLD)
        fAngle = ANGULAR_MOTION_THRESHOLD / dt;

    if (fAngle < 0.001f) {
        // use Taylor's expansions of sync function
        axis = angvel * (0.5f * dt - (dt * dt * dt) * 0.020833333333f * fAngle * fAngle);
    } else {
        axis = angvel * (sinf(0.5f * fAngle * dt) / fAngle);
    }

    Quat dorn = {axis.x, axis.y, axis.z, cosf(fAngle * dt * 0.5f)};
    Quat orn0 = rotmat_to_quat(basis);
    Quat predictedOrn = quat_mul(dorn, orn0);
    predictedOrn = quat_normalize(predictedOrn);
    return quat_to_rotmat(predictedOrn);
}

// ----------------------------------------------------------------------------
// Solver bodies and constraint rows
// ----------------------------------------------------------------------------

// All quantities in BT units. Index -1 = the static fixed body (zero mass).
struct SolverBody {
    Vec3 linVel;            // m_linearVelocity at solve start (post pre-tick impulses, post damping)
    Vec3 angVel;            // m_angularVelocity
    Vec3 extForceImpulse;   // totalForce * invMass * dt  (accumulated accelerations * dt)
    Vec3 extTorqueImpulse;  // invInertiaWorld * totalTorque * dt (accumulated angVel deltas)
    Vec3 dLin, dAng;        // delta velocities (solver internal)
    Vec3 pushVel, turnVel;  // split impulse
    float invMass;
    Mat3 invInertiaWorld;
};

struct ContactRow {
    int bodyA;  // index into bodies array (always valid)
    int bodyB;  // index, or -1 for static world
    Vec3 contactNormal1;        // for body A (world normal)
    Vec3 contactNormal2;        // for body B (= -normal), zero if static
    Vec3 relpos1CrossNormal;
    Vec3 relpos2CrossNormal;
    Vec3 angularComponentA;
    Vec3 angularComponentB;
    float jacDiagABInv;
    float rhs;
    float rhsPenetration;
    float cfm;
    float lowerLimit;
    float upperLimit;
    float appliedImpulse;
    float appliedPushImpulse;
    float friction;
    int frictionIndex;          // for friction rows: index of the owning normal row
    bool isSpecial;             // RocketSim special (ball-world real contacts): split-impulse only
};

__device__ __forceinline__ Vec3 body_vel_at_point_no_delta(const SolverBody& b, Vec3 rel_pos) {
    // btSolverBody::getVelocityInLocalPointNoDelta:
    // m_linearVelocity + m_externalForceImpulse + (m_angularVelocity + m_externalTorqueImpulse) x rel_pos
    return b.linVel + b.extForceImpulse + v3_cross(b.angVel + b.extTorqueImpulse, rel_pos);
}

// internalApplyImpulse on delta velocities
__device__ __forceinline__ void body_apply_impulse(SolverBody& b, Vec3 linearComponent, Vec3 angularComponent, float impulseMagnitude) {
    b.dLin += linearComponent * impulseMagnitude;
    b.dAng += angularComponent * impulseMagnitude;
}

__device__ __forceinline__ void body_apply_push_impulse(SolverBody& b, Vec3 linearComponent, Vec3 angularComponent, float impulseMagnitude) {
    b.pushVel += linearComponent * impulseMagnitude;
    b.turnVel += angularComponent * impulseMagnitude;
}

// ----------------------------------------------------------------------------
// setupContactConstraint transcription.
//  cpDistance: signed contact distance (negative = penetrating), BT units
//  warmImpulse: previous tick's applied impulse for this persistent point
//               (pass 0 for new contacts)
// ----------------------------------------------------------------------------
__device__ inline void setup_contact_row(
    ContactRow& row,
    SolverBody* bodies,
    int bodyAIdx, int bodyBIdx,
    Vec3 normalWorldOnB,
    Vec3 rel_pos1, Vec3 rel_pos2,
    float cpDistance,
    float combinedFriction,
    float combinedRestitution,
    float warmImpulse,
    float invTimeStep,
    bool isSpecial
) {
    SolverBody& bodyA = bodies[bodyAIdx];
    SolverBody* bodyB = (bodyBIdx >= 0) ? &bodies[bodyBIdx] : nullptr;

    row.bodyA = bodyAIdx;
    row.bodyB = bodyBIdx;
    row.isSpecial = isSpecial;
    row.friction = combinedFriction;
    row.cfm = 0.f;  // m_globalCfm default 0, then *= invTimeStep -> still 0

    float erp = ERP2;

    Vec3 torqueAxis0 = v3_cross(rel_pos1, normalWorldOnB);
    row.angularComponentA = mat3_mul(bodyA.invInertiaWorld, torqueAxis0);
    Vec3 torqueAxis1 = v3_cross(rel_pos2, normalWorldOnB);
    row.angularComponentB = bodyB ? mat3_mul(bodyB->invInertiaWorld, -torqueAxis1) : v3_zero();

    {
        float denom0 = 0.f, denom1 = 0.f;
        {
            Vec3 vec = v3_cross(row.angularComponentA, rel_pos1);
            denom0 = bodyA.invMass + v3_dot(normalWorldOnB, vec);
        }
        if (bodyB) {
            Vec3 vec = v3_cross(-row.angularComponentB, rel_pos2);
            denom1 = bodyB->invMass + v3_dot(normalWorldOnB, vec);
        }
        float denom = 1.f / (denom0 + denom1);  // relaxation (m_sor) = 1
        row.jacDiagABInv = denom;
    }

    row.contactNormal1 = normalWorldOnB;
    row.relpos1CrossNormal = torqueAxis0;
    if (bodyB) {
        row.contactNormal2 = -normalWorldOnB;
        row.relpos2CrossNormal = -torqueAxis1;
    } else {
        row.contactNormal2 = v3_zero();
        row.relpos2CrossNormal = v3_zero();
    }

    float restitution = 0.f;
    float penetration = cpDistance + LINEAR_SLOP;

    {
        Vec3 vel1 = bodyA.linVel + v3_cross(bodyA.angVel, rel_pos1);
        Vec3 vel2 = bodyB ? (bodyB->linVel + v3_cross(bodyB->angVel, rel_pos2)) : v3_zero();
        Vec3 vel = vel1 - vel2;
        float rel_vel = v3_dot(normalWorldOnB, vel);

        restitution = restitution_curve(rel_vel, combinedRestitution, RESTITUTION_VELOCITY_THRESHOLD);
        if (restitution <= 0.f)
            restitution = 0.f;
    }

    // Warm starting
    row.appliedImpulse = warmImpulse * WARMSTARTING_FACTOR;
    if (row.appliedImpulse != 0.f) {
        body_apply_impulse(bodyA, row.contactNormal1 * bodyA.invMass, row.angularComponentA, row.appliedImpulse);
        if (bodyB)
            body_apply_impulse(*bodyB, -row.contactNormal2 * bodyB->invMass, -row.angularComponentB, -row.appliedImpulse);
    }

    row.appliedPushImpulse = 0.f;

    {
        Vec3 externalForceImpulseA = bodyA.extForceImpulse;
        Vec3 externalTorqueImpulseA = bodyA.extTorqueImpulse;
        Vec3 externalForceImpulseB = bodyB ? bodyB->extForceImpulse : v3_zero();
        Vec3 externalTorqueImpulseB = bodyB ? bodyB->extTorqueImpulse : v3_zero();

        float vel1Dotn = v3_dot(row.contactNormal1, bodyA.linVel + externalForceImpulseA)
                       + v3_dot(row.relpos1CrossNormal, bodyA.angVel + externalTorqueImpulseA);
        float vel2Dotn = bodyB
            ? (v3_dot(row.contactNormal2, bodyB->linVel + externalForceImpulseB)
               + v3_dot(row.relpos2CrossNormal, bodyB->angVel + externalTorqueImpulseB))
            : 0.f;
        float rel_vel = vel1Dotn + vel2Dotn;

        float positionalError = 0.f;
        float velocityError = restitution - rel_vel;

        if (penetration > 0.f) {
            positionalError = 0.f;
            // ROCKETSIM CHANGE: velocityError -= penetration * invTimeStep  (removed)
        } else {
            positionalError = -penetration * erp * invTimeStep;
        }

        float penetrationImpulse = positionalError * row.jacDiagABInv;
        float velocityImpulse = velocityError * row.jacDiagABInv;

        // splitImpulsePenetrationThreshold = 1e30 => always split
        row.rhs = velocityImpulse;
        row.rhsPenetration = penetrationImpulse;
        row.lowerLimit = 0.f;
        row.upperLimit = 1e10f;
    }
}

// setupFrictionConstraint transcription (desiredVelocity = 0, cfmSlip = 0,
// friction warm starting disabled per setFrictionConstraintImpulse).
__device__ inline void setup_friction_row(
    ContactRow& row,
    SolverBody* bodies,
    int bodyAIdx, int bodyBIdx,
    Vec3 normalAxis,
    Vec3 rel_pos1, Vec3 rel_pos2,
    float combinedFriction,
    int frictionIndex
) {
    SolverBody& bodyA = bodies[bodyAIdx];
    SolverBody* bodyB = (bodyBIdx >= 0) ? &bodies[bodyBIdx] : nullptr;

    row.bodyA = bodyAIdx;
    row.bodyB = bodyBIdx;
    row.isSpecial = false;
    row.friction = combinedFriction;
    row.frictionIndex = frictionIndex;

    row.appliedImpulse = 0.f;
    row.appliedPushImpulse = 0.f;
    row.cfm = 0.f;
    row.rhsPenetration = 0.f;

    row.contactNormal1 = normalAxis;
    Vec3 ftorqueAxis1 = v3_cross(rel_pos1, row.contactNormal1);
    row.relpos1CrossNormal = ftorqueAxis1;
    row.angularComponentA = mat3_mul(bodyA.invInertiaWorld, ftorqueAxis1);

    if (bodyB) {
        row.contactNormal2 = -normalAxis;
        Vec3 ftorqueAxis2 = v3_cross(rel_pos2, row.contactNormal2);
        row.relpos2CrossNormal = ftorqueAxis2;
        row.angularComponentB = mat3_mul(bodyB->invInertiaWorld, ftorqueAxis2);
    } else {
        row.contactNormal2 = v3_zero();
        row.relpos2CrossNormal = v3_zero();
        row.angularComponentB = v3_zero();
    }

    {
        float denom0 = 0.f, denom1 = 0.f;
        {
            Vec3 vec = v3_cross(row.angularComponentA, rel_pos1);
            denom0 = bodyA.invMass + v3_dot(normalAxis, vec);
        }
        if (bodyB) {
            Vec3 vec = v3_cross(-row.angularComponentB, rel_pos2);
            denom1 = bodyB->invMass + v3_dot(normalAxis, vec);
        }
        float denom = 1.f / (denom0 + denom1);  // relaxation = 1
        row.jacDiagABInv = denom;
    }

    {
        float vel1Dotn = v3_dot(row.contactNormal1, bodyA.linVel + bodyA.extForceImpulse)
                       + v3_dot(row.relpos1CrossNormal, bodyA.angVel + bodyA.extTorqueImpulse);
        float vel2Dotn = bodyB
            ? (v3_dot(row.contactNormal2, bodyB->linVel + bodyB->extForceImpulse)
               + v3_dot(row.relpos2CrossNormal, bodyB->angVel + bodyB->extTorqueImpulse))
            : 0.f;
        float rel_vel = vel1Dotn + vel2Dotn;

        float velocityError = 0.f - rel_vel;  // desiredVelocity = 0
        float velocityImpulse = velocityError * row.jacDiagABInv;
        row.rhs = velocityImpulse;
        row.lowerLimit = -row.friction;
        row.upperLimit = row.friction;
    }
}

// Friction direction selection from convertContactInner:
// dir = relative velocity at contact minus normal component, normalized;
// fallback to btPlaneSpace1 first axis when no lateral velocity.
__device__ inline Vec3 friction_direction(
    SolverBody* bodies, int bodyAIdx, int bodyBIdx,
    Vec3 normalWorldOnB, Vec3 rel_pos1, Vec3 rel_pos2
) {
    SolverBody& bodyA = bodies[bodyAIdx];
    Vec3 vel1 = body_vel_at_point_no_delta(bodyA, rel_pos1);
    Vec3 vel2 = (bodyBIdx >= 0) ? body_vel_at_point_no_delta(bodies[bodyBIdx], rel_pos2) : v3_zero();
    Vec3 vel = vel1 - vel2;
    float rel_vel = v3_dot(normalWorldOnB, vel);

    Vec3 lateralFrictionDir = vel - normalWorldOnB * rel_vel;
    float lat_rel_vel = v3_length_sq(lateralFrictionDir);
    if (lat_rel_vel > SIMD_EPSILON_F) {
        return lateralFrictionDir * (1.f / sqrtf(lat_rel_vel));
    } else {
        Vec3 p, q;
        bt_plane_space1(normalWorldOnB, p, q);
        return p;
    }
}

// ----------------------------------------------------------------------------
// Row solvers (scalar reference transcriptions)
// ----------------------------------------------------------------------------

// gResolveSingleConstraintRowGeneric_scalar_reference
__device__ inline float solve_row_generic(SolverBody& bodyA, SolverBody& bodyB, ContactRow& c) {
    float deltaImpulse = c.rhs - c.appliedImpulse * c.cfm;
    float deltaVel1Dotn = v3_dot(c.contactNormal1, bodyA.dLin) + v3_dot(c.relpos1CrossNormal, bodyA.dAng);
    float deltaVel2Dotn = v3_dot(c.contactNormal2, bodyB.dLin) + v3_dot(c.relpos2CrossNormal, bodyB.dAng);

    deltaImpulse -= deltaVel1Dotn * c.jacDiagABInv;
    deltaImpulse -= deltaVel2Dotn * c.jacDiagABInv;
    float sum = c.appliedImpulse + deltaImpulse;
    if (sum < c.lowerLimit) {
        deltaImpulse = c.lowerLimit - c.appliedImpulse;
        c.appliedImpulse = c.lowerLimit;
    } else if (sum > c.upperLimit) {
        deltaImpulse = c.upperLimit - c.appliedImpulse;
        c.appliedImpulse = c.upperLimit;
    } else {
        c.appliedImpulse = sum;
    }

    body_apply_impulse(bodyA, c.contactNormal1 * bodyA.invMass, c.angularComponentA, deltaImpulse);
    body_apply_impulse(bodyB, c.contactNormal2 * bodyB.invMass, c.angularComponentB, deltaImpulse);

    return deltaImpulse * (1.f / c.jacDiagABInv);
}

// gResolveSingleConstraintRowLowerLimit_scalar_reference
__device__ inline float solve_row_lower_limit(SolverBody& bodyA, SolverBody& bodyB, ContactRow& c) {
    float deltaImpulse = c.rhs - c.appliedImpulse * c.cfm;
    float deltaVel1Dotn = v3_dot(c.contactNormal1, bodyA.dLin) + v3_dot(c.relpos1CrossNormal, bodyA.dAng);
    float deltaVel2Dotn = v3_dot(c.contactNormal2, bodyB.dLin) + v3_dot(c.relpos2CrossNormal, bodyB.dAng);

    deltaImpulse -= deltaVel1Dotn * c.jacDiagABInv;
    deltaImpulse -= deltaVel2Dotn * c.jacDiagABInv;
    float sum = c.appliedImpulse + deltaImpulse;
    if (sum < c.lowerLimit) {
        deltaImpulse = c.lowerLimit - c.appliedImpulse;
        c.appliedImpulse = c.lowerLimit;
    } else {
        c.appliedImpulse = sum;
    }
    body_apply_impulse(bodyA, c.contactNormal1 * bodyA.invMass, c.angularComponentA, deltaImpulse);
    body_apply_impulse(bodyB, c.contactNormal2 * bodyB.invMass, c.angularComponentB, deltaImpulse);

    return deltaImpulse * (1.f / c.jacDiagABInv);
}

// gResolveSplitPenetrationImpulse_scalar_reference
__device__ inline float solve_split_penetration(SolverBody& bodyA, SolverBody& bodyB, ContactRow& c) {
    float deltaImpulse = 0.f;

    if (c.rhsPenetration != 0.f) {
        deltaImpulse = c.rhsPenetration - c.appliedPushImpulse * c.cfm;
        float deltaVel1Dotn = v3_dot(c.contactNormal1, bodyA.pushVel) + v3_dot(c.relpos1CrossNormal, bodyA.turnVel);
        float deltaVel2Dotn = v3_dot(c.contactNormal2, bodyB.pushVel) + v3_dot(c.relpos2CrossNormal, bodyB.turnVel);

        deltaImpulse -= deltaVel1Dotn * c.jacDiagABInv;
        deltaImpulse -= deltaVel2Dotn * c.jacDiagABInv;
        float sum = c.appliedPushImpulse + deltaImpulse;
        if (sum < c.lowerLimit) {
            deltaImpulse = c.lowerLimit - c.appliedPushImpulse;
            c.appliedPushImpulse = c.lowerLimit;
        } else {
            c.appliedPushImpulse = sum;
        }
        body_apply_push_impulse(bodyA, c.contactNormal1 * bodyA.invMass, c.angularComponentA, deltaImpulse);
        body_apply_push_impulse(bodyB, c.contactNormal2 * bodyB.invMass, c.angularComponentB, deltaImpulse);
    }
    return deltaImpulse * (1.f / c.jacDiagABInv);
}

// ----------------------------------------------------------------------------
// Solve loop. `bodies[numBodies]` must include a trailing FIXED body at index
// `fixedBodyIdx` (zero invMass, zero inertia) used for static contacts.
// Rows reference it via bodyB == -1; we remap here.
// ----------------------------------------------------------------------------
struct SolverContext {
    SolverBody* bodies;
    int fixedBodyIdx;
    ContactRow* contactRows;
    int numContactRows;
    ContactRow* frictionRows;
    int numFrictionRows;
};

__device__ inline void solve_group(SolverContext& ctx, float dt) {
    SolverBody* bodies = ctx.bodies;
    int fixedIdx = ctx.fixedBodyIdx;

    // --- split impulse penetration iterations (all rows, incl. special) ---
    // (solveGroupCacheFriendlySplitImpulseIterations; early-out on zero residual)
    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        float leastSquaresResidual = 0.f;
        for (int j = 0; j < ctx.numContactRows; j++) {
            ContactRow& c = ctx.contactRows[j];
            int bIdx = (c.bodyB >= 0) ? c.bodyB : fixedIdx;
            float residual = solve_split_penetration(bodies[c.bodyA], bodies[bIdx], c);
            leastSquaresResidual = fmaxf(leastSquaresResidual, residual * residual);
        }
        if (leastSquaresResidual <= 0.f || iteration >= (NUM_ITERATIONS - 1))
            break;
    }

    // --- velocity iterations ---
    // Default solver mode (no interleave): normal rows first (special rows
    // skipped — ROCKETSIM CHANGE), then all friction rows gated on the normal
    // row's current applied impulse.
    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        for (int j = 0; j < ctx.numContactRows; j++) {
            ContactRow& c = ctx.contactRows[j];
            if (c.isSpecial)
                continue;
            int bIdx = (c.bodyB >= 0) ? c.bodyB : fixedIdx;
            solve_row_lower_limit(bodies[c.bodyA], bodies[bIdx], c);
        }

        for (int j = 0; j < ctx.numFrictionRows; j++) {
            ContactRow& c = ctx.frictionRows[j];
            float totalImpulse = ctx.contactRows[c.frictionIndex].appliedImpulse;
            if (totalImpulse > 0.f) {
                c.lowerLimit = -(c.friction * totalImpulse);
                c.upperLimit = c.friction * totalImpulse;
                int bIdx = (c.bodyB >= 0) ? c.bodyB : fixedIdx;
                solve_row_generic(bodies[c.bodyA], bodies[bIdx], c);
            }
        }
    }
}

// Writeback for one body (btSolverBody::writebackVelocityAndTransform +
// solveGroupCacheFriendlyFinish): returns final velocities and applies the
// split-impulse position/orientation correction.
//  - linVelOut/angVelOut are in BT units.
//  - posInOut in BT units; basisInOut updated unless noRot.
__device__ inline void body_writeback(
    SolverBody& b, float dt,
    Vec3& linVelOut, Vec3& angVelOut,
    Vec3& posInOut, RotMat& basisInOut, bool noRot
) {
    Vec3 linVel = b.linVel + b.dLin;
    Vec3 angVel = b.angVel + b.dAng;

    if (b.pushVel.x != 0.f || b.pushVel.y != 0.f || b.pushVel.z != 0.f ||
        b.turnVel.x != 0.f || b.turnVel.y != 0.f || b.turnVel.z != 0.f) {
        // integrateTransform(world, pushVel, turnVel * splitImpulseTurnErp, dt)
        posInOut += b.pushVel * dt;
        if (!noRot) {
            basisInOut = bt_integrate_rotation(basisInOut, b.turnVel * SPLIT_IMPULSE_TURN_ERP, dt);
        }
    }

    linVelOut = linVel + b.extForceImpulse;
    angVelOut = angVel + b.extTorqueImpulse;
}

// resolveSingleCollision (btContactConstraint.cpp) — used by the suspension
// "extra pushback" path in btVehicleRL::rayCast. Returns the normal impulse,
// does NOT apply it (applyImpulses=false in btVehicleRL).
// distance: wheelTraceDistDelta (negative when compressed past limit), BT.
__device__ inline float resolve_single_collision_no_apply(
    Vec3 linVel, Vec3 angVel, float invMass, const Mat3& invInertiaWorld,
    Vec3 bodyPos,
    Vec3 contactPositionWorld, Vec3 contactNormalOnB,
    float distance, float erp, float invTimeStep
) {
    Vec3 rel_pos1 = contactPositionWorld - bodyPos;

    Vec3 vel1 = linVel + v3_cross(angVel, rel_pos1);
    Vec3 vel = vel1;  // body2 static
    float rel_vel = v3_dot(contactNormalOnB, vel);

    float combinedRestitution = 0.f;
    float restitution = combinedRestitution * -rel_vel;

    float positionalError = erp * -distance * invTimeStep;
    float velocityError = -(1.0f + restitution) * rel_vel;

    // computeImpulseDenominator(pos, normal):
    // r0 = pos - center; c0 = r0 x n; vec = (invIW * c0) x r0; denom = invM + n.vec
    float denom0;
    {
        Vec3 r0 = rel_pos1;
        Vec3 c0 = v3_cross(r0, contactNormalOnB);
        Vec3 vec = v3_cross(mat3_mul(invInertiaWorld, c0), r0);
        denom0 = invMass + v3_dot(contactNormalOnB, vec);
    }
    float jacDiagABInv = 1.f / denom0;  // relaxation = 1

    float penetrationImpulse = positionalError * jacDiagABInv;
    float velocityImpulse = velocityError * jacDiagABInv;

    float normalImpulse = penetrationImpulse + velocityImpulse;
    normalImpulse = (0.f > normalImpulse) ? 0.f : normalImpulse;
    return normalImpulse;
}

// resolveSingleBilateral (btContactConstraint.cpp) — lateral wheel friction.
// Body2 is the static ground. All BT units.
__device__ inline float resolve_single_bilateral_static(
    Vec3 linVel, Vec3 angVel, float invMass, Vec3 invInertiaDiagLocal, const RotMat& basis,
    Vec3 bodyPos, Vec3 contactPos, Vec3 normal
) {
    float normalLenSqr = v3_length_sq(normal);
    if (normalLenSqr > 1.1f)
        return 0.f;

    Vec3 rel_pos1 = contactPos - bodyPos;
    Vec3 vel1 = linVel + v3_cross(angVel, rel_pos1);
    // vel2 = 0 (static)
    Vec3 vel = vel1;

    // btJacobianEntry jac(world2A = basis^T, world2B, rel_pos1, rel_pos2, normal,
    //                     invInertiaA_localDiag, invMassA, invInertiaB(0), invMassB(0))
    //  m_aJ = world2A * (rel_pos1 x normal)
    //  m_0MinvJt = invInertiaA_local * m_aJ
    //  diag = invMassA + m_0MinvJt.dot(m_aJ)   [B terms are zero]
    Vec3 aJ = rotmat_dot_vec(basis, v3_cross(rel_pos1, normal));  // basis^T * v
    Vec3 minvJt = {invInertiaDiagLocal.x * aJ.x, invInertiaDiagLocal.y * aJ.y, invInertiaDiagLocal.z * aJ.z};
    float jacDiagAB = invMass + v3_dot(minvJt, aJ);
    float jacDiagABInv = 1.f / jacDiagAB;

    float rel_vel = v3_dot(normal, vel);

    float contactDamping = 0.2f;
    float velocityImpulse = -contactDamping * rel_vel * jacDiagABInv;
    return velocityImpulse;
}

}  // namespace bts
}  // namespace rsc
