#pragma once

#include "CudaMath.cuh"
#include "BulletSolver.cuh"
#include "MeshCollision.cuh"

namespace rsc {

// bt_safe_normalize — btVector3::safeNormalize transcription. Differs from
// v3_safe_normalize: falls back to (1,0,0) instead of zero, and the epsilon
// check matches bullet (l2 >= SIMD_EPSILON*SIMD_EPSILON).
__device__ __forceinline__ Vec3 bt_safe_normalize(Vec3 v) {
    float l2 = v3_length_sq(v);
    if (l2 >= bts::SIMD_EPSILON_F * bts::SIMD_EPSILON_F) {
        return v / sqrtf(l2);
    }
    return v3(1.f, 0.f, 0.f);
}

__device__ __forceinline__ Vec3 car_forward(const GpuCarState& car) { return car.rotMat.forward; }
__device__ __forceinline__ Vec3 car_right(const GpuCarState& car)   { return car.rotMat.right; }
__device__ __forceinline__ Vec3 car_up(const GpuCarState& car)      { return car.rotMat.up; }


__device__ __forceinline__ float car_forward_speed(const GpuCarState& car) {
    return v3_dot(car.vel, car.rotMat.forward);
}



__device__ __forceinline__ void car_compute_inertia(GpuCarState& car) {
    // Matches btBoxShape + btRigidBody exactly:
    //   ctor: m_implicitShapeDimensions = halfExtents - 0.04 (CONVEX_DISTANCE_MARGIN)
    //   then setSafeMargin: margin = min(0.04, 0.1 * minHalfExtent)
    //   calculateLocalInertia uses getHalfExtentsWithMargin() = implicit + margin
    float hx = car.config.hitboxSize.x * 0.5f / 50.f; // UU -> BT
    float hy = car.config.hitboxSize.y * 0.5f / 50.f;
    float hz = car.config.hitboxSize.z * 0.5f / 50.f;

    constexpr float CONVEX_DISTANCE_MARGIN = 0.04f;
    float minHalf = fminf(hx, fminf(hy, hz));
    float safeMargin = fminf(CONVEX_DISTANCE_MARGIN, 0.1f * minHalf);
    float adjust = -CONVEX_DISTANCE_MARGIN + safeMargin;
    hx += adjust;
    hy += adjust;
    hz += adjust;

    float m = PhysConst::CAR_MASS;
    float lx = 2.f * hx, ly = 2.f * hy, lz = 2.f * hz;
    car.localInertia = {
        m / 12.f * (ly*ly + lz*lz),
        m / 12.f * (lx*lx + lz*lz),
        m / 12.f * (lx*lx + ly*ly)
    };
    car.invLocalInertia = {
        1.f / car.localInertia.x,
        1.f / car.localInertia.y,
        1.f / car.localInertia.z
    };
}

// World-space inverse inertia tensor: R * diag(invI) * R^T applied to vector
__device__ __forceinline__ Vec3 apply_world_inv_inertia(const GpuCarState& car, Vec3 torque) {
    // Transform to local space, apply diagonal inverse inertia, transform back
    Vec3 local = rotmat_dot_vec(car.rotMat, torque);
    local = local * car.invLocalInertia;
    return rotmat_mul_vec(car.rotMat, local);
}



__device__ void device_wheel_raycast(
    GpuCarState& car,
    const ArenaSurface* surfaces, int numSurfaces,
    const MeshGridView& meshGrid,
    float dt
) {
    using namespace PhysConst;

    int numWheelsInContact = 0;

    for (int i = 0; i < 4; i++) {
        GpuWheelState& w = car.wheels[i];
        bool isFront = (i < 2);
        bool isLeft  = (i % 2 == 1);

        const GpuWheelPairConfig& wpCfg = isFront ? car.config.frontWheels : car.config.backWheels;
        float wheelRadius = wpCfg.wheelRadius;
        float susRestLen  = wpCfg.suspensionRestLength;

        // Wheel connection point in local space
        Vec3 connPt = wpCfg.connectionPointOffset;
        if (isLeft) connPt.y = -connPt.y;

        // Transform to world space
        Vec3 hardPointWS = car.pos + rotmat_mul_vec(car.rotMat, connPt);
        w.hardPointWS = hardPointWS;

        // Wheel direction (down in car space)
        Vec3 wheelDirWS = rotmat_mul_vec(car.rotMat, v3(0, 0, -1));

        // Total ray length
        float rayLen = susRestLen + MAX_SUSPENSION_TRAVEL + wheelRadius - SUSPENSION_SUBTRACTION;

        // Find closest surface hit
        float bestT = rayLen + 1.f; // Start beyond max
        Vec3 bestNormal = v3(0, 0, 1);
        bool hasHit = false;
        bool hitIsWorld = true;

        if (meshGrid.numTris > 0) {
            // ---- Real geometry: arena triangles + 4 static planes ----
            // (btDefaultVehicleRaycaster -> rayTest against all world bodies)
            Vec3 fromBT = hardPointWS * bts::UU_TO_BT;
            Vec3 toBT = (hardPointWS + wheelDirWS * rayLen) * bts::UU_TO_BT;

            float bestFrac = 1e30f;

            {
                Vec3 aabbMin = {fminf(fromBT.x, toBT.x), fminf(fromBT.y, toBT.y), fminf(fromBT.z, toBT.z)};
                Vec3 aabbMax = {fmaxf(fromBT.x, toBT.x), fmaxf(fromBT.y, toBT.y), fmaxf(fromBT.z, toBT.z)};

                mesh_grid_query_aabb(meshGrid, aabbMin, aabbMax, [&](int triIdx) {
                    Vec3 triNormal;
                    float frac = mesh_ray_triangle(fromBT, toBT, meshGrid.tris[triIdx], triNormal);
                    if (frac >= 0.f && frac < bestFrac) {
                        bestFrac = frac;
                        bestNormal = triNormal;
                        hasHit = true;
                    }
                });
            }

            // Static planes
            for (int s = 0; s < numSurfaces; s++) {
                if (!surfaces[s].isStaticPlane) continue;
                Vec3 n = surfaces[s].normal;
                float denom = v3_dot(wheelDirWS, n);
                if (denom > -1e-6f) continue;
                float dist = v3_dot(hardPointWS, n) - surfaces[s].offset;
                float t = -dist / denom;
                if (t > 0.f && t <= rayLen) {
                    float frac = t / rayLen;
                    if (frac < bestFrac) {
                        bestFrac = frac;
                        bestNormal = n;
                        hasHit = true;
                    }
                }
            }

            if (hasHit)
                bestT = bestFrac * rayLen;
        } else {
            // ---- Analytic fallback ----

            // Check floor (z=0)
            if (wheelDirWS.z < -1e-6f) {
                float t = -hardPointWS.z / wheelDirWS.z;
                if (t > 0.f && t < bestT && t <= rayLen) {
                    bestT = t;
                    bestNormal = v3(0, 0, 1);
                    hasHit = true;
                }
            }

            // Check arena surfaces
            for (int s = 0; s < numSurfaces; s++) {
                Vec3 n = surfaces[s].normal;
                float denom = v3_dot(wheelDirWS, n);
                if (denom > -1e-6f) continue;

                float dist = v3_dot(hardPointWS, n) - surfaces[s].offset;
                float t = -dist / denom;
                if (t > 0.f && t < bestT && t <= rayLen) {
                    // Check bounds
                    Vec3 hitPt = hardPointWS + wheelDirWS * t;
                    Vec3 bmin = surfaces[s].boundsMin;
                    Vec3 bmax = surfaces[s].boundsMax;
                    if (hitPt.x >= bmin.x && hitPt.x <= bmax.x &&
                        hitPt.y >= bmin.y && hitPt.y <= bmax.y &&
                        hitPt.z >= bmin.z && hitPt.z <= bmax.z) {
                        bestT = t;
                        bestNormal = n;
                        hasHit = true;
                    }
                }
            }
        }

        w.isInContact = hasHit;
        w.isInContactWithWorld = hasHit && hitIsWorld;
        w.contactNormal = bestNormal;

        if (hasHit) {
            w.contactPoint = hardPointWS + wheelDirWS * bestT;

            // btVehicleRL::rayCast — suspension length is the trace length
            // projected onto the chassis up axis, minus the wheel radius.
            // (Ray dir is exactly -carUp, so the projection equals bestT, but
            // we keep the reference formulation for exactness on slopes.)
            Vec3 carUpDir = car.rotMat.up;
            float wheelTraceLen = v3_dot(hardPointWS - w.contactPoint, carUpDir);
            w.suspensionLength = wheelTraceLen - wheelRadius;

            // Clamp on max suspension travel
            float minLen = susRestLen - MAX_SUSPENSION_TRAVEL;
            float maxLen = susRestLen + MAX_SUSPENSION_TRAVEL;
            w.suspensionLength = rs_clamp(w.suspensionLength, minLen, maxLen);

            float denominator = v3_dot(bestNormal, carUpDir);

            // Relative velocity at the CONTACT POINT (reference uses the
            // contact point, not the suspension hard point), in BT units.
            Vec3 relposBT = (w.contactPoint - car.pos) * bts::UU_TO_BT;
            Vec3 velAtContactBT = car.vel * bts::UU_TO_BT + v3_cross(car.angVel, relposBT);
            float projVel = v3_dot(bestNormal, velAtContactBT);

            if (denominator > 0.1f) {
                float inv = 1.f / denominator;
                w.suspensionRelVel = projVel * inv;
                w.clippedInvContactDotSuspension = inv;
            } else {
                w.suspensionRelVel = 0.f;
                w.clippedInvContactDotSuspension = 10.f;
            }

            // Extra pushback when the suspension is compressed past its limit
            // (btVehicleRL::rayCast -> resolveSingleCollision, applyImpulses
            // false, divided across all 4 wheels). Static world hits only.
            w.extraPushback = 0.f;
            if (hitIsWorld) {
                float rayPushbackThresh = (susRestLen + wheelRadius - SUSPENSION_SUBTRACTION) * bts::UU_TO_BT;
                float wheelTraceLenBT = wheelTraceLen * bts::UU_TO_BT;
                if (wheelTraceLenBT < rayPushbackThresh) {
                    float wheelTraceDistDelta = wheelTraceLenBT - rayPushbackThresh;
                    bts::Mat3 invIW = bts::make_inv_inertia_world(car.rotMat, car.invLocalInertia);
                    float collisionResult = bts::resolve_single_collision_no_apply(
                        car.vel * bts::UU_TO_BT, car.angVel,
                        1.f / CAR_MASS, invIW,
                        car.pos * bts::UU_TO_BT,
                        w.contactPoint * bts::UU_TO_BT, bestNormal,
                        wheelTraceDistDelta,
                        0.2f /* solverInfo.m_erp default */, 1.f / dt);
                    w.extraPushback = collisionResult / 4.f;
                }
            }

            numWheelsInContact++;
        } else {
            w.suspensionLength = susRestLen + MAX_SUSPENSION_TRAVEL;
            w.suspensionRelVel = 0.f;
            w.clippedInvContactDotSuspension = 1.f;
            w.contactPoint = hardPointWS + wheelDirWS * rayLen;
            w.extraPushback = 0.f;
        }

    }

    // Update isOnGround
    car.isOnGround = (numWheelsInContact >= 3);
}

__device__ void device_update_suspension(GpuCarState& car, float dt) {
    using namespace PhysConst;

    // btVehicleRL::updateSuspension — two passes: compute all wheel forces,
    // then apply impulses. The clipped inverse contact dot scales ONLY the
    // spring term (not the damping term).
    for (int i = 0; i < 4; i++) {
        GpuWheelState& w = car.wheels[i];
        if (!w.isInContact) {
            w.suspensionForce = 0.f;
            continue;
        }

        bool isFront = (i < 2);
        float susRestLen = isFront ? car.config.frontWheels.suspensionRestLength
                                   : car.config.backWheels.suspensionRestLength;
        float forceScale = isFront ? SUSPENSION_FORCE_SCALE_FRONT : SUSPENSION_FORCE_SCALE_BACK;

        float force = (susRestLen - w.suspensionLength) * bts::UU_TO_BT
                    * SUSPENSION_STIFFNESS * w.clippedInvContactDotSuspension;

        float dampingVelScale = (w.suspensionRelVel < 0.f)
            ? WHEELS_DAMPING_COMPRESSION
            : WHEELS_DAMPING_RELAXATION;

        float totalForce = force - (dampingVelScale * w.suspensionRelVel);
        totalForce *= forceScale;

        // RL never uses downwards suspension forces
        if (totalForce < 0.f) totalForce = 0.f;

        w.suspensionForce = totalForce;
    }

    bts::Mat3 invIW = bts::make_inv_inertia_world(car.rotMat, car.invLocalInertia);
    for (int i = 0; i < 4; i++) {
        GpuWheelState& w = car.wheels[i];
        if (w.suspensionForce != 0.f) {
            Vec3 contactPointOffsetBT = (w.contactPoint - car.pos) * bts::UU_TO_BT;
            float baseForceScale = (w.suspensionForce * dt) + w.extraPushback;
            Vec3 impulseBT = w.contactNormal * baseForceScale;

            // btRigidBody::applyImpulse(impulse, rel_pos)
            car.vel += impulseBT * (1.f / CAR_MASS) * bts::BT_TO_UU;
            Vec3 angImpulse = v3_cross(contactPointOffsetBT, impulseBT);
            car.angVel += bts::mat3_mul(invIW, angImpulse);
        }
    }
}



__device__ void device_calc_friction_impulses(GpuCarState& car, float dt) {
    using namespace PhysConst;

    // btVehicleRL::calcFrictionImpulses — all in BT units.
    float frictionScale = CAR_MASS / 3.f;

    Vec3 carPosBT = car.pos * bts::UU_TO_BT;
    Vec3 carVelBT = car.vel * bts::UU_TO_BT;

    for (int i = 0; i < 4; i++) {
        GpuWheelState& w = car.wheels[i];
        if (!w.isInContact) {
            w.frictionImpulse = v3_zero();
            continue;
        }

        // Axle direction (includes steering turn) — column(right) of the
        // wheel's world transform: rotate chassis right by steerAngle about
        // the chassis up axis.
        Vec3 carForward = car.rotMat.forward;
        Vec3 carRight = car.rotMat.right;
        float sa = sinf(w.steerAngle), ca = cosf(w.steerAngle);
        Vec3 axleDir = carRight * ca - carForward * sa;

        Vec3 surfNormalWS = w.contactNormal;
        float proj = v3_dot(axleDir, surfNormalWS);
        axleDir -= surfNormalWS * proj;
        axleDir = bt_safe_normalize(axleDir);

        // Wheel forwards direction
        Vec3 forwardDir = bt_safe_normalize(v3_cross(surfNormalWS, axleDir));

        Vec3 contactPointBT = w.contactPoint * bts::UU_TO_BT;

        // Sideways friction impulse: full resolveSingleBilateral with the
        // bullet jacobian (NOT a simple velocity cancel).
        float sideImpulse = bts::resolve_single_bilateral_static(
            carVelBT, car.angVel, 1.f / CAR_MASS, car.invLocalInertia, car.rotMat,
            carPosBT, contactPointBT, axleDir);

        float rollingFriction;
        if (w.engineForce == 0.f) {
            if (w.brake != 0.f) {
                // Simplified variation of calcRollingFriction()
                Vec3 carRelContactPoint = contactPointBT - carPosBT;
                Vec3 v1 = carVelBT + v3_cross(car.angVel, carRelContactPoint);
                float relVel = v3_dot(v1, forwardDir);

                if (dt > (1.f / 80.f)) {
                    float threshold = -(1.f / (dt * 150.f)) + 0.8f;
                    if (rs_abs(relVel) < threshold)
                        relVel = 0.f;
                }

                constexpr float ROLLING_FRICTION_SCALE_MAGIC = 113.73963f;
                rollingFriction = rs_clamp(-relVel * ROLLING_FRICTION_SCALE_MAGIC, -w.brake, w.brake);
            } else {
                // Don't apply friction when driving with no brake
                rollingFriction = 0.f;
            }
        } else {
            // Engine force already accounts for our mass
            rollingFriction = -w.engineForce / frictionScale;
        }

        Vec3 totalFrictionForce = forwardDir * (rollingFriction * w.longFriction)
                                + axleDir * (sideImpulse * w.latFriction);
        w.frictionImpulse = totalFrictionForce * frictionScale;
    }
}


__device__ void device_apply_friction_impulses(GpuCarState& car, float dt) {
    using namespace PhysConst;

    // btVehicleRL::applyFrictionImpulses — impulse applied at the contact
    // offset projected off the chassis up axis. BT units.
    Vec3 upDir = car.rotMat.up;
    bts::Mat3 invIW = bts::make_inv_inertia_world(car.rotMat, car.invLocalInertia);

    for (int i = 0; i < 4; i++) {
        GpuWheelState& w = car.wheels[i];
        if (v3_is_zero(w.frictionImpulse)) continue;

        Vec3 wheelContactOffsetBT = (w.contactPoint - car.pos) * bts::UU_TO_BT;
        float contactUpDot = v3_dot(upDir, wheelContactOffsetBT);
        Vec3 wheelRelPosBT = wheelContactOffsetBT - upDir * contactUpDot;

        Vec3 impulseBT = w.frictionImpulse * dt;
        car.vel += impulseBT * (1.f / CAR_MASS) * bts::BT_TO_UU;
        Vec3 angImpulse = v3_cross(wheelRelPosBT, impulseBT);
        car.angVel += bts::mat3_mul(invIW, angImpulse);
    }
}



// `accelAccum` accumulates central forces as accelerations (UU/s^2); they are
// integrated by the solver as Bullet's externalForceImpulse (i.e. applied to
// the velocity only after contact solving, like applyCentralForce).
__device__ void device_update_wheels(GpuCarState& car, float dt, Vec3& accelAccum) {
    using namespace PhysConst;

    float forwardSpeed = car_forward_speed(car);
    float absForwardSpeed = rs_abs(forwardSpeed);

    int numWheelsInContact = 0;
    bool wheelsHaveWorldContact = false;
    for (int i = 0; i < 4; i++) {
        numWheelsInContact += car.wheels[i].isInContact ? 1 : 0;
        wheelsHaveWorldContact |= car.wheels[i].isInContactWithWorld;
    }

    // Handbrake rise/fall
    if (car.controls.handbrake) {
        car.handbrakeVal += POWERSLIDE_RISE_RATE * dt;
    } else {
        car.handbrakeVal -= POWERSLIDE_FALL_RATE * dt;
    }
    car.handbrakeVal = rs_clamp(car.handbrakeVal, 0.f, 1.f);

    // Compute real throttle and brake
    float realThrottle = car.controls.throttle;
    float realBrake = 0.f;

    if (car.controls.boost && car.boost > 0.f)
        realThrottle = 1.f;

    float engineThrottle = realThrottle;

    if (car.controls.handbrake) {
        // realThrottle unchanged when powersliding
    } else {
        float absThrottle = rs_abs(realThrottle);
        if (absThrottle >= THROTTLE_DEADZONE) {
            if (absForwardSpeed > STOPPING_FORWARD_VEL &&
                rs_sign(realThrottle) != rs_sign(forwardSpeed)) {
                realBrake = 1.f;
                if (absForwardSpeed > BRAKING_NO_THROTTLE_SPEED_THRESH)
                    engineThrottle = 0.f;
            }
        } else {
            engineThrottle = 0.f;
            bool shouldFullStop = (absForwardSpeed < STOPPING_FORWARD_VEL);
            realBrake = shouldFullStop ? 1.f : COASTING_BRAKE_FACTOR;
        }
    }

    // Drive speed scale
    float driveSpeedScale = curve_drive_torque(absForwardSpeed);
    if (numWheelsInContact < 3)
        driveSpeedScale /= 4.f;

    // Engine and brake forces (in BT-like units for consistency with friction calc)
    float driveEngineForce = engineThrottle * (THROTTLE_TORQUE_AMOUNT / 50.f) * driveSpeedScale;
    float driveBrakeForce = realBrake * (BRAKE_TORQUE_AMOUNT / 50.f);

    for (int i = 0; i < 4; i++) {
        car.wheels[i].engineForce = driveEngineForce;
        car.wheels[i].brake = driveBrakeForce;
    }

    // The wheel world transforms are computed at the START of the vehicle
    // update (btVehicleRL::updateVehicleFirst) from the PREVIOUS tick's steer
    // angle; Car::_UpdateWheels' friction block reads those stale transforms.
    float prevSteerAngle[4];
    for (int i = 0; i < 4; i++)
        prevSteerAngle[i] = car.wheels[i].steerAngle;

    // Steering
    float steerAngle = curve_steer_angle(absForwardSpeed);
    if (car.handbrakeVal > 0.f) {
        float psAngle = curve_powerslide_steer(absForwardSpeed);
        steerAngle += (psAngle - steerAngle) * car.handbrakeVal;
    }
    steerAngle *= car.controls.steer;
    car.wheels[0].steerAngle = steerAngle;
    car.wheels[1].steerAngle = steerAngle;

    // Friction per wheel
    for (int i = 0; i < 4; i++) {
        GpuWheelState& w = car.wheels[i];
        if (!w.isInContact) continue;

        Vec3 latDir, longDir;
        {
            Vec3 carFwd = car.rotMat.forward;
            Vec3 carRt = car.rotMat.right;
            float sa = sinf(prevSteerAngle[i]), ca = cosf(prevSteerAngle[i]);
            Vec3 wheelAxle = carRt * ca - carFwd * sa;
            latDir = wheelAxle - w.contactNormal * v3_dot(wheelAxle, w.contactNormal);
            latDir = v3_safe_normalize(latDir);
            longDir = v3_cross(w.contactNormal, latDir);
        }

        Vec3 velAtContact = car.vel + v3_cross(car.angVel, w.hardPointWS - car.pos);
        Vec3 crossVec = velAtContact; // Already in UU/s

        float baseFriction = rs_abs(v3_dot(crossVec, latDir));
        float frictionCurveInput = 0.f;
        if (baseFriction > 5.f) {
            frictionCurveInput = baseFriction / (rs_abs(v3_dot(crossVec, longDir)) + baseFriction);
        }

        float latFriction = curve_lat_friction(frictionCurveInput);
        float longFriction = curve_long_friction(frictionCurveInput);

        if (car.handbrakeVal > 0.f) {
            float hbAmount = car.handbrakeVal;
            latFriction *= (curve_handbrake_lat(frictionCurveInput) - 1.f) * hbAmount + 1.f;
            longFriction *= (curve_handbrake_long(frictionCurveInput) - 1.f) * hbAmount + 1.f;
        } else {
            longFriction = 1.f;
        }

        bool isContactSticky = (realThrottle != 0.f);
        if (!isContactSticky) {
            float nonStickyScale = curve_non_sticky_friction(w.contactNormal.z);
            latFriction *= nonStickyScale;
            longFriction *= nonStickyScale;
        }

        w.latFriction = latFriction;
        w.longFriction = longFriction;
    }

    // Sticky forces
    if (wheelsHaveWorldContact) {
        // btVehicleRL::getUpwardsDirFromWheelContacts — sums normals of ALL
        // wheels in contact (any object), bullet safeNormalized.
        Vec3 upwardsDir = v3_zero();
        for (int i = 0; i < 4; i++) {
            if (car.wheels[i].isInContact)
                upwardsDir += car.wheels[i].contactNormal;
        }
        if (v3_is_zero(upwardsDir)) {
            upwardsDir = car.rotMat.up;
        } else {
            upwardsDir = bt_safe_normalize(upwardsDir);
        }

        bool fullStick = (realThrottle != 0.f) || (absForwardSpeed > STOPPING_FORWARD_VEL);
        float stickyForceScale = 0.5f;
        if (fullStick)
            stickyForceScale += 1.f - rs_abs(upwardsDir.z);

        // applyCentralForce(upwardsDir * scale * GRAVITY_Z * mass) — accumulate
        // as acceleration, integrated by the solver writeback.
        accelAccum += upwardsDir * (stickyForceScale * GRAVITY_Z);
    }
}


__device__ void device_update_jump(GpuCarState& car, float dt, bool jumpPressed, Vec3& accelAccum) {
    using namespace PhysConst;

    if (car.isOnGround && !car.isJumping) {
        if (car.hasJumped && car.jumpTime < JUMP_MIN_TIME + JUMP_RESET_TIME_PAD) {
            // Don't reset jump yet
        } else {
            car.hasJumped = false;
            car.jumpTime = 0.f;
        }
    }

    if (car.isJumping) {
        if (car.jumpTime < JUMP_MIN_TIME || (car.controls.jump && car.jumpTime < JUMP_MAX_TIME)) {
            car.isJumping = true;
        } else {
            car.isJumping = false;
        }
    } else if (car.isOnGround && jumpPressed) {
        car.isJumping = true;
        car.jumpTime = 0.f;
        // applyCentralImpulse — immediate velocity change
        car.vel += car_up(car) * JUMP_IMMEDIATE_FORCE;
    }

    if (car.isJumping) {
        car.hasJumped = true;

        // Sustained jump force — applyCentralForce, integrated by the solver.
        Vec3 jumpForce = car_up(car) * JUMP_ACCEL;
        if (car.jumpTime < JUMP_MIN_TIME)
            jumpForce *= JUMP_PRE_MIN_ACCEL_SCALE;

        accelAccum += jumpForce;
    }

    if (car.isJumping || car.hasJumped)
        car.jumpTime += dt;
}


// `angAccelAccum` accumulates torques as angular accelerations (rad/s^2);
// applied by the solver as Bullet's externalTorqueImpulse (applyTorque with
// the inverse-inertia round trip cancelled out).
__device__ void device_update_air_torque(GpuCarState& car, float dt, bool updateAirControl,
                                         Vec3& accelAccum, Vec3& angAccelAccum) {
    using namespace PhysConst;

    Vec3 dirPitch = -car_right(car);
    Vec3 dirYaw   = car_up(car);
    Vec3 dirRoll  = -car_forward(car);

    bool doAirControl = false;

    if (car.isFlipping)
        car.isFlipping = car.hasFlipped && car.flipTime < FLIP_TORQUE_TIME;

    if (car.isFlipping) {
        Vec3 relDodgeTorque = car.flipRelTorque;

        if (!v3_is_zero(relDodgeTorque)) {
            float pitchScale = 1.f;
            if (relDodgeTorque.y != 0.f && car.controls.pitch != 0.f) {
                if (rs_sign(relDodgeTorque.y) == rs_sign(car.controls.pitch)) {
                    pitchScale = 1.f - rs_clamp(rs_abs(car.controls.pitch), 0.f, 1.f);
                    doAirControl = true;
                }
            }
            relDodgeTorque.y *= pitchScale;

            // applyTorque(invInertiaWorld.inverse() * basis * dodgeTorque)
            Vec3 dodgeTorque = relDodgeTorque * v3(FLIP_TORQUE_X, FLIP_TORQUE_Y, 0.f);
            angAccelAccum += rotmat_mul_vec(car.rotMat, dodgeTorque);
        } else {
            doAirControl = true; // Stall
        }
    } else {
        doAirControl = true;
    }

    doAirControl &= !car.isAutoFlipping;
    doAirControl &= updateAirControl;

    if (doAirControl) {
        float pitchTorqueScale = 1.f;
        Vec3 torque = v3_zero();

        if (car.controls.pitch != 0.f || car.controls.yaw != 0.f || car.controls.roll != 0.f) {
            if (car.isFlipping) {
                pitchTorqueScale = 0.f;
            } else if (car.hasFlipped) {
                if (car.flipTime < FLIP_TORQUE_TIME + FLIP_PITCHLOCK_EXTRA_TIME)
                    pitchTorqueScale = 0.f;
            }

            torque = dirPitch * (car.controls.pitch * pitchTorqueScale * AIR_CONTROL_TORQUE_PITCH)
                   + dirYaw   * (car.controls.yaw * AIR_CONTROL_TORQUE_YAW)
                   + dirRoll  * (car.controls.roll * AIR_CONTROL_TORQUE_ROLL);
        }

        // Damping
        Vec3 angVel = car.angVel;
        float dampPitch = v3_dot(dirPitch, angVel) * AIR_CONTROL_DAMPING_PITCH
                        * (1.f - rs_abs(car.controls.pitch * pitchTorqueScale));
        float dampYaw   = v3_dot(dirYaw, angVel) * AIR_CONTROL_DAMPING_YAW
                        * (1.f - rs_abs(car.controls.yaw));
        float dampRoll  = v3_dot(dirRoll, angVel) * AIR_CONTROL_DAMPING_ROLL;

        Vec3 damping = dirYaw * dampYaw + dirPitch * dampPitch + dirRoll * dampRoll;

        angAccelAccum += (torque - damping) * CAR_TORQUE_SCALE;
    }

    // Air throttle — applyCentralForce
    if (car.controls.throttle != 0.f) {
        accelAccum += car_forward(car) * (car.controls.throttle * THROTTLE_AIR_ACCEL);
    }
}


__device__ void device_update_double_jump_or_flip(GpuCarState& car, float dt, bool jumpPressed, float forwardSpeed) {
    using namespace PhysConst;

    if (car.isOnGround) {
        car.hasDoubleJumped = false;
        car.hasFlipped = false;
        car.airTime = 0.f;
        car.airTimeSinceJump = 0.f;
        car.flipTime = 0.f;
    } else {
        car.airTime += dt;

        if (car.hasJumped && !car.isJumping) {
            car.airTimeSinceJump += dt;
        } else {
            car.airTimeSinceJump = 0.f;
        }

        if (jumpPressed && car.airTimeSinceJump < DOUBLEJUMP_MAX_DELAY) {
            float inputMag = rs_abs(car.controls.yaw) + rs_abs(car.controls.pitch) + rs_abs(car.controls.roll);
            bool isFlipInput = (inputMag >= car.config.dodgeDeadzone);

            bool canUse = (!car.hasDoubleJumped && !car.hasFlipped);
            if (car.isAutoFlipping) canUse = false;

            if (canUse) {
                if (isFlipInput) {
                    // Begin flip
                    car.flipTime = 0.f;
                    car.hasFlipped = true;
                    car.isFlipping = true;

                    float forwardSpeedRatio = rs_abs(forwardSpeed) / CAR_MAX_SPEED;

                    Vec3 dodgeDir = v3(-car.controls.pitch,
                                       car.controls.yaw + car.controls.roll, 0.f);

                    if (rs_abs(dodgeDir.x) < 0.1f && rs_abs(dodgeDir.y) < 0.1f) {
                        dodgeDir = v3_zero();
                    } else {
                        dodgeDir = v3_safe_normalize(dodgeDir);
                    }

                    car.flipRelTorque = v3(-dodgeDir.y, dodgeDir.x, 0.f);

                    if (rs_abs(dodgeDir.x) < 0.1f) dodgeDir.x = 0.f;
                    if (rs_abs(dodgeDir.y) < 0.1f) dodgeDir.y = 0.f;

                    if (!v3_is_zero(dodgeDir)) {
                        bool shouldDodgeBackwards;
                        if (rs_abs(forwardSpeed) < 100.f) {
                            shouldDodgeBackwards = (dodgeDir.x < 0.f);
                        } else {
                            shouldDodgeBackwards = ((dodgeDir.x >= 0.f) != (forwardSpeed >= 0.f));
                        }

                        Vec3 initDodgeVel = dodgeDir * FLIP_INITIAL_VEL_SCALE;

                        float maxSpeedScaleX = shouldDodgeBackwards
                            ? FLIP_BACKWARD_IMPULSE_MAX_SPEED_SCALE
                            : FLIP_FORWARD_IMPULSE_MAX_SPEED_SCALE;

                        initDodgeVel.x *= (maxSpeedScaleX - 1.f) * forwardSpeedRatio + 1.f;
                        initDodgeVel.y *= (FLIP_SIDE_IMPULSE_MAX_SPEED_SCALE - 1.f) * forwardSpeedRatio + 1.f;

                        if (shouldDodgeBackwards)
                            initDodgeVel.x *= FLIP_BACKWARD_IMPULSE_SCALE_X;

                        Vec3 fwd = car_forward(car);
                        float fwdAng = atan2f(fwd.y, fwd.x);

                        Vec3 xVelDir = v3(cosf(fwdAng), -sinf(fwdAng), 0.f);
                        Vec3 yVelDir = v3(sinf(fwdAng),  cosf(fwdAng), 0.f);

                        Vec3 finalDeltaVel = v3(
                            v3_dot(initDodgeVel, xVelDir),
                            v3_dot(initDodgeVel, yVelDir),
                            0.f
                        );

                        car.vel += finalDeltaVel;
                    }
                } else {
                    // Double jump
                    car.vel += car_up(car) * JUMP_IMMEDIATE_FORCE;
                    car.hasDoubleJumped = true;
                }
            }
        }
    }

    // Flip Z damping
    if (car.isFlipping) {
        car.flipTime += dt;
        if (car.flipTime <= FLIP_TORQUE_TIME) {
            if (car.flipTime >= FLIP_Z_DAMP_START &&
                (car.vel.z < 0.f || car.flipTime < FLIP_Z_DAMP_END)) {
                car.vel.z *= powf(1.f - FLIP_Z_DAMP_120, dt / (1.f / 120.f));
            }
        }
    } else if (car.hasFlipped) {
        car.flipTime += dt;
    }
}

__device__ void device_update_autoflip(GpuCarState& car, float dt, bool jumpPressed) {
    using namespace PhysConst;

    if (jumpPressed && car.worldContactHasContact &&
        car.worldContactNormal.z > CAR_AUTOFLIP_NORMZ_THRESH) {

        EulerAngles angles = rotmat_to_euler(car.rotMat);
        float absRoll = rs_abs(angles.roll);

        if (absRoll > CAR_AUTOFLIP_ROLL_THRESH) {
            car.autoFlipTimer = CAR_AUTOFLIP_TIME * (absRoll / M_PI);
            car.autoFlipTorqueScale = (angles.roll > 0.f) ? 1.f : -1.f;
            car.isAutoFlipping = true;
            car.vel += (-car_up(car)) * CAR_AUTOFLIP_IMPULSE;
        }
    }

    if (car.isAutoFlipping) {
        if (car.autoFlipTimer <= 0.f) {
            car.isAutoFlipping = false;
            car.autoFlipTimer = 0.f;
        } else {
            car.angVel += car_forward(car) * (CAR_AUTOFLIP_TORQUE * car.autoFlipTorqueScale * dt);
            car.autoFlipTimer -= dt;
        }
    }
}



__device__ void device_update_autoroll(GpuCarState& car, float dt, int numWheelsInContact,
                                        Vec3& accelAccum, Vec3& angAccelAccum) {
    using namespace PhysConst;

    Vec3 groundUpDir;
    if (numWheelsInContact > 0) {
        // btVehicleRL::getUpwardsDirFromWheelContacts
        groundUpDir = v3_zero();
        for (int i = 0; i < 4; i++) {
            if (car.wheels[i].isInContact)
                groundUpDir += car.wheels[i].contactNormal;
        }
        if (v3_is_zero(groundUpDir)) groundUpDir = car.rotMat.up;
        else groundUpDir = bt_safe_normalize(groundUpDir);
    } else {
        groundUpDir = car.worldContactNormal;
    }

    Vec3 groundDownDir = -groundUpDir;
    Vec3 fwd = car_forward(car);
    Vec3 rt  = car_right(car);

    Vec3 crossRightDir = v3_cross(groundUpDir, fwd);
    Vec3 crossForwardDir = v3_cross(groundDownDir, crossRightDir);

    float rightTorqueFactor = 1.f - rs_clamp(v3_dot(rt, crossRightDir), 0.f, 1.f);
    float forwardTorqueFactor = 1.f - rs_clamp(v3_dot(fwd, crossForwardDir), 0.f, 1.f);

    Vec3 torqueDirRight = fwd * ((v3_dot(rt, groundUpDir) >= 0.f) ? -1.f : 1.f);
    Vec3 torqueDirForward = rt * ((v3_dot(fwd, groundUpDir) >= 0.f) ? 1.f : -1.f);

    Vec3 torqueRight = torqueDirRight * rightTorqueFactor;
    Vec3 torqueForward = torqueDirForward * forwardTorqueFactor;

    // applyCentralForce: push car toward ground
    accelAccum += groundDownDir * CAR_AUTOROLL_FORCE;

    // applyTorque(invInertiaWorld.inverse() * (tF + tR) * AUTOROLL_TORQUE)
    angAccelAccum += (torqueForward + torqueRight) * CAR_AUTOROLL_TORQUE;
}


__device__ void device_update_boost(GpuCarState& car, float dt,
                                     float boostUsedPerSecond,
                                     float boostAccelGround,
                                     float boostAccelAir,
                                     Vec3& accelAccum) {
    using namespace PhysConst;

    // Update boosting timer
    if (car.timeSpentBoosting > 0.f) {
        if (!car.controls.boost && car.timeSpentBoosting >= BOOST_MIN_TIME) {
            car.timeSpentBoosting = 0.f;
        } else {
            car.timeSpentBoosting += dt;
        }
    } else {
        if (car.controls.boost)
            car.timeSpentBoosting = dt;
    }

    // Apply boost force (applyCentralForce) and consume
    if (car.boost > 0.f && car.timeSpentBoosting > 0.f) {
        car.boost = rs_max(car.boost - boostUsedPerSecond * dt, 0.f);
        float accel = car.isOnGround ? boostAccelGround : boostAccelAir;
        accelAccum += car_forward(car) * accel;
    }

    car.boost = rs_min(car.boost, BOOST_MAX);
}


__device__ void device_car_post_tick(GpuCarState& car, float dt) {
    using namespace PhysConst;

    if (car.isDemoed) return;

    // RotMat already updated during integration

    // Supersonic
    float speedSq = v3_length_sq(car.vel);

    if (car.isSupersonic && car.supersonicTime < SUPERSONIC_MAINTAIN_MAX_TIME) {
        car.isSupersonic = (speedSq >= SUPERSONIC_MAINTAIN_MIN_SPEED * SUPERSONIC_MAINTAIN_MIN_SPEED);
    } else {
        car.isSupersonic = (speedSq >= SUPERSONIC_START_SPEED * SUPERSONIC_START_SPEED);
    }

    if (car.isSupersonic) {
        car.supersonicTime += dt;
    } else {
        car.supersonicTime = 0.f;
    }

    // Car contact cooldown
    if (car.carContactCooldownTimer > 0.f)
        car.carContactCooldownTimer = rs_max(car.carContactCooldownTimer - dt, 0.f);

    car.lastControls = car.controls;
}


__device__ void device_car_finish_tick(GpuCarState& car) {
    using namespace PhysConst;

    if (car.isDemoed) return;

    // Apply velocity impulse cache
    if (!v3_is_zero(car.velocityImpulseCache)) {
        car.vel += car.velocityImpulseCache;
        car.velocityImpulseCache = v3_zero();
    }

    // Clamp velocities
    car.vel = v3_clamp_mag(car.vel, CAR_MAX_SPEED);
    car.angVel = v3_clamp_mag(car.angVel, CAR_MAX_ANG_SPEED);
}



__device__ void device_ball_finish_tick(GpuBallState& ball, float ballMaxSpeed) {
    using namespace PhysConst;

    // Ball::_FinishPhysicsTick — impulse cache + speed clamps. Resting is
    // handled by the zero-velocity sleep rule in the world step (the floor
    // contact constraint cancels the velocity exactly, then the ball sleeps),
    // matching Arena::Step.
    if (!v3_is_zero(ball.velocityImpulseCache)) {
        ball.vel += ball.velocityImpulseCache;
        ball.velocityImpulseCache = v3_zero();
    }

    ball.vel = v3_clamp_mag(ball.vel, ballMaxSpeed);
    ball.angVel = v3_clamp_mag(ball.angVel, BALL_MAX_ANG_SPEED);
}


__device__ void device_boostpad_pre_tick(GpuBoostPadState& pad, float dt) {
    pad.prevLockedCarID = pad.curLockedCarID;
    pad.curLockedCarID = 0;
    if (pad.cooldown > 0.f) {
        pad.cooldown = rs_max(pad.cooldown - dt, 0.f);
        if (pad.cooldown == 0.f)
            pad.isActive = true;
    }
}

__device__ void device_boostpad_check_collide(
    GpuBoostPadState& pad, int padIdx,
    const GpuCarState& car
) {
    using namespace PhysConst;

    if (!pad.isActive && pad.prevLockedCarID != car.id) return;
    if (car.isDemoed) return;

    bool isBig = BoostPadData::IsBig(padIdx);
    Vec3 padPos = BoostPadData::Position(padIdx);

    // Cylinder check: 2D distance + height
    float cylRad = isBig ? PAD_CYL_RAD_BIG : PAD_CYL_RAD_SMALL;
    float dist2d = v3_dist_sq_2d(car.pos, padPos);

    if (dist2d < cylRad * cylRad) {
        float dz = rs_abs(car.pos.z - padPos.z);
        if (dz < PAD_CYL_HEIGHT) {
            pad.curLockedCarID = car.id;
        }
    }
}

__device__ void device_boostpad_post_tick(
    GpuBoostPadState& pad, int padIdx,
    GpuCarState* cars, int numCars, float dt,
    float cooldownBig, float cooldownSmall
) {
    using namespace PhysConst;

    if (pad.curLockedCarID == 0) return;
    if (!pad.isActive) return;

    bool isBig = BoostPadData::IsBig(padIdx);
    float boostAmount = isBig ? PAD_BOOST_BIG : PAD_BOOST_SMALL;

    // Find the car and give boost
    for (int c = 0; c < numCars; c++) {
        if (cars[c].id == pad.curLockedCarID) {
            // Only give if car doesn't already have full boost (for big) or can receive
            cars[c].boost = rs_min(cars[c].boost + boostAmount, BOOST_MAX);
            break;
        }
    }

    pad.isActive = false;
    pad.cooldown = isBig ? cooldownBig : cooldownSmall;
}

__device__ void device_handle_demo(GpuCarState& car, float dt, int arenaIdx) {
    using namespace PhysConst;

    car.demoRespawnTimer = rs_max(car.demoRespawnTimer - dt, 0.f);

    if (car.demoRespawnTimer <= 0.f) {
        // Respawn
        car.isDemoed = false;

        // Pick a respawn position based on car id (deterministic)
        int spawnIdx = (car.id + (int)arenaIdx) % SpawnData::RESPAWN_COUNT;
        SpawnData::SpawnPos sp = SpawnData::RespawnSoccar(spawnIdx);

        float yMul = (car.team == 0) ? 1.f : -1.f;
        float yawOffset = (car.team == 0) ? 0.f : M_PI;

        car.pos = v3(sp.x, sp.y * yMul, CAR_RESPAWN_Z);
        car.rotMat = euler_to_rotmat(sp.yawAng + yawOffset, 0.f, 0.f);
        car.vel = v3_zero();
        car.angVel = v3_zero();
        car.boost = BOOST_SPAWN_AMOUNT;
        car.hasJumped = false;
        car.hasDoubleJumped = false;
        car.hasFlipped = false;
        car.isFlipping = false;
        car.isJumping = false;
        car.jumpTime = 0.f;
        car.flipTime = 0.f;
        car.airTime = 0.f;
        car.airTimeSinceJump = 0.f;
        car.isSupersonic = false;
        car.supersonicTime = 0.f;
        car.handbrakeVal = 0.f;
        car.isAutoFlipping = false;
        car.velocityImpulseCache = v3_zero();
        car.worldContactHasContact = false;
    }
}

} // namespace rsc
