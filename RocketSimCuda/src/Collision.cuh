#pragma once

#include "CudaMath.cuh"
#include "BulletSolver.cuh"
#include "MeshCollision.cuh"

namespace rsc {

// ============================================================================
// ARENA SURFACE SETUP
// Build the analytical arena geometry as a list of planes + quarter-cylinder
// blends. Coordinates are in BT units (1 BT = 50 UU... actually for the soccar
// arena we use UU directly throughout).
// ============================================================================

// Convenience constructor for a flat plane surface.
inline ArenaSurface makePlane(Vec3 normal, float offset, Vec3 bMin, Vec3 bMax,
                              bool isStaticPlane = false) {
    ArenaSurface s = {};
    s.kind = ArenaSurface::kPlane;
    s.isStaticPlane = isStaticPlane ? 1 : 0;
    s.normal = normal;
    s.offset = offset;
    s.boundsMin = bMin;
    s.boundsMax = bMax;
    return s;
}

// Convenience constructor for a quarter-cylinder surface.
// `center`  : 3D centre of the circular slice (a point on the cylinder axis).
// `axis`    : unit vector along the axis (the cylinder runs along this).
// `radius`  : circle radius.
// `quadN1`  : first radial direction spanning the valid quadrant (unit).
// `quadN2`  : second radial direction (unit, perpendicular to both quadN1 and axis).
inline ArenaSurface makeQuarterCylinder(
    Vec3 center, Vec3 axis, float radius,
    Vec3 quadN1, Vec3 quadN2,
    Vec3 bMin, Vec3 bMax)
{
    ArenaSurface s = {};
    s.kind = ArenaSurface::kQuarterCylinder;
    s.center = center;
    s.axis = axis;
    s.radius = radius;
    s.quadN1 = quadN1;
    s.quadN2 = quadN2;
    s.boundsMin = bMin;
    s.boundsMax = bMax;
    return s;
}

inline int buildArenaSurfaces(ArenaSurface* out) {
    using namespace PhysConst;
    int n = 0;
    const float INF = 1e6f;

    // ------------------------------------------------------------------------
    // FLAT PLANES (14 total)
    // ------------------------------------------------------------------------

    // The reference arena uses four btStaticPlaneShape colliders for the
    // floor, ceiling and +-X walls (Arena::_SetupArenaCollisionShapes); the
    // rest of the geometry comes from the BVH triangle meshes.

    // Floor: z >= 0
    out[n++] = makePlane({0,0,1}, 0.f, {-INF,-INF,-INF}, {INF,INF,INF}, true);

    // Ceiling: z <= ARENA_HEIGHT
    out[n++] = makePlane({0,0,-1}, -ARENA_HEIGHT, {-INF,-INF,-INF}, {INF,INF,INF}, true);

    // Left wall: x >= -ARENA_EXTENT_X (normal points +X)
    out[n++] = makePlane({1,0,0}, -ARENA_EXTENT_X, {-INF,-INF,-INF}, {INF,INF,INF}, true);

    // Right wall: x <= ARENA_EXTENT_X (normal points -X)
    out[n++] = makePlane({-1,0,0}, -ARENA_EXTENT_X, {-INF,-INF,-INF}, {INF,INF,INF}, true);

    // Back wall blue: y >= -ARENA_EXTENT_Y (normal +Y)
    // With goal opening: 3 segments (left of goal, right of goal, above goal).
    out[n++] = makePlane({0,1,0}, -ARENA_EXTENT_Y,
        {-INF, -INF, -INF}, {-GOAL_HALF_WIDTH, INF, INF});
    out[n++] = makePlane({0,1,0}, -ARENA_EXTENT_Y,
        {GOAL_HALF_WIDTH, -INF, -INF}, {INF, INF, INF});
    out[n++] = makePlane({0,1,0}, -ARENA_EXTENT_Y,
        {-GOAL_HALF_WIDTH, -INF, GOAL_HEIGHT}, {GOAL_HALF_WIDTH, INF, INF});

    // Back wall orange: y <= ARENA_EXTENT_Y (normal -Y) with goal opening.
    out[n++] = makePlane({0,-1,0}, -ARENA_EXTENT_Y,
        {-INF, -INF, -INF}, {-GOAL_HALF_WIDTH, INF, INF});
    out[n++] = makePlane({0,-1,0}, -ARENA_EXTENT_Y,
        {GOAL_HALF_WIDTH, -INF, -INF}, {INF, INF, INF});
    out[n++] = makePlane({0,-1,0}, -ARENA_EXTENT_Y,
        {-GOAL_HALF_WIDTH, -INF, GOAL_HEIGHT}, {GOAL_HALF_WIDTH, INF, INF});

    // Corner bevels (45 degrees) - 4 vertical corners.
    // The arena corners are at (+-4096, +-5120). Corner bevel offset is ~1152 UU
    // from the sharp corner along each wall, so the bevel plane equation is
    // x + y = 8064 (and mirrors).
    const float cornerOffset = ARENA_EXTENT_X + ARENA_EXTENT_Y - 1152.f; // ~8064
    const float n45 = 0.70710678f; // 1/sqrt(2)

    out[n++] = makePlane({-n45, -n45, 0}, -cornerOffset * n45,
        {-INF, -INF, -INF}, {INF, INF, INF});
    out[n++] = makePlane({n45, -n45, 0}, -cornerOffset * n45,
        {-INF, -INF, -INF}, {INF, INF, INF});
    out[n++] = makePlane({-n45, n45, 0}, -cornerOffset * n45,
        {-INF, -INF, -INF}, {INF, INF, INF});
    out[n++] = makePlane({n45, n45, 0}, -cornerOffset * n45,
        {-INF, -INF, -INF}, {INF, INF, INF});

    // ------------------------------------------------------------------------
    // QUARTER-CYLINDER BLENDS
    //
    // Floor-wall curves:  R = 256 UU, tangent to floor at distance 256 from
    //                     the wall and to the wall at z = 256.
    // Wall-ceiling curves: R = 512 UU, tangent to the wall at z = 1536 and to
    //                      the ceiling at distance 512 from the wall.
    //
    // Verified from collision_meshes/soccar/*.cmf via least-squares circle fit
    // (max residual 0.07 UU for floor-wall, 0.14 UU for wall-ceiling).
    //
    // The X-wall curves run along the Y axis, so y bounds are the wall span
    // between the two corner bevels (|y| <= 3960). The Y-wall curves run along
    // the X axis with a goal-mouth cutout at |x| <= GOAL_HALF_WIDTH for the
    // floor-wall variant only (the wall-ceiling curve sits at z >= 1536, well
    // above the GOAL_HEIGHT = 642.775 mouth).
    // ------------------------------------------------------------------------

    const float FW_R = 256.f;             // floor-wall radius
    const float FW_FROM_WALL = 256.f;     // tangent distance from wall
    const float WC_R = 512.f;             // wall-ceiling radius
    const float WC_Z_TANGENT = 1536.f;    // wall tangent height
    const float CORNER_TRIM = 3960.f;     // y bound for X walls (just inside the bevel)
    const float CORNER_TRIM_X = 2940.f;   // x bound for Y walls (just inside the bevel)

    // -------- Floor-wall, +X wall (axis along Y) --------
    // Tangent points: (4096, _, 256) on the wall, (3840, _, 0) on the floor.
    // Centre at (4096 - 256, _, 256) = (3840, _, 256).
    // Quadrant: +X relative to centre, -Z relative to centre.
    out[n++] = makeQuarterCylinder(
        {ARENA_EXTENT_X - FW_FROM_WALL, 0.f, FW_R}, // center
        {0.f, 1.f, 0.f},                             // axis
        FW_R,                                         // radius
        {1.f, 0.f, 0.f},                             // quadN1 (+X)
        {0.f, 0.f, -1.f},                            // quadN2 (-Z)
        {ARENA_EXTENT_X - FW_FROM_WALL - 50.f, -CORNER_TRIM, -50.f},
        {INF, CORNER_TRIM, FW_R + 50.f});

    // -------- Floor-wall, -X wall --------
    out[n++] = makeQuarterCylinder(
        {-(ARENA_EXTENT_X - FW_FROM_WALL), 0.f, FW_R},
        {0.f, 1.f, 0.f},
        FW_R,
        {-1.f, 0.f, 0.f}, // quadN1 (-X)
        {0.f, 0.f, -1.f}, // quadN2 (-Z)
        {-INF, -CORNER_TRIM, -50.f},
        {-(ARENA_EXTENT_X - FW_FROM_WALL) + 50.f, CORNER_TRIM, FW_R + 50.f});

    // -------- Floor-wall, +Y wall, left of goal (x < -GOAL_HALF_WIDTH) --------
    // Tangent points: (_, 5120, 256) on the wall, (_, 4864, 0) on the floor.
    // Centre at (_, 4864, 256). Axis along X.
    out[n++] = makeQuarterCylinder(
        {0.f, ARENA_EXTENT_Y - FW_FROM_WALL, FW_R},
        {1.f, 0.f, 0.f},
        FW_R,
        {0.f, 1.f, 0.f},  // quadN1 (+Y)
        {0.f, 0.f, -1.f}, // quadN2 (-Z)
        {-CORNER_TRIM_X, ARENA_EXTENT_Y - FW_FROM_WALL - 50.f, -50.f},
        {-GOAL_HALF_WIDTH, INF, FW_R + 50.f});

    // -------- Floor-wall, +Y wall, right of goal (x > GOAL_HALF_WIDTH) --------
    out[n++] = makeQuarterCylinder(
        {0.f, ARENA_EXTENT_Y - FW_FROM_WALL, FW_R},
        {1.f, 0.f, 0.f},
        FW_R,
        {0.f, 1.f, 0.f},
        {0.f, 0.f, -1.f},
        {GOAL_HALF_WIDTH, ARENA_EXTENT_Y - FW_FROM_WALL - 50.f, -50.f},
        {CORNER_TRIM_X, INF, FW_R + 50.f});

    // -------- Floor-wall, -Y wall, left of goal --------
    out[n++] = makeQuarterCylinder(
        {0.f, -(ARENA_EXTENT_Y - FW_FROM_WALL), FW_R},
        {1.f, 0.f, 0.f},
        FW_R,
        {0.f, -1.f, 0.f}, // quadN1 (-Y)
        {0.f, 0.f, -1.f}, // quadN2 (-Z)
        {-CORNER_TRIM_X, -INF, -50.f},
        {-GOAL_HALF_WIDTH, -(ARENA_EXTENT_Y - FW_FROM_WALL) + 50.f, FW_R + 50.f});

    // -------- Floor-wall, -Y wall, right of goal --------
    out[n++] = makeQuarterCylinder(
        {0.f, -(ARENA_EXTENT_Y - FW_FROM_WALL), FW_R},
        {1.f, 0.f, 0.f},
        FW_R,
        {0.f, -1.f, 0.f},
        {0.f, 0.f, -1.f},
        {GOAL_HALF_WIDTH, -INF, -50.f},
        {CORNER_TRIM_X, -(ARENA_EXTENT_Y - FW_FROM_WALL) + 50.f, FW_R + 50.f});

    // -------- Wall-ceiling, +X wall (axis along Y) --------
    // Tangent points: (4096, _, 1536) on the wall, (3584, _, 2048) on the ceiling.
    // Centre at (3584, _, 1536). Quadrant: +X, +Z relative to centre.
    out[n++] = makeQuarterCylinder(
        {ARENA_EXTENT_X - WC_R, 0.f, WC_Z_TANGENT},
        {0.f, 1.f, 0.f},
        WC_R,
        {1.f, 0.f, 0.f}, // quadN1 (+X)
        {0.f, 0.f, 1.f}, // quadN2 (+Z)
        {ARENA_EXTENT_X - WC_R - 50.f, -CORNER_TRIM, WC_Z_TANGENT - 50.f},
        {INF, CORNER_TRIM, ARENA_HEIGHT + 50.f});

    // -------- Wall-ceiling, -X wall --------
    out[n++] = makeQuarterCylinder(
        {-(ARENA_EXTENT_X - WC_R), 0.f, WC_Z_TANGENT},
        {0.f, 1.f, 0.f},
        WC_R,
        {-1.f, 0.f, 0.f}, // quadN1 (-X)
        {0.f, 0.f, 1.f},  // quadN2 (+Z)
        {-INF, -CORNER_TRIM, WC_Z_TANGENT - 50.f},
        {-(ARENA_EXTENT_X - WC_R) + 50.f, CORNER_TRIM, ARENA_HEIGHT + 50.f});

    // -------- Wall-ceiling, +Y wall (unbroken — entirely above the goal mouth) --------
    out[n++] = makeQuarterCylinder(
        {0.f, ARENA_EXTENT_Y - WC_R, WC_Z_TANGENT},
        {1.f, 0.f, 0.f},
        WC_R,
        {0.f, 1.f, 0.f}, // quadN1 (+Y)
        {0.f, 0.f, 1.f}, // quadN2 (+Z)
        {-CORNER_TRIM_X, ARENA_EXTENT_Y - WC_R - 50.f, WC_Z_TANGENT - 50.f},
        {CORNER_TRIM_X, INF, ARENA_HEIGHT + 50.f});

    // -------- Wall-ceiling, -Y wall --------
    out[n++] = makeQuarterCylinder(
        {0.f, -(ARENA_EXTENT_Y - WC_R), WC_Z_TANGENT},
        {1.f, 0.f, 0.f},
        WC_R,
        {0.f, -1.f, 0.f}, // quadN1 (-Y)
        {0.f, 0.f, 1.f},  // quadN2 (+Z)
        {-CORNER_TRIM_X, -INF, WC_Z_TANGENT - 50.f},
        {CORNER_TRIM_X, -(ARENA_EXTENT_Y - WC_R) + 50.f, ARENA_HEIGHT + 50.f});

    return n;
}

// Helper: project (p - center) onto the plane perpendicular to `axis` and
// return the radial vector. axisOut receives the along-axis scalar coordinate
// for callers that need it.
__device__ inline Vec3 projectRadial(
    Vec3 p, Vec3 center, Vec3 axis, float& axisOut)
{
    Vec3 delta = p - center;
    axisOut = v3_dot(delta, axis);
    return delta - axis * axisOut;
}

// ============================================================================
// BALL-WORLD COLLISION
// Sphere vs analytical arena planes
// ============================================================================

__device__ void device_collision_ball_world(
    GpuBallState& ball,
    const ArenaSurface* surfaces, int numSurfaces,
    float ballRadius, float friction, float restitution
) {
    const float effectiveRadius = ballRadius + PhysConst::BALL_WORLD_COLLISION_MARGIN;

    for (int s = 0; s < numSurfaces; s++) {
        const ArenaSurface& surf = surfaces[s];

        // Broadphase AABB cull — identical for both surface kinds.
        {
            Vec3 bmin = surf.boundsMin;
            Vec3 bmax = surf.boundsMax;
            if (ball.pos.x < bmin.x || ball.pos.x > bmax.x ||
                ball.pos.y < bmin.y || ball.pos.y > bmax.y ||
                ball.pos.z < bmin.z || ball.pos.z > bmax.z)
                continue;
        }

        // Compute per-kind contact normal `n` (pointing INTO playable space)
        // and the signed `penetration` amount (positive = overlapping).
        Vec3 n;
        float penetration;

        if (surf.kind == ArenaSurface::kPlane) {
            n = surf.normal;
            float dist = v3_dot(ball.pos, n) - surf.offset;
            if (dist >= effectiveRadius) continue;
            penetration = effectiveRadius - dist;
        } else {
            // Quarter-cylinder.
            // Project ball centre onto the plane perpendicular to the axis.
            float axialT;
            Vec3 radial = projectRadial(ball.pos, surf.center, surf.axis, axialT);

            // Quadrant test — both dots must be non-negative to be in the
            // valid arc of the cylinder.
            float q1 = v3_dot(radial, surf.quadN1);
            float q2 = v3_dot(radial, surf.quadN2);
            if (q1 < 0.f || q2 < 0.f) continue;

            float radialDist = v3_length(radial);
            if (radialDist < 1e-4f) continue; // degenerate, ignore

            // Overlap test: ball centre plus its radius must exceed the
            // curve's radius for collision.
            float overlap = (radialDist + effectiveRadius) - surf.radius;
            if (overlap <= 0.f) continue;

            // Contact normal points from the test point toward the centre
            // (i.e. out of the curve surface, into playable space).
            n = radial * (-1.f / radialDist);
            penetration = overlap;
        }

        // Push out along the contact normal.
        ball.pos += n * penetration;

        // Velocity reflection with restitution.
        float velNormal = v3_dot(ball.vel, n);
        if (velNormal < 0.f) {
            // Prevent an endless tiny bounce loop on the floor. Bullet settles
            // the resting ball to exact zero; our analytical solver needs an
            // explicit low-speed cutoff for the normal component.
            if (n.z > 0.99f && -velNormal <= PhysConst::BALL_FLOOR_REST_BOUNCE_SPEED) {
                ball.vel -= n * velNormal;
                if (rs_abs(ball.pos.z - PhysConst::BALL_REST_Z) < 0.5f)
                    ball.pos.z = PhysConst::BALL_REST_Z;
                continue;
            }

            // Normal impulse.
            ball.vel -= n * ((1.f + restitution) * velNormal);

            // Tangential friction.
            Vec3 velTangent = ball.vel - n * v3_dot(ball.vel, n);
            float tangentSpeed = v3_length(velTangent);
            if (tangentSpeed > 1e-4f) {
                float frictionImpulse = rs_abs(velNormal) * friction;
                if (frictionImpulse > tangentSpeed)
                    frictionImpulse = tangentSpeed;
                ball.vel -= v3_normalize(velTangent) * frictionImpulse;
            }

            // Angular velocity from surface friction (simplified).
            Vec3 contactPt = ball.pos - n * effectiveRadius;
            Vec3 surfaceVelAtContact = v3_cross(ball.angVel, n * (-effectiveRadius));
            Vec3 relSurfaceVel = surfaceVelAtContact - velTangent;
            ball.angVel -= relSurfaceVel * (friction * 0.5f / effectiveRadius);
        }
    }
}

// ============================================================================
// CAR-WORLD COLLISION
// OBB corners vs analytical arena planes
// ============================================================================

__device__ void device_collision_car_world(
    GpuCarState& car,
    const ArenaSurface* surfaces, int numSurfaces
) {
    using namespace PhysConst;

    if (car.isDemoed) return;

    Vec3 halfExtent = car.config.hitboxSize * 0.5f;
    Vec3 offset = car.config.hitboxPosOffset;

    car.worldContactHasContact = false;

    // Test 8 corners of the OBB against every surface.
    for (int cx = -1; cx <= 1; cx += 2) {
        for (int cy = -1; cy <= 1; cy += 2) {
            for (int cz = -1; cz <= 1; cz += 2) {
                Vec3 localPt = offset + v3(halfExtent.x * cx, halfExtent.y * cy, halfExtent.z * cz);
                Vec3 worldPt = car.pos + rotmat_mul_vec(car.rotMat, localPt);

                for (int s = 0; s < numSurfaces; s++) {
                    const ArenaSurface& surf = surfaces[s];

                    // Broadphase AABB cull against the test point.
                    {
                        Vec3 bmin = surf.boundsMin;
                        Vec3 bmax = surf.boundsMax;
                        if (worldPt.x < bmin.x || worldPt.x > bmax.x ||
                            worldPt.y < bmin.y || worldPt.y > bmax.y ||
                            worldPt.z < bmin.z || worldPt.z > bmax.z)
                            continue;
                    }

                    // Compute per-kind contact normal `n` (into playable space)
                    // and the signed `penetration` (positive = corner is behind
                    // the surface).
                    Vec3 n;
                    float penetration;

                    if (surf.kind == ArenaSurface::kPlane) {
                        n = surf.normal;
                        float planeDist = v3_dot(worldPt, n) - surf.offset;
                        if (planeDist >= 0.f) continue;
                        penetration = -planeDist;
                    } else {
                        // Quarter-cylinder.
                        float axialT;
                        Vec3 radial = projectRadial(worldPt, surf.center, surf.axis, axialT);

                        float q1 = v3_dot(radial, surf.quadN1);
                        float q2 = v3_dot(radial, surf.quadN2);
                        if (q1 < 0.f || q2 < 0.f) continue;

                        float radialDist = v3_length(radial);
                        if (radialDist < 1e-4f) continue;

                        // Corner penetrates when its radial distance exceeds
                        // the curve radius (corner is a point, no radius to add).
                        float overlap = radialDist - surf.radius;
                        if (overlap <= 0.f) continue;

                        n = radial * (-1.f / radialDist);
                        penetration = overlap;
                    }

                    // Push the whole car by the penetration along n. Because
                    // we move the car body (not just the corner), subsequent
                    // corner tests see the updated position.
                    car.pos += n * penetration;

                    // Contact response on the CG velocity.
                    float velNormal = v3_dot(car.vel, n);
                    if (velNormal < 0.f) {
                        car.vel -= n * ((1.f + CARWORLD_COLLISION_RESTITUTION) * velNormal);

                        // Tangential friction.
                        Vec3 velT = car.vel - n * v3_dot(car.vel, n);
                        float tSpeed = v3_length(velT);
                        if (tSpeed > 1e-4f) {
                            float fricImpulse = rs_abs(velNormal) * CARWORLD_COLLISION_FRICTION;
                            if (fricImpulse > tSpeed) fricImpulse = tSpeed;
                            car.vel -= v3_normalize(velT) * fricImpulse;
                        }
                    }

                    // Track world contact for autoflip/autoroll.
                    car.worldContactHasContact = true;
                    car.worldContactNormal = n;
                }
            }
        }
    }
}

// ============================================================================
// CAR-BALL COLLISION
// OBB vs Sphere with extra impulse (transcribes Arena::_BtCallback_OnCarBallCollision)
// ============================================================================

__device__ void device_collision_car_ball(
    GpuCarState& car, GpuBallState& ball,
    float ballRadius, float ballHitExtraForceScale,
    uint64_t tickCount
) {
    using namespace PhysConst;

    if (car.isDemoed) return;

    // Transform ball into car's local hitbox space
    Vec3 localBallPos = rotmat_dot_vec(car.rotMat, ball.pos - car.pos) - car.config.hitboxPosOffset;
    Vec3 halfExt = car.config.hitboxSize * 0.5f;

    // Closest point on box
    Vec3 closest;
    closest.x = rs_clamp(localBallPos.x, -halfExt.x, halfExt.x);
    closest.y = rs_clamp(localBallPos.y, -halfExt.y, halfExt.y);
    closest.z = rs_clamp(localBallPos.z, -halfExt.z, halfExt.z);

    Vec3 diff = localBallPos - closest;
    float distSq = v3_length_sq(diff);

    if (distSq >= ballRadius * ballRadius) return; // No collision

    float dist = sqrtf(distSq);

    car.ballHitValid = true;
    car.ballHitTickCount = tickCount;

    // Contact normal (world space, pointing from car to ball)
    Vec3 contactNormal;
    if (dist > 1e-4f) {
        contactNormal = rotmat_mul_vec(car.rotMat, diff / dist);
    } else {
        contactNormal = v3_safe_normalize(ball.pos - car.pos);
    }

    // Penetration resolution
    float penetration = ballRadius - dist;
    ball.pos += contactNormal * penetration;

    // Physical collision response (Bullet-like impulse)
    Vec3 relVel = ball.vel - car.vel;
    float velAlongNormal = v3_dot(relVel, contactNormal);

    if (velAlongNormal < 0.f) {
        // Compute impulse with car-ball friction/restitution
        float invMassCar  = 1.f / CAR_MASS;
        float invMassBall = 1.f / BALL_MASS;
        float j = -(1.f + CARBALL_COLLISION_RESTITUTION) * velAlongNormal / (invMassCar + invMassBall);

        ball.vel += contactNormal * (j * invMassBall);
        car.vel  -= contactNormal * (j * invMassCar);

        // Tangential friction
        Vec3 tangent = relVel - contactNormal * velAlongNormal;
        float tangentSpeed = v3_length(tangent);
        if (tangentSpeed > 1e-4f) {
            Vec3 tangentDir = tangent / tangentSpeed;
            float jt = -tangentSpeed / (invMassCar + invMassBall);
            float maxFric = CARBALL_COLLISION_FRICTION * rs_abs(j);
            jt = rs_clamp(jt, -maxFric, maxFric);
            ball.vel += tangentDir * (jt * invMassBall);
            car.vel  -= tangentDir * (jt * invMassCar);
        }
    }

    // Extra car-ball impulse (RocketSim-specific)
    if (tickCount > car.lastExtraImpulseTick + 1 || car.lastExtraImpulseTick > tickCount) {
        car.lastExtraImpulseTick = tickCount;

        Vec3 relPos = ball.pos - car.pos;
        Vec3 relVelFull = ball.vel - car.vel;
        float relSpeed = rs_min(v3_length(relVelFull), BALL_CAR_EXTRA_IMPULSE_MAXDELTAVEL);

        if (relSpeed > 0.f) {
            Vec3 hitDir = v3_safe_normalize(relPos * v3(1.f, 1.f, BALL_CAR_EXTRA_IMPULSE_Z_SCALE));
            Vec3 carFwd = car_forward(car);
            Vec3 fwdAdj = carFwd * (v3_dot(hitDir, carFwd) * (1.f - BALL_CAR_EXTRA_IMPULSE_FORWARD_SCALE));
            hitDir = v3_safe_normalize(hitDir - fwdAdj);

            Vec3 addedVel = hitDir * relSpeed * curve_ball_car_impulse(relSpeed) * ballHitExtraForceScale;
            ball.velocityImpulseCache += addedVel;
        }
    }
}

// ============================================================================
// CAR-CAR COLLISION (bumps and demos)
// Transcribes Arena::_BtCallback_OnCarCarCollision
// ============================================================================

// ============================================================================
// OBB vs OBB Separating Axis Theorem
//
// Full 15-axis SAT between two oriented bounding boxes. Returns true if the
// boxes overlap, in which case `contactNormal` points from box A toward box B
// (unit vector) and `overlap` holds the minimum penetration depth in UU.
//
// Axes tested:
//   - 3 face normals of A (its local x/y/z basis)
//   - 3 face normals of B
//   - 9 edge-edge cross products (A.axis[i] x B.axis[j] for all i,j)
// Near-parallel edge pairs produce a degenerate cross product and are skipped
// to avoid false separations from floating-point noise.
// ============================================================================
__device__ inline bool obbVsObbSAT(
    Vec3 cA, const Vec3 axA[3], const Vec3 hA,
    Vec3 cB, const Vec3 axB[3], const Vec3 hB,
    Vec3& contactNormal, float& overlap)
{
    Vec3 d = cB - cA;

    // Precompute |dot(axA[i], axB[j])| for the 9 edge-edge axes and projections.
    float R[3][3];
    float absR[3][3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            R[i][j]    = v3_dot(axA[i], axB[j]);
            absR[i][j] = rs_abs(R[i][j]) + 1e-6f; // epsilon for parallel edges
        }
    }

    // Project the separation vector `d` onto A's local frame.
    float tA[3] = { v3_dot(d, axA[0]), v3_dot(d, axA[1]), v3_dot(d, axA[2]) };
    float hAi[3] = { hA.x, hA.y, hA.z };
    float hBi[3] = { hB.x, hB.y, hB.z };

    float minOverlap = 1e30f;
    Vec3  minAxis = v3(1, 0, 0);

    // --- 3 face normals of A ---------------------------------------------------
    for (int i = 0; i < 3; i++) {
        float rA = hAi[i];
        float rB = hBi[0] * absR[i][0] + hBi[1] * absR[i][1] + hBi[2] * absR[i][2];
        float dist = rs_abs(tA[i]);
        if (dist > rA + rB) return false;
        float over = (rA + rB) - dist;
        if (over < minOverlap) { minOverlap = over; minAxis = axA[i]; }
    }

    // --- 3 face normals of B ---------------------------------------------------
    float tB[3] = { v3_dot(d, axB[0]), v3_dot(d, axB[1]), v3_dot(d, axB[2]) };
    for (int i = 0; i < 3; i++) {
        float rA = hAi[0] * absR[0][i] + hAi[1] * absR[1][i] + hAi[2] * absR[2][i];
        float rB = hBi[i];
        float dist = rs_abs(tB[i]);
        if (dist > rA + rB) return false;
        float over = (rA + rB) - dist;
        if (over < minOverlap) { minOverlap = over; minAxis = axB[i]; }
    }

    // --- 9 edge-edge cross products -------------------------------------------
    // For each pair (i,j) the SAT axis is axA[i] x axB[j]. Projections can be
    // computed without materialising the cross vector (standard trick).
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            int i1 = (i + 1) % 3, i2 = (i + 2) % 3;
            int j1 = (j + 1) % 3, j2 = (j + 2) % 3;
            float rA = hAi[i1] * absR[i2][j] + hAi[i2] * absR[i1][j];
            float rB = hBi[j1] * absR[i][j2] + hBi[j2] * absR[i][j1];
            float dist = rs_abs(tA[i2] * R[i1][j] - tA[i1] * R[i2][j]);
            if (dist > rA + rB) return false;

            // Materialise the axis to compute its length so we can normalise
            // the overlap back into world UU space.
            Vec3 L = v3_cross(axA[i], axB[j]);
            float Llen = v3_length(L);
            if (Llen < 1e-4f) continue; // parallel edges — skip as SAT axis
            float over = ((rA + rB) - dist) / Llen;
            if (over < minOverlap) {
                minOverlap = over;
                minAxis = L * (1.f / Llen);
            }
        }
    }

    // Flip axis so it points from A toward B (makes downstream response easier).
    if (v3_dot(d, minAxis) < 0.f) minAxis = minAxis * -1.f;

    contactNormal = minAxis;
    overlap = minOverlap;
    return true;
}

__device__ void device_collision_car_car(
    GpuCarState& car1, GpuCarState& car2,
    float bumpForceScale, float bumpCooldownTime
) {
    using namespace PhysConst;

    if (car1.isDemoed || car2.isDemoed) return;

    // Broadphase: sphere-sphere check using the hitbox diagonals.
    float diag1 = v3_length(car1.config.hitboxSize) * 0.5f;
    float diag2 = v3_length(car2.config.hitboxSize) * 0.5f;
    float maxDist = diag1 + diag2;

    if (v3_dist_sq(car1.pos, car2.pos) > maxDist * maxDist) return;

    // Narrowphase: real OBB-OBB SAT.
    Vec3 cA = car1.pos + rotmat_mul_vec(car1.rotMat, car1.config.hitboxPosOffset);
    Vec3 cB = car2.pos + rotmat_mul_vec(car2.rotMat, car2.config.hitboxPosOffset);
    Vec3 axA[3] = { car1.rotMat.forward, car1.rotMat.right, car1.rotMat.up };
    Vec3 axB[3] = { car2.rotMat.forward, car2.rotMat.right, car2.rotMat.up };
    Vec3 hA = car1.config.hitboxSize * 0.5f;
    Vec3 hB = car2.config.hitboxSize * 0.5f;

    Vec3 contactNormal;
    float overlap;
    if (!obbVsObbSAT(cA, axA, hA, cB, axB, hB, contactNormal, overlap)) return;
    if (overlap <= 0.f) return;

    // Physical collision response
    float velAlongNormal = v3_dot(car2.vel - car1.vel, contactNormal);
    if (velAlongNormal < 0.f) {
        float invMass = 1.f / CAR_MASS;
        float j = -(1.f + CARCAR_COLLISION_RESTITUTION) * velAlongNormal / (2.f * invMass);
        car1.vel -= contactNormal * (j * invMass);
        car2.vel += contactNormal * (j * invMass);
    }

    // Push apart
    car1.pos -= contactNormal * (overlap * 0.5f);
    car2.pos += contactNormal * (overlap * 0.5f);

    // Bump/demo logic (check both directions)
    for (int dir = 0; dir < 2; dir++) {
        GpuCarState& attacker = (dir == 0) ? car1 : car2;
        GpuCarState& victim   = (dir == 0) ? car2 : car1;

        Vec3 deltaPos = victim.pos - attacker.pos;
        if (v3_dot(attacker.vel, deltaPos) <= 0.f) continue;

        // In cooldown?
        if (attacker.carContactOtherID == victim.id && attacker.carContactCooldownTimer > 0.f)
            continue;

        Vec3 velDir = v3_safe_normalize(attacker.vel);
        Vec3 dirToOther = v3_safe_normalize(deltaPos);

        float speedTowards = v3_dot(attacker.vel, dirToOther);
        float otherAwaySpeed = v3_dot(victim.vel, velDir);

        if (speedTowards <= otherAwaySpeed) continue;

        // Check if hit with bumper (forward part of hitbox)
        Vec3 localHitPt = rotmat_dot_vec(attacker.rotMat, deltaPos);
        if (localHitPt.x < BUMP_MIN_FORWARD_DIST) continue;

        // Demo check
        bool isDemo = attacker.isSupersonic && (attacker.team != victim.team);

        if (isDemo) {
            victim.isDemoed = true;
            victim.demoRespawnTimer = DEMO_RESPAWN_TIME;
        } else {
            bool groundHit = victim.isOnGround;
            float baseScale = groundHit
                ? curve_bump_vel_ground(speedTowards)
                : curve_bump_vel_air(speedTowards);

            Vec3 hitUpDir = victim.isOnGround ? victim.rotMat.up : v3(0, 0, 1);
            Vec3 bumpImpulse = (velDir * baseScale + hitUpDir * curve_bump_upward(speedTowards))
                             * bumpForceScale;

            victim.velocityImpulseCache += bumpImpulse;
        }

        attacker.carContactOtherID = victim.id;
        attacker.carContactCooldownTimer = bumpCooldownTime;
    }
}

// ============================================================================
// INTEGRATION STEP
// Semi-implicit Euler (symplectic) matching Bullet's integration
// ============================================================================

__device__ void device_integrate(
    Vec3& pos, Vec3& vel, RotMat& rotMat, Vec3& angVel,
    Vec3 gravity, float linearDamping, float dt
) {
    // Apply gravity
    vel += gravity * dt;

    // Bullet's default damping path uses pow(1 - damping, dt), not a full
    // per-tick linear scale, otherwise the ball loses almost all of its speed.
    if (linearDamping > 0.f)
        vel *= powf(fmaxf(1.f - linearDamping, 0.f), dt);

    // Update position (using new velocity - semi-implicit Euler)
    pos += vel * dt;

    // Update rotation
    rotMat = rotmat_integrate(rotMat, angVel, dt);
}

// ============================================================================
// ============================================================================
//                    BULLET-EXACT WORLD STEP (parity path)
// ----------------------------------------------------------------------------
// Replicates RocketSim's patched bullet3-3.24 tick:
//   1. damping (predictUnconstrainedMotion)
//   2. collision detection at the CURRENT (pre-move) positions
//   3. sequential-impulse solve:
//        - external forces (gravity, boost, throttle, sticky, ...) integrated
//          by the solver as externalForce/TorqueImpulse
//        - per-point split-impulse position recovery (erp2 = 0.8)
//        - ball-world contacts: RocketSim "special" aggregated constraint
//        - warm starting on persistent normal impulses (factor 0.85)
//   4. writeback (vel += deltas + external impulses, pos += pushVel * dt)
//   5. integrate transforms (pos += vel*dt, exact quaternion exp-map rotation)
// ============================================================================
// ============================================================================

// Triangle-mesh shapes carry a 0.04 BT (= 2 UU) collision margin. It applies
// to GJK-based convex-vs-mesh narrowphase (car body), but NOT to the
// specialized SphereTriangleDetector used for the ball (its depth is measured
// to the bare sphere radius).
constexpr float MESH_COLLISION_MARGIN = 2.0f; // UU
// Box shapes: implicit dims = input - 0.04 BT; collision margin reduced by
// setSafeMargin to min(0.04, 0.1 * minHalfExtent).
constexpr float CONVEX_DISTANCE_MARGIN_UU = 2.0f;  // 0.04 BT
// Manifold points are added while distance < breaking threshold. Bullet uses
// a relative threshold (angular motion disc * gContactBreakingThreshold=0.02):
// ball disc = sphere radius 1.825 BT -> 1.825 UU.
constexpr float BALL_CONTACT_ADD_THRESHOLD = 1.825f;  // UU
constexpr float CAR_CONTACT_ADD_THRESHOLD = 1.55f;    // UU (compound hitbox disc * 0.02)

constexpr int MAX_CW_CONTACTS = 12;  // car-world contacts kept per car (mesh corners + plane supports)
constexpr int MAX_BW_CONTACTS = 8;   // ball-world contacts aggregated

__device__ __forceinline__ float car_box_safe_margin(const GpuCarConfig& cfg) {
    // btConvexInternalShape::setSafeMargin: min(0.04 BT, 0.1 * min half extent)
    float minHalf = fminf(cfg.hitboxSize.x, fminf(cfg.hitboxSize.y, cfg.hitboxSize.z)) * 0.5f;
    return fminf(CONVEX_DISTANCE_MARGIN_UU, 0.1f * minHalf);
}

struct GenContact {
    Vec3 posWorld;   // contact point on B (world surface), UU
    Vec3 normal;     // m_normalWorldOnB (into playable space / from B to A), unit
    float distance;  // signed gap in UU (negative = penetrating)
};

// ---- Ball vs arena surfaces (contact generation only) ----------------------
__device__ inline int device_gen_ball_world_contacts(
    const GpuBallState& ball,
    const ArenaSurface* surfaces, int numSurfaces,
    const MeshGridView& meshGrid,
    float ballRadius,
    GenContact* out, int maxOut
) {
    // SphereTriangleDetector: depth = -(radius - distanceToTriangle); no mesh
    // margin involved. Contact kept while distance < radius + breaking
    // threshold. The resting "hover" at ~radius + threshold above the floor
    // emerges from this (the special constraint keeps cancelling gravity while
    // the manifold persists) — matching the reference rest height ~93.1.
    int n = 0;

    if (meshGrid.numTris > 0) {
        // ---- Real arena triangles (BT units) ----
        const float radiusBT = ballRadius * bts::UU_TO_BT;
        const float breakingBT = BALL_CONTACT_ADD_THRESHOLD * bts::UU_TO_BT;
        Vec3 centerBT = ball.pos * bts::UU_TO_BT;

        float reach = radiusBT + breakingBT;
        Vec3 aabbMin = centerBT - v3(reach, reach, reach);
        Vec3 aabbMax = centerBT + v3(reach, reach, reach);

        // Per-mesh manifolds with btPersistentManifold semantics: 4-point
        // capacity, nearest-point replacement within the breaking threshold,
        // and sortCachedPoints (keep deepest, maximize area) when full.
        constexpr int MAX_MESH_MANIFOLDS = 4;
        struct ManifoldPt { Vec3 point, normal; float depth; bool valid; };
        ManifoldPt mani[MAX_MESH_MANIFOLDS][4] = {};
        int maniMeshId[MAX_MESH_MANIFOLDS];
        int numMani = 0;

        mesh_grid_query_aabb(meshGrid, aabbMin, aabbMax, [&](int triIdx) {
            const MeshTriangle& tri = meshGrid.tris[triIdx];
            Vec3 point, normal;
            float depth;
            if (!mesh_sphere_triangle_collide(centerBT, radiusBT, breakingBT, tri, point, normal, depth))
                return;

            // find/create the manifold for this mesh body
            int m = -1;
            for (int i = 0; i < numMani; i++)
                if (maniMeshId[i] == tri.meshId) { m = i; break; }
            if (m < 0) {
                if (numMani >= MAX_MESH_MANIFOLDS) return;
                m = numMani++;
                maniMeshId[m] = tri.meshId;
                for (int i = 0; i < 4; i++) mani[m][i].valid = false;
            }
            ManifoldPt* cache = mani[m];

            // getCacheEntry: nearest valid point within breaking threshold
            float shortest = breakingBT * breakingBT;
            int nearest = -1;
            int firstFree = -1;
            for (int i = 0; i < 4; i++) {
                if (!cache[i].valid) {
                    if (firstFree < 0) firstFree = i;
                    continue;
                }
                Vec3 d = cache[i].point - point;
                float dd = v3_length_sq(d);
                if (dd < shortest) { shortest = dd; nearest = i; }
            }

            if (nearest >= 0) {
                // replaceContactPoint (fresh per-tick: nothing warm to keep)
                cache[nearest].point = point;
                cache[nearest].normal = normal;
                cache[nearest].depth = depth;
                return;
            }

            if (firstFree >= 0) {
                cache[firstFree] = {point, normal, depth, true};
                return;
            }

            // Full: sortCachedPoints — keep the deepest point, replace the
            // slot that maximizes the resulting contact area (calcArea4Points).
            int maxPenetrationIndex = -1;
            float maxPenetration = depth;
            for (int i = 0; i < 4; i++) {
                if (cache[i].depth < maxPenetration) {
                    maxPenetrationIndex = i;
                    maxPenetration = cache[i].depth;
                }
            }

            auto calcArea4 = [](Vec3 p0, Vec3 p1, Vec3 p2, Vec3 p3) {
                Vec3 a0 = p0 - p1, b0 = p2 - p3;
                Vec3 a1 = p0 - p2, b1 = p1 - p3;
                Vec3 a2 = p0 - p3, b2 = p1 - p2;
                float l0 = v3_length_sq(v3_cross(a0, b0));
                float l1 = v3_length_sq(v3_cross(a1, b1));
                float l2 = v3_length_sq(v3_cross(a2, b2));
                return fmaxf(fmaxf(l0, l1), l2);
            };

            float res[4] = {0.f, 0.f, 0.f, 0.f};
            if (maxPenetrationIndex != 0)
                res[0] = calcArea4(point, cache[1].point, cache[2].point, cache[3].point);
            if (maxPenetrationIndex != 1)
                res[1] = calcArea4(point, cache[0].point, cache[2].point, cache[3].point);
            if (maxPenetrationIndex != 2)
                res[2] = calcArea4(point, cache[0].point, cache[1].point, cache[3].point);
            if (maxPenetrationIndex != 3)
                res[3] = calcArea4(point, cache[0].point, cache[1].point, cache[2].point);

            int biggest = 0;
            float biggestArea = res[0];
            for (int i = 1; i < 4; i++) {
                if (res[i] > biggestArea) { biggestArea = res[i]; biggest = i; }
            }
            cache[biggest] = {point, normal, depth, true};
        });

        for (int m = 0; m < numMani; m++) {
            for (int i = 0; i < 4 && n < maxOut; i++) {
                if (!mani[m][i].valid) continue;
                GenContact c;
                c.normal = mani[m][i].normal;
                c.distance = mani[m][i].depth * bts::BT_TO_UU;
                c.posWorld = mani[m][i].point * bts::BT_TO_UU;
                out[n++] = c;
            }
        }

        // ---- The four btStaticPlaneShape colliders ----
        for (int s = 0; s < numSurfaces && n < maxOut; s++) {
            const ArenaSurface& surf = surfaces[s];
            if (!surf.isStaticPlane)
                continue;
            float centerDist = v3_dot(ball.pos, surf.normal) - surf.offset;
            float gap = centerDist - ballRadius;
            if (gap < BALL_CONTACT_ADD_THRESHOLD) {
                GenContact c;
                c.normal = surf.normal;
                c.distance = gap;
                c.posWorld = ball.pos - surf.normal * centerDist;
                out[n++] = c;
            }
        }
        return n;
    }

    // ---- Analytic fallback (no mesh loaded) ----
    for (int s = 0; s < numSurfaces && n < maxOut; s++) {
        const ArenaSurface& surf = surfaces[s];

        {
            Vec3 bmin = surf.boundsMin;
            Vec3 bmax = surf.boundsMax;
            if (ball.pos.x < bmin.x || ball.pos.x > bmax.x ||
                ball.pos.y < bmin.y || ball.pos.y > bmax.y ||
                ball.pos.z < bmin.z || ball.pos.z > bmax.z)
                continue;
        }

        if (surf.kind == ArenaSurface::kPlane) {
            float centerDist = v3_dot(ball.pos, surf.normal) - surf.offset;
            float gap = centerDist - ballRadius;
            if (gap < BALL_CONTACT_ADD_THRESHOLD) {
                GenContact c;
                c.normal = surf.normal;
                c.distance = gap;
                c.posWorld = ball.pos - surf.normal * centerDist;
                out[n++] = c;
                // Floor/ceiling/X-walls exist both in the mesh and as static
                // planes: two manifolds with identical contact data.
                if (surf.isStaticPlane && n < maxOut)
                    out[n++] = c;
            }
        } else {
            float axialT;
            Vec3 radial = projectRadial(ball.pos, surf.center, surf.axis, axialT);
            float q1 = v3_dot(radial, surf.quadN1);
            float q2 = v3_dot(radial, surf.quadN2);
            if (q1 < 0.f || q2 < 0.f) continue;

            float radialDist = v3_length(radial);
            if (radialDist < 1e-4f) continue;

            // Inside of the curve is playable; the surface is at surf.radius.
            float centerDist = surf.radius - radialDist;  // distance from center to surface along inward normal
            float gap = centerDist - ballRadius;
            if (gap < BALL_CONTACT_ADD_THRESHOLD) {
                GenContact c;
                c.normal = radial * (-1.f / radialDist);
                c.distance = gap;
                c.posWorld = surf.center + surf.axis * axialT + radial * (surf.radius / radialDist);
                out[n++] = c;
            }
        }
    }
    return n;
}

// ---- Car (OBB) vs arena surfaces: persistent manifold -----------------------
// Mirrors btPersistentManifold: refreshContactPoints (expire stale points),
// then one new GJK-like point per overlapping surface, merged via cache-entry
// replacement (keeping warm-start impulses).

__device__ inline float car_world_breaking_threshold(const GpuCarConfig& cfg, float safeMargin) {
    // angularMotionDisc * gContactBreakingThreshold(0.02). The compound
    // shape's disc = |with-margin half extents| + |child offset|.
    Vec3 wmHalf = cfg.hitboxSize * 0.5f
        - v3(CONVEX_DISTANCE_MARGIN_UU, CONVEX_DISTANCE_MARGIN_UU, CONVEX_DISTANCE_MARGIN_UU)
        + v3(safeMargin, safeMargin, safeMargin);
    return (v3_length(wmHalf) + v3_length(cfg.hitboxPosOffset)) * 0.02f;
}

__device__ inline void device_manifold_add_point(
    GpuCarState& car, float breakingThreshold,
    Vec3 localPointA, Vec3 worldPointB, Vec3 normal, float distance, uint8_t algo
) {
    auto& mp = car.worldManifold;

    // btPersistentManifold::getCacheEntry — nearest valid point by localPointA
    // within the breaking threshold.
    float shortestDist = breakingThreshold * breakingThreshold;
    int nearest = -1;
    for (int i = 0; i < 4; i++) {
        if (!mp[i].valid) continue;
        Vec3 diffA = mp[i].localPointA - localPointA;
        float distToManiPoint = v3_dot(diffA, diffA);
        if (distToManiPoint < shortestDist) {
            shortestDist = distToManiPoint;
            nearest = i;
        }
    }

    if (nearest >= 0) {
        // replaceContactPoint: geometry replaced, applied impulse preserved
        float warm = mp[nearest].appliedImpulse;
        mp[nearest].localPointA = localPointA;
        mp[nearest].worldPointB = worldPointB;
        mp[nearest].normal = normal;
        mp[nearest].appliedImpulse = warm;
        mp[nearest].algo = algo;
        // distance refreshed at row build
        return;
    }

    // Insert into a free slot, else replace the shallowest point
    int slot = -1;
    for (int i = 0; i < 4; i++) {
        if (!mp[i].valid) { slot = i; break; }
    }
    if (slot < 0) {
        // sortCachedPoints approximation: keep deeper points
        float maxDist = -1e30f;
        for (int i = 0; i < 4; i++) {
            Vec3 posA = car.pos + rotmat_mul_vec(car.rotMat, mp[i].localPointA);
            float d = v3_dot(posA - mp[i].worldPointB, mp[i].normal);
            if (d > maxDist) { maxDist = d; slot = i; }
        }
        if (maxDist <= distance) return;  // new point is shallower than all kept
    }

    mp[slot].valid = 1;
    mp[slot].localPointA = localPointA;
    mp[slot].worldPointB = worldPointB;
    mp[slot].normal = normal;
    mp[slot].appliedImpulse = 0.f;
    mp[slot].algo = algo;
}

__device__ inline void device_update_car_world_manifold(
    GpuCarState& car,
    const ArenaSurface* surfaces, int numSurfaces
) {
    if (car.isDemoed) {
        for (int i = 0; i < 4; i++) car.worldManifold[i].valid = 0;
        return;
    }

    Vec3 implicitHalf = car.config.hitboxSize * 0.5f - v3(CONVEX_DISTANCE_MARGIN_UU, CONVEX_DISTANCE_MARGIN_UU, CONVEX_DISTANCE_MARGIN_UU);
    float safeMargin = car_box_safe_margin(car.config);
    float meshMarginSum = safeMargin + MESH_COLLISION_MARGIN;
    Vec3 offset = car.config.hitboxPosOffset;
    float breakingThreshold = car_world_breaking_threshold(car.config, safeMargin);

    // ---- refreshContactPoints: expire stale points ----
    for (int i = 0; i < 4; i++) {
        auto& pt = car.worldManifold[i];
        if (!pt.valid) continue;
        Vec3 posA = car.pos + rotmat_mul_vec(car.rotMat, pt.localPointA);
        float distance = v3_dot(posA - pt.worldPointB, pt.normal);
        if (distance > breakingThreshold) {
            pt.valid = 0;
            continue;
        }
        Vec3 projectedPoint = posA - pt.normal * distance;
        Vec3 projectedDifference = pt.worldPointB - projectedPoint;
        if (v3_dot(projectedDifference, projectedDifference) > breakingThreshold * breakingThreshold)
            pt.valid = 0;
    }

    if (numSurfaces <= 0)
        return;

    // ---- new contact candidates: one per overlapping surface/algorithm ----
    for (int s = 0; s < numSurfaces; s++) {
        const ArenaSurface& surf = surfaces[s];

        // BVH-mesh narrowphase (deepest implicit-box point, both margins).
        {
            float bestGap = 1e30f;
            Vec3 bestNormal, bestWorldPt, bestLocalPt;

            int cornerIdx = 0;
            for (int cx = -1; cx <= 1; cx += 2) {
                for (int cy = -1; cy <= 1; cy += 2) {
                    for (int cz = -1; cz <= 1; cz += 2, cornerIdx++) {
                        Vec3 localPt = offset + v3(implicitHalf.x * cx, implicitHalf.y * cy, implicitHalf.z * cz);
                        Vec3 worldPt = car.pos + rotmat_mul_vec(car.rotMat, localPt);

                        {
                            Vec3 bmin = surf.boundsMin;
                            Vec3 bmax = surf.boundsMax;
                            if (worldPt.x < bmin.x || worldPt.x > bmax.x ||
                                worldPt.y < bmin.y || worldPt.y > bmax.y ||
                                worldPt.z < bmin.z || worldPt.z > bmax.z)
                                continue;
                        }

                        Vec3 nrm;
                        float surfDist;
                        if (surf.kind == ArenaSurface::kPlane) {
                            nrm = surf.normal;
                            surfDist = v3_dot(worldPt, nrm) - surf.offset;
                        } else {
                            float axialT;
                            Vec3 radial = projectRadial(worldPt, surf.center, surf.axis, axialT);
                            float q1 = v3_dot(radial, surf.quadN1);
                            float q2 = v3_dot(radial, surf.quadN2);
                            if (q1 < 0.f || q2 < 0.f) continue;
                            float radialDist = v3_length(radial);
                            if (radialDist < 1e-4f) continue;
                            nrm = radial * (-1.f / radialDist);
                            surfDist = surf.radius - radialDist;
                        }

                        float gap = surfDist - meshMarginSum;
                        // <= so the LAST minimum wins on exact symmetric ties,
                        // matching the reference GJK pick observed in traces.
                        if (gap <= bestGap) {
                            bestGap = gap;
                            bestNormal = nrm;
                            // GJK reports pointOnB on the triangle's MARGIN
                            // surface (triangle + margin along the normal).
                            bestWorldPt = worldPt - nrm * (surfDist - MESH_COLLISION_MARGIN);
                            bestLocalPt = localPt;
                        }
                    }
                }
            }

            if (bestGap < breakingThreshold) {
                // localPointA: contact on the car (posB + n*dist in car space)
                Vec3 posA = bestWorldPt + bestNormal * bestGap;
                Vec3 localA = rotmat_dot_vec(car.rotMat, posA - car.pos);
                device_manifold_add_point(car, breakingThreshold, localA, bestWorldPt, bestNormal, bestGap, 0);

                car.worldContactHasContact = true;
                car.worldContactNormal = bestNormal;
            }
        }

        // btStaticPlaneShape path (support point, box margin only).
        if (surf.isStaticPlane) {
            Vec3 nrm = surf.normal;
            Vec3 dirLocal = rotmat_dot_vec(car.rotMat, -nrm);
            float sx = (dirLocal.x >= 0.f) ? 1.f : -1.f;
            float sy = (dirLocal.y >= 0.f) ? 1.f : -1.f;
            float sz = (dirLocal.z >= 0.f) ? 1.f : -1.f;
            Vec3 supportLocal = offset + v3(implicitHalf.x * sx, implicitHalf.y * sy, implicitHalf.z * sz);
            Vec3 supportWorld = car.pos + rotmat_mul_vec(car.rotMat, supportLocal) + nrm * (-safeMargin);

            float gap = v3_dot(supportWorld, nrm) - surf.offset;
            if (gap < breakingThreshold) {
                Vec3 posB = supportWorld - nrm * gap;
                Vec3 localA = rotmat_dot_vec(car.rotMat, supportWorld - car.pos);
                device_manifold_add_point(car, breakingThreshold, localA, posB, nrm, gap, 1);

                car.worldContactHasContact = true;
                car.worldContactNormal = nrm;
            }
        }
    }
}

// ---- Car vs ball (sphere-box) ----------------------------------------------
// Returns true and fills `c` when within the add threshold. Also performs the
// _BtCallback_OnCarBallCollision side effects (hit info + extra impulse).
__device__ inline bool device_gen_car_ball_contact(
    GpuCarState& car, GpuBallState& ball,
    float ballRadius, float ballHitExtraForceScale,
    uint64_t tickCount,
    GenContact& c
) {
    using namespace PhysConst;

    if (car.isDemoed) return false;

    // btSphereBoxCollisionAlgorithm::getSphereDistance — closest point on the
    // IMPLICIT box (getHalfExtentsWithoutMargin); intersection distance is
    // sphereRadius + box safe margin; pointOnBox = closest + normal * margin.
    Vec3 localBallPos = rotmat_dot_vec(car.rotMat, ball.pos - car.pos) - car.config.hitboxPosOffset;
    Vec3 halfExt = car.config.hitboxSize * 0.5f - v3(CONVEX_DISTANCE_MARGIN_UU, CONVEX_DISTANCE_MARGIN_UU, CONVEX_DISTANCE_MARGIN_UU);
    float boxMargin = car_box_safe_margin(car.config);

    Vec3 closest;
    closest.x = rs_clamp(localBallPos.x, -halfExt.x, halfExt.x);
    closest.y = rs_clamp(localBallPos.y, -halfExt.y, halfExt.y);
    closest.z = rs_clamp(localBallPos.z, -halfExt.z, halfExt.z);

    float intersectionDist = ballRadius + boxMargin;
    float contactDist = intersectionDist + BALL_CONTACT_ADD_THRESHOLD;
    Vec3 diff = localBallPos - closest;
    float distSq = v3_length_sq(diff);

    if (distSq > contactDist * contactDist) return false;

    float dist;
    Vec3 localNormal;
    if (distSq <= bts::SIMD_EPSILON_F) {
        // Sphere center inside the box: project onto the closest face
        // (btSphereBoxCollisionAlgorithm::getSpherePenetration)
        float faceDist = halfExt.x - localBallPos.x;
        float minDist = faceDist;
        closest = localBallPos; closest.x = halfExt.x;
        localNormal = v3(1.f, 0.f, 0.f);

        faceDist = halfExt.x + localBallPos.x;
        if (faceDist < minDist) {
            minDist = faceDist;
            closest = localBallPos; closest.x = -halfExt.x;
            localNormal = v3(-1.f, 0.f, 0.f);
        }
        faceDist = halfExt.y - localBallPos.y;
        if (faceDist < minDist) {
            minDist = faceDist;
            closest = localBallPos; closest.y = halfExt.y;
            localNormal = v3(0.f, 1.f, 0.f);
        }
        faceDist = halfExt.y + localBallPos.y;
        if (faceDist < minDist) {
            minDist = faceDist;
            closest = localBallPos; closest.y = -halfExt.y;
            localNormal = v3(0.f, -1.f, 0.f);
        }
        faceDist = halfExt.z - localBallPos.z;
        if (faceDist < minDist) {
            minDist = faceDist;
            closest = localBallPos; closest.z = halfExt.z;
            localNormal = v3(0.f, 0.f, 1.f);
        }
        faceDist = halfExt.z + localBallPos.z;
        if (faceDist < minDist) {
            minDist = faceDist;
            closest = localBallPos; closest.z = -halfExt.z;
            localNormal = v3(0.f, 0.f, -1.f);
        }
        dist = -minDist;
    } else {
        dist = sqrtf(distSq);
        localNormal = diff / dist;
    }

    Vec3 contactNormal = rotmat_mul_vec(car.rotMat, localNormal);  // from box toward sphere
    Vec3 pointOnBoxLocal = closest + localNormal * boxMargin;

    c.normal = contactNormal;
    c.distance = dist - intersectionDist;
    c.posWorld = car.pos + rotmat_mul_vec(car.rotMat, pointOnBoxLocal + car.config.hitboxPosOffset);

    {
        // _BulletContactAddedCallback fires whenever a manifold point is
        // added/updated — including non-penetrating cushion contacts within
        // the breaking threshold, NOT just at overlap.
        car.ballHitValid = true;
        car.ballHitTickCount = tickCount;

        // Extra car-ball impulse (Arena::_BtCallback_OnCarBallCollision)
        if (tickCount > car.lastExtraImpulseTick + 1 || car.lastExtraImpulseTick > tickCount) {
            car.lastExtraImpulseTick = tickCount;

            Vec3 relPos = ball.pos - car.pos;
            Vec3 relVelFull = ball.vel - car.vel;
            float relSpeed = rs_min(v3_length(relVelFull), BALL_CAR_EXTRA_IMPULSE_MAXDELTAVEL);

            if (relSpeed > 0.f) {
                Vec3 hitDir = v3_safe_normalize(relPos * v3(1.f, 1.f, BALL_CAR_EXTRA_IMPULSE_Z_SCALE));
                Vec3 carFwd = car.rotMat.forward;
                Vec3 fwdAdj = carFwd * (v3_dot(hitDir, carFwd) * (1.f - BALL_CAR_EXTRA_IMPULSE_FORWARD_SCALE));
                hitDir = v3_safe_normalize(hitDir - fwdAdj);

                Vec3 addedVel = hitDir * relSpeed * curve_ball_car_impulse(relSpeed) * ballHitExtraForceScale;
                ball.velocityImpulseCache += addedVel;
            }
        }
    }
    return true;
}

// ---- Car vs car (single-point manifold from SAT) ----------------------------
// Fills `c` (normal from car1 toward car2) and runs the bump/demo callback.
__device__ inline bool device_gen_car_car_contact(
    GpuCarState& car1, GpuCarState& car2,
    float bumpForceScale, float bumpCooldownTime,
    GenContact& c
) {
    using namespace PhysConst;

    if (car1.isDemoed || car2.isDemoed) return false;

    float diag1 = v3_length(car1.config.hitboxSize) * 0.5f;
    float diag2 = v3_length(car2.config.hitboxSize) * 0.5f;
    float maxDist = diag1 + diag2;
    if (v3_dist_sq(car1.pos, car2.pos) > maxDist * maxDist) return false;

    Vec3 cA = car1.pos + rotmat_mul_vec(car1.rotMat, car1.config.hitboxPosOffset);
    Vec3 cB = car2.pos + rotmat_mul_vec(car2.rotMat, car2.config.hitboxPosOffset);
    Vec3 axA[3] = { car1.rotMat.forward, car1.rotMat.right, car1.rotMat.up };
    Vec3 axB[3] = { car2.rotMat.forward, car2.rotMat.right, car2.rotMat.up };
    Vec3 hA = car1.config.hitboxSize * 0.5f;
    Vec3 hB = car2.config.hitboxSize * 0.5f;

    Vec3 contactNormal;
    float overlap;
    if (!obbVsObbSAT(cA, axA, hA, cB, axB, hB, contactNormal, overlap)) return false;
    if (overlap <= 0.f) return false;

    // Single representative contact point: midpoint of the two deepest
    // support points along the contact normal.
    Vec3 suppA = cA;
    Vec3 suppB = cB;
    float hAi[3] = { hA.x, hA.y, hA.z };
    float hBi[3] = { hB.x, hB.y, hB.z };
    for (int i = 0; i < 3; i++) {
        float dA = v3_dot(axA[i], contactNormal);
        suppA += axA[i] * (hAi[i] * ((dA >= 0.f) ? 1.f : -1.f));
        float dB = v3_dot(axB[i], contactNormal);
        suppB += axB[i] * (hBi[i] * ((dB >= 0.f) ? -1.f : 1.f));
    }
    c.posWorld = (suppA + suppB) * 0.5f;
    c.normal = contactNormal;     // from car1 toward car2
    c.distance = -overlap;

    // ---- Bump / demo logic (Arena::_BtCallback_OnCarCarCollision) ----
    for (int dir = 0; dir < 2; dir++) {
        GpuCarState& attacker = (dir == 0) ? car1 : car2;
        GpuCarState& victim   = (dir == 0) ? car2 : car1;

        Vec3 deltaPos = victim.pos - attacker.pos;
        if (v3_dot(attacker.vel, deltaPos) <= 0.f) continue;

        if (attacker.carContactOtherID == victim.id && attacker.carContactCooldownTimer > 0.f)
            continue;

        Vec3 velDir = v3_safe_normalize(attacker.vel);
        Vec3 dirToOther = v3_safe_normalize(deltaPos);

        float speedTowards = v3_dot(attacker.vel, dirToOther);
        float otherAwaySpeed = v3_dot(victim.vel, velDir);

        if (speedTowards <= otherAwaySpeed) continue;

        // The reference checks the manifold's local point on the attacker.
        Vec3 localHitPt = rotmat_dot_vec(attacker.rotMat, c.posWorld - attacker.pos);
        if (localHitPt.x < BUMP_MIN_FORWARD_DIST) continue;

        bool isDemo = attacker.isSupersonic && (attacker.team != victim.team);

        if (isDemo) {
            victim.isDemoed = true;
            victim.demoRespawnTimer = DEMO_RESPAWN_TIME;
        } else {
            bool groundHit = victim.isOnGround;
            float baseScale = groundHit
                ? curve_bump_vel_ground(speedTowards)
                : curve_bump_vel_air(speedTowards);

            Vec3 hitUpDir = victim.isOnGround ? victim.rotMat.up : v3(0, 0, 1);
            Vec3 bumpImpulse = (velDir * baseScale + hitUpDir * curve_bump_upward(speedTowards))
                             * bumpForceScale;

            victim.velocityImpulseCache += bumpImpulse;
        }

        attacker.carContactOtherID = victim.id;
        attacker.carContactCooldownTimer = bumpCooldownTime;
    }
    return true;
}

// ============================================================================
// Full bullet-exact world step for one arena.
// Call AFTER the car pre-tick updates (which apply impulses directly and fill
// the per-car force/torque accumulators), with positions NOT yet integrated.
// ============================================================================

// rows: cars*8 (car-world) + cars (car-ball) + pairs (car-car) + ball special + aggregate
constexpr int MAX_NORMAL_ROWS =
    MAX_CARS_PER_ARENA * MAX_CW_CONTACTS + MAX_CARS_PER_ARENA +
    (MAX_CARS_PER_ARENA * (MAX_CARS_PER_ARENA - 1)) / 2 + MAX_BW_CONTACTS + 1;
constexpr int MAX_FRICTION_ROWS = MAX_NORMAL_ROWS;  // special rows have none; safe upper bound

// Row bookkeeping for warm-start writeback
enum class RowKind : uint8_t { CarWorld, CarBall, CarCar, BallSpecial, BallAggregate };

__device__ void device_bullet_world_step(
    GpuCarState* cars, int numCars,
    GpuBallState& ball,
    const ArenaSurface* surfaces, int numSurfaces,
    const MeshGridView& meshGrid,
    const Vec3* carAccelAccum,      // per car, UU/s^2 (gravity NOT included)
    const Vec3* carAngAccelAccum,   // per car, rad/s^2
    Vec3 gravity,                   // UU/s^2
    float ballDrag,
    float ballRadius, float ballInvInertia,
    float ballWorldFrictionCombined, float ballWorldRestitutionCombined,
    float carWorldFriction, float carWorldRestitution,
    float ballHitExtraForceScale, float bumpForceScale, float bumpCooldownTime,
    uint64_t tickCount,
    float dt
) {
    using namespace PhysConst;

    const float invDt = 1.f / dt;

    // Ball zero-velocity sleeping (Arena::Step): a ball with exactly zero
    // linear and angular velocity is ISLAND_SLEEPING — no gravity, no
    // damping, no integration — until a car contact wakes its island.
    const bool ballAsleep = v3_is_zero(ball.vel) && v3_is_zero(ball.angVel);

    // ------------------------------------------------------------------
    // 1. Damping (btDiscreteDynamicsWorld::predictUnconstrainedMotion ->
    //    btRigidBody::applyDamping). Ball only; cars have zero damping.
    // ------------------------------------------------------------------
    if (ballDrag > 0.f && !ballAsleep)
        ball.vel *= powf(rs_clamp(1.f - ballDrag, 0.f, 1.f), dt);

    // ------------------------------------------------------------------
    // 2. Solver bodies (BT units)
    // ------------------------------------------------------------------
    bts::SolverBody bodies[MAX_CARS_PER_ARENA + 2];
    const int ballIdx = numCars;
    const int fixedIdx = numCars + 1;

    for (int i = 0; i < numCars; i++) {
        bts::SolverBody& b = bodies[i];
        GpuCarState& car = cars[i];
        b.dLin = v3_zero(); b.dAng = v3_zero();
        b.pushVel = v3_zero(); b.turnVel = v3_zero();
        if (car.isDemoed) {
            b.linVel = v3_zero(); b.angVel = v3_zero();
            b.extForceImpulse = v3_zero(); b.extTorqueImpulse = v3_zero();
            b.invMass = 0.f;
            b.invInertiaWorld = {v3_zero(), v3_zero(), v3_zero()};
            continue;
        }
        b.linVel = car.vel * bts::UU_TO_BT;
        b.angVel = car.angVel;
        b.extForceImpulse = (gravity + carAccelAccum[i]) * (bts::UU_TO_BT * dt);
        b.extTorqueImpulse = carAngAccelAccum[i] * dt;
        b.invMass = 1.f / CAR_MASS;
        b.invInertiaWorld = bts::make_inv_inertia_world(car.rotMat, car.invLocalInertia);
    }
    {
        bts::SolverBody& b = bodies[ballIdx];
        b.linVel = ball.vel * bts::UU_TO_BT;
        b.angVel = ball.angVel;
        // Sleeping bodies are skipped by applyGravity().
        b.extForceImpulse = ballAsleep ? v3_zero() : gravity * (bts::UU_TO_BT * dt);
        b.extTorqueImpulse = v3_zero();
        b.dLin = v3_zero(); b.dAng = v3_zero();
        b.pushVel = v3_zero(); b.turnVel = v3_zero();
        b.invMass = 1.f / BALL_MASS;
        b.invInertiaWorld = {
            {ballInvInertia, 0.f, 0.f},
            {0.f, ballInvInertia, 0.f},
            {0.f, 0.f, ballInvInertia}};
    }
    {
        bts::SolverBody& b = bodies[fixedIdx];
        b.linVel = v3_zero(); b.angVel = v3_zero();
        b.extForceImpulse = v3_zero(); b.extTorqueImpulse = v3_zero();
        b.dLin = v3_zero(); b.dAng = v3_zero();
        b.pushVel = v3_zero(); b.turnVel = v3_zero();
        b.invMass = 0.f;
        b.invInertiaWorld = {v3_zero(), v3_zero(), v3_zero()};
    }

    // ------------------------------------------------------------------
    // 3. Collision detection (current positions) + constraint rows
    // ------------------------------------------------------------------
    bts::ContactRow normalRows[MAX_NORMAL_ROWS];
    bts::ContactRow frictionRows[MAX_FRICTION_ROWS];
    int nNormal = 0, nFriction = 0;

    // bookkeeping for warm-start writeback
    RowKind rowKind[MAX_NORMAL_ROWS];
    uint8_t rowCarIdx[MAX_NORMAL_ROWS];
    uint8_t rowSlot[MAX_NORMAL_ROWS];

    uint8_t newBallContact[MAX_CARS_PER_ARENA] = {};
    uint8_t newCarContact[MAX_CARS_PER_ARENA] = {};

    // --- car-world (persistent manifold) ---
    for (int i = 0; i < numCars; i++) {
        GpuCarState& car = cars[i];
        car.worldContactHasContact = false;
        device_update_car_world_manifold(car, surfaces, numSurfaces);

        for (int k = 0; k < 4 && nNormal < MAX_NORMAL_ROWS; k++) {
            const auto& pt = car.worldManifold[k];
            if (!pt.valid) continue;

            Vec3 posA = car.pos + rotmat_mul_vec(car.rotMat, pt.localPointA);
            float distance = v3_dot(posA - pt.worldPointB, pt.normal);
            Vec3 rel1 = (posA - car.pos) * bts::UU_TO_BT;
            Vec3 rel2 = pt.worldPointB * bts::UU_TO_BT;

            int rowIdx = nNormal;
            bts::setup_contact_row(
                normalRows[nNormal], bodies, i, -1,
                pt.normal, rel1, rel2,
                distance * bts::UU_TO_BT,
                carWorldFriction, carWorldRestitution,
                pt.appliedImpulse, invDt, false);
            rowKind[nNormal] = RowKind::CarWorld;
            rowCarIdx[nNormal] = (uint8_t)i;
            rowSlot[nNormal] = (uint8_t)k;
            nNormal++;

            Vec3 fdir = bts::friction_direction(bodies, i, -1, pt.normal, rel1, rel2);
            bts::setup_friction_row(
                frictionRows[nFriction], bodies, i, -1,
                fdir, rel1, rel2, carWorldFriction, rowIdx);
            nFriction++;
        }
    }

    // --- car-ball ---
    bool ballHasCarContact = false;
    for (int i = 0; i < numCars && nNormal < MAX_NORMAL_ROWS; i++) {
        GpuCarState& car = cars[i];
        GenContact c;
        if (!device_gen_car_ball_contact(car, ball, ballRadius, ballHitExtraForceScale, tickCount, c))
            continue;
        ballHasCarContact = true;

        Vec3 posA = c.posWorld + c.normal * c.distance;
        Vec3 rel1 = (posA - ball.pos) * bts::UU_TO_BT;       // ball is body A (sphere)
        Vec3 rel2 = (c.posWorld - car.pos) * bts::UU_TO_BT;  // car is body B (box)

        float warm = car.wsBallContact ? car.wsBallImpulse : 0.f;

        int rowIdx = nNormal;
        bts::setup_contact_row(
            normalRows[nNormal], bodies, ballIdx, i,
            c.normal, rel1, rel2,
            c.distance * bts::UU_TO_BT,
            CARBALL_COLLISION_FRICTION, CARBALL_COLLISION_RESTITUTION,
            warm, invDt, false);
        rowKind[nNormal] = RowKind::CarBall;
        rowCarIdx[nNormal] = (uint8_t)i;
        rowSlot[nNormal] = 0;
        nNormal++;
        newBallContact[i] = 1;

        Vec3 fdir = bts::friction_direction(bodies, ballIdx, i, c.normal, rel1, rel2);
        bts::setup_friction_row(
            frictionRows[nFriction], bodies, ballIdx, i,
            fdir, rel1, rel2, CARBALL_COLLISION_FRICTION, rowIdx);
        nFriction++;
    }

    // --- car-car ---
    for (int i = 0; i < numCars; i++) {
        for (int j = i + 1; j < numCars && nNormal < MAX_NORMAL_ROWS; j++) {
            GenContact c;
            if (!device_gen_car_car_contact(cars[i], cars[j], bumpForceScale, bumpCooldownTime, c))
                continue;

            // normal points from car i toward car j => body A = j, body B = i
            Vec3 posA = c.posWorld + c.normal * c.distance;
            Vec3 rel1 = (posA - cars[j].pos) * bts::UU_TO_BT;
            Vec3 rel2 = (c.posWorld - cars[i].pos) * bts::UU_TO_BT;

            float warm = cars[i].wsCarContact ? cars[i].wsCarImpulse : 0.f;

            int rowIdx = nNormal;
            bts::setup_contact_row(
                normalRows[nNormal], bodies, j, i,
                c.normal, rel1, rel2,
                c.distance * bts::UU_TO_BT,
                CARCAR_COLLISION_FRICTION, CARCAR_COLLISION_RESTITUTION,
                warm, invDt, false);
            rowKind[nNormal] = RowKind::CarCar;
            rowCarIdx[nNormal] = (uint8_t)i;
            rowSlot[nNormal] = (uint8_t)j;
            nNormal++;
            newCarContact[i] = 1;

            Vec3 fdir = bts::friction_direction(bodies, j, i, c.normal, rel1, rel2);
            bts::setup_friction_row(
                frictionRows[nFriction], bodies, j, i,
                fdir, rel1, rel2, CARCAR_COLLISION_FRICTION, rowIdx);
            nFriction++;
        }
    }

    // A sleeping ball with no car contact is in a sleeping island: its
    // world contacts are not solved and it is not integrated.
    const bool ballActive = !ballAsleep || ballHasCarContact;

    // --- ball-world (RocketSim "special" path) ---
    if (numSurfaces > 0 && ballActive) {
        GenContact bw[MAX_BW_CONTACTS];
        int n = device_gen_ball_world_contacts(ball, surfaces, numSurfaces, meshGrid, ballRadius, bw, MAX_BW_CONTACTS);

        if (n > 0) {
            // Real contacts: split-impulse-only rows (m_isSpecial).
            Vec3 totalNormal = v3_zero();
            float totalDist = 0.f;

            for (int k = 0; k < n && nNormal < MAX_NORMAL_ROWS; k++) {
                const GenContact& c = bw[k];
                Vec3 posA = c.posWorld + c.normal * c.distance;
                Vec3 rel1 = (posA - ball.pos) * bts::UU_TO_BT;
                Vec3 rel2 = c.posWorld * bts::UU_TO_BT;

                bts::setup_contact_row(
                    normalRows[nNormal], bodies, ballIdx, -1,
                    c.normal, rel1, rel2,
                    c.distance * bts::UU_TO_BT,
                    ballWorldFrictionCombined, ballWorldRestitutionCombined,
                    0.f, invDt, true /* special */);
                rowKind[nNormal] = RowKind::BallSpecial;
                rowCarIdx[nNormal] = 0;
                rowSlot[nNormal] = 0;
                nNormal++;

                totalNormal += c.normal;
                totalDist += v3_length(rel1);

                // Ball::_OnWorldCollision is a no-op in SOCCAR.
            }

            // Aggregated constraint (convertContactSpecial): average distance
            // and average (NON-normalized) normal; contact placed at
            // rel_pos1 = normal * -distance; solved like a regular contact.
            if (nNormal < MAX_NORMAL_ROWS) {
                float invNum = 1.f / (float)n;
                float distance = totalDist * invNum;
                Vec3 normal = totalNormal * invNum;

                Vec3 rel1 = normal * -distance;
                Vec3 rel2 = v3_zero();

                int rowIdx = nNormal;
                bts::setup_contact_row(
                    normalRows[nNormal], bodies, ballIdx, -1,
                    normal, rel1, rel2,
                    distance,  // positive -> velocity-only constraint
                    ballWorldFrictionCombined, ballWorldRestitutionCombined,
                    0.f, invDt, false);
                rowKind[nNormal] = RowKind::BallAggregate;
                rowCarIdx[nNormal] = 0;
                rowSlot[nNormal] = 0;
                nNormal++;

                Vec3 fdir = bts::friction_direction(bodies, ballIdx, -1, normal, rel1, rel2);
                bts::setup_friction_row(
                    frictionRows[nFriction], bodies, ballIdx, -1,
                    fdir, rel1, rel2, ballWorldFrictionCombined, rowIdx);
                nFriction++;
            }
        }
    }

    // ------------------------------------------------------------------
    // 4. Solve
    // ------------------------------------------------------------------
    bts::SolverContext ctx;
    ctx.bodies = bodies;
    ctx.fixedBodyIdx = fixedIdx;
    ctx.contactRows = normalRows;
    ctx.numContactRows = nNormal;
    ctx.frictionRows = frictionRows;
    ctx.numFrictionRows = nFriction;
    bts::solve_group(ctx, dt);

    // ------------------------------------------------------------------
    // 5. Writeback + integrate transforms
    // ------------------------------------------------------------------
    for (int i = 0; i < numCars; i++) {
        GpuCarState& car = cars[i];
        if (car.isDemoed)
            continue;

        Vec3 posBT = car.pos * bts::UU_TO_BT;
        RotMat basis = car.rotMat;
        Vec3 linVel, angVel;
        bts::body_writeback(bodies[i], dt, linVel, angVel, posBT, basis, false);

        // integrateTransforms
        posBT += linVel * dt;
        basis = bts::bt_integrate_rotation(basis, angVel, dt);

        car.pos = posBT * bts::BT_TO_UU;
        car.vel = linVel * bts::BT_TO_UU;
        car.angVel = angVel;
        car.rotMat = basis;
    }
    if (ballActive) {
        Vec3 posBT = ball.pos * bts::UU_TO_BT;
        RotMat basis = ball.rotMat;
        Vec3 linVel, angVel;
        bts::body_writeback(bodies[ballIdx], dt, linVel, angVel, posBT, basis, true /* noRot */);

        posBT += linVel * dt;
        // noRot: rotation updates skipped entirely

        ball.pos = posBT * bts::BT_TO_UU;
        ball.vel = linVel * bts::BT_TO_UU;
        ball.angVel = angVel;
    }

    // ------------------------------------------------------------------
    // 6. Persist warm-start impulses (writeBackContacts)
    // ------------------------------------------------------------------
    for (int i = 0; i < numCars; i++) {
        cars[i].wsBallContact = 0;
        cars[i].wsCarContact = 0;
    }
    for (int r = 0; r < nNormal; r++) {
        switch (rowKind[r]) {
        case RowKind::CarWorld: {
            GpuCarState& car = cars[rowCarIdx[r]];
            car.worldManifold[rowSlot[r]].appliedImpulse = normalRows[r].appliedImpulse;
            break;
        }
        case RowKind::CarBall: {
            GpuCarState& car = cars[rowCarIdx[r]];
            car.wsBallImpulse = normalRows[r].appliedImpulse;
            car.wsBallContact = newBallContact[rowCarIdx[r]];
            break;
        }
        case RowKind::CarCar: {
            GpuCarState& car = cars[rowCarIdx[r]];
            car.wsCarImpulse = normalRows[r].appliedImpulse;
            car.wsCarContact = newCarContact[rowCarIdx[r]];
            break;
        }
        default:
            break;  // ball special/aggregate rows are not warm-started
        }
    }
}

} // namespace rsc
