#pragma once

// ============================================================================
// MeshCollision.cuh — real arena collision-mesh queries on the GPU.
//
// The reference arena uses the actual triangle meshes from collision_meshes/
// (CMF files, vertices already in BT units) plus 4 static planes. This module
// provides:
//   - device sphere-vs-triangle contact generation transcribed from
//     RocketSim's patched SphereTriangleDetector (VirxEC closest-point path)
//   - device ray-vs-triangle for suspension raycasts (both-sided, closest hit)
//   - a uniform grid over the arena for triangle culling
//
// All math here is in BT units (1 BT = 50 UU) to match the reference
// operation-for-operation; callers convert at the boundary.
// ============================================================================

#include "CudaMath.cuh"

namespace rsc {

struct MeshTriangle {
    Vec3 v0, v1, v2;  // BT units
    int meshId;       // which CMF file (manifolds are per mesh rigid body)
};

struct MeshGridConfig {
    Vec3 minPos;       // BT
    Vec3 invCellSize;  // 1 / cellSize (BT)
    int nx, ny, nz;
};

struct MeshGridView {
    MeshGridConfig cfg;
    const MeshTriangle* tris;     // [numTris]
    const int* cellStart;         // [numCells + 1] CSR offsets
    const int* cellTris;          // triangle indices per cell
    int numTris;
};

__device__ __forceinline__ int mesh_grid_cell_index(const MeshGridConfig& g, int cx, int cy, int cz) {
    return (cz * g.ny + cy) * g.nx + cx;
}

__device__ __forceinline__ int mesh_grid_clamp(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// ----------------------------------------------------------------------------
// closestPointTriangle (embree variant used by RocketSim's SphereTriangleDetector)
// ----------------------------------------------------------------------------
__device__ inline Vec3 mesh_closest_point_triangle(Vec3 p, Vec3 a, Vec3 b, Vec3 c) {
    Vec3 ab = b - a;
    Vec3 ac = c - a;
    Vec3 ap = p - a;

    float d1 = v3_dot(ab, ap);
    float d2 = v3_dot(ac, ap);
    if (d1 <= 0.f && d2 <= 0.f) return a;

    Vec3 bp = p - b;
    float d3 = v3_dot(ab, bp);
    float d4 = v3_dot(ac, bp);
    if (d3 >= 0.f && d4 <= d3) return b;

    Vec3 cp = p - c;
    float d5 = v3_dot(ab, cp);
    float d6 = v3_dot(ac, cp);
    if (d6 >= 0.f && d5 <= d6) return c;

    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.f && d1 >= 0.f && d3 <= 0.f) {
        float v = d1 / (d1 - d3);
        return a + ab * v;
    }

    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.f && d2 >= 0.f && d6 <= 0.f) {
        float v = d2 / (d2 - d6);
        return a + ac * v;
    }

    float va = d3 * d6 - d5 * d4;
    if (va <= 0.f && (d4 - d3) >= 0.f && (d5 - d6) >= 0.f) {
        float v = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + (c - b) * v;
    }

    float denom = 1.f / (va + vb + vc);
    float v = vb * denom;
    float w = vc * denom;
    return a + ab * v + ac * w;
}

// pointInTriangle (fast variant from RocketSim's SphereTriangleDetector)
__device__ inline bool mesh_point_in_triangle(Vec3 p, Vec3 p0, Vec3 p1, Vec3 p2) {
    Vec3 u = p1 - p0;
    Vec3 v = p2 - p0;
    Vec3 n = v3_cross(u, v);
    float nLenSq = v3_dot(n, n);

    Vec3 w = p - p0;

    float gamma = v3_dot(v3_cross(u, w), n) / nLenSq;
    float beta = v3_dot(v3_cross(w, v), n) / nLenSq;
    float alpha = 1.f - gamma - beta;

    return ((0.f <= alpha) && (alpha <= 1.f) &&
            (0.f <= beta) && (beta <= 1.f) &&
            (0.f <= gamma) && (gamma <= 1.f));
}

// ----------------------------------------------------------------------------
// SphereTriangleDetector::collide transcription. All BT units.
// Returns true with point/normal/depth like the reference:
//   depth = -(radius - distance)  (negative = penetrating)
//   normal points from the triangle toward the sphere center.
// ----------------------------------------------------------------------------
__device__ inline bool mesh_sphere_triangle_collide(
    Vec3 sphereCenter, float radius, float contactBreakingThreshold,
    const MeshTriangle& tri,
    Vec3& point, Vec3& resultNormal, float& depth
) {
    constexpr float SIMD_EPSILON_F = 1.19209290e-07f;

    float radiusWithThreshold = radius + contactBreakingThreshold;

    Vec3 normal = v3_cross(tri.v1 - tri.v0, tri.v2 - tri.v0);
    float l2 = v3_length_sq(normal);
    bool hasContact = false;
    bool faceContact = false;
    Vec3 contactPoint;

    if (l2 >= SIMD_EPSILON_F * SIMD_EPSILON_F) {
        normal = normal / sqrtf(l2);

        Vec3 p1ToCentre = sphereCenter - tri.v0;
        float distanceFromPlane = v3_dot(p1ToCentre, normal);

        if (distanceFromPlane < 0.f) {
            distanceFromPlane *= -1.f;
            normal = -normal;
        }

        bool isInsideContactPlane = distanceFromPlane < radiusWithThreshold;

        if (isInsideContactPlane) {
            if (mesh_point_in_triangle(sphereCenter, tri.v0, tri.v1, tri.v2)) {
                hasContact = true;
                faceContact = true;
                contactPoint = sphereCenter - normal * distanceFromPlane;
            } else {
                float contactCapsuleRadiusSqr = radiusWithThreshold * radiusWithThreshold;
                Vec3 nearestOnEdge = mesh_closest_point_triangle(sphereCenter, tri.v0, tri.v1, tri.v2);
                float distanceSqr = v3_length_sq(nearestOnEdge - sphereCenter);
                if (distanceSqr < contactCapsuleRadiusSqr) {
                    hasContact = true;
                    contactPoint = nearestOnEdge;
                }
            }
        }
    }

    if (hasContact) {
        Vec3 contactToCentre = sphereCenter - contactPoint;
        float distanceSqr = v3_length_sq(contactToCentre);

        if (distanceSqr < radiusWithThreshold * radiusWithThreshold) {
            if (distanceSqr > SIMD_EPSILON_F) {
                float distance = sqrtf(distanceSqr);
                resultNormal = contactToCentre / distance;
                point = contactPoint;
                depth = -(radius - distance);
            } else {
                resultNormal = normal;
                point = contactPoint;
                depth = -radius;
            }

            // btAdjustInternalEdgeContacts (called from RocketSim's contact-
            // added callback; the arena meshes have triangle info maps):
            // edge/vertex contacts on internal edges get their normal snapped
            // to the triangle FACE normal; the distance is kept.
            if (!faceContact) {
                resultNormal = normal;
            }
            return true;
        }
    }
    return false;
}

// ----------------------------------------------------------------------------
// Ray vs triangle (both-sided, like bullet's btTriangleRaycastCallback without
// backface filtering). Returns hit fraction in [0, 1] relative to (from, to),
// or -1 on miss. normalOut faces the ray origin.
// ----------------------------------------------------------------------------
__device__ inline float mesh_ray_triangle(
    Vec3 from, Vec3 to,
    const MeshTriangle& tri,
    Vec3& normalOut
) {
    Vec3 v10 = tri.v1 - tri.v0;
    Vec3 v20 = tri.v2 - tri.v0;
    Vec3 triangleNormal = v3_cross(v10, v20);

    float dist = v3_dot(tri.v0, triangleNormal);
    float dist_a = v3_dot(triangleNormal, from) - dist;
    float dist_b = v3_dot(triangleNormal, to) - dist;

    if (dist_a * dist_b >= 0.f)
        return -1.f;  // same side: no crossing

    float proj_length = dist_a - dist_b;
    float distance = dist_a / proj_length;

    Vec3 point = from + (to - from) * distance;

    // Edge tests (btTriangleRaycastCallback style)
    {
        Vec3 cp0 = v3_cross(tri.v1 - point, tri.v2 - point);
        if (v3_dot(cp0, triangleNormal) < 0.f) return -1.f;
        Vec3 cp1 = v3_cross(tri.v2 - point, tri.v0 - point);
        if (v3_dot(cp1, triangleNormal) < 0.f) return -1.f;
        Vec3 cp2 = v3_cross(tri.v0 - point, tri.v1 - point);
        if (v3_dot(cp2, triangleNormal) < 0.f) return -1.f;
    }

    normalOut = (dist_a > 0.f) ? v3_normalize(triangleNormal) : -v3_normalize(triangleNormal);
    return distance;
}

// Iterate grid cells overlapped by an AABB (BT units), calling the triangle
// visitor for each triangle index. Visitor signature: bool visit(int triIdx)
// — return value ignored; dedup via the per-query stamp is the caller's job
// (triangles can span multiple cells).
template <typename F>
__device__ inline void mesh_grid_query_aabb(
    const MeshGridView& grid, Vec3 aabbMin, Vec3 aabbMax, F&& visit
) {
    const MeshGridConfig& g = grid.cfg;
    int cx0 = mesh_grid_clamp((int)floorf((aabbMin.x - g.minPos.x) * g.invCellSize.x), 0, g.nx - 1);
    int cy0 = mesh_grid_clamp((int)floorf((aabbMin.y - g.minPos.y) * g.invCellSize.y), 0, g.ny - 1);
    int cz0 = mesh_grid_clamp((int)floorf((aabbMin.z - g.minPos.z) * g.invCellSize.z), 0, g.nz - 1);
    int cx1 = mesh_grid_clamp((int)floorf((aabbMax.x - g.minPos.x) * g.invCellSize.x), 0, g.nx - 1);
    int cy1 = mesh_grid_clamp((int)floorf((aabbMax.y - g.minPos.y) * g.invCellSize.y), 0, g.ny - 1);
    int cz1 = mesh_grid_clamp((int)floorf((aabbMax.z - g.minPos.z) * g.invCellSize.z), 0, g.nz - 1);

    for (int cz = cz0; cz <= cz1; cz++) {
        for (int cy = cy0; cy <= cy1; cy++) {
            for (int cx = cx0; cx <= cx1; cx++) {
                int cell = mesh_grid_cell_index(g, cx, cy, cz);
                int begin = grid.cellStart[cell];
                int end = grid.cellStart[cell + 1];
                for (int k = begin; k < end; k++)
                    visit(grid.cellTris[k]);
            }
        }
    }
}

} // namespace rsc
