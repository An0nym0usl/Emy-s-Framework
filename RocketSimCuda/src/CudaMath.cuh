#pragma once

#include "GpuTypes.cuh"
#include <math.h>

namespace rsc {


__device__ __forceinline__ float rs_min(float a, float b) { return fminf(a, b); }
__device__ __forceinline__ float rs_max(float a, float b) { return fmaxf(a, b); }
__device__ __forceinline__ float rs_clamp(float v, float lo, float hi) { return fminf(fmaxf(v, lo), hi); }
__device__ __forceinline__ float rs_sign(float v) { return (v > 0.f) ? 1.f : ((v < 0.f) ? -1.f : 0.f); }
__device__ __forceinline__ float rs_abs(float v) { return fabsf(v); }
__device__ __forceinline__ float rs_sqr(float v) { return v * v; }

__device__ __forceinline__ Vec3 v3(float x, float y, float z) { return {x, y, z}; }
__device__ __forceinline__ Vec3 v3_zero() { return {0.f, 0.f, 0.f}; }

__device__ __forceinline__ Vec3 operator+(Vec3 a, Vec3 b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
__device__ __forceinline__ Vec3 operator-(Vec3 a, Vec3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
__device__ __forceinline__ Vec3 operator*(Vec3 a, float s) { return {a.x*s, a.y*s, a.z*s}; }
__device__ __forceinline__ Vec3 operator*(float s, Vec3 a) { return {a.x*s, a.y*s, a.z*s}; }
__device__ __forceinline__ Vec3 operator*(Vec3 a, Vec3 b) { return {a.x*b.x, a.y*b.y, a.z*b.z}; }
__device__ __forceinline__ Vec3 operator/(Vec3 a, float s) { float inv = 1.f/s; return {a.x*inv, a.y*inv, a.z*inv}; }
__device__ __forceinline__ Vec3 operator-(Vec3 a) { return {-a.x, -a.y, -a.z}; }

__device__ __forceinline__ Vec3& operator+=(Vec3& a, Vec3 b) { a.x+=b.x; a.y+=b.y; a.z+=b.z; return a; }
__device__ __forceinline__ Vec3& operator-=(Vec3& a, Vec3 b) { a.x-=b.x; a.y-=b.y; a.z-=b.z; return a; }
__device__ __forceinline__ Vec3& operator*=(Vec3& a, float s) { a.x*=s; a.y*=s; a.z*=s; return a; }

__device__ __forceinline__ float v3_dot(Vec3 a, Vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }

__device__ __forceinline__ Vec3 v3_cross(Vec3 a, Vec3 b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}

__device__ __forceinline__ float v3_length_sq(Vec3 v) { return v3_dot(v, v); }
__device__ __forceinline__ float v3_length(Vec3 v) { return sqrtf(v3_length_sq(v)); }

__device__ __forceinline__ Vec3 v3_normalize(Vec3 v) {
    float len2 = v3_length_sq(v);
    if (len2 < 1e-10f) return v3_zero();
    return v * rsqrtf(len2);
}

__device__ __forceinline__ Vec3 v3_safe_normalize(Vec3 v) {
    float len2 = v3_length_sq(v);
    if (len2 < 1e-10f) return v3_zero();
    return v * rsqrtf(len2);
}

__device__ __forceinline__ Vec3 v3_clamp_mag(Vec3 v, float maxLen) {
    float len2 = v3_length_sq(v);
    if (len2 > maxLen * maxLen) {
        return v * (maxLen * rsqrtf(len2));
    }
    return v;
}

__device__ __forceinline__ Vec3 v3_lerp(Vec3 a, Vec3 b, float t) {
    return a + (b - a) * t;
}

__device__ __forceinline__ bool v3_is_zero(Vec3 v) {
    return v.x == 0.f && v.y == 0.f && v.z == 0.f;
}

__device__ __forceinline__ float v3_dist_sq(Vec3 a, Vec3 b) { return v3_length_sq(a - b); }

// 2D distance squared (XY plane, for boost pad checks)
__device__ __forceinline__ float v3_dist_sq_2d(Vec3 a, Vec3 b) {
    float dx = a.x - b.x, dy = a.y - b.y;
    return dx*dx + dy*dy;
}


__device__ __forceinline__ RotMat rotmat_identity() {
    return {{1,0,0}, {0,1,0}, {0,0,1}};
}

// Transform local-space vector to world-space: R * v
__device__ __forceinline__ Vec3 rotmat_mul_vec(RotMat R, Vec3 v) {
    return R.forward * v.x + R.right * v.y + R.up * v.z;
}

// Transform world-space vector to local-space: R^T * v
__device__ __forceinline__ Vec3 rotmat_dot_vec(RotMat R, Vec3 v) {
    return {v3_dot(R.forward, v), v3_dot(R.right, v), v3_dot(R.up, v)};
}

// Matrix transpose
__device__ __forceinline__ RotMat rotmat_transpose(RotMat R) {
    return {
        {R.forward.x, R.right.x, R.up.x},
        {R.forward.y, R.right.y, R.up.y},
        {R.forward.z, R.right.z, R.up.z}
    };
}

// Matrix multiply: A * B
__device__ __forceinline__ RotMat rotmat_mul(RotMat A, RotMat B) {
    return {
        rotmat_mul_vec(A, B.forward),
        rotmat_mul_vec(A, B.right),
        rotmat_mul_vec(A, B.up),
    };
}

// Gram-Schmidt orthonormalization (prevent rotation drift)
__device__ __forceinline__ RotMat rotmat_orthonormalize(RotMat R) {
    R.forward = v3_normalize(R.forward);
    R.right = v3_normalize(R.right - R.forward * v3_dot(R.forward, R.right));
    R.up = v3_cross(R.forward, R.right);
    return R;
}

// Integrate rotation matrix by angular velocity * dt
// For small angle: column += cross(angVel, column) * dt
__device__ __forceinline__ RotMat rotmat_integrate(RotMat R, Vec3 angVel, float dt) {
    Vec3 dw = angVel * dt;
    R.forward += v3_cross(dw, R.forward);
    R.right   += v3_cross(dw, R.right);
    R.up      += v3_cross(dw, R.up);
    return rotmat_orthonormalize(R);
}


struct EulerAngles {
    float yaw, pitch, roll;
};

// Extract euler angles (YPR) from rotation matrix
// Matches RocketSim's Angle::FromRotMat
__device__ __forceinline__ EulerAngles rotmat_to_euler(RotMat R) {
    EulerAngles a;
    a.yaw   = atan2f(R.forward.y, R.forward.x);
    a.pitch = asinf(rs_clamp(-R.forward.z, -1.f, 1.f));
    a.roll  = atan2f(R.right.z, R.up.z);
    return a;
}

// Build rotation matrix from yaw, pitch, roll
__device__ __forceinline__ RotMat euler_to_rotmat(float yaw, float pitch, float roll) {
    float cy = cosf(yaw),   sy = sinf(yaw);
    float cp = cosf(pitch), sp = sinf(pitch);
    float cr = cosf(roll),  sr = sinf(roll);

    RotMat R;
    R.forward = {cy*cp, sy*cp, -sp};
    R.right   = {cy*sp*sr - sy*cr, sy*sp*sr + cy*cr, cp*sr};
    R.up      = {cy*sp*cr + sy*sr, sy*sp*cr - cy*sr, cp*cr};
    return R;
}



__device__ __forceinline__ float curve_eval(
    const CurvePoint* points, int numPoints, float input, float defaultOutput = 1.f
) {
    if (numPoints == 0) return defaultOutput;
    if (numPoints == 1) return points[0].output;

    // Clamp to range
    if (input <= points[0].input) return points[0].output;
    if (input >= points[numPoints - 1].input) return points[numPoints - 1].output;

    // Find surrounding points and interpolate
    for (int i = 0; i < numPoints - 1; i++) {
        if (input >= points[i].input && input <= points[i + 1].input) {
            float range = points[i + 1].input - points[i].input;
            if (range < 1e-10f) return points[i].output;
            float t = (input - points[i].input) / range;
            return points[i].output + (points[i + 1].output - points[i].output) * t;
        }
    }
    return points[numPoints - 1].output;
}

// Convenience wrappers for each curve (using constexpr data directly)
__device__ __forceinline__ float curve_steer_angle(float speed) {
    return curve_eval(Curves::STEER_ANGLE, Curves::STEER_ANGLE_N, speed);
}
__device__ __forceinline__ float curve_powerslide_steer(float speed) {
    return curve_eval(Curves::POWERSLIDE_STEER, Curves::POWERSLIDE_STEER_N, speed);
}
__device__ __forceinline__ float curve_drive_torque(float speed) {
    return curve_eval(Curves::DRIVE_TORQUE, Curves::DRIVE_TORQUE_N, speed);
}
__device__ __forceinline__ float curve_non_sticky_friction(float normalZ) {
    return curve_eval(Curves::NON_STICKY_FRICTION, Curves::NON_STICKY_FRICTION_N, normalZ);
}
__device__ __forceinline__ float curve_lat_friction(float input) {
    return curve_eval(Curves::LAT_FRICTION, Curves::LAT_FRICTION_N, input);
}
__device__ __forceinline__ float curve_long_friction(float input) {
    return curve_eval(nullptr, 0, input, 1.f); // Empty curve -> always 1.0
}
__device__ __forceinline__ float curve_handbrake_lat(float input) {
    return curve_eval(Curves::HANDBRAKE_LAT, Curves::HANDBRAKE_LAT_N, input);
}
__device__ __forceinline__ float curve_handbrake_long(float input) {
    return curve_eval(Curves::HANDBRAKE_LONG, Curves::HANDBRAKE_LONG_N, input);
}
__device__ __forceinline__ float curve_ball_car_impulse(float relSpeed) {
    return curve_eval(Curves::BALL_CAR_IMPULSE, Curves::BALL_CAR_IMPULSE_N, relSpeed);
}
__device__ __forceinline__ float curve_bump_vel_ground(float speed) {
    return curve_eval(Curves::BUMP_VEL_GROUND, Curves::BUMP_VEL_GROUND_N, speed);
}
__device__ __forceinline__ float curve_bump_vel_air(float speed) {
    return curve_eval(Curves::BUMP_VEL_AIR, Curves::BUMP_VEL_AIR_N, speed);
}
__device__ __forceinline__ float curve_bump_upward(float speed) {
    return curve_eval(Curves::BUMP_UPWARD, Curves::BUMP_UPWARD_N, speed);
}

} // namespace rsc
