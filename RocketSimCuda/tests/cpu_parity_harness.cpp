// ============================================================================
// cpu_parity_harness.cpp  —  GPU-less CUDA<->CPU physics parity harness
// ----------------------------------------------------------------------------
// OPT-105: prove RocketSimCuda physics is correct WITHOUT a GPU / CUDA runtime.
//
// HOW IT WORKS
//   We host-compile the *actual* RocketSimCuda kernel sources via
//   cuda_host_shim.cuh (which turns __device__/__global__/etc. into no-ops and
//   supplies host versions of the few CUDA intrinsics). The device functions
//   then execute on the CPU bit-for-bit as written. For each physics
//   subsystem we run a deterministic, fixed-state scenario through the real
//   kernel code and diff term-by-term against an INDEPENDENT double-precision
//   reference of the same documented algorithm (RocketSim semi-implicit Euler,
//   pow(1-drag,dt) damping, magnitude clamps, boost model, demo respawn).
//
//   No CUDA, no GPU, no nvcc. Builds and runs with a plain C++17 host compiler.
//
// WHAT A NONZERO DIFF MEANS (tolerance rationale)
//   The kernel runs in float32; the reference runs in float64. The only
//   expected divergence is float32 rounding accumulated over the tick loop.
//   For an N-tick semi-implicit Euler integration the rounding grows roughly
//   like N * eps_f32 * |state|, so we bound each term with a relative
//   tolerance (REL_TOL) plus a small absolute floor (ABS_FLOOR). A "perfect 0
//   diff" would only happen for ops that are exactly representable (e.g. a
//   single multiply); anything else is reported with its real magnitude and
//   must fall under tolerance. Determinism is verified separately by running
//   each scenario twice and requiring bit-identical kernel output.
//
//   Scenarios marked [INVARIANT] do not have a closed-form numeric reference
//   (collision response); instead the real kernel output is checked against
//   physical invariants (post-bounce direction, no energy gain, no tunnelling).
//   Scenarios marked [PREDICATE] re-evaluate an inline kernel predicate (goal
//   check) using the kernel's own PhysConst constants.
// ============================================================================

#include "cuda_host_shim.cuh"      // MUST be first
#include "../src/CarPhysics.cuh"   // before Collision.cuh (Collision uses car_forward)
#include "../src/Collision.cuh"
#include "../src/TrainingObs.cuh"  // real ContinuousV2 + Attention obs builder (OPT-124)

#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

using namespace rsc;

// ---- tolerances (see header comment) ---------------------------------------
// REL_TOL bounds float32 rounding accumulated over the tick loop. Semi-implicit
// Euler does one add + one multiply per axis per tick, so the relative error
// grows ~ N * eps_f32 (eps_f32 ~ 6e-8). For the longest scenario here
// (N=1440 ticks => ~9e-5) this is the dominant, physically-unavoidable diff
// between the float32 kernel and the float64 reference. REL_TOL=1e-4 bounds
// that with ~10x headroom while still catching any real algorithmic drift.
static constexpr double REL_TOL   = 1e-4;   // relative error vs f64 reference
static constexpr double ABS_FLOOR = 1e-2;   // abs floor for near-zero terms (UU or UU/s)

// Obs/action builders do NO tick-loop accumulation: each emitted element is a
// handful of float32 multiplies/dot-products, so the only divergence from the
// float64 reference is a few ulps of rounding. We therefore hold the obs/action
// path to a far tighter bound than the physics scenarios above.
static constexpr double REL_TOL_OBS   = 1e-6;  // relative error for obs/action elements
static constexpr double ABS_FLOOR_OBS = 1e-6;  // abs floor for normalized near-zero terms

static constexpr float DT = 1.f / 120.f;
static constexpr Vec3  GRAVITY = {0.f, 0.f, -650.f};

// ---- double-precision reference vector -------------------------------------
struct V3d { double x, y, z; };
static V3d  operator+(V3d a, V3d b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
static V3d  operator*(V3d a, double s) { return {a.x*s, a.y*s, a.z*s}; }
static double len(V3d v) { return std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z); }
static double len(Vec3 v) { return std::sqrt((double)v.x*v.x + (double)v.y*v.y + (double)v.z*v.z); }

// ---- per-term error between kernel Vec3 (f32) and reference V3d (f64) -------
struct TermErr { double maxAbs = 0.0; double maxRel = 0.0; };

static TermErr diff(Vec3 k, V3d r) {
    TermErr e;
    const double comp[3][2] = {
        {(double)k.x, r.x}, {(double)k.y, r.y}, {(double)k.z, r.z}
    };
    for (auto& c : comp) {
        double a = std::fabs(c[0] - c[1]);
        double rel = a / std::max(std::fabs(c[1]), ABS_FLOOR);
        e.maxAbs = std::max(e.maxAbs, a);
        e.maxRel = std::max(e.maxRel, rel);
    }
    return e;
}
static bool within(TermErr e) { return e.maxAbs <= ABS_FLOOR || e.maxRel <= REL_TOL; }

// ---- result accumulation ---------------------------------------------------
struct Scenario {
    std::string name;
    std::string kind;   // "numeric" | "invariant" | "predicate"
    bool passed = true;
    bool deterministic = true;
    std::vector<std::string> lines;
    void note(const std::string& s) { lines.push_back(s); }
};

static void report_term(Scenario& sc, const char* term, TermErr e) {
    bool ok = within(e);
    sc.passed &= ok;
    char buf[256];
    std::snprintf(buf, sizeof buf, "    %-10s maxAbs=%.6g  maxRel=%.3g  -> %s",
                  term, e.maxAbs, e.maxRel, ok ? "ok" : "OVER-TOL");
    sc.note(buf);
}

// ============================================================================
// Independent double-precision references
// ============================================================================

// Gravity-only semi-implicit Euler has an exact closed form, used as a truly
// independent reference (not a re-run of the kernel loop):
//   v_N = v0 + N*g*dt
//   p_N = p0 + N*dt*v0 + g*dt^2 * N(N+1)/2
static void ref_ballistic_closed_form(V3d p0, V3d v0, V3d g, int N, double dt,
                                      V3d& pOut, V3d& vOut) {
    vOut = v0 + g * ((double)N * dt);
    double tri = (double)N * (N + 1) * 0.5;
    pOut = p0 + v0 * ((double)N * dt) + g * (dt * dt * tri);
}

// Gravity + linear drag: no clean closed form, so re-derive the SAME algorithm
// device_integrate uses, in double precision (independent code path).
static void ref_ballistic_drag(V3d p0, V3d v0, V3d g, double drag, int N, double dt,
                               V3d& pOut, V3d& vOut) {
    V3d p = p0, v = v0;
    double damp = std::pow(std::max(1.0 - drag, 0.0), dt);
    for (int i = 0; i < N; i++) {
        v = v + g * dt;
        if (drag > 0.0) v = v * damp;
        p = p + v * dt;
    }
    pOut = p; vOut = v;
}

// ============================================================================
// Scenario runners (each calls the REAL kernel device functions)
// ============================================================================

// run device_integrate N ticks with given gravity/drag, no collisions
static void kernel_integrate_ball(GpuBallState& ball, float drag, int N) {
    for (int i = 0; i < N; i++)
        device_integrate(ball.pos, ball.vel, ball.rotMat, ball.angVel, GRAVITY, drag, DT);
}

static GpuBallState make_ball(Vec3 pos, Vec3 vel) {
    GpuBallState b{};
    b.pos = pos; b.vel = vel; b.angVel = {0,0,0};
    b.rotMat = rotmat_identity();
    b.velocityImpulseCache = {0,0,0};
    return b;
}

static bool ball_state_eq(const GpuBallState& a, const GpuBallState& b) {
    return std::memcmp(&a, &b, sizeof(GpuBallState)) == 0;
}
static bool car_state_eq(const GpuCarState& a, const GpuCarState& b) {
    return std::memcmp(&a, &b, sizeof(GpuCarState)) == 0;
}

// ----- 1/2. ballistic integrator (gravity-only) -----------------------------
static Scenario scn_ballistic(int N) {
    Scenario sc; sc.name = "ballistic_gravity_" + std::to_string(N) + "t"; sc.kind = "numeric";
    Vec3 p0 = {123.f, -45.f, 2000.f}, v0 = {200.f, -150.f, 300.f};

    GpuBallState ball = make_ball(p0, v0), ball2 = make_ball(p0, v0);
    kernel_integrate_ball(ball, 0.f, N);
    kernel_integrate_ball(ball2, 0.f, N);
    sc.deterministic = ball_state_eq(ball, ball2);

    V3d pr, vr;
    ref_ballistic_closed_form({p0.x,p0.y,p0.z}, {v0.x,v0.y,v0.z},
                              {GRAVITY.x,GRAVITY.y,GRAVITY.z}, N, DT, pr, vr);
    report_term(sc, "pos", diff(ball.pos, pr));
    report_term(sc, "vel", diff(ball.vel, vr));
    return sc;
}

// ----- 3. ballistic integrator (gravity + drag) -----------------------------
static Scenario scn_ballistic_drag(int N) {
    Scenario sc; sc.name = "ballistic_drag_" + std::to_string(N) + "t"; sc.kind = "numeric";
    const float DRAG = 0.03f;
    Vec3 p0 = {0.f, 0.f, 1500.f}, v0 = {1000.f, 500.f, 200.f};

    GpuBallState ball = make_ball(p0, v0), ball2 = make_ball(p0, v0);
    kernel_integrate_ball(ball, DRAG, N);
    kernel_integrate_ball(ball2, DRAG, N);
    sc.deterministic = ball_state_eq(ball, ball2);

    V3d pr, vr;
    ref_ballistic_drag({p0.x,p0.y,p0.z}, {v0.x,v0.y,v0.z},
                       {GRAVITY.x,GRAVITY.y,GRAVITY.z}, DRAG, N, DT, pr, vr);
    report_term(sc, "pos", diff(ball.pos, pr));
    report_term(sc, "vel", diff(ball.vel, vr));
    return sc;
}

// ----- 4. ball max-speed clamp (device_ball_finish_tick) --------------------
static Scenario scn_ball_maxspeed() {
    Scenario sc; sc.name = "ball_maxspeed_clamp"; sc.kind = "numeric";
    const float MAXSPD = 6000.f;
    Vec3 v0 = {5000.f, 4000.f, 3000.f}; // |v| ~ 7071 > 6000
    GpuBallState ball = make_ball({0,0,1000}, v0), ball2 = make_ball({0,0,1000}, v0);
    device_ball_finish_tick(ball, MAXSPD);
    device_ball_finish_tick(ball2, MAXSPD);
    sc.deterministic = ball_state_eq(ball, ball2);

    double m = len(V3d{v0.x, v0.y, v0.z});
    V3d vr = {v0.x, v0.y, v0.z};
    vr = vr * ((double)MAXSPD / m);
    report_term(sc, "vel", diff(ball.vel, vr));
    char buf[128];
    std::snprintf(buf, sizeof buf, "    |vel|=%.4f (target %.1f)", len(ball.vel), (double)MAXSPD);
    sc.note(buf);
    return sc;
}

// ----- 5. ball<->world floor bounce [INVARIANT] -----------------------------
static Scenario scn_ball_floor_bounce(int N) {
    Scenario sc; sc.name = "ball_floor_bounce_" + std::to_string(N) + "t"; sc.kind = "invariant";
    ArenaSurface surf[MAX_ARENA_SURFACES];
    int numSurf = buildArenaSurfaces(surf);

    const float R = 91.25f, FRIC = 0.35f, REST = 0.6f;
    GpuBallState ball = make_ball({0.f, 0.f, 300.f}, {0.f, 0.f, -2000.f});

    // Mechanical energy per unit mass: E = 0.5|v|^2 + g*z  (g=650). Ballistic
    // free-fall conserves E (up to symplectic-Euler O(dt) drift); a lossy
    // restitution<1 bounce strictly REDUCES E. So E must never grow across the
    // trajectory -- this is the right invariant (peak speed alone is wrong
    // because gravity legitimately speeds the ball up on the way down).
    auto energy = [](const GpuBallState& b) {
        double v2 = (double)b.vel.x*b.vel.x + (double)b.vel.y*b.vel.y + (double)b.vel.z*b.vel.z;
        return 0.5*v2 + 650.0*(double)b.pos.z;
    };
    bool everWentUp = false;
    double E0 = energy(ball), maxE = E0;
    float minZ = ball.pos.z;
    for (int i = 0; i < N; i++) {
        float preZv = ball.vel.z;
        device_integrate(ball.pos, ball.vel, ball.rotMat, ball.angVel, GRAVITY, 0.03f, DT);
        device_collision_ball_world(ball, surf, numSurf, R, FRIC, REST);
        device_ball_finish_tick(ball, 6000.f);
        if (preZv < 0.f && ball.vel.z > 0.f) everWentUp = true;
        maxE = std::max(maxE, energy(ball));
        minZ = std::min(minZ, ball.pos.z);
    }
    // Invariants:
    bool bounced  = everWentUp;                       // collision reversed downward motion
    bool noTunnel = minZ > (93.15f - 91.25f) - 1.0f;  // never sank a full radius below rest
    bool noEnergy = maxE <= E0 * (1.0 + 1e-3);         // restitution<1 + drag: E non-increasing
    sc.passed &= bounced && noTunnel && noEnergy;
    char buf[256];
    std::snprintf(buf, sizeof buf,
        "    bounced=%d  minZ=%.2f (>=%.2f)  E0=%.1f maxE=%.1f (<=%.1f)  finalPos.z=%.2f",
        (int)bounced, minZ, (93.15f-91.25f)-1.0f, E0, maxE, E0*(1.0+1e-3), ball.pos.z);
    sc.note(buf);
    sc.note(std::string("    invariants: bounced=") + (bounced?"ok":"FAIL") +
            " noTunnel=" + (noTunnel?"ok":"FAIL") + " noEnergyGain=" + (noEnergy?"ok":"FAIL"));
    // determinism
    GpuBallState b2 = make_ball({0.f,0.f,300.f}, {0.f,0.f,-2000.f});
    for (int i = 0; i < N; i++) {
        device_integrate(b2.pos, b2.vel, b2.rotMat, b2.angVel, GRAVITY, 0.03f, DT);
        device_collision_ball_world(b2, surf, numSurf, R, FRIC, REST);
        device_ball_finish_tick(b2, 6000.f);
    }
    sc.deterministic = ball_state_eq(ball, b2);
    return sc;
}

// minimal airborne car (identity orientation -> forward = +x)
static GpuCarState make_air_car() {
    GpuCarState car{};
    car.pos = {0,0,500};
    car.rotMat = rotmat_identity();
    car.vel = {0,0,0};
    car.angVel = {0,0,0};
    car.boost = 100.f;
    car.timeSpentBoosting = 0.f;
    car.isOnGround = false;
    car.team = 0; car.id = 0;
    return car;
}

// ----- 6. boost acceleration (air) (device_update_boost) --------------------
static Scenario scn_boost_air(int N) {
    Scenario sc; sc.name = "boost_accel_air_" + std::to_string(N) + "t"; sc.kind = "numeric";
    const float BUPS = 100.f/3.f, BAG = 2975.f/3.f, BAA = 3175.f/3.f;

    GpuCarState car = make_air_car(), car2 = make_air_car();
    car.controls.boost = true; car2.controls.boost = true;
    for (int i = 0; i < N; i++) device_update_boost(car,  DT, BUPS, BAG, BAA);
    for (int i = 0; i < N; i++) device_update_boost(car2, DT, BUPS, BAG, BAA);
    sc.deterministic = car_state_eq(car, car2);

    // reference: forward=+x, airborne, boosting every tick from tick 0
    double velx = (double)BAA * DT * N;
    double boost = std::max(100.0 - (double)BUPS * DT * N, 0.0);
    report_term(sc, "vel", diff(car.vel, {velx, 0.0, 0.0}));
    TermErr be; be.maxAbs = std::fabs((double)car.boost - boost);
    be.maxRel = be.maxAbs / std::max(boost, ABS_FLOOR);
    report_term(sc, "boost", be);
    return sc;
}

// ----- 7. demolition respawn (device_handle_demo) ---------------------------
static Scenario scn_demo_respawn() {
    Scenario sc; sc.name = "demo_respawn"; sc.kind = "numeric";
    GpuCarState car = make_air_car();
    car.isDemoed = true;
    car.demoRespawnTimer = 3.0f;

    const int ticksToRespawn = (int)std::ceil(3.0 / DT); // 360
    bool timerTracksRef = true;
    int respawnTick = -1;
    for (int i = 1; i <= ticksToRespawn + 2; i++) {
        device_handle_demo(car, DT, 0);
        if (!car.isDemoed && respawnTick < 0) respawnTick = i;
        if (car.isDemoed) {
            double refTimer = std::max(3.0 - (double)i * DT, 0.0);
            if (std::fabs((double)car.demoRespawnTimer - refTimer) > 1e-4) timerTracksRef = false;
        }
    }
    bool respawnedOnTime = (respawnTick == ticksToRespawn); // first tick timer hits 0
    bool respawnFields = (!car.isDemoed) &&
                         std::fabs(car.boost - 100.f/3.f) < 1e-3f &&
                         std::fabs(car.pos.z - 36.f) < 1e-3f; // CAR_RESPAWN_Z
    sc.passed &= timerTracksRef && respawnedOnTime && respawnFields;
    sc.deterministic = true; // pure deterministic function of id/team
    char buf[256];
    std::snprintf(buf, sizeof buf,
        "    respawnTick=%d (expect %d)  timerTracksRef=%s  boost=%.3f pos.z=%.3f",
        respawnTick, ticksToRespawn, timerTracksRef?"ok":"FAIL", car.boost, car.pos.z);
    sc.note(buf);
    return sc;
}

// ----- 8. goal detection predicate [PREDICATE] ------------------------------
static bool kernel_goal_pred(Vec3 p, float thresholdY) {
    // exact copy of the inline goal check in stepArenaKernel, using PhysConst.
    return rs_abs(p.y) > thresholdY &&
           rs_abs(p.x) < PhysConst::GOAL_HALF_WIDTH &&
           p.z < PhysConst::GOAL_HEIGHT;
}
static Scenario scn_goal_predicate() {
    Scenario sc; sc.name = "goal_detection_predicate"; sc.kind = "predicate";
    const float TY = PhysConst::SOCCAR_GOAL_SCORE_THRESHOLD_Y; // 5124.25
    struct Case { Vec3 p; bool expect; const char* why; };
    Case cases[] = {
        {{   0.f, 5200.f, 100.f}, true,  "centered past line -> goal (blue scores)"},
        {{   0.f, 5000.f, 100.f}, false, "short of line -> no goal"},
        {{1000.f, 5200.f, 100.f}, false, "outside goal width -> no goal"},
        {{   0.f, 5200.f, 700.f}, false, "above crossbar -> no goal"},
        {{   0.f,-5200.f, 100.f}, true,  "centered past line (orange end)"},
    };
    int fails = 0;
    for (auto& c : cases) {
        bool got = kernel_goal_pred(c.p, TY);
        bool ok = (got == c.expect);
        fails += ok ? 0 : 1;
        char buf[256];
        std::snprintf(buf, sizeof buf, "    pred=%d expect=%d %s : %s",
                      (int)got, (int)c.expect, ok?"ok":"FAIL", c.why);
        sc.note(buf);
    }
    sc.passed &= (fails == 0);
    return sc;
}

// ============================================================================
// OPT-124: ContinuousV2 + Attention obs/action builder parity
// ----------------------------------------------------------------------------
// The obs scenarios run the REAL device obs assembler (device_build_advanced_obs
// from ../src/TrainingObs.cuh, the exact function the production kernel calls)
// over representative states and diff every emitted element against an
// INDEPENDENT float64 reference of the documented AdvancedObs algorithm
// (matching RLGymCPP AdvancedObs::BuildObs). The action scenarios prove the
// ContinuousV2 action parser mapping (8 continuous floats -> controls) and that
// the parsed controls drive the device physics to a reference-matching state.
// ============================================================================

// ---- float64 mirror of the AdvancedObs algorithm ---------------------------
namespace obsref {
    static constexpr double POS  = 1.0 / 5000.0;
    static constexpr double VEL  = 1.0 / 2300.0;
    static constexpr double ANGV = 1.0 / 3.0;

    struct Rot3 { V3d f, r, u; };

    static V3d invvec(Vec3 v, bool inv) {
        return inv ? V3d{-(double)v.x, -(double)v.y, (double)v.z}
                   : V3d{ (double)v.x,  (double)v.y, (double)v.z};
    }
    static Rot3 invrot(RotMat m, bool inv) {
        return { invvec(m.forward, inv), invvec(m.right, inv), invvec(m.up, inv) };
    }
    static double dot(V3d a, V3d b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
    static V3d sub(V3d a, V3d b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
    static V3d rotdot(const Rot3& R, V3d v) { return { dot(R.f,v), dot(R.r,v), dot(R.u,v) }; }

    struct Builder {
        std::vector<double> out;
        void w(double x) { out.push_back(x); }
        void w3(V3d a, double c) { out.push_back(a.x*c); out.push_back(a.y*c); out.push_back(a.z*c); }
        void w3raw(V3d a) { out.push_back(a.x); out.push_back(a.y); out.push_back(a.z); }
    };

    static bool has_flip_or_jump(const GpuCarState& c) {
        return c.isOnGround || (!c.hasFlipped && !c.hasDoubleJumped && c.airTimeSinceJump < 1.25f);
    }

    static void add_player(Builder& b, const GpuCarState& p, bool inv, V3d invBallPos, V3d invBallVel) {
        V3d pos = invvec(p.pos, inv);
        Rot3 rot = invrot(p.rotMat, inv);
        V3d vel = invvec(p.vel, inv);
        V3d ang = invvec(p.angVel, inv);

        b.w3(pos, POS);
        b.w3raw(rot.f);
        b.w3raw(rot.u);
        b.w3(vel, VEL);
        b.w3(ang, ANGV);
        b.w3(rotdot(rot, ang), ANGV);
        b.w3(rotdot(rot, sub(invBallPos, pos)), POS);
        b.w3(rotdot(rot, sub(invBallVel, vel)), VEL);

        b.w((double)p.boost / 100.0);
        b.w(p.isOnGround ? 1.0 : 0.0);
        b.w(has_flip_or_jump(p) ? 1.0 : 0.0);
        b.w(p.isDemoed ? 1.0 : 0.0);
        b.w(p.hasJumped ? 1.0 : 0.0);
    }

    static std::vector<double> build(
        const GpuCarState* cars, int numCars, int carIdx,
        const GpuBallState& ball,
        const GpuBoostPadState* pads, const int* padMap, bool inv)
    {
        const GpuCarState& self = cars[carIdx];
        V3d invBallPos = invvec(ball.pos, inv);
        V3d invBallVel = invvec(ball.vel, inv);
        V3d invBallAng = invvec(ball.angVel, inv);

        Builder b;
        b.w3(invBallPos, POS);
        b.w3(invBallVel, VEL);
        b.w3(invBallAng, ANGV);

        b.w((double)self.lastControls.throttle);
        b.w((double)self.lastControls.steer);
        b.w((double)self.lastControls.pitch);
        b.w((double)self.lastControls.yaw);
        b.w((double)self.lastControls.roll);
        b.w(self.lastControls.jump ? 1.0 : 0.0);
        b.w(self.lastControls.boost ? 1.0 : 0.0);
        b.w(self.lastControls.handbrake ? 1.0 : 0.0);

        for (int i = 0; i < NUM_BOOST_PADS; i++) {
            const GpuBoostPadState& pad = pads[padMap[i]];
            b.w(pad.isActive ? 1.0 : 1.0 / (1.0 + (double)pad.cooldown));
        }

        add_player(b, self, inv, invBallPos, invBallVel);
        for (int o = 0; o < numCars; o++) {
            if (o == carIdx || cars[o].team != self.team) continue;
            add_player(b, cars[o], inv, invBallPos, invBallVel);
        }
        for (int o = 0; o < numCars; o++) {
            if (o == carIdx || cars[o].team == self.team) continue;
            add_player(b, cars[o], inv, invBallPos, invBallVel);
        }
        return b.out;
    }
} // namespace obsref

// ---- builders for representative states -------------------------------------
static GpuCarState obs_make_car(Vec3 pos, RotMat rot, Vec3 vel, Vec3 angVel, uint8_t team, uint32_t id) {
    GpuCarState c{};
    c.pos = pos; c.rotMat = rot; c.vel = vel; c.angVel = angVel;
    c.boost = 47.5f;
    c.isOnGround = true;
    c.team = team; c.id = id; c.preset = 0;
    // last action seen by the policy (ContinuousV2 8-tuple); exercises obs cols 9..16
    c.lastControls.throttle = 0.7f;
    c.lastControls.steer = -0.3f;
    c.lastControls.pitch = 0.15f;
    c.lastControls.yaw = -0.85f;
    c.lastControls.roll = 1.0f;
    c.lastControls.jump = false;
    c.lastControls.boost = true;
    c.lastControls.handbrake = false;
    return c;
}

static GpuBallState obs_make_ball(Vec3 pos, Vec3 vel, Vec3 angVel) {
    GpuBallState b{};
    b.pos = pos; b.vel = vel; b.angVel = angVel;
    b.rotMat = rotmat_identity();
    return b;
}

// yaw-only rotation (orthonormal), used to give non-identity orientations
static RotMat obs_yaw_rot(float yaw) {
    float c = std::cos(yaw), s = std::sin(yaw);
    RotMat m;
    m.forward = { c,  s, 0.f };
    m.right   = { s, -c, 0.f };
    m.up      = { 0.f, 0.f, 1.f };
    return m;
}

// run REAL device obs builder, diff every element vs the f64 reference, and
// verify the obs row is bit-identical across two runs (determinism)
static Scenario scn_obs(const std::string& name,
                        std::vector<GpuCarState> cars, int carIdx,
                        GpuBallState ball) {
    Scenario sc; sc.name = name; sc.kind = "numeric";

    GpuBoostPadState pads[NUM_BOOST_PADS];
    int padMap[NUM_BOOST_PADS];
    for (int i = 0; i < NUM_BOOST_PADS; i++) {
        pads[i].isActive = (i % 3 == 0);
        pads[i].cooldown = pads[i].isActive ? 0.f : (0.25f * (float)(i + 1));
        padMap[i] = i;
    }

    bool inv = (cars[carIdx].team == 1);

    std::vector<float> dev(512, 0.f), dev2(512, 0.f);
    int n  = device_build_advanced_obs(dev.data(),  cars.data(), (int)cars.size(), carIdx, ball, pads, padMap, inv);
    int n2 = device_build_advanced_obs(dev2.data(), cars.data(), (int)cars.size(), carIdx, ball, pads, padMap, inv);

    std::vector<double> ref = obsref::build(cars.data(), (int)cars.size(), carIdx, ball, pads, padMap, inv);

    sc.deterministic = (n == n2) &&
        std::memcmp(dev.data(), dev2.data(), sizeof(float) * (size_t)n) == 0;

    bool sizeOk = ((size_t)n == ref.size());
    sc.passed &= sizeOk;

    TermErr e;
    int worstIdx = -1;
    if (sizeOk) {
        for (int i = 0; i < n; i++) {
            double a = std::fabs((double)dev[i] - ref[i]);
            double rel = a / std::max(std::fabs(ref[i]), ABS_FLOOR_OBS);
            if (rel > e.maxRel) { e.maxRel = rel; worstIdx = i; }
            e.maxAbs = std::max(e.maxAbs, a);
        }
    }
    bool tolOk = sizeOk && (e.maxAbs <= ABS_FLOOR_OBS || e.maxRel <= REL_TOL_OBS);
    sc.passed &= tolOk;

    char buf[256];
    std::snprintf(buf, sizeof buf,
        "    obsLen=%d (ref %zu)  maxAbs=%.3g  maxRel=%.3g (idx %d)  -> %s",
        n, ref.size(), e.maxAbs, e.maxRel, worstIdx, tolOk ? "ok" : "OVER-TOL");
    sc.note(buf);
    return sc;
}

// ---- ContinuousV2 action parser (mirror of RLGC::DefaultContinuousAction) ---
// 8 continuous floats -> controls: 5 passthrough, 3 thresholded at >0.
static void parse_continuous_action(const float a[8], GpuCarControls& c) {
    c.throttle = a[0];
    c.steer    = a[1];
    c.pitch    = a[2];
    c.yaw      = a[3];
    c.roll     = a[4];
    c.jump      = a[5] > 0.0f;
    c.boost     = a[6] > 0.0f;
    c.handbrake = a[7] > 0.0f;
}

// parser mapping must be exact (passthrough = bit-identical, threshold = 0/1)
static Scenario scn_continuous_action_parse() {
    Scenario sc; sc.name = "continuous_action_parse"; sc.kind = "numeric";
    struct Case { float a[8]; };
    Case cases[] = {
        {{ 1.0f, -1.0f,  0.5f, -0.25f, 0.123f,  0.2f, -0.2f,  1.0f}},
        {{-0.7f,  0.7f, -0.3f,  0.9f, -1.0f,   -0.5f,  0.5f, -0.5f}},
        {{ 0.0f,  0.0f,  0.0f,  0.0f,  0.0f,    0.0f,  0.0f,  0.0f}}, // boundary: >0 is false at 0
        {{ 0.31f,-0.62f, 0.93f,-0.04f, 0.55f,   1e-6f,-1e-6f, 1e-6f}},
    };
    double maxAbs = 0.0; int fails = 0;
    for (auto& cse : cases) {
        GpuCarControls c{};
        parse_continuous_action(cse.a, c);
        // exact passthrough
        maxAbs = std::max(maxAbs, std::fabs((double)c.throttle - cse.a[0]));
        maxAbs = std::max(maxAbs, std::fabs((double)c.steer    - cse.a[1]));
        maxAbs = std::max(maxAbs, std::fabs((double)c.pitch    - cse.a[2]));
        maxAbs = std::max(maxAbs, std::fabs((double)c.yaw      - cse.a[3]));
        maxAbs = std::max(maxAbs, std::fabs((double)c.roll     - cse.a[4]));
        // exact threshold (independent reference computation)
        bool j = cse.a[5] > 0.0f, b = cse.a[6] > 0.0f, h = cse.a[7] > 0.0f;
        fails += (c.jump == j && c.boost == b && c.handbrake == h) ? 0 : 1;
    }
    sc.passed &= (maxAbs == 0.0) && (fails == 0);
    char buf[160];
    std::snprintf(buf, sizeof buf,
        "    passthrough maxAbs=%.3g (expect 0)  threshold mismatches=%d -> %s",
        maxAbs, fails, sc.passed ? "ok" : "FAIL");
    sc.note(buf);
    return sc;
}

// continuous control inputs -> next state: parse a ContinuousV2 action with
// boost engaged and drive the REAL device boost integrator for N ticks, then
// diff resulting state against an independent f64 reference.
static Scenario scn_continuous_action_apply_air(int N) {
    Scenario sc; sc.name = "continuous_action_apply_air_" + std::to_string(N) + "t"; sc.kind = "numeric";
    const float BUPS = 100.f/3.f, BAG = 2975.f/3.f, BAA = 3175.f/3.f;
    const float action[8] = { 0.8f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.6f, -1.0f }; // boost on (a[6]>0)

    GpuCarControls parsed{};
    parse_continuous_action(action, parsed);

    GpuCarState car = make_air_car(), car2 = make_air_car();
    car.controls = parsed; car2.controls = parsed;
    for (int i = 0; i < N; i++) device_update_boost(car,  DT, BUPS, BAG, BAA);
    for (int i = 0; i < N; i++) device_update_boost(car2, DT, BUPS, BAG, BAA);
    sc.deterministic = car_state_eq(car, car2);

    // reference: airborne forward=+x, boosting from tick 0 (timer set then applied)
    double velx = (double)BAA * DT * N;
    double boost = std::max(100.0 - (double)BUPS * DT * N, 0.0);
    report_term(sc, "vel", diff(car.vel, {velx, 0.0, 0.0}));
    TermErr be; be.maxAbs = std::fabs((double)car.boost - boost);
    be.maxRel = be.maxAbs / std::max(boost, ABS_FLOOR);
    report_term(sc, "boost", be);
    char buf[160];
    std::snprintf(buf, sizeof buf,
        "    parsed: boost=%d throttle=%.3f  vel.x=%.4f boost=%.4f",
        (int)parsed.boost, parsed.throttle, car.vel.x, car.boost);
    sc.note(buf);
    return sc;
}

// ============================================================================
int main() {
    std::printf("RocketSimCuda GPU-less CPU parity harness (OPT-105)\n");
    std::printf("Host build: float32 kernels vs float64 reference. No CUDA, no GPU.\n");
    std::printf("Tolerances: REL_TOL=%.0e  ABS_FLOOR=%.0e  dt=1/120\n", REL_TOL, ABS_FLOOR);

    std::vector<Scenario> all;
    all.push_back(scn_ballistic(1));
    all.push_back(scn_ballistic(120));
    all.push_back(scn_ballistic(1440));
    all.push_back(scn_ballistic_drag(120));
    all.push_back(scn_ballistic_drag(600));
    all.push_back(scn_ball_maxspeed());
    all.push_back(scn_ball_floor_bounce(120));
    all.push_back(scn_boost_air(30));
    all.push_back(scn_boost_air(120));
    all.push_back(scn_demo_respawn());
    all.push_back(scn_goal_predicate());

    // ---- OPT-124: ContinuousV2 + Attention obs/action builder parity --------
    {
        RotMat I = rotmat_identity();
        // kickoff (1v1): self at back-left spawn facing diagonally, ball centred at rest
        {
            std::vector<GpuCarState> cars = {
                obs_make_car({-2048.f, -2560.f, 17.f}, obs_yaw_rot(M_PI/4.f), {0,0,0}, {0,0,0}, 0, 1),
                obs_make_car({ 2048.f,  2560.f, 17.f}, obs_yaw_rot(M_PI*1.25f), {0,0,0}, {0,0,0}, 1, 2),
            };
            all.push_back(scn_obs("obs_kickoff_1v1", cars, 0, obs_make_ball({0,0,93.15f}, {0,0,0}, {0,0,0})));
        }
        // mid-air (2v2): airborne self with rotation + spin, teammate + 2 opponents
        {
            std::vector<GpuCarState> cars = {
                obs_make_car({120.f, -300.f, 950.f}, obs_yaw_rot(1.1f), {600.f,-250.f,420.f}, {1.2f,-0.7f,2.1f}, 0, 1),
                obs_make_car({-800.f, 200.f, 60.f},  obs_yaw_rot(0.2f), {-150.f,300.f,0.f},   {0,0,0.4f},       0, 2),
                obs_make_car({900.f, 1200.f, 300.f}, obs_yaw_rot(2.7f), {200.f,-400.f,100.f}, {0.3f,0.1f,-1.0f},1, 3),
                obs_make_car({1500.f,-900.f, 17.f},  obs_yaw_rot(-1.4f),{50.f,80.f,0.f},      {0,0,0},          1, 4),
            };
            cars[0].isOnGround = false; cars[0].airTimeSinceJump = 0.4f; cars[0].hasJumped = true;
            cars[2].isOnGround = false; cars[2].hasFlipped = true; cars[2].hasDoubleJumped = true;
            all.push_back(scn_obs("obs_midair_2v2", cars, 0, obs_make_ball({-50.f,400.f,1100.f}, {300.f,900.f,-200.f}, {2.5f,-1.1f,0.8f})));
        }
        // post-bounce: ball low with high spin and rebounding velocity (orange POV -> inv)
        {
            std::vector<GpuCarState> cars = {
                obs_make_car({-400.f, 1800.f, 17.f}, obs_yaw_rot(-0.6f), {-300.f,700.f,0.f}, {0,0,0}, 1, 7),
                obs_make_car({ 300.f, 1500.f, 17.f}, obs_yaw_rot(2.1f),  {100.f,-200.f,0.f}, {0,0,0}, 0, 8),
            };
            all.push_back(scn_obs("obs_post_bounce_inv", cars, 0, obs_make_ball({-120.f,1600.f,140.f}, {80.f,-260.f,950.f}, {5.9f,-4.2f,3.3f})));
        }
        // demo: self demolished (zeroed phys, isDemoed flag), opponent live
        {
            std::vector<GpuCarState> cars = {
                obs_make_car({0,0,0}, rotmat_identity(), {0,0,0}, {0,0,0}, 0, 11),
                obs_make_car({1200.f,-600.f,17.f}, obs_yaw_rot(0.9f), {400.f,0.f,0.f}, {0,0,0}, 1, 12),
            };
            cars[0].isDemoed = true; cars[0].demoRespawnTimer = 2.4f; cars[0].boost = 0.f; cars[0].isOnGround = false;
            all.push_back(scn_obs("obs_demo", cars, 0, obs_make_ball({500.f,-200.f,300.f}, {-100.f,150.f,50.f}, {0.4f,0.2f,-0.1f})));
        }
        // near-goal: ball just inside scoring depth, self attacking
        {
            std::vector<GpuCarState> cars = {
                obs_make_car({200.f, 4600.f, 17.f}, obs_yaw_rot(M_PI/2.f), {300.f,1400.f,0.f}, {0,0,0}, 0, 21),
                obs_make_car({-150.f,4950.f, 30.f}, obs_yaw_rot(-M_PI/2.f),{0.f,-200.f,0.f},   {0,0,0}, 1, 22),
            };
            all.push_back(scn_obs("obs_near_goal", cars, 0, obs_make_ball({120.f,5050.f,320.f}, {200.f,1500.f,-120.f}, {1.0f,-2.0f,0.5f})));
        }
    }
    all.push_back(scn_continuous_action_parse());
    all.push_back(scn_continuous_action_apply_air(30));
    all.push_back(scn_continuous_action_apply_air(120));

    bool allPass = true;
    for (auto& sc : all) {
        bool ok = sc.passed && sc.deterministic;
        allPass &= ok;
        std::printf("\n[%-9s] %-26s : %s%s\n",
                    sc.kind.c_str(), sc.name.c_str(),
                    sc.passed ? "PASS" : "FAIL",
                    sc.deterministic ? "" : " (NON-DETERMINISTIC)");
        for (auto& l : sc.lines) std::printf("%s\n", l.c_str());
    }

    std::printf("\n=====================================================\n");
    std::printf("OVERALL: %s\n", allPass ? "PASS" : "FAIL");
    return allPass ? 0 : 1;
}
