#pragma once
// stepArenaKernel (main sim)
// Extracted from RocketSimCuda.cu (modular split; same TU via include).

// ============================================================================
// MAIN SIMULATION KERNEL
// 1 thread = 1 arena
// ============================================================================

__global__ void stepArenaKernel(
    GpuCarState* allCars,       // [numArenas * maxCarsPerArena]
    GpuBallState* allBalls,     // [numArenas]
    GpuBoostPadState* allPads,  // [numArenas * NUM_BOOST_PADS]
    GpuArenaState* allArenas,   // [numArenas]
    const GpuCarControls* allControls, // [numArenas * maxCarsPerArena]
    int numArenas,
    int maxCarsPerArena,
    int ticksToSimulate
) {
    int arenaIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (arenaIdx >= numArenas) return;

    // Pointers to this arena's data
    GpuCarState* cars = &allCars[arenaIdx * maxCarsPerArena];
    GpuBallState& ball = allBalls[arenaIdx];
    GpuBoostPadState* pads = &allPads[arenaIdx * BoostPadData::NUM_TOTAL];
    GpuArenaState& arena = allArenas[arenaIdx];

    int numCars = arena.numCars;
    bool isVoid = (arena.gameMode == 1);
    float dt = c_tickTime;

    for (int tick = 0; tick < ticksToSimulate; tick++) {

        // Per-tick force accumulators (Bullet applyCentralForce/applyTorque:
        // integrated by the solver, applied to velocities at writeback).
        Vec3 carAccel[MAX_CARS_PER_ARENA];
        Vec3 carAngAccel[MAX_CARS_PER_ARENA];
        for (int c = 0; c < numCars; c++) {
            carAccel[c] = v3_zero();
            carAngAccel[c] = v3_zero();
        }

        // ============================================================
        // 1. CAR PRE-TICK UPDATE (Car::_PreTickUpdate)
        // ============================================================
        for (int c = 0; c < numCars; c++) {
            GpuCarState& car = cars[c];
            car.controls = allControls[arenaIdx * maxCarsPerArena + c];

            // Clamp controls
            car.controls.throttle = rs_clamp(car.controls.throttle, -1.f, 1.f);
            car.controls.steer    = rs_clamp(car.controls.steer, -1.f, 1.f);
            car.controls.pitch    = rs_clamp(car.controls.pitch, -1.f, 1.f);
            car.controls.yaw      = rs_clamp(car.controls.yaw, -1.f, 1.f);
            car.controls.roll     = rs_clamp(car.controls.roll, -1.f, 1.f);

            // Handle demo state
            if (car.isDemoed) {
                device_handle_demo(car, dt, arenaIdx);
                continue;
            }

            bool jumpPressed = car.controls.jump && !car.lastControls.jump;

            // Wheel raycasts
            if (!isVoid) {
                device_wheel_raycast(car, c_surfaces, c_numSurfaces, c_meshGrid, dt);
                // Match RocketSim's updateVehicleFirst(): compute friction impulses
                // using wheel state from the previous tick before controls update.
                device_calc_friction_impulses(car, dt);
            } else {
                car.isOnGround = false;
                for (int w = 0; w < 4; w++) car.wheels[w].isInContact = false;
            }

            // Count wheels in contact
            int numWheelsInContact = 0;
            for (int w = 0; w < 4; w++)
                numWheelsInContact += car.wheels[w].isInContact ? 1 : 0;

            float forwardSpeed = car_forward_speed(car);

            // Update wheels (throttle, brake, steer, friction, sticky force)
            if (!isVoid)
                device_update_wheels(car, dt, carAccel[c]);

            // Air torque (if not fully grounded)
            if (numWheelsInContact < 3) {
                device_update_air_torque(car, dt, numWheelsInContact == 0,
                                         carAccel[c], carAngAccel[c]);
            } else {
                car.isFlipping = false;
            }

            // Jump
            device_update_jump(car, dt, jumpPressed, carAccel[c]);

            // Auto-flip
            device_update_autoflip(car, dt, jumpPressed);

            // Double jump / Flip
            device_update_double_jump_or_flip(car, dt, jumpPressed, forwardSpeed);

            // Auto-roll
            if (car.controls.throttle != 0.f &&
                ((numWheelsInContact > 0 && numWheelsInContact < 4) || car.worldContactHasContact)) {
                device_update_autoroll(car, dt, numWheelsInContact,
                                       carAccel[c], carAngAccel[c]);
            }

            car.worldContactHasContact = false;

            // Suspension + friction (direct impulses, like btVehicleRL)
            if (!isVoid) {
                device_update_suspension(car, dt);
                device_apply_friction_impulses(car, dt);
            }

            // Boost
            device_update_boost(car, dt, c_boostUsedPerSecond, c_boostAccelGround, c_boostAccelAir,
                                carAccel[c]);
        }

        // ============================================================
        // 2. BOOST PADS PRE-TICK
        // ============================================================
        if (!isVoid) {
            for (int p = 0; p < BoostPadData::NUM_TOTAL; p++)
                device_boostpad_pre_tick(pads[p], dt);
        }

        // ============================================================
        // 3+4. BULLET WORLD STEP
        // (damping -> collision detection at pre-move positions ->
        //  sequential impulse solve -> writeback -> integrate transforms)
        // ============================================================
        device_bullet_world_step(
            cars, numCars, ball,
            c_surfaces, isVoid ? 0 : c_numSurfaces,
            c_meshGrid,
            carAccel, carAngAccel,
            c_gravity,
            c_ballDrag,
            c_ballRadius, c_ballInvInertia,
            c_ballWorldFrictionCombined, c_ballWorldRestitutionCombined,
            PhysConst::CARWORLD_COLLISION_FRICTION, PhysConst::CARWORLD_COLLISION_RESTITUTION,
            c_ballHitExtraForceScale, c_bumpForceScale, c_bumpCooldownTime,
            arena.tickCount, dt);

        // ============================================================
        // 5. POST-TICK
        // ============================================================
        for (int c = 0; c < numCars; c++) {
            device_car_post_tick(cars[c], dt);
            device_car_finish_tick(cars[c]);
        }

        // Boost pad collisions + pickup
        if (!isVoid) {
            for (int c = 0; c < numCars; c++) {
                if (cars[c].isDemoed) continue;
                for (int p = 0; p < BoostPadData::NUM_TOTAL; p++)
                    device_boostpad_check_collide(pads[p], p, cars[c]);
            }
            for (int p = 0; p < BoostPadData::NUM_TOTAL; p++)
                device_boostpad_post_tick(pads[p], p, cars, numCars, dt,
                                         c_boostPadCooldownBig, c_boostPadCooldownSmall);
        }

        // Ball finish tick
        device_ball_finish_tick(ball, c_ballMaxSpeed);

        // ============================================================
        // 6. GOAL CHECK
        // ============================================================
        if (!isVoid) {
            if (rs_abs(ball.pos.y) > c_goalThresholdY &&
                rs_abs(ball.pos.x) < PhysConst::GOAL_HALF_WIDTH &&
                ball.pos.z < PhysConst::GOAL_HEIGHT) {
                arena.goalScored = true;
                arena.goalTeam = (ball.pos.y > 0.f) ? 0 : 1; // 0=blue scored (ball in orange end)
            }
        }

        arena.tickCount++;
    }
}

