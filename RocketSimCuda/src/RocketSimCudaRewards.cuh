#pragma once
// Training rewards + terminals
// Extracted from RocketSimCuda.cu (modular split; same TU via include).

// ============================================================================
// Training rewards + terminals
// ============================================================================

static constexpr int MAX_TRAINING_CARS_PER_ARENA = 8;

__device__ __forceinline__ float training_vec_length(Vec3 v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ __forceinline__ bool training_ball_touched_step(const GpuCarState& car, uint64_t tickCount, int stepTicks) {
    return car.ballHitValid && car.ballHitTickCount >= (tickCount - (uint64_t)stepTicks);
}

__device__ __forceinline__ int training_find_car_idx_by_id(const GpuCarState* cars, int numCars, uint32_t id) {
    for (int i = 0; i < numCars; i++) {
        if (cars[i].id == id)
            return i;
    }
    return -1;
}

__device__ __forceinline__ bool training_is_new_contact(
    const GpuCarState& cur,
    const GpuCarState& prev
) {
    if (!cur.carContactOtherID || cur.carContactCooldownTimer <= 0.f)
        return false;

    return
        prev.carContactOtherID != cur.carContactOtherID ||
        prev.carContactCooldownTimer <= 0.f ||
        prev.carContactCooldownTimer < cur.carContactCooldownTimer;
}

__device__ float training_eval_reward(
    const TrainingRewardEntry& entry,
    const GpuCarState* cars,
    const GpuCarState* prevCars,
    const GpuBallState& ball,
    const GpuBallState& prevBall,
    const GpuArenaState& arena,
    int numCars,
    int carIdx,
    int stepTicks,
    GpuRewardAuxState* rewardAuxRow,
    uint32_t lastTouchCarID
) {
    const GpuCarState& car = cars[carIdx];
    const GpuCarState& prevCar = prevCars[carIdx];
    bool touchedStep = training_ball_touched_step(car, arena.tickCount, stepTicks);

    switch (entry.id) {
    case TrainingRewardID::GOAL_REWARD:
        if (!arena.goalScored || arena.goalTeam < 0)
            return 0.f;
        return ((int)car.team == arena.goalTeam) ? 1.f : entry.params[0];

    case TrainingRewardID::VELOCITY_BALL_TO_GOAL: {
        bool targetOrangeGoal = (car.team == (uint8_t)Team::BLUE);
        if (entry.params[0] > 0.5f)
            targetOrangeGoal = !targetOrangeGoal;

        float targetY = targetOrangeGoal ? 6000.f : -6000.f;
        Vec3 dir = { -ball.pos.x, targetY - ball.pos.y, 321.3875f - ball.pos.z };
        dir = v3_safe_normalize(dir);
        Vec3 normBallVel = ball.vel / PhysConst::BALL_MAX_SPEED;
        return v3_dot(dir, normBallVel);
    }

    case TrainingRewardID::VELOCITY_PLAYER_TO_BALL: {
        Vec3 dirToBall = v3_safe_normalize(ball.pos - car.pos);
        Vec3 normVel = car.vel / PhysConst::CAR_MAX_SPEED;
        return v3_dot(dirToBall, normVel);
    }

    case TrainingRewardID::FACE_BALL: {
        Vec3 dirToBall = v3_safe_normalize(ball.pos - car.pos);
        return v3_dot(car.rotMat.forward, dirToBall);
    }

    case TrainingRewardID::TOUCH_BALL:
        return touchedStep ? 1.f : 0.f;

    case TrainingRewardID::TOUCH_ACCEL: {
        if (!touchedStep)
            return 0.f;

        constexpr float MAX_REWARDED_BALL_SPEED = 110.f * (100000.f / 36000.f);
        float prevSpeedFrac = rs_min(1.f, training_vec_length(prevBall.vel) / MAX_REWARDED_BALL_SPEED);
        float curSpeedFrac = rs_min(1.f, training_vec_length(ball.vel) / MAX_REWARDED_BALL_SPEED);
        return (curSpeedFrac > prevSpeedFrac) ? (curSpeedFrac - prevSpeedFrac) : 0.f;
    }

    case TrainingRewardID::STRONG_TOUCH: {
        if (!touchedStep)
            return 0.f;

        float hitForce = training_vec_length(ball.vel - prevBall.vel);
        if (hitForce < entry.params[0])
            return 0.f;
        return rs_min(1.f, hitForce / rs_max(entry.params[1], 1e-6f));
    }

    case TrainingRewardID::SPEED:
        return training_vec_length(car.vel) / PhysConst::CAR_MAX_SPEED;

    case TrainingRewardID::VELOCITY:
        return (training_vec_length(car.vel) / PhysConst::CAR_MAX_SPEED) * entry.params[0];

    case TrainingRewardID::AIR:
        return car.isOnGround ? 0.f : 1.f;

    case TrainingRewardID::WAVEDASH:
        return (car.isOnGround && prevCar.isFlipping && !prevCar.isOnGround) ? 1.f : 0.f;

    case TrainingRewardID::PICKUP_BOOST:
        if (car.boost > prevCar.boost)
            return sqrtf(car.boost / 100.f) - sqrtf(prevCar.boost / 100.f);
        return 0.f;

    case TrainingRewardID::SAVE_BOOST:
        return rs_clamp(powf(car.boost / 100.f, entry.params[0]), 0.f, 1.f);

    case TrainingRewardID::BUMP: {
        if (!training_is_new_contact(car, prevCar))
            return 0.f;
        int otherIdx = training_find_car_idx_by_id(cars, numCars, car.carContactOtherID);
        if (otherIdx < 0 || cars[otherIdx].team == car.team)
            return 0.f;
        return 1.f;
    }

    case TrainingRewardID::BUMPED_PENALTY: {
        for (int otherIdx = 0; otherIdx < numCars; otherIdx++) {
            if (otherIdx == carIdx || cars[otherIdx].team == car.team)
                continue;
            if (!training_is_new_contact(cars[otherIdx], prevCars[otherIdx]))
                continue;
            if (cars[otherIdx].carContactOtherID == car.id)
                return -1.f;
        }
        return 0.f;
    }

    case TrainingRewardID::DEMO: {
        if (!training_is_new_contact(car, prevCar))
            return 0.f;
        int otherIdx = training_find_car_idx_by_id(cars, numCars, car.carContactOtherID);
        if (otherIdx < 0 || cars[otherIdx].team == car.team)
            return 0.f;
        return (cars[otherIdx].isDemoed && !prevCars[otherIdx].isDemoed) ? 1.f : 0.f;
    }

    case TrainingRewardID::DEMOED_PENALTY:
        return (car.isDemoed && !prevCar.isDemoed) ? -1.f : 0.f;

    case TrainingRewardID::KICKOFF_PROXIMITY: {
        constexpr float KICKOFF_POS_XY_TOLERANCE = 1.0f;
        if (rs_abs(ball.pos.x) >= KICKOFF_POS_XY_TOLERANCE || rs_abs(ball.pos.y) >= KICKOFF_POS_XY_TOLERANCE)
            return 0.f;

        float playerDistSq = v3_dist_sq(car.pos, ball.pos);

        float minOpponentDistSq = 3.402823466e+38f;
        bool opponentFound = false;
        for (int i = 0; i < numCars; i++) {
            if (cars[i].id == car.id)
                continue;
            if (cars[i].team != car.team) {
                opponentFound = true;
                float d = v3_dist_sq(cars[i].pos, ball.pos);
                minOpponentDistSq = rs_min(minOpponentDistSq, d);
            }
        }

        if (!opponentFound || playerDistSq < minOpponentDistSq) {
            if (opponentFound && playerDistSq == minOpponentDistSq)
                return 0.f;
            return 1.f;
        }
        return -1.f;
    }

    case TrainingRewardID::TEAMMATE_BUMP_PENALTY: {
        uint32_t curId = car.carContactOtherID;
        float curTimer = car.carContactCooldownTimer;
        if (curTimer <= 0.f || curId == 0)
            return 0.f;

        uint32_t prevId = prevCar.carContactOtherID;
        float prevTimer = prevCar.carContactCooldownTimer;
        bool isNewBump = (curTimer > prevTimer) || (curId != prevId && prevTimer <= 0.f);
        if (!isNewBump)
            return 0.f;

        for (int i = 0; i < numCars; i++) {
            if (cars[i].id == curId) {
                if (cars[i].team == car.team)
                    return -1.f;
                break;
            }
        }
        return 0.f;
    }

    // Reserved team/aerial IDs â€” public stubs (no proprietary pack).
    case TrainingRewardID::PASS_INCENTIVE:
    case TrainingRewardID::TEAM_PASSING:
    case TrainingRewardID::MIXED_INITIATE:
    case TrainingRewardID::MIXED_DOUBLETAP:
    case TrainingRewardID::SPEED_AERIAL:
        (void)rewardAuxRow;
        (void)lastTouchCarID;
        return 0.f;

    case TrainingRewardID::PRESSURE_FLICK: {
        // Approx Leak PressureFlickReward: flip-touch near ball under pressure, ballâ†’goal speed.
        if (!touchedStep || !car.isFlipping)
            return 0.f;
        float dist = sqrtf(v3_dist_sq(prevCar.pos, prevBall.pos));
        if (dist > 250.f)
            return 0.f;
        float speedDiff = rs_abs(training_vec_length(prevCar.vel) - training_vec_length(prevBall.vel));
        if (speedDiff > 500.f)
            return 0.f;
        float closestOpp = 1e9f;
        for (int i = 0; i < numCars; i++) {
            if (cars[i].team == car.team || cars[i].id == car.id)
                continue;
            closestOpp = rs_min(closestOpp, sqrtf(v3_dist_sq(car.pos, cars[i].pos)));
        }
        if (closestOpp > 700.f)
            return 0.f;
        bool targetOrange = (car.team == (uint8_t)Team::BLUE);
        float targetY = targetOrange ? 5120.f : -5120.f;
        Vec3 dir = v3_safe_normalize(Vec3{ 0.f, targetY - ball.pos.y, 0.f });
        float velGoal = v3_dot(ball.vel, dir);
        if (velGoal <= 1000.f)
            return 0.f;
        float ratio = velGoal / 2920.f;
        float r = ratio * ratio * sqrtf(rs_max(ratio, 0.f)); // ~pow 2.5
        return rs_min(r, 2.f);
    }

    case TrainingRewardID::MAWKZY_FLICK: {
        // Lighter flick proxy: flip-touch that accelerates ball toward goal.
        if (!touchedStep || !car.isFlipping)
            return 0.f;
        bool targetOrange = (car.team == (uint8_t)Team::BLUE);
        float targetY = targetOrange ? 5120.f : -5120.f;
        Vec3 dir = v3_safe_normalize(Vec3{ 0.f, targetY - ball.pos.y, 0.f });
        float accel = v3_dot(ball.vel - prevBall.vel, dir);
        return rs_clamp(accel / 1500.f, 0.f, 1.f);
    }

    case TrainingRewardID::AIR_CARRY: {
        // Air + ball touch / soft carry shaping.
        if (car.isOnGround)
            return 0.f;
        if (touchedStep)
            return 1.f;
        float dist = sqrtf(v3_dist_sq(car.pos, ball.pos));
        if (dist > 300.f)
            return 0.f;
        float rel = training_vec_length(car.vel - ball.vel);
        return rs_clamp(1.f - rel / 1200.f, 0.f, 0.35f);
    }

    case TrainingRewardID::ANTI_BALL_STACK: {
        // Discourage sitting on ball / stacking (lightweight proxy).
        float dz = rs_abs(car.pos.z - ball.pos.z);
        float dxy = sqrtf((car.pos.x - ball.pos.x) * (car.pos.x - ball.pos.x)
            + (car.pos.y - ball.pos.y) * (car.pos.y - ball.pos.y));
        if (dxy < 120.f && car.pos.z > ball.pos.z && dz < 180.f && car.isOnGround)
            return -1.f;
        return 0.f;
    }

    default:
        return 0.f;
    }
}

__global__ void buildRewardsAndTerminalsKernel(
    const GpuCarState* allCars,
    const GpuCarState* prevCars,
    const GpuBallState* allBalls,
    const GpuBallState* prevBalls,
    const GpuArenaState* allArenas,
    GpuRewardAuxState* allRewardAuxStates,
    int numArenas,
    int maxCarsPerArena,
    int stepTicks,
    float* outRewards,
    uint8_t* outTerminals,
    float* noTouchTimers
) {
    int arenaIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (arenaIdx >= numArenas)
        return;

    const GpuArenaState& arena = allArenas[arenaIdx];
    const GpuCarState* cars = &allCars[arenaIdx * maxCarsPerArena];
    const GpuCarState* prevArenaCars = &prevCars[arenaIdx * maxCarsPerArena];
    const GpuBallState& ball = allBalls[arenaIdx];
    const GpuBallState& prevBall = prevBalls[arenaIdx];
    GpuRewardAuxState* rewardAuxRow = &allRewardAuxStates[arenaIdx * maxCarsPerArena];
    float* rewardRow = outRewards + arenaIdx * maxCarsPerArena;

    for (int i = 0; i < maxCarsPerArena; i++)
        rewardRow[i] = 0.f;

    // GameState.lastTouchCarID: the most recent ball toucher.
    uint32_t lastTouchCarID = 0;
    {
        uint64_t bestTick = 0;
        for (int i = 0; i < arena.numCars; i++) {
            if (cars[i].ballHitValid && cars[i].ballHitTickCount != ~0ULL &&
                cars[i].ballHitTickCount >= bestTick) {
                bestTick = cars[i].ballHitTickCount;
                lastTouchCarID = cars[i].id;
            }
        }
    }

    (void)rewardAuxRow;

    bool supportedArena = arena.numCars <= MAX_TRAINING_CARS_PER_ARENA;
    if (supportedArena) {
        float totalRewards[MAX_TRAINING_CARS_PER_ARENA] = {};
        float rawRewards[MAX_TRAINING_CARS_PER_ARENA] = {};

        for (int rewardIdx = 0; rewardIdx < c_numTrainingRewardEntries; rewardIdx++) {
            const TrainingRewardEntry& entry = c_trainingRewardEntries[rewardIdx];

            for (int carIdx = 0; carIdx < arena.numCars; carIdx++) {
                rawRewards[carIdx] = training_eval_reward(
                    entry, cars, prevArenaCars, ball, prevBall, arena, arena.numCars, carIdx, stepTicks,
                    rewardAuxRow, lastTouchCarID
                );
            }

            if (entry.isZeroSum) {
                for (int carIdx = 0; carIdx < arena.numCars; carIdx++) {
                    float teamSum = 0.f;
                    int teamCount = 0;
                    float oppSum = 0.f;
                    int oppCount = 0;
                    uint8_t myTeam = cars[carIdx].team;

                    for (int otherIdx = 0; otherIdx < arena.numCars; otherIdx++) {
                        if (cars[otherIdx].team == myTeam) {
                            teamSum += rawRewards[otherIdx];
                            teamCount++;
                        } else {
                            oppSum += rawRewards[otherIdx];
                            oppCount++;
                        }
                    }

                    float avgTeam = teamCount > 0 ? (teamSum / (float)teamCount) : 0.f;
                    float avgOpp = oppCount > 0 ? (oppSum / (float)oppCount) : 0.f;
                    float zeroSumReward =
                        rawRewards[carIdx] * (1.f - entry.teamSpirit) +
                        avgTeam * entry.teamSpirit -
                        avgOpp * entry.opponentScale;
                    totalRewards[carIdx] += zeroSumReward * entry.weight;
                }
            } else {
                for (int carIdx = 0; carIdx < arena.numCars; carIdx++)
                    totalRewards[carIdx] += rawRewards[carIdx] * entry.weight;
            }
        }

        for (int carIdx = 0; carIdx < arena.numCars; carIdx++)
            rewardRow[carIdx] = totalRewards[carIdx];
    }

    bool touchedThisStep = false;
    for (int carIdx = 0; carIdx < arena.numCars; carIdx++) {
        if (training_ball_touched_step(cars[carIdx], arena.tickCount, stepTicks)) {
            touchedThisStep = true;
            break;
        }
    }

    float noTouchTime = noTouchTimers[arenaIdx];
    if (touchedThisStep) {
        noTouchTime = 0.f;
    } else {
        noTouchTime += (float)stepTicks * c_tickTime;
    }
    noTouchTimers[arenaIdx] = noTouchTime;

    uint8_t terminal = 0;
    if (c_trainingTerminalConfig.useGoalScore && arena.goalScored)
        terminal = 1;
    if (!terminal &&
        c_trainingTerminalConfig.noTouchTimeoutSeconds > 0.f &&
        noTouchTime >= c_trainingTerminalConfig.noTouchTimeoutSeconds) {
        terminal = 2;
    }

    outTerminals[arenaIdx] = terminal;
}

