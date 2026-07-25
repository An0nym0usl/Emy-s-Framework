#pragma once

#include <RLGymCPP/RewardCore/Reward.h>
#include <RLGymCPP/CommonValues.h>

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <unordered_map>

namespace RLGC {

// OPTIONAL recipe file (not wired into the blank handoff defaults).
// Include and push into EnvCreateDefault / CudaEnvSet only if you want these.
// Ported historically from GigaLearnCPP-Leak default reward stack.
class PressureFlickReward : public Reward {
public:
	const float PANIC_DISTANCE = 700.0f;
	const float MIN_FLICK_SPEED = 1000.0f;
	const float TARGET_FLICK_SPEED = 2920.0f;
	const float EXPONENT = 2.5f;

	virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
		(void)isFinal;
		if (!player.ballTouchedStep || !player.isFlipping)
			return 0.0f;
		if (!state.prev || !player.prev)
			return 0.0f;

		float dist = player.prev->pos.Dist(state.prev->ball.pos);
		if (dist > 250.0f)
			return 0.0f;

		float speedDiff = std::abs(player.prev->vel.Length() - state.prev->ball.vel.Length());
		if (speedDiff > 500.0f)
			return 0.0f;

		float closestOppDist = 100000.0f;
		for (const auto& p : state.players) {
			if (p.team != player.team) {
				float d = player.pos.Dist(p.pos);
				if (d < closestOppDist)
					closestOppDist = d;
			}
		}
		if (closestOppDist > PANIC_DISTANCE)
			return 0.0f;

		bool targetOrange = player.team == Team::BLUE;
		Vec targetPos = targetOrange ? CommonValues::ORANGE_GOAL_BACK : CommonValues::BLUE_GOAL_BACK;
		Vec dirToGoal = (targetPos - state.ball.pos).Normalized();
		float velTowardsGoal = state.ball.vel.Dot(dirToGoal);

		if (velTowardsGoal > MIN_FLICK_SPEED) {
			float ratio = velTowardsGoal / TARGET_FLICK_SPEED;
			float reward = std::pow(ratio, EXPONENT);
			return std::min(reward, 2.0f);
		}
		return 0.0f;
	}
};

// Test separately vs PressureFlickReward — default: don't run both at once.
class PunishingFlickReward : public Reward {
public:
	const float CARRY_DIST_MAX = 250.0f;
	const float CARRY_SPEED_DIFF_MAX = 500.0f;
	const float PANIC_DISTANCE = 700.0f;
	const float MIN_FLICK_SPEED = 1000.0f;

	const float GOAL_HALF_WIDTH = 892.755f;
	const float GOAL_HEIGHT = 642.775f;

	const float POWER_EXPONENT = 1.5f;
	const float PLACEMENT_EXPONENT = 2.0f;
	const float SPEED_DIVISOR = 2300.0f;

	struct FlickData {
		bool didFlick = false;
		bool allCriteriaMet = false;
		float flickReward = 0.0f;
		Vec flickBallVel = Vec(0, 0, 0);
		Vec flickBallPos = Vec(0, 0, 0);
		Vec nearestOppPosAtFlick = Vec(0, 0, 0);
		Vec nearestOppVelAtFlick = Vec(0, 0, 0);
		int ticksSinceFlick = 9999;
	};

	std::unordered_map<uint32_t, FlickData> playerFlickData;

	virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
		(void)isFinal;
		FlickData& data = playerFlickData[player.carId];
		float reward = 0.0f;

		if (player.ballTouchedStep && player.isFlipping && state.prev) {
			float prevDist = player.prev->pos.Dist(state.prev->ball.pos);
			float prevSpeedDiff = std::abs(player.prev->vel.Length() - state.prev->ball.vel.Length());

			bool carryDistOk = prevDist < CARRY_DIST_MAX;
			bool speedDiffOk = prevSpeedDiff < CARRY_SPEED_DIFF_MAX;

			float closestOppDist = FLT_MAX;
			Vec closestOppPos, closestOppVel;
			for (const auto& opp : state.players) {
				if (opp.team == player.team)
					continue;
				float d = player.pos.Dist(opp.pos);
				if (d < closestOppDist) {
					closestOppDist = d;
					closestOppPos = opp.pos;
					closestOppVel = opp.vel;
				}
			}
			bool pressureOk = closestOppDist < PANIC_DISTANCE;

			int criteriaMet = (int)carryDistOk + (int)speedDiffOk + (int)pressureOk;

			if (criteriaMet >= 1) {
				bool targetOrange = player.team == Team::BLUE;
				Vec targetPos = targetOrange ? CommonValues::ORANGE_GOAL_BACK : CommonValues::BLUE_GOAL_BACK;
				Vec dirToGoal = (targetPos - state.ball.pos).Normalized();
				float velTowardsGoal = state.ball.vel.Dot(dirToGoal);

				if (velTowardsGoal > MIN_FLICK_SPEED) {
					data.didFlick = true;
					data.allCriteriaMet = (criteriaMet == 3);
					data.flickBallVel = state.ball.vel;
					data.flickBallPos = state.ball.pos;
					data.nearestOppPosAtFlick = closestOppPos;
					data.nearestOppVelAtFlick = closestOppVel;
					data.ticksSinceFlick = 0;

					float powerFactor = std::pow(state.ball.vel.Length() / SPEED_DIVISOR, POWER_EXPONENT);
					float directionFactor = velTowardsGoal / state.ball.vel.Length();

					reward = powerFactor * (1.0f + directionFactor);
					if (!data.allCriteriaMet)
						reward *= 0.5f;

					data.flickReward = reward;
				}
			}
		}

		data.ticksSinceFlick++;
		if (data.ticksSinceFlick > 600)
			data.didFlick = false;

		if (state.goalScored && data.didFlick) {
			bool scoredOnOrange = state.ball.pos.y > 0;
			bool weAreBlue = player.team == Team::BLUE;
			bool weScored = (weAreBlue && scoredOnOrange) || (!weAreBlue && !scoredOnOrange);

			if (weScored) {
				Vec goalCenter = weAreBlue ? CommonValues::ORANGE_GOAL_CENTER : CommonValues::BLUE_GOAL_CENTER;
				float goalY = goalCenter.y;

				float ballGoalX = state.ball.pos.x;
				float ballGoalZ = state.ball.pos.z;

				Vec oppPos = data.nearestOppPosAtFlick;
				Vec oppVel = data.nearestOppVelAtFlick;

				float predictionTime = 0.3f;
				Vec oppPredictedPos = oppPos + oppVel * predictionTime;

				float oppGoalX, oppGoalZ;
				float oppDistToGoal = std::abs(oppPos.y - goalY);

				if (oppDistToGoal < 1000.0f) {
					oppGoalX = oppPredictedPos.x;
					oppGoalZ = oppPredictedPos.z;
				} else {
					Vec flickDir = data.flickBallVel.Normalized();
					float t = (goalY - data.flickBallPos.y) / (flickDir.y + 0.001f);
					Vec ballAtGoal = data.flickBallPos + flickDir * std::max(t, 0.0f);
					Vec oppToIntercept = (ballAtGoal - oppPos).Normalized();
					Vec oppProjected = oppPos + oppToIntercept * oppVel.Length() * 0.5f;
					oppGoalX = oppProjected.x;
					oppGoalZ = std::max(oppProjected.z, 100.0f);
				}

				oppGoalX = std::clamp(oppGoalX, -GOAL_HALF_WIDTH, GOAL_HALF_WIDTH);
				oppGoalZ = std::clamp(oppGoalZ, 0.0f, GOAL_HEIGHT);

				float placementDistX = std::abs(ballGoalX - oppGoalX);
				float placementDistZ = std::abs(ballGoalZ - oppGoalZ);
				float placementDist = std::sqrt(placementDistX * placementDistX + placementDistZ * placementDistZ);

				float maxPlacementDist = std::sqrt(GOAL_HALF_WIDTH * 2 * GOAL_HALF_WIDTH * 2 + GOAL_HEIGHT * GOAL_HEIGHT);
				float placementRatio = placementDist / maxPlacementDist;

				float cornerBonus = 1.0f;
				bool oppWasLeft = oppGoalX < 0;
				bool ballWentRight = ballGoalX > 0;
				bool oppWasLow = oppGoalZ < GOAL_HEIGHT * 0.5f;
				bool ballWentHigh = ballGoalZ > GOAL_HEIGHT * 0.5f;

				if (oppWasLeft != ballWentRight)
					cornerBonus += 0.5f;
				if (oppWasLow != ballWentHigh)
					cornerBonus += 0.3f;

				bool inCorner = (std::abs(ballGoalX) > GOAL_HALF_WIDTH * 0.6f)
					&& (ballGoalZ > GOAL_HEIGHT * 0.6f || ballGoalZ < GOAL_HEIGHT * 0.25f);
				if (inCorner)
					cornerBonus += 0.3f;

				float placementFactor = std::pow(1.0f + placementRatio * 2.0f, PLACEMENT_EXPONENT);

				reward += data.flickReward * 2.0f * placementFactor * cornerBonus;
			}

			data.didFlick = false;
		}

		return reward;
	}
};

} // namespace RLGC
