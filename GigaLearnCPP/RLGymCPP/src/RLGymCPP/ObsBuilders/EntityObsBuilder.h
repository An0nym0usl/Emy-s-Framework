#pragma once

#include "AdvancedObs.h"
#include <RLGymCPP/CommonValues.h>
#include <RLGymCPP/Gamestates/StateUtil.h>

namespace RLGC {

	// Observation builder for TRUE multi-entity attention (see GGL::AttentionModel with entityMode).
	//
	// Instead of one flat vector, the observation is laid out as a fixed grid of equal-width tokens:
	//   token 0            : ball / global context (ball physics + own previous action + boost pads)
	//   token 1            : the acting player ("self")
	//   tokens 2..         : teammates (zero-padded up to MAX_TEAMMATES)
	//   tokens ..          : opponents (zero-padded up to MAX_OPPONENTS)
	//
	// Every token has the same width ENTITY_FEAT and ends with:
	//   [.. features ..][role_self, role_teammate, role_opponent, role_ball][validity]
	// The validity flag (last feature) is 1 for real entities and 0 for padding; the attention head
	// uses it to mask out missing players. The grid is returned flattened (row-major) so it flows
	// through the existing flat-observation pipeline unchanged; the model reshapes it back to
	// (numEntities, ENTITY_FEAT) internally.
	class EntityObsBuilder : public AdvancedObs {
	public:
		static constexpr int MAX_TEAMMATES = 2;
		static constexpr int MAX_OPPONENTS = 3;

		// 1 self + teammates + opponents, plus a dedicated ball/context token.
		static constexpr int NUM_CARS = 1 + MAX_TEAMMATES + MAX_OPPONENTS; // 6
		static constexpr int NUM_ENTITIES = 1 + NUM_CARS;                  // 7

		// Role one-hot (self, teammate, opponent, ball) + validity flag.
		static constexpr int ROLE_COUNT = 4;
		static constexpr int ENTITY_FEAT = 64; // Token width (>= largest raw token); rest zero-padded
		static constexpr int FEATURE_AREA = ENTITY_FEAT - ROLE_COUNT - 1; // Feature slots per token (59)

		enum Role { ROLE_SELF = 0, ROLE_TEAMMATE = 1, ROLE_OPPONENT = 2, ROLE_BALL = 3 };

		// Total flattened observation size (what you pass as obsSize to the Learner / InferUnit).
		static constexpr int GetObsSize() { return NUM_ENTITIES * ENTITY_FEAT; }

		virtual FList BuildObs(const Player& player, const GameState& state) override {
			FList obs = {};
			obs.reserve(GetObsSize());

			bool inv = player.team == Team::ORANGE;
			auto ball = InvertPhys(state.ball, inv);
			auto& pads = state.GetBoostPads(inv);
			auto& padTimers = state.GetBoostPadTimers(inv);

			// Append one fixed-width token: feature area (zero-padded) + role one-hot + validity.
			auto pushToken = [&](const FList& feat, int role, float valid) {
				for (int i = 0; i < FEATURE_AREA; i++)
					obs += (i < (int)feat.size() ? feat[i] : 0.f);
				for (int r = 0; r < ROLE_COUNT; r++)
					obs += (r == role ? 1.f : 0.f);
				obs += valid;
			};

			// --- Ball / global context token ---
			{
				FList feat = {};
				feat += ball.pos * POS_COEF;
				feat += ball.vel * VEL_COEF;
				feat += ball.angVel * ANG_VEL_COEF;
				for (int i = 0; i < player.prevAction.ELEM_AMOUNT; i++)
					feat += player.prevAction[i];
				for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++)
					feat += pads[i] ? 1.f : (1.f / (1.f + padTimers[i]));
				pushToken(feat, ROLE_BALL, 1.f);
			}

			// --- Self token ---
			{
				FList feat = {};
				AddPlayerToObs(feat, player, inv, ball);
				pushToken(feat, ROLE_SELF, 1.f);
			}

			// --- Teammates (padded to MAX_TEAMMATES) ---
			int mateCount = 0;
			for (auto& other : state.players) {
				if (other.carId == player.carId || other.team != player.team)
					continue;
				if (mateCount >= MAX_TEAMMATES)
					break;
				FList feat = {};
				AddPlayerToObs(feat, other, inv, ball);
				pushToken(feat, ROLE_TEAMMATE, 1.f);
				mateCount++;
			}
			for (; mateCount < MAX_TEAMMATES; mateCount++)
				pushToken(FList{}, ROLE_TEAMMATE, 0.f);

			// --- Opponents (padded to MAX_OPPONENTS) ---
			int oppCount = 0;
			for (auto& other : state.players) {
				if (other.carId == player.carId || other.team == player.team)
					continue;
				if (oppCount >= MAX_OPPONENTS)
					break;
				FList feat = {};
				AddPlayerToObs(feat, other, inv, ball);
				pushToken(feat, ROLE_OPPONENT, 1.f);
				oppCount++;
			}
			for (; oppCount < MAX_OPPONENTS; oppCount++)
				pushToken(FList{}, ROLE_OPPONENT, 0.f);

			return obs;
		}
	};
}
