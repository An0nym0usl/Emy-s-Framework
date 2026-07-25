#pragma once

#include "AdvancedObs.h"
#include <RLGymCPP/CommonValues.h>
#include <RLGymCPP/Gamestates/StateUtil.h>

namespace RLGC {

	class AdvancedObsPadded : public AdvancedObs {
	public:
		static constexpr int MAX_TEAMMATES = 2; 
		static constexpr int MAX_OPPONENTS = 3; 
		static constexpr int PLAYER_FEAT_SIZE = 29; 

		virtual FList BuildObs(const Player& player, const GameState& state) override {
			FList obs = {};

			bool inv = player.team == Team::ORANGE;

			auto ball = InvertPhys(state.ball, inv);
			auto& pads = state.GetBoostPads(inv);
			auto& padTimers = state.GetBoostPadTimers(inv);

			obs += ball.pos * POS_COEF;
			obs += ball.vel * VEL_COEF;
			obs += ball.angVel * ANG_VEL_COEF;
			for (int i = 0; i < player.prevAction.ELEM_AMOUNT; i++)
				obs += player.prevAction[i];

			for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
				if (pads[i]) {
					obs += 1.f;
				} else {
					obs += 1.f / (1.f + padTimers[i]);
				}
			}

			AddPlayerToObs(obs, player, inv, ball);

			FList teammates = {}, opponents = {};
			for (auto& otherPlayer : state.players) {
				if (otherPlayer.carId == player.carId)
					continue;
				AddPlayerToObs(
					(otherPlayer.team == player.team) ? teammates : opponents,
					otherPlayer, inv, ball
				);
			}

			auto padZeros = [&](FList& out, int playersMissing) {
				int elems = playersMissing * PLAYER_FEAT_SIZE;
				for (int i = 0; i < elems; i++) out += 0.f;
			};

			int curTMates = (int)(teammates.size() / PLAYER_FEAT_SIZE);
			int missTMates = MAX_TEAMMATES - curTMates;
			if (missTMates < 0) missTMates = 0; 
			obs += teammates;
			padZeros(obs, missTMates);

			int curOpps = (int)(opponents.size() / PLAYER_FEAT_SIZE);
			int missOpps = MAX_OPPONENTS - curOpps;
			if (missOpps < 0) missOpps = 0;
			obs += opponents;
			padZeros(obs, missOpps);

			return obs;
		}
	};
}
