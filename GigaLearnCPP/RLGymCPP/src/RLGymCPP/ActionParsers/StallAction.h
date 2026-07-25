#pragma once
#include "ActionParser.h"

namespace RLGC {

	class StallAction : public ActionParser {
	public:

		std::vector<Action> actions;
		std::vector<uint8_t> groundMask, airMask, jumpMask, boostMask;

		StallAction();

		virtual Action ParseAction(int index, const Player& player, const GameState& state) override {
			return actions[index];
		}

		virtual int GetActionAmount() override {
			return actions.size();
		}

		virtual std::vector<uint8_t> GetActionMask(const Player& player, const GameState& state) override;
	};
}
