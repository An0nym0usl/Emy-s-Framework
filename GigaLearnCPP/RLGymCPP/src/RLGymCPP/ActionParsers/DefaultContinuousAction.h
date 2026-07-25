#pragma once
#include "ActionParser.h"

namespace RLGC {

	// Default continuous action parser for Rocket League
	// Maps 8 continuous float values [-1, 1] to Action struct
	// throttle, steer, pitch, yaw, roll: use float directly
	// jump, boost, handbrake: threshold at 0 (>0 -> 1, <=0 -> 0)
	class DefaultContinuousAction : public ContinuousActionParser {
	public:

		virtual Action ParseContinuousAction(
			const float* actionValues, int actionSize,
			const Player& player, const GameState& state
		) override {
			RG_ASSERT(actionSize >= 8);

			Action action;
			action.throttle = actionValues[0];   // [-1, 1]
			action.steer    = actionValues[1];   // [-1, 1]
			action.pitch    = actionValues[2];   // [-1, 1]
			action.yaw      = actionValues[3];   // [-1, 1]
			action.roll     = actionValues[4];   // [-1, 1]

			// Binary controls: threshold at 0
			action.jump      = actionValues[5] > 0.0f ? 1.0f : 0.0f;
			action.boost     = actionValues[6] > 0.0f ? 1.0f : 0.0f;
			action.handbrake = actionValues[7] > 0.0f ? 1.0f : 0.0f;

			return action;
		}

		virtual int GetContinuousActionSize() override { return 8; }
	};
}
