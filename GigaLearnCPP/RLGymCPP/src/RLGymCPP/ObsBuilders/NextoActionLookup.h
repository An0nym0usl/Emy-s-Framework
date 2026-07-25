#pragma once

#include "../Framework.h"
#include <array>
#include <vector>

namespace RLGC {

	// Discrete action lookup table for Nexto (ported from RLBotPack Nexto/agent.py).
	inline std::vector<std::array<float, 8>> MakeNextoLookupTable() {
		std::vector<std::array<float, 8>> actions;

		// Ground
		for (int throttle : { -1, 0, 1 }) {
			for (int steer : { -1, 0, 1 }) {
				for (int boost : { 0, 1 }) {
					for (int handbrake : { 0, 1 }) {
						if (boost == 1 && throttle != 1)
							continue;
						actions.push_back({
							(float)(throttle ? throttle : boost),
							(float)steer,
							0.f, (float)steer, 0.f, 0.f, (float)boost, (float)handbrake
						});
					}
				}
			}
		}

		// Aerial
		for (int pitch : { -1, 0, 1 }) {
			for (int yaw : { -1, 0, 1 }) {
				for (int roll : { -1, 0, 1 }) {
					for (int jump : { 0, 1 }) {
						for (int boost : { 0, 1 }) {
							if (jump == 1 && yaw != 0)
								continue;
							if (pitch == 0 && roll == 0 && jump == 0)
								continue;
							int handbrake = (jump == 1 && (pitch != 0 || yaw != 0 || roll != 0)) ? 1 : 0;
							actions.push_back({
								(float)boost,
								(float)yaw,
								(float)pitch,
								(float)yaw,
								(float)roll,
								(float)jump,
								(float)boost,
								(float)handbrake
							});
						}
					}
				}
			}
		}

		return actions;
	}

}
