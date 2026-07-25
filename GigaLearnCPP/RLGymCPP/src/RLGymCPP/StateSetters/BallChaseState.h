#pragma once
#include "StateSetter.h"

namespace RLGC {

	// Spawns cars within maxSpawnDist of the ball (Necto BetterRandom-style early curriculum).
	class BallChaseState : public StateSetter {
	public:
		float maxSpawnDist;
		bool randBallSpeed;
		bool carsOnGround;

		BallChaseState(float maxSpawnDist = 2200.f, bool randBallSpeed = true, bool carsOnGround = true)
			: maxSpawnDist(maxSpawnDist), randBallSpeed(randBallSpeed), carsOnGround(carsOnGround) {}

		virtual void ResetArena(Arena* arena) override;
	};
}
