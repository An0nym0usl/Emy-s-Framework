#include "BallChaseState.h"
#include "../Math.h"
#include "../CommonValues.h"

using RocketSim::Math::RandFloat;
using RLGC::Math::RandVec;

void RLGC::BallChaseState::ResetArena(Arena* arena) {
	arena->ResetToRandomKickoff();

	constexpr float
		X_MAX = 3200.f,
		Y_MAX = 3800.f,
		Z_MAX = 1600.f,
		CAR_Z_GROUND = 17.f,
		BALL_Z_MIN = CommonValues::BALL_RADIUS;

	// Ball near field center with moderate spread
	{
		BallState bs = {};
		bs.pos = RandVec(
			Vec(-X_MAX * 0.6f, -Y_MAX * 0.6f, BALL_Z_MIN),
			Vec(X_MAX * 0.6f, Y_MAX * 0.6f, RS_MIN(Z_MAX, 900.f))
		);
		if (randBallSpeed) {
			Vec dir = RandVec(Vec(-1, -1, -1), Vec(1, 1, 1)).Normalized();
			bs.vel = dir * RandFloat(0.f, 2200.f);
			bs.angVel = RandVec(Vec(-3, -3, -3), Vec(3, 3, 3));
		}
		arena->ball->SetState(bs);
	}

	Vec ballPos = arena->ball->GetState().pos;

	for (Car* car : arena->_cars) {
		CarState cs = {};
		Vec offset = RandVec(Vec(-1, -1, -0.2f), Vec(1, 1, 0.2f)).Normalized();
		offset *= RandFloat(400.f, maxSpawnDist);
		cs.pos = ballPos + offset;
		cs.pos.x = RS_CLAMP(cs.pos.x, -X_MAX, X_MAX);
		cs.pos.y = RS_CLAMP(cs.pos.y, -Y_MAX, Y_MAX);

		if (carsOnGround) {
			cs.pos.z = CAR_Z_GROUND;
			cs.vel = RandVec(Vec(-800, -800, 0), Vec(800, 800, 0));
			cs.angVel = {};
			cs.rotMat = Angle(RandFloat(-M_PI, M_PI), 0, 0).ToRotMat();
		} else {
			cs.pos.z = RS_CLAMP(cs.pos.z, CAR_Z_GROUND, Z_MAX);
			cs.vel = RandVec(Vec(-1200, -1200, -200), Vec(1200, 1200, 800));
			cs.angVel = RandVec(Vec(-4, -4, -4), Vec(4, 4, 4));
			cs.rotMat = Angle(
				RandFloat(-M_PI, M_PI),
				RandFloat(-M_PI / 4, M_PI / 4),
				RandFloat(-M_PI / 4, M_PI / 4)
			).ToRotMat();
		}

		cs.boost = RandFloat(30.f, 100.f);
		car->SetState(cs);
	}
}
