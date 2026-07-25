#include "GameState.h"

#include "../Math.h"

using namespace RLGC;

static int boostPadIndexMap[CommonValues::BOOST_LOCATIONS_AMOUNT] = {};
static bool boostPadIndexMapBuilt = false;
static std::mutex boostPadIndexMapMutex = {};
void _BuildBoostPadIndexMap(Arena* arena) {
	constexpr const char* ERROR_PREFIX = "_BuildBoostPadIndexMap(): ";
#ifdef RG_VERBOSE
	RG_LOG("Building boost pad index map...");
#endif

	if (arena->_boostPads.size() != CommonValues::BOOST_LOCATIONS_AMOUNT) {
		RG_ERR_CLOSE(
			ERROR_PREFIX << "Arena boost pad count does not match CommonValues::BOOST_LOCATIONS_AMOUNT " <<
			"(" << arena->_boostPads.size() << "/" << CommonValues::BOOST_LOCATIONS_AMOUNT << ")"
		);
	}
	
	bool found[CommonValues::BOOST_LOCATIONS_AMOUNT] = {};
	for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
		Vec targetPos = CommonValues::BOOST_LOCATIONS[i];
		for (int j = 0; j < arena->_boostPads.size(); j++) {
			Vec padPos = arena->_boostPads[j]->config.pos;

			if (padPos.DistSq2D(targetPos) < 10) {
				if (!found[i]) {
					found[i] = true;
					boostPadIndexMap[i] = j;
				} else {
					RG_ERR_CLOSE(
						ERROR_PREFIX << "Matched duplicate boost pad at " << targetPos << "=" << padPos
					);
				}
				break;
			}
		}

		if (!found[i])
			RS_ERR_CLOSE(ERROR_PREFIX << "Failed to find matching pad at " << targetPos);
	}

#ifdef RG_VERBOSE
	RG_LOG(" > Done");
#endif
	boostPadIndexMapBuilt = true;
}

void RLGC::GameState::ResetBeforeStep() {
	for (auto& player : players)
		player.ResetBeforeStep();
}

void RLGC::GameState::UpdateFromArena(Arena* arena, const std::vector<Action>& actions, GameState* prev) {
	this->prev = prev;
	if (prev)
		prev->prev = NULL;

	lastArena = arena;
	int tickSkip = RS_MAX(arena->tickCount - lastTickCount, 0);
	deltaTime = tickSkip * (1 / 120.f);

	ball = arena->ball->GetState();

	players.resize(arena->_cars.size());

	auto carItr = arena->_cars.begin();
	for (int i = 0; i < players.size(); i++) {
		auto& player = players[i];
		player.index = i;
		player.UpdateFromCar(*carItr, arena->tickCount, tickSkip, actions[i], prev ? &prev->players[i] : NULL);
		if (player.ballTouchedStep)
			lastTouchCarID = player.carId;

		carItr++;
	}

	if (!boostPadIndexMapBuilt) {
		boostPadIndexMapMutex.lock();
		// Check again? This seems stupid but also makes sense to me
		//	Without this, we could lock as the index map is building, then go build again
		//	I would like to keep the mutex inside the if statement so it is only checked a few times
		if (!boostPadIndexMapBuilt) 
			_BuildBoostPadIndexMap(arena);
		boostPadIndexMapMutex.unlock();
	}

	int numBoostPads = arena->_boostPads.size();
	boostPads.resize(numBoostPads);
	boostPadsInv.resize(numBoostPads);
	boostPadTimers.resize(numBoostPads);
	boostPadTimersInv.resize(numBoostPads);
	for (int i = 0; i < arena->_boostPads.size(); i++) {
		int idx = boostPadIndexMap[i];
		int invIdx = boostPadIndexMap[CommonValues::BOOST_LOCATIONS_AMOUNT - i - 1];

		auto state = arena->_boostPads[idx]->GetState();
		auto stateInv = arena->_boostPads[invIdx]->GetState();

		boostPads[i] = state.isActive;
		boostPadsInv[i] = stateInv.isActive;

		boostPadTimers[i] = state.cooldown;
		boostPadTimersInv[i] = stateInv.cooldown;
	}

	// Update goal scoring
	// If you don't have a GoalScoreCondition then that's not my problem lmao
	goalScored = arena->IsBallScored();

	lastTickCount = arena->tickCount;
}

#ifdef GIGA_USE_CUDA_SIM
#include <RocketSimCuda.h>

namespace {
	inline ::RocketSim::Vec ToRSVec(const rsc::Vec3& v) { return ::RocketSim::Vec(v.x, v.y, v.z); }
	inline ::RocketSim::RotMat ToRSRot(const rsc::RotMat& m) {
		::RocketSim::RotMat r;
		r.forward = ToRSVec(m.forward);
		r.right   = ToRSVec(m.right);
		r.up      = ToRSVec(m.up);
		return r;
	}
}

GameState::GameState(rsc::RocketSimCudaBatch& batch, int arenaIdx) {
	// Default-initialize boost pad vectors like the default ctor, then fill from the batch.
	boostPads = std::vector<bool>(CommonValues::BOOST_LOCATIONS_AMOUNT, true);
	boostPadsInv = std::vector<bool>(CommonValues::BOOST_LOCATIONS_AMOUNT, true);
	boostPadTimers = std::vector<float>(CommonValues::BOOST_LOCATIONS_AMOUNT, 0);
	boostPadTimersInv = std::vector<float>(CommonValues::BOOST_LOCATIONS_AMOUNT, 0);

	std::vector<Action> defaultActions(batch.GetMaxCarsPerArena());
	UpdateFromCudaBatch(batch, arenaIdx, defaultActions, nullptr);
}

void GameState::UpdateFromCudaBatch(rsc::RocketSimCudaBatch& batch, int arenaIdx,
	const std::vector<Action>& actions, GameState* prev) {

	this->prev = prev;
	if (prev)
		prev->prev = NULL;

	lastArena = nullptr; // No CPU arena backs this state; rewards must not rely on lastArena here.

	rsc::ArenaInfo info = batch.GetArenaInfo(arenaIdx);
	uint64_t tickCount = info.tickCount;
	int tickSkip = (int)RS_MAX((int64_t)tickCount - (int64_t)lastTickCount, (int64_t)0);
	deltaTime = tickSkip * (1 / 120.f);

	// --- Ball ---
	rsc::BallState gBall = batch.GetBallState(arenaIdx);
	ball.pos = ToRSVec(gBall.pos);
	ball.rotMat = ToRSRot(gBall.rotMat);
	ball.vel = ToRSVec(gBall.vel);
	ball.angVel = ToRSVec(gBall.angVel);

	// --- Cars ---
	int numCars = batch.GetMaxCarsPerArena();
	players.resize(numCars);
	for (int i = 0; i < numCars; i++) {
		auto& player = players[i];
		rsc::CarState g = batch.GetCarState(arenaIdx, i);

		player.index = i;
		player.carId = g.id;
		player.team = (Team)(uint8_t)g.team;

		// Map the GPU car state into the underlying RocketSim CarState (mirrors CopyGpuToCpuCarState).
		::RocketSim::CarState cs = {};
		cs.pos = ToRSVec(g.pos);
		cs.rotMat = ToRSRot(g.rotMat);
		cs.vel = ToRSVec(g.vel);
		cs.angVel = ToRSVec(g.angVel);
		cs.boost = g.boost;
		cs.timeSpentBoosting = g.timeSpentBoosting;
		cs.isOnGround = g.isOnGround;
		cs.hasJumped = g.hasJumped;
		cs.hasDoubleJumped = g.hasDoubleJumped;
		cs.hasFlipped = g.hasFlipped;
		cs.isFlipping = g.isFlipping;
		cs.isJumping = g.isJumping;
		cs.flipRelTorque = ToRSVec(g.flipRelTorque);
		cs.jumpTime = g.jumpTime;
		cs.flipTime = g.flipTime;
		cs.airTime = g.airTime;
		cs.airTimeSinceJump = g.airTimeSinceJump;
		cs.isSupersonic = g.isSupersonic;
		cs.supersonicTime = g.supersonicTime;
		cs.handbrakeVal = g.handbrakeVal;
		cs.isAutoFlipping = g.isAutoFlipping;
		cs.autoFlipTimer = g.autoFlipTimer;
		cs.autoFlipTorqueScale = g.autoFlipTorqueScale;
		cs.isDemoed = g.isDemoed;
		cs.demoRespawnTimer = g.demoRespawnTimer;
		cs.ballHitInfo.isValid = g.ballHitValid;
		cs.ballHitInfo.tickCountWhenHit = g.ballHitTickCount;

		*(::RocketSim::CarState*)&player = cs;

		if (cs.ballHitInfo.isValid) {
			player.ballTouchedStep = cs.ballHitInfo.tickCountWhenHit >= (tickCount - tickSkip);
			player.ballTouchedTick = cs.ballHitInfo.tickCountWhenHit == (tickCount - 1);
		} else {
			player.ballTouchedStep = player.ballTouchedTick = false;
		}

		player.prevAction = (i < (int)actions.size()) ? actions[i] : Action{};

		if (player.ballTouchedStep)
			lastTouchCarID = player.carId;
	}

	// --- Boost pads (GPU returns them in CommonValues::BOOST_LOCATIONS order) ---
	const int numPads = CommonValues::BOOST_LOCATIONS_AMOUNT;
	std::vector<rsc::BoostPadState> pads(numPads);
	batch.GetBoostPadStates(arenaIdx, pads.data());
	boostPads.resize(numPads);
	boostPadsInv.resize(numPads);
	boostPadTimers.resize(numPads);
	boostPadTimersInv.resize(numPads);
	for (int i = 0; i < numPads; i++) {
		int invI = numPads - i - 1;
		boostPads[i] = pads[i].isActive;
		boostPadsInv[i] = pads[invI].isActive;
		boostPadTimers[i] = pads[i].cooldown;
		boostPadTimersInv[i] = pads[invI].cooldown;
	}

	goalScored = info.goalScored;
	lastTickCount = tickCount;
}
#endif // GIGA_USE_CUDA_SIM
