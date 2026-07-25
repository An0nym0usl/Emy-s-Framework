#include "NectoObsBuilder.h"
#include "../Gamestates/StateUtil.h"
#include <cmath>

using namespace RocketSim;

namespace RLGC {

	namespace {
		constexpr float BOOST_LOCS[NectoObsTuple::NUM_BOOSTS][3] = {
			{0.f, -4240.f, 70.f}, {-1792.f, -4184.f, 70.f}, {1792.f, -4184.f, 70.f},
			{-3072.f, -4096.f, 73.f}, {3072.f, -4096.f, 73.f}, {-940.f, -3308.f, 70.f},
			{940.f, -3308.f, 70.f}, {0.f, -2816.f, 70.f}, {-3584.f, -2484.f, 70.f},
			{3584.f, -2484.f, 70.f}, {-1788.f, -2300.f, 70.f}, {1788.f, -2300.f, 70.f},
			{-2048.f, -1036.f, 70.f}, {0.f, -1024.f, 70.f}, {2048.f, -1036.f, 70.f},
			{-3584.f, 0.f, 73.f}, {-1024.f, 0.f, 70.f}, {1024.f, 0.f, 70.f},
			{3584.f, 0.f, 73.f}, {-2048.f, 1036.f, 70.f}, {0.f, 1024.f, 70.f},
			{2048.f, 1036.f, 70.f}, {-1788.f, 2300.f, 70.f}, {1788.f, 2300.f, 70.f},
			{-3584.f, 2484.f, 70.f}, {3584.f, 2484.f, 70.f}, {0.f, 2816.f, 70.f},
			{-940.f, 3310.f, 70.f}, {940.f, 3308.f, 70.f}, {-3072.f, 4096.f, 73.f},
			{3072.f, 4096.f, 73.f}, {-1792.f, 4184.f, 70.f}, {1792.f, 4184.f, 70.f},
			{0.f, 4240.f, 70.f},
		};
	}

	NectoObsBuilder::NectoObsBuilder() {
		for (int i = 0; i < 5; i++) _norm[i] = 1.f;
		for (int i = 0; i < 6; i++) {
			_norm[POS + i] = 2300.f;
			_norm[LIN_VEL + i] = 2300.f;
		}
		for (int i = 0; i < 6; i++) _norm[FW + i] = 1.f;
		for (int i = 0; i < 3; i++) _norm[ANG_VEL + i] = 5.5f;
		for (int i = 20; i < 24; i++) _norm[i] = 1.f;

		for (int i = 0; i < 5; i++) _invert[i] = 1.f;
		for (int g = 0; g < 5; g++) {
			_invert[POS + g * 3 + 0] = -1.f;
			_invert[POS + g * 3 + 1] = -1.f;
			_invert[POS + g * 3 + 2] = 1.f;
		}
		for (int i = 20; i < 24; i++) _invert[i] = 1.f;

		InitBoostLocations();
	}

	void NectoObsBuilder::InitBoostLocations() {
		for (int i = 0; i < NectoObsTuple::NUM_BOOSTS; i++) {
			_boostLocations[i] = Vec(BOOST_LOCS[i][0], BOOST_LOCS[i][1], BOOST_LOCS[i][2]);
			_boostTypes[i] = BOOST_LOCS[i][2] > 72.f;
		}
	}

	void NectoObsBuilder::Reset(const GameState& state) {
		_demoTimers.clear();
		_boostTimers.fill(0.f);
		_timersInit = true;
		(void)state;
	}

	void NectoObsBuilder::UpdateSharedQkv(
		const GameState& state,
		std::vector<std::array<float, NectoObsTuple::FEAT>>& rows,
		int& nPlayers) {

		if (!_timersInit)
			Reset(state);

		nPlayers = RS_MIN((int)state.players.size(), MAX_PLAYERS);
		const int nEntities = 1 + nPlayers + NectoObsTuple::NUM_BOOSTS;
		rows.assign(nEntities, {});
		for (auto& row : rows)
			row.fill(0.f);

		// Ball (index 0)
		{
			auto& row = rows[0];
			row[IS_BALL] = 1.f;
			row[POS + 0] = state.ball.pos.x;
			row[POS + 1] = state.ball.pos.y;
			row[POS + 2] = state.ball.pos.z;
			row[LIN_VEL + 0] = state.ball.vel.x;
			row[LIN_VEL + 1] = state.ball.vel.y;
			row[LIN_VEL + 2] = state.ball.vel.z;
			row[ANG_VEL + 0] = state.ball.angVel.x;
			row[ANG_VEL + 1] = state.ball.angVel.y;
			row[ANG_VEL + 2] = state.ball.angVel.z;
		}

		// Players (index 1..nPlayers)
		for (int i = 0; i < nPlayers; i++) {
			const auto& p = state.players[i];
			auto& row = rows[1 + i];

			if (p.team == Team::BLUE)
				row[IS_MATE] = 1.f;
			else
				row[IS_OPP] = 1.f;

			row[POS + 0] = p.pos.x;
			row[POS + 1] = p.pos.y;
			row[POS + 2] = p.pos.z;
			row[LIN_VEL + 0] = p.vel.x;
			row[LIN_VEL + 1] = p.vel.y;
			row[LIN_VEL + 2] = p.vel.z;
			row[FW + 0] = p.rotMat.forward.x;
			row[FW + 1] = p.rotMat.forward.y;
			row[FW + 2] = p.rotMat.forward.z;
			row[UP + 0] = p.rotMat.up.x;
			row[UP + 1] = p.rotMat.up.y;
			row[UP + 2] = p.rotMat.up.z;
			row[ANG_VEL + 0] = p.angVel.x;
			row[ANG_VEL + 1] = p.angVel.y;
			row[ANG_VEL + 2] = p.angVel.z;
			row[BOOST] = p.boost / 100.f;
			row[ON_GROUND] = p.isOnGround ? 1.f : 0.f;
			row[HAS_FLIP] = p.HasFlipOrJump() ? 1.f : 0.f;

			float& demoT = _demoTimers[p.carId];
			if (demoT <= 0.f)
				demoT = 3.f;
			else
				demoT = RS_MAX(demoT - TICK_SKIP / 120.f, 0.f);
			row[DEMO] = demoT / 10.f;
		}

		// Boost pads
		const auto& pads = state.boostPads;
		for (int i = 0; i < NectoObsTuple::NUM_BOOSTS; i++) {
			auto& row = rows[1 + nPlayers + i];
			row[IS_BOOST] = 1.f;
			row[POS + 0] = _boostLocations[i].x;
			row[POS + 1] = _boostLocations[i].y;
			row[POS + 2] = _boostLocations[i].z;
			row[BOOST] = 0.12f + 0.88f * (_boostTypes[i] ? 1.f : 0.f);
		}

		// Boost pad timers (matches necto_obs.py)
		for (int i = 0; i < NectoObsTuple::NUM_BOOSTS; i++) {
			bool padActive = i < (int)pads.size() ? pads[i] : true;
			bool newGrab = padActive && _boostTimers[i] == 0.f;
			if (newGrab)
				_boostTimers[i] = 0.4f + 0.6f * (_boostTypes[i] ? 1.f : 0.f);
			if (!padActive)
				_boostTimers[i] = 0.f;
			rows[1 + nPlayers + i][DEMO] = _boostTimers[i];
		}
		for (int i = 0; i < NectoObsTuple::NUM_BOOSTS; i++)
			_boostTimers[i] = RS_MAX(_boostTimers[i] - TICK_SKIP / 1200.f, 0.f);

		// Normalize
		for (auto& row : rows) {
			for (int f = 0; f < NectoObsTuple::FEAT; f++)
				row[f] /= _norm[f];
		}
	}

	NectoObsTuple NectoObsBuilder::Build(
		const Player& player, const GameState& state, const std::array<float, 8>& prevAction) {

		std::vector<std::array<float, NectoObsTuple::FEAT>> rows;
		int nPlayers = 0;
		UpdateSharedQkv(state, rows, nPlayers);
		const int nEntities = (int)rows.size();

		int mainN = player.index + 1;
		if (mainN <= 0 || mainN > nPlayers) {
			for (int i = 0; i < nPlayers; i++) {
				if (state.players[i].carId == player.carId) {
					mainN = i + 1;
					break;
				}
			}
		}
		if (mainN <= 0)
			mainN = 1;

		rows[mainN][IS_SELF] = 1.f;

		const bool invert = player.team == Team::ORANGE;
		if (invert) {
			for (auto& row : rows) {
				std::swap(row[IS_MATE], row[IS_OPP]);
				for (int f = 0; f < NectoObsTuple::FEAT; f++)
					row[f] *= _invert[f];
			}
		}

		NectoObsTuple result;
		result.nEntities = nEntities;
		result.kv.resize(nEntities * NectoObsTuple::FEAT);
		for (int r = 0; r < nEntities; r++)
			for (int f = 0; f < NectoObsTuple::FEAT; f++)
				result.kv[r * NectoObsTuple::FEAT + f] = rows[r][f];

		result.q.assign(NectoObsTuple::Q_SIZE, 0.f);
		for (int f = 0; f < NectoObsTuple::FEAT; f++)
			result.q[f] = rows[mainN][f];
		for (int i = 0; i < 8; i++)
			result.q[NectoObsTuple::FEAT + i] = prevAction[i];

		// Relative coords: kv[:, 5:11] -= q[5:11]
		for (int r = 0; r < nEntities; r++) {
			for (int d = 0; d < 6; d++)
				result.kv[r * NectoObsTuple::FEAT + POS + d] -= result.q[POS + d];
		}

		result.m.assign(nEntities, 0.f);

		return result;
	}

}
