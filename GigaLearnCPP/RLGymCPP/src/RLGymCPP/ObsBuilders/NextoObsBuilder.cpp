#include "NextoObsBuilder.h"
#include "../CommonValues.h"
#include "../Gamestates/StateUtil.h"
#include <cmath>

using namespace RocketSim;

namespace RLGC {

	namespace {
		constexpr float BOOST_LOCS[NextoObsTuple::NUM_BOOSTS][3] = {
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

	NextoObsBuilder::NextoObsBuilder() {
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

	void NextoObsBuilder::InitBoostLocations() {
		for (int i = 0; i < NextoObsTuple::NUM_BOOSTS; i++) {
			_boostLocations[i] = Vec(BOOST_LOCS[i][0], BOOST_LOCS[i][1], BOOST_LOCS[i][2]);
			_boostTypes[i] = BOOST_LOCS[i][2] > 72.f;
		}
	}

	void NextoObsBuilder::Reset(const GameState& state) {
		(void)state;
	}

	void NextoObsBuilder::ConvertToRelative(
		std::array<float, NextoObsTuple::FEAT>& q,
		std::vector<std::array<float, NextoObsTuple::FEAT>>& kvRows) const {

		for (auto& row : kvRows) {
			row[POS + 0] -= q[POS + 0];
			row[POS + 1] -= q[POS + 1];
			row[POS + 2] -= q[POS + 2];
		}

		float theta = std::atan2(q[FW + 0], q[FW + 1]);
		float ct = std::cos(theta);
		float st = std::sin(theta);

		for (auto& row : kvRows) {
			for (int base = POS; base < ANG_VEL + 3; base += 3) {
				float xs = row[base + 0];
				float ys = row[base + 1];
				row[base + 0] = ct * xs - st * ys;
				row[base + 1] = st * xs + ct * ys;
			}
		}
	}

	NextoObsTuple NextoObsBuilder::Build(
		const Player& player, const GameState& state, const std::array<float, 8>& prevAction) {

		const int nPlayers = RS_MIN((int)state.players.size(), MAX_PLAYERS);
		const int nEntities = nPlayers + 1 + NextoObsTuple::NUM_BOOSTS;
		const int selBall = nPlayers;

		NextoObsTuple result;
		result.nEntities = nEntities;
		result.kv.assign(nEntities * NextoObsTuple::FEAT, 0.f);
		result.m.assign(nEntities, 0.f);

		std::vector<std::array<float, NextoObsTuple::FEAT>> kvRows(nEntities);
		for (auto& row : kvRows)
			row.fill(0.f);

		// Ball
		{
			auto& row = kvRows[selBall];
			auto ball = InvertPhys(state.ball, player.team == Team::ORANGE);
			row[IS_BALL] = 1.f;
			row[POS + 0] = ball.pos.x;
			row[POS + 1] = ball.pos.y;
			row[POS + 2] = ball.pos.z;
			row[LIN_VEL + 0] = ball.vel.x;
			row[LIN_VEL + 1] = ball.vel.y;
			row[LIN_VEL + 2] = ball.vel.z;
			row[ANG_VEL + 0] = ball.angVel.x;
			row[ANG_VEL + 1] = ball.angVel.y;
			row[ANG_VEL + 2] = ball.angVel.z;
		}

		// Boost pads
		const auto& pads = state.GetBoostPads(player.team == Team::ORANGE);
		for (int i = 0; i < NextoObsTuple::NUM_BOOSTS; i++) {
			auto& row = kvRows[nPlayers + 1 + i];
			row[IS_BOOST] = 1.f;
			row[POS + 0] = _boostLocations[i].x;
			row[POS + 1] = _boostLocations[i].y;
			row[POS + 2] = _boostLocations[i].z;
			row[BOOST] = 0.12f + 0.88f * (_boostTypes[i] ? 1.f : 0.f);
			row[DEMO] = pads[i] ? 0.f : 1.f;
		}

		// Players
		for (int i = 0; i < nPlayers; i++) {
			const auto& p = state.players[i];
			auto& row = kvRows[i];
			const bool inv = p.team == Team::ORANGE;
			auto phys = InvertPhys(p, inv);

			if (p.team == player.team)
				row[IS_MATE] = 1.f;
			else
				row[IS_OPP] = 1.f;

			if (p.carId == player.carId)
				row[IS_SELF] = 1.f;

			row[POS + 0] = phys.pos.x;
			row[POS + 1] = phys.pos.y;
			row[POS + 2] = phys.pos.z;
			row[LIN_VEL + 0] = phys.vel.x;
			row[LIN_VEL + 1] = phys.vel.y;
			row[LIN_VEL + 2] = phys.vel.z;
			row[FW + 0] = phys.rotMat.forward.x;
			row[FW + 1] = phys.rotMat.forward.y;
			row[FW + 2] = phys.rotMat.forward.z;
			row[UP + 0] = phys.rotMat.up.x;
			row[UP + 1] = phys.rotMat.up.y;
			row[UP + 2] = phys.rotMat.up.z;
			row[ANG_VEL + 0] = phys.angVel.x;
			row[ANG_VEL + 1] = phys.angVel.y;
			row[ANG_VEL + 2] = phys.angVel.z;
			row[BOOST] = p.boost / 100.f;
			row[DEMO] = p.isDemoed ? 1.f : 0.f;
			row[ON_GROUND] = p.isOnGround ? 1.f : 0.f;
			row[HAS_FLIP] = p.HasFlipOrJump() ? 1.f : 0.f;
		}

		if (player.team == Team::ORANGE) {
			for (auto& row : kvRows) {
				std::swap(row[IS_MATE], row[IS_OPP]);
				for (int f = 0; f < NextoObsTuple::FEAT; f++)
					row[f] *= _invert[f];
			}
		}

		for (auto& row : kvRows) {
			for (int f = 0; f < NextoObsTuple::FEAT; f++)
				row[f] /= _norm[f];
		}

		int selfIdx = player.index;
		if (selfIdx < 0 || selfIdx >= nPlayers) {
			for (int i = 0; i < nPlayers; i++) {
				if (state.players[i].carId == player.carId) {
					selfIdx = i;
					break;
				}
			}
		}
		if (selfIdx < 0)
			selfIdx = 0;

		std::array<float, NextoObsTuple::FEAT> qArr = kvRows[selfIdx];
		ConvertToRelative(qArr, kvRows);

		result.q.assign(NextoObsTuple::Q_SIZE, 0.f);
		for (int f = 0; f < NextoObsTuple::FEAT; f++)
			result.q[f] = qArr[f];
		for (int i = 0; i < 8; i++)
			result.q[ACTIONS + i] = prevAction[i];

		for (int r = 0; r < nEntities; r++) {
			for (int f = 0; f < NextoObsTuple::FEAT; f++)
				result.kv[r * NextoObsTuple::FEAT + f] = kvRows[r][f];
		}

		return result;
	}

}
