#pragma once

#include "../Framework.h"
#include "../Gamestates/GameState.h"
#include <array>
#include <vector>

namespace RLGC {

	// Nexto attention observation (q, kv, mask) for TorchScript nexto-model.pt.
	// Ported from RLBotPack Nexto/nexto_obs.py (NextoObsBuilder).
	struct NextoObsTuple {
		static constexpr int FEAT = 24;
		static constexpr int Q_SIZE = 32;
		static constexpr int NUM_BOOSTS = 34;

		std::vector<float> q;   // [1, 1, 32] flattened
		std::vector<float> kv;  // [1, nEntities, 24] flattened
		std::vector<float> m;   // [1, nEntities] flattened
		int nEntities = 0;
	};

	class NextoObsBuilder {
	public:
		static constexpr int IS_SELF = 0;
		static constexpr int IS_MATE = 1;
		static constexpr int IS_OPP = 2;
		static constexpr int IS_BALL = 3;
		static constexpr int IS_BOOST = 4;
		static constexpr int POS = 5;
		static constexpr int LIN_VEL = 8;
		static constexpr int FW = 11;
		static constexpr int UP = 14;
		static constexpr int ANG_VEL = 17;
		static constexpr int BOOST = 20;
		static constexpr int DEMO = 21;
		static constexpr int ON_GROUND = 22;
		static constexpr int HAS_FLIP = 23;
		static constexpr int ACTIONS = 24;

		NextoObsBuilder();

		void Reset(const GameState& state);

		NextoObsTuple Build(const Player& player, const GameState& state, const std::array<float, 8>& prevAction);

	private:
		static constexpr int MAX_PLAYERS = 6;

		std::array<float, NextoObsTuple::FEAT> _norm{};
		std::array<float, NextoObsTuple::FEAT> _invert{};
		Vec _boostLocations[NextoObsTuple::NUM_BOOSTS]{};
		bool _boostTypes[NextoObsTuple::NUM_BOOSTS]{};

		void InitBoostLocations();
		void ConvertToRelative(std::array<float, NextoObsTuple::FEAT>& q, std::vector<std::array<float, NextoObsTuple::FEAT>>& kv) const;
	};

}
