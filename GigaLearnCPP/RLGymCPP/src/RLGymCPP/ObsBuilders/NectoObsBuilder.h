#pragma once

#include "../Framework.h"
#include "../Gamestates/GameState.h"
#include <array>
#include <unordered_map>
#include <vector>

namespace RLGC {

	// Necto attention observation for TorchScript necto-model.pt (rlbot-support/Necto/necto_obs.py).
	struct NectoObsTuple {
		static constexpr int FEAT = 24;
		static constexpr int Q_SIZE = 32; // 24 entity + 8 previous action
		static constexpr int NUM_BOOSTS = 34;

		std::vector<float> q;   // [1, 1, 32]
		std::vector<float> kv;  // [1, nEntities, 24]
		std::vector<float> m;   // [1, nEntities]
		int nEntities = 0;
	};

	class NectoObsBuilder {
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

		NectoObsBuilder();

		void Reset(const GameState& state);

		NectoObsTuple Build(const Player& player, const GameState& state, const std::array<float, 8>& prevAction);

	private:
		static constexpr int MAX_PLAYERS = 6;
		static constexpr int TICK_SKIP = 8;

		std::array<float, NectoObsTuple::FEAT> _norm{};
		std::array<float, NectoObsTuple::FEAT> _invert{};
		Vec _boostLocations[NectoObsTuple::NUM_BOOSTS]{};
		bool _boostTypes[NectoObsTuple::NUM_BOOSTS]{};

		std::unordered_map<uint32_t, float> _demoTimers;
		std::array<float, NectoObsTuple::NUM_BOOSTS> _boostTimers{};
		bool _timersInit = false;

		void InitBoostLocations();
		void UpdateSharedQkv(const GameState& state, std::vector<std::array<float, NectoObsTuple::FEAT>>& rows, int& nPlayers);
	};

}
