#pragma once

#include <GigaLearnCPP/LearnerConfig.h>
#include <GigaLearnCPP/Util/Models.h>
#include <GigaLearnCPP/PPO/PPOLearnerConfig.h>
#include <RLGymCPP/EnvSet/EnvSet.h>
#include <RLGymCPP/ObsBuilders/NextoObsBuilder.h>
#include <RLGymCPP/ObsBuilders/NectoObsBuilder.h>
#include <RLGymCPP/ObsBuilders/ObsBuilder.h>
#include <RLGymCPP/ActionParsers/ActionParser.h>
#include <torch/script.h>
#include <array>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace GGL {

	class PPOLearner;

	enum class OpponentKind {
		GIGALEARN,
		NEXTO_JIT,
		NECTO_JIT,
	};

	struct OpponentEntry {
		std::string name;
		OpponentKind kind = OpponentKind::NEXTO_JIT;
		float weight = 1.f;
		float beta = 1.f;
		// Scales OpponentPoolConfig::beatBonus / concedePenalty (ContinuousV2 parity).
		float beatBonusScale = 1.f;
		float concedePenaltyScale = 1.f;

		// Arch-aware GigaLearn external (Requiem / NextoTled): own obs + discrete→continuous.
		// obsSize==0 → reuse student batch obs (same-arch clone path).
		int obsSize = 0;
		PolicyType policyType = PolicyType::DISCRETE;
		RLGC::ObsBuilder* obsBuilder = nullptr;
		RLGC::ActionParser* actionParser = nullptr;

		ModelSet models = {};
		std::unique_ptr<torch::jit::script::Module> jitModule;

		bool UsesOwnObs() const { return obsSize > 0 && obsBuilder != nullptr && actionParser != nullptr; }
	};

	// Loads external sparring bots and runs inference on the opponent team.
	class OpponentPool {
	public:
		OpponentPool() = default;
		std::vector<OpponentEntry> entries;

		void Load(
			const OpponentPoolConfig& config,
			PPOLearner* ppo,
			at::Device device
		);

		bool Empty() const { return entries.empty(); }

		float TotalWeight() const;

		// Weighted random pick. Returns nullptr if empty.
		OpponentEntry* Pick();

		// Infer 8-dim continuous controls for opponent-team players.
		void InferOpponentActions(
			OpponentEntry& opponent,
			PPOLearner* ppo,
			const std::vector<RLGC::GameState>& gameStates,
			const std::vector<int>& arenaPlayerStartIdx,
			const std::vector<int>& opponentPlayerIndices,
			torch::Tensor tStates,
			torch::Tensor tActionMasks,
			int contActionDim,
			torch::Tensor* outActions
		);

		void InferOpponentActions(
			OpponentEntry& opponent,
			PPOLearner* ppo,
			RLGC::EnvSet* envSet,
			const std::vector<int>& opponentPlayerIndices,
			torch::Tensor tStates,
			torch::Tensor tActionMasks,
			int contActionDim,
			torch::Tensor* outActions
		) {
			InferOpponentActions(
				opponent, ppo, envSet->state.gameStates, envSet->state.arenaPlayerStartIdx,
				opponentPlayerIndices, tStates, tActionMasks, contActionDim, outActions);
		}

		void ClearPrevActions();
		void SetPrevAction(int globalPlayerIdx, const std::array<float, 8>& action);

		~OpponentPool();

		OpponentPool(const OpponentPool&) = delete;
		OpponentPool& operator=(const OpponentPool&) = delete;

	private:
		at::Device _device = at::kCPU;
		std::vector<std::array<float, 8>> _nextoLookup;
		std::unordered_map<int, std::array<float, 8>> _prevActions;
		std::unordered_map<int, RLGC::NectoObsBuilder> _nectoObsBuilders;

		void LoadManifest(const std::filesystem::path& manifestPath, const OpponentPoolConfig& config, PPOLearner* ppo);
		void TryAutoDiscover(const OpponentPoolConfig& config, PPOLearner* ppo);
		void FreeEntry(OpponentEntry& e);

		static void ResolvePlayerArena(
			const std::vector<RLGC::GameState>& gameStates,
			const std::vector<int>& arenaPlayerStartIdx,
			int globalIdx,
			int& arenaIdx,
			int& localIdx);

		static void ResolvePlayerArena(RLGC::EnvSet* envSet, int globalIdx, int& arenaIdx, int& localIdx) {
			ResolvePlayerArena(envSet->state.gameStates, envSet->state.arenaPlayerStartIdx, globalIdx, arenaIdx, localIdx);
		}

		torch::Tensor InferNextoBatch(
			OpponentEntry& opponent,
			const std::vector<RLGC::NextoObsTuple>& obsBatch,
			int batchSize
		) const;

		torch::Tensor InferNectoBatch(
			OpponentEntry& opponent,
			const std::vector<RLGC::NectoObsTuple>& obsBatch,
			int batchSize
		) const;

		void InferGigalearnOwnObs(
			OpponentEntry& opponent,
			PPOLearner* ppo,
			const std::vector<RLGC::GameState>& gameStates,
			const std::vector<int>& arenaPlayerStartIdx,
			const std::vector<int>& opponentPlayerIndices,
			int contActionDim,
			torch::Tensor* outActions
		) const;

		std::array<float, 8> LookupNextoAction(int actionIdx) const;
		static std::array<float, 8> ParseNectoControls(const int64_t* actions5);
		static std::array<float, 8> ActionToContinuous(const RLGC::Action& a);
		// Map 8-dim continuous controls → nearest DefaultAction discrete index (discrete students).
		static int ContinuousToDefaultActionIndex(const std::array<float, 8>& controls);
	};

}
