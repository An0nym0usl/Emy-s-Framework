#pragma once

#include "ModelConfig.h"
#include <GigaLearnCPP/PPO/PPOLearnerConfig.h>

namespace GGL {
	struct RG_IMEXPORT InferUnit {
		int obsSize;
		RLGC::ObsBuilder* obsBuilder;
		RLGC::ActionParser* actionParser;
		RLGC::ContinuousActionParser* continuousActionParser;
		struct ModelSet* models;
		bool useGPU;
		PolicyType policyType;
		float varMin, varMax;
		int continuousActionSize;

		// NOTE: Reset() will never be called on your obs 
		InferUnit(
			RLGC::ObsBuilder* obsBuilder, int obsSize, RLGC::ActionParser* actionParser,
			PartialModelConfig sharedHeadConfig, PartialModelConfig policyConfig,
			std::filesystem::path modelsFolder, bool useGPU,
			PolicyType policyType = PolicyType::DISCRETE,
			float varMin = 0.1f, float varMax = 1.0f,
			int continuousActionSize = 8,
			RLGC::ContinuousActionParser* continuousActionParser = nullptr,
			AttentionModelConfig* attentionHeadConfig = nullptr);


		RLGC::Action InferAction(const RLGC::Player& player, const RLGC::GameState& state, bool deterministic, float temperature = 1);
		std::vector<RLGC::Action> BatchInferActions(const std::vector<RLGC::Player>& players, const std::vector<RLGC::GameState>& states, bool deterministic, float temperature = 1);

		// Frees the owned model set. Note: obsBuilder/actionParser/continuousActionParser are
		// NOT owned by InferUnit (they are provided by the caller) and are intentionally not freed.
		~InferUnit();

		// Non-copyable: owns raw model pointers, copying would double-free.
		InferUnit(const InferUnit&) = delete;
		InferUnit& operator=(const InferUnit&) = delete;
	};
}