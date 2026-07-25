#pragma once

#include <GigaLearnCPP/Util/InferUnit.h>

namespace GGL {

	// RAII convenience wrapper around InferUnit for deployment and standalone inference.
	// Unlike a bare InferUnit, an InferenceSession OWNS the observation/action builders you pass in
	// and frees everything (builders + models) on destruction, so there is nothing to leak.
	struct RG_IMEXPORT InferenceSession {
		InferUnit* inferUnit = nullptr;
		RLGC::ObsBuilder* obsBuilder = nullptr;
		RLGC::ActionParser* actionParser = nullptr;
		RLGC::ContinuousActionParser* continuousActionParser = nullptr;

		// Loads a checkpoint and prepares it for inference. Takes ownership of the provided builders.
		// Make sure obsSize / policyType / attentionHeadConfig match the checkpoint you are loading.
		InferenceSession(
			RLGC::ObsBuilder* obsBuilder, int obsSize, RLGC::ActionParser* actionParser,
			PartialModelConfig sharedHeadConfig, PartialModelConfig policyConfig,
			std::filesystem::path checkpointFolder, bool useGPU,
			PolicyType policyType = PolicyType::DISCRETE,
			float varMin = 0.1f, float varMax = 1.0f, int continuousActionSize = 8,
			RLGC::ContinuousActionParser* continuousActionParser = nullptr,
			AttentionModelConfig* attentionHeadConfig = nullptr);

		RLGC::Action InferAction(
			const RLGC::Player& player, const RLGC::GameState& state,
			bool deterministic = true, float temperature = 1.f);

		std::vector<RLGC::Action> BatchInferActions(
			const std::vector<RLGC::Player>& players, const std::vector<RLGC::GameState>& states,
			bool deterministic = true, float temperature = 1.f);

		~InferenceSession();

		InferenceSession(const InferenceSession&) = delete;
		InferenceSession& operator=(const InferenceSession&) = delete;
	};
}
