#include "InferenceSession.h"

GGL::InferenceSession::InferenceSession(
	RLGC::ObsBuilder* obsBuilder, int obsSize, RLGC::ActionParser* actionParser,
	PartialModelConfig sharedHeadConfig, PartialModelConfig policyConfig,
	std::filesystem::path checkpointFolder, bool useGPU,
	PolicyType policyType, float varMin, float varMax, int continuousActionSize,
	RLGC::ContinuousActionParser* continuousActionParser,
	AttentionModelConfig* attentionHeadConfig)
	: obsBuilder(obsBuilder), actionParser(actionParser), continuousActionParser(continuousActionParser) {

	inferUnit = new InferUnit(
		obsBuilder, obsSize, actionParser,
		sharedHeadConfig, policyConfig, checkpointFolder, useGPU,
		policyType, varMin, varMax, continuousActionSize,
		continuousActionParser, attentionHeadConfig
	);
}

RLGC::Action GGL::InferenceSession::InferAction(
	const RLGC::Player& player, const RLGC::GameState& state, bool deterministic, float temperature) {
	return inferUnit->InferAction(player, state, deterministic, temperature);
}

std::vector<RLGC::Action> GGL::InferenceSession::BatchInferActions(
	const std::vector<RLGC::Player>& players, const std::vector<RLGC::GameState>& states,
	bool deterministic, float temperature) {
	return inferUnit->BatchInferActions(players, states, deterministic, temperature);
}

GGL::InferenceSession::~InferenceSession() {
	delete inferUnit;             // frees the loaded models (InferUnit destructor)
	delete obsBuilder;
	delete actionParser;
	delete continuousActionParser;
}
