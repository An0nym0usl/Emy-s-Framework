#include "InferUnit.h"

#include <GigaLearnCPP/Util/Models.h>
#include <GigaLearnCPP/PPO/PPOLearner.h>

GGL::InferUnit::InferUnit(
	RLGC::ObsBuilder* obsBuilder, int obsSize, RLGC::ActionParser* actionParser,
	PartialModelConfig sharedHeadConfig, PartialModelConfig policyConfig, 
	std::filesystem::path modelsFolder, bool useGPU,
	PolicyType policyType, float varMin, float varMax,
	int continuousActionSize,
	RLGC::ContinuousActionParser* continuousActionParser,
	AttentionModelConfig* attentionHeadConfig) : 
	obsBuilder(obsBuilder), obsSize(obsSize), actionParser(actionParser), useGPU(useGPU),
	policyType(policyType), varMin(varMin), varMax(varMax), 
	continuousActionSize(continuousActionSize),
	continuousActionParser(continuousActionParser) {

	this->models = new ModelSet();

	// For continuous, set up the policy output size
	int numOutputs;
	if (policyType == PolicyType::CONTINUOUS) {
		numOutputs = PPOLearner::GetContinuousPolicyOutputSize(continuousActionSize);
	} else {
		numOutputs = actionParser->GetActionAmount();
	}

	try {
		auto device = useGPU ? torch::kCUDA : torch::kCPU;
		if (attentionHeadConfig) {
			PPOLearner::MakeModelsWithAttention(
				false, obsSize, numOutputs, *attentionHeadConfig, policyConfig, {},
				device, *this->models, policyType, continuousActionSize
			);
		} else {
			PPOLearner::MakeModels(
				false, obsSize, numOutputs,
				sharedHeadConfig, policyConfig, {},
				device, *this->models,
				policyType, continuousActionSize
			);
		}
	} catch (std::exception& e) {
		RG_ERR_CLOSE("InferUnit: Exception when trying to construct models: " << e.what());
	}

	try {
		this->models->Load(modelsFolder, false, false);
	} catch (std::exception& e) {
		RG_ERR_CLOSE("InferUnit: Exception when trying to load models: " << e.what());
	}
}

GGL::InferUnit::~InferUnit() {
	if (models) {
		models->Free();
		delete models;
		models = nullptr;
	}
}

RLGC::Action GGL::InferUnit::InferAction(const RLGC::Player& player, const RLGC::GameState& state, bool deterministic, float temperature) {
	return BatchInferActions({ player }, { state }, deterministic, temperature)[0];
}

std::vector<RLGC::Action> GGL::InferUnit::BatchInferActions(const std::vector<RLGC::Player>& players, const std::vector<RLGC::GameState>& states, bool deterministic, float temperature) {
	RG_ASSERT(players.size() > 0 && states.size() > 0);
	RG_ASSERT(players.size() == states.size());

	int batchSize = players.size();
	std::vector<float> allObs;
	std::vector<uint8_t> allActionMasks;
	for (int i = 0; i < batchSize; i++) {
		FList curObs = obsBuilder->BuildObs(players[i], states[i]);
		if (curObs.size() != obsSize) {
			RG_ERR_CLOSE(
				"InferUnit: Obs builder produced an obs that differs from the provided size (expected: " << obsSize << ", got: " << curObs.size() << ")\n" <<
				"Make sure you provided the correct obs size to the InferUnit constructor.\n" <<
				"Also, make sure there aren't an incorrect number of players (there are " << states[i].players.size() << " in this state)"
			);
		}
		allObs += curObs;

		if (policyType != PolicyType::CONTINUOUS) {
			allActionMasks += actionParser->GetActionMask(players[i], states[i]);
		}
	}
	
	std::vector<RLGC::Action> results = {};

	try {
		RG_NO_GRAD;

		auto device = useGPU ? torch::kCUDA : torch::kCPU;

		auto tObs = torch::tensor(allObs).reshape({(int64_t)players.size(), obsSize});
		tObs = tObs.to(device);

		if (policyType == PolicyType::CONTINUOUS) {
			// === Continuous path ===
			torch::Tensor tActions, tLogProbs;
			PPOLearner::SampleContinuousActions(*models, tObs, deterministic, false, varMin, varMax, &tActions, &tLogProbs);

			auto actionValues = TENSOR_TO_VEC<float>(tActions.cpu().flatten());
			
			for (int i = 0; i < batchSize; i++) {
				const float* actionPtr = &actionValues[i * continuousActionSize];
				if (continuousActionParser) {
					results.push_back(continuousActionParser->ParseContinuousAction(actionPtr, continuousActionSize, players[i], states[i]));
				} else {
					// Default mapping
					RLGC::Action action;
					action.throttle = actionPtr[0];
					action.steer    = actionPtr[1];
					action.pitch    = actionPtr[2];
					action.yaw      = actionPtr[3];
					action.roll     = actionPtr[4];
					action.jump      = actionPtr[5] > 0.0f ? 1.0f : 0.0f;
					action.boost     = actionPtr[6] > 0.0f ? 1.0f : 0.0f;
					action.handbrake = actionPtr[7] > 0.0f ? 1.0f : 0.0f;
					results.push_back(action);
				}
			}
		} else {
			// === Discrete path (unchanged) ===
			auto tActionMasks = torch::tensor(allActionMasks).reshape({(int64_t)players.size(), this->actionParser->GetActionAmount()});
			tActionMasks = tActionMasks.to(device);
			torch::Tensor tActions, tLogProbs;

			PPOLearner::InferActionsFromModels(*models, tObs, tActionMasks, deterministic, temperature, false, &tActions, &tLogProbs);

			auto actionIndices = TENSOR_TO_VEC<int>(tActions);
			
			for (int i = 0; i < batchSize; i++) 
				results.push_back(actionParser->ParseAction(actionIndices[i], players[i], states[i]));
		}

	} catch (std::exception& e) {
		RG_ERR_CLOSE("InferUnit: Exception when inferring model: " << e.what());
	}

	return results;
}
