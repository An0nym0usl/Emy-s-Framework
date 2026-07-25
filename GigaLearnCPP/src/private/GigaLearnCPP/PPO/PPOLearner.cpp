#include "PPOLearner.h"

#include <torch/nn/utils/convert_parameters.h>
#include <torch/nn/utils/clip_grad.h>
#include <torch/csrc/api/include/torch/serialize.h>
#include <public/GigaLearnCPP/Util/AvgTracker.h>
#include <algorithm>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace torch;

namespace {
	constexpr int HYBRID_BUTTON_ACTION_DIMS = 3;
	constexpr float ACTION_MIN_PROB = 1e-11f;
	constexpr float ACTION_DISABLED_LOGIT = -1e10f;

	int GetHybridAnalogActionDims(int totalActionDims) {
		RG_ASSERT(totalActionDims > HYBRID_BUTTON_ACTION_DIMS);
		return totalActionDims - HYBRID_BUTTON_ACTION_DIMS;
	}

	int GetHybridPolicyOutputDims(int totalActionDims) {
		int analogActionDims = GetHybridAnalogActionDims(totalActionDims);
		return analogActionDims * 2 + HYBRID_BUTTON_ACTION_DIMS;
	}

	torch::Tensor BernoulliLogProbFromLogits(torch::Tensor buttonActions01, torch::Tensor buttonLogits) {
		auto buttonProbs = torch::sigmoid(buttonLogits).clamp(1e-6f, 1.0f - 1e-6f);
		return buttonActions01 * buttonProbs.log() + (1.0f - buttonActions01) * (1.0f - buttonProbs).log();
	}

	torch::Tensor BernoulliEntropyFromLogits(torch::Tensor buttonLogits) {
		auto buttonProbs = torch::sigmoid(buttonLogits).clamp(1e-6f, 1.0f - 1e-6f);
		return -(buttonProbs * buttonProbs.log() + (1.0f - buttonProbs) * (1.0f - buttonProbs).log());
	}

	torch::Tensor BernoulliKLDivergenceFromLogits(torch::Tensor firstLogits, torch::Tensor secondLogits) {
		auto firstProbs = torch::sigmoid(firstLogits).clamp(1e-6f, 1.0f - 1e-6f);
		auto secondProbs = torch::sigmoid(secondLogits).clamp(1e-6f, 1.0f - 1e-6f);
		return
			firstProbs * (firstProbs.log() - secondProbs.log()) +
			(1.0f - firstProbs) * ((1.0f - firstProbs).log() - (1.0f - secondProbs).log());
	}

	torch::Tensor NormalizeMaskedDiscreteProbs(torch::Tensor probs, torch::Tensor actionMasks) {
		auto mask = actionMasks.to(torch::kBool);
		auto maskedProbs = probs * mask.to(probs.dtype());
		return maskedProbs / maskedProbs.sum(-1, true).clamp_min(ACTION_MIN_PROB);
	}
}

int GGL::PPOLearner::GetContinuousPolicyOutputSize(int continuousActionSize) {
	return GetHybridPolicyOutputDims(continuousActionSize);
}

GGL::PPOLearner::PPOLearner(int obsSize, int numActions, PPOLearnerConfig _config, Device _device) : config(_config), device(_device) {

	if (config.miniBatchSize == 0)
		config.miniBatchSize = config.batchSize;

	if (config.batchSize % config.miniBatchSize != 0)
		RG_ERR_CLOSE("PPOLearner: config.batchSize (" << config.batchSize << ") must be a multiple of config.miniBatchSize (" << config.miniBatchSize << ")");

	// In CONTINUOUS mode we now use a hybrid head:
	// analog controls: mean + std, binary controls: Bernoulli logits.
	int policyOutputSize = numActions;
	if (config.policyType == PolicyType::CONTINUOUS)
		policyOutputSize = GetContinuousPolicyOutputSize(config.continuousActionSize);

	if (config.useAttentionHead && config.attentionHeadConfig) {
		// Use attention-based shared head (Refinement->Think blocks)
		RG_LOG("Using AttentionModel shared head");
		MakeModelsWithAttention(true, obsSize, policyOutputSize, *config.attentionHeadConfig, config.policy, config.critic, device, models, config.policyType, config.continuousActionSize);
	} else {
		MakeModels(true, obsSize, policyOutputSize, config.sharedHead, config.policy, config.critic, device, models, config.policyType, config.continuousActionSize);
	}

	SetLearningRates(config.policyLR, config.criticLR);

	// Print param counts
	RG_LOG("Model parameter counts:");
	uint64_t total = 0;
	for (auto model : this->models) {
		uint64_t count = model->GetParamCount();
		RG_LOG("\t\"" << model->modelName << "\": " << Utils::NumToStr(count));
		total += count;
	}
	RG_LOG("\t[Total]: " << Utils::NumToStr(total));

	if (config.policyType == PolicyType::CONTINUOUS) {
		RG_LOG("Policy type: CONTINUOUS/HYBRID (action dims: " << config.continuousActionSize << ", policy outputs: " << policyOutputSize << ", var range: [" << config.varMin << ", " << config.varMax << "])");
	} else {
		RG_LOG("Policy type: DISCRETE");
	}

	if (config.useGuidingPolicy) {
		RG_LOG("Guiding policy enabled, loading from " << config.guidingPolicyPath << "...");
		if (config.useAttentionHead && config.attentionHeadConfig) {
			MakeModelsWithAttention(false, obsSize, policyOutputSize, *config.attentionHeadConfig, config.policy, config.critic, device, guidingPolicyModels, config.policyType, config.continuousActionSize);
		} else {
			MakeModels(false, obsSize, policyOutputSize, config.sharedHead, config.policy, config.critic, device, guidingPolicyModels, config.policyType, config.continuousActionSize);
		}
		guidingPolicyModels.Load(config.guidingPolicyPath, false, false);
	}
}

void GGL::PPOLearner::MakeModels(
	bool makeCritic,
	int obsSize, int numActions, 
	PartialModelConfig sharedHeadConfig, PartialModelConfig policyConfig, PartialModelConfig criticConfig,
	torch::Device device, 
	ModelSet& outModels,
	PolicyType policyType,
	int continuousActionSize) {

	ModelConfig fullPolicyConfig = policyConfig;
	fullPolicyConfig.numInputs = obsSize;
	fullPolicyConfig.numOutputs = numActions;

	ModelConfig fullCriticConfig = criticConfig;
	fullCriticConfig.numInputs = obsSize;
	fullCriticConfig.numOutputs = 1;

	if (sharedHeadConfig.IsValid()) {

		ModelConfig fullSharedHeadConfig = sharedHeadConfig;
		fullSharedHeadConfig.numInputs = obsSize;
		fullSharedHeadConfig.numOutputs = 0;

		RG_ASSERT(!sharedHeadConfig.addOutputLayer);

		fullPolicyConfig.numInputs = fullSharedHeadConfig.layerSizes.back();
		fullCriticConfig.numInputs = fullSharedHeadConfig.layerSizes.back();

		outModels.Add(new Model("shared_head", fullSharedHeadConfig, device));
	}

	outModels.Add(new Model("policy", fullPolicyConfig, device));

	if (makeCritic)
		outModels.Add(new Model("critic", fullCriticConfig, device));
}

void GGL::PPOLearner::MakeModelsWithAttention(
	bool makeCritic,
	int obsSize, int numActions,
	AttentionModelConfig attentionConfig, PartialModelConfig policyConfig, PartialModelConfig criticConfig,
	torch::Device device,
	ModelSet& outModels,
	PolicyType policyType,
	int continuousActionSize) {

	// Auto-detect numInputs if not set
	if (attentionConfig.numInputs <= 0)
		attentionConfig.numInputs = obsSize;

	// Attention shared head outputs numDims
	int headOutputDims = attentionConfig.GetOutputDims();

	// Create attention model as shared head
	outModels.Add(new AttentionModel("shared_head", attentionConfig, device));

	// Policy and critic take attention head output
	ModelConfig fullPolicyConfig = policyConfig;
	fullPolicyConfig.numInputs = headOutputDims;
	fullPolicyConfig.numOutputs = numActions;

	ModelConfig fullCriticConfig = criticConfig;
	fullCriticConfig.numInputs = headOutputDims;
	fullCriticConfig.numOutputs = 1;

	outModels.Add(new Model("policy", fullPolicyConfig, device));

	if (makeCritic)
		outModels.Add(new Model("critic", fullCriticConfig, device));
}

// =============================================
// === Discrete Policy (unchanged from original)
// =============================================

torch::Tensor GGL::PPOLearner::InferPolicyProbsFromModels(
	ModelSet& models,
	torch::Tensor obs, torch::Tensor actionMasks,
	float temperature, bool halfPrec) {

	actionMasks = actionMasks.to(torch::kBool);

	if (models["shared_head"])
		obs = models["shared_head"]->Forward(obs, halfPrec);

	auto logits = models["policy"]->Forward(obs, halfPrec) / temperature;

	auto result = torch::softmax(logits + ACTION_DISABLED_LOGIT * actionMasks.logical_not(), -1);
	return result.view({ -1, models["policy"]->config.numOutputs }).clamp(ACTION_MIN_PROB, 1);
}

void GGL::PPOLearner::InferActionsFromModels(
	ModelSet& models,
	torch::Tensor obs, torch::Tensor actionMasks, 
	bool deterministic, float temperature, bool halfPrec,
	torch::Tensor* outActions, torch::Tensor* outLogProbs) {

	auto probs = InferPolicyProbsFromModels(models, obs, actionMasks, temperature, halfPrec);

	if (deterministic) {
		auto action = probs.argmax(1);
		if (outActions)
			*outActions = action.flatten();
	} else {
		auto action = torch::multinomial(probs, 1, true);
		auto logProb = torch::log(probs).gather(-1, action);
		if (outActions)
			*outActions = action.flatten();

		if (outLogProbs)
			*outLogProbs = logProb.flatten();
	}
}

// =============================================
// === Continuous Policy (new)
// =============================================

torch::Tensor GGL::PPOLearner::GaussianLogPdf(torch::Tensor x, torch::Tensor mean, torch::Tensor stddev) {
	// Manual Gaussian log-pdf matching rlgym-ppo's custom implementation
	// log(N(x | mean, std)) = -0.5 * log(2*pi) - log(std) - 0.5 * ((x - mean) / std)^2
	auto variance = stddev * stddev;
	auto logScale = stddev.log();
	return -((x - mean).pow(2)) / (2 * variance) - logScale - 0.5f * std::log(2.0 * M_PI);
}

torch::Tensor ComputeHybridLogProb(
	torch::Tensor actions,
	torch::Tensor mean,
	torch::Tensor stddev,
	torch::Tensor buttonLogits) {

	int analogActionDims = mean.size(-1);
	int buttonActionDims = buttonLogits.size(-1);

	auto analogActions = actions.slice(-1, 0, analogActionDims);
	auto buttonActions = actions.slice(-1, analogActionDims, analogActionDims + buttonActionDims);
	auto buttonActions01 = (buttonActions > 0).to(mean.dtype());

	// Squashed Gaussian log-prob correction
	// log P(y) = log P(x) - sum(log(1 - tanh^2(x) + eps))
	// To find x, we invert y: x = atanh(clamp(y, -1+eps, 1-eps))
	float eps = 1e-6f;
	auto y_clamped = analogActions.clamp(-1.0f + eps, 1.0f - eps);
	auto x = torch::atanh(y_clamped);

	// Compute base Gaussian log-pdf on the unbounded x
	auto baseLogProb = GGL::PPOLearner::GaussianLogPdf(x, mean, stddev).sum(-1);

	// Subtract the log derivative of the tanh transform
	auto squashedCorrection = torch::log(1.0f - y_clamped.pow(2) + eps).sum(-1);
	auto analogLogProb = baseLogProb - squashedCorrection;

	auto buttonLogProb = BernoulliLogProbFromLogits(buttonActions01, buttonLogits).sum(-1);
	return analogLogProb + buttonLogProb;
}

torch::Tensor ComputeHybridActionTableLogProb(
	torch::Tensor actionTable,
	torch::Tensor mean,
	torch::Tensor stddev,
	torch::Tensor buttonLogits) {

	int analogActionDims = mean.size(-1);
	int totalActionDims = actionTable.size(-1);

	auto analogActions = actionTable.slice(-1, 0, analogActionDims);
	auto buttonActions = actionTable.slice(-1, analogActionDims, totalActionDims);
	auto buttonActions01 = (buttonActions > 0).to(mean.dtype());

	auto analogLogProb =
		GGL::PPOLearner::GaussianLogPdf(analogActions, mean.unsqueeze(1), stddev.unsqueeze(1)).sum(-1);
	auto buttonLogProb =
		BernoulliLogProbFromLogits(buttonActions01, buttonLogits.unsqueeze(1)).sum(-1);
	return analogLogProb + buttonLogProb;
}

torch::Tensor ComputeHybridKLDivergence(
	torch::Tensor firstMean,
	torch::Tensor firstStd,
	torch::Tensor firstButtonLogits,
	torch::Tensor secondMean,
	torch::Tensor secondStd,
	torch::Tensor secondButtonLogits) {

	auto analogKL =
		(secondStd / firstStd).log() +
		(firstStd.pow(2) + (firstMean - secondMean).pow(2)) / (2 * secondStd.pow(2)) -
		0.5f;
	auto buttonKL = BernoulliKLDivergenceFromLogits(firstButtonLogits, secondButtonLogits);
	return analogKL.sum(-1) + buttonKL.sum(-1);
}

void GGL::PPOLearner::InferContinuousPolicyFromModels(
	ModelSet& models, torch::Tensor obs, bool halfPrec,
	torch::Tensor& outMean, torch::Tensor& outStd, torch::Tensor& outButtonLogits,
	float varMin, float varMax) {

	if (models["shared_head"])
		obs = models["shared_head"]->Forward(obs, halfPrec);

	auto rawOutput = models["policy"]->Forward(obs, halfPrec);
	int policyOutputDims = rawOutput.size(-1);
	RG_ASSERT(policyOutputDims > HYBRID_BUTTON_ACTION_DIMS);
	RG_ASSERT((policyOutputDims - HYBRID_BUTTON_ACTION_DIMS) % 2 == 0);
	int analogActionDims = (policyOutputDims - HYBRID_BUTTON_ACTION_DIMS) / 2;

	outMean = rawOutput.slice(-1, 0, analogActionDims).clamp(-4.0f, 4.0f);
	auto rawStd = torch::tanh(rawOutput.slice(-1, analogActionDims, analogActionDims * 2));
	outButtonLogits = rawOutput.slice(-1, analogActionDims * 2, policyOutputDims);

	// Map raw_std from [-1, 1] to [varMin, varMax]
	float m = (varMax - varMin) / 2.0f;
	float b = varMin + m;
	outStd = rawStd * m + b;
}

void GGL::PPOLearner::SampleContinuousActions(
	ModelSet& models, torch::Tensor obs,
	bool deterministic, bool halfPrec,
	float varMin, float varMax,
	torch::Tensor* outActions, torch::Tensor* outLogProbs,
	float esNoiseScale) {

	torch::Tensor mean, stddev, buttonLogits;
	InferContinuousPolicyFromModels(models, obs, halfPrec, mean, stddev, buttonLogits, varMin, varMax);

	// ES exploration: inflate rollout std. PPO trains on-policy against these log-probs.
	if (!deterministic && esNoiseScale > 0.f)
		stddev = (stddev + esNoiseScale).clamp_max(varMax * 2.f);

	if (deterministic) {
		if (outActions) {
			auto buttonActions = torch::where(
				buttonLogits > 0,
				torch::ones_like(buttonLogits),
				-torch::ones_like(buttonLogits)
			);
			*outActions = torch::cat({ torch::tanh(mean), buttonActions }, -1);
		}
		if (outLogProbs)
			*outLogProbs = torch::zeros({ mean.size(0) }, mean.options());
	} else {
		auto preSquash = mean + torch::randn_like(mean) * stddev;
		auto analogActions = torch::tanh(preSquash);
		
		auto buttonProbs = torch::sigmoid(buttonLogits);
		auto buttonSamples = (torch::rand_like(buttonProbs) < buttonProbs).to(mean.dtype());
		auto buttonActions = buttonSamples * 2.0f - 1.0f;
		auto action = torch::cat({ analogActions, buttonActions }, -1);

		if (outActions)
			*outActions = action;

		if (outLogProbs)
			*outLogProbs = ComputeHybridLogProb(action, mean, stddev, buttonLogits);
	}
}

torch::Tensor GGL::PPOLearner::ComputeContinuousEntropy(torch::Tensor mean, torch::Tensor stddev, torch::Tensor buttonLogits, float varMax) {
	// Per-dimension entropy of Normal: 0.5 + 0.5*log(2π) + log(σ)
	auto analogEntropyPerDim = 0.5f + 0.5f * std::log(2.0 * M_PI) + stddev.log();

	// Squash correction: the actual policy output is tanh-squashed, so true entropy is
	// H(Y) = H(X) + E[log|dy/dx|] = H(Normal) + E[log(1 - tanh²(X))]
	// We approximate using the deterministic mean: log(1 - tanh²(mean) + eps)
	auto squashCorrection = torch::log(1.0f - mean.tanh().pow(2) + 1e-6f);
	analogEntropyPerDim = analogEntropyPerDim + squashCorrection;

	// Average entropy across all action dimensions (analog + button)
	auto totalEntropy = torch::cat({
		analogEntropyPerDim, 
		BernoulliEntropyFromLogits(buttonLogits)
	}, -1).mean(-1);

	return totalEntropy.mean();
}

// =============================================
// === Unified Inference
// =============================================

void GGL::PPOLearner::InferActions(torch::Tensor obs, torch::Tensor actionMasks, torch::Tensor* outActions, torch::Tensor* outLogProbs, ModelSet* models) {
	auto& mdls = models ? *models : this->models;

	if (config.policyType == PolicyType::CONTINUOUS) {
		SampleContinuousActions(mdls, obs, config.deterministic, config.useHalfPrecision, config.varMin, config.varMax, outActions, outLogProbs, config.esNoiseScale);
	} else {
		InferActionsFromModels(mdls, obs, actionMasks, config.deterministic, config.policyTemperature, config.useHalfPrecision, outActions, outLogProbs);
	}
}

torch::Tensor GGL::PPOLearner::InferCritic(torch::Tensor obs) {
	return InferCritic(obs, -1);
}

torch::Tensor GGL::PPOLearner::InferCritic(torch::Tensor obs, int halfPrecOverride) {
	const bool useHalf = halfPrecOverride < 0 ? config.useHalfPrecision : (halfPrecOverride != 0);

	if (models["shared_head"])
		obs = models["shared_head"]->Forward(obs, useHalf);

	return models["critic"]->Forward(obs, useHalf).flatten();
}

// =============================================
// === Entropy (discrete - unchanged)
// =============================================

torch::Tensor ComputeEntropy(torch::Tensor probs, torch::Tensor actionMasks, bool maskEntropy) {
	// Compute log probs and entropy
	auto entropy = -(probs.log() * probs).sum(-1);

	if (maskEntropy) {
		// Account for action masking in entropy
		// We will effectively narrow the entropy to the scope of the valid actions
		// This way states with more masked actions don't just have inherently lower entropy
		entropy /= actionMasks.to(torch::kFloat32).sum(-1).log();
	} else {
		entropy /= logf(actionMasks.size(-1));
	}

	return entropy.mean();
}

// =============================================
// === Learn
// =============================================

namespace {
	torch::Tensor NormalizeAdvantages(torch::Tensor adv, int mode, float eps) {
		if (mode <= 0 || !adv.defined())
			return adv;
		if (mode == 2)
			return adv - adv.mean();
		if (mode == 3) {
			auto scale = adv.abs().max().clamp_min(eps);
			return adv / scale;
		}
		// mode == 1 (default): batch mean/std
		return (adv - adv.mean()) / (adv.std() + eps);
	}

	torch::Tensor DiscreteGuidingKL(torch::Tensor teacherProbs, torch::Tensor studentProbs) {
		// KL(teacher ‖ student) — BC-style distill (matches TransferLearn KL direction).
		auto t = teacherProbs.clamp_min(ACTION_MIN_PROB);
		auto s = studentProbs.clamp_min(ACTION_MIN_PROB);
		return (t * (t.log() - s.log())).sum(-1).mean();
	}
}

void GGL::PPOLearner::Learn(ExperienceBuffer& experience, Report& report, bool isFirstIteration) {
	bool isContinuous = (config.policyType == PolicyType::CONTINUOUS);
	auto mseLoss = torch::nn::MSELoss();
	const bool skipMetrics = config.skipPPOMetrics;
	const int advNormMode = config.advantageNormMode;
	const float advNormEps = (std::max)(config.advantageNormEps, 1e-12f);
	const float logClamp = (config.ratioLogClamp > 0.f) ? config.ratioLogClamp : 20.f;
	const int gradAccum = (std::max)(1, config.gradAccumulationSteps);
	const bool useGuiding = config.useGuidingPolicy && !guidingPolicyModels.map.empty();

	MutAvgTracker
		avgEntropy,
		avgDivergence,
		avgPolicyLoss,
		avgRelEntropyLoss,
		avgCriticLoss,
		avgGuidingLoss,
		avgRatio,
		avgClip;

	// Continuous-specific metrics
	MutAvgTracker avgMeanStd;

	torch::Tensor policyBefore, criticBefore, sharedBefore;
	if (!skipMetrics) {
		policyBefore = models["policy"]->CopyParams();
		criticBefore = models["critic"]->CopyParams();
		if (models["shared_head"])
			sharedBefore = models["shared_head"]->CopyParams();
	}

	bool trainPolicy = config.policyLR != 0;
	bool trainCritic = config.criticLR != 0;
	bool trainSharedHead = models["shared_head"] && (trainPolicy || trainCritic);

	const float savedPolicyLR = config.policyLR;
	const float savedCriticLR = config.criticLR;
	bool klEarlyStop = false;
	int accumCounter = 0;
	bool haveEntropySample = false;

	for (int epoch = 0; epoch < config.epochs && !klEarlyStop; epoch++) {

		// Get randomly-ordered timesteps for PPO
		auto batches = experience.GetAllBatchesShuffled(
			config.batchSize, config.overbatching, config.prioritySampling, config.skipExperienceShuffle);

		for (auto& batch : batches) {
			if (klEarlyStop)
				break;

			auto batchActs = batch.actions;
			auto batchOldProbs = batch.logProbs;
			auto batchObs = batch.states;
			auto batchActionMasks = batch.actionMasks;
			auto batchTargetValues = batch.targetValues;
			// Move to device only when needed. Never force copy=true — device-XP banks
			// already live on CUDA; a forced D2D of the full buffer was a Cons regression.
			auto ensureDev = [&](torch::Tensor t) -> torch::Tensor {
				return t.device() == device ? t : t.to(device, /*non_blocking=*/true);
			};
			if (device.is_cuda()) {
				batchActs = ensureDev(batchActs);
				batchOldProbs = ensureDev(batchOldProbs);
				batchObs = ensureDev(batchObs);
				batchActionMasks = ensureDev(batchActionMasks);
				batchTargetValues = ensureDev(batchTargetValues);
			}
			auto batchAdvantages = ensureDev(batch.advantages);
			torch::Tensor normalizedAdvantages = NormalizeAdvantages(batchAdvantages, advNormMode, advNormEps);

			// Actual rows in this batch (may exceed batchSize when overbatching).
			const int64_t actualBatchSize = batchObs.size(0);
			if (actualBatchSize <= 0)
				continue;

			torch::Tensor batchApproxKl; // filled when targetKl or metrics need it

			auto fnRunMinibatch = [&](int start, int stop) {

				// Scale to configured batchSize so overbatch / last partial still match Adam step size.
				float batchSizeRatio = (stop - start) / (float)config.batchSize;
				if (gradAccum > 1)
					batchSizeRatio /= (float)gradAccum;

				// bf16 autocast: forward + loss run in mixed precision (CUDA only), backward stays fp32.
				// No GradScaler is required because bf16 retains fp32 dynamic range.
				const bool useAutocast = config.useBF16Autocast && device.is_cuda();
				if (useAutocast) RG_AUTOCAST_ON();

				const bool alreadyOnDevice = batchObs.device() == device;
				auto acts = alreadyOnDevice ? batchActs.slice(0, start, stop) : batchActs.slice(0, start, stop).to(device, true, true);
				auto obs = alreadyOnDevice ? batchObs.slice(0, start, stop) : batchObs.slice(0, start, stop).to(device, true, true);
				auto actionMasks = alreadyOnDevice ? batchActionMasks.slice(0, start, stop) : batchActionMasks.slice(0, start, stop).to(device, true, true);
				
				auto advantages = normalizedAdvantages.slice(0, start, stop);
				auto oldProbs = alreadyOnDevice ? batchOldProbs.slice(0, start, stop) : batchOldProbs.slice(0, start, stop).to(device, true, true);
				auto targetValues = alreadyOnDevice ? batchTargetValues.slice(0, start, stop) : batchTargetValues.slice(0, start, stop).to(device, true, true);

				torch::Tensor logProbs, entropy, ratio, clipped, policyLoss, ppoLoss;
				torch::Tensor criticLoss;
				torch::Tensor discreteProbs; // reused by discrete guiding (avoid 2nd forward)
				torch::Tensor sharedOut; // fused shared_head for discrete policy+critic
				const bool fuseShared =
					!isContinuous && trainPolicy && trainCritic && models["shared_head"] != nullptr;

				if (fuseShared)
					sharedOut = models["shared_head"]->Forward(obs, false);

				if (trainPolicy) {

					float curEntropy = 0.f;
					torch::Tensor contMean, contStd, contBtnLogits; // Hoisted for reuse in guiding block
					if (isContinuous) {
						// === Hybrid policy path ===
						InferContinuousPolicyFromModels(models, obs, false, contMean, contStd, contBtnLogits, config.varMin, config.varMax);

						// Compute log probs of stored actions
						logProbs = ComputeHybridLogProb(acts, contMean, contStd, contBtnLogits);

						// Compute entropy
						entropy = ComputeContinuousEntropy(contMean, contStd, contBtnLogits, config.varMax);
						if (!skipMetrics || config.targetEntropy >= 0.f) {
							curEntropy = entropy.detach().item<float>();
							avgEntropy += curEntropy;
							haveEntropySample = true;
						}
						if (!skipMetrics)
							avgMeanStd += contStd.mean().detach().cpu().item<float>();
					} else {
						// === Discrete policy path ===
						if (fuseShared) {
							if (actionMasks.dtype() != torch::kBool)
								actionMasks = actionMasks.to(torch::kBool);
							auto logits = models["policy"]->Forward(sharedOut, false) / config.policyTemperature;
							discreteProbs = torch::softmax(
								logits + ACTION_DISABLED_LOGIT * actionMasks.logical_not(), -1
							).view({ -1, models["policy"]->config.numOutputs }).clamp(ACTION_MIN_PROB, 1);
						} else {
							discreteProbs = InferPolicyProbsFromModels(models, obs, actionMasks, config.policyTemperature, false);
						}
						logProbs = discreteProbs.log().gather(-1, acts.unsqueeze(-1));
						if (config.entropyScale > 1e-12f || !skipMetrics || config.targetEntropy >= 0.f)
							entropy = ComputeEntropy(discreteProbs, actionMasks, config.maskEntropy);
						else
							entropy = torch::zeros({}, discreteProbs.options());
						if (!skipMetrics || config.targetEntropy >= 0.f) {
							curEntropy = entropy.detach().item<float>();
							avgEntropy += curEntropy;
							haveEntropySample = true;
						}
					}

					logProbs = logProbs.view_as(oldProbs);

					// Numerically stable PPO ratio
					auto logRatio = (logProbs - oldProbs).clamp(-logClamp, logClamp);
					ratio = exp(logRatio);
					clipped = clamp(
						ratio, 1 - config.clipRange, 1 + config.clipRange
					);

					// Compute policy loss
					policyLoss = -min(
						ratio * advantages, clipped * advantages
					).mean();
					if (!skipMetrics) {
						avgRatio += ratio.mean().detach().cpu().item<float>();
						float curPolicyLoss = policyLoss.detach().cpu().item<float>();
						avgPolicyLoss += curPolicyLoss;
						avgRelEntropyLoss += (curEntropy * config.entropyScale) / (std::abs(curPolicyLoss) + 1e-8f);
					}

					ppoLoss = (policyLoss - entropy * config.entropyScale) * batchSizeRatio;

					if (useGuiding && config.guidingStrength > 1e-12f) {
						torch::Tensor guidingLoss;
						if (isContinuous) {
							torch::Tensor guidingMean, guidingStd, guidingButtonLogits;
							{
								RG_NO_GRAD;
								InferContinuousPolicyFromModels(
									guidingPolicyModels, obs, config.useHalfPrecision,
									guidingMean, guidingStd, guidingButtonLogits,
									config.varMin, config.varMax
								);
							}
							// KL(student ‖ teacher) for continuous (same as prior hybrid path)
							guidingLoss = ComputeHybridKLDivergence(
								contMean, contStd, contBtnLogits,
								guidingMean, guidingStd, guidingButtonLogits
							).mean();
						} else {
							torch::Tensor guidingProbs;
							{
								RG_NO_GRAD;
								guidingProbs = InferPolicyProbsFromModels(
									guidingPolicyModels, obs, actionMasks,
									config.policyTemperature, config.useHalfPrecision);
							}
							if (config.guidingUseKL)
								guidingLoss = DiscreteGuidingKL(guidingProbs, discreteProbs);
							else
								guidingLoss = (guidingProbs - discreteProbs).abs().mean();
						}
						if (config.guidingLossClip > 0.f)
							guidingLoss = guidingLoss.clamp_max(config.guidingLossClip);
						if (!skipMetrics)
							avgGuidingLoss.Add(guidingLoss.detach().cpu().item<float>());
						ppoLoss = ppoLoss + guidingLoss * config.guidingStrength * batchSizeRatio;
					}
				}

				if (trainCritic) {
					torch::Tensor vals;
					if (fuseShared)
						vals = models["critic"]->Forward(sharedOut, false).flatten();
					else
						vals = InferCritic(obs);

					// Compute value loss
					vals = vals.view_as(targetValues);
					criticLoss = mseLoss(vals, targetValues) * batchSizeRatio;
					if (!skipMetrics)
						avgCriticLoss += criticLoss.detach().cpu().item<float>();
				}

				if (trainPolicy && (!skipMetrics || config.targetKl > 0.f)) {
					RG_NO_GRAD;
					auto logRatio = (logProbs - oldProbs).clamp(-logClamp, logClamp);
					auto klTensor = (exp(logRatio) - 1) - logRatio;
					auto klMean = klTensor.mean();
					if (!batchApproxKl.defined())
						batchApproxKl = klMean.detach();
					else
						batchApproxKl = batchApproxKl + klMean.detach();
					if (!skipMetrics) {
						avgDivergence += klMean.detach().cpu().item<float>();
						auto clipFraction = mean((abs(ratio - 1) > config.clipRange).to(kFloat));
						avgClip += clipFraction.cpu().item<float>();
					}
				}

				// Disable autocast before the backward pass and clear the per-iteration weight-cast cache.
				if (useAutocast) RG_AUTOCAST_OFF();

				if (trainPolicy && trainCritic) {
					auto combinedLoss = ppoLoss + criticLoss * config.vfCoef;
					combinedLoss.backward();
				} else {
					if (trainPolicy)
						ppoLoss.backward();
					if (trainCritic)
						criticLoss.backward();
				}
			};

			const int64_t mini = device.is_cpu()
				? actualBatchSize
				: (std::max)((int64_t)1, (int64_t)config.miniBatchSize);
			for (int64_t mbs = 0; mbs < actualBatchSize; mbs += mini) {
				int64_t start = mbs;
				int64_t stop = (std::min)(start + mini, actualBatchSize);
				fnRunMinibatch((int)start, (int)stop);
			}

			accumCounter++;
			const bool shouldStep = (accumCounter % gradAccum) == 0;
			if (shouldStep) {
				// maxGradNorm <= 0 disables clipping (pure throughput mode).
				if (config.maxGradNorm > 1e-6f) {
					const float gradClip = config.maxGradNorm;
					if (trainPolicy)
						nn::utils::clip_grad_norm_(models["policy"]->parameters(), gradClip);
					if (trainCritic)
						nn::utils::clip_grad_norm_(models["critic"]->parameters(), gradClip);
					if (trainSharedHead)
						nn::utils::clip_grad_norm_(models["shared_head"]->parameters(), gradClip);
				}
				models.StepOptims();
			}

			// Adaptive KL early-stop (after a full optimizer batch).
			if (config.targetKl > 0.f && batchApproxKl.defined() && trainPolicy) {
				// Average KL across minibatches in this optimizer batch.
				const int64_t nMini = (actualBatchSize + mini - 1) / mini;
				float kl = (batchApproxKl / (float)(std::max)((int64_t)1, nMini)).item<float>();
				if (kl > config.targetKl * 1.5f) {
					klEarlyStop = true;
					if (config.adaptiveKlLrScale > 0.f && config.adaptiveKlLrScale < 1.f) {
						SetLearningRates(
							savedPolicyLR * config.adaptiveKlLrScale,
							savedCriticLR * config.adaptiveKlLrScale);
					}
					if (!skipMetrics)
						report["PPO/KL Early Stop"] = 1.f;
				}
				if (!skipMetrics)
					report["PPO/Approx KL"] = kl;
			}
		}
	}

	// Flush leftover accumulated grads if epoch ended mid-accumulation.
	if ((accumCounter % gradAccum) != 0 && accumCounter > 0) {
		if (config.maxGradNorm > 1e-6f) {
			const float gradClip = config.maxGradNorm;
			if (trainPolicy)
				nn::utils::clip_grad_norm_(models["policy"]->parameters(), gradClip);
			if (trainCritic)
				nn::utils::clip_grad_norm_(models["critic"]->parameters(), gradClip);
			if (trainSharedHead)
				nn::utils::clip_grad_norm_(models["shared_head"]->parameters(), gradClip);
		}
		models.StepOptims();
	}

	// Restore LRs if adaptive KL temporarily scaled them.
	if (config.policyLR != savedPolicyLR || config.criticLR != savedCriticLR)
		SetLearningRates(savedPolicyLR, savedCriticLR);

	// Soft target-entropy adaptation (persists in config for AutoTrainer visibility).
	if (config.targetEntropy >= 0.f && haveEntropySample) {
		float ent = avgEntropy.Get();
		float err = config.targetEntropy - ent;
		float next = config.entropyScale + config.entropyAdaptRate * err;
		config.entropyScale = RS_CLAMP(next, config.entropyScaleMin, config.entropyScaleMax);
		if (!skipMetrics)
			report["PPO/Adapted Entropy Scale"] = config.entropyScale;
	}

	if (skipMetrics)
		return;

	// Compute magnitude of updates made to the policy and value estimator
	auto policyAfter = models["policy"]->CopyParams();
	auto criticAfter = models["critic"]->CopyParams();

	float policyUpdateMagnitude = (policyBefore - policyAfter).norm().item<float>();
	float criticUpdateMagnitude = (criticBefore - criticAfter).norm().item<float>();

	// Assemble and return report
	report["Policy Entropy"] = avgEntropy.Get();
	report["Mean KL Divergence"] = avgDivergence.Get();
	report["KL Div Loss"] = avgDivergence.Get(); // Leak Display / AutoTrainer alias
	if (!isFirstIteration) {
		// These metrics give bad data on the first iteration, which will mess up graph scaling
		// So we'll just skip them for the first iteration
		report["Policy Loss"] = avgPolicyLoss.Get();
		report["Policy Relative Entropy Loss"] = avgRelEntropyLoss.Get();
		report["Critic Loss"] = avgCriticLoss.Get();

		if (useGuiding)
			report["Guiding Loss"] = avgGuidingLoss.Get();

		report["SB3 Clip Fraction"] = avgClip.Get();
		report["Policy Update Magnitude"] = policyUpdateMagnitude;
		report["Critic Update Magnitude"] = criticUpdateMagnitude;
		if (sharedBefore.defined() && models["shared_head"]) {
			auto sharedAfter = models["shared_head"]->CopyParams();
			report["Shared Head Update Magnitude"] = (sharedBefore - sharedAfter).norm().item<float>();
		}

		// Continuous-specific metrics
		if (isContinuous)
			report["Mean Policy Std"] = avgMeanStd.Get();
	}
}

void GGL::PPOLearner::TransferLearn(
	ModelSet& oldModels,
	torch::Tensor newObs, torch::Tensor oldObs,
	torch::Tensor newActionMasks, torch::Tensor oldActionMasks,
	torch::Tensor actionMaps,
	torch::Tensor teacherActionTable,
	torch::Tensor teacherActionTargets,
	Report& report,
	const TransferLearnConfig& tlConfig
) {

	bool isContinuous = (config.policyType == PolicyType::CONTINUOUS);
	bool oldIsContinuous = (tlConfig.oldPolicyType == PolicyType::CONTINUOUS);

	if (isContinuous) {
		for (auto& model : GetPolicyModels())
			model->SetOptimLR(tlConfig.lr);

		auto policyBefore = models["policy"]->CopyParams();

		if (oldIsContinuous) {
			torch::Tensor oldMean, oldStd, oldButtonLogits;
			{
				RG_NO_GRAD;
				InferContinuousPolicyFromModels(
					oldModels, oldObs, config.useHalfPrecision,
					oldMean, oldStd, oldButtonLogits,
					config.varMin, config.varMax
				);
				report["Old Policy Entropy"] = ComputeContinuousEntropy(oldMean, oldStd, oldButtonLogits, config.varMax).detach().cpu().item<float>();
			}

		for (int i = 0; i < tlConfig.epochs; i++) {
			torch::Tensor newMean, newStd, newButtonLogits;
			InferContinuousPolicyFromModels(
				models, newObs, false,
				newMean, newStd, newButtonLogits,
				config.varMin, config.varMax
			);

			// KL divergence between old and new Gaussian policies
			torch::Tensor transferLearnLoss;
			if (tlConfig.useKLDiv) {
				// KL(old || new) = log(σ_new/σ_old) + (σ_old² + (μ_old - μ_new)²)/(2σ_new²) - 0.5
				transferLearnLoss = ComputeHybridKLDivergence(
					oldMean, oldStd, oldButtonLogits,
					newMean, newStd, newButtonLogits
				).mean();
			} else {
				auto buttonProbDelta =
					(torch::sigmoid(oldButtonLogits) - torch::sigmoid(newButtonLogits)).abs().mean();
				transferLearnLoss =
					(oldMean - newMean).abs().mean() +
					(oldStd - newStd).abs().mean() +
					buttonProbDelta;
			}
			transferLearnLoss = transferLearnLoss.pow(tlConfig.lossExponent);
			transferLearnLoss *= tlConfig.lossScale;

			if (i == 0) {
				RG_NO_GRAD;
				report["Transfer Learn Loss"] = transferLearnLoss.detach().cpu().item<float>();
				report["Policy Entropy"] = ComputeContinuousEntropy(newMean, newStd, newButtonLogits, config.varMax).detach().cpu().item<float>();
				report["Mean Policy Std"] = newStd.mean().detach().cpu().item<float>();
			}

			transferLearnLoss.backward();
			models.StepOptims();
		}

		} else {
			RG_ASSERT(teacherActionTargets.defined() || teacherActionTable.defined());

			torch::Tensor oldProbs;
			{
				RG_NO_GRAD;
				oldProbs = InferPolicyProbsFromModels(oldModels, oldObs, oldActionMasks, config.policyTemperature, config.useHalfPrecision);
				report["Old Policy Entropy"] = ComputeEntropy(oldProbs, oldActionMasks, config.maskEntropy).detach().cpu().item<float>();
			}

			for (int i = 0; i < tlConfig.epochs; i++) {
				torch::Tensor newMean, newStd, newButtonLogits;
				InferContinuousPolicyFromModels(
					models, newObs, false,
					newMean, newStd, newButtonLogits,
					config.varMin, config.varMax
				);

				auto teacherTableProbs = NormalizeMaskedDiscreteProbs(oldProbs, oldActionMasks).to(newMean.dtype());
				auto teacherSafeProbs = teacherTableProbs.clamp_min(ACTION_MIN_PROB);
				auto teacherEntropy = -(teacherSafeProbs.log() * teacherTableProbs).sum(-1);
				if (config.maskEntropy) {
					teacherEntropy /= oldActionMasks.to(torch::kFloat32).sum(-1).clamp_min(2.0f).log();
				} else {
					teacherEntropy /= std::log((float)oldProbs.size(-1));
				}
				auto teacherConfidence = (1.0f - teacherEntropy).clamp(0.0f, 1.0f).to(newMean.dtype());
				auto sampleWeights = teacherConfidence.pow(2.0f);

				torch::Tensor actionTableLoss = torch::zeros({}, newMean.options());
				torch::Tensor actionTableAccuracy = torch::zeros({}, newMean.options());
				torch::Tensor analogTargets;
				torch::Tensor buttonTargetProbs;
				torch::Tensor aerialTargetRatio = torch::zeros({}, newMean.options());

				if (teacherActionTable.defined()) {
					RG_ASSERT(teacherActionTable.size(0) == newMean.size(0));
					RG_ASSERT(teacherActionTable.size(1) == oldProbs.size(1));
					RG_ASSERT(teacherActionTable.size(2) == config.continuousActionSize);

					auto workingTeacherTableProbs = teacherTableProbs;
					auto workingTeacherSafeProbs = teacherSafeProbs;
					auto workingTeacherActionTable = teacherActionTable;
					auto workingActionMask = oldActionMasks.to(torch::kBool);
					torch::Tensor workingTeacherIndices = {};

					int64_t actionTableTopK = tlConfig.continuousActionTableTopK;
					if (actionTableTopK > 0 && actionTableTopK < teacherTableProbs.size(-1)) {
						auto topK = teacherTableProbs.topk(actionTableTopK, -1);
						workingTeacherTableProbs = std::get<0>(topK);
						workingTeacherIndices = std::get<1>(topK);
						workingTeacherTableProbs =
							workingTeacherTableProbs /
							workingTeacherTableProbs.sum(-1, true).clamp_min(ACTION_MIN_PROB);
						workingTeacherSafeProbs = workingTeacherTableProbs.clamp_min(ACTION_MIN_PROB);
						workingActionMask = oldActionMasks.gather(-1, workingTeacherIndices).to(torch::kBool);
						workingTeacherActionTable = teacherActionTable.gather(
							1,
							workingTeacherIndices.unsqueeze(-1).expand({ -1, -1, config.continuousActionSize })
						);
					}

					auto studentTableLogits =
						ComputeHybridActionTableLogProb(workingTeacherActionTable, newMean, newStd, newButtonLogits);
					auto maskedStudentTableLogits =
						studentTableLogits +
						ACTION_DISABLED_LOGIT * workingActionMask.logical_not().to(studentTableLogits.dtype());
					auto studentTableLogProbs = torch::log_softmax(maskedStudentTableLogits, -1);
					auto studentTableProbs = studentTableLogProbs.exp();

					auto analogTeacherTable = workingTeacherActionTable.slice(-1, 0, newMean.size(-1)).to(newMean.dtype());
					analogTargets = (workingTeacherTableProbs.unsqueeze(-1) * analogTeacherTable).sum(1);

					auto buttonTeacherTable =
						(workingTeacherActionTable.slice(-1, newMean.size(-1), config.continuousActionSize) > 0)
						.to(newButtonLogits.dtype());
					buttonTargetProbs = (workingTeacherTableProbs.unsqueeze(-1) * buttonTeacherTable).sum(1);

					auto aerialActionMask =
						(workingTeacherActionTable.select(-1, 2).abs() > 0.25f) |
						(workingTeacherActionTable.select(-1, 4).abs() > 0.25f) |
						(workingTeacherActionTable.select(-1, 5) > 0.0f) |
						(workingTeacherActionTable.select(-1, 6) > 0.0f) |
						(workingTeacherActionTable.select(-1, 7) > 0.0f);
					auto aerialTargetMass =
						(workingTeacherTableProbs * aerialActionMask.to(workingTeacherTableProbs.dtype())).sum(-1);
					aerialTargetRatio = aerialTargetMass.mean();
					if (tlConfig.continuousAerialSampleWeight != 1.0f) {
						sampleWeights =
							sampleWeights *
							(1.0f + (tlConfig.continuousAerialSampleWeight - 1.0f) * aerialTargetMass);
					}

					if (tlConfig.useKLDiv) {
						actionTableLoss =
							((workingTeacherSafeProbs * (workingTeacherSafeProbs.log() - studentTableLogProbs)).sum(-1) * sampleWeights).mean();
					} else {
						actionTableLoss =
							((workingTeacherTableProbs - studentTableProbs).abs().mean(-1) * sampleWeights).mean();
					}
					if (workingTeacherIndices.defined()) {
						auto studentBestOffsets = studentTableProbs.argmax(-1, true);
						auto studentBestIndices = workingTeacherIndices.gather(-1, studentBestOffsets).squeeze(-1);
						actionTableAccuracy =
							(studentBestIndices == teacherTableProbs.argmax(-1)).to(newMean.dtype()).mean();
					} else {
						actionTableAccuracy =
							(studentTableProbs.argmax(-1) == teacherTableProbs.argmax(-1)).to(newMean.dtype()).mean();
					}
				} else {
					auto aerialMask =
						(teacherActionTargets.select(-1, 2).abs() > 0.25f) |
						(teacherActionTargets.select(-1, 4).abs() > 0.25f) |
						(teacherActionTargets.select(-1, 5) > 0.0f) |
						(teacherActionTargets.select(-1, 6) > 0.0f) |
						(teacherActionTargets.select(-1, 7) > 0.0f);
					aerialTargetRatio = aerialMask.to(torch::kFloat32).mean().to(newMean.dtype());
					if (tlConfig.continuousAerialSampleWeight != 1.0f) {
						auto aerialWeights = torch::full_like(sampleWeights, tlConfig.continuousAerialSampleWeight);
						sampleWeights = torch::where(aerialMask, aerialWeights, sampleWeights);
					}

					analogTargets = teacherActionTargets.slice(-1, 0, newMean.size(-1)).to(newMean.dtype());
					buttonTargetProbs =
						(teacherActionTargets.slice(-1, newMean.size(-1), config.continuousActionSize) > 0)
						.to(newButtonLogits.dtype());
				}

				auto actionMSE = (((newMean - analogTargets).pow(2)).mean(-1) * sampleWeights).mean();
				auto actionMAE = (((newMean - analogTargets).abs()).mean(-1) * sampleWeights).mean();
				auto actionNLL = ((-GaussianLogPdf(analogTargets, newMean, newStd).sum(-1)) * sampleWeights).mean();
				float targetStdValue = tlConfig.continuousTargetStd > 0 ? tlConfig.continuousTargetStd : config.varMin;
				targetStdValue = RS_CLAMP(targetStdValue, config.varMin, config.varMax);
				auto stdTarget = torch::full_like(newStd, targetStdValue);
				auto stdLoss = (((newStd - stdTarget).pow(2)).mean(-1) * sampleWeights).mean();

				torch::Tensor buttonBCELoss = torch::zeros({}, newMean.options());
				if (tlConfig.continuousButtonBCELossWeight > 0) {
					auto perDimButtonBCE = -BernoulliLogProbFromLogits(buttonTargetProbs, newButtonLogits);
					auto buttonPositiveWeights = torch::tensor({ 3.0f, 2.5f, 2.0f }, newButtonLogits.options()).view({ 1, -1 });
					auto buttonWeights = 1.0f + buttonTargetProbs * (buttonPositiveWeights - 1.0f);
					auto perSampleButtonBCE =
						(perDimButtonBCE * buttonWeights).sum(-1) /
						buttonWeights.sum(-1).clamp_min(1e-6f);
					buttonBCELoss = (perSampleButtonBCE * sampleWeights).mean();
				}

				torch::Tensor transferLearnLoss =
					actionTableLoss * tlConfig.continuousActionTableLossWeight +
					actionMSE * tlConfig.continuousActionMSEWeight +
					actionNLL * tlConfig.continuousActionNLLWeight +
					stdLoss * tlConfig.continuousStdLossWeight +
					buttonBCELoss * tlConfig.continuousButtonBCELossWeight;
				transferLearnLoss = transferLearnLoss.pow(tlConfig.lossExponent);
				transferLearnLoss *= tlConfig.lossScale;

				if (i == 0) {
					RG_NO_GRAD;
					report["Transfer Learn Accuracy"] = actionTableAccuracy.detach().cpu().item<float>();
					report["Transfer Learn Table Loss"] = actionTableLoss.detach().cpu().item<float>();
					report["Transfer Learn MSE"] = actionMSE.detach().cpu().item<float>();
					report["Transfer Learn MAE"] = actionMAE.detach().cpu().item<float>();
					report["Transfer Learn NLL"] = actionNLL.detach().cpu().item<float>();
					report["Transfer Learn Std Loss"] = stdLoss.detach().cpu().item<float>();
					report["Transfer Learn Button BCE"] = buttonBCELoss.detach().cpu().item<float>();
					report["Transfer Learn Loss"] = transferLearnLoss.detach().cpu().item<float>();
					report["Policy Entropy"] = ComputeContinuousEntropy(newMean, newStd, newButtonLogits, config.varMax).detach().cpu().item<float>();
					report["Mean Policy Std"] = newStd.mean().detach().cpu().item<float>();
					report["Aerial Target Ratio"] = aerialTargetRatio.detach().cpu().item<float>();
					report["Teacher Confidence"] = teacherConfidence.mean().detach().cpu().item<float>();
				}

				transferLearnLoss.backward();
				models.StepOptims();
			}
		}

		auto policyAfter = models["policy"]->CopyParams();
		report["Policy Update Magnitude"] = (policyBefore - policyAfter).norm().item<float>();
	} else {
		if (oldIsContinuous)
			RG_ERR_CLOSE("TransferLearn: continuous-teacher -> discrete-student transfer learning is not implemented");

		// === Discrete transfer learning (unchanged) ===
		torch::Tensor oldProbs;
		{ // No grad for old model inference
			RG_NO_GRAD;
			oldProbs = InferPolicyProbsFromModels(oldModels, oldObs, oldActionMasks, config.policyTemperature, config.useHalfPrecision);
			report["Old Policy Entropy"] = ComputeEntropy(oldProbs, oldActionMasks, config.maskEntropy).detach().cpu().item<float>();

			if (actionMaps.defined())
				oldProbs = oldProbs.gather(1, actionMaps);
		}

		for (auto& model : GetPolicyModels())
			model->SetOptimLR(tlConfig.lr);

		auto policyBefore = models["policy"]->CopyParams();
		
		for (int i = 0; i < tlConfig.epochs; i++) {
			torch::Tensor newProbs = InferPolicyProbsFromModels(models, newObs, newActionMasks, config.policyTemperature, false);

			// Non-summative KL div	loss
			torch::Tensor transferLearnLoss;
			if (tlConfig.useKLDiv) {
				transferLearnLoss = (oldProbs * torch::log(oldProbs / newProbs)).abs();
			} else {
				transferLearnLoss = (oldProbs - newProbs).abs();
			}
			transferLearnLoss = transferLearnLoss.pow(tlConfig.lossExponent);
			transferLearnLoss = transferLearnLoss.mean();
			transferLearnLoss *= tlConfig.lossScale;

			if (i == 0) {
				RG_NO_GRAD;
				torch::Tensor matchingActionsMask = (newProbs.detach().argmax(-1) == oldProbs.detach().argmax(-1));
				report["Transfer Learn Accuracy"] = matchingActionsMask.to(torch::kFloat).mean().cpu().item<float>();
				report["Transfer Learn Loss"] = transferLearnLoss.detach().cpu().item<float>();

				report["Policy Entropy"] = ComputeEntropy(newProbs, newActionMasks, config.maskEntropy).detach().cpu().item<float>();
			}

			transferLearnLoss.backward();

			models.StepOptims();
		}

		auto policyAfter = models["policy"]->CopyParams();
		report["Policy Update Magnitude"] = (policyBefore - policyAfter).norm().item<float>();
	}
}

void GGL::PPOLearner::SaveTo(std::filesystem::path folderPath) {
	models.Save(folderPath);
}

void GGL::PPOLearner::LoadFrom(std::filesystem::path folderPath)  {
	if (!std::filesystem::is_directory(folderPath))
		RG_ERR_CLOSE("PPOLearner:LoadFrom(): Path " << folderPath << " is not a valid directory");

	models.Load(folderPath, true, true);

	SetLearningRates(config.policyLR, config.criticLR);
}

void GGL::PPOLearner::SetLearningRates(float policyLR, float criticLR) {
	// Log only when values change (AutoTrainer reasserts every iter; config may already match).
	static float s_loggedPol = -1.f, s_loggedCrit = -1.f;
	const bool logChange = (policyLR != s_loggedPol) || (criticLR != s_loggedCrit);

	config.policyLR = policyLR;
	config.criticLR = criticLR;

	models["policy"]->SetOptimLR(policyLR);
	models["critic"]->SetOptimLR(criticLR);

	if (models["shared_head"])
		models["shared_head"]->SetOptimLR(RS_MIN(policyLR, criticLR));

	if (logChange) {
		s_loggedPol = policyLR;
		s_loggedCrit = criticLR;
		RG_LOG("PPOLearner: " << RS_STR(std::scientific << "Set learning rate to [" << policyLR << ", " << criticLR << "]"));
	}
}

bool GGL::PPOLearner::TrySetGuidingPolicy(
	std::filesystem::path folderPath, float strength, int obsSize, int policyOutputSize
) {
	if (folderPath.empty() || !std::filesystem::is_directory(folderPath))
		return false;
	if (!std::filesystem::exists(folderPath / "POLICY.lt"))
		return false;
	if (obsSize <= 0 || policyOutputSize <= 0)
		return false;

	strength = RS_CLAMP(strength, 0.f, 1.f);
	config.guidingStrength = strength;
	config.guidingPolicyPath = folderPath;

	try {
		if (guidingPolicyModels.map.empty()) {
			if (config.useAttentionHead && config.attentionHeadConfig) {
				MakeModelsWithAttention(
					false, obsSize, policyOutputSize, *config.attentionHeadConfig,
					config.policy, config.critic, device, guidingPolicyModels,
					config.policyType, config.continuousActionSize);
			} else {
				MakeModels(
					false, obsSize, policyOutputSize, config.sharedHead,
					config.policy, config.critic, device, guidingPolicyModels,
					config.policyType, config.continuousActionSize);
			}
		}
		guidingPolicyModels.Load(folderPath, false, false);
		config.useGuidingPolicy = true;
		RG_LOG("PPOLearner: soft distill guiding ON path=" << folderPath
			<< " strength=" << strength);
		return true;
	} catch (const std::exception& e) {
		RG_LOG("PPOLearner: soft distill guiding FAILED (arch/path) — " << e.what());
		guidingPolicyModels.Free();
		config.useGuidingPolicy = false;
		return false;
	}
}

void GGL::PPOLearner::ClearGuidingPolicy() {
	if (!config.useGuidingPolicy && guidingPolicyModels.map.empty())
		return;
	config.useGuidingPolicy = false;
	config.guidingStrength = 0.f;
	guidingPolicyModels.Free();
	RG_LOG("PPOLearner: soft distill guiding OFF");
}

GGL::ModelSet GGL::PPOLearner::GetPolicyModels() {
	ModelSet result = {};
	for (Model* model : models) {
		if (model->modelName == "critic")
			continue;
		
		result.Add(model);
	}
	return result;
}
