#pragma once
#include <RLGymCPP/BasicTypes/Lists.h>

#include "../Util/ModelConfig.h"
#include "../Util/AttentionModelConfig.h"

namespace GGL {

	// Policy type: discrete softmax policy or hybrid analog+button policy.
	enum class PolicyType { DISCRETE, CONTINUOUS };

	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/ppo/ppo_learner.py
	struct PPOLearnerConfig {

		// === Policy Type ===
		PolicyType policyType = PolicyType::DISCRETE; // Default: discrete actions

		// === Hybrid Action Config (only used when policyType == CONTINUOUS) ===
		// External action layout remains 8 floats:
		// throttle, steer, pitch, yaw, roll, jump, boost, handbrake.
		// Internally, the first 5 controls use a diagonal Gaussian and the last 3 use Bernoulli logits.
		int continuousActionSize = 8;
		float varMin = 0.1f;            // Minimum std for the analog Gaussian controls
		float varMax = 1.0f;            // Maximum std for the analog Gaussian controls

		int64_t tsPerItr = 50'000;
		int64_t batchSize = 50'000;
		int64_t miniBatchSize = 0; // Set to 0 to just use batchSize

		// On the last batch of the iteration, 
		//	if the amount of remaining experience exceeds the batch size, 
		//	all remaining experience is used as a larger batch.
		// This prevents experience loss due to batch size rounding.
		// This will only happen if the amount of remaining experience is < batchSize*2.
		bool overbatching = true;

		double maxEpisodeDuration = 120; // In seconds

		// Actions with the highest probability are always chosen, instead of being more likely
		// This will make your bot play better (usually), but is horrible for learning
		// Trying to run a PPO learn iteration with deterministic mode will throw an exception
		bool deterministic = false;

		// Use half-precision (bf16) models for INFERENCE via the maintained half-weight mirror.
		// This is much faster on GPU, not so much for CPU.
		bool useHalfPrecision = false;

		// Enable bf16 autocast during the TRAINING forward/backward pass (CUDA only).
		// bf16 keeps fp32-like dynamic range, so no GradScaler is needed; this typically gives
		// a large speedup and lower memory use on modern GPUs (Ampere+) at no stability cost.
		// Master weights and the optimizer state remain fp32 (mixed precision).
		bool useBF16Autocast = false;

		PartialModelConfig policy, critic, sharedHead;

		// === Attention Head Config ===
		// When true, the shared head uses an AttentionModel (Refinement->Think blocks)
		// instead of a plain MLP Model. Ignores sharedHead config when enabled.
		bool useAttentionHead = false;
		// Config for the attention shared head (only used when useAttentionHead = true)
		// Set numInputs to -1 to auto-detect from obsSize
		AttentionModelConfig* attentionHeadConfig = nullptr;

		int epochs = 2;
		float policyLR = 3e-4f; // Policy learning rate
		float criticLR = 3e-4f; // Critic learning rate

		float entropyScale = 0.018f; // The scale of the normalized entropy loss
		// Whether to ignore invalid actions in the entropy calculation.
		// True means that entropy will be determined only from available actions.
		// False means that entropy for unavailable actions will be zero, 
		//	meaning the entropy of the state is limited to the fraction of available actions in that state.
		// Prefer true with DefaultAction masking (Kue/GigaLearn guide): entropy over valid actions only.
		bool maskEntropy = true; 

		float clipRange = 0.2f;
		// Dynamic gradient clipping threshold (PBT / meta-evolvable via runtime_overrides).
		// Set <= 0 to disable clipping (pure throughput mode).
		float maxGradNorm = 0.5f;
		float vfCoef = 0.5f; // Value function loss coefficient (balances critic vs policy gradient in shared head)

		// Skip per-minibatch .item() syncs, CopyParams magnitude tracking, and full-buffer
		// shuffle when batch covers the whole experience (pure80 / max SPS).
		bool skipPPOMetrics = false;
		// When true and !prioritySampling, skip the expensive index_select shuffle if the
		// (only) batch consumes the entire buffer.
		bool skipExperienceShuffle = false;

		// Advantage normalization before policy loss:
		// 0 = none, 1 = batch mean/std (default), 2 = subtract mean only, 3 = / max(|A|).
		// Applied even in skipPPOMetrics (no host .item(); quality > fake SPS).
		int advantageNormMode = 1;
		float advantageNormEps = 1e-8f;

		// Clamp |log π − log π_old| before exp(ratio) — prevents Inf ratio on rare stale actions.
		float ratioLogClamp = 20.f;

		// Accumulate this many optimizer-batches before Adam step (effective batch *= N).
		// 1 = step every batch (default). Safe way to grow effective batch without OOM.
		int gradAccumulationSteps = 1;

		// Adaptive KL: if approx KL exceeds target, stop remaining epochs this Learn call.
		// 0 = disabled. Typical range 0.01–0.05 (SB3-style early stop).
		float targetKl = 0.f;
		// When targetKl > 0 and KL spikes, shrink policy LR for this Learn by this factor (then restore).
		float adaptiveKlLrScale = 0.5f;

		// Target entropy (normalized [0,1] discrete / continuous mean). <0 = disabled.
		// Softly adapts entropyScale each Learn toward the target (AutoTrainer can still override).
		float targetEntropy = -1.f;
		float entropyAdaptRate = 0.02f;
		float entropyScaleMin = 1e-5f;
		float entropyScaleMax = 0.1f;
		
		// Temperature of the policy's softmax distribution
		float policyTemperature = 1;

		float gaeLambda = 0.95f;
		float gaeGamma = 0.99f;
		float rewardClipRange = 10; // Clip range for normalized rewards, set 0 to disable

		// === OP training (ES + event-driven, free / no extra deps) ===
		// Evolution-strategy style exploration: inflate action std during rollout collection.
		// Combined with PPO gradient descent on the same batch (ES explores, PPO exploits).
		float esNoiseScale = 0.f;
		// Multiply |advantage| on high-signal event steps (terminals / reward spikes).
		float eventAdvantageBoost = 1.f;
		// When true, shuffle experience with |advantage|-weighted permutation (event-rich first).
		bool prioritySampling = true;

		bool useGuidingPolicy = false;
		std::filesystem::path guidingPolicyPath = "guiding_policy/"; // Path of the guiding policy model(s)
		float guidingStrength = 0.03f;
		// Discrete distill: true → KL(teacher‖student); false → legacy L1(|p_t−p_s|).
		bool guidingUseKL = true;
		// Clamp guiding loss contribution before strength scale (numerical safety).
		float guidingLossClip = 25.f;

		PPOLearnerConfig() {
			policy = {};
			policy.layerSizes = { 256, 256, 256 };
			critic = {};
			critic.layerSizes = { 256, 256, 256 };
			sharedHead = {};
			sharedHead.layerSizes = { 256 };
			sharedHead.addOutputLayer = false;
		}
	};
}
