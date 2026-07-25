#pragma once
#include "../FrameworkTorch.h"

namespace GGL {

	struct ExperienceTensors {
		torch::Tensor
			states, actions, logProbs, targetValues, actionMasks, advantages;

		auto begin() { return &states; }
		auto end() { return &advantages + 1; }
		auto begin() const { return &states; }
		auto end() const { return &advantages + 1; }
	};

	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/ppo/experience_buffer.py
	class ExperienceBuffer {
	public:

		torch::Device device;
		int seed;

		ExperienceTensors data;

		std::default_random_engine rng;

		ExperienceBuffer(int seed, torch::Device device);

		ExperienceTensors _GetSamples(const int64_t* indices, size_t size) const;

		// Not const because it uses our random engine.
		// If prioritySampling and advantages are set, uses |A|-weighted Gumbel-top-k permutation
		// so event-rich transitions appear earlier / more uniformly across mini-batches.
		// skipShuffle: when true and !prioritySampling and batchSize covers the buffer, returns
		// the raw buffer as a single batch (no index_select) — pure throughput mode.
		std::vector<ExperienceTensors> GetAllBatchesShuffled(
			int64_t batchSize, bool overbatching, bool prioritySampling = false, bool skipShuffle = false);
	};
}