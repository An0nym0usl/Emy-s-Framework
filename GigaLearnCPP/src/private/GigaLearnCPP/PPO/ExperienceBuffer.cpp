#include "ExperienceBuffer.h"

#include <numeric>
#include <algorithm>

using namespace torch;

GGL::ExperienceBuffer::ExperienceBuffer(int seed, torch::Device device) :
	seed(seed), device(device), rng(seed) {

}

GGL::ExperienceTensors GGL::ExperienceBuffer::_GetSamples(const int64_t* indices, size_t size) const {

	// Gathers an arbitrary set of rows. Kept for API compatibility; the hot path
	// (GetAllBatchesShuffled) now uses a single pre-shuffled contiguous blob instead.
	Tensor tIndices = torch::from_blob(
		const_cast<int64_t*>(indices), { (int64_t)size }, torch::kInt64
	).to(data.states.device());

	ExperienceTensors result;
	auto* toItr = result.begin();
	auto* fromItr = data.begin();
	for (; toItr != result.end(); toItr++, fromItr++) {
		if (fromItr->defined() && fromItr->size(0) == data.states.size(0))
			*toItr = torch::index_select(*fromItr, 0, tIndices);
		else
			*toItr = *fromItr; // pass through empty/undefined tensors (e.g. action masks in continuous mode)
	}

	return result;
}

std::vector<GGL::ExperienceTensors> GGL::ExperienceBuffer::GetAllBatchesShuffled(
	int64_t batchSize, bool overbatching, bool prioritySampling, bool skipShuffle
) {

	RG_NO_GRAD;

	const int64_t expSize = data.states.size(0);

	// Throughput path: one full-buffer batch, no priority — avoid O(N) index_select shuffle.
	const bool canSkipShuffle =
		skipShuffle && !prioritySampling && batchSize >= expSize && expSize > 0;
	if (canSkipShuffle) {
		std::vector<ExperienceTensors> result;
		ExperienceTensors view = data;
		// Guarantee contiguous rows for Learn narrow()/slice (device-XP bank is already
		// contiguous at offset 0; host path may hand non-contiguous views after cat/narrow).
		auto ensureContig = [](torch::Tensor& t) {
			if (t.defined() && !t.is_contiguous())
				t = t.contiguous();
		};
		ensureContig(view.states);
		ensureContig(view.actions);
		ensureContig(view.logProbs);
		ensureContig(view.targetValues);
		ensureContig(view.actionMasks);
		ensureContig(view.advantages);
		result.push_back(std::move(view));
		return result;
	}

	// Build a permutation of all rows (uniform shuffle, or |advantage|-weighted).
	std::vector<int64_t> indices(expSize);
	std::iota(indices.begin(), indices.end(), 0);

	if (prioritySampling && data.advantages.defined() && data.advantages.size(0) == expSize) {
		// Gumbel-top-k weighted permutation (no replacement): P(i) ∝ |A_i|^0.6
		auto adv = data.advantages.detach().to(torch::kCPU).contiguous().view({-1}).to(torch::kFloat32);
		auto w = adv.abs().pow(0.6f) + 1e-3f;
		auto u = torch::rand({expSize}, torch::kFloat32).clamp(1e-8f, 1.f - 1e-8f);
		auto gumbel = -torch::log(-torch::log(u));
		auto scores = w.log() + gumbel;
		auto order = std::get<1>(scores.sort(/*dim=*/0, /*descending=*/true));
		auto orderAcc = order.accessor<int64_t, 1>();
		for (int64_t i = 0; i < expSize; i++)
			indices[i] = orderAcc[i];
	} else {
		std::shuffle(indices.begin(), indices.end(), rng);
	}

	Tensor perm = torch::from_blob(
		indices.data(), { expSize }, torch::kInt64
	).clone().to(data.states.device());

	// Gather the ENTIRE buffer once into a contiguous, shuffled blob (one index_select per tensor).
	// After this, every mini-batch is just a contiguous narrow() view: no per-batch gather, no copies.
	ExperienceTensors shuffled;
	{
		auto* sItr = shuffled.begin();
		auto* dItr = data.begin();
		for (; dItr != data.end(); ++dItr, ++sItr) {
			if (dItr->defined() && dItr->size(0) == expSize)
				*sItr = dItr->index_select(0, perm).contiguous();
			else
				*sItr = *dItr; // pass through empty/undefined tensors (e.g. action masks in continuous mode)
		}
	}

	std::vector<ExperienceTensors> result;
	for (int64_t startIdx = 0; startIdx + batchSize <= expSize; startIdx += batchSize) {

		int64_t curBatchSize = batchSize;
		if (overbatching && (startIdx + batchSize * 2 > expSize)) {
			// Last batch of the iteration: extend to absorb the remainder.
			curBatchSize = expSize - startIdx;
		}

		ExperienceTensors batch;
		auto* bItr = batch.begin();
		auto* shItr = shuffled.begin();
		for (; shItr != shuffled.end(); ++shItr, ++bItr) {
			if (shItr->defined() && shItr->size(0) == expSize)
				*bItr = shItr->narrow(0, startIdx, curBatchSize);
			else
				*bItr = *shItr;
		}
		result.push_back(std::move(batch));

		if (curBatchSize > batchSize)
			break; // overbatched final batch consumed the rest
	}

	return result;
}