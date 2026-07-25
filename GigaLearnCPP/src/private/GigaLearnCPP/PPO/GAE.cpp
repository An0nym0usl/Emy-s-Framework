#include "GAE.h"

#include <RLGymCPP/TerminalConditions/TerminalCondition.h>

void GGL::GAE::Compute(
	torch::Tensor rews, torch::Tensor terminals, torch::Tensor valPreds, torch::Tensor truncValPreds,
	torch::Tensor& outAdvantages, torch::Tensor& outTargetValues, torch::Tensor& outReturns, float& outRewClipPortion,
	float gamma, float lambda, float returnStd, float clipRange
) {

	bool hasTruncValPreds = truncValPreds.defined();

	int numReturns = (int)rews.size(0);
	int truncCount = 0;

	float totalRew = 0, totalClippedRew = 0;

	// Device path: all-truncated TD(0). Supports optional returnStd + clip without host scan.
	// When truncValPreds.size == rews.size and every step is truncated by construction
	// (pure80 / device-XP bank), λ-GAE collapses to per-step TD(0).
	if (hasTruncValPreds
		&& truncValPreds.size(0) == numReturns
		&& numReturns > 0
		&& rews.is_cuda()
		&& valPreds.is_cuda()
		&& truncValPreds.is_cuda()) {

		torch::Tensor curReward = rews;
		if (returnStd > 0.f) {
			curReward = rews / returnStd;
			if (clipRange > 0.f) {
				// Clip portion for metrics: cheap host summary (optional).
				auto absBefore = curReward.abs();
				curReward = curReward.clamp(-clipRange, clipRange);
				auto absAfter = curReward.abs();
				// Avoid per-iter .item() when clip disabled for accounting; still compute if clip on.
				totalRew = absBefore.sum().item<float>();
				totalClippedRew = absAfter.sum().item<float>();
				outRewClipPortion = (totalRew - totalClippedRew) / RS_MAX(totalRew, 1e-7f);
			} else {
				outRewClipPortion = 0.f;
			}
		} else {
			outRewClipPortion = 0.f;
		}

		auto predReturn = curReward + gamma * truncValPreds;
		outAdvantages = predReturn - valPreds;
		outReturns = rews; // truncated → no Monte-Carlo return bootstrap chain
		outTargetValues = predReturn;
		return;
	}

	outAdvantages = torch::zeros(numReturns);
	outReturns = torch::zeros(numReturns);

	// GAE is a strictly sequential backward scan (each step depends on the previous one),
	// so it cannot be parallelized into a fast GPU kernel without a complex parallel-scan that
	// also has to special-case truncations. The CPU raw-pointer loop below is ~10x faster than
	// the equivalent tensor ops, so we keep GAE on the CPU and instead pin the inputs here:
	// any tensor arriving on the GPU is moved to a contiguous CPU tensor so data_ptr<> is valid.
	auto toCpu = [](torch::Tensor t) {
		return t.is_cuda() ? t.to(torch::kCPU).contiguous() : t.contiguous();
	};
	rews = toCpu(rews);
	terminals = toCpu(terminals);
	valPreds = toCpu(valPreds);
	if (hasTruncValPreds)
		truncValPreds = toCpu(truncValPreds);

	auto _terminals = terminals.const_data_ptr<int8_t>();
	auto _rews = rews.const_data_ptr<float>();
	auto _valPreds = valPreds.const_data_ptr<float>();

	const float* _truncValPreds;
	int numTruncs;
	if (hasTruncValPreds) {
		_truncValPreds = truncValPreds.const_data_ptr<float>();
		numTruncs = (int)truncValPreds.size(0);
	} else {
		_truncValPreds = NULL;
		numTruncs = 0;
	}

	// Fast path: every step is truncated (pure80 / maxEpisodeDuration 1-step bulk bank).
	// Episodes do not chain, so λ-GAE collapses to per-step TD(0) and is fully vectorizable.
	bool allTruncated = hasTruncValPreds && numTruncs == numReturns && numReturns > 0;
	if (allTruncated) {
		for (int i = 0; i < numReturns; i++) {
			if (_terminals[i] != (int8_t)RLGC::TerminalType::TRUNCATED) {
				allTruncated = false;
				break;
			}
		}
	}

	if (allTruncated) {
		auto _outReturns = std::vector<float>(numReturns, 0);
		auto _outAdvantages = std::vector<float>(numReturns, 0);

		for (int step = 0; step < numReturns; step++) {
			float curReward;
			if (returnStd != 0) {
				curReward = _rews[step] / returnStd;
				totalRew += abs(curReward);
				if (clipRange > 0)
					curReward = RS_CLAMP(curReward, -clipRange, clipRange);
				totalClippedRew += abs(curReward);
			} else {
				curReward = _rews[step];
				totalRew += abs(curReward);
			}

			// Trunc bootstrap matches the same index (forward-collected nextStates).
			float nextValPred = _truncValPreds[step];
			float predReturn = curReward + gamma * nextValPred;
			_outAdvantages[step] = predReturn - _valPreds[step];
			_outReturns[step] = _rews[step]; // truncated → no return bootstrap chain
		}

		outReturns = torch::tensor(_outReturns);
		outAdvantages = torch::tensor(_outAdvantages);
		outTargetValues = valPreds.slice(0, 0, numReturns) + outAdvantages;
		outRewClipPortion = (totalRew - totalClippedRew) / RS_MAX(totalRew, 1e-7f);
		return;
	}

	float prevLambda = 0;
	float prevRet = 0;
	auto _outReturns = std::vector<float>(numReturns, 0);
	auto _outAdvantages = std::vector<float>(numReturns, 0);

	for (int step = numReturns - 1; step >= 0; step--) {
		uint8_t terminal = _terminals[step];
		float done = terminal == RLGC::TerminalType::NORMAL;
		float trunc = terminal == RLGC::TerminalType::TRUNCATED;

		float curReward;
		if (returnStd != 0) {
			curReward = _rews[step] / returnStd;

			totalRew += abs(curReward);

			// We only clip if returns are standardized
			if (clipRange > 0)
				curReward = RS_CLAMP(curReward, -clipRange, clipRange);

			totalClippedRew += abs(curReward);
		} else {
			curReward = _rews[step];
			totalRew += abs(curReward);
		}

		float nextValPred;
		if (terminal == RLGC::TerminalType::TRUNCATED) {
			// Walking backwards encounters truncations in reverse append order, so consume
			// truncValPreds from the end (last truncated episode first).
			if (!hasTruncValPreds)
				RG_ERR_CLOSE("GAE encountered a truncated terminal, but has no truncated val pred");

			if (truncCount >= numTruncs)
				RG_ERR_CLOSE("GAE encountered too many truncated terminals, not enough val preds (max: " << numTruncs << ")")

			nextValPred = _truncValPreds[numTruncs - 1 - truncCount];
			truncCount++;
		} else if (step >= numReturns - 1) {
			// Edge case: last buffer step without explicit trunc/terminal (should not happen
			// with force-truncate). Treat as truncated bootstrap with V=0 — never OOB-read.
			nextValPred = 0.f;
			trunc = 1.f;
		} else {
			nextValPred = _valPreds[step + 1];
		}

		// TD residual: r + γ V(s') − V(s). Truncation bootstraps V(s') from truncValPreds;
		// true episode end (NORMAL) zeroes the bootstrap.
		float predReturn = curReward + gamma * nextValPred * (1 - done);
		float delta = predReturn - _valPreds[step];
		// Monte-Carlo-style return for critic targets (stops at done OR trunc).
		float curReturn = _rews[step] + prevRet * gamma * (1 - done) * (1 - trunc);
		_outReturns[step] = curReturn;

		// GAE(λ): δ_t + γλ (1−done)(1−trunc) Â_{t+1}
		prevLambda = delta + gamma * lambda * (1 - done) * (1 - trunc) * prevLambda;
		_outAdvantages[step] = prevLambda;

		prevRet = curReturn;
	}
	
	if (hasTruncValPreds)
		if (truncCount != truncValPreds.size(0))
			RG_ERR_CLOSE("GAE didn't receive expected truncation count (only " << truncCount << "/" << truncValPreds.size(0) << ")");

	outReturns = torch::tensor(_outReturns);
	outAdvantages = torch::tensor(_outAdvantages);
	outTargetValues = valPreds.slice(0, 0, numReturns) + outAdvantages;
	outRewClipPortion = (totalRew - totalClippedRew) / RS_MAX(totalRew, 1e-7f);
}
