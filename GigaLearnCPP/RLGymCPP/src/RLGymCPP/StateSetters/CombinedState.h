#pragma once

#include "StateSetter.h"

#include "../Math.h"

namespace RLGC {
	// Combine state setters with weights
	// On env reset, one of the child state setters will be chosen at random, weighted by their corresponding weight
	class CombinedState : public StateSetter {
	private:
		std::vector<StateSetter*> setters = {};
		std::vector<float> cumulativeWeights = {};
		float totalWeight;

	public:
		CombinedState(const std::vector<std::pair<StateSetter*, float>>& setters) {

			RG_ASSERT(!setters.empty());

			totalWeight = 0;
			for (auto& pair : setters) {
				this->setters.push_back(pair.first);

				if (pair.second < 0 || isnan(pair.second) || isinf(pair.second)) {
					RG_ERR_CLOSE("CombinedState: State setter \"" << typeid(*pair.first).name() << "\" has a negative or invalid weight of " << pair.second);
				}

				totalWeight += pair.second;
				cumulativeWeights.push_back(totalWeight);
			}
		}

		void ResetArena(Arena* arena) override {
			arena->SetMutatorConfig(MutatorConfig(arena->gameMode)); // Reset any lingering mutators from previous states
			float f = RocketSim::Math::RandFloat(0, totalWeight);

			for (int i = 0; i < setters.size(); i++) {
				if (f <= cumulativeWeights[i]) {
					setters[i]->ResetArena(arena);
					return;
				}
			}

			RG_ERR_CLOSE("CombinedState ran out of setters before the matching cumulative weight was found (this should never happen)");
		}

		// Rebuild sampling weights in-place (same setter count). Used by Apex / curricula.
		void SetWeights(const std::vector<float>& weights) {
			RG_ASSERT(weights.size() == setters.size());
			totalWeight = 0;
			cumulativeWeights.clear();
			for (float w : weights) {
				if (w < 0 || isnan(w) || isinf(w))
					RG_ERR_CLOSE("CombinedState::SetWeights: invalid weight " << w);
				totalWeight += w;
				cumulativeWeights.push_back(totalWeight);
			}
			RG_ASSERT(totalWeight > 0);
		}

		size_t NumSetters() const { return setters.size(); }
	};
}