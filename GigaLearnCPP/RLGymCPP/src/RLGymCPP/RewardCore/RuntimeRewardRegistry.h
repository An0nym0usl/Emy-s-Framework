#pragma once

#include <string>
#include <unordered_map>
#include <mutex>

namespace RLGC {

	// Global runtime multipliers for reward GetName() keys.
	// AutoTrainer orchestrator writes reward_weights in runtime_overrides.json;
	// EnvSet multiplies base weight by GetMultiplier(name) each step.
	class RuntimeRewardRegistry {
	public:
		static RuntimeRewardRegistry& Instance();

		float GetMultiplier(const std::string& rewardName) const;
		void SetMultiplier(const std::string& rewardName, float mult);
		void SetAll(const std::unordered_map<std::string, float>& weights);
		void ClearOverrides();

		std::unordered_map<std::string, float> Snapshot() const;

	private:
		mutable std::mutex _mutex;
		std::unordered_map<std::string, float> _multipliers;
	};

	inline float RuntimeRewardScale(const std::string& rewardName, float baseWeight) {
		return baseWeight * RuntimeRewardRegistry::Instance().GetMultiplier(rewardName);
	}

}
