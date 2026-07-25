#include "RuntimeRewardRegistry.h"

#include <algorithm>
#include <cmath>

namespace RLGC {

	namespace {
		// AutoTrainer / guide aliases → canonical GetName() keys used by EnvSet / GPU remap.
		const char* ResolveRewardAlias(const std::string& name) {
			if (name == "AirDribbleReward")
				return "AirReward";
			if (name == "BoostPickupReward")
				return "PickupBoostReward";
			if (name == "PressureFlickReward")
				return "PressureFlick";
			if (name == "MawkzyFlickReward")
				return "MawkzyFlick";
			if (name == "BallPossessionReward")
				return "PossessionReward";
			if (name == "StrongTouch")
				return "StrongTouchReward";
			if (name == "TouchAccel")
				return "TouchAccelReward";
			return nullptr;
		}
	}

	RuntimeRewardRegistry& RuntimeRewardRegistry::Instance() {
		static RuntimeRewardRegistry inst;
		return inst;
	}

	float RuntimeRewardRegistry::GetMultiplier(const std::string& rewardName) const {
		std::lock_guard lock(_mutex);
		auto it = _multipliers.find(rewardName);
		if (it != _multipliers.end())
			return it->second;
		// Also accept guide/CPU aliases when the canonical key is what EnvSet/GPU asks for.
		if (rewardName == "AirReward") {
			auto a = _multipliers.find("AirDribbleReward");
			if (a != _multipliers.end())
				return a->second;
		}
		if (rewardName == "PickupBoostReward") {
			auto a = _multipliers.find("BoostPickupReward");
			if (a != _multipliers.end())
				return a->second;
		}
		if (rewardName == "PressureFlick") {
			auto a = _multipliers.find("PressureFlickReward");
			if (a != _multipliers.end())
				return a->second;
		}
		if (rewardName == "MawkzyFlick") {
			auto a = _multipliers.find("MawkzyFlickReward");
			if (a != _multipliers.end())
				return a->second;
		}
		return 1.f;
	}

	void RuntimeRewardRegistry::SetMultiplier(const std::string& rewardName, float mult) {
		std::lock_guard lock(_mutex);
		if (!std::isfinite(mult))
			mult = 1.f;
		mult = (std::max)(-1e6f, (std::min)(1e6f, mult));
		_multipliers[rewardName] = mult;
		if (const char* canon = ResolveRewardAlias(rewardName))
			_multipliers[canon] = mult;
	}

	void RuntimeRewardRegistry::SetAll(const std::unordered_map<std::string, float>& weights) {
		std::lock_guard lock(_mutex);
		_multipliers.clear();
		for (const auto& [k, v] : weights) {
			float mult = v;
			if (!std::isfinite(mult))
				mult = 1.f;
			mult = (std::max)(-1e6f, (std::min)(1e6f, mult));
			_multipliers[k] = mult;
			if (const char* canon = ResolveRewardAlias(k)) {
				auto existing = _multipliers.find(canon);
				if (existing == _multipliers.end())
					_multipliers[canon] = mult;
				else
					existing->second = (std::max)(existing->second, mult);
			}
		}
	}

	void RuntimeRewardRegistry::ClearOverrides() {
		std::lock_guard lock(_mutex);
		_multipliers.clear();
	}

	std::unordered_map<std::string, float> RuntimeRewardRegistry::Snapshot() const {
		std::lock_guard lock(_mutex);
		return _multipliers;
	}

}
