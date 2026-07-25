#pragma once

#include <GigaLearnCPP/Learner.h>
#include <GigaLearnCPP/Util/Report.h>
#include <RLGymCPP/RewardCore/Reward.h>

#include <filesystem>
#include <string>
#include <vector>

namespace GGL {

	// Full hot-reload bridge: overrides, commands (pause/save), reward weights, status export.
	struct AutoTrainerBridge {
		static std::filesystem::path RootDir(const Learner* learner);

		static void ProcessCommands(Learner* learner);
		static bool IsPaused(const Learner* learner);
		static void WaitWhilePaused(Learner* learner);

		static void OnIterationStart(Learner* learner);
		// Re-apply AutoTrainer overrides AFTER Apex curriculum each iter.
		// When autotrainer_full_control / full_control (default ON) or ssl_guide_post_apex:
		// rewards, entropy, LR, gamma, league mix, GPU reset, skill-eval all win over Apex.
		// Opt-out: GIGA_AT_READONLY=1 (bridge ignores overrides) or full_control:false in YAML.
		static void ReapplyPostApex(Learner* learner);

		// After Collect||Learn join: push any CUDA surface updates deferred while Learn ran.
		static void FlushDeferredCuda(Learner* learner);

		static void WriteStatus(Learner* learner, const Report& report);

		static void WriteRewardManifest(
			const Learner* learner,
			const std::vector<RLGC::WeightedReward>& rewards);

		static void AppendMetricsHistory(const Learner* learner, const Report& report);
	};

}
