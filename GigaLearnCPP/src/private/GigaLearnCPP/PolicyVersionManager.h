#pragma once

#include "Util/Models.h"
#include <GigaLearnCPP/SkillTrackerConfig.h>
#include <GigaLearnCPP/Eval/InSimEval.h>
#include <GigaLearnCPP/Util/Report.h>
#include <GigaLearnCPP/Util/RenderSender.h>

#include <nlohmann/json.hpp>

namespace GGL {
	class PPOLearner;

	struct SkillRating {
		std::map<std::string, float> data;

		static std::string GetModeName(const RLGC::GameState& state) {
			int playersOnTeams[2] = { 0, 0 };
			for (auto& player : state.players)
				playersOnTeams[(int)player.team]++;

			int minPlayersOnTeam = RS_MIN(playersOnTeams[0], playersOnTeams[1]);
			int maxPlayersOnTeam = RS_MAX(playersOnTeams[0], playersOnTeams[1]);

			std::string name = RS_STR(minPlayersOnTeam << "v" << maxPlayersOnTeam);
			return name;
		}

		float& GetRating(std::string name, float defaultRating) {
			if (data.contains(name)) {
				return data[name];
			} else {
				data[name] = defaultRating;
				return data[name];
			}
		}

		float& GetRating(const RLGC::GameState& state, float defaultRating) {
			return GetRating(GetModeName(state), defaultRating);
		}

		nlohmann::json ToJSON() {
			nlohmann::json j = {};
			for (auto pair : data)
				j[pair.first] = pair.second;
			return j;
		}

		void ReadFromJSON(const nlohmann::json& j) {
			data = {};
			for (auto& pair : j.items())
				data[pair.key()] = pair.value();
		}
	};

	struct PolicyVersion {
		uint64_t timesteps;
		ModelSet models;
		SkillRating ratings;
	};

	struct PolicyVersionManager {
		std::vector<PolicyVersion> versions;
		std::filesystem::path saveFolder;
		int maxVersions;
		uint64_t tsPerVersion;

		//////////////////

		struct {
			SkillTrackerConfig config;

			RLGC::EnvSet* envSet;
			RLGC::EnvSet* envSet1v1 = NULL;
			int curGoals = 0;

			bool doContinuation = false;
			int prevOldVersionIndex;
			Team prevNewTeam;
			float prevSimTime;

			InSimEvalGate gate = {};
			InSimEvalResult lastResult = {};

			SkillRating curRatings = {};
		} skill;

		RenderSender* renderSender;

		PolicyVersionManager(
			std::filesystem::path saveFolder, int maxVersions, uint64_t tsPerVersion,
			const SkillTrackerConfig& skillTrackerConfig, const RLGC::EnvSetConfig& envSetConfig,
			RenderSender* renderSender = NULL);

		// NOTE: Passed models should not be already cloned
		PolicyVersion& AddVersion(ModelSet modelsToClone, uint64_t timesteps);

		void SaveVersions();
		void LoadVersions(ModelSet modelsTemplate, uint64_t curTimesteps);

		void SortVersions();

		void RunSkillMatches(PPOLearner* ppo, Report& report);
		void RunSkillMatchesOnEnvSet(PPOLearner* ppo, RLGC::EnvSet* envSet, bool allowContinuation);

		void OnIteration(PPOLearner* ppo, Report& report, int64_t totalTimesteps, int64_t prevTotalTimesteps);

		// Hot-reload SkillTracker knobs from LearnerConfig (AutoTrainer / curriculum).
		// Lazily builds RocketSim skill arenas the first time enabled becomes true
		// (unless allowArenaCreate=false — used while Collect||Learn owns the GPU).
		void SyncSkillConfig(
			const SkillTrackerConfig& cfg,
			const RLGC::EnvSetConfig& trainEnvConfig,
			bool allowArenaCreate = true);
		// True if skill is enabled but arenas are not built yet (needs a safe flush).
		bool NeedsSkillArenaCreate() const {
			if (!skill.config.enabled)
				return false;
			if (!skill.envSet)
				return true;
			return skill.config.evaluate1v1 && skill.config.envCreateFn1v1 && !skill.envSet1v1;
		}

		void AddRunningStatsToJSON(nlohmann::json& json);
		void LoadRunningStatsFromJSON(const nlohmann::json& json);

		// Frees the skill-tracking environments and the cloned model sets held by each policy version.
		// NOTE: renderSender is owned by the Learner and is intentionally not freed here.
		~PolicyVersionManager();

		// Non-copyable: owns EnvSet pointers and cloned models.
		PolicyVersionManager(const PolicyVersionManager&) = delete;
		PolicyVersionManager& operator=(const PolicyVersionManager&) = delete;
	};
}
