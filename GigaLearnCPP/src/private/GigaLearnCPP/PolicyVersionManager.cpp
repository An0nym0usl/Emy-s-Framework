#include "PolicyVersionManager.h"
#include <nlohmann/json.hpp>

#include <GigaLearnCPP/Util/Utils.h>

#include <RLGymCPP/StateSetters/FuzzedKickoffState.h>
#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>
#include <RLGymCPP/RewardCore/CommonRewards.h>

#include <private/GigaLearnCPP/PPO/PPOLearner.h>

using namespace nlohmann;

namespace {
	void ConfigureSkillArenaRewards(RLGC::EnvSet* envSet) {
		if (!envSet)
			return;
		for (int i = 0; i < (int)envSet->arenas.size(); i++) {
			// Keep a cheap dense signal for Eval/AvgReward (Elo still from goals only).
			envSet->rewards[i].clear();
			envSet->rewards[i].push_back({ new RLGC::VelocityPlayerToBallReward(), 1.f });
			envSet->stateSetters[i] = { new RLGC::FuzzedKickoffState() };
			envSet->terminalConditions[i] = { new RLGC::GoalScoreCondition() };
		}
	}
}

GGL::PolicyVersionManager::PolicyVersionManager(
	std::filesystem::path saveFolder, int maxVersions, uint64_t tsPerVersion, 
	const SkillTrackerConfig& skillTrackerConfig, const RLGC::EnvSetConfig& envSetConfig, RenderSender* renderSender) : 
	saveFolder(saveFolder), maxVersions(maxVersions), tsPerVersion(tsPerVersion), 
	renderSender(renderSender) {

	skill.config = skillTrackerConfig;

	skill.gate.updateInterval = skillTrackerConfig.updateInterval;
	skill.gate.tsPerEval = skillTrackerConfig.tsPerEval;
	skill.gate.iterationsSinceRan = 0;
	skill.gate.lastEvalTimesteps = 0;

	if (!std::filesystem::exists(saveFolder))
		std::filesystem::create_directories(saveFolder);

	if (skill.config.enabled) {
		RLGC::EnvSetConfig skillEnvSetConfig = envSetConfig;
		skillEnvSetConfig.numArenas = skill.config.numArenas;
		skill.envSet = new RLGC::EnvSet(skillEnvSetConfig);
		ConfigureSkillArenaRewards(skill.envSet);

		if (skill.config.evaluate1v1 && skill.config.envCreateFn1v1) {
			RLGC::EnvSetConfig skillEnvSetConfig1v1 = envSetConfig;
			skillEnvSetConfig1v1.envCreateFn = skill.config.envCreateFn1v1;
			skillEnvSetConfig1v1.numArenas = skill.config.numArenas1v1;
			skill.envSet1v1 = new RLGC::EnvSet(skillEnvSetConfig1v1);
			ConfigureSkillArenaRewards(skill.envSet1v1);
		}
	} else {
		skill.envSet = NULL;
		skill.envSet1v1 = NULL;
	}
}

GGL::PolicyVersionManager::~PolicyVersionManager() {
	// Each policy version holds a cloned ModelSet of raw Model* pointers that must be freed explicitly.
	for (auto& version : versions)
		version.models.Free();
	versions.clear();

	delete skill.envSet;
	skill.envSet = NULL;
	delete skill.envSet1v1;
	skill.envSet1v1 = NULL;
	// renderSender is owned by the Learner; do not delete it here.
}

GGL::PolicyVersion& GGL::PolicyVersionManager::AddVersion(ModelSet modelsToClone, uint64_t timesteps) {
	RG_NO_GRAD;

	auto models = modelsToClone.CloneAll();

	auto newVersion = PolicyVersion{
		timesteps,
		models
	};

	newVersion.ratings = skill.curRatings;

	versions.push_back(newVersion);

	SortVersions();

	// Remove old versions
	while (versions.size() > maxVersions) {
		auto& toRemove = versions[0];
		toRemove.models.Free();
		versions.erase(versions.begin());
	}

	return versions.back();
}

void GGL::PolicyVersionManager::SaveVersions() {
	RG_NO_GRAD;

	// Remove old saved versions
	std::set<int64_t> allSavedTimesteps = Utils::FindNumberedDirs(saveFolder);

	for (int64_t savedTimesteps : allSavedTimesteps) {
		bool matchesVersion = false;
		for (auto& version : versions)
			matchesVersion |= (savedTimesteps == version.timesteps);

		if (matchesVersion) {
			// We want to keep this
			allSavedTimesteps.insert(savedTimesteps);
		} else {
			// Get rid of it
			std::filesystem::remove_all(saveFolder / std::to_string(savedTimesteps));
		}
	}

	for (auto& version : versions) {
		if (allSavedTimesteps.contains(version.timesteps))
			continue;
		auto versionSaveFolder = saveFolder / std::to_string(version.timesteps);
		std::filesystem::create_directories(versionSaveFolder);

		version.models.Save(versionSaveFolder, false);

		{ // Save JSON
			auto jsonPath = versionSaveFolder / "STATS.json";

			std::ofstream fOut(jsonPath);
			RG_ASSERT(fOut.good());

			json j = {};
			j["skill_ratings"] = version.ratings.ToJSON();
			std::string jStr = j.dump(4);
			fOut << jStr;
		}
	}
}

void GGL::PolicyVersionManager::LoadVersions(ModelSet modelsTemplate, uint64_t curTimesteps) {

	RG_NO_GRAD;

	RG_LOG("PolicyVersionManager::LoadVersions():");

	for (auto& version : versions)
		version.models.Free();
	versions.clear();

	std::set<int64_t> allSavedTimesteps = Utils::FindNumberedDirs(saveFolder);

	std::vector<int64_t> eligibleTimesteps = {};
	for (int64_t savedTimesteps : allSavedTimesteps) {
		if (savedTimesteps > curTimesteps) {
			RG_LOG(" > Skipping policy version " << savedTimesteps << " (newer than checkpoint " << curTimesteps << ")");
			continue;
		}
		eligibleTimesteps.push_back(savedTimesteps);
	}

	// Only load the most recent N versions — loading dozens of full GPU models OOMs at startup
	int skipCount = RS_MAX(0, (int)eligibleTimesteps.size() - maxVersions);
	for (int i = skipCount; i < (int)eligibleTimesteps.size(); i++) {
		int64_t savedTimesteps = eligibleTimesteps[i];
		auto path = saveFolder / std::to_string(savedTimesteps);

		try {
			PolicyVersion& version = AddVersion(modelsTemplate, savedTimesteps);
			version.models.Load(path, false, false);

			auto jsonPath = path / "STATS.json";
			std::ifstream fIn(jsonPath);
			if (fIn.good()) {
				json j = json::parse(fIn);
				if (j.contains("skill_ratings"))
					version.ratings.ReadFromJSON(j["skill_ratings"]);
			}
		} catch (std::exception& e) {
			RG_LOG(" > Failed to load policy version " << savedTimesteps << " (" << e.what() << "), skipping");
			if (!versions.empty() && versions.back().timesteps == (uint64_t)savedTimesteps) {
				versions.back().models.Free();
				versions.pop_back();
			}
		}
	}

	if (skipCount > 0) {
		RG_LOG(" > Skipped " << skipCount << " older policy version(s) on disk (maxVersions=" << maxVersions << ")");
	}

	SortVersions();

	RG_LOG(" > Loaded " << versions.size() << " versions(s)");
}

void GGL::PolicyVersionManager::SortVersions() {
	auto fnCompareVersions = [](const PolicyVersion& a, const PolicyVersion& b) {
		return a.timesteps < b.timesteps;
	};

	std::sort(versions.begin(), versions.end(), fnCompareVersions);
}

/////////////////////////////////////////////////////////////////////

void GGL::PolicyVersionManager::RunSkillMatchesOnEnvSet(PPOLearner* ppo, RLGC::EnvSet* envSet, bool allowContinuation) {
	RG_NO_GRAD;
	bool isContinuous = (ppo->config.policyType == PolicyType::CONTINUOUS);

	auto fnUpdateRatings = [this](SkillRating& winner, SkillRating& loser, RLGC::GameState& state) {
		float& winnerRating = winner.GetRating(state, skill.config.initialRating);
		float& loserRating = loser.GetRating(state, skill.config.initialRating);

		float expDelta = (loserRating - winnerRating) / 400;
		float expected = 1 / (powf(10, expDelta) + 1);

		winnerRating += skill.config.ratingInc * (1 - expected);
		loserRating += skill.config.ratingInc * (expected - 1);
	};

	Team newTeam;
	int oldVersionIndex;
	float totalSimTime;
	if (allowContinuation && skill.doContinuation) {
		RG_ASSERT(skill.prevOldVersionIndex < versions.size());
		oldVersionIndex = skill.prevOldVersionIndex;
		newTeam = skill.prevNewTeam;
		totalSimTime = skill.prevSimTime;
	} else {
		oldVersionIndex = Math::RandInt(0, versions.size());
		newTeam = (Team)Math::RandInt(0, 2);
		totalSimTime = 0;

		// Force-reset every arena for a fresh match (Reset() alone only clears terminal ones).
		for (int i = 0; i < (int)envSet->arenas.size(); i++)
			envSet->state.terminals[i] = 1;
		envSet->Reset();
	}
	if (allowContinuation)
		skill.doContinuation = false;

	auto& oldVersion = versions[oldVersionIndex];
	skill.lastResult.opponentVersionTs = (int)oldVersion.timesteps;

	std::vector<int>
		newPlayers = {},
		oldPlayers = {};
	for (int i = 0; i < envSet->arenas.size(); i++) {
		auto& state = envSet->state.gameStates[i];
		for (int j = 0; j < state.players.size(); j++) {
			int playerIdx = envSet->state.arenaPlayerStartIdx[i] + j;
			bool isNew = (state.players[j].team == newTeam);
			(isNew ? newPlayers : oldPlayers).push_back(playerIdx);
		}
	}

	torch::Tensor
		tNewPlayers = torch::tensor(newPlayers),
		tOldPlayers = torch::tensor(oldPlayers);

	int goalsThisRun = 0;
	double rewardSum = 0.0;
	int rewardCount = 0;

	float stepTime = envSet->config.tickSkip * RLGC::CommonValues::TICK_TIME;
	for (float t = 0;
		t < skill.config.simTime && totalSimTime < skill.config.maxSimTime && goalsThisRun < envSet->arenas.size();
		t += stepTime, totalSimTime += stepTime) {

		envSet->Reset();

		torch::Tensor tStates = DIMLIST2_TO_TENSOR<float>(envSet->state.obs);
		torch::Tensor tActionMasks = DIMLIST2_TO_TENSOR<uint8_t>(envSet->state.actionMasks);

		torch::Tensor tNewStates = tStates.index_select(0, tNewPlayers);
		torch::Tensor tOldStates = tStates.index_select(0, tOldPlayers);

		torch::Tensor tNewActionMasks = tActionMasks.index_select(0, tNewPlayers);
		torch::Tensor tOldActionMasks = tActionMasks.index_select(0, tOldPlayers);

		envSet->StepFirstHalf(true);

		torch::Tensor tNewActions, tOldActions;
		torch::Tensor _tLogProbs;

		if (isContinuous) {
			int actionDim = ppo->config.continuousActionSize;
			PPOLearner::SampleContinuousActions(
				ppo->models, tNewStates.to(ppo->device, true),
				skill.config.deterministic, ppo->config.useHalfPrecision,
				ppo->config.varMin, ppo->config.varMax,
				&tNewActions, &_tLogProbs
			);
			PPOLearner::SampleContinuousActions(
				oldVersion.models, tOldStates.to(ppo->device, true),
				skill.config.deterministic, ppo->config.useHalfPrecision,
				ppo->config.varMin, ppo->config.varMax,
				&tOldActions, &_tLogProbs
			);

			auto newActions = TENSOR_TO_VEC<float>(tNewActions.flatten());
			auto oldActions = TENSOR_TO_VEC<float>(tOldActions.flatten());

			auto combinedActions = std::vector<float>(envSet->state.numPlayers * actionDim, 0.0f);
			for (int i = 0; i < newPlayers.size(); i++) {
				for (int d = 0; d < actionDim; d++)
					combinedActions[newPlayers[i] * actionDim + d] = newActions[i * actionDim + d];
			}
			for (int i = 0; i < oldPlayers.size(); i++) {
				for (int d = 0; d < actionDim; d++)
					combinedActions[oldPlayers[i] * actionDim + d] = oldActions[i * actionDim + d];
			}

			envSet->Sync();
			envSet->StepSecondHalfContinuous(combinedActions, actionDim, false);
		} else {
			PPOLearner::InferActionsFromModels(
				ppo->models, tNewStates.to(ppo->device, true), tNewActionMasks.to(ppo->device, true),
				skill.config.deterministic, ppo->config.policyTemperature, ppo->config.useHalfPrecision,
				&tNewActions, &_tLogProbs);
			PPOLearner::InferActionsFromModels(
				oldVersion.models, tOldStates.to(ppo->device, true), tOldActionMasks.to(ppo->device, true),
				skill.config.deterministic, ppo->config.policyTemperature, ppo->config.useHalfPrecision,
				&tOldActions, &_tLogProbs);

			auto newActions = TENSOR_TO_VEC<int>(tNewActions);
			auto oldActions = TENSOR_TO_VEC<int>(tOldActions);

			auto combinedActions = std::vector<int>(envSet->state.numPlayers, -1);
			for (int i = 0; i < newActions.size(); i++)
				combinedActions[newPlayers[i]] = newActions[i];
			for (int i = 0; i < oldActions.size(); i++)
				combinedActions[oldPlayers[i]] = oldActions[i];

			envSet->Sync();
			envSet->StepSecondHalf(combinedActions, false);
		}

		// Avg step reward for current-policy players (host rewards after StepSecondHalf).
		if (!envSet->state.rewards.empty() && !newPlayers.empty()) {
			for (int pi : newPlayers) {
				if (pi >= 0 && pi < (int)envSet->state.rewards.size()) {
					rewardSum += envSet->state.rewards[pi];
					rewardCount++;
				}
			}
		}

		for (int i = 0; i < envSet->arenas.size(); i++) {
			auto& gs = envSet->state.gameStates[i];
			if (gs.goalScored) {
				if (RS_TEAM_FROM_Y(gs.ball.pos.y) != newTeam) {
					fnUpdateRatings(skill.curRatings, oldVersion.ratings, gs);
					skill.lastResult.goalsFor++;
				} else {
					fnUpdateRatings(oldVersion.ratings, skill.curRatings, gs);
					skill.lastResult.goalsAgainst++;
				}

				goalsThisRun++;
			}
		}

		if (renderSender)
			renderSender->Send(envSet->state.gameStates[0]);
	}

	skill.lastResult.simSeconds += totalSimTime;
	if (rewardCount > 0) {
		int prevN = skill.lastResult.rewardSamples;
		double prevMean = skill.lastResult.avgStepReward;
		double blockSum = rewardSum;
		int newN = prevN + rewardCount;
		if (prevN <= 0)
			skill.lastResult.avgStepReward = (float)(blockSum / (double)rewardCount);
		else
			skill.lastResult.avgStepReward = (float)((prevMean * (double)prevN + blockSum) / (double)newN);
		skill.lastResult.rewardSamples = newN;
	}

	if (allowContinuation) {
		if (goalsThisRun < envSet->arenas.size() && totalSimTime < skill.config.maxSimTime) {
			RG_LOG(" > Forcing continuation (" << goalsThisRun << "/" << envSet->arenas.size() << ")");
			skill.doContinuation = true;
			skill.prevOldVersionIndex = oldVersionIndex;
			skill.prevNewTeam = newTeam;
			skill.prevSimTime = totalSimTime;
		} else {
			skill.curGoals = 0;
		}
	}
}

void GGL::PolicyVersionManager::RunSkillMatches(PPOLearner* ppo, Report& report) {
	RG_LOG("Running in-sim eval (simTime=" << skill.config.simTime
		<< "s, arenas=" << (skill.envSet ? (int)skill.envSet->arenas.size() : 0) << ")...");

	SkillRating prevCurRatings = skill.curRatings;

	// Reset per-run accumulators (keep Elo ratings).
	skill.lastResult.goalsFor = 0;
	skill.lastResult.goalsAgainst = 0;
	skill.lastResult.avgStepReward = 0.f;
	skill.lastResult.rewardSamples = 0;
	skill.lastResult.simSeconds = 0.f;
	skill.lastResult.eloDelta = 0.f;

	RunSkillMatchesOnEnvSet(ppo, skill.envSet, true);

	if (skill.envSet1v1) {
		skill.doContinuation = false;
		skill.curGoals = 0;
		RunSkillMatchesOnEnvSet(ppo, skill.envSet1v1, false);
	}

	skill.lastResult.ratings = skill.curRatings.data;
	float primary = skill.config.initialRating;
	if (skill.curRatings.data.count("1v1"))
		primary = skill.curRatings.data["1v1"];
	else if (skill.curRatings.data.count("2v2"))
		primary = skill.curRatings.data["2v2"];
	else if (!skill.curRatings.data.empty())
		primary = skill.curRatings.data.begin()->second;
	skill.lastResult.primaryElo = primary;

	float prevPrimary = skill.config.initialRating;
	if (prevCurRatings.data.count("1v1"))
		prevPrimary = prevCurRatings.data["1v1"];
	else if (prevCurRatings.data.count("2v2"))
		prevPrimary = prevCurRatings.data["2v2"];
	else if (!prevCurRatings.data.empty())
		prevPrimary = prevCurRatings.data.begin()->second;
	skill.lastResult.eloDelta = primary - prevPrimary;

	const int decided = skill.lastResult.goalsFor + skill.lastResult.goalsAgainst;
	skill.lastResult.winRate = decided > 0
		? (float)skill.lastResult.goalsFor / (float)decided
		: 0.5f;

	for (auto& pair : skill.curRatings.data) {
		float prevRating = prevCurRatings.GetRating(pair.first, skill.config.initialRating);
		float delta = pair.second - prevRating;

		std::stringstream ratingLine;
		ratingLine << " > " << pair.first << " = " << prevRating;
		if (delta != 0)
			ratingLine << " (" << (delta >= 0 ? '+' : '-') << abs(delta) << ")";

		RG_LOG(" > " << ratingLine.str());

		report["Rating/" + pair.first] = pair.second;
	}

	if (skill.config.writeEvalMetrics) {
		skill.lastResult.WriteToReport(report, "Eval");
		RG_LOG(" > Eval Elo=" << skill.lastResult.primaryElo
			<< " delta=" << skill.lastResult.eloDelta
			<< " WR=" << skill.lastResult.winRate
			<< " avgR=" << skill.lastResult.avgStepReward
			<< " GF/GA=" << skill.lastResult.goalsFor << "/" << skill.lastResult.goalsAgainst);
	}
}

void GGL::PolicyVersionManager::SyncSkillConfig(
	const SkillTrackerConfig& cfg, const RLGC::EnvSetConfig& trainEnvConfig, bool allowArenaCreate) {

	skill.config.enabled = cfg.enabled;
	skill.config.updateInterval = cfg.updateInterval;
	skill.config.tsPerEval = cfg.tsPerEval;
	skill.config.simTime = cfg.simTime;
	skill.config.maxSimTime = cfg.maxSimTime;
	skill.config.numArenas = cfg.numArenas;
	skill.config.numArenas1v1 = cfg.numArenas1v1;
	skill.config.evaluate1v1 = cfg.evaluate1v1;
	skill.config.deterministic = cfg.deterministic;
	skill.config.ratingInc = cfg.ratingInc;
	skill.config.initialRating = cfg.initialRating;
	skill.config.writeEvalMetrics = cfg.writeEvalMetrics;
	// AutoTrainer cannot pass C++ envCreateFn — keep construction-time fn if override is null.
	if (cfg.envCreateFn1v1)
		skill.config.envCreateFn1v1 = cfg.envCreateFn1v1;

	skill.gate.updateInterval = skill.config.updateInterval;
	skill.gate.tsPerEval = skill.config.tsPerEval;

	if (!skill.config.enabled)
		return;

	// Lazy arena build: from-scratch boots with skill OFF (fast iters); AutoTrainer enables later.
	// Never allocate arenas while Collect||Learn owns the GPU (caller sets allowArenaCreate=false).
	if (!allowArenaCreate)
		return;

	if (!skill.envSet) {
		int n = (std::max)(1, skill.config.numArenas);
		RG_LOG("PolicyVersionManager: lazy-creating skill arenas (num=" << n << ")");
		RLGC::EnvSetConfig skillEnvSetConfig = trainEnvConfig;
		skillEnvSetConfig.numArenas = n;
		skill.envSet = new RLGC::EnvSet(skillEnvSetConfig);
		ConfigureSkillArenaRewards(skill.envSet);
	}

	if (skill.config.evaluate1v1 && skill.config.envCreateFn1v1 && !skill.envSet1v1) {
		int n1 = (std::max)(1, skill.config.numArenas1v1);
		RG_LOG("PolicyVersionManager: lazy-creating 1v1 skill arenas (num=" << n1 << ")");
		RLGC::EnvSetConfig skillEnvSetConfig1v1 = trainEnvConfig;
		skillEnvSetConfig1v1.envCreateFn = skill.config.envCreateFn1v1;
		skillEnvSetConfig1v1.numArenas = n1;
		skill.envSet1v1 = new RLGC::EnvSet(skillEnvSetConfig1v1);
		ConfigureSkillArenaRewards(skill.envSet1v1);
	}
}

void GGL::PolicyVersionManager::OnIteration(PPOLearner* ppo, Report& report, int64_t totalTimesteps, int64_t prevTotalTimesteps) {
	if ((totalTimesteps / tsPerVersion > prevTotalTimesteps / tsPerVersion) || (prevTotalTimesteps == 0)) {
		// Save version
		AddVersion(ppo->GetPolicyModels(), totalTimesteps);
	}

	if (skill.config.enabled && skill.envSet) {
		skill.gate.OnIteration();

		if (skill.gate.Due(totalTimesteps) && !versions.empty()) {
			skill.gate.MarkRan(totalTimesteps);
			RunSkillMatches(ppo, report);
		}
	}
}

void GGL::PolicyVersionManager::AddRunningStatsToJSON(nlohmann::json& json) {
	if (skill.config.enabled)
		json["skill_ratings"] = skill.curRatings.ToJSON();
}

void GGL::PolicyVersionManager::LoadRunningStatsFromJSON(const nlohmann::json& json) {
	if (skill.config.enabled)
		if (json.contains("skill_ratings"))
			skill.curRatings.ReadFromJSON(json["skill_ratings"]);
}
