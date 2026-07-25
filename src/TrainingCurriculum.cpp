#include "TrainingCurriculum.h"

#include <RLGymCPP/EnvSet/EnvSet.h>

namespace GGL {

	TrainingCurriculumConfig TrainingCurriculum::config = {};
	TrainingPhase TrainingCurriculum::currentPhase = TrainingPhase::CHASE;
	RLGC::EnvCreateFn TrainingCurriculum::chaseEnvCreateFn = nullptr;
	RLGC::EnvCreateFn TrainingCurriculum::advancedEnvCreateFn = nullptr;

	namespace {
		int _lastPlayersPerTeam = 1;
	}

	TrainingPhase TrainingCurriculum::GetPhase(int64_t totalTimesteps) {
		if (totalTimesteps < config.chaseEndSteps)
			return TrainingPhase::CHASE;
		if (totalTimesteps < config.foundationEndSteps)
			return TrainingPhase::FOUNDATION;
		return TrainingPhase::ADVANCED;
	}

	int TrainingCurriculum::PlayersPerTeamForPhase(TrainingPhase phase) {
		return phase == TrainingPhase::ADVANCED ? 2 : 1;
	}

	static void ApplyPhaseHyperparams(Learner* learner, TrainingPhase phase) {
		auto& ppo = learner->config.ppo;

		switch (phase) {
		case TrainingPhase::CHASE:
			ppo.epochs = 1;
			ppo.gaeGamma = 0.99f;
			ppo.entropyScale = 0.015f;
			ppo.varMax = 0.35f;
			ppo.miniBatchSize = TrainingCurriculum::config.ppoMiniBatchSize;
			learner->config.savePolicyVersions = false;
			learner->config.opponentPool.enabled = (TrainingCurriculum::config.chaseOpponentChance > 0);
			learner->config.opponentPool.chance = TrainingCurriculum::config.chaseOpponentChance;
			learner->config.trainAgainstOldVersions = (TrainingCurriculum::config.chaseOldVersionChance > 0);
			learner->config.trainAgainstOldChance = TrainingCurriculum::config.chaseOldVersionChance;
			learner->config.skillTracker.enabled = TrainingCurriculum::config.chaseSkillTracker;
			break;
		case TrainingPhase::FOUNDATION:
			ppo.epochs = 1;
			ppo.gaeGamma = 0.992f;
			ppo.entropyScale = 0.020f;
			ppo.varMax = 0.65f;
			ppo.miniBatchSize = TrainingCurriculum::config.ppoMiniBatchSize;
			learner->config.savePolicyVersions = false;
			learner->config.opponentPool.enabled = true;
			learner->config.opponentPool.chance = TrainingCurriculum::config.foundationOpponentChance;
			learner->config.trainAgainstOldVersions = true;
			learner->config.trainAgainstOldChance = TrainingCurriculum::config.foundationOldVersionChance;
			learner->config.skillTracker.enabled = TrainingCurriculum::config.foundationSkillTracker;
			break;
		case TrainingPhase::ADVANCED:
			ppo.epochs = 2;
			ppo.gaeGamma = 0.993f;
			ppo.entropyScale = 0.025f;
			ppo.varMax = 1.0f;
			ppo.miniBatchSize = TrainingCurriculum::config.ppoMiniBatchSize;
			learner->config.savePolicyVersions = true;
			learner->config.opponentPool.enabled = true;
			learner->config.opponentPool.chance = TrainingCurriculum::config.advancedOpponentChance;
			learner->config.trainAgainstOldVersions = true;
			learner->config.trainAgainstOldChance = TrainingCurriculum::config.advancedOldVersionChance;
			learner->config.skillTracker.enabled = TrainingCurriculum::config.advancedSkillTracker;
			break;
		}

		if (ppo.miniBatchSize <= 0)
			ppo.miniBatchSize = ppo.batchSize;
		else if (ppo.miniBatchSize > ppo.batchSize)
			ppo.miniBatchSize = ppo.batchSize;
	}

	void TrainingCurriculum::OnIterationStart(Learner* learner) {
		TrainingPhase newPhase = GetPhase((int64_t)learner->totalTimesteps);
		int playersPerTeam = PlayersPerTeamForPhase(newPhase);

		bool phaseChanged = (newPhase != currentPhase);
		bool teamSizeChanged = (playersPerTeam != _lastPlayersPerTeam);

		if (phaseChanged || teamSizeChanged || learner->totalTimesteps == 0) {
			RG_LOG("TrainingCurriculum: phase="
				<< (int)newPhase << " (timesteps=" << learner->totalTimesteps
				<< "), playersPerTeam=" << playersPerTeam);

			ApplyPhaseHyperparams(learner, newPhase);
			learner->SyncRuntimePPOConfig();

			// Recreate env when phase or team size changes (not on first tick — Learner already created matching env).
			if (chaseEnvCreateFn && (phaseChanged || teamSizeChanged))
				learner->RecreateEnvSet(chaseEnvCreateFn);

			currentPhase = newPhase;
			_lastPlayersPerTeam = playersPerTeam;
		}
	}

}
