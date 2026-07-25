#pragma once

#include <GigaLearnCPP/Learner.h>
#include <RLGymCPP/RewardCore/Reward.h>

namespace GGL {

	enum class TrainingPhase : int {
		CHASE = 0,       // 0 – 300M: chase ball, 1v1, dense shaping
		FOUNDATION = 1,  // 300M – 2B: kickoff, boost, scoring basics
		ADVANCED = 2       // 2B+: 2v2, sparring, fuller reward stack
	};

	struct TrainingCurriculumConfig {
		int64_t chaseEndSteps = 300'000'000;
		int64_t foundationEndSteps = 2'000'000'000;

		// Sparring (overrides LearnerConfig when phase allows)
		float chaseOpponentChance = 0.f;
		float chaseOldVersionChance = 0.f;
		float foundationOpponentChance = 0.08f;
		float foundationOldVersionChance = 0.10f;
		float advancedOpponentChance = 0.15f;
		float advancedOldVersionChance = 0.20f;

		bool chaseSkillTracker = false;
		bool foundationSkillTracker = false;
		bool advancedSkillTracker = true;

		// PPO learn chunk size. 0 = full batch (needs 24GB+ with attention head).
		int ppoMiniBatchSize = 50'000;

		// LEVEL 4 env architect (POET/PLR) — hot-applied from runtime_overrides.
		float ballChaseWeight = 0.85f;
		float randomStateWeight = 0.10f;
		float kickoffWeight = 0.05f;
		// SSL guide §3 — GPU reset curriculum (kickoff / fuzzed / aerial).
		float fuzzedWeight = 0.40f;
		float aerialWeight = 0.10f;
		float noTouchSecondsChase = 4.f;
		float noTouchSecondsAdvanced = 8.f;
		float wIcm = 0.05f;
		float wRnd = 0.03f;
		// When true, ExampleMain re-applies AutoTrainer league/scenario after Apex.
		bool sslGuidePostApex = false;
	};

	class TrainingCurriculum {
	public:
		static TrainingCurriculumConfig config;
		static TrainingPhase currentPhase;

		static TrainingPhase GetPhase(int64_t totalTimesteps);
		static int PlayersPerTeamForPhase(TrainingPhase phase);
		static void OnIterationStart(Learner* learner);

		static RLGC::EnvCreateFn chaseEnvCreateFn;
		static RLGC::EnvCreateFn advancedEnvCreateFn;
	};

	// Reward wrapper: active only when current training phase >= minPhase.
	class CurriculumGateReward : public RLGC::Reward {
	public:
		RLGC::Reward* inner;
		TrainingPhase minPhase;

		CurriculumGateReward(RLGC::Reward* inner, TrainingPhase minPhase)
			: inner(inner), minPhase(minPhase) {}

		~CurriculumGateReward() { delete inner; }

		virtual void Reset(const RLGC::GameState& initialState) override {
			inner->Reset(initialState);
		}

		virtual void PreStep(const RLGC::GameState& state) override {
			if ((int)TrainingCurriculum::currentPhase >= (int)minPhase)
				inner->PreStep(state);
		}

		virtual float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
			if ((int)TrainingCurriculum::currentPhase < (int)minPhase)
				return 0.f;
			return inner->GetReward(player, state, isFinal);
		}
	};

}
