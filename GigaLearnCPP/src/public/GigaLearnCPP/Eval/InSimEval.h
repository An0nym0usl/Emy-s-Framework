#pragma once

#include <GigaLearnCPP/Export.h>
#include <GigaLearnCPP/Util/Report.h>

#include <cstdint>
#include <map>
#include <string>

namespace GGL {

	// Results from a short in-sim arena eval (current policy vs old-version / fixed opponent).
	// Not a live Rocket League client eval — RocketSim arenas only.
	struct RG_IMEXPORT InSimEvalResult {
		// Primary Elo-like rating after the match block (mode key → rating).
		std::map<std::string, float> ratings;
		float primaryElo = 0.f;
		float eloDelta = 0.f;

		int goalsFor = 0;
		int goalsAgainst = 0;
		float winRate = 0.f; // goalsFor / (goalsFor+goalsAgainst), 0.5 if none
		float avgStepReward = 0.f;
		int rewardSamples = 0;
		float simSeconds = 0.f;
		int opponentVersionTs = 0; // timesteps of old-version opponent (0 if none)

		void WriteToReport(Report& report, const std::string& prefix = "Eval") const;
	};

	// Interval gating for in-sim eval so train SPS stays healthy.
	struct InSimEvalGate {
		// Iteration-based (SkillTracker classic). 0 = ignore.
		int updateInterval = 16;
		int iterationsSinceRan = 0;

		// Timestep-based (preferred for AutoTrainer). 0 = ignore.
		// Example: 5'000'000 → every ~5M steps.
		int64_t tsPerEval = 5'000'000;
		int64_t lastEvalTimesteps = 0;

		bool Due(int64_t totalTimesteps) const {
			bool byIter = updateInterval > 0 && iterationsSinceRan >= updateInterval;
			bool byTs = tsPerEval > 0
				&& (lastEvalTimesteps == 0 || (totalTimesteps - lastEvalTimesteps) >= tsPerEval);
			// First call after enable: wait for at least one gate to fire (avoid boot spam).
			if (lastEvalTimesteps == 0 && iterationsSinceRan == 0)
				return byIter || (tsPerEval > 0 && totalTimesteps >= tsPerEval);
			return byIter || byTs;
		}

		void MarkRan(int64_t totalTimesteps) {
			iterationsSinceRan = 0;
			lastEvalTimesteps = totalTimesteps;
		}

		void OnIteration() { iterationsSinceRan++; }
	};

}
