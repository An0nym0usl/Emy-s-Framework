#pragma once

#include "Framework.h"

namespace GGL {
	struct SkillTrackerConfig {
		bool enabled = false;

		// Number of arenas for evaluation
		// Don't put this much higher than your CPU thread count
		int numArenas = 16;

		// Time (in seconds) to simulate each skill rating run, per arena
		float simTime = 45;

		// Maximum time (in seconds) to simulate games for (games are reset after the maximum time is exceeded)
		// If this max time is too low, the rating quality may be poor because the bots wont have enough time to score goals
		float maxSimTime = 240; 

		// Number of iterations between running skill rating games (0 = disable iter gate)
		int updateInterval = 16;

		// Timestep gate for in-sim eval (0 = disable). Default 5M — keeps train SPS healthy.
		// Eval runs when EITHER updateInterval OR tsPerEval fires (and old versions exist).
		int64_t tsPerEval = 5'000'000;

		// Rating increment scale per-goal
		float ratingInc = 5; 

		// Initial rating of the first version
		float initialRating = 0; 

		// Policies will be inferred deterministically
		// Off by default since the learning algorithm is trying to optimize the stochastic policy
		bool deterministic = false;

		// Optional 1v1 skill evaluation (logs Rating/1v1 alongside Rating/2v2 on wandb)
		bool evaluate1v1 = false;
		int numArenas1v1 = 8;
		RLGC::EnvCreateFn envCreateFn1v1 = nullptr;

		// Also write Eval/* metrics (Elo, WinRate, AvgReward) for AutoTrainer / best_skill.
		bool writeEvalMetrics = true;
	};
}
