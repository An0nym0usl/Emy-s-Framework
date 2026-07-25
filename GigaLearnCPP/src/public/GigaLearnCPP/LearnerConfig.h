#pragma once
#include <RLGymCPP/BasicTypes/Lists.h>
#include "PPO/PPOLearnerConfig.h"
#include "SkillTrackerConfig.h"
#include "Util/Report.h"
#include <functional>

namespace GGL {
	// C++ PPO uses stock libtorch. Backends available to this Learner:
	//   AUTO / GPU_CUDA → CUDA (or ROCm libtorch that exposes torch.cuda)
	//   CPU             → multi-threaded CPU (OpenMP / MKL via torch::set_num_threads)
	// DirectML (torch-directml / PrivateUse1) is Python-only — Microsoft does not ship a
	// libtorch+DirectML distribution. GIGA_TORCH_DIRECTML=1 is detected and logged, then
	// falls back to CPU. See docs/DIRECTML.md and docs/AMD_WSL2.md.
	enum class LearnerDeviceType {
		AUTO,
		CPU,
		GPU_CUDA
	};

	// External sparring bots (Nexto JIT, GigaLearn checkpoints in opponents/).
	struct OpponentPoolConfig {
		bool enabled = false;
		float chance = 0.15f;
		// ContinuousV2-parity: dense SE on top of GoalReward when beating / conceding vs externals.
		float beatBonus = 100.f;
		float concedePenalty = -40.f;
		std::filesystem::path folder = "opponents";
		std::filesystem::path manifest = "opponents/opponents.json";
		// Suppress load banners. Errors still print.
		bool quiet = false;
	};

	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/learner.py
	struct LearnerConfig {
		int numGames = 300;

		int tickSkip = 8;
		int actionDelay = 7;

		bool renderMode = false;
		// If renderMode, this is the scaling of time for the game
		// 1.0 = Run the game at real time
		// 2.0 = Run the game twice as fast as real time
		float renderTimeScale = 1.0f; 

		PPOLearnerConfig ppo = {};

		// Checkpoints are saved here as timestep-numbered subfolders
		//	e.g. a checkpoint at 20,000 steps will save to a subfolder called "20000"
		// Set empty to disable saving
		std::filesystem::path checkpointFolder = "checkpoints"; 

		// Save every timestep
		// Set to zero to just use timestepsPerIteration
		int64_t tsPerSave = 1'000'000;

		int64_t randomSeed = -1; // Set to -1 to use the current time
		int checkpointsToKeep = 8; // Checkpoint storage limit before old checkpoints are deleted, set to -1 to disable
		LearnerDeviceType deviceType = LearnerDeviceType::AUTO; // Auto will use your CUDA GPU if available

		// Standardize the obs values (doesn't seem to help much from my testing)
		bool standardizeObs = false;
		float minObsSTD = 1 / 10.f;
		float maxObsMeanRange = 3;
		int maxObsSamples = 100;

		// Standardize the returns to help the critic (don't disable this unless you know what you're doing)
		bool standardizeReturns = true;
		int maxReturnSamples = 150;
		// When standardizeReturns is on, also divide advantages by running return STD after GAE
		// (complements PPO batch advantageNormMode). Off by default — batch norm is usually enough.
		bool standardizeAdvantages = false;

		// Skip per-step NaN/inf scan of the full obs buffer (pure throughput mode).
		bool skipObsIntegrityChecks = false;

		// Will automatically add the rewards to metrics
		bool addRewardsToMetrics = true;
		int maxRewardSamples = 50; // Maximum reward samples per step for reward metrics
		int rewardSampleRandInterval = 8; // Randomized interval range between sampling rewards (per step)

		// Send metrics to the python metrics receiver
		// The receiver can then log them to wandb or whatever
		bool sendMetrics = true;
		std::string metricsProjectName = "gigalearncpp"; // Project name for the python metrics receiver
		std::string metricsGroupName = "unnamed-runs"; // Group name for the python metrics receiver
		std::string metricsRunName = "gigalearncpp-run"; // Run name for the python metrics receiver

		bool savePolicyVersions = false;
		int64_t tsPerVersion = 25'000'000;
		int maxOldVersions = 32;

		bool trainAgainstOldVersions = false;
		float trainAgainstOldChance = 0.15f; // Chance (from 0 - 1) that an iteration will train against an old version

		SkillTrackerConfig skillTracker = {};

		OpponentPoolConfig opponentPool = {};

		// Called at the start of each training iteration (curriculum phase transitions, etc.).
		std::function<void(class Learner*)> onIterationStart = nullptr;

		// Called after Collect||Learn join (GPU free). Use for AutoTrainer cudaMemcpyToSymbol
		// reward/terminal pushes deferred while async Learn was in flight.
		std::function<void(class Learner*)> onAfterAsyncLearnJoin = nullptr;

		// Called after metrics are finalized for the iteration (AutoTrainer status export, etc.).
		std::function<void(class Learner*, const Report&)> onIterationComplete = nullptr;

#ifdef GIGA_USE_CUDA_SIM
		// GPU physics via RocketSimCuda.
		bool useCudaSim = false;
		// true: AdvancedObs + rewards/terminals on device (fast). false: hybrid CPU rewards.
		bool cudaPreferGpuNative = true;
		// 0 = dense chase SE stack; 1 = default-approx (mapped GPU reward IDs).
		int cudaRewardProfile = 0;
		float cudaNoTouchSeconds = 4.f;
		// Skip D2H of AdvancedObs/masks every step; bank experience on CUDA tensors.
		// Requires gpuNative + bulk 1-step truncate (pure80) or short multi-step with device bank.
		bool cudaDeviceExperience = false;

		// Collect∥Learn overlap mode:
		//  0 = off (two-phase; safest default for large arena counts)
		//  1 = on  (requires cudaDeviceExperience)
		//  2 = auto — enable when device XP + CUDA + numGames <= asyncOverlapMaxArenas
		// Safe by construction: Collect Infer uses a frozen shared_head+policy clone; Learn updates
		// fp32 masters on a double-buffered device XP bank. Join Learn before critic/GAE.
		// Env GIGA_ASYNC_OVERLAP=0|1|auto overrides at Learner::Start resolve time.
		int asyncLearnOverlapMode = 2; // auto
		int asyncOverlapMaxArenas = 4096;
		// Resolved bool (set by Learner::Start). Prefer reading this after Start().
		bool asyncLearnOverlap = false;
#endif

		// Optional Collect/Learn JSON profile dump (also GIGA_PROFILE=1 / GIGA_PROFILE_PATH=...).
		bool profileIters = false;
		std::string profileDumpPath = "profile_iters.jsonl";
		int profileDumpEvery = 1;
	};
}