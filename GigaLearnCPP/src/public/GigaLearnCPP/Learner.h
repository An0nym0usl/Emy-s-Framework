#pragma once

#include <RLGymCPP/EnvSet/EnvSet.h>
#include "Util/MetricSender.h"
#include "Util/RenderSender.h"
#include "LearnerConfig.h"
#include "PPO/TransferLearnConfig.h"

#include <atomic>

namespace GGL {

	typedef std::function<void(class Learner*, const std::vector<RLGC::GameState>& states, Report& report)> StepCallbackFn;

	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/learner.py
	class RG_IMEXPORT Learner {
	public:
		LearnerConfig config;

		RLGC::EnvSet* envSet;

		RLGC::EnvSetConfig storedEnvSetConfig = {};

		class PPOLearner* ppo;
		struct PolicyVersionManager* versionMgr;
		class OpponentPool* opponentPool = nullptr;

		RLGC::EnvCreateFn envCreateFn;
		MetricSender* metricSender;
		RenderSender* renderSender;

		int obsSize;
		int numActions;

		struct WelfordStat* returnStat;
		struct BatchedWelfordStat* obsStat;

		std::string runID = {};

		uint64_t
			totalTimesteps = 0,
			totalIterations = 0;

		StepCallbackFn stepCallback = NULL;

		// Set true by RecreateEnvSet(); cleared by Start() after resizing trajectory buffers.
		bool envRecreatedThisIteration = false;

#ifdef GIGA_USE_CUDA_SIM
		class CudaEnvSet* cudaEnvSet = nullptr;
#endif

		// Collect||Learn: true while async PPO Learn is in flight on the CUDA device.
		// AutoTrainer must NOT cudaMemcpyToSymbol / Adam SetLR / Save / state-ring D2H until this clears
		// (otherwise train hangs — seen after iter 2 on first reward apply, and again ~iter 13).
		std::atomic<bool> asyncLearnInFlight{ false };
		// Set by AutoTrainerBridge when a CUDA/optimizer surface apply was deferred; flushed after join.
		std::atomic<bool> deferredCudaSurfaceApply{ false };
		// Checkpoint Save() requested while Learn owned the GPU — flushed after join.
		std::atomic<bool> deferredCheckpointSave{ false };
		// Mid-run OpponentPool::Load (Nexto JIT) / PolicyVersionManager create — NEVER while
		// asyncLearnInFlight or during Collect||Learn overlap; flushed after Learn join.
		std::atomic<bool> deferredOpponentPoolLoad{ false };
		std::atomic<bool> deferredVersionMgrCreate{ false };

		// True if this Learner initialized the embedded Python interpreter (only when metrics/render are enabled).
		// Used to avoid finalizing an interpreter we never started.
		bool _pythonInitialized = false;

		Learner(RLGC::EnvCreateFn envCreateFunc, LearnerConfig config, StepCallbackFn stepCallback = NULL);
		void Start();

		// Rebuild all arenas (e.g. 1v1 -> 2v2 curriculum transition). Obs size must stay compatible.
		void RecreateEnvSet(RLGC::EnvCreateFn newCreateFn);

		// Push LearnerConfig.ppo changes (curriculum) into the live PPOLearner.
		void SyncRuntimePPOConfig();

		// Mid-run: queue PolicyVersionManager + OpponentPool when AutoTrainer enables
		// skill-eval / league / opponents after a from-scratch OFF boot.
		// Heavy create/Load is NEVER done here while Collect||Learn owns the GPU —
		// call FlushDeferredRuntimeSubsystems() after Learn join (FlushDeferredCuda).
		void EnsureRuntimeSubsystems();
		// GPU-idle window only: perform deferred OpponentPool::Load / PolicyVersion create.
		void FlushDeferredRuntimeSubsystems();

		// Hot-reload Adam LRs (AutoTrainer / OP stack).
		void SetLearningRates(float policyLR, float criticLR);

		// Hot-reload entropy (Apex curriculum).
		void SetEntropyScale(float entropyScale);

		// Hot-reload PPO clip / grad clip / guiding strength (AutoTrainer hooks).
		void SetClipRange(float clipRange);
		void SetMaxGradNorm(float maxGradNorm);
		void SetGuidingStrength(float strength);
		void SetTargetKl(float targetKl);
		void SetTargetEntropy(float targetEntropy);

		// Hot-set OpponentPool entry weight by name (Apex curriculum).
		void SetOpponentWeight(const std::string& name, float weight);

		void StartTransferLearn(const TransferLearnConfig& transferLearnConfig);

		void Save();
		void Load();
		void SaveStats(std::filesystem::path path);
		void LoadStats(std::filesystem::path path);

		// Competitive PBT: save/load live policy weights to an arbitrary folder (agent slots).
		void SavePolicyTo(std::filesystem::path folder);
		void LoadPolicyFrom(std::filesystem::path folder);

		// Mid-run soft TransferLearn / BC distill via PPO guiding loss (same-arch only).
		// Returns false if teacher path incompatible — caller should keep sparring pressure.
		bool SetSoftDistillTeacher(std::filesystem::path teacherFolder, float strength);
		void ClearSoftDistillTeacher();
		bool HasSoftDistillTeacher() const;

#ifdef GIGA_USE_CUDA_SIM
		// Error savestate replay (−1.5s ring). Returns arenas restored (0 if ring empty/disabled).
		int TriggerErrorStateReplay(float restoreFraction = -1.f, int lookbackSlots = -1, float fuzzScale = 1.f, int maxArenas = -1);
		void ConfigureErrorStateRing(bool enable, int depth = 10, int captureEvery = 3, float restoreFraction = 0.05f, int maxArenas = 64);

		bool AsyncLearnInFlight() const { return asyncLearnInFlight.load(std::memory_order_acquire); }
		void RequestDeferredCudaSurfaceApply() { deferredCudaSurfaceApply.store(true, std::memory_order_release); }
		bool ConsumeDeferredCudaSurfaceApply() {
			return deferredCudaSurfaceApply.exchange(false, std::memory_order_acq_rel);
		}
#endif

		RG_NO_COPY(Learner);

		~Learner();
	};
}
