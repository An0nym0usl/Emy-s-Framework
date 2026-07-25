#pragma once
#ifdef GIGA_USE_CUDA_SIM

#include <GigaLearnCPP/Export.h>

#include <RLGymCPP/Gamestates/GameState.h>
#include <RLGymCPP/ObsBuilders/ObsBuilder.h>
#include <RLGymCPP/ActionParsers/ActionParser.h>
#include <RLGymCPP/RewardCore/Reward.h>
#include <RLGymCPP/TerminalConditions/TerminalCondition.h>
#include <RLGymCPP/BasicTypes/Lists.h>

#include <RocketSimCuda.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <vector>

namespace GGL {

	// Per-arena environment pieces (the observation builder, weighted rewards and terminal
	// conditions). Used for hybrid CPU-bridge mode and for action parsers in GPU-native mode.
	struct CudaEnvCreateResult {
		RLGC::ObsBuilder* obsBuilder = nullptr;
		std::vector<RLGC::WeightedReward> rewards;
		std::vector<RLGC::TerminalCondition*> terminalConditions;
		RLGC::ActionParser* actionParser = nullptr;
		RLGC::ContinuousActionParser* continuousActionParser = nullptr;
	};
	typedef std::function<CudaEnvCreateResult(int arenaIdx)> CudaEnvCreateFn;

	// Hybrid OR GPU-native environment set.
	// GPU-native (default when AdvancedObs + DefaultAction): physics, AdvancedObs, masks,
	// rewards and terminals all on device — only final buffers copy to host.
	struct RG_IMEXPORT CudaEnvSet {
		int numArenas;
		int carsPerArena;
		int tickSkip;
		int numPlayers;
		int obsSize = 0;
		int actionCount = 0;
		bool gpuNative = true; // BuildAdvancedObs + GPU rewards/terminals (no per-arena CPU bridge)
		int rewardProfile = 0; // 0=chase SE, 1=default-approx
		float noTouchSeconds = 4.f;
		// When true, StepDiscrete skips host obs/mask/reward/terminal copies (device ptrs for Infer/bank).
		// Also disables per-kernel cudaDeviceSynchronize (chain + sync once before Infer).
		// Terminal arena resets are skipped (pure80 banks 1-step truncate on device anyway).
		bool skipHostObsCopy = false;

		rsc::RocketSimCudaBatch* batch = nullptr;

		std::vector<RLGC::ObsBuilder*> obsBuilders;
		std::vector<std::vector<RLGC::WeightedReward>> rewards;
		std::vector<std::vector<RLGC::TerminalCondition*>> terminalConditions;
		std::vector<RLGC::ActionParser*> actionParsers;
		std::vector<RLGC::ContinuousActionParser*> contActionParsers;

		std::vector<RLGC::GameState> gameStates;
		std::vector<RLGC::GameState> prevGameStates;

		std::vector<int> arenaPlayerStartIdx = {};

		RLGC::DimList2<float> obs;        // [numPlayers, obsSize]
		RLGC::DimList2<uint8_t> actionMasks;
		std::vector<float> rewards_out;   // [numPlayers]
		std::vector<std::vector<float>> lastRewards; // per-arena reward breakdown (hybrid only)
		std::vector<uint8_t> terminals;   // [numArenas] (RLGC::TerminalType)

		CudaEnvSet(
			int numArenas, int carsPerTeam, int tickSkip, CudaEnvCreateFn createFn,
			bool preferGpuNative = true, int rewardProfile = 0, float noTouchSeconds = 4.f);
		~CudaEnvSet();

		CudaEnvSet(const CudaEnvSet&) = delete;
		CudaEnvSet& operator=(const CudaEnvSet&) = delete;

		void StepContinuous(const RLGC::FList& actions, int actionDim);
		void StepDiscrete(const std::vector<int>& actions);
		// GPU-native: expand int32 action indices already on device (no H2D / host vector).
		void StepDiscreteDevice(const int* deviceActionIndices);

		void Reset();
		void ResetAll();
		// Enable chained kernels (no per-op device sync). Call after setting skipHostObsCopy.
		void EnableThroughputSyncMode(bool enable);

		// Hot-reload GPU reward base×RuntimeRewardRegistry multipliers + no-touch terminal.
		// No-op when !gpuNative or batch is null. Safe to call every iteration.
		void ApplyRuntimeGpuRewards();
		void SetNoTouchSeconds(float seconds);

		// --- Error savestate / replay (−1.5s style) ---
		// Host ring-buffer of car/ball/pad states via RocketSimCuda Get/Set APIs.
		// Restores arena physics only (not full RL traj / GAE); pads restored
		// when SetBoostPadStates is available; lookback is discrete env-step slots.
		struct StateRingConfig {
			bool enabled = false;
			int depth = 10;           // slots (~1.5s with captureEvery=3, tickSkip=8)
			int captureEvery = 3;     // env steps between captures (amortize D2H)
			float restoreFraction = 0.05f; // fraction of arenas to rewind on trigger
			int maxArenasPerRestore = 64;  // hard cap (also ≤5% of batch)
			float fuzzPos = 25.f;     // UU position jitter on restore
			float fuzzVel = 80.f;     // UU/s velocity jitter
			int lookbackSlots = 7;    // ~1.4s at captureEvery=3, tickSkip=8
		};

		void ConfigureStateRing(const StateRingConfig& cfg);
		bool StateRingEnabled() const { return stateRing.enabled; }
		bool StateRingReplayDisabled() const { return stateRing.replayDisabled; }
		// While Collect||Learn overlap: skip ring D2H + cudaDeviceSynchronize (deadlocks PyTorch Learn).
		void SetSuppressStateRingCapture(bool suppress) {
			suppressStateRingCapture.store(suppress, std::memory_order_release);
		}
		bool StateRingCaptureSuppressed() const {
			return suppressStateRingCapture.load(std::memory_order_acquire);
		}
		// Call after each env Step (amortized internally). Cheap no-op when disabled.
		void MaybeCaptureStateRing();
		// Rewind a fraction of arenas to lookback slot + fuzz; returns arenas restored.
		// Optional maxArenas overrides cfg.maxArenasPerRestore when > 0.
		int RestoreStateRingReplay(float restoreFraction = -1.f, int lookbackSlots = -1, float fuzzScale = 1.f, int maxArenas = -1);
		// Convenience: enable + restore in one call (AutoTrainer error_pressure path).
		int TriggerErrorReplay(float restoreFraction = -1.f, int lookbackSlots = -1, float fuzzScale = 1.f, int maxArenas = -1);
		uint64_t StateRingCaptures() const { return stateRing.captures; }
		uint64_t StateRingRestores() const { return stateRing.restores; }
		uint64_t StateRingRestoreCalls() const { return stateRing.restoreCalls; }
		void SetStateRingReplayDisabled(bool disabled) { stateRing.replayDisabled = disabled; }

	private:
		void BridgeArena(int arenaIdx, const std::vector<RLGC::Action>& arenaActions, bool usePrev);
		void BuildArenaObs(int arenaIdx);
		void BuildArenaActionMasks(int arenaIdx);
		void ResetArenaEnv(int arenaIdx);
		void ConfigureGpuTrainingRewards();
		void CopyGpuTrainingBuffers();
		void ResetTerminalArenasGpu();
		void BuildDiscreteActionLut();
		void CaptureStateRingSlot();
		void ApplyArenaSnap(int arenaIdx, int slot, float fuzzScale);

		std::vector<rsc::CarControls> discreteActionLut;
		std::vector<rsc::CarControls> controlsBuf;

		struct ArenaSnap {
			std::vector<rsc::CarState> cars;
			rsc::BallState ball = {};
			std::vector<rsc::BoostPadState> pads;
			bool valid = false;
		};
		struct StateRing {
			StateRingConfig cfg;
			bool enabled = false;
			bool replayDisabled = false; // tripwire when restores explode
			int writeIdx = 0;
			int filled = 0;
			int stepsSinceCapture = 0;
			uint64_t captures = 0;
			uint64_t restores = 0;       // arenas restored (sum)
			uint64_t restoreCalls = 0;   // Trigger/Restore invocations
			uint32_t rng = 0xA5A5A5A5u;
			// slots[slot][arena]
			std::vector<std::vector<ArenaSnap>> slots;
			// Staging for bulk GetAll*
			std::vector<rsc::CarState> stagingCars;
			std::vector<rsc::BallState> stagingBalls;
			std::vector<rsc::BoostPadState> stagingPads;
		} stateRing;
		std::atomic<bool> suppressStateRingCapture{ false };
	};
}

#endif // GIGA_USE_CUDA_SIM
