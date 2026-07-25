#ifdef GIGA_USE_CUDA_SIM

#include "CudaEnvSet.h"

#include <RocketSimCuda.h>
#include <RLGymCPP/RewardCore/RuntimeRewardRegistry.h>
#include <RLGymCPP/Framework.h>

#include <cmath>
#include <cstring>
#include <algorithm>
#include <utility>

using namespace RLGC;

namespace {
	inline rsc::CarControls ToControls(const Action& a) {
		rsc::CarControls c;
		c.throttle = a.throttle;
		c.steer = a.steer;
		c.pitch = a.pitch;
		c.yaw = a.yaw;
		c.roll = a.roll;
		c.jump = a.jump > 0.5f;
		c.boost = a.boost > 0.5f;
		c.handbrake = a.handbrake > 0.5f;
		return c;
	}
}

namespace {
	// Map GPU TrainingRewardID → AutoTrainer / CPU reward class names (aliases accepted in registry).
	const char* GpuRewardRegistryName(rsc::TrainingRewardID id) {
		using ID = rsc::TrainingRewardID;
		switch (id) {
		case ID::GOAL_REWARD: return "GoalReward";
		case ID::VELOCITY_BALL_TO_GOAL: return "VelocityBallToGoalReward";
		case ID::VELOCITY_PLAYER_TO_BALL: return "VelocityPlayerToBallReward";
		case ID::FACE_BALL: return "FaceBallReward";
		case ID::TOUCH_BALL: return "TouchBallReward";
		case ID::TOUCH_ACCEL: return "TouchAccelReward";
		case ID::STRONG_TOUCH: return "StrongTouchReward";
		case ID::PICKUP_BOOST: return "PickupBoostReward";
		case ID::SAVE_BOOST: return "SaveBoostReward";
		case ID::KICKOFF_PROXIMITY: return "KickoffProximity";
		case ID::PRESSURE_FLICK: return "PressureFlick";
		case ID::MAWKZY_FLICK: return "MawkzyFlick";
		case ID::AIR_CARRY: return "AirReward";
		case ID::ANTI_BALL_STACK: return "AntiBallStack";
		case ID::AIR: return "AirReward";
		case ID::WAVEDASH: return "WavedashReward";
		case ID::DEMO: return "Demo";
		default: return nullptr;
		}
	}

	float GpuRewardMult(rsc::TrainingRewardID id) {
		const char* name = GpuRewardRegistryName(id);
		if (!name)
			return 1.f;
		// GetMultiplier resolves AirReward/BoostPickup/etc. aliases → canonical keys.
		return RLGC::RuntimeRewardRegistry::Instance().GetMultiplier(name);
	}
}

void GGL::CudaEnvSet::ConfigureGpuTrainingRewards() {
	if (!batch)
		return;
	rsc::TrainingRewardConfig rc = {};
	auto add = [&](rsc::TrainingRewardID id, float w, bool zeroSum = false) {
		if (rc.numEntries >= rsc::MAX_TRAINING_REWARD_ENTRIES)
			return;
		// Live AutoTrainer: base profile weight × RuntimeRewardRegistry multiplier.
		// Clamp NaN/Inf and extreme remaps so GPU reward kernels never see garbage weights.
		float scaled = w * GpuRewardMult(id);
		if (!std::isfinite(scaled))
			scaled = w;
		scaled = (std::max)(-1e6f, (std::min)(1e6f, scaled));
		rc.entries[rc.numEntries].id = id;
		rc.entries[rc.numEntries].weight = scaled;
		rc.entries[rc.numEntries].isZeroSum = zeroSum ? 1 : 0;
		rc.numEntries++;
	};

	if (rewardProfile == 1) {
		// Default from-scratch GPU stack — blank default template.
		// Add your own IDs/weights here (or via AutoTrainer reward_weights).
		// Optional community reward IDs exist in RocketSimCuda but are NOT defaults.
		add(rsc::TrainingRewardID::GOAL_REWARD, 100.f);
		add(rsc::TrainingRewardID::TOUCH_BALL, 5.f, true);
		add(rsc::TrainingRewardID::VELOCITY_BALL_TO_GOAL, 10.f, true);
		add(rsc::TrainingRewardID::VELOCITY_PLAYER_TO_BALL, 2.f);
	} else {
		// Dense chase stack (GPU-native IDs) — SE weapon vs Leak Goal=400.
		add(rsc::TrainingRewardID::VELOCITY_PLAYER_TO_BALL, 10.f);
		add(rsc::TrainingRewardID::FACE_BALL, 3.f);
		add(rsc::TrainingRewardID::TOUCH_BALL, 18.f);
		add(rsc::TrainingRewardID::VELOCITY_BALL_TO_GOAL, 2.f);
		add(rsc::TrainingRewardID::GOAL_REWARD, 25.f);
	}
	batch->ConfigureTrainingRewards(rc);

	rsc::TrainingTerminalConfig tc = {};
	tc.useGoalScore = 1;
	tc.noTouchTimeoutSeconds = noTouchSeconds;
	batch->ConfigureTrainingTerminals(tc);
}

void GGL::CudaEnvSet::ApplyRuntimeGpuRewards() {
	if (!gpuNative || !batch)
		return;
	ConfigureGpuTrainingRewards();
}

void GGL::CudaEnvSet::SetNoTouchSeconds(float seconds) {
	if (seconds <= 0.f)
		return;
	noTouchSeconds = seconds;
	if (!gpuNative || !batch)
		return;
	rsc::TrainingTerminalConfig tc = {};
	tc.useGoalScore = 1;
	tc.noTouchTimeoutSeconds = noTouchSeconds;
	batch->ConfigureTrainingTerminals(tc);
}

void GGL::CudaEnvSet::CopyGpuTrainingBuffers() {
	if (skipHostObsCopy) {
		// Device-XP: no host mirror, but must fence chained kernels before Infer/bank reads.
		if (!batch->GetKernelSync())
			batch->Synchronize();
		return;
	}
	// Kernels were chained without per-op sync — fence once before D2H.
	if (!batch->GetKernelSync())
		batch->Synchronize();
	batch->CopyBuiltAdvancedObs(obs.data.data());
	batch->CopyBuiltDefaultActionMasks(actionMasks.data.data());
	batch->CopyBuiltRewards(rewards_out.data());
	batch->CopyBuiltTerminals(terminals.data());
	batch->Synchronize();
}

void GGL::CudaEnvSet::EnableThroughputSyncMode(bool enable) {
	if (!batch)
		return;
	// Device-XP: skip host obs/mask/reward/terminal mirrors + chain Step/Obs/Rewards
	// (single fence in CopyGpuTrainingBuffers). Disable chain with GIGA_KERNEL_CHAIN=0.
	bool chain = true;
	if (const char* env = std::getenv("GIGA_KERNEL_CHAIN"))
		chain = std::atoi(env) != 0;
	batch->SetKernelSync(!(enable && chain));
	skipHostObsCopy = enable;
}

void GGL::CudaEnvSet::ResetTerminalArenasGpu() {
	bool any = false;
	for (int a = 0; a < numArenas; a++) {
		if (terminals[a] == (uint8_t)NOT_TERMINAL)
			continue;
		batch->ResetArena(a);
		terminals[a] = (uint8_t)NOT_TERMINAL;
		any = true;
	}
	if (!any)
		return;
	// Rebuild obs/masks for freshly reset arenas (full batch rebuild is cheapest).
	batch->BuildAdvancedObsAndDefaultMasks();
	if (!skipHostObsCopy) {
		batch->CopyBuiltAdvancedObs(obs.data.data());
		batch->CopyBuiltDefaultActionMasks(actionMasks.data.data());
		batch->Synchronize();
	}
}

GGL::CudaEnvSet::CudaEnvSet(
	int numArenas, int carsPerTeam, int tickSkip, CudaEnvCreateFn createFn,
	bool preferGpuNative, int rewardProfileIn, float noTouchSecondsIn)
	: numArenas(numArenas), carsPerArena(carsPerTeam * 2), tickSkip(tickSkip),
	  gpuNative(preferGpuNative), rewardProfile(rewardProfileIn), noTouchSeconds(noTouchSecondsIn) {

	numPlayers = numArenas * carsPerArena;

	batch = new rsc::RocketSimCudaBatch();
	rsc::BatchConfig cfg = {};
	cfg.numArenas = numArenas;
	cfg.maxCarsPerArena = carsPerArena;
	cfg.tickRate = 120.f;
	cfg.gameMode = rsc::GameMode::SOCCAR;
	cfg.obsMode = rsc::ObsMode::ADVANCED;
	batch->Init(cfg);

	for (int a = 0; a < numArenas; a++) {
		for (int t = 0; t < carsPerTeam; t++) {
			batch->AddCar(a, rsc::Team::BLUE, rsc::CarPreset::OCTANE);
			batch->AddCar(a, rsc::Team::ORANGE, rsc::CarPreset::OCTANE);
		}
	}
	batch->ResetAllArenas();

	obsBuilders.resize(numArenas, nullptr);
	rewards.resize(numArenas);
	terminalConditions.resize(numArenas);
	actionParsers.resize(numArenas, nullptr);
	contActionParsers.resize(numArenas, nullptr);
	gameStates.resize(numArenas);
	prevGameStates.resize(numArenas);
	arenaPlayerStartIdx.resize(numArenas);
	for (int a = 0; a < numArenas; a++)
		arenaPlayerStartIdx[a] = a * carsPerArena;

	if (gpuNative) {
		// ONE createFn call — do not instantiate thousands of CPU reward/obs stacks.
		// DefaultAction::ParseAction ignores GameState; GPU owns obs/rewards/terminals.
		CudaEnvCreateResult res = createFn(0);
		for (auto& w : res.rewards) delete w.reward;
		for (auto* tc : res.terminalConditions) delete tc;
		delete res.obsBuilder;
		actionParsers[0] = res.actionParser;
		contActionParsers[0] = res.continuousActionParser;
		for (int a = 1; a < numArenas; a++) {
			actionParsers[a] = actionParsers[0];
			contActionParsers[a] = contActionParsers[0];
		}
		actionCount = actionParsers[0] ? actionParsers[0]->GetActionAmount() : rsc::DEFAULT_ACTION_COUNT;
		obsSize = batch->GetObsRowSize();
		ConfigureGpuTrainingRewards();
		// Chain Step->Obs->Rewards without per-kernel device sync (CopyGpuTrainingBuffers fences).
		batch->SetKernelSync(false);
	} else {
		for (int a = 0; a < numArenas; a++) {
			CudaEnvCreateResult res = createFn(a);
			obsBuilders[a] = res.obsBuilder;
			rewards[a] = res.rewards;
			terminalConditions[a] = res.terminalConditions;
			actionParsers[a] = res.actionParser;
			contActionParsers[a] = res.continuousActionParser;
			gameStates[a] = GameState(*batch, a);
			ResetArenaEnv(a);
		}
		actionCount = actionParsers[0] ? actionParsers[0]->GetActionAmount() : rsc::DEFAULT_ACTION_COUNT;
		obsSize = (int)obsBuilders[0]->BuildObs(gameStates[0].players[0], gameStates[0]).size();
	}

	obs = DimList2<float>(numPlayers, obsSize);
	actionMasks = DimList2<uint8_t>(numPlayers, actionCount);
	rewards_out.assign(numPlayers, 0.f);
	lastRewards.resize(numArenas);
	terminals.assign(numArenas, (uint8_t)NOT_TERMINAL);

	controlsBuf.resize((size_t)numPlayers);
	if (gpuNative) {
		BuildDiscreteActionLut();
		batch->BuildAdvancedObsAndDefaultMasks();
		CopyGpuTrainingBuffers();
	} else {
		BuildDiscreteActionLut();
		for (int a = 0; a < numArenas; a++) {
			BuildArenaObs(a);
			BuildArenaActionMasks(a);
		}
	}
}

void GGL::CudaEnvSet::BuildDiscreteActionLut() {
	discreteActionLut.resize((size_t)actionCount);
	static Player dummyPlayer = {};
	static GameState dummyGs = {};
	auto* parser = actionParsers.empty() ? nullptr : actionParsers[0];
	if (!parser) return;
	for (int i = 0; i < actionCount; i++) {
		Action act = parser->ParseAction(i, dummyPlayer, dummyGs);
		discreteActionLut[(size_t)i] = ToControls(act);
	}
	if (batch && !discreteActionLut.empty()) {
		batch->UploadDiscreteActionLut(discreteActionLut.data(), actionCount);
	}
}

GGL::CudaEnvSet::~CudaEnvSet() {
	if (batch) {
		batch->Destroy();
		delete batch;
		batch = nullptr;
	}
	for (auto* o : obsBuilders) delete o;
	for (auto& set : rewards)
		for (auto& w : set) delete w.reward;
	for (auto& set : terminalConditions)
		for (auto* tc : set) delete tc;
	if (gpuNative) {
		// Parsers are shared across arenas — free once.
		if (!actionParsers.empty())
			delete actionParsers[0];
		if (!contActionParsers.empty())
			delete contActionParsers[0];
	} else {
		for (auto* p : actionParsers) delete p;
		for (auto* p : contActionParsers) delete p;
	}
}

void GGL::CudaEnvSet::ResetArenaEnv(int arenaIdx) {
	GameState& gs = gameStates[arenaIdx];
	if (obsBuilders[arenaIdx])
		obsBuilders[arenaIdx]->Reset(gs);
	for (auto& w : rewards[arenaIdx])
		w.reward->Reset(gs);
	for (auto* tc : terminalConditions[arenaIdx])
		tc->Reset(gs);
}

void GGL::CudaEnvSet::BridgeArena(int arenaIdx, const std::vector<Action>& arenaActions, bool usePrev) {
	gameStates[arenaIdx].UpdateFromCudaBatch(
		*batch, arenaIdx, arenaActions, usePrev ? &prevGameStates[arenaIdx] : nullptr);
}

void GGL::CudaEnvSet::BuildArenaObs(int arenaIdx) {
	GameState& gs = gameStates[arenaIdx];
	int base = arenaIdx * carsPerArena;
	for (int c = 0; c < carsPerArena; c++) {
		FList row = obsBuilders[arenaIdx]->BuildObs(gs.players[c], gs);
		obs.Set(base + c, row);
	}
}

void GGL::CudaEnvSet::BuildArenaActionMasks(int arenaIdx) {
	if (!actionParsers[arenaIdx])
		return;
	GameState& gs = gameStates[arenaIdx];
	int base = arenaIdx * carsPerArena;
	for (int c = 0; c < carsPerArena; c++) {
		auto mask = actionParsers[arenaIdx]->GetActionMask(gs.players[c], gs);
		actionMasks.Set(base + c, mask);
	}
}

void GGL::CudaEnvSet::StepDiscreteDevice(const int* deviceActionIndices) {
	if (!gpuNative || !batch || !deviceActionIndices || discreteActionLut.empty())
		return;
	batch->SnapshotTrainingState();
	batch->SetDiscreteActionsFromDevice(deviceActionIndices, numPlayers);
	batch->Step(tickSkip);
	batch->BuildAdvancedObsAndDefaultMasks();
	batch->BuildRewardsAndTerminals(tickSkip);
	CopyGpuTrainingBuffers();
	if (!skipHostObsCopy)
		ResetTerminalArenasGpu();
	MaybeCaptureStateRing();
}

void GGL::CudaEnvSet::StepDiscrete(const std::vector<int>& actions) {
	if ((int)controlsBuf.size() != numPlayers)
		controlsBuf.resize((size_t)numPlayers);
	// LUT: discrete index → CarControls (no per-step ParseAction).
	// GPU-native path expands indices on device; skip host controlsBuf fill.
	if (gpuNative && !discreteActionLut.empty()) {
		// device LUT already uploaded in BuildDiscreteActionLut
	} else if (!discreteActionLut.empty()) {
		for (int p = 0; p < numPlayers; p++) {
			int idx = actions[(size_t)p];
			if (idx < 0) idx = 0;
			if (idx >= actionCount) idx = actionCount - 1;
			controlsBuf[(size_t)p] = discreteActionLut[(size_t)idx];
		}
	} else {
		static Player dummyPlayer = {};
		static GameState dummyGs = {};
		for (int p = 0; p < numPlayers; p++) {
			int a = p / carsPerArena;
			int c = p % carsPerArena;
			int idx = actions[(size_t)p];
			if (idx < 0) idx = 0;
			if (idx >= actionCount) idx = actionCount - 1;
			Action act = actionParsers[a]->ParseAction(idx, dummyPlayer, dummyGs);
			controlsBuf[(size_t)p] = ToControls(act);
		}
	}

	if (gpuNative) {
		batch->SnapshotTrainingState();
		if (!discreteActionLut.empty()) {
			batch->SetDiscreteActionsFromIndices(actions.data(), numPlayers);
		} else {
			batch->SetAllCarControls(controlsBuf.data());
		}
		batch->Step(tickSkip);
		batch->BuildAdvancedObsAndDefaultMasks();
		batch->BuildRewardsAndTerminals(tickSkip);
		CopyGpuTrainingBuffers();
		if (!skipHostObsCopy)
			ResetTerminalArenasGpu();
		MaybeCaptureStateRing();
		return;
	}

	// Hybrid fallback: GPU physics + CPU obs/rewards
	std::vector<std::vector<Action>> arenaActions(numArenas);
	for (int a = 0; a < numArenas; a++)
		arenaActions[a].resize(carsPerArena);
	for (int p = 0; p < numPlayers; p++) {
		int a = p / carsPerArena;
		int c = p % carsPerArena;
		arenaActions[a][c] = actionParsers[a]->ParseAction(
			actions[p], gameStates[a].players[c], gameStates[a]);
	}

	batch->SetAllCarControls(controlsBuf.data());
	batch->Step(tickSkip);

	for (int a = 0; a < numArenas; a++) {
		prevGameStates[a] = gameStates[a];
		BridgeArena(a, arenaActions[a], true);

		GameState& gs = gameStates[a];
		uint8_t ttype = (uint8_t)NOT_TERMINAL;
		for (auto* tc : terminalConditions[a]) {
			if (tc->IsTerminal(gs)) {
				ttype = tc->IsTruncation() ? (uint8_t)TRUNCATED : (uint8_t)NORMAL;
				break;
			}
		}
		terminals[a] = ttype;
		bool isFinal = (ttype != (uint8_t)NOT_TERMINAL);

		for (auto& w : rewards[a])
			w.reward->PreStep(gs);

		lastRewards[a].resize(rewards[a].size());
		int base = a * carsPerArena;
		for (int c = 0; c < carsPerArena; c++) {
			float r = 0.f;
			for (size_t ri = 0; ri < rewards[a].size(); ri++) {
				float component = RuntimeRewardScale(
					rewards[a][ri].reward->GetName(), rewards[a][ri].weight)
					* rewards[a][ri].reward->GetReward(gs.players[c], gs, isFinal);
				lastRewards[a][ri] = component;
				r += component;
			}
			rewards_out[base + c] = r;
		}

		BuildArenaObs(a);
		BuildArenaActionMasks(a);

		if (isFinal) {
			batch->ResetArena(a);
			gameStates[a] = GameState(*batch, a);
			ResetArenaEnv(a);
			BuildArenaObs(a);
			BuildArenaActionMasks(a);
		}
	}
	MaybeCaptureStateRing();
}

void GGL::CudaEnvSet::StepContinuous(const FList& actions, int actionDim) {
	if ((int)controlsBuf.size() != numPlayers)
		controlsBuf.resize((size_t)numPlayers);
	std::vector<std::vector<Action>> arenaActions;
	if (!gpuNative) {
		arenaActions.resize(numArenas);
		for (int a = 0; a < numArenas; a++)
			arenaActions[a].resize(carsPerArena);
	}

	static Player dummyPlayer = {};
	static GameState dummyGs = {};
	for (int p = 0; p < numPlayers; p++) {
		int a = p / carsPerArena;
		int c = p % carsPerArena;
		const float* ptr = &actions[(size_t)p * actionDim];
		const Player& pl = gpuNative ? dummyPlayer : gameStates[a].players[c];
		const GameState& gs = gpuNative ? dummyGs : gameStates[a];
		Action act = contActionParsers[a]->ParseContinuousAction(ptr, actionDim, pl, gs);
		if (!gpuNative)
			arenaActions[a][c] = act;
		controlsBuf[(size_t)p] = ToControls(act);
	}

	if (gpuNative) {
		batch->SnapshotTrainingState();
		batch->SetAllCarControls(controlsBuf.data());
		batch->Step(tickSkip);
		batch->BuildAdvancedObsAndDefaultMasks();
		batch->BuildRewardsAndTerminals(tickSkip);
		CopyGpuTrainingBuffers();
		ResetTerminalArenasGpu();
		MaybeCaptureStateRing();
		return;
	}

	batch->SetAllCarControls(controlsBuf.data());
	batch->Step(tickSkip);

	for (int a = 0; a < numArenas; a++) {
		prevGameStates[a] = gameStates[a];
		BridgeArena(a, arenaActions[a], true);

		GameState& gs = gameStates[a];

		uint8_t ttype = (uint8_t)NOT_TERMINAL;
		for (auto* tc : terminalConditions[a]) {
			if (tc->IsTerminal(gs)) {
				ttype = tc->IsTruncation() ? (uint8_t)TRUNCATED : (uint8_t)NORMAL;
				break;
			}
		}
		terminals[a] = ttype;
		bool isFinal = (ttype != (uint8_t)NOT_TERMINAL);

		for (auto& w : rewards[a])
			w.reward->PreStep(gs);

		lastRewards[a].resize(rewards[a].size());
		int base = a * carsPerArena;
		for (int c = 0; c < carsPerArena; c++) {
			float r = 0.f;
			for (size_t ri = 0; ri < rewards[a].size(); ri++) {
				float component = RuntimeRewardScale(
					rewards[a][ri].reward->GetName(), rewards[a][ri].weight)
					* rewards[a][ri].reward->GetReward(gs.players[c], gs, isFinal);
				lastRewards[a][ri] = component;
				r += component;
			}
			rewards_out[base + c] = r;
		}

		BuildArenaObs(a);
		BuildArenaActionMasks(a);

		if (isFinal) {
			batch->ResetArena(a);
			gameStates[a] = GameState(*batch, a);
			ResetArenaEnv(a);
			BuildArenaObs(a);
			BuildArenaActionMasks(a);
		}
	}
	MaybeCaptureStateRing();
}

void GGL::CudaEnvSet::Reset() {
	if (gpuNative) {
		if (skipHostObsCopy)
			return; // device-XP: open-loop physics; Learner banks 1-step truncate
		ResetTerminalArenasGpu();
		return;
	}
	for (int a = 0; a < numArenas; a++) {
		if (!terminals[a])
			continue;
		batch->ResetArena(a);
		gameStates[a] = GameState(*batch, a);
		ResetArenaEnv(a);
		BuildArenaObs(a);
		BuildArenaActionMasks(a);
		terminals[a] = (uint8_t)NOT_TERMINAL;
	}
}

void GGL::CudaEnvSet::ResetAll() {
	batch->ResetAllArenas();
	if (gpuNative) {
		batch->BuildAdvancedObsAndDefaultMasks();
		CopyGpuTrainingBuffers();
		std::fill(rewards_out.begin(), rewards_out.end(), 0.f);
		std::fill(terminals.begin(), terminals.end(), (uint8_t)NOT_TERMINAL);
		return;
	}
	for (int a = 0; a < numArenas; a++) {
		gameStates[a] = GameState(*batch, a);
		ResetArenaEnv(a);
		BuildArenaObs(a);
		BuildArenaActionMasks(a);
	}
	std::fill(rewards_out.begin(), rewards_out.end(), 0.f);
	std::fill(terminals.begin(), terminals.end(), (uint8_t)NOT_TERMINAL);
}

namespace {
	inline float RingRand01(uint32_t& rng) {
		rng ^= rng << 13;
		rng ^= rng >> 17;
		rng ^= rng << 5;
		return (rng & 0xFFFFFFu) / float(0xFFFFFFu);
	}
	inline float RingRandRange(uint32_t& rng, float lo, float hi) {
		return lo + (hi - lo) * RingRand01(rng);
	}
}

void GGL::CudaEnvSet::ConfigureStateRing(const StateRingConfig& cfg) {
	stateRing.cfg = cfg;
	stateRing.enabled = cfg.enabled && batch != nullptr && numArenas > 0;
	if (!stateRing.enabled) {
		stateRing.slots.clear();
		stateRing.filled = 0;
		stateRing.writeIdx = 0;
		return;
	}
	const int depth = (std::max)(2, cfg.depth);
	stateRing.cfg.depth = depth;
	stateRing.cfg.captureEvery = (std::max)(1, cfg.captureEvery);
	stateRing.cfg.lookbackSlots = (std::max)(1, (std::min)(cfg.lookbackSlots, depth - 1));
	stateRing.cfg.restoreFraction = (std::max)(0.01f, (std::min)(0.05f, cfg.restoreFraction > 0.f ? cfg.restoreFraction : 0.05f));
	stateRing.cfg.maxArenasPerRestore = (std::max)(1, cfg.maxArenasPerRestore > 0 ? cfg.maxArenasPerRestore : 64);
	stateRing.slots.assign((size_t)depth, std::vector<ArenaSnap>((size_t)numArenas));
	for (int s = 0; s < depth; s++) {
		for (int a = 0; a < numArenas; a++) {
			auto& snap = stateRing.slots[(size_t)s][(size_t)a];
			snap.cars.assign((size_t)carsPerArena, rsc::CarState{});
			snap.pads.assign((size_t)rsc::NUM_BOOST_PADS, rsc::BoostPadState{});
			snap.valid = false;
		}
	}
	stateRing.stagingCars.assign((size_t)numPlayers, rsc::CarState{});
	stateRing.stagingBalls.assign((size_t)numArenas, rsc::BallState{});
	stateRing.stagingPads.assign((size_t)numArenas * rsc::NUM_BOOST_PADS, rsc::BoostPadState{});
	stateRing.writeIdx = 0;
	stateRing.filled = 0;
	stateRing.stepsSinceCapture = 0;
	RG_LOG("CudaEnvSet state-ring: depth=" << depth
		<< " captureEvery=" << stateRing.cfg.captureEvery
		<< " lookbackSlots=" << stateRing.cfg.lookbackSlots
		<< " restoreFrac=" << stateRing.cfg.restoreFraction
		<< " maxArenas=" << stateRing.cfg.maxArenasPerRestore
		<< " (~" << (stateRing.cfg.lookbackSlots * stateRing.cfg.captureEvery * tickSkip / 120.f)
		<< "s lookback)");
}

void GGL::CudaEnvSet::MaybeCaptureStateRing() {
	if (!stateRing.enabled || !batch)
		return;
	// Never cudaDeviceSynchronize / bulk D2H while async PPO Learn owns the device.
	if (suppressStateRingCapture.load(std::memory_order_acquire))
		return;
	if (++stateRing.stepsSinceCapture < stateRing.cfg.captureEvery)
		return;
	stateRing.stepsSinceCapture = 0;
	CaptureStateRingSlot();
}

void GGL::CudaEnvSet::CaptureStateRingSlot() {
	if (!stateRing.enabled || !batch || stateRing.slots.empty())
		return;
	// Fence chained kernels before D2H state dump. When GetKernelSync() is on, each
	// prior kernel already synchronized — skip the redundant full-device sync.
	if (!batch->GetKernelSync())
		batch->Synchronize();
	batch->GetAllCarStates(stateRing.stagingCars.data());
	batch->GetAllBallStates(stateRing.stagingBalls.data());
	batch->GetAllBoostPadStates(stateRing.stagingPads.data());

	const int slot = stateRing.writeIdx;
	auto& arenas = stateRing.slots[(size_t)slot];
	for (int a = 0; a < numArenas; a++) {
		auto& snap = arenas[(size_t)a];
		for (int c = 0; c < carsPerArena; c++)
			snap.cars[(size_t)c] = stateRing.stagingCars[(size_t)(a * carsPerArena + c)];
		snap.ball = stateRing.stagingBalls[(size_t)a];
		const int padBase = a * rsc::NUM_BOOST_PADS;
		for (int p = 0; p < rsc::NUM_BOOST_PADS; p++)
			snap.pads[(size_t)p] = stateRing.stagingPads[(size_t)(padBase + p)];
		snap.valid = true;
	}
	stateRing.writeIdx = (slot + 1) % stateRing.cfg.depth;
	if (stateRing.filled < stateRing.cfg.depth)
		stateRing.filled++;
	stateRing.captures++;
}

void GGL::CudaEnvSet::ApplyArenaSnap(int arenaIdx, int slot, float fuzzScale) {
	if (!batch || arenaIdx < 0 || arenaIdx >= numArenas)
		return;
	if (slot < 0 || slot >= stateRing.cfg.depth)
		return;
	const auto& snap = stateRing.slots[(size_t)slot][(size_t)arenaIdx];
	if (!snap.valid)
		return;

	const float fp = stateRing.cfg.fuzzPos * fuzzScale;
	const float fv = stateRing.cfg.fuzzVel * fuzzScale;

	for (int c = 0; c < carsPerArena && c < (int)snap.cars.size(); c++) {
		rsc::CarState car = snap.cars[(size_t)c];
		if (fp > 0.f) {
			car.pos.x += RingRandRange(stateRing.rng, -fp, fp);
			car.pos.y += RingRandRange(stateRing.rng, -fp, fp);
			car.pos.z = (std::max)(17.f, car.pos.z + RingRandRange(stateRing.rng, -fp * 0.25f, fp * 0.35f));
		}
		if (fv > 0.f) {
			car.vel.x += RingRandRange(stateRing.rng, -fv, fv);
			car.vel.y += RingRandRange(stateRing.rng, -fv, fv);
			car.vel.z += RingRandRange(stateRing.rng, -fv * 0.5f, fv * 0.5f);
		}
		batch->SetCarState(arenaIdx, c, car);
	}

	rsc::BallState ball = snap.ball;
	if (fp > 0.f) {
		ball.pos.x += RingRandRange(stateRing.rng, -fp, fp);
		ball.pos.y += RingRandRange(stateRing.rng, -fp, fp);
		ball.pos.z = (std::max)(93.f, ball.pos.z + RingRandRange(stateRing.rng, -fp * 0.2f, fp * 0.4f));
	}
	if (fv > 0.f) {
		ball.vel.x += RingRandRange(stateRing.rng, -fv, fv);
		ball.vel.y += RingRandRange(stateRing.rng, -fv, fv);
		ball.vel.z += RingRandRange(stateRing.rng, -fv * 0.5f, fv * 0.5f);
	}
	batch->SetBallState(arenaIdx, ball);

	if (!snap.pads.empty())
		batch->SetBoostPadStates(arenaIdx, snap.pads.data());

	if (!gpuNative && arenaIdx < (int)gameStates.size()) {
		gameStates[arenaIdx] = GameState(*batch, arenaIdx);
		ResetArenaEnv(arenaIdx);
	}
}

int GGL::CudaEnvSet::RestoreStateRingReplay(float restoreFraction, int lookbackSlots, float fuzzScale, int maxArenas) {
	if (!stateRing.enabled || stateRing.replayDisabled || !batch || stateRing.filled < 2)
		return 0;

	float frac = restoreFraction >= 0.f ? restoreFraction : stateRing.cfg.restoreFraction;
	// Hard ceiling: ≤5% of batch (never rewind nearly all arenas)
	frac = (std::max)(0.01f, (std::min)(0.05f, frac));
	int lookback = lookbackSlots >= 0 ? lookbackSlots : stateRing.cfg.lookbackSlots;
	lookback = (std::max)(1, (std::min)(lookback, stateRing.filled - 1));

	// writeIdx points to next write = oldest among filled once full; last written = writeIdx-1
	int lastWritten = stateRing.writeIdx - 1;
	if (lastWritten < 0)
		lastWritten = stateRing.cfg.depth - 1;
	int slot = lastWritten - lookback;
	while (slot < 0)
		slot += stateRing.cfg.depth;

	int cap = maxArenas > 0 ? maxArenas : stateRing.cfg.maxArenasPerRestore;
	if (cap <= 0) cap = 64;
	// Also ≤5% of arenas
	int fivePct = (std::max)(1, (int)std::lround(0.05 * (double)numArenas));
	cap = (std::min)(cap, fivePct);

	int nRestore = (std::max)(1, (int)std::lround(frac * (double)numArenas));
	nRestore = (std::min)(nRestore, numArenas);
	nRestore = (std::min)(nRestore, cap);

	// Fisher-Yates partial shuffle over arena indices
	std::vector<int> order((size_t)numArenas);
	for (int i = 0; i < numArenas; i++)
		order[(size_t)i] = i;
	for (int i = 0; i < nRestore; i++) {
		int j = i + (int)(RingRand01(stateRing.rng) * (numArenas - i));
		if (j >= numArenas) j = numArenas - 1;
		std::swap(order[(size_t)i], order[(size_t)j]);
	}

	for (int i = 0; i < nRestore; i++)
		ApplyArenaSnap(order[(size_t)i], slot, fuzzScale);

	// Refresh obs/masks so next Infer sees restored states
	if (gpuNative) {
		batch->BuildAdvancedObsAndDefaultMasks();
		if (!skipHostObsCopy)
			CopyGpuTrainingBuffers();
		else
			batch->Synchronize();
	}

	stateRing.restores += (uint64_t)nRestore;
	stateRing.restoreCalls++;
	// Tripwire: if restore calls explode vs captures, disable further replay
	if (stateRing.restoreCalls > 32 && stateRing.captures > 0
		&& stateRing.restoreCalls > stateRing.captures * 4) {
		stateRing.replayDisabled = true;
		RG_LOG("CudaEnvSet state-ring: replay DISABLED (calls=" << stateRing.restoreCalls
			<< " captures=" << stateRing.captures << " arenas_restored=" << stateRing.restores << ")");
	}
	return nRestore;
}

int GGL::CudaEnvSet::TriggerErrorReplay(float restoreFraction, int lookbackSlots, float fuzzScale, int maxArenas) {
	if (stateRing.replayDisabled)
		return 0;
	if (!stateRing.enabled) {
		StateRingConfig cfg = stateRing.cfg;
		cfg.enabled = true;
		if (cfg.depth < 2) cfg.depth = 10;
		if (cfg.captureEvery < 1) cfg.captureEvery = 3;
		if (cfg.restoreFraction <= 0.f || cfg.restoreFraction > 0.05f) cfg.restoreFraction = 0.05f;
		if (cfg.maxArenasPerRestore <= 0) cfg.maxArenasPerRestore = 64;
		ConfigureStateRing(cfg);
		// Need at least one capture before restore can work
		CaptureStateRingSlot();
		return 0;
	}
	return RestoreStateRingReplay(restoreFraction, lookbackSlots, fuzzScale, maxArenas);
}

#endif // GIGA_USE_CUDA_SIM
