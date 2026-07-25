#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <GigaLearnCPP/Learner.h>
#ifdef GIGA_USE_CUDA_SIM
#include <GigaLearnCPP/Sim/CudaEnvSet.h>
#endif

#include <RLGymCPP/RewardCore/CommonRewards.h>
#include <RLGymCPP/RewardCore/ZeroSumReward.h>
#include <RLGymCPP/RewardCore/Reward.h>
#include <RLGymCPP/TerminalConditions/NoTouchCondition.h>
#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>
#include <RLGymCPP/OBSBuilders/DefaultObs.h>
#include <RLGymCPP/OBSBuilders/AdvancedObs.h>
#include <RLGymCPP/OBSBuilders/AdvancedObsPadded.h>
#include <RLGymCPP/StateSetters/KickoffState.h>
#include <RLGymCPP/StateSetters/FuzzedKickoffState.h>
#include <RLGymCPP/StateSetters/CombinedState.h>
#include <RLGymCPP/StateSetters/RandomState.h>
#include <RLGymCPP/StateSetters/BallChaseState.h>
#include <RLGymCPP/ActionParsers/DefaultAction.h>
#include <RLGymCPP/CommonValues.h>
#include <GigaLearnCPP/Util/InferUnit.h>
#include <RLGymCPP/Gamestates/GameState.h>
#include "RLBotClient.h"
#include "TrainingCurriculum.h"
#include "AutoTrainerBridge.h"
// Optional extra reward recipe headers under src/ are not included by default.

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <set>
#include <unordered_map>
#include <vector>
#include <nlohmann/json.hpp>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif
#ifdef receiver
#undef receiver
#endif

using namespace GGL;
using namespace RLGC;

// =============================================================================
// TRAINING SIZE - edit these, then REBUILD (tools\build_amd.bat or cmake).
// Arenas / PPO bank / episode wall / ckpt folder.
// Reference PC: Win11 - A520-M - Ryzen 5 3600 - RX 6600 XT 8GB - see docs/AMD.md
// Locked profile: amd_win_20k -> normal train Overall >= 20,000 SPS with HIP working.
// =============================================================================
// Precedence (default POWER path: from-scratch / normal / continue-leak):
//   1) kForceTrainingSize == true  -> numbers below ALWAYS win (ignore hw_probe + GIGA_ENV_*).
//   2) Else GIGA_ENV_ARENAS / TS / EPOCHS / MAXEP / STEPS -> env wins (no rebuild).
//   3) Else kUseHwProbeArenas && hw_probe arenas -> probe wins for arenas only.
//   4) Else -> kArenas / kTsPerItr / kPpoEpochs / kMaxEpisodeSeconds / kStepsPerItr below.
// Tip: set kUseHwProbeArenas = false to lock arenas to kArenas without forcing env off.
// After ANY edit here you must REBUILD - changing this file alone does nothing to the exe.
// =============================================================================
namespace TrainingSize {
	// Profile name (hw_probe / run_fresh_train.bat on AMD HIP).
	constexpr const char* kProfileName = "amd_win_20k";
	// Target Overall for Win11 + RX 6600 XT + 3600 with HIP + CPU PPO.
	// Overall = 1/(1/Coll+1/Cons). Without HIP, 20k is NOT guaranteed.
	constexpr float kTargetMinOverallSps = 20000.f;
	// Secondary Overall target when HIP is healthy.
	constexpr float kStretchOverallSps = 25000.f;

	// true  = this block is the single source (beats hw_probe and GIGA_ENV_*).
	// false = allow probe/env overrides (good until you lock a size).
	constexpr bool kForceTrainingSize = false;

	// When false, ignore hw_probe arena recommendation and use kArenas
	// (GIGA_ENV_ARENAS still overrides unless kForceTrainingSize).
	constexpr bool kUseHwProbeArenas = true;

	// Parallel RocketSim arenas (LearnerConfig::numGames).
	// amd_win_20k: 2048 = Overall sweet spot on 8GB (1536 safer / 2560 Coll-heavy).
	// Do NOT jump to 8192 on 8GB VRAM - Learn+Infer on the 3600 will choke Overall.
	// Auto-downgrade ladder if Overall stays <20k: 2048 -> 1536 -> 1024 (next restart).
	constexpr int kArenas = 2048;

	// Env steps in the experience bank. Auto ts ~ arenas * 2 * steps (1v1).
	// 2048 x 2 x 3 = 12288 ts/iter - leaner Cons on 3600 vs steps=4 (16384).
	constexpr int kStepsPerItr = 3;

	// Timesteps per PPO iteration. 0 = auto (arenas * 2 * kStepsPerItr).
	constexpr int kTsPerItr = 0;

	// PPO epochs per iteration (1 = full-batch SPS path; 2 = Leak-style reuse).
	constexpr int kPpoEpochs = 1;

	// Minibatch size. 0 = auto (full batch when epochs==1; Leak-style when epochs>=2).
	constexpr int kMiniBatchSize = 0;

	// Episode wall-clock truncate (seconds). Shorter = more truncates / denser banks.
	constexpr double kMaxEpisodeSeconds = 1.5;

	// Pure self-play until this many steps (opp/old/skill stay off - SPS tax).
	constexpr int64_t kSparringDeferTs = 50'000'000;

	// From-scratch checkpoint folder name under the exe dir (e.g. "checkpoints").
	constexpr const char* kCheckpointDirName = "checkpoints";
}

// CUDA POWER (NVIDIA normal / from-scratch) - gated separately from amd_win_20k.
// 5060 Ti 16GB class: 8192 arenas + lean512 + half Infer. Does NOT touch AMD HIP defaults.
// Sustained target: median Overall ~ smoke steady (~946k+), not a lucky peak.
namespace CudaPower {
	constexpr int kArenas = 8192;
	// 16 steps -> ts=262144: better amortizes Infer/Learn fixed cost than steps=8 on 5060 Ti.
	constexpr int kStepsPerItr = 16;
	constexpr int kPpoEpochs = 1;
	// 1.0s -> maxEpisodeLength=20; iterBudgetFull truncates at ~steps so Collect stays lean.
	constexpr double kMaxEpisodeSeconds = 1.0;
	// Sustained Overall target (median/mean), not peak.
	constexpr float kTargetMinOverallSps = 900000.f;
	// Defer disk Save hitches (~every 4 iters at 1M) so long-run median stays near peak path.
	constexpr int64_t kTsPerSave = 25'000'000;
	// Pure self-play until this many steps - Apex must not re-arm opp/old/skill early.
	constexpr int64_t kSparringDeferTs = 50'000'000;
}

// NORMAL TRAIN (default, no flags): 1v1 **discrete** PPO + GPU high-SPS path.
// Continuous / entity-attention not wired.
// NVIDIA CUDA: high Overall. AMD Win HIP: >= 20k (amd_win_20k).
static bool g_Hyperpower = false;
static bool g_ForceCpuSim = false;
// Pure SPS benchmark only (1-step episode collapse) - NOT default training.
static bool g_Pure80 = false;
// Default 1v1 training path (blank stub rewards). Target Overall >= 400k (CUDA) / >=20k (AMD HIP).
static bool g_DefaultEnv = true;
// Full Leak-parity metric payload (console + optional wandb). Off by default for >=400k SPS.
static bool g_FullMetrics = false;
// Send metrics to wandb via MetricSender (requires creds). Implies g_FullMetrics.
static bool g_SendWandb = false;
// Apex-style phased curriculum (opponent mix / entropy / LR). On by default for normal train.
static bool g_ApexCurriculum = false;
// Opt-out: --no-apex / GIGA_NO_APEX=1 (keeps >=400k floor path without sparring tax).
static bool g_ForceNoApex = false;
// Continue-Leak: Leak 1024+LN weights + GigaLearnRL POWER train (docs/POWER_80X.md).
// Flags: --continue-leak | --like-leak | --max-learn | --resume-leak | env GIGA_CONTINUE_LEAK=1
// Size defaults come from TrainingSize{} above (RX 6600 XT 8GB -> 2048 arenas / amd_win_20k).
// Classic Leak batches: GIGA_ENV_ARENAS=2056 GIGA_ENV_TS=500000
//   GIGA_ENV_MAXEP=120 GIGA_ENV_EPOCHS=2 (+ optional GIGA_NO_APEX=1).
static bool g_MaxLearn = false;
// From-scratch (DEFAULT): POWER normal train into empty checkpoints/.
// Plain GigaLearnBot.exe and --from-scratch / --fresh are the same path.
// Does NOT load Leak checkpoints_default (~120B) or old lean ckpts.
// Opt out of blank box: --lean-resume (lean resume) | --continue-leak | GIGA_FROM_SCRATCH=0
static bool g_FromScratch = true;
// CLI wins over stale shell env (e.g. leftover GIGA_FROM_SCRATCH vs --continue-leak).
static bool g_CliFromScratch = false;
static bool g_CliContinueLeak = false;
// Extra per-iter SPS banners (off by default).
static bool g_Verbose = false;
// AutoTrainer sidecar: OFF by default (no self-spawn). Opt in: GIGA_AUTOTRAINER=1.
// Prefer run_with_autotrainer.bat / start_autotrainer.bat (they set EXTERNAL=1).
// Opt out still: --no-autotrainer / GIGA_NO_AUTOTRAINER=1.
static bool g_NoAutoTrainer = false;
// Hardware profile (tools/hw_probe.py -> hw_profile.json). Opt out: --no-hw-profile / GIGA_NO_HW_PROFILE=1.
static bool g_NoHwProfile = false;
// Explicit --cpu wins over hw_profile CUDA recommendations.
static bool g_ExplicitCpu = false;
// Arenas from hw_profile when GIGA_ENV_ARENAS unset and TrainingSize::kUseHwProbeArenas (-1 = unused).
static int g_ProfileArenas = -1;
// Prefer CPU libtorch when profile says no CUDA GPU (avoid hard GPU_CUDA require).
static bool g_ProfileTorchCpu = false;
// Win AMD HIP: gpuNative env + CPU PPO; Overall target = TrainingSize::kTargetMinOverallSps.
static bool g_AmdWin20k = false;
static bool g_AmdHipGpuNativeConfirmed = false;
static int g_AmdLowSpsStreak = 0;
static int g_AmdAutoDowngradeLevel = 0;
static std::filesystem::path g_AmdAutosizePath;
static std::filesystem::path g_TeacherPath;
static std::vector<std::filesystem::path> g_TeacherPaths; // multi-teacher registry (OpponentPool / hints)
static constexpr float kDefaultTargetOverallSPS = 400000.f;
// Last light skill-eval rating (from Report) - steers curriculum pressure without wandb.
static float g_LastSkillRating1v1 = 0.f;
static int64_t g_RandomSeed = -1; // --seed / GIGA_SEED

// Switches kickoff/random/ball-chase init by training phase.
class PhasedStateSetter : public StateSetter {
	StateSetter* chaseSetter;
	StateSetter* advancedSetter;
public:
	PhasedStateSetter(StateSetter* chaseSetter, StateSetter* advancedSetter)
		: chaseSetter(chaseSetter), advancedSetter(advancedSetter) {}

	~PhasedStateSetter() {
		delete chaseSetter;
		delete advancedSetter;
	}

	void ResetArena(Arena* arena) override {
		if (TrainingCurriculum::currentPhase == TrainingPhase::ADVANCED)
			advancedSetter->ResetArena(arena);
		else
			chaseSetter->ResetArena(arena);
	}

	StateSetter* ActiveSetter() const {
		return TrainingCurriculum::currentPhase == TrainingPhase::ADVANCED ? advancedSetter : chaseSetter;
	}

	StateSetter* AdvancedSetter() const { return advancedSetter; }
	StateSetter* ChaseSetter() const { return chaseSetter; }
};

// Hot-update CombinedState kickoff/random mix (Apex). Prefer Advanced 2-way mix.
static void SetKickoffStateMix(Learner* learner, float kickoffWeight, float randomWeight) {
	if (!learner || !learner->envSet)
		return;
	for (auto* setter : learner->envSet->stateSetters) {
		auto* phased = dynamic_cast<PhasedStateSetter*>(setter);
		CombinedState* combined = nullptr;
		if (phased) {
			combined = dynamic_cast<CombinedState*>(phased->AdvancedSetter());
			if (!combined || combined->NumSetters() != 2)
				combined = dynamic_cast<CombinedState*>(phased->ActiveSetter());
		} else {
			combined = dynamic_cast<CombinedState*>(setter);
		}
		if (!combined || combined->NumSetters() != 2)
			continue;
		combined->SetWeights({ kickoffWeight, randomWeight });
	}
}

// Hot-update default 3-way CombinedState (kickoff / fuzzed / aerial) on CPU hybrid path.
static void SetDefaultStateMix(Learner* learner, float kickoffW, float fuzzedW, float aerialW) {
	if (!learner || !learner->envSet)
		return;
	for (auto* setter : learner->envSet->stateSetters) {
		auto* combined = dynamic_cast<CombinedState*>(setter);
		if (!combined || combined->NumSetters() != 3)
			continue;
		combined->SetWeights({ kickoffW, fuzzedW, aerialW });
	}
}

// GPU-native reset curriculum (shapes normal CUDA train - CombinedState alone does not).
static void SetGpuResetCurriculum(Learner* learner, float kickoffW, float fuzzedW, float aerialW) {
#ifdef GIGA_USE_CUDA_SIM
	if (!learner || !learner->cudaEnvSet || !learner->cudaEnvSet->batch)
		return;
	learner->cudaEnvSet->batch->SetResetCurriculum(kickoffW, fuzzedW, aerialW);
#else
	(void)learner; (void)kickoffW; (void)fuzzedW; (void)aerialW;
#endif
}

// Curriculum auto-scale: densify pressure as timesteps advance (and optional skill hint).
static float CurriculumPressureScale(const Learner* learner, bool spsSafe) {
	if (!learner)
		return 1.f;
	const double tsB = learner->totalTimesteps / 1e9;
	float s = 1.f;
	if (tsB >= 0.040) s = 1.12f;
	if (tsB >= 0.120) s = 1.28f;
	if (tsB >= 0.220) s = 1.42f;
	if (tsB >= 0.350) s = 1.55f;
	// Light skill feedback: higher 1v1 rating -> slightly denser pressure (cheap; no wandb).
	if (g_LastSkillRating1v1 > 0.f) {
		float skillBoost = (std::min)(0.15f, g_LastSkillRating1v1 * 0.0025f);
		s *= (1.f + skillBoost);
	}
	if (const char* env = std::getenv("GIGA_CURRICULUM_SCALE")) {
		float v = (float)std::atof(env);
		if (v >= 0.5f && v <= 2.5f)
			s *= v;
	}
	// Normal train: cap pressure so sparring tax does not sink Overall below 400k for long.
	if (spsSafe)
		s = (std::min)(s, 1.28f);
	return s;
}

// Apex curriculum - hot-updates opponent mix / entropy / LR / gamma / beat SE by timesteps.
// Includes Requiem / NextoTled arch-aware weights when present in OpponentPool.
// spsSafe=true: normal/default path - lower external chance, prefer old-versions + lean teachers.
static int ApplyApexCurriculum(Learner* learner, Report* report, bool spsSafe = false) {
	if (!learner || learner->config.renderMode)
		return -1;

	const double tsB = learner->totalTimesteps / 1e9;
	const float pressure = CurriculumPressureScale(learner, spsSafe);
	int phase = 0;
	// SPS-safe (normal/8192): old-versions carry most pressure; externals rare + lean-only.
	// Requiem 2048 / Nexto JIT on half of 8192 arenas stalls Coll - weight 0 on normal.
	// Early old% kept modest so Collect stays fast (iter cadence); ramps in later phases.
	float trainOldChance = spsSafe ? 0.12f : 0.30f;
	float opponentChance = spsSafe ? 0.06f : 0.60f;
	float nextoW = spsSafe ? 0.f : 0.90f;
	float nectoW = spsSafe ? 0.f : 0.70f;
	float nextoTledW = spsSafe ? 1.0f : 1.05f; // lean 512 - only external on normal
	float requiemW = spsSafe ? 0.f : 1.15f;    // heavy 2048 - off on SPS-safe normal
	float beatBonus = spsSafe ? 70.f : 100.f;
	float concedePenalty = spsSafe ? -28.f : -40.f;
	float entropy = spsSafe ? 0.024f : 0.025f;
	float gamma = spsSafe ? 0.997f : 0.993f;
	float lr = 1e-4f;
	float kickoffW = 0.55f;
	float randomW = 0.45f;
	float envKickoff = 0.50f;
	float envFuzzed = 0.40f;
	float envAerial = 0.10f;

	if (tsB < 0.120) {
		phase = 0; // A Kickoff / foundations
	} else if (tsB < 0.180) {
		phase = 1; // B Mid strength
		trainOldChance = spsSafe ? 0.18f : 0.34f;
		opponentChance = spsSafe ? 0.07f : 0.55f;
		nextoW = spsSafe ? 0.f : 0.85f;
		nectoW = spsSafe ? 0.f : 0.75f;
		nextoTledW = spsSafe ? 1.0f : 1.10f;
		requiemW = spsSafe ? 0.f : 1.25f;
		beatBonus = spsSafe ? 75.f : 95.f;
		concedePenalty = spsSafe ? -30.f : -42.f;
		entropy = spsSafe ? 0.022f : 0.022f;
		gamma = spsSafe ? 0.997f : 0.994f;
		kickoffW = 0.50f;
		randomW = 0.50f;
		envKickoff = 0.40f;
		envFuzzed = 0.35f;
		envAerial = 0.25f;
	} else if (tsB < 0.280) {
		phase = 2; // C Hard mode
		trainOldChance = spsSafe ? 0.22f : 0.38f;
		opponentChance = spsSafe ? 0.08f : 0.62f;
		nextoW = spsSafe ? 0.f : 0.75f;
		nectoW = spsSafe ? 0.f : 0.85f;
		nextoTledW = spsSafe ? 1.0f : 1.05f;
		requiemW = spsSafe ? 0.f : 1.40f;
		beatBonus = spsSafe ? 70.f : 85.f;
		concedePenalty = spsSafe ? -32.f : -48.f;
		entropy = spsSafe ? 0.019f : 0.018f;
		gamma = spsSafe ? 0.9973f : 0.994f;
		lr = spsSafe ? 8e-5f : 5e-5f;
		kickoffW = 0.40f;
		randomW = 0.60f;
		envKickoff = 0.30f;
		envFuzzed = 0.30f;
		envAerial = 0.40f;
	} else {
		phase = 3; // D Apex
		trainOldChance = spsSafe ? 0.26f : 0.42f;
		opponentChance = spsSafe ? 0.09f : 0.45f;
		nextoW = spsSafe ? 0.f : 0.65f;
		nectoW = spsSafe ? 0.f : 0.90f;
		nextoTledW = spsSafe ? 1.0f : 1.00f;
		requiemW = spsSafe ? 0.f : 1.55f;
		beatBonus = spsSafe ? 60.f : 70.f;
		concedePenalty = spsSafe ? -26.f : -38.f;
		entropy = spsSafe ? 0.016f : 0.015f;
		gamma = spsSafe ? 0.9975f : 0.995f;
		lr = spsSafe ? 7e-5f : 5e-5f;
		kickoffW = 0.35f;
		randomW = 0.65f;
		envKickoff = 0.25f;
		envFuzzed = 0.25f;
		envAerial = 0.50f;
	}

	// Auto-scale: denser league / external / beat SE without inventing SPS.
	const float oldCap = spsSafe ? 0.34f : 0.55f;
	const float oppCap = spsSafe ? 0.12f : 0.85f;
	trainOldChance = (std::min)(oldCap, trainOldChance * pressure);
	opponentChance = (std::min)(oppCap, opponentChance * pressure);
	beatBonus *= pressure;
	concedePenalty *= (0.85f + 0.15f * pressure);

	static int lastPhase = -999;
	static float lastLr = -1.f;
	static float lastEntropy = -1.f;

	learner->config.trainAgainstOldVersions = true;
	learner->config.trainAgainstOldChance = trainOldChance;
	learner->config.opponentPool.enabled = true;
	learner->config.opponentPool.chance = opponentChance;
	learner->config.opponentPool.beatBonus = beatBonus;
	learner->config.opponentPool.concedePenalty = concedePenalty;
	learner->config.ppo.gaeGamma = gamma;
	learner->SetOpponentWeight("nexto", nextoW);
	learner->SetOpponentWeight("necto", nectoW);
	learner->SetOpponentWeight("nexto_tled", nextoTledW);
	learner->SetOpponentWeight("requiem", requiemW);
	SetKickoffStateMix(learner, kickoffW, randomW);
	if (spsSafe || g_DefaultEnv) {
		SetDefaultStateMix(learner, envKickoff, envFuzzed, envAerial);
		SetGpuResetCurriculum(learner, envKickoff, envFuzzed, envAerial);
	}

	if (lr != lastLr) {
		learner->SetLearningRates(lr, lr);
		lastLr = lr;
	}
	if (entropy != lastEntropy) {
		learner->SetEntropyScale(entropy);
		lastEntropy = entropy;
	}

	if (phase != lastPhase) {
		if (g_Verbose) {
			RG_LOG((spsSafe ? "NORMAL Apex curriculum: phase " : "Apex curriculum: phase ")
				<< phase << " @ " << tsB << "B"
				<< " | oldChance=" << trainOldChance
				<< " oppChance=" << opponentChance
				<< " nextoW=" << nextoW << " nectoW=" << nectoW
				<< " tledW=" << nextoTledW << " requiemW=" << requiemW
				<< " beat=" << beatBonus << " concede=" << concedePenalty
				<< " pressure=" << pressure
				<< " entropy=" << entropy << " gamma=" << gamma << " lr=" << lr
				<< " gpuReset(k/f/a)=" << envKickoff << "/" << envFuzzed << "/" << envAerial
				<< (spsSafe ? " [SPS-safe]" : ""));
		}
		lastPhase = phase;
	}

	// Multi-teacher cycle hint by Apex phase (eval harness / TL index consumers).
	const int teacherHint = g_TeacherPaths.empty() ? 0 : (phase % (int)g_TeacherPaths.size());

	if (report) {
		(*report)["Curriculum/Phase"] = (float)phase;
		(*report)["Curriculum/OpponentChance"] = opponentChance;
		(*report)["Curriculum/TrainAgainstOldChance"] = trainOldChance;
		(*report)["Curriculum/EntropyScale"] = entropy;
		(*report)["Curriculum/GaeGamma"] = gamma;
		(*report)["Curriculum/LearningRate"] = lr;
		(*report)["Curriculum/BeatBonus"] = beatBonus;
		(*report)["Curriculum/ConcedePenalty"] = concedePenalty;
		(*report)["Curriculum/KickoffWeight"] = kickoffW;
		(*report)["Curriculum/PressureScale"] = pressure;
		(*report)["Curriculum/RequiemWeight"] = requiemW;
		(*report)["Curriculum/GpuResetKickoff"] = envKickoff;
		(*report)["Curriculum/GpuResetFuzzed"] = envFuzzed;
		(*report)["Curriculum/GpuResetAerial"] = envAerial;
		(*report)["Curriculum/TeacherIndexHint"] = (float)teacherHint;
		(*report)["Curriculum/TeacherCount"] = (float)g_TeacherPaths.size();
		(*report)["Curriculum/SpsSafe"] = spsSafe ? 1.f : 0.f;
	}
	return phase;
}

static ObsBuilder* MakeObsBuilder() {
	// Discrete: flat AdvancedObs.
	return new AdvancedObs();
}

static std::set<int64_t> FindNumberedCheckpointDirs(const std::filesystem::path& basePath) {
	std::set<int64_t> results = {};
	if (!std::filesystem::exists(basePath))
		return results;
	for (const auto& entry : std::filesystem::directory_iterator(basePath)) {
		if (!entry.is_directory())
			continue;
		auto name = entry.path().filename().string();
		bool allDigits = !name.empty();
		for (char c : name) {
			if (!std::isdigit(static_cast<unsigned char>(c))) {
				allDigits = false;
				break;
			}
		}
		if (allDigits)
			results.insert(std::stoll(name));
	}
	return results;
}

static std::filesystem::path FindLatestValidCheckpoint(const std::filesystem::path& checkpointFolder) {
	int64_t bestTs = -1;
	std::filesystem::path bestPath;
	for (int64_t ts : FindNumberedCheckpointDirs(checkpointFolder)) {
		auto folder = checkpointFolder / std::to_string(ts);
		if (!std::filesystem::exists(folder / "POLICY.lt"))
			continue;
		if (!std::filesystem::exists(folder / "SHARED_HEAD.lt"))
			continue;
		if (ts > bestTs) {
			bestTs = ts;
			bestPath = folder;
		}
	}
	return bestPath;
}

static int ReadRLBotPort(const std::filesystem::path& exeDir) {
	const std::filesystem::path candidates[] = {
		exeDir / "port.cfg",
		exeDir / ".." / "rlbot" / "port.cfg",
		exeDir / ".." / ".." / "rlbot" / "port.cfg",
	};
	for (const auto& path : candidates) {
		std::ifstream fIn(path);
		if (!fIn.good())
			continue;
		int port = 0;
		if (fIn >> port)
			return port;
	}
	return 42653;
}

static void DiscoverTeacherCheckpoints(const std::filesystem::path& exeDir) {
	// Keep any CLI --teachers seeds, then expand search roots.
	// Local checkpoint boxes only.
	std::vector<std::filesystem::path> seeds = g_TeacherPaths;
	g_TeacherPaths.clear();
	std::vector<std::filesystem::path> roots = seeds;
	roots.push_back(g_TeacherPath);
	roots.push_back(exeDir / "checkpoints");
	roots.push_back(exeDir / "checkpoints" / "best_skill");
	roots.push_back(exeDir / "checkpoints_reconstructed");
	// Optional Leak resume boxes only when explicitly continuing Leak / lean.
	if (g_MaxLearn || !g_FromScratch) {
		roots.push_back(exeDir / "checkpoints_default_lean");
		roots.push_back(exeDir / "checkpoints_default");
	}
	std::set<std::string> seen;
	for (const auto& root : roots) {
		if (root.empty())
			continue;
		std::filesystem::path path = root;
		if (!std::filesystem::exists(path / "POLICY.lt"))
			path = FindLatestValidCheckpoint(root);
		if (path.empty() || !std::filesystem::exists(path / "POLICY.lt"))
			continue;
		std::string key = path.lexically_normal().string();
		if (!seen.insert(key).second)
			continue;
		g_TeacherPaths.push_back(path);
	}
	if (!g_TeacherPaths.empty()) {
		if (g_Verbose) {
			RG_LOG("Multi-teacher registry: " << g_TeacherPaths.size() << " checkpoint(s)");
			for (size_t i = 0; i < g_TeacherPaths.size(); i++)
				RG_LOG("  [" << i << "] " << g_TeacherPaths[i]);
		}
		try {
			nlohmann::json j;
			j["count"] = (int)g_TeacherPaths.size();
			nlohmann::json arr = nlohmann::json::array();
			for (const auto& p : g_TeacherPaths)
				arr.push_back(p.string());
			j["teachers"] = arr;
			j["includes_requiem"] = true;
			j["includes_nexto_tled"] = true;
			j["multi_seed_ready"] = true;
			std::ofstream out(exeDir / "teachers_manifest.json");
			out << j.dump(2);
		} catch (...) {}
	}
}

static void FillDiscretePolicyConfig(PartialModelConfig& policy) {
	// Match from-scratch lean 512 MLP used by default discrete train.
	policy.layerSizes = { 512, 512 };
	policy.optimType = ModelOptimType::ADAM;
	policy.activationType = ModelActivationType::RELU;
	policy.addLayerNorm = false;
}

static bool KickoffBallActive(const GameState& state) {
	const auto& b = state.ball;
	const float ballXY = std::hypot(b.pos.x, b.pos.y);
	if (ballXY > 460.f || b.pos.z > 235.f)
		return false;
	return b.vel.Length() <= 280.f;
}

// Blank-template phase rewards (CommonRewards only). Add your own headers locally.
static std::vector<WeightedReward> BuildChaseRewards() {
	return {
		{ new VelocityPlayerToBallReward(), 12.f },
		{ new FaceBallReward(), 4.0f },
		{ new TouchBallReward(), 22.f },
		{ new VelocityBallToGoalReward(), 2.0f },
		{ new GoalReward(-0.50f), 20.f },
	};
}

static std::vector<WeightedReward> BuildFoundationRewards() {
	return {
		{ new VelocityPlayerToBallReward(), 7.5f },
		{ new FaceBallReward(), 2.0f },
		{ new TouchBallReward(), 14.f },
		{ new VelocityBallToGoalReward(), 8.f },
		{ new GoalReward(-0.60f), 55.f },
		{ new PickupBoostReward(), 8.f },
		{ new SaveBoostReward(0.5f), 0.4f },
	};
}

static std::vector<WeightedReward> BuildAdvancedRewards() {
	auto r = BuildFoundationRewards();
	r.insert(r.end(), {
		{ new ZeroSumReward(new TouchBallReward(), 0.0f), 2.0f },
		{ new AirReward(), 0.25f },
	});
	return r;
}

static std::vector<WeightedReward> RewardsForPhase(TrainingPhase phase) {
	switch (phase) {
	case TrainingPhase::CHASE: return BuildChaseRewards();
	case TrainingPhase::FOUNDATION: return BuildFoundationRewards();
	default: return BuildAdvancedRewards();
	}
}

static EnvCreateResult MakeEnv(int playersPerTeam, int noTouchSeconds, std::vector<WeightedReward> rewards) {
	auto arena = Arena::Create(GameMode::SOCCAR);
	for (int i = 0; i < playersPerTeam; i++) {
		arena->AddCar(Team::BLUE);
		arena->AddCar(Team::ORANGE);
	}

	// LEVEL 4 env architect weights (defaults match dense BallChase SE stack).
	float bc = TrainingCurriculum::config.ballChaseWeight;
	float rs = TrainingCurriculum::config.randomStateWeight;
	float ko = TrainingCurriculum::config.kickoffWeight;
	float wSum = bc + rs + ko;
	if (wSum < 1e-6f) { bc = 0.85f; rs = 0.10f; ko = 0.05f; wSum = 1.f; }
	bc /= wSum; rs /= wSum; ko /= wSum;

	EnvCreateResult result = {};
	result.actionParser = new DefaultAction();
	result.continuousActionParser = nullptr; // discrete only
	result.obsBuilder = MakeObsBuilder();
	result.stateSetter = new PhasedStateSetter(
		// Chase: mostly BallChase - Leak uses kickoff (hard). This alone is multi-x SE.
		new CombinedState({
			{ new BallChaseState(2400.f, true, true), bc },
			{ new RandomState(true, true, true), rs },
			{ new KickoffState(), ko },
		}),
		new CombinedState({
			{ new KickoffState(), 0.40f },
			{ new RandomState(true, true, false), 0.60f },
		})
	);
	result.terminalConditions = {
		new NoTouchCondition(noTouchSeconds),
		new GoalScoreCondition()
	};
	// Soft ICM/RND-style densification via curriculum intrinsic coeffs.
	float wIcm = TrainingCurriculum::config.wIcm;
	float wRnd = TrainingCurriculum::config.wRnd;
	if (wIcm > 0.f || wRnd > 0.f) {
		for (auto& wr : rewards) {
			// Scale dense chase heads slightly.
			std::string name = wr.reward ? wr.reward->GetName() : std::string();
			if (name.find("VelocityPlayerToBall") != std::string::npos)
				wr.weight *= (1.f + 0.5f * wIcm);
			else if (name.find("TouchBall") != std::string::npos)
				wr.weight *= (1.f + 0.4f * wRnd);
		}
	}
	result.rewards = std::move(rewards);
	result.arena = arena;
	return result;
}

static EnvCreateResult EnvCreateTraining(int index) {
	(void)index;
	TrainingPhase phase = TrainingCurriculum::currentPhase;
	int playersPerTeam = TrainingCurriculum::PlayersPerTeamForPhase(phase);
	int noTouch = (phase == TrainingPhase::ADVANCED)
		? (int)TrainingCurriculum::config.noTouchSecondsAdvanced
		: (int)TrainingCurriculum::config.noTouchSecondsChase;
	return MakeEnv(playersPerTeam, noTouch, RewardsForPhase(phase));
}

// Leak ExampleMain aerial setup - used in default CombinedState (0.25 weight).
class AerialDribbleSetupStateSetter : public StateSetter {
public:
	void ResetArena(Arena* arena) override {
		arena->ResetToRandomKickoff();

		const float ballX = RocketSim::Math::RandFloat(-1600.f, 1600.f);
		const float ballY = RocketSim::Math::RandFloat(-1800.f, 1800.f);
		const float ballZ = RocketSim::Math::RandFloat(420.f, 780.f);

		BallState ballState = {};
		ballState.pos = Vec(ballX, ballY, ballZ);
		ballState.vel = Vec(
			RocketSim::Math::RandFloat(-120.f, 120.f),
			RocketSim::Math::RandFloat(-120.f, 120.f),
			RocketSim::Math::RandFloat(-40.f, 60.f)
		);
		arena->ball->SetState(ballState);

		std::vector<Car*> cars(arena->_cars.begin(), arena->_cars.end());
		if (cars.empty())
			return;

		const int setupIdx = (int)RocketSim::Math::RandFloat(0.f, (float)(cars.size() - 1));

		for (int i = 0; i < (int)cars.size(); i++) {
			Car* car = cars[i];
			CarState cs = {};

			if (i == setupIdx) {
				const Vec ballDir = Vec(
					RocketSim::Math::RandFloat(-1.f, 1.f),
					RocketSim::Math::RandFloat(-1.f, 1.f),
					0.f
				).Normalized();

				cs.pos = ballState.pos - ballDir * RocketSim::Math::RandFloat(180.f, 340.f);
				cs.pos.z = ballZ - RocketSim::Math::RandFloat(40.f, 160.f);

				const float yaw = std::atan2(ballY - cs.pos.y, ballX - cs.pos.x);
				cs.rotMat = Angle(yaw, RocketSim::Math::RandFloat(-0.25f, 0.35f), RocketSim::Math::RandFloat(-0.2f, 0.2f)).ToRotMat();

				const Vec toBall = (ballState.pos - cs.pos).Normalized();
				cs.vel = toBall * RocketSim::Math::RandFloat(500.f, 900.f);
				cs.angVel = {};

				cs.isOnGround = false;
				cs.hasJumped = RocketSim::Math::RandFloat() < 0.55f;
				cs.hasFlipped = cs.hasJumped && (RocketSim::Math::RandFloat() < 0.65f);
				cs.hasDoubleJumped = false;
				cs.airTime = 0.45f;
				cs.airTimeSinceJump = cs.hasJumped ? 0.9f : 0.f;
				cs.boost = RocketSim::Math::RandFloat(55.f, 100.f);
			} else {
				const float awayX = ballX + (ballX >= 0.f ? 3200.f : -3200.f);
				cs.pos = Vec(awayX, ballY, 17.f);
				cs.vel = {};
				cs.angVel = {};
				cs.isOnGround = true;
				cs.boost = RocketSim::Math::RandFloat(30.f, 70.f);
				cs.rotMat = Angle(std::atan2(ballY - cs.pos.y, ballX - cs.pos.x), 0.f, 0.f).ToRotMat();
			}

			car->SetState(cs);
		}
	}
};

// Default from-scratch 1v1 template - replace with YOUR rewards (see docs/CUSTOMIZE.md).
// Framework still exposes many reward classes under RLGymCPP; none of the owner packs ship as defaults.
static EnvCreateResult EnvCreateDefault(int index) {
	(void)index;

	// TODO: add your rewards here (Goal / Touch / VelBallToGoal stubs compile & train).
	std::vector<WeightedReward> rewards;
	rewards.push_back({ new GoalReward(), 100.f });
	rewards.push_back({ new ZeroSumReward(new TouchBallReward(), 0.0f), 5.0f });
	rewards.push_back({ new VelocityBallToGoalReward(), 10.f });
	rewards.push_back({ new VelocityPlayerToBallReward(), 2.0f });

	auto arena = Arena::Create(GameMode::SOCCAR);
	arena->AddCar(Team::BLUE);
	arena->AddCar(Team::ORANGE);

	EnvCreateResult result = {};
	result.actionParser = new DefaultAction();
	result.continuousActionParser = nullptr; // discrete only
	result.obsBuilder = new AdvancedObs();
	result.stateSetter = new CombinedState({
		{ new KickoffState(), 0.375f },
		{ new FuzzedKickoffState(), 0.375f },
		{ new AerialDribbleSetupStateSetter(), 0.25f },
	});
	result.terminalConditions = {
		new NoTouchCondition(10),
		new GoalScoreCondition()
	};
	result.rewards = std::move(rewards);
	result.arena = arena;
	return result;
}

static int GetTrainingObsSize() {
	EnvCreateResult env = EnvCreateTraining(0);
	env.stateSetter->ResetArena(env.arena);
	GameState gs(env.arena);
	env.obsBuilder->Reset(gs);
	return (int)env.obsBuilder->BuildObs(gs.players[0], gs).size();
}

void StepCallback(Learner* learner, const std::vector<GameState>& states, Report& report) {
	(void)learner;
	// Match GigaLearnCPP-Leak sampling cadence (every ~8th step batch).
	bool doExpensiveMetrics = (rand() % 8) == 0;

	if (!g_DefaultEnv)
		report.AddAvg("Curriculum/Phase", (float)TrainingCurriculum::currentPhase);

	for (auto& state : states) {
		if (doExpensiveMetrics) {
			for (auto& player : state.players) {
				report.AddAvg("Player/In Air Ratio", !player.isOnGround);
				report.AddAvg("Player/Ball Touch Ratio", player.ballTouchedStep);
				report.AddAvg("Player/Demoed Ratio", player.isDemoed);
				report.AddAvg("Player/Boost", player.boost);
				report.AddAvg("Player/Speed", player.vel.Length());
				if (!player.isOnGround && player.pos.z > 220.f)
					report.AddAvg("Aerial/InAirHigh", 1.f);
			}
			if (KickoffBallActive(state)) {
				for (auto& player : state.players) {
					const float d = (player.pos - state.ball.pos).Length();
					const Vec toBall = (state.ball.pos - player.pos).Normalized();
					report.AddAvg("Kickoff/PhaseActive", 1.f);
					report.AddAvg("Kickoff/DistToBall", d);
					report.AddAvg("Kickoff/VelTowardBall", RS_MAX(0.f, player.vel.Dot(toBall)));
				}
			}
		}

		if (state.goalScored)
			report.AddAvg("Game/Goal Speed", state.ball.vel.Length());
	}
}

static std::filesystem::path FindRepoRoot(const std::filesystem::path& exeDir) {
	// Release layout: <repo>/build/Release/GigaLearnBot.exe
	const std::filesystem::path candidates[] = {
		exeDir.parent_path().parent_path(),
		exeDir.parent_path(),
		exeDir,
		std::filesystem::path(R"(C:\GigaLearnRL)"), // optional clean CUDA mirror
	};
	for (const auto& root : candidates) {
		if (std::filesystem::exists(root / "autotrainer" / "orchestrator.py")
			|| std::filesystem::exists(root / "tools" / "hw_probe.py"))
			return root;
	}
	return {};
}

static void PutEnvForce(const char* key, const char* value) {
	if (!key || !value)
		return;
#ifdef _WIN32
	_putenv_s(key, value);
#else
	setenv(key, value, 1);
#endif
}

static void PutEnvIfUnset(const char* key, const char* value) {
	if (!key || !value || !key[0])
		return;
	if (std::getenv(key))
		return;
#ifdef _WIN32
	_putenv_s(key, value);
#else
	setenv(key, value, 0);
#endif
}

// Detect NVIDIA vs AMD and write hw_profile.json next to the exe.
// Prefer tools/hw_probe.py; fall back to a tiny Windows adapter probe so launching
// GigaLearnBot.exe alone (no bat) still picks CudaPower vs amd_win_20k.
static bool WriteFallbackHwProfile(const std::filesystem::path& outPath) {
	bool hasNvidia = false;
	bool hasAmd = false;
	std::string gpuName = "unknown";
#ifdef _WIN32
	{
		char buf[MAX_PATH] = {};
		hasNvidia = (SearchPathA(nullptr, "nvidia-smi.exe", nullptr, MAX_PATH, buf, nullptr) != 0);
	}
	// Adapter names via PowerShell (no admin). Quiet / short timeout.
	{
		std::string cmd =
			"powershell -NoProfile -Command \"try { "
			"(Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join '; ' "
			"} catch { '' }\"";
		FILE* pipe = _popen(cmd.c_str(), "r");
		if (pipe) {
			char line[1024] = {};
			std::string names;
			while (fgets(line, sizeof(line), pipe))
				names += line;
			_pclose(pipe);
			for (char& c : names) c = (char)std::tolower((unsigned char)c);
			if (names.find("nvidia") != std::string::npos || names.find("geforce") != std::string::npos
				|| names.find("rtx") != std::string::npos || names.find("quadro") != std::string::npos)
				hasNvidia = true;
			if (names.find("amd") != std::string::npos || names.find("radeon") != std::string::npos
				|| names.find("rx ") != std::string::npos)
				hasAmd = true;
			gpuName = names.empty() ? "unknown" : names;
			while (!gpuName.empty() && (gpuName.back() == '\n' || gpuName.back() == '\r'))
				gpuName.pop_back();
		}
	}
#else
	(void)outPath;
	return false;
#endif
	// Prefer NVIDIA if both present (laptop mux / dual GPU).
	const bool useCuda = hasNvidia;
	const bool useHip = !useCuda && hasAmd;
	const char* primary = useCuda ? "nvidia" : (useHip ? "amd" : "none");
	const char* backend = useCuda ? "cuda" : (useHip ? "hip" : "cpu");
	const char* profile = useCuda ? "power_cuda" : (useHip ? "amd_win_20k" : "cpu_fallback");
	const int arenas = useCuda ? 8192 : (useHip ? 2048 : 512);
	const int steps = useCuda ? 16 : (useHip ? 3 : 1);
	const bool forceCpu = !useCuda && !useHip;
	try {
		std::error_code ec;
		std::filesystem::create_directories(outPath.parent_path(), ec);
		std::ofstream f(outPath, std::ios::trunc);
		if (!f)
			return false;
		f << "{\n  \"schema\": 2,\n  \"source\": \"exe_fallback_probe\",\n"
			<< "  \"recommendations\": {\n"
			<< "    \"primary_gpu\": \"" << primary << "\",\n"
			<< "    \"gpu_backend\": \"" << backend << "\",\n"
			<< "    \"profile\": \"" << profile << "\",\n"
			<< "    \"cuda_available\": " << (useCuda ? "true" : "false") << ",\n"
			<< "    \"hip_available\": " << (useHip ? "true" : "false") << ",\n"
			<< "    \"gpu_sim_available\": " << ((useCuda || useHip) ? "true" : "false") << ",\n"
			<< "    \"force_cpu_sim\": " << (forceCpu ? "true" : "false") << ",\n"
			<< "    \"torch_device\": \"" << (useCuda ? "cuda" : "cpu") << "\",\n"
			<< "    \"torch_threads\": " << (useHip || forceCpu ? 10 : 0) << ",\n"
			<< "    \"num_arenas\": " << arenas << ",\n"
			<< "    \"steps_per_itr\": " << steps << ",\n"
			<< "    \"note\": \"Auto-detected at GigaLearnBot startup: " << gpuName << "\"\n"
			<< "  }\n}\n";
		RG_LOG("HW: auto-detect (fallback) primary=" << primary
			<< " backend=" << backend << " profile=" << profile
			<< " arenas=" << arenas << " - " << gpuName);
		return true;
	} catch (...) {
		return false;
	}
}

static bool RunPythonHwProbe(const std::filesystem::path& exeDir) {
	const auto root = FindRepoRoot(exeDir);
	if (root.empty())
		return false;
	const auto probePy = root / "tools" / "hw_probe.py";
	if (!std::filesystem::exists(probePy))
		return false;
	const auto outPath = exeDir / "hw_profile.json";
#ifdef _WIN32
	std::wstring cmd = L"python -u \"";
	cmd += probePy.wstring();
	cmd += L"\" --out \"";
	cmd += outPath.wstring();
	cmd += L"\" --quiet";
	STARTUPINFOW si{};
	si.cb = sizeof(si);
	si.dwFlags = STARTF_USESHOWWINDOW;
	si.wShowWindow = SW_HIDE;
	PROCESS_INFORMATION pi{};
	std::vector<wchar_t> buf(cmd.begin(), cmd.end());
	buf.push_back(L'\0');
	if (!CreateProcessW(nullptr, buf.data(), nullptr, nullptr, FALSE,
			CREATE_NO_WINDOW, nullptr, root.wstring().c_str(), &si, &pi)) {
		return false;
	}
	WaitForSingleObject(pi.hProcess, 60000);
	DWORD code = 1;
	GetExitCodeProcess(pi.hProcess, &code);
	CloseHandle(pi.hThread);
	CloseHandle(pi.hProcess);
	if (code != 0 || !std::filesystem::exists(outPath))
		return false;
	RG_LOG("HW: auto-detect via tools/hw_probe.py -> " << outPath.filename().string());
	return true;
#else
	(void)outPath;
	return false;
#endif
}

// Always refresh GPU vendor detection when training starts (NVIDIA -> CUDA POWER, AMD -> amd_win_20k).
static void EnsureAutoGpuDetect(const std::filesystem::path& exeDir) {
	if (g_NoHwProfile)
		return;
	if (const char* env = std::getenv("GIGA_NO_HW_PROFILE")) {
		if (std::atoi(env) != 0)
			return;
	}
	if (const char* skip = std::getenv("GIGA_SKIP_AUTO_HW_PROBE")) {
		if (std::atoi(skip) != 0)
			return;
	}

	RG_LOG("HW: detecting GPU (NVIDIA / AMD) automatically...");
	const auto outPath = exeDir / "hw_profile.json";
	bool ok = RunPythonHwProbe(exeDir);
	if (!ok)
		ok = WriteFallbackHwProfile(outPath);
	if (!ok) {
		RG_LOG("WARN HW: auto-detect failed - using built-in defaults "
			"(install Python for full probe, or set GIGA_ENV_ARENAS).");
		return;
	}

	// Push probe env into this process so ApplyHwProfile + LearnerConfig see NVIDIA vs AMD.
	try {
		std::ifstream f(outPath);
		nlohmann::json j;
		f >> j;
		auto rec = j.value("recommendations", nlohmann::json::object());
		auto env = rec.value("env", nlohmann::json::object());
		if (env.is_object()) {
			for (auto it = env.begin(); it != env.end(); ++it) {
				if (!it.value().is_string() && !it.value().is_number())
					continue;
				const std::string key = it.key();
				const std::string val = it.value().is_string()
					? it.value().get<std::string>()
					: it.value().dump();
				PutEnvIfUnset(key.c_str(), val.c_str());
			}
		}
		// Force vendor keys from this run (stale GIGA_* from an old shell must not stick).
		// Opt out: GIGA_LOCK_HW_PROFILE=1 keeps existing env.
		const bool lockHw = std::getenv("GIGA_LOCK_HW_PROFILE")
			&& std::atoi(std::getenv("GIGA_LOCK_HW_PROFILE")) != 0;
		const std::string backend = rec.value("gpu_backend", "");
		const std::string profile = rec.value("profile", "");
		const std::string primary = rec.value("primary_gpu", "");
		if (!lockHw) {
			if (primary == "amd" || backend == "hip" || profile == "amd_win_20k") {
				PutEnvForce("GIGA_TRAIN_PROFILE", "amd_win_20k");
				PutEnvForce("GIGA_GPU_BACKEND", backend.empty() ? "hip" : backend.c_str());
				RG_LOG("HW: AMD path selected automatically (amd_win_20k / HIP env + CPU PPO)");
			} else if (primary == "nvidia" || backend == "cuda") {
				PutEnvForce("GIGA_GPU_BACKEND", "cuda");
				PutEnvForce("GIGA_TRAIN_PROFILE", profile.empty() ? "power_cuda" : profile.c_str());
				RG_LOG("HW: NVIDIA path selected automatically (CUDA POWER)");
			} else {
				PutEnvForce("GIGA_GPU_BACKEND", "cpu");
				PutEnvForce("GIGA_TRAIN_PROFILE", "cpu_fallback");
				RG_LOG("HW: no discrete GPU detected - CPU fallback");
			}
		} else {
			RG_LOG("HW: GIGA_LOCK_HW_PROFILE=1 - keeping existing GPU env");
		}
	} catch (...) {
		RG_LOG("WARN HW: could not apply auto-detect env from hw_profile.json");
	}
}

static float ActiveOverallTargetSps() {
	if (g_AmdWin20k)
		return TrainingSize::kTargetMinOverallSps;
#ifdef GIGA_USE_CUDA_SIM
	if (!g_ForceCpuSim && !g_ProfileTorchCpu)
		return CudaPower::kTargetMinOverallSps;
#endif
	return kDefaultTargetOverallSPS;
}

// NVIDIA CUDA POWER path (not amd_win_20k / CPU-forced). Used to lock sustained Overall.
static bool IsCudaPowerRuntime() {
#ifdef GIGA_USE_CUDA_SIM
	return !g_AmdWin20k && !g_ForceCpuSim && !g_ProfileTorchCpu && !g_Pure80;
#else
	return false;
#endif
}

// Undo Apex mid-run re-arm of opp/old/skill while CUDA POWER is still in the sustained-SPS window.
static void ClampCudaPowerSustainedSps(Learner* learner) {
	if (!learner || !IsCudaPowerRuntime())
		return;
	learner->config.skillTracker.enabled = false;
	if (learner->totalTimesteps < (uint64_t)CudaPower::kSparringDeferTs) {
		learner->config.opponentPool.enabled = false;
		learner->config.opponentPool.chance = 0.f;
		learner->config.trainAgainstOldVersions = false;
		learner->config.trainAgainstOldChance = 0.f;
		learner->config.savePolicyVersions = false;
	}
}

// amd_win_20k: keep early iters pure self-play (opp/old/skill tax Overall on CPU PPO).
static void ClampAmdWin20kSustainedSps(Learner* learner) {
	if (!learner || !g_AmdWin20k)
		return;
	learner->config.skillTracker.enabled = false;
	if (learner->totalTimesteps < (uint64_t)TrainingSize::kSparringDeferTs) {
		learner->config.opponentPool.enabled = false;
		learner->config.opponentPool.chance = 0.f;
		learner->config.trainAgainstOldVersions = false;
		learner->config.trainAgainstOldChance = 0.f;
		learner->config.savePolicyVersions = false;
	}
}

static void WriteAmdAutosizeHint(int arenas, int steps) {
	if (g_AmdAutosizePath.empty())
		return;
	try {
		std::ofstream f(g_AmdAutosizePath, std::ios::trunc);
		if (!f)
			return;
		f << "GIGA_ENV_ARENAS=" << arenas << "\n";
		f << "GIGA_ENV_STEPS=" << steps << "\n";
		f << "GIGA_TRAIN_PROFILE=amd_win_20k\n";
		f << "# Auto-written because Overall SPS stayed under "
			<< (int)TrainingSize::kTargetMinOverallSps
			<< ". Restart run_fresh_train.bat to apply arenas.\n";
		RG_LOG("amd_win_20k: wrote autosize hint " << g_AmdAutosizePath.string()
			<< " (arenas=" << arenas << " steps=" << steps
			<< ") - restart train to apply arena change");
	} catch (...) {
	}
}

// Mid-run lean when Overall <20k (arenas cannot shrink live - ts/maxEp can).
static void MaybeAutoDowngradeAmdWin20k(Learner* learner, float overallSps) {
	if (!g_AmdWin20k || !learner || overallSps <= 1.f)
		return;
	if (learner->totalIterations < 3)
		return;

	if (overallSps >= TrainingSize::kTargetMinOverallSps) {
		g_AmdLowSpsStreak = 0;
		return;
	}
	++g_AmdLowSpsStreak;
	if (g_AmdLowSpsStreak < 3)
		return;
	g_AmdLowSpsStreak = 0;

	const int arenasNow = learner->config.numGames > 0 ? learner->config.numGames : TrainingSize::kArenas;
	const int minTs = std::max(4096, arenasNow * 2 * 2);

	if (g_AmdAutoDowngradeLevel == 0) {
		int64_t ts = learner->config.ppo.tsPerItr;
		int64_t neu = (int64_t)(ts * 0.75);
		if (neu < minTs)
			neu = minTs;
		if (neu < ts) {
			learner->config.ppo.tsPerItr = neu;
			learner->config.ppo.batchSize = neu;
			++g_AmdAutoDowngradeLevel;
			RG_LOG("amd_win_20k AUTO: Overall<" << (int)TrainingSize::kTargetMinOverallSps
				<< " - shrink tsPerItr " << ts << " -> " << neu
				<< " (live Cons lean; arenas unchanged until restart)");
			return;
		}
		++g_AmdAutoDowngradeLevel;
	}
	if (g_AmdAutoDowngradeLevel == 1) {
		if (learner->config.ppo.maxEpisodeDuration > 1.05) {
			learner->config.ppo.maxEpisodeDuration = 1.0;
			++g_AmdAutoDowngradeLevel;
			RG_LOG("amd_win_20k AUTO: maxEpisodeDuration -> 1.0s (dense truncates)");
			return;
		}
		++g_AmdAutoDowngradeLevel;
	}
	if (g_AmdAutoDowngradeLevel == 2) {
		const int nextA = (arenasNow > 1536) ? 1536 : ((arenasNow > 1024) ? 1024 : arenasNow);
		WriteAmdAutosizeHint(nextA, 3);
		++g_AmdAutoDowngradeLevel;
		return;
	}
	if (g_AmdAutoDowngradeLevel == 3 && arenasNow > 1024) {
		WriteAmdAutosizeHint(1024, 2);
		++g_AmdAutoDowngradeLevel;
	}
}

static void ApplyHwProfile(const std::filesystem::path& exeDir) {
	if (g_NoHwProfile)
		return;
	g_AmdAutosizePath = exeDir / "amd_win_20k_next.env";
	// Apply previous auto-downgrade arena hint before JSON (next restart after low Overall).
	if (std::filesystem::exists(g_AmdAutosizePath) && !std::getenv("GIGA_ENV_ARENAS")) {
		try {
			std::ifstream hint(g_AmdAutosizePath);
			std::string line;
			while (std::getline(hint, line)) {
				if (line.empty() || line[0] == '#')
					continue;
				const auto eq = line.find('=');
				if (eq == std::string::npos)
					continue;
				const std::string key = line.substr(0, eq);
				const std::string val = line.substr(eq + 1);
				if (key == "GIGA_ENV_ARENAS" || key == "GIGA_ENV_STEPS" || key == "GIGA_TRAIN_PROFILE") {
#ifdef _WIN32
					_putenv_s(key.c_str(), val.c_str());
#else
					setenv(key.c_str(), val.c_str(), 1);
#endif
					RG_LOG("HW: applied autosize " << key << "=" << val
						<< " from " << g_AmdAutosizePath.filename().string());
				}
			}
		} catch (...) {
		}
	}
	if (const char* env = std::getenv("GIGA_NO_HW_PROFILE")) {
		if (std::atoi(env) != 0)
			return;
	}

	if (const char* envCpu = std::getenv("GIGA_FORCE_CPU")) {
		if (std::atoi(envCpu) != 0 && !g_ExplicitCpu) {
			g_ForceCpuSim = true;
			RG_LOG("HW: GIGA_FORCE_CPU=1 -> CPU RocketSim (cudaSim=off, gpuNative=off)");
		}
	}

	// Explicit profile name from bat/probe (even before JSON read).
	if (const char* tp = std::getenv("GIGA_TRAIN_PROFILE")) {
		std::string p = tp;
		for (char& c : p) c = (char)std::tolower((unsigned char)c);
		if (p == "amd_win_20k" || p == TrainingSize::kProfileName)
			g_AmdWin20k = true;
	}
	if (const char* be = std::getenv("GIGA_GPU_BACKEND")) {
		std::string b = be;
		for (char& c : b) c = (char)std::tolower((unsigned char)c);
		if (b == "hip")
			g_AmdWin20k = true;
	}

	std::filesystem::path profilePath;
	if (const char* envP = std::getenv("GIGA_HW_PROFILE")) {
		if (envP[0])
			profilePath = envP;
	}
	if (profilePath.empty())
		profilePath = exeDir / "hw_profile.json";
	if (!std::filesystem::exists(profilePath)) {
		auto root = FindRepoRoot(exeDir);
		if (!root.empty() && std::filesystem::exists(root / "build" / "Release" / "hw_profile.json"))
			profilePath = root / "build" / "Release" / "hw_profile.json";
	}
	if (!std::filesystem::exists(profilePath)) {
		if (g_AmdWin20k) {
			// Bat already forced HIP - still lock lean knobs without JSON.
			PutEnvIfUnset("GIGA_ENV_ARENAS", std::to_string(TrainingSize::kArenas).c_str());
			PutEnvIfUnset("GIGA_ENV_STEPS", std::to_string(TrainingSize::kStepsPerItr).c_str());
			PutEnvIfUnset("GIGA_ENV_EPOCHS", std::to_string(TrainingSize::kPpoEpochs).c_str());
			PutEnvIfUnset("GIGA_ENV_MAXEP", "1.5");
			PutEnvIfUnset("GIGA_ENV_FP32", "1");
			PutEnvIfUnset("GIGA_ASYNC_OVERLAP", "0");
			PutEnvIfUnset("GIGA_TORCH_THREADS", "10");
			PutEnvIfUnset("OMP_NUM_THREADS", "10");
			PutEnvIfUnset("MKL_NUM_THREADS", "10");
			g_ProfileTorchCpu = true;
			g_AmdHipGpuNativeConfirmed = true; // bat/env claimed HIP; still verify CUDA DLLs below
			RG_LOG("HW: amd_win_20k (no hw_profile.json) - arenas="
				<< TrainingSize::kArenas << " steps=" << TrainingSize::kStepsPerItr
				<< " target Overall>=" << (int)TrainingSize::kTargetMinOverallSps);
			if (std::filesystem::exists(exeDir / "cudart64_12.dll")
				|| std::filesystem::exists(exeDir / "c10_cuda.dll")) {
				const char* be = std::getenv("GIGA_GPU_BACKEND");
				std::string b = be ? be : "";
				for (char& c : b) c = (char)std::tolower((unsigned char)c);
				if (b != "hip") {
					g_AmdHipGpuNativeConfirmed = false;
					RG_LOG("WARN amd_win_20k: CUDA DLLs present without GIGA_GPU_BACKEND=hip - "
						"rebuild tools\\build_amd.bat (docs/AMD.md).");
				}
			}
		}
		return;
	}

	try {
		std::ifstream f(profilePath);
		nlohmann::json j;
		f >> j;
		auto rec = j.value("recommendations", nlohmann::json::object());
		const bool forceCpu = rec.value("force_cpu_sim", false);
		const bool cudaAvail = rec.value("cuda_available", false);
		const bool hipAvail = rec.value("hip_available", false);
		const bool gpuSimAvail = rec.value("gpu_sim_available", cudaAvail || hipAvail);
		const int arenas = rec.value("num_arenas", -1);
		const std::string torchDev = rec.value("torch_device", "auto");
		const std::string note = rec.value("note", "");
		const std::string primary = rec.value("primary_gpu", "unknown");
		const std::string backend = rec.value("gpu_backend", cudaAvail ? "cuda" : (hipAvail ? "hip" : "cpu"));
		const std::string profileName = rec.value("profile", "");
		const int torchThreads = rec.value("torch_threads", -1);
		const bool directmlPython = rec.value("directml_python_available", false);

		RG_LOG("HW profile: " << profilePath.filename().string()
			<< " primary_gpu=" << primary
			<< " backend=" << backend
			<< " profile=" << (profileName.empty() ? "-" : profileName)
			<< " cuda=" << (cudaAvail ? "1" : "0")
			<< " hip=" << (hipAvail ? "1" : "0")
			<< " gpu_sim=" << (gpuSimAvail ? "1" : "0")
			<< " arenas_rec=" << arenas);

		if (forceCpu && !g_ExplicitCpu) {
			g_ForceCpuSim = true;
			RG_LOG("HW: CPU RocketSim fallback"
				<< (note.empty() ? "" : (std::string(" - ") + note)));
		} else if (gpuSimAvail && !g_ExplicitCpu) {
			g_ForceCpuSim = false;
			RG_LOG("HW: GPU sim path enabled (" << backend << ")"
				<< (note.empty() ? "" : (std::string(" - ") + note)));
		}
		// Only force torch CPU when profile says so - ROCm PyTorch still reports cuda.
		if (torchDev == "cpu" || torchDev == "directml")
			g_ProfileTorchCpu = true;

		// AMD HIP Windows = amd_win_20k Overall target.
		if (hipAvail || backend == "hip" || primary == "amd"
			|| profileName == "amd_win_20k" || profileName == "hip_scaled"
			|| profileName == "power_hip") {
			g_AmdWin20k = true;
		}

		// Apply recommended CPU torch threads for Win AMD PPO (libtorch CPU path).
		if (torchThreads > 0 && !std::getenv("GIGA_TORCH_THREADS")) {
#ifdef _WIN32
			_putenv_s("GIGA_TORCH_THREADS", std::to_string(torchThreads).c_str());
#else
			setenv("GIGA_TORCH_THREADS", std::to_string(torchThreads).c_str(), 0);
#endif
			RG_LOG("HW: GIGA_TORCH_THREADS=" << torchThreads << " (CPU PPO; override env to change)");
		}
		if (directmlPython && primary == "amd") {
			RG_LOG("HW: torch-directml importable in Python - NOT used by C++ Learner/PPO "
				"(docs/DIRECTML.md). Win HIP: docs/AMD.md");
		}

		if (g_AmdWin20k) {
			// Lock lean Overall knobs unless user already overrode.
			const int a = (arenas >= 128 && arenas <= 16384) ? arenas : TrainingSize::kArenas;
			PutEnvIfUnset("GIGA_ENV_ARENAS", std::to_string(a).c_str());
			PutEnvIfUnset("GIGA_ENV_STEPS", std::to_string(TrainingSize::kStepsPerItr).c_str());
			PutEnvIfUnset("GIGA_ENV_EPOCHS", std::to_string(TrainingSize::kPpoEpochs).c_str());
			PutEnvIfUnset("GIGA_ENV_MAXEP", "1.5");
			PutEnvIfUnset("GIGA_ENV_FP32", "1");
			PutEnvIfUnset("GIGA_ASYNC_OVERLAP", "0");
			PutEnvIfUnset("GIGA_TRAIN_PROFILE", TrainingSize::kProfileName);
			if (torchThreads <= 0)
				PutEnvIfUnset("GIGA_TORCH_THREADS", "10");
			{
				const char* tt = std::getenv("GIGA_TORCH_THREADS");
				if (!tt || !tt[0])
					tt = "10";
				PutEnvIfUnset("OMP_NUM_THREADS", tt);
				PutEnvIfUnset("MKL_NUM_THREADS", tt);
			}
			g_ProfileTorchCpu = true;
			RG_LOG("HW: amd_win_20k ON - target Overall>="
				<< (int)TrainingSize::kTargetMinOverallSps
				<< " (HIP env + CPU PPO lean512; AT/skill/opponents off at boot)");
			if (forceCpu || !hipAvail) {
				g_AmdHipGpuNativeConfirmed = false;
				RG_LOG("WARN amd_win_20k: HIP gpuNative NOT confirmed - Overall>=20k NOT guaranteed. "
					"Install HIP SDK + rebuild tools\\build_amd.bat (docs/AMD.md). "
					"If Overall <20k: see INIZIO_RAPIDO_IT.md / docs/AMD.md");
			} else {
				g_AmdHipGpuNativeConfirmed = true;
				RG_LOG("HW: HIP gpuNative=ON (probe) - target Overall>="
					<< (int)TrainingSize::kTargetMinOverallSps
					<< " (stretch ~" << (int)TrainingSize::kStretchOverallSps << ")");
			}
			// CUDA prebuilt on AMD PC: RocketSimCuda is NVIDIA CUDA - rebuild HIP required.
			if (std::filesystem::exists(exeDir / "cudart64_12.dll")
				|| std::filesystem::exists(exeDir / "cudart64_11.dll")
				|| std::filesystem::exists(exeDir / "c10_cuda.dll")) {
				const char* be = std::getenv("GIGA_GPU_BACKEND");
				std::string b = be ? be : "";
				for (char& c : b) c = (char)std::tolower((unsigned char)c);
				if (b != "hip") {
					g_AmdHipGpuNativeConfirmed = false;
					RG_LOG("WARN amd_win_20k: NVIDIA CUDA DLLs next to exe with AMD GPU - "
						"this build will NOT use the Radeon. Run tools\\build_amd.bat "
						"(docs/AMD.md). Overall>=20k NOT guaranteed until HIP rebuild.");
				}
			}
		}

		if (arenas >= 128 && arenas <= 16384) {
			if (!std::getenv("GIGA_ENV_ARENAS"))
				g_ProfileArenas = arenas;
		}
		if (g_AmdWin20k && g_ProfileArenas < 0 && !std::getenv("GIGA_ENV_ARENAS"))
			g_ProfileArenas = TrainingSize::kArenas;
		if (g_ProfileArenas > 0)
			RG_LOG("HW: recommended arenas=" << g_ProfileArenas
				<< " (override: TRAINING SIZE / GIGA_ENV_ARENAS)");
	} catch (const std::exception& e) {
		RG_LOG("HW: failed to read " << profilePath << " (" << e.what() << ")");
	} catch (...) {
		RG_LOG("HW: failed to read " << profilePath);
	}
}

static void MaybeStartAutoTrainer(const std::filesystem::path& exeDir) {
	if (g_NoAutoTrainer)
		return;
	if (const char* env = std::getenv("GIGA_NO_AUTOTRAINER")) {
		if (std::atoi(env) != 0) {
			RG_LOG("AutoTrainer: skipped (GIGA_NO_AUTOTRAINER=1)");
			return;
		}
	}
	if (const char* ext = std::getenv("GIGA_AUTOTRAINER_EXTERNAL")) {
		if (std::atoi(ext) != 0) {
			RG_LOG("AutoTrainer: external launcher already started (GIGA_AUTOTRAINER_EXTERNAL=1)");
			return;
		}
	}
	// Self-spawn is opt-in only.
	{
		bool allowSpawn = false;
		if (const char* at = std::getenv("GIGA_AUTOTRAINER")) {
			if (std::atoi(at) != 0)
				allowSpawn = true;
		}
		if (!allowSpawn) {
			RG_LOG("AutoTrainer: skipped (off by default; set GIGA_AUTOTRAINER=1 or use run_with_autotrainer.bat)");
			return;
		}
	}

	const auto root = FindRepoRoot(exeDir);
	if (root.empty()) {
		RG_LOG("AutoTrainer: repo root not found (orchestrator.py missing) - skipped");
		return;
	}
	const auto orch = root / "autotrainer" / "orchestrator.py";
	const auto watch = exeDir / "autotrainer";
	std::error_code ec;
	std::filesystem::create_directories(watch, ec);
	std::filesystem::create_directories(watch / "profiles", ec);
	// Best-effort copy of default config/profiles next to the exe watch dir.
	try {
		const auto srcProfiles = root / "autotrainer" / "profiles";
		if (std::filesystem::exists(srcProfiles)) {
			for (auto& ent : std::filesystem::directory_iterator(srcProfiles)) {
				if (!ent.is_regular_file()) continue;
				std::filesystem::copy_file(
					ent.path(), watch / "profiles" / ent.path().filename(),
					std::filesystem::copy_options::overwrite_existing, ec);
			}
		}
		const auto cfgSrc = root / "autotrainer" / "config.default.yaml";
		if (std::filesystem::exists(cfgSrc))
			std::filesystem::copy_file(
				cfgSrc, watch / "config.default.yaml",
				std::filesystem::copy_options::overwrite_existing, ec);
	} catch (...) {}

	const char* profile = std::getenv("GIGA_AUTOTRAINER_PROFILE");
	if (!profile || !profile[0])
		profile = "default";

#ifdef _WIN32
	// Non-blocking spawn via visible console + tee helper (same UX as launch_autotrainer.bat).
	// Never wait on the child - Learner Start must not hang if Python is slow/missing.
	const auto tee = root / "tools" / "run_autotrainer_console.py";
	std::wstring cmd;
	if (std::filesystem::exists(tee)) {
		cmd = L"cmd.exe /k set PYTHONUNBUFFERED=1&& set PYTHONIOENCODING=utf-8&& python -u \""
			+ tee.wstring() + L"\" \"" + root.wstring() + L"\" "
			+ std::wstring(profile, profile + std::strlen(profile))
			+ L" \"" + watch.wstring() + L"\"";
	} else {
		cmd = L"cmd.exe /k set PYTHONUNBUFFERED=1&& set PYTHONIOENCODING=utf-8&& python -u \""
			+ orch.wstring() + L"\" --profile "
			+ std::wstring(profile, profile + std::strlen(profile))
			+ L" --watch-dir \"" + watch.wstring() + L"\"";
	}
	std::vector<wchar_t> cmdBuf(cmd.begin(), cmd.end());
	cmdBuf.push_back(L'\0');

	STARTUPINFOW si{};
	si.cb = sizeof(si);
	PROCESS_INFORMATION pi{};
	std::wstring workDir = root.wstring();
	BOOL ok = CreateProcessW(
		nullptr,
		cmdBuf.data(),
		nullptr,
		nullptr,
		FALSE,
		CREATE_NEW_CONSOLE | CREATE_NEW_PROCESS_GROUP,
		nullptr,
		workDir.c_str(),
		&si,
		&pi
	);
	if (!ok) {
		RG_LOG("AutoTrainer: CreateProcess failed (is python on PATH?) - skipped");
		return;
	}
	const DWORD pid = pi.dwProcessId;
	CloseHandle(pi.hThread);
	CloseHandle(pi.hProcess);
	RG_LOG("AutoTrainer started PID=" << pid
		<< " profile=" << profile
		<< " watch=" << watch);
	RG_LOG("AutoTrainer: disable with GIGA_NO_AUTOTRAINER=1 or --no-autotrainer (or omit GIGA_AUTOTRAINER)");
#else
	(void)profile;
	RG_LOG("AutoTrainer: auto-spawn is Windows-only - start manually: python -m autotrainer");
#endif
}

int main(int argc, char* argv[]) {
	std::filesystem::path exeDir = std::filesystem::absolute(argv[0]).parent_path();
	// Windows often launches with cwd=system32 (or elsewhere). OpponentPool and other
	// relative assets must resolve next to the exe - pin cwd so create_directories /
	// manifest loads cannot abort after LoadVersions.
	{
		std::error_code cwdEc;
		std::filesystem::current_path(exeDir, cwdEc);
		if (cwdEc)
			RG_LOG("Warning: could not chdir to exe dir " << exeDir << " (" << cwdEc.message() << ")");
	}
	std::filesystem::path meshDir = exeDir / "collision_meshes";
	if (!std::filesystem::exists(meshDir))
		RG_ERR_CLOSE("collision_meshes not found at " << meshDir);
	RocketSim::Init(meshDir.string());

	LearnerConfig cfg = {};
	bool renderMode = false;
	bool rlbotMode = false;
	std::filesystem::path checkpointOverride;
	for (int i = 1; i < argc; i++) {
		std::string arg = argv[i];
		if (arg == "--render") {
			cfg.sendMetrics = false;
			cfg.ppo.deterministic = true;
			cfg.renderMode = true;
			renderMode = true;
		} else if (arg == "--metrics") {
			// Full Leak-parity local metrics (console Report each iter). Does not require wandb.
			g_FullMetrics = true;
		} else if (arg == "--wandb") {
			// wandb via MetricSender (needs creds). Implies full metrics.
			g_SendWandb = true;
			g_FullMetrics = true;
		} else if (arg == "--verbose") {
			g_Verbose = true;
		} else if (arg == "--continuous" || arg == "--quality"
			|| arg == "--transfer-discrete" || arg == "--attention"
			|| arg == "--no-auto-tl") {
			RG_ERR_CLOSE("Unsupported flag: " << arg
				<< " (discrete only; continuous/attention not wired)");
		} else if (arg == "--teacher" && i + 1 < argc) {
			g_TeacherPath = argv[++i];
		} else if (arg == "--teachers" && i + 1 < argc) {
			// Semi-colon or comma separated roots for multi-teacher discovery.
			std::string list = argv[++i];
			for (char& c : list) {
				if (c == ',') c = ';';
			}
			std::stringstream ss(list);
			std::string part;
			while (std::getline(ss, part, ';')) {
				if (part.empty()) continue;
				if (g_TeacherPath.empty())
					g_TeacherPath = part;
				g_TeacherPaths.push_back(part);
			}
		} else if (arg == "--apex") {
			g_ApexCurriculum = true;
		} else if (arg == "--no-apex") {
			g_ForceNoApex = true;
			g_ApexCurriculum = false;
		} else if (arg == "--rlbot") {
			rlbotMode = true;
		} else if (arg == "--hyperpower") {
			// Dense BallChase curriculum path (not default; lower Overall than blank POWER path).
			g_Hyperpower = true;
			g_DefaultEnv = false;
			g_Pure80 = false;
			g_ApexCurriculum = true;
		} else if (arg == "--cpu") {
			// Force CPU RocketSim even when CUDA sim is compiled in.
			g_ForceCpuSim = true;
			g_ExplicitCpu = true;
		} else if (arg == "--no-autotrainer") {
			g_NoAutoTrainer = true;
		} else if (arg == "--no-hw-profile") {
			g_NoHwProfile = true;
		} else if (arg == "--pure80") {
			// Optional SPS-only benchmark (1-step truncate). Not normal training.
			g_Hyperpower = true;
			g_Pure80 = true;
			g_DefaultEnv = false;
			g_ForceCpuSim = false;
		} else if (arg == "--lean-resume") {
			// Resume old lean box (checkpoints_default_lean). Not the default blank train path.
			g_DefaultEnv = true;
			g_FromScratch = false;
			g_Pure80 = false;
			g_Hyperpower = false;
			g_ForceCpuSim = false;
		} else if (arg == "--continue-leak" || arg == "--like-leak"
			|| arg == "--max-learn" || arg == "--resume-leak") {
			// Continue Leak 1v1: 1024+LN into checkpoints_default + POWER train
			// (8192/CUDA/Apex). Classic Leak: GIGA_ENV_* + optional GIGA_NO_APEX=1.
			g_CliContinueLeak = true;
			g_MaxLearn = true;
			g_FromScratch = false;
			g_DefaultEnv = true;
			g_Pure80 = false;
			g_Hyperpower = false;
			g_ForceCpuSim = false;
			g_ForceNoApex = false;
			g_ApexCurriculum = true;
		} else if (arg == "--from-scratch" || arg == "--fresh") {
			// Explicit alias of default: POWER normal into empty checkpoints/.
			// Never loads checkpoints_default / checkpoints_default_lean.
			g_CliFromScratch = true;
			g_FromScratch = true;
			g_MaxLearn = false;
			g_DefaultEnv = true;
			g_Pure80 = false;
			g_Hyperpower = false;
			g_ForceCpuSim = false;
			g_ForceNoApex = false;
		} else if (arg == "--legacy-hyperpower") {
			// Old Cap-augmented gate path (not pure).
			g_Hyperpower = true;
			g_Pure80 = false;
			g_DefaultEnv = false;
		} else if (arg == "--checkpoint" && i + 1 < argc) {
			checkpointOverride = argv[++i];
		} else if (arg == "--seed" && i + 1 < argc) {
			g_RandomSeed = (int64_t)std::atoll(argv[++i]);
		}
	}
	if (const char* envSeed = std::getenv("GIGA_SEED")) {
		if (g_RandomSeed < 0)
			g_RandomSeed = (int64_t)std::atoll(envSeed);
	}
	if (const char* envNoApex = std::getenv("GIGA_NO_APEX")) {
		if (std::atoi(envNoApex) != 0)
			g_ForceNoApex = true;
	}
	if (const char* envFS = std::getenv("GIGA_FROM_SCRATCH")) {
		// Env only if CLI did not pick a mode (stale GIGA_FROM_SCRATCH must not beat --continue-leak).
		if (!g_CliContinueLeak && !g_CliFromScratch)
			g_FromScratch = std::atoi(envFS) != 0;
	}
	if (const char* envCL = std::getenv("GIGA_CONTINUE_LEAK")) {
		if (std::atoi(envCL) != 0 && !g_CliFromScratch) {
			g_MaxLearn = true;
			g_FromScratch = false;
		}
	}
	if (const char* envML = std::getenv("GIGA_MAX_LEARN")) {
		if (std::atoi(envML) != 0 && !g_CliFromScratch) {
			g_MaxLearn = true;
			g_FromScratch = false;
		}
	}
	if (const char* envV = std::getenv("GIGA_VERBOSE")) {
		if (std::atoi(envV) != 0)
			g_Verbose = true;
	}
	// CLI / continue-leak wins over from-scratch when both somehow set.
	if (g_CliContinueLeak || (g_MaxLearn && !g_CliFromScratch)) {
		g_FromScratch = false;
		g_MaxLearn = true;
	} else if (g_CliFromScratch || g_FromScratch) {
		g_FromScratch = true;
		g_MaxLearn = false;
		g_DefaultEnv = true;
		g_Pure80 = false;
		g_Hyperpower = false;
		g_ForceNoApex = false;
	}
	// --cpu must survive later mode flags that clear g_ForceCpuSim.
	if (g_ExplicitCpu)
		g_ForceCpuSim = true;
	if (g_MaxLearn) {
		g_DefaultEnv = true;
		g_Pure80 = false;
		g_Hyperpower = false;
		// POWER continue-leak: full Apex (docs). Classic Leak self-play: GIGA_NO_APEX=1.
		if (!g_ForceNoApex)
			g_ApexCurriculum = true;
		// Leak-style Report.Display each iter (same field order/labels). Opt out: GIGA_SKIP_METRICS=1.
		// wandb still requires --wandb (Leak local also keeps sendMetrics=false).
		if (!renderMode) {
			bool skipMetrics = false;
			if (const char* envSM = std::getenv("GIGA_SKIP_METRICS"))
				skipMetrics = std::atoi(envSM) != 0;
			if (!skipMetrics)
				g_FullMetrics = true;
		}
	}
	// From-scratch: Leak-style Report.Display each iter when wired. Opt out: GIGA_SKIP_METRICS=1.
	if (g_FromScratch && !renderMode && !g_FullMetrics) {
		bool skipMetrics = false;
		if (const char* envSM = std::getenv("GIGA_SKIP_METRICS"))
			skipMetrics = std::atoi(envSM) != 0;
		if (!skipMetrics)
			g_FullMetrics = true;
	}
	// Normal/from-scratch: Apex SPS-safe. Continue-leak: full Apex (already set above).
	// Opt out: --no-apex / GIGA_NO_APEX=1.
	if (!g_MaxLearn && g_DefaultEnv && !g_ForceNoApex && !g_Pure80 && !renderMode && !rlbotMode) {
		g_ApexCurriculum = true;
	}

	// Hardware probe: auto-detect NVIDIA vs AMD, then apply profile (CUDA POWER / amd_win_20k).
	// Works when launching GigaLearnBot.exe directly - no bat / manual GPU flags required.
	if (!renderMode && !rlbotMode) {
		EnsureAutoGpuDetect(exeDir);
		ApplyHwProfile(exeDir);
	}

	if (rlbotMode) {
		std::filesystem::path modelsFolder = checkpointOverride;
		if (modelsFolder.empty())
			modelsFolder = FindLatestValidCheckpoint(exeDir / "checkpoints");
		if (modelsFolder.empty())
			RG_ERR_CLOSE("RLBot: No valid checkpoint in " << (exeDir / "checkpoints"));

		RG_LOG("RLBot: Loading checkpoint " << modelsFolder << "...");

		const int tickSkip = g_Hyperpower ? 6 : 8;
		const int actionDelay = tickSkip - 1;
		const int obsSize = GetTrainingObsSize();

		PartialModelConfig policyConfig = {};
		FillDiscretePolicyConfig(policyConfig);
		PartialModelConfig sharedConfig = {};
		sharedConfig.layerSizes = { 512 };
		sharedConfig.optimType = ModelOptimType::ADAM;
		sharedConfig.activationType = ModelActivationType::RELU;
		sharedConfig.addLayerNorm = false;
		sharedConfig.addOutputLayer = false;

		auto* obsBuilder = MakeObsBuilder();
		auto* actionParser = new DefaultAction();

		PolicyType deployPolicy = PolicyType::DISCRETE;
		auto* inferUnit = new InferUnit(
			obsBuilder, obsSize, actionParser,
			sharedConfig, policyConfig, modelsFolder, true,
			deployPolicy, 0.1f, 1.0f, 8,
			nullptr,
			nullptr
		);

		RLBotParams rlParams = {};
		rlParams.port = ReadRLBotPort(exeDir);
		rlParams.tickSkip = tickSkip;
		rlParams.actionDelay = actionDelay;
		rlParams.deterministic = true;
		rlParams.inferUnit = inferUnit;

		RG_LOG("RLBot: Listening on port " << rlParams.port << " (tickSkip=" << tickSkip << ", obsSize=" << obsSize << ")");
		RLBotClient::Run(rlParams);
		return EXIT_SUCCESS;
	}

	TrainingCurriculum::chaseEnvCreateFn = EnvCreateTraining;
	TrainingCurriculum::advancedEnvCreateFn = EnvCreateTraining;
	TrainingCurriculum::config.ppoMiniBatchSize = g_Hyperpower ? 25'000 : 50'000;
	TrainingCurriculum::config.chaseSkillTracker = false;
	TrainingCurriculum::config.foundationSkillTracker = false;

	cfg.onIterationStart = [](Learner* learner) {
		AutoTrainerBridge::ProcessCommands(learner);
		AutoTrainerBridge::OnIterationStart(learner);
		if (!g_DefaultEnv)
			TrainingCurriculum::OnIterationStart(learner);
		// Normal (default) uses SPS-safe Apex; --max-learn uses full Apex.
		if (g_ApexCurriculum && !g_Pure80)
			ApplyApexCurriculum(learner, nullptr, /*spsSafe=*/g_DefaultEnv && !g_MaxLearn);
		// SSL AutoTrainer guide: league 60/30/10 + GPU reset mix must win over Apex when flagged.
		AutoTrainerBridge::ReapplyPostApex(learner);
		// CUDA POWER: keep early iters pure self-play (Apex would re-enable opp/old every iter).
		ClampCudaPowerSustainedSps(learner);
		// amd_win_20k: same - CPU PPO cannot afford early league/skill tax.
		ClampAmdWin20kSustainedSps(learner);
		if (TrainingCurriculum::config.sslGuidePostApex || g_DefaultEnv) {
			float ko = TrainingCurriculum::config.kickoffWeight;
			float fz = TrainingCurriculum::config.fuzzedWeight;
			float ae = TrainingCurriculum::config.aerialWeight;
			float s = ko + fz + ae;
			if (s > 1e-6f && TrainingCurriculum::config.sslGuidePostApex) {
				ko /= s; fz /= s; ae /= s;
				SetDefaultStateMix(learner, ko, fz, ae);
				SetGpuResetCurriculum(learner, ko, fz, ae);
			}
		}
		if (g_Pure80 || (g_Hyperpower && !g_ApexCurriculum
			&& TrainingCurriculum::currentPhase == TrainingPhase::CHASE)) {
			learner->config.skillTracker.enabled = false;
			learner->config.savePolicyVersions = false;
			if (!g_ApexCurriculum) {
				learner->config.trainAgainstOldChance = 0.f;
				learner->config.opponentPool.enabled = false;
			}
		}
	};
	// Flush AT GPU reward/terminal/distill/LR updates deferred while Collect||Learn held CUDA.
	// Also SyncRuntimePPOConfig + deferred Save/Load (see FlushDeferredCuda).
	cfg.onAfterAsyncLearnJoin = [](Learner* learner) {
		AutoTrainerBridge::FlushDeferredCuda(learner);
	};
	cfg.onIterationComplete = [](Learner* learner, const Report& report) {
		float ourSPS = 0.f, collectionSPS = 0.f, consumptionSPS = 0.f;
		auto it = report.data.find("Overall Steps/Second");
		if (it != report.data.end()) ourSPS = (float)it->second;
		auto itC = report.data.find("Collection Steps/Second");
		if (itC != report.data.end()) collectionSPS = (float)itC->second;
		auto itL = report.data.find("Consumption Steps/Second");
		if (itL != report.data.end()) consumptionSPS = (float)itL->second;

		{
			auto itR = report.data.find("Rating/1v1");
			if (itR == report.data.end())
				itR = report.data.find("Rating/2v2");
			if (itR != report.data.end())
				g_LastSkillRating1v1 = (float)itR->second;
		}

		if (g_DefaultEnv) {
			const float targetSps = ActiveOverallTargetSps();
			bool met = ourSPS >= targetSps;
			Report powered = report;
			powered["Power/PureOverallSPS"] = ourSPS;
			powered["Power/TargetOverallSPS"] = targetSps;
			powered["Power/CollectionSPS"] = collectionSPS;
			powered["Power/ConsumptionSPS"] = consumptionSPS;
			const bool learnOn = g_ApexCurriculum;
			// AMD Overall warning (not only --verbose).
			if (g_AmdWin20k && ourSPS > 1.f && ourSPS < TrainingSize::kTargetMinOverallSps
				&& learner && learner->totalIterations >= 2
				&& (learner->totalIterations <= 8 || (learner->totalIterations % 10) == 0)) {
				RG_LOG("WARN amd_win_20k: OverallSPS=" << ourSPS
					<< " < " << (int)TrainingSize::kTargetMinOverallSps
					<< " (Coll=" << collectionSPS << " Cons=" << consumptionSPS << "). "
					<< (g_AmdHipGpuNativeConfirmed
						? "Auto-lean may shrink ts/maxEp; if still low restart with "
						  "GIGA_ENV_ARENAS=1536 (or check amd_win_20k_next.env). "
						: "HIP gpuNative NOT confirmed - rebuild tools\\build_amd.bat. ")
					<< "See INIZIO_RAPIDO_IT.md / docs/AMD.md");
			}
			if (g_AmdWin20k)
				MaybeAutoDowngradeAmdWin20k(learner, ourSPS);
			// Report.Display each iter; extra banners with --verbose.
			if (g_Verbose) {
				if (g_MaxLearn) {
					RG_LOG("CONTINUE-LEAK train: OverallSPS=" << ourSPS
						<< " Coll=" << collectionSPS
						<< " Cons=" << consumptionSPS
						<< " [Leak knobs: oldChance=" << (learner ? learner->config.trainAgainstOldChance : 0.f)
						<< " maxEp=" << (learner ? learner->config.ppo.maxEpisodeDuration : 0.0) << "s]");
				} else {
					RG_LOG("NORMAL train: OverallSPS=" << ourSPS
						<< " Coll=" << collectionSPS
						<< " Cons=" << consumptionSPS
						<< " need>=" << targetSps
						<< (met ? " [TARGET MET]" : (g_AmdWin20k ? " [pushing >=20k Overall]" : " [pushing >=400k Overall]"))
						<< (learnOn ? " [curriculum+opponents+teachers ON]" : "")
						<< (g_AmdWin20k ? " [amd_win_20k]" : ""));
				}
				if (learner && (learner->totalIterations % 10 == 0) && (g_MaxLearn || learnOn)) {
					RG_LOG((g_MaxLearn ? "CONTINUE-LEAK" : "NORMAL") << " learn: oldChance="
						<< learner->config.trainAgainstOldChance
						<< " oppChance=" << learner->config.opponentPool.chance
						<< " beat=" << learner->config.opponentPool.beatBonus
						<< " teachers=" << g_TeacherPaths.size()
						<< " skill1v1=" << g_LastSkillRating1v1
						<< " entropy=" << learner->config.ppo.entropyScale
						<< " maxEp=" << learner->config.ppo.maxEpisodeDuration << "s"
						<< " epochs=" << learner->config.ppo.epochs);
				}
			}
			AutoTrainerBridge::WriteStatus(learner, powered);
			return;
		}

		// Hyperpower / non-default path: SPS only.
		Report powered = report;
		powered["Power/PureOverallSPS"] = ourSPS;
		powered["Power/CollectionSPS"] = collectionSPS;
		powered["Power/ConsumptionSPS"] = consumptionSPS;
		if (g_Verbose) {
			RG_LOG("train: OverallSPS=" << ourSPS
				<< " Coll=" << collectionSPS
				<< " Cons=" << consumptionSPS);
		}
		AutoTrainerBridge::WriteStatus(learner, powered);
	};

	// AUTO uses CUDA when libtorch sees a GPU; CPU when hw_profile says no CUDA (AMD/etc.).
	// Avoid hard GPU_CUDA require so AMD boxes do not abort at Learner ctor.
	// DirectML: GIGA_TORCH_DIRECTML / device=directml -> CPU (libtorch has no DirectML; see docs/AMD.md).
	if (g_ProfileTorchCpu || g_ForceCpuSim)
		cfg.deviceType = LearnerDeviceType::AUTO;
	else
		cfg.deviceType = LearnerDeviceType::GPU_CUDA;
	if (const char* envDev = std::getenv("GIGA_TORCH_DEVICE")) {
		std::string d = envDev;
		for (char& c : d) c = (char)std::tolower((unsigned char)c);
		if (d == "cpu")
			cfg.deviceType = LearnerDeviceType::CPU;
		else if (d == "cuda" || d == "gpu")
			cfg.deviceType = LearnerDeviceType::GPU_CUDA;
		else if (d == "directml" || d == "dml" || d == "privateuse1"
			|| d.find("privateuse") != std::string::npos) {
			cfg.deviceType = LearnerDeviceType::CPU;
			RG_LOG("GIGA_TORCH_DEVICE=" << d
				<< " -> CPU PPO (no C++ LibTorch-DirectML; pip torch-directml does NOT speed GigaLearnBot). "
				<< "Win HIP: docs/AMD.md / docs/DIRECTML.md");
		} else if (d == "auto")
			cfg.deviceType = LearnerDeviceType::AUTO;
	}
	if (const char* dml = std::getenv("GIGA_TORCH_DIRECTML")) {
		if (dml[0] == '1' || dml[0] == 't' || dml[0] == 'T' || dml[0] == 'y' || dml[0] == 'Y') {
			cfg.deviceType = LearnerDeviceType::CPU;
			RG_LOG("GIGA_TORCH_DIRECTML=1 -> CPU PPO (DirectML not in C++ stack; docs/DIRECTML.md + AMD_WSL2.md)");
		}
	}
	// Separate checkpoint boxes so modes never clash on load.
	if (g_DefaultEnv)
		cfg.checkpointFolder = exeDir / "checkpoints_default";
	else
		cfg.checkpointFolder = exeDir / (g_Hyperpower ? "checkpoints_hyperpower" : "checkpoints_quality");
	cfg.checkpointsToKeep = 24;
	cfg.randomSeed = 123;
	cfg.addRewardsToMetrics = false;

	auto optim = ModelOptimType::ADAM;
	auto activation = ModelActivationType::RELU;

	if (g_DefaultEnv) {
		if (g_FromScratch) {
			RG_LOG("=== FROM-SCRATCH 1v1 TRAIN (POWER normal, empty checkpoints/) ===");
			RG_LOG("Note: blank reward stack + empty ckpt box - new model at iter 0.");
		} else if (g_MaxLearn) {
			RG_LOG("=== CONTINUE-LEAK (1024+LN + POWER train, Leak console) ===");
			RG_LOG("Note: --render is view-only (same as Leak). Train with --continue-leak; watch with a separate --render process.");
			RG_LOG("Classic Leak batches: GIGA_ENV_ARENAS=2056 GIGA_ENV_TS=500000 GIGA_ENV_MAXEP=120 GIGA_ENV_EPOCHS=2");
		} else {
			RG_LOG("=== NORMAL 1v1 TRAIN (blank template, Overall >= 400k target) ===");
		}
		cfg.tickSkip = 6;
		cfg.actionDelay = cfg.tickSkip - 1; // Leak
		cfg.ppo.policyType = PolicyType::DISCRETE;
#ifdef GIGA_USE_CUDA_SIM
		cfg.useCudaSim = !g_ForceCpuSim;
		// GPU-native maps default state weights onto device IDs (framework power underneath Leak knobs).
		// Exact Leak CPU rewards: add --cpu (much slower).
		cfg.cudaPreferGpuNative = !g_ForceCpuSim;
		cfg.cudaRewardProfile = 1;
		cfg.cudaNoTouchSeconds = 10.f; // Leak NoTouchCondition(10); CUDA POWER overrides below
#else
		cfg.useCudaSim = false;
#endif
		// TRAINING SIZE knobs (top of this file) -> LearnerConfig. See namespace TrainingSize.
		// Force-truncate at tsPerItr (Learner) + maxEp wall - never wait for full episode waves.
		// NVIDIA CUDA normal: CudaPower{} (8192/8/1.0s). AMD HIP: TrainingSize{} (2048/4/1.5s).
		const bool cudaPower =
#ifdef GIGA_USE_CUDA_SIM
			!g_AmdWin20k && !g_ForceCpuSim && !g_ProfileTorchCpu;
#else
			false;
#endif
#ifdef GIGA_USE_CUDA_SIM
		// CUDA POWER: disable GPU no-touch terminals. At 8192 arenas, a synchronized 10s
		// no-touch wave calls ResetArenax8k (~4s Env hitch every ~12 iters); staggering
		// that wave spreads ResetArena cost into every step and tanks sustained Overall
		// to ~350k. Host maxEp truncate + goals still end episodes; amd_win_20k keeps 10s.
		if (cudaPower && cfg.useCudaSim)
			cfg.cudaNoTouchSeconds = 0.f;
#endif
		int stepsPerItr = cudaPower ? CudaPower::kStepsPerItr : TrainingSize::kStepsPerItr;
		if (!TrainingSize::kForceTrainingSize && !g_FromScratch && !g_MaxLearn && !cudaPower && !g_AmdWin20k)
			stepsPerItr = 4; // lean-resume legacy bank (non-AMD)
		int epochs = cudaPower ? CudaPower::kPpoEpochs : TrainingSize::kPpoEpochs;
		double maxEp = cudaPower ? CudaPower::kMaxEpisodeSeconds : TrainingSize::kMaxEpisodeSeconds;
		if (!TrainingSize::kForceTrainingSize && g_MaxLearn)
			maxEp = 3.0; // continue-leak POWER wall unless forced
		cfg.numGames = renderMode ? 512
			: (cudaPower ? CudaPower::kArenas : TrainingSize::kArenas);
		int tsPerItr = (TrainingSize::kTsPerItr > 0)
			? TrainingSize::kTsPerItr
			: (cfg.numGames * 2 * stepsPerItr);

		if (!TrainingSize::kForceTrainingSize) {
			if (const char* envS = std::getenv("GIGA_ENV_STEPS")) {
				int v = std::atoi(envS);
				if (v >= 1 && v <= 64) stepsPerItr = v;
			}
			if (const char* envA = std::getenv("GIGA_ENV_ARENAS")) {
				int v = std::atoi(envA);
				if (v >= 256 && v <= 16384) cfg.numGames = v;
			} else if (!cudaPower && TrainingSize::kUseHwProbeArenas
				&& g_ProfileArenas >= 128 && g_ProfileArenas <= 16384) {
				// hw_probe may recommend 8192 on NVIDIA; keep AMD TrainingSize when amd_win_20k.
				// CUDA POWER locks CudaPower::kArenas (do not let a stale probe shrink the bank).
				cfg.numGames = g_ProfileArenas;
			}
			// Keep ts bank proportional after arena/hw/step scaling (unless GIGA_ENV_TS / kTsPerItr).
			if (TrainingSize::kTsPerItr > 0)
				tsPerItr = TrainingSize::kTsPerItr;
			else if (!std::getenv("GIGA_ENV_TS"))
				tsPerItr = cfg.numGames * 2 * stepsPerItr;
			if (const char* envT = std::getenv("GIGA_ENV_TS")) {
				int v = std::atoi(envT);
				if (v >= 8192 && v <= 4'000'000) tsPerItr = v;
			}
			if (const char* envEpochs = std::getenv("GIGA_ENV_EPOCHS")) {
				int v = std::atoi(envEpochs);
				if (v >= 1 && v <= 8) epochs = v;
			}
			if (const char* envE = std::getenv("GIGA_ENV_MAXEP")) {
				double v = std::atof(envE);
				if (v >= 0.25 && v <= 120.0) maxEp = v;
			}
		} else if (TrainingSize::kTsPerItr <= 0) {
			tsPerItr = cfg.numGames * 2 * stepsPerItr;
		}
		cfg.ppo.tsPerItr = tsPerItr;
		cfg.ppo.batchSize = tsPerItr;
		cfg.ppo.epochs = epochs;
		if (TrainingSize::kMiniBatchSize > 0) {
			cfg.ppo.miniBatchSize = TrainingSize::kMiniBatchSize;
			cfg.ppo.skipExperienceShuffle = (epochs < 2);
		} else if (epochs >= 2) {
			// Leak: miniBatchSize = 25'000 with sample reuse.
			int mb = 25'000;
			if (tsPerItr % mb != 0) {
				mb = 8192;
				for (int cand = 32768; cand >= 8192; cand /= 2) {
					if (tsPerItr % cand == 0) { mb = cand; break; }
				}
				if (tsPerItr % mb != 0)
					mb = tsPerItr;
			}
			cfg.ppo.miniBatchSize = mb;
			cfg.ppo.skipExperienceShuffle = false;
		} else {
			cfg.ppo.miniBatchSize = tsPerItr;
			cfg.ppo.skipExperienceShuffle = true;
		}
		// Leak entropy on continue-leak; slightly higher on from-scratch/normal POWER.
		cfg.ppo.entropyScale = g_MaxLearn ? 0.020f : 0.024f;
		cfg.ppo.gaeGamma = 0.99730f;
		cfg.ppo.gaeLambda = 0.975f;
		cfg.ppo.policyLR = 1e-4f;
		cfg.ppo.criticLR = 1e-4f;
		// Guide/Kue: with DefaultAction masking, entropy must be over valid actions only
		// (otherwise entropyScale wastes exploration on impossible flips / dodges).
		cfg.ppo.maskEntropy = true;
		cfg.ppo.useBF16Autocast = true;
		// Half + bf16 on CUDA POWER paths. amd_win_20k / CPU Infer+Learn: FP32 (half on host is slow).
		cfg.ppo.useHalfPrecision = !g_AmdWin20k && !g_ProfileTorchCpu;
		if (const char* envFp = std::getenv("GIGA_ENV_FP32")) {
			if (std::atoi(envFp) != 0)
				cfg.ppo.useHalfPrecision = false;
		}
		cfg.ppo.clipRange = 0.2f;
		cfg.ppo.useAttentionHead = false;
		cfg.ppo.esNoiseScale = 0.f;
		cfg.ppo.prioritySampling = false;
		// Leak has no eventAdvantageBoost - leave 1.0 on continue-leak.
		cfg.ppo.eventAdvantageBoost = g_MaxLearn ? 1.0f : 1.18f;
		if (const char* envEv = std::getenv("GIGA_ENV_EVENT")) {
			float v = (float)std::atof(envEv);
			if (v >= 1.f && v <= 2.5f) cfg.ppo.eventAdvantageBoost = v;
		}
		cfg.ppo.maxGradNorm = 0.f;
		cfg.ppo.skipPPOMetrics = true;
		cfg.ppo.overbatching = false;
		cfg.ppo.maxEpisodeDuration = maxEp;
		cfg.sendMetrics = false;
		cfg.addRewardsToMetrics = false;
		cfg.skipObsIntegrityChecks = true;
		cfg.rewardSampleRandInterval = 16; // Leak
#ifdef GIGA_USE_CUDA_SIM
		// Stable fast path on Win AMD: no Collect||Learn overlap (CPU Infer+Learn + HIP env).
		if (g_AmdWin20k) {
			cfg.cudaDeviceExperience = false;
			cfg.asyncLearnOverlapMode = 0;
			cfg.asyncLearnOverlap = false;
		} else if (cfg.useCudaSim && cudaPower) {
			// Normal CUDA: host multi-step bank (real GAE). deviceXP stays pure80/1-step.
			// Async overlap needs deviceXP - leave auto (no-ops without it).
			cfg.cudaDeviceExperience = false;
			cfg.asyncLearnOverlapMode = 2;
			cfg.asyncLearnOverlap = false;
			cfg.asyncOverlapMaxArenas = 4096;
			// Sustained median: defer Save hitches + hard-off state-ring (AT cannot re-arm without env).
			cfg.tsPerSave = CudaPower::kTsPerSave;
			PutEnvIfUnset("GIGA_STATE_RING", "0");
			if (const char* envO = std::getenv("GIGA_ASYNC_OVERLAP")) {
				std::string v(envO);
				for (char& c : v) c = (char)tolower((unsigned char)c);
				if (v == "1" || v == "on" || v == "true") {
					// Opt-in: 1-step device bank + Collect||Learn (SPS; changes GAE to TD(0)).
					cfg.ppo.maxEpisodeDuration = 0.05;
					maxEp = 0.05;
					cfg.cudaDeviceExperience = true;
					cfg.asyncLearnOverlapMode = 1;
					RG_LOG("CUDA POWER: GIGA_ASYNC_OVERLAP=1 -> deviceXP + 1-step bank + Collect||Learn");
				} else if (v == "0" || v == "off" || v == "false") {
					cfg.asyncLearnOverlapMode = 0;
				}
			}
		}
#endif
		// Default lean 512 MLP hits >=400k Overall on CUDA; amd_win_20k uses same lean net for >=20k.
		// --continue-leak always forces Leak 1024+LN.
		// --from-scratch / --fresh always lean 512 into empty checkpoints/.
		bool leanNet = !g_MaxLearn;
		if (g_FromScratch) {
			leanNet = true;
		} else if (!g_MaxLearn) {
			if (const char* envL = std::getenv("GIGA_ENV_LEAN"))
				leanNet = std::atoi(envL) != 0;
		} else {
			leanNet = false; // Leak 1024+LN (never lean-512 for checkpoint load)
		}
		if (leanNet) {
			cfg.ppo.sharedHead.layerSizes = { 512 };
			cfg.ppo.policy.layerSizes = { 512, 512 };
			cfg.ppo.critic.layerSizes = { 512, 512 };
			cfg.ppo.policy.addLayerNorm = false;
			cfg.ppo.critic.addLayerNorm = false;
			cfg.ppo.sharedHead.addLayerNorm = false;
			cfg.checkpointFolder = exeDir / "checkpoints_default_lean";
		} else {
			// Leak: shared {1024,1024}, policy/critic {1024,1024,1024}, LayerNorm, Adam, ReLU
			cfg.ppo.sharedHead.layerSizes = { 1024, 1024 };
			cfg.ppo.policy.layerSizes = { 1024, 1024, 1024 };
			cfg.ppo.critic.layerSizes = { 1024, 1024, 1024 };
			cfg.ppo.policy.addLayerNorm = true;
			cfg.ppo.critic.addLayerNorm = true;
			cfg.ppo.sharedHead.addLayerNorm = true;
			cfg.checkpointFolder = exeDir / "checkpoints_default";
		}
		if (g_FromScratch) {
			// Neutral blank box - never touches Leak/lean dirs. Empty -> new model; else resumes this box only.
			const char* ckptName = TrainingSize::kCheckpointDirName;
			if (!ckptName || !*ckptName) ckptName = "checkpoints";
			cfg.checkpointFolder = exeDir / ckptName;
			std::error_code mkEc;
			std::filesystem::create_directories(cfg.checkpointFolder, mkEc);
			if (mkEc)
				RG_LOG("Warning: could not create " << cfg.checkpointFolder << " (" << mkEc.message() << ")");
			bool hasCkpt = false;
			std::error_code itEc;
			if (std::filesystem::exists(cfg.checkpointFolder)) {
				for (auto& e : std::filesystem::directory_iterator(cfg.checkpointFolder, itEc)) {
					if (!e.is_directory()) continue;
					if (std::filesystem::exists(e.path() / "POLICY.lt")
						&& std::filesystem::exists(e.path() / "SHARED_HEAD.lt")) {
						hasCkpt = true;
						break;
					}
				}
			}
			RG_LOG("FROM-SCRATCH ckpt box: " << cfg.checkpointFolder
				<< (hasCkpt ? " (resume existing ckpts)" : " (empty -> start new model)"));
		}
		const char* modeTag = g_FromScratch ? "FROM-SCRATCH" : (g_MaxLearn ? "CONTINUE-LEAK" : "NORMAL");
		if (g_Verbose) {
			RG_LOG(modeTag << " cfg: arenas=" << cfg.numGames
				<< " stepsPerItr=" << stepsPerItr
				<< " tsPerItr=" << tsPerItr
				<< " epochs=" << epochs
				<< " miniBatch=" << cfg.ppo.miniBatchSize
				<< " cudaSim=" << (cfg.useCudaSim ? "on" : "off")
				<< " gpuNativeDefaultRewards=" << (cfg.useCudaSim && cfg.cudaPreferGpuNative ? "1" : "0")
				<< " maxEpDur=" << maxEp << "s"
				<< " entropy=" << cfg.ppo.entropyScale
				<< " leanNet=" << (leanNet ? "1" : "0")
				<< " apex=" << (g_ApexCurriculum ? "sps-safe" : "off")
				<< " metrics=" << (g_FullMetrics ? "leak-iter" : "skip")
				<< " ckpt=" << cfg.checkpointFolder);
		} else {
			RG_LOG(modeTag
				<< ": arenas=" << cfg.numGames
				<< " tsPerItr=" << tsPerItr
				<< " epochs=" << epochs
				<< " maxEp=" << maxEp << "s"
				<< (cfg.useCudaSim ? " CUDA" : " CPU")
				<< (leanNet ? " lean512" : " 1024+LN")
				<< (g_AmdWin20k ? " amd_win_20k" : "")
				<< (g_ApexCurriculum ? " Apex=sps-safe" : (g_MaxLearn ? " like-Leak" : ""))
				<< (g_FullMetrics ? " metrics=on" : "")
				<< " ckpt=" << cfg.checkpointFolder.filename().string());
			if (g_AmdWin20k) {
				RG_LOG("amd_win_20k: target Overall>="
					<< (int)TrainingSize::kTargetMinOverallSps
					<< " with HIP gpuNative + CPU PPO (threads via GIGA_TORCH_THREADS). "
					<< "If Overall stays low: GIGA_ENV_ARENAS=1536 and verify HIP (docs/AMD.md).");
			} else if (cudaPower && cfg.useCudaSim) {
				RG_LOG("CUDA POWER: target Overall>="
					<< (int)CudaPower::kTargetMinOverallSps
					<< " arenas=" << cfg.numGames
					<< " steps=" << stepsPerItr
					<< " half=" << (cfg.ppo.useHalfPrecision ? 1 : 0)
					<< " tsPerSave=" << cfg.tsPerSave
					<< " sparringDefer=" << CudaPower::kSparringDeferTs
					<< " noTouch=" << cfg.cudaNoTouchSeconds
					<< "s ring=off (amd_win_20k untouched)");
			}
		}
	} else if (g_Hyperpower) {
		if (g_Pure80) {
			RG_LOG("=== PURE80 SPS BENCHMARK (1-step truncate; not normal train) ===");
		} else {
			RG_LOG("=== HYPERPOWER MODE ===");
		}
		cfg.tickSkip = 6;
		cfg.actionDelay = cfg.tickSkip - 1;
		cfg.ppo.policyType = PolicyType::DISCRETE;
#ifdef GIGA_USE_CUDA_SIM
		cfg.useCudaSim = !g_ForceCpuSim;
#else
		cfg.useCudaSim = false;
#endif
		if (g_Pure80 && cfg.useCudaSim) {
			// Max-out 5060 Ti 16GB: as many arenas as stable, truncate ASAP, lean PPO.
			// Target near-term: Overall >= 20x Leak (~908k). Ceiling ~35-50x on this box.
			// 8192 fills VRAM; 4096-6144 often steadier. Override: GIGA_PURE80_ARENAS.
			cfg.numGames = 8192;
			if (const char* envA = std::getenv("GIGA_PURE80_ARENAS")) {
				int v = std::atoi(envA);
				if (v >= 256 && v <= 16384) cfg.numGames = v;
			}
			// steps=6 measured: Overall peak ~3.53M, Cons peak >=8M, Coll>=6M (vs steps=4 Cons~6M).
			int stepsPerItr = 6;
			if (const char* envS = std::getenv("GIGA_PURE80_STEPS")) {
				int v = std::atoi(envS);
				if (v >= 1 && v <= 64) stepsPerItr = v;
			}
			int tsPerItr = cfg.numGames * 2 * stepsPerItr;
			cfg.ppo.tsPerItr = tsPerItr;
			cfg.ppo.batchSize = tsPerItr;
			cfg.ppo.miniBatchSize = tsPerItr;
			cfg.ppo.epochs = 1;
			cfg.ppo.overbatching = false;
			cfg.ppo.prioritySampling = false;
			cfg.ppo.eventAdvantageBoost = 1.f;
			cfg.ppo.maxGradNorm = 0.f;
			cfg.ppo.skipPPOMetrics = true;
			cfg.ppo.skipExperienceShuffle = true;
			cfg.ppo.rewardClipRange = 0.f; // no clip work in GAE (std=1 path still divides by 1)
			// SPS: skip entropy term in PPO loss (still full policy+critic Learn).
			cfg.ppo.entropyScale = 0.f;
			// 1 env-step truncate -> bank every step (Learner only appends on truncate).
			cfg.ppo.maxEpisodeDuration = 0.05;
			cfg.skipObsIntegrityChecks = true;
			cfg.standardizeReturns = false;
			cfg.addRewardsToMetrics = false;
			cfg.sendMetrics = false;
#ifdef GIGA_USE_CUDA_SIM
			// Architectural leap: bank traj on CUDA, skip host obs/mask D2H every step.
			// Two-phase default for huge arena counts; Collect||Learn auto when deviceXP + arenas<=4096.
			// Opt-in/out: GIGA_ASYNC_OVERLAP=0|1|auto (Learner resolves; mode default=auto).
			cfg.cudaDeviceExperience = true;
			cfg.asyncLearnOverlapMode = 2; // auto
			cfg.asyncLearnOverlap = false; // resolved in Learner::Start
			cfg.asyncOverlapMaxArenas = 4096;
#endif
			RG_LOG("PURE80 MAX-PC: arenas=" << cfg.numGames << " stepsPerItr=" << stepsPerItr
				<< " tsPerItr=" << tsPerItr
				<< " maxEpDur=" << cfg.ppo.maxEpisodeDuration << "s"
				<< " skipMetrics+skipShuffle+fusedCritic+deviceXP"
				<< " asyncMode=" << cfg.asyncLearnOverlapMode);
		} else if (cfg.useCudaSim) {
			cfg.numGames = 4096;
			int tsPerItr = 500'000;
			cfg.ppo.tsPerItr = tsPerItr;
			cfg.ppo.batchSize = tsPerItr;
			cfg.ppo.miniBatchSize = 25'000;
			cfg.ppo.epochs = 1;
#ifdef GIGA_USE_CUDA_SIM
			// Safe single-GPU Collect||Learn: frozen rollout Infer + double XP bank.
			// Default auto in Learner (ON only with deviceXP + arenas<=max). Keep host obs for skill path
			// unless overlap explicitly requested (then deviceXP is required).
			cfg.cudaDeviceExperience = false;
			cfg.asyncLearnOverlapMode = 2; // auto (no-ops without deviceXP)
			cfg.asyncLearnOverlap = false;
			cfg.asyncOverlapMaxArenas = 2048;
			if (const char* envO = std::getenv("GIGA_ASYNC_OVERLAP")) {
				std::string v(envO);
				for (char& c : v) c = (char)tolower((unsigned char)c);
				if (v == "1" || v == "on" || v == "true" || (v == "auto" && cfg.numGames <= 2048)) {
					cfg.cudaDeviceExperience = true; // required for safe overlap
					cfg.asyncLearnOverlapMode = (v == "auto") ? 2 : 1;
					RG_LOG("Collect||Learn: deviceXP enabled for overlap (mode="
						<< cfg.asyncLearnOverlapMode << ")");
				} else if (v == "0" || v == "off" || v == "false") {
					cfg.asyncLearnOverlapMode = 0;
				}
			}
#endif
			RG_LOG("CUDA GPU-native @ " << cfg.numGames << " arenas"
				<< " asyncMode=" << cfg.asyncLearnOverlapMode
				<< (cfg.cudaDeviceExperience ? " deviceXP=1" : ""));
		} else {
			cfg.numGames = 2056;
			int tsPerItr = 500'000;
			cfg.ppo.tsPerItr = tsPerItr;
			cfg.ppo.batchSize = tsPerItr;
			cfg.ppo.miniBatchSize = 25'000;
			cfg.ppo.epochs = 1;
			RG_LOG("CPU discrete @ 2056");
		}
		cfg.ppo.entropyScale = 0.020f;
		cfg.ppo.maskEntropy = true; // DefaultAction masking -> explore valid actions only
		cfg.ppo.useBF16Autocast = true;
		cfg.ppo.useHalfPrecision = true;
		cfg.ppo.gaeGamma = 0.997f;
		cfg.ppo.gaeLambda = 0.975f;
		cfg.ppo.policyLR = 1e-4f;
		cfg.ppo.criticLR = 1e-4f;
		cfg.ppo.esNoiseScale = 0.f;
		if (!g_Pure80) {
			cfg.ppo.eventAdvantageBoost = 1.85f;
			cfg.ppo.prioritySampling = true;
		}
		cfg.ppo.clipRange = 0.2f;
		cfg.ppo.useAttentionHead = false;
	} else {
		RG_ERR_CLOSE("Discrete only (no continuous/attention). Use default train or --hyperpower/--pure80.");
	}

	if (!g_DefaultEnv) {
		if (g_Pure80 && g_Hyperpower) {
			// Lean MLP: 192 default (Cons reclaim vs 256; Coll stays >=6M class). GIGA_PURE80_NET overrides.
			int net = 192;
			if (const char* envN = std::getenv("GIGA_PURE80_NET")) {
				int v = std::atoi(envN);
				if (v == 128 || v == 192 || v == 256 || v == 384) net = v;
			}
			cfg.ppo.sharedHead.layerSizes = { net };
			cfg.ppo.policy.layerSizes = { net, net };
			cfg.ppo.critic.layerSizes = { net, net };
			cfg.ppo.policy.addLayerNorm = false;
			cfg.ppo.critic.addLayerNorm = false;
			cfg.ppo.sharedHead.addLayerNorm = false;
			cfg.checkpointFolder = exeDir / ("checkpoints_pure80_" + std::to_string(net));
			RG_LOG("PURE80 net width=" << net);
		} else {
			cfg.ppo.sharedHead.layerSizes = { 1024, 1024 };
			cfg.ppo.policy.layerSizes = { 1024, 1024, 1024 };
			cfg.ppo.critic.layerSizes = { 1024, 1024, 1024 };
			cfg.ppo.policy.addLayerNorm = true;
			cfg.ppo.critic.addLayerNorm = true;
			cfg.ppo.sharedHead.addLayerNorm = true;
		}
	}
	cfg.ppo.policy.optimType = optim;
	cfg.ppo.critic.optimType = optim;
	cfg.ppo.sharedHead.optimType = optim;
	cfg.ppo.policy.activationType = activation;
	cfg.ppo.critic.activationType = activation;
	cfg.ppo.sharedHead.activationType = activation;

	// Default: wandb off for pure80 / default unattended (SPS floor).
	// --wandb / --metrics overrides below. Render never sends wandb.
	cfg.sendMetrics = (g_Pure80 || g_DefaultEnv) ? false : !renderMode;
	cfg.renderMode = renderMode;

	cfg.skillTracker.enabled = false;
	cfg.skillTracker.updateInterval = 16;
	cfg.skillTracker.tsPerEval = 5'000'000; // in-sim eval every ~5M steps when enabled
	cfg.skillTracker.numArenas = 12;
	cfg.skillTracker.evaluate1v1 = true;
	cfg.skillTracker.numArenas1v1 = 8;
	cfg.skillTracker.envCreateFn1v1 = g_DefaultEnv ? EnvCreateDefault : EnvCreateTraining;

	// Apex on normal (SPS-safe) + continue-leak (full) + quality: policy league + opponent pressure.
	const bool apexLearn = g_ApexCurriculum && !g_Pure80;
	const bool denseLearn = apexLearn && (!g_DefaultEnv || g_MaxLearn); // hyperpower OR POWER continue-leak
	if (g_MaxLearn && apexLearn) {
		// POWER continue-leak: full Apex league + 1024 teachers/experts (docs/POWER_80X.md).
		cfg.savePolicyVersions = true;
		cfg.tsPerVersion = 4'000'000;
		cfg.maxOldVersions = 40;
		cfg.trainAgainstOldVersions = true;
		cfg.trainAgainstOldChance = 0.36f;
		cfg.opponentPool.enabled = true;
		cfg.opponentPool.chance = 0.55f;
		cfg.opponentPool.beatBonus = 110.f;
		cfg.opponentPool.concedePenalty = -42.f;
	} else if (g_MaxLearn) {
		// Classic Leak self-play (GIGA_NO_APEX=1): old versions only, OpponentPool off.
		cfg.savePolicyVersions = true;
		cfg.tsPerVersion = 8'000'000;
		cfg.maxOldVersions = 16;
		cfg.trainAgainstOldVersions = true;
		cfg.trainAgainstOldChance = 0.12f;
		cfg.opponentPool.enabled = false;
		cfg.opponentPool.chance = 0.f;
		cfg.opponentPool.beatBonus = 0.f;
		cfg.opponentPool.concedePenalty = 0.f;
	} else {
		cfg.savePolicyVersions = apexLearn;
		cfg.tsPerVersion = denseLearn ? 4'000'000 : (g_DefaultEnv ? 6'000'000 : (g_ApexCurriculum ? 4'000'000 : 8'000'000));
		cfg.maxOldVersions = denseLearn ? 40 : (g_DefaultEnv ? 24 : (g_ApexCurriculum ? 40 : 8));
		cfg.trainAgainstOldVersions = apexLearn;
		cfg.trainAgainstOldChance = denseLearn ? 0.36f : (g_DefaultEnv ? 0.12f : (apexLearn ? 0.36f : 0.f));
		cfg.opponentPool.enabled = apexLearn;
		cfg.opponentPool.chance = denseLearn ? 0.55f : (g_DefaultEnv ? 0.06f : (apexLearn ? 0.62f : 0.f));
		cfg.opponentPool.beatBonus = denseLearn ? 110.f : (g_DefaultEnv ? 70.f : (apexLearn ? 110.f : 0.f));
		cfg.opponentPool.concedePenalty = denseLearn ? -42.f : (g_DefaultEnv ? -28.f : (apexLearn ? -42.f : 0.f));
		// From-scratch boot: pure self-play. Old-version sparring + OpponentPool (Nexto JIT)
		// can stall wall-clock iters / fight a second GigaLearnBot for the GPU.
		if (g_FromScratch) {
			cfg.savePolicyVersions = false;
			cfg.trainAgainstOldVersions = false;
			cfg.trainAgainstOldChance = 0.f;
			cfg.opponentPool.enabled = false;
			cfg.opponentPool.chance = 0.f;
		}
	}
	// Absolute under exeDir - never depend on process cwd (system32 launch used to abort here).
	cfg.opponentPool.folder = exeDir / "opponents";
	cfg.opponentPool.manifest = exeDir / "opponents" / "opponents.json";
	cfg.opponentPool.quiet = !g_Verbose;
	if (cfg.opponentPool.enabled && g_Verbose) {
		RG_LOG("OpponentPool ON" << (g_DefaultEnv ? " (NORMAL SPS-safe)" : " (Apex)")
			<< ": chance=" << cfg.opponentPool.chance
			<< " beat=" << cfg.opponentPool.beatBonus
			<< " concede=" << cfg.opponentPool.concedePenalty
			<< " manifest=" << cfg.opponentPool.manifest
			<< " (nexto/necto JIT + nexto_tled/requiem arch-aware)");
	} else if (g_MaxLearn && g_Verbose && !cfg.opponentPool.enabled) {
		RG_LOG("OpponentPool OFF (CONTINUE-LEAK classic / GIGA_NO_APEX)");
	}

	// Skill-arena: Leak-style on continue-leak; light on normal; dense on hyperpower.
	if (g_MaxLearn) {
		cfg.skillTracker.enabled = true;
		cfg.skillTracker.updateInterval = 16; // Leak
		cfg.skillTracker.numArenas = 8;
		cfg.skillTracker.evaluate1v1 = true;
		cfg.skillTracker.numArenas1v1 = 8;
		cfg.skillTracker.simTime = 30.f;
		cfg.skillTracker.maxSimTime = 120.f;
		cfg.skillTracker.envCreateFn1v1 = EnvCreateDefault;
		if (g_Verbose) {
			RG_LOG("Skill-arena ON (CONTINUE-LEAK / Leak): arenas=" << cfg.skillTracker.numArenas
				<< " 1v1=" << cfg.skillTracker.numArenas1v1
				<< " interval=" << cfg.skillTracker.updateInterval);
		}
	} else if (apexLearn && denseLearn) {
		cfg.skillTracker.enabled = true;
		cfg.skillTracker.updateInterval = 10;
		cfg.skillTracker.numArenas = 14;
		cfg.skillTracker.evaluate1v1 = true;
		cfg.skillTracker.numArenas1v1 = 12;
		cfg.skillTracker.simTime = 42.f;
		cfg.skillTracker.maxSimTime = 220.f;
		cfg.skillTracker.envCreateFn1v1 = EnvCreateTraining;
		if (g_Verbose) {
			RG_LOG("Skill-arena ON: arenas=" << cfg.skillTracker.numArenas
				<< " 1v1=" << cfg.skillTracker.numArenas1v1
				<< " interval=" << cfg.skillTracker.updateInterval);
		}
	} else if (apexLearn && g_DefaultEnv) {
		// CUDA POWER: skill-eval OFF - Elo stalls wall-clock and pulls sustained Overall median down.
		// From-scratch / AMD resume: same early OFF; light eval only on non-CUDA POWER resume.
		if (g_FromScratch || IsCudaPowerRuntime()) {
			cfg.skillTracker.enabled = false;
			if (g_Verbose) {
				if (IsCudaPowerRuntime()) {
					RG_LOG("Skill-arena OFF (CUDA POWER sustained median)");
				} else {
					RG_LOG("Skill-arena OFF (FROM-SCRATCH, fast iters)");
				}
			}
		} else {
			cfg.skillTracker.enabled = true;
			cfg.skillTracker.updateInterval = 128;
			cfg.skillTracker.numArenas = 2;
			cfg.skillTracker.evaluate1v1 = true;
			cfg.skillTracker.numArenas1v1 = 2;
			cfg.skillTracker.simTime = 12.f;
			cfg.skillTracker.maxSimTime = 36.f;
			cfg.skillTracker.envCreateFn1v1 = EnvCreateDefault;
			if (g_Verbose) {
				RG_LOG("Skill-arena LIGHT (NORMAL): arenas=" << cfg.skillTracker.numArenas
					<< " 1v1=" << cfg.skillTracker.numArenas1v1
					<< " interval=" << cfg.skillTracker.updateInterval);
			}
		}
	}

	if (g_RandomSeed >= 0) {
		cfg.randomSeed = g_RandomSeed;
		RG_LOG("Multi-seed: randomSeed=" << g_RandomSeed);
	}

	// --metrics / --wandb / continue-leak: Leak Report.Display each iter (PPO fields).
	// Do NOT enable StepCallback here - walking 8192 arenas/step kills Coll (Leak Display
	// does not show Player/* anyway). Rewards/* sampling only when sending wandb.
	if (g_FullMetrics && !g_Pure80) {
		cfg.ppo.skipPPOMetrics = false;
		cfg.addRewardsToMetrics = g_SendWandb && !renderMode;
		cfg.rewardSampleRandInterval = 16; // Leak
		cfg.maxRewardSamples = 50;
		cfg.sendMetrics = g_SendWandb && !renderMode;
		// From-scratch / CUDA POWER: keep console Report, but do not re-arm heavy skill-eval /
		// version dumps - that stalls wall-clock and sinks sustained Overall median.
		if (!g_FromScratch && !IsCudaPowerRuntime()) {
			cfg.skillTracker.enabled = true;
			cfg.savePolicyVersions = true;
		} else {
			cfg.savePolicyVersions = false; // resume Apex league later; prefer fast iters at boot
			if (IsCudaPowerRuntime())
				cfg.skillTracker.enabled = false;
		}
		if (!g_MaxLearn && !g_FromScratch && !IsCudaPowerRuntime()) {
			// Standalone --metrics on normal: match Leak league knobs.
			cfg.maxOldVersions = 16;
			cfg.tsPerVersion = 8'000'000;
			if (g_DefaultEnv) {
				cfg.trainAgainstOldVersions = true;
				cfg.trainAgainstOldChance = 0.12f; // Leak
			}
		}
		if (g_Verbose) {
			RG_LOG("Leak Report.Display ON"
				<< (cfg.sendMetrics ? " (wandb+rewards)" : " (console PPO fields)")
				<< (g_FromScratch ? " [from-scratch: light skill-eval, no policy_versions]" : "")
				<< (g_MaxLearn ? " [continue-leak default; GIGA_SKIP_METRICS=1 to disable]" : ""));
		}
	}

	if (renderMode) {
		cfg.renderMode = true;
		cfg.sendMetrics = false;
		cfg.numGames = 1;
		cfg.addRewardsToMetrics = true; // RocketSimVis reward breakdown
		cfg.ppo.deterministic = true;
		cfg.ppo.skipPPOMetrics = true; // render loop does not Learn (same as Leak)
#ifdef GIGA_USE_CUDA_SIM
		cfg.useCudaSim = false; // CPU EnvSet for stable RocketSimVis stream
#endif
		RG_LOG("Render mode (view-only, like Leak): no Collect/Learn. Start RocketSimVis (UDP 9273).");
		RG_LOG("To train iterations like Leak: run without --render (e.g. --continue-leak / --like-leak).");
	}

	// Start AutoTrainer before heavy Learner/CUDA arena setup so the brain is ready
	// when the first trainer_status.json appears.
	if (!renderMode && !rlbotMode)
		MaybeStartAutoTrainer(exeDir);

	// Skip per-step callback on default train - walks all arenas and kills Coll / can hang iters.
	// Leak console Display uses PPO/Learner fields only (no Player/* rows).
	// --from-scratch/--verbose/--metrics must NOT enable StepCallback (was stalling wall-clock).
	// Render still needs callback for RocketSimVis; non-default + metrics+verbose opts into Player/*.
	const bool useStepCallback = !g_Pure80 && (renderMode || (!g_DefaultEnv && g_FullMetrics && g_Verbose));
	auto envCreate = g_DefaultEnv ? EnvCreateDefault : EnvCreateTraining;
	Learner* learner = new Learner(envCreate, cfg, useStepCallback ? StepCallback : nullptr);
	{
		EnvCreateResult peek = envCreate(0);
		AutoTrainerBridge::WriteRewardManifest(learner, peek.rewards);
		delete peek.arena;
		delete peek.obsBuilder;
		delete peek.actionParser;
		delete peek.continuousActionParser;
		delete peek.stateSetter;
		for (auto& w : peek.rewards) delete w.reward;
		for (auto* tc : peek.terminalConditions) delete tc;
	}
#ifdef GIGA_USE_CUDA_SIM
	// Error savestate ring (-1.5s): host car/ball/pad snapshots for AutoTrainer replay.
	// Default OFF on CUDA normal/from-scratch - ring D2H across 8k arenas tanks Collection SPS.
	// Opt-in: GIGA_STATE_RING=1 (AutoTrainer error_pressure can still enable at runtime).
	if (learner->cudaEnvSet && !renderMode && !rlbotMode) {
		bool enableRing = false;
		if (const char* envR = std::getenv("GIGA_STATE_RING"))
			enableRing = std::atoi(envR) != 0;
		if (enableRing) {
			int depth = 10, every = 3;
			if (g_Pure80) { depth = 6; every = 6; }
			learner->ConfigureErrorStateRing(true, depth, every, 0.05f, 64);
		}
	}
#endif
	// Leak CombinedState mix (kickoff/fuzzed/aerial). Pin GPU reset curriculum too -
	// CUDA defaults to kickoff-only; Apex hot-updates normal path, continue-leak does not.
	if (g_DefaultEnv) {
		SetDefaultStateMix(learner, 0.375f, 0.375f, 0.25f);
		SetGpuResetCurriculum(learner, 0.375f, 0.375f, 0.25f);
	}
	if (g_DefaultEnv) {
		std::cout
			<< "Default 1v1 template (self-play) - customize in ExampleMain / CudaEnvSet\n"
			<< "  Goal 100 | Touch 5 (ZS) | VelBallToGoal 10 (ZS) | VelPlayerToBall 2\n"
			<< "  See docs/CUSTOMIZE.md - add your rewards / opponents / AT profile.\n";
	}
	// Optional teacher discovery for Apex / resume paths (discrete PPO only).
	if (!renderMode && !rlbotMode && (g_ApexCurriculum || g_DefaultEnv)) {
		if (g_TeacherPaths.empty())
			DiscoverTeacherCheckpoints(exeDir);
		if (g_DefaultEnv && !g_TeacherPaths.empty() && g_Verbose)
			RG_LOG((g_MaxLearn ? "CONTINUE-LEAK" : "NORMAL") << " multi-teacher registry: "
				<< g_TeacherPaths.size() << " checkpoint(s)");
	}

	learner->Start();

	return EXIT_SUCCESS;
}
