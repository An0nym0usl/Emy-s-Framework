#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "TrainCli.h"
#include "TrainProfiles.h"
#include "TrainEnv.h"
#include "TrainingCurriculum.h"
#include "AutoTrainerBridge.h"

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

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace GGL;
using namespace RLGC;

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
		return TrainingCurriculum::currentPhase == TrainingPhase::ADVANCED
			? advancedSetter : chaseSetter;
	}

	StateSetter* AdvancedSetter() const { return advancedSetter; }
	StateSetter* ChaseSetter() const { return chaseSetter; }
};

// Hot-update CombinedState kickoff/random mix (Apex). Prefer Advanced 2-way mix.
void SetKickoffStateMix(Learner* learner, float kickoffWeight, float randomWeight) {
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

// Hot-update default CombinedState on CPU hybrid path.
// 4-way: chase / kickoff / fuzzed / aerial. 3-way legacy: kickoff / fuzzed / aerial.
void SetDefaultStateMix(Learner* learner, float kickoffW, float fuzzedW, float aerialW, float chaseW) {
	if (!learner || !learner->envSet)
		return;
	for (auto* setter : learner->envSet->stateSetters) {
		auto* combined = dynamic_cast<CombinedState*>(setter);
		if (!combined)
			continue;
		if (combined->NumSetters() == 4)
			combined->SetWeights({ chaseW, kickoffW, fuzzedW, aerialW });
		else if (combined->NumSetters() == 3)
			combined->SetWeights({ kickoffW, fuzzedW, aerialW });
	}
}

// GPU-native reset curriculum (shapes normal CUDA train - CombinedState alone does not).
// chaseW: ball-near ground spawn (dense VelPlayerToBall). Default 0 keeps old 3-way mix.
void SetGpuResetCurriculum(Learner* learner, float kickoffW, float fuzzedW, float aerialW, float chaseW) {
#ifdef GIGA_USE_CUDA_SIM
	if (!learner || !learner->cudaEnvSet || !learner->cudaEnvSet->batch)
		return;
	learner->cudaEnvSet->batch->SetResetCurriculum(kickoffW, fuzzedW, aerialW, chaseW);
#else
	(void)learner; (void)kickoffW; (void)fuzzedW; (void)aerialW; (void)chaseW;
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
int ApplyApexCurriculum(Learner* learner, Report* report, bool spsSafe) {
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

std::filesystem::path FindLatestValidCheckpoint(const std::filesystem::path& checkpointFolder) {
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

int ReadRLBotPort(const std::filesystem::path& exeDir) {
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

void DiscoverTeacherCheckpoints(const std::filesystem::path& exeDir) {
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

void FillDiscretePolicyConfig(PartialModelConfig& policy) {
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

// =============================================================================
// 2v2 Wazne roadmap — auto milestones (GPU rewards + entropy / old versions).
//
// START (<550M):
//   Goal(-0.80)×150 | Touch×5 | Face×0.1 | VelP2B×1 | VelB2G×5 | Air×0.15
//   LR=1e-4 | epochs=2 | entropyScale=0.035
//
//   550M  → entropyScale 0.025
//   1.0B  → remove FaceBall | trainAgainstOldVersions chance 0.15
//   1.5B  → add KickoffProximity×1
//   1.8B  → entropyScale 0.02
//   2.0B  → TouchBall zero-sum
//   2.5B  → PickupBoost×50
// =============================================================================
static int WazneRewardPhase(uint64_t ts) {
	if (ts >= 2'500'000'000ull) return 4;
	if (ts >= 2'000'000'000ull) return 3;
	if (ts >= 1'500'000'000ull) return 2;
	if (ts >= 1'000'000'000ull) return 1;
	return 0;
}

static float WazneEntropyScale(uint64_t ts) {
	// Keep exploration high through early motor skills (was 0.025@550M → jump collapse).
	if (ts >= 1'800'000'000ull) return 0.022f;
	if (ts >= 1'000'000'000ull) return 0.028f;
	return 0.040f;
}

static std::vector<WeightedReward> BuildDefault2v2Rewards(int phase = 0) {
	std::vector<WeightedReward> r;
	r.push_back({ new GoalReward(-0.80f), 150.f });
	if (phase < 1) {
		// Early: dense drive-to-ball. No Air (farms jump under short horizons).
		r.push_back({ new TouchBallReward(), 12.f });
		r.push_back({ new FaceBallReward(), 3.f });
		r.push_back({ new VelocityPlayerToBallReward(), 10.f });
		r.push_back({ new VelocityBallToGoalReward(), 5.f });
		return r;
	}
	if (phase >= 3)
		r.push_back({ new ZeroSumReward(new TouchBallReward(), 0.0f), 5.0f });
	else
		r.push_back({ new TouchBallReward(), 5.0f });
	r.push_back({ new VelocityPlayerToBallReward(), 1.0f });
	r.push_back({ new VelocityBallToGoalReward(), 5.0f });
	r.push_back({ new AirReward(), 0.15f });
	if (phase >= 2)
		r.push_back({ new KickoffProximityReward(), 1.0f });
	if (phase >= 4)
		r.push_back({ new PickupBoostReward(), 50.f });
	return r;
}

// Hot-apply Wazne milestones each iter (replaces Apex on default 2v2).
void ApplyWazne2v2Roadmap(Learner* learner) {
	if (!learner || !g_DefaultEnv)
		return;

	const uint64_t ts = learner->totalTimesteps;
	const int phase = WazneRewardPhase(ts);
	const float entropy = WazneEntropyScale(ts);

	static int lastPhase = -1;
	static float lastEntropy = -1.f;
	static bool lastOld = false;

	learner->SetLearningRates(1e-4f, 1e-4f);
	if (entropy != lastEntropy) {
		learner->SetEntropyScale(entropy);
		lastEntropy = entropy;
		RG_LOG("Wazne roadmap: entropyScale=" << entropy << " @ ts=" << ts);
	}

	const bool oldOn = ts >= 1'000'000'000ull;
	// Light Rating/2v2 (Elo vs old self) — more accurate than 1v1 for this train mode.
	learner->config.skillTracker.enabled = true;
	learner->config.skillTracker.evaluate1v1 = false;
	learner->config.skillTracker.numArenas = 4;
	learner->config.skillTracker.updateInterval = 64;
	learner->config.skillTracker.tsPerEval = 20'000'000; // ~every 20M steps
	learner->config.skillTracker.simTime = 30.f;
	learner->config.skillTracker.maxSimTime = 90.f;
	learner->config.skillTracker.writeEvalMetrics = true;
	// In-RAM policy snapshots for Elo (disk SaveVersions only after 1B / oldOn).
	learner->config.tsPerVersion = 25'000'000;
	learner->config.maxOldVersions = 8;
	if (oldOn) {
		learner->config.trainAgainstOldVersions = true;
		learner->config.trainAgainstOldChance = 0.15f;
		learner->config.savePolicyVersions = true;
		learner->config.opponentPool.enabled = false;
		learner->config.opponentPool.chance = 0.f;
	} else {
		learner->config.trainAgainstOldVersions = false;
		learner->config.trainAgainstOldChance = 0.f;
		learner->config.savePolicyVersions = false;
		learner->config.opponentPool.enabled = false;
		learner->config.opponentPool.chance = 0.f;
	}
	if (oldOn != lastOld) {
		RG_LOG("Wazne roadmap: trainAgainstOld=" << (oldOn ? "ON chance=0.15" : "OFF")
			<< " @ ts=" << ts);
		lastOld = oldOn;
	}

#ifdef GIGA_USE_CUDA_SIM
	if (learner->cudaEnvSet && learner->cudaEnvSet->gpuNative
		&& learner->config.cudaRewardProfile == 1) {
		if (learner->cudaEnvSet->waznePhase != phase) {
			learner->cudaEnvSet->waznePhase = phase;
			learner->RequestDeferredCudaSurfaceApply();
			RG_LOG("Wazne roadmap: GPU reward phase=" << phase
				<< " (0=start 1=noFace 2=+kickoff 3=zsTouch 4=+boost) @ ts=" << ts);
		}
	}
#endif

	if (phase != lastPhase) {
		const char* names[] = {
			"START (dense chase, no Air)",
			"1.0B +Air / lighter VelP2B",
			"1.5B +KickoffProximity",
			"2.0B zero-sum Touch",
			"2.5B +PickupBoost×50"
		};
		RG_LOG("Wazne roadmap: entered " << names[(std::max)(0, (std::min)(phase, 4))]
			<< " | entropy=" << entropy << " | ts=" << ts);
		lastPhase = phase;
	}
}

// Hyperpower (--hyperpower) still uses phased TrainingCurriculum rewards.
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

EnvCreateResult EnvCreateTraining(int index) {
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

// Default train: 2v2 + dense chase start (CPU path). GPU uses cudaRewardProfile.
EnvCreateResult EnvCreateDefault(int index) {
	(void)index;

	// From-scratch: chase stack (no Air). Later Wazne phases via BuildDefault2v2Rewards.
	std::vector<WeightedReward> rewards = g_FromScratch
		? BuildChaseRewards()
		: BuildDefault2v2Rewards(0);

	auto arena = Arena::Create(GameMode::SOCCAR);
	arena->AddCar(Team::BLUE);
	arena->AddCar(Team::BLUE);
	arena->AddCar(Team::ORANGE);
	arena->AddCar(Team::ORANGE);

	EnvCreateResult result = {};
	result.actionParser = new DefaultAction();
	result.continuousActionParser = nullptr; // discrete only
	result.obsBuilder = new AdvancedObs();
	// Heavy BallChase so VelPlayerToBall has gradient inside short bank horizons.
	result.stateSetter = new CombinedState({
		{ new BallChaseState(1800.f, true, true), 0.60f },
		{ new KickoffState(), 0.15f },
		{ new FuzzedKickoffState(), 0.15f },
		{ new AerialDribbleSetupStateSetter(), 0.10f },
	});
	result.terminalConditions = {
		new NoTouchCondition(10),
		new GoalScoreCondition()
	};
	result.rewards = std::move(rewards);
	result.arena = arena;
	return result;
}

int GetTrainingObsSize() {
	EnvCreateResult env = EnvCreateDefault(0);
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

