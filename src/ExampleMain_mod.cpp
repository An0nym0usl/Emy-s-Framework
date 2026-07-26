#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <GigaLearnCPP/Learner.h>
#ifdef GIGA_USE_CUDA_SIM
#include <GigaLearnCPP/Sim/CudaEnvSet.h>
#endif

#include <GigaLearnCPP/Util/InferUnit.h>
#include <RLGymCPP/ActionParsers/DefaultAction.h>
#include <RLGymCPP/OBSBuilders/AdvancedObs.h>

#include "RLBotClient.h"
#include "TrainingCurriculum.h"
#include "AutoTrainerBridge.h"
#include "TrainCli.h"
#include "TrainProfiles.h"
#include "TrainEnv.h"
#include "TrainHw.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

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

using namespace GGL;
using namespace RLGC;

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

	ParsedCli parsed = ParseTrainCli(argc, argv);
	ResolveTrainMode(parsed);
	LearnerConfig cfg = parsed.cfg;
	bool renderMode = parsed.renderMode;
	bool rlbotMode = parsed.rlbotMode;
	std::filesystem::path checkpointOverride = parsed.checkpointOverride;

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

		auto* obsBuilder = new AdvancedObs();
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
		// Default 2v2: Wazne roadmap (not Apex). Apex only for --max-learn / explicit --apex.
		if (!(g_DefaultEnv && !g_Pure80 && !g_MaxLearn) && g_ApexCurriculum && !g_Pure80)
			ApplyApexCurriculum(learner, nullptr, /*spsSafe=*/g_DefaultEnv && !g_MaxLearn);
		// SSL AutoTrainer guide: league 60/30/10 + GPU reset mix must win over Apex when flagged.
		AutoTrainerBridge::ReapplyPostApex(learner);
		// CUDA POWER: keep early iters pure self-play (Apex would re-enable opp/old every iter).
		ClampCudaPowerSustainedSps(learner);
		// amd_win_20k: same - CPU PPO cannot afford early league/skill tax.
		ClampAmdWin20kSustainedSps(learner);
		// Belt: below 1B never arm league/opp (skill Rating/2v2 stays on via Wazne).
		if (g_DefaultEnv && learner->totalTimesteps < 1'000'000'000ull) {
			learner->config.opponentPool.enabled = false;
			learner->config.opponentPool.chance = 0.f;
			learner->config.trainAgainstOldVersions = false;
			learner->config.trainAgainstOldChance = 0.f;
			learner->config.savePolicyVersions = false;
		}
		// Wazne last so entropy / 1B+ old-versions / GPU reward phase stick.
		if (g_DefaultEnv && !g_Pure80 && !g_MaxLearn)
			ApplyWazne2v2Roadmap(learner);
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
			RG_LOG("FROM-SCRATCH POWER train (dense chase, no Air early)");
		} else if (g_MaxLearn) {
			RG_LOG("CONTINUE-LEAK (secondary resume path; see docs/EXPERIMENTAL.md)");
		} else {
			RG_LOG("NORMAL 2v2 train");
		}
		cfg.tickSkip = 6;
		cfg.actionDelay = cfg.tickSkip - 1; // Leak
		cfg.ppo.policyType = PolicyType::DISCRETE;
#ifdef GIGA_USE_CUDA_SIM
		cfg.useCudaSim = !g_ForceCpuSim;
		// GPU-native maps default state weights onto device IDs (framework power underneath Leak knobs).
		// Exact Leak CPU rewards: add --cpu (much slower).
		cfg.cudaPreferGpuNative = !g_ForceCpuSim;
		// From-scratch: profile 0 = dense chase (VelP2B/Face/Touch). Profile 1 = Wazne (Air→jumps).
		cfg.cudaRewardProfile = g_FromScratch ? 0 : 1;
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
			: (cfg.numGames * 4 * stepsPerItr); // 2v2: 4 cars per arena

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
				tsPerItr = cfg.numGames * 4 * stepsPerItr; // 2v2
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
			tsPerItr = cfg.numGames * 4 * stepsPerItr; // 2v2
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
		// Leak entropy on continue-leak; higher floor on from-scratch so policy doesn't collapse to jump.
		cfg.ppo.entropyScale = g_MaxLearn ? 0.020f : (g_FromScratch ? 0.040f : 0.035f);
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
		// wandb ON by default; Final enable happens in g_FullMetrics block below.
		cfg.sendMetrics = false;
		cfg.addRewardsToMetrics = false;
		cfg.metricsProjectName = "GigaLearnRL-2v2";
		cfg.metricsGroupName = "wazne-base";
		cfg.metricsRunName = "2v2-from-scratch";
		// Always pin save cadence (default LearnerConfig is 1M — freezes train every ~2 iters).
		cfg.tsPerSave = CudaPower::kTsPerSave; // 50M
		cfg.checkpointsToKeep = -1; // never delete mid-train (AV/OneDrive remove_all hangs)
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
			// Sustained median: hard-off state-ring (AT cannot re-arm without env).
			// tsPerSave already pinned above for all default 2v2 CUDA/AMD paths.
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
			// Prefer a local (non-OneDrive) path — saves on synced folders hang the train loop.
			if (const char* envCk = std::getenv("GIGA_CHECKPOINT_DIR")) {
				if (envCk[0])
					cfg.checkpointFolder = envCk;
			} else {
				const std::filesystem::path localMirror = "C:\\GigaLearnRL\\build\\Release\\checkpoints";
				std::error_code ecExist;
				if (std::filesystem::exists(localMirror, ecExist)) {
					cfg.checkpointFolder = localMirror;
					RG_LOG("Using local checkpoint dir (avoid OneDrive hang): " << cfg.checkpointFolder);
				}
			}
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
				<< " wazne=" << (g_DefaultEnv && !g_MaxLearn ? "roadmap" : "off")
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
		RG_LOG((g_Pure80
			? "PURE80 benchmark (experimental; docs/EXPERIMENTAL.md)"
			: "HYPERPOWER curriculum (experimental; docs/EXPERIMENTAL.md)"));
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
	cfg.skillTracker.updateInterval = 64;
	cfg.skillTracker.tsPerEval = 20'000'000;
	cfg.skillTracker.numArenas = 4;
	cfg.skillTracker.evaluate1v1 = false; // 2v2 train → Rating/2v2 only (more accurate)
	cfg.skillTracker.numArenas1v1 = 0;
	cfg.skillTracker.simTime = 30.f;
	cfg.skillTracker.maxSimTime = 90.f;
	cfg.skillTracker.writeEvalMetrics = true;
	cfg.skillTracker.envCreateFn1v1 = nullptr;

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
		// From-scratch boot: pure self-play. No policy_versions / OpponentPool until 1B
		// (ClampCudaPowerSustainedSps). Early version dumps freeze the train loop on disk.
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

	// Skill-arena: Leak-style on continue-leak; light Rating/2v2 on default Wazne; dense on hyperpower.
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
	} else if (g_DefaultEnv) {
		// Wazne 2v2: light Elo Rating/2v2 vs old self (in-RAM versions). No 1v1.
		cfg.skillTracker.enabled = true;
		cfg.skillTracker.updateInterval = 64;
		cfg.skillTracker.tsPerEval = 20'000'000;
		cfg.skillTracker.numArenas = 4;
		cfg.skillTracker.evaluate1v1 = false;
		cfg.skillTracker.numArenas1v1 = 0;
		cfg.skillTracker.simTime = 30.f;
		cfg.skillTracker.maxSimTime = 90.f;
		cfg.skillTracker.writeEvalMetrics = true;
		cfg.skillTracker.envCreateFn1v1 = nullptr;
		cfg.tsPerVersion = 25'000'000;
		cfg.maxOldVersions = 8;
		RG_LOG("Skill-arena LIGHT Rating/2v2: arenas=4 every ~20M steps (no 1v1)");
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
		// Keep light Rating/2v2; do not re-arm disk policy_versions dumps before 1B.
		if (!g_FromScratch && !IsCudaPowerRuntime() && !g_DefaultEnv) {
			cfg.skillTracker.enabled = true;
			cfg.savePolicyVersions = true;
		} else if (g_DefaultEnv) {
			cfg.savePolicyVersions = false; // Elo uses in-RAM versions until 1B
			cfg.skillTracker.enabled = true;
			cfg.skillTracker.evaluate1v1 = false;
		} else {
			cfg.savePolicyVersions = false;
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
	// Pin GPU reset curriculum — CUDA defaults to kickoff-only without this.
	// From-scratch: heavy ball-chase so short bank horizons still teach drive-to-ball.
	if (g_DefaultEnv) {
		if (g_FromScratch) {
			SetDefaultStateMix(learner, 0.15f, 0.15f, 0.10f, 0.60f);
			SetGpuResetCurriculum(learner, 0.15f, 0.15f, 0.10f, 0.60f);
		} else {
			SetDefaultStateMix(learner, 0.375f, 0.375f, 0.25f, 0.f);
			SetGpuResetCurriculum(learner, 0.375f, 0.375f, 0.25f, 0.f);
		}
	}
	if (g_DefaultEnv && g_Verbose) {
		RG_LOG((g_FromScratch
			? "FROM-SCRATCH: Goal20/Touch22/Face4/VelP2B12/VelB2G2 | NO Air | maxEp=45s | chase reset 60%"
			: "Default 2v2 Wazne roadmap (see docs/CUSTOMIZE.md)"));
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
