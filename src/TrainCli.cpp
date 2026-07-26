#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "TrainCli.h"

#include <RLGymCPP/Framework.h> // RG_LOG / RG_ERR_CLOSE

#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

using namespace GGL;

bool g_Hyperpower = false;
bool g_ForceCpuSim = false;
bool g_Pure80 = false;
bool g_DefaultEnv = true;
bool g_FullMetrics = true;
bool g_SendWandb = true;
bool g_ApexCurriculum = false;
bool g_ForceNoApex = true;
bool g_MaxLearn = false;
bool g_FromScratch = true;
bool g_CliFromScratch = false;
bool g_CliContinueLeak = false;
bool g_Verbose = false;
bool g_NoAutoTrainer = false;
bool g_NoHwProfile = false;
bool g_ExplicitCpu = false;
int g_ProfileArenas = -1;
bool g_ProfileTorchCpu = false;
bool g_AmdWin20k = false;
bool g_AmdHipGpuNativeConfirmed = false;
int g_AmdLowSpsStreak = 0;
int g_AmdAutoDowngradeLevel = 0;
std::filesystem::path g_AmdAutosizePath;
std::filesystem::path g_TeacherPath;
std::vector<std::filesystem::path> g_TeacherPaths;
float g_LastSkillRating1v1 = 0.f;
int64_t g_RandomSeed = -1;

void PutEnvForce(const char* key, const char* value) {
#ifdef _WIN32
	_putenv_s(key, value);
#else
	setenv(key, value, 1);
#endif
}

void PutEnvIfUnset(const char* key, const char* value) {
	if (!std::getenv(key))
		PutEnvForce(key, value);
}

ParsedCli ParseTrainCli(int argc, char* argv[]) {
	ParsedCli out;
	for (int i = 1; i < argc; i++) {
		std::string arg = argv[i];
		if (arg == "--render") {
			out.cfg.sendMetrics = false;
			out.cfg.ppo.deterministic = true;
			out.cfg.renderMode = true;
			out.renderMode = true;
		} else if (arg == "--metrics") {
			g_FullMetrics = true;
		} else if (arg == "--wandb") {
			g_SendWandb = true;
			g_FullMetrics = true;
		} else if (arg == "--no-wandb") {
			g_SendWandb = false;
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
			out.rlbotMode = true;
		} else if (arg == "--hyperpower") {
			// Secondary: dense BallChase curriculum (docs/EXPERIMENTAL.md).
			g_Hyperpower = true;
			g_DefaultEnv = false;
			g_Pure80 = false;
			g_ApexCurriculum = true;
		} else if (arg == "--cpu") {
			g_ForceCpuSim = true;
			g_ExplicitCpu = true;
		} else if (arg == "--no-autotrainer") {
			g_NoAutoTrainer = true;
		} else if (arg == "--no-hw-profile") {
			g_NoHwProfile = true;
		} else if (arg == "--pure80") {
			// Secondary: SPS-only benchmark (docs/EXPERIMENTAL.md).
			g_Hyperpower = true;
			g_Pure80 = true;
			g_DefaultEnv = false;
			g_ForceCpuSim = false;
		} else if (arg == "--lean-resume") {
			g_DefaultEnv = true;
			g_FromScratch = false;
			g_Pure80 = false;
			g_Hyperpower = false;
			g_ForceCpuSim = false;
		} else if (arg == "--continue-leak" || arg == "--like-leak"
			|| arg == "--max-learn" || arg == "--resume-leak") {
			// Secondary resume path (still supported).
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
			g_CliFromScratch = true;
			g_FromScratch = true;
			g_MaxLearn = false;
			g_DefaultEnv = true;
			g_Pure80 = false;
			g_Hyperpower = false;
			g_ForceCpuSim = false;
			g_ForceNoApex = false;
		} else if (arg == "--legacy-hyperpower") {
			g_Hyperpower = true;
			g_Pure80 = false;
			g_DefaultEnv = false;
		} else if (arg == "--checkpoint" && i + 1 < argc) {
			out.checkpointOverride = argv[++i];
		} else if (arg == "--seed" && i + 1 < argc) {
			g_RandomSeed = (int64_t)std::atoll(argv[++i]);
		}
	}
	return out;
}

void ResolveTrainMode(ParsedCli& parsed) {
	if (const char* envSeed = std::getenv("GIGA_SEED")) {
		if (g_RandomSeed < 0)
			g_RandomSeed = (int64_t)std::atoll(envSeed);
	}
	if (const char* envNoApex = std::getenv("GIGA_NO_APEX")) {
		if (std::atoi(envNoApex) != 0)
			g_ForceNoApex = true;
	}
	if (const char* envFS = std::getenv("GIGA_FROM_SCRATCH")) {
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
	if (g_CliContinueLeak || (g_MaxLearn && !g_CliFromScratch)) {
		g_FromScratch = false;
		g_MaxLearn = true;
	} else if (g_CliFromScratch || g_FromScratch) {
		g_FromScratch = true;
		g_MaxLearn = false;
		g_DefaultEnv = true;
		g_Pure80 = false;
		g_Hyperpower = false;
		g_ForceNoApex = true;
		g_ApexCurriculum = false;
	}
	if (g_ExplicitCpu)
		g_ForceCpuSim = true;
	if (g_MaxLearn) {
		g_DefaultEnv = true;
		g_Pure80 = false;
		g_Hyperpower = false;
		if (!g_ForceNoApex)
			g_ApexCurriculum = true;
		if (!parsed.renderMode) {
			bool skipMetrics = false;
			if (const char* envSM = std::getenv("GIGA_SKIP_METRICS"))
				skipMetrics = std::atoi(envSM) != 0;
			if (!skipMetrics)
				g_FullMetrics = true;
			else {
				g_FullMetrics = false;
				g_SendWandb = false;
			}
		}
	}
	if (const char* envW = std::getenv("GIGA_WANDB")) {
		if (std::atoi(envW) != 0)
			g_SendWandb = true;
	}
	if (const char* envNW = std::getenv("GIGA_NO_WANDB")) {
		if (std::atoi(envNW) != 0)
			g_SendWandb = false;
	}
	if (const char* envSM = std::getenv("GIGA_SKIP_METRICS")) {
		if (std::atoi(envSM) != 0) {
			g_FullMetrics = false;
			g_SendWandb = false;
		}
	}
	if (g_FromScratch && !parsed.renderMode && !g_FullMetrics)
		g_FullMetrics = true;
	if (g_SendWandb && !parsed.renderMode)
		g_FullMetrics = true;
	if (parsed.renderMode || parsed.rlbotMode || g_Pure80)
		g_SendWandb = false;
	if (g_SendWandb && !parsed.renderMode)
		PutEnvIfUnset("WANDB_DISABLE_SERVICE", "true");
	if (!g_MaxLearn && g_DefaultEnv && !g_ForceNoApex && !g_Pure80
		&& !parsed.renderMode && !parsed.rlbotMode) {
		g_ApexCurriculum = true;
	}
}
