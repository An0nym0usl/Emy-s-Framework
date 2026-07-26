#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <GigaLearnCPP/Learner.h>

// Runtime mode flags for GigaLearnBot. Default path = from-scratch POWER train.
// Experimental / secondary flags: see docs/EXPERIMENTAL.md

extern bool g_Hyperpower;
extern bool g_ForceCpuSim;
extern bool g_Pure80;
extern bool g_DefaultEnv;
extern bool g_FullMetrics;
extern bool g_SendWandb;
extern bool g_ApexCurriculum;
extern bool g_ForceNoApex;
extern bool g_MaxLearn;
extern bool g_FromScratch;
extern bool g_CliFromScratch;
extern bool g_CliContinueLeak;
extern bool g_Verbose;
extern bool g_NoAutoTrainer;
extern bool g_NoHwProfile;
extern bool g_ExplicitCpu;
extern int g_ProfileArenas;
extern bool g_ProfileTorchCpu;
extern bool g_AmdWin20k;
extern bool g_AmdHipGpuNativeConfirmed;
extern int g_AmdLowSpsStreak;
extern int g_AmdAutoDowngradeLevel;
extern std::filesystem::path g_AmdAutosizePath;
extern std::filesystem::path g_TeacherPath;
extern std::vector<std::filesystem::path> g_TeacherPaths;
extern float g_LastSkillRating1v1;
extern int64_t g_RandomSeed;

constexpr float kDefaultTargetOverallSPS = 400000.f;

struct ParsedCli {
	GGL::LearnerConfig cfg = {};
	bool renderMode = false;
	bool rlbotMode = false;
	std::filesystem::path checkpointOverride;
};

// Parse argv + env into mode flags and base LearnerConfig knobs (render / metrics).
ParsedCli ParseTrainCli(int argc, char* argv[]);

// Apply env vars and mode precedence after ParseTrainCli.
void ResolveTrainMode(ParsedCli& parsed);

void PutEnvForce(const char* key, const char* value);
void PutEnvIfUnset(const char* key, const char* value);
