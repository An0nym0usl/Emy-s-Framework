#pragma once

#include <filesystem>
#include <GigaLearnCPP/Learner.h>

void EnsureAutoGpuDetect(const std::filesystem::path& exeDir);
void ApplyHwProfile(const std::filesystem::path& exeDir);
void MaybeStartAutoTrainer(const std::filesystem::path& exeDir);
float ActiveOverallTargetSps();
bool IsCudaPowerRuntime();
void ClampCudaPowerSustainedSps(GGL::Learner* learner);
void ClampAmdWin20kSustainedSps(GGL::Learner* learner);
void MaybeAutoDowngradeAmdWin20k(GGL::Learner* learner, float overallSps);
