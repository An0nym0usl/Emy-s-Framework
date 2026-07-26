#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

#include <GigaLearnCPP/Learner.h>
#include <GigaLearnCPP/Util/ModelConfig.h>
#include <GigaLearnCPP/Util/Report.h>
#include <RLGymCPP/EnvSet/EnvSet.h>
#include <RLGymCPP/Gamestates/GameState.h>

void SetKickoffStateMix(GGL::Learner* learner, float kickoffWeight, float randomWeight);
void SetDefaultStateMix(GGL::Learner* learner, float kickoffW, float fuzzedW, float aerialW, float chaseW = 0.f);
void SetGpuResetCurriculum(GGL::Learner* learner, float kickoffW, float fuzzedW, float aerialW, float chaseW = 0.f);
int ApplyApexCurriculum(GGL::Learner* learner, GGL::Report* report, bool spsSafe = false);
void ApplyWazne2v2Roadmap(GGL::Learner* learner);
RLGC::EnvCreateResult EnvCreateTraining(int index);
RLGC::EnvCreateResult EnvCreateDefault(int index);
int GetTrainingObsSize();
void FillDiscretePolicyConfig(GGL::PartialModelConfig& policy);
void DiscoverTeacherCheckpoints(const std::filesystem::path& exeDir);
std::filesystem::path FindLatestValidCheckpoint(const std::filesystem::path& checkpointFolder);
int ReadRLBotPort(const std::filesystem::path& exeDir);
void StepCallback(GGL::Learner* learner, const std::vector<RLGC::GameState>& states, GGL::Report& report);
