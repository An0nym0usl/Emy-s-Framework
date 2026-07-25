#include "InSimEval.h"

void GGL::InSimEvalResult::WriteToReport(Report& report, const std::string& prefix) const {
	report[prefix + "/Elo"] = primaryElo;
	report[prefix + "/EloDelta"] = eloDelta;
	report[prefix + "/WinRate"] = winRate;
	report[prefix + "/AvgReward"] = avgStepReward;
	report[prefix + "/GoalsFor"] = (float)goalsFor;
	report[prefix + "/GoalsAgainst"] = (float)goalsAgainst;
	report[prefix + "/SimSeconds"] = simSeconds;
	if (opponentVersionTs > 0)
		report[prefix + "/OpponentTs"] = (float)opponentVersionTs;
	if (rewardSamples > 0)
		report[prefix + "/RewardSamples"] = (float)rewardSamples;

	for (const auto& pair : ratings)
		report["Rating/" + pair.first] = pair.second;
}
