#include "IterProfiler.h"

#include <algorithm>
#include <cstdlib>
#include <sstream>

GGL::IterProfiler& GGL::IterProfiler::FromEnv() {
	static IterProfiler inst;
	static bool inited = false;
	if (!inited) {
		inited = true;
		const char* en = std::getenv("GIGA_PROFILE");
		if (en && std::atoi(en) != 0) {
			inst.enabled = true;
			inst.dumpEvery = (std::max)(1, std::atoi(en));
		}
		if (const char* path = std::getenv("GIGA_PROFILE_PATH")) {
			inst.dumpPath = path;
			inst.enabled = true;
		} else if (inst.enabled) {
			inst.dumpPath = "profile_iters.jsonl";
		}
		if (const char* every = std::getenv("GIGA_PROFILE_EVERY"))
			inst.dumpEvery = (std::max)(1, std::atoi(every));
	}
	return inst;
}

void GGL::IterProfiler::FinishIteration(Report& report, int64_t timesteps, float wallSec) {
	if (!enabled)
		return;
	if (hasActive)
		End();

	iteration++;
	double sumSpans = 0.0;
	for (const auto& kv : spans) {
		report["Profile/" + kv.first] = (float)kv.second;
		sumSpans += kv.second;
	}
	report["Profile/Wall"] = wallSec;
	report["Profile/Accounted"] = (float)sumSpans;
	if (wallSec > 1e-9)
		report["Profile/Unaccounted"] = (float)(wallSec - sumSpans);

	if (!dumpPath.empty() && (iteration % dumpEvery) == 0) {
		std::ostringstream oss;
		oss << "{\"iter\":" << iteration
			<< ",\"timesteps\":" << timesteps
			<< ",\"wall\":" << wallSec;
		for (const auto& kv : spans)
			oss << ",\"" << kv.first << "\":" << kv.second;
		oss << "}\n";

		static std::mutex dumpMu;
		std::lock_guard lock(dumpMu);
		std::ofstream f(dumpPath, std::ios::app);
		if (f.good())
			f << oss.str();
	}

	ResetSpans();
}
