#pragma once

#include <GigaLearnCPP/Export.h>
#include <GigaLearnCPP/Util/Report.h>
#include <GigaLearnCPP/Util/Timer.h>

#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace GGL {

	// Lightweight Collect/Learn phase profiler.
	// Enable with GIGA_PROFILE=1 (or LearnerConfig.profileDumpPath set).
	// Writes one JSON object per iteration to the dump path (append).
	struct RG_IMEXPORT IterProfiler {
		bool enabled = false;
		std::string dumpPath;
		int dumpEvery = 1; // dump every N iterations
		int64_t iteration = 0;

		std::unordered_map<std::string, double> spans;
		Timer active;
		std::string activeName;
		bool hasActive = false;

		static IterProfiler& FromEnv();

		void Begin(const std::string& name) {
			if (!enabled)
				return;
			if (hasActive)
				End();
			activeName = name;
			active.Reset();
			hasActive = true;
		}

		void End() {
			if (!enabled || !hasActive)
				return;
			spans[activeName] += active.Elapsed();
			hasActive = false;
		}

		void Add(const std::string& name, double seconds) {
			if (!enabled)
				return;
			spans[name] += seconds;
		}

		void ResetSpans() {
			spans.clear();
			hasActive = false;
		}

		// Merge key spans into Report (Profile/*) and optionally append JSON line.
		void FinishIteration(Report& report, int64_t timesteps, float wallSec);
	};

}
