#include "MetricSender.h"

#include "Timer.h"
#include <filesystem>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace py = pybind11;
using namespace GGL;

GGL::MetricSender::MetricSender(std::string _projectName, std::string _groupName, std::string _runName, std::string runID) :
	projectName(_projectName), groupName(_groupName), runName(_runName) {

	RG_LOG("Initializing MetricSender...");

	// Ensure Python can resolve the local python_scripts package regardless of launch cwd.
	// Same candidate strategy as RenderSender (exe-relative + repo layouts).
	try {
		auto sys = py::module::import("sys");
		auto path = sys.attr("path");

		std::vector<std::filesystem::path> candidates = {
			std::filesystem::current_path(),
			std::filesystem::current_path() / "build",
			std::filesystem::current_path() / "GigaLearnCPP",
			std::filesystem::current_path().parent_path(),
			std::filesystem::current_path().parent_path() / "build",
			std::filesystem::current_path().parent_path() / "GigaLearnCPP",
			std::filesystem::current_path().parent_path().parent_path() / "GigaLearnCPP",
		};

#ifdef _WIN32
		char exeBuf[MAX_PATH];
		DWORD n = GetModuleFileNameA(nullptr, exeBuf, MAX_PATH);
		if (n > 0 && n < MAX_PATH) {
			std::filesystem::path exeDir = std::filesystem::path(exeBuf).parent_path();
			candidates.push_back(exeDir);
			candidates.push_back(exeDir.parent_path());
			candidates.push_back(exeDir.parent_path() / "GigaLearnCPP");
			candidates.push_back(exeDir.parent_path().parent_path() / "GigaLearnCPP");
		}
#endif

		for (const auto& candidate : candidates) {
			auto metricReceiverPath = candidate / "python_scripts" / "metric_receiver.py";
			if (std::filesystem::exists(metricReceiverPath)) {
				path.attr("append")(candidate.string());
				RG_LOG("MetricSender: found python_scripts at " << candidate);
			}
		}
	} catch (...) {
		// If path setup fails, import below will emit the real actionable error.
	}

	try {
		pyMod = py::module::import("python_scripts.metric_receiver");
	} catch (std::exception& e) {
		RG_ERR_CLOSE(
			"MetricSender: Failed to import metrics receiver, exception: " << e.what() << "\n"
			"Ensure python_scripts/metric_receiver.py is discoverable "
			"(e.g. C:\\GigaLearnRL\\build\\python_scripts or GigaLearnCPP\\python_scripts).\n"
			"Enable with: GigaLearnBot.exe --metrics   or   --wandb"
		);
	}

	try {
		auto returedRunID = pyMod.attr("init")(PY_EXEC_PATH, projectName, groupName, runName, runID);
		curRunID = returedRunID.cast<std::string>();
		RG_LOG(" > " << (runID.empty() ? "Starting" : "Continuing") << " run with ID : \"" << curRunID << "\"...");

	} catch (std::exception& e) {
		RG_ERR_CLOSE("MetricSender: Failed to initialize in Python, exception: " << e.what());
	}

	RG_LOG(" > MetricSender initalized.");
}

void GGL::MetricSender::Send(const Report& report) {
	// Calling into Python every iter contends for the GIL. Log sparsely; always log #1.
	++sendCounter;
	if (sendEveryN > 1 && sendCounter != 1 && (sendCounter % sendEveryN) != 0)
		return;

	py::dict reportDict = {};

	for (auto& pair : report.data)
		reportDict[pair.first.c_str()] = pair.second;

	try {
		pyMod.attr("add_metrics")(reportDict);
	} catch (std::exception& e) {
		RG_LOG("MetricSender: add_metrics failed (continuing train): " << e.what());
	}
}

GGL::MetricSender::~MetricSender() {
	
}
