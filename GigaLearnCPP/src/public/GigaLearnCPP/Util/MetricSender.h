#pragma once
#include "Report.h"
#include <pybind11/pybind11.h>

namespace GGL {
	struct RG_IMEXPORT MetricSender {
		std::string curRunID;
		std::string projectName, groupName, runName;
		pybind11::module pyMod;
		// wandb GIL contention — throttle; always send the first sample so Charts populate.
		int sendCounter = 0;
		int sendEveryN = 10;

		MetricSender(std::string projectName = {}, std::string groupName = {}, std::string runName = {}, std::string runID = {});
		
		RG_NO_COPY(MetricSender);

		void Send(const Report& report);

		~MetricSender();
	};
}