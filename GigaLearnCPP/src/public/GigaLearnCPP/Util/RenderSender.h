#pragma once
#include "Report.h"
#include <pybind11/pybind11.h>
#include <RLGymCPP/Gamestates/GameState.h>
#include <RLGymCPP/BasicTypes/Action.h>
#include <GigaLearnCPP/Util/Timer.h>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace GGL {
	struct RG_IMEXPORT RenderSender {
		pybind11::module pyMod;

		float timeScale;
		double adaptiveRenderDelay = -1;
		Timer renderTimer = {};

		RenderSender(float timeScale);

		RG_NO_COPY(RenderSender);

		// Optional playerLabels: one name per car (e.g. "You", "Opponent", "OldVersion").
		// totalTimesteps: forwarded to RocketSimVis training HUD when > 0.
		// trainingDiag: optional micro-diagnostics (reward components, etc.).
		void Send(
			const RLGC::GameState& state,
			const std::vector<std::string>& playerLabels = {},
			uint64_t totalTimesteps = 0,
			const nlohmann::json& trainingDiag = nlohmann::json()
		);

		~RenderSender();
	};
}
