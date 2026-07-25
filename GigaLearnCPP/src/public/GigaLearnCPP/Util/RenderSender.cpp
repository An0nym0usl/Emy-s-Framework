#include "RenderSender.h"

#include <nlohmann/json.hpp>
#include <filesystem>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

using namespace nlohmann;
using namespace RLGC;

GGL::RenderSender::RenderSender(float timeScale) : timeScale(timeScale) {
	RG_LOG("Initializing RenderSender...");

	// Ensure Python can resolve the local python_scripts package regardless of launch cwd.
	// Typical layouts:
	//   build/Release/GigaLearnBot.exe  →  build/python_scripts or GigaLearnCPP/python_scripts
	//   cwd = repo root                 →  GigaLearnCPP/python_scripts
	try {
		auto sys = pybind11::module::import("sys");
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
			candidates.push_back(exeDir.parent_path()); // build/
			candidates.push_back(exeDir.parent_path() / "GigaLearnCPP");
			candidates.push_back(exeDir.parent_path().parent_path() / "GigaLearnCPP");
		}
#endif

		for (const auto& candidate : candidates) {
			auto renderReceiverPath = candidate / "python_scripts" / "render_receiver.py";
			if (std::filesystem::exists(renderReceiverPath)) {
				path.attr("append")(candidate.string());
				RG_LOG("RenderSender: found python_scripts at " << candidate);
			}
		}
	} catch (...) {
		// Import below will provide actionable error if this setup fails.
	}

	try {
		RG_LOG("Current dir: " << std::filesystem::current_path());
		pyMod = pybind11::module::import("python_scripts.render_receiver");
	} catch (std::exception& e) {
		RG_ERR_CLOSE(
			"RenderSender: Failed to import render receiver, exception: " << e.what() << "\n"
			"Ensure python_scripts/render_receiver.py is next to the build tree "
			"(e.g. C:\\GigaLearnRL\\build\\python_scripts or GigaLearnCPP\\python_scripts).\n"
			"Start RocketSimVis (UDP 9273) then: GigaLearnBot.exe --render"
		);
	}

	RG_LOG(" > RenderSender initalized.");
}

FList VecToList(const Vec& vec) {
	return FList({ vec.x, vec.y, vec.z });
}

json PhysToJSON(const PhysState& obj) {
	json j = {};

	j["pos"] = VecToList(obj.pos);

	j["forward"] = VecToList(obj.rotMat.forward);
	j["right"] = VecToList(obj.rotMat.right);
	j["up"] = VecToList(obj.rotMat.up);

	j["vel"] = VecToList(obj.vel);
	j["ang_vel"] = VecToList(obj.angVel);

	return j;
}

json ControlsToJSON(const Action& action) {
	json j = {};
	j["throttle"] = action.throttle;
	j["steer"] = action.steer;
	j["pitch"] = action.pitch;
	j["yaw"] = action.yaw;
	j["roll"] = action.roll;
	j["jump"] = action.jump > 0;
	j["boost"] = action.boost > 0;
	j["handbrake"] = action.handbrake > 0;
	return j;
}

json PlayerToJSON(const Player& player) {
	json j = {};

	j["car_id"] = player.carId;
	j["team_num"] = (int)player.team;
	j["phys"] = PhysToJSON(player);
	j["is_demoed"] = player.isDemoed;
	j["on_ground"] = player.isOnGround;
	j["ball_touched"] = player.ballTouchedStep;
	j["has_flip"] = player.HasFlipOrJump();
	j["boost_amount"] = player.boost / 100;
	j["controls"] = ControlsToJSON(player.prevAction);

	return j;
}

json GameStateToJSON(const GameState& state) {
	json j = {};
	
	j["ball"] = PhysToJSON(state.ball);

	std::vector<json> players;
	for (auto& player : state.players)
		players.push_back(PlayerToJSON(player));

	j["players"] = players;
	j["boost_pads"] = state.boostPads;

	return j;
}

std::vector<json> ActionSetToJSON(const std::vector<Action>& actions) {
	std::vector<json> js = {};
	for (auto& action : actions) {
		FList vals  = FList(action.begin(), action.end());
		js.push_back(json(vals));
	}

	return js;
}

void GGL::RenderSender::Send(
	const GameState& state,
	const std::vector<std::string>& playerLabels,
	uint64_t totalTimesteps,
	const nlohmann::json& trainingDiag
) {
	json j = {};
	j["gamemode"] = state.lastArena ? GAMEMODE_STRS[(int)state.lastArena->gameMode] : "soccar";
	j["state"] = GameStateToJSON(state);

	if (!playerLabels.empty()) {
		auto& players = j["state"]["players"];
		for (size_t i = 0; i < players.size() && i < playerLabels.size(); i++)
			players[i]["name"] = playerLabels[i];
	}

	std::vector<Action> actions = {};
	for (auto& player : state.players)
		actions.push_back(player.prevAction);

	j["actions"] = ActionSetToJSON(actions);
	if (totalTimesteps > 0)
		j["total_timesteps"] = totalTimesteps;
	if (!trainingDiag.is_null() && !trainingDiag.empty())
		j["training"] = trainingDiag;

	std::string jStr = j.dump();

	try {
		pyMod.attr("render_state")(jStr);
	} catch (std::exception& e) {
		RG_ERR_CLOSE("RenderSender: Failed to send gamestate, exception: " << e.what());
	}

	// Delay
	{
		namespace chr = std::chrono;

		// Determine the desired delay and the actual delay (in seconds)
		double targetDelay = state.deltaTime / timeScale;
		double realDelay = renderTimer.Elapsed();
		renderTimer.Reset();

		constexpr double CORRECTION_SCALE = 0.3f; // Portion of the error we wil compensate for each step
		double error = targetDelay - realDelay;

		if (adaptiveRenderDelay == -1) {
			// Just initialize render delay as target delay
			adaptiveRenderDelay = targetDelay;
		} else {
			adaptiveRenderDelay += error * CORRECTION_SCALE;
		}
		adaptiveRenderDelay = RS_CLAMP(adaptiveRenderDelay, 0, targetDelay);

		// Sleep for the new adaptive delay
		int64_t sleepMics = (int64_t)(adaptiveRenderDelay * 1'000'000);
		std::this_thread::sleep_for(chr::microseconds(sleepMics));
	}
}

GGL::RenderSender::~RenderSender() {}
