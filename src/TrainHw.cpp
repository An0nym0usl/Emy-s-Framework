#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "TrainHw.h"
#include "TrainCli.h"
#include "TrainProfiles.h"
#include "AutoTrainerBridge.h"

#include <GigaLearnCPP/Learner.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <nlohmann/json.hpp>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

using namespace GGL;

std::filesystem::path FindRepoRoot(const std::filesystem::path& exeDir) {
	// Release layout: <repo>/build/Release/GigaLearnBot.exe
	const std::filesystem::path candidates[] = {
		exeDir.parent_path().parent_path(),
		exeDir.parent_path(),
		exeDir,
		std::filesystem::path(R"(C:\GigaLearnRL)"), // optional clean CUDA mirror
	};
	for (const auto& root : candidates) {
		if (std::filesystem::exists(root / "autotrainer" / "orchestrator.py")
			|| std::filesystem::exists(root / "tools" / "hw_probe.py"))
			return root;
	}
	return {};
}

// Detect NVIDIA vs AMD and write hw_profile.json next to the exe.
// Prefer tools/hw_probe.py; fall back to a tiny Windows adapter probe so launching
// GigaLearnBot.exe alone (no bat) still picks CudaPower vs amd_win_20k.
bool WriteFallbackHwProfile(const std::filesystem::path& outPath) {
	bool hasNvidia = false;
	bool hasAmd = false;
	std::string gpuName = "unknown";
#ifdef _WIN32
	{
		char buf[MAX_PATH] = {};
		hasNvidia = (SearchPathA(nullptr, "nvidia-smi.exe", nullptr, MAX_PATH, buf, nullptr) != 0);
	}
	// Adapter names via PowerShell (no admin). Quiet / short timeout.
	{
		std::string cmd =
			"powershell -NoProfile -Command \"try { "
			"(Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join '; ' "
			"} catch { '' }\"";
		FILE* pipe = _popen(cmd.c_str(), "r");
		if (pipe) {
			char line[1024] = {};
			std::string names;
			while (fgets(line, sizeof(line), pipe))
				names += line;
			_pclose(pipe);
			for (char& c : names) c = (char)std::tolower((unsigned char)c);
			if (names.find("nvidia") != std::string::npos || names.find("geforce") != std::string::npos
				|| names.find("rtx") != std::string::npos || names.find("quadro") != std::string::npos)
				hasNvidia = true;
			if (names.find("amd") != std::string::npos || names.find("radeon") != std::string::npos
				|| names.find("rx ") != std::string::npos)
				hasAmd = true;
			gpuName = names.empty() ? "unknown" : names;
			while (!gpuName.empty() && (gpuName.back() == '\n' || gpuName.back() == '\r'))
				gpuName.pop_back();
		}
	}
#else
	(void)outPath;
	return false;
#endif
	// Prefer NVIDIA if both present (laptop mux / dual GPU).
	const bool useCuda = hasNvidia;
	const bool useHip = !useCuda && hasAmd;
	const char* primary = useCuda ? "nvidia" : (useHip ? "amd" : "none");
	const char* backend = useCuda ? "cuda" : (useHip ? "hip" : "cpu");
	const char* profile = useCuda ? "power_cuda" : (useHip ? "amd_win_20k" : "cpu_fallback");
	const int arenas = useCuda ? 8192 : (useHip ? 2048 : 512);
	const int steps = useCuda ? 16 : (useHip ? 3 : 1);
	const bool forceCpu = !useCuda && !useHip;
	try {
		std::error_code ec;
		std::filesystem::create_directories(outPath.parent_path(), ec);
		std::ofstream f(outPath, std::ios::trunc);
		if (!f)
			return false;
		f << "{\n  \"schema\": 2,\n  \"source\": \"exe_fallback_probe\",\n"
			<< "  \"recommendations\": {\n"
			<< "    \"primary_gpu\": \"" << primary << "\",\n"
			<< "    \"gpu_backend\": \"" << backend << "\",\n"
			<< "    \"profile\": \"" << profile << "\",\n"
			<< "    \"cuda_available\": " << (useCuda ? "true" : "false") << ",\n"
			<< "    \"hip_available\": " << (useHip ? "true" : "false") << ",\n"
			<< "    \"gpu_sim_available\": " << ((useCuda || useHip) ? "true" : "false") << ",\n"
			<< "    \"force_cpu_sim\": " << (forceCpu ? "true" : "false") << ",\n"
			<< "    \"torch_device\": \"" << (useCuda ? "cuda" : "cpu") << "\",\n"
			<< "    \"torch_threads\": " << (useHip || forceCpu ? 10 : 0) << ",\n"
			<< "    \"num_arenas\": " << arenas << ",\n"
			<< "    \"steps_per_itr\": " << steps << ",\n"
			<< "    \"note\": \"Auto-detected at GigaLearnBot startup: " << gpuName << "\"\n"
			<< "  }\n}\n";
		RG_LOG("HW: auto-detect (fallback) primary=" << primary
			<< " backend=" << backend << " profile=" << profile
			<< " arenas=" << arenas << " - " << gpuName);
		return true;
	} catch (...) {
		return false;
	}
}

bool RunPythonHwProbe(const std::filesystem::path& exeDir) {
	const auto root = FindRepoRoot(exeDir);
	if (root.empty())
		return false;
	const auto probePy = root / "tools" / "hw_probe.py";
	if (!std::filesystem::exists(probePy))
		return false;
	const auto outPath = exeDir / "hw_profile.json";
#ifdef _WIN32
	std::wstring cmd = L"python -u \"";
	cmd += probePy.wstring();
	cmd += L"\" --out \"";
	cmd += outPath.wstring();
	cmd += L"\" --quiet";
	STARTUPINFOW si{};
	si.cb = sizeof(si);
	si.dwFlags = STARTF_USESHOWWINDOW;
	si.wShowWindow = SW_HIDE;
	PROCESS_INFORMATION pi{};
	std::vector<wchar_t> buf(cmd.begin(), cmd.end());
	buf.push_back(L'\0');
	if (!CreateProcessW(nullptr, buf.data(), nullptr, nullptr, FALSE,
			CREATE_NO_WINDOW, nullptr, root.wstring().c_str(), &si, &pi)) {
		return false;
	}
	WaitForSingleObject(pi.hProcess, 60000);
	DWORD code = 1;
	GetExitCodeProcess(pi.hProcess, &code);
	CloseHandle(pi.hThread);
	CloseHandle(pi.hProcess);
	if (code != 0 || !std::filesystem::exists(outPath))
		return false;
	RG_LOG("HW: auto-detect via tools/hw_probe.py -> " << outPath.filename().string());
	return true;
#else
	(void)outPath;
	return false;
#endif
}

// Always refresh GPU vendor detection when training starts (NVIDIA -> CUDA POWER, AMD -> amd_win_20k).
void EnsureAutoGpuDetect(const std::filesystem::path& exeDir) {
	if (g_NoHwProfile)
		return;
	if (const char* env = std::getenv("GIGA_NO_HW_PROFILE")) {
		if (std::atoi(env) != 0)
			return;
	}
	if (const char* skip = std::getenv("GIGA_SKIP_AUTO_HW_PROBE")) {
		if (std::atoi(skip) != 0)
			return;
	}

	RG_LOG("HW: detecting GPU (NVIDIA / AMD) automatically...");
	const auto outPath = exeDir / "hw_profile.json";
	bool ok = RunPythonHwProbe(exeDir);
	if (!ok)
		ok = WriteFallbackHwProfile(outPath);
	if (!ok) {
		RG_LOG("WARN HW: auto-detect failed - using built-in defaults "
			"(install Python for full probe, or set GIGA_ENV_ARENAS).");
		return;
	}

	// Push probe env into this process so ApplyHwProfile + LearnerConfig see NVIDIA vs AMD.
	try {
		std::ifstream f(outPath);
		nlohmann::json j;
		f >> j;
		auto rec = j.value("recommendations", nlohmann::json::object());
		auto env = rec.value("env", nlohmann::json::object());
		if (env.is_object()) {
			for (auto it = env.begin(); it != env.end(); ++it) {
				if (!it.value().is_string() && !it.value().is_number())
					continue;
				const std::string key = it.key();
				const std::string val = it.value().is_string()
					? it.value().get<std::string>()
					: it.value().dump();
				PutEnvIfUnset(key.c_str(), val.c_str());
			}
		}
		// Force vendor keys from this run (stale GIGA_* from an old shell must not stick).
		// Opt out: GIGA_LOCK_HW_PROFILE=1 keeps existing env.
		const bool lockHw = std::getenv("GIGA_LOCK_HW_PROFILE")
			&& std::atoi(std::getenv("GIGA_LOCK_HW_PROFILE")) != 0;
		const std::string backend = rec.value("gpu_backend", "");
		const std::string profile = rec.value("profile", "");
		const std::string primary = rec.value("primary_gpu", "");
		if (!lockHw) {
			if (primary == "amd" || backend == "hip" || profile == "amd_win_20k") {
				PutEnvForce("GIGA_TRAIN_PROFILE", "amd_win_20k");
				PutEnvForce("GIGA_GPU_BACKEND", backend.empty() ? "hip" : backend.c_str());
				RG_LOG("HW: AMD path selected automatically (amd_win_20k / HIP env + CPU PPO)");
			} else if (primary == "nvidia" || backend == "cuda") {
				PutEnvForce("GIGA_GPU_BACKEND", "cuda");
				PutEnvForce("GIGA_TRAIN_PROFILE", profile.empty() ? "power_cuda" : profile.c_str());
				RG_LOG("HW: NVIDIA path selected automatically (CUDA POWER)");
			} else {
				PutEnvForce("GIGA_GPU_BACKEND", "cpu");
				PutEnvForce("GIGA_TRAIN_PROFILE", "cpu_fallback");
				RG_LOG("HW: no discrete GPU detected - CPU fallback");
			}
		} else {
			RG_LOG("HW: GIGA_LOCK_HW_PROFILE=1 - keeping existing GPU env");
		}
	} catch (...) {
		RG_LOG("WARN HW: could not apply auto-detect env from hw_profile.json");
	}
}

float ActiveOverallTargetSps() {
	if (g_AmdWin20k)
		return TrainingSize::kTargetMinOverallSps;
#ifdef GIGA_USE_CUDA_SIM
	if (!g_ForceCpuSim && !g_ProfileTorchCpu)
		return CudaPower::kTargetMinOverallSps;
#endif
	return kDefaultTargetOverallSPS;
}

// NVIDIA CUDA POWER path (not amd_win_20k / CPU-forced). Used to lock sustained Overall.
bool IsCudaPowerRuntime() {
#ifdef GIGA_USE_CUDA_SIM
	return !g_AmdWin20k && !g_ForceCpuSim && !g_ProfileTorchCpu && !g_Pure80;
#else
	return false;
#endif
}

// Undo Apex mid-run re-arm of opp/old/skill while CUDA POWER is still in the sustained-SPS window.
// Wazne 2v2: keep versions/old/opp OFF until 1B (manual milestone) — disk saves every few
// million steps were freezing the train loop ("ogni tot iterazioni").
void ClampCudaPowerSustainedSps(Learner* learner) {
	if (!learner || !IsCudaPowerRuntime())
		return;
	// Keep Rating/2v2 skill-eval ON; only strip league tax that kills Overall SPS.
	const uint64_t ts = learner->totalTimesteps;
	const uint64_t oldVersionsAt = 1'000'000'000ull; // roadmap: trainAgainstOld @ 1.0B
	if (ts < oldVersionsAt) {
		learner->config.opponentPool.enabled = false;
		learner->config.opponentPool.chance = 0.f;
		learner->config.trainAgainstOldVersions = false;
		learner->config.trainAgainstOldChance = 0.f;
		// savePolicyVersions stays false until 1B (in-RAM versions for Elo only).
		if (!learner->config.skillTracker.enabled)
			learner->config.savePolicyVersions = false;
	}
}

// amd_win_20k: keep early iters pure self-play (opp/old/skill tax Overall on CPU PPO).
// Same 1B milestone as CUDA POWER for default 2v2 — 50M was re-arming Apex league and
// freezing mid-train via PolicyVersionManager disk saves.
void ClampAmdWin20kSustainedSps(Learner* learner) {
	if (!learner || !g_AmdWin20k)
		return;
	const uint64_t oldVersionsAt = g_DefaultEnv
		? 1'000'000'000ull
		: (uint64_t)TrainingSize::kSparringDeferTs;
	if (learner->totalTimesteps < oldVersionsAt) {
		learner->config.opponentPool.enabled = false;
		learner->config.opponentPool.chance = 0.f;
		learner->config.trainAgainstOldVersions = false;
		learner->config.trainAgainstOldChance = 0.f;
		if (!learner->config.skillTracker.enabled)
			learner->config.savePolicyVersions = false;
	}
}

void WriteAmdAutosizeHint(int arenas, int steps) {
	if (g_AmdAutosizePath.empty())
		return;
	try {
		std::ofstream f(g_AmdAutosizePath, std::ios::trunc);
		if (!f)
			return;
		f << "GIGA_ENV_ARENAS=" << arenas << "\n";
		f << "GIGA_ENV_STEPS=" << steps << "\n";
		f << "GIGA_TRAIN_PROFILE=amd_win_20k\n";
		f << "# Auto-written because Overall SPS stayed under "
			<< (int)TrainingSize::kTargetMinOverallSps
			<< ". Restart run_fresh_train.bat to apply arenas.\n";
		RG_LOG("amd_win_20k: wrote autosize hint " << g_AmdAutosizePath.string()
			<< " (arenas=" << arenas << " steps=" << steps
			<< ") - restart train to apply arena change");
	} catch (...) {
	}
}

// Mid-run lean when Overall <20k (arenas cannot shrink live - ts/maxEp can).
void MaybeAutoDowngradeAmdWin20k(Learner* learner, float overallSps) {
	if (!g_AmdWin20k || !learner || overallSps <= 1.f)
		return;
	if (learner->totalIterations < 3)
		return;

	if (overallSps >= TrainingSize::kTargetMinOverallSps) {
		g_AmdLowSpsStreak = 0;
		return;
	}
	++g_AmdLowSpsStreak;
	if (g_AmdLowSpsStreak < 3)
		return;
	g_AmdLowSpsStreak = 0;

	const int arenasNow = learner->config.numGames > 0 ? learner->config.numGames : TrainingSize::kArenas;
	const int minTs = std::max(4096, arenasNow * 2 * 2);

	if (g_AmdAutoDowngradeLevel == 0) {
		int64_t ts = learner->config.ppo.tsPerItr;
		int64_t neu = (int64_t)(ts * 0.75);
		if (neu < minTs)
			neu = minTs;
		if (neu < ts) {
			learner->config.ppo.tsPerItr = neu;
			learner->config.ppo.batchSize = neu;
			++g_AmdAutoDowngradeLevel;
			RG_LOG("amd_win_20k AUTO: Overall<" << (int)TrainingSize::kTargetMinOverallSps
				<< " - shrink tsPerItr " << ts << " -> " << neu
				<< " (live Cons lean; arenas unchanged until restart)");
			return;
		}
		++g_AmdAutoDowngradeLevel;
	}
	if (g_AmdAutoDowngradeLevel == 1) {
		if (learner->config.ppo.maxEpisodeDuration > 1.05) {
			learner->config.ppo.maxEpisodeDuration = 1.0;
			++g_AmdAutoDowngradeLevel;
			RG_LOG("amd_win_20k AUTO: maxEpisodeDuration -> 1.0s (dense truncates)");
			return;
		}
		++g_AmdAutoDowngradeLevel;
	}
	if (g_AmdAutoDowngradeLevel == 2) {
		const int nextA = (arenasNow > 1536) ? 1536 : ((arenasNow > 1024) ? 1024 : arenasNow);
		WriteAmdAutosizeHint(nextA, 3);
		++g_AmdAutoDowngradeLevel;
		return;
	}
	if (g_AmdAutoDowngradeLevel == 3 && arenasNow > 1024) {
		WriteAmdAutosizeHint(1024, 2);
		++g_AmdAutoDowngradeLevel;
	}
}

void ApplyHwProfile(const std::filesystem::path& exeDir) {
	if (g_NoHwProfile)
		return;
	g_AmdAutosizePath = exeDir / "amd_win_20k_next.env";
	// Apply previous auto-downgrade arena hint before JSON (next restart after low Overall).
	if (std::filesystem::exists(g_AmdAutosizePath) && !std::getenv("GIGA_ENV_ARENAS")) {
		try {
			std::ifstream hint(g_AmdAutosizePath);
			std::string line;
			while (std::getline(hint, line)) {
				if (line.empty() || line[0] == '#')
					continue;
				const auto eq = line.find('=');
				if (eq == std::string::npos)
					continue;
				const std::string key = line.substr(0, eq);
				const std::string val = line.substr(eq + 1);
				if (key == "GIGA_ENV_ARENAS" || key == "GIGA_ENV_STEPS" || key == "GIGA_TRAIN_PROFILE") {
#ifdef _WIN32
					_putenv_s(key.c_str(), val.c_str());
#else
					setenv(key.c_str(), val.c_str(), 1);
#endif
					RG_LOG("HW: applied autosize " << key << "=" << val
						<< " from " << g_AmdAutosizePath.filename().string());
				}
			}
		} catch (...) {
		}
	}
	if (const char* env = std::getenv("GIGA_NO_HW_PROFILE")) {
		if (std::atoi(env) != 0)
			return;
	}

	if (const char* envCpu = std::getenv("GIGA_FORCE_CPU")) {
		if (std::atoi(envCpu) != 0 && !g_ExplicitCpu) {
			g_ForceCpuSim = true;
			RG_LOG("HW: GIGA_FORCE_CPU=1 -> CPU RocketSim (cudaSim=off, gpuNative=off)");
		}
	}

	// Explicit profile name from bat/probe (even before JSON read).
	if (const char* tp = std::getenv("GIGA_TRAIN_PROFILE")) {
		std::string p = tp;
		for (char& c : p) c = (char)std::tolower((unsigned char)c);
		if (p == "amd_win_20k" || p == TrainingSize::kProfileName)
			g_AmdWin20k = true;
	}
	if (const char* be = std::getenv("GIGA_GPU_BACKEND")) {
		std::string b = be;
		for (char& c : b) c = (char)std::tolower((unsigned char)c);
		if (b == "hip")
			g_AmdWin20k = true;
	}

	std::filesystem::path profilePath;
	if (const char* envP = std::getenv("GIGA_HW_PROFILE")) {
		if (envP[0])
			profilePath = envP;
	}
	if (profilePath.empty())
		profilePath = exeDir / "hw_profile.json";
	if (!std::filesystem::exists(profilePath)) {
		auto root = FindRepoRoot(exeDir);
		if (!root.empty() && std::filesystem::exists(root / "build" / "Release" / "hw_profile.json"))
			profilePath = root / "build" / "Release" / "hw_profile.json";
	}
	if (!std::filesystem::exists(profilePath)) {
		if (g_AmdWin20k) {
			// Bat already forced HIP - still lock lean knobs without JSON.
			PutEnvIfUnset("GIGA_ENV_ARENAS", std::to_string(TrainingSize::kArenas).c_str());
			PutEnvIfUnset("GIGA_ENV_STEPS", std::to_string(TrainingSize::kStepsPerItr).c_str());
			PutEnvIfUnset("GIGA_ENV_EPOCHS", std::to_string(TrainingSize::kPpoEpochs).c_str());
			PutEnvIfUnset("GIGA_ENV_MAXEP", "1.5");
			PutEnvIfUnset("GIGA_ENV_FP32", "1");
			PutEnvIfUnset("GIGA_ASYNC_OVERLAP", "0");
			PutEnvIfUnset("GIGA_TORCH_THREADS", "10");
			PutEnvIfUnset("OMP_NUM_THREADS", "10");
			PutEnvIfUnset("MKL_NUM_THREADS", "10");
			g_ProfileTorchCpu = true;
			g_AmdHipGpuNativeConfirmed = true; // bat/env claimed HIP; still verify CUDA DLLs below
			RG_LOG("HW: amd_win_20k (no hw_profile.json) - arenas="
				<< TrainingSize::kArenas << " steps=" << TrainingSize::kStepsPerItr
				<< " target Overall>=" << (int)TrainingSize::kTargetMinOverallSps);
			if (std::filesystem::exists(exeDir / "cudart64_12.dll")
				|| std::filesystem::exists(exeDir / "c10_cuda.dll")) {
				const char* be = std::getenv("GIGA_GPU_BACKEND");
				std::string b = be ? be : "";
				for (char& c : b) c = (char)std::tolower((unsigned char)c);
				if (b != "hip") {
					g_AmdHipGpuNativeConfirmed = false;
					RG_LOG("WARN amd_win_20k: CUDA DLLs present without GIGA_GPU_BACKEND=hip - "
						"rebuild tools\\build_amd.bat (docs/AMD.md).");
				}
			}
		}
		return;
	}

	try {
		std::ifstream f(profilePath);
		nlohmann::json j;
		f >> j;
		auto rec = j.value("recommendations", nlohmann::json::object());
		const bool forceCpu = rec.value("force_cpu_sim", false);
		const bool cudaAvail = rec.value("cuda_available", false);
		const bool hipAvail = rec.value("hip_available", false);
		const bool gpuSimAvail = rec.value("gpu_sim_available", cudaAvail || hipAvail);
		const int arenas = rec.value("num_arenas", -1);
		const std::string torchDev = rec.value("torch_device", "auto");
		const std::string note = rec.value("note", "");
		const std::string primary = rec.value("primary_gpu", "unknown");
		const std::string backend = rec.value("gpu_backend", cudaAvail ? "cuda" : (hipAvail ? "hip" : "cpu"));
		const std::string profileName = rec.value("profile", "");
		const int torchThreads = rec.value("torch_threads", -1);
		const bool directmlPython = rec.value("directml_python_available", false);

		RG_LOG("HW profile: " << profilePath.filename().string()
			<< " primary_gpu=" << primary
			<< " backend=" << backend
			<< " profile=" << (profileName.empty() ? "-" : profileName)
			<< " cuda=" << (cudaAvail ? "1" : "0")
			<< " hip=" << (hipAvail ? "1" : "0")
			<< " gpu_sim=" << (gpuSimAvail ? "1" : "0")
			<< " arenas_rec=" << arenas);

		if (forceCpu && !g_ExplicitCpu) {
			g_ForceCpuSim = true;
			RG_LOG("HW: CPU RocketSim fallback"
				<< (note.empty() ? "" : (std::string(" - ") + note)));
		} else if (gpuSimAvail && !g_ExplicitCpu) {
			g_ForceCpuSim = false;
			RG_LOG("HW: GPU sim path enabled (" << backend << ")"
				<< (note.empty() ? "" : (std::string(" - ") + note)));
		}
		// Only force torch CPU when profile says so - ROCm PyTorch still reports cuda.
		if (torchDev == "cpu" || torchDev == "directml")
			g_ProfileTorchCpu = true;

		// AMD HIP Windows = amd_win_20k Overall target.
		// Do NOT key off hipAvail alone: NVIDIA + AMD iGPU machines report hip=1 from
		// amdhip64.dll and wrongly flipped CUDA POWER -> amd_win_20k (sparring at 50M hang).
		const bool nvidiaPrimary = (primary == "nvidia" || backend == "cuda"
			|| profileName == "power_cuda");
		if (!nvidiaPrimary && (backend == "hip" || primary == "amd"
			|| profileName == "amd_win_20k" || profileName == "hip_scaled"
			|| profileName == "power_hip")) {
			g_AmdWin20k = true;
		} else if (nvidiaPrimary) {
			g_AmdWin20k = false;
		}

		// Apply recommended CPU torch threads for Win AMD PPO (libtorch CPU path).
		if (torchThreads > 0 && !std::getenv("GIGA_TORCH_THREADS")) {
#ifdef _WIN32
			_putenv_s("GIGA_TORCH_THREADS", std::to_string(torchThreads).c_str());
#else
			setenv("GIGA_TORCH_THREADS", std::to_string(torchThreads).c_str(), 0);
#endif
			RG_LOG("HW: GIGA_TORCH_THREADS=" << torchThreads << " (CPU PPO; override env to change)");
		}
		if (directmlPython && primary == "amd") {
			RG_LOG("HW: torch-directml importable in Python - NOT used by C++ Learner/PPO "
				"(docs/DIRECTML.md). Win HIP: docs/AMD.md");
		}

		if (g_AmdWin20k) {
			// Lock lean Overall knobs unless user already overrode.
			const int a = (arenas >= 128 && arenas <= 16384) ? arenas : TrainingSize::kArenas;
			PutEnvIfUnset("GIGA_ENV_ARENAS", std::to_string(a).c_str());
			PutEnvIfUnset("GIGA_ENV_STEPS", std::to_string(TrainingSize::kStepsPerItr).c_str());
			PutEnvIfUnset("GIGA_ENV_EPOCHS", std::to_string(TrainingSize::kPpoEpochs).c_str());
			PutEnvIfUnset("GIGA_ENV_MAXEP", "1.5");
			PutEnvIfUnset("GIGA_ENV_FP32", "1");
			PutEnvIfUnset("GIGA_ASYNC_OVERLAP", "0");
			PutEnvIfUnset("GIGA_TRAIN_PROFILE", TrainingSize::kProfileName);
			if (torchThreads <= 0)
				PutEnvIfUnset("GIGA_TORCH_THREADS", "10");
			{
				const char* tt = std::getenv("GIGA_TORCH_THREADS");
				if (!tt || !tt[0])
					tt = "10";
				PutEnvIfUnset("OMP_NUM_THREADS", tt);
				PutEnvIfUnset("MKL_NUM_THREADS", tt);
			}
			g_ProfileTorchCpu = true;
			RG_LOG("HW: amd_win_20k ON - target Overall>="
				<< (int)TrainingSize::kTargetMinOverallSps
				<< " (HIP env + CPU PPO lean512; AT/skill/opponents off at boot)");
			if (forceCpu || !hipAvail) {
				g_AmdHipGpuNativeConfirmed = false;
				RG_LOG("WARN amd_win_20k: HIP gpuNative NOT confirmed - Overall>=20k NOT guaranteed. "
					"Install HIP SDK + rebuild tools\\build_amd.bat (docs/AMD.md). "
					"If Overall <20k: see INIZIO_RAPIDO_IT.md / docs/AMD.md");
			} else {
				g_AmdHipGpuNativeConfirmed = true;
				RG_LOG("HW: HIP gpuNative=ON (probe) - target Overall>="
					<< (int)TrainingSize::kTargetMinOverallSps
					<< " (stretch ~" << (int)TrainingSize::kStretchOverallSps << ")");
			}
			// CUDA prebuilt on AMD PC: RocketSimCuda is NVIDIA CUDA - rebuild HIP required.
			if (std::filesystem::exists(exeDir / "cudart64_12.dll")
				|| std::filesystem::exists(exeDir / "cudart64_11.dll")
				|| std::filesystem::exists(exeDir / "c10_cuda.dll")) {
				const char* be = std::getenv("GIGA_GPU_BACKEND");
				std::string b = be ? be : "";
				for (char& c : b) c = (char)std::tolower((unsigned char)c);
				if (b != "hip") {
					g_AmdHipGpuNativeConfirmed = false;
					RG_LOG("WARN amd_win_20k: NVIDIA CUDA DLLs next to exe with AMD GPU - "
						"this build will NOT use the Radeon. Run tools\\build_amd.bat "
						"(docs/AMD.md). Overall>=20k NOT guaranteed until HIP rebuild.");
				}
			}
		}

		if (arenas >= 128 && arenas <= 16384) {
			if (!std::getenv("GIGA_ENV_ARENAS"))
				g_ProfileArenas = arenas;
		}
		if (g_AmdWin20k && g_ProfileArenas < 0 && !std::getenv("GIGA_ENV_ARENAS"))
			g_ProfileArenas = TrainingSize::kArenas;
		if (g_ProfileArenas > 0)
			RG_LOG("HW: recommended arenas=" << g_ProfileArenas
				<< " (override: TRAINING SIZE / GIGA_ENV_ARENAS)");
	} catch (const std::exception& e) {
		RG_LOG("HW: failed to read " << profilePath << " (" << e.what() << ")");
	} catch (...) {
		RG_LOG("HW: failed to read " << profilePath);
	}
}

void MaybeStartAutoTrainer(const std::filesystem::path& exeDir) {
	if (g_NoAutoTrainer)
		return;
	if (const char* env = std::getenv("GIGA_NO_AUTOTRAINER")) {
		if (std::atoi(env) != 0) {
			RG_LOG("AutoTrainer: skipped (GIGA_NO_AUTOTRAINER=1)");
			return;
		}
	}
	if (const char* ext = std::getenv("GIGA_AUTOTRAINER_EXTERNAL")) {
		if (std::atoi(ext) != 0) {
			RG_LOG("AutoTrainer: external launcher already started (GIGA_AUTOTRAINER_EXTERNAL=1)");
			return;
		}
	}
	// Self-spawn is opt-in only.
	{
		bool allowSpawn = false;
		if (const char* at = std::getenv("GIGA_AUTOTRAINER")) {
			if (std::atoi(at) != 0)
				allowSpawn = true;
		}
		if (!allowSpawn) {
			RG_LOG("AutoTrainer: skipped (off by default; set GIGA_AUTOTRAINER=1 or use run_with_autotrainer.bat)");
			return;
		}
	}

	const auto root = FindRepoRoot(exeDir);
	if (root.empty()) {
		RG_LOG("AutoTrainer: repo root not found (orchestrator.py missing) - skipped");
		return;
	}
	const auto orch = root / "autotrainer" / "orchestrator.py";
	const auto watch = exeDir / "autotrainer";
	std::error_code ec;
	std::filesystem::create_directories(watch, ec);
	std::filesystem::create_directories(watch / "profiles", ec);
	// Best-effort copy of default config/profiles next to the exe watch dir.
	try {
		const auto srcProfiles = root / "autotrainer" / "profiles";
		if (std::filesystem::exists(srcProfiles)) {
			for (auto& ent : std::filesystem::directory_iterator(srcProfiles)) {
				if (!ent.is_regular_file()) continue;
				std::filesystem::copy_file(
					ent.path(), watch / "profiles" / ent.path().filename(),
					std::filesystem::copy_options::overwrite_existing, ec);
			}
		}
		const auto cfgSrc = root / "autotrainer" / "config.default.yaml";
		if (std::filesystem::exists(cfgSrc))
			std::filesystem::copy_file(
				cfgSrc, watch / "config.default.yaml",
				std::filesystem::copy_options::overwrite_existing, ec);
	} catch (...) {}

	const char* profile = std::getenv("GIGA_AUTOTRAINER_PROFILE");
	if (!profile || !profile[0])
		profile = "default";

#ifdef _WIN32
	// Non-blocking spawn via visible console + tee helper (same UX as launch_autotrainer.bat).
	// Never wait on the child - Learner Start must not hang if Python is slow/missing.
	const auto tee = root / "tools" / "run_autotrainer_console.py";
	std::wstring cmd;
	if (std::filesystem::exists(tee)) {
		cmd = L"cmd.exe /k set PYTHONUNBUFFERED=1&& set PYTHONIOENCODING=utf-8&& python -u \""
			+ tee.wstring() + L"\" \"" + root.wstring() + L"\" "
			+ std::wstring(profile, profile + std::strlen(profile))
			+ L" \"" + watch.wstring() + L"\"";
	} else {
		cmd = L"cmd.exe /k set PYTHONUNBUFFERED=1&& set PYTHONIOENCODING=utf-8&& python -u \""
			+ orch.wstring() + L"\" --profile "
			+ std::wstring(profile, profile + std::strlen(profile))
			+ L" --watch-dir \"" + watch.wstring() + L"\"";
	}
	std::vector<wchar_t> cmdBuf(cmd.begin(), cmd.end());
	cmdBuf.push_back(L'\0');

	STARTUPINFOW si{};
	si.cb = sizeof(si);
	PROCESS_INFORMATION pi{};
	std::wstring workDir = root.wstring();
	BOOL ok = CreateProcessW(
		nullptr,
		cmdBuf.data(),
		nullptr,
		nullptr,
		FALSE,
		CREATE_NEW_CONSOLE | CREATE_NEW_PROCESS_GROUP,
		nullptr,
		workDir.c_str(),
		&si,
		&pi
	);
	if (!ok) {
		RG_LOG("AutoTrainer: CreateProcess failed (is python on PATH?) - skipped");
		return;
	}
	const DWORD pid = pi.dwProcessId;
	CloseHandle(pi.hThread);
	CloseHandle(pi.hProcess);
	RG_LOG("AutoTrainer started PID=" << pid
		<< " profile=" << profile
		<< " watch=" << watch);
	RG_LOG("AutoTrainer: disable with GIGA_NO_AUTOTRAINER=1 or --no-autotrainer (or omit GIGA_AUTOTRAINER)");
#else
	(void)profile;
	RG_LOG("AutoTrainer: auto-spawn is Windows-only - start manually: python -m autotrainer");
#endif
}

