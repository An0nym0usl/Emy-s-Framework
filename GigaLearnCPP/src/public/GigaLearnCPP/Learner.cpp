#include "Learner.h"

#include <GigaLearnCPP/PPO/PPOLearner.h>
#include <GigaLearnCPP/PPO/ExperienceBuffer.h>

#include <algorithm>
#include <cstring>
#include <future>
#include <torch/cuda.h>
#include <nlohmann/json.hpp>
#include <pybind11/embed.h>

#ifdef RG_CUDA_SUPPORT
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#endif
#include <private/GigaLearnCPP/PPO/ExperienceBuffer.h>
#include <private/GigaLearnCPP/PPO/GAE.h>
#include <private/GigaLearnCPP/PolicyVersionManager.h>
#include <private/GigaLearnCPP/OpponentPool.h>

#ifdef GIGA_USE_CUDA_SIM
#include <GigaLearnCPP/Sim/CudaEnvSet.h>
#endif

#include "Util/KeyPressDetector.h"
#include "Util/IterProfiler.h"
#include <private/GigaLearnCPP/Util/WelfordStat.h>
#include "Util/AvgTracker.h"

#include <atomic>
#include <cctype>
#include <cstdlib>
#include <string>
#include <thread>

using namespace RLGC;

namespace {
	// Maximize CPU PPO on Win AMD (HIP env on GPU, learn on CPU libtorch).
	// GIGA_TORCH_THREADS / OMP_NUM_THREADS override; else leave ~2 logical cores for host/HIP.
	inline void ConfigureCpuTorchThreads() {
		unsigned hw = std::thread::hardware_concurrency();
		if (hw == 0)
			hw = 4;
		int threads = 0;
		if (const char* env = std::getenv("GIGA_TORCH_THREADS")) {
			if (env[0])
				threads = std::atoi(env);
		}
		if (threads <= 0) {
			if (const char* omp = std::getenv("OMP_NUM_THREADS")) {
				if (omp[0])
					threads = std::atoi(omp);
			}
		}
		if (threads <= 0) {
			// Ryzen 5 3600 = 12 logical: use 10. Weak 4c laptops: leave 1 free.
			const unsigned reserve = (hw >= 8) ? 2u : 1u;
			threads = (int)((hw > reserve) ? (hw - reserve) : 1u);
		}
		if (threads < 1)
			threads = 1;
		if (threads > (int)hw)
			threads = (int)hw;

		int interop = 0;
		if (const char* envI = std::getenv("GIGA_TORCH_INTEROP_THREADS")) {
			if (envI[0])
				interop = std::atoi(envI);
		}
		if (interop <= 0)
			interop = std::max(1, std::min(4, threads / 4));
		if (interop > threads)
			interop = threads;

		torch::set_num_threads(threads);
		torch::set_num_interop_threads(interop);
		RG_LOG("\tCPU torch threads: intra-op=" << threads
			<< " interop=" << interop
			<< " (hw=" << hw
			<< "; override GIGA_TORCH_THREADS / GIGA_TORCH_INTEROP_THREADS)");
	}

	inline bool EnvFlagTruthy(const char* name) {
		const char* v = std::getenv(name);
		if (!v || !v[0])
			return false;
		return v[0] == '1' || v[0] == 't' || v[0] == 'T' || v[0] == 'y' || v[0] == 'Y';
	}
}

namespace {
	inline void AppendContinuousTeacherTarget(const Action& action, std::vector<float>& outValues) {
		outValues.push_back(action.throttle);
		outValues.push_back(action.steer);
		outValues.push_back(action.pitch);
		outValues.push_back(action.yaw);
		outValues.push_back(action.roll);
		outValues.push_back(action.jump > 0 ? 1.0f : -1.0f);
		outValues.push_back(action.boost > 0 ? 1.0f : -1.0f);
		outValues.push_back(action.handbrake > 0 ? 1.0f : -1.0f);
	}

	inline void AppendContinuousTeacherActionTable(
		ActionParser& parser,
		const Player& player,
		const GameState& state,
		std::vector<float>& outValues) {

		int numActions = parser.GetActionAmount();
		for (int actionIdx = 0; actionIdx < numActions; actionIdx++)
			AppendContinuousTeacherTarget(parser.ParseAction(actionIdx, player, state), outValues);
	}

	struct CollectEnvView {
		RLGC::EnvSet* cpu = nullptr;
#ifdef GIGA_USE_CUDA_SIM
		GGL::CudaEnvSet* cuda = nullptr;
#endif

		RLGC::DimList2<float>* obs = nullptr;
		RLGC::DimList2<uint8_t>* actionMasks = nullptr;
		std::vector<float>* rewards = nullptr;
		std::vector<uint8_t>* terminals = nullptr;
		std::vector<RLGC::GameState>* gameStates = nullptr;
		std::vector<int>* arenaPlayerStartIdx = nullptr;
		std::vector<std::vector<RLGC::WeightedReward>>* rewardSets = nullptr;
		std::vector<std::vector<float>>* lastRewards = nullptr;
		int numPlayers = 0;
		int numArenas = 0;

		static CollectEnvView FromLearner(GGL::Learner* learner) {
			CollectEnvView v = {};
#ifdef GIGA_USE_CUDA_SIM
			if (learner->cudaEnvSet) {
				auto* e = learner->cudaEnvSet;
				v.cuda = e;
				v.obs = &e->obs;
				v.actionMasks = &e->actionMasks;
				v.rewards = &e->rewards_out;
				v.terminals = &e->terminals;
				v.gameStates = &e->gameStates;
				v.arenaPlayerStartIdx = &e->arenaPlayerStartIdx;
				v.rewardSets = &e->rewards;
				v.lastRewards = &e->lastRewards;
				v.numPlayers = e->numPlayers;
				v.numArenas = e->numArenas;
				return v;
			}
#endif
			v.cpu = learner->envSet;
			v.obs = &learner->envSet->state.obs;
			v.actionMasks = &learner->envSet->state.actionMasks;
			v.rewards = &learner->envSet->state.rewards;
			v.terminals = &learner->envSet->state.terminals;
			v.gameStates = &learner->envSet->state.gameStates;
			v.arenaPlayerStartIdx = &learner->envSet->state.arenaPlayerStartIdx;
			v.rewardSets = &learner->envSet->rewards;
			v.lastRewards = &learner->envSet->state.lastRewards;
			v.numPlayers = learner->envSet->state.numPlayers;
			v.numArenas = (int)learner->envSet->arenas.size();
			return v;
		}

		void Reset() {
#ifdef GIGA_USE_CUDA_SIM
			if (cuda) { cuda->Reset(); return; }
#endif
			cpu->Reset();
		}

		void StepFirstHalf(bool async) {
#ifdef GIGA_USE_CUDA_SIM
			if (cuda) return;
#endif
			cpu->StepFirstHalf(async);
		}

		void Sync() {
#ifdef GIGA_USE_CUDA_SIM
			if (cuda) return;
#endif
			cpu->Sync();
		}

		void StepSecondHalfContinuous(const RLGC::FList& actions, int actionDim, bool async) {
#ifdef GIGA_USE_CUDA_SIM
			if (cuda) {
				cuda->StepContinuous(actions, actionDim);
				return;
			}
#endif
			cpu->StepSecondHalfContinuous(actions, actionDim, async);
		}

		void StepSecondHalfDiscrete(const std::vector<int>& actions, bool async) {
#ifdef GIGA_USE_CUDA_SIM
			if (cuda) {
				cuda->StepDiscrete(actions);
				return;
			}
#endif
			cpu->StepSecondHalf(actions, async);
		}

#ifdef GIGA_USE_CUDA_SIM
		void StepSecondHalfDiscreteDevice(const int* deviceActionIndices) {
			if (cuda)
				cuda->StepDiscreteDevice(deviceActionIndices);
		}
#endif
	};

	void SyncRolloutPolicy(GGL::ModelSet& rollout, GGL::ModelSet& train) {
		RG_NO_GRAD;
		for (GGL::Model* src : train) {
			if (!src)
				continue;
			const char* name = src->modelName;
			if (!name)
				continue;
			// Rollout only needs shared_head + policy (Infer path).
			if (std::strcmp(name, "critic") == 0)
				continue;
			GGL::Model* dst = rollout[name];
			if (!dst)
				continue;
			auto fromParams = src->parameters();
			auto toParams = dst->parameters();
			for (size_t i = 0; i < fromParams.size() && i < toParams.size(); i++)
				toParams[i].copy_(fromParams[i], /*non_blocking=*/true);
			dst->_seqHalfOutdated = true;
		}
	}
}

GGL::Learner::Learner(EnvCreateFn envCreateFn, LearnerConfig config, StepCallbackFn stepCallback) :
	envCreateFn(envCreateFn), config(config), stepCallback(stepCallback)
{
	// The embedded Python interpreter is only needed for wandb metrics (MetricSender)
	// and for rendering (RenderSender). When neither is enabled we skip initializing it,
	// so training does not require a working Python/wandb install and won't crash if absent.
	if (config.sendMetrics || config.renderMode) {
		pybind11::initialize_interpreter();
		_pythonInitialized = true;
	}

#ifndef NDEBUG
	RG_LOG("===========================");
	RG_LOG("WARNING: GigaLearn runs extremely slowly in debug, and there are often bizzare issues with debug-mode torch.");
	RG_LOG("It is recommended that you compile in release mode without optimization for debugging.");
	RG_SLEEP(1000);
#endif

	if (config.tsPerSave == 0)
		config.tsPerSave = config.ppo.tsPerItr;

	RG_LOG("Learner::Learner():");

	if (config.randomSeed == -1)
		config.randomSeed = RS_CUR_MS();

	RG_LOG("\tCheckpoint Save/Load Dir: " << config.checkpointFolder);

	torch::manual_seed(config.randomSeed);

	at::Device device = at::Device(at::kCPU);

	// DirectML gate: torch-directml is Python PrivateUse1 only — no C++ libtorch build.
	// Env GIGA_TORCH_DIRECTML=1 (or GIGA_TORCH_DEVICE=directml) is acknowledged then CPU is used.
	const bool wantDirectML =
		EnvFlagTruthy("GIGA_TORCH_DIRECTML") ||
		[]() {
			if (const char* d = std::getenv("GIGA_TORCH_DEVICE")) {
				std::string s = d;
				for (char& c : s) c = (char)std::tolower((unsigned char)c);
				return s == "directml" || s == "dml" || s == "privateuse1"
					|| s.find("privateuse") != std::string::npos;
			}
			return false;
		}();
	if (wantDirectML) {
		RG_LOG("\tDirectML requested (GIGA_TORCH_DIRECTML / GIGA_TORCH_DEVICE=directml):");
		RG_LOG("\t  C++ Learner uses stock libtorch — Microsoft ships no libtorch+DirectML.");
		RG_LOG("\t  pip torch-directml does NOT accelerate GigaLearnBot. See docs/DIRECTML.md.");
		RG_LOG("\t  PPO learn stays on CPU (threaded). See docs/AMD.md.");
		RG_LOG("\t  Native Win: HIP gpuNative env (tools/build_amd.bat) + GIGA_TORCH_THREADS.");
		if (config.deviceType == LearnerDeviceType::GPU_CUDA)
			config.deviceType = LearnerDeviceType::CPU;
	}

	if (
		config.deviceType == LearnerDeviceType::GPU_CUDA || 
		(config.deviceType == LearnerDeviceType::AUTO && torch::cuda::is_available())
		) {
		RG_LOG("\tUsing CUDA GPU device...");

		// Test out moving a tensor to GPU and back to make sure the device is working
		torch::Tensor t;
		bool deviceTestFailed = false;
		try {
			t = torch::tensor(0);
			t = t.to(at::Device(at::kCUDA));
			t = t.cpu();
		} catch (...) {
			deviceTestFailed = true;
		}

		// Pick fast cuDNN algos for steady PPO gemms (helps Cons SPS after warmup).
		torch::globalContext().setBenchmarkCuDNN(true);
		torch::globalContext().setAllowTF32CuDNN(true);
		torch::globalContext().setAllowTF32CuBLAS(true);

		if (!torch::cuda::is_available() || deviceTestFailed)
			RG_ERR_CLOSE(
				"Learner::Learner(): Can't use CUDA GPU because " <<
				(torch::cuda::is_available() ? "libtorch cannot access the GPU" : "CUDA is not available to libtorch") << ".\n" <<
				"Make sure your libtorch comes with CUDA support, and that CUDA is installed properly."
			)
		device = at::Device(at::kCUDA);
	} else {
		RG_LOG("\tUsing CPU device...");
		if (!torch::cuda::is_available()) {
			RG_LOG("\t  (libtorch has no CUDA/ROCm GPU — typical on Windows AMD HIP builds)");
		}
		ConfigureCpuTorchThreads();
		device = at::Device(at::kCPU);
	}

	if (RocketSim::GetStage() != RocketSimStage::INITIALIZED) {
		RG_LOG("\tInitializing RocketSim...");
		RocketSim::Init("collision_meshes", true);
	}

	{
		RG_LOG("\tCreating envs...");
		storedEnvSetConfig = {};
		storedEnvSetConfig.envCreateFn = envCreateFn;
		storedEnvSetConfig.numArenas = config.renderMode ? 1 : config.numGames;
		storedEnvSetConfig.tickSkip = config.tickSkip;
		storedEnvSetConfig.actionDelay = config.actionDelay;
		storedEnvSetConfig.saveRewards = config.addRewardsToMetrics || config.renderMode;
		// ContinuousV2/Leak: stable lowest-id reward sample in render (RocketSimVis "You").
		storedEnvSetConfig.shuffleRewardSampling = !config.renderMode;

#ifdef GIGA_USE_CUDA_SIM
		bool useCuda =
			config.useCudaSim &&
			!config.renderMode;

		if (useCuda) {
			RG_LOG("\tUsing CUDA GPU-native env (physics + AdvancedObs + rewards on device)...");
			int carsPerTeam = 1;
			// Peek at first arena to count players per team
			{
				EnvCreateResult peek = envCreateFn(0);
				carsPerTeam = (int)peek.arena->_cars.size() / 2;
				delete peek.arena;
				for (auto& w : peek.rewards) delete w.reward;
				for (auto* tc : peek.terminalConditions) delete tc;
				delete peek.obsBuilder;
				delete peek.actionParser;
				delete peek.continuousActionParser;
				delete peek.stateSetter;
			}

			auto cudaCreateFn = [envCreateFn](int arenaIdx) -> CudaEnvCreateResult {
				EnvCreateResult cpu = envCreateFn(arenaIdx);
				CudaEnvCreateResult r = {};
				r.obsBuilder = cpu.obsBuilder;
				r.rewards = cpu.rewards;
				r.terminalConditions = cpu.terminalConditions;
				r.actionParser = cpu.actionParser;
				r.continuousActionParser = cpu.continuousActionParser;
				delete cpu.arena;
				delete cpu.stateSetter;
				cpu.obsBuilder = nullptr;
				cpu.rewards.clear();
				cpu.terminalConditions.clear();
				cpu.actionParser = nullptr;
				cpu.continuousActionParser = nullptr;
				return r;
			};

			cudaEnvSet = new CudaEnvSet(
				storedEnvSetConfig.numArenas, carsPerTeam, config.tickSkip, cudaCreateFn,
				config.cudaPreferGpuNative, config.cudaRewardProfile, config.cudaNoTouchSeconds);
			if (config.cudaDeviceExperience && cudaEnvSet->gpuNative) {
				cudaEnvSet->skipHostObsCopy = true;
				RG_LOG("\tCUDA device-experience bank ON (skip host obs/mask D2H)");
			}
			envSet = nullptr;
			obsSize = cudaEnvSet->obsSize;
			if (cudaEnvSet->gpuNative) {
				RG_LOG("\tCUDA gpuNative rewardProfile=" << config.cudaRewardProfile
					<< " noTouch=" << config.cudaNoTouchSeconds << "s"
					<< (config.cudaDeviceExperience ? " deviceXP=1" : ""));
			} else {
				RG_LOG("\tCUDA hybrid (CPU rewards/obs bridge)");
			}
		} else {
			envSet = new RLGC::EnvSet(storedEnvSetConfig);
			obsSize = envSet->state.obs.size[1];
		}
#else
		envSet = new RLGC::EnvSet(storedEnvSetConfig);
		obsSize = envSet->state.obs.size[1];
#endif

		if (config.ppo.policyType == PolicyType::CONTINUOUS) {
			numActions = config.ppo.continuousActionSize;
		} else {
#ifdef GIGA_USE_CUDA_SIM
			if (cudaEnvSet)
				numActions = cudaEnvSet->actionParsers[0]->GetActionAmount();
			else
#endif
				numActions = envSet->actionParsers[0]->GetActionAmount();
		}
	}

	{
		if (config.standardizeReturns) {
			this->returnStat = new WelfordStat();
		} else {
			this->returnStat = NULL;
		}

		if (config.standardizeObs) {
			this->obsStat = new BatchedWelfordStat(obsSize);
		} else {
			this->obsStat = NULL;
		}
	}

	try {
		RG_LOG("\tMaking PPO learner...");
		ppo = new PPOLearner(obsSize, numActions, config.ppo, device);
	} catch (std::exception& e) {
		RG_ERR_CLOSE("Failed to create PPO learner: " << e.what());
	}

	if (config.renderMode) {
		renderSender = new RenderSender(config.renderTimeScale);
	} else {
		renderSender = NULL;
	}

	if (config.skillTracker.enabled || config.trainAgainstOldVersions)
		config.savePolicyVersions = true;

	if (config.savePolicyVersions && !config.renderMode) {
		if (config.checkpointFolder.empty())
			RG_ERR_CLOSE("Cannot save/load old policy versions with no checkpoint save folder");
		versionMgr = new PolicyVersionManager(
			config.checkpointFolder / "policy_versions", config.maxOldVersions, config.tsPerVersion,
			config.skillTracker, storedEnvSetConfig
		);
	} else {
		versionMgr = NULL;
	}

	if (!config.checkpointFolder.empty())
		Load();

	if (config.savePolicyVersions && !config.renderMode) {
		if (config.checkpointFolder.empty())
			RG_ERR_CLOSE("Cannot save/load old policy versions with no checkpoint save folder");
		auto models = ppo->GetPolicyModels();
		versionMgr->LoadVersions(models, totalTimesteps);
	}

	if (config.opponentPool.enabled && !config.renderMode) {
		opponentPool = new OpponentPool();
		opponentPool->Load(config.opponentPool, ppo, device);
	} else {
		opponentPool = nullptr;
	}

	if (config.sendMetrics && !config.renderMode) {
		if (!runID.empty())
			RG_LOG("\tRun ID: " << runID);
		metricSender = new MetricSender(config.metricsProjectName, config.metricsGroupName, config.metricsRunName, runID);
	} else {
		metricSender = NULL;
	}

	RG_LOG(RG_DIVIDER);
}

void GGL::Learner::SaveStats(std::filesystem::path path) {
	using namespace nlohmann;

	constexpr const char* ERROR_PREFIX = "Learner::SaveStats(): ";

	// Remove a half-written stats file from an interrupted save (all-zero RUNNING_STATS.json)
	std::error_code removeEc;
	std::filesystem::remove(path, removeEc);

	json j = {};
	j["total_timesteps"] = totalTimesteps;
	j["total_iterations"] = totalIterations;

	if (config.sendMetrics)
		j["run_id"] = metricSender->curRunID;

	if (returnStat)
		j["return_stat"] = returnStat->ToJSON();
	if (obsStat)
		j["obs_stat"] = obsStat->ToJSON();

	if (versionMgr)
		versionMgr->AddRunningStatsToJSON(j);

	std::string jStr = j.dump(4);

	// Atomic write: crash mid-save must not leave a zero-filled RUNNING_STATS.json
	std::filesystem::path tempPath = path;
	tempPath += ".tmp";
	{
		std::ofstream fTemp(tempPath, std::ios::binary | std::ios::trunc);
		if (!fTemp.good())
			RG_ERR_CLOSE(ERROR_PREFIX << "Can't open temp file at " << tempPath);
		fTemp << jStr;
		fTemp.flush();
		if (!fTemp.good())
			RG_ERR_CLOSE(ERROR_PREFIX << "Failed to write temp stats at " << tempPath);
	}
	std::error_code ec;
	std::filesystem::rename(tempPath, path, ec);
	if (ec) {
		std::filesystem::remove(path, ec);
		std::filesystem::rename(tempPath, path, ec);
		if (ec)
			RG_ERR_CLOSE(ERROR_PREFIX << "Failed to replace stats at " << path << ", exception: " << ec.message());
	}
}

void GGL::Learner::LoadStats(std::filesystem::path path) {
	// TODO: Repetitive code, merge repeated code into one function called from both SaveStats() and LoadStats()

	using namespace nlohmann;
	constexpr const char* ERROR_PREFIX = "Learner::LoadStats(): ";

	std::ifstream fIn(path);
	if (!fIn.good())
		RG_ERR_CLOSE(ERROR_PREFIX << "Can't open file at " << path);

	json j = json::parse(fIn);
	totalTimesteps = j["total_timesteps"];
	totalIterations = j["total_iterations"];

	if (j.contains("run_id"))
		runID = j["run_id"];

	if (returnStat)
		returnStat->ReadFromJSON(j["return_stat"]);
	if (obsStat)
		obsStat->ReadFromJSON(j["obs_stat"]);

	if (versionMgr)
		versionMgr->LoadRunningStatsFromJSON(j);
}

// Different than RLGym-PPO to show that they are not compatible
constexpr const char* STATS_FILE_NAME = "RUNNING_STATS.json";

namespace {

bool CheckpointHasModelFiles(const std::filesystem::path& folder) {
	const char* required[] = { "POLICY.lt", "CRITIC.lt", "SHARED_HEAD.lt" };
	for (const char* name : required) {
		auto path = folder / name;
		if (!std::filesystem::exists(path))
			return false;
		if (std::filesystem::file_size(path) < 1024)
			return false;
	}
	return true;
}

bool IsValidRunningStatsFile(const std::filesystem::path& path) {
	if (!std::filesystem::exists(path))
		return false;

	std::ifstream fIn(path, std::ios::binary);
	if (!fIn.good())
		return false;

	char first = 0;
	fIn.read(&first, 1);
	if (!fIn.good() || first != '{')
		return false;

	fIn.seekg(0);
	try {
		nlohmann::json j = nlohmann::json::parse(fIn);
		return j.contains("total_timesteps") && j["total_timesteps"].is_number_integer();
	} catch (...) {
		return false;
	}
}

} // namespace

void GGL::Learner::Save() {
	if (config.checkpointFolder.empty())
		RG_ERR_CLOSE("Learner::Save(): Cannot save because config.checkpointSaveFolder is not set");

#ifdef GIGA_USE_CUDA_SIM
	if (AsyncLearnInFlight()) {
		deferredCheckpointSave.store(true, std::memory_order_release);
		RequestDeferredCudaSurfaceApply();
		RG_LOG("Learner::Save(): deferred (async Learn in flight)");
		return;
	}
#endif

	std::filesystem::path saveFolder = config.checkpointFolder / std::to_string(totalTimesteps);
	std::filesystem::create_directories(saveFolder);

	RG_LOG("Saving to folder " << saveFolder << "...");
	Timer saveTimer = {};
	// Models first, stats last — avoids a numbered folder with corrupt/empty RUNNING_STATS.json
	ppo->SaveTo(saveFolder);
	SaveStats(saveFolder / STATS_FILE_NAME);

	// Remove old checkpoints (optional — mid-train delete often hangs under AV/OneDrive)
	if (config.checkpointsToKeep != -1) {
		std::set<int64_t> allSavedTimesteps = Utils::FindNumberedDirs(config.checkpointFolder);
		while (allSavedTimesteps.size() > config.checkpointsToKeep) {
			int64_t lowestCheckpointTS = INT64_MAX;
			for (int64_t savedTimesteps : allSavedTimesteps)
				lowestCheckpointTS = RS_MIN(lowestCheckpointTS, savedTimesteps);

			std::filesystem::path removePath = config.checkpointFolder / std::to_string(lowestCheckpointTS);
			try {
				std::filesystem::remove_all(removePath);
			} catch (std::exception& e) {
				// OneDrive / AV locks must not kill training mid-save.
				RG_LOG("Warning: could not remove old checkpoint " << removePath
					<< " (" << e.what() << ") - continuing");
				allSavedTimesteps.erase(lowestCheckpointTS);
				break;
			}
			allSavedTimesteps.erase(lowestCheckpointTS);
		}
	}

	// Only flush policy_versions when that feature is actually enabled (otherwise disk thrash).
	if (versionMgr && config.savePolicyVersions)
		versionMgr->SaveVersions();

	RG_LOG(" > Done. (" << (saveTimer.Elapsed() * 1000.0) << "ms)");
}

void GGL::Learner::Load() {
	if (config.checkpointFolder.empty())
		RG_ERR_CLOSE("Learner::Load(): Cannot load because config.checkpointLoadFolder is not set");

	RG_LOG("Loading most recent checkpoint in " << config.checkpointFolder << "...");

	std::set<int64_t> allSavedTimesteps = Utils::FindNumberedDirs(config.checkpointFolder);
	bool loaded = false;
	for (auto it = allSavedTimesteps.rbegin(); it != allSavedTimesteps.rend(); ++it) {
		std::filesystem::path loadFolder = config.checkpointFolder / std::to_string(*it);
		auto statsPath = loadFolder / STATS_FILE_NAME;

		if (!CheckpointHasModelFiles(loadFolder)) {
			RG_LOG(" > Skipping checkpoint " << *it << " (missing or incomplete model files)");
			continue;
		}
		if (!IsValidRunningStatsFile(statsPath)) {
			RG_LOG(" > Skipping checkpoint " << *it << " (invalid or corrupt " << STATS_FILE_NAME << ")");
			continue;
		}

		RG_LOG(" > Loading checkpoint " << loadFolder << "...");
		LoadStats(statsPath);
		ppo->LoadFrom(loadFolder);
		RG_LOG(" > Done.");
		loaded = true;
		break;
	}

	if (!loaded) {
		RG_LOG(" > No valid checkpoints found, starting new model.")
	}
}

// Non-blocking poll for the 'Q' (save and quit) key. Returns true once Q has been pressed.
// Replaces the old detached infinite-loop thread, allowing a clean shutdown via loop break.
static bool PollQuitKey(bool& quitPressed) {
	if (!quitPressed) {
		char c = toupper(GGL::KeyPressDetector::GetPressedCharNonBlocking());
		if (c == 'Q') {
			RG_LOG("Save queued, will save and exit at the end of this iteration.");
			quitPressed = true;
		}
	}
	return quitPressed;
}

void GGL::Learner::StartTransferLearn(const TransferLearnConfig& tlConfig) {

	RG_LOG("Starting transfer learning...");
	bool isContinuous = (config.ppo.policyType == PolicyType::CONTINUOUS);
	bool oldIsContinuous = (tlConfig.oldPolicyType == PolicyType::CONTINUOUS);
	bool distillDiscreteToContinuous = isContinuous && !oldIsContinuous;
	auto fnGetStudentRolloutProb = [&]() {
		float result = tlConfig.discreteToContinuousStudentRolloutProb;
		if (distillDiscreteToContinuous && tlConfig.discreteToContinuousStudentRolloutWarmupTimesteps > 0) {
			float warmupFrac = RS_CLAMP(
				totalTimesteps / (float)tlConfig.discreteToContinuousStudentRolloutWarmupTimesteps,
				0.0f, 1.0f
			);
			result *= warmupFrac;
		}
		return result;
	};

	if (distillDiscreteToContinuous && config.ppo.continuousActionSize != 8) {
		RG_ERR_CLOSE("StartTransferLearn: discrete-teacher -> continuous-student distillation currently expects continuousActionSize == 8");
	}
	if (tlConfig.discreteToContinuousStudentRolloutProb < 0 || tlConfig.discreteToContinuousStudentRolloutProb > 1) {
		RG_ERR_CLOSE("StartTransferLearn: tlConfig.discreteToContinuousStudentRolloutProb must be in [0, 1]");
	}
	if (distillDiscreteToContinuous && tlConfig.discreteToContinuousStudentRolloutProb > 0) {
		RG_LOG("Discrete -> continuous TL rollout mixing enabled (base student rollout probability: " << tlConfig.discreteToContinuousStudentRolloutProb << ")");
	}

	// TODO: Lots of manual obs builder stuff going on which is quite volatile
	//	Although I can't really think another way to do this

	std::vector<ObsBuilder*> oldObsBuilders = {};
	for (int i = 0; i < envSet->arenas.size(); i++)
		oldObsBuilders.push_back(tlConfig.makeOldObsFn());

	// Reset all obs builders initially
	for (int i = 0; i < envSet->arenas.size(); i++)
		oldObsBuilders[i]->Reset(envSet->state.gameStates[0]);

	std::vector<ActionParser*> oldActionParsers = {};
	if (!oldIsContinuous) {
		for (int i = 0; i < envSet->arenas.size(); i++)
			oldActionParsers.push_back(tlConfig.makeOldActFn());
	}

	int oldNumActions = oldIsContinuous ? tlConfig.oldContinuousActionSize : oldActionParsers[0]->GetActionAmount();
	bool canTeacherForceContinuousRollout =
		isContinuous && oldIsContinuous &&
		(tlConfig.oldContinuousActionSize == config.ppo.continuousActionSize);
	bool canTeacherForceDiscreteRollout =
		!isContinuous && !oldIsContinuous &&
		(oldNumActions == numActions) &&
		!tlConfig.mapActsFn;

	if (!isContinuous && !oldIsContinuous && oldNumActions != numActions) {
		if (!tlConfig.mapActsFn) {
			RG_ERR_CLOSE(
				"StartTransferLearn: Old and new action parsers have a different number of actions, but tlConfig.mapActsFn is NULL.\n" <<
				"You must implement this function to translate the action indices."
			);
		};
	}

	// Determine old obs size
	int oldObsSize;
	{
		GameState testState = envSet->state.gameStates[0];
		oldObsSize = oldObsBuilders[0]->BuildObs(testState.players[0], testState).size();
	}

	ModelSet oldModels = {};
	{
		RG_NO_GRAD;
		int oldPolicyOutputs =
			oldIsContinuous ? PPOLearner::GetContinuousPolicyOutputSize(tlConfig.oldContinuousActionSize) : oldNumActions;
		PPOLearner::MakeModels(
			false, oldObsSize, oldPolicyOutputs,
			tlConfig.oldSharedHeadConfig, tlConfig.oldPolicyConfig, {},
			ppo->device, oldModels,
			tlConfig.oldPolicyType, tlConfig.oldContinuousActionSize
		);

		oldModels.Load(tlConfig.oldModelsPath, false, false);
	}

	if (oldIsContinuous && isContinuous && !canTeacherForceContinuousRollout)
		RG_LOG("Transfer learning rollout will fall back to student actions because old/new continuous action sizes differ.");
	if (!distillDiscreteToContinuous && !oldIsContinuous && !isContinuous && !canTeacherForceDiscreteRollout)
		RG_LOG("Transfer learning rollout will fall back to student actions because teacher discrete actions cannot be applied directly to the new action space.");

	try {
		bool saveQueued = false;
		RG_LOG("Press 'Q' to save and quit!");

		while (true) {
			PollQuitKey(saveQueued);
			Report report = {};

			// Collect obs
			std::vector<float> allNewObs = {};
			std::vector<float> allOldObs = {};
			std::vector<float> allTeacherActionTable = {};
			std::vector<float> allTeacherActionTargets = {};
			std::vector<uint8_t> allNewActionMasks = {};
			std::vector<uint8_t> allOldActionMasks = {};
			std::vector<int> allActionMaps = {};
			int64_t studentRolloutActionsUsed = 0;
			int64_t totalDiscreteToContinuousRolloutActions = 0;

			// Pre-reserve vectors to estimated sizes to avoid reallocation overhead
			{
				int64_t estSamples = tlConfig.batchSize;
				allNewObs.reserve(estSamples * obsSize);
				allOldObs.reserve(estSamples * oldObsSize);
				if (!oldIsContinuous)
					allOldActionMasks.reserve(estSamples * oldNumActions);
				if (distillDiscreteToContinuous) {
					allTeacherActionTable.reserve(estSamples * oldNumActions * config.ppo.continuousActionSize);
					allTeacherActionTargets.reserve(estSamples * config.ppo.continuousActionSize);
				}
			}
			int stepsCollected;
			{
				RG_NO_GRAD;
				for (stepsCollected = 0; stepsCollected < tlConfig.batchSize; stepsCollected += envSet->state.numPlayers) {
					float studentRolloutProb = fnGetStudentRolloutProb();
					struct TeacherActionCtx {
						ActionParser* parser;
						const Player* player;
						const GameState* state;
					};
					std::vector<float> stepOldObsForTeacher = {};
					std::vector<uint8_t> stepOldMasksForTeacher = {};
					std::vector<TeacherActionCtx> teacherActionCtx = {};
					std::vector<float> stepRolloutActions = {};
					
					auto terminals = envSet->state.terminals; // Backup
					envSet->Reset();
					for (int i = 0; i < envSet->arenas.size(); i++) // Manually reset old obs builders
						if (terminals[i])
							oldObsBuilders[i]->Reset(envSet->state.gameStates[i]);

					if (!config.renderMode && obsStat) {
						// Cap samples (was RS_MAX — blew past maxObsSamples on large EnvSets).
						int numSamples = RS_MIN(envSet->state.numPlayers, config.maxObsSamples);
						for (int i = 0; i < numSamples; i++) {
							int idx = Math::RandInt(0, envSet->state.numPlayers);
							obsStat->IncrementRow(&envSet->state.obs.At(idx, 0));
						}

						std::vector<double> mean = obsStat->GetMean();
						std::vector<double> std = obsStat->GetSTD();
						for (double& f : mean)
							f = RS_CLAMP(f, -config.maxObsMeanRange, config.maxObsMeanRange);
						for (double& f : std)
							f = RS_MAX(f, config.minObsSTD);
						for (int i = 0; i < envSet->state.numPlayers; i++) {
							for (int j = 0; j < obsSize; j++) {
								float& obsVal = envSet->state.obs.At(i, j);
								obsVal = (obsVal - mean[j]) / std[j];
							}
						}
					}

					torch::Tensor tActions, tLogProbs;
					torch::Tensor tStates = DIMLIST2_TO_TENSOR<float>(envSet->state.obs);
					torch::Tensor tActionMasks = DIMLIST2_TO_TENSOR<uint8_t>(envSet->state.actionMasks);

					envSet->StepFirstHalf(true);

					allNewObs += envSet->state.obs.data;
					allNewActionMasks += envSet->state.actionMasks.data;

					// Run all old obs and old action parser on each player
					// TODO: Could be multithreaded
					for (int arenaIdx = 0; arenaIdx < envSet->arenas.size(); arenaIdx++) {
						auto& gs = envSet->state.gameStates[arenaIdx];
						for (auto& player : gs.players) {
							auto oldObs = oldObsBuilders[arenaIdx]->BuildObs(player, gs);
							allOldObs += oldObs;
							if (oldIsContinuous) {
								if (canTeacherForceContinuousRollout)
									stepOldObsForTeacher += oldObs;
							} else {
								auto oldMask = oldActionParsers[arenaIdx]->GetActionMask(player, gs);
								allOldActionMasks += oldMask;

								if (distillDiscreteToContinuous || canTeacherForceDiscreteRollout) {
									stepOldObsForTeacher += oldObs;
									stepOldMasksForTeacher += oldMask;
								}
								if (distillDiscreteToContinuous) {
									teacherActionCtx.push_back({ oldActionParsers[arenaIdx], &player, &gs });
								}

								if (!isContinuous && tlConfig.mapActsFn) {
									auto curMap = tlConfig.mapActsFn(player, gs);
									if (curMap.size() != numActions)
										RG_ERR_CLOSE("StartTransferLearn: Your action map must have the same size as the new action parser's actions");
									allActionMaps += curMap;
								}
							}
						}
					}

					if (distillDiscreteToContinuous) {
						std::vector<float> studentRolloutActions = {};
						if (studentRolloutProb > 0) {
							torch::Tensor tStudentActions;
							PPOLearner::SampleContinuousActions(
								ppo->models, tStates.to(ppo->device, true),
								true, config.ppo.useHalfPrecision,
								config.ppo.varMin, config.ppo.varMax,
								&tStudentActions, NULL
							);
							studentRolloutActions = TENSOR_TO_VEC<float>(tStudentActions.cpu().flatten());
						}

						torch::Tensor tTeacherObs = torch::tensor(stepOldObsForTeacher).reshape({ -1, oldObsSize });
						torch::Tensor tTeacherMasks = torch::tensor(stepOldMasksForTeacher).reshape({ -1, oldNumActions });
						torch::Tensor tTeacherActionIndices;
						PPOLearner::InferActionsFromModels(
							oldModels, tTeacherObs.to(ppo->device, true), tTeacherMasks.to(ppo->device, true),
							true, config.ppo.policyTemperature, config.ppo.useHalfPrecision,
							&tTeacherActionIndices, NULL
						);
						auto teacherActionIndices = TENSOR_TO_VEC<int>(tTeacherActionIndices.cpu());

						RG_ASSERT(teacherActionIndices.size() == teacherActionCtx.size());
						if (!studentRolloutActions.empty())
							RG_ASSERT(studentRolloutActions.size() == teacherActionCtx.size() * config.ppo.continuousActionSize);
						stepRolloutActions.reserve(teacherActionCtx.size() * config.ppo.continuousActionSize);
						allTeacherActionTable.reserve(allTeacherActionTable.size() + teacherActionCtx.size() * oldNumActions * config.ppo.continuousActionSize);

						for (int sampleIdx = 0; sampleIdx < teacherActionIndices.size(); sampleIdx++) {
							auto& ctx = teacherActionCtx[sampleIdx];
							RG_ASSERT(ctx.parser->GetActionAmount() == oldNumActions);
							AppendContinuousTeacherActionTable(*ctx.parser, *ctx.player, *ctx.state, allTeacherActionTable);
							Action teacherStepAction = ctx.parser->ParseAction(teacherActionIndices[sampleIdx], *ctx.player, *ctx.state);
							AppendContinuousTeacherTarget(teacherStepAction, allTeacherActionTargets);

							bool useStudentRolloutAction =
								!studentRolloutActions.empty() &&
								(RocketSim::Math::RandFloat() < studentRolloutProb);
							totalDiscreteToContinuousRolloutActions++;
							if (useStudentRolloutAction) {
								studentRolloutActionsUsed++;
								int studentActionOffset = sampleIdx * config.ppo.continuousActionSize;
								for (int dim = 0; dim < config.ppo.continuousActionSize; dim++)
									stepRolloutActions.push_back(studentRolloutActions[studentActionOffset + dim]);
							} else {
								AppendContinuousTeacherTarget(teacherStepAction, stepRolloutActions);
							}
						}
					}

					envSet->Sync();
					if (distillDiscreteToContinuous) {
						envSet->StepSecondHalfContinuous(stepRolloutActions, config.ppo.continuousActionSize, false);
					} else if (canTeacherForceContinuousRollout) {
						torch::Tensor tTeacherObs = torch::tensor(stepOldObsForTeacher).reshape({ -1, oldObsSize });
						torch::Tensor tTeacherActions;
						PPOLearner::SampleContinuousActions(
							oldModels, tTeacherObs.to(ppo->device, true),
							true, config.ppo.useHalfPrecision,
							config.ppo.varMin, config.ppo.varMax,
							&tTeacherActions, NULL
						);
						auto teacherActionsCont = TENSOR_TO_VEC<float>(tTeacherActions.cpu().flatten());
						envSet->StepSecondHalfContinuous(teacherActionsCont, config.ppo.continuousActionSize, false);
					} else if (canTeacherForceDiscreteRollout) {
						torch::Tensor tTeacherObs = torch::tensor(stepOldObsForTeacher).reshape({ -1, oldObsSize });
						torch::Tensor tTeacherMasks = torch::tensor(stepOldMasksForTeacher).reshape({ -1, oldNumActions });
						torch::Tensor tTeacherActionIndices;
						PPOLearner::InferActionsFromModels(
							oldModels, tTeacherObs.to(ppo->device, true), tTeacherMasks.to(ppo->device, true),
							true, config.ppo.policyTemperature, config.ppo.useHalfPrecision,
							&tTeacherActionIndices, NULL
						);
						auto teacherActions = TENSOR_TO_VEC<int>(tTeacherActionIndices.cpu());
						envSet->StepSecondHalf(teacherActions, false);
					} else {
						ppo->InferActions(
							tStates.to(ppo->device, true), tActionMasks.to(ppo->device, true), 
							&tActions, &tLogProbs
						);

						if (isContinuous) {
							auto curActionsCont = TENSOR_TO_VEC<float>(tActions.flatten());
							envSet->StepSecondHalfContinuous(curActionsCont, config.ppo.continuousActionSize, false);
						} else {
							auto curActions = TENSOR_TO_VEC<int>(tActions);
							envSet->StepSecondHalf(curActions, false);
						}
					}

					if (stepCallback)
						stepCallback(this, envSet->state.gameStates, report);
				}
			}

			uint64_t prevTimesteps = totalTimesteps;
			totalTimesteps += stepsCollected;
			report["Total Timesteps"] = totalTimesteps;
			report["Collected Timesteps"] = stepsCollected;
			totalIterations++;
			report["Total Iterations"] = totalIterations;

			// Make tensors
			torch::Tensor tNewObs = torch::tensor(allNewObs).reshape({ -1, obsSize }).to(ppo->device);
			torch::Tensor tOldObs = torch::tensor(allOldObs).reshape({ -1, oldObsSize }).to(ppo->device);
			torch::Tensor tNewActionMasks;
			torch::Tensor tOldActionMasks;
			if (isContinuous) {
				auto maskOptions = torch::TensorOptions().dtype(torch::kUInt8).device(ppo->device);
				tNewActionMasks = torch::ones({ tNewObs.size(0), 1 }, maskOptions);
				if (oldIsContinuous)
					tOldActionMasks = torch::ones({ tOldObs.size(0), 1 }, maskOptions);
				else
					tOldActionMasks = torch::tensor(allOldActionMasks).reshape({ -1, oldNumActions }).to(ppo->device);
			} else {
				tNewActionMasks = torch::tensor(allNewActionMasks).reshape({ -1, numActions }).to(ppo->device);
				if (oldIsContinuous) {
					auto maskOptions = torch::TensorOptions().dtype(torch::kUInt8).device(ppo->device);
					tOldActionMasks = torch::ones({ tOldObs.size(0), 1 }, maskOptions);
				} else {
					tOldActionMasks = torch::tensor(allOldActionMasks).reshape({ -1, oldNumActions }).to(ppo->device);
				}
			}

			torch::Tensor tActionMaps = {};
			if (!allActionMaps.empty())
				tActionMaps = torch::tensor(allActionMaps).reshape({ -1, numActions }).to(ppo->device);

			torch::Tensor tTeacherActionTargets = {};
			if (!allTeacherActionTargets.empty())
				tTeacherActionTargets = torch::tensor(allTeacherActionTargets).reshape({ -1, config.ppo.continuousActionSize }).to(ppo->device);

			torch::Tensor tTeacherActionTable = {};
			if (!allTeacherActionTable.empty())
				tTeacherActionTable = torch::tensor(allTeacherActionTable).reshape({ -1, oldNumActions, config.ppo.continuousActionSize }).to(ppo->device);

			// Transfer learn
			ppo->TransferLearn(
				oldModels, tNewObs, tOldObs,
				tNewActionMasks, tOldActionMasks,
				tActionMaps, tTeacherActionTable, tTeacherActionTargets,
				report, tlConfig
			);
			if (totalDiscreteToContinuousRolloutActions > 0)
				report["Student Rollout Ratio"] = studentRolloutActionsUsed / (float)totalDiscreteToContinuousRolloutActions;

			if (versionMgr && (config.savePolicyVersions || config.skillTracker.enabled))
				versionMgr->OnIteration(ppo, report, totalTimesteps, prevTimesteps);

			if (saveQueued) {
				if (!config.checkpointFolder.empty())
					Save();
				RG_LOG("Quit requested: saved and exiting training loop cleanly.");
				break;
			}

			if (!config.checkpointFolder.empty()) {
				if (totalTimesteps / config.tsPerSave > prevTimesteps / config.tsPerSave) {
					// Auto-save
					Save();
				}
			}

			report.Finish();

			if (metricSender)
				metricSender->Send(report);

			report.Display(
				{
					"Transfer Learn Accuracy",
					"Transfer Learn Loss",
					"Transfer Learn Table Loss",
					"Transfer Learn MSE",
					"Transfer Learn MAE",
					"Transfer Learn NLL",
					"Transfer Learn Std Loss",
					"Transfer Learn Button BCE",
					"Teacher Confidence",
					"",
					"Policy Entropy",
					"Mean Policy Std",
					"Old Policy Entropy",
					"Aerial Target Ratio",
					"Student Rollout Ratio",
					"Policy Update Magnitude",
					"",
					"Collected Timesteps",
					"Total Timesteps",
					"Total Iterations"
				}
			);
		}

	} catch (std::exception& e) {
		RG_ERR_CLOSE("Exception thrown during transfer learn loop: " << e.what());
	}
}

void GGL::Learner::RecreateEnvSet(RLGC::EnvCreateFn newCreateFn) {
	if (config.renderMode)
		return;

	envCreateFn = newCreateFn;
	storedEnvSetConfig.envCreateFn = newCreateFn;

#ifdef GIGA_USE_CUDA_SIM
	if (cudaEnvSet) {
		delete cudaEnvSet;
		cudaEnvSet = nullptr;

		int carsPerTeam = 1;
		{
			EnvCreateResult peek = envCreateFn(0);
			carsPerTeam = (int)peek.arena->_cars.size() / 2;
			delete peek.arena;
			for (auto& w : peek.rewards) delete w.reward;
			for (auto* tc : peek.terminalConditions) delete tc;
			delete peek.obsBuilder;
			delete peek.actionParser;
			delete peek.continuousActionParser;
			delete peek.stateSetter;
		}

		auto cudaCreateFn = [this](int arenaIdx) -> CudaEnvCreateResult {
			EnvCreateResult cpu = envCreateFn(arenaIdx);
			CudaEnvCreateResult r = {};
			r.obsBuilder = cpu.obsBuilder;
			r.rewards = cpu.rewards;
			r.terminalConditions = cpu.terminalConditions;
			r.actionParser = cpu.actionParser;
			r.continuousActionParser = cpu.continuousActionParser;
			delete cpu.arena;
			delete cpu.stateSetter;
			return r;
		};

		cudaEnvSet = new CudaEnvSet(
			storedEnvSetConfig.numArenas, carsPerTeam, config.tickSkip, cudaCreateFn,
			config.cudaPreferGpuNative, config.cudaRewardProfile, config.cudaNoTouchSeconds);
		if (config.cudaDeviceExperience && cudaEnvSet->gpuNative)
			cudaEnvSet->EnableThroughputSyncMode(true);
		envRecreatedThisIteration = true;
		return;
	}
#endif

	delete envSet;
	envSet = new RLGC::EnvSet(storedEnvSetConfig);
	envRecreatedThisIteration = true;
}

void GGL::Learner::SyncRuntimePPOConfig() {
	if (!ppo)
		return;

#ifdef GIGA_USE_CUDA_SIM
	// Adam SetOptimLR while Collect||Learn owns the device deadlocks the same way as
	// cudaMemcpyToSymbol (hangs ~iter 2 on first AT apply, again ~iter 13 on later surfaces).
	if (AsyncLearnInFlight()) {
		RequestDeferredCudaSurfaceApply();
		return;
	}
#endif

	// Preserve loaded distill teacher models — a bare config copy can flip useGuidingPolicy
	// without models (or clear the flag while tensors are still loaded).
	const bool hadGuiding = ppo->HasGuidingPolicy();
	const auto guidingPath = ppo->config.guidingPolicyPath;
	const float guidingStrength = ppo->config.guidingStrength;
	const float oldPol = ppo->config.policyLR;
	const float oldCrit = ppo->config.criticLR;

	ppo->config = config.ppo;

	if (hadGuiding) {
		ppo->config.useGuidingPolicy = true;
		ppo->config.guidingPolicyPath = guidingPath;
		if (config.ppo.guidingStrength > 0.f)
			ppo->config.guidingStrength = config.ppo.guidingStrength;
		else
			ppo->config.guidingStrength = guidingStrength;
	} else if (!ppo->HasGuidingPolicy()) {
		// Flag set without a loaded teacher — avoid Learn using an empty ModelSet.
		ppo->config.useGuidingPolicy = false;
	}

	// Apply Adam LRs to live optimizers when curriculum/AutoTrainer changed them.
	if (ppo->config.policyLR != oldPol || ppo->config.criticLR != oldCrit)
		ppo->SetLearningRates(ppo->config.policyLR, ppo->config.criticLR);
}

void GGL::Learner::EnsureRuntimeSubsystems() {
	if (config.renderMode || !ppo)
		return;

	// Early AT flaps (false entropy_death / reward_crash) used to force mid-run
	// PolicyVersionManager + OpponentPool::Load (Nexto JIT) during Collect||Learn and
	// freeze the bot while AutoTrainer kept polling (~iter 9–13). Soft warmup still
	// suppresses queueing until enough timesteps; race safety is FlushDeferredRuntimeSubsystems.
	int64_t warmupTs = 5'000'000;
	if (const char* envW = std::getenv("GIGA_SUBSYSTEM_WARMUP_TS")) {
		char* end = nullptr;
		long long v = std::strtoll(envW, &end, 10);
		if (end != envW && v >= 0)
			warmupTs = (int64_t)v;
	}
	const bool earlyWarmup = totalTimesteps < warmupTs;
	if (earlyWarmup) {
		static bool s_loggedDefer = false;
		if (!s_loggedDefer
			&& (config.opponentPool.enabled || config.skillTracker.enabled
				|| config.trainAgainstOldVersions || config.savePolicyVersions)) {
			RG_LOG("Learner: deferring mid-run OpponentPool/PolicyVersion create until "
				<< warmupTs << " timesteps (early AT flap guard)");
			s_loggedDefer = true;
		}
		if (versionMgr)
			versionMgr->SyncSkillConfig(config.skillTracker, storedEnvSetConfig, /*allowArenaCreate=*/false);
		return;
	}

	const bool wantVersionMgr =
		config.savePolicyVersions || config.skillTracker.enabled || config.trainAgainstOldVersions;
	const bool wantOppPool = config.opponentPool.enabled && !opponentPool;
	const bool wantPvmCreate = wantVersionMgr && !versionMgr && !config.checkpointFolder.empty();
	const bool wantSkillArenas = versionMgr && versionMgr->NeedsSkillArenaCreate();

	// Light skill-gate sync only (no arena alloc) while Collect||Learn may own the GPU.
	if (versionMgr)
		versionMgr->SyncSkillConfig(config.skillTracker, storedEnvSetConfig, /*allowArenaCreate=*/false);

	if (wantPvmCreate || wantSkillArenas)
		deferredVersionMgrCreate.store(true, std::memory_order_release);
	if (wantOppPool)
		deferredOpponentPoolLoad.store(true, std::memory_order_release);

#ifdef GIGA_USE_CUDA_SIM
	const bool gpuBusy = AsyncLearnInFlight();
#else
	const bool gpuBusy = false;
#endif

	if (gpuBusy) {
		if (wantPvmCreate || wantSkillArenas || wantOppPool) {
			RequestDeferredCudaSurfaceApply();
			static bool s_loggedQueue = false;
			if (!s_loggedQueue) {
				RG_LOG("Learner: queued deferred runtime subsystems"
					<< " (PVM=" << (wantPvmCreate || wantSkillArenas ? 1 : 0)
					<< " OpponentPool=" << (wantOppPool ? 1 : 0)
					<< ") — flush after Learn join / GPU-idle");
				s_loggedQueue = true;
			}
		}
		return;
	}

	// GPU idle (no async Learn): safe to flush immediately (non-overlap / post-join callers).
	FlushDeferredRuntimeSubsystems();
}

void GGL::Learner::FlushDeferredRuntimeSubsystems() {
	if (config.renderMode || !ppo)
		return;

#ifdef GIGA_USE_CUDA_SIM
	if (AsyncLearnInFlight()) {
		RequestDeferredCudaSurfaceApply();
		RG_LOG("Learner: FlushDeferredRuntimeSubsystems skipped (async Learn still in flight)");
		return;
	}
#endif

	int64_t warmupTs = 5'000'000;
	if (const char* envW = std::getenv("GIGA_SUBSYSTEM_WARMUP_TS")) {
		char* end = nullptr;
		long long v = std::strtoll(envW, &end, 10);
		if (end != envW && v >= 0)
			warmupTs = (int64_t)v;
	}
	if (totalTimesteps < warmupTs)
		return;

	const bool wantVersionMgr =
		config.savePolicyVersions || config.skillTracker.enabled || config.trainAgainstOldVersions;
	const bool doPvm = deferredVersionMgrCreate.exchange(false, std::memory_order_acq_rel)
		|| (wantVersionMgr && !versionMgr && !config.checkpointFolder.empty())
		|| (versionMgr && versionMgr->NeedsSkillArenaCreate());
	const bool doOpp = deferredOpponentPoolLoad.exchange(false, std::memory_order_acq_rel)
		|| (config.opponentPool.enabled && !opponentPool);

	if (!doPvm && !doOpp)
		return;

	// Learn thread may have left CUDA work; sync before Nexto JIT / model clone.
#ifdef RG_CUDA_SUPPORT
	if (ppo->device.is_cuda()) {
		c10::cuda::device_synchronize();
	}
#endif

	if (doPvm && wantVersionMgr && !config.checkpointFolder.empty()) {
		if (!versionMgr) {
			// Disk SaveVersions only when league/old-versions need persistence.
			// Skill-rating alone keeps versions in RAM (avoids mid-train freezes).
			if (config.trainAgainstOldVersions)
				config.savePolicyVersions = true;
			RG_LOG("Learner: flushing deferred PolicyVersionManager create "
				<< "(skill=" << (config.skillTracker.enabled ? 1 : 0)
				<< " old=" << (config.trainAgainstOldVersions ? 1 : 0)
				<< " saveVers=" << (config.savePolicyVersions ? 1 : 0)
				<< " ts=" << totalTimesteps << ")");
			versionMgr = new PolicyVersionManager(
				config.checkpointFolder / "policy_versions",
				config.maxOldVersions,
				(uint64_t)config.tsPerVersion,
				config.skillTracker,
				storedEnvSetConfig,
				renderSender
			);
			auto models = ppo->GetPolicyModels();
			versionMgr->LoadVersions(models, totalTimesteps);
		} else {
			versionMgr->maxVersions = config.maxOldVersions;
			versionMgr->tsPerVersion = (uint64_t)config.tsPerVersion;
			versionMgr->SyncSkillConfig(config.skillTracker, storedEnvSetConfig, /*allowArenaCreate=*/true);
		}
	} else if (versionMgr) {
		versionMgr->maxVersions = config.maxOldVersions;
		versionMgr->tsPerVersion = (uint64_t)config.tsPerVersion;
		versionMgr->SyncSkillConfig(config.skillTracker, storedEnvSetConfig, /*allowArenaCreate=*/true);
	}

	if (doOpp && config.opponentPool.enabled && !opponentPool) {
		RG_LOG("Learner: flushing deferred OpponentPool load (chance="
			<< config.opponentPool.chance << " ts=" << totalTimesteps << ")");
		opponentPool = new OpponentPool();
		opponentPool->Load(config.opponentPool, ppo, ppo->device);
#ifdef RG_CUDA_SUPPORT
		if (ppo->device.is_cuda())
			c10::cuda::device_synchronize();
#endif
		RG_LOG("Learner: OpponentPool load complete (entries="
			<< (opponentPool ? (int)opponentPool->entries.size() : 0) << ")");
	}
}

void GGL::Learner::SetLearningRates(float policyLR, float criticLR) {
	config.ppo.policyLR = policyLR;
	config.ppo.criticLR = criticLR;
#ifdef GIGA_USE_CUDA_SIM
	if (AsyncLearnInFlight()) {
		RequestDeferredCudaSurfaceApply();
		return;
	}
#endif
	if (ppo)
		ppo->SetLearningRates(policyLR, criticLR);
}

void GGL::Learner::SetEntropyScale(float entropyScale) {
	config.ppo.entropyScale = entropyScale;
	if (ppo)
		ppo->config.entropyScale = entropyScale;
}

void GGL::Learner::SetClipRange(float clipRange) {
	clipRange = RS_CLAMP(clipRange, 0.01f, 2.f);
	config.ppo.clipRange = clipRange;
	if (ppo)
		ppo->config.clipRange = clipRange;
}

void GGL::Learner::SetMaxGradNorm(float maxGradNorm) {
	config.ppo.maxGradNorm = maxGradNorm;
	if (ppo)
		ppo->config.maxGradNorm = maxGradNorm;
}

void GGL::Learner::SetGuidingStrength(float strength) {
	strength = RS_CLAMP(strength, 0.f, 1.f);
	config.ppo.guidingStrength = strength;
	if (ppo)
		ppo->config.guidingStrength = strength;
}

void GGL::Learner::SetTargetKl(float targetKl) {
	config.ppo.targetKl = (std::max)(0.f, targetKl);
	if (ppo)
		ppo->config.targetKl = config.ppo.targetKl;
}

void GGL::Learner::SetTargetEntropy(float targetEntropy) {
	config.ppo.targetEntropy = targetEntropy;
	if (ppo)
		ppo->config.targetEntropy = targetEntropy;
}

void GGL::Learner::SetOpponentWeight(const std::string& name, float weight) {
	if (!opponentPool)
		return;
	for (auto& e : opponentPool->entries) {
		if (e.name == name) {
			e.weight = weight;
			return;
		}
	}
}

void GGL::Learner::SavePolicyTo(std::filesystem::path folder) {
	if (!ppo)
		RG_ERR_CLOSE("Learner::SavePolicyTo(): no PPO learner");
#ifdef GIGA_USE_CUDA_SIM
	if (AsyncLearnInFlight()) {
		RequestDeferredCudaSurfaceApply();
		RG_LOG("Learner::SavePolicyTo(): deferred (async Learn in flight) → " << folder);
		return;
	}
#endif
	std::filesystem::create_directories(folder);
	ppo->SaveTo(folder);
}

void GGL::Learner::LoadPolicyFrom(std::filesystem::path folder) {
	if (!ppo)
		RG_ERR_CLOSE("Learner::LoadPolicyFrom(): no PPO learner");
	if (!std::filesystem::is_directory(folder))
		RG_ERR_CLOSE("Learner::LoadPolicyFrom(): not a directory: " << folder);
#ifdef GIGA_USE_CUDA_SIM
	if (AsyncLearnInFlight()) {
		RequestDeferredCudaSurfaceApply();
		RG_LOG("Learner::LoadPolicyFrom(): deferred (async Learn in flight) ← " << folder);
		return;
	}
#endif
	ppo->LoadFrom(folder);
}

bool GGL::Learner::SetSoftDistillTeacher(std::filesystem::path teacherFolder, float strength) {
	if (!ppo)
		return false;
#ifdef GIGA_USE_CUDA_SIM
	if (AsyncLearnInFlight()) {
		RequestDeferredCudaSurfaceApply();
		return false;
	}
#endif
	int policyOut = numActions;
	if (config.ppo.policyType == PolicyType::CONTINUOUS)
		policyOut = PPOLearner::GetContinuousPolicyOutputSize(config.ppo.continuousActionSize);
	strength = RS_CLAMP(strength, 0.f, 1.f);
	bool ok = ppo->TrySetGuidingPolicy(teacherFolder, strength, obsSize, policyOut);
	config.ppo.useGuidingPolicy = ok;
	config.ppo.guidingStrength = ok ? strength : 0.f;
	config.ppo.guidingUseKL = ppo->config.guidingUseKL;
	if (ok)
		config.ppo.guidingPolicyPath = teacherFolder;
	return ok;
}

void GGL::Learner::ClearSoftDistillTeacher() {
#ifdef GIGA_USE_CUDA_SIM
	if (AsyncLearnInFlight()) {
		config.ppo.useGuidingPolicy = false;
		config.ppo.guidingStrength = 0.f;
		RequestDeferredCudaSurfaceApply();
		return;
	}
#endif
	config.ppo.useGuidingPolicy = false;
	config.ppo.guidingStrength = 0.f;
	if (ppo)
		ppo->ClearGuidingPolicy();
}

bool GGL::Learner::HasSoftDistillTeacher() const {
	return ppo && ppo->HasGuidingPolicy();
}

#ifdef GIGA_USE_CUDA_SIM
int GGL::Learner::TriggerErrorStateReplay(float restoreFraction, int lookbackSlots, float fuzzScale, int maxArenas) {
	if (!cudaEnvSet)
		return 0;
	int n = cudaEnvSet->TriggerErrorReplay(restoreFraction, lookbackSlots, fuzzScale, maxArenas);
	if (n > 0) {
		RG_LOG("Learner: error state-replay restored " << n << " arenas"
			<< " (ring captures=" << cudaEnvSet->StateRingCaptures()
			<< " restores=" << cudaEnvSet->StateRingRestores()
			<< " calls=" << cudaEnvSet->StateRingRestoreCalls() << ")");
	}
	return n;
}

void GGL::Learner::ConfigureErrorStateRing(bool enable, int depth, int captureEvery, float restoreFraction, int maxArenas) {
	if (!cudaEnvSet)
		return;
	CudaEnvSet::StateRingConfig cfg = {};
	cfg.enabled = enable;
	cfg.depth = depth;
	cfg.captureEvery = captureEvery;
	cfg.restoreFraction = (restoreFraction > 0.f && restoreFraction <= 0.05f) ? restoreFraction : 0.05f;
	cfg.maxArenasPerRestore = maxArenas > 0 ? maxArenas : 64;
	cfg.lookbackSlots = (std::max)(1, depth - 3);
	cudaEnvSet->ConfigureStateRing(cfg);
}
#endif

void GGL::Learner::Start() {

	bool render = config.renderMode;

	RG_LOG("Learner::Start():");
	RG_LOG("\tObs size: " << obsSize);
	RG_LOG("\tAction amount: " << numActions);

	if (render)
		RG_LOG("\t(Render mode enabled)");

#ifdef GIGA_USE_CUDA_SIM
	// Resolve Collect∥Learn overlap: env override → mode → conditions.
	{
		int mode = config.asyncLearnOverlapMode;
		if (const char* envO = std::getenv("GIGA_ASYNC_OVERLAP")) {
			std::string v(envO);
			for (char& c : v) c = (char)tolower((unsigned char)c);
			if (v == "auto")
				mode = 2;
			else if (v == "0" || v == "off" || v == "false")
				mode = 0;
			else if (v == "1" || v == "on" || v == "true")
				mode = 1;
			else
				mode = (std::atoi(envO) != 0) ? 1 : 0;
		}
		config.asyncLearnOverlapMode = mode;

		const bool haveDeviceXp = config.cudaDeviceExperience && cudaEnvSet
			&& cudaEnvSet->gpuNative && ppo && ppo->device.is_cuda();
		bool enable = false;
		if (mode == 1) {
			enable = haveDeviceXp;
			if (!haveDeviceXp)
				RG_LOG("Collect||Learn: requested ON but requires cudaDeviceExperience+gpuNative+CUDA — staying two-phase");
		} else if (mode == 2) {
			enable = haveDeviceXp && !render && config.numGames <= config.asyncOverlapMaxArenas;
			if (!enable && haveDeviceXp && config.numGames > config.asyncOverlapMaxArenas) {
				RG_LOG("Collect||Learn auto: arenas=" << config.numGames
					<< " > max=" << config.asyncOverlapMaxArenas << " — two-phase (safer)");
			} else if (enable) {
				RG_LOG("Collect||Learn auto: ON (deviceXP, arenas=" << config.numGames
					<< " <= " << config.asyncOverlapMaxArenas << ")");
			}
		}
		config.asyncLearnOverlap = enable;
	}
#endif

	// Profiler (optional).
	IterProfiler& profiler = IterProfiler::FromEnv();
	if (config.profileIters) {
		profiler.enabled = true;
		if (!config.profileDumpPath.empty())
			profiler.dumpPath = config.profileDumpPath;
		profiler.dumpEvery = (std::max)(1, config.profileDumpEvery);
	}

	// Hoisted so exception paths can drain Collect||Learn before PPO teardown.
	std::future<void> learnFuture;
	ModelSet rolloutModels = {};
	bool haveRollout = false;

	try {
		bool saveQueued = false;
		RG_LOG("Press 'Q' to save and quit!");

		ExperienceBuffer experience = ExperienceBuffer(config.randomSeed, torch::kCPU);
		ExperienceBuffer experiencePrev = ExperienceBuffer(config.randomSeed, torch::kCPU);
		Report learnReportPending = {};
		std::atomic<float> asyncLearnDuration{ 0.f }; // exclusive Learn wall time (not join residual)
		bool asyncOverlap =
#ifdef GIGA_USE_CUDA_SIM
			config.asyncLearnOverlap && config.cudaDeviceExperience;
#else
			false;
#endif
		bool havePrevExperience = false;
		if (asyncOverlap) {
			// Frozen Infer clone: Collect never touches train fp32 while Learn updates it.
			rolloutModels = ppo->models.CloneAll();
			// Drop critic from rollout (not needed for Infer); free to save VRAM.
			if (rolloutModels["critic"]) {
				delete rolloutModels["critic"];
				rolloutModels.map.erase("critic");
			}
			haveRollout = true;
			RG_LOG("Collect||Learn overlap: frozen rollout clone (shared_head+policy)");
		}

		int numPlayers =
#ifdef GIGA_USE_CUDA_SIM
			cudaEnvSet ? cudaEnvSet->numPlayers :
#endif
			envSet->state.numPlayers;

		bool isContinuous = (config.ppo.policyType == PolicyType::CONTINUOUS);
		int contActionDim = config.ppo.continuousActionSize;

		// Preallocated CUDA experience bank (pure80 / 1-step truncate + gpuNative).
		// Double-buffer when Collect||Learn overlap is on so Learn never reads the fill bank.
		struct DeviceXpBank {
			torch::Tensor states, nextStates, masks, actions, logProbs, rewards, terminals;
			int64_t filled = 0;
		};
		DeviceXpBank dXpBanks[2] = {};
		int dXpFillIdx = 0;
		int64_t dXpCapacity = 0;
		torch::Tensor dCriticCat; // persistent [2*N, obs] — avoid torch::cat alloc each Cons
		const bool wantDeviceXp =
#ifdef GIGA_USE_CUDA_SIM
			config.cudaDeviceExperience && cudaEnvSet && cudaEnvSet->gpuNative
			&& ppo->device.is_cuda() && !isContinuous;
#else
			false;
#endif
		const int dXpBankCount = (wantDeviceXp && asyncOverlap) ? 2 : (wantDeviceXp ? 1 : 0);
		if (wantDeviceXp) {
			dXpCapacity = config.ppo.tsPerItr + (int64_t)numPlayers; // slack for last step
			auto optsF = torch::TensorOptions().dtype(torch::kFloat32).device(ppo->device);
			auto optsM = torch::TensorOptions().dtype(torch::kUInt8).device(ppo->device);
			auto optsA = torch::TensorOptions().dtype(torch::kInt64).device(ppo->device);
			auto optsT = torch::TensorOptions().dtype(torch::kInt8).device(ppo->device);
			for (int b = 0; b < dXpBankCount; b++) {
				dXpBanks[b].states = torch::empty({ dXpCapacity, (int64_t)obsSize }, optsF);
				dXpBanks[b].nextStates = torch::empty({ dXpCapacity, (int64_t)obsSize }, optsF);
				dXpBanks[b].masks = torch::empty({ dXpCapacity, (int64_t)numActions }, optsM);
				dXpBanks[b].actions = torch::empty({ dXpCapacity }, optsA);
				dXpBanks[b].logProbs = torch::empty({ dXpCapacity }, optsF);
				dXpBanks[b].rewards = torch::empty({ dXpCapacity }, optsF);
				dXpBanks[b].terminals = torch::empty({ dXpCapacity }, optsT);
			}
			dCriticCat = torch::empty({ 2 * dXpCapacity, (int64_t)obsSize }, optsF);
			RG_LOG("Device experience bank: capacity=" << dXpCapacity
				<< " obs=" << obsSize << " actions=" << numActions
				<< " buffers=" << dXpBankCount
				<< (asyncOverlap ? " asyncOverlap=1" : "")
				<< " throughputSync=1");
		}

		struct Trajectory {
			FList states, nextStates, rewards, logProbs;
			std::vector<uint8_t> actionMasks;
			std::vector<int8_t> terminals;
			std::vector<int32_t> actions;        // Discrete actions
			FList continuousActions;              // Continuous actions (flat: N floats per step)
			int _contActionDim = 0;               // Continuous action dimension (for Length())

			void Clear() {
				int contActionDim = _contActionDim;
				*this = Trajectory();
				_contActionDim = contActionDim;
			}

			void Append(const Trajectory& other) {
				states += other.states;
				nextStates += other.nextStates;
				rewards += other.rewards;
				logProbs += other.logProbs;
				actionMasks += other.actionMasks;
				terminals += other.terminals;
				actions += other.actions;
				continuousActions += other.continuousActions;
				if (other._contActionDim > 0) _contActionDim = other._contActionDim;
			}

			size_t Length() const {
				if (_contActionDim > 0 && !continuousActions.empty())
					return continuousActions.size() / _contActionDim;
				return actions.size();
			}
		};

		auto trajectories = std::vector<Trajectory>(numPlayers, Trajectory{});
		for (auto& traj : trajectories)
			traj._contActionDim = contActionDim;
		int maxEpisodeLength = (int)(config.ppo.maxEpisodeDuration * (120.f / config.tickSkip));

		while (true) {
			PollQuitKey(saveQueued);

			if (config.onIterationStart)
				config.onIterationStart(this);

			// After AutoTrainer/curriculum overrides: queue skill-eval / league / opponents
			// that were intentionally OFF at from-scratch boot. Heavy Load only in
			// FlushDeferredRuntimeSubsystems (after Learn join / GPU-idle).
			EnsureRuntimeSubsystems();

			if (envRecreatedThisIteration) {
#ifdef GIGA_USE_CUDA_SIM
				numPlayers = cudaEnvSet ? cudaEnvSet->numPlayers : envSet->state.numPlayers;
#else
				numPlayers = envSet->state.numPlayers;
#endif
				trajectories = std::vector<Trajectory>(numPlayers, Trajectory{});
				for (auto& traj : trajectories)
					traj._contActionDim = contActionDim;
				envRecreatedThisIteration = false;
			}

			CollectEnvView envView = CollectEnvView::FromLearner(this);

			Report report = {};

			bool isFirstIteration = (totalTimesteps == 0);
			float asyncLearnJoinTime = 0.f; // Learn(N-1) duration measured after Collect(N) overlap

#ifdef GIGA_USE_CUDA_SIM
			// Belt-and-suspenders: never state-ring D2H during Collect||Learn even if a
			// prior path cleared suppress early (Learn finished / exception race).
			if (AsyncLearnInFlight() && cudaEnvSet)
				cudaEnvSet->SetSuppressStateRingCapture(true);
#endif

			// Sparring: external OpponentPool (Nexto, etc.) or old self-play versions
			GGL::PolicyVersion* oldVersion = NULL;
			GGL::OpponentEntry* activeOpponent = NULL;
			std::vector<bool> oldVersionPlayerMask;
			std::vector<int> newPlayerIndices = {}, oldPlayerIndices = {};
			torch::Tensor tNewPlayerIndices, tOldPlayerIndices;
			Team learningTeam = Team::BLUE;

			for (int i = 0; i < numPlayers; i++)
				newPlayerIndices.push_back(i);

			if (config.opponentPool.enabled && opponentPool && !opponentPool->Empty() && !render) {
				const float oppChance = (std::clamp)(config.opponentPool.chance, 0.f, 1.f);
				if (RocketSim::Math::RandFloat() < oppChance) {
					activeOpponent = opponentPool->Pick();
					if (activeOpponent)
						report["Training Vs Fixed Opponent"] = 1.f;
				}
			}

			if (!activeOpponent && config.trainAgainstOldVersions) {
				const float oldChance = (std::clamp)(config.trainAgainstOldChance, 0.f, 1.f);
				bool shouldTrainAgainstOld =
					(RocketSim::Math::RandFloat() < oldChance)
					&& versionMgr && !versionMgr->versions.empty()
					&& !render;

				if (shouldTrainAgainstOld) {
					int oldVersionIdx = RocketSim::Math::RandInt(0, versionMgr->versions.size());
					oldVersion = &versionMgr->versions[oldVersionIdx];
					report["Training Vs Old Version"] = 1.f;
				}
			}

			if (activeOpponent || oldVersion) {
				Team opponentTeam = Team(RocketSim::Math::RandInt(0, 2));
				learningTeam = opponentTeam == Team::BLUE ? Team::ORANGE : Team::BLUE;

				newPlayerIndices.clear();
				oldVersionPlayerMask.resize(numPlayers);
				int i = 0;
				for (auto& state : *envView.gameStates) {
					for (auto& player : state.players) {
						if (player.team == opponentTeam) {
							oldVersionPlayerMask[i] = true;
							oldPlayerIndices.push_back(i);
						} else {
							oldVersionPlayerMask[i] = false;
							newPlayerIndices.push_back(i);
						}
						i++;
					}
				}

				tNewPlayerIndices = torch::tensor(newPlayerIndices);
				tOldPlayerIndices = torch::tensor(oldPlayerIndices);
			}

			bool sparring = (activeOpponent != NULL) || (oldVersion != NULL);
			int numRealPlayers = sparring ? (int)newPlayerIndices.size() : envView.numPlayers;

			int stepsCollected = 0;
			{ // Generate experience

				// Only contains complete episodes
				auto combinedTraj = Trajectory();
				combinedTraj._contActionDim = contActionDim;
				// Pure80-style: truncate every step → avoid O(n²) per-player Append into combinedTraj.
				const bool bulkTruncateBank =
					!render && !sparring && !isContinuous && maxEpisodeLength <= 1;
				if (bulkTruncateBank) {
					const size_t T = (size_t)config.ppo.tsPerItr;
					combinedTraj.states.reserve(T * (size_t)obsSize);
					combinedTraj.nextStates.reserve(T * (size_t)obsSize);
					combinedTraj.actionMasks.reserve(T * (size_t)numActions);
					combinedTraj.actions.reserve(T);
					combinedTraj.rewards.reserve(T);
					combinedTraj.logProbs.reserve(T);
					combinedTraj.terminals.reserve(T);
				} else if (!render && !sparring && !isContinuous) {
					// Multi-step normal: pre-size combined + per-player trajs (cuts realloc on 8k arenas).
					const size_t T = (size_t)config.ppo.tsPerItr;
					const size_t epCap = (size_t)std::max(1, maxEpisodeLength);
					combinedTraj.states.reserve(T * (size_t)obsSize);
					combinedTraj.nextStates.reserve(T * (size_t)obsSize);
					combinedTraj.actionMasks.reserve(T * (size_t)numActions);
					combinedTraj.actions.reserve(T);
					combinedTraj.rewards.reserve(T);
					combinedTraj.logProbs.reserve(T);
					combinedTraj.terminals.reserve(T);
					for (auto& traj : trajectories) {
						traj.states.reserve(epCap * (size_t)obsSize);
						traj.nextStates.reserve(epCap * (size_t)obsSize);
						traj.actionMasks.reserve(epCap * (size_t)numActions);
						traj.actions.reserve(epCap);
						traj.rewards.reserve(epCap);
						traj.logProbs.reserve(epCap);
						traj.terminals.reserve(epCap);
					}
				}

				Timer collectionTimer = {};
				Timer iterWallTimer = {};
				const bool useDeviceXp = wantDeviceXp && bulkTruncateBank && !render && !sparring;
				DeviceXpBank& dXp = dXpBanks[dXpFillIdx];
				if (useDeviceXp)
					dXp.filled = 0;
				profiler.Begin("Collect");
				{ // Collect timesteps
					RG_NO_GRAD;

					float inferTime = 0;
					float envStepTime = 0;

					auto trajLen = [&]() -> int64_t {
						return useDeviceXp ? dXp.filled : (int64_t)combinedTraj.Length();
					};

					for (int step = 0; trajLen() < config.ppo.tsPerItr || render; step++, stepsCollected += numRealPlayers) {
						Timer stepTimer = {};
						envView.Reset();
						envStepTime += stepTimer.Elapsed();

						if (!config.skipObsIntegrityChecks) {
							for (float f : envView.obs->data)
								if (isnan(f) || isinf(f))
									RG_ERR_CLOSE("Obs builder produced a NaN/inf value");
						}

						if (!render && obsStat) {
							// Only update running stats from learning-team players (not opponents /
							// old-version sparring partners). Cap count with RS_MIN (was RS_MAX).
							const int poolN = sparring ? (int)newPlayerIndices.size() : envView.numPlayers;
							if (poolN > 0) {
								int numSamples = RS_MIN(poolN, config.maxObsSamples);
								for (int i = 0; i < numSamples; i++) {
									int idx = sparring
										? newPlayerIndices[Math::RandInt(0, poolN)]
										: Math::RandInt(0, envView.numPlayers);
									obsStat->IncrementRow(&envView.obs->At(idx, 0));
								}
							}

							std::vector<double> mean = obsStat->GetMean();
							std::vector<double> std = obsStat->GetSTD();
							for (double& f : mean)
								f = RS_CLAMP(f, -config.maxObsMeanRange, config.maxObsMeanRange);
							for (double& f : std)
								f = RS_MAX(f, config.minObsSTD);
							// Still normalize every player so Infer sees a consistent scale.
							for (int i = 0; i < envView.numPlayers; i++) {
								for (int j = 0; j < obsSize; j++) {
									float& obsVal = envView.obs->At(i, j);
									obsVal = (obsVal - mean[j]) / std[j];
								}
							}
						}

						torch::Tensor tActions, tLogProbs;

						if (!render && !useDeviceXp) {
							if (bulkTruncateBank) {
								// One contiguous append of all players (no per-traj copies).
								combinedTraj.states.insert(
									combinedTraj.states.end(),
									envView.obs->data.begin(), envView.obs->data.end());
								combinedTraj.actionMasks.insert(
									combinedTraj.actionMasks.end(),
									envView.actionMasks->data.begin(), envView.actionMasks->data.end());
							} else {
								const float* obsBase = envView.obs->data.data();
								const uint8_t* maskBase = envView.actionMasks->data.data();
								const size_t obsRow = (size_t)obsSize;
								const size_t maskRow = envView.actionMasks->size[1];
								for (int newPlayerIdx : newPlayerIndices) {
									auto& traj = trajectories[newPlayerIdx];
									const float* orow = obsBase + (size_t)newPlayerIdx * obsRow;
									traj.states.insert(traj.states.end(), orow, orow + obsRow);
									const uint8_t* mrow = maskBase + (size_t)newPlayerIdx * maskRow;
									traj.actionMasks.insert(traj.actionMasks.end(), mrow, mrow + maskRow);
								}
							}
						}

						envView.StepFirstHalf(true);

						Timer inferTimer = {};

						// CUDA-native: infer from device obs/masks (skip host→device copy).
						// Host buffers stay for trajectories; device ptrs remain valid until next Step.
						const bool inferFromCudaDevice =
#ifdef GIGA_USE_CUDA_SIM
							envView.cuda && envView.cuda->gpuNative && envView.cuda->batch
							&& !obsStat && !sparring && ppo->device.is_cuda();
#else
							false;
#endif

						if (inferFromCudaDevice) {
#ifdef GIGA_USE_CUDA_SIM
							auto* batch = envView.cuda->batch;
							// StepDiscrete already fences when skipHostObsCopy (see CopyGpuTrainingBuffers).
							torch::Tensor tdStates = torch::from_blob(
								batch->GetBuiltAdvancedObsDevicePtr(),
								{ (int64_t)envView.numPlayers, (int64_t)obsSize },
								torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
							torch::Tensor tdActionMasks = torch::from_blob(
								batch->GetBuiltDefaultActionMasksDevicePtr(),
								{ (int64_t)envView.numPlayers, (int64_t)envView.actionMasks->size[1] },
								torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
							if (useDeviceXp) {
								const int64_t n = (int64_t)envView.numPlayers;
								if (dXp.filled + n > dXpCapacity)
									RG_ERR_CLOSE("Device experience bank overflow");
								dXp.states.narrow(0, dXp.filled, n).copy_(tdStates, /*non_blocking=*/true);
								dXp.masks.narrow(0, dXp.filled, n).copy_(tdActionMasks, /*non_blocking=*/true);
							}
							ModelSet* inferMdls = (haveRollout && asyncOverlap) ? &rolloutModels : nullptr;
							ppo->InferActions(tdStates, tdActionMasks, &tActions, &tLogProbs, inferMdls);
							if (useDeviceXp && tLogProbs.defined()) {
								const int64_t n = (int64_t)envView.numPlayers;
								dXp.actions.narrow(0, dXp.filled, n).copy_(tActions.to(torch::kInt64), true);
								dXp.logProbs.narrow(0, dXp.filled, n).copy_(tLogProbs.to(torch::kFloat32), true);
							}
							// Device action expand (default ON): skip action D2H↔H2D via LUT kernel.
							// Disable with GIGA_DEVICE_ACTIONS=0.
							static int s_devActs = -1;
							if (s_devActs < 0) {
								const char* e = std::getenv("GIGA_DEVICE_ACTIONS");
								s_devActs = (!e || std::atoi(e) != 0) ? 1 : 0;
							}
							if (!(useDeviceXp && envView.cuda->skipHostObsCopy && !isContinuous && s_devActs))
								tActions = tActions.to(torch::kCPU, /*non_blocking=*/false);
#endif
						} else if (sparring) {
							// from_blob → single H2D copy (avoid torch::tensor host clone).
							torch::Tensor tStates = torch::from_blob(
								envView.obs->data.data(),
								{ (int64_t)envView.numPlayers, (int64_t)obsSize },
								torch::kFloat32);
							torch::Tensor tActionMasks = torch::from_blob(
								envView.actionMasks->data.data(),
								{ (int64_t)envView.numPlayers, (int64_t)envView.actionMasks->size[1] },
								torch::kUInt8);
							torch::Tensor tdStates = tStates.to(ppo->device, /*non_blocking=*/false);
							torch::Tensor tdActionMasks = tActionMasks.to(ppo->device, false);

							torch::Tensor tdNewStates = tdStates.index_select(0, tNewPlayerIndices.to(ppo->device));
							torch::Tensor tdOldActionMasks = tdActionMasks.index_select(0, tOldPlayerIndices.to(ppo->device));
							torch::Tensor tdNewActionMasks = tdActionMasks.index_select(0, tNewPlayerIndices.to(ppo->device));

							torch::Tensor tNewActions;
							torch::Tensor tOldActions;

							ppo->InferActions(tdNewStates, tdNewActionMasks, &tNewActions, &tLogProbs);

							if (activeOpponent) {
								opponentPool->InferOpponentActions(
									*activeOpponent, ppo, *envView.gameStates, *envView.arenaPlayerStartIdx, oldPlayerIndices,
									tStates, tActionMasks, contActionDim, &tOldActions);
							} else {
								torch::Tensor tdOldStates = tdStates.index_select(0, tOldPlayerIndices.to(ppo->device));
								ppo->InferActions(tdOldStates, tdOldActionMasks, &tOldActions, NULL, &oldVersion->models);
							}

							if (isContinuous) {
								tActions = torch::zeros({ numPlayers, contActionDim }, tNewActions.options().device(torch::kCPU));
							} else {
								tActions = torch::zeros({ numPlayers }, tNewActions.options().device(torch::kCPU));
							}
							auto tNewCpu = tNewActions.cpu().to(tActions.dtype());
							auto tOldCpu = tOldActions.cpu().to(tActions.dtype());
							if (!isContinuous && tOldCpu.dim() == 2)
								tOldCpu = tOldCpu.select(-1, 0).contiguous(); // safety: never index_copy 2D into 1D
							tActions.index_copy_(0, tNewPlayerIndices, tNewCpu);
							tActions.index_copy_(0, tOldPlayerIndices, tOldCpu);
						} else {
							torch::Tensor tStates = torch::from_blob(
								envView.obs->data.data(),
								{ (int64_t)envView.numPlayers, (int64_t)obsSize },
								torch::kFloat32);
							torch::Tensor tActionMasks = torch::from_blob(
								envView.actionMasks->data.data(),
								{ (int64_t)envView.numPlayers, (int64_t)envView.actionMasks->size[1] },
								torch::kUInt8);
							torch::Tensor tdStates = tStates.to(ppo->device, false);
							torch::Tensor tdActionMasks = tActionMasks.to(ppo->device, false);
							ppo->InferActions(tdStates, tdActionMasks, &tActions, &tLogProbs);
							tActions = tActions.to(torch::kCPU, false);
						}
						inferTime += inferTimer.Elapsed();

						std::vector<int> curActionsDiscrete;
						FList curActionsContinuous;
						torch::Tensor tActionsDeviceInt32;
						const bool stepActionsOnDevice =
#ifdef GIGA_USE_CUDA_SIM
							useDeviceXp && inferFromCudaDevice && envView.cuda && envView.cuda->skipHostObsCopy
							&& !isContinuous && tActions.defined() && tActions.is_cuda();
#else
							false;
#endif
						if (isContinuous) {
							curActionsContinuous = TENSOR_TO_VEC<float>(tActions.flatten());
						} else if (stepActionsOnDevice) {
							tActionsDeviceInt32 = tActions.contiguous().to(torch::kInt32);
						} else {
							// Contiguous int32 host view (no raw memcpy into wrong dtype — that crashed before).
							auto tAct = tActions.contiguous().to(torch::kInt32);
							auto* ap = tAct.data_ptr<int>();
							curActionsDiscrete.assign(ap, ap + tAct.numel());
						}

						FList newLogProbs;
						if (tLogProbs.defined() && !render && !useDeviceXp) {
							auto tLp = tLogProbs.contiguous().to(torch::kCPU).to(torch::kFloat32);
							auto* lp = tLp.data_ptr<float>();
							newLogProbs.assign(lp, lp + tLp.numel());
						}

						stepTimer.Reset();
						envView.Sync(); // Make sure the first half is done
						if (isContinuous) {
							envView.StepSecondHalfContinuous(curActionsContinuous, contActionDim, false);
						} else if (stepActionsOnDevice) {
#ifdef GIGA_USE_CUDA_SIM
							envView.StepSecondHalfDiscreteDevice(tActionsDeviceInt32.data_ptr<int>());
#endif
						} else {
							envView.StepSecondHalfDiscrete(curActionsDiscrete, false);
						}
						envStepTime += stepTimer.Elapsed();

						if (stepCallback)
							stepCallback(this, *envView.gameStates, report);

						if (render) {
							const auto& renderState = (*envView.gameStates)[0];
							std::vector<std::string> playerLabels;
							playerLabels.reserve(renderState.players.size());
							for (size_t pi = 0; pi < renderState.players.size(); pi++) {
								bool isOpponent = !oldVersionPlayerMask.empty()
									&& (int)pi < (int)oldVersionPlayerMask.size()
									&& oldVersionPlayerMask[(int)pi];
								if (isOpponent && activeOpponent)
									playerLabels.push_back(activeOpponent->name);
								else if (isOpponent && oldVersion)
									playerLabels.push_back("Old " + std::to_string(oldVersion->timesteps / 1'000'000'000ULL) + "B");
								else if (isOpponent)
									playerLabels.push_back("Opponent");
								else
									playerLabels.push_back("You");
							}

							nlohmann::json trainingDiag = nlohmann::json::object();
							trainingDiag["timesteps"] = totalTimesteps;
							nlohmann::json rewardObj = nlohmann::json::object();
							float totalWeighted = 0.f;
							if (envView.lastRewards && envView.rewardSets
								&& !envView.lastRewards->empty() && !envView.rewardSets->empty()) {
								auto& prevRewards = (*envView.lastRewards)[0];
								auto& arenaRewards = (*envView.rewardSets)[0];
								for (size_t rj = 0; rj < arenaRewards.size() && rj < prevRewards.size(); rj++) {
									std::string rName = arenaRewards[rj].reward->GetName();
									float raw = prevRewards[rj];
									float weight = arenaRewards[rj].weight;
									float weighted = raw * weight;
									totalWeighted += weighted;
									rewardObj[rName] = {
										{"raw", raw},
										{"weight", weight},
										{"weighted", weighted},
									};
								}
							}
							trainingDiag["rewards"] = rewardObj;
							trainingDiag["total_reward"] = totalWeighted;
							if (envView.rewards && !envView.rewards->empty())
								trainingDiag["player0_total"] = (*envView.rewards)[0];

							renderSender->Send(renderState, playerLabels, totalTimesteps, trainingDiag);
							continue;
						}

						// Calc average rewards
						if (config.addRewardsToMetrics && (Math::RandInt(0, config.rewardSampleRandInterval) == 0)) {
							int numSamples = RS_MIN(envView.numArenas, config.maxRewardSamples);
							std::unordered_map<std::string, AvgTracker> avgRewards = {};
							for (int i = 0; i < numSamples; i++) {
								int arenaIdx = Math::RandInt(0, envView.numArenas);
								auto& prevRewards = (*envView.lastRewards)[arenaIdx];

								for (int j = 0; j < (*envView.rewardSets)[arenaIdx].size(); j++) {
									std::string rewardName = (*envView.rewardSets)[arenaIdx][j].reward->GetName();
									avgRewards[rewardName] += prevRewards[j];
								}
							}

							for (auto& pair : avgRewards)
								report.AddAvg("Rewards/" + pair.first, pair.second.Get());
						}

						// Now that we've inferred and stepped the env, we can add that stuff to the trajectories
						if (useDeviceXp) {
#ifdef GIGA_USE_CUDA_SIM
							auto* batch = envView.cuda->batch;
							const int64_t n = (int64_t)envView.numPlayers;
							torch::Tensor tdNext = torch::from_blob(
								batch->GetBuiltAdvancedObsDevicePtr(),
								{ n, (int64_t)obsSize },
								torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
							torch::Tensor tdRew = torch::from_blob(
								batch->GetBuiltRewardsDevicePtr(),
								{ n },
								torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
							dXp.nextStates.narrow(0, dXp.filled, n).copy_(tdNext, true);
							dXp.rewards.narrow(0, dXp.filled, n).copy_(tdRew, true);
							dXp.terminals.narrow(0, dXp.filled, n).fill_((int8_t)RLGC::TerminalType::TRUNCATED);
							dXp.filled += n;
#endif
						} else if (bulkTruncateBank) {
							// Bank every player as truncated in bulk (pure80 / maxEpisodeDuration 1-step).
							combinedTraj.actions.insert(
								combinedTraj.actions.end(),
								curActionsDiscrete.begin(), curActionsDiscrete.end());
							combinedTraj.rewards.insert(
								combinedTraj.rewards.end(),
								envView.rewards->begin(), envView.rewards->end());
							combinedTraj.logProbs.insert(
								combinedTraj.logProbs.end(),
								newLogProbs.begin(), newLogProbs.end());

							const size_t n = (size_t)envView.numPlayers;
							const size_t termOff = combinedTraj.terminals.size();
							combinedTraj.terminals.resize(termOff + n,
								(int8_t)RLGC::TerminalType::TRUNCATED);
							// Truncation bootstrap: nextStates = post-step obs (one per truncated step).
							combinedTraj.nextStates.insert(
								combinedTraj.nextStates.end(),
								envView.obs->data.begin(), envView.obs->data.end());
						} else {
							// ContinuousV2: dense SE bonus/penalty vs fixed externals on goal events.
							std::vector<float> beatBonusPerPlayer(numPlayers, 0.f);
							if (activeOpponent) {
								const float beatBonus =
									config.opponentPool.beatBonus * activeOpponent->beatBonusScale;
								const float concedePenalty =
									config.opponentPool.concedePenalty * activeOpponent->concedePenaltyScale;
								if (beatBonus != 0.f || concedePenalty != 0.f) {
									for (int arenaIdx = 0; arenaIdx < envView.numArenas; arenaIdx++) {
										auto& gs = (*envView.gameStates)[arenaIdx];
										if (!gs.goalScored)
											continue;
										const bool learningConceded =
											(RS_TEAM_FROM_Y(gs.ball.pos.y) == learningTeam);
										const float delta = learningConceded ? concedePenalty : beatBonus;
										if (delta == 0.f)
											continue;
										int startIdx = (*envView.arenaPlayerStartIdx)[arenaIdx];
										int playersInArena = (int)gs.players.size();
										for (int p = 0; p < playersInArena; p++) {
											int playerIdx = startIdx + p;
											if (!oldVersionPlayerMask.empty() && oldVersionPlayerMask[playerIdx])
												continue;
											beatBonusPerPlayer[playerIdx] = delta;
										}
										if (learningConceded) {
											report.AddAvg("Rewards/ConcedeExternalPenalty", delta);
											report.AddAvg(
												"Rewards/ConcedeExternalPenalty/" + activeOpponent->name, delta);
										} else {
											report.AddAvg("Rewards/BeatExternalBonus", delta);
											report.AddAvg(
												"Rewards/BeatExternalBonus/" + activeOpponent->name, delta);
										}
									}
								}
							}

							int i = 0;
							for (int newPlayerIdx : newPlayerIndices) {
								if (isContinuous) {
									// Append N floats for this player
									int offset = newPlayerIdx * contActionDim;
									for (int d = 0; d < contActionDim; d++)
										trajectories[newPlayerIdx].continuousActions.push_back(curActionsContinuous[offset + d]);
								} else {
									trajectories[newPlayerIdx].actions.push_back(curActionsDiscrete[newPlayerIdx]);
								}
								trajectories[newPlayerIdx].rewards +=
									(*envView.rewards)[newPlayerIdx] + beatBonusPerPlayer[newPlayerIdx];
								trajectories[newPlayerIdx].logProbs += newLogProbs[i];
								i++;
							}

							auto curTerminals = std::vector<uint8_t>(numPlayers, 0);
							const int playersInArena = envView.numArenas > 0
								? (envView.numPlayers / envView.numArenas) : 2;
							for (int idx = 0; idx < envView.numArenas; idx++) {
								uint8_t terminalType = (*envView.terminals)[idx];
								if (!terminalType)
									continue;

								auto playerStartIdx = (*envView.arenaPlayerStartIdx)[idx];
								for (int i = 0; i < playersInArena; i++)
									curTerminals[playerStartIdx + i] = terminalType;
							}

							// Combined traj only grows on terminal/truncate. With maxEp > 1 step,
							// trajLen() stays 0 until every open episode hits maxEpisodeLength
							// (e.g. 3s → 60 steps → 8192*2*60 = 983k). That overshoots tsPerItr
							// (131k) and makes iter-2+ old-version sparring look hung. Truncate
							// open episodes once this step would meet the iteration budget.
							const bool iterBudgetFull =
								((int64_t)stepsCollected + (int64_t)numRealPlayers)
								>= (int64_t)config.ppo.tsPerItr;

							for (int newPlayerIdx : newPlayerIndices) {
								int8_t terminalType = curTerminals[newPlayerIdx];
								auto& traj = trajectories[newPlayerIdx];

								if (!terminalType && traj.Length() >= maxEpisodeLength) {
									// Episode is too long, truncate it here
									// This won't actually reset the env, but rather will just add it to experience buffer as truncated
									terminalType = RLGC::TerminalType::TRUNCATED;
								}
								if (!terminalType && iterBudgetFull && traj.Length() > 0)
									terminalType = RLGC::TerminalType::TRUNCATED;

								traj.terminals.push_back(terminalType);
								if (terminalType) {

									if (terminalType == RLGC::TerminalType::TRUNCATED) {
										const float* nrow = envView.obs->data.data()
											+ (size_t)newPlayerIdx * (size_t)obsSize;
										traj.nextStates.insert(traj.nextStates.end(), nrow, nrow + obsSize);
									}

									combinedTraj.Append(traj);
									traj.Clear();
								}
							}
						}
					}

					report["Inference Time"] = inferTime;
					report["Env Step Time"] = envStepTime;
				}
				float collectionTime = collectionTimer.Elapsed();
				profiler.End(); // Collect

				// Join Learn(N-1) AFTER Collect(N): Infer used frozen rollout, so Collect||Learn overlapped.
				// Must finish before critic/GAE touch train weights.
				if (asyncOverlap && learnFuture.valid()) {
					profiler.Begin("AsyncJoin");
					Timer joinTimer = {};
					try {
						learnFuture.get();
					} catch (const std::exception& e) {
						asyncLearnInFlight.store(false, std::memory_order_release);
#ifdef GIGA_USE_CUDA_SIM
						if (cudaEnvSet)
							cudaEnvSet->SetSuppressStateRingCapture(false);
#endif
						RG_ERR_CLOSE("Collect||Learn: async Learn threw: " << e.what());
					} catch (...) {
						asyncLearnInFlight.store(false, std::memory_order_release);
#ifdef GIGA_USE_CUDA_SIM
						if (cudaEnvSet)
							cudaEnvSet->SetSuppressStateRingCapture(false);
#endif
						RG_ERR_CLOSE("Collect||Learn: async Learn threw unknown exception");
					}
					asyncLearnInFlight.store(false, std::memory_order_release);
#ifdef GIGA_USE_CUDA_SIM
					if (cudaEnvSet)
						cudaEnvSet->SetSuppressStateRingCapture(false);
#endif
					asyncLearnJoinTime = joinTimer.Elapsed(); // residual wait (may be ~0 if Learn finished early)
					profiler.End();
					if (haveRollout)
						SyncRolloutPolicy(rolloutModels, ppo->models);
					for (auto& kv : learnReportPending.data)
						report.data[kv.first] = kv.second;
					learnReportPending = {};
#ifdef GIGA_USE_CUDA_SIM
					// Default flush if no callback (keeps GigaLearnCPP usable without AutoTrainer).
					if (config.onAfterAsyncLearnJoin) {
						config.onAfterAsyncLearnJoin(this);
					} else if (ConsumeDeferredCudaSurfaceApply() && cudaEnvSet) {
						cudaEnvSet->ApplyRuntimeGpuRewards();
						if (cudaEnvSet->noTouchSeconds > 0.f)
							cudaEnvSet->SetNoTouchSeconds(cudaEnvSet->noTouchSeconds);
					}
					if (deferredCheckpointSave.exchange(false, std::memory_order_acq_rel)
						&& !config.checkpointFolder.empty()) {
						Save();
					}
					// Safe GPU-idle window: deferred OpponentPool / PolicyVersion / skill arenas.
					// (onAfterAsyncLearnJoin → FlushDeferredCuda also calls this; safe to repeat.)
					FlushDeferredRuntimeSubsystems();
#else
					if (config.onAfterAsyncLearnJoin)
						config.onAfterAsyncLearnJoin(this);
#endif
				}

				Timer consumptionTimer = {};
				profiler.Begin("Cons");
				{ // Process timesteps
					RG_NO_GRAD;

					const bool throughputMode = config.skipObsIntegrityChecks && ppo->config.skipPPOMetrics;
					const int64_t nSteps = useDeviceXp ? dXp.filled : (int64_t)combinedTraj.Length();

					torch::Tensor tStates, tActionMasks, tActions, tLogProbs, tRewards, tTerminals, tNextTruncStates;

					if (useDeviceXp) {
						// Already on CUDA — narrow views (offset 0 ⇒ contiguous, no copy).
						tStates = dXp.states.narrow(0, 0, nSteps);
						tNextTruncStates = dXp.nextStates.narrow(0, 0, nSteps);
						tActionMasks = dXp.masks.narrow(0, 0, nSteps);
						tActions = dXp.actions.narrow(0, 0, nSteps);
						tLogProbs = dXp.logProbs.narrow(0, 0, nSteps);
						tRewards = dXp.rewards.narrow(0, 0, nSteps);
						tTerminals = dXp.terminals.narrow(0, 0, nSteps);
					} else {
						// from_blob + clone avoids torch::tensor's dtype/device probing overhead on huge trajs.
						auto blobF32 = [](FList& v, at::IntArrayRef shape) {
							return torch::from_blob(v.data(), shape, torch::kFloat32).clone();
						};

						tStates = blobF32(combinedTraj.states, { nSteps, (int64_t)obsSize });
						if (!isContinuous) {
							tActionMasks = torch::from_blob(
								combinedTraj.actionMasks.data(),
								{ nSteps, (int64_t)numActions },
								torch::kUInt8
							).clone();
						} else {
							tActionMasks = torch::ones({ nSteps, 1 }, torch::kUInt8);
						}
						if (isContinuous) {
							tActions = blobF32(combinedTraj.continuousActions, { nSteps, (int64_t)contActionDim });
						} else {
							tActions = torch::from_blob(
								combinedTraj.actions.data(),
								{ (int64_t)combinedTraj.actions.size() },
								torch::kInt32
							).clone().to(torch::kInt64);
						}
						tLogProbs = blobF32(combinedTraj.logProbs, { nSteps });
						tRewards = blobF32(combinedTraj.rewards, { nSteps });
						tTerminals = torch::from_blob(
							combinedTraj.terminals.data(),
							{ (int64_t)combinedTraj.terminals.size() },
							torch::kInt8
						).clone();

						if (!combinedTraj.nextStates.empty()) {
							const int64_t nTrunc = (int64_t)combinedTraj.nextStates.size() / (int64_t)obsSize;
							tNextTruncStates = blobF32(combinedTraj.nextStates, { nTrunc, (int64_t)obsSize });
						}
					}

					if (!throughputMode)
						report["Average Step Reward"] = tRewards.mean().item<float>();
					report["Collected Timesteps"] = stepsCollected;
					
					torch::Tensor tValPreds;
					torch::Tensor tTruncValPreds;

					if (useDeviceXp
						&& tNextTruncStates.defined()
						&& tNextTruncStates.size(0) == nSteps
						&& ppo->config.miniBatchSize >= nSteps
						&& dCriticCat.defined()
					) {
						// Device bank: fused critic on CUDA → GPU TD0 GAE.
						// Chain reuse: with skip-reset bulk truncate, nextStates[step]==states[step+1]
						// (step-major, P=numPlayers). Critic once on (steps+1)*P instead of 2*steps*P.
						const int64_t P = (int64_t)numPlayers;
						const bool chainOk = P > 0 && (nSteps % P) == 0 && nSteps >= P;
						torch::InferenceMode inferGuard;
						if (chainOk) {
							const int64_t chainN = nSteps + P; // s0..s{T-1} + final next
							dCriticCat.narrow(0, 0, P).copy_(tStates.narrow(0, 0, P), true);
							dCriticCat.narrow(0, P, nSteps).copy_(tNextTruncStates, true);
							// half critic: Collect already keeps half mirror warm; faster gemm on 5060 Ti.
							auto allVals = ppo->InferCritic(
								dCriticCat.narrow(0, 0, chainN), /*halfPrecOverride=*/1);
							// V(s_t)=allVals[t*P:(t+1)*P], V(s'_t)=allVals[(t+1)*P:(t+2)*P]
							tValPreds = allVals.narrow(0, 0, nSteps);
							tTruncValPreds = allVals.narrow(0, P, nSteps);
						} else {
							dCriticCat.narrow(0, 0, nSteps).copy_(tStates, true);
							dCriticCat.narrow(0, nSteps, nSteps).copy_(tNextTruncStates, true);
							auto allVals = ppo->InferCritic(
								dCriticCat.narrow(0, 0, 2 * nSteps), /*halfPrecOverride=*/1);
							tValPreds = allVals.narrow(0, 0, nSteps);
							tTruncValPreds = allVals.narrow(0, nSteps, nSteps);
						}
					} else if (ppo->device.is_cpu()) {
						tValPreds = ppo->InferCritic(tStates.to(ppo->device, true, true)).cpu();
						if (tNextTruncStates.defined())
							tTruncValPreds = ppo->InferCritic(tNextTruncStates.to(ppo->device, true, true)).cpu();
					} else if (
						tNextTruncStates.defined()
						&& tNextTruncStates.size(0) == nSteps
						&& bulkTruncateBank
						&& ppo->config.miniBatchSize >= nSteps
					) {
						// Fuse states + nextStates into one critic forward (one H2D, one kernel, one D2H).
						auto catStates = torch::cat({ tStates, tNextTruncStates }, 0);
						auto allVals = ppo->InferCritic(catStates.to(ppo->device, true, true)).cpu();
						tValPreds = allVals.slice(0, 0, nSteps).contiguous();
						tTruncValPreds = allVals.slice(0, nSteps, 2 * nSteps).contiguous();
					} else {
						// Predict values using minibatching
						tValPreds = torch::zeros({ nSteps });
						for (int64_t i = 0; i < nSteps; i += ppo->config.miniBatchSize) {
							int64_t end = RS_MIN(i + ppo->config.miniBatchSize, nSteps);
							auto valPredsPart = ppo->InferCritic(
								tStates.slice(0, i, end).to(ppo->device, true, true)).cpu();
							RG_ASSERT(valPredsPart.size(0) == (end - i));
							tValPreds.slice(0, i, end).copy_(valPredsPart, true);
						}

						if (tNextTruncStates.defined()) {
							const int64_t nTrunc = tNextTruncStates.size(0);
							tTruncValPreds = torch::zeros({ nTrunc });
							for (int64_t i = 0; i < nTrunc; i += ppo->config.miniBatchSize) {
								int64_t end = RS_MIN(i + ppo->config.miniBatchSize, nTrunc);
								auto part = ppo->InferCritic(
									tNextTruncStates.slice(0, i, end).to(ppo->device, true, true)).cpu();
								tTruncValPreds.slice(0, i, end).copy_(part, true);
							}
						}
					}

					if (!throughputMode)
						report["Episode Length"] = 1.f / (tTerminals == 1).to(torch::kFloat32).mean().item<float>();

					Timer gaeTimer = {};
					torch::Tensor tAdvantages, tTargetVals, tReturns;
					float rewClipPortion;
					// returnStd=0 skips normalize/clip accounting (pure80 sets rewardClipRange=0).
					const float gaeReturnStd = returnStat
						? (float)returnStat->GetSTD()
						: (config.ppo.rewardClipRange > 0.f ? 1.f : 0.f);
					GAE::Compute(
						tRewards, tTerminals, tValPreds, tTruncValPreds,
						tAdvantages, tTargetVals, tReturns, rewClipPortion,
						config.ppo.gaeGamma, config.ppo.gaeLambda, gaeReturnStd, config.ppo.rewardClipRange
					);
					report["GAE Time"] = gaeTimer.Elapsed();
					profiler.Add("GAE", report["GAE Time"]);
					if (!throughputMode)
						report["Clipped Reward Portion"] = rewClipPortion;

					// Event-driven advantage boost: amplify learning on terminals and reward spikes
					// (goals/demos/etc.) instead of treating every physics tick equally.
					if (config.ppo.eventAdvantageBoost > 1.f && tAdvantages.defined()) {
						auto rewAbs = tRewards.abs();
						auto rewThresh = rewAbs.mean() + rewAbs.std().clamp_min(1e-6f);
						auto isTerminal = (tTerminals != 0);
						auto isSpike = rewAbs > rewThresh;
						auto eventMask = (isTerminal | isSpike).to(tAdvantages.dtype());
						auto boost = 1.f + (config.ppo.eventAdvantageBoost - 1.f) * eventMask;
						tAdvantages = tAdvantages * boost;
						report["OP/Event Advantage Boost"] = config.ppo.eventAdvantageBoost;
						if (!throughputMode)
							report["OP/Event Step Portion"] = eventMask.mean().item<float>();
					}
					// Optional running-STD advantage scale (complements PPO batch advantageNormMode).
					if (config.standardizeAdvantages && returnStat && tAdvantages.defined()) {
						float advStd = (float)returnStat->GetSTD();
						if (advStd > 1e-6f)
							tAdvantages = tAdvantages / advStd;
					}
					if (config.ppo.esNoiseScale > 0.f)
						report["OP/ES Noise Scale"] = config.ppo.esNoiseScale;

					if (returnStat) {
						report["GAE/Returns STD"] = returnStat->GetSTD();

						int numToIncrement = RS_MIN(config.maxReturnSamples, tReturns.size(0));
						if (numToIncrement > 0) {
							auto selectedReturns = tReturns.index_select(0, torch::randint(tReturns.size(0), { (int64_t)numToIncrement }));
							returnStat->Increment(TENSOR_TO_VEC<float>(selectedReturns));
						}
					}
					if (!throughputMode) {
						report["GAE/Avg Return"] = tReturns.abs().mean().item<float>();
						report["GAE/Avg Advantage"] = tAdvantages.abs().mean().item<float>();
						report["GAE/Avg Val Target"] = tTargetVals.abs().mean().item<float>();
					}

					// Device bank keeps tensors on CUDA; host path stays on CPU (Learn H2Ds).
					experience.data.actions = tActions;
					experience.data.logProbs = tLogProbs;
					experience.data.actionMasks = tActionMasks;
					experience.data.states = tStates;
					experience.data.advantages = tAdvantages;
					experience.data.targetValues = tTargetVals;
					if (useDeviceXp)
						report["Power/DeviceExperience"] = 1.f;
				}

				// Free CUDA cache (every iteration is very costly; periodic trim is enough).
				// Skip entirely in pure throughput mode — emptyCache stalls the GPU.
				// Must run while async Learn is NOT in flight (before launch below).
#ifdef RG_CUDA_SUPPORT
				if (ppo->device.is_cuda() && !ppo->config.skipPPOMetrics && (totalIterations % 8 == 0))
					c10::cuda::CUDACachingAllocator::emptyCache();
#endif

				Timer learnTimer = {};
				float consumptionTime = 0.f;
				float gaeOnlyTime = consumptionTimer.Elapsed(); // critic+GAE (Learn joined before this block)
				profiler.End(); // Cons (critic+GAE)

				// --- GPU-idle window: checkpoint / policy-version / skill-eval ---
				// Previously these ran AFTER async Learn launch → torch Save/clone/skill
				// Infer raced Collect||Learn and froze train (~iter 13 / tsPerSave cadence).
				uint64_t prevTimesteps = totalTimesteps;
				totalTimesteps += stepsCollected;
				totalIterations++;
				report["Total Timesteps"] = totalTimesteps;
				report["Total Iterations"] = totalIterations;

				if (versionMgr && (config.savePolicyVersions || config.skillTracker.enabled))
					versionMgr->OnIteration(ppo, report, totalTimesteps, prevTimesteps);

				bool quitAfterSave = false;
				if (saveQueued) {
					if (!config.checkpointFolder.empty())
						Save();
					RG_LOG("Quit requested: saved and exiting transfer-learn loop cleanly.");
					quitAfterSave = true;
				} else if (!config.checkpointFolder.empty()) {
					if (totalTimesteps / config.tsPerSave > prevTimesteps / config.tsPerSave)
						Save();
				}

				if (quitAfterSave)
					break;

				// Launch Learn async so it overlaps the next Collect (frozen rollout Infer).
				// Double-buffer device XP: flip fill bank so next Collect won't clobber Learn's tensors.
				if (asyncOverlap) {
					experiencePrev.data = experience.data;
					if (dXpBankCount > 1)
						dXpFillIdx = 1 - dXpFillIdx;
					const bool firstLearn = !havePrevExperience;
					learnReportPending = {};
					if (firstLearn) {
						// Warmup Learn (no prior overlap yet).
						profiler.Begin("Learn");
						ppo->Learn(experiencePrev, report, isFirstIteration);
						profiler.End();
						report["PPO Learn Time"] = learnTimer.Elapsed();
						consumptionTime = gaeOnlyTime + learnTimer.Elapsed();
						if (haveRollout)
							SyncRolloutPolicy(rolloutModels, ppo->models);
						havePrevExperience = true;
					} else {
						// Classic Cons = GAE + exclusive Learn duration (NOT join residual — that undercounts).
						const float learnExcl = asyncLearnDuration.load() > 0.f
							? asyncLearnDuration.load() : asyncLearnJoinTime;
						report["PPO Learn Time"] = learnExcl;
						report["Power/AsyncLearnJoinResidual"] = asyncLearnJoinTime;
						consumptionTime = gaeOnlyTime + learnExcl;
						asyncLearnInFlight.store(true, std::memory_order_release);
#ifdef GIGA_USE_CUDA_SIM
						if (cudaEnvSet)
							cudaEnvSet->SetSuppressStateRingCapture(true);
#endif
						learnFuture = std::async(std::launch::async,
							[this, &experiencePrev, &learnReportPending, &asyncLearnDuration]() {
								Timer lt;
								try {
									ppo->Learn(experiencePrev, learnReportPending, false);
								} catch (...) {
									asyncLearnInFlight.store(false, std::memory_order_release);
#ifdef GIGA_USE_CUDA_SIM
									if (cudaEnvSet)
										cudaEnvSet->SetSuppressStateRingCapture(false);
#endif
									throw;
								}
								asyncLearnDuration.store(lt.Elapsed());
							});
					}
					report["Power/AsyncLearnOverlap"] = 1.f;
				} else {
					profiler.Begin("Learn");
					ppo->Learn(experience, report, isFirstIteration);
					profiler.End();
					report["PPO Learn Time"] = learnTimer.Elapsed();
					consumptionTime = consumptionTimer.Elapsed();
				}

				// Set metrics — Wall reflects real elapsed (Collect||Learn overlap).
				float wallTime = iterWallTimer.Elapsed();
				report["Collection Time"] = collectionTime;
				report["Consumption Time"] = consumptionTime;
				report["Collection Steps/Second"] = stepsCollected / RS_MAX(collectionTime, 1e-9f);
				report["Consumption Steps/Second"] = stepsCollected / RS_MAX(consumptionTime, 1e-9f);
				// Classic Overall: steps / (t_coll + t_cons).
				report["Overall Steps/Second"] = stepsCollected / RS_MAX(collectionTime + consumptionTime, 1e-9f);
				report["Wall Steps/Second"] = stepsCollected / RS_MAX(wallTime, 1e-9f);
				if (useDeviceXp)
					report["Power/DeviceExperience"] = 1.f;

				profiler.Add("Infer", report.data.count("Inference Time") ? report.data.at("Inference Time") : 0.0);
				profiler.Add("EnvStep", report.data.count("Env Step Time") ? report.data.at("Env Step Time") : 0.0);
				profiler.FinishIteration(report, (int64_t)totalTimesteps, wallTime);

				report.Finish();

				if (config.onIterationComplete)
					config.onIterationComplete(this, report);

				if (metricSender)
					metricSender->Send(report);

				// Match GigaLearnCPP-Leak Report.Display field order/labels (console UX).
				// Extra keys (GAE/*, Player/*, Power/*, etc.) still go to MetricSender when enabled.
				report.Display(
					{
						"Average Step Reward",
						"Policy Entropy",
						"KL Div Loss",
						"First Accuracy",
						"",
						"Policy Update Magnitude",
						"Critic Update Magnitude",
						"Shared Head Update Magnitude",
						"",
						"Collection Steps/Second",
						"Consumption Steps/Second",
						"Overall Steps/Second",
						"",
						"Collection Time",
						"-Inference Time",
						"-Env Step Time",
						"Consumption Time",
						"-GAE Time",
						"-PPO Learn Time",
						"",
						"Collected Timesteps",
						"Total Timesteps",
						"Total Iterations"
					}
				);
			}
		}

		if (learnFuture.valid()) {
			try {
				learnFuture.get();
			} catch (const std::exception& e) {
				RG_LOG("Collect||Learn: draining async Learn after loop: " << e.what());
			}
		}
		if (haveRollout)
			rolloutModels.Free();
		
	} catch (std::exception& e) {
		// Drain async Learn before tearing down PPO (avoids use-after-free hang).
		try {
			if (learnFuture.valid())
				learnFuture.get();
		} catch (...) {}
		if (haveRollout) {
			rolloutModels.Free();
			haveRollout = false;
		}
		RG_ERR_CLOSE("Exception thrown during main learner loop: " << e.what());
	}
}

GGL::Learner::~Learner() {
	delete opponentPool;
	delete ppo;
	delete versionMgr;
	delete metricSender;
	delete renderSender;
	delete envSet;
#ifdef GIGA_USE_CUDA_SIM
	delete cudaEnvSet;
#endif
	delete returnStat;
	delete obsStat;

	// Only finalize the Python interpreter if we actually initialized it (see ctor).
	if (_pythonInitialized) {
		pybind11::finalize_interpreter();
		_pythonInitialized = false;
	}
}
