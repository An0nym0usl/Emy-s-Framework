// Optional pybind11 binding exposing the C++ GGL::CheckpointManager to Python as a native
// acceleration of giga_sdk. The pure-Python giga_sdk.checkpoints works without this module;
// build it only if you want the SDK to call straight into the C++ core.
//
// Build (after the main framework is built):
//   cmake -S sdk/python/native -B build_sdk -DCMAKE_PREFIX_PATH=<path-to-GigaLearnRL-build>
//   cmake --build build_sdk --config Release
// then place the resulting `giga_sdk_native` module on your PYTHONPATH.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <GigaLearnCPP/SDK/CheckpointManager.h>

namespace py = pybind11;
using namespace GGL;

PYBIND11_MODULE(giga_sdk_native, m) {
	m.doc() = "Native bindings for the GigaLearnRL checkpoint SDK";

	py::class_<CheckpointInfo>(m, "CheckpointInfo")
		.def_property_readonly("path", [](const CheckpointInfo& c) { return c.path.string(); })
		.def_readonly("timesteps", &CheckpointInfo::timesteps)
		.def_readonly("has_policy", &CheckpointInfo::hasPolicy)
		.def_readonly("has_critic", &CheckpointInfo::hasCritic)
		.def_readonly("has_shared_head", &CheckpointInfo::hasSharedHead)
		.def_readonly("has_optimizers", &CheckpointInfo::hasOptimizers)
		.def_readonly("has_stats", &CheckpointInfo::hasStats)
		.def_readonly("total_timesteps", &CheckpointInfo::totalTimesteps)
		.def_readonly("total_iterations", &CheckpointInfo::totalIterations)
		.def_readonly("run_id", &CheckpointInfo::runId)
		.def_readonly("total_bytes", &CheckpointInfo::totalBytes)
		.def_property_readonly("is_valid_for_inference", &CheckpointInfo::IsValidForInference)
		.def_property_readonly("is_valid_for_resume", &CheckpointInfo::IsValidForResume);

	py::class_<CheckpointManager>(m, "CheckpointManager")
		.def(py::init<std::filesystem::path>(), py::arg("folder"))
		.def("read", &CheckpointManager::Read)
		.def("list", &CheckpointManager::List)
		.def("list_policy_versions", &CheckpointManager::ListPolicyVersions)
		.def("latest", &CheckpointManager::Latest)
		.def("latest_valid_for_inference", &CheckpointManager::LatestValidForInference)
		.def("get", &CheckpointManager::Get, py::arg("timesteps"))
		.def("validate", [](const CheckpointManager& m, std::filesystem::path dir) {
			std::string err;
			bool ok = m.Validate(dir, err);
			return py::make_tuple(ok, err);
		})
		.def("export", [](const CheckpointManager& m, int64_t ts, std::filesystem::path dest) {
			std::string err;
			bool ok = m.Export(ts, dest, err);
			return py::make_tuple(ok, err);
		})
		.def("prune", &CheckpointManager::Prune, py::arg("keep_n"));
}
