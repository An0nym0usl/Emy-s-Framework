cmake_minimum_required(VERSION 3.18)
project(giga_sdk_native)

# Optional native pybind11 binding for the GigaLearnRL checkpoint SDK.
# This wraps GGL::CheckpointManager (a torch-free part of the core) so giga_sdk can call into C++.
# The pure-Python giga_sdk works without this; build only if you want native acceleration.

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point this at the repo root so we can find the core sources and the bundled pybind11.
set(GIGA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." CACHE PATH "GigaLearnRL repository root")

add_subdirectory("${GIGA_ROOT}/GigaLearnCPP/pybind11" pybind11_build)

pybind11_add_module(giga_sdk_native
    binding.cpp
    "${GIGA_ROOT}/GigaLearnCPP/src/public/GigaLearnCPP/SDK/CheckpointManager.cpp"
)

# CheckpointManager is intentionally torch-free and depends only on its own header + nlohmann/json,
# so the binding needs just these include paths. WITHIN_GGL makes the class symbols exported
# (rather than dllimport) when compiled directly into this module on MSVC.
target_compile_definitions(giga_sdk_native PRIVATE WITHIN_GGL)
target_include_directories(giga_sdk_native PRIVATE
    "${GIGA_ROOT}/GigaLearnCPP/src/public"
    "${GIGA_ROOT}/GigaLearnCPP/libsrc/json"
)
