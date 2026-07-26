cmake_minimum_required (VERSION 3.18)

# GigaLearnRL - unified C++ reinforcement learning framework for Rocket League bots.
# Builds a single executable (GigaLearnBot) that can train, render, transfer-learn, and run as an
# RLBot bot, plus the GigaLearnCPP shared library (PPO + RLGymCPP + RocketSim + libtorch).
project("GigaLearnRL" CXX)

# libtorch is required by GigaLearnCPP. Provide it in one of these ways:
#   - place an unpacked libtorch in GigaLearnCPP/libtorch/  (auto-detected), or
#   - configure with -DCMAKE_PREFIX_PATH=<path-to-libtorch>/share/cmake/Torch
# See README.md for details (it is intentionally NOT vendored to keep the repo small).

# Optional GPU-resident simulation backend (RocketSimCuda). Declared up front so the option is
# visible to all subdirectories (RLGymCPP enables its GPU bridge when this is ON).
# Backend: GIGA_GPU_BACKEND=cuda (NVIDIA) | hip (AMD HIP SDK / ROCm). See docs/AMD.md + docs/CUDA_SIM.md.
# NVIDIA nvcc cannot handle paths with apostrophes/spaces — use a clean mirror (e.g. C:\GigaLearnRL).
option(GIGA_USE_CUDA_SIM "Build and link the RocketSimCuda GPU simulation backend (CUDA or HIP)" OFF)
set(GIGA_GPU_BACKEND "cuda" CACHE STRING "GPU backend when GIGA_USE_CUDA_SIM=ON: cuda | hip")
set_property(CACHE GIGA_GPU_BACKEND PROPERTY STRINGS cuda hip)
if(GIGA_USE_CUDA_SIM)
    message(STATUS "GIGA_USE_CUDA_SIM=ON: building RocketSimCuda GPU backend (${GIGA_GPU_BACKEND})")
    add_subdirectory(RocketSimCuda)
endif()

file(GLOB_RECURSE FILES_SRC "src/*.cpp" "src/*.h" "src/*.hpp")
# Canonical entry is ExampleMain_mod.cpp (modular). Exclude locked/legacy monoliths + scraps.
list(FILTER FILES_SRC EXCLUDE REGEX "ExampleMain\\.cpp$")
list(FILTER FILES_SRC EXCLUDE REGEX "ExampleMain_drivefix\\.cpp$")
list(FILTER FILES_SRC EXCLUDE REGEX "ExampleMain_canonical\\.cpp$")
list(FILTER FILES_SRC EXCLUDE REGEX "_main_body_snip\\.cpp$")
add_executable(GigaLearnBot ${FILES_SRC})

set_target_properties(GigaLearnBot PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(GigaLearnBot PROPERTIES CXX_STANDARD 20)

# Build GigaLearnCPP (and its DLL) into the same directory as the executable so it loads at runtime.
set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}")

# Core learning library (PPO, models, attention, RLGymCPP, RocketSim, embedded Python for metrics).
add_subdirectory(GigaLearnCPP)
target_link_libraries(GigaLearnBot GigaLearnCPP)

# RLBot client (used by the --rlbot deployment mode).
add_subdirectory(RLBotCPP)
target_link_libraries(GigaLearnBot RLBotCPP)

# Link the GPU backend into the core DLL (the GameState bridge lives in RLGymCPP, inside the DLL)
# and into the executable, and enable the bridge code path everywhere.
if(GIGA_USE_CUDA_SIM)
    target_link_libraries(GigaLearnCPP PUBLIC RocketSimCuda)
    target_link_libraries(GigaLearnBot RocketSimCuda)
    target_compile_definitions(GigaLearnBot PRIVATE GIGA_USE_CUDA_SIM)
endif()

# Optional ImGui AutoTrainer monitor (off by default — not needed for discrete train).
option(GIGA_AUTOTRAINER_DASHBOARD "Build ImGui AutoTrainer dashboard executable" OFF)
if(GIGA_AUTOTRAINER_DASHBOARD AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/autotrainer/dashboard/CMakeLists.txt")
    add_subdirectory(autotrainer/dashboard)
endif()
