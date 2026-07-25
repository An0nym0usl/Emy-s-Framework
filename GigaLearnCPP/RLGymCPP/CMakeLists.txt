cmake_minimum_required (VERSION 3.8)

project("RLGymCPP")

# Add all headers and code files
file(GLOB_RECURSE FILES_SRC "src/*.cpp" "src/*.h")
add_library(RLGymCPP STATIC ${FILES_SRC})
target_include_directories(RLGymCPP PUBLIC "src/")

# Set C++ version to 20
set_target_properties(RLGymCPP PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(RLGymCPP PROPERTIES CXX_STANDARD 20)

# Include RocketSim
add_subdirectory("RocketSim")
target_link_libraries(RLGymCPP RocketSim)

# Include thread pool library (https://github.com/DeveloperPaul123/thread-pool)
target_include_directories(RLGymCPP PUBLIC "thread_pool")

# Optional GPU backend bridge (RocketSimCuda): enable the GameState::UpdateFromCudaBatch path.
# Only the public header is needed to compile (POD structs); the GPU library is linked at the
# core/executable level. PUBLIC so dependents (GigaLearnCPP, the executable) see the same macro.
if(GIGA_USE_CUDA_SIM)
    target_compile_definitions(RLGymCPP PUBLIC GIGA_USE_CUDA_SIM)
    target_include_directories(RLGymCPP PUBLIC "${CMAKE_SOURCE_DIR}/RocketSimCuda/include")
endif()