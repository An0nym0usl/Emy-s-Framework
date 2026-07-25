cmake_minimum_required(VERSION 3.21)

# ---------------------------------------------------------------------------
# GPU backend: cuda (NVIDIA, default) | hip (AMD ROCm / HIP SDK) | none
# GIGA_USE_CUDA_SIM remains the public ON switch (legacy name = GPU sim).
# ---------------------------------------------------------------------------
set(GIGA_GPU_BACKEND "cuda" CACHE STRING "GPU backend for RocketSimCuda: cuda, hip, or none")
set_property(CACHE GIGA_GPU_BACKEND PROPERTY STRINGS cuda hip none)

option(RSCUDA_BUILD_CPU_PARITY
    "Build the GPU-less CPU parity harness only (no CUDA/HIP toolkit required)" OFF)

if(RSCUDA_BUILD_CPU_PARITY)
    project(RocketSimCuda LANGUAGES CXX)

    add_executable(RocketSimCudaCpuParity tests/cpu_parity_harness.cpp)
    target_include_directories(RocketSimCudaCpuParity PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
    set_target_properties(RocketSimCudaCpuParity PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(RocketSimCudaCpuParity PRIVATE -ffp-contract=off)
    endif()

    enable_testing()
    add_test(NAME RocketSimCudaCpuParity COMMAND RocketSimCudaCpuParity)
    return()
endif()

# ---- HIP (AMD) backend ------------------------------------------------------
if(GIGA_GPU_BACKEND STREQUAL "hip")
    project(RocketSimCuda LANGUAGES CXX HIP)

    find_package(hip REQUIRED CONFIG)

    file(GLOB_RECURSE GPU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")
    file(GLOB_RECURSE GPU_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cuh"
                                   "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
                                   "${CMAKE_CURRENT_SOURCE_DIR}/include/*.cuh")

    # Compile the same .cu sources as HIP (hipcc / amdclang).
    set_source_files_properties(${GPU_SOURCES} PROPERTIES LANGUAGE HIP)

    add_library(RocketSimCuda STATIC ${GPU_SOURCES} ${GPU_HEADERS})

    target_include_directories(RocketSimCuda PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_compile_definitions(RocketSimCuda PUBLIC
        GIGA_GPU_HIP=1
        GIGA_USE_CUDA_SIM=1
        GIGA_GPU_BACKEND_HIP=1
    )
    target_link_libraries(RocketSimCuda PUBLIC hip::host hip::device)

    # RDNA2 / RDNA3 / RDNA4 common consumer targets (override with -DAMDGPU_TARGETS=...)
    if(NOT DEFINED AMDGPU_TARGETS AND NOT DEFINED CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES "gfx1030;gfx1100;gfx1101;gfx1200;gfx1201" CACHE STRING
            "AMD GPU ISA targets for RocketSimCuda HIP build")
    endif()
    if(DEFINED AMDGPU_TARGETS)
        set_property(TARGET RocketSimCuda PROPERTY HIP_ARCHITECTURES "${AMDGPU_TARGETS}")
    else()
        set_property(TARGET RocketSimCuda PROPERTY HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
    endif()

    set_target_properties(RocketSimCuda PROPERTIES
        CXX_STANDARD 17
        HIP_STANDARD 17
        HIP_STANDARD_REQUIRED ON
    )

    # HIP: disable contraction where supported for RocketSim parity with CPU.
    target_compile_options(RocketSimCuda PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-ffp-contract=off>
    )

    message(STATUS "RocketSimCuda: HIP/ROCm backend (GIGA_GPU_HIP)")
    return()
endif()

# ---- CUDA (NVIDIA) backend --------------------------------------------------
if(GIGA_GPU_BACKEND STREQUAL "none")
    message(FATAL_ERROR "GIGA_GPU_BACKEND=none but RocketSimCuda subdirectory was added. Set GIGA_USE_CUDA_SIM=OFF.")
endif()

project(RocketSimCuda LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

file(GLOB_RECURSE CUDA_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")
file(GLOB_RECURSE CUDA_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cuh"
                               "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
                               "${CMAKE_CURRENT_SOURCE_DIR}/include/*.cuh")

add_library(RocketSimCuda STATIC ${CUDA_SOURCES} ${CUDA_HEADERS})

target_include_directories(RocketSimCuda PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions(RocketSimCuda PUBLIC
    GIGA_GPU_CUDA=1
    GIGA_USE_CUDA_SIM=1
)

set_target_properties(RocketSimCuda PROPERTIES
    CUDA_STANDARD 17
    CXX_STANDARD 17
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES "75;86;89"
)

# Physics parity: MSVC (/fp:precise) does not contract mul+add into FMA, but
# nvcc does by default. Disable contraction so GPU arithmetic matches the
# reference RocketSim build operation-for-operation.
target_compile_options(RocketSimCuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
)

target_link_libraries(RocketSimCuda PUBLIC CUDA::cudart)

message(STATUS "RocketSimCuda: CUDA backend (NVIDIA)")

# ---- Test executable ----
add_executable(RocketSimCudaTest tests/test_physics.cpp)
target_link_libraries(RocketSimCudaTest PRIVATE RocketSimCuda)
set_target_properties(RocketSimCudaTest PROPERTIES
    CXX_STANDARD 17
    CUDA_ARCHITECTURES "75;86;89"
)

add_executable(RocketSimCudaGaeParity tests/test_gae_parity.cpp)
target_link_libraries(RocketSimCudaGaeParity PRIVATE RocketSimCuda)
set_target_properties(RocketSimCudaGaeParity PROPERTIES
    CXX_STANDARD 17
    CUDA_ARCHITECTURES "75;86;89"
)

add_executable(RocketSimCudaBenchmark1024 tests/benchmark_1024.cpp)
target_link_libraries(RocketSimCudaBenchmark1024 PRIVATE RocketSimCuda)
target_compile_definitions(RocketSimCudaBenchmark1024 PRIVATE
    RSCUDA_WORKSPACE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
)
set_target_properties(RocketSimCudaBenchmark1024 PROPERTIES
    CXX_STANDARD 17
    CUDA_ARCHITECTURES "75;86;89"
)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    add_subdirectory("../GigaLearnCPP/RLGymCPP" RLGymCPPBuild)

    add_executable(RocketSimCpuCudaCompare tests/compare_cpu_cuda.cpp)
    target_include_directories(RocketSimCpuCudaCompare PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../GigaLearnCPP/RLGymCPP/RocketSim/src
    )
    target_compile_definitions(RocketSimCpuCudaCompare PRIVATE
        RSCUDA_WORKSPACE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
    )
    target_link_libraries(RocketSimCpuCudaCompare PRIVATE RocketSimCuda RocketSim)
    set_target_properties(RocketSimCpuCudaCompare PROPERTIES
        CXX_STANDARD 20
        CUDA_ARCHITECTURES "75;86;89"
    )

    add_executable(RocketSimEnvPipelineBenchmark tests/benchmark_env_pipeline.cpp)
    target_include_directories(RocketSimEnvPipelineBenchmark PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../GigaLearnCPP/RLGymCPP/src
        ${CMAKE_CURRENT_SOURCE_DIR}/../GigaLearnCPP/RLGymCPP/RocketSim/src
    )
    target_compile_definitions(RocketSimEnvPipelineBenchmark PRIVATE
        RSCUDA_WORKSPACE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
    )
    target_link_libraries(RocketSimEnvPipelineBenchmark PRIVATE RocketSimCuda RLGymCPP RocketSim)
    set_target_properties(RocketSimEnvPipelineBenchmark PROPERTIES
        CXX_STANDARD 20
        CUDA_ARCHITECTURES "75;86;89"
    )

    add_executable(RocketSimCudaVantaParity tests/test_vanta_parity.cpp)
    target_include_directories(RocketSimCudaVantaParity PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../GigaLearnCPP/RLGymCPP/src
        ${CMAKE_CURRENT_SOURCE_DIR}/../GigaLearnCPP/RLGymCPP/RocketSim/src
    )
    target_compile_definitions(RocketSimCudaVantaParity PRIVATE
        RSCUDA_WORKSPACE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
    )
    target_link_libraries(RocketSimCudaVantaParity PRIVATE RocketSimCuda RLGymCPP RocketSim)
    set_target_properties(RocketSimCudaVantaParity PROPERTIES
        CXX_STANDARD 20
        CUDA_ARCHITECTURES "75;86;89"
    )

    add_executable(RocketSimCudaTraceDivergence tests/trace_divergence.cpp)
    target_include_directories(RocketSimCudaTraceDivergence PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../GigaLearnCPP/RLGymCPP/RocketSim/src
    )
    target_compile_definitions(RocketSimCudaTraceDivergence PRIVATE
        RSCUDA_WORKSPACE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
    )
    target_link_libraries(RocketSimCudaTraceDivergence PRIVATE RocketSimCuda RocketSim)
    set_target_properties(RocketSimCudaTraceDivergence PROPERTIES
        CXX_STANDARD 20
        CUDA_ARCHITECTURES "75;86;89"
    )

    add_executable(RocketSimCudaGameStateBridgeTest tests/test_gamestate_cuda.cpp)
    target_include_directories(RocketSimCudaGameStateBridgeTest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../GigaLearnCPP/RLGymCPP/src
        ${CMAKE_CURRENT_SOURCE_DIR}/../GigaLearnCPP/RLGymCPP/RocketSim/src
    )
    target_compile_definitions(RocketSimCudaGameStateBridgeTest PRIVATE
        RSCUDA_WORKSPACE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
    )
    target_link_libraries(RocketSimCudaGameStateBridgeTest PRIVATE RocketSimCuda RLGymCPP RocketSim)
    set_target_properties(RocketSimCudaGameStateBridgeTest PROPERTIES
        CXX_STANDARD 20
        CUDA_ARCHITECTURES "75;86;89"
    )
endif()
