cmake_minimum_required(VERSION 3.24)  # 3.24+ for CMAKE_CUDA_ARCHITECTURES=native

project(ikeda_cuda
    VERSION 1.0.0
    DESCRIPTION "GPU-accelerated parameter sweeps of the Ikeda optical-cavity map"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
set(IKEDA_ENABLE_CUDA AUTO CACHE STRING "Build the CUDA backend: ON, OFF, or AUTO (if nvcc is found)")
set_property(CACHE IKEDA_ENABLE_CUDA PROPERTY STRINGS AUTO ON OFF)
option(IKEDA_BUILD_TESTS "Build the unit tests" ON)
option(IKEDA_BUILD_BENCHMARKS "Build the benchmark driver" ON)
option(IKEDA_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(IKEDA_NATIVE_CPU "Compile the CPU code with -march=native" ON)

# ---------------------------------------------------------------------------
# CUDA detection
# ---------------------------------------------------------------------------
set(IKEDA_HAVE_CUDA OFF)
if(NOT IKEDA_ENABLE_CUDA STREQUAL "OFF")
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            # Build for the GPU in this machine. Pass e.g.
            # -DCMAKE_CUDA_ARCHITECTURES="75;86;89" to cross-compile instead.
            set(CMAKE_CUDA_ARCHITECTURES native)
        endif()
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 20)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        find_package(CUDAToolkit REQUIRED)
        set(IKEDA_HAVE_CUDA ON)
    elseif(IKEDA_ENABLE_CUDA STREQUAL "ON")
        message(FATAL_ERROR "IKEDA_ENABLE_CUDA=ON but no CUDA compiler was found")
    endif()
endif()
message(STATUS "Ikeda: CUDA backend ${IKEDA_HAVE_CUDA}")

find_package(OpenMP COMPONENTS CXX)
if(NOT OpenMP_CXX_FOUND)
    message(WARNING "OpenMP not found: the CPU reference will run single-threaded. "
                    "Use GCC, or install libomp for Clang.")
endif()

# ---------------------------------------------------------------------------
# Shared compile settings
# ---------------------------------------------------------------------------
add_library(ikeda_warnings INTERFACE)
target_compile_options(ikeda_warnings INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra --Werror=cross-execution-space-call>)
if(IKEDA_WARNINGS_AS_ERRORS)
    target_compile_options(ikeda_warnings INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:-Werror>
        $<$<COMPILE_LANGUAGE:CUDA>:-Werror=all-warnings>)
endif()

# ---------------------------------------------------------------------------
# Core library: map, CPU reference sweep, analysis, I/O (no CUDA dependency)
# ---------------------------------------------------------------------------
add_library(ikeda_core
    src/sweep_cpu.cpp
    src/analysis.cpp
    src/io.cpp
    src/options.cpp)
target_include_directories(ikeda_core PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(ikeda_core PRIVATE ikeda_warnings)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ikeda_core PUBLIC OpenMP::OpenMP_CXX)
endif()
if(IKEDA_NATIVE_CPU)
    target_compile_options(ikeda_core PRIVATE -march=native)
endif()

# ---------------------------------------------------------------------------
# CUDA backend
# ---------------------------------------------------------------------------
if(IKEDA_HAVE_CUDA)
    add_library(ikeda_gpu src/sweep_gpu.cu)
    target_link_libraries(ikeda_gpu PUBLIC ikeda_core CUDA::cudart PRIVATE ikeda_warnings)
    target_compile_definitions(ikeda_gpu PUBLIC IKEDA_HAVE_CUDA)
    # No --use_fast_math. It swaps sincos/log/div for low-accuracy intrinsics,
    # which visibly shifts bifurcation points and Lyapunov exponents. Speed is
    # gained by choosing float precision explicitly instead.
    target_compile_options(ikeda_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
    set(IKEDA_BACKEND ikeda_gpu)
else()
    set(IKEDA_BACKEND ikeda_core)
endif()

# ---------------------------------------------------------------------------
# Executables
# ---------------------------------------------------------------------------
add_executable(ikeda_sweep src/apps/ikeda_sweep.cpp)
target_link_libraries(ikeda_sweep PRIVATE ${IKEDA_BACKEND} ikeda_warnings)

add_executable(ikeda_thresholds src/apps/ikeda_thresholds.cpp)
target_link_libraries(ikeda_thresholds PRIVATE ikeda_core ikeda_warnings)

if(IKEDA_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(IKEDA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
