# Copyright (c) 2025 Alessandro Baretta
# All rights reserved.

cmake_minimum_required(VERSION 4.0)  # Lowered requirement for hip::host usage

# Set C++ compiler before project()
set(CMAKE_CXX_COMPILER "g++-14")
set(CMAKE_CUDA_HOST_COMPILER "g++-14")
# set(CMAKE_CXX_COMPILER "clang-19")
# set(CMAKE_CUDA_HOST_COMPILER "clang-19")

project(gpu_algos LANGUAGES CUDA CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set default parallel jobs for building
set(CMAKE_BUILD_PARALLEL_LEVEL 12)

# Set CUDA specific flags
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "89")  # Compute capability 8.9


include(CheckLanguage)
check_language(HIP)

if(CMAKE_HIP_COMPILER)
    # Try to enable HIP if available
    enable_language(HIP)
    find_package(hip REQUIRED)
    # Set HIP specific flags
    set(CMAKE_HIP_STANDARD 20)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    set(CMAKE_HIP_ARCHITECTURES "gfx1201")  # AMD Radeon RX 9070 (RDNA 3+)

    # # Add ROCm to CMAKE_PREFIX_PATH for proper package discovery
    # list(APPEND CMAKE_PREFIX_PATH
    #     "${ROCM_PATH}"
    #     "${ROCM_PATH}/lib/cmake"
    #     "${ROCM_PATH}/hip"
    #     "${ROCM_PATH}/hip/lib/cmake"
    # )

    message(STATUS "HIP compiler found and enabled")
    set(HIP_AVAILABLE TRUE)
else()
    message(STATUS "HIP compiler not found or not working, continuing without HIP support")
    set(HIP_AVAILABLE FALSE)
endif()

# Set output directory for executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Configure ROCm/HIP paths (only if HIP is available)
if(HIP_AVAILABLE)

endif()

find_package(CUDAToolkit REQUIRED)

# Add OpenMP flags for CUDA (avoid duplication)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fopenmp")

# Add compiler warnings (avoid duplication by checking if already set)
if(NOT CMAKE_CXX_FLAGS MATCHES "-Werror")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall")
endif()
if(NOT CMAKE_CUDA_FLAGS MATCHES "--Werror")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --Werror all-warnings -diag-suppress 20012,20015")
endif()
if(HIP_AVAILABLE AND NOT CMAKE_HIP_FLAGS MATCHES "-Werror")
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Werror -Wall")
endif()

# Set OpenMP flags and libraries
set(OpenMP_CXX_FLAGS "-fopenmp")
set(OpenMP_CXX_LIB_NAMES "omp")
set(OpenMP_CUDA_FLAGS "-Xcompiler=-fopenmp")
set(OpenMP_CUDA_LIB_NAMES "omp")
if(HIP_AVAILABLE)
    set(OpenMP_HIP_FLAGS "-fopenmp")
    set(OpenMP_HIP_LIB_NAMES "omp")
endif()
set(OpenMP_omp_LIBRARY "omp")

# Find Eigen package
find_package(Eigen3 REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Add include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/third-party/cxxopts/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third-party/cutlass/include
    ${EIGEN3_INCLUDE_DIR}
)

# Conditionally add HIP include directory
if(HIP_AVAILABLE)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/hip)
endif()

# Include all tools
add_subdirectory(src/benchmarks)
