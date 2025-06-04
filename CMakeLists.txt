# Copyright (c) 2025 Alessandro Baretta
# All rights reserved.

cmake_minimum_required(VERSION 3.18)

# Set C++ compiler before project()
set(CMAKE_CXX_COMPILER "g++-14")
set(CMAKE_CUDA_HOST_COMPILER "g++-14")
# set(CMAKE_CXX_COMPILER "clang-19")
# set(CMAKE_CUDA_HOST_COMPILER "clang-19")

project(gpu_algos CUDA CXX)

# Set default parallel jobs for building
set(CMAKE_BUILD_PARALLEL_LEVEL 12)

# Set CUDA specific flags
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "89")  # Compute capability 8.9

# Set output directory for executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Find CUDA and cuBLAS
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

# Set OpenMP flags and libraries
set(OpenMP_CXX_FLAGS "-fopenmp")
set(OpenMP_CXX_LIB_NAMES "omp")
set(OpenMP_CUDA_FLAGS "-Xcompiler=-fopenmp")
set(OpenMP_CUDA_LIB_NAMES "omp")
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

# Include all tools
add_subdirectory(src/benchmarks)
