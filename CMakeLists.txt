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

# Try to enable HIP if available
include(CheckLanguage)
check_language(HIP)

if(CMAKE_HIP_COMPILER)
    # Enable HIP language
    enable_language(HIP)

    # Set language mappings for HIP files
    set_property(GLOBAL PROPERTY CMAKE_HIP_LANGUAGE_EXTENSION ".hip.cpp")

    # Add ROCm cmake path so find_package can locate HIP
    list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")

    # Find HIP package using the modern CMake approach
    find_package(hip CONFIG QUIET)

    if(hip_FOUND)
        message(STATUS "HIP found and enabled")
        # Set HIP specific flags
        set(CMAKE_HIP_STANDARD 20)
        set(CMAKE_HIP_STANDARD_REQUIRED ON)
        set(CMAKE_HIP_ARCHITECTURES "gfx1201")  # AMD Radeon RX 9070 (RDNA 3+)
        set(HIP_AVAILABLE TRUE)

        # Find additional ROCm libraries that HIP targets may need
        find_package(rocblas CONFIG QUIET)
        if(rocblas_FOUND)
            message(STATUS "rocBLAS found")
        endif()
    else()
        message(WARNING "HIP compiler found but HIP package not found - HIP support disabled")
        set(HIP_AVAILABLE FALSE)
    endif()
else()
    message(STATUS "HIP compiler not found - HIP support disabled")
    set(HIP_AVAILABLE FALSE)
endif()

# Set output directory for executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Force override CUDAToolkit's librt detection to prevent linking with it
# (librt is unnecessary in modern glibc and causes nvlink issues)
# Clear any existing cached rt library references
unset(CUDAToolkit_rt_LIBRARY CACHE)
set(CUDAToolkit_rt_LIBRARY "" CACHE FILEPATH "Force disable librt linking" FORCE)
set(CUDAToolkit_rt_LIBRARY_RELEASE "" CACHE FILEPATH "Force disable librt linking (Release)" FORCE)
set(CUDAToolkit_rt_LIBRARY_DEBUG "" CACHE FILEPATH "Force disable librt linking (Debug)" FORCE)

find_package(CUDAToolkit REQUIRED)

# Force override CUDAToolkit's librt detection to prevent linking with it
# (librt is unnecessary in modern glibc and causes nvlink issues)
set(CUDAToolkit_rt_LIBRARY "" CACHE FILEPATH "Force disable librt linking" FORCE)

# Find OpenMP and let CMake automatically detect the correct implementation
find_package(OpenMP REQUIRED)

# Add OpenMP flags for CUDA (avoid duplication)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fopenmp")
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -fopenmp")

# Add OpenMP flags for HIP (if available)
if(HIP_AVAILABLE)
    # set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -fopenmp -D_OPENMP")
endif()

# Link OpenMP to all targets
link_libraries(OpenMP::OpenMP_CXX)

# Add compiler warnings (avoid duplication by checking if already set)
if(NOT CMAKE_CXX_FLAGS MATCHES "-Werror")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall")
endif()
if(NOT CMAKE_CUDA_FLAGS MATCHES "--Werror")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --Werror cross-execution-space-call,reorder,ext-lambda-captures-this,deprecated-declarations -diag-suppress 20012,20015")
endif()
if(HIP_AVAILABLE AND NOT CMAKE_HIP_FLAGS MATCHES "-Werror")
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Werror -Wall")
endif()

# Find Eigen package
find_package(Eigen3 REQUIRED)

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
