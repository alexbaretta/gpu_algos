cmake_minimum_required(VERSION 3.18)

# Set C++ compiler before project()
set(CMAKE_CXX_COMPILER "g++-14")

project(gpu_algos CUDA CXX)

# Set CUDA specific flags
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89)  # Compute capability 8.9

# Find CUDA package
find_package(CUDA 12.9 REQUIRED)

# Add include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/common
    ${CMAKE_CURRENT_SOURCE_DIR}/third-party/cxxopts/include
)

# Add executable
add_executable(matrix_multiply
    src/matrix_multiply/matrix_multiply.cu
    src/common/cuda/cuda_utils.cu
)

# Set CUDA specific properties
set_target_properties(matrix_multiply PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 89
)
