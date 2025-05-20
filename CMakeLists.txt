cmake_minimum_required(VERSION 3.18)

# Set C++ compiler before project()
set(CMAKE_CXX_COMPILER "g++-14")

project(gpu_algos CUDA CXX)

# Set CUDA specific flags
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75)  # Compute capability 7.5

# Add debug flags if DEBUG is set
if(DEFINED ENV{DEBUG})
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
endif()

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
    CUDA_ARCHITECTURES 75
)

# Link CUDA libraries
target_link_libraries(matrix_multiply
    cudart
    cuda
)
