cmake_minimum_required(VERSION 3.18)

# Set C++ compiler before project()
set(CMAKE_CXX_COMPILER "g++-14")
set(CMAKE_CUDA_HOST_COMPILER "g++-14")
# set(CMAKE_CXX_COMPILER "clang-19")
# set(CMAKE_CUDA_HOST_COMPILER "clang-19")

project(gpu_algos CUDA CXX)

# Set CUDA specific flags
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75)  # Compute capability 7.5

# Add compiler warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror -Wall -Wextra -diag-suppress 20012,20015")

# Add debug flags if DEBUG is set
if(DEFINED ENV{DEBUG})
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
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
    ${CMAKE_CURRENT_SOURCE_DIR}/include/common
    ${CMAKE_CURRENT_SOURCE_DIR}/third-party/cxxopts/include
    ${EIGEN3_INCLUDE_DIR}
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
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
)
