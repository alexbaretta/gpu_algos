cmake_minimum_required(VERSION 3.18)
project(cuda_matrix_multiply CUDA CXX)

# Set CUDA specific flags
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89)  # Compute capability 8.9

# Find CUDA package
find_package(CUDA 12.9 REQUIRED)

# Add executable
add_executable(matrix_multiply src/main.cu)

# Include directories
target_include_directories(matrix_multiply PRIVATE include)

# Set CUDA specific properties
set_target_properties(matrix_multiply PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 89
)
