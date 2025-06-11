# Copyright (c) 2025 Alessandro Baretta
# All rights reserved.

# source path: src/benchmarks/matrix_transpose/CMakeLists.txt

# Matrix product executables
add_hip_executable(hip_matrix_transpose_naive matrix_transpose_naive.hip.cpp)
add_hip_executable(hip_matrix_transpose_striped matrix_transpose_striped.hip.cpp)
add_hip_executable(hip_matrix_transpose_tiled matrix_transpose_tiled.hip.cpp)
add_hip_executable(hip_matrix_transpose_rocblas matrix_transpose_rocblas.hip.cpp)

# Additional rocBLAS linking for rocBLAS implementation
target_link_libraries(hip_matrix_transpose_rocblas PRIVATE
    rocblas
)
