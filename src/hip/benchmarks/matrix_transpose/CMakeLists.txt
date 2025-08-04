# Copyright (c) 2025 Alessandro Baretta

# source path: src/hip/benchmarks/matrix_transpose/CMakeLists.txt

# Matrix product executables
add_hip_executable(hip_matrix_transpose_naive matrix_transpose_naive.hip)
add_hip_executable(hip_matrix_transpose_striped matrix_transpose_striped.hip)
add_hip_executable(hip_matrix_transpose_tiled matrix_transpose_tiled.hip)
add_hip_executable(hip_matrix_transpose_rocblas matrix_transpose_rocblas.hip)

# Additional rocBLAS linking for rocBLAS implementation
target_link_libraries(hip_matrix_transpose_rocblas PRIVATE
    rocblas
)
