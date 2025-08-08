# Copyright (c) 2025 Alessandro Baretta

# source path: src/cuda/benchmarks/matrix_transpose/CMakeLists.txt

# Matrix product executables
add_cuda_executable(matrix_transpose_naive matrix_transpose_naive.cu)
add_cuda_executable(matrix_transpose_striped matrix_transpose_striped.cu)
add_cuda_executable(matrix_transpose_tiled matrix_transpose_tiled.cu)
add_cuda_executable(matrix_transpose_cublas matrix_transpose_cublas.cu)

# Additional cuBLAS linking for cuBLAS implementation
target_link_libraries(matrix_transpose_cublas PRIVATE
    CUDA::cublas
)
