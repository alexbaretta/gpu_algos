# Copyright (c) 2025 Alessandro Baretta
# All rights reserved.

# source path: src/tools/matrix_product/CMakeLists.txt

# Matrix product executables
add_gpu_executable(matrix_product_naive matrix_product_naive.cu)
add_gpu_executable(matrix_product_tiled matrix_product_tiled.cu)
add_gpu_executable(matrix_product_warp matrix_product_warp.cu)

add_gpu_executable(matrix_product_cublas matrix_product_cublas.cu)

# Additional cuBLAS linking for tensor core implementation
target_link_libraries(matrix_product_cublas PRIVATE
    CUDA::cublas
    CUDA::cublasLt
)
