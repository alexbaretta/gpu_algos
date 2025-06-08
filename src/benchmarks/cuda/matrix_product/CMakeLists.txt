# Copyright (c) 2025 Alessandro Baretta
# All rights reserved.

# source path: src/benchmarks/matrix_product/CMakeLists.txt

# Matrix product executables
add_gpu_executable(matrix_product_naive matrix_product_naive.cu)
add_gpu_executable(matrix_product_tiled matrix_product_tiled.cu)
add_gpu_executable(matrix_product_warp matrix_product_warp.cu)
add_gpu_executable(matrix_product_tensor matrix_product_tensor.cu)
add_gpu_executable(matrix_product_cublas matrix_product_cublas.cu)

# Additional cuBLAS linking
target_link_libraries(matrix_product_cublas PRIVATE
    CUDA::cublas
    CUDA::cublasLt
)


# CUTLASS causes errors due to initializer order mismatch
set(DEFAULT_CUDA_FLAGS_ORIGINAL "${DEFAULT_CUDA_FLAGS}")
string(REPLACE ",reorder" "" DEFAULT_CUDA_FLAGS "${DEFAULT_CUDA_FLAGS}")
add_gpu_executable(matrix_product_cutlass matrix_product_cutlass.cu)
set(DEFAULT_CUDA_FLAGS "${DEFAULT_CUDA_FLAGS_ORIGINAL}")


# # Additional compilation flags for CUTLASS implementation
# target_compile_options(matrix_product_cutlass PRIVATE
#     # $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
#     # $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress 20013>
#     CMAKE_CXX_FLAGS string(REPLACE <match-string> <replace-string> <out-var> <input>...)
# )
