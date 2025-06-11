# Copyright (c) 2025 Alessandro Baretta
# All rights reserved.

# source path: src/benchmarks/matrix_product/CMakeLists.txt

# Matrix product executables
add_hip_executable(hip_matrix_product_naive matrix_product_naive.hip.cpp)
add_hip_executable(hip_matrix_product_tiled matrix_product_tiled.hip.cpp)
add_hip_executable(hip_matrix_product_warp matrix_product_warp.hip.cpp)
add_hip_executable(hip_matrix_product_tensor matrix_product_tensor.hip.cpp)
add_hip_executable(hip_matrix_product_rocblas matrix_product_rocblas.hip.cpp)

# Additional rocBLAS linking
target_link_libraries(hip_matrix_product_rocblas PRIVATE
    rocblas
)
