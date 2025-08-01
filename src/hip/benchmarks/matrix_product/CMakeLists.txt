# Copyright (c) 2025 Alessandro Baretta

# source path: src/benchmarks/matrix_product/CMakeLists.txt

# Matrix product executables
add_hip_executable(hip_matrix_product_naive matrix_product_naive.hip)
add_hip_executable(hip_matrix_product_tiled matrix_product_tiled.hip)
add_hip_executable(hip_matrix_product_warp matrix_product_warp.hip)

# Disabling executables that do not currently build successfully
# add_hip_executable(hip_matrix_product_tensor matrix_product_tensor.hip)
# add_hip_executable(hip_matrix_product_rocblas matrix_product_rocblas.hip)

# # Additional rocBLAS linking
# target_link_libraries(hip_matrix_product_rocblas PRIVATE
#     rocblas
# )
