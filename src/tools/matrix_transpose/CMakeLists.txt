# Copyright (c) 2025 Alessandro Baretta
# All rights reserved.

# source path: src/tools/matrix_product/CMakeLists.txt

# Matrix product executables
add_gpu_executable(matrix_transpose_naive matrix_transpose_naive.cu)
add_gpu_executable(matrix_transpose_tiled matrix_transpose_tiled.cu)
