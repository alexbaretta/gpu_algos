# Copyright (c) 2025 Alessandro Baretta
# All rights reserved.

# source path: src/tools/matrix_product/CMakeLists.txt

# Matrix product executables
add_gpu_executable(matrix_product_naive matrix_product_naive.cu)
add_gpu_executable(matrix_product_opt matrix_product_opt.cu)
