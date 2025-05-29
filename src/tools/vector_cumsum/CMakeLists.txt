# Copyright (c) 2025 Alessandro Baretta
# All rights reserved.

# source path: src/tools/matrix_product/CMakeLists.txt

# Matrix product executables
add_gpu_executable(vector_cumsum_serial vector_cumsum_serial.cu)
add_gpu_executable(vector_cumsum_parallel vector_cumsum_parallel.cu)
add_gpu_executable(vector_cumsum_cooperative vector_cumsum_cooperative.cu)
