# Copyright (c) 2025 Alessandro Baretta
# All rights reserved.

# source path: src/benchmarks/cuda/gradient/CMakeLists.txt

add_gpu_executable(gradient_leastsquares_elasticnet_naive gradient_leastsquares_elasticnet_naive.cu)
add_gpu_executable(matrix_gradient_elasticnet matrix_gradient_elasticnet.cu)
