# Copyright (c) 2025 Alessandro Baretta

# source path: src/cuda/benchmarks/vector_reduction/CMakeLists.txt

# Matrix product executables
add_cuda_executable(vector_reduction_recursive vector_reduction_recursive.cu)
add_cuda_executable(vector_sum_atomic vector_sum_atomic.cu)
