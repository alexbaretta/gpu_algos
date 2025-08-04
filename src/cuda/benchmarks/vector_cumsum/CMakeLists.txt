# Copyright (c) 2025 Alessandro Baretta

# source path: src/cuda/benchmarks/vector_cumsum/CMakeLists.txt

# Matrix product executables
add_gpu_executable(vector_cumsum_serial vector_cumsum_serial.cu)
add_gpu_executable(vector_cumsum_parallel vector_cumsum_parallel.cu)
# add_gpu_executable(vector_cumsum_cooperative vector_cumsum_cooperative.cu)
