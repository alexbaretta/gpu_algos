# Copyright (c) 2025 Alessandro Baretta
# All rights reserved.

# source path: src/benchmarks/vector_cumsum/CMakeLists.txt

# Matrix product executables
add_hip_executable(hip_vector_cumsum_serial vector_cumsum_serial.hip.cpp)
add_hip_executable(hip_vector_cumsum_parallel vector_cumsum_parallel.hip.cpp)
# add_hip_executable(hip_vector_cumsum_cooperative vector_cumsum_cooperative.hip.cpp)
