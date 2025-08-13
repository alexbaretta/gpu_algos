# Copyright (c) 2025 Alessandro Baretta

# source path: src/cuda/benchmarks/glm/CMakeLists.txt

add_cuda_executable(glm_predict_naive glm_predict_naive.cu)
add_cuda_executable(glm_gradient_naive glm_gradient_naive.cu)
add_cuda_executable(glm_gradient_xyyhat glm_gradient_xyyhat.cu)
add_cuda_executable(glm_loss_myyhat glm_loss_myyhat.cu)
