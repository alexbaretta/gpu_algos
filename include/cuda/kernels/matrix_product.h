// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

#include "cuda/concepts.h"

template <CUDA_floating_point CUDA_FLOAT>
__global__ void matrix_product_naive(
    const CUDA_FLOAT* A,
    const CUDA_FLOAT* B,
    CUDA_FLOAT* C,
    unsigned int m,
    unsigned int n,
    unsigned int k
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        CUDA_FLOAT sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = sum;
    }
}
