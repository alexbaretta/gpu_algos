// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

#include "cuda/concepts.h"

template <CUDA_floating_point CUDA_FLOAT>
__global__ void matrix_product_naive(
    const CUDA_FLOAT* A,
    const CUDA_FLOAT* B,
    CUDA_FLOAT* C,
    unsigned int nrows,
    unsigned int ncols
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < nrows && col < nrows) {
        CUDA_FLOAT sum = 0.0f;
        for (int i = 0; i < ncols; ++i) {
            sum += A[row * ncols + i] * B[i * nrows + col];
        }
        C[row * nrows + col] = sum;
    }
}
