// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

#pragma once
#include <cuda_runtime.h>

#include "cuda/kernel_api.h"
#include "cuda/type_traits.h"

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

class Matrix_product_naive {
    public:
    using NUMBER = float;

    const NUMBER* gpu_data_A;
    const NUMBER* gpu_data_B;
    NUMBER* gpu_data_C;
    const unsigned int nrows;
    const unsigned int ncols;
    const dim3 block_dim;
    const dim3 grid_dim;
    const size_t shared_mem_size;

    Matrix_product_naive(
        const NUMBER* gpu_data_A,
        const NUMBER* gpu_data_B,
        NUMBER* gpu_data_C,
        const unsigned int nrows,
        const unsigned int ncols
    ) : gpu_data_A(gpu_data_A),
        gpu_data_B(gpu_data_B),
        gpu_data_C(gpu_data_C),
        nrows(nrows),
        ncols(ncols),
        block_dim(16, 16),
        grid_dim(
            (nrows + block_dim.x - 1) / block_dim.x,
            (nrows + block_dim.y - 1) / block_dim.y
        ),
        shared_mem_size(0)
    {}

    void run_kernel(cudaStream_t stream) {
        matrix_product_naive<<<grid_dim, block_dim, shared_mem_size, stream>>>(gpu_data_A, gpu_data_B, gpu_data_C, nrows, ncols);
    }
};

static_assert(Check_kernel_spec<Matrix_product_naive>::check_passed, "Matrix_product_naive is not a valid kernel");
