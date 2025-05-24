// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/cuda/kernels/matrix/matrix_product_opt.h

#pragma once
#include <cuda_runtime.h>

#include "cuda/kernel_api.h"
#include "cuda/type_traits.h"

template <CUDA_floating_point CUDA_FLOAT>
__global__ void matrix_product_opt(
    const CUDA_FLOAT* A,
    const CUDA_FLOAT* B,
    CUDA_FLOAT* C,
    unsigned int m,
    unsigned int n,
    unsigned int k
) {
    // Tile size (matches block dimensions)
    const int TILE_SIZE = 16;

    // Shared memory for caching tiles of A and B
    __shared__ CUDA_FLOAT tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ CUDA_FLOAT tile_B[TILE_SIZE][TILE_SIZE];

    // Thread indices within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global indices for the output element this thread computes
    int global_col = blockIdx.x * blockDim.x + threadIdx.x;
    int global_row = blockIdx.y * blockDim.y + threadIdx.y;

    CUDA_FLOAT accumulator = 0.0f;

    // Process matrix multiplication in tiles across the shared dimension n
    int num_tiles = (n + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        // Load tile of matrix A into shared memory
        int a_col = tile_idx * TILE_SIZE + tx;
        if (global_row < m && a_col < n) {
            tile_A[ty][tx] = A[global_row * n + a_col];
        } else {
            tile_A[ty][tx] = 0.0f;  // Padding for out-of-bounds
        }

        // Load tile of matrix B into shared memory
        int b_row = tile_idx * TILE_SIZE + ty;
        if (b_row < n && global_col < k) {
            tile_B[ty][tx] = B[b_row * k + global_col];
        } else {
            tile_B[ty][tx] = 0.0f;  // Padding for out-of-bounds
        }

        // Synchronize threads to ensure tiles are fully loaded
        __syncthreads();

        // Compute partial dot product using the cached tiles
        for (int i = 0; i < TILE_SIZE; ++i) {
            accumulator += tile_A[ty][i] * tile_B[i][tx];
        }

        // Synchronize before loading the next tile
        __syncthreads();
    }

    // Write the final result to global memory
    if (global_row < m && global_col < k) {
        C[global_row * k + global_col] = accumulator;
    }
}

struct Matrix_product_opt_spec {
    const std::string type_;

    const unsigned int m_;    // Rows of first matrix
    const unsigned int n_;    // Columns of first matrix and rows of second matrix
    const unsigned int k_;    // Columns of second matrix

    const unsigned int n_rows_A_;
    const unsigned int n_cols_A_;

    const unsigned int n_rows_B_;
    const unsigned int n_cols_B_;

    const unsigned int n_rows_C_;
    const unsigned int n_cols_C_;

    const dim3 block_dim_;
    const dim3 grid_dim_;
    const size_t shared_mem_size_ = 0;  // Will be calculated dynamically by kernel

    constexpr static int DEFAULT_M = 3000; // Rows of first matrix
    constexpr static int DEFAULT_N = 300;  // Columns of first matrix / Rows of second matrix
    constexpr static int DEFAULT_K = 1000; // Columns of second matrix

    inline static void add_kernel_spec_options(cxxopts::Options& options) {
        options.add_options()
            ("m", "Number of rows in first matrix", cxxopts::value<int>()->default_value(std::to_string(DEFAULT_M)))
            ("n", "Number of columns in first matrix and rows of the second matrix", cxxopts::value<int>()->default_value(std::to_string(DEFAULT_N)))
            ("k", "Number of columns in the second matrix", cxxopts::value<int>()->default_value(std::to_string(DEFAULT_K)))
            ("type", "Numeric type (half, single/float, double)", cxxopts::value<std::string>()->default_value("float"));
        ;
    }

    inline static Matrix_product_opt_spec make(
        const cxxopts::ParseResult& options_parsed
    ) {
        // Validate the type option
        const auto& type = options_parsed["type"].as<std::string>();
        if (type != "half" && type != "single" && type != "float" && type != "double") {
            std::cerr << "[ERROR] --type must be one of: half, single/float, double" << std::endl;
            throw cxxopts::exceptions::exception("Invalid --type: " + type);
        }
        return Matrix_product_opt_spec(
            type,
            options_parsed["m"].as<int>(),
            options_parsed["n"].as<int>(),
            options_parsed["k"].as<int>()
        );
    }

    inline Matrix_product_opt_spec(
        const std::string& type,
        const unsigned int m,
        const unsigned int n,
        const unsigned int k
    ) : type_(type),
        m_(m),
        n_(n),
        k_(k),
        n_rows_A_(m),
        n_cols_A_(n),
        n_rows_B_(n),
        n_cols_B_(k),
        n_rows_C_(m),
        n_cols_C_(k),
        block_dim_(16, 16),
        grid_dim_(
            (k_ + block_dim_.x - 1) / block_dim_.x,
            (m_ + block_dim_.y - 1) / block_dim_.y
        )
    {}
};

static_assert(Check_kernel_spec<Matrix_product_opt_spec>::check_passed, "Matrix_product_opt_spec is not a valid kernel spec");


template <CUDA_floating_point NUMBER_>
class Matrix_product_opt_kernel {
    public:
    using NUMBER = NUMBER_;
    using KERNEL_SPEC = Matrix_product_opt_spec;

    const KERNEL_SPEC spec_;

    const NUMBER* const gpu_data_A_;
    const NUMBER* const gpu_data_B_;
    NUMBER* const gpu_data_C_;
    cudaStream_t& stream_;

    // Calculate shared memory size based on data type
    static constexpr size_t get_shared_mem_size() {
        // Two 16x16 tiles in shared memory
        constexpr int TILE_SIZE = 16;
        return 2 * TILE_SIZE * TILE_SIZE * sizeof(NUMBER);
    }

    Matrix_product_opt_kernel(
        const KERNEL_SPEC spec,
        const NUMBER* const gpu_data_A,
        const NUMBER* const gpu_data_B,
        NUMBER* const gpu_data_C,
        cudaStream_t& stream
    ) : spec_(spec),
        gpu_data_A_(gpu_data_A),
        gpu_data_B_(gpu_data_B),
        gpu_data_C_(gpu_data_C),
        stream_(stream)
    {}

    void run_kernel() {
        matrix_product_opt<<<
            spec_.grid_dim_,
            spec_.block_dim_,
            get_shared_mem_size(),
            stream_
        >>>(gpu_data_A_, gpu_data_B_, gpu_data_C_, spec_.m_, spec_.n_, spec_.k_);
    }

};
static_assert(Check_kernel_template<Matrix_product_opt_kernel>::check_passed, "Matrix_product_opt is not a valid kernel template");
