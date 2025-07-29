// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/cuda/kernels/matrix/matrix_product_tiled.hpp

#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include <cxxopts.hpp>
#include <Eigen/Dense>
#include <cxxopts.hpp>

#include "cuda/kernel_api/matrix_2in_1out.cuh"
#include "cuda/type_traits.cuh"

constexpr long TILE_SIZE = 16;

template <CUDA_scalar CUDA_Number>
__global__ void matrix_product_tiled(
    const CUDA_Number* A,
    const CUDA_Number* B,
    CUDA_Number* C,
    const long m,
    const long k,
    const long n
) {
    // Shared memory for caching tiles of A and B
    __shared__ CUDA_Number tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ CUDA_Number tile_B[TILE_SIZE][TILE_SIZE];

    // Thread indices within the block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Global indices for the output element this thread computes
    // NOTE: This algorithm uses "one thread per output value" strategy
    // Each thread computes exactly one element C[global_row][global_col]
    const int global_col = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_row = blockIdx.y * blockDim.y + threadIdx.y;

    CUDA_Number accumulator = 0.0f;

    // TILING STRATEGY EXPLANATION:
    //
    // The tiling algorithm processes the shared dimension 'k' in chunks of TILE_SIZE.
    // While we call them "tiles", the actual access pattern creates "slivers":
    //
    // Matrix A (m×k):               Matrix B (k×n):
    // ┌─────────────────┐          ┌─────┬─────┬─────┐
    // │     │     │     │          │  │  │  │  │  │  │
    // ├─────┼─────┼─────┤          │  │  │  │  │  │  │
    // │█████│█████│█████│ ← sliver │  │  │  │  │  │  │
    // ├─────┼─────┼─────┤          │  │  │  │  │  │  │
    // │     │     │     │          │  │  │  │  │  │  │
    // └─────────────────┘          └─────┴─────┴─────┘
    //                                     ↑ sliver
    //
    // Each thread block accesses:
    // - Matrix A: horizontal sliver (fixed rows, all columns across iterations)
    // - Matrix B: vertical sliver (all rows across iterations, fixed columns)
    // - Matrix C: square 16x16 output tile (true 2D tiling)

    // Process matrix multiplication in tiles across the shared dimension k
    const int num_tiles = (k + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        // COOPERATIVE LOADING PHASE:
        // All threads in the block work together to load two 16x16 square tiles
        // into shared memory for maximum data reuse.

        // Load tile of matrix A into shared memory
        // This loads A^(I,t) where I=blockIdx.y, t=tile_idx
        // Accessing: A[blockIdx.y*16:(blockIdx.y+1)*16, tile_idx*16:(tile_idx+1)*16]
        int a_col = tile_idx * TILE_SIZE + tx;
        if (global_row < m && a_col < k) {
            tile_A[ty][tx] = A[global_row * k + a_col];
        } else {
            tile_A[ty][tx] = 0.0f;  // Padding for out-of-bounds
        }

        // Load tile of matrix B into shared memory
        // This loads B^(t,J) where t=tile_idx, J=blockIdx.x
        // Accessing: B[tile_idx*16:(tile_idx+1)*16, blockIdx.x*16:(blockIdx.x+1)*16]
        int b_row = tile_idx * TILE_SIZE + ty;
        if (b_row < k && global_col < n) {
            tile_B[ty][tx] = B[b_row * n + global_col];
        } else {
            tile_B[ty][tx] = 0.0f;  // Padding for out-of-bounds
        }

        // Synchronize threads to ensure tiles are fully loaded
        __syncthreads();

        // INDEPENDENT COMPUTATION PHASE:
        // Now each thread computes its portion of the dot product using the
        // cached tiles. Thread (tx,ty) computes element C[global_row][global_col]
        // by accumulating: tile_A[ty][i] * tile_B[i][tx] for i=0..15
        for (int i = 0; i < TILE_SIZE; ++i) {
            accumulator += tile_A[ty][i] * tile_B[i][tx];
        }

        // Synchronize before loading the next tile to prevent race conditions
        __syncthreads();
    }

    // Write the final accumulated result to global memory
    // Each thread writes exactly one output element
    if (global_row < m && global_col < n) {
        C[global_row * n + global_col] = accumulator;
    }
}

struct Matrix_product_tiled_spec {
    const std::string type_;

    const long m_;    // Rows of first matrix
    const long k_;    // Columns of first matrix and rows of second matrix
    const long n_;    // Columns of second matrix

    const long n_rows_A_;
    const long n_cols_A_;

    const long n_rows_B_;
    const long n_cols_B_;

    const long n_rows_C_;
    const long n_cols_C_;

    const long n_rows_temp_;
    const long n_cols_temp_;

    const dim3 block_dim_;
    const dim3 grid_dim_;
    constexpr static size_t dynamic_shared_mem_words_ = 0;
    constexpr static int DEFAULT_M = 3000; // Rows of first matrix
    constexpr static int DEFAULT_K = 300;  // Columns of first matrix / Rows of second matrix
    constexpr static int DEFAULT_N = 1000; // Columns of second matrix

    inline static void add_kernel_spec_options(cxxopts::Options& options) {
        options.add_options()
            ("m", "Number of rows in first matrix", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_M)))
            ("k", "Number of columns in first matrix and rows of the second matrix", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_K)))
            ("n", "Number of columns in the second matrix", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_N)))
            ("type", "Numeric type (half, single/float, double, int<n>, uint<n>)", cxxopts::value<std::string>()->default_value("float"));
        ;
    }

    inline static Matrix_product_tiled_spec make(
        const cxxopts::ParseResult& options_parsed
    ) {
        // Validate the type option
        const auto& type = options_parsed["type"].as<std::string>();
        if (type != "half" && type != "single" && type != "float" && type != "double" && type != "int8" && type != "int16" && type != "int32" && type != "int64" && type != "uint8" && type != "uint16" && type != "uint32" && type != "uint64") {
            std::cerr << "[ERROR] --type must be one of: half, single/float, double, int<n>, uint<n>" << std::endl;
            throw cxxopts::exceptions::exception("Invalid --type: " + type);
        }
        return Matrix_product_tiled_spec(
            type,
            options_parsed["m"].as<long>(),
            options_parsed["k"].as<long>(),
            options_parsed["n"].as<long>()
        );
    }

    inline Matrix_product_tiled_spec(
        const std::string& type,
        const long m,
        const long k,
        const long n
    ) : type_(type),
        m_(m),
        k_(k),
        n_(n),
        n_rows_A_(m),
        n_cols_A_(k),
        n_rows_B_(k),
        n_cols_B_(n),
        n_rows_C_(m),
        n_cols_C_(n),
        n_rows_temp_(0),
        n_cols_temp_(0),
        block_dim_(16, 16),
        grid_dim_(
            (n_ + block_dim_.x - 1) / block_dim_.x,
            (m_ + block_dim_.y - 1) / block_dim_.y
        )
    {}
};

static_assert(Check_matrix_kernel_spec_2In_1Out<Matrix_product_tiled_spec>::check_passed, "Matrix_product_tiled_spec is not a valid kernel spec");


template <CUDA_scalar Number_>
class Matrix_product_tiled_kernel {
    public:
    using Number = Number_;
    using Kernel_spec = Matrix_product_tiled_spec;

    const Kernel_spec spec_;

    Matrix_product_tiled_kernel(
        const Kernel_spec spec
    ) : spec_(spec)
    {}

    void run_device_kernel(
        const Number* gpu_data_A,
        const Number* gpu_data_B,
        Number* gpu_data_C,
        Number* gpu_data_temp,
        cudaStream_t stream
    ) const {
        matrix_product_tiled<<<
            spec_.grid_dim_,
            spec_.block_dim_,
            spec_.dynamic_shared_mem_words_ * sizeof(Number),
            stream
        >>>(gpu_data_A, gpu_data_B, gpu_data_C, spec_.m_, spec_.k_, spec_.n_);
    }
    Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> run_host_kernel(
        const Eigen::Map<Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& A,
        const Eigen::Map<Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& B
    ) {
        return (A * B).eval();
    }

};
static_assert(Check_matrix_kernel_2In_1Out_template<Matrix_product_tiled_kernel>::check_passed, "matrix_product_tiled is not a valid kernel template");
