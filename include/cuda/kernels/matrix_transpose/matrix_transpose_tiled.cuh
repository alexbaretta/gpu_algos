// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/cuda/kernels/matrix/matrix_transpose_tiled.hpp

#pragma once

#include <cmath>
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cxxopts.hpp>
#include <Eigen/Dense>
#include <cassert>

#include "cuda/kernel_api/matrix_1in_1out.cuh"
#include "cuda/type_traits.cuh"
#include "cuda/check_errors.cuh"

constexpr long TILE_DIM = 32;


template <CUDA_scalar CUDA_Number>
__global__ void matrix_transpose_tiled(
    const CUDA_Number* A,
    CUDA_Number* C,
    const long A_ncols, // cols of A, rows of C
    const long A_nrows  // rows of A, cols of C
) {
    constexpr bool debug = (
    #ifdef DEBUG
        true
    #else
        false
    #endif
    );

    /*
        The key idea of this algorithm is that warps should, as much as possible,
        work on rows to take advantage of the coalescing of memory operations. So we don't
        want the same thread to operate first on A(i, j) then on C(j, i), otherwise
        either the reads or the writes will not be coalesced.
    */
    /*
        CUDA doesn't immediately 'support' dynamically-allocated shared memory arrays in templated functions, as it (apparently)
        generates actual definitions of those extern's. If you instantiate a templated function for multiple types, the
        definitions would conflict.

        Quoted from: See https://stackoverflow.com/questions/20497209/getting-cuda-error-declaration-is-incompatible-with-previous-variable-name

        As a consequence we need to use a statically-allocated shared memory array with a maximum size of TILE_DIM * TILE_DIM.
    */
    __shared__ CUDA_Number shm[TILE_DIM][TILE_DIM + 1];
    #ifdef DEBUG
    [[maybe_unused]] const CUDA_Number* __shm = (CUDA_Number*)shm;
    #endif

    // row-major matrix of shape (blockDim.x, blockDim.y)
    // We add one element of padding to each row to avoid bank conflicts
    // const auto block_size = blockDim.x * blockDim.y;

    const auto A_row = blockIdx.y * blockDim.y + threadIdx.y;
    const auto A_col = blockIdx.x * blockDim.x + threadIdx.x;

    if (A_row < A_nrows && A_col < A_ncols) {
        // We can read
        const auto A_idx = long(A_row) * A_ncols + A_col;
        if constexpr (debug) {
            const auto n_elems = long(A_ncols)*A_nrows;
            if (A_idx >= n_elems) {
                printf("Bad indexing: A_idx=%ld at xy=(%d,%d) when A is %ldx%ld = %ld\n",
                A_idx, A_row, A_col, A_nrows, A_ncols, n_elems);
            }
        }
        const auto shared_value = A[A_idx];
        shm[threadIdx.x][threadIdx.y] = shared_value;
    } else if constexpr(debug) {
        printf("No can read at xy=(%d,%d)\n", A_row, A_col);
    }

    __syncthreads();

    #define C_ncols A_nrows
    #define C_nrows A_ncols

    const auto tid_in_block = threadIdx.y * blockDim.x + threadIdx.x;
    const auto C_tid_x = tid_in_block % blockDim.y;
    const auto C_tid_y = tid_in_block / blockDim.y;
    const auto C_row = (
        // Transpose the blocks in the grid...
        blockIdx.x * blockDim.x +

        // ...and rotate each block while maintaining Row major order
        C_tid_y
    );
    const auto C_col = (
        // Transpose the blocks in the grid...
        blockIdx.y * blockDim.y +

        // ...and rotate each block while maintaining Row major order
        C_tid_x
    );

    if (C_row < C_nrows && C_col < C_ncols) {
        // We can write
        const auto C_idx = long(C_row) * C_ncols + C_col;
        if constexpr (debug) {
            const auto n_elems = long(A_ncols)*A_nrows;
            if (C_idx >= n_elems) {
                printf("Bad indexing: C_idx=%ld at xy=(%d,%d) when A is %ldx%ld = %ld\n", C_idx, C_row, C_col, C_nrows, C_ncols, n_elems);
            }
        }

        // This is where we transpose the tile in shm
        const auto shared_value = shm[C_tid_y][C_tid_x];
        if constexpr (debug) {
            C[C_idx] = shared_value;
        }
        C[C_idx] = shared_value;
    } else if constexpr(debug) {
        printf("No can write at xy=(%d,%d)\n", C_row, C_col);
    }
}

struct Matrix_transpose_tiled_spec {
    const std::string type_;

    const long m_;    // Rows of input matrix, cols of output matrix
    const long n_;    // Columns of input matrix, rows of output matrix
    constexpr static long k_ = 0;  // unused

    const long n_rows_A_;
    const long n_cols_A_;

    const long n_rows_C_;
    const long n_cols_C_;

    const long n_rows_temp_;
    const long n_cols_temp_;

    const dim3 block_dim_;
    const dim3 grid_dim_;
    const size_t dynamic_shared_mem_words_;

    constexpr static int DEFAULT_M = 3000; // rows of A, cols of C
    constexpr static int DEFAULT_N = 300;  // cols of A, rows of C
    constexpr static int DEFAULT_K = 1000; // unused
    constexpr static int DEFAULT_BLOCK_DIM = 32;

    inline static void add_kernel_spec_options(cxxopts::Options& options) {
        options.add_options()
            ("m", "Number of rows in input matrix", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_M)))
            ("n", "Number of columns in input matrix", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_N)))
            ("k", "Unused", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_K)))
            ("block-dim,x", "Number of threads per block", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_BLOCK_DIM)))
            ("type", "Numeric type (half, single/float, double, int<n>, uint<n>)", cxxopts::value<std::string>()->default_value("float"));
        ;
    }

    inline static Matrix_transpose_tiled_spec make(
        const cxxopts::ParseResult& options_parsed
    ) {
        // Validate the type option
        const auto& type = options_parsed["type"].as<std::string>();
        if (type != "half" && type != "single" && type != "float" && type != "double" && type != "int8" && type != "int16" && type != "int32" && type != "int64" && type != "uint8" && type != "uint16" && type != "uint32" && type != "uint64") {
            std::cerr << "[ERROR] --type must be one of: half, single/float, double, int<n>, uint<n>" << std::endl;
            throw cxxopts::exceptions::exception("Invalid --type: " + type);
        }
        return Matrix_transpose_tiled_spec(
            type,
            options_parsed["m"].as<long>(),
            options_parsed["n"].as<long>(),
            options_parsed["block-dim"].as<long>()
        );
    }

    inline Matrix_transpose_tiled_spec(
        const std::string& type,
        const long m,
        const long n,
        const long block_dim
    ) : type_(type),
        m_(m),
        n_(n),
        n_rows_A_(m),
        n_cols_A_(n),
        n_rows_C_(n),
        n_cols_C_(m),
        n_rows_temp_(0),
        n_cols_temp_(0),

        // Threads per tile
        block_dim_(TILE_DIM, TILE_DIM, 1),

        grid_dim_(
            (
                // Number of tiles to cover the columns of A
                ((n_cols_A_ + TILE_DIM - 1) / TILE_DIM)
            ),
            (
                // Number of tiles to cover the rows of A
                ((n_rows_A_ + TILE_DIM - 1) / TILE_DIM)
            )
        ),
        dynamic_shared_mem_words_(0)
    {
        assert(grid_dim_.x * grid_dim_.y * block_dim_.x * block_dim_.y * TILE_DIM >= n_ * m_);
    }
};

static_assert(Check_matrix_kernel_spec_1In_1Out<Matrix_transpose_tiled_spec>::check_passed, "Matrix_transpose_tiled_spec is not a valid kernel spec");


template <CUDA_scalar Number_>
class Matrix_transpose_tiled_kernel {
    public:
    using Number = Number_;
    using Kernel_spec = Matrix_transpose_tiled_spec;

    const Kernel_spec spec_;

    Matrix_transpose_tiled_kernel(
        const Kernel_spec spec
    ) : spec_(spec) {}

    void run_device_kernel(
        const Number* const gpu_data_A,
        Number* const gpu_data_C,
        Number* const gpu_data_temp,
        cudaStream_t stream
    ) {
        int max_block_size = 0;
        int opt_grid_size = 0;
        int max_active_blocks_per_multiprocessor = 0;
        cuda_check_error(cudaOccupancyMaxPotentialBlockSize(
            &max_block_size,
            &opt_grid_size,
            matrix_transpose_tiled<Number>,
            0,
            0
        ), "cudaOccupancyMaxPotentialBlockSize");
        cuda_check_error(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_active_blocks_per_multiprocessor,
            matrix_transpose_tiled<Number>,
            max_block_size,
            0
        ), "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
        const unsigned nrows_A = spec_.m_, ncols_A = spec_.n_;
        const unsigned desired_block_size = spec_.block_dim_.x * spec_.block_dim_.y;
        const auto [block_dim, grid_dim] = [&](){
            if (int(desired_block_size) <= int(max_block_size)) {
                return std::make_tuple(spec_.block_dim_, spec_.grid_dim_);
            } else {
                const unsigned max_block_dim = unsigned(std::ceil(std::sqrt(max_block_size)));
                const dim3 constrained_block_dim = {max_block_dim, max_block_dim};
                const dim3 constrained_grid_dim = {(ncols_A + max_block_dim - 1)/max_block_dim, (nrows_A + max_block_dim - 1)/max_block_dim};
                return std::make_tuple(constrained_block_dim, constrained_grid_dim);
            }
        }();

        const auto shared_mem_size = spec_.dynamic_shared_mem_words_ * sizeof(Number);
        std::cout << "[INFO] matrix_transpose_tiled<<<(" <<
            grid_dim.x << ", " << grid_dim.y << ", " << grid_dim.z << "), " <<
            "(" << block_dim.x << ", " << block_dim.y << ", " << block_dim.z << "), " <<
            shared_mem_size << ">>>(gpu_data_A, gpu_data_C, " << spec_.m_ << ", " << spec_.n_ << ")" <<
            std::endl;
        matrix_transpose_tiled<<<
            grid_dim,
            block_dim,
            shared_mem_size,
            stream
        >>>(gpu_data_A, gpu_data_C, ncols_A, nrows_A);
    }

    Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> run_host_kernel(
        const Eigen::Map<Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& A
    ) {
        return A.transpose().eval();
    }

};
static_assert(Check_matrix_kernel_1In_1Out_template<Matrix_transpose_tiled_kernel>::check_passed, "Matrix_transpose_tiled is not a valid kernel template");
