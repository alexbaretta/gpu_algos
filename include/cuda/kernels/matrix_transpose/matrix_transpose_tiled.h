// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/cuda/kernels/matrix/matrix_transpose_tiled.h

#pragma once
#include <stdio.h>
#include <cuda_runtime.h>

#include "cuda/kernel_api.h"
#include "cuda/type_traits.h"

constexpr unsigned int TILE_DIM = 32;

template <CUDA_floating_point CUDA_FLOAT>
__global__ void matrix_transpose_tiled(
    const CUDA_FLOAT* A,
    CUDA_FLOAT* C,
    const unsigned int m, // rows of A, cols of C
    const unsigned int n  // cols of A, rows of C
) {
    /*
        CUDA doesn't immediately 'support' dynamically-allocated shared memory arrays in templated functions, as it (apparently)
        generates actual definitions of those extern's. If you instantiate a templated function for multiple types, the
        definitions would conflict.

        Quoted from: See https://stackoverflow.com/questions/20497209/getting-cuda-error-declaration-is-incompatible-with-previous-variable-name

        As a consequence we need to use a statically-allocated shared memory array with a maximum size of TILE_DIM * TILE_DIM.
    */
    __shared__ CUDA_FLOAT shared_mem[(TILE_DIM + 1) * TILE_DIM]; // row-major matrix of shape (blockDim.x, blockDim.y)

    // for readability
    const unsigned int nrows_A = m;
    const unsigned int ncols_A = n;
    const unsigned int nrows_C = n;
    const unsigned int ncols_C = m;

    // const unsigned int n_elems = nrows_A * ncols_A;

    const unsigned int col_A = threadIdx.x + blockIdx.x * blockDim.x;

    const unsigned int stripe_idx = threadIdx.x;
    const unsigned int stripe_col_A = stripe_idx;
    const unsigned int stripe_row_C = stripe_idx;

    // Iterate over the rows to cooperatively load a tile from A
    const unsigned int start_row_A = blockIdx.y * TILE_DIM;

    if (col_A < ncols_A) {
        for (int tile_row_A = 0; tile_row_A < TILE_DIM; tile_row_A++) {
            const unsigned int row_A = start_row_A + tile_row_A;
            if (row_A >= nrows_A) {
                break;
            }
            const unsigned int A_idx = col_A + row_A * ncols_A;
            const unsigned shm_idx = tile_row_A * (TILE_DIM + 1) + stripe_col_A;
            const auto value = A[A_idx];
            // printf("A_idx:%u col_A:%u row_A:%u tile_row_A=%d stripe_col_A=%d shared_mem[%u]=%f\n",
            //     A_idx, col_A, row_A, tile_row_A, stripe_col_A, shm_idx, float(value));
            shared_mem[shm_idx] = value;
        }
    }
    __syncthreads();

    // Iterate over the columns to cooperatively store a tile to C
    const int start_col_C = start_row_A;
    const int row_C = col_A;
    if (row_C < nrows_C) {
        for (int tile_col_C = 0; tile_col_C < TILE_DIM; tile_col_C++) {
            const unsigned int col_C = start_col_C + tile_col_C;
            if (col_C >= ncols_C) {
                break;
            }
            const unsigned int C_idx = col_C + row_C * ncols_C;
            const unsigned shm_idx = tile_col_C * (TILE_DIM + 1) + stripe_row_C;
            const auto value = shared_mem[shm_idx];
            // printf("C_idx:%u col_C:%d row_C:%d tile_col_C=%d stripe_row_C=%d shared_mem[%u]=%f\n",
            //     C_idx, col_C, row_C, tile_col_C, stripe_row_C, shm_idx, float(value));
            C[C_idx] = value;
        }
    }
}

struct Matrix_transpose_tiled_spec {
    const std::string type_;

    const unsigned int m_;    // Rows of input matrix, cols of output matrix
    const unsigned int n_;    // Columns of input matrix, rows of output matrix
    constexpr static unsigned int k_ = 0;  // unused

    const unsigned int n_rows_A_;
    const unsigned int n_cols_A_;

    const unsigned int n_rows_C_;
    const unsigned int n_cols_C_;

    const dim3 block_dim_;
    const dim3 grid_dim_;
    const size_t dynamic_shared_mem_words_;

    constexpr static int DEFAULT_M = 3000; // rows of A, cols of C
    constexpr static int DEFAULT_N = 300;  // cols of A, rows of C
    constexpr static int DEFAULT_K = 1000; // unused
    constexpr static int DEFAULT_BLOCK_DIM = 32;

    inline static void add_kernel_spec_options(cxxopts::Options& options) {
        options.add_options()
            ("m", "Number of rows in input matrix", cxxopts::value<int>()->default_value(std::to_string(DEFAULT_M)))
            ("n", "Number of columns in input matrix", cxxopts::value<int>()->default_value(std::to_string(DEFAULT_N)))
            ("k", "Unused", cxxopts::value<int>()->default_value(std::to_string(DEFAULT_K)))
            ("block_dim,x", "Number of threads of the block", cxxopts::value<int>()->default_value(std::to_string(DEFAULT_BLOCK_DIM)))
            ("type", "Numeric type (half, single/float, double)", cxxopts::value<std::string>()->default_value("float"));
        ;
    }

    inline static Matrix_transpose_tiled_spec make(
        const cxxopts::ParseResult& options_parsed
    ) {
        // Validate the type option
        const auto& type = options_parsed["type"].as<std::string>();
        if (type != "half" && type != "single" && type != "float" && type != "double") {
            std::cerr << "[ERROR] --type must be one of: half, single/float, double" << std::endl;
            throw cxxopts::exceptions::exception("Invalid --type: " + type);
        }
        return Matrix_transpose_tiled_spec(
            type,
            options_parsed["m"].as<int>(),
            options_parsed["n"].as<int>(),
            options_parsed["block_dim"].as<int>()
        );
    }

    inline Matrix_transpose_tiled_spec(
        const std::string& type,
        const unsigned int m,
        const unsigned int n,
        const unsigned int block_dim
    ) : type_(type),
        m_(m),
        n_(n),
        n_rows_A_(m),
        n_cols_A_(n),
        n_rows_C_(n),
        n_cols_C_(m),

        // Threads per tile
        block_dim_(TILE_DIM),

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

static_assert(Check_kernel_spec_1In_1Out<Matrix_transpose_tiled_spec>::check_passed, "Matrix_transpose_tiled_spec is not a valid kernel spec");


template <CUDA_floating_point Number_>
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
        cudaStream_t stream
    ) {
        const auto shared_mem_size = spec_.dynamic_shared_mem_words_ * sizeof(Number);
        std::cout << "[INFO] matrix_transpose_tiled<<<(" <<
            spec_.grid_dim_.x << ", " << spec_.grid_dim_.y << ", " << spec_.grid_dim_.z << "), " <<
            "(" << spec_.block_dim_.x << ", " << spec_.block_dim_.y << ", " << spec_.block_dim_.z << "), " <<
            shared_mem_size << ">>>(gpu_data_A, gpu_data_C, " << spec_.m_ << ", " << spec_.n_ << ")" <<
            std::endl;
        matrix_transpose_tiled<<<
            spec_.grid_dim_,
            spec_.block_dim_,
            shared_mem_size,
            stream
        >>>(gpu_data_A, gpu_data_C, spec_.m_, spec_.n_);
    }

    Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> run_host_kernel(
        const Eigen::Map<Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& A
    ) {
        return A.transpose().eval();
    }

};
static_assert(Check_kernel_1In_1Out_template<Matrix_transpose_tiled_kernel>::check_passed, "Matrix_transpose_tiled is not a valid kernel template");
