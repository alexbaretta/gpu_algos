// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/cuda/kernels/matrix/matrix_transpose_naive.hpp

#pragma once
#include <iostream>

#include <cuda_runtime.h>
#include <cxxopts.hpp>
#include <Eigen/Dense>
#include "cuda/kernel_api/matrix_1in_1out.cuh"
#include "cuda/type_traits.cuh"

template <CUDA_scalar CUDA_Number>
__global__ void matrix_transpose_naive(
    const CUDA_Number* A,
    CUDA_Number* C,
    const long m, // rows of A, cols of C
    const long n  // cols of A, rows of C
) {
    // for readability
    // const long nrows_A = m;
    const long ncols_A = n;
    // const long nrows_C = n;
    const long ncols_C = m;

    int i = blockIdx.x * blockDim.x + threadIdx.x; // col of A, row of B
    int j = blockIdx.y * blockDim.y + threadIdx.y; // row of A, col of B

    // for readability
    const long col_A = i;
    const long row_A = j;
    const long col_C = j;
    const long row_C = i;

    if (!(col_A < n && row_A < m)) return;
    C[col_C + ncols_C * row_C] = A[col_A + ncols_A * row_A];
}

struct Matrix_transpose_naive_spec {
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
    const size_t dynamic_shared_mem_words_ = 0;

    constexpr static int DEFAULT_M = 3000; // rows of A, cols of C
    constexpr static int DEFAULT_N = 300;  // cols of A, rows of C
    constexpr static int DEFAULT_K = 1000; // unused
    constexpr static int DEFAULT_BLOCK_DIM_X = 16;
    constexpr static int DEFAULT_BLOCK_DIM_Y = 16;

    inline static void add_kernel_spec_options(cxxopts::Options& options) {
        options.add_options()
            ("M", "Number of rows in input matrix", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_M)))
            ("N", "Number of columns in input matrix", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_N)))
            ("block-dim-x,x", "Number of threads in the x dimension per block", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_BLOCK_DIM_X)))
            ("block-dim-y,y", "Number of threads in the y dimension per block", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_BLOCK_DIM_Y)))
            ("type", "Numeric type (half, single/float, double, int<n>, uint<n>)", cxxopts::value<std::string>()->default_value("float"));
        ;
    }

    inline static Matrix_transpose_naive_spec make(
        const cxxopts::ParseResult& options_parsed
    ) {
        // Validate the type option
        const auto& type = options_parsed["type"].as<std::string>();
        if (type != "half" && type != "single" && type != "float" && type != "double" && type != "int8" && type != "int16" && type != "int32" && type != "int64" && type != "uint8" && type != "uint16" && type != "uint32" && type != "uint64") {
            std::cerr << "[ERROR] --type must be one of: half, single/float, double, int<n>, uint<n>" << std::endl;
            throw cxxopts::exceptions::exception("Invalid --type: " + type);
        }
        return Matrix_transpose_naive_spec(
            type,
            options_parsed["m"].as<long>(),
            options_parsed["n"].as<long>(),
            options_parsed["block-dim-x"].as<long>(),
            options_parsed["block-dim-y"].as<long>()
        );
    }

    inline Matrix_transpose_naive_spec(
        const std::string& type,
        const long m,
        const long n,
        const long block_dim_x,
        const long block_dim_y
    ) : type_(type),
        m_(m),
        n_(n),
        n_rows_A_(m),
        n_cols_A_(n),
        n_rows_C_(n),
        n_cols_C_(m),
        n_rows_temp_(0),
        n_cols_temp_(0),
        block_dim_(block_dim_x, block_dim_y),
        grid_dim_(
            (n_ + block_dim_.x - 1) / block_dim_.x,
            (m_ + block_dim_.y - 1) / block_dim_.y
        )
    {}
};

static_assert(Check_matrix_kernel_spec_1In_1Out<Matrix_transpose_naive_spec>::check_passed, "Matrix_transpose_naive_spec is not a valid kernel spec");


template <CUDA_scalar Number_>
class Matrix_transpose_naive_kernel {
    public:
    using Number = Number_;
    using Kernel_spec = Matrix_transpose_naive_spec;

    const Kernel_spec spec_;

    Matrix_transpose_naive_kernel(
        const Kernel_spec spec
    ) : spec_(spec) {}

    void run_device_kernel(
        const Number* const gpu_data_A,
        Number* const gpu_data_C,
        Number* const gpu_data_temp,
        cudaStream_t stream
    ) {
        matrix_transpose_naive<<<
            spec_.grid_dim_,
            spec_.block_dim_,
            spec_.dynamic_shared_mem_words_ * sizeof(Number),
            stream
        >>>(gpu_data_A, gpu_data_C, spec_.m_, spec_.n_);
    }

    Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> run_host_kernel(
        const Eigen::Map<Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& A
    ) {
        return A.transpose().eval();
    }

};
static_assert(Check_matrix_kernel_1In_1Out_template<Matrix_transpose_naive_kernel>::check_passed, "Matrix_transpose_naive is not a valid kernel template");
