// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/cuda/kernels/matrix/matrix_product_naive.h

#pragma once
#include <cuda_runtime.h>

#include "cuda/kernel_api.h"
#include "cuda/type_traits.h"

template <CUDA_floating_point CUDA_FLOAT>
__global__ void matrix_product_naive(
    const CUDA_FLOAT* A,
    const CUDA_FLOAT* B,
    CUDA_FLOAT* C,
    unsigned int m,
    unsigned int n,
    unsigned int k
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < k) {
        CUDA_FLOAT sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = sum;
    }
}

struct Matrix_product_naive_spec {
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
    const size_t shared_mem_size_ = 0;

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

    inline static Matrix_product_naive_spec make(
        const cxxopts::ParseResult& options_parsed
    ) {
        // Validate the type option
        const auto& type = options_parsed["type"].as<std::string>();
        if (type != "half" && type != "single" && type != "float" && type != "double") {
            std::cerr << "[ERROR] --type must be one of: half, single/float, double" << std::endl;
            throw cxxopts::exceptions::exception("Invalid --type: " + type);
        }
        return Matrix_product_naive_spec(
            type,
            options_parsed["m"].as<int>(),
            options_parsed["n"].as<int>(),
            options_parsed["k"].as<int>()
        );
    }

    inline Matrix_product_naive_spec(
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

static_assert(Check_kernel_spec<Matrix_product_naive_spec>::check_passed, "Matrix_product_naive_spec is not a valid kernel spec");


template <CUDA_floating_point NUMBER_>
class Matrix_product_naive_kernel {
    public:
    using NUMBER = NUMBER_;
    using KERNEL_SPEC = Matrix_product_naive_spec;

    const KERNEL_SPEC spec_;

    Matrix_product_naive_kernel(
        const KERNEL_SPEC spec
    ) : spec_(spec) {}

    void run_kernel(
        const NUMBER* const gpu_data_A,
        const NUMBER* const gpu_data_B,
        NUMBER* const gpu_data_C,
        cudaStream_t stream
    ) {
        matrix_product_naive<<<
            spec_.grid_dim_,
            spec_.block_dim_,
            spec_.shared_mem_size_,
            stream
        >>>(gpu_data_A, gpu_data_B, gpu_data_C, spec_.m_, spec_.n_, spec_.k_);
    }

};
static_assert(Check_kernel_template<Matrix_product_naive_kernel>::check_passed, "Matrix_product_naive is not a valid kernel template");
