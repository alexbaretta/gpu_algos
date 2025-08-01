// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/cuda/kernels/matrix/matrix_product_warp.hpp

#pragma once

#include <iostream>

#include <cuda_runtime.h>
#include <cxxopts.hpp>
#include <Eigen/Dense>

#include "cuda/kernel_api/matrix_2in_1out.cuh"
#include "cuda/type_traits.cuh"
#include "cuda/cuda_utils.cuh"

/*
    This kernel uses a "one warp per result element" threading strategy.
    Conceptually, we want to allocate one warp per result element, and each warp
    will compute the result element for a single row of the left matrix and a single
    column of the right matrix.

    The grid is then defined by the Number of result elements, which is the product of
    the Number of rows of the left matrix and the Number of columns of the right matrix.

    The threads in each warp collaborate to compute the result element for a single row
    of the left matrix and a single column of the right matrix: each thread processes
    the products whose index mod WARP_SIZE is equal to the thread's index within the warp.

    Warp-reduction is then used to compute the sum of the partial results computed by
    the threads in the warp.
*/

struct Matrix_product_warp_spec {
    constexpr static int TILE_SIZE = 4;
    constexpr static int DEFAULT_M = 3000; // Rows of first matrix
    constexpr static int DEFAULT_K = 300;  // Columns of first matrix / Rows of second matrix
    constexpr static int DEFAULT_N = 1000; // Columns of second matrix

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

    const long warp_size_;
    const dim3 block_dim_;
    const dim3 grid_dim_;
    const size_t dynamic_shared_mem_words_ = 0;

    inline static void add_kernel_spec_options(cxxopts::Options& options) {
        options.add_options()
            ("M", "Number of rows in first matrix", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_M)))
            ("K", "Number of columns in first matrix and rows of the second matrix", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_K)))
            ("N", "Number of columns in the second matrix", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_N)))
            ("type", "Numeric type (half, single/float, double, int<n>, uint<n>)", cxxopts::value<std::string>()->default_value("float"));
        ;
    }

    inline static Matrix_product_warp_spec make(
        const cxxopts::ParseResult& options_parsed
    ) {
        // Validate the type option
        const auto& type = options_parsed["type"].as<std::string>();
        if (type != "half" && type != "single" && type != "float" && type != "double" && type != "int8" && type != "int16" && type != "int32" && type != "int64" && type != "uint8" && type != "uint16" && type != "uint32" && type != "uint64") {
            std::cerr << "[ERROR] --type must be one of: half, single/float, double, int<n>, uint<n>" << std::endl;
            throw cxxopts::exceptions::exception("Invalid --type: " + type);
        }
        return Matrix_product_warp_spec(
            type,
            options_parsed["M"].as<long>(),
            options_parsed["K"].as<long>(),
            options_parsed["N"].as<long>()
        );
    }

    inline Matrix_product_warp_spec(
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
        warp_size_(get_warp_size()),
        block_dim_(warp_size_, TILE_SIZE, TILE_SIZE),
        grid_dim_((n_ + TILE_SIZE - 1) / TILE_SIZE, (m_ + TILE_SIZE - 1) / TILE_SIZE)
    {}
};

static_assert(Check_matrix_kernel_spec_2In_1Out<Matrix_product_warp_spec>::check_passed, "Matrix_product_warp_spec is not a valid kernel spec");

template <CUDA_scalar CUDA_Number>
__global__ void matrix_product_warp(
    const CUDA_Number* A, // m x k
    const CUDA_Number* B, // k x n
    CUDA_Number* C, // m x n
    const long m,
    const long k, // shared dimension
    const long n
) {
    // thread id is (x + y Dx + z Dx Dy, see https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy
    // int thread_id = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    // but blockDim.x is 32, so the lane id is just threadIdx.x

    int tid_warp = threadIdx.x;
    int col = threadIdx.y + blockIdx.x * blockDim.y;
    int row = threadIdx.z + blockIdx.y * blockDim.z;

    // Bounds checking to prevent out-of-bounds memory access
    if (row >= m || col >= n) {
        return;
    }

    CUDA_Number sum{0};

    // Compute the partial sum for the current thread by iterating over shared dimension k
    for (int i = tid_warp; i < k; i += warpSize) {
        sum += A[row * k + i] * B[i * n + col];
    }

    // Reduce the partial sum using warp-reduction
    for (int offset = warpSize/2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(__activemask(), sum, offset);
    }

    // Store the result
    if (tid_warp == 0) {
        C[row * n + col] = sum;
    }
}

template <CUDA_scalar Number_>
class Matrix_product_warp_kernel {
    public:
    using Number = Number_;
    using Kernel_spec = Matrix_product_warp_spec;

    const Kernel_spec spec_;

    Matrix_product_warp_kernel(
        const Kernel_spec spec
    ) : spec_(spec) {}

    void run_device_kernel(
        const Number* const gpu_data_A,
        const Number* const gpu_data_B,
        Number* const gpu_data_C,
        Number* const gpu_data_temp,
        cudaStream_t stream
    ) {
        matrix_product_warp<<<
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
static_assert(Check_matrix_kernel_2In_1Out_template<Matrix_product_warp_kernel>::check_passed, "Matrix_product_warp is not a valid kernel template");
