// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/hip/kernels/matrix/matrix_product_tensor.hip.h

#pragma once
#include <hip/hip_runtime.h>

#include "hip/kernel_api.hip.hpp
#include "hip/type_traits.hip.hpp

struct Matrix_product_tensor_spec {
    const std::string type_;

    const long m_;    // Rows of first matrix
    const long n_;    // Columns of first matrix and rows of second matrix
    const long k_;    // Columns of second matrix

    const long n_rows_A_;
    const long n_cols_A_;

    const long n_rows_B_;
    const long n_cols_B_;

    const long n_rows_C_;
    const long n_cols_C_;

    const long n_rows_temp_;
    const long n_cols_temp_;

    // Note: block_dim and grid_dim are not used with tensor cores but kept for compatibility
    const dim3 block_dim_;
    const dim3 grid_dim_;
    const size_t dynamic_shared_mem_words_ = 0;

    constexpr static int DEFAULT_M = 3000; // Rows of first matrix
    constexpr static int DEFAULT_N = 300;  // Columns of first matrix / Rows of second matrix
    constexpr static int DEFAULT_K = 1000; // Columns of second matrix
    constexpr static int DEFAULT_BLOCK_DIM_X = 16;
    constexpr static int DEFAULT_BLOCK_DIM_Y = 16;

    inline static void add_kernel_spec_options(cxxopts::Options& options) {
        options.add_options()
            ("m", "Number of rows in first matrix", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_M)))
            ("n", "Number of columns in first matrix and rows of the second matrix", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_N)))
            ("k", "Number of columns in the second matrix", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_K)))
            ("block_dim_x,x", "Number of threads in the x dimension per block", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_BLOCK_DIM_X)))
            ("block_dim_y,y", "Number of threads in the y dimension per block", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_BLOCK_DIM_Y)))
            ("type", "Numeric type (half*, int8*, single/float, double, int16, int32, int64, uint8, uint16, uint32, uint64) (* = matrix cores when available)", cxxopts::value<std::string>()->default_value("float"));
        ;
    }

    inline static Matrix_product_tensor_spec make(
        const cxxopts::ParseResult& options_parsed
    ) {
        // Validate the type option - now accepts all types supported by the main program
        const auto& type = options_parsed["type"].as<std::string>();
        if (type != "half" && type != "single" && type != "float" && type != "double" &&
            type != "int8" && type != "int16" && type != "int32" && type != "int64" &&
            type != "uint8" && type != "uint16" && type != "uint32" && type != "uint64") {
            std::cerr << "[ERROR] --type must be one of: half, single/float, double, int8, int16, int32, int64, uint8, uint16, uint32, uint64" << std::endl;
            throw cxxopts::exceptions::exception("Invalid --type: " + type);
        }
        return Matrix_product_tensor_spec(
            type,
            options_parsed["m"].as<long>(),
            options_parsed["n"].as<long>(),
            options_parsed["k"].as<long>(),
            options_parsed["block_dim_x"].as<long>(),
            options_parsed["block_dim_y"].as<long>()
        );
    }

    inline Matrix_product_tensor_spec(
        const std::string& type,
        const long m,
        const long n,
        const long k,
        const long block_dim_x,
        const long block_dim_y
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
        n_rows_temp_(0),
        n_cols_temp_(0),
        block_dim_(block_dim_x, block_dim_y),
        grid_dim_(
            (k_ + block_dim_.x - 1) / block_dim_.x,
            (m_ + block_dim_.y - 1) / block_dim_.y
        )
    {}
};

static_assert(Check_kernel_spec_2In_1Out<Matrix_product_tensor_spec>::check_passed, "Matrix_product_tensor_spec is not a valid kernel spec");

// Matrix multiplication kernel optimized for tensor operations
// Note: HIP doesn't have direct WMMA equivalent, but this kernel is structured
// to be compatible with potential future matrix core accelerations
template <typename Number>
__global__ void matrix_product_tensor_accelerated(
    const Number* A,
    const Number* B,
    Number* C,
    const int m,
    const int n,
    const int k
) {
    // Use shared memory for tiling to improve memory access patterns
    const int TILE_SIZE = 16;
    __shared__ Number shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ Number shared_B[TILE_SIZE][TILE_SIZE];

    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    Number sum = static_cast<Number>(0);

    // Tile across the K dimension
    for (int tile = 0; tile < (n + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load tiles into shared memory with bounds checking
        if (row < m && tile * TILE_SIZE + threadIdx.x < n) {
            shared_A[threadIdx.y][threadIdx.x] = A[row * n + tile * TILE_SIZE + threadIdx.x];
        } else {
            shared_A[threadIdx.y][threadIdx.x] = static_cast<Number>(0);
        }

        if (col < k && tile * TILE_SIZE + threadIdx.y < n) {
            shared_B[threadIdx.y][threadIdx.x] = B[(tile * TILE_SIZE + threadIdx.y) * k + col];
        } else {
            shared_B[threadIdx.y][threadIdx.x] = static_cast<Number>(0);
        }

        __syncthreads();

        // Compute partial dot product for this tile
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += shared_A[threadIdx.y][i] * shared_B[i][threadIdx.x];
        }

        __syncthreads();
    }

    // Write result to global memory
    if (row < m && col < k) {
        C[row * k + col] = sum;
    }
}

template <HIP_scalar Number_>
class Matrix_product_tensor_kernel {
    public:
    using Number = Number_;
    using Kernel_spec = Matrix_product_tensor_spec;

    const Kernel_spec spec_;
    Matrix_product_tensor_kernel(
        const Kernel_spec spec
    ) : spec_(spec) {}

    void run_device_kernel(
        const Number* const gpu_data_A,
        const Number* const gpu_data_B,
        Number* const gpu_data_C,
        Number* const gpu_data_temp,
        hipStream_t stream
    ) {
        // Launch accelerated matrix multiplication kernel
        hipLaunchKernelGGL(
            matrix_product_tensor_accelerated<Number>,
            spec_.grid_dim_,
            spec_.block_dim_,
            spec_.dynamic_shared_mem_words_ * sizeof(Number),
            stream,
            gpu_data_A, gpu_data_B, gpu_data_C, spec_.m_, spec_.n_, spec_.k_
        );
    }
    Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> run_host_kernel(
        const Eigen::Map<Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& A,
        const Eigen::Map<Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& B
    ) {
        return (A * B).eval();
    }

};
static_assert(Check_kernel_2In_1Out_template<Matrix_product_tensor_kernel>::check_passed, "Matrix_product_tensor is not a valid kernel template");
