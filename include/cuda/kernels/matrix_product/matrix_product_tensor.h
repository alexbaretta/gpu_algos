// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/cuda/kernels/matrix/matrix_product_tensor.h

#pragma once
#include <cuda_runtime.h>
#include <mma.h>
#include <cxxopts.hpp>
#include <Eigen/Dense>
#include <type_traits>
#include <cstdint>
#include <cuda_fp16.h>

#include "cuda/kernel_api.h"
#include "cuda/type_traits.h"

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

    // Note: block_dim and grid_dim are not used with cuBLAS but kept for compatibility
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
            ("type", "Numeric type (half*, int8*, single/float, double, int16, int32, int64, uint8, uint16, uint32, uint64) (* = tensor cores)", cxxopts::value<std::string>()->default_value("float"));
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
            (k_ + 15) / 16,  // Each warp handles 16 columns for WMMA
            (m_ + 15) / 16   // Each warp handles 16 rows for WMMA
        )
    {}
};

static_assert(Check_kernel_spec_2In_1Out<Matrix_product_tensor_spec>::check_passed, "Matrix_product_tensor_spec is not a valid kernel spec");

// WMMA tensor core matrix multiplication kernel
template <typename Number>
__global__ void matrix_product_tensor_wmma(
    const Number* A,
    const Number* B,
    Number* C,
    const int m,
    const int n,
    const int k
) {
    // WMMA fragment dimensions
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;

    // Shared memory for int8 conversion
    __shared__ int shared_temp[WMMA_M * WMMA_N];

    // Calculate which 16x16 tile this block handles
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;

    // Each block handles one 16x16 output tile
    const int row_base = block_row * WMMA_M;
    const int col_base = block_col * WMMA_N;

    if constexpr (std::is_same_v<Number, __half>) {
        using namespace nvcuda::wmma;

        // Get warp ID within the block
        const int warpId = (threadIdx.y * blockDim.x + threadIdx.x) / warpSize;

        // Only first warp per block performs WMMA operations
        if (warpId == 0) {
            // Declare WMMA fragments
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, row_major> b_frag;
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, __half> c_frag;

            // Initialize accumulator to zero
            fill_fragment(c_frag, 0.0f);

            // Loop over K dimension in chunks of WMMA_K
            for (int i = 0; i < n; i += WMMA_K) {
                // Bounds checking for this tile
                if (row_base < m && col_base < k &&
                    row_base + WMMA_M <= m && col_base + WMMA_N <= k &&
                    i + WMMA_K <= n) {

                    // Load fragments from global memory
                    load_matrix_sync(a_frag, A + row_base * n + i, n);
                    load_matrix_sync(b_frag, B + i * k + col_base, k);

                    // Perform tensor core matrix multiplication
                    mma_sync(c_frag, a_frag, b_frag, c_frag);
                }
            }

            // Store the result
            if (row_base < m && col_base < k &&
                row_base + WMMA_M <= m && col_base + WMMA_N <= k) {
                store_matrix_sync(C + row_base * k + col_base, c_frag, k, mem_row_major);
            }
        }
    } else if constexpr (std::is_same_v<Number, std::int8_t>) {
        using namespace nvcuda::wmma;

        // Get warp ID within the block
        const int warpId = (threadIdx.y * blockDim.x + threadIdx.x) / warpSize;

        // Only first warp per block performs WMMA operations
        if (warpId == 0) {
            // Declare WMMA fragments for INT8 inputs with INT32 accumulation
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, signed char, row_major> a_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, signed char, row_major> b_frag;
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, int> c_frag;

            // Initialize accumulator to zero
            fill_fragment(c_frag, 0);

            // Loop over K dimension in chunks of WMMA_K
            for (int i = 0; i < n; i += WMMA_K) {
                // Bounds checking for this tile
                if (row_base < m && col_base < k &&
                    row_base + WMMA_M <= m && col_base + WMMA_N <= k &&
                    i + WMMA_K <= n) {

                    // Load fragments from global memory
                    load_matrix_sync(a_frag,
                        reinterpret_cast<const signed char*>(A) + row_base * n + i, n);
                    load_matrix_sync(b_frag,
                        reinterpret_cast<const signed char*>(B) + i * k + col_base, k);

                    // Perform tensor core matrix multiplication
                    mma_sync(c_frag, a_frag, b_frag, c_frag);
                }
            }

            // Store the result (convert from INT32 back to INT8)
            if (row_base < m && col_base < k &&
                row_base + WMMA_M <= m && col_base + WMMA_N <= k) {

                // Store to shared memory instead of local memory
                store_matrix_sync(shared_temp, c_frag, WMMA_N, mem_row_major);

                // Synchronize threads within the block
                __syncthreads();

                // Convert and store as INT8
                for (int i = 0; i < WMMA_M; ++i) {
                    for (int j = 0; j < WMMA_N; ++j) {
                        if (row_base + i < m && col_base + j < k) {
                            // Clamp to INT8 range
                            int val = shared_temp[i * WMMA_N + j];
                            val = max(-128, min(127, val));
                            C[(row_base + i) * k + (col_base + j)] = static_cast<Number>(val);
                        }
                    }
                }
            }
        }
    } else {
        // Fallback to naive implementation for unsupported types
        const int row = blockIdx.y * blockDim.y + threadIdx.y;
        const int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < m && col < k) {
            Number sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += A[row * n + i] * B[i * k + col];
            }
            C[row * k + col] = sum;
        }
    }
}

template <CUDA_scalar Number_>
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
        cudaStream_t stream
    ) {
        // Launch tensor core matrix multiplication kernel
        matrix_product_tensor_wmma<<<
            spec_.grid_dim_,
            spec_.block_dim_,
            spec_.dynamic_shared_mem_words_ * sizeof(Number),
            stream
        >>>(gpu_data_A, gpu_data_B, gpu_data_C, spec_.m_, spec_.n_, spec_.k_);
    }
    Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> run_host_kernel(
        const Eigen::Map<Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& A,
        const Eigen::Map<Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& B
    ) {
        return (A * B).eval();
    }

};
static_assert(Check_kernel_2In_1Out_template<Matrix_product_tensor_kernel>::check_passed, "Matrix_product_tensor is not a valid kernel template");
