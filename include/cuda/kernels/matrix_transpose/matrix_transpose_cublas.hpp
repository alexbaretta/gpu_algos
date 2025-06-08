// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/cuda/kernels/matrix_transpose/matrix_transpose_cublas.hpp

#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <string>
#include <iostream>
#include <cxxopts.hpp>
#include <Eigen/Dense>
#include <cuda_fp16.h>

#include "cuda/kernel_api/matrix_1in_1out.hpp"
#include "cuda/type_traits.hpp"

struct Matrix_transpose_cublas_spec {
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

    // Note: block_dim and grid_dim are not used with cuBLAS but kept for compatibility
    const dim3 block_dim_;
    const dim3 grid_dim_;
    const size_t dynamic_shared_mem_words_;

    constexpr static int DEFAULT_M = 3000; // rows of A, cols of C
    constexpr static int DEFAULT_N = 300;  // cols of A, rows of C
    constexpr static int DEFAULT_K = 1000; // unused
    constexpr static int DEFAULT_BLOCK_DIM_X = 16;
    constexpr static int DEFAULT_BLOCK_DIM_Y = 16;

    inline static void add_kernel_spec_options(cxxopts::Options& options) {
        options.add_options()
            ("m", "Number of rows in input matrix", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_M)))
            ("n", "Number of columns in input matrix", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_N)))
            ("k", "Unused", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_K)))
            ("block_dim_x,x", "Number of threads in the x dimension per block", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_BLOCK_DIM_X)))
            ("block_dim_y,y", "Number of threads in the y dimension per block", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_BLOCK_DIM_Y)))
            ("type", "Numeric type (half, single/float, double, int<n>, uint<n>)", cxxopts::value<std::string>()->default_value("float"));
        ;
    }

    inline static Matrix_transpose_cublas_spec make(
        const cxxopts::ParseResult& options_parsed
    ) {
        // Validate the type option
        const auto& type = options_parsed["type"].as<std::string>();
        if (type != "half" && type != "single" && type != "float" && type != "double" && type != "int8" && type != "int16" && type != "int32" && type != "int64" && type != "uint8" && type != "uint16" && type != "uint32" && type != "uint64") {
            std::cerr << "[ERROR] --type must be one of: half, single/float, double, int<n>, uint<n>" << std::endl;
            throw cxxopts::exceptions::exception("Invalid --type: " + type);
        }
        return Matrix_transpose_cublas_spec(
            type,
            options_parsed["m"].as<long>(),
            options_parsed["n"].as<long>(),
            options_parsed["block_dim_x"].as<long>(),
            options_parsed["block_dim_y"].as<long>()
        );
    }

    inline Matrix_transpose_cublas_spec(
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
        ),
        dynamic_shared_mem_words_(0)
    {}
};

static_assert(Check_matrix_kernel_spec_1In_1Out<Matrix_transpose_cublas_spec>::check_passed, "Matrix_transpose_cublas_spec is not a valid kernel spec");


template <CUDA_scalar Number_>
class Matrix_transpose_cublas_kernel {
    public:
    using Number = Number_;
    using Kernel_spec = Matrix_transpose_cublas_spec;

    const Kernel_spec spec_;
    cublasHandle_t cublas_handle_;

    Matrix_transpose_cublas_kernel(
        const Kernel_spec spec
    ) : spec_(spec) {
        // Initialize cuBLAS handle
        cublasCreate(&cublas_handle_);

        // Set math mode to enable tensor cores when available
        cublasSetMathMode(cublas_handle_, CUBLAS_DEFAULT_MATH);
    }

    ~Matrix_transpose_cublas_kernel() {
        if (cublas_handle_) {
            cublasDestroy(cublas_handle_);
        }
    }

    void run_device_kernel(
        const Number* const gpu_data_A,
        Number* const gpu_data_C,
        Number* const gpu_data_temp,
        cudaStream_t stream
    ) {
        // Set the stream for cuBLAS operations
        cublasSetStream(cublas_handle_, stream);

        const Number alpha = static_cast<Number>(1.0);
        const Number beta = static_cast<Number>(0.0);

        // Matrix transpose using cuBLAS geam
        // For row-major matrices: A is m×n, C is n×m
        // cuBLAS expects column-major, so we need to account for this

        if constexpr (std::is_same_v<Number, __half>) {
            // cuBLAS doesn't provide cublasHgeam function for half precision
            std::cerr << "Error: Half precision matrix transpose is not supported with cuBLAS geam" << std::endl;
            throw std::runtime_error("Half precision not supported in cuBLAS geam operations");
        } else if constexpr (std::is_same_v<Number, float>) {
            // For single precision, use cublasSgeam
            // C = alpha * A^T + beta * 0
            // Since we're working with row-major data but cuBLAS expects column-major:
            // - A (m×n row-major) appears as A^T (n×m column-major) to cuBLAS
            // - We want C = A^T, so C (n×m row-major) appears as C^T (m×n column-major) to cuBLAS
            // - So cuBLAS should compute: C^T = (A^T)^T = A
            cublasSgeam(cublas_handle_,
                       CUBLAS_OP_T, CUBLAS_OP_T,   // transpose the input to undo the implicit transpose
                       spec_.m_, spec_.n_,          // output dimensions as seen by cuBLAS (C^T is m×n)
                       &alpha,
                       gpu_data_A, spec_.n_,        // A with leading dimension n (as row-major m×n)
                       &beta,
                       gpu_data_A, spec_.n_,        // dummy (beta=0)
                       gpu_data_C, spec_.m_);       // C with leading dimension m (as row-major n×m)
        } else if constexpr (std::is_same_v<Number, double>) {
            // For double precision, use cublasDgeam
            cublasDgeam(cublas_handle_,
                       CUBLAS_OP_T, CUBLAS_OP_T,   // transpose the input to undo the implicit transpose
                       spec_.m_, spec_.n_,          // output dimensions as seen by cuBLAS (C^T is m×n)
                       &alpha,
                       gpu_data_A, spec_.n_,        // A with leading dimension n
                       &beta,
                       gpu_data_A, spec_.n_,        // dummy (beta=0)
                       gpu_data_C, spec_.m_);       // C with leading dimension m
        }
    }

    Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> run_host_kernel(
        const Eigen::Map<Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& A
    ) {
        return A.transpose().eval();
    }

};
static_assert(Check_matrix_kernel_1In_1Out_template<Matrix_transpose_cublas_kernel>::check_passed, "Matrix_transpose_cublas is not a valid kernel template");
