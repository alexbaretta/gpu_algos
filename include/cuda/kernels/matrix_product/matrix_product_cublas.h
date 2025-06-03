// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/cuda/kernels/matrix/matrix_product_cublas.h

#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#include "cuda/kernel_api.h"
#include "cuda/type_traits.h"

struct Matrix_product_cublas_spec {
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
            ("type", "Numeric type (half, single/float, double, int<n>, uint<n>)", cxxopts::value<std::string>()->default_value("float"));
        ;
    }

    inline static Matrix_product_cublas_spec make(
        const cxxopts::ParseResult& options_parsed
    ) {
        // Validate the type option
        const auto& type = options_parsed["type"].as<std::string>();
        if (type != "half" && type != "single" && type != "float" && type != "double" && type != "int8" && type != "int16" && type != "int32" && type != "int64" && type != "uint8" && type != "uint16" && type != "uint32" && type != "uint64") {
            std::cerr << "[ERROR] --type must be one of: half, single/float, double, int<n>, uint<n>" << std::endl;
            throw cxxopts::exceptions::exception("Invalid --type: " + type);
        }
        return Matrix_product_cublas_spec(
            type,
            options_parsed["m"].as<long>(),
            options_parsed["n"].as<long>(),
            options_parsed["k"].as<long>(),
            options_parsed["block_dim_x"].as<long>(),
            options_parsed["block_dim_y"].as<long>()
        );
    }

    inline Matrix_product_cublas_spec(
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

static_assert(Check_kernel_spec_2In_1Out<Matrix_product_cublas_spec>::check_passed, "Matrix_product_cublas_spec is not a valid kernel spec");


template <CUDA_scalar Number_>
class Matrix_product_cublas_kernel {
    public:
    using Number = Number_;
    using Kernel_spec = Matrix_product_cublas_spec;

    const Kernel_spec spec_;
    cublasLtHandle_t cublaslt_handle_;
    cublasHandle_t cublas_handle_;

    Matrix_product_cublas_kernel(
        const Kernel_spec spec
    ) : spec_(spec) {
        // Initialize cuBLAS and cuBLASLt handles
        cublasCreate(&cublas_handle_);
        cublasLtCreate(&cublaslt_handle_);

        // Set math mode to enable tensor cores
        cublasSetMathMode(cublas_handle_, CUBLAS_TENSOR_OP_MATH);
    }

    ~Matrix_product_cublas_kernel() {
        if (cublas_handle_) {
            cublasDestroy(cublas_handle_);
        }
        if (cublaslt_handle_) {
            cublasLtDestroy(cublaslt_handle_);
        }
    }

    void run_device_kernel(
        const Number* const gpu_data_A,
        const Number* const gpu_data_B,
        Number* const gpu_data_C,
        Number* const gpu_data_temp,
        cudaStream_t stream
    ) {
        // Set the stream for cuBLAS operations
        cublasSetStream(cublas_handle_, stream);

        const Number alpha = static_cast<Number>(1.0);
        const Number beta = static_cast<Number>(0.0);

        if constexpr (std::is_same_v<Number, __half>) {
            // Use cuBLASLt for half precision to leverage tensor cores
            cublasLtMatmulDesc_t matmul_desc;
            cublasLtMatrixLayout_t A_desc, B_desc, C_desc;

            // Create matrix descriptors
            cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_16F, CUDA_R_16F);
            cublasLtMatrixLayoutCreate(&A_desc, CUDA_R_16F, spec_.m_, spec_.n_, spec_.n_);
            cublasLtMatrixLayoutCreate(&B_desc, CUDA_R_16F, spec_.n_, spec_.k_, spec_.k_);
            cublasLtMatrixLayoutCreate(&C_desc, CUDA_R_16F, spec_.m_, spec_.k_, spec_.k_);

            // Perform matrix multiplication: C = A * B
            cublasLtMatmul(cublaslt_handle_,
                          matmul_desc,
                          &alpha,
                          gpu_data_A, A_desc,
                          gpu_data_B, B_desc,
                          &beta,
                          gpu_data_C, C_desc,
                          gpu_data_C, C_desc,
                          nullptr, nullptr, 0, stream);

            // Clean up descriptors
            cublasLtMatrixLayoutDestroy(A_desc);
            cublasLtMatrixLayoutDestroy(B_desc);
            cublasLtMatrixLayoutDestroy(C_desc);
            cublasLtMatmulDescDestroy(matmul_desc);
        } else if constexpr (std::is_same_v<Number, float>) {
            // Use cuBLAS for single precision
            cublasSgemm(cublas_handle_,
                       CUBLAS_OP_N, CUBLAS_OP_N,
                       spec_.k_, spec_.m_, spec_.n_,
                       &alpha,
                       gpu_data_B, spec_.k_,
                       gpu_data_A, spec_.n_,
                       &beta,
                       gpu_data_C, spec_.k_);
        } else if constexpr (std::is_same_v<Number, double>) {
            // Use cuBLAS for double precision
            cublasDgemm(cublas_handle_,
                       CUBLAS_OP_N, CUBLAS_OP_N,
                       spec_.k_, spec_.m_, spec_.n_,
                       &alpha,
                       gpu_data_B, spec_.k_,
                       gpu_data_A, spec_.n_,
                       &beta,
                       gpu_data_C, spec_.k_);
        }
    }
    Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> run_host_kernel(
        const Eigen::Map<Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& A,
        const Eigen::Map<Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& B
    ) {
        return (A * B).eval();
    }

};
static_assert(Check_kernel_2In_1Out_template<Matrix_product_cublas_kernel>::check_passed, "Matrix_product_cublas is not a valid kernel template");
