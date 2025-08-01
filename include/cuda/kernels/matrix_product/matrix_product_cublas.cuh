// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/cuda/kernels/matrix/matrix_product_cublas.hpp

#pragma once
#include <iostream>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_fp16.h>

// #include "cuda/kernel_api/matrix_2in_1out.cuh"

#include <cxxopts.hpp>
#include <Eigen/Dense>

#include "cuda/kernel_api.cuh"
#include "cuda/type_traits.cuh"
#include "cuda/wmma.cuh"

struct Matrix_product_cublas_spec {
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

    // Note: block_dim and grid_dim are not used with cuBLAS but kept for compatibility
    const dim3 block_dim_ = dim3(0);
    const dim3 grid_dim_ = dim3(0);
    const size_t dynamic_shared_mem_words_ = 0;

    constexpr static int DEFAULT_M = 3000; // Rows of first matrix
    constexpr static int DEFAULT_K = 300;  // Columns of first matrix / Rows of second matrix
    constexpr static int DEFAULT_N = 1000; // Columns of second matrix

    inline static void add_kernel_spec_options(cxxopts::Options& options) {
        options.add_options()
            ("M", "Number of rows in first matrix", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_M)))
            ("K", "Number of columns in first matrix and rows of the second matrix", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_K)))
            ("N", "Number of columns in the second matrix", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_N)))
            ("type", "Numeric type (half, single/float, double)", cxxopts::value<std::string>()->default_value("float"));
        ;
    }

    inline static Matrix_product_cublas_spec make(
        const cxxopts::ParseResult& options_parsed
    ) {
        // Validate the type option
        const auto& type = options_parsed["type"].as<std::string>();
        if (type != "half" && type != "single" && type != "float" && type != "double"  ) {
            std::cerr << "[ERROR] --type must be one of: half, single/float, double" << std::endl;
            throw cxxopts::exceptions::exception("Invalid --type: " + type);
        }
        return Matrix_product_cublas_spec(
            type,
            options_parsed["M"].as<long>(),
            options_parsed["K"].as<long>(),
            options_parsed["N"].as<long>()
        );
    }

    inline Matrix_product_cublas_spec(
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
        n_cols_temp_(0)
    {}
};

static_assert(Check_matrix_kernel_spec_2In_1Out<Matrix_product_cublas_spec>::check_passed, "Matrix_product_cublas_spec is not a valid kernel spec");

inline const char* string_of_cublasStatus_t(cublasStatus_t status) {
    /* CUBLAS status type returns */
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
    }

    throw std::runtime_error("CUBLASLT STATUS UNKNOWN: " + std::to_string(status));
}

#define check_cublaslt_status(status) \
    do { \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            const auto message = std::string("[ERROR][cuBLASLt][") + __FILE__ + ":" + std::to_string(__LINE__) + "] status = " + string_of_cublasStatus_t(status); \
            std::cerr << message << std::endl; \
            throw (std::runtime_error(string_of_cublasStatus_t(status))); \
        } \
    } while (0)
#define check_cublaslt_status_noexcept(status) \
    do { \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            const auto message = std::string("[ERROR][cuBLASLt][") + __FILE__ + ":" + std::to_string(__LINE__) + "] status = " + string_of_cublasStatus_t(status); \
            std::cerr << message << std::endl; \
        } \
    } while (0)


template <CUDA_scalar Number_>
class Matrix_product_cublas_kernel {
public:
    using Number = Number_;
    using Wmma_config = wmma_config<Number>;
    using NumberA = typename Wmma_config::argument_type;
    using NumberB = typename Wmma_config::argument_type;
    using NumberC = typename Wmma_config::result_type;
    using NumberTemp = typename Wmma_config::result_type; // Unused, but required
    using NumberInternal = typename Wmma_config::operand_type;

    using Kernel_spec = Matrix_product_cublas_spec;

    const Kernel_spec spec_;
    cublasLtHandle_t cublaslt_handle_;

    const Number alpha{1};
    const Number beta{0};

    cublasLtMatmulDesc_t matmul_desc;
    cublasLtMatrixLayout_t A_layout = nullptr, B_layout = nullptr, C_layout = nullptr;

    Matrix_product_cublas_kernel(
        const Kernel_spec spec
    ) : spec_(spec) {
        // Initialize cuBLAS and cuBLASLt handles
        check_cublaslt_status(cublasLtCreate(&cublaslt_handle_));
        // Create matrix descriptors
        check_cublaslt_status(cublasLtMatmulDescCreate(&matmul_desc, Wmma_config::cublas_compute_type, Wmma_config::cublas_scale_type));
        check_cublaslt_status(cublasLtMatrixLayoutCreate(&A_layout, Wmma_config::cublas_operand_type, spec_.m_, spec_.k_, spec_.k_));
        check_cublaslt_status(cublasLtMatrixLayoutCreate(&B_layout, Wmma_config::cublas_operand_type, spec_.k_, spec_.n_, spec_.n_));
        check_cublaslt_status(cublasLtMatrixLayoutCreate(&C_layout, Wmma_config::cublas_scale_type, spec_.m_, spec_.n_, spec_.n_));

        const cublasLtOrder_t row_major_order = CUBLASLT_ORDER_ROW;
        check_cublaslt_status(cublasLtMatrixLayoutSetAttribute(A_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major_order, sizeof(row_major_order)));
        check_cublaslt_status(cublasLtMatrixLayoutSetAttribute(B_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major_order, sizeof(row_major_order)));
        check_cublaslt_status(cublasLtMatrixLayoutSetAttribute(C_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major_order, sizeof(row_major_order)));
    }

    ~Matrix_product_cublas_kernel() {
        if (cublaslt_handle_) {
            check_cublaslt_status_noexcept(cublasLtDestroy(cublaslt_handle_));
        }
        // Clean up descriptors
        check_cublaslt_status_noexcept(cublasLtMatrixLayoutDestroy(A_layout));
        check_cublaslt_status_noexcept(cublasLtMatrixLayoutDestroy(B_layout));
        check_cublaslt_status_noexcept(cublasLtMatrixLayoutDestroy(C_layout));
        check_cublaslt_status_noexcept(cublasLtMatmulDescDestroy(matmul_desc));
    }

    void run_device_kernel(
        const NumberA* const gpu_data_A,
        const NumberB* const gpu_data_B,
        NumberC* const gpu_data_C,
        NumberTemp* const gpu_data_temp,
        cudaStream_t stream
    ) {
        // Perform matrix multiplication: C = A * B
        check_cublaslt_status(cublasLtMatmul(
            cublaslt_handle_,
            matmul_desc,
            &alpha,
            gpu_data_A, A_layout,
            gpu_data_B, B_layout,
            &beta,
            gpu_data_C, C_layout,
            gpu_data_C, C_layout,
            nullptr, nullptr, 0, stream
        ));
    }
    Eigen::Matrix<NumberA, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> run_host_kernel(
        const Eigen::Map<Eigen::Matrix<NumberA, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& A,
        const Eigen::Map<Eigen::Matrix<NumberB, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& B
    ) {
        return (A * B).eval();
    }

};

static_assert(Check_matrix_kernel_2In_1Out_template<Matrix_product_cublas_kernel>::check_passed, "Matrix_product_cublas is not a valid kernel template");
