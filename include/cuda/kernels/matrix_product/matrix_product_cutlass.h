// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/cuda/kernels/matrix_product/matrix_product_cutlass.h

#pragma once
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>

#include "cuda/kernel_api.h"
#include "cuda/type_traits.h"

struct Matrix_product_cutlass_spec {
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

    // Note: block_dim and grid_dim are not used with CUTLASS but kept for compatibility
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

    inline static Matrix_product_cutlass_spec make(
        const cxxopts::ParseResult& options_parsed
    ) {
        // Validate the type option
        const auto& type = options_parsed["type"].as<std::string>();
        if (type != "half" && type != "single" && type != "float" && type != "double" && type != "int8" && type != "int16" && type != "int32" && type != "int64" && type != "uint8" && type != "uint16" && type != "uint32" && type != "uint64") {
            std::cerr << "[ERROR] --type must be one of: half, single/float, double, int<n>, uint<n>" << std::endl;
            throw cxxopts::exceptions::exception("Invalid --type: " + type);
        }
        return Matrix_product_cutlass_spec(
            type,
            options_parsed["m"].as<long>(),
            options_parsed["n"].as<long>(),
            options_parsed["k"].as<long>(),
            options_parsed["block_dim_x"].as<long>(),
            options_parsed["block_dim_y"].as<long>()
        );
    }

    inline Matrix_product_cutlass_spec(
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

static_assert(Check_kernel_spec_2In_1Out<Matrix_product_cutlass_spec>::check_passed, "Matrix_product_cutlass_spec is not a valid kernel spec");

template <CUDA_scalar Number_>
class Matrix_product_cutlass_kernel {
    public:
    using Number = Number_;
    using Kernel_spec = Matrix_product_cutlass_spec;

    const Kernel_spec spec_;

    Matrix_product_cutlass_kernel(
        const Kernel_spec spec
    ) : spec_(spec) {}

    void run_device_kernel(
        const Number* const gpu_data_A,
        const Number* const gpu_data_B,
        Number* const gpu_data_C,
        Number* const gpu_data_temp,
        cudaStream_t stream
    ) {
        // CUTLASS implementation for matrix multiplication
        // Using row-major layout to match the framework's conventions
        using RowMajor = cutlass::layout::RowMajor;

        // Define CUTLASS GEMM type based on Number template parameter
        using CutlassGemm = cutlass::gemm::device::Gemm<
            Number,     // Data-type of A matrix
            RowMajor,   // Layout of A matrix
            Number,     // Data-type of B matrix
            RowMajor,   // Layout of B matrix
            Number,     // Data-type of C matrix
            RowMajor    // Layout of C matrix
        >;

        // Create CUTLASS GEMM operator
        CutlassGemm gemm_operator;

        // Scalars for GEMM operation (alpha=1.0, beta=0.0 for C = A * B)
        Number alpha = static_cast<Number>(1.0);
        Number beta = static_cast<Number>(0.0);

        // Leading dimensions for row-major layout
        int lda = spec_.n_;  // A is m x n, so leading dimension is n
        int ldb = spec_.k_;  // B is n x k, so leading dimension is k
        int ldc = spec_.k_;  // C is m x k, so leading dimension is k

        // Construct CUTLASS GEMM arguments
        typename CutlassGemm::Arguments args(
            {static_cast<int>(spec_.m_), static_cast<int>(spec_.k_), static_cast<int>(spec_.n_)}, // Problem dimensions (M, N, K)
            {gpu_data_A, lda},    // Tensor-ref for source matrix A
            {gpu_data_B, ldb},    // Tensor-ref for source matrix B
            {gpu_data_C, ldc},    // Tensor-ref for source matrix C
            {gpu_data_C, ldc},    // Tensor-ref for destination matrix D
            {alpha, beta}         // Scalars for epilogue
        );

        // Initialize and run the CUTLASS GEMM kernel
        cutlass::Status status = gemm_operator.initialize(args);

        if (status != cutlass::Status::kSuccess) {
            // Handle initialization error
            std::cerr << "CUTLASS GEMM initialization failed" << std::endl;
            return;
        }

        // Run the kernel on the specified stream
        status = gemm_operator.run(stream);

        if (status != cutlass::Status::kSuccess) {
            // Handle execution error
            std::cerr << "CUTLASS GEMM execution failed" << std::endl;
            return;
        }
    }

    Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> run_host_kernel(
        const Eigen::Map<Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& A,
        const Eigen::Map<Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& B
    ) {
        return (A * B).eval();
    }

};
static_assert(Check_kernel_2In_1Out_template<Matrix_product_cutlass_kernel>::check_passed, "Matrix_product_cutlass is not a valid kernel template");
