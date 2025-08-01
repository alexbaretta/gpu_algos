// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/cuda/kernels/matrix_product/matrix_product_cutlass.cuh

#pragma once

#include <string>

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/arch/arch.h>
#include <cxxopts.hpp>
#include <Eigen/Dense>

#include "cuda/kernel_api/matrix_2in_1out.cuh"
#include "cuda/type_traits.cuh"
#include "cutlass/half.h"
#include "cutlass/tfloat32.h"



struct Matrix_product_cutlass_spec {
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

    // Note: block_dim and grid_dim are not used with CUTLASS but kept for compatibility
    const dim3 block_dim_ = dim3(0);
    const dim3 grid_dim_  = dim3(0);
    const size_t dynamic_shared_mem_words_ = 0;

    constexpr static int DEFAULT_M = 3000; // Rows of first matrix
    constexpr static int DEFAULT_K = 300;  // Columns of first matrix / Rows of second matrix
    constexpr static int DEFAULT_N = 1000; // Columns of second matrix

    inline static void add_kernel_spec_options(cxxopts::Options& options) {
        options.add_options()
            ("M", "Number of rows in first matrix", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_M)))
            ("K", "Number of columns in first matrix and rows of the second matrix", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_K)))
            ("N", "Number of columns in the second matrix", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_N)))
            ("type", "Numeric type (half, single/float, double, int<>, uint<>)", cxxopts::value<std::string>()->default_value("float"));
        ;
    }

    inline static Matrix_product_cutlass_spec make(
        const cxxopts::ParseResult& options_parsed
    ) {
        // Validate the type option
        const auto& type = options_parsed["type"].as<std::string>();
        if (type != "half" && type != "single" && type != "float" && type != "double" && type != "int8" && type != "int16" && type != "int32" && type != "int64" && type != "uint8" && type != "uint16" && type != "uint32" && type != "uint64" ) {
            std::cerr << "[ERROR] --type must be one of: half, single/float, double, int<>, uint<>" << std::endl;
            throw cxxopts::exceptions::exception("Invalid --type: " + type);
        }
        return Matrix_product_cutlass_spec(
            type,
            options_parsed["M"].as<long>(),
            options_parsed["K"].as<long>(),
            options_parsed["N"].as<long>()
        );
    }

    inline Matrix_product_cutlass_spec(
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

static_assert(Check_matrix_kernel_spec_2In_1Out<Matrix_product_cutlass_spec>::check_passed, "Matrix_product_cutlass_spec is not a valid kernel spec");

template <CUDA_scalar Number_>
class Matrix_product_cutlass_kernel {
    public:
    using Number = Number_;
    using Kernel_spec = Matrix_product_cutlass_spec;

    const Kernel_spec spec_;

    // Define CUTLASS GEMM type based on Number template parameter
    using Gemm = cutlass::gemm::device::Gemm<
        Number, cutlass::layout::RowMajor, // datatype and layout of A matrix
        Number, cutlass::layout::RowMajor, // datatype and layout of B matrix
        Number, cutlass::layout::RowMajor, // datatype and layout of C matrix
        Number // tensor core accumulator type
    >;

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
        // Scalars for GEMM operation (alpha=1.0, beta=0.0 for C = A * B)
        // The following cannot be declared constexpr as __half does not support constexpr
        // const Number alpha = Number(1);
        // const Number beta = Number(0);

        // Leading dimensions for row-major layout
        const int m = static_cast<int>(spec_.m_);
        const int k = static_cast<int>(spec_.k_);
        const int n = static_cast<int>(spec_.n_);
        const int lda = k;  // A is m x k, so leading dimension is k
        const int ldb = n;  // B is k x n, so leading dimension is n
        const int ldc = n;  // C is m x n, so leading dimension is n

        // Use the exact element types from the GEMM template
        using GemmElementA = typename Gemm::ElementA;
        using GemmElementB = typename Gemm::ElementB;
        using GemmElementC = typename Gemm::ElementC;

        const GemmElementA* const cutlass_data_A = reinterpret_cast<const GemmElementA*>(gpu_data_A);
        const GemmElementB* const cutlass_data_B = reinterpret_cast<const GemmElementB*>(gpu_data_B);
        const GemmElementC* const cutlass_data_C = reinterpret_cast<const GemmElementC*>(gpu_data_C);
        GemmElementC* const cutlass_data_D = reinterpret_cast<GemmElementC*>(gpu_data_C);

        const cutlass::gemm::GemmCoord problem_size{m, n, k};
        const cutlass::TensorRef<const GemmElementA, cutlass::layout::RowMajor> tensor_ref_A {cutlass_data_A, lda};
        const cutlass::TensorRef<const GemmElementB, cutlass::layout::RowMajor> tensor_ref_B {cutlass_data_B, ldb};
        const cutlass::TensorRef<const GemmElementC, cutlass::layout::RowMajor> tensor_ref_C {cutlass_data_C, ldc};
        cutlass::TensorRef<GemmElementC, cutlass::layout::RowMajor> tensor_ref_D {cutlass_data_D, ldc};

        // Create epilogue parameters (alpha=1.0, beta=0.0 for C = A * B)
        // typename Gemm::EpilogueOutputOp::Params epilogue_params{};

        // Construct CUTLASS GEMM arguments
        typename Gemm::Arguments args(
            problem_size, tensor_ref_A, tensor_ref_B, tensor_ref_C, tensor_ref_D
        );

        // Create the GEMM kernel object
        Gemm gemm_operator;

        // Run the CUTLASS GEMM kernel
        cutlass::Status status = gemm_operator(args);

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
static_assert(Check_matrix_kernel_2In_1Out_template<Matrix_product_cutlass_kernel>::check_passed, "Matrix_product_cutlass is not a valid kernel template");
