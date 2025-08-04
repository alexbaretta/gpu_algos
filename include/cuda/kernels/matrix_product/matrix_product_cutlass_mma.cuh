/*
    Copyright (c) 2025 Alessandro Baretta <alex@baretta.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/


// source path: include/cuda/kernels/matrix_product/matrix_product_cutlass_mma.cuh

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
#include "cuda/wmma.cuh"
#include "cutlass/half.h"
#include "cutlass/tfloat32.h"



template <typename T>
struct cutlass_mma_config;

// https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-shape
// Notice that the tile sizes for PTX MMA do not always match the tile sizes for CUDA WMMA
template <>
struct cutlass_mma_config<std::int8_t> {
    // mma.m16n16k16.s8
    using argument_type = std::int8_t;
    using operand_type = std::int8_t;
    using accumulator_type = int;
    using result_type = int;
    constexpr static unsigned M = 16;
    constexpr static unsigned N = 16;
    constexpr static unsigned K = 16;
};

template <>
struct cutlass_mma_config<std::uint8_t> {
    // mma.m16n16k16.u8
    using argument_type = uint8_t;
    using operand_type = std::uint8_t;
    using accumulator_type = int;
    using result_type = int;
    using temp_type = int;
    constexpr static unsigned M = 16;
    constexpr static unsigned N = 16;
    constexpr static unsigned K = 16;
};

template <>
struct cutlass_mma_config<__half> {
    // mma.m16n16k16.f16
    using argument_type = __half;
    using operand_type = cutlass::half_t;
    using accumulator_type = cutlass::half_t;
    using result_type = __half;
    using temp_type = __half;
    constexpr static unsigned M = 16;
    constexpr static unsigned N = 16;
    constexpr static unsigned K = 16;
};

template <>
struct cutlass_mma_config<float> {
    // mma.m16n16k8.tf32
    using argument_type = float;
    using operand_type = cutlass::tfloat32_t;
    using accumulator_type = float;
    using result_type = float;
    using temp_type = float;
    constexpr static unsigned M = 16;
    constexpr static unsigned N = 16;
    constexpr static unsigned K = 8;
};

template <>
struct cutlass_mma_config<double> {
    // mma.m16n8k16.f64
    using argument_type = double;
    using operand_type = double;
    using accumulator_type = double;
    using result_type = double;
    using temp_type = double;
    constexpr static unsigned M = 16;
    constexpr static unsigned N = 8;
    constexpr static unsigned K = 16;
};

template <typename T>
concept cutlass_mma_type = requires {
    typename cutlass_mma_config<T>::argument_type;
};

static_assert(cutlass_mma_type<std::int8_t>, "float is not recognized as a cutlass_mma_type");
static_assert(cutlass_mma_type<std::uint8_t>, "float is not recognized as a cutlass_mma_type");

static_assert(cutlass_mma_type<__half>, "float is not recognized as a cutlass_mma_type");
static_assert(cutlass_mma_type<float>, "float is not recognized as a cutlass_mma_type");
static_assert(cutlass_mma_type<double>, "float is not recognized as a cutlass_mma_type");


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
            ("type", "Numeric type (half, single/float, double, int8, uint8)", cxxopts::value<std::string>()->default_value("float"));
        ;
    }

    inline static Matrix_product_cutlass_spec make(
        const cxxopts::ParseResult& options_parsed
    ) {
        // Validate the type option
        const auto& type = options_parsed["type"].as<std::string>();
        if (type != "half" && type != "single" && type != "float" && type != "double" && type != "int8" && type != "uint8") {
            std::cerr << "[ERROR] --type must be one of: half, single/float, double, int8, uint8" << std::endl;
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
    using Mma_config = cutlass_mma_config<Number>;
    using NumberA = typename Mma_config::argument_type;
    using NumberB = typename Mma_config::argument_type;
    using NumberC = typename Mma_config::result_type;
    using NumberTemp = typename Mma_config::result_type; // Unused, but required
    using NumberOperand = typename Mma_config::operand_type;
    using NumberAccumulator = typename Mma_config::accumulator_type;
    using Kernel_spec = Matrix_product_cutlass_spec;

    const Kernel_spec spec_;

    // Define CUTLASS GEMM type based on Number template parameter
    using Gemm = cutlass::gemm::device::Gemm<
        NumberOperand, cutlass::layout::RowMajor, // datatype and layout of A matrix
        NumberOperand, cutlass::layout::RowMajor, // datatype and layout of B matrix
        NumberC, cutlass::layout::RowMajor, // datatype and layout of C matrix
        typename Mma_config::accumulator_type, // tensor core accumulator type
        cutlass::arch::OpClassTensorOp
    >;

    Matrix_product_cutlass_kernel(
        const Kernel_spec spec
    ) : spec_(spec) {}

    void run_device_kernel(
        const NumberA* const gpu_data_A,
        const NumberB* const gpu_data_B,
        NumberC* const gpu_data_C,
        NumberTemp* const gpu_data_temp,
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

    Eigen::Matrix<NumberA, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> run_host_kernel(
        const Eigen::Map<Eigen::Matrix<NumberA, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& A,
        const Eigen::Map<Eigen::Matrix<NumberB, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& B
    ) {
        return (A * B).eval();
    }

};
static_assert(Check_matrix_kernel_2In_1Out_template<Matrix_product_cutlass_kernel>::check_passed, "Matrix_product_cutlass is not a valid kernel template");
