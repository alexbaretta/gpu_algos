// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/hip/kernels/matrix/matrix_product_warp.hip.h

#pragma once
#include <hip/hip_runtime.h>

#include "hip/kernel_api.hip.hpp
#include "hip/type_traits.hip.hpp

/*
    This kernel uses a "one wavefront per result element" threading strategy.
    Conceptually, we want to allocate one wavefront per result element, and each wavefront
    will compute the result element for a single row of the left matrix and a single
    column of the right matrix.

    The grid is then defined by the Number of result elements, which is the product of
    the Number of rows of the left matrix and the Number of columns of the right matrix.

    The threads in each wavefront collaborate to compute the result element for a single row
    of the left matrix and a single column of the right matrix: each thread processes
    the products whose index mod WAVEFRONT_SIZE is equal to the thread's index within the wavefront.

    Wavefront-reduction is then used to compute the sum of the partial results computed by
    the threads in the wavefront.
*/

struct Matrix_product_warp_spec {
    constexpr static int WAVEFRONT_SIZE = 64;  // AMD wavefront size (different from NVIDIA warp size)
    constexpr static int TILE_SIZE = 4;
    constexpr static int DEFAULT_M = 3000; // Rows of first matrix
    constexpr static int DEFAULT_N = 300;  // Columns of first matrix / Rows of second matrix
    constexpr static int DEFAULT_K = 1000; // Columns of second matrix

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

    const long wavefront_size_;
    const dim3 block_dim_;
    const dim3 grid_dim_;
    const size_t dynamic_shared_mem_words_ = 0;

    inline static void add_kernel_spec_options(cxxopts::Options& options) {
        options.add_options()
            ("m", "Number of rows in first matrix", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_M)))
            ("n", "Number of columns in first matrix and rows of the second matrix", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_N)))
            ("k", "Number of columns in the second matrix", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_K)))
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
            options_parsed["m"].as<long>(),
            options_parsed["n"].as<long>(),
            options_parsed["k"].as<long>()
        );
    }

    inline Matrix_product_warp_spec(
        const std::string& type,
        const long m,
        const long n,
        const long k
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
        wavefront_size_(WAVEFRONT_SIZE),
        block_dim_(wavefront_size_, TILE_SIZE, TILE_SIZE),
        grid_dim_(wavefront_size_, (k_ + TILE_SIZE - 1) / TILE_SIZE, (m_ + TILE_SIZE - 1) / TILE_SIZE)
    {}
};

static_assert(Check_kernel_spec_2In_1Out<Matrix_product_warp_spec>::check_passed, "Matrix_product_warp_spec is not a valid kernel spec");

template <HIP_scalar HIP_Number>
__global__ void matrix_product_warp(
    const HIP_Number* A,
    const HIP_Number* B,
    HIP_Number* C,
    const long m,
    const long n, // shared dimension
    const long k
) {
    // thread id is (x + y Dx + z Dx Dy, see https://rocm.docs.amd.com/en/latest/understand/gpu_arch/mi200.html#wavefront-execution
    // int thread_id = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    // but blockDim.x is 64, so the lane id is just threadIdx.x
    int thread_id = threadIdx.x;
    int lane = thread_id % 64; // AMD wavefront size
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    int row = threadIdx.z + blockIdx.z * blockDim.z;

    HIP_Number sum = static_cast<HIP_Number>(0);

    // Compute the partial sum for the current thread
    for (int i = lane; i < n; i += 64) { // AMD wavefront size
        sum += A[row * n + i] * B[i * k + col];
    }

    // Reduce the partial sum using wavefront-reduction
    // HIP provides __shfl_down but it works within a 32-thread group on AMD
    // For 64-thread wavefronts, we need to handle this differently
    __shared__ HIP_Number shared_mem[64]; // One per wavefront lane

    shared_mem[lane] = sum;
    __syncthreads();

    // Manual reduction within wavefront
    for (int offset = 32; offset > 0; offset >>= 1) {
        if (lane < offset) {
            shared_mem[lane] += shared_mem[lane + offset];
        }
        __syncthreads();
    }

    // Store the result
    if (lane == 0) {
        C[row * k + col] = shared_mem[0];
    }
}

template <HIP_scalar Number_>
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
        hipStream_t stream
    ) {
        hipLaunchKernelGGL(
            matrix_product_warp<Number>,
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
static_assert(Check_kernel_2In_1Out_template<Matrix_product_warp_kernel>::check_passed, "Matrix_product_warp is not a valid kernel template");
