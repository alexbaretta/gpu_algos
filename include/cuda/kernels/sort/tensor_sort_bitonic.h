// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/cuda/kernels/sort/tensor_sort_bitonic.h

#pragma once
#include <cuda_runtime.h>
#include <string>
#include <iostream>
#include <cxxopts.hpp>
#include <Eigen/Dense>

#include "common/kernel_api/tensor3d_1inout.h"
#include "cuda/type_traits.h"

template <CUDA_scalar CUDA_Number>
__global__ void tensor_sort_bitonic(
    CUDA_Number* data,
    const long n_rows,
    const long n_cols,
    const long n_sheets
) {
    // TODO: Implement bitonic sort algorithm
}

struct tensor_sort_bitonic_spec {
    const std::string type_;

    // Input/output tensor dimensions - these correspond to n_rows_C_, n_cols_C_, n_sheets_C_
    const long n_rows_C_;
    const long n_cols_C_;
    const long n_sheets_C_;

    // Additional members expected by the benchmark interface
    const long n_rows_A_;
    const long n_cols_A_;
    const long n_sheets_A_;

    const long n_rows_temp_;
    const long n_cols_temp_;
    const long n_sheets_temp_;

    const dim3 block_dim_;
    const dim3 grid_dim_;
    const size_t dynamic_shared_mem_words_;

    constexpr static int DEFAULT_M = 3000;
    constexpr static int DEFAULT_N = 1;
    constexpr static int DEFAULT_K = 1;
    constexpr static int DEFAULT_BLOCK_DIM_X = 16;
    constexpr static int DEFAULT_BLOCK_DIM_Y = 16;

    inline static void add_kernel_spec_options(cxxopts::Options& options) {
        options.add_options()
            ("m", "Number of rows", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_M)))
            ("n", "Number of columns", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_N)))
            ("k", "Number of sheets", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_K)))
            ("block_dim_x,x", "Number of threads in the x dimension per block", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_BLOCK_DIM_X)))
            ("block_dim_y,y", "Number of threads in the y dimension per block", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_BLOCK_DIM_Y)))
            ("type", "Numeric type (half, single/float, double, int<n>, uint<n>)", cxxopts::value<std::string>()->default_value("float"));
        ;
    }

    inline static tensor_sort_bitonic_spec make(
        const cxxopts::ParseResult& options_parsed
    ) {
        // Validate the type option
        const auto& type = options_parsed["type"].as<std::string>();
        if (type != "half" && type != "single" && type != "float" && type != "double" && type != "int8" && type != "int16" && type != "int32" && type != "int64" && type != "uint8" && type != "uint16" && type != "uint32" && type != "uint64") {
            std::cerr << "[ERROR] --type must be one of: half, single/float, double, int<n>, uint<n>" << std::endl;
            throw cxxopts::exceptions::exception("Invalid --type: " + type);
        }
        return tensor_sort_bitonic_spec(
            type,
            options_parsed["m"].as<long>(),
            options_parsed["n"].as<long>(),
            options_parsed["k"].as<long>(),
            options_parsed["block_dim_x"].as<long>(),
            options_parsed["block_dim_y"].as<long>()
        );
    }

    inline tensor_sort_bitonic_spec(
        const std::string& type,
        const long m,
        const long n,
        const long k,
        const long block_dim_x,
        const long block_dim_y
    ) : type_(type),
        n_rows_C_(m),
        n_cols_C_(n),
        n_sheets_C_(k),
        n_rows_A_(m),  // For this kernel, input == output dimensions
        n_cols_A_(n),
        n_sheets_A_(k),
        n_rows_temp_(0),  // No temporary storage needed for bitonic sort
        n_cols_temp_(0),
        n_sheets_temp_(0),
        block_dim_(block_dim_x, block_dim_y, 1),
        grid_dim_(
            (n + block_dim_.x - 1) / block_dim_.x,
            (m + block_dim_.y - 1) / block_dim_.y,
            (k + block_dim_.z - 1) / block_dim_.z
        ),
        dynamic_shared_mem_words_(0)
    {}
};

static_assert(Check_tensor3d_kernel_spec_1InOut<tensor_sort_bitonic_spec>::check_passed, "tensor_sort_bitonic_spec is not a valid tensor3d kernel spec");

template <CUDA_scalar Number_>
class tensor_sort_bitonic_kernel {
    public:
    using Number = Number_;
    using Kernel_spec = tensor_sort_bitonic_spec;

    const Kernel_spec spec_;

    tensor_sort_bitonic_kernel(
        const Kernel_spec spec
    ) : spec_(spec) {}

    void run_device_kernel(
        Number* const gpu_data_A,
        Number* const gpu_data_temp,
        cudaStream_t stream
    ) {
        tensor_sort_bitonic<<<
            spec_.grid_dim_,
            spec_.block_dim_,
            spec_.dynamic_shared_mem_words_ * sizeof(Number),
            stream
        >>>(gpu_data_A, spec_.n_rows_C_, spec_.n_cols_C_, spec_.n_sheets_C_);
    }

    void run_host_kernel(
        Tensor3D<Number>& tensor
    ) {
        // No-op for now as requested
        // TODO: Implement host reference implementation
    }
};

static_assert(Check_tensor3d_kernel_1InOut<tensor_sort_bitonic_kernel<float>>::check_passed, "tensor_sort_bitonic_kernel is not a valid tensor3d kernel");
static_assert(Check_tensor3d_kernel_1InOut_template<tensor_sort_bitonic_kernel>::check_passed, "tensor_sort_bitonic_kernel is not a valid tensor3d kernel template");
