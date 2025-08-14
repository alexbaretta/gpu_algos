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


// source path: include/cuda/kernels/vector_cumsum/vector_cumsum_serial.cuh

#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include <cxxopts.hpp>
#include <Eigen/Dense>

#include "cuda/kernel_api/vector_1in_1out.cuh"
#include "cuda/type_traits.cuh"

template <CUDA_scalar CUDA_Number>
__global__ void vector_cumsum_serial(
    const CUDA_Number* A,
    CUDA_Number* C,
    const long n  // size of vector
) {

    CUDA_Number sum = 0.0f;
    for (long i = 0; i < n; ++i) {
        sum += A[i];
        C[i] = sum;
    }
}

struct Vector_cumsum_serial_spec {
    const std::string type_;

    const long n_;    // size of vector

    const long n_A_;
    const long n_C_;
    const long n_temp_;

    const dim3 block_dim_;
    const dim3 grid_dim_;
    const size_t dynamic_shared_mem_words_ = 0;

    constexpr static int DEFAULT_N = 3000; // size of vector
    constexpr static int DEFAULT_BLOCK_DIM_X = 32;

    inline static void add_kernel_spec_options(cxxopts::Options& options) {
        options.add_options()
            ("N", "Size of vector", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_N)))
            ("type", "Numeric type (half, single/float, double, int<n>, uint<n>)", cxxopts::value<std::string>()->default_value("float"));
        ;
    }

    inline static Vector_cumsum_serial_spec make(
        const cxxopts::ParseResult& options_parsed
    ) {
        // Validate the type option
        const auto& type = options_parsed["type"].as<std::string>();
        if (type != "half" && type != "single" && type != "float" && type != "double" && type != "int8" && type != "int16" && type != "int32" && type != "int64" && type != "uint8" && type != "uint16" && type != "uint32" && type != "uint64") {
            std::cerr << "[ERROR] --type must be one of: half, single/float, double, int<n>, uint<n>" << std::endl;
            throw cxxopts::exceptions::exception("Invalid --type: " + type);
        }
        return Vector_cumsum_serial_spec(
            type,
            options_parsed["N"].as<long>()
        );
    }

    inline Vector_cumsum_serial_spec(
        const std::string& type,
        const long size
    ) : type_(type),
        n_(size),
        n_A_(size),
        n_C_(size),
        n_temp_(0),
        block_dim_(1),
        grid_dim_(1)
    {}
};

static_assert(Check_vector_kernel_spec_1In_1Out<Vector_cumsum_serial_spec>::check_passed, "Vector_cumsum_serial_spec is not a valid kernel spec");


template <CUDA_scalar Number_>
class Vector_cumsum_serial_kernel {
    public:
    using Number = Number_;
    using Kernel_spec = Vector_cumsum_serial_spec;

    const Kernel_spec spec_;

    Vector_cumsum_serial_kernel(
        const Kernel_spec spec
    ) : spec_(spec) {}

    void run_device_kernel(
        const Number* const gpu_data_A,
        Number* const gpu_data_C,
        Number* const gpu_data_temp,
        cudaStream_t stream
    ) {
        vector_cumsum_serial<<<
            spec_.grid_dim_,
            spec_.block_dim_,
            spec_.dynamic_shared_mem_words_ * sizeof(Number),
            stream
        >>>(gpu_data_A, gpu_data_C, spec_.n_);
    }

    Eigen::Vector<Number, Eigen::Dynamic> run_host_kernel(
        const Eigen::Map<Eigen::Vector<Number, Eigen::Dynamic>>& A
    ) {
        // Compute cumulative sum for a vector (treat matrix as a flattened vector)
        Eigen::Vector<Number, Eigen::Dynamic> result(A.rows(), A.cols());
        Number accu = A(0);
        result(0) = accu;
        for (long i = 1; i < A.size(); ++i) {
            accu += A(i);
            result(i) = accu;
        }
        return result;
    }

};
static_assert(Check_vector_kernel_1In_1Out_template<Vector_cumsum_serial_kernel>::check_passed, "Vector_cumsum_serial is not a valid kernel template");
