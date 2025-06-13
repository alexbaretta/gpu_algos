// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/hip/kernels/matrix/vector_cumsum_serial.hpp

#pragma once
#include <iostream>
#include <hip/hip_runtime.h>
#include <cxxopts.hpp>
#include <Eigen/Dense>

#include "hip/kernel_api/vector_1in_1out.hip.hpp"
#include "hip/type_traits.hip.hpp"

template <HIP_scalar HIP_Number>
__global__ void vector_cumsum_serial(
    const HIP_Number* A,
    HIP_Number* C,
    const long n  // size of vector
) {

    HIP_Number sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += A[i];
        C[i] = sum;
    }
}

struct Vector_cumsum_serial_spec {
    const std::string type_;

    const long m_;    // unused for vector cumsum
    const long n_;    // size of vector
    const long k_;    // unused for vector cumsum

    const long n_A_;
    const long n_C_;
    const long n_temp_;

    const dim3 block_dim_;
    const dim3 grid_dim_;
    const size_t dynamic_shared_mem_words_ = 0;

    constexpr static int DEFAULT_M = 0;    // unused
    constexpr static int DEFAULT_N = 3000; // size of vector
    constexpr static int DEFAULT_K = 0;    // unused
    constexpr static int DEFAULT_BLOCK_DIM_X = 32;

    inline static void add_kernel_spec_options(cxxopts::Options& options) {
        options.add_options()
            ("n", "Size of vector", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_N)))
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
            options_parsed["n"].as<long>()
        );
    }

    inline Vector_cumsum_serial_spec(
        const std::string& type,
        const long n
    ) : type_(type),
        m_(0),  // unused
        n_(n),
        k_(0),  // unused
        n_A_(n),
        n_C_(n),
        n_temp_(0),
        block_dim_(1),
        grid_dim_(1)
    {}
};

static_assert(Check_vector_kernel_spec_1In_1Out<Vector_cumsum_serial_spec>::check_passed, "Vector_cumsum_serial_spec is not a valid kernel spec");


template <HIP_scalar Number_>
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
        hipStream_t stream
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
        for (int i = 1; i < A.size(); ++i) {
            accu += A(i);
            result(i) = accu;
        }
        return result;
    }

};
static_assert(Check_vector_kernel_1In_1Out_template<Vector_cumsum_serial_kernel>::check_passed, "Vector_cumsum_serial is not a valid kernel template");
