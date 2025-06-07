// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/hip/kernels/vector_scan_generic/vector_scan_parallel.hip.hpp

#pragma once
#include <iostream>
#include <hip/hip_runtime.h>
#include <cxxopts.hpp>
#include <Eigen/Dense>

#include "hip/hip_utils.hip.hpp"

// Simple Vector scan spec without CUDA dependencies
struct Vector_scan_parallel_spec {
    const std::string type_;
    const std::string operation_;
    const long n_;    // size of vector
    const dim3 block_dim_;
    const dim3 grid_dim_;

    constexpr static int DEFAULT_N = 3000;
    constexpr static int DEFAULT_BLOCK_DIM_X = 1024;

    inline static void add_kernel_spec_options(cxxopts::Options& options) {
        options.add_options()
            ("operation,op", "Operation to perform (max, min, sum, prod)", cxxopts::value<std::string>()->default_value("sum"))
            ("n", "Size of vector", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_N)))
            ("block_dim,x", "Number of threads in the x dimension per block", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_BLOCK_DIM_X)))
            ("type", "Numeric type (half, single/float, double, int<n>, uint<n>)", cxxopts::value<std::string>()->default_value("float"));
        ;
    }

    inline static Vector_scan_parallel_spec make(const cxxopts::ParseResult& options_parsed) {
        const auto& type = options_parsed["type"].as<std::string>();
        const auto& operation = options_parsed["operation"].as<std::string>();
        const auto n = options_parsed["n"].as<long>();
        const auto block_dim = options_parsed["block_dim"].as<long>();
        return Vector_scan_parallel_spec(type, operation, n, block_dim);
    }

    Vector_scan_parallel_spec(const std::string& type, const std::string& operation, const long n, const long block_size)
        : type_(type), operation_(operation), n_(n), block_dim_(block_size), grid_dim_((n + block_size - 1) / block_size) {}
};

// Simple kernel placeholder - actual implementation would go here
template <typename Number_, typename Operation_>
class Vector_scan_parallel_kernel {
    public:
    using Number = Number_;
    using Operation = Operation_;
    using Kernel_spec = Vector_scan_parallel_spec;

    const Kernel_spec spec_;

    Vector_scan_parallel_kernel(const Kernel_spec spec) : spec_(spec) {}

    // Placeholder implementation
    void run_device_kernel(const Number* A, Number* C, Number* temp, hipStream_t stream) {
        // TODO: Implement actual HIP kernel
    }

    Eigen::Vector<Number, Eigen::Dynamic> run_host_kernel(const Eigen::Vector<Number, Eigen::Dynamic>& A) {
        Eigen::Vector<Number, Eigen::Dynamic> C(A.size());
        if (A.size() == 0) return C;
        C[0] = A[0];
        for (int i = 1; i < A.size(); ++i) {
            C[i] = Operation::apply(C[i-1], A[i]);
        }
        return C;
    }
};
