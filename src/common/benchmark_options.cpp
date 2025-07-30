// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: src/common/benchmark_options.cpp

#include "common/benchmark_options.hpp"

void add_benchmark_options(cxxopts::Options& options) {
    options.add_options()
        ("i,init-method", "How to generate data: random, increasing, decreasing", cxxopts::value<std::string>()->default_value("random"))
        ("gpumem", "GPU memory size", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_GPU_MEM)))
        ("seed", "Random seed", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_SEED)))
        ("tol-bits", "Number of bits of precision lost due to arithmetic rounding", cxxopts::value<int>()->default_value(std::to_string(DEFAULT_TOL_BITS)))
        ("errors", "Display errors", cxxopts::value<bool>()->default_value("false"))
        ("verbose", "Verbose output", cxxopts::value<bool>()->default_value("false"))
        ("force,f", "Allow verbose output even if the input matrix is large", cxxopts::value<bool>()->default_value("false"))
        ("h,help", "Print usage");
}
