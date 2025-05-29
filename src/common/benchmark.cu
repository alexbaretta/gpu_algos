// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: src/common/benchmark.cu

#include "common/benchmark.h"


void add_benchmark_options(cxxopts::Options& options) {
    options.add_options()
        ("i,init-method", "How to generate data: random, sequential", cxxopts::value<std::string>()->default_value("random"))
        ("gpumem", "GPU memory size", cxxopts::value<int>()->default_value(std::to_string(DEFAULT_GPU_MEM)))
        ("seed", "Random seed", cxxopts::value<int>()->default_value(std::to_string(DEFAULT_SEED)))
        ("verbose", "Verbose output", cxxopts::value<bool>()->default_value("false"))
        ("h,help", "Print usage");
}
