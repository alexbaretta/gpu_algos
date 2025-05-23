// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

#include "common/benchmark.h"

constexpr int DEFAULT_NROWS = 3000; // Rows of first matrix
constexpr int DEFAULT_NCOLS = 300;  // Columns of first matrix / Rows of second matrix
constexpr int DEFAULT_GPU_MEM = 16; // GPU memory size in GB
constexpr int DEFAULT_SEED = 42;

void add_benchmark_options(cxxopts::Options& options) {
    options.add_options()
        ("nrows", "Number of rows in first matrix", cxxopts::value<int>()->default_value(std::to_string(DEFAULT_NROWS)))
        ("ncols", "Number of columns in first matrix", cxxopts::value<int>()->default_value(std::to_string(DEFAULT_NCOLS)))
        ("gpumem", "GPU memory size", cxxopts::value<int>()->default_value(std::to_string(DEFAULT_GPU_MEM)))
        ("seed", "Random seed", cxxopts::value<int>()->default_value(std::to_string(DEFAULT_SEED)))
        ("verbose", "Verbose output", cxxopts::value<bool>()->default_value("false"))
        ("h,help", "Print usage");
}
