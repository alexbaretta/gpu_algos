// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/common/benchmark_options.hpp

#pragma once

#include <cxxopts.hpp>

constexpr int DEFAULT_GPU_MEM = 16; // GPU memory size in GB
constexpr int DEFAULT_SEED = 42;
constexpr int DEFAULT_TOL_BITS = 4;

void add_benchmark_options(cxxopts::Options& options);
