// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

#include <cxxopts.hpp>

#include "common/benchmark.h"

#include "cuda/kernels/matrix_product.h"

int main(int argc, char** argv) {
    cxxopts::Options options("matrix_product_naive", "Naive matrix multiplication");
    add_benchmark_options(options);

    try {
        Benchmark<Matrix_product_naive<float>> benchmark(options, argc, argv);
        return benchmark.run();
    } catch (const cxxopts::exceptions::exception& e) {
       std::cerr << "Error parsing options: " << e.what() << std::endl;
       std::cout << options.help() << std::endl;
       return 1;
    }
}
