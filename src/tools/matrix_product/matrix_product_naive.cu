// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

#include <cxxopts.hpp>

#include "common/benchmark.h"

#include "cuda/kernels/matrix_product.h"


// Default matrix dimensions
constexpr int DEFAULT_NROWS = 3000; // Rows of first matrix
constexpr int DEFAULT_NCOLS = 300;  // Columns of first matrix / Rows of second matrix
constexpr int DEFAULT_GPU_MEM = 16; // GPU memory size in GB
constexpr int DEFAULT_SEED = 42;

int main(int argc, char** argv) {
    // Parse command line arguments
    cxxopts::Options options("matrix_multiply", "CUDA Matrix Multiplication");
    options.add_options()
        ("nrows", "Number of rows in first matrix", cxxopts::value<int>()->default_value(std::to_string(DEFAULT_NROWS)))
        ("ncols", "Number of columns in first matrix", cxxopts::value<int>()->default_value(std::to_string(DEFAULT_NCOLS)))
        ("gpumem", "GPU memory size", cxxopts::value<int>()->default_value(std::to_string(DEFAULT_GPU_MEM)))
        ("seed", "Random seed", cxxopts::value<int>()->default_value(std::to_string(DEFAULT_SEED)))
        ("verbose", "Verbose output", cxxopts::value<bool>()->default_value("false"))
        ("h,help", "Print usage");

    try {
        auto result = options.parse(argc, argv);

        if (result.count("help")) {
            std::cout << options.help() << std::endl;
            return 0;
        }

        // Parse CLI options
        const int nrows = result["nrows"].as<int>();
        const int ncols = result["ncols"].as<int>();
        const int seed = result["seed"].as<int>();
        const int gpu_mem = result["gpumem"].as<int>();
        const bool verbose = result["verbose"].as<bool>();

        Benchmark<Matrix_product_naive<float>> benchmark(nrows, ncols, seed, gpu_mem, verbose);
        return benchmark.run();
    } catch (const cxxopts::exceptions::exception& e) {
       std::cerr << "Error parsing options: " << e.what() << std::endl;
       std::cout << options.help() << std::endl;
       return 1;
    }
}
