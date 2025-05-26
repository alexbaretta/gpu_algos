// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: src/tools/matrix_product/matrix_product_naive.cu

#include <cxxopts.hpp>

#include "common/benchmark.h"

#include "cuda/kernels/matrix_product/matrix_product_naive.h"

int main(int argc, char** argv) {
    cxxopts::Options options("matrix_product_naive", "Naive matrix multiplication");
    add_benchmark_options(options);
    Matrix_product_naive_spec::add_kernel_spec_options(options);

    try {
        cxxopts::ParseResult options_parsed = options.parse(argc, argv);

        Matrix_product_naive_spec spec = Matrix_product_naive_spec::make(options_parsed);

        if (spec.type_ == "half") {
            return Benchmark_2In_1Out<Matrix_product_naive_kernel<__half>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "single" || spec.type_ == "float") {
            return Benchmark_2In_1Out<Matrix_product_naive_kernel<float>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "double") {
            return Benchmark_2In_1Out<Matrix_product_naive_kernel<double>>(spec, options, options_parsed).run();
        }
    } catch (const cxxopts::exceptions::exception& e) {
       std::cerr << "Error parsing options: " << e.what() << std::endl;
       std::cout << options.help() << std::endl;
       return 1;
    }
}
