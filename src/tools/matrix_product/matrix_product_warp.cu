// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: src/tools/matrix_product/matrix_product_warp.cu

#include <cxxopts.hpp>

#include "common/benchmark.h"

#include "cuda/kernels/matrix_product/matrix_product_warp.h"

int main(int argc, char** argv) {
    cxxopts::Options options("matrix_product_warp", "warp matrix multiplication");
    add_benchmark_options(options);
    Matrix_product_warp_spec::add_kernel_spec_options(options);

    try {
        cxxopts::ParseResult options_parsed = options.parse(argc, argv);

        Matrix_product_warp_spec spec = Matrix_product_warp_spec::make(options_parsed);

        if (spec.type_ == "half") {
            return Benchmark_2In_1Out<Matrix_product_warp_kernel<__half>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "single" || spec.type_ == "float") {
            return Benchmark_2In_1Out<Matrix_product_warp_kernel<float>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "double") {
            return Benchmark_2In_1Out<Matrix_product_warp_kernel<double>>(spec, options, options_parsed).run();
        }
    } catch (const cxxopts::exceptions::exception& e) {
       std::cerr << "Error parsing options: " << e.what() << std::endl;
       std::cout << options.help() << std::endl;
       return 1;
    }
}
