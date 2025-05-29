// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: src/tools/vector_cumsum/vector_cumsum_parallel.cu

#include <cxxopts.hpp>

#include "common/benchmark.h"

#include "cuda/kernels/vector_cumsum/vector_cumsum_parallel.h"

int main(int argc, char** argv) {
    cxxopts::Options options("vector_cumsum_parallel", "Naive matrix multiplication");
    add_benchmark_options(options);
    Vector_cumsum_parallel_spec::add_kernel_spec_options(options);

    try {
        cxxopts::ParseResult options_parsed = options.parse(argc, argv);

        Vector_cumsum_parallel_spec spec = Vector_cumsum_parallel_spec::make(options_parsed);

        if (spec.type_ == "half") {
            return Benchmark_1In_1Out<Vector_cumsum_parallel_kernel<__half>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "single" || spec.type_ == "float") {
            return Benchmark_1In_1Out<Vector_cumsum_parallel_kernel<float>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "double") {
            return Benchmark_1In_1Out<Vector_cumsum_parallel_kernel<double>>(spec, options, options_parsed).run();
        }
    } catch (const cxxopts::exceptions::exception& e) {
       std::cerr << "Error parsing options: " << e.what() << std::endl;
       std::cout << options.help() << std::endl;
       return 1;
    }
}
