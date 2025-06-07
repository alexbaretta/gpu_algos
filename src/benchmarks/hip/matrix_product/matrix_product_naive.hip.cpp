// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: src/benchmarks/hip/matrix_product/matrix_product_naive.cu

#include <cxxopts.hpp>

#include "common/benchmark_options.hpp"

#include "hip/benchmark.hip.hpp"

#include "hip/kernels/matrix_product/matrix_product_naive.hip.hpp"

int main(int argc, char** argv) {
    cxxopts::Options options("matrix_product_naive", "Matrix multiplication (naive algorithm)");
    add_benchmark_options(options);
    Matrix_product_naive_spec::add_kernel_spec_options(options);

    try {
        cxxopts::ParseResult options_parsed = options.parse(argc, argv);

        Matrix_product_naive_spec spec = Matrix_product_naive_spec::make(options_parsed);

        if (spec.type_ == "half") {
            return Benchmark_2In_1Out<Matrix_product_naive_kernel<_Float16>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "single" || spec.type_ == "float") {
            return Benchmark_2In_1Out<Matrix_product_naive_kernel<float>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "double") {
            return Benchmark_2In_1Out<Matrix_product_naive_kernel<double>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "int8") {
            return Benchmark_2In_1Out<Matrix_product_naive_kernel<std::int8_t>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "int16") {
            return Benchmark_2In_1Out<Matrix_product_naive_kernel<std::int16_t>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "int32") {
            return Benchmark_2In_1Out<Matrix_product_naive_kernel<std::int32_t>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "int64") {
            return Benchmark_2In_1Out<Matrix_product_naive_kernel<std::int64_t>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "uint8") {
            return Benchmark_2In_1Out<Matrix_product_naive_kernel<std::uint8_t>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "uint16") {
            return Benchmark_2In_1Out<Matrix_product_naive_kernel<std::uint16_t>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "uint32") {
            return Benchmark_2In_1Out<Matrix_product_naive_kernel<std::uint32_t>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "uint64") {
            return Benchmark_2In_1Out<Matrix_product_naive_kernel<std::uint64_t>>(spec, options, options_parsed).run();
        } else {
            throw cxxopts::exceptions::exception("Invalid type: " + spec.type_);
        }
    } catch (const cxxopts::exceptions::exception& e) {
       std::cerr << "Error parsing options: " << e.what() << std::endl;
       std::cout << options.help() << std::endl;
       return 1;
    }
}
