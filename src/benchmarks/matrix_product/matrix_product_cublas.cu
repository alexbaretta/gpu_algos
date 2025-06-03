// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: src/benchmarks/matrix_product/matrix_product_cublas.cu

#include <cxxopts.hpp>

#include "common/benchmark.h"

#include "cuda/kernels/matrix_product/matrix_product_cublas.h"

int main(int argc, char** argv) {
    cxxopts::Options options("matrix_product_cublas", "Matrix multiplication using cuBLAS");
    add_benchmark_options(options);
    Matrix_product_cublas_spec::add_kernel_spec_options(options);

    try {
        cxxopts::ParseResult options_parsed = options.parse(argc, argv);

        Matrix_product_cublas_spec spec = Matrix_product_cublas_spec::make(options_parsed);

        if (spec.type_ == "half") {
            return Benchmark_2In_1Out<Matrix_product_cublas_kernel<__half>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "single" || spec.type_ == "float") {
            return Benchmark_2In_1Out<Matrix_product_cublas_kernel<float>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "double") {
            return Benchmark_2In_1Out<Matrix_product_cublas_kernel<double>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "int8") {
            return Benchmark_2In_1Out<Matrix_product_cublas_kernel<std::int8_t>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "int16") {
            return Benchmark_2In_1Out<Matrix_product_cublas_kernel<std::int16_t>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "int32") {
            return Benchmark_2In_1Out<Matrix_product_cublas_kernel<std::int32_t>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "int64") {
            return Benchmark_2In_1Out<Matrix_product_cublas_kernel<std::int64_t>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "uint8") {
            return Benchmark_2In_1Out<Matrix_product_cublas_kernel<std::uint8_t>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "uint16") {
            return Benchmark_2In_1Out<Matrix_product_cublas_kernel<std::uint16_t>>(spec, options, options_parsed).run();
        } else {
            throw cxxopts::exceptions::exception("Invalid type: " + spec.type_);
        }
    } catch (const cxxopts::exceptions::exception& e) {
       std::cerr << "Error parsing options: " << e.what() << std::endl;
       std::cout << options.help() << std::endl;
       return 1;
    }
}
