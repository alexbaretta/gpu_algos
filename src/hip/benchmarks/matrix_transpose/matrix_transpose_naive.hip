// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: src/benchmarks/matrix_transpose/matrix_transpose_naive.hip

#include <cxxopts.hpp>

#include "common/benchmark_options.hpp"
#include "hip/benchmark/benchmark_matrix_1in_1out.hiph"

#include "hip/kernels/matrix_transpose/matrix_transpose_naive.hiph"

int main(int argc, char** argv) {
    cxxopts::Options options("matrix_transpose_naive", "Matrix transpose (naive algorithm)");
    add_benchmark_options(options);
    Matrix_transpose_naive_spec::add_kernel_spec_options(options);

    try {
        cxxopts::ParseResult options_parsed = options.parse(argc, argv);

        Matrix_transpose_naive_spec spec = Matrix_transpose_naive_spec::make(options_parsed);

        if (spec.type_ == "half") {
            return Benchmark_Matrix_1In_1Out<Matrix_transpose_naive_kernel<__half>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "single" || spec.type_ == "float") {
            return Benchmark_Matrix_1In_1Out<Matrix_transpose_naive_kernel<float>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "double") {
            return Benchmark_Matrix_1In_1Out<Matrix_transpose_naive_kernel<double>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "int8") {
            return Benchmark_Matrix_1In_1Out<Matrix_transpose_naive_kernel<std::int8_t>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "int16") {
            return Benchmark_Matrix_1In_1Out<Matrix_transpose_naive_kernel<std::int16_t>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "int32") {
            return Benchmark_Matrix_1In_1Out<Matrix_transpose_naive_kernel<std::int32_t>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "int64") {
            return Benchmark_Matrix_1In_1Out<Matrix_transpose_naive_kernel<std::int64_t>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "uint8") {
            return Benchmark_Matrix_1In_1Out<Matrix_transpose_naive_kernel<std::uint8_t>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "uint16") {
            return Benchmark_Matrix_1In_1Out<Matrix_transpose_naive_kernel<std::uint16_t>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "uint32") {
            return Benchmark_Matrix_1In_1Out<Matrix_transpose_naive_kernel<std::uint32_t>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "uint64") {
            return Benchmark_Matrix_1In_1Out<Matrix_transpose_naive_kernel<std::uint64_t>>(spec, options, options_parsed).run();
        } else {
            throw cxxopts::exceptions::exception("Invalid type: " + spec.type_);
        }
    } catch (const cxxopts::exceptions::exception& e) {
       std::cerr << "Error parsing options: " << e.what() << std::endl;
       std::cout << options.help() << std::endl;
       return 1;
    }
}
