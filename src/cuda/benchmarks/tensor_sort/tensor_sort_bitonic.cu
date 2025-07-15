// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: src/benchmarks/cuda/tensor_sort/tensor_sort_bitonic.cu

#include <cxxopts.hpp>

#include "common/benchmark_options.hpp"
#include "cuda/benchmark/benchmark_tensor3d_1inout.cuh"

#include "cuda/kernels/sort/tensor_sort_bitonic.cuh"

int main(int argc, char** argv) {
    cxxopts::Options options("tensor_sort_bitonic", "Tensor sort (bitonic algorithm)");
    add_benchmark_options(options);
    tensor3d_sort_bitonic_spec::add_kernel_spec_options(options);

    try {
        cxxopts::ParseResult options_parsed = options.parse(argc, argv);

        tensor3d_sort_bitonic_spec spec = tensor3d_sort_bitonic_spec::make(options_parsed);

        if (spec.type_ == "half") {
            return Benchmark_Tensor3D_1Inout<tensor3d_sort_bitonic_kernel<__half>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "single" || spec.type_ == "float") {
            return Benchmark_Tensor3D_1Inout<tensor3d_sort_bitonic_kernel<float>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "double") {
            return Benchmark_Tensor3D_1Inout<tensor3d_sort_bitonic_kernel<double>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "int8") {
            return Benchmark_Tensor3D_1Inout<tensor3d_sort_bitonic_kernel<std::int8_t>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "int16") {
            return Benchmark_Tensor3D_1Inout<tensor3d_sort_bitonic_kernel<std::int16_t>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "int32") {
            return Benchmark_Tensor3D_1Inout<tensor3d_sort_bitonic_kernel<std::int32_t>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "int64") {
            return Benchmark_Tensor3D_1Inout<tensor3d_sort_bitonic_kernel<std::int64_t>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "uint8") {
            return Benchmark_Tensor3D_1Inout<tensor3d_sort_bitonic_kernel<std::uint8_t>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "uint16") {
            return Benchmark_Tensor3D_1Inout<tensor3d_sort_bitonic_kernel<std::uint16_t>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "uint32") {
            return Benchmark_Tensor3D_1Inout<tensor3d_sort_bitonic_kernel<std::uint32_t>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "uint64") {
            return Benchmark_Tensor3D_1Inout<tensor3d_sort_bitonic_kernel<std::uint64_t>>(spec, options, options_parsed).run();
        } else {
            throw cxxopts::exceptions::exception("Invalid type: " + spec.type_);
        }
    } catch (const cxxopts::exceptions::exception& e) {
       std::cerr << "Error parsing options: " << e.what() << std::endl;
       std::cout << options.help() << std::endl;
       return 1;
    }
}
