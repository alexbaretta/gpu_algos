// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: src/tools/vector_scan_generic/vector_scan_parallel.cu

#include <cxxopts.hpp>

#include "common/benchmark.h"

#include "cuda/kernels/vector_scan_generic/vector_scan_parallel.h"

int main(int argc, char** argv) {
    cxxopts::Options options("vector_scan_parallel", "Vector cumulative sum (parallel algorithm)");
    add_benchmark_options(options);
    Vector_scan_parallel_spec::add_kernel_spec_options(options);

    try {
        cxxopts::ParseResult options_parsed = options.parse(argc, argv);

        Vector_scan_parallel_spec spec = Vector_scan_parallel_spec::make(options_parsed);

        if (spec.type_ == "half") {
            if (spec.operation_ == "max") {
                return Benchmark_1In_1Out<Vector_scan_parallel_kernel<__half, cuda_max_op<__half>>>(spec, options, options_parsed).run();
            } else if (spec.operation_ == "min") {
                return Benchmark_1In_1Out<Vector_scan_parallel_kernel<__half, cuda_min_op<__half>>>(spec, options, options_parsed).run();
            } else if (spec.operation_ == "sum") {
                return Benchmark_1In_1Out<Vector_scan_parallel_kernel<__half, cuda_sum_op<__half>>>(spec, options, options_parsed).run();
            } else if (spec.operation_ == "prod") {
                return Benchmark_1In_1Out<Vector_scan_parallel_kernel<__half, cuda_prod_op<__half>>>(spec, options, options_parsed).run();
            }
        } else if (spec.type_ == "single" || spec.type_ == "float") {
            if (spec.operation_ == "max") {
                return Benchmark_1In_1Out<Vector_scan_parallel_kernel<float, cuda_max_op<float>>>(spec, options, options_parsed).run();
            } else if (spec.operation_ == "min") {
                return Benchmark_1In_1Out<Vector_scan_parallel_kernel<float, cuda_min_op<float>>>(spec, options, options_parsed).run();
            } else if (spec.operation_ == "sum") {
                return Benchmark_1In_1Out<Vector_scan_parallel_kernel<float, cuda_sum_op<float>>>(spec, options, options_parsed).run();
            } else if (spec.operation_ == "prod") {
                return Benchmark_1In_1Out<Vector_scan_parallel_kernel<float, cuda_prod_op<float>>>(spec, options, options_parsed).run();
            }
        } else if (spec.type_ == "double") {
            if (spec.operation_ == "max") {
                return Benchmark_1In_1Out<Vector_scan_parallel_kernel<double, cuda_max_op<double>>>(spec, options, options_parsed).run();
            } else if (spec.operation_ == "min") {
                return Benchmark_1In_1Out<Vector_scan_parallel_kernel<double, cuda_min_op<double>>>(spec, options, options_parsed).run();
            } else if (spec.operation_ == "sum") {
                return Benchmark_1In_1Out<Vector_scan_parallel_kernel<double, cuda_sum_op<double>>>(spec, options, options_parsed).run();
            } else if (spec.operation_ == "prod") {
                return Benchmark_1In_1Out<Vector_scan_parallel_kernel<double, cuda_prod_op<double>>>(spec, options, options_parsed).run();
            }
        } else if (spec.type_ == "int8") {
            if (spec.operation_ == "max") {
                return Benchmark_1In_1Out<Vector_scan_parallel_kernel<std::int8_t, cuda_max_op<std::int8_t>>>(spec, options, options_parsed).run();
            } else if (spec.operation_ == "min") {
                return Benchmark_1In_1Out<Vector_scan_parallel_kernel<std::int8_t, cuda_min_op<std::int8_t>>>(spec, options, options_parsed).run();
            } else if (spec.operation_ == "sum") {
                return Benchmark_1In_1Out<Vector_scan_parallel_kernel<std::int8_t, cuda_sum_op<std::int8_t>>>(spec, options, options_parsed).run();
            } else if (spec.operation_ == "prod") {
                return Benchmark_1In_1Out<Vector_scan_parallel_kernel<std::int8_t, cuda_prod_op<std::int8_t>>>(spec, options, options_parsed).run();
            }
        } else if (spec.type_ == "int16") {
            if (spec.operation_ == "max") {
                return Benchmark_1In_1Out<Vector_scan_parallel_kernel<std::int16_t, cuda_max_op<std::int16_t>>>(spec, options, options_parsed).run();
            } else if (spec.operation_ == "min") {
                return Benchmark_1In_1Out<Vector_scan_parallel_kernel<std::int16_t, cuda_min_op<std::int16_t>>>(spec, options, options_parsed).run();
            } else if (spec.operation_ == "sum") {
                return Benchmark_1In_1Out<Vector_scan_parallel_kernel<std::int16_t, cuda_sum_op<std::int16_t>>>(spec, options, options_parsed).run();
            } else if (spec.operation_ == "prod") {
                return Benchmark_1In_1Out<Vector_scan_parallel_kernel<std::int16_t, cuda_prod_op<std::int16_t>>>(spec, options, options_parsed).run();
            }
        } else if (spec.type_ == "int32") {
            if (spec.operation_ == "max") {
                return Benchmark_1In_1Out<Vector_scan_parallel_kernel<std::int32_t, cuda_max_op<std::int32_t>>>(spec, options, options_parsed).run();
            } else if (spec.operation_ == "min") {
                return Benchmark_1In_1Out<Vector_scan_parallel_kernel<std::int32_t, cuda_min_op<std::int32_t>>>(spec, options, options_parsed).run();
            } else if (spec.operation_ == "sum") {
                return Benchmark_1In_1Out<Vector_scan_parallel_kernel<std::int32_t, cuda_sum_op<std::int32_t>>>(spec, options, options_parsed).run();
            } else if (spec.operation_ == "prod") {
                return Benchmark_1In_1Out<Vector_scan_parallel_kernel<std::int32_t, cuda_prod_op<std::int32_t>>>(spec, options, options_parsed).run();
            }
        } else if (spec.type_ == "int64") {
            if (spec.operation_ == "max") {
                return Benchmark_1In_1Out<Vector_scan_parallel_kernel<std::int64_t, cuda_max_op<std::int64_t>>>(spec, options, options_parsed).run();
            } else if (spec.operation_ == "min") {
                return Benchmark_1In_1Out<Vector_scan_parallel_kernel<std::int64_t, cuda_min_op<std::int64_t>>>(spec, options, options_parsed).run();
            } else if (spec.operation_ == "sum") {
                return Benchmark_1In_1Out<Vector_scan_parallel_kernel<std::int64_t, cuda_sum_op<std::int64_t>>>(spec, options, options_parsed).run();
            } else if (spec.operation_ == "prod") {
                return Benchmark_1In_1Out<Vector_scan_parallel_kernel<std::int64_t, cuda_prod_op<std::int64_t>>>(spec, options, options_parsed).run();
            }
        } else if (spec.type_ == "uint8") {
            if (spec.operation_ == "max") {
                return Benchmark_1In_1Out<Vector_scan_parallel_kernel<std::uint8_t, cuda_max_op<std::uint8_t>>>(spec, options, options_parsed).run();
            } else if (spec.operation_ == "min") {
                return Benchmark_1In_1Out<Vector_scan_parallel_kernel<std::uint8_t, cuda_min_op<std::uint8_t>>>(spec, options, options_parsed).run();
            } else if (spec.operation_ == "sum") {
                return Benchmark_1In_1Out<Vector_scan_parallel_kernel<std::uint8_t, cuda_sum_op<std::uint8_t>>>(spec, options, options_parsed).run();
            } else if (spec.operation_ == "prod") {
                return Benchmark_1In_1Out<Vector_scan_parallel_kernel<std::uint8_t, cuda_prod_op<std::uint8_t>>>(spec, options, options_parsed).run();
            }
        } else if (spec.type_ == "uint16") {
            if (spec.operation_ == "max") {
                return Benchmark_1In_1Out<Vector_scan_parallel_kernel<std::uint16_t, cuda_max_op<std::uint16_t>>>(spec, options, options_parsed).run();
            } else if (spec.operation_ == "min") {
                return Benchmark_1In_1Out<Vector_scan_parallel_kernel<std::uint16_t, cuda_min_op<std::uint16_t>>>(spec, options, options_parsed).run();
            } else if (spec.operation_ == "sum") {
                return Benchmark_1In_1Out<Vector_scan_parallel_kernel<std::uint16_t, cuda_sum_op<std::uint16_t>>>(spec, options, options_parsed).run();
            } else if (spec.operation_ == "prod") {
                return Benchmark_1In_1Out<Vector_scan_parallel_kernel<std::uint16_t, cuda_prod_op<std::uint16_t>>>(spec, options, options_parsed).run();
            }
        } else if (spec.type_ == "uint32") {
            if (spec.operation_ == "max") {
                return Benchmark_1In_1Out<Vector_scan_parallel_kernel<std::uint32_t, cuda_max_op<std::uint32_t>>>(spec, options, options_parsed).run();
            } else if (spec.operation_ == "min") {
                return Benchmark_1In_1Out<Vector_scan_parallel_kernel<std::uint32_t, cuda_min_op<std::uint32_t>>>(spec, options, options_parsed).run();
            } else if (spec.operation_ == "sum") {
                return Benchmark_1In_1Out<Vector_scan_parallel_kernel<std::uint32_t, cuda_sum_op<std::uint32_t>>>(spec, options, options_parsed).run();
            } else if (spec.operation_ == "prod") {
                return Benchmark_1In_1Out<Vector_scan_parallel_kernel<std::uint32_t, cuda_prod_op<std::uint32_t>>>(spec, options, options_parsed).run();
            }
        } else if (spec.type_ == "uint64") {
            if (spec.operation_ == "max") {
                return Benchmark_1In_1Out<Vector_scan_parallel_kernel<std::uint64_t, cuda_max_op<std::uint64_t>>>(spec, options, options_parsed).run();
            } else if (spec.operation_ == "min") {
                return Benchmark_1In_1Out<Vector_scan_parallel_kernel<std::uint64_t, cuda_min_op<std::uint64_t>>>(spec, options, options_parsed).run();
            } else if (spec.operation_ == "sum") {
                return Benchmark_1In_1Out<Vector_scan_parallel_kernel<std::uint64_t, cuda_sum_op<std::uint64_t>>>(spec, options, options_parsed).run();
            } else if (spec.operation_ == "prod") {
                return Benchmark_1In_1Out<Vector_scan_parallel_kernel<std::uint64_t, cuda_prod_op<std::uint64_t>>>(spec, options, options_parsed).run();
            }
        } else {
            throw cxxopts::exceptions::exception("Invalid type: " + spec.type_);
        }

        throw cxxopts::exceptions::exception("Invalid operation: " + spec.operation_);
    } catch (const cxxopts::exceptions::exception& e) {
       std::cerr << "Error parsing options: " << e.what() << std::endl;
       std::cout << options.help() << std::endl;
       return 1;
    }
}
