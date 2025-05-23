// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

#include <cxxopts.hpp>

#include "common/benchmark.h"

#include "cuda/kernels/matrix_product.h"

int main(int argc, char** argv) {
    cxxopts::Options options("matrix_product_naive", "Naive matrix multiplication");
    add_benchmark_options(options);
    options.add_options()
        ("type", "Numeric type (half, single/float, double)", cxxopts::value<std::string>()->default_value("float"));

    try {
        cxxopts::ParseResult options_parsed = options.parse(argc, argv);

        // Validate the type option
        std::string type = options_parsed["type"].as<std::string>();
        if (type != "half" && type != "single" && type != "float" && type != "double") {
            std::cerr << "[ERROR] --type must be one of: half, single/float, double" << std::endl;
            return 1;
        }

        if (type == "half") {
            return Benchmark<Matrix_product_naive<half>>(options_parsed).run();
        } else if (type == "single" || type == "float") {
            return Benchmark<Matrix_product_naive<float>>(options_parsed).run();
        } else if (type == "double") {
            return Benchmark<Matrix_product_naive<double>>(options_parsed).run();
        }
    } catch (const cxxopts::exceptions::exception& e) {
       std::cerr << "Error parsing options: " << e.what() << std::endl;
       std::cout << options.help() << std::endl;
       return 1;
    }
}
